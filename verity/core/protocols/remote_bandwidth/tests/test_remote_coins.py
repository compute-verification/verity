"""The audit's coins from the proof service's challenger as a service (`sampled_proofs.remote`), served on a thread over a
socket pair: a session whose coins that service released is accepted, and what it signed holds them
(`Session.released_reason`). Under the auditor's ledger (`Session.trusted_reason`) a session's coins are those of the one
challenger session the ledger opens for the session's own window R inside its period, so coins from a second challenger
session over the same registration are refused, as is a ledger that opens a second session for R; and a coin opened
before its window's root is refused."""
from __future__ import annotations

import hashlib
import itertools
import json
import socket
import threading

import pytest

from test_remote_bandwidth import SCHEDULE, Entropy, drive, finish, params, store  # noqa: F401
from verity.core.protocols.remote_bandwidth import coins
from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

KEY = lms.keygen(lms.Params(64, 4, 4), bytes([7]) * 64)
"""The challenger key: sixteen leaves, enough for a period's open and two sessions' opens, records, closes and ends."""
HOST = "vy-test-challenger"
LANE = "memory-accounting"
APPENDS = 8
CIRCUIT = hashlib.sha512(b"verity/remote-bandwidth/v0").hexdigest()


def key() -> C.ChallengerKey:
    return C.ChallengerKey(KEY.params.W, KEY.params.H, lms.public_key(KEY))


def verify(k: C.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, k.W, k.H)
    return lms.verify(p, k.public, message, lms.Signature.decode(p, signature))


@pytest.fixture
def sign(tmp_path):
    """The key's signer at its next leaf: the host's signatures and the ledger's, in the order they're made."""
    path = tmp_path / "state.json"
    lms.LeafState.create(path, lms.public_key(KEY), KEY.params.H)
    with lms.LeafState(path, lms.public_key(KEY), KEY.params.H) as state:
        yield lambda m: lms.sign(KEY, state, m).encode()


class Service:
    """Challenger session `sid` on its host: epoch `epoch`'s coins over `windows` windows, its record signed and written
    to `directory` before it serves (`remote.open_session`), served on a thread over a socket pair whose other end is
    `client`, the verifier's `remote.RemoteChallenger`."""

    def __init__(self, directory, sign, sid: str, epoch: int, windows: int, label: bytes) -> None:
        directory.mkdir()
        self.directory, self.id, self.reason = directory, sid, None
        challenger = C.LiveChallenger(coins.SCHEME, epoch, coins.families(windows, APPENDS), Entropy(label))
        self.session = R.open_session(challenger, key(), f"{HOST}.{sid}", sign, directory, sid)
        (directory / R.HELLOS).write_text("")
        ours, theirs = socket.socketpair()
        self.thread = threading.Thread(target=self._serve, args=(ours,))
        self.thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket) -> None:
        with sock:
            self.reason = R.serve(sock, self.session, itertools.count().__next__)

    def coins(self, store, session: int, windows: int) -> coins.Coins:
        return coins.Coins(self.client, store.registration.digest, session, windows, clock=itertools.count().__next__)

    def close(self) -> str | None:
        self.client.close()
        self.thread.join()
        return self.reason

    def signed(self):
        """The commitment the challenger signed and the roots its signed release log shows, with no ledger."""
        files = R.session_files(self.directory)
        commitment, received, _ = C.session_of(key(), files["record"], files["log"], files["close"], verify)
        return commitment, received


class Ledger:
    """The auditor's ledger in `path`, each entry signed with `sign` under the challenger key."""

    def __init__(self, path, sign) -> None:
        self.path, self.sign, self.lines = path, sign, []
        self.key_sha512 = hashlib.sha512(lms.public_key(KEY)).hexdigest()

    def append(self, kind: str, session: str | None, body: dict) -> None:
        self.lines.append(C.ledger_line(self.lines, kind, session, body, lms.public_key(KEY)[:16], self.sign))
        self.path.write_text("".join(line + "\n" for line in self.lines))

    def period(self, name: str = "p-1") -> None:
        self.append("period-open", None, {"period": name, "lane": LANE, "key_sha512": self.key_sha512, "cap": 4,
                                          "circuit": CIRCUIT})

    def open(self, sid: str, root: bytes) -> None:
        self.append("open", sid, {"window": {"root": root.hex()}, "host": f"{HOST}.{sid}", "pinned": "ab" * 20,
                                  "key_sha512": self.key_sha512, "port": 7811, "by": LANE})

    def end(self, service: Service) -> None:
        d = service.directory
        self.append("end", service.id, {
            "closed": True, "rc": 0, "complete": False, "ended": "closed by the client", "refused_hellos": 0,
            "files": {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()},
            "record_leaf": int(json.loads((d / R.RECORD).read_text())["signature"][:8], 16),
            "commitment": service.session.challenger.commitment.root.hex()})


def served_session(store, service: Service):
    """An honest session over `store` whose coins `service` releases, every window verified: the session and verdicts."""
    p = params()
    s, prover, passes, clock = drive(store, p, SCHEDULE, coins_=service.coins(store, 0, p.windows))
    return s, finish(s, prover, passes, clock)


def test_a_session_through_a_challenger_service_is_accepted_and_holds_to_what_it_signed(store, tmp_path, sign):
    service = Service(tmp_path / "s-0", sign, "s-0", 0, params().windows, b"service")
    s, verdicts = served_session(store, service)
    assert all(v.accepted for v in verdicts) and s.outcome == "accepted"
    assert service.close() is None
    signed, received = service.signed()
    assert signed == s.commitment and s.released_reason(signed, received) is None
    assert received[coins.CHALLENGE][0] == coins.window_root(store.registration.digest, 0)
    assert received[coins.OPENINGS] == tuple(w.roots[-1] for w in s.windows)
    other = Service(tmp_path / "s-1", sign, "s-1", 0, params().windows, b"another")
    assert other.close() is None
    assert "another coin commitment" in s.released_reason(*other.signed())


def test_coins_from_a_second_session_for_the_registration_are_refused(store, tmp_path, sign):
    """The ledger opens one challenger session for R, the session's window over its registration, inside the lane's
    period: a session whose coins came from another challenger session (opened in the same period for another root)
    is refused against R's, and a ledger that opens a second session for R is refused whole."""
    r = coins.window_root(store.registration.digest, 0)
    ledger = Ledger(tmp_path / "windows.jsonl", sign)
    ledger.period()
    ledger.open("s-r", r)
    first = Service(tmp_path / "s-r", sign, "s-r", 0, params().windows, b"first")
    assert first.close() is None
    ledger.end(first)
    ledger.open("s-other", hashlib.sha512(b"another window").digest())
    second = Service(tmp_path / "s-other", sign, "s-other", 0, params().windows, b"second")
    s, verdicts = served_session(store, second)
    assert all(v.accepted for v in verdicts) and second.close() is None
    ledger.end(second)
    assert s.released_reason(*second.signed()) is None
    why = s.trusted_reason(first.directory, [ledger.path], key(), verify)
    assert why == "the challenger signed another coin commitment than the one the session's coins open against"
    why = s.trusted_reason(second.directory, [ledger.path], key(), verify)
    assert why is not None and why.startswith("the challenger's session for this session's window is not trusted")
    ledger.open("s-again", r)
    why = s.trusted_reason(first.directory, [ledger.path], key(), verify)
    assert why is not None and "a second open for root" in why


def test_the_session_the_ledger_opens_for_its_window_is_trusted(store, tmp_path, sign):
    ledger = Ledger(tmp_path / "windows.jsonl", sign)
    ledger.period()
    ledger.open("s-r", coins.window_root(store.registration.digest, 0))
    service = Service(tmp_path / "s-r", sign, "s-r", 0, params().windows, b"service")
    s, verdicts = served_session(store, service)
    assert all(v.accepted for v in verdicts) and service.close() is None
    ledger.end(service)
    assert s.trusted_reason(service.directory, [ledger.path], key(), verify) is None


def test_an_openings_coin_opened_before_its_windows_root_is_refused(store, tmp_path, sign):
    """Window 0's openings coin asked for after its first root, not its last (it has four): the challenger's signed log
    shows the root it followed, and the session's coins are refused."""
    service = Service(tmp_path / "s-0", sign, "s-0", 0, params().windows, b"service")
    s, _, _, _ = drive(store, params(), SCHEDULE, coins_=service.coins(store, 0, params().windows))
    assert len(s.windows[0].roots) == 4
    s.coins.openings(0, s.windows[0].roots[0])
    assert service.close() is None
    assert s.released_reason(*service.signed()) == "window 0's openings coin was opened before its window's root"


def test_a_challenge_opened_before_the_sessions_window_root_is_refused(store, tmp_path, sign):
    service = Service(tmp_path / "s-0", sign, "s-0", 0, params().windows, b"service")
    c = service.coins(store, 0, params().windows)
    c.challenge(0, bytes(64))
    assert service.close() is None
    signed, received = service.signed()
    assert coins.released_reason(c, [], [], signed, received) == "σ_0 was not opened after the session's window root R"
