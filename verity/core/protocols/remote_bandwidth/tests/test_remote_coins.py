"""The audit's coins from the proof system's challenger as a service (`sampled_proofs.remote`), served on a thread over a
socket pair: a session whose coins that service released is accepted, and what it signed holds them
(`Session.released_reason`). Under the auditor's ledger (`Session.trusted_reason`) a session's coins are those of the one
challenger session the ledger opens for the session's own window R inside its period, run from a tree the host's pins
log records on main's first-parent line (`challenger.pinned_session`), so coins from a second challenger session over
the same registration are refused, as is a ledger that opens a second session for R, and a session with no pins log,
an unsealed open, an unlogged pin, a pin off main's first-parent line or no copy of main; a coin opened before its
window's root is refused; a coin asked for ahead (`Coins.ahead`) goes in one round trip with the coin before it, the
challenger's log as if each had gone alone; and a session that appends nothing reserves no append family, so its
challenger releases every coin it committed to."""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import socket
import threading
from pathlib import Path

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
MAIN = R.main_lines(Path(__file__).resolve().parents[2] / "sampled_proofs" / "tests" / "fixtures" /
                    "main-first-parent.txt")
"""Main's first-parent history, newest first, from c81c31e6d5 back to 6c2de12715: sampled_proofs' fixture."""
PIN = MAIN[0]
PINS = [C.pin_line(PIN, PIN, "2026-10-22T11:07:28Z")]
"""The host's pins log as infra published it when it repinned vy-challenger-1 to `PIN`; every open seals it unless a
test says otherwise."""
BRANCH, MERGE = "882770e6009e73bdf015f4069fa77ef698d7f326", "1018753ee5ff8c659fa8e8c5ccbc31b0d2b1a366"
"""A 20 Oct pin, #2953's head: in main's history through the queue's merge `MERGE`, never on its first-parent line."""


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

    def __init__(self, directory, sign, sid: str, epoch: int, windows: int, label: bytes, appends: int = APPENDS,
                 per_coin: int = 1) -> None:
        directory.mkdir()
        self.directory, self.id, self.reason = directory, sid, None
        challenger = C.LiveChallenger(coins.SCHEME, epoch, coins.families(windows, appends, per_coin), Entropy(label))
        self.session = R.open_session(challenger, key(), f"{HOST}.{sid}", sign, directory, sid)
        (directory / R.HELLOS).write_text("")
        ours, theirs = socket.socketpair()
        self.thread = threading.Thread(target=self._serve, args=(ours,))
        self.thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket) -> None:
        with sock:
            self.reason = R.serve(sock, self.session, itertools.count().__next__)

    def coins(self, store, session: int, windows: int, per_coin: int = 1) -> coins.Coins:
        return coins.Coins(self.client, store.registration.digest, session, windows, per_coin=per_coin,
                           clock=itertools.count().__next__)

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

    def open(self, sid: str, root: bytes, pinned: str = PIN, pins: list[str] | None = PINS) -> None:
        """Session `sid`'s open for window `root` on a host pinned to `pinned`, sealing its pins log `pins` (None: no
        seal, as every gate's open before RT4-INT-40)."""
        body = {"window": {"root": root.hex()}, "host": f"{HOST}.{sid}", "pinned": pinned,
                "key_sha512": self.key_sha512, "port": 7811, "by": LANE}
        self.append("open", sid, body if pins is None else {**body, "pins": C.pins_seal(pins)})

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


@pytest.mark.parametrize("appends", [0, APPENDS])
def test_a_session_that_appends_nothing_reserves_no_append_family_so_its_challenger_releases_every_coin(
        store, tmp_path, sign, appends):
    """A session reserves exactly the coins it will use (`coins.families`): with no append coins the layout is W
    challenges and W openings coins, every one of which the honest session's challenger released, so the gate's end
    reads it complete. Append coins reserved and never taken stay unreleased, and an honest session reads incomplete,
    as every gate run's window did before."""
    service = Service(tmp_path / "s-0", sign, "s-0", 0, params().windows, b"service", appends)
    s, verdicts = served_session(store, service)
    assert all(v.accepted for v in verdicts) and service.close() is None
    signed, received = service.signed()
    assert s.released_reason(signed, received) is None and coins.layout_reason(signed, 0, params().windows) is None
    assert [label for label, _ in signed.families] == [coins.CHALLENGE, coins.OPENINGS] + [coins.APPEND] * (appends > 0)
    assert all(len(received.get(label, ())) == n for label, n in signed.families) is (appends == 0)


def test_a_coin_layout_is_challenge_and_openings_with_an_append_family_only_for_a_session_that_appends():
    w = params().windows

    def layout(families) -> C.LiveCommitment:
        return C.LiveCommitment(coins.SCHEME, 0, tuple(families), bytes(64))

    assert coins.families(w, 0) == ((coins.CHALLENGE, w), (coins.OPENINGS, w))
    assert coins.families(w, 3) == ((coins.CHALLENGE, w), (coins.OPENINGS, w), (coins.APPEND, 3))
    assert coins.layout_reason(layout(coins.families(w, 0)), 0, w) is None
    assert coins.layout_reason(layout(coins.families(w, 3)), 0, w) is None
    for bad in [[(coins.CHALLENGE, w)], [(coins.CHALLENGE, w), (coins.APPEND, 3)],
                [(coins.OPENINGS, w), (coins.CHALLENGE, w)], [(coins.CHALLENGE, w), (coins.OPENINGS, w + 1)],
                [(coins.CHALLENGE, w), (coins.OPENINGS, w), ("other", 1)]]:
        assert "not challenge" in coins.layout_reason(layout(bad), 0, w)


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
    why = s.trusted_reason(first.directory, [ledger.path], key(), PINS, MAIN, verify)
    assert why == "the challenger signed another coin commitment than the one the session's coins open against"
    why = s.trusted_reason(second.directory, [ledger.path], key(), PINS, MAIN, verify)
    assert why is not None and why.startswith("the challenger's session for this session's window is not trusted")
    ledger.open("s-again", r)
    why = s.trusted_reason(first.directory, [ledger.path], key(), PINS, MAIN, verify)
    assert why is not None and "a second open for root" in why


def test_the_session_the_ledger_opens_for_its_window_is_trusted(store, tmp_path, sign):
    """An honest session whose open sealed the host's pins log, which records its pin, c81c31e6d5, on main's
    first-parent line: trusted, inside the audit's period or with none named, and under a later log that extends the
    one the open sealed."""
    ledger = Ledger(tmp_path / "windows.jsonl", sign)
    ledger.period()
    ledger.open("s-r", coins.window_root(store.registration.digest, 0))
    service = Service(tmp_path / "s-r", sign, "s-r", 0, params().windows, b"service")
    s, verdicts = served_session(store, service)
    assert all(v.accepted for v in verdicts) and service.close() is None
    ledger.end(service)
    assert s.trusted_reason(service.directory, [ledger.path], key(), PINS, MAIN, verify) is None
    assert s.trusted_reason(service.directory, [ledger.path], key(), PINS, MAIN, verify, "p-1") is None
    later = PINS + [C.pin_line(MAIN[0], MAIN[0], "2026-10-23T09:00:00Z")]
    assert s.trusted_reason(service.directory, [ledger.path], key(), later, MAIN, verify, "p-1") is None


OFF_MAIN = ("session s-r ran commit {}, which the host's pins log records but which is not on main's first-parent line "
            "in the verifier's copy (`git rev-list --first-parent origin/main`): a branch commit merged into main is in "
            "main's history but no tree main's check ran on; repin the challenger host to a commit on main's "
            "first-parent line (vy-challenger-pin) and rerun the window there")


@pytest.mark.parametrize("case", ["no-pins", "unsealed", "unlogged", "branch", "no-main", "stale-main"])
def test_a_session_is_trusted_only_from_a_logged_pin_on_mains_first_parent_line(store, tmp_path, sign, case):
    """`coins.trusted` reads the window's session through `challenger.pinned_session`, so an honest session whose coins
    are the challenger's is refused, with the reason, when the verifier has no pins log; when its open sealed none
    (every gate before RT4-INT-40); when the log its open sealed doesn't record its pin (c81c31e6d5's parent); when the
    log records a pin in main's history but off its first-parent line (882770e600); when the verifier has no copy of
    main; and when its copy predates the pin, as a verifier host's list infra wrote before it repinned the challenger.
    The pin is read before the period, so the refusal is the same inside the audit's period."""
    assert MERGE in MAIN and BRANCH not in MAIN and MAIN[-1] == "6c2de12715770b0cb1662e5036e5ee239ff191c1"
    pinned, sealed, pins, main = PIN, PINS, PINS, MAIN
    if case == "no-pins":
        pins = None
    elif case == "unsealed":
        sealed = None
    elif case == "unlogged":
        pinned = MAIN[1]
    elif case == "branch":
        pinned = BRANCH
        sealed = pins = [C.pin_line(BRANCH, MERGE, "2026-10-20T21:05:00Z"), *PINS]
    elif case == "no-main":
        main = None
    else:
        main = MAIN[1:]
    ledger = Ledger(tmp_path / "windows.jsonl", sign)
    ledger.period()
    ledger.open("s-r", coins.window_root(store.registration.digest, 0), pinned, sealed)
    service = Service(tmp_path / "s-r", sign, "s-r", 0, params().windows, b"service")
    s, verdicts = served_session(store, service)
    assert all(v.accepted for v in verdicts) and service.close() is None
    ledger.end(service)
    assert s.released_reason(*service.signed()) is None
    why = {"no-pins": C.NO_PINS, "no-main": C.NO_MAIN,
           "unsealed": ("the auditor's ledger's open of session s-r seals no pins log (pins: {lines, sha512}): open the "
                        "window on a host whose gate seals its pins log in each open"),
           "unlogged": (f"session s-r ran commit {MAIN[1]}, which the host's pins log doesn't record as a commit of "
                        "main's history: repin the challenger host to a commit origin/main contains "
                        "(vy-challenger-pin) and rerun the window there"),
           "branch": OFF_MAIN.format(BRANCH), "stale-main": OFF_MAIN.format(PIN)}[case]
    for period in (None, "p-1"):
        assert s.trusted_reason(service.directory, [ledger.path], key(), pins, main, verify, period) == (
            f"the challenger's session for this session's window is not trusted: {why}")
        with pytest.raises(C.CoinError) as e:
            coins.trusted(s.coins, service.directory, [ledger.path], key(), pins, main, verify, period)
        assert str(e.value) == why


@pytest.mark.parametrize("held", ["p-2", None])
def test_a_session_is_bound_to_the_audits_period(store, tmp_path, sign, held):
    """The audit names its period on the ledger: a session whose window the ledger holds in another period, or in none,
    is refused though its coins are the challenger's."""
    ledger = Ledger(tmp_path / "windows.jsonl", sign)
    if held is not None:
        ledger.period(held)
    ledger.open("s-r", coins.window_root(store.registration.digest, 0))
    service = Service(tmp_path / "s-r", sign, "s-r", 0, params().windows, b"service")
    s, verdicts = served_session(store, service)
    assert all(v.accepted for v in verdicts) and service.close() is None
    ledger.end(service)
    assert s.trusted_reason(service.directory, [ledger.path], key(), PINS, MAIN, verify) is None
    why = s.trusted_reason(service.directory, [ledger.path], key(), PINS, MAIN, verify, "p-1")
    assert why == ("the challenger's session for this session's window is not trusted: the auditor's ledger holds the "
                   f"session's window in period {held}, not the audit's period p-1")


def test_an_openings_coin_opened_before_its_windows_root_is_refused(store, tmp_path, sign):
    """Window 0's openings coin asked for after its first root, not its last (it has four): the challenger's signed log
    shows the root it followed, and the session's coins are refused."""
    service = Service(tmp_path / "s-0", sign, "s-0", 0, params().windows, b"service")
    s, _, _, _ = drive(store, params(), SCHEDULE, coins_=service.coins(store, 0, params().windows))
    assert len(s.windows[0].roots) == 4
    s.coins.openings(0, s.windows[0].roots[0])
    assert service.close() is None
    assert s.released_reason(*service.signed()) == "window 0's openings coin was opened before its window's root"


def test_a_coin_asked_for_ahead_goes_in_one_round_trip_with_the_coin_before_it(store, tmp_path, sign):
    """`Coins.ahead`: window 0's openings coin and σ_1 after its root go in one `open_many`, the openings coin first,
    and σ_1 is then taken from the batch with no frame sent: the same coins, the same roots recorded for each family
    and the challenger's release log byte for byte as asking them one after the other. σ_1 asked for after another
    root, or a coin asked for ahead that is not its family's next, is refused, the latter before any frame is sent."""
    root = hashlib.sha512(b"window 0's root").digest()

    def ask(directory, ahead: bool):
        service = Service(directory, sign, "s-0", 0, params().windows, b"service")
        asked, many = [], service.client.open_many
        service.client.open_many = lambda requests: asked.append([r[:2] for r in requests]) or many(requests)
        c = service.coins(store, 0, params().windows)
        c.challenge(0, c.window)
        with c.ahead(coins.CHALLENGE, 1, root) if ahead else contextlib.nullcontext():
            got = [c.openings(0, root)]
        before = (directory / R.RELEASES).read_bytes()
        got.append(c.challenge(1, root))
        assert service.close() is None
        log = (directory / R.RELEASES).read_bytes()
        return asked, got, {label: (c.afters(label), c.opened(label)) for label in coins.LABELS}, before == log, log

    one = ask(tmp_path / "one", False)
    many = ask(tmp_path / "many", True)
    assert one[0] == [[("challenge", 0)], [("openings", 0)], [("challenge", 1)]]
    assert many[0] == [[("challenge", 0)], [("openings", 0), ("challenge", 1)]]
    assert many[1:3] == one[1:3] and many[4] == one[4]
    assert (one[3], many[3]) == (False, True)

    service = Service(tmp_path / "other", sign, "s-1", 0, params().windows, b"service")
    c = service.coins(store, 0, params().windows)
    c.challenge(0, c.window)
    with c.ahead(coins.CHALLENGE, 1, root):
        c.openings(0, root)
    with pytest.raises(C.CoinError, match="challenge coin 1 was asked for ahead after another root"):
        c.challenge(1, c.window)
    sent = (service.directory / R.RELEASES).read_bytes()
    with c.ahead(coins.CHALLENGE, 3, root), pytest.raises(C.CoinError, match="the challenge coin asked for ahead is "
                                                                             "3, not the next, 2"):
        c.openings(1, root)
    assert (service.directory / R.RELEASES).read_bytes() == sent and c.afters(coins.OPENINGS) == (root,)
    assert service.close() is None


def test_a_challenge_opened_before_the_sessions_window_root_is_refused(store, tmp_path, sign):
    service = Service(tmp_path / "s-0", sign, "s-0", 0, params().windows, b"service")
    c = service.coins(store, 0, params().windows)
    c.challenge(0, bytes(64))
    assert service.close() is None
    signed, received = service.signed()
    assert coins.released_reason(c, [], [], signed, received) == "σ_0 was not opened after the session's window root R"
