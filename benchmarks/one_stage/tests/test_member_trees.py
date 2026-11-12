"""A window of several members' sessions on the window's own challenger session (`member_trees`, `a5_coins`'
`hold`, `release`, `tree` and `tree_open`), against the challenger as its host serves it (`remote.serve`, withheld):
a rank holds the session open past its sample; each verifier reaches it through the relay, as `serve --tree-from`
does, under the relay's own secret; each session's tree is closed after its evidence once its files are written; and the
session's release log then shows the sample, then each session's rounds and its closing round, one tree after another,
which `trees_reason` holds the sessions' directories to. The relay passes on tree frames alone, and what it refuses
ends its own connection, never the challenger's session. A stand-in for the verifier's side only: the client here is
`remote.RemoteChallenger`, which checks each round against its tree as Flock's does."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
import threading
from pathlib import Path

import pytest

from verity.commitments import coin_tree as CT
from verity.commitments.crypto import lms
from verity.protocols.spot.device import challenger as C, remote as R
from verity.protocols.spot.verifier.one_stage import closing as K

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402
import member_trees as MT  # noqa: E402

KEY = lms.keygen(lms.Params(64, 4, 4), bytes([7]) * 64)
WINDOW = "ab" * 64
REGISTRATION = bytes([5]) * 64
SPEC = {"block": 512, "rounds_log": 9, "scheme": CT.SCHEME, "streams": [CT.SESSION, "circuit/rep0", "circuit/rep1"]}
ROOT_F, ROOT_B = bytes([20]) * 64, {"circuit": bytes([21]) * 64}


class Host:
    """The challenger's host: one withheld session of two coins of the samples' family, served to the one client that connects to
    `address`; `log` is its release log once `ended`."""

    def __init__(self, tmp: Path) -> None:
        self.directory = tmp / "challenger"
        self.directory.mkdir()
        state = tmp / "key.lms"
        lms.LeafState.create(state, lms.public_key(KEY), KEY.params.H)
        self._state = lms.LeafState(state, lms.public_key(KEY), KEY.params.H)
        ch = C.LiveChallenger(AC.SCHEME, 0, [(AC.FAMILY, 2)], random.Random("host").randbytes)
        key = C.ChallengerKey(KEY.params.W, KEY.params.H, lms.public_key(KEY))
        session = R.open_session(ch, key, "auditor-1", lambda m: lms.sign(KEY, self._state, m).encode(), self.directory,
                                 "s-1", withhold=True)
        listener = MT.listener()
        self.address = "127.0.0.1:%d" % listener.getsockname()[1]
        self._ended: list[str | None] = []

        def run() -> None:
            with listener:
                sock, _ = listener.accept()
            with sock:
                self._ended.append(R.serve(sock, session, itertools.count().__next__))
        self._thread = threading.Thread(target=run)
        self._thread.start()

    def ended(self) -> str | None:
        """How the session ended (`remote.serve`'s), once it has."""
        if self._thread.is_alive() or self._state is not None:
            self._thread.join()
            self._state.close()
            self._state = None
        return self._ended[0]

    @property
    def log(self) -> list[str]:
        return R.session_files(self.directory)["log"]


@pytest.fixture
def held(tmp_path):
    """The host, and rank 0's proxy on its session: planned, connected, held for its sessions' trees, its sample of
    one coin read after `REGISTRATION` and ended. A test releases the rank, or the fixture does."""
    host = Host(tmp_path)
    proxy = AC.Draws(WINDOW, [0])
    planning = threading.Thread(target=proxy.plan, args=(0, 1))
    planning.start()
    assert proxy.wait_planned() == {0: 1}
    proxy.connect(host.address, None, None, True)
    planning.join()
    proxy.hold(0)
    proxy.commit(0, REGISTRATION.hex())
    assert len(proxy.read(0, C.COIN_BYTES)) == C.COIN_BYTES
    assert proxy.drawn(0)["label"] == AC.FAMILY
    yield host, proxy
    proxy.skip(0)
    host.ended()


def verifier(relay: MT.Relay) -> R.RemoteChallenger:
    """A verifier's connection to the relay, as `serve` makes it from the relay's flags."""
    flags = dict(zip(relay.flags[::2], relay.flags[1::2]))
    return R.RemoteChallenger.connect(flags["--tree-from"], bytes.fromhex(Path(flags["--tree-secret-file"]).read_text()))


def hang_up(client: R.RemoteChallenger) -> None:
    """The verifier's close, as `serve` sends it when it exits: a close frame, answered ``closed``."""
    client._ask({"type": "close"}, "closed", lambda _: None)
    client._end("the verifier closed its connection")


def session(client: R.RemoteChallenger, directory: Path, nonce: bytes) -> Path:
    """One ``--zk`` session through `client`, as its verifier asks for its coins: its tree, its link points and a
    round of each rep, each checked against the tree; and its files, as its server writes them to `directory`."""
    nonce, root, key = client.tree(SPEC, nonce)
    points, _ = client.tree_open(root, CT.SESSION, 0, 2, C.link_after(ROOT_F, ROOT_B))
    streams = []
    for rep in SPEC["streams"][1:]:
        message = hashlib.sha512(rep.encode() + root).digest()
        coins, _ = client.tree_open(root, rep, 0, 3, message)
        streams.append({"stream": rep, "rounds": [{"coins": [f"{c:032x}" for c in coins], "k": 0,
                                                   "msg_sha512": message.hex(), "n": 3}]})
    record = {"coin_tree": {"key": key.hex(), "nonce": nonce.hex(), "root": root.hex(), "spec": SPEC},
              "link": {"points": [[f"{c:032x}"] for c in points], "root_b": {t: r.hex() for t, r in ROOT_B.items()},
                       "root_f": ROOT_F.hex()}, "streams": streams}
    directory.mkdir()
    (directory / "session.json").write_text(json.dumps(record, sort_keys=True))
    for rep in SPEC["streams"][1:]:
        (directory / f"{rep.replace('/', '.')}.bin").write_bytes(b"proof of " + rep.encode() + root)
    return directory


def test_two_sessions_on_the_windows_session_each_closed_after_its_evidence_pass_its_log(held, tmp_path):
    """Two verifiers, one a template, each on its own connection to the relay: the first's session runs and is closed
    after its evidence, then the second's. The challenger's session stays open through both and closes at the rank's
    release, under its signed close; its log is the sample, then the link, then the other rounds, and holds the two
    sessions in the order they ran, and in no other. A session's file changed after its closing round, or a session
    left open, is refused."""
    host, proxy = held
    relay = MT.Relay(proxy, MT.listener())
    first, second = verifier(relay), verifier(relay)
    assert first.layout() == second.layout() == proxy.layout() == {"epoch": 0, "families": [[AC.FAMILY, 2]],
                                                                    "scheme": AC.SCHEME}
    a = session(first, tmp_path / "a", bytes(range(32)))
    after_a = MT.closing_round(proxy, a)
    assert after_a == K.closing_after(a).hex() == hashlib.sha512(K.session_evidence(
        (a / "session.json").read_bytes(), [(f.name, f.read_bytes()) for f in sorted(a.glob("*.bin"))])).hexdigest()
    b = session(second, tmp_path / "b", bytes(range(32, 64)))
    still_open = MT.trees_reason(R.log_lines(host.directory / R.RELEASES), [("a", a), ("b", b)])
    assert still_open == f"session b's coins are not the challenger's: {C.NO_CLOSING}"
    MT.closing_round(proxy, b)
    hang_up(first)
    hang_up(second)
    assert relay.close() == [] and not Path(relay.flags[3]).exists()
    assert not (host.directory / R.CLOSE).exists()
    proxy.release(0)
    assert proxy.wait_closed() is None and host.ended() is None
    log = host.log
    events = [json.loads(line)["event"] for line in log]
    assert events == ["commitment", "root", "release", "root", "release", *(["coin-tree"] + ["tree-open"] * 4) * 2]
    closing = [json.loads(line) for line in log if json.loads(line).get("round") == C.CLOSING_ROUND]
    assert [(x["stream"], x["n"], x["after"], x["coins"]) for x in closing] == [
        (CT.SESSION, 0, after, hashlib.sha512(b"").hexdigest()) for after in (after_a, K.closing_after(b).hex())]
    assert C.sample_link_first(AC.FAMILY, REGISTRATION, log[1:])
    assert MT.trees_reason(log, [("a", a), ("b", b)]) is None
    assert MT.trees_reason(log, [("b", b), ("a", a)]) == C.MEMBERS_ORDER
    assert "the sessions use 1" in MT.trees_reason(log, [("a", a)])
    (b / "circuit.rep1.bin").write_bytes(b"another proof")
    assert MT.trees_reason(log, [("a", a), ("b", b)]) == f"session b's coins are not the challenger's: {C.NO_CLOSING}"


def test_the_relay_passes_on_tree_frames_alone_and_a_refusal_ends_its_own_connection(held, tmp_path):
    """A hello without the relay's secret is refused before anything is asked of the challenger; a coin's ``open``
    through the relay is refused by the relay, which ends that connection alone: the challenger's session takes the
    next verifier's tree. A closing round asked twice is the challenger's to refuse, which ends its session."""
    host, proxy = held
    relay = MT.Relay(proxy, MT.listener())
    with pytest.raises(C.CoinError, match="the hello's secret is not the session's"):
        R.RemoteChallenger.connect(relay.address, bytes(32))
    with pytest.raises(C.CoinError, match="a client's first frame is hello with the session's secret"):
        R.RemoteChallenger.connect(relay.address)
    stray = verifier(relay)
    with pytest.raises(C.CoinError, match="the challenger refused: a relay's frames are hello, then tree"):
        stray.open(AC.FAMILY, 1, bytes(64))
    left_open = verifier(relay)
    a = session(left_open, tmp_path / "a", bytes(range(32)))
    MT.closing_round(proxy, a)
    assert sorted(relay.close()) == sorted(["the hello's secret is not the session's",
                                            "a client's first frame is hello with the session's secret", MT.FRAMES,
                                            "the stream closed"])
    with pytest.raises(C.CoinError, match="the challenger refused: the next round of the coin tree's session stream is 2"):
        MT.closing_round(proxy, a)
    with pytest.raises(C.CoinError, match="a retry never redraws"):
        proxy.tree(SPEC, bytes(32))
    proxy.release(0)
    assert proxy.wait_closed() is not None
    assert "the next round of the coin tree's session stream is 2" in host.ended()


def test_a_rank_holds_the_session_before_its_sample_ends_and_takes_no_tree_once_it_closed(tmp_path):
    """Without a hold the session closes with the last rank's sample, as it always did; a hold after that is refused,
    and so is a tree."""
    host = Host(tmp_path)
    proxy = AC.Draws(WINDOW, [0])
    planning = threading.Thread(target=proxy.plan, args=(0, 1))
    planning.start()
    proxy.wait_planned()
    proxy.connect(host.address, None, None, True)
    planning.join()
    proxy.commit(0, REGISTRATION.hex())
    proxy.read(0, C.COIN_BYTES)
    proxy.drawn(0)
    assert proxy.wait_closed() is None and host.ended() is None
    with pytest.raises(C.CoinError, match="rank 0 holds it for its sessions' coin trees before its sample ends"):
        proxy.hold(0)
    with pytest.raises(C.CoinError, match="one rank's sessions take their coin trees from it"):
        AC.Draws(WINDOW, [0, 1]).hold(0)
    with pytest.raises(C.CoinError, match="the challenger's session is not open"):
        proxy.tree(SPEC, bytes(32))
    with pytest.raises(C.CoinError, match="the challenger's session is not open"):
        proxy.layout()
