"""a5's trees session (`a5_coins.start`'s ``trees``, one rank's audit): its drawn member's ``--zk`` server takes its
session's coin tree from the step's session through the relay (`a5_coins.Draws.trees`, `TreeRelay`), in the frames
``flock-circuit serve --tree-from`` sends (`flock_live::tree_from`), so the session's release log holds the tree the
member's record must show (`challenger.sessions_tree_reason`, which the challenger difftest holds to v2's Lean,
`OneStageZk.treesHeld`). The relay's close ends its connection alone: the session stays open until the rank's sessions
are done (`Draws.sessions_done`), and only once it has closed does a5's tie run (`a5.tie_after_sessions`), whose failure
refuses the audit (`a5.refuse_after_sessions`). Each test opens a local challenger session (`test_a5_reserve.local`); the
one through a real ``serve --zk --tree-from`` is opt-in."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.sampled_proofs import challenger as Ch, coins as CC, remote as R, service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as REG
from verity.ml.commitments import coin_tree as CT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a0  # noqa: E402
import a5  # noqa: E402
import a5_coins as AC  # noqa: E402
import test_a5_reserve as RS  # noqa: E402

TREE_SPEC = {"block": 3, "rounds_log": 2, "scheme": CT.SCHEME, "streams": [CT.SESSION, "t/rep0", "t/rep1"]}
NONCE, OTHER_NONCE = bytes(range(32)), bytes(range(32, 64))
ROOT_F, ROOT_B = bytes([20]) * 64, {"t": bytes([21]) * 64}
RULE = "stratified:4"
ROPE = AC.TREE / "benchmarks/numerical/tests/bench/data/captured-101/slices/rope-head-d64"


def drawn(draws, after: bytes = RS.AFTER) -> tuple[AC.RankCoins, dict]:
    """Rank 0's draw of SiluMul's window (`test_a5_reserve.SILU_MUL`) on the step's session, its coins after `after`: its
    coins and its segment."""
    coins = AC.RankCoins(draws, 0)
    coins.plan(RS.need_of(RULE, RS.SILU_MUL))
    coins.commit(after)
    S.select_draw(S.Law(RULE), RS.law_of(RULE, RS.SILU_MUL), sum(RS.SILU_MUL), coins)
    return coins, coins.drawn()


class Server:
    """The relay at `address` as ``flock-circuit serve --tree-from ADDRESS --tree-secret-file FILE`` asks it: one connection,
    its hello carrying `secret` (`hello`, the relay's answer), then one frame at a time, each answered."""

    def __init__(self, address: str, secret: str) -> None:
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.wire = R._Wire(self.sock)
        self.hello = self.ask({"type": "hello", "secret": secret})

    def ask(self, head: dict) -> dict:
        self.wire.send(head)
        return self.wire.recv()

    def session(self, nonce: bytes) -> dict:
        """A ``--zk`` session's coin tree, sampled at its Hello under `nonce`, and two of its rounds, each opening checked
        against the tree as the server checks it (`coin_tree.check`): its link points (the session stream's round 0,
        after its link's roots) and t/rep0's round 0 after that round's message. -> the session's record on them."""
        got = self.ask({"type": "tree", "nonce": nonce.hex(), "spec": TREE_SPEC})
        assert sorted(got) == ["tree", "type"] and got["tree"]["nonce"] == nonce.hex()
        t, spec = got["tree"], CT.Spec.of_json(TREE_SPEC)
        rules = CT.Rules(bytes.fromhex(t["nonce"]), bytes.fromhex(t["key"]))

        def opened(stream: str, n: int, after: bytes) -> list[str]:
            ws = self.ask({"type": "tree-open", "after": after.hex(), "n": n, "round": 0, "stream": stream, "tree": t["root"]})
            assert sorted(ws) == ["tree-coins", "type"]
            coins = CT.check(spec, rules, bytes.fromhex(t["root"]), spec.streams.index(stream), 0, n,
                             [int(w, 16) for w in ws["tree-coins"]])
            return [f"{c:032x}" for c in coins]
        msg = hashlib.sha512(b"t/rep0 round 0").digest()
        points, rep = opened(CT.SESSION, 2, Ch.link_after(ROOT_F, ROOT_B)), opened("t/rep0", 3, msg)
        return {"coin_tree": {**t, "spec": TREE_SPEC},
                "link": {"points": [[c] for c in points], "root_b": {k: v.hex() for k, v in ROOT_B.items()},
                         "root_f": ROOT_F.hex()},
                "streams": [{"stream": "t/rep0", "rounds": [{"coins": rep, "msg_sha512": msg.hex(), "n": 3}]}]}

    def close(self) -> dict:
        reply = self.ask({"type": "close"})
        self.sock.close()
        return reply


def secret_of(file: str) -> str:
    return Path(file).read_text().strip()


def test_the_relays_close_leaves_the_session_open_until_the_ranks_sessions_are_done(tmp_path, monkeypatch):
    """A trees session stays open past the rank's draw. Its relay listens on 127.0.0.1, its secret fresh in a 0600 file
    beside the session's evidence directory (`start`'s ``dest``), never in it; a hello without the secret is refused and
    the next connection taken. The
    server's coin tree and rounds go through it to the step's one client, and back as the challenger answered them, each
    opening its tree. The server's close ends the relay's connection alone: the session is still open and its challenger
    has written no close. Once the rank's sessions are done the session closes and the secret file is gone; the release
    log holds the tree, the member's record holds to it (`sessions_tree_reason`), and the draw's coins hold beside the
    tree's lines (`coins_reason`)."""
    manager, draws, opener = RS.local(tmp_path, monkeypatch, [0], trees=True)
    try:
        coins, segment = drawn(draws)
        assert draws.serves_trees() and not draws.closed() and draws.relayed() is None
        address, file = draws.trees()
        assert address.startswith("127.0.0.1:") and Path(file) == tmp_path / "challenger.relay-secret"
        assert stat.S_IMODE(Path(file).stat().st_mode) == 0o600
        wrong = Server(address, "00" * 32)
        assert wrong.hello == {"type": "refused", "reason": "the hello's secret is not the session's"}
        wrong.sock.close()
        server = Server(address, secret_of(file))
        assert server.hello == {"type": "commitment", "commitment": {}}
        record = server.session(NONCE)
        assert server.close() == {"type": "closed"}
        assert draws.relayed() == {"trees": 1, "rounds": 2, "ended": "the server closed it"}
        (host_out,) = tmp_path.glob("a5-challenger-*/var/sessions/*/out")
        assert not draws.closed() and (host_out / R.RELEASES).exists() and not (host_out / R.CLOSE).exists()
        draws.sessions_done(0)
        assert draws.closed() and (host_out / R.CLOSE).exists() and not Path(file).exists()
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert "error" not in published, published.get("error")
    assert published["trusted"]["close"] is None
    assert published["trusted"]["released"] == {AC.FAMILY: 1 + len(segment["openings"])}
    assert CC.coins_reason(AC.evidence(published, segment), RS.ROOT, RS.AFTER, bytes(coins.read), AC.lms_verify) is None
    trees = Ch.coin_trees_of(published["session"]["log"])
    assert list(trees) == [record["coin_tree"]["root"]]
    assert Ch.sessions_tree_reason(trees, [("m0", record)]) is None


def test_a_second_session_on_the_relay_is_refused(tmp_path, monkeypatch):
    """The relay is one server's: `trees` again is refused, and a frame other than a tree's (a coin's open, which would
    read the step's draw coins) is refused with nothing forwarded, ending its connection with the session still open and
    no coin released. A second coin tree the server sampled through it (a second session, or a redrawn one) leaves the
    sessions' records refused: one session's record leaves the other tree unused (`UNUSED_TREE`), and one tree recorded
    for two sessions is shared (`SHARED_TREE`); each session on a tree of its own holds."""
    manager, draws, opener = RS.local(tmp_path, monkeypatch, [0], trees=True)
    try:
        coins, segment = drawn(draws)
        address, file = draws.trees()
        with pytest.raises(Ch.CoinError, match="^the step's session relays one server's coin trees, and its relay has started$"):
            draws.trees()
        server = Server(address, secret_of(file))
        first, second = server.session(NONCE), server.session(OTHER_NONCE)
        index = 1 + len(segment["openings"])
        refused = server.ask({"type": "open", "label": AC.FAMILY, "index": index, "after": RS.AFTER.hex()})
        server.sock.close()
        assert refused == {"type": "refused", "reason": AC.RELAYED}
        assert draws.relayed() == {"trees": 2, "rounds": 4, "ended": f"it refused a frame: {AC.RELAYED}"}
        assert not draws.closed()
        draws.sessions_done(0)
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert "error" not in published, published.get("error")
    assert published["trusted"]["released"] == {AC.FAMILY: index}
    trees = Ch.coin_trees_of(published["session"]["log"])
    assert list(trees) == [first["coin_tree"]["root"], second["coin_tree"]["root"]]
    assert Ch.sessions_tree_reason(trees, [("m0", first)]) == (
        f"the challenger's release log shows 2 coin trees and the sessions use 1: {Ch.UNUSED_TREE}")
    assert Ch.sessions_tree_reason(trees, [("m0", first), ("m1", first)]) == (
        f"session m1's coins are not the challenger's: {Ch.SHARED_TREE}")
    assert Ch.sessions_tree_reason(trees, [("m0", first), ("m1", second)]) is None


def test_a_tie_failing_after_the_sessions_refuses_the_audit_and_writes_its_summary_alone(tmp_path, monkeypatch):
    """a5's tie runs once the member's server has proved through the relay and the session has closed
    (`tie_after_sessions`). On the registration the coins were released after it holds, and each of its negatives is
    refused (`record_negatives`). On a registration of another root it fails by name, and the audit is refused
    (`refuse_after_sessions`): only its summary is written, naming the check, and no audit record, case, release or
    negatives file, so nothing built from the files that registration's roots completed leaves the auditor."""
    rec = {"window": {"kind": "served", "run": RS.ROOT, "row": 0},
           "roots": [{"port": "x", "root": "01" * 64}, {"port": "out", "root": "02" * 64}]}
    moved = {**rec, "roots": [{"port": "x", "root": "03" * 64}, rec["roots"][1]]}
    a, mine = SimpleNamespace(window_root=RS.ROOT), SimpleNamespace(run=RS.ROOT)
    manager, a.coins, opener = RS.local(tmp_path, monkeypatch, [0], trees=True)
    try:
        coins, segment = drawn(a.coins, bytes.fromhex(REG.digest(rec)))
        address, file = a.coins.trees()
        server = Server(address, secret_of(file))
        server.session(NONCE)
        assert server.close() == {"type": "closed"}
        a.coins.sessions_done(0)
        read = bytes(coins.read)
        held, ev = a5.tie_after_sessions(a, rec, mine, segment, read, 0)
        failed, _ = a5.tie_after_sessions(a, moved, mine, segment, read, 0)
        negatives = a5.record_negatives(a, rec, mine, ev, read, 0)
        opener.join()
    finally:
        manager.shutdown()
    assert held == {"ok": True, "codes": [], "detail": {}, "record": "auditor"}
    unopened = "the draw's coins were not opened after the prover's commitment the verifier holds"
    assert negatives == {
        "registration-window-run-not-the-verifier-s (record)": {
            "refused": True, "why": f"the registration's window is run {a5.OTHER_RUN[:16]}…, not the verifier's {RS.ROOT[:16]}…"},
        "registration-of-another-digest (record)": {"refused": True, "why": unopened},
        "window-run-the-auditor-s-ledger-opens-no-session-on (record)": {
            "refused": True, "why": "the auditor's ledger opens no challenger session for this window"}}
    assert failed == {"ok": False, "codes": ["record"], "detail": {"record": unopened}, "record": "auditor"}
    out = tmp_path / "window"
    out.mkdir()
    checks = {"registration": {"ok": True, "codes": []}, "record": failed}
    population = sum(RS.SILU_MUL)
    summary, ok = a5.refuse_after_sessions(out, {"population": population}, checks, {})
    assert ok is False and sorted(p.name for p in out.iterdir()) == ["summary.json"]
    assert json.loads((out / "summary.json").read_text()) == {"population": population, "checks": checks,
                                                               "negatives": {}} == summary


@pytest.mark.skipif(not (os.environ.get("VERITY_ONE_STAGE_TREE_FROM") and os.environ.get("FLOCK_CIRCUIT")),
                    reason="VERITY_ONE_STAGE_TREE_FROM and FLOCK_CIRCUIT (a release build of flock-circuit) unset: a real "
                           "serve --zk --tree-from session on RoPE's two captured heads through the relay, about a minute")
def test_a_zk_servers_coin_tree_through_the_relay_holds_its_record_to_the_steps_session(tmp_path, monkeypatch):
    """``flock-circuit serve --zk --tree-from`` (``FLOCK_CIRCUIT``) on RoPE's two captured heads, through the relay as a5
    hands it to the server (`a0.session`'s ``tree_from``): the prover's session is accepted, the server's one coin tree
    came through the relay and its close ended the relay alone, and once the session has closed its release log holds
    the tree the server's record shows (`sessions_tree_reason`)."""
    m0 = os.pathsep.join(str(AC.TREE / p) for p in (".", "verity/ml/flock/python", "integrations/vllm",
                                                       "benchmarks/numerical/python"))
    st = a0.stage(str(ROPE), [2], tmp_path / "stage", m0)
    manager, draws, opener = RS.local(tmp_path, monkeypatch, [0], trees=True)
    try:
        drawn(draws)
        with socket.create_server(("127.0.0.1", 0)) as s:
            port = s.getsockname()[1]
        record, live = a0.session(os.environ["FLOCK_CIRCUIT"], st, 2, "subset:2", port, tmp_path / "session", zk=True,
                                  tree_from=draws.trees())
        relayed = draws.relayed()
        draws.sessions_done(0)
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert live["accepted"] is True, live
    assert "error" not in published, published.get("error")
    trees = Ch.coin_trees_of(published["session"]["log"])
    assert list(trees) == [record["coin_tree"]["root"]]
    assert relayed == {"trees": 1, "rounds": len(trees[record["coin_tree"]["root"]].opened), "ended": "the server closed it"}
    assert Ch.sessions_tree_reason(trees, [("m0", record)]) is None
