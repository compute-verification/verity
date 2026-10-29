"""The Commit's coins from the auditor host's challenger (`verity_vllm.pipeline.commit_coins`): one
challenge phase a Commit, reserved through the gate before any root, opened on its first run's root of record, and each
run's coins opened after that run's root; the verifier holds them to the one challenge phase the auditor's signed ledger opened
for the Commit's root (`challenger.ledger_head`, `challenger.pinned_session`), run from a commit the host's pins log
records on the verifier's first-parent history of main, ended closed, and to the roots the challenger's signed release log shows it received.

A Commit whose openings come from the challenge phase opened on its own root passes. Refused: openings sampled before their run's
root, a challenge phase opened on another Commit's root, a second challenge phase for the Commit's root, coins held in the Commit's
process, a ledger cut before the challenge phase's end, and a challenge phase read with no pins log or one that doesn't record its
tree, or with no first-parent history of main or one without its pin."""
from __future__ import annotations

import hashlib
import json
import shlex
import socket
import threading
from types import SimpleNamespace

import pytest

from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.core.protocols.sampled_proofs.challenger import CoinError
from verity.ml.crypto import lms
from verity_vllm.commit.challenge import challenge_positions
from verity_vllm.pipeline import commit_coins as CC

from tests.protocols import challenger_host as CH

PRIVATE, KEY = CH.challenger_key(3, H=5)
GATE = "ssh -i KEY -o IdentitiesOnly=yes research@auditor"
PEER = "vy-nebius-2"


def _root(name: str) -> bytes:
    return hashlib.sha512(name.encode()).digest()


class _Auditor:
    """The challenger gate (`challenger_host.Gate`) and the auditor host's challenger in this process: `run` answers the Commit's
    gate requests (`CC.Gate`'s argv, one shell word after the command) as the gate's forced command does, refusing a
    second open for a root; an open starts that challenge phase's challenger, signed under `KEY` at the gate's next leaves, which
    the Commit reaches over a socket pair. `end` appends the challenge phase's end entry once its client closed it, and `heads`
    writes the ledger (or the lines given) where a verifier reads it."""

    def __init__(self, tmp_path, monkeypatch) -> None:
        self.gate, self.tmp, self.argv, self._heads = CH.Gate(PRIVATE, KEY), tmp_path, [], 0
        self.sessions: dict[str, dict] = {}
        self.client = None
        monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(lambda cls, at, secret=None: self.client))

    def run(self, argv: list[str]) -> tuple[int, str]:
        self.argv.append(argv)
        words = shlex.split(argv[-1])
        opts = dict(zip(words[1::2], words[2::2]))
        if words[0] == "reserve":
            sid = f"s20261021T1200{len(self.sessions):02d}Z-{len(self.sessions):04x}"
            families = json.loads(opts["--families"])
            e = self.gate.reserve(sid, families, scheme=opts["--scheme"], peer=opts["--peer"])
            self.sessions[sid] = {"families": families, "scheme": opts["--scheme"]}
            return 0, json.dumps({"session": sid, "seq": e["seq"], "peer": opts["--peer"], "by": "vllm",
                                  "expires_utc": e["body"]["expires_utc"], "period": None})
        if words[0] == "open":
            sid, root = opts["--session"], opts["--root"]
            if any(s.get("root") == root for s in self.sessions.values()):
                return 3, json.dumps({"refused": f"root {root[:16]} has a session"})
            answer = self.gate.open(sid, root)
            self._serve(sid, root)
            return 0, json.dumps(answer)
        if words[0] == "status":
            return 0, json.dumps({"session": words[1], "commitment": self.sessions[words[1]]["commitment"]})
        return 3, json.dumps({"refused": f"no verb {words[0]}"})

    def _serve(self, sid: str, root: str) -> None:
        s = self.sessions[sid]
        challenger = LC.LiveChallenger(s["scheme"], 0, [tuple(f) for f in s["families"]],
                                       CH.entropy(len(self.sessions) + 40))
        directory = self.tmp / sid / "out"
        directory.mkdir(parents=True)
        (directory / R.HELLOS).write_text("")
        leaves = [self.gate.take()]

        def sign(m: bytes) -> bytes:
            return lms.sign_at(PRIVATE, leaves.pop() if leaves else self.gate.take(), m).encode()
        session = R.open_session(challenger, KEY, f"{CH.HOST}.{sid}", sign, directory, sid)
        ours, theirs = socket.socketpair()

        def serve() -> None:
            with ours:
                s["reason"] = R.serve(ours, session, iter(range(1 << 20)).__next__)
        s.update(root=root, directory=directory, commitment=challenger.commitment.root.hex(),
                 thread=threading.Thread(target=serve))
        s["thread"].start()
        self.client = R.RemoteChallenger(theirs)

    def end(self, sid: str) -> None:
        s = self.sessions[sid]
        s["thread"].join()
        assert s["reason"] is None
        self.gate.end(sid, s["directory"])

    def heads(self, lines=None):
        self._heads += 1
        path = self.tmp / f"windows-{self._heads}.jsonl"
        path.write_text("".join(ln + "\n" for ln in (self.gate.lines if lines is None else lines)))
        return [path]


def _options(**kw):
    return SimpleNamespace(**{"challenger_gate": GATE, "challenger_peer": PEER, "replay_challenge": None, **kw})


def _commit(auditor: _Auditor, names: list[str]) -> tuple[list[dict], list[bytes], str]:
    """A Commit of one run per name, each run's root that name's, its coins from its own challenge phase: the runs' coins
    records, their roots and the challenge phase's id, the challenge phase closed and ended in the ledger."""
    coins = CC.of_options(_options(), len(names), gate_run=auditor.run)
    records, roots = [], []
    for name in names:
        rc = SimpleNamespace(run_root=_root(name), extra={})
        kw = CC.positions(rc, coins)
        assert kw["legacy"] is False and len(challenge_positions(rc.run_root, [8, 8], 4, b"0", **kw)) == 4
        assert CC.identities(rc)["source"] == bytes.fromhex(rc.extra["coins"]["keys"][CC.IDENTITIES])
        chal, rec = CC.replay_challenge(rc, None)
        assert rec == {"source": CC.CHALLENGER, "family": CC.REPLAY, "index": len(records)}
        assert chal == rc.extra["coins"]["keys"][CC.REPLAY]
        records.append(rc.extra["coins"])
        roots.append(rc.run_root)
    assert coins.close() is None
    sid = coins.reserved["session"]
    auditor.end(sid)
    return records, roots, sid


def _reason(auditor, records, roots, sid, heads=None, pins=CH.PINS, main=CH.MAIN):
    published = None if pins is None else [CH.write_pins(auditor.tmp / "pins.jsonl", pins)]
    return CC.reason(records, roots, heads or auditor.heads(), auditor.sessions[sid]["directory"], KEY, pins=published,
                     main=main)


def test_a_commit_whose_openings_come_from_a_session_opened_on_its_own_root_passes(tmp_path, monkeypatch):
    """The challenge phase is reserved through the gate's command before any root, opened on run 0's root of record, and each
    run's coins are its coin of every family, opened after its own root: the verifier finds them the challenger's."""
    auditor = _Auditor(tmp_path, monkeypatch)
    records, roots, sid = _commit(auditor, ["run 0", "run 1"])
    assert _reason(auditor, records, roots, sid) is None
    assert _reason(auditor, records, roots, sid, pins=None) == (
        f"the Commit's challenger session is not trusted: {LC.NO_PINS}")
    other = LC.pin_line("ab" * 20, "ab" * 20, "2026-10-20T10:00:00Z")
    assert _reason(auditor, records, roots, sid, pins=[other]).startswith(
        "the Commit's challenger session is not trusted: the host's pins log is not the one the open of session "
        f"{sid} sealed")
    assert _reason(auditor, records, roots, sid, main=None) == (
        f"the Commit's challenger session is not trusted: {LC.NO_MAIN}")
    assert _reason(auditor, records, roots, sid, main=CH.MAIN[:1]) == (
        f"the Commit's challenger session is not trusted: {LC.first_parent_reason(sid, CH.PINNED, list(CH.MAIN[:1]))}")
    verbs = [shlex.split(a[-1])[0] for a in auditor.argv]
    assert verbs == ["reserve", "open", "status"] and all(a[:-1] == shlex.split(GATE) for a in auditor.argv)
    assert json.loads(shlex.split(auditor.argv[0][-1])[6]) == CC.families(2)
    assert shlex.split(auditor.argv[1][-1])[4] == roots[0].hex()
    assert [r["index"] for r in records] == [0, 1] and all(r["session"]["session"] == sid for r in records)
    assert "secret" not in records[0]["session"]


def test_openings_drawn_before_their_runs_root_are_refused(tmp_path, monkeypatch):
    """Run 1's coins opened after run 0's root (before run 1's existed), then recorded against run 1's: the release log
    shows the challenger never received run 1's root."""
    auditor = _Auditor(tmp_path, monkeypatch)
    coins = CC.of_options(_options(), 2, gate_run=auditor.run)
    early = [coins.draw(_root("run 0")), coins.draw(_root("run 0"))]
    early[1]["root"] = _root("run 1").hex()
    coins.close()
    auditor.end(coins.reserved["session"])
    why = _reason(auditor, early, [_root("run 0"), _root("run 1")], coins.reserved["session"])
    assert why is not None and "a coin drawn before its run's root" in why


def test_a_session_opened_on_another_commits_root_is_refused(tmp_path, monkeypatch):
    """Commit B carrying Commit A's challenge phase, or A's coins held to B's roots: the ledger opens B's root for B's challenge phase
    only, and A's challenge phase for A's root."""
    auditor = _Auditor(tmp_path, monkeypatch)
    rec_a, roots_a, sid_a = _commit(auditor, ["A 0"])
    rec_b, roots_b, sid_b = _commit(auditor, ["B 0"])
    assert _reason(auditor, rec_b, roots_b, sid_b) is None
    borrowed = [{**rec_b[0], "session": rec_a[0]["session"]}]
    assert _reason(auditor, borrowed, roots_b, sid_b).startswith(f"run 0's coins are session {sid_a}'s")
    assert _reason(auditor, rec_a, roots_b, sid_b).startswith(f"run 0's coins are session {sid_a}'s, opened for root "
                                                              f"{roots_a[0].hex()[:16]}")
    (tmp_path / "alone").mkdir()
    alone = _Auditor(tmp_path / "alone", monkeypatch)
    rec_c, _roots_c, sid_c = _commit(alone, ["C 0"])
    assert _reason(alone, rec_c, roots_b, sid_c) == ("the Commit's challenger session is not trusted: the auditor's "
                                                     "ledger opens no challenger session for this window")


def test_a_second_session_for_the_commits_root_is_refused(tmp_path, monkeypatch):
    """The gate refuses a second open for a root, and so does the verifier, whichever challenge phase the Commit carries, when a
    ledger shows one."""
    auditor = _Auditor(tmp_path, monkeypatch)
    records, roots, sid = _commit(auditor, ["run 0"])
    again = CC.of_options(_options(), 1, gate_run=auditor.run)
    with pytest.raises(CoinError, match="the challenger gate refused open .*has a session"):
        again.draw(roots[0])
    auditor.gate.open(again.reserved["session"], roots[0].hex())
    why = _reason(auditor, records, roots, sid)
    assert why.startswith("the Commit's challenger session is not trusted: the auditor's ledger is refused: ") and (
        "a second open for root" in why)


def test_coins_held_in_the_commits_process_credit_nothing(tmp_path, monkeypatch):
    """Without a gate the run's coins are this process's own bytes: the Commit runs, as a measurement, and the verifier
    refuses its coins by name; ``--replay-challenge`` then names the C2 replay's challenge."""
    auditor = _Auditor(tmp_path, monkeypatch)
    records, roots, sid = _commit(auditor, ["run 0"])
    assert CC.of_options(_options(challenger_gate=None, challenger_peer=None), 1) is None
    rc = SimpleNamespace(run_root=roots[0], extra={})
    kw = CC.positions(rc, None)
    assert rc.extra["coins"]["source"] == CC.IN_PROCESS and kw["source"] == bytes.fromhex(rc.extra["coins"]["keys"][CC.POSITIONS])
    assert CC.replay_challenge(rc, "abcd") == ("abcd", {"source": "flag"})
    assert CC.replay_challenge(rc, None)[1] == {"source": CC.IN_PROCESS, "family": CC.REPLAY, "index": None}
    assert _reason(auditor, [rc.extra["coins"]], roots, sid) == (
        "the Commit's coins were drawn in its own process, not by the auditor host's challenger: a measurement, which "
        "credits nothing")
    assert _reason(auditor, [*records, rc.extra["coins"]], [*roots, roots[0]], sid).startswith(
        "the Commit's coins were drawn in its own process")


def test_a_ledger_cut_before_the_sessions_end_is_refused(tmp_path, monkeypatch):
    """The ledger as published before the challenge phase's end entry (or a cut head beside the full one that doesn't extend
    it) seals none of its files: refused."""
    auditor = _Auditor(tmp_path, monkeypatch)
    records, roots, sid = _commit(auditor, ["run 0", "run 1"])
    cut = auditor.heads(auditor.gate.lines[:-1])
    why = _reason(auditor, records, roots, sid, heads=cut)
    assert why.startswith(f"the Commit's challenger session is not trusted: the auditor's ledger has no end for session {sid}")
    assert _reason(auditor, records, roots, sid, heads=[*cut, *auditor.heads()]) is None
    forked = [*auditor.gate.lines[:-1], auditor.gate.lines[-1].replace('"rc":0', '"rc":2')]
    assert "not trusted" in _reason(auditor, records, roots, sid, heads=[*auditor.heads(forked), *auditor.heads()])


def test_the_commits_options_and_draws_refuse_by_name(tmp_path, monkeypatch):
    """The gate and the peer go together, the gate is one source of the C2 replay's challenge, a Commit with no
    instrumented run reserves nothing, a vllm-v1 root draws no coins, the challenge phase flips one coin of each family a run
    and no more, a run's samples read only coins flipped after its own root, and a gate refusal names itself. The gate's
    ``ledger`` reads as its lines."""
    for kw, match in (({"challenger_peer": None}, "go together"), ({"challenger_gate": None}, "go together"),
                      ({"replay_challenge": "ab"}, "two sources")):
        with pytest.raises(SystemExit, match=match):
            CC.of_options(_options(**kw), 1)
    with pytest.raises(SystemExit, match="not without one"):
        CC.of_options(_options(), 0, gate_run=lambda argv: (0, "{}"))
    with pytest.raises(SystemExit, match="--challenger-gate: the challenger gate refused reserve \\(exit 3\\): busy"):
        CC.of_options(_options(), 1, gate_run=lambda argv: (3, json.dumps({"refused": "busy"})))
    with pytest.raises(CoinError, match="refused status \\(exit 255\\): ssh: connect"):
        CC.Gate(GATE, lambda argv: (255, "ssh: connect to host auditor")).call("status", "s")
    lines, asked = ['{"kind":"period-open"}', '{"kind":"reserve"}'], []
    assert CC.Gate(GATE, lambda argv: asked.append(argv) or (0, "".join(f"{x}\n" for x in lines))).ledger() == lines
    assert asked[0][-1] == "ledger"
    with pytest.raises(CoinError, match="refused ledger \\(exit 3\\)"):
        CC.Gate(GATE, lambda argv: (3, json.dumps({"refused": "no such verb"}))).ledger()
    auditor = _Auditor(tmp_path, monkeypatch)
    coins = CC.of_options(_options(), 1, gate_run=auditor.run)
    with pytest.raises(CoinError, match="not this run's 32-byte root \\(vllm-v1\\)"):
        coins.draw(hashlib.sha256(b"v1").digest())
    coins.draw(_root("run 0"))
    with pytest.raises(CoinError, match="holds coins for 1 runs, and each is drawn"):
        coins.draw(_root("run 1"))
    coins.close()
    rc = SimpleNamespace(run_root=_root("run 1"), extra={"coins": {"root": _root("run 0").hex(), "keys": {}}})
    with pytest.raises(ValueError, match="no coins drawn after it"):
        CC.positions(rc)
