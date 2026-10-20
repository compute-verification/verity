"""RT4-INT's findings on H1's coins from the auditor host's challenger (#2940 @ 79c0a054b1, VLLM-55), from the witnesses at
367b7e552b, each now the refusal its fix intends; RT4-INT-23 is infra's (a signed close), sealed under its ledger.

* RT4-INT-17: a window is held to the one session infra's challenger gate opened for its Commit's root, in the gate's
  signed v2 ledger (`pouw_h1_challenger.session_reason`, `Ledger`), ended closed and complete under the coin commitment
  the gate's answer gave it, which the serving side checked before its first coin; its tree is read through the gate's
  ``fetch`` and sealed by its end entry. Two sessions for one root, a reused session, a key off the ledger's pin, leaves
  out of order, a record leaf outside its session, a two-step open off its reserve and two published heads that don't
  extend one another are refused, and so is a tampered byte of infra's own v2 head. A period (`period_reason`) is its
  lane's windows between its signed period-open and period-close: one with no close, an incomplete one, an accepted
  window outside it and a window of it abandoned after its coins are refused.
* RT4-INT-19: the window's draw coin opens once, after a root over the manifest, the last receipts and every round's root
  and coin; an early draw, a draw from another family and a session closed with no draw are refused.
* RT4-INT-20: round s's root commits the A, y and z tables of its calls and round s - 1's receipts.
* RT4-INT-21: a window whose coins were held in the serving process is no session's and credits nothing; pouw-serve and
  the Commit's C2 replay challenge take the auditor host's challenger.
* RT4-INT-22: a window admits in round 0 only unless its ``admission`` is ``per-round``.
* RT4-INT-23: the release log, the only evidence that a round's coin followed its root, is an unsigned file; roots
  chosen after every coin pass once the log is rewritten to name them. Infra's end entry now seals it (a rewritten log
  is refused under the ledger, `test_a_window_is_held_to_the_one_session...`); read without a ledger it stays a witness.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as LC  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import LiveCommitment, record_message  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity_vllm.pipeline import pouw_serve  # noqa: E402
from verity_vllm.pipeline import replay_coin as RCN  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_challenger as PHC  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402

from tests.protocol_options import challenger_host as CH  # noqa: E402
from tests.protocol_options import test_pouw_h1 as T  # noqa: E402

PRIVATE, KEY = CH.challenger_key(1, H=5)
COMMIT_A, COMMIT_B = "ca" * 64, "cb" * 64
#: infra's v2 head (art:fbe4bcb48b, 31 entries: v1 opens, one-step v2 opens, then periods infra-smoke-20261020a to d,
#: c's and d's windows two-step) and seven of its sessions' trees as the gate's ``fetch`` gave them; registered
#: (fixtures/artifacts.json), fetched by `research data fetch-fixtures`
SMOKE = Path(__file__).parent / "fixtures" / "challenger_periods"
#: the SHA-512 of the challenger key infra's gate signed tonight's ledger with (its 09:55Z note); a verifier reads its
#: pin from config, never from code
SMOKE_PIN = ("27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3a"
             "e065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc")


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (T.Linear,))


class _Session:
    """One session `name` of the auditor's challenger under its one key `KEY`, its record signed at LMS leaf `leaf` for
    the gate's host ``vy-challenger-1.<name>``, with its own coins (`seed`), reached at `T.AUDITOR` until the next
    session is made, each hello's secret kept in `secrets`; its tree (the gate's ``fetch``) is `tree`, `directory` its
    ``out/`` and `commitment` its coin commitment's root, which `T.SESSION` names while it is reached."""

    def __init__(self, tmp_path, monkeypatch, name: str, leaf: int, seed: int, challenger=None) -> None:
        self.tree, self.directory = tmp_path / name, tmp_path / name / "out"
        self.challenger = challenger or Wd.live_challenger(0, 3, CH.entropy(seed))
        self.directory.mkdir(parents=True)
        R.open_session(self.challenger, KEY, f"{CH.HOST}.{name}", lambda m: lms.sign_at(PRIVATE, leaf, m).encode(),
                       self.directory)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._serve, args=(ours,))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)
        self.commitment, self.secrets = self.challenger.commitment.root.hex(), []
        monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(lambda cls, at, secret=None: self._reach(at, secret)))
        monkeypatch.setitem(T.SESSION, "commitment", self.commitment)

    def _reach(self, at, secret):
        self.secrets.append(secret)
        return self.client if at == T.AUDITOR else None

    def _serve(self, sock: socket.socket) -> None:
        with sock:
            self.reason = R.serve(sock, self.challenger, self.directory, iter(range(1 << 20)).__next__)

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason


def _serve(settings, seed: int = 9, prompt: int = 5, run: bool = False):
    """`test_pouw_h1._serve_window`'s two served steps of a 3-step window, its activations drawn from `seed` and request
    a's one prompt token `prompt`: the commitment (and the run, with `run`)."""
    model, runner = T.Model(T.Linear(32, 1), T.Linear(40, 2)), settings["pouw"]["runner"]
    comp = T.PO.install(model, T.PO.Context(stage="serve", enabled=(T._entry(),), settings=settings))
    try:
        g = torch.Generator().manual_seed(seed)
        for so, rows in ((T._so(("a", [prompt], 4)), (1, 5)),
                         (T._so(("b", [7], 2), ("c", list(range(8, 39)), 3), running={"a": 1}), (33, 2))):
            runner.execute_model(so)
            for m, layer in zip(rows, model.layers):
                layer.quant_method.apply(layer, (torch.randn(m, T.K, generator=g) * 3).bfloat16())
        h1 = pouw.h1_run(T._entry())
    finally:
        comp.close()
    return (comp.installed[0].commitment(), h1) if run else comp.installed[0].commitment()


def _coins(commitment):
    return [r["coin"]["coin"] for r in commitment["rounds"]]


def _window(gate, tmp_path, monkeypatch, name: str, seed: int, commit: str = COMMIT_A):
    """A window served in session `name`, which `gate` opened for `commit` (its answer's commitment, ``status
    SESSION``'s, once the session's record is signed) and ended over its ``out/``: its commitment and its session."""
    reply = gate.open(name, commit, Wd.live_families(3))
    session = _Session(tmp_path, monkeypatch, name, gate.take(), seed)
    served = _serve(T._remote(session={**reply, "commitment": session.commitment}))
    assert session.ended() is None and session.secrets == [bytes.fromhex(reply["secret"])]
    gate.end(name, session.directory)
    return served, session


def _gate():
    return CH.Gate(PRIVATE, KEY)


def test_a_window_is_held_to_the_one_session_the_auditors_ledger_opened_for_its_commits_root(tmp_path, monkeypatch):
    """RT4-INT-17: two sessions under the auditor's one key both pass `trusted_rounds_reason`, but the window carries the
    gate's answer and the ledger opened one session for its Commit's root: another session's window claiming it is
    refused, and so are an answer's seq off the ledger's, a root the ledger has no session for, a window with no session
    or another Commit, weights other than the period's registered ones, a session tree off its end entry's seals
    (another session's, a rewritten release log, a missing tree), a key or a host other than the open entry pins, a
    session that never ended, ended unclosed or short of its coins, an end entry under another commitment than the
    window's, and a record whose commitment, leaf or release counts aren't the ones its end entry records."""
    gate = _gate()
    first, a = _window(gate, tmp_path, monkeypatch, "s-a", 11)
    second, b = _window(gate, tmp_path, monkeypatch, "s-b", 12, commit=COMMIT_B)
    assert PHC.trusted_rounds_reason(first, a.directory, KEY) is None
    assert PHC.trusted_rounds_reason(second, b.directory, KEY) is None
    assert _coins(first) != _coins(second) and first["session"]["session"] == "s-a" and first["session"]["seq"] == 0
    assert "secret" not in first["session"] and first["session"]["commitment"] == a.commitment
    ledger, fetch = gate.ledger(), {"s-a": a.tree, "s-b": b.tree}.get
    assert PHC.session_reason(first, ledger, fetch) is None and PHC.session_reason(second, ledger, fetch) is None
    claims_a = {**second, "session": {**second["session"], "root": COMMIT_A}}
    assert PHC.session_reason(claims_a, ledger, fetch) == (
        f"the window's session s-b is not s-a, the one the auditor's ledger opened for Commit {COMMIT_A}")
    assert PHC.session_reason({**first, "session": {**first["session"], "seq": 3}}, ledger, fetch) == (
        "the window's session s-a is ledger entry 3 in its reply, but entry 0 opened it")
    assert PHC.session_reason({**first, "session": {**first["session"], "root": "cc" * 64}}, ledger, fetch) == (
        f"the auditor's ledger has no session for Commit {'cc' * 64}, so the window's coins are no session's of record")
    assert PHC.session_reason({k: v for k, v in first.items() if k != "session"}, ledger, fetch).startswith(
        "the window carries no challenger session")
    assert PHC.session_reason({**first, "session": {"id": "s-a"}}, ledger, fetch).startswith(
        "the window's challenger session is malformed")
    assert PHC.session_reason(first, ledger, fetch, commit=COMMIT_B) == (
        f"the window's session is for Commit {COMMIT_A}, not the window's Commit {COMMIT_B}")
    assert PHC.session_reason(first, ledger, fetch, registered=bytes(64)) == (
        "the window's weights' root is not the one the period registered")
    assert PHC.session_reason(first, ledger, fetch, registered=bytes.fromhex(first["weights_root"])) is None
    assert PHC.session_reason({**first, "session": {**first["session"], "commitment": b.commitment}}, ledger, fetch) == (
        f"session s-a's end entry records coin commitment {a.commitment[:16]}..., not {b.commitment[:16]}..., the "
        "window's (the gate's answer): a refused window")
    assert PHC.session_reason(first, ledger, {"s-a": b.tree}.get) == (
        "session s-a's evidence is not trusted: session s-a's record.json is not the one its end entry sealed")
    assert PHC.session_reason(first, ledger, {}.get) == (
        "session s-a's evidence is not trusted: session s-a has no tree on the auditor's side (the gate's fetch)")
    rewritten = tmp_path / "rewritten"
    shutil.copytree(a.tree, rewritten)
    (rewritten / "out" / R.RELEASES).write_text((a.directory / R.RELEASES).read_text().replace(
        first["rounds"][0]["root"], "ee" * 64))
    assert PHC.session_reason(first, ledger, {"s-a": rewritten}.get) == (
        "session s-a's evidence is not trusted: session s-a's releases.jsonl is not the one its end entry sealed")

    def ended_over(out, closed=True, end=True, **override):
        g = _gate()
        g.open("s-a", COMMIT_A, Wd.live_families(3))
        if end:
            g.end("s-a", out, closed, **override)
        return g.ledger()
    assert PHC.session_reason({**first, "session": {**first["session"], "commitment": b.commitment}},
                              ended_over(b.directory), {"s-a": b.tree}.get) == (
        "session s-a's evidence is not trusted: session s-a's challenger record is signed for host vy-challenger-1.s-b, "
        "not vy-challenger-1.s-a, the one its open entry names")
    _p, other = CH.challenger_key(2)
    keyed = tmp_path / "keyed"
    shutil.copytree(a.tree, keyed)
    (keyed / "out" / R.KEY).write_text(json.dumps(other.to_json()))
    assert PHC.session_reason(first, ended_over(keyed / "out"), {"s-a": keyed}.get) == (
        "session s-a's evidence is not trusted: session s-a's key is not the verifier's challenger key, the one its open "
        "entry pins")
    assert PHC.session_reason(first, ended_over(a.directory, end=False), fetch) == (
        "session s-a never ended (the auditor's ledger has its open entry and no end): a refused window")
    assert PHC.session_reason(first, ended_over(a.directory, closed=False), fetch) == (
        "session s-a ended without its client closing it (serve.py exited 1): a refused window")
    short = {Wd.DRAW: 1, Wd.ROUND: 2, Wd.SALT: 1}
    assert PHC.session_reason(first, ended_over(a.directory, complete=False, released=short), fetch) == (
        f"session s-a ended without releasing every coin of its families (released {short}): a refused window")
    assert PHC.session_reason(first, ended_over(a.directory, released=short), fetch) == (
        f"session s-a's evidence is not trusted: session s-a's release log shows {{'{Wd.SALT}': 1, '{Wd.DRAW}': 1, "
        f"'{Wd.ROUND}': 3}} coins released, not {short}, the count its end entry records")
    assert PHC.session_reason({**first, "session": {**first["session"], "commitment": "dd" * 64}},
                              ended_over(a.directory, commitment="dd" * 64), fetch) == (
        f"session s-a's evidence is not trusted: session s-a's record commits to coins {a.commitment[:16]}..., not "
        f"{'dd' * 8}..., the commitment its end entry records")
    assert PHC.session_reason(first, ended_over(a.directory, record_leaf=3), fetch) == (
        "session s-a's evidence is not trusted: session s-a's record is signed at LMS leaf 1, not 3, the leaf its end "
        "entry records")


def test_the_ledger_refuses_two_sessions_for_one_root_a_reused_session_another_key_and_a_broken_chain():
    """RT4-INT-17: `Ledger.verify` (infra's v2 ``verify_ledger``) refuses a whole ledger with a second session for one
    root, one session for two roots, an open of no root or under another key's pin, an end with no open or a second end,
    an entry signed under another key, edited after signing or at a leaf not after the last one's, a v1 entry after a v2,
    a record leaf outside its session's open and end, used twice or also an entry's, a line dropped, and an entry of
    another kind; v1 entries before v2 ones are one chain. `Ledger.newest` reads the longest of every published head and
    refuses two that don't extend one another."""
    families = Wd.live_families(3)

    def refused(build, match):
        g = _gate()
        build(g)
        with pytest.raises(LC.CoinError, match=match):
            g.ledger()
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_A, families)),
            r"entry 1 opens a second session \(s-b, after s-a\) for root (ca)+: one window, one session")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-a", COMMIT_B, families)),
            "entry 1 opens session s-a for a second window")
    refused(lambda g: g.open("s-a", COMMIT_A.upper(), families), "entry 0 opens session s-a for no window")
    refused(lambda g: g.open("s-a", COMMIT_A, families, key_sha512="00" * 64),
            r"entry 0 \(open of s-a\) is under the key pin 0+, not the verifier's")
    refused(lambda g: g.append("end", "s-a", {"files": {}}), "entry 0 ends session s-a, which has no open entry")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.append("end", "s-a", {}), g.append("end", "s-a", {})),
            "entry 2 ends session s-a, which has no open entry or has ended")
    refused(lambda g: g.append("close", "s-a", {}), "entry 0 is of kind 'close', not reserve, open, end, period-open")
    refused(lambda g: (g.open("s-a", COMMIT_A, families, leaf=5), g.open("s-b", COMMIT_B, families, leaf=3)),
            "entry 1 is signed at LMS leaf 3, not after the last entry's 5: a key state restored or forked")
    refused(lambda g: (g.open("s-a", COMMIT_A, families, leaf=5), g.open("s-b", COMMIT_B, families, leaf=5)),
            "entry 1 is signed at LMS leaf 5, not after the last entry's 5")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_B, families, v1=True)),
            "entry 1 is not entry 1 of a verity/challenger-auditor/v2 ledger chained to its predecessor")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.append("end", "s-a", {"record_leaf": 0})),
            "entry 1 ends session s-a with its record signed at leaf 0, not between its open's and its end's leaves, once")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.append("end", "s-a", {"record_leaf": "2"}, leaf=3)),
            "entry 1 ends session s-a with its record signed at leaf '2'")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_B, families),
                       g.append("end", "s-a", {"record_leaf": 2}, leaf=3), g.append("end", "s-b", {"record_leaf": 2})),
            "entry 3 ends session s-b with its record signed at leaf 2")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_B, families),
                       g.append("end", "s-a", {"record_leaf": 1})),
            "entry 2's record leaf 1 is also an entry's leaf")
    g = _gate()
    g.open("s-0", "c0" * 64, families, v1=True)
    g.open("s-a", COMMIT_A, families)
    g.open("s-b", COMMIT_B, families)
    assert set(g.ledger().windows) == {"c0" * 64, COMMIT_A, COMMIT_B}
    assert g.ledger().window(COMMIT_A)["open"]["session"] == "s-a"
    with pytest.raises(LC.CoinError, match="entry 0's signature doesn't verify"):
        PHC.Ledger.verify(g.lines, CH.challenger_key(2)[1])
    with pytest.raises(LC.CoinError, match="entry 1's signature doesn't verify"):
        PHC.Ledger.verify([g.lines[0], g.lines[1].replace(b'"port":7811', b'"port":7812'), g.lines[2]], KEY)
    with pytest.raises(LC.CoinError, match="entry 0 is not entry 0 of a verity/challenger-auditor/v2 ledger"):
        PHC.Ledger.verify(g.lines[1:], KEY)
    with pytest.raises(LC.CoinError, match="entry 0 is malformed"):
        PHC.Ledger.verify([b"not json"], KEY)
    assert set(PHC.Ledger.newest([g.lines[:1], g.lines, g.lines[:2]], KEY).windows) == {"c0" * 64, COMMIT_A, COMMIT_B}
    fork = _gate()
    fork.lines = g.lines[:2]
    fork._leaf = 2
    fork.open("s-c", "cc" * 64, families)
    assert set(PHC.Ledger.newest([fork.lines], KEY).windows) == {"c0" * 64, COMMIT_A, "cc" * 64}
    with pytest.raises(LC.CoinError, match="the published ledger heads of 3 and 3 entries don't extend one another"):
        PHC.Ledger.newest([g.lines, fork.lines], KEY)
    with pytest.raises(LC.CoinError, match="entry 1's signature doesn't verify"):
        PHC.Ledger.newest([g.lines, [g.lines[0], g.lines[1].replace(b'"port":7811', b'"port":7812')]], KEY)
    with pytest.raises(LC.CoinError, match="no published challenger ledger head"):
        PHC.Ledger.newest([], KEY)


def test_a_period_is_its_lanes_windows_between_its_signed_open_and_close_and_credits_only_if_complete(tmp_path,
                                                                                                    monkeypatch):
    """RT4-INT-17: a period is its lane's windows between its period-open and period-close entries (a window of another
    lane's isn't one); it credits only when the ledger closes it complete and every window of it was accepted. Refused: a
    period the ledger doesn't have or doesn't close, one closed with no window, an incomplete window (one that ended
    short of its coins, or never ended), an accepted window that isn't the period's, and a window of it abandoned after
    its coins."""
    gate = _gate()
    gate.period_open("p0")
    gate.period_close("p0")
    gate.period_open("p1")
    kept, a = _window(gate, tmp_path, monkeypatch, "s-a", 11)
    _dropped, b = _window(gate, tmp_path, monkeypatch, "s-b", 12, commit=COMMIT_B)
    gate.open("s-k", "ce" * 64, Wd.live_families(3), by="kernels")
    gate.period_close("p1")
    gate.period_open("p2")
    gate.open("s-i", "cc" * 64, Wd.live_families(3))
    idle = _Session(tmp_path, monkeypatch, "s-i", gate.take(), seed=13)
    idle.client.close()
    assert idle.ended() is None
    gate.end("s-i", idle.directory)
    gate.period_close("p2")
    gate.period_open("p3")
    gate.open("s-n", "cd" * 64, Wd.live_families(3))
    gate.period_close("p3")
    gate.period_open("p4")
    gate.open("s-o", "cf" * 64, Wd.live_families(3))
    ledger = gate.ledger()
    assert {n: p["windows"] for n, p in ledger.periods.items()} == {
        "p0": [], "p1": ["s-a", "s-b"], "p2": ["s-i"], "p3": ["s-n"], "p4": ["s-o"]}
    assert ledger.periods["p1"]["close"]["body"]["complete"] is True and ledger.periods["p4"]["close"] is None
    assert PHC.session_reason(kept, ledger, {"s-a": a.tree}.get) is None
    assert PHC.period_reason(ledger, "p1", ["s-a", "s-b"]) == []
    assert PHC.period_reason(ledger, "p1", ["s-a"]) == [
        f"period p1's window s-b (Commit {COMMIT_B}) ended complete and has no accepted window: a window abandoned after "
        "its coins"]
    assert PHC.period_reason(ledger, "p1", ["s-a", "s-b", "s-k"]) == [
        "session s-k, an accepted window, is not one of period p1's windows (['s-a', 's-b'], lane vllm's between its "
        "period-open and its period-close)"]
    released = ledger.window("cc" * 64)["end"]["body"]["released"]
    assert set(released.values()) == {0} and PHC.period_reason(ledger, "p2", ["s-i"]) == [
        f"period p2's window s-i is incomplete (Commit {'cc' * 64}, ended 'closed by the client', released {released}): a "
        "refused window refuses its period"]
    assert PHC.period_reason(ledger, "p3", []) == [
        f"period p3's window s-n is incomplete (Commit {'cd' * 64}, never ended): a refused window refuses its period"]
    assert PHC.period_reason(ledger, "p0", []) == ["period p0 closed with no window"]
    assert PHC.period_reason(ledger, "p4", ["s-o"]) == [
        "period p4 has no period-close in the auditor's ledger, so its windows aren't all known: an open period credits "
        "nothing"]
    assert PHC.period_reason(ledger, "p5", []) == ["the auditor's ledger has no period p5"]


def test_the_ledger_refuses_periods_and_reserves_off_the_gates_rules():
    """RT4-INT-17 (infra's ``verify_periods`` and two-step windows): `Ledger.verify` refuses a period entry with a
    session, a period-open or a reserve under another key's pin, a period off its name and lane, a period name used
    twice, a lane's second open period, a cap or circuit off form, a window past its cap, a close of a period that isn't
    its lane's open one or that doesn't list exactly its windows and their outcomes; a reserve that isn't first in its
    session or whose expiry isn't its start plus its minutes; and an open that binds no reserve of its session, doesn't
    carry its request, isn't by its prover, falls outside its minutes or after its period's close, or is one-step after
    the first reserve. A two-step window joins its prover's (``for``'s) period at its reserve, and a reserve never opened
    is listed with no root."""
    families, circuit = Wd.live_families(3), "c1" * 64
    s, t = "s20261020T110800Z-000a", "s20261020T110900Z-000b"

    def refused(build, match):
        g = _gate()
        build(g)
        with pytest.raises(LC.CoinError, match=match):
            g.ledger()
    refused(lambda g: g.append("period-open", "s-a", {"period": "p", "lane": "vllm", "key_sha512": g.pin}),
            "entry 0 is malformed .*a period's none")
    refused(lambda g: g.period_open("p", key_sha512="00" * 64), r"entry 0 \(period-open of p\) is under the key pin 0+")
    refused(lambda g: g.period_open("p/1"), "entry 0 is a period-open that doesn't name its period and its lane")
    refused(lambda g: (g.period_open("p"), g.period_close("p"), g.period_open("p")),
            "entry 2 opens period p a second time: a period's name is used once")
    refused(lambda g: (g.period_open("p"), g.period_open("q")), "entry 1 opens period q while lane vllm's period p is open")
    refused(lambda g: g.append("period-close", None, {"period": "p", "lane": "vllm", "windows": [], "complete": False}),
            "entry 0 closes period p, which isn't lane vllm's open period")
    refused(lambda g: (g.period_open("p"), g.period_close("p", lane="kernels")),
            "entry 1 closes period p, which isn't lane kernels's open period")
    refused(lambda g: (g.period_open("p"), g.open("s-a", COMMIT_A, families), g.period_close("p", windows=[])),
            "entry 2 closes period p without listing exactly its windows and their outcomes")
    refused(lambda g: (g.period_open("p"), g.period_close("p", complete=True)), "entry 1 closes period p without listing")
    for cap, c in ((0, circuit), (2, circuit.upper()), (2, None), (True, circuit)):
        refused(lambda g: g.period_open("p", cap=cap, circuit=c),
                "entry 0: period p's cap is a positive integer and its circuit a SHA-512 in lowercase")
    refused(lambda g: (g.period_open("p", cap=1, circuit=circuit), g.open("s-a", COMMIT_A, families),
                       g.open("s-b", COMMIT_B, families)), "entry 2 puts window 1 in period p, past its cap 1")
    refused(lambda g: g.reserve("s-a", families), "entry 0 reserves session s-a, which is not a v2 entry first in its")
    refused(lambda g: (g.reserve(s, families), g.reserve(s, families)), f"entry 1 reserves session {s}, which is not")
    refused(lambda g: (g.open(s, COMMIT_A, families), g.reserve(s, families)), f"entry 1 reserves session {s}, which is")
    refused(lambda g: g.reserve(s, families, expires_utc="2026-10-20T11:09:00Z"),
            f"entry 0 reserves session {s} without its prover, its minutes, its start and its expiry")
    refused(lambda g: g.reserve(s, families, minutes=0), f"entry 0 reserves session {s} without its prover")
    refused(lambda g: g.reserve(s, families, key_sha512="00" * 64), rf"entry 0 \(reserve of {s}\) is under the key pin")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, reserve=5)),
            f"entry 1 opens session {s} binding no reserve of it")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, peer="vy-nebius-2")),
            f"entry 1 opens session {s} without its reserve's request")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, by="kernels")),
            f"entry 1 opens session {s} by 'kernels', not its reserve's prover")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, started_utc="2026-10-20T11:18:01Z")),
            f"entry 1 opens session {s} outside its reserve's minutes")
    refused(lambda g: (g.period_open("p"), g.reserve(s, families), g.period_close("p"), g.open(s, COMMIT_A)),
            f"entry 3 opens session {s} after its period p closed")
    refused(lambda g: (g.reserve(s, families), g.open(t, COMMIT_B, families)),
            f"entry 1 opens session {t} without a reserve, after the ledger's first reserve")
    g = _gate()
    g.open("s-0", COMMIT_A, families)
    g.period_open("p", cap=2, circuit=circuit)
    g.reserve(s, families)
    g.reserve(t, families, by="vy-admin", for_="vllm")
    reply = g.open(t, COMMIT_B, started_utc="2026-10-20T11:17:59Z")
    g.append("end", t, {"complete": False})
    g.period_close("p")
    ledger = g.ledger()
    assert reply["reserve"] == 3 and reply["for"] == "vllm"
    assert PHC.session_of({**reply, "commitment": COMMIT_A})["opened_by"] == "vllm"
    assert ledger.periods["p"]["windows"] == [s, t] and (ledger.periods["p"]["cap"], ledger.periods["p"]["circuit"]) == (
        2, circuit)
    assert ledger.periods["p"]["close"]["body"]["windows"] == [{"root": None, "session": s, "complete": False},
                                                               {"root": COMMIT_B, "session": t, "complete": False}]
    assert ledger.window(COMMIT_B)["reserve"]["seq"] == 3 and ledger.window(COMMIT_A)["reserve"] is None
    assert PHC.period_reason(ledger, "p", []) == [
        f"period p's window {s} is incomplete (reserved and never opened): a refused window refuses its period",
        f"period p's window {t} is incomplete (Commit {COMMIT_B}, ended None, released None): a refused window refuses "
        "its period"]


def test_the_session_setting_is_the_gates_open_answer_for_the_window():
    """RT4-INT-17's serving half: the window carries the gate's open answer (session, seq, root and coin commitment),
    without the hello's secret; any other form is refused."""
    reply = {"session": "s20261020T110845Z-1355", "seq": 8, "root": COMMIT_A, "commitment": COMMIT_B}
    full = {**reply, "host": "vy-challenger-1.s", "port": 7811, "peer": "vy-nebius-1", "by": "vy-admin", "for": "vllm",
            "reserve": 7, "opened_by": "vllm", "secret": "5e" * 32}
    assert PHC.session_of(full)["seq"] == 8 and PHC.carried(full) == {k: v for k, v in full.items() if k != "secret"}
    for bad in ({**reply, "session": ""}, {**reply, "seq": True}, {**reply, "seq": -1}, {**reply, "root": "ab"},
                {**reply, "root": COMMIT_A.upper()}, {**reply, "commitment": None}, {**reply, "secret": "ab"},
                {**reply, "secret": "5E" * 32}, {**reply, "period": "7"}, {k: v for k, v in reply.items() if k != "seq"},
                {"session": "s", "seq": 0, "period": "7", "commit": COMMIT_A}):
        with pytest.raises(ValueError, match="pouw setting 'session'"):
            PHC.session_of(bad)


def test_a_challenger_off_the_answers_commitment_is_refused_before_its_first_coin(tmp_path, monkeypatch):
    """RT4-INT-17 (and RT4-INT-29's secret): the serving side's hello carries the answer's secret, and a challenger whose
    coin commitment isn't the one the gate's answer gives the session (another process's on that port) is closed and
    refused before its salt coin; the session it reached released nothing."""
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=19)
    other = {**T.SESSION, "commitment": "dd" * 64, "secret": "5e" * 32}
    with pytest.raises(ValueError, match=f"commits to coins {host.commitment[:16]}..., not {'dd' * 8}..., the commitment "
                                         "the gate's answer gives session"):
        _serve(T._remote(session=other))
    assert host.ended() is None and host.secrets == [bytes.fromhex("5e" * 32)]
    assert R.released_roots(host.directory / R.RELEASES, host.challenger.commitment) == {
        label: () for label, _n in host.challenger.commitment.families}


def test_infras_v2_head_and_sessions_bind_under_the_gates_key():
    """RT4-INT-17 on the gate's own bytes (vy-challenger-1's v2 head, art:fbe4bcb48b, and seven sessions as its fetch gave
    them): the head verifies under the key its sessions' key.json holds and tonight's pin, v1 entries then v2, and binds a
    window by its root; session s20261020T110845Z-1355 ended closed and complete under the commitment its end entry
    records, every file of its out/ is the one its end entry sealed, and its record is signed for the open entry's host
    and layout at the end's record leaf; s20261020T105232Z-1281 (a hello without the secret), s20261020T105206Z-9738
    (before ends recorded their commitment) and s20261020T114752Z-9e52 (a salt after a root not its window's) are refused
    windows. Its periods: infra-smoke-20261020a and b closed complete, each with one window, whose hellos.jsonl its end
    seals; c's two-step windows bind their reserves, one complete and one killed (refused); d's one window was reserved
    and never opened (refused). One edited byte of the head, or a published head with an entry cut, is refused."""
    sid = "s20261020T110845Z-1355"
    root = ("34aaf35c44afbbeaf9919a1228b0405c67fa2218a25b6c578087719ba8e89b5587c7d3f3d705a490bac294499691f1af5b0450fc3c7b6c"
            "14c3d8c90584456a9a")
    commitment = ("d573a082258c1cbad6c03486976124d1626b82154c9c97d3add9e3fdbe6f15cd709add3e2968cb9059a1ee371e7ee0ff6a4677"
                  "e344321622df1c2f17200be868")
    assert (SMOKE / sid / "out" / R.KEY).is_file(), (
        f"{SMOKE} is not fetched: `uv run research data fetch-fixtures "
        "integrations/vllm/tests/protocol_options/fixtures/challenger_periods`")
    key = PHC.pinned_key(SMOKE / sid / "out" / R.KEY, SMOKE_PIN)
    assert (key.W, key.H) == (1, 20)
    head = PHC.lines_of(SMOKE / "ledger" / "windows.jsonl")
    ledger = PHC.Ledger.newest([head[:4], head, head[:9], head[:23]], key)
    assert len(head) == 31 and len(ledger.windows) == 10
    assert [json.loads(line)["schema"] for line in head] == [PHC.LEDGER_V1] * 4 + [PHC.LEDGER_SCHEMA] * 27
    window = ledger.window(root)
    assert window["open"]["session"] == sid and window["open"]["seq"] == 8
    assert window["open"]["body"]["host"] == f"vy-challenger-1.{sid}" and window["end"]["body"]["complete"] is True
    fetch = lambda s: SMOKE / s if (SMOKE / s).is_dir() else None  # noqa: E731
    given = {"session": sid, "seq": 8, "root": root, "commitment": commitment}
    assert PHC.held(given, ledger, root) == (None, window)
    out = PHC.evidence(window, fetch, key)
    trusted, received = R.trusted_session(out, key, PHC.lms_verify)
    assert trusted.root.hex() == commitment and trusted.scheme == "verity/smoke/coins/v1" and len(received["replay"]) == 2
    smoke = {"coin_source": "challenger", "session": given}
    assert PHC.session_reason(smoke, ledger, fetch).startswith("the window's round_coins are not a live coin commitment")
    unclosed = json.loads(head[6])
    assert PHC.held({**given, "session": unclosed["session"], "seq": 6, "root": PHC.root_of(unclosed)}, ledger)[0] == (
        f"session {unclosed['session']} ended without its client closing it (a client's first frame is hello with the "
        "session's secret): a refused window")
    early = json.loads(head[4])
    assert PHC.held({**given, "session": early["session"], "seq": 4, "root": PHC.root_of(early)}, ledger)[0].startswith(
        f"session {early['session']}'s end entry records coin commitment None..., not")

    def answer(seq):
        opened, end = json.loads(head[seq]), json.loads(head[seq + 1])
        return {"session": opened["session"], "seq": seq, "root": PHC.root_of(opened),
                "commitment": end["body"]["commitment"]}
    salted = answer(10)
    assert PHC.held(salted, ledger)[0].startswith(
        f"session {salted['session']} ended without its client closing it (a pouw/salt coin opens only after the "
        f"window's Commit root {salted['root'][:16]}…")
    periods = {n: (p["windows"], p["close"]["body"]["complete"]) for n, p in ledger.periods.items()}
    a, b, c, d = (f"infra-smoke-20261020{x}" for x in "abcd")
    w4080, w306d, w6e92, wb89c = answer(13), answer(17), answer(25), answer(23)
    assert periods == {a: ([w4080["session"]], True), b: ([w306d["session"]], True),
                       c: ([w6e92["session"], wb89c["session"]], False), d: (["s20261020T123859Z-90aa"], False)}
    assert ledger.periods[b]["cap"] == 1 and ledger.periods[a]["cap"] is None
    for given in (w4080, w306d, wb89c):
        why, w = PHC.held(given, ledger)
        assert why is None and "hellos.jsonl" in w["end"]["body"]["files"]
        assert PHC.evidence(w, fetch, key) == SMOKE / given["session"] / "out"
    assert w4080["commitment"].startswith("43a09815f5424466")
    assert ledger.window(wb89c["root"])["reserve"]["seq"] == 22 and ledger.window(w4080["root"])["reserve"] is None
    assert PHC.period_reason(ledger, a, [w4080["session"]]) == [] and PHC.period_reason(ledger, b, [w306d["session"]]) == []
    assert PHC.period_reason(ledger, a, []) == [
        f"period {a}'s window {w4080['session']} (Commit {w4080['root']}) ended complete and has no accepted window: a "
        "window abandoned after its coins"]
    assert PHC.period_reason(ledger, a, [w306d["session"]])[0].startswith(
        f"session {w306d['session']}, an accepted window, is not one of period {a}'s windows")
    assert PHC.period_reason(ledger, c, [wb89c["session"]]) == [
        f"period {c}'s window {w6e92['session']} is incomplete (Commit {w6e92['root']}, ended 'serve.py exited -15', "
        "released {'pouw/round': 0, 'pouw/salt': 0}): a refused window refuses its period"]
    assert PHC.period_reason(ledger, d, []) == [
        f"period {d}'s window s20261020T123859Z-90aa is incomplete (reserved and never opened): a refused window refuses "
        "its period"]
    with pytest.raises(LC.CoinError, match="entry 9's signature doesn't verify"):
        PHC.Ledger.verify(head[:9] + [head[9].replace(b'"replay":2', b'"replay":3')], key)
    with pytest.raises(LC.CoinError, match="entry 30's signature doesn't verify"):
        PHC.Ledger.verify(head[:30] + [head[30].replace(b'"complete":false', b'"complete":true')], key)
    with pytest.raises(LC.CoinError, match="entry 8 is not entry 8 of a"):
        PHC.Ledger.newest([head, head[:8] + [head[9]]], key)
    with pytest.raises(LC.CoinError, match="not the auditor's pinned"):
        PHC.pinned_key(SMOKE / sid / "out" / R.KEY, "00" * 64)


def test_an_early_draw_is_refused_and_the_windows_own_draw_fails_visibly(tmp_path, monkeypatch):
    """RT4-INT-19: a draw coin opened after a root of the prover's, before the window's first round, is the challenger's one
    draw coin; the window's own draw at `finish` is then refused by the challenger (it has no second draw coin, and a
    retry never redraws), the window records the refusal, and the trusted check refuses the early draw."""
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=13)
    probe = hashlib.sha512(b"a root of the prover's, before any round").digest()
    draw = host.client.open(Wd.DRAW, 0, probe)
    commitment = _serve(T._remote())
    nothing = f"the next {Wd.DRAW} coin is 1, not 0"
    assert nothing in host.ended()
    assert nothing in commitment["draw"]["refused"] and commitment["draw"]["coin"] is None
    received = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
    assert received[Wd.DRAW] == (probe,) and draw.after == probe
    assert PHC.trusted_rounds_reason(commitment, host.directory, KEY) == (
        "the challenger's log does not show the window's draw root as the one root its draw coin followed: a draw opened "
        "before the window's rounds, or after another root")


def test_a_draw_from_another_family_off_its_root_or_never_opened_is_refused(tmp_path, monkeypatch):
    """RT4-INT-19: the honest window's draw passes; its draw coin swapped for a round coin, a draw root not over the
    window's parts, and a session closed at `finish` with no draw opened are refused."""
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=14)
    honest = _serve(T._remote())
    assert host.ended() is None
    assert PHC.trusted_rounds_reason(honest, host.directory, KEY) is None
    assert honest["draw"]["root"] == RQ.draw_root(bytes.fromhex(honest["calls_root"]), bytes.fromhex(
        honest["draw"]["receipts"]), honest["rounds"]).hex()
    swapped = {**honest, "draw": {**honest["draw"], "coin": honest["rounds"][0]["coin"]}}
    assert PHC.trusted_rounds_reason(swapped, host.directory, KEY) == (
        f"the window's draw coin: coin is the epoch's {Wd.ROUND} coin 0, not its {Wd.DRAW} coin 0")
    bent = {**honest, "draw": {**honest["draw"], "receipts": "ee" * 64}}
    assert PHC.trusted_rounds_reason(bent, host.directory, KEY) == (
        "the window's draw root is not over its manifest, its last round's receipts and every round's root and coin")
    quiet = _Session(tmp_path, monkeypatch, "quiet", leaf=1, seed=15)
    weights = bytes.fromhex(honest["weights_root"])
    salt = quiet.client.open(Wd.SALT, 0, weights)
    rounds = [quiet.client.open(Wd.ROUND, s, bytes.fromhex(r["root"])) for s, r in enumerate(honest["rounds"])]
    quiet.client.close()
    assert quiet.ended() is None
    c = quiet.challenger.commitment.to_json()
    closed = {**honest, "coins": c, "round_coins": c, "salt_opening": salt.to_json(),
              "rounds": [{**r, "coin": o.to_json()} for r, o in zip(honest["rounds"], rounds)]}
    assert PHC.trusted_rounds_reason(closed, quiet.directory, KEY) == (
        "the challenger's session closed with no draw coin opened, so the window has no draw of the challenger's")


def test_round_roots_commit_the_windows_a_y_and_z_and_the_receipts_before_them(monkeypatch):
    """RT4-INT-20: round s's root, the one round s's coin follows, changes with the A the window serves (its A, y and z
    tables are in the step's manifest rows), is the same for the same A, and changes with round s - 1's receipts."""
    monkeypatch.setattr(PH.os, "urandom", lambda n: bytes(n))
    served = [_serve(T._settings(), seed) for seed in (9, 10, 9)]
    roots = [[r["root"] for r in c["rounds"]] for c in served]
    assert roots[0] == roots[2] and len(roots[0]) == 3
    assert roots[0][0] != roots[1][0] and roots[0][1] != roots[1][1] and roots[0][2] == roots[1][2]
    after = [[r["coin"]["after"] for r in c["rounds"]] for c in served]
    assert after == roots and after[0][:2] != after[1][:2]
    other_request = [r["root"] for r in _serve(T._settings(), 9, prompt=6)["rounds"]]
    assert other_request[0] != roots[0][0]

    def window(receipt):
        rng = np.random.default_rng(12)
        torch.manual_seed(12)
        run = PH.H1Run("t", T.SALT, {0: "q", 1: "o"}, [(32, T.K), (40, T.K)], 32, 2, rounds=T._rounds(2), key=T.KEY)
        run.record(0, *T._tap(rng, 3, 32))
        with pytest.raises(RuntimeError, match="call 0's receipt before round 0's coin"):
            run.receipt(0, bytes(64))
        run.record(1, *T._tap(rng, 3, 40))
        if receipt:
            run.receipt(0, b"\1" * 64)
            with pytest.raises(RuntimeError, match="or a second one"):
                run.receipt(0, b"\1" * 64)
            with pytest.raises(ValueError, match="64 bytes"):
                run.receipt(1, b"\1" * 32)
        run.record(0, *T._tap(rng, 3, 32))
        run.record(1, *T._tap(rng, 3, 40))
        with pytest.raises(RuntimeError, match="after the root that commits round 0's receipts"):
            run.receipt(1, b"\2" * 64)
        run.receipt(3, b"\3" * 64)
        run.finish()
        return run
    plain, signed = window(False), window(True)
    assert plain.round_log[0]["root"] == signed.round_log[0]["root"]
    assert plain.round_log[1]["root"] != signed.round_log[1]["root"]
    assert signed.round_log[1]["receipts"] == signed.receipts_of(0).hex() != RQ.NO_RECEIPTS.hex()
    assert signed.commits["draw"]["receipts"] == signed.receipts_of(1).hex()
    with pytest.raises(RuntimeError, match="after the root that commits round 1's receipts"):
        signed.receipt(2, b"\4" * 64)
    row = signed.manifest()[1]
    assert set(row["tables"]) == {"a", "y", "z"} and row["tables"]["z"] is None and row["tables"]["y"] is None
    assert signed.manifest()[-1]["tables"]["a"] != row["tables"]["a"]


def test_coins_held_in_the_serving_process_are_no_session_and_credit_nothing(tmp_path, monkeypatch):
    """RT4-INT-21 (and the witness at 367b7e552b: the round rule of record passes in-process coins): a window whose coins
    were held in the serving process passes `rounds_reason`, but no session binds it and it credits nothing."""
    local, run = _serve(T._settings(), run=True)
    assert local["coin_source"] == "in-process"
    assert RQ.rounds_reason(local["rounds"], LiveCommitment.from_json(local["round_coins"])) is None
    assert PHC.session_reason(local, _gate().ledger(), {}.get).startswith(
        "the window's coins were held in the serving process")
    assert run.credit()["coins"] == "in-process" and run.credit()["credited"] == {"rows": 0, "macs": 0}
    auditor = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=14)
    auditor.client.close()
    assert auditor.ended() is None
    assert "coins from a challenger the verifier doesn't hold" in PHC.trusted_rounds_reason(local, auditor.directory, KEY)


def test_pouw_serve_and_ncp_v2_take_the_auditor_hosts_challenger():
    """RT4-INT-21: pouw-serve's ``--challenger`` and ``--session`` are the pouw settings ncp-v2's salt comes from, one
    never without the other; ncp-v2 accepts the setting and ncp-v1 doesn't."""
    opts = pouw_serve.Options(workload="w.json", challenger=T.AUDITOR, session=json.dumps(T.SESSION))
    assert pouw_serve.protocols_config(opts) == {"pouw": {"challenger": T.AUDITOR, "session": T.SESSION, "draws": 16}}
    assert "coins" in pouw_serve.protocols_config(pouw_serve.Options(workload="w.json"))["pouw"]
    with pytest.raises(ValueError, match="--challenger needs --session"):
        pouw_serve.protocols_config(pouw_serve.Options(workload="w.json", challenger=T.AUDITOR))
    with pytest.raises(ValueError, match="--session is the session"):
        pouw_serve.protocols_config(pouw_serve.Options(workload="w.json", session=json.dumps(T.SESSION)))

    def refused(entry, settings):
        return T.PO.refusals(T.PO.Context(stage="serve", enabled=(entry,), settings=settings))
    remote = {"pouw": {"challenger": T.AUDITOR, "session": T.SESSION}}
    assert not any("challenger" in r or "session" in r for r in refused("pouw:ncp-v2", remote))
    assert any("needs 'session'" in r for r in refused("pouw:ncp-v2", {"pouw": {"challenger": T.AUDITOR}}))
    assert any("'challenger' is eps1-sm120-h1fx1 and ncp-v2's" in r for r in refused("pouw:ncp-v1", remote))


class _NcpLinear:
    def __init__(self, weight):
        self.weight, self.quant_method = weight, SimpleNamespace(apply=lambda layer, x, bias=None: None)


def test_ncp_v2s_salt_is_the_auditor_hosts_salt_coin_after_the_weights_root(tmp_path, monkeypatch):
    """RT4-INT-21: ncp-v2 under the setting ``challenger`` takes its epoch's salt from the auditor host's salt coin, opened
    after the weights' root, carries its session, and closes the session at release; pouw-serve parses the flags."""
    from verity.core.protocols.pouw.audit import salt_of

    from verity_vllm.pipeline import cli
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=18)
    monkeypatch.setattr(pouw, "linear_types", lambda: (_NcpLinear,))
    w = (torch.randn(32, 64, generator=torch.Generator().manual_seed(3))).bfloat16().float()
    model = SimpleNamespace(modules=lambda: iter([layer]), named_modules=lambda: iter([("up", layer)]))
    layer = _NcpLinear(w)
    ctx = T.PO.Context(stage="serve", enabled=("pouw:ncp-v2",), settings={"pouw": {"challenger": T.AUDITOR,
                                                                                   "session": T.SESSION}})
    with T.PO.install(model, ctx) as comp:
        c = comp.get("pouw").commitment()
    assert host.ended() is None and c["coin_source"] == PH.CHALLENGER and c["session"] == T.SESSION
    received = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
    weights = bytes.fromhex(c["weights_root"])
    assert received[Wd.SALT] == (weights,)
    opening = LC.LiveOpening.from_json(c["salt_opening"])
    assert c["salt"] == salt_of(host.challenger.commitment, opening.coin, weights).hex()
    parsed = cli.parse("pouw-serve", ["--workload", "w.json", "--challenger", T.AUDITOR, "--session", json.dumps(T.SESSION)])
    assert pouw_serve.protocols_config(parsed)["pouw"]["session"] == T.SESSION


def test_the_commits_replay_challenge_is_the_auditor_hosts_replay_coin_after_the_run_root(tmp_path, monkeypatch):
    """RT4-INT-21: with ``--replay-challenger`` the C2 replay's challenge is the challenger's replay coin 0, opened after
    the run root, and the verifier holds it to the commitment the challenger signed; another run root, a challenge of
    this process's or of the flag, and a challenger of another layout are refused. RT4-INT-17: the Commit's record
    carries the gate's answer (``--replay-session``) without its secret, which the hello carried, held to the session
    the ledger opened for its Commit's root, ended closed and complete under the answer's commitment; a challenger off
    that commitment is refused before the coin."""
    run_root = hashlib.sha256(b"a run root").digest()
    gate = _gate()
    opened = gate.open("s-c", COMMIT_A, [(RCN.FAMILY, 1)], scheme=RCN.SCHEME)
    with pytest.raises(ValueError, match="pouw setting 'session'"):
        RCN.challenge(T.AUDITOR, run_root, {"id": "s-c"})
    other = _Session(tmp_path, monkeypatch, "other", leaf=30, seed=21,
                     challenger=LC.LiveChallenger(RCN.SCHEME, 0, [(RCN.FAMILY, 1)], CH.entropy(21)))
    with pytest.raises(ValueError, match="commits to coins .*, not dddd"):
        RCN.challenge(T.AUDITOR, run_root, {**opened, "commitment": "dd" * 64})
    assert other.ended() is None
    host = _Session(tmp_path, monkeypatch, "s-c", leaf=gate.take(), seed=16,
                    challenger=LC.LiveChallenger(RCN.SCHEME, 0, [(RCN.FAMILY, 1)], CH.entropy(16)))
    reply = {**opened, "commitment": host.commitment}
    chal, rec = RCN.challenge(T.AUDITOR, run_root, reply)
    assert host.ended() is None and rec["source"] == "challenger" and rec["session"] == PHC.carried(reply)
    assert host.secrets == [bytes.fromhex(reply["secret"])] and "secret" not in rec["session"]
    gate.end("s-c", host.directory)
    assert RCN.session_reason({**rec, "session": reply}, chal, run_root, gate.ledger(), {"s-c": host.tree}.get) is None
    ledger, fetch = gate.ledger(), {"s-c": host.tree}.get
    assert RCN.session_reason(rec, chal, run_root, ledger, fetch, commit=COMMIT_A) is None
    assert RCN.session_reason({**rec, "session": {**reply, "seq": 4}}, chal, run_root, ledger, fetch) == (
        "the replay challenge: the window's session s-c is ledger entry 4 in its reply, but entry 0 opened it")
    assert RCN.session_reason(rec, chal, run_root, ledger, fetch, commit=COMMIT_B).startswith(
        f"the replay challenge: the window's session is for Commit {COMMIT_A}")
    assert RCN.session_reason(rec, chal, run_root, ledger, {}.get).startswith(
        "the replay challenge's session s-c is not trusted: session s-c has no tree")
    assert RCN.session_reason(rec, "00" * 64, run_root, ledger, fetch) == "the replay challenge is not the replay coin"
    trusted, received = R.trusted_session(host.directory, KEY, PHC.lms_verify)
    assert received[RCN.FAMILY] == (RCN.after(run_root),)
    assert RCN.reason(rec, chal, run_root, trusted) is None
    assert RCN.reason(rec, chal, hashlib.sha256(b"another").digest(), trusted) == (
        "the replay coin: coin was not opened after what the verifier recorded")
    assert RCN.reason(rec, "00" * 64, run_root, trusted) == "the replay challenge is not the replay coin"
    for given, source in ((None, "in-process"), ("abcd", "flag")):
        c, r = RCN.local(given, "drawn")
        assert r == {"source": source} and RCN.reason(r, c, run_root, trusted).endswith("not the auditor host's challenger")
    _Session(tmp_path, monkeypatch, "window", leaf=1, seed=17)
    with pytest.raises(ValueError, match=f"not {RCN.SCHEME} with a replay family"):
        RCN.challenge(T.AUDITOR, run_root)
    short = _gate()
    opened = short.open("s-d", COMMIT_B, [(RCN.FAMILY, 2)], scheme=RCN.SCHEME)
    two = _Session(tmp_path, monkeypatch, "s-d", leaf=short.take(), seed=20,
                   challenger=LC.LiveChallenger(RCN.SCHEME, 0, [(RCN.FAMILY, 2)], CH.entropy(20)))
    chal, rec = RCN.challenge(T.AUDITOR, run_root, {**opened, "commitment": two.commitment})
    assert two.ended() is None
    short.end("s-d", two.directory)
    assert RCN.session_reason(rec, chal, run_root, short.ledger(), {"s-d": two.tree}.get) == (
        f"the replay challenge: session s-d ended without releasing every coin of its families (released "
        f"{{'{RCN.FAMILY}': 1}}): a refused window")


def test_a_window_admits_in_round_0_only_unless_its_admission_is_per_round():
    """RT4-INT-22: by default (#2722's live tiles are fixed before the first coin) a request admitted after round 0 gets
    no entry, its rows are a gap and the window credits nothing; the verifier refuses an entry in a later round under
    ``round-0``, and accepts it under ``per-round``; an unknown admission is refused."""
    once, run = _serve(T._settings(admission=RQ.ROUND0), run=True)
    assert once["admission"] == RQ.ROUND0 and [e.round for e in run.requests] == [0]
    assert run.refused_requests == [{"round": 1, "reason": RQ.late_admission(1)}] * 2
    assert run.request_gaps() == ["step 1: 1 row of a request with no entry", "step 1: 31 rows of a request with no entry"]
    assert run.credit()["credited"] == {"rows": 0, "macs": 0}
    _c, every = _serve(T._settings(), run=True)
    coins = every.rounds.challenger.commitment
    e = every.requests[1]
    args = (every.round_log, coins, every.salt, 64, e, every.open_request(1), every.open_first(1))
    assert e.round == 1 and RQ.request_reason(*args, admission=RQ.PER_ROUND) is None
    assert RQ.request_reason(*args) == f"request 1 was {RQ.late_admission(1)}"
    owner = every.open_slot(1, 1)
    assert RQ.slot_reason(every.round_log, coins, every.salt, 64, [m for m in every.manifest() if m["step"] == 1], owner,
                          e, every.open_request(1), every.open_first(1)) == f"request 1 was {RQ.late_admission(1)}"
    with pytest.raises(ValueError, match="pouw setting 'admission'"):
        pouw.Settings.of({"admission": "late"})
    with pytest.raises(ValueError, match="an admission is one of"):
        PH.H1Run("t", T.SALT, {0: "q"}, [(32, T.K)], 32, 1, admission="late")


def test_the_order_of_roots_and_coins_is_an_unsigned_file_and_a_rewrite_of_it_passes(tmp_path, monkeypatch):
    """RT4-INT-23 (infra's: a signed close), still a witness: roots chosen after every coin, with a draw opened after a
    root over them, are refused against the honest log and pass once the log is rewritten to name them."""
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=15)
    weights = hashlib.sha512(b"the weights' root").digest()
    salt = host.client.open(Wd.SALT, 0, weights)
    early = [hashlib.sha512(b"before any round, %d" % s).digest() for s in range(3)]
    coins = [host.client.open(Wd.ROUND, s, early[s]) for s in range(3)]
    late = [hashlib.sha512(b"a root chosen after every coin, %d" % s + c.coin).digest() for s, c in enumerate(coins)]
    log = [{"round": s, "root": late[s].hex(), "coin": {**coins[s].to_json(), "after": late[s].hex()}} for s in range(3)]
    calls_root = hashlib.sha512(b"the manifest").digest()
    draw_root = RQ.draw_root(calls_root, RQ.NO_RECEIPTS, log)
    draw = host.client.open(Wd.DRAW, 0, draw_root)
    host.client.close()
    assert host.ended() is None
    c = host.challenger.commitment.to_json()
    commits = {"coins": c, "round_coins": c, "weights_root": weights.hex(), "salt_opening": salt.to_json(), "steps": 3,
               "rounds": log, "calls_root": calls_root.hex(),
               "draw": {"root": draw_root.hex(), "receipts": RQ.NO_RECEIPTS.hex(), "coin": draw.to_json()}}
    assert PHC.trusted_rounds_reason(commits, host.directory, KEY) == (
        "the window's round roots are not the roots the challenger's log shows it received, in order")
    path = host.directory / R.RELEASES
    text = path.read_text()
    for e, x in zip(early, late):
        text = text.replace(e.hex(), x.hex())
    path.write_text(text)
    assert PHC.trusted_rounds_reason(commits, host.directory, KEY) is None


def test_the_challengers_record_is_tied_to_its_session_by_the_gates_host_string_and_end_entry(tmp_path, monkeypatch):
    """RT4-INT-17: the challenger's signed record names its host and coin commitment, not the session or Commit; the
    gate's host string ``vy-challenger-1.<session>`` ties it to its session's open entry, and the v2 end entry records
    its commitment's root, the leaf it was signed at (between the open's and the end's) and the coins each family
    released (none here, so not complete)."""
    gate = _gate()
    reply = gate.open("s-a", COMMIT_A, Wd.live_families(3))
    a = _Session(tmp_path, monkeypatch, "s-a", leaf=gate.take(), seed=11)
    a.client.close()
    assert a.ended() is None
    record = json.loads((a.directory / R.RECORD).read_text())
    assert sorted(record) == ["commitment", "host", "signature"] and record["host"] == reply["host"]
    c = LiveCommitment.from_json(record["commitment"])
    assert PHC.lms_verify(KEY, record_message(KEY.public[:16], record["host"], c), bytes.fromhex(record["signature"]))
    end = gate.end("s-a", a.directory)["body"]
    assert reply["commitment"] is None and end["commitment"] == c.root.hex() == a.commitment
    assert end["record_leaf"] == PHC.leaf_of(record["signature"]) == 1 and end["complete"] is False
    assert end["released"] == {label: 0 for label, _n in c.families}
    assert gate.ledger().window(COMMIT_A)["end"]["body"] == end
