"""RT4-INT's findings on H1's coins from the auditor host's challenger (#2940 @ 79c0a054b1, VLLM-55), from the witnesses at
367b7e552b, each now the refusal its fix intends; RT4-INT-23 is infra's (a signed close), sealed under its ledger.

* RT4-INT-17: a window is held to the one session infra's challenger gate opened for its period and Commit, in the
  gate's signed ledger (`pouw_h1_challenger.session_reason`, `Ledger`), its tree read through the gate's ``fetch`` and
  sealed by its end entry; two sessions for one window, a reused session, a key off the ledger's pin and a session
  abandoned after its coins (`period_reason`) are refused, and so is a tampered byte of infra's own smoke ledger.
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
#: the one-time keys of `PRIVATE` a test's gate signs its ledger with, apart from those its sessions' records take
GATE_LEAVES = range(8, 32)
COMMIT_A, COMMIT_B = "ca" * 64, "cb" * 64
SMOKE = Path(__file__).parent / "fixtures" / "challenger_smoke"
#: the SHA-512 of the challenger key infra's gate signed tonight's smoke ledger with (its 09:55Z note); a verifier reads
#: its pin from config, never from code
SMOKE_PIN = ("27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3a"
             "e065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc")


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (T.Linear,))


class _Session:
    """One session `name` of the auditor's challenger under its one key `KEY`, its record signed at LMS leaf `leaf` for
    the gate's host ``vy-challenger-1.<name>``, with its own coins (`seed`), reached at `T.AUDITOR` until the next
    session is made; its tree (the gate's ``fetch``) is `tree`, and `directory` its ``out/``."""

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
        monkeypatch.setattr(R.RemoteChallenger, "connect",
                            classmethod(lambda cls, at: self.client if at == T.AUDITOR else None))

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


def _window(gate, tmp_path, monkeypatch, name: str, leaf: int, seed: int, commit: str = COMMIT_A, period: str = "7"):
    """A window served in session `name`, which `gate` opened for (`period`, `commit`) and ended over its ``out/``: its
    commitment and its session."""
    reply = gate.open(name, period, commit, Wd.live_families(3))
    session = _Session(tmp_path, monkeypatch, name, leaf, seed)
    served = _serve(T._remote(session=reply))
    assert session.ended() is None
    gate.end(name, session.directory)
    return served, session


def _gate():
    return CH.Gate(PRIVATE, KEY, GATE_LEAVES)


def test_a_window_is_held_to_the_one_session_the_auditors_ledger_opened_for_its_period_and_commit(tmp_path, monkeypatch):
    """RT4-INT-17: two sessions under the auditor's one key both pass `trusted_rounds_reason`, but the window carries the
    gate's reply and the ledger opened one session for its period and Commit: another session's window claiming it is
    refused, and so are a reply's seq off the ledger's, a window with no session or another Commit, weights other than the
    period's registered ones, a session tree off its end entry's seals (another session's, a rewritten release log, a
    missing tree), a key or a host other than the open entry pins, and a session that never ended or ended unclosed."""
    gate = _gate()
    first, a = _window(gate, tmp_path, monkeypatch, "s-a", 0, 11)
    second, b = _window(gate, tmp_path, monkeypatch, "s-b", 1, 12, commit=COMMIT_B)
    assert PHC.trusted_rounds_reason(first, a.directory, KEY) is None
    assert PHC.trusted_rounds_reason(second, b.directory, KEY) is None
    assert _coins(first) != _coins(second) and first["session"]["session"] == "s-a" and first["session"]["seq"] == 0
    ledger, fetch = gate.ledger(), {"s-a": a.tree, "s-b": b.tree}.get
    assert PHC.session_reason(first, ledger, fetch) is None and PHC.session_reason(second, ledger, fetch) is None
    claims_a = {**second, "session": {**second["session"], "commit": COMMIT_A}}
    assert PHC.session_reason(claims_a, ledger, fetch) == (
        f"the window's session s-b is not s-a, the one the auditor's ledger opened for Commit {COMMIT_A} in period 7")
    assert PHC.session_reason({**first, "session": {**first["session"], "seq": 3}}, ledger, fetch) == (
        "the window's session s-a is ledger entry 3 in its reply, but entry 0 opened it")
    assert PHC.session_reason({**first, "session": {**first["session"], "period": "8"}}, ledger, fetch) == (
        f"the auditor's ledger has no session for Commit {COMMIT_A} in period 8, so the window's coins are no session's of "
        "record")
    assert PHC.session_reason({k: v for k, v in first.items() if k != "session"}, ledger, fetch).startswith(
        "the window carries no challenger session")
    assert PHC.session_reason({**first, "session": {"id": "s-a"}}, ledger, fetch).startswith(
        "the window's challenger session is malformed")
    assert PHC.session_reason(first, ledger, fetch, commit=COMMIT_B) == (
        f"the window's session is for Commit {COMMIT_A}, not the window's Commit {COMMIT_B}")
    assert PHC.session_reason(first, ledger, fetch, registered=bytes(64)) == (
        "the window's weights' root is not the one the period registered")
    assert PHC.session_reason(first, ledger, fetch, registered=bytes.fromhex(first["weights_root"])) is None
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

    def ended_over(out, closed=True, end=True):
        g = _gate()
        g.open("s-a", "7", COMMIT_A, Wd.live_families(3))
        if end:
            g.end("s-a", out, closed)
        return g.ledger()
    assert PHC.session_reason(first, ended_over(b.directory), {"s-a": b.tree}.get) == (
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


def test_the_ledger_refuses_two_sessions_for_one_window_a_reused_session_another_key_and_a_broken_chain():
    """RT4-INT-17: `Ledger.verify` (infra's ``verify_ledger``) refuses a whole ledger with a second session for one window,
    one session for two windows, an open under another key's pin, an end with no open or a second end, an entry signed
    under another key or edited after signing, a line dropped, and an entry of another kind."""
    families = Wd.live_families(3)

    def refused(build, match):
        g = _gate()
        build(g)
        with pytest.raises(LC.CoinError, match=match):
            g.ledger()
    refused(lambda g: (g.open("s-a", "7", COMMIT_A, families), g.open("s-b", "7", COMMIT_A, families)),
            r"entry 1 opens a second session \(s-b, after s-a\) for Commit (ca)+ in period 7: one window, one session")
    refused(lambda g: (g.open("s-a", "7", COMMIT_A, families), g.open("s-a", "7", COMMIT_B, families)),
            "entry 1 opens session s-a for a second window")
    refused(lambda g: g.open("s-a", "7", COMMIT_A, families, key_sha512="00" * 64),
            "entry 0 opens session s-a under the key pin 0+, not the verifier's")
    refused(lambda g: g.append("end", "s-a", {"files": {}}), "entry 0 ends session s-a, which has no open entry")
    refused(lambda g: (g.open("s-a", "7", COMMIT_A, families), g.append("end", "s-a", {}), g.append("end", "s-a", {})),
            "entry 2 ends session s-a, which has no open entry or has ended")
    refused(lambda g: g.append("close", "s-a", {}), "entry 0 is of kind 'close', not open or end")
    g = _gate()
    g.open("s-a", "7", COMMIT_A, families)
    g.open("s-b", "7", COMMIT_B, families)
    assert set(g.ledger().windows) == {("7", COMMIT_A), ("7", COMMIT_B)}
    with pytest.raises(LC.CoinError, match="entry 0's signature doesn't verify"):
        PHC.Ledger.verify(g.lines, CH.challenger_key(2)[1])
    with pytest.raises(LC.CoinError, match="entry 0's signature doesn't verify"):
        PHC.Ledger.verify([g.lines[0].replace(b'"port":7811', b'"port":7812'), g.lines[1]], KEY)
    with pytest.raises(LC.CoinError, match="entry 0 is not entry 0 of a verity/challenger-auditor/v1 ledger"):
        PHC.Ledger.verify(g.lines[1:], KEY)
    with pytest.raises(LC.CoinError, match="entry 0 is malformed"):
        PHC.Ledger.verify([b"not json"], KEY)


def test_a_session_that_released_coins_with_no_accepted_window_refuses_its_period(tmp_path, monkeypatch):
    """RT4-INT-17: a window abandoned after its coins (its session released coins and no window of it was accepted) is
    refused; so are a session that never ended and one whose evidence the verifier can't read; a session that released
    nothing is not."""
    gate = _gate()
    kept, a = _window(gate, tmp_path, monkeypatch, "s-a", 0, 11)
    _dropped, b = _window(gate, tmp_path, monkeypatch, "s-b", 1, 12, commit=COMMIT_B)
    gate.open("s-i", "7", "cc" * 64, Wd.live_families(3))
    idle = _Session(tmp_path, monkeypatch, "s-i", leaf=2, seed=13)
    idle.client.close()
    assert idle.ended() is None
    gate.end("s-i", idle.directory)
    gate.open("s-n", "7", "cd" * 64, Wd.live_families(3))
    ledger, fetch = gate.ledger(), {"s-a": a.tree, "s-b": b.tree, "s-i": idle.tree}.get
    assert PHC.session_reason(kept, ledger, fetch) is None
    abandoned = (f"session s-b (Commit {COMMIT_B}) released 5 coins and has no accepted window: a window abandoned after "
                 "its coins")
    never = (f"session s-n (Commit {'cd' * 64}): its evidence is not trusted (session s-n never ended: the ledger has its "
             "open entry and no end, so nothing seals its out/), so nothing shows it released no coin")
    assert PHC.period_reason(ledger, "7", ["s-a"], fetch) == [abandoned, never]
    assert PHC.period_reason(ledger, "7", ["s-a", "s-b", "s-n"], fetch) == []
    assert PHC.period_reason(ledger, "8", [], fetch) == []
    assert PHC.period_reason(ledger, "7", ["s-a", "s-n"], {"s-a": a.tree, "s-i": idle.tree}.get) == [
        f"session s-b (Commit {COMMIT_B}): its evidence is not trusted (session s-b has no tree on the auditor's side (the "
        "gate's fetch)), so nothing shows it released no coin"]


def test_the_session_setting_is_the_gates_reply_for_the_window():
    """RT4-INT-17's serving half: the window carries the gate's reply (session and seq) with the period and Commit root its
    open named; any other form is refused."""
    reply = {"session": "s20261020T094547Z-287a", "seq": 0, "period": "p", "commit": COMMIT_A}
    assert PHC.session_of({**reply, "host": "vy-challenger-1.s", "port": 7811, "peer": "vy-nebius-1"})["seq"] == 0
    for bad in ({**reply, "session": ""}, {**reply, "seq": True}, {**reply, "seq": -1}, {**reply, "period": 7},
                {**reply, "commit": "ab"}, {**reply, "commit": COMMIT_A.upper()}, {**reply, "root": "ab" * 64},
                {k: v for k, v in reply.items() if k != "seq"}, {"id": "s", "commit": "c", "period": 1}):
        with pytest.raises(ValueError, match="pouw setting 'session'"):
            PHC.session_of(bad)


def test_infras_smoke_ledger_and_session_bind_under_the_gates_key():
    """RT4-INT-17 on the gate's own bytes (vy-challenger-1's smoke at 09:45Z, art:822a00ad and art:42d947fc): the
    ledger verifies under the key its session's key.json holds and tonight's pin, its one window is session
    s20261020T094547Z-287a, ended and closed, every file of its out/ is the one its end entry sealed, its record is signed
    for the open entry's host and layout, and the period refuses it as abandoned (it released 2 coins and no window was
    accepted); one edited byte of the ledger or the tree is refused."""
    sid, period = "s20261020T094547Z-287a", "smoke-20261020"
    commit = ("96278d36587262f84b2b66d848b8b52f339e88ebd5a7d8573e19ad5765bdbb2a1c7f33c29f46e86e87c984557d430b1754d58223"
              "2dd692bd0ed425f684057ada")
    key = PHC.pinned_key(SMOKE / sid / "out" / R.KEY, SMOKE_PIN)
    assert (key.W, key.H) == (1, 20)
    ledger = PHC.Ledger.load(SMOKE / "ledger" / "windows.jsonl", key)
    window = ledger.window(period, commit)
    assert list(ledger.windows) == [(period, commit)] and window["open"]["session"] == sid
    assert window["open"]["body"]["host"] == f"vy-challenger-1.{sid}" and window["end"]["body"]["closed"] is True
    fetch = {sid: SMOKE / sid}.get
    out = PHC.evidence(window, fetch, key)
    trusted, received = R.trusted_session(out, key, PHC.lms_verify)
    assert trusted.scheme == "verity/smoke/coins/v1" and len(received["replay"]) == 2
    assert PHC.period_reason(ledger, period, [], fetch) == [
        f"session {sid} (Commit {commit}) released 2 coins and has no accepted window: a window abandoned after its coins"]
    assert PHC.period_reason(ledger, period, [sid], fetch) == []
    smoke = {"coin_source": "challenger", "session": {"session": sid, "seq": 0, "period": period, "commit": commit}}
    assert PHC.session_reason(smoke, ledger, fetch).startswith("the window's round_coins are not a live coin commitment")
    lines = (SMOKE / "ledger" / "windows.jsonl").read_bytes().split(b"\n")[:2]
    with pytest.raises(LC.CoinError, match="entry 0's signature doesn't verify"):
        PHC.Ledger.verify([lines[0].replace(b'"port":7811', b'"port":7812'), lines[1]], key)
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
    carries the gate's reply (``--replay-session``), held to the session the ledger opened for its period and Commit."""
    run_root = hashlib.sha256(b"a run root").digest()
    gate = _gate()
    reply = gate.open("s-c", "c1", COMMIT_A, [(RCN.FAMILY, 1)], scheme=RCN.SCHEME)
    with pytest.raises(ValueError, match="pouw setting 'session'"):
        RCN.challenge(T.AUDITOR, run_root, {"id": "s-c"})
    host = _Session(tmp_path, monkeypatch, "s-c", leaf=0, seed=16,
                    challenger=LC.LiveChallenger(RCN.SCHEME, 0, [(RCN.FAMILY, 1)], CH.entropy(16)))
    chal, rec = RCN.challenge(T.AUDITOR, run_root, reply)
    assert host.ended() is None and rec["source"] == "challenger" and rec["session"] == reply
    gate.end("s-c", host.directory)
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


def test_the_challengers_record_names_its_session_only_through_the_gates_host_string(tmp_path, monkeypatch):
    """What infra must add for RT4-INT-17: the challenger's signed record names its host and coin commitment, not the
    session, Commit or period; the gate's host string ``vy-challenger-1.<session>`` ties it to its session's open entry,
    and the coin commitment's root is in no ledger entry, only in the record its end entry seals."""
    gate = _gate()
    reply = gate.open("s-a", "7", COMMIT_A, Wd.live_families(3))
    a = _Session(tmp_path, monkeypatch, "s-a", leaf=0, seed=11)
    a.client.close()
    assert a.ended() is None
    record = json.loads((a.directory / R.RECORD).read_text())
    assert sorted(record) == ["commitment", "host", "signature"] and record["host"] == reply["host"]
    c = LiveCommitment.from_json(record["commitment"])
    assert PHC.lms_verify(KEY, record_message(KEY.public[:16], record["host"], c), bytes.fromhex(record["signature"]))
    assert c.root.hex() not in b"".join(gate.lines).decode() and "root" not in reply
