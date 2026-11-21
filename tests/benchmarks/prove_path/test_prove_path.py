"""The prove path's compiler and its served windows' pieces, without proving.

* The compiler is total: every byte string is the requests or a `ValueError` naming why, the same on every run; random and
  truncated advice is refused, and advice over the registered cap ℓ is refused before it is read.
* A forged window's verdict (`records_audit.forged_outcome`), on records written as a5 and a0 write an audit's: behind the
  ZK server it passes on no proof of the forged unit's member released, and in a plain audit on Lean refusing the proof it
  read. A Lean verdict on no proof, refused at setup or for want of the session's files, counts for neither.
* I2's adapter: its requests, its refusal by name, and its Program bodies by digest, held to the Build that holds them.
* A served window's pieces on a small synthetic serve-openai run dir (no Commit): its requests' bodies by I1 id, φ of the served
  engine, the registered compiler and its default seed, the weights registration, and ``window-commit``, which writes its
  own Builds beside ``--out`` (`own_builds`, `naming.write_build`) and binds the advice the server committed."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import socket
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity.commitments import docket_digest as DD
from verity.commitments import domains as DM
from verity.commitments import salt_file as SF
from verity.commitments.periods import interface as IF
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.commit import step_rows

HERE = (Path(__file__).resolve().parent)
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "prove_path"))
import prove_path as PP  # noqa: E402
import records_audit as RA  # noqa: E402

sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
import a5 as A5  # noqa: E402

PROMPTS = (5, 3)


class Inner:
    """A compiler's name and digest, for the advice side alone (no Programs to load)."""

    name = "test-compiler"
    digest = "ab" * 64

    def load(self, rank, world, requests):
        raise AssertionError("the advice side loads no Program")


def _workload(tmp_path: Path, arrive: tuple[int, ...] = (0, 2), caps: tuple[int, ...] = (3, 2)) -> Path:
    f = tmp_path / "workload.json"
    f.write_text(json.dumps({"requests": [{"prompt_token_ids": list(range(lp)), "max_tokens": t, "arrive_step": s}
                                          for lp, t, s in zip(PROMPTS, caps, arrive, strict=True)]}))
    return f


def _advice(tmp_path: Path, **kw) -> bytes:
    assert PP.main(["advice", "--workload", str(_workload(tmp_path, **kw)), "--out", str(tmp_path / "advice.bin")]) == 0
    return (tmp_path / "advice.bin").read_bytes()


def _outcome(f, *args):
    """What the compiler does on an input: ("ok", its result) or ("refused", its message); any other exception fails the test."""
    try:
        return "ok", f(*args)
    except ValueError as e:
        return "refused", str(e)


def test_a_declared_runs_advice_compiles_to_the_requests_tp_commit_served(tmp_path):
    assert PP.compile_advice(PP.Capped(Inner()), _advice(tmp_path)) == [
        {"request_id": "r0", "arrive_step": 0, "prompt_len": 5, "max_tokens": 3},
        {"request_id": "r1", "arrive_step": 2, "prompt_len": 3, "max_tokens": 2}]


def test_the_cap_is_in_the_compilers_registered_digest():
    a, b = PP.Capped(Inner(), 1024), PP.Capped(Inner(), 2048)
    assert a.digest != b.digest and a.digest == PP.Capped(Inner(), 1024).digest and len(a.digest) == 128
    for bad in (True, 7, 1.5, "1024"):
        with pytest.raises(ValueError, match="the cap ℓ is a whole number of bits"):
            PP.Capped(Inner(), bad)


def test_advice_over_the_cap_is_refused_before_it_is_read(tmp_path):
    data = _advice(tmp_path)
    assert _outcome(PP.compile_advice, PP.Capped(Inner(), 8 * len(data)), data)[0] == "ok"
    for junk in (data + b"\0", b"\xff" * (len(data) + 1)):
        assert _outcome(PP.compile_advice, PP.Capped(Inner(), 8 * len(data)), junk) == (
            "refused", f"the advice is {8 * len(junk)} bits, over the registered cap ℓ = {8 * len(data)} bits")


def test_the_compiler_is_total_and_deterministic_on_random_truncated_and_flipped_advice(tmp_path):
    """Every input is the requests or a refusal by name, never another exception, and two runs give the same answer. Every random
    string and every strict prefix of the advice is refused; a flipped bit is either refused or other advice."""
    data, compiler = _advice(tmp_path), PP.Capped(Inner(), 4096)
    rng = random.Random(20261024)
    randoms = [bytes(rng.randrange(256) for _ in range(rng.randrange(0, 160))) for _ in range(1500)]
    prefixes = [data[:k] for k in range(len(data))]
    flips = [bytes(b ^ (1 << j) if i == k else b for i, b in enumerate(data)) for k in range(len(data)) for j in range(8)]
    tails = [data[:k] + bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40))) for k in range(1, len(data)) for _ in range(40)]
    for inputs, all_refused in ((randoms, True), (prefixes, True), (flips, False), (tails, False),
                                ([None, "advice", bytearray(data)], True)):
        first = [_outcome(PP.compile_advice, compiler, x) for x in inputs]
        assert first == [_outcome(PP.compile_advice, compiler, x) for x in inputs]
        if all_refused:
            assert all(kind == "refused" and why for kind, why in first)
    assert len({why for kind, why in first if kind == "refused"}) >= 1


def test_the_transfer_gates_are_in_the_compilers_registered_digest(monkeypatch):
    """I2's digest covers the templates it lists as holding transfer gates, and `Capped` passes the list and its digest on, so a
    docket under a compiler proving other units whole names another compiler."""
    monkeypatch.setattr("verity_vllm.program.predict.score.code_version", lambda: "c0de")
    r = SimpleNamespace(phi=SimpleNamespace(to_json=lambda: {"model": "m"}), default_seed=0, cap_bits=PP.CAP_BITS)
    bare, listing = (PP.I2Compiler(r, {}, "/nonexistent", transfer_gates=tg) for tg in ((), ("RoPEHead_v1{D=64}",)))
    assert bare.digest != listing.digest and len(listing.digest) == 128
    assert PP.Capped(listing).transfer_gates == ("RoPEHead_v1{D=64}",) and PP.Capped(bare).transfer_gates == ()
    assert PP.Capped(listing).digest != PP.Capped(bare).digest
    assert PP.Capped(Inner()).transfer_gates == ()


def test_an_audit_listens_on_ports_no_other_audit_holds():
    with socket.socket() as held, socket.socket() as spare:
        held.bind(("127.0.0.1", 0))
        spare.bind(("127.0.0.1", 0))
        taken, free = held.getsockname()[1], spare.getsockname()[1]
        spare.close()
        assert PP.free_ports(1, [taken, free]) == free
        with pytest.raises(SystemExit):
            PP.free_ports(1, [taken])
    base = PP.free_ports(2 * A5.RANK_PORTS)
    assert PP.PORTS.start <= base and base + 2 * A5.RANK_PORTS <= PP.PORTS.stop


#: the ZK server's reason for holding back a forged unit's proof, as the prover's record of the session gives it
HELD = RA.zk_server_refused() + "circuit: the constraints lincheck final fail on a rep's pads"
#: the verifier's reason on a session whose proof never arrived
NO_PROOF = "R7: no proof for circuit/rep0 (lone rep)"
#: Lean's refusals on no proof: for want of a session's proof file, and at the statement's setup
NO_FILE = "session files: no such file or directory (error code: 2)\n  file: session-m0/sessions/l0000/circuit.rep0.bin"
AT_SETUP = "setup: one statement per table expected"
#: Lean's refusal of a forged unit's proof it read, and the verifier's own of the same proof
ON_THE_PROOF = "lincheck: ConsistencyFailed (sumcheck-final)"
PROOF_FAILS = 'Lincheck(ConsistencyFailed { which: "sumcheck-final" })'
#: a forged window's three checks on where it was stopped (`records_audit.forged_outcome`)
STOPPED = ("no_proof_released", "lean_refused", "read_refused")


def _summary(sample: list[int], before: bool = True) -> dict:
    """What a5's summary of an audit of one member of 8 units says of its sample, ``sample``, and of the checks before
    it: each passed, but the docket's where not ``before``."""
    return {"members": [{"k": 0, "template": "T", "n": 8, "base": 0}], "draw": sample,
            "checks_before_draw": {"registration": before, "source": True}}


def _stopped(row: dict) -> dict:
    """A forged window's verdict ``row``'s three checks on where it was stopped (`STOPPED`), its two others holding: the
    one on the checks before the sample, and the record rejecting."""
    checks = dict(row["checks"])
    stopped = {k: checks.pop(k) for k in STOPPED}
    assert list(checks.values()) == [True, True], row
    return stopped


def _forged(d: Path, lean_why: str | None, *, zk_server: bool, proofs: int | None = 0, stopped: str | None = None,
            digest: bool = True, sample: tuple[int, ...] = (0, 1, 2), before: bool = True) -> dict:
    """A forged rank's audit as a5 leaves one in ``d``, its unit 1 of member 0 forged -> a5's summary of it. Member 0's
    session has ``proofs`` proof streams in the verifier's record of it (no record where None) and a prover's log, written
    as `a0.run` writes one, whose ``LIVE`` line gives ``stopped`` as why it stopped short. Lean's verdict on the session
    refuses with ``lean_why`` (it accepts where None, and the audit with it), its own where ``digest``, else a0's stand-in
    for none. ``zk_server``: the sessions ran behind the ZK server; without it, plain."""
    import a0

    accepted = lean_why is None
    m0_why = None if accepted else NO_PROOF if not proofs else PROOF_FAILS
    session = d / "session-m0"
    if proofs is not None:
        (session / "sessions" / "l0000").mkdir(parents=True)
        (session / "sessions" / "l0000" / "session.json").write_text(json.dumps({
            "proof_sha512": [{"stream": f"circuit/rep{r}", "bytes": 9, "sha512": "cd" * 64} for r in range(proofs)],
            "verdict": {"accepted": accepted, "why": m0_why or ""}}))
    session.mkdir(parents=True, exist_ok=True)
    live = {"accepted": accepted, "why": m0_why or "", **({"prover_stopped": stopped} if stopped else {})}
    a0.run([sys.executable, "-c", f"print({'LIVE' + chr(9) + json.dumps(live)!r})"], session / "prove.log")
    verdicts = {"m0-verifier/m0": {"accepted": accepted, "why": m0_why},
                "lean-verify/m0": {"accepted": accepted, "why": lean_why, "of_record": True,
                                   "digest": "ab" * 64 if digest else None}}
    (d / "audit.json").write_text(json.dumps({"accepted": accepted, "verdicts": verdicts}))
    return {**_summary(list(sample), before),
            "arm": "fw" if zk_server else None, **({} if zk_server else {"plain": {"coins": "tree"}}),
            "audit": {"accepted": accepted, "verdicts": {k: v["accepted"] for k, v in verdicts.items()},
                      "why": {k: v.get("why") for k, v in verdicts.items() if not v["accepted"]}}}


def _forged_row(s: dict, d: Path) -> dict:
    """`records_audit.forged_outcome` of `_forged`'s rank, its forged unit 1 of member 0, with the caller's check that
    the unit was sampled."""
    return RA.forged_outcome(s, d, 0, {"forged_unit_drawn": 1 in s["draw"]})


@pytest.mark.parametrize("lean_why, kw, ok, no_proof, lean", [
    (NO_FILE, {"stopped": HELD}, True, True, None),
    (AT_SETUP, {"stopped": HELD}, True, True, None),
    (NO_FILE, {}, False, False, None),
    ("no session recorded", {"proofs": None, "stopped": HELD, "digest": False}, False, False, None),
    (ON_THE_PROOF, {"proofs": 2}, False, False, True),
    (None, {"proofs": 2}, False, False, False),
    (NO_FILE, {"stopped": HELD, "sample": (0, 2)}, False, True, None),
    (NO_FILE, {"stopped": HELD, "before": False}, False, True, None),
], ids=["no-proof-released", "lean-refused-at-setup", "the-prover-stopped-on-its-own", "no-session-recorded",
        "a-proof-left-and-lean-refused-it", "accepted", "forged-unit-not-sampled", "refused-before-the-sample"])
def test_behind_the_zk_server_a_forged_rank_passes_on_no_proof_released(tmp_path, lean_why, kw, ok, no_proof, lean):
    """Under `--zk` the ZK server holds a false member's proof back, so the rank passes on that: the verifier's record of
    the forged unit's session lists no proof and the prover's gives the ZK server's refusal. Lean's verdict on that session
    is on no proof, at setup or for want of the session's files, and is no value of ``lean_refused``; a proof that left
    fails the rank even where Lean refused it."""
    row = _forged_row(_forged(tmp_path, lean_why, zk_server=True, **kw), tmp_path)
    checks = row["checks"]
    assert (row["ok"], row["passes_on"]) == (ok, "no_proof_released"), row
    assert (checks["no_proof_released"], checks["lean_refused"]) == (no_proof, lean), row
    before, sampled = (v for k, v in checks.items() if k not in ("rejected", *STOPPED))
    assert (before, sampled) == ("before" not in kw, "sample" not in kw), row
    assert checks["rejected"] is (lean_why is not None)
    assert sorted(row["no_value"]) == (["lean_refused"] if lean is None else []), row
    if lean is None:
        assert row["no_value"]["lean_refused"] == f"the Lean verifier read no proof: lean-verify/m0: {lean_why.splitlines()[0]}"
    assert row["sessions"] == {"m0": {"proofs": kw.get("proofs", 0), "prover_stopped": kw.get("stopped")}}


@pytest.mark.parametrize("lean_why, kw, ok, lean", [
    (ON_THE_PROOF, {"proofs": 2}, True, True),
    (AT_SETUP, {"proofs": 2}, False, None),
    (NO_FILE, {}, False, None),
    ("no verdict (rc 139): ", {"proofs": 2, "digest": False}, False, None),
    (None, {"proofs": 2}, False, False),
    (ON_THE_PROOF, {"proofs": 2, "sample": (0, 2)}, False, True),
], ids=["lean-refused-the-proof", "lean-refused-at-setup", "lean-had-no-proof-file", "no-verdict-of-leans-own", "accepted",
        "forged-unit-not-sampled"])
def test_a_plain_forged_rank_passes_on_lean_refusing_the_proof_it_read(tmp_path, lean_why, kw, ok, lean):
    """Under ``--plain`` no ZK server stands before the verifier, so the proof leaves and the rank passes on Lean's own
    verdict refusing it. A verdict that read no proof (refused at setup or for want of the session's files, or a0's
    stand-in for none) is no value of ``lean_refused`` and passes nothing, and ``no_proof_released`` has no value."""
    row = _forged_row(_forged(tmp_path, lean_why, zk_server=False, **kw), tmp_path)
    checks = row["checks"]
    assert (row["ok"], row["passes_on"]) == (ok, "lean_refused"), row
    assert (checks["no_proof_released"], checks["lean_refused"]) == (None, lean), row
    assert row["no_value"]["no_proof_released"] == "the sessions ran behind no ZK server"
    assert ("lean_refused" in row["no_value"]) is (lean is None), row
    assert row["sessions"] == {"m0": {"proofs": kw.get("proofs", 0), "prover_stopped": None}}


def _two_members(tmp_path: Path) -> tuple[dict, Path]:
    """A forged window of two sampled members under ``tmp_path/window``, each session a process pair's (`_forged`): member
    0 honest, its proof left and Lean accepted it; member 1 false, its proof held back by the ZK server -> a5's summary of
    it for `records_audit.forged_outcome`, and the window's audit directory."""
    _forged(tmp_path / "honest", None, zk_server=True, proofs=2)
    false = _forged(tmp_path / "false", NO_FILE, zk_server=True, stopped=HELD)
    d = tmp_path / "window"
    d.mkdir()
    shutil.copytree(tmp_path / "honest" / "session-m0", d / "session-m0")
    shutil.copytree(tmp_path / "false" / "session-m0", d / "session-m1")
    verdicts = {f"{k.split('/')[0]}/m{i}": v for i, w in enumerate(("honest", "false"))
                for k, v in json.loads((tmp_path / w / "audit.json").read_text())["verdicts"].items()}
    (d / "audit.json").write_text(json.dumps({"accepted": False, "verdicts": verdicts}))
    return {**false, "audit": {"accepted": False, "why": {}}}, d


def test_a_forged_window_of_several_members_passes_on_the_one_whose_proof_was_held(tmp_path):
    """The records audit knows no forged member, so it reads every sampled member's session: the honest member's proof
    left and Lean accepted it, the false member's was held back, and Lean's refusal for want of its files is the only
    one, so ``lean_refused`` has no value."""
    import records_audit as RA

    s, d = _two_members(tmp_path)
    row = RA.forged_outcome(s, d)
    assert row["ok"] is True and _stopped(row) == {"no_proof_released": True, "lean_refused": None, "read_refused": False}, row
    assert row["sessions"] == {"m0": {"proofs": 2, "prover_stopped": None}, "m1": {"proofs": 0, "prover_stopped": HELD}}
    assert row["no_value"] == {"lean_refused": "the Lean verifier read no proof: lean-verify/m1: " + NO_FILE.splitlines()[0]}
    (d / "session-m1" / "prove.log").write_text((d / "session-m0" / "prove.log").read_text())
    assert RA.forged_outcome(s, d)["ok"] is False


def test_a_session_the_audits_proof_server_proved_has_its_stop_in_that_servers_provers_log(tmp_path):
    """Behind the ZK server an audit's sessions are proved by its proof server's one prover where the members' statements
    agree on its setup (`a5.proof_server`). No session directory then holds a prover's log: every session's ``LIVE`` line
    is in that prover's log beside the sessions, between the lines of its stderr, each labelled with its session's
    directory. The false member's stop is the line under its own label, and the window passes as one of process pairs
    does. A stop under another label is not its stop: another window's session of the same name, the honest member's, or
    no line of its own at all. Nor is a line of another form than the prover's: one not opened ``LIVE`` and a tab, or
    holding JSON that is no object, gives no stop, and one opened so that holds no JSON raises, so in neither case does
    the window pass. A session directory that holds a prover's log was a process pair's, and that log is read."""
    import records_audit as RA

    s, d = _two_members(tmp_path)
    m0, m1 = ((d / f"session-m{k}").resolve() for k in (0, 1))
    for session in (m0, m1):
        (session / "prove.log").unlink()

    def served(*lives: tuple[Path, str | None]) -> dict:
        """The verdict with the proof server's prover's log holding one ``LIVE`` line a (label, stop) of ``lives``."""
        (d / "proof-server").mkdir(exist_ok=True)
        (d / "proof-server" / "prover-cpu.log").write_text("".join(
            f"prove: a line of the prover's stderr\nLIVE\t"
            f"{json.dumps({'accepted': stop is None, 'label': str(label), **({'prover_stopped': stop} if stop else {})})}\n"
            for label, stop in lives))
        return RA.forged_outcome(s, d)

    assert RA.forged_outcome(s, d)["ok"] is False, "a window with no prover's log anywhere"
    row = served((m0, None), (m1, HELD))
    assert row["ok"] is True and _stopped(row) == {"no_proof_released": True, "lean_refused": None, "read_refused": False}, row
    assert row["sessions"] == {"m0": {"proofs": 2, "prover_stopped": None}, "m1": {"proofs": 0, "prover_stopped": HELD}}
    for lives in (((m0, None), (tmp_path / "other" / "session-m1", HELD)), ((m0, HELD), (m1, None)), ((m0, None),)):
        row = served(*lives)
        assert row["ok"] is False and row["checks"]["no_proof_released"] is False, (lives, row)
        assert row["sessions"]["m1"] == {"proofs": 0, "prover_stopped": None}, (lives, row)
    line = json.dumps({"accepted": False, "label": str(m1), "prover_stopped": HELD})
    for other in (f"LIVE {line}", f"live\t{line}", f" LIVE\t{line}", f"SESSION\t{line}", f"LIVE\t[{line}]",
                  f"LIVE\t{json.dumps(line)}"):
        (d / "proof-server" / "prover-cpu.log").write_text(f"{other}\n")
        assert RA.forged_outcome(s, d)["sessions"]["m1"] == {"proofs": 0, "prover_stopped": None}, other
    for other in (f"LIVE\t{line[:-1]}", f"LIVE\t{line} and more"):
        (d / "proof-server" / "prover-cpu.log").write_text(f"{other}\n")
        with pytest.raises(ValueError):
            RA.forged_outcome(s, d)
    (d / "proof-server" / "prover-cpu.log").write_bytes(b"\xff\xfe a line that is no text\nLIVE\t" + line.encode() + b"\n")
    assert RA.forged_outcome(s, d)["sessions"]["m1"] == {"proofs": 0, "prover_stopped": HELD}
    (m1 / "prove.log").write_text(f"LIVE\t{json.dumps({'accepted': False})}\n")
    assert served((m0, None), (m1, HELD))["sessions"]["m1"] == {"proofs": 0, "prover_stopped": None}




def test_i2s_requests_are_its_compiled_units_once_per_advice_and_its_refusal_is_named(monkeypatch):
    from verity_vllm.program.predict import steps as ST
    from verity_vllm.program.predict.common import Unexpressible

    calls = []

    def compile_steps(registered, data, ingress, *, named=False):
        calls.append(data)
        if data == b"no":
            raise Unexpressible("a chunked prefill")
        return SimpleNamespace(requests=[ST.RequestUnit(0, "d0" * 64, 5, 2, {}, None, 0, "r0"),
                                         ST.RequestUnit(1, "d1" * 64, 3, 1, {}, None, 2, "r1")], named={})

    monkeypatch.setattr(ST, "compile_steps", compile_steps)
    i2 = PP.I2Compiler(registered=None, ingress={}, programs_root="/nonexistent")
    want = [{"request_id": "r0", "arrive_step": 0, "prompt_len": 5, "max_tokens": 3, "digest": "d0" * 64},
            {"request_id": "r1", "arrive_step": 2, "prompt_len": 3, "max_tokens": 2, "digest": "d1" * 64}]
    assert i2.requests(b"ok") == want and i2.requests(b"ok") == want and calls == [b"ok"]
    with pytest.raises(ValueError, match="I2 refuses the advice: a chunked prefill"):
        i2.requests(b"no")


def test_i2_loads_only_a_single_gpu_session_and_only_programs_a_build_dir_holds(tmp_path):
    from verity_vllm.program.predict import steps as ST

    for name, ok, digest in (("build_request_LP5_T2", True, "d0" * 64), ("build_request_LP3_T1", False, "d1" * 64),
                             ("build_step", True, "d2" * 64)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps({"ok": ok, "program": {"digest": digest}}))
    assert PP.bodies_by_digest(str(tmp_path)) == {"d0" * 64: str(tmp_path / "build_request_LP5_T2")}
    i2 = PP.I2Compiler(registered=None, ingress={}, programs_root=str(tmp_path))
    for rank, world in ((1, 2), (0, 2), (1, 1)):
        with pytest.raises(ValueError, match="I2 compiles a single-GPU session"):
            i2.load(rank, world, [])
    i2.units["d1" * 64] = ST.RequestUnit(1, "d1" * 64, 3, 1, {"kind": "greedy"}, None, 0, "r1", stop={"max_tokens": 2})
    with pytest.raises(ValueError, match=r"request r1: no Build dir under .* holds I2's Program d1d1.*\(its trace: "
                                         r"build_request_LP3_T1_greedy_s[0-9a-f]{12}/; the Builds there: build_request_LP5_T2 d0d0.*\)"):
        i2.load(0, 1, [{"request_id": "r1", "arrive_step": 0, "digest": "d1" * 64}])


#: adapters/vllm's `tests.program.predict.test_predict.TINY`: a two-layer Llama whose request Programs are I2's in a second
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
STOCHASTIC = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": 32, "rid": 4}


def _request_build(sampling: dict | None = None, named: bool = False, **shape):
    from verity_vllm.program.predict.build import build_program
    from verity_vllm.program.predict.naming import named_program
    from verity_vllm.program.predict.phi import Phi

    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, workload={"requests": []},
              target={"compute_capability": [12, 0], "num_sms": 188})
    phi = Phi.from_json({**phi.to_json(), "sampling": sampling or {"kind": "greedy"}})
    return named_program(phi, **shape) if named else build_program(phi, "request", **shape)


def _request_program(sampling: dict | None = None, **shape):
    return _request_build(sampling, **shape).program


@pytest.fixture(scope="module")
def requests_of_tiny():
    """TINY's greedy request Programs at (LP, T) = (4, 1) and (4, 2), its stochastic one (a seed and its seed rule's inputs beside the prompt)
    and its cached one (a prefix of 16 rows beside the prompt)."""
    return {"greedy": _request_program(LP=4, T=1), "longer": _request_program(LP=4, T=2),
            "stochastic": _request_program(STOCHASTIC, LP=4, T=2), "cached": _request_program(LP=24, T=2, C=16)}


def _build(d: Path, program, *, desc: dict | None = None, digest: str | None = None) -> str:
    """Build dir ``d`` of ``program`` as the Build writes one (`build_instances.write_instances`): its descriptor (``desc``, the
    program's by default), and its instances and ``result.json`` under ``digest`` (the descriptor's by default); the digest."""
    import gzip

    from verity.circuits.codec import encode_program, program_digest
    from verity_vllm.pipeline.build_instances import write_instances

    d.mkdir(parents=True, exist_ok=True)
    desc = encode_program(program) if desc is None else desc
    digest = digest or program_digest(desc)
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    write_instances(str(d), program, digest, None, {})
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": digest}}))
    return digest


def _edit_instances(d: Path, edit) -> None:
    import gzip

    with gzip.open(d / "instances.json.gz", "rt") as f:
        inst = json.load(f)
    edit(inst)
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(inst, f)


def _load(root: Path, digest: str):
    return PP.I2Compiler(registered=None, ingress={}, programs_root=str(root)).load(
        0, 1, [{"request_id": "r0", "arrive_step": 0, "digest": digest}])


def test_i2_loads_a_build_whose_body_is_the_program_it_compiled(tmp_path, requests_of_tiny):
    """I2's adapter holds each Build dir to the digest I2 compiled (`build_of`) before the window reads its rows: an honest
    Build of each request Program loads, its Input batches read as the parameters they bind (prompt, seed, prefix,
    weights)."""
    for name, program in requests_of_tiny.items():
        dg = _build(tmp_path / name / "build_request", program)
        p = _load(tmp_path / name, dg)
        assert p.digs == {"r0": dg} and len(p.progs["r0"]["rows"]) == len(program.fn.body.nodes), name


def test_i2_refuses_a_build_whose_body_was_swapped_under_its_old_digest(tmp_path, requests_of_tiny):
    """The Build's ``result.json`` (and its instances head) keep the digest I2 compiled, and its body is another Program's:
    each is refused by name, the descriptor by its recomputed digest, the instances by the descriptor's rows."""
    from verity.circuits.codec import encode_program, program_digest

    greedy, longer = requests_of_tiny["greedy"], requests_of_tiny["longer"]
    dg, other = (program_digest(encode_program(p)) for p in (greedy, longer))
    d = tmp_path / "desc" / "build_request"
    _build(d, greedy, desc=encode_program(longer), digest=dg)
    with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))}/descriptor.json.gz is Program {other[:16]}, "
                                         rf"not I2's {dg[:16]}$"):
        _load(tmp_path / "desc", dg)

    d = tmp_path / "inst" / "build_request"
    _build(d, longer, desc=encode_program(greedy), digest=dg)
    with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))}'s instances.json.gz has {len(longer.fn.body.nodes)} "
                                         rf"rows, and its descriptor.json.gz {len(greedy.fn.body.nodes)} nodes$"):
        _load(tmp_path / "inst", dg)

    k = next(k for k, n in enumerate(greedy.fn.body.nodes) if len(n.args) == 2 and n.args[0].refs != n.args[1].refs)
    for what, edit in (("args", lambda r: r["args"].reverse()), ("spec", lambda r: r.update(spec=r["spec"] + "x")),
                       ("i", lambda r: r.update(i=r["i"] + 1))):
        d = tmp_path / what / "build_request"
        _build(d, greedy)
        _edit_instances(d, lambda inst: edit(inst["rows"][k]))
        with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))}'s instances.json.gz row {k} is not its "
                                             rf"descriptor.json.gz's node {k}: they differ in {what}$"):
            _load(tmp_path / what, dg)


def test_i2_refuses_a_build_that_records_another_digest_or_has_no_descriptor(tmp_path, requests_of_tiny):
    greedy = requests_of_tiny["greedy"]
    d = tmp_path / "head" / "build_request"
    dg = _build(d, greedy)
    _edit_instances(d, lambda inst: inst.update(digest="ab" * 64))
    with pytest.raises(ValueError, match=rf"'s instances.json.gz records Program {'ab' * 8}, and its descriptor.json.gz is "
                                         rf"Program {dg[:16]}$"):
        _load(tmp_path / "head", dg)

    d = tmp_path / "result" / "build_request"
    _build(d, greedy)
    assert PP.build_of(str(d), dg)["instances"]["digest"] == dg
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "ab" * 64}}))
    with pytest.raises(ValueError, match=rf"'s result.json records Program {'ab' * 8}, and its descriptor.json.gz is Program"):
        PP.build_of(str(d), dg)

    d = tmp_path / "boolean" / "build_request"
    _build(d, greedy)
    _edit_instances(d, lambda inst: inst.update(ir="boolean"))
    with pytest.raises(ValueError, match="is a Build in IR 'boolean', and I2 compiles word Programs"):
        _load(tmp_path / "boolean", dg)
    (d / "descriptor.json.gz").unlink()
    with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))} has no descriptor.json.gz: its Program's digest "
                                         rf"can't be recomputed$"):
        _load(tmp_path / "boolean", dg)


def test_a_builds_input_batches_are_its_parameters_as_src_shapes_them():
    src = 'Derived[X]_v1{SRC={"dict":{"inputs":[[2],[],[0],[2,2]]}}}'
    assert PP.input_layout({"program_fn": src}, [2, 1, 3, 1, 5, 7]) == [(0, 0), (1, 0), (3, 0), (3, 3), (4, 0), (4, 5)]
    with pytest.raises(ValueError, match="Input batch 1 runs past parameter 1's 1 leaves"):
        PP.input_layout({"program_fn": src}, [2, 2, 4])
    with pytest.raises(ValueError, match="the root's 2 Input batches don't fill the 4 parameters SRC shapes"):
        PP.input_layout({"program_fn": src}, [2, 1])
    for pf in ("Derived[X]_v1", 'Derived[X]_v1{SRC={"dict":{}}}', 'Derived[X]_v1{SRC={"dict":{"inputs":[1]}}}'):
        with pytest.raises(ValueError, match="records no SRC input shapes: not one of I2's Programs"):
            PP.input_layout({"program_fn": pf}, [1])
    assert PP.merged([["p", 1, 0, 3], ["p", 1, 3, 5], ["p", 0, 5, 6], ["n", 2, 0, 1], ["n", 2, 2, 3]]) == \
        [["p", 1, 0, 5], ["p", 0, 5, 6], ["n", 2, 0, 1], ["n", 2, 2, 3]]


TARGET = {"device_name": "test-gpu", "execution_label": "bi-eager", "batch_invariant": True}


def _serve_request(i: int, prompt: list[int], cap: int, windows: list[int], user: str | None = None) -> dict:
    return {"http_id": f"cmpl-{i}", "engine_id": f"cmpl-{i}-0", "user": user, "prompt_ids": prompt, "windows": windows,
            "ingress": {"brought": {"max_tokens": cap, "temperature": 0.0},
                        "params": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": cap, "min_tokens": 0,
                                   "stop": []}}}


#: the server's registered compiler in these tests (`served_identity`'s, from `verity-vllm advice` on the server's tree)
SERVER = ("verity_vllm.program.predict.steps.compile_steps", "5e" * 64)


def _served(tmp_path: Path, monkeypatch, *, root: str = "aa" * 64, scheme: str = step_rows.NAME, published: str | None = None,
            uncapped: bool = False, gates: tuple[str, ...] | None = None) -> SimpleNamespace:
    """A serve-openai run dir with one closed window (w0, the stand-in workload's two requests by their `user`) and a request of
    another window, its advice committed as `verity-vllm advice` commits it under the served compiler (or, ``uncapped``, under
    the inner compiler alone; ``gates``, under the server's listing those instead of the model's of record), and its row dir's
    request Programs as vllm's registered compiler writes them;
    the server's compiler identity, I2's compile and the row dir's record stubbed."""
    from verity_vllm.program.predict import steps as ST

    monkeypatch.setattr(PP, "served_identity", lambda: SERVER)
    run, rows = tmp_path / "serve", tmp_path / "rows"
    (run / "advice" / "w0" / "private").mkdir(parents=True)
    rows.mkdir()
    pub = root if published is None else published
    (run / "serve.json").write_text(json.dumps({"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da",
                                                "default_seed": DEFAULT_SEED, "engine": {"execution": "bi-eager", "pin": {"world": 1}},
                                                "target": TARGET, "run_id": "RUN"}))
    reqs = [_serve_request(0, list(range(5)), 3, [0], "r0"), _serve_request(1, list(range(3)), 2, [0], "r1"),
            _serve_request(2, [9], 1, [1])]
    (run / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs))
    (run / "windows.jsonl").write_text("".join(json.dumps({"window": w, "run_id": f"RUN/w{w}", "run_root": r}) + "\n"
                                               for w, r in ((0, pub), (1, "bb" * 64))))
    data = _advice(tmp_path)
    compiler = PP.Capped(PP.i2_of_serve(str(run), 0, str(rows), None, None))
    if gates is not None:
        compiler = PP.Capped(replace(compiler.inner, transfer_gates=tuple(gates)))
    inner = SimpleNamespace(name=compiler.inner.name, digest=compiler.inner.digest)
    _r, bound, reg = AW.of_advice(inner if uncapped else compiler, data)
    adv = run / "advice" / "w0"
    reg_doc = {"schema": "verity-vllm/advice-registration/v1", "scheme": AW.SCHEME, "binding": bound,
               "compiler": {**bound["compiler"], "cap_bits": PP.CAP_BITS, "inner": {"name": SERVER[0], "digest": SERVER[1]}},
               "window": 0, "commit": {"run_id": "RUN/w0", "run_root": pub}}
    (adv / "registration.json").write_text(json.dumps(reg_doc))
    (adv / "private" / "advice.bin").write_bytes(data)
    (adv / "private" / "opening.json").write_text(json.dumps(AW.opening(data, reg)))

    def compile_steps(registered, data, ingress, *, named=False):
        assert set(ingress) == {"r0", "r1"} and registered.default_seed == DEFAULT_SEED
        return SimpleNamespace(requests=[ST.RequestUnit(0, "d0" * 64, 5, 2, {}, None, 0, "r0"),
                                         ST.RequestUnit(1, "d1" * 64, 3, 1, {}, None, 2, "r1")], named={})

    monkeypatch.setattr(ST, "compile_steps", compile_steps)
    record = SimpleNamespace(run=SimpleNamespace(run_root=bytes.fromhex(root)), store=SimpleNamespace(commit_scheme=scheme),
                             digests={"r0": "d0" * 64, "r1": "d1" * 64})
    monkeypatch.setattr(PP, "record1", lambda bundle, row_dir: record)
    a = argparse.Namespace(serve_run=str(run), window=0, row_dir=str(rows), bundle=str(rows / "b"),
                           hf_config=None, target=None, cap_bits=PP.CAP_BITS, opening=str(adv / "private" / "opening.json"),
                           tp_commit=None, workload=None)
    return SimpleNamespace(run=run, a=a, data=data, bound=bound, compiler=compiler, root=root)


#: the scope a naming test's partition takes: layer 0's MLP, as the Llama partitions do (the ``model.`` wrapper's prefix stripped)
LAYER0 = {"templates": ["SiluMul_v2{I=128}"], "law": "subset:2", "scope": {"op_path_prefix": "model.layers.0.mlp."}}


def _tiny_served(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """`_served`'s window with its two requests running TINY's request Programs as I2 compiles them (r0 at LP 4, T 1; r1 at LP 4,
    T 2), each with the names I2's compile gives it where it is asked for them (``named``, `naming`); the served model lists no
    transfer gates (TINY has no ``RoPEHead_v1{D=64}``). -> `_served`'s, with the Programs and their names by (LP, T), the units,
    and each compile's ``named``."""
    from verity_vllm.program.predict import steps as ST

    monkeypatch.setattr(PP, "GATES_OF_RECORD", {})
    s = _served(tmp_path, monkeypatch)
    builds = {(4, 1): _request_build(LP=4, T=1, named=True), (4, 2): _request_build(LP=4, T=2, named=True)}
    units = [ST.RequestUnit(0, builds[(4, 1)].digest, 4, 1, {"kind": "greedy"}, None, 0, "r0"),
             ST.RequestUnit(1, builds[(4, 2)].digest, 4, 2, {"kind": "greedy"}, None, 2, "r1")]
    compiles: list[bool] = []

    def compile_steps(registered, data, ingress, *, named=False):
        compiles.append(named)
        return SimpleNamespace(requests=units, programs={b.digest: b.program for b in builds.values()},
                               named={b.digest: b.named for b in builds.values()} if named else {})

    monkeypatch.setattr(ST, "compile_steps", compile_steps)
    return SimpleNamespace(**vars(s), programs={k: b.program for k, b in builds.items()}, named={k: b.named for k, b in builds.items()},
                           units=units, compiles=compiles)


def test_a_served_windows_bodies_are_its_requests_ingress_by_i1_id(tmp_path, monkeypatch):
    s = _served(tmp_path, monkeypatch)
    w = PP.serve_window(str(s.run), 0)
    assert [PP.serve_id(q) for q in w.requests] == ["r0", "r1"] and w.commit["run_root"] == s.root
    assert PP.serve_ingress(w.requests) == {"r0": {"max_tokens": 3, "temperature": 0.0, "prompt": [0, 1, 2, 3, 4]},
                                            "r1": {"max_tokens": 2, "temperature": 0.0, "prompt": [0, 1, 2]}}
    assert [PP.serve_id(q) for q in PP.serve_requests(s.run, 1)] == ["cmpl-2"]
    with pytest.raises(ValueError, match="request r0 is recorded twice"):
        PP.serve_ingress([w.requests[0], w.requests[0]])
    assert PP.serve_ingress([{**w.requests[0], "ingress": {}}]) == {"r0": {"prompt": [0, 1, 2, 3, 4]}}
    with pytest.raises(ValueError, match="0 lines for window 2, not one"):
        PP.serve_window(str(s.run), 2)


def test_phi_of_a_served_engine_reads_its_target_and_execution(tmp_path):
    serve = {"model": "m/x", "revision": "r", "default_seed": DEFAULT_SEED, "engine": {"execution": "bi-eager"}, "target": TARGET}
    phi = PP.phi_of_serve(serve)
    assert (phi.model["repo"], phi.model["revision"], phi.target, phi.execution) == ("m/x", "r", TARGET, "bi-eager")
    assert phi.model["hf_config"] == json.loads(PP.LLAMA32_1B_CONFIG.read_text()) and not phi.requests
    with pytest.raises(ValueError, match="serve.json records no target profile .* --target F gives it"):
        PP.phi_of_serve({**serve, "target": None})
    (tmp_path / "target.json").write_text(json.dumps({**TARGET, "device_name": "other"}))
    assert PP.phi_of_serve({**serve, "target": None}, target=str(tmp_path / "target.json")).target["device_name"] == "other"
    with pytest.raises(ValueError, match="no execution label"):
        PP.phi_of_serve({**serve, "engine": {}})
    assert PP.phi_of_serve({**serve, "execution": "eager"}).execution == "eager"


#: a served engine's record as ``serve.json`` holds it (`LLM.engine_record`'s): an engine serving below its model's length
ENGINE = {"execution": "bi-eager", "env_flags": {},
          "engine_kwargs": {"max_model_len": 512, "max_num_batched_tokens": 512, "max_num_seqs": 8, "enable_prefix_caching": False,
                            "tensor_parallel_size": 1}}


@pytest.mark.parametrize("label", ["serve.json's", "its engine record's"])
def test_a_served_engines_phi_is_the_one_its_row_dirs_builds_are_compiled_under(tmp_path, label):
    """The prover's φ of a served engine (`predict.phi.served_phi`) and the verifier's (`phi_of_serve`) are one φ of one
    ``serve.json``, its engine's args and world included, wherever
    it states its execution: a request Program's length rule reads the engine's ``max_model_len``, so under a φ stating
    none I2 compiles, for an engine serving below its model's length, a Program no Build holds. A launch that left the
    length to vLLM states the length its engine fitted on both sides. The bodies the two compile are one too."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config

    from verity_vllm.pipeline import advice as ADV
    from verity_vllm.program.predict.phi import served_phi

    serve = {"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "default_seed": DEFAULT_SEED,
             "target": TARGET, "engine": ENGINE, **({"execution": "bi-eager"} if label == "serve.json's" else {})}
    (tmp_path / "serve.json").write_text(json.dumps(serve))
    (tmp_path / "advice" / "w0").mkdir(parents=True)
    (tmp_path / "advice" / "w0" / "registration.json").write_text(json.dumps({"window": 0}))
    reqs = [_serve_request(0, list(range(5)), 3, [0], "r0"), _serve_request(1, list(range(3)), 2, [0], "r1")]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs))
    bodies = ADV.bodies(tmp_path, 0)
    meta = {"model": serve["model"], "revision": serve["revision"], "target": serve["target"],
            "execution": serve.get("execution") or serve["engine"].get("execution"), "default_seed": DEFAULT_SEED,
            "engine_record": serve["engine"]}
    built = served_phi(meta, json.loads(pinned_hf_config(serve["model"], serve["revision"]).read_text()), 0)
    phi = PP.phi_of_serve(serve)
    assert phi.to_json() == built.to_json()
    assert phi.engine == {"max_model_len": 512, "max_num_batched_tokens": 512, "max_num_seqs": 8, "enable_prefix_caching": False,
                          "env": {}}
    assert (phi.tp, phi.execution) == ({"world": 1, "rank": 0}, "bi-eager")
    assert PP.serve_ingress(PP.serve_requests(tmp_path, 0)) == bodies
    fitted = {**ENGINE, "effective": {"max_model_len": 4032}, "engine_kwargs": {**ENGINE["engine_kwargs"], "max_model_len": -1}}
    assert PP.phi_of_serve({**serve, "engine": fitted}).engine == {**phi.engine, "max_model_len": 4032}
    assert served_phi({**meta, "engine_record": fitted}, phi.hf_config, 0).engine == {**phi.engine, "max_model_len": 4032}


def test_a_served_pins_phi_reads_its_config_of_record():
    """A pinned served model's φ reads the pin's config.json of record (`pinned_hf_config`), so an int8
    serve compiles on both sides with its quantization config beside the target's int8 scope, never the bf16 fixture's."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config

    rev = "15b0976d400d796ee589e7737d03a919526bf800"
    serve = {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": rev, "execution": "bi-eager",
             "target": {**TARGET, "int8_exact": "linears"}}
    phi = PP.phi_of_serve(serve)
    assert phi.model["hf_config"] == json.loads(pinned_hf_config(serve["model"], rev).read_text())
    assert phi.model["hf_config"]["quantization_config"]["quant_method"] == "verity_int8" and phi.target["int8_exact"] == "linears"
    assert phi.digest() != PP.phi_of_serve(serve, hf_config=str(PP.LLAMA32_1B_CONFIG)).digest()
    bf16 = {**serve, "model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "target": TARGET}
    assert PP.phi_of_serve(bf16).model["hf_config"] == json.loads(PP.LLAMA32_1B_CONFIG.read_text())


def test_the_ran_with_line_names_the_coins_and_the_parameters_of_their_key(tmp_path, capsys):
    """A local challenger's key is at its development parameters; a gate's is its key record's, deployed only at
    `lms.DEPLOYED`'s, and without the record no word is stated. The challenger party's host on this machine, which
    `verity run` hands the path prover as ``loopback:HOST:PORT``, is named a development session's, never an auditor's."""
    from verity.protocols.proof.spot.verifier.challenger import challenger as Ch

    assert PP.ran_with("local") == ("ran with: coins local challenger; parameters development (the local challenger's LMS key "
                                    "at W=4, H=5, not the deployed W=1, H=20)")
    pub = bytes(range(Ch.PUBLIC_BYTES))

    def record(W: int, H: int) -> dict:
        return {"schema": Ch.KEY_RECORD_SCHEMA, "W": W, "H": H, "pubkey": pub.hex(), "pubkey_sha512": hashlib.sha512(pub).hexdigest()}

    gate = "gate:vy@auditor-2"
    assert PP.ran_with(gate, record(1, 20)).endswith("parameters deployed (the challenger's LMS key at W=1, H=20, its key record's)")
    assert "parameters development (the challenger's LMS key at W=4, H=10" in PP.ran_with(gate, record(4, 10))
    assert PP.ran_with(gate) == f"ran with: coins the auditor device's challenger ({gate}); parameters not stated (no challenger key record given)"
    party = "loopback:127.0.0.1:4455"
    assert PP.ran_with(party) == (f"ran with: coins the challenger party's host on this machine ({party}), a development "
                                  "session's, not an auditor's; parameters development (the party's own LMS key, no key "
                                  "record given)")
    at_deployed = PP.ran_with(party, record(1, 20))
    assert at_deployed.endswith("parameters development (the party's own LMS key at W=1, H=20, its key record's)")
    assert "deployed" not in at_deployed and "auditor device" not in at_deployed
    with pytest.raises(ValueError, match="local, loopback:HOST:PORT or gate:USER@HOST"):
        PP.ran_with("remote")
    assert PP.main(["ran-with", "--challenger", party]) == 0
    assert capsys.readouterr().out.strip() == PP.ran_with(party)
    (f := tmp_path / "key.json").write_text(json.dumps(record(1, 20)))
    assert PP.main(["ran-with", "--challenger", gate, "--challenger-key", str(f)]) == 0
    assert capsys.readouterr().out.strip() == PP.ran_with(gate, record(1, 20))


def test_a_served_windows_compiler_is_a_single_gpu_i2s_over_the_servers_code(monkeypatch):
    """The served I2 is named as the server names its compiler, and its digest is a single-GPU window's I2's (one
    `I2Compiler.digest`) with the server's compiler (`served_identity`) as its code: φ (the engine's args among its
    fields), the default seed, ℓ and the served model name move it as they move a single-GPU one's."""
    serve = {"model": "m/x", "revision": "r", "default_seed": DEFAULT_SEED, "execution": "bi-eager", "target": TARGET}
    reg = PP.registered_of_serve(serve, None, None, PP.CAP_BITS)
    one = PP.I2Compiler(reg, {}, "", name=SERVER[0], code=SERVER[1])
    assert one.name == SERVER[0] and one.digest != SERVER[1] and len(one.digest) == 128
    monkeypatch.setattr(PP, "code_identity", lambda: SERVER[1])
    assert PP.I2Compiler(reg, {}, "", name=SERVER[0]).digest == one.digest
    assert PP.I2Compiler(reg, {}, "", name=SERVER[0], code="c0" * 64).digest != one.digest
    others = [PP.registered_of_serve(other, None, None, cap) for other, cap in (
        ({**serve, "model": "m/y"}, PP.CAP_BITS), ({**serve, "revision": "s"}, PP.CAP_BITS),
        ({**serve, "execution": "eager"}, PP.CAP_BITS), ({**serve, "target": {**TARGET, "device_name": "other"}}, PP.CAP_BITS),
        (serve, 2 * PP.CAP_BITS))] + [replace(reg, default_seed=DEFAULT_SEED + 1), replace(reg, served_model="llama")]
    others += [replace(reg, phi=PP.phi_of_serve({**serve, "engine": engine})) for engine in (
        ENGINE, {**ENGINE, "engine_kwargs": {**ENGINE["engine_kwargs"], "max_model_len": 1024}})]
    digests = {PP.I2Compiler(r, {}, "", name=SERVER[0], code=SERVER[1]).digest for r in others}
    assert len(digests) == len(others) and one.digest not in digests


#: a served run's ``serve.json`` edited after its advice was registered, in each field the compile reads of it (`phi_of_serve`,
#: `registered_of_serve`, `served_gates`), and the verifier's own ``--hf-config`` other than the pin's config of record
SERVE_EDITS = {"model": {"model": "verity-int8/Llama-3.2-1B-INT8"}, "revision": {"revision": "0" * 40},
               "default_seed": {"default_seed": DEFAULT_SEED + 1}, "execution": {"execution": "eager"},
               "engine-execution": {"engine": {"execution": "eager", "pin": {"world": 1}}},
               "engine-args": {"engine": {"execution": "bi-eager", "pin": {"world": 1}, "engine_kwargs": {"max_model_len": 512}}},
               "target": {"target": {**TARGET, "device_name": "other-gpu"}}, "hf_config": None}


def test_a_served_run_registers_the_public_default_seed_and_one_that_records_another_or_none_is_refused(tmp_path, monkeypatch,
                                                                                                         capsys):
    """A served window's docket fixes `engine.served_seed.DEFAULT_SEED`, as vllm's registered compiler does: a run
    whose serve.json says its server filled in another seed (7, 0, the public one as a string or a float) or none is refused
    by name, by the served compiler, never compiled under the seed the run recorded."""
    serve = {"model": "m/x", "revision": "r", "default_seed": DEFAULT_SEED, "execution": "bi-eager", "target": TARGET}
    assert PP.registered_seed(serve) == PP.registered_of_serve(serve, None, None, PP.CAP_BITS).default_seed == DEFAULT_SEED
    for seed in (7, 0, None, str(DEFAULT_SEED), float(DEFAULT_SEED)):
        why = f"serve.json: the run's server filled in default seed {seed!r}, and the registered one is engine.served_seed."
        with pytest.raises(ValueError, match=re.escape(why)):
            PP.registered_of_serve({**serve, "default_seed": seed}, None, None, PP.CAP_BITS)
    with pytest.raises(ValueError, match="filled in default seed None"):
        PP.registered_seed({k: v for k, v in serve.items() if k != "default_seed"})
    s = _served(tmp_path, monkeypatch)
    assert s.compiler.inner.registered.default_seed == DEFAULT_SEED
    recorded = json.loads((s.run / "serve.json").read_text())
    (tmp_path / "advice.bin").write_bytes(s.data)
    for doc in ({**recorded, "default_seed": 7}, {k: v for k, v in recorded.items() if k != "default_seed"}):
        (s.run / "serve.json").write_text(json.dumps(doc))
        why = f"filled in default seed {doc.get('default_seed')!r}"
        with pytest.raises(ValueError, match=re.escape(why)):
            PP.i2_of_serve(str(s.run), 0, s.a.row_dir, None, None)


def test_the_served_identity_is_verity_vllm_advices_and_refused_where_the_tree_has_none(monkeypatch):
    import types

    PP.served_identity.cache_clear()
    fake = types.ModuleType("verity_vllm.pipeline.advice")
    fake.registered_compiler = lambda: SimpleNamespace(name=PP.CAPPED, digest="ff" * 64, inner=SimpleNamespace(name=SERVER[0], digest=SERVER[1]))
    monkeypatch.setitem(sys.modules, "verity_vllm.pipeline.advice", fake)
    assert PP.served_identity() == SERVER
    PP.served_identity.cache_clear()
    fake.registered_compiler = lambda: SimpleNamespace(name=SERVER[0], digest=SERVER[1])
    with pytest.raises(ValueError, match="registers no capped compiler"):
        PP.served_identity()
    monkeypatch.setitem(sys.modules, "verity_vllm.pipeline.advice", None)
    with pytest.raises(ValueError, match="which this tree lacks"):
        PP.served_identity()
    PP.served_identity.cache_clear()


def test_the_served_compiler_is_what_the_registration_must_name(tmp_path, monkeypatch, capsys):
    s = _served(tmp_path, monkeypatch)
    doc = PP.compiler_doc(str(s.run), None, None, PP.CAP_BITS)
    assert {"name": doc["name"], "digest": doc["digest"]} == s.bound["compiler"] == {"name": PP.CAPPED, "digest": s.compiler.digest}
    assert doc["served"] == {"name": SERVER[0], "digest": SERVER[1]} and doc["default_seed"] == DEFAULT_SEED and doc["phi"]["target"] == TARGET
    assert doc["transfer_gates"] == list(PP.GATES_OF_RECORD["unsloth/Llama-3.2-1B"])
    assert PP.compiler_doc(str(s.run), None, None, 2 * PP.CAP_BITS)["digest"] != doc["digest"]
    capsys.readouterr()
    assert PP.main(["compiler", "--serve-run", str(s.run)]) == 0
    assert json.loads(capsys.readouterr().out)["digest"] == doc["digest"]


#: `engine.int8_checkpoint`'s v1 recipe's quantization config, the one Llama-3.2-1B-INT8's config.json carries
INT8_V1 = {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 127}


def _tiny_checkpoint(d: Path, model: str, revision: str, int8: bool = False) -> Path:
    """TINY's checkpoint under ``d`` (random bf16 weights by their HF names, one safetensors shard, and its config.json) and a
    manifest pinning the shard at ``model`` and ``revision``: the manifest. ``int8``: as `engine.int8_checkpoint`'s v1 recipe
    makes one, each decoder linear an I8 weight under its row rule and its F32 ``weight_scale`` [N, 1], under ``INT8_V1``."""
    import struct

    import numpy as np

    rng, H, I, D = np.random.default_rng(3), TINY["hidden_size"], TINY["intermediate_size"], 16

    def bf16(*shape):
        return (rng.random(shape, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def linear(name, n, k):
        if not int8:
            return {name + ".weight": bf16(n, k)}
        w = rng.random((n, k), dtype=np.float32) - np.float32(0.5)
        s = np.maximum(np.abs(w).max(axis=1, keepdims=True), np.float32(1e-30)) / np.float32(127)
        return {name + ".weight": np.clip(np.rint(w / s), -127, 127).astype(np.int8), name + ".weight_scale": s}
    t = {"model.embed_tokens.weight": bf16(TINY["vocab_size"], H), "model.norm.weight": bf16(H)}
    for i in range(TINY["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name, n, k in (("self_attn.q_proj", 4 * D, H), ("self_attn.k_proj", 2 * D, H), ("self_attn.v_proj", 2 * D, H),
                           ("self_attn.o_proj", H, 4 * D), ("mlp.gate_proj", I, H), ("mlp.up_proj", I, H), ("mlp.down_proj", H, I)):
            t |= linear(p + name, n, k)
        t |= {p + "input_layernorm.weight": bf16(H), p + "post_attention_layernorm.weight": bf16(H)}
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    for name, a in t.items():
        hdr[name] = {"dtype": {"uint16": "BF16", "int8": "I8", "float32": "F32"}[a.dtype.name], "shape": list(a.shape),
                     "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    head = json.dumps(hdr).encode()
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"".join(a.tobytes() for a in t.values()))
    (d / "config.json").write_text(json.dumps({**TINY, "rope_theta": 10000.0, **({"quantization_config": INT8_V1} if int8 else {})}))
    sha = hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
    man = d.parent / "checkpoints.json"
    man.write_text(json.dumps({"checkpoints": [{"repo": model, "revision": revision, "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": sha}}}]}))
    return man


def test_a_served_runs_weights_registration_is_the_committers_under_fresh_salts(tmp_path, capsys, monkeypatch):
    """``weights-registration``, the committer's: the served run's weights (`window_weights.weights_registration` under the
    φ its window circuit lowers under, here at a grain TINY lowers at) committed under fresh OS salts kept in the salt file
    beside ``--out`` (0700, no salt in the registration), once the committer has checked the checkpoint's shards
    against its manifest. Written to ``--out``: a root for each weight parameter but the rotary table's, what `verity-vllm
    advice --weights-of-record` records and ``window-circuit --pins`` takes (the embedding's table its own, its ``row-seg``
    rows), and what the verifier holds to its own weights
    record's shapes (`registration_pins`) without computing a root. With the engine's served rotary table in the run's
    ``private/cos_sin_cache.npy`` (serve-openai's) it roots that too, and refuses by name a table outside the analytic
    table's bound. Registering again gives other roots for every weight; a checkpoint the manifest pins as another model
    than the run served, or a shard other than the one the manifest pins, is refused by name, nothing registered."""
    import functools

    import numpy as np

    from verity_vllm.pipeline import advice as ADV
    from verity_vllm.program import model as M
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.window_weights import Grain

    grain = Grain(gemm_words=512, rows=18, select=64)
    monkeypatch.setattr(WW, "weights_registration", functools.partial(WW.weights_registration, grain=grain))
    run, hf, ck = tmp_path / "serve", tmp_path / "tiny.json", tmp_path / "ck" / "snap"
    run.mkdir()
    serve = {"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "default_seed": DEFAULT_SEED,
             "execution": "bi-eager", "target": {"compute_capability": [12, 0], "num_sms": 188}}
    (run / "serve.json").write_text(json.dumps(serve))
    hf.write_text(json.dumps(TINY))
    man = _tiny_checkpoint(ck, serve["model"], serve["revision"])
    out = tmp_path / "weights_registration.json"
    salts = SF.beside(out)

    def register() -> int:
        return PP.main(["weights-registration", "--serve-run", str(run), "--checkpoint-dir", str(ck), "--checkpoints",
                        str(man), "--hf-config", str(hf), "--out", str(out)])

    capsys.readouterr()
    assert register() == 0
    reg, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    assert salts.stat().st_mode & 0o777 == 0o700 and set(SF.params(salts)) == set(reg["values"])
    assert all(SF.read(salts, q, [0]).hex() not in out.read_text() for q in reg["values"])
    pins = ADV.registered_weights(reg, serve)
    assert said["roots"] == len(pins) == 16 and PP.pins_of((), [str(out)]) == pins
    assert [p for p in pins if p.endswith("/table")] == ["weights/model.embed_tokens.weight@64/table"]
    assert said["unrooted"] == sorted(reg["unrooted"]) and all("rotary_emb" in p for p in said["unrooted"])
    registered = PP.registered_of_serve(serve, str(hf), None, PP.CAP_BITS)
    record = WW.weights_record(registered, str(ck), str(man), grain=grain)
    assert WW.registration_pins(reg, record) == pins
    assert register() == 0
    again = ADV.registered_weights(json.loads(out.read_text()), serve)
    assert sorted(again) == sorted(pins) and all(again[q] != pins[q] for q in pins)
    reg = json.loads(out.read_text())
    C = next(f for f in record["fields"] if f.get("pinned") == "analytic")["analytic"]["C"]
    table = run / "private" / "cos_sin_cache.npy"
    table.parent.mkdir()
    np.save(table, M.cos_sin_table(C))
    capsys.readouterr()
    assert register() == 0
    rooted, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    assert said["roots"] == 17 and said["unrooted"] == [] == list(rooted["unrooted"])
    assert set(rooted["values"]) == set(reg["values"]) | set(reg["unrooted"])
    assert WW.registration_pins(rooted, record) == ADV.registered_weights(rooted, serve)
    np.save(table, M.cos_sin_table({**C, "ROPE_THETA": 2 * C["ROPE_THETA"]}))
    assert register() == PP.REFUSED
    assert "served words outside the analytic table's bound" in capsys.readouterr().err
    table.unlink()
    out.unlink()
    (run / "serve.json").write_text(json.dumps({**serve, "model": "org/other"}))
    assert register() == PP.REFUSED
    assert f"manifest entry is {serve['model']}@{serve['revision']}, not the registered model org/other@" in capsys.readouterr().err
    assert not out.exists()
    (run / "serve.json").write_text(json.dumps(serve))
    shard = ck / "model.safetensors"
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(bytes(body))
    assert register() == PP.REFUSED
    assert "the checkpoint is not of record" in capsys.readouterr().err and not out.exists()


def test_the_weights_registration_is_also_the_first_record_of_the_serves_registration_log(tmp_path, capsys, monkeypatch):
    """``weights-registration --deployment-record FILE`` writes the committer's registration also as the first record of the
    serve's registration log (`window_weights.deployment_record`): a `verity/registration/v2` record chained after nothing
    that registers each rooted weight parameter's entry as the docket states it, by name, and names the unrooted
    ones in its header, with none of its salts; the command says its digest (`registration.digest`), which Pearl+'s W
    names as its ``previous`` on a serve tied to it. So each weight's root is in the log's first record under its value's
    name, where the inference-only policy's check looks a registered parameter up, by name and root, in the log it is
    given (its Lean's `isTheta`). Without the flag no such record is
    written; a docket with a value that is no weight parameter gives none, and a name under a weight's with a
    suffix `_param` takes no form of (``/bnk``, ``/b<b>/table``, ``/b<b>/vocab``) is such a value: the record is refused
    whole, never written without the name."""
    import functools

    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.window_weights import Grain

    grain = Grain(gemm_words=512, rows=18, select=64)
    monkeypatch.setattr(WW, "weights_registration", functools.partial(WW.weights_registration, grain=grain))
    run, hf, ck = tmp_path / "serve", tmp_path / "tiny.json", tmp_path / "ck" / "snap"
    run.mkdir()
    serve = {"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "default_seed": DEFAULT_SEED,
             "execution": "bi-eager", "target": {"compute_capability": [12, 0], "num_sms": 188}}
    (run / "serve.json").write_text(json.dumps(serve))
    hf.write_text(json.dumps(TINY))
    man = _tiny_checkpoint(ck, serve["model"], serve["revision"])
    out, first = tmp_path / "weights_registration.json", tmp_path / "first.json"
    argv = ["weights-registration", "--serve-run", str(run), "--checkpoint-dir", str(ck), "--checkpoints", str(man),
            "--hf-config", str(hf), "--out", str(out)]
    capsys.readouterr()
    assert PP.main(argv) == 0 and "deployment" not in json.loads(capsys.readouterr().out) and not first.exists()
    assert PP.main([*argv, "--deployment-record", str(first)]) == 0
    reg, rec, said = json.loads(out.read_text()), json.loads(first.read_text()), json.loads(capsys.readouterr().out)
    assert rec == WW.deployment_record(reg) and said["deployment"] == DD.digest(rec) and said["roots"] == 16
    assert (rec["format"], rec["protocol"], rec["previous"]) == (DM.REGISTRATION, DM.ONE_STAGE_PROTOCOL, None)
    assert [e["value"] for e in rec["registered"]] == sorted(reg["values"])
    assert {e["value"]: e for e in rec["registered"]} == reg["values"]
    assert all(set(e) == {"value", "root", "leaves", "words", "domain"} for e in rec["registered"])
    assert rec["deployment"] == {"kind": WW.DEPLOYMENT, "program": reg["program"], "grain": reg["grain"],
                                 "model": {"repo": serve["model"], "revision": serve["revision"]},
                                 "unrooted": sorted(reg["unrooted"])}
    assert rec["deployment"]["unrooted"] and not {e["value"] for e in rec["registered"]} & set(reg["unrooted"])
    assert all(SF.read(SF.beside(out), q, [0]).hex() not in first.read_text() for q in reg["values"])
    stem = next(iter(reg["values"])).rpartition("@")[0]
    for name in ("x", f"{stem}@192/bnk", f"{stem}@3072/b0/table", f"{stem}@192/b0/vocab"):
        odd = {**reg, "values": {**reg["values"], name: {**next(iter(reg["values"].values())), "value": name}}}
        with pytest.raises(ValueError, match=re.escape(f"{name!r} is no weight parameter")):
            WW.deployment_record(odd)
    with pytest.raises(ValueError, match="never a weights record"):
        WW.deployment_record({"schema": "other"})


def test_an_int8_served_runs_weights_registration_roots_its_scales_fused_linears_and_tied_embedding(tmp_path, capsys, monkeypatch):
    """The int8 Llama-3.2-1B is an exact-int8 checkpoint (`engine.int8_checkpoint`'s v1: each decoder linear an s8 weight and an f32
    row scale, the embedding bf16 and tied to the LM head), served under a target whose ``int8_exact`` is ``linears``, and
    ``weights-registration`` roots its weight parameters as the window circuit reads them: the s8 weights and their scales of
    q|k|v and gate|up fused as vLLM serves them, o and down, the norms, and the tied embedding at the embedding's and the LM
    head's row widths and as the embedding's table (its own parameter, `window_builder.table_param`), each the checkpoint's
    words; only the rotary table is unrooted. The verifier holds it to its own weights record's shapes (`registration_pins`)."""
    import functools

    from verity_vllm.pipeline import advice as ADV
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.window_weights import Grain

    grain = Grain(gemm_words=512, rows=18, select=64)
    monkeypatch.setattr(WW, "weights_registration", functools.partial(WW.weights_registration, grain=grain))
    run, hf, ck = tmp_path / "serve", tmp_path / "tiny-int8.json", tmp_path / "ck" / "snap"
    run.mkdir()
    serve = {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": "15b0976d400d796ee589e7737d03a919526bf800", "default_seed": DEFAULT_SEED,
             "execution": "bi-eager", "target": {"compute_capability": [12, 0], "num_sms": 188, "int8_exact": "linears"}}
    (run / "serve.json").write_text(json.dumps(serve))
    hf.write_text(json.dumps({**TINY, "quantization_config": INT8_V1}))
    man = _tiny_checkpoint(ck, serve["model"], serve["revision"], int8=True)
    out = tmp_path / "weights_registration.json"
    capsys.readouterr()
    assert PP.main(["weights-registration", "--serve-run", str(run), "--checkpoint-dir", str(ck), "--checkpoints", str(man),
                    "--hf-config", str(hf), "--out", str(out)]) == 0
    reg, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    pins = ADV.registered_weights(reg, serve)
    assert said["roots"] == len(pins) == 24 and said["unrooted"] == ["weights/model.layers.0.self_attn.rotary_emb.cos_sin_cache@16"]
    assert [p for p in pins if p.endswith("/table")] == ["weights/model.embed_tokens.weight@64/table"]
    words = {p.removeprefix("weights/"): e["words"] * e["leaves"] for p, e in reg["values"].items() if not p.endswith("/table")}
    H, I, QKV = TINY["hidden_size"], TINY["intermediate_size"], 8 * 16
    for i in range(TINY["num_hidden_layers"]):
        p = f"model.layers.{i}."
        assert {k: v for k, v in words.items() if k.startswith(p) and "_proj" in k} == {
            p + "self_attn.qkv_proj.weight@512": QKV * H // 2, p + "self_attn.qkv_proj.weight_scale@32": 2 * QKV,
            p + "self_attn.o_proj.weight@512": H * H // 2, p + "self_attn.o_proj.weight_scale@32": 2 * H,
            p + "mlp.gate_up_proj.weight@512": 2 * I * H // 2, p + "mlp.gate_up_proj.weight_scale@32": 2 * 2 * I,
            p + "mlp.down_proj.weight@512": H * I // 2, p + "mlp.down_proj.weight_scale@16": 2 * H}
    assert words["model.embed_tokens.weight@64"] == words["model.embed_tokens.weight@512"] == TINY["vocab_size"] * H
    assert not any("lm_head" in p for p in words)
    registered = PP.registered_of_serve(serve, str(hf), None, PP.CAP_BITS)
    assert WW.registration_pins(reg, WW.weights_record(registered, str(ck), str(man), grain=grain)) == pins


def test_the_transfer_gates_of_a_served_model_are_its_compiler_of_records():
    """`served_gates`: Llama-3.2-1B's compiler of record lists its RoPE heads (the TP1 stand-in), ``--transfer-gate`` may restate
    the list and nothing else (another template, or the record's with one more), and a model with no line lists none."""
    llama, tg = {"model": "unsloth/Llama-3.2-1B"}, "RoPEHead_v1{D=64}"
    assert PP.served_gates(llama) == PP.served_gates(llama, [tg, tg]) == (tg,)
    for other in (["SiluMul_v2{I=8192}"], [tg, "SiluMul_v2{I=8192}"]):
        with pytest.raises(ValueError, match=r"unsloth/Llama-3.2-1B's compiler of record lists \['RoPEHead_v1\{D=64\}'\]"):
            PP.served_gates(llama, other)
    assert PP.served_gates({"model": "m/x"}) == ()
    with pytest.raises(ValueError, match=r"m/x's compiler of record lists \[\]"):
        PP.served_gates({"model": "m/x"}, [tg])


def _wstep(*rows: tuple[int, int, int]) -> WA.StepAdvice:
    return WA.StepAdvice(tuple(WA.Row(r, "decode" if n == 1 and s >= 3 else "prefill", s, n, 0, False, ()) for r, s, n in rows))


QA, QB = WA.RequestAdvice("a", 3, 4, "length"), WA.RequestAdvice("b", 3, 3, "length")
# window 0 admits a and b (prompts of 3) and c, which finishes in it; a keeps 2 outputs and b 1, so both must carry
W0 = WA.Advice((QA, QB, WA.RequestAdvice("c", 3, 1, "eos")), (_wstep((0, 0, 3), (1, 0, 3)), _wstep((0, 3, 1), (2, 0, 3))),
               (0, 0, 1, 1))
AT_A, AT_B = WA.Carried(0, 4, 2, ((0, 4),)), WA.Carried(0, 3, 1, ((0, 3),))
ON = (_wstep((0, 4, 1)), _wstep((1, 3, 1)))
# window 1 carries both on from where window 0 left them
W1 = WA.Advice((QA, QB), ON, (0, 1), (AT_A, AT_B))


def test_window_commit_loads_its_own_builds_beside_its_out_while_the_row_dir_holds_none(tmp_path, monkeypatch):
    """At a tied window's close (`serve_tie`, before its sample coin) the row dir holds the seal's commit and no Build:
    so window-commit's I2 loads each Program from its own Build,
    written from its compile with its names and no vLLM (`own_builds`) into a directory of its own beside ``--out``, gone
    once it returns, and writes none into the row dir."""
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict import window_weights  # noqa: F401  imported before `lower`'s stub, which it binds by name

    monkeypatch.setitem(sys.modules, "vllm", None)
    s = _tiny_served(tmp_path, monkeypatch)
    out, seen, load = tmp_path / "v3c-w0" / "window", {}, PP.I2Compiler.load

    class Loaded(Exception):
        pass

    def loaded(i2, rank, world, requests):
        p = load(i2, rank, world, requests)
        seen.update(root=Path(i2.programs_root), held={Path(d).parent for d in p.dirs.values()}, digs=p.digs)
        raise Loaded

    monkeypatch.setattr(PP.I2Compiler, "load", loaded)
    monkeypatch.setattr(WC, "takes", lambda *a, **k: None)
    monkeypatch.setattr(WC, "lower", lambda *a, **k: SimpleNamespace(circuit=None))
    monkeypatch.setattr(WC, "check_served", lambda *a, **k: {})
    (tmp_path / "advice.bin").write_bytes(s.data)
    with pytest.raises(Loaded):
        PP.main(["window-commit", "--serve-run", s.a.serve_run, "--window", "0", "--row-dir", s.a.row_dir, "--advice",
                 str(tmp_path / "advice.bin"), "--registration", "unread",
                 "--checkpoint-dir", "unread", "--sampling-design", "rate:1/4", "--out", str(out)],
                kept_ingress=lambda serve_dir: None, scan_state_file="unread")
    assert seen["root"].parent == out.parent and seen["held"] == {seen["root"]} and not seen["root"].exists()
    assert seen["digs"] == {u.id: u.digest for u in s.units} and s.compiles == [True, False]
    assert not list(Path(s.a.row_dir).glob("build_request*"))


def test_window_commit_without_the_attesters_reader_or_state_file_is_refused_by_name():
    """SPOT imports neither the attester nor a benchmark, so window-commit takes the reader of the run a serve kept and
    the name the attester's scan reads the opened scan state under from its caller, and refuses by name without them."""
    for given in ({}, {"kept_ingress": lambda serve_dir: None}, {"scan_state_file": "unread"}):
        with pytest.raises(ValueError, match="by what its caller passes .*run benchmarks/one_stage/window_commit.py"):
            PP.window_commit_of_builds(argparse.Namespace(), **given)


#: adapters/vllm's `tests.program.predict.test_steps.TINY`, as test_class_rows has it
TINY_CONFIG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
               "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256,
               "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}


@pytest.mark.slow
def test_window_commit_fills_the_served_circuits_slots_and_rows_from_its_served_requests(tmp_path, monkeypatch):
    """`window_commit_of_builds` on TINY's served window at class 2,4,2 (r0 at a prompt of 4 keeping 1 output, r1 at 4
    keeping 2): it lowers the served circuit of the requests it takes, builds one `Slot` per Program from the slot its
    anchor and lane name, each of its served request's values, registers the weights and the scan state, and fills the
    circuit's every unit (`Rows`), up to the window it closes. The record, store, checkpoint and scan readers are stood
    in, and TINY's templates are below what Flock's composer proves; the lowering, the slots, the weights' values and
    the rows are window-commit's own. Slow (about two minutes) for TINY's rows, which every unit of the circuit fills:
    what no cheaper test reaches is window-commit's own path from the lowering to the rows, which none runs past a slot."""
    import numpy as np
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check import weights_of_record as WOR
    from verity_vllm.check.replay import index as RI
    from verity_vllm.check.replay import opening as OP
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.program.predict import scan_state as SS
    from verity_vllm.program.predict import steps as ST
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict import window_rows as WRS
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.phi import Phi

    grain = WC.Grain(gemm_words=1024, rows=256, select=64)
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY_CONFIG},
              target={"compute_capability": [12, 0], "num_sms": 188})
    registered, cls = ST.Registered.of(phi, default_seed=7), WC.WindowClass(2, 4, 2)
    served = {"r0": (4, 1, [1, 4, 17, 3, 250, 9]), "r1": (4, 2, [1, 4, 5, 77, 42, 0])}
    requests = tuple(WC.Request(cls.anchor(k), cls.lane(k), lp, m) for k, (lp, m, _row) in enumerate(served.values()))
    lower, weight_values, rows_of = WC.lower, WW.weight_values, WRS.Rows
    widths, zeros = {8: np.uint8, 16: np.uint16, 32: np.uint32}, {}

    def lowered(r, c, programs=None, g=grain, requests=None):
        low = lower(r, c, programs, g, requests=requests)
        zeros.update({label: np.zeros(ty.leaves, dtype=widths[next(iter(ty.width_set()))])
                      for label, ty in dict(low.fn.signature()[0])[WC.WEIGHTS].fields})
        return low

    monkeypatch.setattr(WC, "lower", lowered)
    monkeypatch.setattr(WC, "takes", lambda compiled, c: requests)
    monkeypatch.setattr(WC, "check_served", lambda circuit: {})
    monkeypatch.setattr(WW, "weight_values", lambda r, s, _reg, keep_rows=False: weight_values(r, s, None, grain, keep_rows=keep_rows))
    monkeypatch.setattr(ST, "compile_steps", lambda *a, **k: SimpleNamespace(requests=[SimpleNamespace(id=i) for i in served]))
    p = SimpleNamespace(progs={i: i for i in served}, dirs={i: str(tmp_path) for i in served}, arrivals={i: 0 for i in served})
    compiler = SimpleNamespace(inner=SimpleNamespace(registered=registered, ingress={}), load=lambda *a: p)
    monkeypatch.setattr(PP, "compiler_of_args", lambda a: compiler)
    monkeypatch.setattr(PP, "compile_advice", lambda c, d: [])
    monkeypatch.setattr(PP, "kept_scan", lambda *a, **k: None)
    monkeypatch.setattr(PP, "place_slots", lambda *a: [(i, row, m) for i, (_lp, m, row) in served.items()])
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "tree.json").write_text("{}")
    (tmp_path / "binding_map_p0.json").write_text("{}")
    monkeypatch.setattr(PP, "record1", lambda *a: SimpleNamespace(
        run=SimpleNamespace(run_root=bytes(64)), bdir=tmp_path, row={"commit": {}}, out=tmp_path, pair=0, store=None,
        manifest=None, aliases=None))
    monkeypatch.setattr(WOR, "CheckpointIndex", lambda *a, **k: None)
    monkeypatch.setattr(CPU, "OnRead", lambda *a: None)
    monkeypatch.setattr(CPU, "provider", lambda *a: lambda label, lo, hi: zeros[label][lo:hi])
    monkeypatch.setattr(RI, "ProgramIndex", lambda *a, **k: SimpleNamespace(prefix=""))
    monkeypatch.setattr(OC, "OpenedReader", lambda *a: None)
    monkeypatch.setattr(OP.CommittedStore, "of_record", staticmethod(lambda *a: None))
    stand_in = {i: (lp, lp + m - 1) for i, (lp, m, _row) in served.items()}
    class Computed:
        """A served request whose every node the committer computes."""

        def __init__(self, rid: str) -> None:
            self.LP, self.positions = stand_in[rid]

        def __call__(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(WRS.Served, "of_record", classmethod(lambda c, pi, store, rid, arrive: Computed(rid)))
    monkeypatch.setattr(WW.WeightsRegistration, "of", staticmethod(
        lambda *a: SimpleNamespace(repo="org/m", revision="r", values={}, unrooted=[])))
    filled = {}

    class Closed(Exception):
        pass

    def rows(low, slots, values):
        filled.update(low=low, slots=slots, values=values, rows=rows_of(low, slots, values))
        return filled["rows"]

    def window(*a, **k):
        filled.update(advice=k.get("advice"))
        raise Closed

    bound = {"compiler": {"name": "c", "digest": "cd" * 64}, "advice": {"root": "ef" * 64, "bits": 0}}
    monkeypatch.setattr(PP, "served_binding", lambda a, c, data, r: bound)
    monkeypatch.setattr(WRS, "Rows", rows)
    monkeypatch.setattr(IF, "Window", window)
    (tmp_path / "advice.bin").write_bytes(b"")
    a = argparse.Namespace(sampling_design="rate:1/4", advice=str(tmp_path / "advice.bin"), window_class="2,4,2", cap=64, bundle="unread",
                           row_dir="unread", checkpoint_dir="unread", checkpoints=None, serve_run=str(tmp_path),
                           registration=str(tmp_path / "registration.json"), window=0, workers=1, out=str(tmp_path / "out"))
    (tmp_path / "registration.json").write_text("{}")
    with pytest.raises(Closed):
        PP.window_commit_of_builds(a, kept_ingress=lambda serve_dir: None, scan_state_file="unread")
    low, slots = filled["low"], filled["slots"]
    assert [P.positions for P in low.programs] == [5, 4] and len(slots) == 2
    assert [s.fn for s in slots] == [P.fn for P in low.programs]
    assert SS.SCAN_STATE in filled["values"] and sum(filled["rows"].n.values()) > 0
    assert filled["advice"] is bound


def test_a_served_window_commit_binds_the_advice_the_server_committed(tmp_path, monkeypatch):
    """`served_binding`: the window commit's advice binding is the server's own (the root its opening opens, under the served
    compiler), committed nothing a second time; advice bytes other than the opening's are refused by name, and so is a
    docket binding another root."""
    s = _served(tmp_path, monkeypatch)
    a = argparse.Namespace(**vars(s.a), advice=str(tmp_path / "advice.bin"))
    r = PP.record1(s.a.bundle, s.a.row_dir)
    assert PP.served_binding(a, s.compiler, s.data, r) == s.bound
    with pytest.raises(ValueError, match="^advice: .* other advice than window 0's committed opening"):
        PP.served_binding(a, s.compiler, s.data + b"\0", r)
    reg = s.run / "advice" / "w0" / "registration.json"
    doc = json.loads(reg.read_text())
    reg.write_text(json.dumps({**doc, "binding": {**doc["binding"], "advice": {**doc["binding"]["advice"], "root": "ee" * 64}}}))
    with pytest.raises(ValueError, match=r"published records are refused \(advice\)"):
        PP.served_binding(a, s.compiler, s.data, r)


def test_window_commits_design_is_the_one_its_rate_names_over_the_circuit_and_a_fixed_count_is_refused():
    """`window_design`: window-commit's ``--sampling-design`` is the rate the window's audit states to its stream, and the window is
    committed under the sampling design that rate names over the circuit's own proof blocks, so its docket is under
    the design the stream holds it to: a subset design under a rate, one K for every stratum under a stratified rate, at
    1 where the rate's denominator is no smaller than the largest stratum. A fixed count, which no stream takes, is
    refused by name."""
    from verity.circuits.defs import bind
    from verity.circuits.ops import gemm as G
    from verity.protocols.proof.spot.verifier import service as S

    small, large = (bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8) for k in (64, 128))
    circuit = IF.Circuit("test", ((small, 3), (large, 37)))
    program, query = circuit.program()
    assert PP.window_design("rate:1/40", circuit) == {"law": "subset", "k": 1}
    assert PP.window_design("rate:1/8", circuit) == {"law": "subset", "k": 5}
    assert PP.window_design("stratified-rate:1/40", circuit) == {"law": "stratified", "k": 1}
    dense = PP.window_design("stratified-rate:1/8", circuit)
    assert dense["law"] == "stratified" and dense["k"] > 1 and dense == S.params_sampling_design("stratified-rate:1/8", program, query, None)[0]
    for fixed in ("subset:1", "stratified:1", "work:1"):
        with pytest.raises(ValueError, match="a fixed count") as e:
            PP.window_design(fixed, circuit)
        assert repr(fixed) in str(e.value)
