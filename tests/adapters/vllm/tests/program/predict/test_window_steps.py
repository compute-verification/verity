"""The runner's window steps (`served_verdicts.sh steps`, a plain run's) through `verity run`'s own calls
(`served_table.served_window`) on a window's record made on the processor."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_cli import run as RUN
from verity_cli import served_table as ST

from tests.program.predict.test_close_record import _close
from tests.program.predict.test_window_rows import LANES
from tests.program.predict.test_window_rows import RUN as RUN_ROOT
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.predict.close_record import Record

ROOT = Path(__file__).parents[6]
COMPILER = {"name": "c", "digest": "cd" * 64}
#: what a card's serve writes in serve.json beside the application's checkpoint and its device (`serve_seal.target_of`), the
#: default profile standing in for the card's
SERVED = {"execution": "bi-eager", "default_seed": DEFAULT_SEED,
          "engine": {"execution": "bi-eager", "engine_kwargs": {"max_num_batched_tokens": 512}}}
PROGRAM, WORKLOAD = "a1" * 64, "12" * 64


def _serve_dir(tmp_path, monkeypatch, application):
    """TINY's window at class 2,4,2 closed as a serve's close leaves it (`Record.handoff`, then `Record.make` from the
    handoff alone, as close_record's finish does after the lease), under ``tmp_path/serve``; what only a card's serve writes
    beside it (its configuration of ``application``'s checkpoint, its requests, seal, commit verdict, advice registration
    and predict row's Build) stood in."""
    d = tmp_path / "serve"
    low, units, served, plan, binding, compiled = _close(d, monkeypatch)
    rec = Record("rate:1/4", "unread", tmp_path / "salt_stream", sample=2, workers=2)
    h = rec.handoff(plan, 0, low, units, binding)
    got = rec.make(plan, 0, compiled, low, units, served, SimpleNamespace(rc=binding.rc, com=None), handoff=Path(h["out"]))
    line = {"window": 0, "advice": "ef" * 64, "handoff": {"out": h["out"]}, "record": {"pending": "made after the lease"}}
    (d / "plans.jsonl").write_text(json.dumps(line) + "\n")
    params = RUN.load(ROOT / "applications" / f"{application}.toml", RUN.registry())["workload"]["params"]
    checkpoint = {k: params[k] for k in ("model", "revision", "precision")}
    (d / "serve.json").write_text(json.dumps({"scheme": "rows-sha512", **checkpoint, **SERVED,
                                              "target": TargetProfile.default().to_json()}))
    (d / "requests.jsonl").write_text("".join(json.dumps({"user": f"q{k}", "windows": [0], "prompt_ids": p[:lp],
                                                         "ingress": {"brought": {"max_tokens": m}}}) + "\n"
                                              for k, (_o, lp, m, p) in enumerate(LANES)))
    (d / "seals.jsonl").write_text(json.dumps({"window": 0, "run_root": RUN_ROOT}) + "\n")
    commit = d / "windows" / "w0" / "commit"
    commit.mkdir(parents=True)
    (commit / "verdict.json").write_text(json.dumps({"run_root": got["run"], "commit_scheme": "rows-sha512"}))
    (commit / "runs.jsonl").write_text(json.dumps({"commit": {"binding": {"program_digest": PROGRAM}}}) + "\n")
    (d / "advice" / "w0" / "registration.json").write_text(json.dumps(
        {"compiler": {**COMPILER, "cap_bits": 64}, "row_bits": 8, "binding": {"advice": {"root": "ef" * 64, "bits": 0}}}))
    (d / "advice" / "w0" / "private" / "advice.bin").write_bytes(b"")
    (d / "weights_registration.json").write_text("{}")
    (tmp_path / "predicted").mkdir()
    (tmp_path / "predicted" / "predicted.json").write_text(json.dumps({"compiler": COMPILER, "workload_digest": WORKLOAD}))
    return d


def _road(tmp_path, monkeypatch, application, d, law=None, **env):
    """``application``'s runner as `verity run`'s accounting and policy stages launch it on a node whose environment names
    the serve dir, the auditor's sampling design ``law`` where given: the accounting call's exit, both calls' lines in that
    order and its public record (``state/out``)."""
    cfg = RUN.load(ROOT / "applications" / f"{application}.toml", RUN.registry())
    if law is not None:
        cfg = {**cfg, "service": {**cfg["service"], "law": law}}
    for k, v in {"D": str(d), "PRED": str(tmp_path / "predicted"), "SERVE_ROOT": str(tmp_path), "WINDOW_CLASS": "2,4,2",
                 "VENV": sys.prefix, "CUDA_HOME": str(tmp_path / "no-cuda"), "HF_HOME": str(tmp_path / "hf"), **env}.items():
        monkeypatch.setenv(k, v)
    state = tmp_path / f"state-{application}"
    rc = ST.served_window(cfg, state, {}, ST.ACCOUNTING, "served-window")
    if rc in ST.ENDED:
        ST.served_window(cfg, state, {}, ST.POLICY, "policy")
    logs = [state / "logs" / f"{x}.log" for x in ("served-window", "policy")]
    return rc, "".join(f.read_text() for f in logs if f.is_file()), state / "out"


def _digests(doc, path=""):
    """Each 128-hex string of ``doc`` with its path."""
    if isinstance(doc, dict):
        return [x for k, v in doc.items() for x in _digests(v, f"{path}.{k}" if path else str(k))]
    if isinstance(doc, list):
        return [x for i, v in enumerate(doc) for x in _digests(v, f"{path}[{i}]")]
    return [(path, doc)] if isinstance(doc, str) and len(doc) == 128 and set(doc) <= set("0123456789abcdef") else []


def _plain(d):
    """Every plain digest the window's registration carries (in its record's registration.json and index.json, and the
    registration's own docket digest), by path, the registered compiler's digest among them: only a hiding commitment's
    root may leave."""
    from verity.commitments import docket_digest as DD

    record = d / "windows" / "w0" / "record"
    reg = json.loads((record / "registration.json").read_text())
    found = {f"registration.json {k}": v for k, v in _digests(reg)}
    found |= {f"index.json {k}": v for k, v in _digests(json.loads((record / "index.json").read_text()))}
    found["the registration's docket digest"] = DD.digest(reg)
    return {k: v for k, v in found.items()
            if not any(w in ("run", "root", "roots") or w.endswith(("_root", "_roots")) for w in re.split(r"[ .\[]", k))}


def _secret(d):
    """The declared circuit's description as the auditor must not see it: its templates' names and its digests, the
    registered compiler's included."""
    record = d / "windows" / "w0" / "record"
    doc = json.loads((record / "circuit.json").read_text())
    return {str(t) for t, _ in doc["templates"]} | {hashlib.sha512((record / "circuit.json").read_bytes()).hexdigest(),
                                                    PROGRAM, WORKLOAD} | set(_plain(d).values())


def _published(out):
    return [(str(f.relative_to(out)), f.read_text(errors="replace")) for f in sorted(out.rglob("*")) if f.is_file()]


def _end(log):
    return next(x for x in log.splitlines() if x.startswith("ROAD END"))


#: what no public line says of the window's advice or its compiler, which nothing checks (candidate 135)
UNSAID = ("compiled by the registered compiler", "charged", "bounded by", "capped at")


@pytest.mark.slow
def test_the_window_steps_runs_remote_from_the_record_and_publishes_no_description(tmp_path, monkeypatch):
    """remote through the preflight on the window's record (no ROW read), its seal held to the record and predict row's
    compiler, each step after it run or refused by name: the sampled proofs refused or not run on this fixture, each saying why, and never an accept, Pearl+ as needing a card (a processor has no gpu-lease), the
    record and the composite after them; the block counts count the record's circuit beside the registration that binds
    it. The end line counts what ran, what refused or was undecided and what did not run, and neither it nor any
    published file names a template or carries the description's digest. What no cheaper test reaches: the runner itself
    on a real window's record. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch, "remote")
    (tmp_path / "run").mkdir()
    path = os.pathsep.join(x for x in os.environ["PATH"].split(os.pathsep) if not (Path(x) / "gpu-lease").exists())
    rc, log, out = _road(tmp_path, monkeypatch, "remote", d, law="rate:1/4", RESEARCH_RUN_DIR=str(tmp_path / "run"), PATH=path,
                         DEVELOPMENT="1")
    lines = log
    err = "\n".join(f.read_text()[-2000:] for f in sorted(d.glob("window-circuit-*.err")))
    log += f"\n(the window circuit's log, which stays on the node:)\n{err}"
    assert "PREFLIGHT" not in log and "ROW" not in log.replace("no ROW read", ""), log
    assert "SEAL: the window's seal" in log and ": PASS" in log.split("SEAL: ")[1].splitlines()[0], log
    assert "COMPILER: the record's compiler digest and predict row's are equal strings: PASS" in log, log
    assert ("ADVICE: committed (its root a hiding commitment); its content and length are held to no registered compiler "
            "(candidate 135)") in log and "its circuit registered by the prover" in log, log
    published = lines + "".join(text for _, text in _published(out))
    assert not [p for p in UNSAID if p in published.lower()], [p for p in UNSAID if p in published.lower()]
    sampled = [x for x in lines.splitlines() if x.startswith("sampled-proofs: ")]
    assert sampled and sampled[0].startswith(("sampled-proofs: refused: ", "sampled-proofs: not run: ")) and not sampled[0].endswith(
        (": refused: ", ": not run: ")), log
    assert not [x for x in lines.splitlines() if "the Lean checks pass" in x or x.startswith("WINDOW-RECORD accept")], log
    assert ("pearl-plus: not run: needs a card: its piece takes its own gpu-lease, and this node has no gpu-lease: "
            "compute accounting's") in log, log
    assert re.search(r"^record: (refused|not run): ", lines, re.M) and "composite: not run:" in log, log
    end = _end(log)
    ran = set(end.split("; refused ")[0].split("(")[1].rstrip(")").split(","))
    assert end.startswith("ROAD END (the window's record): ran ") and "pearl-plus" in end and rc == 20, log
    assert not {"sampled-proofs", "record"} & ran and "no verdict stands for a step that did not run" in end, end
    assert end.endswith("no proof is made, and Lean verifies nothing; no sampled verdict, so no δ or ε: the sampled proofs did not run; the circuit digest is kept private and NO public value fixes it before the coins"), end
    assert [x for x in lines.splitlines() if x.startswith("CLASS ")] == [
        "CLASS 2,4,2: 8.00 bits leave with it (its prompt length, one of the 256 that fit 2 slots in 512 tokens a step; its "
        "slots and outputs are the application's)"], lines
    assert "prove_path.py window-circuit: truncated: a 8-bit field at bit 0 of 0" in err, err
    counts = [x for x in lines.splitlines() if x.startswith("BLOCK-COUNTS")]
    assert counts and not [x for x in counts if "refused" in x], (counts, [f.read_text()[-600:] for f in d.glob("block-counts-*.err")])
    secret = _secret(d)
    assert _plain(d) and hashlib.sha512((d / "windows/w0/record/circuit.json").read_bytes()).hexdigest() in _plain(d).values()
    assert not [s for s in secret if s in lines or s[:16] in lines], lines
    assert not [(f, s) for f, text in _published(out) for s in secret if s in f or s in text or s[:16] in text]
    names = json.loads((tmp_path / "run" / "private-paths.json").read_text())["names"]
    assert {"plans.jsonl", "handoff/", "circuit.json", "predicted.json"} <= set(names), names


def _preflight(tmp_path, steps=("steps", "preflight"), **env):
    (tmp_path / "serve").mkdir(exist_ok=True)
    env = {**os.environ, "D": str(tmp_path / "serve"), "PRED": str(tmp_path / "predicted"), "OUT": str(tmp_path / "out"),
           "SERVE_ROOT": str(tmp_path), "VENV": sys.prefix, "PRODUCT": "remote", "CUDA_HOME": str(tmp_path),
           "PEARLC_CHALLENGER_GATE": "", **env}
    p = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh", *steps], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    return p.returncode, p.stdout + p.stderr


def test_compute_accounting_is_held_to_the_application_list_both_ways(tmp_path):
    """Where the application lists compute accounting, no scheme is refused by name; where it lists none, a scheme set
    is refused too."""
    rc, out = _preflight(tmp_path, APPLICATION_PROTOCOLS="sampled-proofs pearl-plus inference-only", POUW="")
    assert rc == 3 and "PREFLIGHT: the application lists compute accounting (pearl-plus), and POUW names no scheme" in out, out
    rc, out = _preflight(tmp_path, APPLICATION_PROTOCOLS="sampled-proofs network-attester wipe", POUW="pearlplus")
    assert rc == 3 and ("PREFLIGHT: POUW=pearlplus names a compute-accounting scheme, and the application lists no "
                        "compute accounting") in out, out
    for protocols, pouw in (("sampled-proofs pearl-plus", "pearlplus"), ("sampled-proofs wipe", "")):
        rc, out = _preflight(tmp_path, APPLICATION_PROTOCOLS=protocols, POUW=pouw)
        assert "compute accounting" not in out and "the window's record lacks" in out, out


def test_the_window_steps_refuses_by_name_each_thing_its_record_lacks(tmp_path):
    """An empty serve dir: the preflight names every file it wanted, and nothing named ROW."""
    rc, out = _preflight(tmp_path, POUW="pearlplus", APPLICATION_PROTOCOLS="sampled-proofs pearl-plus")
    assert rc == 3, out
    assert ("PREFLIGHT: the window's record lacks WINDOW_CLASS, the class the serve took (its --window-class): "
            "serve.json's window_class, absent from a serve made before it recorded it, so pass WINDOW_CLASS=S,P,O; "
            "D/plans.jsonl, the close's plan of each window; PRED/predicted.json, predict row's Build before the lease: "
            "PRED is serve.json's committer_options.row_build, else D/predicted, so pass PRED=<the directory that holds "
            "predicted.json>") in out, out
    assert "ROW" not in out


def _from_serve_dir(tmp_path, record):
    """The runner's preflight with only the serve dir named (``D``): no ``PRED``, ``WINDOW_CLASS`` or ``SERVE_ROOT``, a
    serve.json of ``record`` in it."""
    d = tmp_path / "serve"
    d.mkdir(exist_ok=True)
    (d / "serve.json").write_text(json.dumps(record))
    env = {k: v for k, v in os.environ.items() if k not in ("PRED", "WINDOW_CLASS", "SERVE_ROOT")}
    env |= {"D": str(d), "OUT": str(tmp_path / "out"), "VENV": sys.prefix, "PRODUCT": "remote", "CUDA_HOME": str(tmp_path),
            "PEARLC_CHALLENGER_GATE": "", "POUW": "pearlplus", "APPLICATION_PROTOCOLS": "sampled-proofs pearl-plus"}
    p = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh", "steps", "preflight"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_the_serve_dir_alone_gives_the_class_and_the_predicted_build_it_took(tmp_path):
    """serve.json holds the class the serve took and the directory of predict row's Build it was given, so a run that
    names only the serve dir needs neither WINDOW_CLASS nor PRED; each is refused by name, with where it comes from, where
    the serve dir doesn't hold it."""
    (tmp_path / "predicted").mkdir()
    (tmp_path / "predicted" / "predicted.json").write_text("{}")
    rc, out = _from_serve_dir(tmp_path, {"window_class": "2,4,2", "committer_options": {"row_build": str(tmp_path / "predicted")}})
    assert rc == 3 and "D/plans.jsonl" in out, out
    assert "WINDOW_CLASS" not in out and "PRED/predicted.json" not in out, out
    rc, out = _from_serve_dir(tmp_path / "bare", {"scheme": "rows-sha512"})
    assert rc == 3 and "WINDOW_CLASS, the class the serve took" in out and "PRED/predicted.json" in out, out


def test_a_run_naming_no_application_is_refused_by_name(tmp_path):
    """The runner is `verity run`'s, which hands it the application's list: with none, what it lists can't be checked, so
    the run is refused before the window's preflight, compute accounting's scheme set or not."""
    for pouw in ("", "pearlplus"):
        env = {k: v for k, v in os.environ.items() if k != "APPLICATION_PROTOCOLS"}
        rc, out = _preflight(tmp_path, (), **env, POUW=pouw)
        assert rc == 3 and out.strip().splitlines()[-1] == (
            "PREFLIGHT: the runner is verity run's: no application was named (APPLICATION_PROTOCOLS), so what it lists "
            "cannot be checked"), out


def test_a_run_naming_no_step_is_refused_by_name(tmp_path):
    """The steps and their order are `verity run`'s stages': a run naming none, an unknown one or one twice is refused
    before the window's preflight and any step."""
    for steps in ((), ("steps",)):
        rc, out = _preflight(tmp_path, steps, APPLICATION_PROTOCOLS="sampled-proofs", POUW="")
        assert rc == 3 and out.strip().splitlines()[-1] == (
            "PREFLIGHT: the runner's steps and their order are verity run's stages': none was named (served_verdicts.sh "
            "steps STEP...)"), out
    for steps, line in ((("steps", "preflight", "seal", "preflight"), "the step preflight is named twice"),
                        (("steps", "seal", "audit"), "the window has no step audit")):
        rc, out = _preflight(tmp_path, steps, APPLICATION_PROTOCOLS="sampled-proofs", POUW="")
        assert rc == 3 and out.strip().splitlines()[-1] == f"PREFLIGHT: {line}: nothing run", out


def _sealed(tmp_path, window, drop=()):
    d, pred, body = tmp_path / "serve", tmp_path / "predicted", b'{"templates": []}'
    rec = d / "windows" / "w0" / "record"
    rec.mkdir(parents=True)
    (d / "windows" / "w0" / "commit").mkdir()
    pred.mkdir()
    (rec / "circuit.json").write_bytes(body)
    good = {"run": RUN_ROOT, "circuit": hashlib.sha512(body).hexdigest(), "compiler": COMPILER}
    decoy = {"run": "00" * 64, "circuit": "00" * 64, "compiler": {"name": "c", "digest": "00" * 64}}
    (rec / "registration.json").write_text(json.dumps({**decoy, "window": {k: v for k, v in {**good, **window}.items()
                                                                           if k not in drop}}))
    (d / "seals.jsonl").write_text(json.dumps({"window": 0, "run_root": RUN_ROOT}) + "\n")
    (d / "windows" / "w0" / "commit" / "verdict.json").write_text(json.dumps({"run_root": RUN_ROOT}))
    (d / "windows" / "w0" / "commit" / "runs.jsonl").write_text(json.dumps({"commit": {"binding": {}}}) + "\n")
    (pred / "predicted.json").write_text(json.dumps({"compiler": COMPILER, "workload_digest": WORKLOAD}))
    env = {**os.environ, "D": str(d), "PRED": str(pred), "OUT": str(tmp_path / "out"), "SERVE_ROOT": str(tmp_path),
           "VENV": sys.prefix, "PRODUCT": "remote", "CUDA_HOME": str(tmp_path), "PEARLC_CHALLENGER_GATE": ""}
    p = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh", "window-seal", "0"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_the_seal_reads_each_field_at_its_path_in_the_registration_and_refuses_a_record_without_it(tmp_path):
    """The seal holds the registration's ``window`` run, circuit and compiler, never a field of the same name elsewhere in
    it: a record whose top level carries other values passes on its ``window``'s, one whose ``window`` differs fails, and
    one whose ``window`` lacks a field is refused by its path."""
    rc, out = _sealed(tmp_path / "a", {})
    assert rc == 0 and "and its record: PASS" in out and "predict row's are equal strings: PASS" in out, out
    assert "BOUND-PROGRAM: the commit's bound Program digest and the predicted workload digest are kept side by side and " \
           "compared by nothing, so the seal does not tie the window's circuit to the served one" in out, out
    rc, out = _sealed(tmp_path / "b", {"circuit": "11" * 64})
    assert rc == 1 and "the record's circuit.json is not the circuit its registration binds" in out, out
    rc, out = _sealed(tmp_path / "c", {"compiler": {"name": "c", "digest": "11" * 64}})
    assert rc == 1 and "predict row's are equal strings: FAIL" in out, out
    rc, out = _sealed(tmp_path / "d", {"run": None})
    assert rc == 1 and "the seal's run root" in out, out
    rc, out = _sealed(tmp_path / "e", {}, drop=("compiler",))
    assert rc == 1 and "the record's registration has no window.compiler" in out, out


SCRIPT = ROOT / "benchmarks" / "one_stage" / "served_verdicts.sh"
#: what the road's steps call beyond the functions under test, stood in: our tools' public lines, and the counted
#: circuit's paths
STUBS = """say() { printf '%s\\n' "$*" | tee -a "$OUT/verdicts.txt"; }
failed() { echo "(a tool's refusal, exit $1)"; }
said() { echo "counted"; }
counted_pair() { echo "$D/circuit.json $D/registration.json"; }
io_class() { echo 2,4,2; }
gpu-lease() { :; }
nvidia-smi() { echo "GPU 0: a card"; }
RAN=() REFUSED=() NOT_RUN=()
declare -A WHY=()
"""


def _fns(*names):
    """The definitions of ``names`` as `served_verdicts.sh` holds them, one-line or to their closing brace."""
    lines, out = SCRIPT.read_text().splitlines(), []
    for n in names:
        i = next(i for i, x in enumerate(lines) if x.startswith(f"{n}() {{"))
        j = i if not re.match(rf"{re.escape(n)}\(\) \{{\s*(#.*)?$", lines[i]) else lines.index("}", i)
        out += lines[i:j + 1]
    return "\n".join(out)


def _steps(tmp_path, body, py, **env):
    """``body`` after the road's step functions, with ``py`` (bash) as the venv's python where it is not ``-c`` or one of
    our tools: its exit and what it said."""
    d, out = tmp_path / "serve", tmp_path / "out"
    d.mkdir(parents=True)
    out.mkdir()
    (d / "serve.json").write_text(json.dumps({"clocks": {"time": 1_700_000_000}}))
    (d / "circuit.json").write_text("{}")
    (tmp_path / "policy.json").write_text("{}")
    stub = tmp_path / "python"
    stub.write_text(f'#!/bin/bash\n[ "$1" = -c ] && exec {sys.executable} "$@"\n{py}\n')
    stub.chmod(0o755)
    env = {**os.environ, "D": str(d), "OUT": str(out), "PY": str(stub), "CLASSES": "2,4,2", "WREG": str(d / "none.json"),
           "IO_POLICY": str(tmp_path / "policy.json"), "IO_LEAN": str(tmp_path / "no-lean"), "BASIS_129": "basis",
           "POUW": "", "POUS_CMD": "", "BANDWIDTH_CMD": "", **env}
    fns = _fns("step_not_run", "step_refused", "road_step", "road_end", "block_counts", "inference_only", "road_pieces")
    p = subprocess.run(["bash", "-c", f"{STUBS}\n{fns}\n{body}"], cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_a_step_whose_command_refuses_is_refused_by_its_first_line_and_one_that_counts_nothing_did_not_run(tmp_path):
    """block-counts' command exits 1 on the window's record: the step is refused, by the line it said, and not among
    what ran; with no policy declared it did not run, and says why; either way the road's exit is not 0."""
    refuse = 'echo "ValueError: verity-vllm/window: no one session Definition" >&2; exit 1'
    rc, out = _steps(tmp_path / "a", "RAN+=(seal); road_step block-counts block_counts; road_end 20", refuse)
    assert rc == 20, out
    assert ("ROAD END (the window's record): ran 1 (seal); refused or undecided 1 (block-counts); not run 0 ()") in out, out
    assert "  REFUSED block-counts: BLOCK-COUNTS class 2,4,2 refused (a tool's refusal, exit 1) (exit 1)" in out.splitlines(), out
    rc, out = _steps(tmp_path / "b", "road_step block-counts block_counts; road_end 20", refuse, IO_POLICY="")
    assert rc == 20 and "not run 1 (block-counts)" in out, out
    assert "  NOT RUN block-counts: the run declares no c, k and n" in out.splitlines(), out


def test_an_undecided_verdict_is_refused_by_its_own_line_and_a_decided_refusal_ran(tmp_path):
    """inference-only's command ends 0 on an undecided verdict: the step is refused or undecided, its line verbatim; a
    verdict that refuses on a count is decided, so the step ran."""
    undecided = ("INFERENCE-ONLY refuse: undecided: the policy was written at 1, not before the serve began at 2")
    rc, out = _steps(tmp_path / "a", "road_step inference-only inference_only; road_end 20", f'echo "{undecided}"')
    assert rc == 20 and "refused or undecided 1 (inference-only)" in out and "ran 0 ()" in out, out
    assert f"  REFUSED inference-only: {undecided} (exit 0)" in out.splitlines(), out
    decided = "INFERENCE-ONLY refuse: the blocks' exact count is more than n = 3"
    rc, out = _steps(tmp_path / "b", "road_step inference-only inference_only; road_end 20", f'echo "{decided}"')
    assert rc == 0 and "ran 1 (inference-only); refused or undecided 0 (); not run 0 ()" in out, out


def test_a_listed_pearl_plus_whose_piece_refuses_is_refused_by_its_sentence_and_an_unlisted_one_is_no_step(tmp_path):
    """Pearl+ is a step where the application lists it (POUW=pearlplus): a piece that answers no verdict is refused by
    its own sentence (its refused.txt), not hidden; with POUW empty it is no step at all, and a road whose every step ran
    exits 0."""
    sentence = "pearlplus-no-verdict: served_passes.py serve wrote no document"
    piece = (f'pieces() {{ mkdir -p "$OUT/pieces/pouw"; echo \'{{"rc": 2}}\' > "$OUT/pieces/pouw/piece.json"; '
             f'echo "{sentence}" > "$OUT/pieces/pouw/refused.txt"; }}\n')
    rc, out = _steps(tmp_path / "a", f"{piece}RAN+=(seal); road_pieces; road_end 20", "exit 9", POUW="pearlplus")
    assert rc == 20 and "refused or undecided 1 (pearl-plus)" in out, out
    assert f"  REFUSED pearl-plus: {sentence}" in out.splitlines(), out
    rc, out = _steps(tmp_path / "b", f"{piece}RAN+=(seal); road_pieces; road_end 20", "exit 9")
    assert rc == 0 and "pearl-plus" not in out and "ran 1 (seal)" in out, out


def test_a_count_an_earlier_call_left_is_cleared_by_the_circuits_step_so_inference_only_does_not_run(tmp_path):
    """The circuits step removes the window's public counts before it counts: one that fails leaves none, and the
    inference-only step, which reads them, does not run on an earlier call's."""
    body = ('touch "$OUT/window-circuit-2-4-2.json"; circuits() { echo "WINDOW-CIRCUIT class 2,4,2 refused"; return 1; }\n'
            "road_step_named circuits; RAN+=(inference-policy); road_step_named inference-only; road_end 20")
    rc, out = _steps(tmp_path, f"{_fns('road_step_named', 'circuits_counted')}\n{body}", "exit 9")
    assert rc == 20 and not (tmp_path / "out" / "window-circuit-2-4-2.json").exists(), out
    assert ("  NOT RUN inference-only: its verdict reads the circuits step's window circuit and the inference-only policy, "
            "and one of them did not run") in out.splitlines(), out
