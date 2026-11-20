"""The runner's window road (`served_verdicts.sh`, a plain run's) through `verity run`'s own call (`served_table.served_window`)
on a window's record made on the processor."""
from __future__ import annotations

import hashlib
import json
import os
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
SERVED = {"execution": "bi-eager", "default_seed": DEFAULT_SEED}
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


def _road(tmp_path, monkeypatch, application, d, **env):
    """``application``'s runner as `verity run` launches it on a node whose environment names the serve dir: its exit, its
    lines and its public record (``state/out``)."""
    cfg = RUN.load(ROOT / "applications" / f"{application}.toml", RUN.registry())
    for k, v in {"D": str(d), "PRED": str(tmp_path / "predicted"), "SERVE_ROOT": str(tmp_path), "WINDOW_CLASS": "2,4,2",
                 "VENV": sys.prefix, "CUDA_HOME": str(tmp_path / "no-cuda"), "HF_HOME": str(tmp_path / "hf"), **env}.items():
        monkeypatch.setenv(k, v)
    state = tmp_path / f"state-{application}"
    rc = ST.served_window(cfg, state, {})
    return rc, (state / "logs" / "served-window.log").read_text(), state / "out"


def _secret(d):
    """The declared circuit's description as the auditor must not see it: its templates' names and its digests (the
    registered compiler that builds it is public)."""
    record = d / "windows" / "w0" / "record"
    doc = json.loads((record / "circuit.json").read_text())
    return {str(t) for t, _ in doc["templates"]} | {hashlib.sha512((record / "circuit.json").read_bytes()).hexdigest(),
                                                    PROGRAM, WORKLOAD}


def _published(out):
    return [(str(f.relative_to(out)), f.read_text(errors="replace")) for f in sorted(out.rglob("*")) if f.is_file()]


def _end(log):
    return next(x for x in log.splitlines() if x.startswith("ROAD END"))


@pytest.mark.slow
def test_the_window_road_runs_remote_from_the_record_and_publishes_no_description(tmp_path, monkeypatch):
    """remote through the preflight on the window's record (no ROW read), its seal held to the record and predict row's
    compiler, each step after it run or refused by name: the sampled proofs as proofs', Pearl+ as needing a card (a processor has no gpu-lease), the
    record and the composite after them. The end line counts what ran, refused and failed, and neither it nor any
    published file names a template or carries the description's digest. What no cheaper test reaches: the runner itself
    on a real window's record. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch, "remote")
    (tmp_path / "run").mkdir()
    path = os.pathsep.join(x for x in os.environ["PATH"].split(os.pathsep) if not (Path(x) / "gpu-lease").exists())
    rc, log, out = _road(tmp_path, monkeypatch, "remote", d, RESEARCH_RUN_DIR=str(tmp_path / "run"), PATH=path)
    err = "\n".join(f.read_text()[-2000:] for f in sorted(d.glob("window-circuit-*.err")))
    log += f"\n(the window circuit's log, which stays on the node:)\n{err}"
    assert "PREFLIGHT" not in log and "ROW" not in log.replace("no ROW read", ""), log
    assert "SEAL: the window's seal" in log and ": PASS" in log.split("SEAL: ")[1].splitlines()[0], log
    assert "COMPILER: the record's registered compiler is predict row's: PASS" in log, log
    assert ("sampled-proofs: not run: step 3's prove path (prove_path/run.sh) proves the units a partition file names "
            "(PP_PARTITION), and the window's record has none: its proof of the window's circuit is proofs'") in log, log
    assert ("pearl-plus: not run: needs a card: its piece takes its own gpu-lease, and this node has no gpu-lease: "
            "compute accounting's") in log, log
    assert "record: not run:" in log and "composite: not run:" in log, log
    end = _end(log)
    assert end.startswith("ROAD END (the window's record): ran ") and "pearl-plus" in end and rc == 20, log
    assert "no verdict stands for a step that did not run" in end
    secret = _secret(d)
    assert not [s for s in secret if s in end or s[:16] in end], end
    assert not [(f, s) for f, text in _published(out) for s in secret if s in f or s in text or s[:16] in text]
    names = json.loads((tmp_path / "run" / "private-paths.json").read_text())["names"]
    assert {"plans.jsonl", "handoff/", "circuit.json", "predicted.json"} <= set(names), names


@pytest.mark.slow
def test_the_window_road_runs_an_onsite_application_without_compute_accounting(tmp_path, monkeypatch):
    """onsite_network lists no compute accounting, so POUW is empty and the preflight passes it; the network attester's
    own piece refuses in its own words (`attester_piece.NOT_RUN`), and the wipe, whose answer the card launcher did not
    write, refuses as that launcher's. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch, "onsite_network")
    rc, log, _ = _road(tmp_path, monkeypatch, "onsite_network", d)
    assert "POUW" not in log and "pearl-plus" not in log, log
    assert "network-attester: the network audit is not run: its checks are being moved into the workload's circuit" in log, log
    assert ("wipe: not run: D/pieces/wipe/answer.json is absent: the card launcher did not run the wipe under the serve's "
            "lease (vllm's h100_smoke.sh)") in log, log
    assert rc == 20 and "network-attester" in _end(log), log


def _preflight(tmp_path, **env):
    (tmp_path / "serve").mkdir(exist_ok=True)
    env = {**os.environ, "D": str(tmp_path / "serve"), "PRED": str(tmp_path / "predicted"), "OUT": str(tmp_path / "out"),
           "SERVE_ROOT": str(tmp_path), "VENV": sys.prefix, "PRODUCT": "remote", "CUDA_HOME": str(tmp_path),
           "PEARLC_CHALLENGER_GATE": "", **env}
    p = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh"], cwd=ROOT, env=env, capture_output=True, text=True)
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


def test_the_window_road_refuses_by_name_each_thing_its_record_lacks(tmp_path):
    """An empty serve dir: the preflight names every file it wanted, and nothing named ROW."""
    rc, out = _preflight(tmp_path, POUW="pearlplus")
    assert rc == 3, out
    assert ("PREFLIGHT: the window's record lacks WINDOW_CLASS, the class the serve took (its --window-class), which "
            "serve.json doesn't record; D/plans.jsonl, the close's plan of each window; PRED/predicted.json, predict "
            "row's Build before the lease") in out, out
    assert "ROW" not in out
