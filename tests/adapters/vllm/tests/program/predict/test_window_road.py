"""The runner's window road (`served_verdicts.sh`) through `verity run`'s own call (`served_table.served_window`) on a
window's record made on the processor."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_cli import run as RUN
from verity_cli import served_table as ST

from tests.program.predict.test_close_record import _close
from tests.program.predict.test_window_rows import RUN as RUN_ROOT
from verity_vllm.program.predict.close_record import Record

ROOT = Path(__file__).parents[6]
COMPILER = {"name": "c", "digest": "cd" * 64}
#: a Definition's name as the circuit spells it
TEMPLATE = re.compile(r"\b[A-Z][A-Za-z0-9]*_v[0-9]+\b")


def _serve_dir(tmp_path, monkeypatch):
    """TINY's window at class 2,4,2 closed as a serve's close leaves it (`Record.handoff`, then `Record.make` from the
    handoff alone, as close_record's finish does after the lease), under ``tmp_path/serve``; what only a card's serve writes
    beside it (its seal, commit verdict, advice registration and predict row's Build) stood in."""
    d = tmp_path / "serve"
    low, units, served, plan, binding, compiled = _close(d, monkeypatch)
    rec = Record("rate:1/4", "unread", tmp_path / "salt_stream", sample=2, workers=2)
    h = rec.handoff(plan, 0, low, units, binding)
    got = rec.make(plan, 0, compiled, low, units, served, SimpleNamespace(rc=binding.rc, com=None), handoff=Path(h["out"]))
    line = {"window": 0, "advice": "ef" * 64, "handoff": {"out": h["out"]}, "record": {"pending": "made after the lease"}}
    (d / "plans.jsonl").write_text(json.dumps(line) + "\n")
    (d / "seals.jsonl").write_text(json.dumps({"window": 0, "run_root": RUN_ROOT}) + "\n")
    commit = d / "windows" / "w0" / "commit"
    commit.mkdir(parents=True)
    (commit / "verdict.json").write_text(json.dumps({"run_root": got["run"], "commit_scheme": "rows-sha512"}))
    (commit / "runs.jsonl").write_text(json.dumps({"commit": {"binding": {"program_digest": "ab" * 64}}}) + "\n")
    (d / "advice" / "w0" / "registration.json").write_text(json.dumps(
        {"compiler": {**COMPILER, "cap_bits": 64}, "row_bits": 8, "binding": {"advice": {"root": "ef" * 64, "bits": 0}}}))
    (d / "advice" / "w0" / "private" / "advice.bin").write_bytes(b"")
    (d / "weights_registration.json").write_text("{}")
    (tmp_path / "predicted").mkdir()
    (tmp_path / "predicted" / "predicted.json").write_text(json.dumps({"compiler": COMPILER, "workload_digest": "12" * 64}))
    return d


def _road(tmp_path, monkeypatch, application, d):
    """``application``'s runner as `verity run` launches it on a node whose environment names the serve dir, the serve's
    configuration its workload's model, revision and precision; its lines and its public record."""
    cfg = RUN.load(ROOT / "applications" / f"{application}.toml", RUN.registry())
    w = cfg["workload"]["params"]
    (d / "serve.json").write_text(json.dumps({"scheme": "rows-sha512", "model": w["model"], "revision": w["revision"],
                                              "precision": w["precision"]}))
    run = tmp_path / f"run-{application}"
    run.mkdir()
    for k, v in {"D": str(d), "PRED": str(tmp_path / "predicted"), "SERVE_ROOT": str(tmp_path), "WINDOW_CLASS": "2,4,2",
                 "VENV": sys.prefix, "CUDA_HOME": str(tmp_path / "no-cuda"), "HF_HOME": str(tmp_path / "hf"),
                 "RESEARCH_RUN_DIR": str(run)}.items():
        monkeypatch.setenv(k, v)
    state = tmp_path / f"state-{application}"
    rc = ST.served_window(cfg, state, {})
    return rc, (state / "logs" / "served-window.log").read_text(), state / "out", run


def _secrets(d):
    """The template names and the description's digests the window's record holds."""
    rec = d / "windows" / "w0" / "record"
    text = "".join(p.read_text() for p in (rec / "circuit.json", rec / "index.json", d / "windows" / "w0" / "handoff" / "handoff.json"))
    digests = {hashlib.sha512((rec / "circuit.json").read_bytes()).hexdigest()[:16]}
    digests |= {hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in d.glob("window-circuit-w0-*.json")}
    return set(TEMPLATE.findall(text)), digests


def _public(log, out, d, run):
    """Nothing public names a template or carries the description's digest; the plan and the handoff are private."""
    templates, digests = _secrets(d)
    assert templates, "the record names no template, so this test would pass on nothing"
    published = {p.relative_to(out).as_posix(): p.read_text(errors="replace") for p in out.rglob("*") if p.is_file()}
    for where, text in {"the log": log, **published}.items():
        assert not (found := {t for t in templates if t in text} | {x for x in digests if x in text}), (where, found)
    assert not {n for n in published if Path(n).name in ("plans.jsonl", "handoff.json", "circuit.json")}, published
    names = json.loads((run / "private-paths.json").read_text())["names"]
    assert {"plans.jsonl", "handoff/", "circuit.json"} <= set(names), names
    kept = (d / "private" / "verdicts.txt").read_text()
    assert "COMPILER: the record's c cdcdcdcdcdcdcdcd…" in kept and "BOUND-PROGRAM: the commit's bound Program digest" in kept
    assert "COMPILER:" not in log and "BOUND-PROGRAM:" not in log


def _card():
    """Whether this node has a gpu-lease and a card nvidia-smi lists."""
    if shutil.which("gpu-lease") is None or shutil.which("nvidia-smi") is None:
        return False
    p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return p.returncode == 0 and any(x.startswith("GPU ") for x in p.stdout.splitlines())


def _end(log):
    return next(x for x in log.splitlines() if "ROAD END" in x)


@pytest.mark.slow
def test_the_window_road_runs_remote_from_the_record_as_far_as_it_goes(tmp_path, monkeypatch):
    """remote through the preflight on the window's record (no ROW read), its seal held to the record and predict row's
    compiler, Pearl+ through its own launcher or refused as needing a card, the sampled proofs, the record and the
    composite refused by name; the end line counts each kind. What no cheaper test reaches: the runner itself on a real
    window's record. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch)
    rc, log, out, run = _road(tmp_path, monkeypatch, "remote", d)
    assert "PREFLIGHT" not in log and "ROW" not in log.replace("no ROW read", ""), log
    assert "SEAL: the window's seal" in log and ": PASS" in log.split("SEAL: ")[1].splitlines()[0], log
    assert ("sampled-proofs: not run: step 3's prove path (prove_path/run.sh) proves the units a partition file names "
            "(PP_PARTITION), and the window's record has none: its proof of the window's circuit is proofs'") in log, log
    if not _card():
        line = next(x for x in log.splitlines() if x.startswith("pearl-plus: not run: "))
        assert line.startswith("pearl-plus: not run: needs a card: Pearl+'s piece takes its own gpu-lease, and ") and \
            line.endswith(": compute-accounting's"), log
        assert "pearl-plus" not in _end(log).split("refused")[0], _end(log)
    end = _end(log)
    assert re.search(r"ran \d+ \(preflight,seal,window,advice", end) and re.search(r"refused \d+ \(", end), end
    assert "sampled-proofs" in end and "record" in end and "composite" in end and "pearl-plus" in end and rc == 20, log
    _public(log, out, d, run)


@pytest.mark.slow
def test_the_window_road_runs_an_onsite_application_as_far_as_it_goes(tmp_path, monkeypatch):
    """onsite_network lists no compute accounting, so POUW is empty and the preflight passes it; the network attester's
    piece is called and refuses in its own words (its refused.txt, `attester_piece.NOT_RUN`), and the wipe, with no
    answer in the serve dir, refuses by name. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch)
    rc, log, out, run = _road(tmp_path, monkeypatch, "onsite_network", d)
    assert "POUW" not in log and "pearl-plus" not in log, log
    said = (out / "pieces" / "certifier" / "refused.txt").read_text().strip()
    assert said.startswith("the network audit is not run") and f"NETWORK-ATTESTER: {said}" in log, log
    assert "wipe: not run: the serve dir holds no wipe answer (D/pieces/wipe/answer.json)" in log, log
    assert "sampled-proofs: not run:" in log and "network-attester" in _end(log) and rc == 20, log
    _public(log, out, d, run)


def _preflight(tmp_path, **env):
    env = {**os.environ, "D": str(tmp_path / "serve"), "PRED": str(tmp_path / "predicted"), "OUT": str(tmp_path / "out"),
           "SERVE_ROOT": str(tmp_path), "VENV": sys.prefix, "PRODUCT": "remote", "CUDA_HOME": str(tmp_path),
           "PEARLC_CHALLENGER_GATE": "", **env}
    (tmp_path / "serve").mkdir(exist_ok=True)
    p = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh"], cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_the_window_road_refuses_by_name_each_thing_its_record_lacks(tmp_path):
    """An empty serve dir: the preflight names every file it wanted, and nothing named ROW."""
    rc, out = _preflight(tmp_path, POUW="pearlplus")
    assert rc == 3, out
    assert ("PREFLIGHT: the window's record lacks APPLICATION_PROTOCOLS, the application's list, which verity run sets; "
            "WINDOW_CLASS, the class the serve took (its --window-class), which serve.json doesn't record; D/plans.jsonl, "
            "the close's plan of each window; PRED/predicted.json, predict row's Build before the lease") in out, out
    assert "ROW" not in out


def test_the_runner_reads_the_applications_list_both_ways(tmp_path):
    """Where the application lists compute accounting, an absent scheme is refused by name; where it lists none, a scheme
    that is set is refused too."""
    rc, out = _preflight(tmp_path, APPLICATION_PROTOCOLS="sampled-proofs pearl-plus inference-only", POUW="")
    assert rc == 3 and ("PREFLIGHT: the application lists compute accounting (pearl-plus), and POUW is unset: Pearl+'s "
                        "hooks set POUW=pearlplus") in out, out
    rc, out = _preflight(tmp_path, APPLICATION_PROTOCOLS="sampled-proofs wipe", POUW="pearlplus")
    assert rc == 3 and "PREFLIGHT: the application lists no compute accounting, and POUW is pearlplus: unset it" in out, out
