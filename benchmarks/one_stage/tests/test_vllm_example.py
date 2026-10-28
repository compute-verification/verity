"""The vLLM example's endpoint serves under its row's Build of record, the ahead-of-time compiler's (`verity-vllm predict row`),
never its Commit of record's traced Build, which is only the cross-check (`predict compare --predicted`): `serve.sh` in a toy tree
whose `verity_vllm.pipeline.cli` is a stub that records each call, and `run.sh --dry-run` handing `served_verdicts.sh` the
example's workloads dir, where its `predict row` finds the row."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "vllm"
ROW = "llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager"

FAKE_CLI = '''import json, os, sys
from pathlib import Path
a = sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps({"argv": a, "VERITY_WORKLOADS": os.environ.get("VERITY_WORKLOADS")}) + "\\n")
answer = json.loads(Path(os.environ["ANSWERS"]).read_text())
if a[:2] == ["predict", "row"]:
    if answer["row"] != "write":
        sys.exit("predict row: refused")
    out = Path(a[a.index("--out") + 1]); out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}")
    (out / "predicted.json").write_text(json.dumps({"config": "config.json"}))
elif a[:2] == ["predict", "compare"]:
    print("[predict] row: " + ("EQUAL on every rank" if answer["compare"] == "equal" else "NOT EQUAL") + " -> out")
    sys.exit(0 if answer["compare"] == "equal" else 1)
elif a[0] == "serve-openai":
    sys.exit(3)
'''


def _serve(tmp_path: Path, row: str = "write", compare: str = "equal") -> tuple[int, list[dict], str]:
    """`serve.sh` in a toy tree under a stub vLLM venv whose `verity_vllm.pipeline.cli` answers `predict row` with a Build or a
    refusal (``row``) and `predict compare` equal or not (``compare``), and whose `serve-openai` exits before it listens, under a
    `sleep` that returns at once: the exit status, the cli's calls in order, and serve.sh's output."""
    tree, venv, bin_ = tmp_path / "tree", tmp_path / "venv" / "bin", tmp_path / "bin"
    for d in (tree / "integrations" / "vllm", venv, bin_):
        d.mkdir(parents=True)
    (tmp_path / "cli.py").write_text(FAKE_CLI)
    (tmp_path / "answers.json").write_text(json.dumps({"row": row, "compare": compare}))
    stub = (f'case "$1 $2" in\n"-m research") exit 0;;\n'
            f'"-m verity_vllm.pipeline.cli") shift 2; exec {sys.executable} {tmp_path / "cli.py"} "$@";;\nesac\n'
            f'exec {sys.executable} "$@"')
    for path, body in ((venv / "python", stub), (bin_ / "sleep", "exit 0")):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "OUT": str(tmp_path / "out"), "RESEARCH_RUN_ID": "r-toy",
           "CALLS": str(tmp_path / "calls.jsonl"), "ANSWERS": str(tmp_path / "answers.json"), "DISK_FLOOR_GB": "0",
           "REPO": "unsloth/Llama-3.2-1B", "REV": "9535bd9b", "ROW": str(tmp_path / "commit" / ROW), "VENV": str(venv.parent),
           "HIDDEN_SO": "hidden.so", "NORM_TAP_SO": "norm.so", "CUDA_HOME": str(tmp_path), "HF_HOME": str(tmp_path),
           "SERVE_ROOT": str(tmp_path / "serve")}
    p = subprocess.run(["bash", str(EXAMPLE / "serve.sh")], cwd=tree, env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    log = tmp_path / "calls.jsonl"
    calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return p.returncode, calls, p.stdout


def test_the_endpoint_serves_under_predicts_row_held_to_the_traced_one(tmp_path):
    rc, calls, out = _serve(tmp_path)
    assert rc == 10 and "SERVER EXITED before listening" in out, out
    pred, traced = str(tmp_path / "serve" / "r-toy" / "predicted"), str(tmp_path / "commit" / ROW)
    (row, compare, serve), workloads = [c["argv"] for c in calls], {c["VERITY_WORKLOADS"] for c in calls}
    assert row[:2] == ["predict", "row"] and row[row.index("--row") + 1] == ROW and row[row.index("--out") + 1] == pred
    assert compare[:3] == ["predict", "compare", traced] and compare[compare.index("--predicted") + 1] == pred
    assert compare[compare.index("--config") + 1] == "config.json"
    assert serve[0] == "serve-openai" and serve[serve.index("--required-manifest") + 1] == f"{pred}/manifest.json"
    assert workloads == {str(tmp_path / "tree" / "examples" / "vllm" / "workloads")}
    assert "PREDICT-COMPARE [predict] row: EQUAL on every rank" in (tmp_path / "out" / "steps.txt").read_text()


@pytest.mark.parametrize("row, compare, rc, said", [
    ("refuse", "equal", 17, "PREDICT-ROW refused: no Build of record to serve under"),
    ("write", "differ", 18, "PREDICT-COMPARE: predict's Build of record is not ROW's traced Build"),
])
def test_the_endpoint_doesnt_serve_without_predicts_row_or_when_the_trace_disagrees(tmp_path, row, compare, rc, said):
    got, calls, out = _serve(tmp_path, row, compare)
    assert got == rc and said in out, out
    assert not [c for c in calls if c["argv"][0] == "serve-openai"]


def test_run_hands_every_run_the_example_workloads_from_the_tree_root(tmp_path):
    """`served_verdicts.sh`, `window`'s script, runs `predict row` from the tree root, and the example's row is found only through
    `VERITY_WORKLOADS`: the dir `run.sh` passes holds the row file the example's SHA256SUMS lists."""
    here = tmp_path / "examples" / "vllm"
    (here / "workloads").mkdir(parents=True)
    for f in ("run.sh", "placement.toml"):
        shutil.copy(EXAMPLE / f, here / f)
    doc = json.dumps({"model": "unsloth/Llama-3.2-1B", "case": "LLAMA32_1B", "engine": {"revision": "9535bd9b"}})
    (here / "workloads" / f"{ROW}.json").write_text(doc)
    (here / "workloads" / "SHA256SUMS").write_text(f"{hashlib.sha256(doc.encode()).hexdigest()}  {ROW}.json\n")
    for run in ("window", "serve", "commit"):
        p = subprocess.run(["bash", str(here / "run.sh"), run, "--dry-run"], capture_output=True, text=True, check=True)
        argv = shlex.split(p.stdout)
        env = dict(argv[i + 1].split("=", 1) for i, x in enumerate(argv) if x == "--env")
        assert (tmp_path / env["VERITY_WORKLOADS"] / f"{ROW}.json").is_file(), (run, env)
