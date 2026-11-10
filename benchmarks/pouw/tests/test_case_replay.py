"""`pearl_c_vllm/case_replay.sh`, Pearl+'s verdict taken again off the served node from the run's case, with stand-ins for
the tree's verifiers: what it hands them, what it writes, and what it refuses."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "pearl_c_vllm" / "case_replay.sh"

#: `served_passes.py` in the verifier's tree: `verify` reads the manifests, the coins and the files TOY_READS names under
#: the passes, and answers; `lean` answers. TOY_DRAWS has `verify` ask for a draw, as a verdict without a coin would.
VERIFIER = r'''
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
opt = lambda k: a[a.index(k) + 1] if k in a else None
passes = Path(opt("--passes"))
if a[0] == "verify":
    assert "PEARLC_CHALLENGER_GATE" not in os.environ and "--fast" in a
    mans = sorted(passes.glob("pass-*/manifest.json"))
    for m in mans:
        json.loads(m.read_text())
    json.loads((passes / "coins.json").read_text())
    for rel in filter(None, os.environ.get("TOY_READS", "").split(",")):
        open(passes / rel, "rb").read()
    coin = json.loads(Path(opt("--coin-record")).read_text())
    if os.environ.get("TOY_DRAWS"):
        Path(opt("--coin-record") + ".asked").write_text("{}")
    w = json.loads(Path(opt("--tie")).read_text())
    answer = {"verdict": "accept", "reason": None, "commitment": coin["commitment"]}
    doc = {"answer": answer, "tie": {"record": w["digest"], "weights": "weights untied (W v1)"}, "draw": "challenger",
           "coin_record": opt("--coin-record"), "draws": int(opt("--draws")), "commit": opt("--commit"),
           "line": f"ACCEPT toy: {len(mans)} passes"}
    Path(opt("--out")).write_text(json.dumps(doc))
    Path(opt("--answer")).write_text(json.dumps(answer))
    print(doc["line"])
else:
    assert Path(opt("--exe")).is_file()
    out = Path(opt("--out"))
    out.write_text(json.dumps({"verdict": "accept", "tie": {"record": "w" * 8, "weights": "weights untied (W v1)"}}))
    out.with_name("pearlplus-lean-check.json").write_text(json.dumps({"reason": None, "differ": []}))
    print("lean ACCEPT toy")
'''


def _case(d: Path) -> Path:
    blobs = d / "blobs"
    blobs.mkdir(parents=True)
    files = []
    for i in range(2):
        mine = {}
        for rel in ("calls/0.rows.bf16", "tree_t"):
            data = f"{i}/{rel}".encode()
            h = hashlib.sha512(data).hexdigest()
            (blobs / h).write_bytes(data)
            mine[rel] = h
        files.append(mine)
    parts = {"schema": "verity-pouw/pearl-plus-case-parts/v1", "params": {"coins": {"root": "c0"}, "draws": 3},
             "manifests": [{"pass_index": i} for i in range(2)], "evidence": {"files": files}}
    (d / "parts.json").write_text(json.dumps(parts))
    return d


def _replay(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    tree, bins, out, scratch = (tmp_path / n for n in ("tree", "bin", "out", "scratch"))
    for d in (tree / ".venv" / "bin", tree / "benchmarks" / "pouw" / "pearl_c_vllm",
              tree / "verity" / "core" / "protocols" / "pouw" / "lean" / "pearlplus", bins, scratch):
        d.mkdir(parents=True, exist_ok=True)
    (tree / "benchmarks" / "pouw" / "pearl_c_vllm" / "served_passes.py").write_text(VERIFIER)
    py = tree / ".venv" / "bin" / "python"
    py.unlink(missing_ok=True)
    py.symlink_to(sys.executable)
    for name, body in (("uv", "exit 0"), ("lake", "mkdir -p .lake/build/bin && echo exe > .lake/build/bin/pouw-pearlplus")):
        (bins / name).write_text(f"#!/bin/sh\n{body}\n")
        (bins / name).chmod(0o755)
    case = _case(tmp_path / "case") if not (tmp_path / "case").exists() else tmp_path / "case"
    (tmp_path / "W.json").write_text(json.dumps({"digest": "w" * 8}))
    (tmp_path / "coin.json").write_text(json.dumps({"commitment": "run", "source": "src"}))
    run = subprocess.run(["bash", str(SCRIPT), str(case), str(tmp_path / "W.json"), str(tmp_path / "coin.json"), str(out)],
                         cwd=tree, capture_output=True, text=True, check=False,
                         env={"PATH": f"{bins}:{os.defpath}", "HOME": str(tmp_path), "REPLAY_SCRATCH": str(scratch),
                              "REPLAY_JOBS": "1", "RESEARCH_SOURCE_SHA": "abc123", "PEARLC_CHALLENGER_GATE": "ssh gate", **env})
    return run, out, scratch


def test_the_replay_hands_the_verifiers_the_rebuilt_passes_and_the_recorded_coin_and_keeps_no_opening(tmp_path):
    """Both verifiers run on the case's passes with W and the coin record as given and no challenger gate; the answers
    and what the Python opened go to OUT, and the scratch holding the openings is gone when it ends."""
    run, out, scratch = _replay(tmp_path, TOY_READS="pass-00000/tree_t,pass-00001/calls/0.rows.bf16")
    assert run.returncode == 0, run.stdout + run.stderr
    replay = json.loads((out / "replay.json").read_text())
    coin = hashlib.sha512((tmp_path / "coin.json").read_bytes()).hexdigest()
    assert (replay["verifier_commit"], replay["coin_sha512"], replay["coin_unchanged"]) == ("abc123", coin, True)
    assert replay["python"]["verdict"] == replay["lean"]["verdict"] == "accept"
    assert replay["python"]["tie"]["weights"] == replay["lean"]["tie"]["weights"] == "weights untied (W v1)"
    assert replay["exe_sha256"] == hashlib.sha256(b"exe\n").hexdigest()
    opened = json.loads((out / "opened.json").read_text())
    assert opened == replay["opened"] and (opened["case_files"], opened["opened_case_files"]) == (4, 2)
    assert opened["opened_outside_the_case"] == [] and len(opened["opened_other"]) == 3
    assert opened["case_files_not_opened"] == ["pass-00000/calls/0.rows.bf16", "pass-00001/tree_t"]
    assert list(scratch.iterdir()) == []
    assert not [p for p in out.rglob("*") if p.is_symlink() or p.name in ("blobs", "passes")]
    assert "REPLAY python opened 5 files under the passes: 2 of the case's 4 opened tiles' files" in run.stdout


@pytest.mark.parametrize("env, why", [
    ({"TOY_READS": "pass-00000/calls/1.rows.bf16"}, "python: served_passes.py verify exit 1"),
    ({"TOY_DRAWS": "1"}, "a draw was asked for"),
])
def test_the_replay_fails_on_a_file_outside_the_case_and_on_a_draw(tmp_path, env, why):
    """A verifier that reads a file the case doesn't hold fails on it, and one that asks for a draw fails the step: no
    answer of either is taken as the replay's."""
    run, out, scratch = _replay(tmp_path, **env)
    assert run.returncode != 0 and why in run.stdout, run.stdout + run.stderr
    assert list(scratch.iterdir()) == []
    if "TOY_DRAWS" in env:
        assert not (out / "replay.json").exists()
    else:
        assert not (out / "verify.json").exists() and json.loads((out / "replay.json").read_text())["python"]["verdict"] is None


def test_the_replay_stops_on_a_case_whose_blob_changed(tmp_path):
    case = _case(tmp_path / "case")
    blob = next((case / "blobs").iterdir())
    blob.write_bytes(b"other")
    run, out, scratch = _replay(tmp_path)
    assert run.returncode == 2 and "REPLAY FAILED: case_passes REFUSED: blob" in run.stdout, run.stdout + run.stderr
    assert list(scratch.iterdir()) == [] and not (out / "verify.json").exists()
