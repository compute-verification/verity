"""[r18-int-3; admit F-r18-admit-1 / S-r18-3 C2; coord M-0061 (2)] row_pod.sh's ADVISORY admission hook.

The hook sits right after `precheck-target` and before the row starts; it writes $D/admission.json and one row.log line (+ a WARN line
when a stage would be refused) and NEVER stops the row: every path exits 0, including the planner's own failure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
ROW_POD = TREE / "verity_vllm" / "ops" / "row_pod.sh"
WL = TREE / "workloads"

HOOK = r'''
set -euo pipefail
ROW=$1; WL=$2; D=$3; STAGES=build,match,commit; PAIRS=3; SNAP_DEF=0,1
log() { echo "$(date -u +%FT%TZ) [row $ROW] $*" | tee -a "$D/row.log"; }
tl() { echo "tl $*" >> "$D/tl.txt"; }
'''


def _hook_block() -> str:
    src = ROW_POD.read_text()
    m = re.search(r"tl admission start\n.*?tl admission end rc=0 advisory=true\n", src, re.DOTALL)
    assert m, "admission hook block not found in row_pod.sh"
    return m.group(0)


def test_hook_sits_after_precheck_target_and_before_the_row_starts_and_never_exits():
    src = ROW_POD.read_text()
    block = _hook_block()
    i_tf_end = src.index("tl precheck-target end rc=0")
    i_hook = src.index(block)
    i_row_start = src.index("tl row start")
    i_build = src.index("tl stage.build start")
    assert i_tf_end < i_hook < i_row_start < i_build
    # advisory: nothing in the block can stop the row
    assert "exit" not in block and "verdict " not in block  # no stages.txt line, no exit
    assert "|| ADM_LINE=" in block and "--out \"$D/admission.json\"" in block
    assert '${MATCH_SNAP_MAX_BYTES:+--snapshot-max-bytes "$MATCH_SNAP_MAX_BYTES"}' in block  # the cap the Match will run with
    assert subprocess.run(["bash", "-n", str(ROW_POD)], capture_output=True, check=False).returncode == 0


def _run_hook(tmp_path: Path, row: str, wl: str, env_extra: dict[str, str] | None = None) -> tuple[int, str]:
    d = tmp_path / row
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(TREE) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""), **(env_extra or {})}
    env.pop("MATCH_SNAP_MAX_BYTES", None) if not (env_extra and "MATCH_SNAP_MAX_BYTES" in env_extra) else None
    r = subprocess.run(["bash", "-c", HOOK + _hook_block(), "hook", row, wl, str(d)], capture_output=True, text=True, env=env, cwd=str(TREE), check=False)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.skipif(not WL.exists(), reason="sweep workloads not in this checkout")
def test_hook_writes_admission_json_logs_one_line_and_warns_without_stopping(tmp_path):
    row = "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
    rc, out = _run_hook(tmp_path, row, str(WL / f"{row}.json"), {"MATCH_SNAP_MAX_BYTES": "33554432"})
    assert rc == 0, out
    d = tmp_path / row
    doc = json.loads((d / "admission.json").read_text())
    assert doc["schema"] == "verity-sweep/admission/v1" and doc["advisory"] is True and doc["model"] == "mistral-7b"
    assert doc["config"]["snapshot_max_bytes"] == 33554432 and doc["config"]["snapshot_steps"] == "0,1" and doc["config"]["pairs"] == 3
    assert set(doc["stages"]) == {"build", "match", "commit"}
    for s in doc["stages"].values():
        assert s["reserved_bytes"] > 0 and s["verdict"] in ("ADMIT", "REFUSE", "LOWER_BOUND", "NO_LIMIT") and s["terms"]
    log = (d / "row.log").read_text()
    assert log.count("ADMISSION advisory") == 1
    # on this laptop there is no cgroup limit -> NO_LIMIT verdicts and no WARN; with a limit the same hook warns and still exits 0
    tl = (d / "tl.txt").read_text()
    assert "tl admission start" in tl and "tl admission end rc=0 advisory=true" in tl


@pytest.mark.skipif(not WL.exists(), reason="sweep workloads not in this checkout")
def test_hook_warns_when_a_stage_would_be_refused_and_the_row_continues(tmp_path):
    row = "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
    # simulate the 188 GB L40S host by giving the planner a limit through the workload-independent knob the CLI exposes
    block = _hook_block().replace('--out "$D/admission.json"', '--host-limit-bytes 188000000000 --out "$D/admission.json"')
    d = tmp_path / row
    d.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(TREE)}
    env.pop("MATCH_SNAP_MAX_BYTES", None)
    r = subprocess.run(["bash", "-c", HOOK + block + "\necho ROW-CONTINUES\n", "hook", row, str(WL / f"{row}.json"), str(d)], capture_output=True, text=True, env=env, cwd=str(TREE), check=False)
    assert r.returncode == 0 and "ROW-CONTINUES" in r.stdout, r.stdout + r.stderr
    log = (d / "row.log").read_text()
    assert "WARN would-refuse commit" in log and "WARN admission advisory" in log
    doc = json.loads((d / "admission.json").read_text())
    assert doc["would_refuse"] == ["commit"] and doc["stages"]["commit"]["decision"]["short_by_bytes"] > 0
    assert (d / "row.log").read_text().count("\n") == 2  # exactly the ADMISSION line and the WARN line


def test_hook_survives_a_broken_planner_input(tmp_path):
    rc, out = _run_hook(tmp_path, "nosuch-row__x", str(tmp_path / "missing.json"))
    assert rc == 0, out
    log = (tmp_path / "nosuch-row__x" / "row.log").read_text()
    assert "ADMISSION advisory unavailable" in log and "WARN" not in log
