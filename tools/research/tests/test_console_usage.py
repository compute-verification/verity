"""The console's GPU usage: node 2's sampler reduced to minutes with a timed window's samples unmeasured, both nodes' minutes and
node 1's pool leases turned into `cluster.usage` rows, the held-idle lines derived from them, and both files recorded."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location("verity_console", Path(__file__).parents[1] / "console" / "verity_console.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)
usage = vc.usage_module()

H0 = datetime(2026, 10, 3, 8, tzinfo=timezone.utc).timestamp()
NOW = datetime(2026, 10, 3, 9, 10, tzinfo=timezone.utc)


def test_the_console_and_the_module_share_their_labels_and_the_record():
    assert (vc.NO_POD, vc.USAGE_RECORD) == (usage.NO_POD, Path(usage.RECORD))
    assert set(vc.NODE_NAMES.values()) == set(usage.NODES.values())


def test_node_2s_minutes_count_a_timed_windows_samples_unmeasured(tmp_path):
    util = tmp_path / "util"
    util.mkdir()
    lease = "pid={} run=- who={} since=2026-10-03T07:00:00Z cmd=x"
    samples = [{"t": "2026-10-03T08:00:00Z", "gpus": [
                    {"i": 0, "util": 40, "mem_mib": 2048, "lease": lease.format(1, "fill:bc-15ada664"), "procs": [{"pid": 1}]},
                    {"i": 1, "util": 0, "mem_mib": 0, "lease": None, "procs": []}]},
               {"t": "2026-10-03T08:00:10Z", "timed": True, "gpus": [{"i": 0, "lease": lease.format(2, "circuits-tp8")},
                                                                       {"i": 1, "lease": lease.format(2, "circuits-tp8")}]}]
    f = util / "2026-10-03.jsonl"
    f.write_text("".join(json.dumps(s) + "\n" for s in samples))
    os.utime(f, (H0 + 100, H0 + 100))
    out = subprocess.run([sys.executable, "-", str(int(H0)), str(util)], input=vc.N2_MINUTES, capture_output=True, text=True,
                         check=True).stdout
    assert json.loads(out) == [[H0, 0, 0.5, 0, 0, 0, "fill:bc-15ada664", 0.5], [H0, 1, 0, 0, 0, 0.5, "circuits-tp8", 0.5]]


def test_lanes_and_kinds_of_each_nodes_holders():
    assert [(vc.lane_of("node 1", h), vc.kind_of("node 1", h)) for h in (
        "gpu-pool-circuits-1791-x", "gpu-pool-1791-y", "Kueue vllm-epoch-run", "SkyPilot m0-*", vc.NO_POD)] == [
        ("circuits", "gpu-lease pool"), ("proofs", "gpu-lease pool"), ("circuits", "Kueue vllm-epoch-run"),
        ("proofs", "SkyPilot m0-*"), (vc.NO_POD, vc.NO_POD)]
    assert [(vc.lane_of("node 2", h), vc.kind_of("node 2", h)) for h in (
        "fill bc-15ada664", "bc-b8aaadaa", "circuits-tp8", "(process, no lease)")] == [
        ("memory-accounting", "fill"), ("circuits", "gpu-lease"), ("circuits-tp8", "gpu-lease"),
        ("(process, no lease)", "(process, no lease)")]


MINUTES = {
    "node 1": [(H0, 0, 1.0, 0, 0, 0, "gpu-pool-circuits-a"), (H0 + 60, 0, 0, 1.0, 0, 0, "gpu-pool-circuits-a"),
               (H0, 1, 0, 1.0, 0, 0, "Kueue vllm-epoch-run"), (H0 + 60, 1, 0, 1.0, 0, 0, "Kueue vllm-epoch-run"),
               (H0, 2, 0, 1.0, 0, 0, "gpu-pool-1791-y"), (H0, 3, 1.0, 0, 0, 0, vc.NO_POD)],
    "node 2": [(H0, 0, 0, 0, 0, 0, "circuits-tp8", 1.0), (H0, 1, 0.5, 0.5, 0, 0, "fill bc-15ada664", 0.0)]}
SPANS = {"node 1": [(0, H0, H0 + 120, "circuits")]}


def test_held_idle_lines_come_from_the_modules_rows():
    rows = vc.usage_rows(MINUTES, SPANS)
    assert {(r["node"], r["owner"], r["kind"]) for r in rows} == {
        ("vy-nebius-1", "circuits", "gpu-lease"), ("vy-nebius-1", "circuits", "Kueue vllm-epoch-run"),
        ("vy-nebius-1", "proofs", usage.POOL_UNLEASED), ("vy-nebius-1", vc.NO_POD, vc.NO_POD),
        ("vy-nebius-2", "circuits-tp8", "gpu-lease"), ("vy-nebius-2", "memory-accounting", "fill")}
    got = {(d["node"], d["lane"]): (d["held_gpu_h"], d["held_idle_gpu_h"], d["held_idle_pct"], d["unmeasured_gpu_h"],
                                    d["target_pct"], d["complete"]) for d in vc.idle_readings(rows, NOW)}
    assert got == {
        ("node 1", "(all holders)"): (0.1, 0.07, 66.7, 0.0, None, True),
        ("node 1", "circuits"): (0.07, 0.05, 75.0, 0.0, 40.0, True),
        ("node 1", "lease pool, not leased"): (0.02, 0.02, 100.0, 0.0, None, True),
        ("node 1", vc.NO_POD): (0.02, 0.0, 0.0, 0.0, None, True),
        ("node 2", "(all holders)"): (0.02, 0.01, 50.0, 0.02, None, True),
        ("node 2", "circuits-tp8"): (0.0, 0.0, None, 0.02, None, True),
        ("node 2", "memory-accounting"): (0.02, 0.01, 50.0, 0.0, None, True)}


def test_a_node_whose_minutes_were_not_read_has_no_rows():
    assert vc.usage_rows({"node 1": None, "node 2": MINUTES["node 2"]}, SPANS) == vc.usage_rows({"node 2": MINUTES["node 2"]})


def test_record_usage_writes_both_files_once_per_complete_hour(tmp_path):
    rows = vc.usage_rows(MINUTES, SPANS)
    readings = vc.idle_readings(rows, NOW)
    idle, rec = tmp_path / "held-idle-hourly.jsonl", tmp_path / "usage-hourly.jsonl"
    assert vc.record_usage(rows, readings, NOW, H0 - 600, idle, rec) == (len(readings), len(rows))
    assert vc.record_usage(rows, readings, NOW, H0 - 600, idle, rec) == (0, 0)
    lines = [json.loads(ln) for ln in idle.read_text().splitlines()]
    assert lines == [{k: v for k, v in r.items() if k != "complete"} for r in readings]
    assert {r["schema"] for r in map(json.loads, rec.read_text().splitlines())} == {usage.ROW}
