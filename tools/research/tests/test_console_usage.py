"""The console's GPU usage: node 2's sampler reduced to minutes with a timed window's samples unmeasured, both nodes' minutes and
node 1's pool leases turned into `cluster.usage` rows, the held-idle lines derived from them, and both files recorded."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def test_a_pools_leases_end_its_own_idle_before_each_shrink(tmp_path):
    # red team Q: the circuits pool shrinks a GPU 30 s after it frees, the provers pool 120 s; one 120 for both cut 22.8 % of
    # circuits' lease time on node 1 (2-3 Oct) and booked it to the pool, not leased
    circuits, provers = tmp_path / "gpu-lease-circuits", tmp_path / "gpu-lease"
    for d, g, shrink in ((circuits, 3, "08:01:40"), (provers, 5, "08:04:00")):
        d.mkdir()
        (d / "n1-lease.log").write_text(f'{{"t": "2026-10-03T08:00:00Z", "ev": "unfenced", "gpu": {g}}}\n'
                                        f'{{"t": "2026-10-03T{shrink}Z", "ev": "shrink", "gpu": {g}}}\n')
    with open(circuits / "n1-lease.log", "a") as f:
        f.write('{"t": "2026-10-03T08:10:00Z", "ev": "unfenced", "gpu": 4}\n')
    (circuits / "n1-lease.json").write_text(json.dumps({"t": "2026-10-03T08:20:00Z", "gpus": [{"index": 4, "lease": None}]}))
    pools = {str(circuits): "circuits", str(provers): "proofs"}
    spans = vc.n1_lease_spans(H0 + 3600, pools, {str(circuits): 30, str(provers): 120})
    assert sorted(spans) == [(3, H0, H0 + 70, "circuits"), (4, H0 + 600, H0 + 1170, "circuits"), (5, H0, H0 + 120, "proofs")]
    minutes = {"node 1": [(H0, 3, 1.0, 0, 0, 0, "gpu-pool-circuits-1791-x"), (H0 + 60, 3, 1.0, 0, 0, 0, "gpu-pool-circuits-1791-x")]}
    rows = {(r["owner"], r["kind"]): r for r in vc.usage_rows(minutes, {"node 1": spans[:1]})}
    assert (rows[("circuits", "gpu-lease")]["useful_s"], rows[("circuits", usage.POOL_UNLEASED)]["busy_s"]) == (70, 50)


def fake_proc(tmp_path, procs: dict) -> Path:
    """A /proc of `procs`, {pid: (argv, environ)}; an environ of None can't be read."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    for pid, (argv, env) in procs.items():
        d = proc / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
        if env is None:
            (d / "environ").mkdir()
        else:
            (d / "environ").write_bytes(b"".join(f"{k}={v}".encode() + b"\0" for k, v in env.items()))
    return proc


LOOP = ["python3", "/workspace/verity-guest/bin/n1_lease.py", "loop", "--every", "5"]


def test_each_pools_idle_is_its_running_loops_and_nothing_else_is_read(tmp_path):
    proc = fake_proc(tmp_path, {
        101: (LOOP, {"GPU_LEASE_DIR": "/run/gpu-lease-circuits", "VY_POOL_IDLE_S": "30", "HF_TOKEN": "dummy",
                     "PEM": "-----BEGIN\nVY_POOL_IDLE_S=999"}),
        102: (["/usr/bin/python3", "-u", *LOOP[1:]], {"PATH": "/usr/bin"}),               # n1_lease.py's defaults
        103: (["bash", "-c", " ".join(LOOP)], {"GPU_LEASE_DIR": "/run/gpu-lease", "VY_POOL_IDLE_S": "7"}),  # its pane, not it
        104: (["python3", LOOP[1], "status"], {"VY_POOL_IDLE_S": "7"}),
        105: (LOOP, {})})                                                                 # exited after its cmdline was read
    loops = vc.n1_pool_loops(proc)
    assert loops == {"/run/gpu-lease-circuits": {("30", "")}, "/run/gpu-lease": {("120", "")}}
    assert [vc.n1_pool_idle(d, loops) for d in vc.N1_LEASE_POOLS] == [30, 120]


def test_a_pools_idle_that_isnt_one_known_number_raises_and_so_does_its_spans(tmp_path, monkeypatch):
    circuits = {"GPU_LEASE_DIR": "/run/gpu-lease-circuits", "VY_POOL_IDLE_S": "30"}
    for k, (procs, why) in enumerate((({}, "no n1_lease.py loop runs for /run/gpu-lease-circuits"),
                                      ({1: (LOOP, circuits), 2: (LOOP, {**circuits, "VY_POOL_IDLE_S": "120"})}, "loops differ"),
                                      ({1: (LOOP, {**circuits, "VY_LEASE_BRAIN": "ssh vy-cluster cluster grant"})}, "VY_LEASE_BRAIN"),
                                      ({1: (LOOP, {**circuits, "VY_POOL_IDLE_S": "30s"})}, "whole number"))):
        with pytest.raises(LookupError, match=why):
            vc.n1_pool_idle("/run/gpu-lease-circuits", vc.n1_pool_loops(fake_proc(tmp_path / str(k), procs)))
    with pytest.raises(LookupError, match="can't read the environment of n1_lease.py loop 1"):
        vc.n1_pool_loops(fake_proc(tmp_path / "unreadable", {1: (LOOP, None)}))
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "n1-lease.log").write_text("")
    monkeypatch.setattr(vc, "n1_pool_loops", lambda: {})
    with pytest.raises(LookupError):
        vc.n1_lease_spans(H0, {str(pool): "circuits"})
    assert vc.n1_lease_spans(H0, {str(tmp_path / "no-log"): "circuits"}) == []


def test_the_loop_defaults_are_n1_lease_pys():
    src = (Path(__file__).parents[1] / "src" / "research" / "pods" / "nebius" / "n1_lease.py").read_text()
    assert f'IDLE_S = int(os.environ.get("VY_POOL_IDLE_S", "{vc.N1_POOL_IDLE_S}"))' in src
    assert f'DIR = Path(os.environ.get("GPU_LEASE_DIR", "{vc.N1_POOL_DIR}"))' in src
    assert 'BRAIN = os.environ.get("VY_LEASE_BRAIN", "")' in src
    assert set(re.findall(r'os\.environ\.get\("(GPU_LEASE_DIR|VY_POOL_IDLE_S|VY_LEASE_BRAIN)"', src)) == set(vc.N1_LOOP_ENV)


def test_record_usage_writes_both_files_once_per_complete_hour(tmp_path):
    rows = vc.usage_rows(MINUTES, SPANS)
    readings = vc.idle_readings(rows, NOW)
    idle, rec = tmp_path / "held-idle-hourly.jsonl", tmp_path / "usage-hourly.jsonl"
    assert vc.record_usage(rows, readings, NOW, H0 - 600, idle, rec) == (len(readings), len(rows))
    assert vc.record_usage(rows, readings, NOW, H0 - 600, idle, rec) == (0, 0)
    lines = [json.loads(ln) for ln in idle.read_text().splitlines()]
    assert lines == [{k: v for k, v in r.items() if k != "complete"} for r in readings]
    assert {r["schema"] for r in map(json.loads, rec.read_text().splitlines())} == {usage.ROW}
