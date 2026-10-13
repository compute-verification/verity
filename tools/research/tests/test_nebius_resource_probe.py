"""The resource steward's probe (`pods/nebius/resource_probe.py`): its watermarks, rates and sources, on synthetic readings."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"
_spec = importlib.util.spec_from_file_location("nebius_resource_probe", NEBIUS / "resource_probe.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

GB = 10**9
T0 = 1_790_800_000.0


def reading(node: str = "n1", *, root_free_gb: float = 150, ws_pct: float = 70.0, ipct: float = 35.0, iused: int = 7_000_000,
            avail_pct: float = 70.0, replay_gb: float = 0.0, reserved_gb: float = 0.0, idle=(), unpres: int = 0, t: float = T0):
    total = 1800 * GB
    ws_size = 5000 * GB
    return {"node": node, "t": t,
            "disk": {"root": {"path": "/", "free_b": root_free_gb * GB, "pct": 40.0, "ipct": 5.0, "iused": 1_000_000, "t": t},
                     "workspace": {"path": "/workspace", "free_b": ws_size * (1 - ws_pct / 100), "pct": ws_pct, "ipct": ipct,
                                   "inodes": 20_000_000, "iused": iused, "t": t}},
            "mem": {"total_b": total, "avail_b": total * avail_pct / 100, "avail_pct": avail_pct},
            "ramlock": {"reserved_gb": reserved_gb},
            "replay_queue": {"queued": 1 if replay_gb else 0, "gb": replay_gb},
            "idle_gpus": list(idle), "unpreserved": {"old": unpres, "oldest": ["r1", "r2", "r3"]}}


def lim(node: str) -> dict:
    return {**P.COMMON, **P.LIMITS[node]}


def test_all_clear_under_every_watermark():
    assert P.evaluate(reading("n1"), None, lim("n1")) == []
    assert P.evaluate(reading("n2", ws_pct=40.0), None, lim("n2")) == []


def test_workspace_watermarks_differ_by_node():
    assert P.evaluate(reading("n1", ws_pct=60.0), None, lim("n1")) == []
    assert P.evaluate(reading("n2", ws_pct=60.0), None, lim("n2"))[0].startswith("HARD /workspace")
    assert P.evaluate(reading("n1", ws_pct=81.0), None, lim("n1"))[0].startswith("/workspace 81.0%")
    assert P.evaluate(reading("n1", ws_pct=86.0), None, lim("n1"))[0].startswith("HARD /workspace")


def test_root_free_alert_and_hard_stop():
    assert P.evaluate(reading(root_free_gb=55), None, lim("n1"))[0].startswith("root free 55 GB")
    assert P.evaluate(reading(root_free_gb=40), None, lim("n1"))[0].startswith("HARD root free 40 GB")


def test_inodes_ram_idle_gpu_and_custody():
    b = P.evaluate(reading(ipct=81.0, avail_pct=8.0, idle=[{"gpu": 3, "min_gib": 50.0}], unpres=11), None, lim("n1"))
    assert [x.split()[0] for x in b] == ["/workspace", "RAM", "GPU", "11"]
    assert P.evaluate(reading(unpres=10), None, lim("n1")) == []


def test_inode_growth_rate_uses_the_previous_reading():
    prev = reading(iused=7_000_000, t=T0)
    assert P.evaluate(reading(iused=7_030_000, t=T0 + 1200), prev, lim("n1")) == []          # 30k in 20 min = 90k/h
    b = P.evaluate(reading(iused=7_060_000, t=T0 + 1200), prev, lim("n1"))                  # 60k in 20 min = 180k/h
    assert len(b) == 1 and "inodes/h" in b[0]
    assert not b[0].startswith("HARD")                                                     # 80% of 20M is 52 h away
    assert P.evaluate(reading(iused=7_060_000, t=T0 + 300), prev, lim("n1")) == []           # a 5-min burst is not a rate
    prev = reading(iused=14_000_000, t=T0)
    b = P.evaluate(reading(iused=15_000_000, t=T0 + 3000), prev, lim("n1"))                 # 1.2M/h over 50 min, 1M left
    assert b[0].startswith("HARD /workspace gaining 1,200,000 inodes/h over 50 min: 80% of inodes in 0.8 h")


def test_an_inode_rate_over_a_young_base_warns_and_pages_only_once_it_holds():
    prev = [reading(iused=14_000_000, t=T0)]
    b = P.evaluate(reading(iused=14_400_000, t=T0 + 1200), prev, lim("n1"))                 # 1.2M/h over 20 min
    assert len(b) == 1 and not b[0].startswith("HARD") and "pages once it holds for 45 min" in b[0]
    base = None
    for i, iused in enumerate((7_000_000, 7_010_000, 7_020_000, 7_030_000, 7_040_000, 7_400_000)):
        r = reading(iused=iused, t=T0 + 600 * i)                                            # a burst in the last 10 min
        b = P.evaluate(r, base, lim("n1"))
        base = P.next_base(r, base, lim("n1"))
    assert b == ["/workspace gaining 480,000 inodes/h > 100,000"]                            # over the oldest base: 50 min
    assert [x["t"] for x in base] == [T0 + 600 * i for i in range(6)]
    assert [x["t"] for x in P.next_base(reading(t=T0 + 3800), base, lim("n1"))][0] == T0 + 600


def test_the_inode_line_names_the_directories_that_grew_most(tmp_path):
    ws = "/workspace"
    (tmp_path / "inode-dirs.prev.txt").write_text(f"10\t{ws}/a/x\n5\t{ws}/a/y\n15\t{ws}/a\n7\t{ws}/b\n40\t{ws}\n")
    (tmp_path / "inode-dirs.txt").write_text(f"10\t{ws}/a/x\n905\t{ws}/a/y\n915\t{ws}/a\n507\t{ws}/b\n9\t{ws}/c/new\n"
                                             f"9\t{ws}/c\n1440\t{ws}\n")
    for f, age in (("inode-dirs.prev.txt", 3700), ("inode-dirs.txt", 100)):
        os.utime(tmp_path / f, (T0 - age, T0 - age))
    g = P.inode_dirs(tmp_path, ws, T0, 3600, walk=False)
    assert g == {"over_s": 3600, "top": [{"dir": "a/y", "gained": 900}, {"dir": "b", "gained": 500},
                                         {"dir": "c/new", "gained": 9}]}
    r = reading(iused=15_000_000, t=T0 + 3000)
    r["inode_dirs"] = g
    line = P.evaluate(r, reading(iused=14_000_000, t=T0), lim("n1"))[0]
    assert line.endswith("(growing a/y +900, b +500, c/new +9 in 60 min)")
    assert P.inode_dirs(tmp_path / "none", ws, T0, 3600, walk=False) is None


def test_replay_queue_against_half_of_available_ram_after_reservations():
    # 1800 GB * 40% = 720 GB available; half is 360
    assert P.evaluate(reading(avail_pct=40.0, replay_gb=350), None, lim("n1")) == []
    assert "queued replay" in P.evaluate(reading(avail_pct=40.0, replay_gb=350, reserved_gb=100), None, lim("n1"))[0]


def test_replay_queue_kueue_counts_pending_replays_at_a_floor():
    def wl(name, conds, mem):
        return {"metadata": {"name": name}, "status": {"conditions": [{"type": k, "status": "True"} for k in conds]},
                "spec": {"podSets": [{"count": 1, "template": {"spec": {"containers": [{"resources": {"requests": {"memory": mem}}}]}}}]}}
    ws = [wl("job-a-replay-0", [], "64G"), wl("job-b-replay-0", ["QuotaReserved"], "64G"), wl("job-c-replay-0", ["Finished"], "64G"),
          wl("job-d-build-0", [], "64G"), wl("job-e-replay-0", [], "128Gi")]
    q = P.replay_queue_kueue(ws, floor_gb=90.0)
    assert q["queued"] == 2
    assert q["gb"] == round(90.0 + 128 * 2**30 / GB, 1)


def test_replay_queue_fill_counts_queued_replays_and_running_commits(tmp_path):
    (tmp_path / "queue").mkdir()
    (tmp_path / "running").mkdir()
    (tmp_path / "queue/verity-replay-y.sh").write_text("#!/usr/bin/env bash\n# fill: owner=o gpus=0 mem_gb=200\n")
    (tmp_path / "queue/verity-replay-z.sh").write_text("#!/usr/bin/env bash\n# fill: owner=o gpus=0 mem_gb=64\n")
    (tmp_path / "queue/verity-commit-q.sh").write_text("# fill: owner=o gpus=1\n")          # not running: reserves nothing
    (tmp_path / "running/verity-commit-r.sh").write_text("# fill: owner=o gpus=1\n")
    (tmp_path / "running/.verity-commit-r.sh.pgid").write_text("4242\n")                    # the runner's, not a job
    (tmp_path / "running/.verity-commit-r.sh.unit").write_text("fill-verity-commit-r\n")
    (tmp_path / "queue/aw-other.sh").write_text("# fill: owner=o mem_gb=500\n")
    q = P.replay_queue_fill(tmp_path, floor_gb=90.0)
    assert (q["queued"], q["gb"]) == (3, 380.0)


def test_idle_gpus_sampler_needs_a_full_window_of_zero_util(tmp_path):
    import time
    def line(t, g0, g1):
        return json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
                           "gpus": [{"i": 0, "util": g0[0], "mem_mib": g0[1]}, {"i": 1, "util": g1[0], "mem_mib": g1[1]}]})
    now = T0
    lines = [line(now - s, (0, 30_000), (0 if s > 60 else 5, 30_000)) for s in range(1000, -1, -10)]
    idle = P.idle_gpus_sampler(lines, now, 900, 20.0)
    assert [g["gpu"] for g in idle] == [0]
    assert P.idle_gpus_sampler(lines[-30:], now, 900, 20.0) == []                           # 5 min of history is not 15
    f = tmp_path / "u.jsonl"
    f.write_text("\n".join(lines) + "\n")
    assert P.tail_lines(f, nbytes=500)[-1] == lines[-1]
    assert all(x.startswith("{") for x in P.tail_lines(f, nbytes=500))


def test_idle_from_series_needs_one_pod_for_the_whole_window():
    util = {(0, "new-pod"): 0.0, (0, "old-pod"): 0.0, (1, "idle-pod"): 0.0, (2, "busy-pod"): 40.0, (3, ""): 0.0}
    fb = {(0, "new-pod"): 50_000, (0, "old-pod"): 50_000, (1, "idle-pod"): 88_000, (2, "busy-pod"): 90_000, (3, ""): 90_000}
    n = {(0, "new-pod"): 16, (0, "old-pod"): 44, (1, "idle-pod"): 60, (2, "busy-pod"): 60, (3, ""): 60}
    assert P.idle_from_series(util, fb, n, 20.0) == [{"gpu": 1, "min_gib": 85.9, "pod": "idle-pod"}]


def test_pool_flagged_reads_only_a_fresh_pool_file(tmp_path):
    import time
    f = tmp_path / "pool.json"
    def write(age_s):
        f.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 - age_s)),
                                 "nodes": {"n1": {"monitors": [{"kind": "gpu-idle-in-lease", "gpu": 1},
                                                               {"kind": "gpu-unleased", "gpu": 2}, {"kind": "template-drift"}]}}}))
    write(60)
    assert P.pool_flagged(f, T0) == {1}
    write(1800)
    assert P.pool_flagged(f, T0) == set()
    assert P.pool_flagged(tmp_path / "absent.json", T0) == set()


def test_unpreserved_counts_finished_runs_without_a_marker(tmp_path):
    def run(name, state, age_s, marker=None):
        d = tmp_path / name
        d.mkdir()
        (d / "status.json").write_text(json.dumps({"transitions": [{"state": state}]}))
        os.utime(d / "status.json", (T0 - age_s, T0 - age_s))
        if marker:
            (d / marker).write_text("{}")
    run("r20260930-010000-aaaa", "done", 7200)
    run("r20260930-020000-bbbb", "failed", 7200)
    run("r20260930-030000-cccc", "done", 7200, ".custody")
    run("r20260930-040000-dddd", "done", 7200, ".fetched")
    run("r20260930-050000-eeee", "running", 7200)
    run("r20260930-060000-ffff", "done", 60)
    (tmp_path / "cfgtp2-cpu").mkdir()
    u = P.unpreserved(tmp_path, T0, 3600)
    assert (u["old"], u["recent"], u["oldest"]) == (2, 1, ["r20260930-010000-aaaa", "r20260930-020000-bbbb"])


def test_ramlock_skips_stale_reservations(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"owner": "x", "gb": 90, "pid": os.getpid()}))
    (tmp_path / "b.json").write_text(json.dumps({"owner": "y", "gb": 50}))
    (tmp_path / "c.json").write_text(json.dumps({"owner": "z", "gb": 70, "pid": 2**22 + 12345}))
    r = P.ramlock(tmp_path)
    assert (r["reserved_gb"], r["stale"]) == (140.0, 1)


def test_meminfo_parses_available():
    m = P.meminfo("MemTotal:       1000000 kB\nMemFree:  1 kB\nMemAvailable:    250000 kB\n")
    assert m["avail_pct"] == 25.0
