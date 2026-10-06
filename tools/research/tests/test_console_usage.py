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
    assert (vc.NO_POD, vc.POOL_UNLEASED, vc.USAGE_RECORD) == (usage.NO_POD, usage.POOL_UNLEASED, Path(usage.RECORD))
    assert set(vc.NODE_NAMES.values()) == set(usage.NODES.values())


def test_a_console_without_its_cluster_tree_says_which_tree_to_install(tmp_path):
    # red team T: node 1 has neither /workspace/research/console/cluster nor /workspace/cluster/src
    console = tmp_path / "research" / "console"
    console.mkdir(parents=True)
    (console / "verity_console.py").write_bytes((Path(__file__).parents[1] / "console" / "verity_console.py").read_bytes())
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import verity_console as v\n"
             "try:\n    v.usage_rows({'node 2': []})\nexcept ModuleNotFoundError as e:\n    print(e)")
    out = subprocess.run([sys.executable, "-I", "-S", "-c", probe, str(console)], capture_output=True, text=True, check=True).stdout
    assert f"install deploy.toml's tree {console / 'cluster'} (research deploy install vy-nebius-1 --by WHO" in out


LEASE = "pid={} run=- who={} since=2026-10-03T07:00:00Z cmd=x"


def n2_minutes_of(tmp_path, samples: list[dict]) -> list:
    util = tmp_path / "util"
    util.mkdir()
    f = util / "2026-10-03.jsonl"
    f.write_text("".join(json.dumps(s) + "\n" for s in samples))
    os.utime(f, (H0 + 100, H0 + 100))
    return json.loads(subprocess.run([sys.executable, "-", str(int(H0)), str(util)], input=vc.N2_MINUTES, capture_output=True,
                                     text=True, check=True).stdout)


def test_node_2s_minutes_count_a_timed_windows_samples_unmeasured(tmp_path):
    samples = [{"t": "2026-10-03T08:00:00Z", "gpus": [
                    {"i": 0, "util": 40, "mem_mib": 2048, "lease": LEASE.format(1, "fill:bc-15ada664"), "procs": [{"pid": 1}]},
                    {"i": 1, "util": 0, "mem_mib": 0, "lease": None, "procs": []}]},
               {"t": "2026-10-03T08:00:10Z", "timed": True, "gpus": [{"i": 0, "lease": LEASE.format(2, "circuits-tp8")},
                                                                       {"i": 1, "lease": LEASE.format(2, "circuits-tp8")}]}]
    assert n2_minutes_of(tmp_path, samples) == [[H0, 0, 0.5, 0, 0, 0, "fill:bc-15ada664", 0.5],
                                                [H0, 1, 0, 0, 0, 0.5, "circuits-tp8", 0.5]]


def test_a_util_nvidia_smi_couldnt_read_is_unmeasured_and_a_minute_goes_to_the_holder_of_most_of_it(tmp_path):
    # red team R: the sampler writes `"util": null` for "[N/A]", which counted as held idle
    # red team S: one sample of a process before its lease file booked the minute to "(process, no lease)", never useful
    fill = {"util": 50, "mem_mib": 40000, "lease": LEASE.format(1, "fill:bc-15ada664"), "procs": [{"pid": 1}]}
    samples = [{"t": "2026-10-03T08:00:00Z", "gpus": [{"i": 0, **fill, "lease": None}, {"i": 1, **fill, "util": None}]}]
    samples += [{"t": f"2026-10-03T08:00:{s}Z", "gpus": [{"i": 0, **fill}, {"i": 1, **fill, "util": None}]} for s in (10, 20, 30)]
    samples += [{"t": f"2026-10-03T08:00:{s}Z", "gpus": [{"i": 0, **fill, "lease": LEASE.format(2, "verity:bc-b8aaadaa")},
                                                         {"i": 1, **fill, "util": None}]} for s in (40, 50)]
    assert n2_minutes_of(tmp_path, samples) == [[H0, 0, 1.0, 0, 0, 0, "fill:bc-15ada664", 0.0],
                                                [H0, 1, 0, 0, 0, 0, "fill:bc-15ada664", 1.0]]


def test_node_1s_minutes_are_read_at_each_minutes_end_and_memory_holds_by_the_minute(monkeypatch):
    # red team N: a pass's own seconds set the instants, so two passes read a minute from different scrapes
    asked = []

    def prom_range(query: str, start: float, end: float) -> list[dict]:
        asked.append(start)
        ts = range(int(start), int(end) + 1, 60)
        if "FB_USED" in query:
            return [{"metric": {"gpu": "0"}, "values": [[t, "2048"] for t in ts]}]
        return [{"metric": {"gpu": "0"}, "values": [[t, "50"] for t in ts]},
                {"metric": {"gpu": "1", "pod": "nd-vllm-epoch-run-0123456789-x", "namespace": "kueue"},
                 "values": [[t, "0"] for t in ts]}]

    monkeypatch.setattr(vc, "prom_range", prom_range)
    rows = vc.n1_minutes(H0 + 37, H0 + 130)
    assert asked == [H0 + 59, H0 + 59]
    assert rows == [(H0, 0, 1.0, 0, 0, 0, vc.NO_POD), (H0, 1, 0, 1.0, 0, 0, "Kueue vllm-epoch-run"),
                    (H0 + 60, 0, 1.0, 0, 0, 0, vc.NO_POD), (H0 + 60, 1, 0, 1.0, 0, 0, "Kueue vllm-epoch-run")]


def test_lanes_and_kinds_of_each_nodes_holders():
    assert [(vc.lane_of("node 1", h), vc.kind_of("node 1", h)) for h in (
        "gpu-pool-circuits-1791-x", "gpu-pool-1791-y", "Kueue vllm-epoch-run", "SkyPilot m0-*", vc.NO_POD)] == [
        ("circuits", "gpu-lease pool"), (vc.POOL_UNLEASED, "gpu-lease pool"), ("circuits", "Kueue vllm-epoch-run"),
        ("proofs", "SkyPilot m0-*"), (vc.NO_POD, vc.NO_POD)]
    assert [(vc.lane_of("node 2", h), vc.kind_of("node 2", h)) for h in (
        "fill bc-15ada664", "bc-b8aaadaa", "circuits-tp8", "(process, no lease)")] == [
        ("memory-accounting", "fill"), ("circuits", "gpu-lease"), ("circuits-tp8", "gpu-lease"),
        ("(process, no lease)", "(process, no lease)")]


def test_a_leases_who_names_its_lane():
    assert [vc.who_lane(w) for w in (
        "keeper:memory-accounting", "keeper:circuits", "fill:bc-15ada664", "fill:network-accounting", "verity:bc-ecea50f6",
        "memory-accounting:erase-calib", "keeper:someone", "adhoc:ubuntu", "infra2:x", "circuits-tp8")] == [
        "memory-accounting", "circuits", "memory-accounting", "network-accounting", "network-accounting",
        "memory-accounting", "keeper:someone", "adhoc:ubuntu", "infra2:x", "circuits-tp8"]


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
        ("vy-nebius-1", usage.POOL_UNLEASED, usage.POOL_UNLEASED), ("vy-nebius-1", vc.NO_POD, vc.NO_POD),
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
    spans = vc.n1_lease_spans(H0 + 3600, pools, {str(circuits): 30, str(provers): 120}, tmp_path / "lease-usage.jsonl")
    assert sorted(spans) == [(3, H0, H0 + 70, "circuits"), (4, H0 + 600, H0 + 1170, "circuits"), (5, H0, H0 + 120, "proofs")]
    minutes = {"node 1": [(H0, 3, 1.0, 0, 0, 0, "gpu-pool-circuits-1791-x"), (H0 + 60, 3, 1.0, 0, 0, 0, "gpu-pool-circuits-1791-x")]}
    rows = {(r["owner"], r["kind"]): r for r in vc.usage_rows(minutes, {"node 1": spans[:1]})}
    assert (rows[("circuits", "gpu-lease")]["useful_s"], rows[("circuits", usage.POOL_UNLEASED)]["busy_s"]) == (70, 50)


def usage_record(gpu: int, who: str, since: str, end: str) -> str:
    return json.dumps({"schema": "gpu-lease/usage/v1", "pid": 1, "scope": None, "who": who, "run": None, "timed": False,
                       "preemptible": False, "since": f"2026-10-03T{since}Z", "end": f"2026-10-03T{end}Z", "rc": 0, "cmd": "x",
                       "gpus": [{"index": gpu, "uuid": f"GPU-{gpu}", "held_s": 1, "busy_s": 1, "sampled_s": 1}]}) + "\n"


def test_a_pool_span_counts_to_the_lanes_of_the_leases_in_it(tmp_path):
    # the provers pool serves every lane: its leases were memory-accounting's keeper cells, network-accounting's runs and
    # adhoc:ubuntu, and the report named proofs for all of them
    pool = tmp_path / "gpu-lease"
    pool.mkdir()
    (pool / "n1-lease.log").write_text("".join(json.dumps({"t": f"2026-10-03T{t}Z", "ev": ev, "gpu": g}) + "\n" for t, ev, g in (
        ("08:00:00", "unfenced", 5), ("08:00:00", "unfenced", 6), ("08:10:00", "shrink", 6), ("08:30:00", "shrink", 5),
        ("08:40:00", "unfenced", 7))))
    (pool / "n1-lease.json").write_text(json.dumps({"t": "2026-10-03T08:59:00Z", "gpus": [
        {"index": 5, "lease": None},
        {"index": 7, "lease": {"who": "adhoc:ubuntu", "run": None, "since": "2026-10-03T08:40:05Z", "until": None}}]}))
    log = tmp_path / "lease-usage.jsonl"
    log.write_text(usage_record(5, "keeper:memory-accounting", "08:00:10", "08:12:00") + "torn{\n"
                   + usage_record(5, "verity:bc-ecea50f6", "08:15:00", "08:28:00")
                   + usage_record(5, "keeper:circuits", "09:30:00", "09:40:00")       # after the span: not in it
                   + json.dumps({"schema": "other", "who": "x"}) + "\n")
    spans = vc.n1_lease_spans(H0 + 3600, {str(pool): "proofs"}, {str(pool): 120}, log)
    assert sorted(spans) == [(5, H0 + 10, H0 + 720, "memory-accounting"), (5, H0 + 900, H0 + 1680, "network-accounting"),
                             (6, H0, H0 + 480, "proofs"),                       # no record names its who: the pool's lane
                             (7, H0 + 2405, H0 + 3600, "adhoc:ubuntu")]         # live: until now, not the state's time
    minutes = {"node 1": [(H0, 5, 1.0, 0, 0, 0, "gpu-pool-1791-y"), (H0 + 720, 5, 0, 1.0, 0, 0, "gpu-pool-1791-y")]}
    rows = {(r["owner"], r["kind"]): r for r in vc.usage_rows(minutes, {"node 1": spans})}
    assert rows[("memory-accounting", "gpu-lease")]["useful_s"] == 50
    assert (rows[(usage.POOL_UNLEASED, usage.POOL_UNLEASED)]["busy_s"], rows[(usage.POOL_UNLEASED, usage.POOL_UNLEASED)]["idle_s"],
            rows[(usage.POOL_UNLEASED, usage.POOL_UNLEASED)]["useful_s"]) == (10, 60, 0)
    assert not any(r["owner"] == "proofs" for r in rows.values() if r["kind"] != "gpu-lease")


def test_a_live_lease_a_record_already_has_counts_once(tmp_path):
    pool = tmp_path / "gpu-lease"
    pool.mkdir()
    (pool / "n1-lease.json").write_text(json.dumps({"t": "2026-10-03T08:20:00Z", "gpus": [
        {"index": 0, "lease": {"who": "keeper:circuits", "since": "2026-10-03T08:00:00Z"}}]}))
    log = tmp_path / "lease-usage.jsonl"
    log.write_text(usage_record(0, "keeper:circuits", "08:00:00", "08:10:00"))
    assert vc.n1_leases(H0 + 3600, {str(pool): "proofs"}, log) == [(0, H0, H0 + 600, "keeper:circuits")]
    assert vc.n1_leases(H0 + 3600, {str(pool): "proofs"}, tmp_path / "none") == [(0, H0, H0 + 3600, "keeper:circuits")]


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
