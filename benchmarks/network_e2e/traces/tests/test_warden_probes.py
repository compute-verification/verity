"""What a warden trace records to tell a late tick's cause: the warden's probe (each tick's lateness and work, each
publish, each collection), the host sampler, the cores each process gets, and the summary's layout of the late ticks."""
from __future__ import annotations

import asyncio
import gc
import itertools
import json
import os
import types
from types import SimpleNamespace as NS

import pytest


def test_probe_records_every_ticks_work_each_publish_and_the_collections(mod):
    probe_mod = mod("warden_probe")

    class Warden:
        def __init__(self) -> None:
            self.origin, self.slack, self.lateness, self.params = 10.0, 0.025, [], NS(bucket_ms=100, window=600)

        async def run(self) -> None:
            for t in range(3):
                self.lateness.append(0.001 * t)
                self._tick(t)

        def _tick(self, t: int) -> None:
            if t == 2:
                raise ValueError("a tick that fails is still timed")

    class Publisher:
        def keep(self, record, grid) -> None:
            pass

    ticks = itertools.count(10.0, 0.5)
    probe = probe_mod.Probe(clock=lambda: next(ticks))
    probe.install(Warden, Publisher)
    w = Warden()
    with pytest.raises(ValueError):
        asyncio.run(w.run())
    Publisher().keep(NS(link="requests", window=0), None)
    probe.collections("start", {})
    probe.collections("stop", {"generation": 0, "collected": 0})
    probe.collections("start", {})
    probe.collections("stop", {"generation": 2, "collected": 7})
    doc = probe.doc()
    assert probe.warden is w and doc["lateness_ms"] == [0.0, 1.0, 2.0] and doc["tick_ms"] == [500.0] * 3
    assert doc["publish"] == [[3.0, 500.0, "requests", 0]]
    assert doc["gc"] == [[4.0, 0, 500.0, 0], [5.0, 2, 500.0, 7]] and doc["gc_counts"] == [1, 0, 1]
    assert (doc["origin"], doc["bucket_ms"], doc["window"], doc["slack_ms"]) == (10.0, 100, 600, 25.0)


def test_collections_sees_the_interpreters_own_collections(mod):
    ticks = itertools.count()
    c = mod("warden_probe").Collections(clock=lambda: next(ticks))
    gc.callbacks.append(c)
    try:
        gc.collect(1)
    finally:
        gc.callbacks.remove(c)
    assert c.counts[1] >= 1 and any(g == 1 for _, g, _, _ in c.events)


def test_host_sample_reads_the_hosts_load_and_each_watched_process(mod):
    hs = mod("host_sample")
    assert hs.watches(["warden:12", "server=34"]) == {"warden": (12, False), "server": (34, True)}
    with pytest.raises(SystemExit):
        hs.watches(["12"])
    assert hs.busy([10, 0, 10, 70, 10, 0, 0, 0], [20, 0, 20, 140, 20, 0, 0, 0]) == (0.2, 0.0)
    pid, prev = os.getpid(), {}
    watched = {"me": (pid, False), "tree": (pid, True)}
    first = hs.sample([0], watched, prev)
    second = hs.sample([0], watched, prev)
    assert "busy" not in first and first["procs"]["me"]["cpu_s"] is None
    assert "busy" in second and 0 in second["cpus"] and second["procs"]["me"]["cpu_s"] >= 0
    assert second["procs"]["me"]["threads"] >= 1 and second["procs"]["tree"]["threads"] >= second["procs"]["me"]["threads"]
    assert hs.tree(pid)[0] == pid and len(second["load"]) == 3


def test_cpu_plan_gives_the_warden_and_the_client_cores_of_their_own(mod):
    corpus = mod("corpus")
    assert corpus.cpu_plan(list(range(16, 32))) == {"server": list(range(16, 29)), "client": [29], "warden": [30, 31]}
    assert corpus.cpu_plan([3, 1, 0, 2]) == {"server": [0, 1], "client": [2], "warden": [3]}
    assert corpus.cpu_plan([0, 1, 2]) is None and corpus.taskset(None) == []
    assert corpus.taskset([4, 5]) == ["taskset", "-c", "4,5"]
    assert corpus.egress_base(types.ModuleType("w", "PAD in every slot before a + D")).startswith("release a")
    assert corpus.egress_base(types.ModuleType("w", "PAD in every slot before s + D")).startswith("admission s")
    assert corpus.egress_base(types.ModuleType("w")).startswith("unknown")


def test_summary_lays_each_late_tick_against_the_publishes_and_the_host(tmp_path, mod):
    summary = mod("summary")
    w = tmp_path / "warden"
    w.mkdir()
    origin = 999.95
    (w / "ticks.json").write_text(json.dumps({
        "origin": 5.0, "bucket_ms": 100, "window": 600, "slack_ms": 25.0, "gc_enabled": False, "switch_ms": 1.0,
        "lateness_ms": [0.2, 0.5, 30.0, 0.1, 300.0], "tick_ms": [1.0, 2.0, 3.0, 120.0, 1.0],
        "publish": [[0.35, 40.0, "requests", 0]], "gc": [], "gc_counts": [0, 0, 0]}))
    (w / "host.jsonl").write_text("".join(json.dumps(
        {"t": origin + k / 10, "load": [2.5, 3, 900], "busy": [0.5, 0.0], "cpus": {"0": 0.9, "3": 0.4}, "psi": 1.5,
         "procs": {"warden": {"cpu_s": 0.05, "invol": 2}, "client": {"cpu_s": 0.01, "invol": 0}}}) + "\n"
        for k in range(1, 9)))
    (w / "client_loop.json").write_text(json.dumps({"cpus": [2], "lags": [[0.5, 25.0]], "max_lag_ms": 25.0,
                                                    "lag_ms_at_least": 20.0, "gc_counts": [0, 0, 0]}))
    t = summary.late_ticks(w, origin, summary.warden_cores({"cpus": {"server": [0, 1], "client": [2], "warden": [3]}}))
    assert (t["ticks"], t["over_slack"], t["host"]["samples"], t["host"]["load_1m"]["max"]) == (5, 2, 8, 2.5)
    assert t["lateness_ms"] == {"<1": 3, "1-5": 0, "5-10": 0, "10-25": 0, "25-50": 1, "50-100": 0, "100-250": 0, ">=250": 1}
    late = {x["tick"]: x for x in t["late"]}
    assert sorted(late) == [2, 4] and late[2]["publish"] == [] and late[2]["tick_before_ms"] == 2.0
    assert late[4]["publish"] == [[0.35, 40.0, "requests", 0]] and late[4]["tick_before_ms"] == 120.0
    assert [h["t"] for h in late[4]["host"]] == [0.4, 0.5, 0.6, 0.7]
    assert late[4]["host"][0]["warden_cores_busy"] == {"3": 0.4} and late[4]["host"][0]["warden"]["invol"] == 2
    assert summary.client_loop(w)["stalls"] == 1 and summary.late_ticks(tmp_path, origin, []) is None
    assert summary.warden_cores({"cpus": "not pinned: the job has 2 cores"}) == []
