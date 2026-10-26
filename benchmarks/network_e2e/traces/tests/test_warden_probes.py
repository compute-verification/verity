"""What a warden trace records to tell a late tick's cause: the warden's probe (each tick's lateness and work, each
publish, each collection), the host sampler, the cores each process gets, and the summary's layout of the late ticks."""
from __future__ import annotations

import asyncio
import gc
import itertools
import json
import os
import threading
import types
from pathlib import Path
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
    assert doc["publisher"] is None and doc["cpus"] == sorted(os.sched_getaffinity(0))


def test_probe_follows_a_spawned_publisher_from_both_sides(tmp_path, mod):
    """The warden's side (handoffs, the close's wait, the pid) and the publisher process's (`time_keeps`), joined in
    ticks.json's `publisher`."""
    probe_mod = mod("warden_probe")

    class Warden:
        def __init__(self) -> None:
            self.origin, self.slack, self.lateness, self.params = 10.0, 0.025, [0.0], NS(bucket_ms=100, window=600)

        async def run(self) -> None:
            pass

        def _tick(self, t: int) -> None:
            pass

    class Publisher:
        def __init__(self, out) -> None:
            self.out = out

        def keep(self, record, grid) -> None:
            pass

    class PublisherProcess:
        def __init__(self, out) -> None:
            self.process, self.got = NS(pid=4321), []

        def __call__(self, record, grid) -> None:
            self.got.append(record.window)

        def close(self) -> list:
            return ["refused"]

    ticks = itertools.count(10.0, 0.5)
    probe = probe_mod.Probe(clock=lambda: next(ticks), out=tmp_path)
    probe.install(Warden, type("Inline", (), {"keep": lambda self, r, g: None}), PublisherProcess)
    asyncio.run(Warden().run())
    p = PublisherProcess(tmp_path)
    p(NS(link="requests", window=0), None)
    p(NS(link="responses", window=0), None)
    assert p.got == [0, 0] and p.close() == ["refused"]
    assert (tmp_path / probe_mod.PUBLISHER_PID).read_text() == "4321"
    child = itertools.count(10.6, 1.25)
    probe_mod.time_keeps(Publisher, clock=lambda: next(child))
    Publisher(tmp_path).keep(NS(link="requests", window=0), None)
    Publisher(tmp_path).keep(NS(link="responses", window=0), None)
    assert probe.doc()["publisher"] == {"pid": 4321, "handoff": [[0.0, "requests", 0], [0.5, "responses", 0]],
                                        "keep": [[0.6, 1250.0, "requests", 0], [3.1, 1250.0, "responses", 0]],
                                        "close": [1.0, 0.5]}


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


def test_the_client_grows_its_descriptor_table_first_and_pins_every_thread(mod):
    corpus = mod("corpus")
    n = corpus.reserve_fds(700)
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    assert 0 < n <= 700 and int(status["FDSize"]) >= n
    mine = os.sched_getaffinity(0)
    one = {min(mine)}
    stop = threading.Event()
    other = threading.Thread(target=stop.wait)
    other.start()
    try:
        corpus.pin_threads(one)
        assert os.sched_getaffinity(other.native_id) == one and os.sched_getaffinity(0) == one
    finally:
        corpus.pin_threads(mine)
        stop.set()
        other.join()
    assert os.sched_getaffinity(0) == mine
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
    assert t["publisher"] is None


def test_summary_says_whether_a_spawned_publisher_kept_up(tmp_path, mod):
    summary = mod("summary")
    w = tmp_path / "warden"
    w.mkdir()
    (w / "ticks.json").write_text(json.dumps({
        "origin": 5.0, "bucket_ms": 100, "window": 10, "slack_ms": 25.0, "lateness_ms": [0.1] * 29 + [40.0],
        "tick_ms": [1.0] * 30, "publish": [], "gc": [], "gc_counts": [0, 0, 0],
        "publisher": {"pid": 7, "handoff": [[0.95, "requests", 0], [0.95, "responses", 0], [1.95, "requests", 1],
                                            [1.95, "responses", 1], [2.95, "requests", 2], [2.95, "responses", 2]],
                      "keep": [[0.96, 400.0, "requests", 0], [1.36, 500.0, "responses", 0], [1.96, 400.0, "requests", 1],
                               [2.36, 800.0, "responses", 1], [3.16, 400.0, "requests", 2], [3.56, 200.0, "responses", 2]],
                      "close": [3.0, 0.8]}}))
    t = summary.late_ticks(w, 0.0, [])
    p = t["publisher"]
    assert (p["handed"], p["kept"], p["behind_a_window"], p["close_wait_s"], p["end_past_last_tick_s"]) == (6, 6, 1, 0.8, 0.8)
    assert p["most_behind"] == {"link": "responses", "window": 1, "lag_s": 1.21} and p["lag_s"]["min"] == 0.41
    assert (p["window_busy"]["min"], p["window_busy"]["max"]) == (0.6, 1.2)
    assert t["publish_ms"]["n"] == 6 and [x[2:] for x in t["late"][0]["publish"]] == [["responses", 1]]
