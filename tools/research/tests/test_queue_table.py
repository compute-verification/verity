"""`research queue stats` with no `--log`: the queue's table (`queue_table`), its pure parts on a small made-up day."""
import ast
import importlib.util
import json
import types

import pytest
from research import queue_table as qt

DAY = "2026-11-10"
T0 = qt.ts(f"{DAY}T00:00:00Z")


def at(hhmm: str) -> str:
    return f"{DAY}T{hhmm}:00Z"


def lab(key: str, value, hhmm: str) -> dict:
    return {"key": key, "value": value, "by": "x", "ts": at(hhmm)}


def test_a_quantile_interpolates_between_the_two_nearest_values():
    assert qt.pct([], .5) is None
    assert qt.pct([7], .9) == 7
    assert qt.pct([4, 1, 3, 2], .5) == 2.5
    assert qt.pct(list(range(11)), .9) == 9
    assert qt.pct([0, 10], .9) == 9


def test_a_run_ids_time_is_when_it_was_asked_for():
    assert qt.rid_time("r20261110-201216-abcd") == qt.ts(at("20:12")) + 16


def test_a_mark_is_the_first_true_after_the_last_false():
    labs = [lab("ready", True, "01:00"), lab("ready", False, "02:00"), lab("ready", "true", "03:00"), lab("ready", True, "04:00"),
            lab("reviewed", "land", "05:00")]
    assert qt.mark_at(labs, qt.ts(at("06:00"))) == qt.ts(at("03:00"))
    assert qt.mark_at(labs, qt.ts(at("02:30"))) is None      # ejected then, not yet marked again
    assert qt.mark_at(labs, qt.ts(at("01:30"))) == qt.ts(at("01:00"))
    assert qt.mark_at([], T0) is None


def test_a_verdict_is_the_first_land_and_a_newest_hold_is_a_hold():
    assert qt.verdict_at([lab("reviewed", "land", "02:00"), lab("reviewed", "land", "03:00")]) == (qt.ts(at("02:00")), False)
    assert qt.verdict_at([lab("reviewed", "land", "02:00"), lab("reviewed", "hold", "03:00")]) == (qt.ts(at("02:00")), True)
    assert qt.verdict_at([lab("ready", True, "02:00")]) == (None, False)


def test_a_drop_is_read_from_the_loops_reason():
    assert qt.drop_cause("on main") == "on main"
    assert qt.drop_cause("its base was dropped") == qt.drop_cause("its base failed") == "dropped base"
    assert qt.drop_cause("not on main's chain: its base dddddddddddd is neither main nor a chain member") == "dropped base"
    assert qt.drop_cause("#12's ready mark was withdrawn") == "withdrawn mark"
    assert qt.drop_cause("failed: step lean-audit, step lean-suites") == "red"
    assert qt.drop_cause("#13 isn't taken now: ejected at 0a0a0a0a: candidate cccccccccccc fails pytest") == "red"
    # a hold whose text speaks of a withdrawn copy is still a hold
    assert qt.drop_cause("#14 isn't taken now: held by a lead (red in candidate dddd: a withdrawn copy reads WITHDRAWN)") == "hold"
    assert qt.drop_cause("ci dropped it by hand: its base dddddddddddd is red") == "by hand"
    assert qt.drop_cause("eeeeeeeeeeee's check ended in infra errors: the run beside its check registered no run") == "infrastructure"
    assert qt.drop_cause("something new") == "other"


def test_a_reds_kind_is_read_from_its_failed_tasks():
    assert qt.red_kind("failed: step pytest", ["step pytest"]) == "test"
    assert qt.red_kind("failed: step lean-audit, step lean-suites", []) == "Lean audit"
    assert qt.red_kind("failed: step preflight-fixtures", []) == "preflight"
    assert qt.red_kind("x's check ended in infra errors", []) == "infrastructure"
    assert qt.red_kind("#7 isn't taken now: held by top (red in candidate c, repository suite): tests/test_x.py::test_y fails: "
                       "security/lean-audit.json is unwitnessed", []) == "test"         # a test that names a Lean file


def test_the_check_nodes_are_the_loops_on_flag():
    assert qt.check_nodes("--batch 16\n# a comment\n--on vy-nebius-cpu-13,vy-nebius-cpu-7\n--gate check\n") == ["vy-nebius-cpu-13", "vy-nebius-cpu-7"]
    assert qt.check_nodes("") == []
    appended = "--on vy-nebius-cpu-1,vy-nebius-cpu-2\n# it runs --on vy-nebius-cpu-7\n--shards auto:5 --depth 2 --on vy-nebius-cpu-13,vy-nebius-cpu-9\n"
    assert qt.check_nodes(appended) == ["vy-nebius-cpu-13", "vy-nebius-cpu-9"]       # a deploy appends a line, and the last flag wins


def test_a_shards_gate_ends_with_its_last_step_on_the_gate():
    off = ("circuit-check",)
    shard = {"steps": [{"step": "pytest", "state": "passed", "end": 100.0}, {"step": "circuit-check", "state": "passed", "end": 900.0}]}
    assert qt.gate_end(shard, off) == 100.0
    assert qt.gate_end({"steps": [{"step": "pytest", "state": "stopped", "end": 100.0}]}, off) is None   # cancelled
    assert qt.gate_end({"steps": [{"step": "pytest", "state": "running"}]}, off) is None
    assert qt.gate_end({"steps": []}, off) is None


def test_the_reading_an_hour_back_is_the_nearest_to_an_hour():
    now = 10_000.0
    readings = [{"now": now - 60}, {"now": now - 2000}, {"now": now - 3500}, {"now": now - 9000}]
    assert qt.earlier(readings, now) == {"now": now - 3500}
    assert qt.earlier([{"now": now - 60}], now) is None


def decisions() -> list[dict]:
    """Candidate A (#1, #2) goes green; B (#3, #4) on top of it is red on a test, C on top of B falls with it; D rebuilds #4."""
    def row(action, hhmm, sha, **kw):
        return {"action": action, "at": at(hhmm), "sha": sha * 40, "shadow": False, "done": True, "why": "", "failed": [], "ids": [],
                "verdict": None, **kw}

    return [
        row("build", "01:00", "a", members=[1, 2], heads=["h1", "h2"]),
        row("slim", "01:10", "a", ids=["r20261110-010500-aaaa"]),
        row("build", "01:20", "b", members=[3, 4], heads=["h3", "h4"]),
        row("slim", "01:30", "b", ids=["r20261110-012500-bbbb"]),
        row("build", "01:40", "c", members=[5], heads=["h5"]),
        row("slim", "02:00", "a", ids=["r20261110-010500-aaaa"], verdict="pass"),
        row("advance", "02:05", "a", members=[1, 2], heads=["h1", "h2"]),
        row("drop", "02:06", "a", why="on main", cancel="cancelled"),
        row("drop", "02:30", "b", why="failed: step pytest", failed=["step pytest"], cancel="cancelled"),
        row("drop", "02:31", "c", why="its base was dropped"),
        row("build", "02:40", "d", members=[4], heads=["h4"]),
    ]


def runs() -> dict[str, dict]:
    took = qt.ts(at("01:06"))
    return {
        "r20261110-010500-aaaa": {"id": "r20261110-010500-aaaa", "kind": "shard", "shard": "1/1", "took": took, "end": took + 3600, "rc": 0,
                                  "failed": [], "suites": [["tools/research", 2700.0, 8]],
                                  "slow_test": ["tools/research", "tests/test_x.py::test_slow", 2400.0], "steps": [{"step": "pytest", "state": "passed", "seconds": 3000, "end": took + 3000},
                                                          {"step": "circuit-check", "state": "passed", "seconds": 3500, "end": took + 3600}]},
        "r20261110-012500-bbbb": {"id": "r20261110-012500-bbbb", "kind": "shard", "shard": "1/1", "took": qt.ts(at("01:26")), "end": None,
                                  "rc": None, "failed": [["tools/research", qt.ts(at("02:00"))]], "slot": "b", "slot_cores": 96,
                                  "scope_cpu_s": 94 * 60 * 12.0, "scope_anon_gb": 40.0, "scope_peak_gb": 120.0,
                                  "parallel": [{"suite": "tools/wide", "worker_gb": 8.0, "default_gb": 9.0, "own": [20.0, 22.0, 21.0],
                                                "tree": [20.0, 22.5, 21.0]},
                                               {"suite": "tools/plain", "worker_gb": None, "default_gb": 9.0, "own": [3.0], "tree": [7.5]}],
                                  "steps": [{"step": "pytest", "state": "stopped", "seconds": 3800, "end": qt.ts(at("02:30"))}]},
    }


def test_candidates_carry_their_outcome_their_shards_and_a_reds_times():
    cs = {c["sha"][0]: c for c in qt.candidates(decisions(), runs(), T0)}
    assert [cs[k]["outcome"] for k in "abcd"] == ["green", "red", "dropped", "running"]
    assert cs["a"]["verdict"] == qt.ts(at("02:00")) and cs["a"]["landed"] == qt.ts(at("02:05"))
    b = cs["b"]
    assert (b["cause"], b["kind"]) == ("red", "test") and cs["c"]["cause"] == "dropped base"
    assert b["first_failure"] == qt.ts(at("02:00")) and b["dropped"] == qt.ts(at("02:30")) and b["rebuilt"] == qt.ts(at("02:40"))
    assert [s["id"] for s in cs["a"]["shards"]] == ["r20261110-010500-aaaa"]
    rs = runs()
    rs["r20261110-012500-bbbb"]["steps"].append({"step": "ratchet-gate", "state": "failed", "seconds": 60, "end": qt.ts(at("01:40"))})
    first = lambda **kw: {c["sha"][0]: c for c in qt.candidates(decisions(), rs, T0, **kw)}["b"]["first_failure"]   # noqa: E731
    assert first() == qt.ts(at("02:00"))                         # a step off the gate fails without making its candidate red
    assert first(off_gate=()) == qt.ts(at("01:40"))


def doc() -> dict:
    labels = {
        "1@h1": [lab("ready", True, "00:10"), lab("reviewed", "land", "00:40")],      # 115 min from mark to main
        "2@h2": [lab("ready", True, "00:50")],                                        # low-risk: no verdict; 75 min
        "3@h3": [lab("ready", True, "01:00"), lab("reviewed", "land", "01:05")],
        "4@h4": [lab("ready", True, "01:10"), lab("reviewed", "hold", "01:15")],
        "5@h5": [lab("ready", True, "01:30")],
        "6@h6": [lab("ready", False, "01:00")],
    }
    node = {"cores": 100, "busy": 0.25, "stat": [5000, 20000], "hz": 100, "load": ["1", "1", "1"], "now": qt.ts(at("03:00")),
            "slots": [{"pool": "check", "slots": 2, "busy": 1, "blocked": 0, "waiting": 0, "mem": {"b": 300}},
                      {"pool": "research", "slots": 4, "busy": 4, "blocked": 0, "waiting": 3, "mem": {}}],
            "runs": [*runs().values(),
                     {"id": "r20261110-020000-cccc", "kind": "tier", "took": qt.ts(at("02:20")), "end": qt.ts(at("02:50")), "slot_state": "held",
                      "slot": "r0", "slot_cores": 16, "cpu_s": 3600.0, "mem_peak_gb": 8.0},
                     {"id": "r20261110-024500-dddd", "kind": "tier", "took": None, "end": None, "slot_state": "waiting"},
                     {"id": "r20261110-023000-eeee", "kind": "worker", "took": qt.ts(at("02:30")), "end": qt.ts(at("02:40")), "slot": "g0",
                      "slot_cores": 32, "cpu_s": 1200.0, "mem_peak_gb": 2.0, "tasks": 3, "first_own_s": 5.0,
                      "results": [qt.ts(at("02:30")) + 15, qt.ts(at("02:35")), qt.ts(at("02:40"))]}]}
    return {"now": qt.ts(at("03:00")), "day": DAY, "errors": [], "check_nodes": ["vy-nebius-cpu-1"], "off_gate": ["circuit-check"],
            "control": {"decisions": decisions(), "ticks": [at("02:50"), at("02:54"), at("02:58")], "args": "",
                        "executor": [{"at": "20261110T010000", "sha": "a" * 12, "tasks": {"hit": 30, "done": 10}},
                                     {"at": "20261110T024000", "sha": "d" * 12, "tasks": {"hit": 10, "waiting": 6, "queued": 3, "leased": 1}},
                                     {"at": "20261110T005000", "sha": "e" * 12, "tasks": {"queued": 2}}]},
            "nodes": {"vy-nebius-cpu-1": node}, "open": {"3": "h3", "4": "h4", "5": "h5", "6": "h6"}, "labels": labels,
            "lean": {"a" * 12: [5, 0], "b" * 12: [3, 2]}}


def test_the_table_holds_each_number_over_its_windows():
    d = doc()
    rows = {r["name"]: r for r in qt.table(d, {"now": d["now"] - 3600, "nodes": {"vy-nebius-cpu-1": {"stat": [5000 - 3600 * 100 * 30, 0]}}})}
    m2m = rows["Mark to `main`"]
    assert m2m["hour"] == m2m["day"] == "p50 95, p90 111 min (n=2)"            # 115 and 75 min, landed 55 min ago
    assert rows["Share on `main` within 2 h of the mark"]["day"] == "2 of 2 landed (100%)"
    assert rows["Marked and waiting now (the backlog)"]["now"].startswith("3 PRs: 1 with a land verdict, 2 without; 1 in a candidate being checked")
    assert rows["Mark to land verdict (the review wait)"]["day"] == "p50 18, p90 28 min (n=2)"     # 30 and 5 min
    assert rows["Marked heads with no land verdict now"]["now"] == "2: 1 held by a reviewer, 1 not looked at, the oldest of those marked 90 min ago"
    assert rows["Quick tier: waiting for a slot"]["now"] == "1 in line, longest 15 min so far"
    assert rows["Quick tier: waiting for a slot"]["day"] == "p50 20, p90 20 min (n=1)"
    assert rows["Quick tier: running"]["hour"] == "p50 30, p90 30 min (n=1)"
    assert rows["Candidate: build to verdict"]["day"] == "p50 60, p90 60 min (n=1)"
    plain = rows["Round with no Lean change: build to verdict"]
    assert plain["day"] == ("p50 60, p90 60 min (n=1), of 1 with a verdict; slowest step `pytest`, p50 50 min, the slowest in 1 of 1; "
                              "slowest test `tests/test_x.py::test_slow` of suite `tools/research`, p50 40 min, the slowest in 1 of 1")
    assert plain["now"] == ("0 of the 1 being checked change no Lean; the newest verdict: built 01:00Z in 1 shards, 60 min, 0 of its 5 "
                            "changed files a Lean change, slowest step `pytest` 50 min")
    assert plain["target"] == "30"
    assert rows["The floor: a round's slowest suite and slowest test"]["day"] == (
        "slowest suite `tools/research`, p50 45 min, the slowest in 1 of 1; slowest test `tests/test_x.py::test_slow` of suite "
        "`tools/research`, p50 40 min, the slowest in 1 of 1")
    assert rows["Candidates by outcome"]["day"] == "4 built: 1 green, 1 red, 1 dropped for another reason, 1 being checked"
    assert rows["Check that started no shard: build to its `infra` verdict"]["day"] == "none"
    shard = rows["Shard 1/1: its slot to the end of its last step on the gate"]["day"]
    assert shard.startswith("p50 50, p90 50 min (n=1); slowest step `pytest` in 1 of 1, p50 50 min; then it holds the slot p50 10, p90 10 min more")
    assert rows["Red: first failure to the candidate's drop"]["day"] == "p50 30, p90 30 min (n=1)"
    assert rows["Red: drop to the next candidate built"]["day"] == "p50 10, p90 10 min (n=1)"
    assert rows["Reds by kind"]["day"] == "1 test, 0 Lean audit, 0 preflight, 0 infrastructure"
    assert rows["PR-rounds lost, by cause"]["day"] == "3: 2 red, 1 dropped base"
    assert rows["Executor (in shadow, not the gate): tasks finished a minute"]["hour"] == "0.1"             # 3 results in 60 min
    assert rows["Executor: cores working in a slot"]["day"] == "2.0 on average, p50 2.0 (n=1 runs)"
    assert rows["Executor: overhead before a run's first task result"]["day"] == "p50 10, p90 10 s (n=1)"
    cache = rows["Executor: share served from the cache; tasks waiting on other tasks"]
    assert cache["now"] == ("6 waiting on other tasks, 5 waiting for a slot, 1 given to a worker, in 2 checks; 1 of those checks, with 2 of "
                            "the tasks, are of no candidate being checked now")
    assert cache["hour"] == "50% from the cache (10 of 20 tasks in 1 checks)" and cache["day"] == "65% from the cache (40 of 62 tasks in 3 checks)"
    assert rows["Cores busy over cores paid, the check nodes"]["now"] == "25 of 100 (25%)"
    assert rows["Cores busy over cores paid, the check nodes"]["hour"] == "30 of 100 (30%), the average over the last 60 min"
    assert rows["Slots busy over slots, the check nodes"]["now"] == ("check slots: 1 of 2 busy, 0 free but blocked, 0 runs in line; "
                                                                    "lane slots: 4 of 4 busy, 0 free but blocked, 3 runs in line")
    assert rows["Reserved against used, a check's shard"]["now"] == ("1 of the 1 running, the ones whose cgroup was found: cores p50 12, most 12, of 96 in its slot; memory without the "
                                                                    "page cache p50 40 GiB, most 40 GiB; peak with the page cache most 120 GiB, "
                                                                    "of 300 GiB reserved")
    held = rows["Parallel suites running in a shard: a worker's memory against what it declares"]
    assert held["now"] == ("`tools/wide`: 3 workers in 1 shard, a worker holds p50 21.0, most 22.0 GiB; declares 8 GiB: **over, 2.8 "
                           "times**; `tools/plain`: 1 workers in 1 shard, a worker holds p50 3.0, most 3.0 GiB (7.5 GiB with the "
                           "processes it started); declares none, so the runner counts 9 GiB: within it")
    assert held["hour"] == held["day"] == ("`tools/wide`: most 22.0 GiB a worker, running at 1 of 1 readings; declares 8 GiB: **over, "
                                             "2.8 times**; `tools/plain`: most 3.0 GiB a worker, running at 1 of 1 readings; declares "
                                             "none, so the runner counts 9 GiB: within it")
    assert rows["Reserved against used, a quick tier"]["day"].startswith("cores: p50 2.0 used of 16 in its slot; memory: p50 8 GiB peak")
    assert rows["Ready to built"]["day"] == "p50 15, p90 19 min (n=2)"          # #1: verdict 00:40 to 01:00; #2: mark 00:50 to 01:00
    assert rows["The loop's tick: minutes between tick ends"]["hour"] == "p50 4.0, p90 4.0 min (n=2)"
    assert rows["Green candidate: verdict to `main`"]["day"] == "p50 5, p90 5 min (n=1)"
    assert all(set(r) == {"row", "name", "now", "hour", "day", "target", "how"} for r in rows.values())


def test_a_round_is_plain_only_when_its_candidate_is_known_to_change_no_lean():
    d = doc()
    d["lean"] = {"a" * 12: [qt.COMPARE_FILES, 0]}       # the compare's list was full, so a Lean file may be past its end
    row = {r["name"]: r for r in qt.table(d)}["Round with no Lean change: build to verdict"]
    assert row["day"] == "none of the 1 with a verdict (1 not known)"
    d["lean"] = {"a" * 12: [qt.COMPARE_FILES, 4]}
    assert {r["name"]: r for r in qt.table(d)}["Round with no Lean change: build to verdict"]["day"] == "none of the 1 with a verdict"


def test_where_a_landed_prs_time_went_adds_up_to_its_total():
    ps = {p["pr"]: p for p in qt.parts(doc())}
    one = ps["1"]
    assert (one["review"], one["to_build"], one["rework"], one["check"], one["to_main"]) == (30, 20, 0, 60, 5)
    assert sum(one[k] for k in ("review", "to_build", "rework", "check", "to_main")) == one["total"] == 115
    assert "the landing candidate's check" in qt.largest(doc())


def test_the_rendered_table_is_markdown_and_names_what_was_not_read():
    d = doc()
    d["errors"] = ["vy-nebius-cpu-2: no answer in 100 s"]
    text = qt.render(qt.table(d), d)
    assert "| # | Number | Now | Last hour | Last 24 h | Target | How it is measured |" in text
    assert "Not read: vy-nebius-cpu-2: no answer in 100 s" in text
    assert "not measured: needs a run of this command" in text      # no earlier reading of the CPU counters
    d["now"] += (qt.SHARD_LAYOUT_H + 1) * 3600                      # a layout no shard has used lately has no rows; b's still runs
    assert "| Shard 1/1: its slot" in qt.render(qt.table(d), d)
    for run in d["nodes"]["vy-nebius-cpu-1"]["runs"]:
        run["end"] = run.get("end") or qt.ts(at("02:30"))
    assert "| Shard 1/1: its slot" not in qt.render(qt.table(d), d)
    d = doc()
    d["errors"] = ["vy-nebius-cpu-2: no answer in 100 s"]
    short = qt.render(qt.table(d), d, short=True)
    assert "| # | Number | Now | Last hour | Last 24 h | Target |\n|---|---|---|---|---|---|\n" in short and "How it is measured" not in short
    d["control"]["executor"], d["control"]["executor_why"] = [], "its status printed nothing"
    assert "| not measured: its status printed nothing |" in qt.render(qt.table(d), d)


def test_a_workers_memory_is_held_to_its_suites_declaration_over_the_saved_readings():
    shards = [{"parallel": [{"suite": "a", "worker_gb": 8.0, "default_gb": 9.0, "own": [10.0, 30.0], "tree": [10.0, 30.0]}]},
              {"parallel": [{"suite": "a", "worker_gb": 8.0, "default_gb": 9.0, "own": [20.0], "tree": [20.0]},
                            {"suite": "a", "worker_gb": 32.0, "default_gb": 9.0, "own": [25.0], "tree": [25.0]}]},   # another candidate's
              {"parallel": []}, {}]
    now, other = qt.worker_memory(shards)
    assert now == {"suite": "a", "declared": 8.0, "by_default": False, "workers": 3, "shards": 2, "p50": 20.0, "most": 30.0, "tree": 30.0}
    assert (other["declared"], other["workers"]) == (32.0, 1) and qt.declares(other) == "declares 32 GiB: within it"
    assert qt.declares(dict(now, declared=None)) == "its declaration was not read"
    d = doc()
    for run in d["nodes"]["vy-nebius-cpu-1"]["runs"]:
        run.pop("parallel", None)
    name = "Parallel suites running in a shard: a worker's memory against what it declares"
    earlier = [{"now": d["now"] - 1800, "workers": [dict(now, most=12.0)]}, {"now": d["now"] - 7200, "workers": [now]},
               {"now": d["now"] - 900, "workers": []}, {"now": d["now"] - 600}]              # the last from before workers were saved
    row = {r["name"]: r for r in qt.table(d, None, earlier)}[name]
    assert row["now"] == "no shard is running"                                              # none whose processes were read
    assert row["hour"] == "`a`: most 12.0 GiB a worker, running at 1 of 3 readings; declares 8 GiB: **over, 1.5 times**"
    assert row["day"] == "`a`: most 30.0 GiB a worker, running at 2 of 4 readings; declares 8 GiB: **over, 3.8 times**"
    d["nodes"]["vy-nebius-cpu-1"]["runs"][1]["parallel"] = []
    row = {r["name"]: r for r in qt.table(d)}[name]
    assert (row["now"], row["hour"]) == ("no parallel suite is running in the 1 shard running",
                                         "no parallel suite was running in a shard at any of 1 readings")



def test_a_check_that_started_no_shard_is_timed_from_its_build_to_its_infra_verdict():
    d = doc()
    d["control"]["decisions"] += [
        {"action": "slim", "at": at("02:49"), "sha": "d" * 40, "why": "judged infra", "verdict": "infra", "ids": []},
        {"action": "slim", "at": at("02:55"), "sha": "d" * 40, "why": "judged infra", "verdict": "infra", "ids": ["r20261110-024500-ffff"]},
        {"action": "slim", "at": at("02:56"), "sha": "f" * 40, "why": "judged infra", "verdict": "infra", "ids": []}]   # no such build
    built = next(qt.ts(r["at"]) for r in decisions() if r["action"] == "build" and r["sha"].startswith("d"))
    row = {r["name"]: r for r in qt.table(d)}["Check that started no shard: build to its `infra` verdict"]
    minutes = (qt.ts(at("02:49")) - built) / 60
    assert row["hour"] == row["day"] == f"p50 {minutes:.0f}, p90 {minutes:.0f} min (n=1)"


def test_the_long_window_is_the_last_day_and_a_shard_still_running_is_never_outside_it():
    d = doc()
    assert qt.window_start(d) == T0 and qt.window_start(dict(d, day="2026-11-10T02:15:00Z")) == qt.ts(at("02:15"))
    d["day"] = "2026-11-10T02:15:00Z"                      # read later: only D's build (02:40) is inside the window
    rows = {r["name"]: r for r in qt.table(d)}
    assert rows["Candidates by outcome"]["day"] == "1 built: 0 green, 0 red, 0 dropped for another reason, 1 being checked"
    assert rows["Reserved against used, a check's shard"]["now"].startswith("1 of the 1 running")     # b's shard took its slot at 01:26


def test_a_change_of_the_lean_checks_own_code_is_a_lean_change():
    rule = qt.LEAN_FILE
    for f in ("security/Proofs/A.lean", "x/lakefile.toml", "lake-manifest.json", "a/lean-toolchain", "security/lean-audit.json",
              "tools/verity/lean/check.py", "tools/verity/check/common.py"):
        assert rule.search(f), f
    for f in ("tools/verity/check/suites.py", "verity/ml/circuits/lean.py", "docs/tools/verity/lean/x.md", "a/leanish.py",
              "tools/verity/lean/tests/test_audit.py", "tools/verity/lean/README.md"):
        assert not rule.search(f), f


def test_a_head_the_low_risk_rule_admits_needs_no_land_verdict():
    d = doc()
    d["open"]["7"], d["labels"]["7@h7"] = "h7", [lab("ready", True, "02:00")]
    row = {r["name"]: r for r in qt.table(dict(d, lowrisk={"5@h5": True, "7@h7": None}))}["Marked heads with no land verdict now"]
    assert row["now"] == ("2: 1 held by a reviewer, 1 not looked at, the oldest of those marked 60 min ago (1 of them with files GitHub "
                          "did not list); 1 more need none: low-risk, admitted on the mark alone")
    assert "low-risk" not in {r["name"]: r for r in qt.table(d)}["Marked heads with no land verdict now"]["now"]


def test_the_low_risk_heads_are_the_marked_ones_with_no_verdict_and_no_hold(monkeypatch, tmp_path):
    files = {"5": ["benchmarks/spot_figures/a.py"], "7": ["verity/core/x.py"], "8": None}
    asked = []
    monkeypatch.setattr(qt, "pr_files", lambda pr: asked.append(pr) or files[pr])
    monkeypatch.setattr(qt, "CHECKOUT", tmp_path)
    flags = tmp_path / "tools/verity/check/drive.toml"
    flags.parent.mkdir(parents=True)
    d = doc()
    for n in ("7", "8"):
        d["open"][n], d["labels"][f"{n}@h{n}"] = f"h{n}", [lab("ready", True, "02:00")]
    assert qt.lowrisk_heads(d["open"], d["labels"], d["now"]) == {}             # no drive.toml: no head is low-risk
    flags.write_text('lowrisk_lane = true\nlowrisk_excludes = ["verity/core/", "*.lean"]\n')
    assert qt.lowrisk_heads(d["open"], d["labels"], d["now"]) == {"5@h5": True, "7@h7": False, "8@h8": None}
    assert sorted(asked) == ["5", "7", "8"]         # not #3 (a land verdict), #4 (held) or #6 (its mark withdrawn)
    flags.write_text('lowrisk_lane = false\nlowrisk_excludes = ["verity/core/"]\n')
    assert qt.lowrisk_heads(d["open"], d["labels"], d["now"]) == {}


def test_a_restacks_last_commit_names_the_head_it_was_made_from():
    old, pins, onto = "a1" * 20, "b2" * 20, "c3" * 20
    assert qt.restack_of(f"Merge {onto[:12]} (the layout move) into {old}, restacked", [old, "d4" * 20]) == old
    assert qt.restack_of(f"Merge {onto[:12]} (the layout move) into {old[:12]}, restacked", [pins, "e5" * 20, old]) == old
    assert qt.restack_of(f"Merge {onto[:12]} (the layout move) into {old}, restacked", [pins]) is None      # not one of its parents
    assert qt.restack_of(f"Merge {onto[:12]} (the layout move) into cursor/a-branch, restacked", [old]) is None
    assert qt.restack_of(f"Merge {onto[:12]} into {old}", [old]) is None                                     # a merge of `main`
    assert qt.restack_of("restack: regenerate stale pins", [old]) is None


def test_restacks_follow_a_head_across_each_move_it_crossed():
    subject = "Merge cccccccccccc (the layout move) into {}, restacked"
    known = {"n3": (subject.format("abc1234"), ["abc1234" + "0" * 33]), "abc1234" + "0" * 33: (subject.format("def5678"), ["x", "def5678" + "1" * 33]),
             "def5678" + "1" * 33: ("a change", ["p"]), "h4": ("a change", ["q"])}
    asked = []

    def commits(shas):
        asked.append(sorted(shas))
        return {s: known[s] for s in shas if s in known}, [f"no commit for {s}" for s in shas if s not in known]

    made, unread = qt.restacks(["n3", "h4", "gone"], commits)
    assert made == {"n3": "abc1234" + "0" * 33, "abc1234" + "0" * 33: "def5678" + "1" * 33}
    assert unread == ["no commit for gone"]                                                          # said, not dropped
    assert asked == [["gone", "h4", "n3"], ["abc1234" + "0" * 33], ["def5678" + "1" * 33]]         # each head once
    assert qt.chain(made, "n3") == ["n3", "abc1234" + "0" * 33, "def5678" + "1" * 33] and qt.chain(made, "h4") == ["h4"]
    assert qt.chain({"a": "b", "b": "a"}, "a") == ["a", "b"]                                         # never round a loop


def test_the_commits_github_did_not_give_are_counted_with_what_went_wrong(monkeypatch):
    a, b, c = "a1" * 20, "b2" * 20, "c3" * 20
    subject = f"Merge cccccccccccc (the layout move) into {b}, restacked"
    answer = {"data": {"repository": {
        "c0": {"message": subject + "\n\nits body", "parents": {"nodes": [{"oid": b, "parents": {"nodes": [{"oid": c, "parents": {"nodes": []}}]}}]}},
        "c1": None}}}
    monkeypatch.setattr(qt.subprocess, "run", lambda argv, **kw: types.SimpleNamespace(stdout=json.dumps(answer)))
    got, unread = qt._commits([a, c, "a-branch"])
    assert got == {a: (subject, [b, c])}
    assert unread == ["GitHub: 1 heads whose name is no commit's, which count as not restacked (first 'a-branch')",
                      "GitHub: 1 heads it has no commit for, which count as not restacked (first c3c3c3c3c3c3)"]
    monkeypatch.setattr(qt.subprocess, "run", lambda argv, **kw: types.SimpleNamespace(stdout="rate limited"))
    assert qt._commits([a, c]) == ({}, ["GitHub: the commits of 2 heads, which count as not restacked "
                                        "(JSONDecodeError('Expecting value: line 1 column 1 (char 0)'))"])
    monkeypatch.setattr(qt.subprocess, "run", lambda argv, **kw: types.SimpleNamespace(stdout='{"data": null, "errors": []}'))
    made, unread = qt.restacks([a])                                                # no restack is read from an answer that failed
    assert made == {} and unread == ["GitHub: the commits of 1 heads, which count as not restacked "
                                     "(TypeError(\"'NoneType' object is not subscriptable\"))"]

    def down(argv, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'gh'")

    monkeypatch.setattr(qt.subprocess, "run", down)
    assert qt._commits([a])[1] == ["GitHub: the commits of 1 heads, which count as not restacked "
                                   "(FileNotFoundError(2, \"No such file or directory: 'gh'\"))"]


def restacked() -> dict:
    """The day with #1 landed at `n1` and #3 open at `n3`, each a restack of the head its owner marked; #1's owner marked `n1`
    again 45 minutes after the first mark."""
    d = doc()
    for r in d["control"]["decisions"]:
        if r["sha"].startswith("a") and r.get("heads"):
            r["heads"] = ["n1", "h2"]
    d["labels"]["1@n1"], d["open"]["3"] = [lab("ready", True, "00:55")], "n3"
    d["restacks"] = {"n1": "h1", "n3": "h3"}
    return d


def test_a_mark_and_a_land_verdict_carry_across_a_restack():
    d = restacked()
    rows = {r["name"]: r for r in qt.table(d)}
    assert rows["Mark to `main`"]["day"] == "p50 95, p90 111 min (n=2); 1 of the heads a restack's, 1 marked before it"     # still 115 and 75
    assert rows["Marked and waiting now (the backlog)"]["now"].startswith("3 PRs: 1 with a land verdict, 2 without")           # #3 at n3
    assert rows["Marked heads with no land verdict now"]["now"].startswith("2: 1 held by a reviewer, 1 not looked at")
    bare = dict(d, labels={k: v for k, v in d["labels"].items() if k != "3@h3"})    # #3 marked at h3 alone would need a verdict at n3
    bare["labels"]["3@h3"] = [lab("ready", True, "01:00")]
    assert {r["name"]: r for r in qt.table(bare)}["Marked heads with no land verdict now"]["now"].startswith("3: 1 held by a reviewer, 2 not looked at")
    one = {p["pr"]: p for p in qt.parts(d)}["1"]
    assert (one["total"], one["review"], one["to_build"]) == (115, 30, 20)          # the verdict at h1, the build of n1
    assert qt.carried(d["labels"], d["restacks"], "3", "n3") == d["labels"]["3@h3"] and qt.carried(d["labels"], {}, "3", "n3") == []
    plain = dict(d, restacks={})                                                    # read at the landed head alone: 70 and 75 min
    assert {r["name"]: r for r in qt.table(plain)}["Mark to `main`"]["day"] == "p50 72, p90 74 min (n=2)"
    d["labels"]["3@n3"] = [lab("ready", False, "02:00")]                            # a mark withdrawn at the new head is withdrawn
    assert {r["name"]: r for r in qt.table(d)}["Marked and waiting now (the backlog)"]["now"].startswith("2 PRs: 0 with a land verdict")
    d["labels"]["1@h1"] = d["labels"]["1@n1"] = []                                  # a restack's head nobody marked was not marked before it
    assert {r["name"]: r for r in qt.table(d)}["Mark to `main`"]["day"] == "p50 75, p90 75 min (n=1); 1 of the heads a restack's, 0 marked before it"


def left() -> list[dict]:
    """The loop's lines leaving heads out: #4 before the build that took it, #5 for `main`, #3 for a PR built before it, and a
    closed PR's head."""
    def line(action, hhmm, number, head, why):
        return {"action": action, "at": at(hhmm), "number": number, "heads": [head], "why": why, "shadow": False, "done": True}

    return [line("conflict", "02:10", "4", "h4", "conflicts with `main` in a.py"), line("conflict", "02:15", "5", "h5", "conflicts with `main` in a.py"),
            line("skip", "02:45", "3", "h3", "conflicts with #4, merged before it, in b.py (it merges onto `main`)"),
            line("skip", "02:50", "9", "h9", "conflicts with `main` in c.py")]


def test_the_heads_the_loop_leaves_out_are_a_row_of_their_own():
    d = doc()
    d["control"]["decisions"] = sorted(d["control"]["decisions"] + left(), key=lambda r: r["at"])
    out = qt.left_out(d["control"]["decisions"], d["open"])
    assert {pr: (r["action"], qt.with_main(r)) for pr, r in out.items()} == {"5": ("conflict", True), "3": ("skip", False)}   # #4 was built since
    rows = {r["name"]: r for r in qt.table(d)}
    row = rows["Marked heads the loop leaves out now"]
    assert row["now"] == ("1 refused until the owner merges `main` in: conflict with `main`, 0 of them with a land verdict, marked up to "
                          "1.5 h ago; 1 more out of the newest candidate: conflict with a PR built before them")
    assert row["hour"] == "3 heads for a conflict with `main`, 1 for one with a PR built before them"
    assert rows["Marked heads with no land verdict now"]["now"] == "1: 1 held by a reviewer, 0 not looked at"       # #5 waits on its owner
    assert rows["Mark to `main`"]["day"] == "p50 95, p90 111 min (n=2)"


def test_ready_to_built_is_split_by_the_cause_the_loops_lines_give_for_each_minute():
    """Each landed PR's minutes from ready to its first candidate go to the cause of the loop's newest line leaving its head
    out since its mark (`leave`, or a `conflict` or `skip` of a build), and to none logged before its first: #1 is ready at
    its verdict 00:40, out for a full chain from 00:45, built 01:00; #2 is ready at its mark 00:50, its batch full from 00:55,
    built 01:00.  A line from before the mark, or of the shadow, says nothing."""
    def leave(hhmm: str, number: int, head: str, cause: str, shadow: bool = False) -> dict:
        return {"action": "leave", "at": at(hhmm), "number": number, "heads": [head], "cause": cause, "why": cause, "shadow": shadow}

    d = doc()
    d["control"]["decisions"] = sorted(d["control"]["decisions"] + [
        leave("00:05", 1, "h1", "no land verdict"), leave("00:45", 1, "h1", "chain full"),
        leave("00:52", 2, "h2", "held", shadow=True), leave("00:55", 2, "h2", "batch full")], key=lambda r: r["at"])
    row = {r["name"]: r for r in qt.table(d)}["Ready to built, by the loop's reason"]
    assert row["hour"] == row["day"] == "30 PR-min over 2 PRs: chain full 15 (50%), none logged 10 (33%), batch full 5 (17%)"
    assert {r["name"]: r for r in qt.table(doc())}["Ready to built, by the loop's reason"]["day"] == (
        "30 PR-min over 2 PRs: none logged 30 (100%)")
    script = qt.CONTROL_SCRIPT[:qt.CONTROL_SCRIPT.index("ticks = ")]
    assert '"leave"' in script and '"cause"' in script, "the control pod's reading keeps the loop's leave lines and their cause"


def test_a_window_that_ended_is_cut_to_what_was_known_then():
    d = qt.cut(dict(doc(), read=qt.ts(at("03:00"))), qt.ts(at("02:00")))
    assert d["now"] == d["until"] == qt.ts(at("02:00"))
    assert [r["action"] for r in d["control"]["decisions"]][-1] == "slim" and d["control"]["ticks"] == []
    assert d["open"] == {"1": "h1", "2": "h2", "3": "h3", "4": "h4", "5": "h5", "6": "h6"}            # #1 and #2 landed at 02:05
    runs = {r["id"][-4:]: r for r in d["nodes"]["vy-nebius-cpu-1"]["runs"]}
    assert sorted(runs) == ["aaaa", "bbbb", "cccc"]                                                     # the others were asked for later
    assert (runs["cccc"]["took"], runs["cccc"]["slot_state"], runs["cccc"]["end"]) == (None, "waiting", None)   # its slot came at 02:20
    assert (runs["aaaa"]["end"], runs["aaaa"]["state"]) == (None, "running")                            # it ended at 02:06
    assert runs["aaaa"]["steps"] == [{"step": "pytest", "state": "passed", "seconds": 3000, "end": qt.ts(at("01:56"))},
                                     {"step": "circuit-check", "state": "running", "start": None}]
    assert runs["bbbb"]["failed"] == [["tools/research", qt.ts(at("02:00"))]] and "parallel" not in runs["bbbb"]
    rows = {r["name"]: r for r in qt.table(d)}
    assert rows["Mark to `main`"]["day"] == "none" and rows["Candidates by outcome"]["day"].startswith("3 built: 0 green, 0 red, 0 dropped")
    assert rows["Marked and waiting now (the backlog)"]["now"].startswith("5 PRs: 2 with a land verdict, 3 without; 5 in a candidate being checked")
    assert "a floor" in rows["Marked and waiting now (the backlog)"]["now"]
    for name in ("Cores busy over cores paid, the check nodes", "Slots busy over slots, the check nodes", "Reserved against used, a check's shard",
                 "Parallel suites running in a shard: a worker's memory against what it declares",
                 "Executor: share served from the cache; tasks waiting on other tasks"):
        assert rows[name]["now"] == qt.PAST, name
    assert rows["Cores busy over cores paid, the check nodes"]["hour"] == qt.PAST
    text = qt.render(list(rows.values()), d, short=True)
    assert text.startswith("The merge queue's numbers for the windows that ended at 2026-11-10 02:00Z, read at 2026-11-10 03:00Z.")
    assert "| # | Number | At its end | Its last hour | Its 24 h | Target |" in text and "in the 24 hours to 2026-11-10 02:00Z" in text
    later = qt.cut(dict(doc(), read=qt.ts(at("03:00"))), qt.ts(at("02:35")))                          # after B's drop, before D's build
    assert {r["name"]: r for r in qt.table(later)}["Candidates by outcome"]["day"] == "3 built: 1 green, 1 red, 1 dropped for another reason, 0 being checked"
    assert [r["id"][-4:] for r in later["nodes"]["vy-nebius-cpu-1"]["runs"]] == ["aaaa", "bbbb", "cccc", "eeee"]
    assert later["nodes"]["vy-nebius-cpu-1"]["runs"][2]["end"] is None                                  # the quick tier ended at 02:50


def test_a_windows_end_is_a_past_time_with_its_offset(capsys):
    now = qt.ts(at("03:00"))
    assert qt.window_end("2026-11-10T02:00:00Z", now) == qt.window_end("2026-11-09T18:00:00-08:00", now) == qt.ts(at("02:00"))
    for until, why in (("2026-11-10T02:00:00", "it has no UTC offset"), ("2026-11-10T03:00:00Z", "it is not in the past"),
                       ("soon", "Invalid isoformat string: 'soon'")):
        with pytest.raises(ValueError, match=why):
            qt.window_end(until, now)
    # the command refuses each before it reads anything, saying which it is and the form to write
    assert qt.main(until="2026-11-10T02:00:00") == 2 and qt.main(until="9999-01-01T00:00:00Z") == 2 and qt.main(until="soon") == 2
    said = capsys.readouterr().err.splitlines()
    assert [line.split(". Write a past time")[0] for line in said] == [
        "research queue stats: --until 2026-11-10T02:00:00: it has no UTC offset",
        "research queue stats: --until 9999-01-01T00:00:00Z: it is not in the past",
        "research queue stats: --until soon: Invalid isoformat string: 'soon'"]
    assert all(line.endswith("like 2026-01-31T08:00:00Z or 2026-01-31T00:00:00-08:00") for line in said)


def test_an_audit_logs_lines_say_what_it_reused():
    log = {"staged": "security/Proofs: 857 of 4982 keyed modules restored from the artifact cache; missed: 3553 reads a module not restored",
           "audited": "security, security/Proofs, security/Properties audited in 3957 s (exit 0)", "verdict": "lean-audit: PASS: 0 of 3 packages reused"}
    assert qt.reuse(log) == {"staged": [857, 4982], "reused": [0, 3], "passed": True, "audit_s": 3957, "unread": None}
    failed = qt.reuse({"verdict": "lean-audit: FAIL (security, security/Properties): 1 of 3 packages reused"})
    assert failed == {"staged": None, "reused": [1, 3], "passed": False, "audit_s": None, "unread": None}
    assert qt.reuse({}) == {"staged": None, "reused": None, "passed": None, "audit_s": None, "unread": None}


def node_function(name: str, tmp_path, **given):
    """One function of the script a node runs, as a module of its own under ``tmp_path``, with ``given`` in place of the
    globals it would find."""
    fn = next(n for n in ast.parse(qt.NODE_SCRIPT).body if isinstance(n, ast.FunctionDef) and n.name == name)
    path = tmp_path / f"node_{name}.py"
    path.write_text("import os\n\n\n" + ast.unparse(fn) + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in given.items():
        setattr(module, key, value)
    return getattr(module, name)


def test_a_node_says_why_an_audit_log_that_is_there_was_not_read(tmp_path):
    log = tmp_path / "lean-audit.log"
    assert node_function("audit", tmp_path)(str(log)) == {}                 # a shard that ran no Lean audit has no log
    log.write_text("stage: /scratch/x/security: 3 of 3 keyed modules restored from the artifact cache\n"
                   "stage: /scratch/x/security/Proofs: 10 of 40 keyed modules restored from the artifact cache; missed: 30 its source\n"
                   "lean-audit: security audited in 12 s (exit 0)\nlean-audit: PASS: 2 of 3 packages reused\n")
    read = node_function("audit", tmp_path)(str(log))
    assert read == {"staged": "security/Proofs: 10 of 40 keyed modules restored from the artifact cache; missed: 30 its source",
                    "audited": "lean-audit: security audited in 12 s (exit 0)", "verdict": "lean-audit: PASS: 2 of 3 packages reused"}
    assert qt.reuse(read) == {"staged": [10, 40], "reused": [2, 3], "passed": True, "audit_s": 12, "unread": None}

    def broken(path, **kw):
        raise OSError(5, "Input/output error")

    unread = node_function("audit", tmp_path, open=broken)(str(log))
    assert unread == {"unread": "OSError(5, 'Input/output error')"}
    d = rounds_doc()
    d["nodes"]["vy-nebius-cpu-1"]["runs"][0]["audit"] = unread
    row = qt.rounds(d)[0]
    assert (row["staged"], row["reused"], row["unread"]) == (None, None, "OSError(5, 'Input/output error')")
    cell = "log not read: OSError(5, 'Input/output error')"
    assert f"| 3,000 | 2,000 | {cell} | {cell} | not run / not run |" in qt.render_rounds(qt.rounds(d), d)


def rounds_doc() -> dict:
    """The day with each candidate's base, A's Lean audit (ended 01:39), B judged red early, and a shard of D's the loop never named."""
    d = doc()
    bases = {"a": "", "b": "a" * 40, "c": "b" * 40, "d": "a" * 40}
    for r in d["control"]["decisions"]:
        if r["action"] == "build":
            r["base"] = bases[r["sha"][0]]
    d["control"]["decisions"].insert(8, {"action": "slim", "at": at("02:25"), "sha": "b" * 40, "why": "judged fail", "verdict": None,
                                          "ids": ["r20261110-012500-bbbb"], "shadow": False})
    a = d["nodes"]["vy-nebius-cpu-1"]["runs"][0]
    a["steps"].append({"step": "lean-audit", "state": "passed", "seconds": 2000, "end": a["took"] + 2000})
    a["audit"] = {"staged": "security/Proofs: 10 of 40 keyed modules restored from the artifact cache", "verdict": "lean-audit: PASS: 2 of 3 packages reused",
                  "audited": "security audited in 1500 s (exit 0)"}
    d["nodes"]["vy-nebius-cpu-1"]["runs"].append(
        {"id": "r20261110-024200-dddd", "kind": "shard", "sha": "d" * 12, "shard": "2/3", "took": qt.ts(at("02:43")), "end": None, "failed": [],
         "steps": [{"step": "lean-changed", "state": "passed", "seconds": 3, "end": qt.ts(at("02:44"))},
                   {"step": "lean-suites", "state": "running", "start": qt.ts(at("02:44"))}]})
    return d


def test_a_round_says_what_it_changed_what_it_was_checked_on_and_how_it_went():
    d = rounds_doc()
    a, b, c, e = qt.rounds(d)
    assert (a["lean"], a["below"], a["first"]) == (0, [], {"node": "vy-nebius-cpu-1", "id": "r20261110-010500-aaaa"})
    assert (a["outcome"], a["verdict_min"], a["landed_min"], a["staged"], a["reused"], a["audit_s"]) == ("green", 60, 65, [10, 40], [2, 3], 1500)
    assert a["steps"]["pytest"] == {"state": "passed", "seconds": 3000} and a["steps"]["lean-audit"] == {"state": "passed", "seconds": 2000}
    assert a["pole"] == {"shard": "1/1", "step": "pytest", "state": "passed", "min": 56}           # circuit-check is off the gate
    # B's first shard was asked for at 01:25, before A's audit ended: B audits what A changed, and C, built at 01:40, only B's
    assert (b["lean"], b["below"], b["below_lean"]) == (2, ["a" * 12], 0) and (c["lean"], c["below"], c["below_lean"]) == (None, ["b" * 12], 2)
    assert (b["outcome"], b["why"], b["verdict_min"], b["dropped_min"]) == ("red", "test", 65, 70)   # judged in the loop's reason alone
    assert c["first"] is None and c["steps"]["pytest"] is None and c["pole"] is None
    assert (e["below"], e["first"], e["shards"]) == ([], None, 1)                                    # `main` had moved to A; its shard by its commit
    assert e["steps"]["lean-changed"] == {"state": "passed", "seconds": 3} and e["steps"]["lean-suites"] == {"state": "running", "seconds": 960}
    text = qt.render_rounds(qt.rounds(d), d)
    assert text.startswith("The candidates built in the last 24 hours, oldest first: 4.")
    assert ("| `aaaaaaaaaaaa` 11-10 01:00Z | 2 | 0 | 0 | cpu-1 `r20261110-010500-aaaa` | **green**: verdict at 60.0, on `main` at 65.0 | 3,000 | "
            "2,000 (1,500) | 10 of 40 | 2 of 3 | not run / not run | 1/1 `pytest` at 56.0 |") in text
    assert ("| `cccccccccccc` 11-10 01:40Z | 1 | not known | 2 (`bbbb`) | none of its 0 ran `pytest` | **dropped** (dropped base): no verdict, "
            "dropped at 51.0 | not run | not run | no line | no line | not run / not run | none ended |") in text
    assert "being checked: no verdict, built 20 min ago | not run | not run | no line | no line | 3 / 960 (running) | 2/3 `lean-changed` at 4.0 so far |" in text
