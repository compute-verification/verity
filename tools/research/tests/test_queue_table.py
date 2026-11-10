"""`research queue stats` with no `--log`: the queue's table (`queue_table`), its pure parts on a small made-up day."""
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
    assert qt.rid_time("r20261110-201216-fd89") == qt.ts(at("20:12")) + 16


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
    assert qt.drop_cause("#8328's ready mark was withdrawn") == "withdrawn mark"
    assert qt.drop_cause("failed: step lean-audit, step lean-suites") == "red"
    assert qt.drop_cause("#8633 isn't taken now: ejected at 4ddfaf5c: candidate 3ddd53312267 fails pytest") == "red"
    # a hold whose text speaks of a withdrawn copy is still a hold
    assert qt.drop_cause("#8488 isn't taken now: held by top (red in candidate 47f2: a withdrawn copy reads WITHDRAWN)") == "hold"
    assert qt.drop_cause("ci dropped it by hand: its base 47f2c95d4548 is red") == "by hand"
    assert qt.drop_cause("9328fef5a055's check ended in infra errors: the run beside its check registered no run") == "infrastructure"
    assert qt.drop_cause("something new") == "other"


def test_a_reds_kind_is_read_from_its_failed_tasks():
    assert qt.red_kind("failed: step pytest", ["step pytest"]) == "test"
    assert qt.red_kind("failed: step lean-audit, step lean-suites", []) == "Lean audit"
    assert qt.red_kind("failed: step preflight-fixtures", []) == "preflight"
    assert qt.red_kind("x's check ended in infra errors", []) == "infrastructure"


def test_the_check_nodes_are_the_loops_on_flag():
    assert qt.check_nodes("--batch 16\n# a comment\n--on vy-nebius-cpu-13,vy-nebius-cpu-7\n--gate check\n") == ["vy-nebius-cpu-13", "vy-nebius-cpu-7"]
    assert qt.check_nodes("") == []


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
                                  "failed": [], "steps": [{"step": "pytest", "state": "passed", "seconds": 3000, "end": took + 3000},
                                                          {"step": "circuit-check", "state": "passed", "seconds": 3500, "end": took + 3600}]},
        "r20261110-012500-bbbb": {"id": "r20261110-012500-bbbb", "kind": "shard", "shard": "1/1", "took": qt.ts(at("01:26")), "end": None,
                                  "rc": None, "failed": [["tools/research", qt.ts(at("02:00"))]],
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


def doc() -> dict:
    labels = {
        "1@h1": [lab("ready", True, "00:10"), lab("reviewed", "land", "00:40")],      # 115 min from mark to main
        "2@h2": [lab("ready", True, "00:50")],                                        # low-risk: no verdict; 75 min
        "3@h3": [lab("ready", True, "01:00"), lab("reviewed", "land", "01:05")],
        "4@h4": [lab("ready", True, "01:10"), lab("reviewed", "hold", "01:15")],
        "5@h5": [lab("ready", True, "01:30")],
        "6@h6": [lab("ready", False, "01:00")],
    }
    node = {"cores": 100, "busy": 0.25, "stat": [5000, 20000], "hz": 100, "load": ["1", "1", "1"],
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
            "control": {"decisions": decisions(), "ticks": [at("02:50"), at("02:54"), at("02:58")], "args": ""},
            "nodes": {"vy-nebius-cpu-1": node}, "open": {"3": "h3", "4": "h4", "5": "h5", "6": "h6"}, "labels": labels}


def test_the_table_holds_each_number_over_its_windows():
    d = doc()
    rows = {r["name"]: r for r in qt.table(d, {"now": d["now"] - 3600, "nodes": {"vy-nebius-cpu-1": {"stat": [5000 - 3600 * 100 * 30, 0]}}})}
    m2m = rows["Mark to `main`"]
    assert m2m["hour"] == m2m["today"] == "p50 95, p90 111 min (n=2)"            # 115 and 75 min, landed 55 min ago
    assert rows["Share on `main` within 2 h of the mark"]["today"] == "2 of 2 landed (100%)"
    assert rows["Marked and waiting now (the backlog)"]["now"].startswith("3 PRs: 1 with a land verdict, 2 without; 1 in a candidate being checked")
    assert rows["Mark to land verdict (the review wait)"]["today"] == "p50 18, p90 28 min (n=2)"     # 30 and 5 min
    assert rows["Marked heads with no land verdict now"]["now"] == "2: 1 held by a reviewer, 1 not looked at, the oldest of those marked 90 min ago"
    assert rows["Quick tier: waiting for a slot"]["now"] == "1 in line, longest 15 min so far"
    assert rows["Quick tier: waiting for a slot"]["today"] == "p50 20, p90 20 min (n=1)"
    assert rows["Quick tier: running"]["hour"] == "p50 30, p90 30 min (n=1)"
    assert rows["Candidate: build to verdict"]["today"] == "p50 60, p90 60 min (n=1)"
    assert rows["Candidates by outcome"]["today"] == "4 built: 1 green, 1 red, 1 dropped for another reason, 1 being checked"
    shard = rows["Shard 1/1: its slot to the end of its last step on the gate"]["today"]
    assert shard.startswith("p50 50, p90 50 min (n=1); slowest step `pytest` in 1 of 1, p50 50 min; then it holds the slot p50 10, p90 10 min more")
    assert rows["Red: first failure to the candidate's drop"]["today"] == "p50 30, p90 30 min (n=1)"
    assert rows["Red: drop to the next candidate built"]["today"] == "p50 10, p90 10 min (n=1)"
    assert rows["Reds by kind"]["today"] == "1 test, 0 Lean audit, 0 preflight, 0 infrastructure"
    assert rows["PR-rounds lost, by cause"]["today"] == "3: 2 red, 1 dropped base"
    assert rows["Executor: tasks finished a minute"]["hour"] == "0.1"             # 3 results in 60 min
    assert rows["Executor: cores working in a slot"]["today"] == "2.0 on average, p50 2.0 (n=1 runs)"
    assert rows["Executor: overhead before a run's first task result"]["today"] == "p50 10, p90 10 s (n=1)"
    assert rows["Cores busy over cores paid, the check nodes"]["now"] == "25 of 100 (25%)"
    assert rows["Cores busy over cores paid, the check nodes"]["hour"] == "30 of 100 (30%), the average over the last 60 min"
    assert rows["Slots busy over slots, the check nodes"]["now"] == ("check slots: 1 of 2 busy, 0 free but blocked, 0 runs in line; "
                                                                    "lane slots: 4 of 4 busy, 0 free but blocked, 3 runs in line")
    assert rows["Reserved against used, a quick tier"]["today"].startswith("cores: p50 2.0 used of 16 in its slot; memory: p50 8 GB peak")
    assert rows["Ready to built"]["today"] == "p50 15, p90 19 min (n=2)"          # #1: verdict 00:40 to 01:00; #2: mark 00:50 to 01:00
    assert rows["The loop's tick: minutes between tick ends"]["hour"] == "p50 4.0, p90 4.0 min (n=2)"
    assert rows["Green candidate: verdict to `main`"]["today"] == "p50 5, p90 5 min (n=1)"
    assert all(set(r) == {"row", "name", "now", "hour", "today", "target", "how"} for r in rows.values())


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
    assert "| # | Number | Now | Last hour | Today | Target | How it is measured |" in text
    assert "Not read: vy-nebius-cpu-2: no answer in 100 s" in text
    assert "not measured: needs a run of this command" in text      # no earlier reading of the CPU counters
