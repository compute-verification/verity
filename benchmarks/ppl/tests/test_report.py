import json
import math
import re

import pytest

import report
import run


def results(arms: dict, point=("llama-3.1-8b", "c4", 2048), passed=True, run_id="r20261016-000000-abcd") -> dict:
    """A results.json: per arm, its plain arm and per seed its windows' mean NLLs."""
    out = {}
    for name, (plain, seeds) in arms.items():
        out[name] = {"plain": plain, "seeds": {str(s): {"ppl": math.exp(sum(w) / len(w)), "window_nll": w}
                                               for s, w in seeds.items()}}
    model, set_, ctx = point
    return {"model": model, "set": set_, "ctx": ctx, "n_windows": 16, "scored_tokens": 16 * (ctx - 1), "passed": passed,
            "gates": {"int8_range": {"checked": 5, "failed": 0, "first_failure": None}}, "env": {"run": run_id},
            "arms": out}


def standard(shift: float = 0.0) -> dict:
    return {"bf16": ("", {0: [2.0 + shift] * 16}),
            "int8-linears": ("", {0: [2.01 + shift] * 16}),
            "adopted-linears": ("int8-linears", {s: [2.004 + 0.001 * s + shift] * 16 for s in range(3)})}


def test_the_table_reports_both_deltas_and_the_verdict():
    t = report.table(results(standard()))
    assert "### llama-3.1-8b · c4 · 2048 (16 windows, 32,752 tokens)" in t
    rows = {line.split(" | ")[0].strip("| "): line for line in t.splitlines() if line.startswith("| ")}
    assert rows["int8-linears"].endswith(f"| {100 * (math.exp(0.01) - 1):+.2f} | - | - |")
    a = rows["adopted-linears"]
    assert "| 3 |" in a and f"({math.exp(2.004):.4f}–{math.exp(2.006):.4f})" in a
    assert a.endswith(f"| {100 * (math.exp(0.005) - 1):+.2f} | {100 * (math.exp(-0.005) - 1):+.2f} | < |")


def test_pct():
    assert report.pct(1.01, 1.0) == "+1.00" and report.pct(0.99, 1.0) == "-1.00"
    assert report.pct(5.9792, 5.9793) == "+0.00" and report.pct(9.169, 9.1753, 3) == "-0.069"


def test_a_failed_run_has_no_table():
    r = results(standard(), passed=False)
    r["gates"]["peel"] = {"checked": 3, "failed": 1, "first_failure": {"where": "layer 0"}}
    t = report.table(r)
    assert "GATES FAILED" in t and "layer 0" in t and "| bf16 |" not in t


def test_the_readme_shows_the_locked_baseline():
    lk = json.loads(run.BASELINE.read_text())
    text = run.README.read_text()
    assert text.split(run.MARK[0])[1].split(run.MARK[1])[0].strip() == report.baseline_table(lk), \
        "the README's baseline is not baseline.json's: `run.py baseline --lock` rewrites both"
    assert run.readme_with(report.baseline_table(lk), text) == text


def test_the_lock_is_well_formed():
    lk = json.loads(run.BASELINE.read_text())
    keys = {report.point_key(*p) for p in report.BASELINE_POINTS}
    assert set(lk["tonight"]) <= keys
    for t in lk["tonight"].values():
        for arm, e in t.items():
            assert arm in report.BASELINE_ARMS and e["min"] <= e["ppl"] <= e["max"] and e["n"] >= 1 and e["as"]
            assert re.fullmatch(r"r\d{8}-\d{6}-[0-9a-f]{4}", e["run"])
    assert [(p["model"], p["set"], p["ctx"]) for p in lk["points"]] == \
        [k for k in report.BASELINE_POINTS if any((p["model"], p["set"], p["ctx"]) == k for p in lk["points"])]
    for p in lk["points"]:
        assert "bf16" in p["arms"] and "adopted" in p["arms"], p
        assert all(not s["plain"] or s["plain"] in p["arms"] for s in p["arms"].values())


def test_lock_replaces_a_point_in_order_and_refuses_failed_runs():
    lk = {"tonight": {}, "points": []}
    later = results(standard(), point=("qwen2.5-7b", "wikitext2", 2048))
    first = results(standard())
    lk = report.lock(report.lock(lk, [later]), [first])
    assert [p["model"] for p in lk["points"]] == ["llama-3.1-8b", "qwen2.5-7b"]
    p = lk["points"][0]
    assert p["run"] == "r20261016-000000-abcd" and set(p["arms"]) == {"bf16", "int8-linears", "adopted-linears"}
    assert p["arms"]["adopted-linears"]["n"] == 3 and p["arms"]["adopted-linears"]["plain"] == "int8-linears"
    with pytest.raises(ValueError, match="gates failed"):
        report.lock(lk, [results(standard(), passed=False)])
    with pytest.raises(ValueError, match="not a baseline point"):
        report.lock(lk, [results(standard(), point=("llama-3.1-8b", "c4", 8192))])


def test_baseline_check():
    lk = report.lock({"tonight": {}, "points": []}, [results(standard())])
    text, ok = report.baseline_check(lk, [results(standard())])
    assert ok and text.count("| yes |") == 3
    _, ok = report.baseline_check(lk, [results(standard(shift=math.log(1.0008)))])
    assert ok
    text, ok = report.baseline_check(lk, [results(standard(shift=math.log(1.003)))])
    assert not ok and "| NO |" in text
    _, ok = report.baseline_check(lk, [results(standard(), passed=False)])
    assert not ok
    _, ok = report.baseline_check(lk, [results(standard(), point=("qwen2.5-7b", "wikitext2", 2048))])
    assert not ok


def test_the_baseline_table_compares_with_tonight():
    lk = report.lock({"tonight": {report.point_key("llama-3.1-8b", "c4", 2048): {
        "bf16": {"ppl": math.exp(2.0), "min": math.exp(2.0), "max": math.exp(2.0), "n": 1, "as": "bf16", "run": "r1"},
        "int8-linears": {"ppl": 7.5, "min": 7.5, "max": 7.5, "n": 1, "as": "r127", "run": "r1", "note": "eager"}}},
        "points": []}, [results(standard())])
    t = report.baseline_table(lk)
    rows = {line.split(" | ")[1]: line for line in t.splitlines() if line.startswith("| ") and " | " in line}
    assert rows["bf16"].endswith("| `bf16` | +0.00 | yes |")
    assert "`r127` (eager)" in rows["int8-linears"] and rows["int8-linears"].endswith("| NO |")
    assert rows["adopted-linears"].endswith("| - | - | - | - |")
    assert "bench `r20261016-000000-abcd`; tonight `r1`" in t
