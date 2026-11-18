import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[3] / "benchmarks" / "dsa_topk")))
import units


def stub(clean_at: tuple | None = None, idle_at: tuple | None = None, wrong_at: tuple | None = None):
    """A timer: every forced row falls back and no base row does, but at ``clean_at`` a base row fails, at
    ``idle_at`` a forced line reaches no fallback and at ``wrong_at`` the passes' output is wrong."""
    def timer(values, rows, keys, how):
        fell = how is not None and idle_at != (rows, keys) or how is None and clean_at == (rows, keys)
        counts = {"rows_long": rows, "rows_tied": 0, "rows_bisected": rows if fell else 0, "rows_refined": 0}
        ok, us = wrong_at != (rows, keys), rows * keys / 1e4
        if how is None:
            return {"fast_us": us, **{p: {"ok": ok, "us": 2 * us + i, "added_us": us + i, "per_call": counts}
                                      for i, p in enumerate(("present", "cut"))}}
        return {"copy_us": 1.0, **{p: {"ok": ok, "one_call": counts, "us": 3 * us + i}
                                   for i, p in enumerate(("present", "cut"))}}
    return timer


def run(tmp_path, shapes, timed):
    ok = units.write(tmp_path / "out", tmp_path / "out" / ".." / "result.json", shapes, timed, "stub")
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["run_id"] == tmp_path.name
    return ok, doc


@pytest.mark.parametrize("shapes", ["decode", "prefill"])
def test_units_are_read_by_priced_at_every_shape(tmp_path, shapes):
    ok, doc = run(tmp_path, shapes, units.lines(shapes, stub()))
    assert ok and doc["kind"] == units.KIND and doc["validation"]["status"] == "passed"
    assert doc["workload_fingerprint"]["shapes"] == shapes and doc["measurement_files"] == ["out/units.jsonl"]
    rows, keys, _ = units.SHAPES[shapes]
    lines = [json.loads(s) for s in (tmp_path / "out" / "units.jsonl").read_text().splitlines()]
    assert {(r["rows"], r["keys"]) for r in lines if "forced" in r} == {(r, n) for r in rows for n in keys}
    assert all("dist" not in r for r in lines if "forced" in r)


def test_a_shape_with_no_clean_time_fails(tmp_path):
    rows, keys, _ = units.SHAPES["decode"]
    ok, doc = run(tmp_path, "decode", units.lines("decode", stub(clean_at=(rows[0], keys[-1]))))
    assert not ok and doc["validation"]["status"] == "failed"
    assert f"({rows[0]}, {keys[-1]})" in doc["validation"]["detail"]


def test_a_forced_line_that_reached_no_fallback_fails(tmp_path):
    ok, doc = run(tmp_path, "prefill", units.lines("prefill", stub(idle_at=(2048, 16384))))
    assert not ok and "no row reached the fallback" in doc["validation"]["detail"]


def test_a_pass_whose_output_is_wrong_fails(tmp_path):
    ok, doc = run(tmp_path, "decode", units.lines("decode", stub(wrong_at=(12, 40000))))
    assert not ok and "at 12 x 40000: not reference_rows'" in doc["validation"]["detail"]


def test_counters_that_count_no_long_row_fail(tmp_path):
    def timer(v, r, n, how):
        t = stub()(v, r, n, how)
        for p in ("present", "cut"):
            t[p]["per_call" if how is None else "one_call"] = {"rows_long": 0, "rows_bisected": int(how is not None)}
        return t
    ok, doc = run(tmp_path, "decode", units.lines("decode", timer))
    assert not ok and "its counters count no row longer than k" in doc["validation"]["detail"]


def test_priced_refuses_lines_with_no_failed_time(tmp_path):
    ok, doc = run(tmp_path, "decode", [r for r in units.lines("decode", stub()) if "forced" not in r])
    assert not ok and "times with and without a failed row" in doc["validation"]["detail"]
