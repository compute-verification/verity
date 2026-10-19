import json
import math

import pytest

import dither_plot
from test_report import results


def test_arms_are_read_by_side_and_size():
    assert dither_plot.classify("adopted-bd-lm") == ("both", "both", 1)
    assert dither_plot.classify("adopted-bd-lm-d3") == ("both", "both", 3)
    assert dither_plot.classify("side-lin-act") == ("act", "both", 1)
    assert dither_plot.classify("side-lin-w-d2") == ("w", "both", 2)
    assert dither_plot.classify("side-qk-k-d4") == ("both", "k", 4)
    assert dither_plot.classify("adopted") == ("record", "record", 1)
    assert dither_plot.classify("side-act-k-d2") == ("act", "k", 2)
    assert dither_plot.classify("co32c-d2") is None and dither_plot.classify("int8-all") is None


def test_rows_give_each_arms_increase_over_its_runs_bf16(tmp_path):
    seeds = lambda base: {s: [base + 0.001 * s] * 16 for s in range(3)}  # noqa: E731
    r1 = results({"bf16": ("", {0: [2.0] * 16}), "side-lin-act": ("int8-all", seeds(2.01)),
                  "adopted-bd-lm": ("int8-all", seeds(2.04)), "int8-all": ("", seeds(2.02))},
                 run_id="r20261017-000000-aaaa")
    r2 = results({"bf16": ("", {0: [2.0] * 16}), "side-lin-act": ("int8-all", seeds(2.03)),
                  "adopted-bd-lm-d2": ("int8-all", seeds(2.005))}, run_id="r20261017-000000-bbbb")
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    p1.write_text(json.dumps(r1))
    p2.write_text(json.dumps(r2))
    rows = {r["arm"]: r for r in dither_plot.rows([str(p1), str(p2)])}
    assert set(rows) == {"side-lin-act", "adopted-bd-lm", "adopted-bd-lm-d2"}
    act = rows["side-lin-act"]
    diff, verdict = dither_plot.against_both(list(rows.values()), act)
    assert diff == pytest.approx(act["inc"] - rows["adopted-bd-lm"]["inc"]) and diff < 0 and verdict == "<"
    assert dither_plot.against_both(list(rows.values()), rows["adopted-bd-lm-d2"]) is None
    assert act["run"] == "r20261017-000000-bbbb"
    assert act["inc_min"] == pytest.approx(100 * (math.exp(0.03) - 1))
    assert act["inc_max"] == pytest.approx(100 * (math.exp(0.032) - 1))
    assert (rows["adopted-bd-lm-d2"]["b"], rows["adopted-bd-lm-d2"]["linears"]) == (2, "both")
    pt = ("llama-3.1-8b", "c4", 2048)
    assert [r["arm"] for r in dither_plot.series(list(rows.values()), pt, "linears", "both")] == [
        "adopted-bd-lm", "adopted-bd-lm-d2"]
    assert [r["arm"] for r in dither_plot.series(list(rows.values()), pt, "linears", "act")] == ["side-lin-act"]
    assert dither_plot.series(list(rows.values()), pt, "qk", "q") == []
    t = dither_plot.table(list(rows.values()))
    assert "| llama-3.1-8b · c4 · 2K | side-lin-act | act | both | 1 | 3 |" in t
    assert f"| {diff:+.2f} < | r20261017-000000-bbbb |" in t
    pytest.importorskip("matplotlib")
    png = tmp_path / "p.png"
    dither_plot.plot(list(rows.values()), str(png))
    assert png.stat().st_size > 0
