"""`figures.py` end to end."""
import functools
import json

import figure1
import figure2
import figure3
import figures
import pytest
import total_cost


@pytest.mark.slow
def test_figures_writes_every_figure(tmp_path, monkeypatch):
    """Every figure, its JSON and the break-even's table, Figure 1's planned on one process over its primary scope alone;
    the break-even's search is `test_total_cost`'s, stubbed here because it plans each line thousands of times."""
    monkeypatch.setattr(total_cost, "BETAS", (1e-2, 1.0))
    for planner in (total_cost, figure1):
        monkeypatch.setattr(planner, "run", functools.partial(planner.run, scope_names=(figure1.PRIMARY,)))
    be = {"beta": 1e-2, "targets": {"FP8": 1e3}, "rel": 1e-2, "spans": total_cost.SPANS,
          "rows": [{"config": "tokens", "registration": "machine", "scope": "world week", "overhead": 2.0,
                    "floor": 8e6, "h": {"FP8": None}, "beta": {"FP8": 3.0}}]}
    seen = []
    monkeypatch.setattr(total_cost, "breakeven", lambda *a, **k: seen.append(k) or be)
    figures.main(["--model", "GLM-5.3", "--out", str(tmp_path), "--points", "2", "--workers", "1", "--no-lean"])
    assert seen[0]["lean"] is False
    assert json.loads((tmp_path / "figure1.json").read_text())["breakeven"] == json.loads(json.dumps(be))
    assert (tmp_path / "breakeven.txt").read_text() == total_cost.breakeven_table(be)
    for name, schema in (("figure1", total_cost.SCHEMA), ("figure1_overhead", figure1.SCHEMA),
                         ("figure2", figure2.SCHEMA), ("figure3", figure3.SCHEMA)):
        assert json.loads((tmp_path / f"{name}.json").read_text())["schema"] == schema
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (tmp_path / f"{name}.svg").exists()
    for pattern in ("figure1_overhead_appendix*", "figure1_skipped_work*"):
        assert len(list(tmp_path.glob(f"{pattern}.png"))) == 1, pattern
    assert len(list(tmp_path.glob("figure1*.png"))) == 4
