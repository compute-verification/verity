"""`figures.py` end to end: what it writes for one model, with Figure 1 planned on one process over its primary scope alone
and the break-even's search, `test_total_cost`'s, stubbed because it plans each line thousands of times."""
import functools
import json

import figure1
import figure2
import figure3
import figure4
import figures
import pytest
import settings
import total_cost

#: the bounds against overhead that take most of a minute each to plan at two overheads: the two-stage designs stratified
#: by width that charge a request at most what it serves
SLOWEST = tuple(d for d, (stratified, form) in figure1.TWO_STAGE.items() if stratified and form)
DESIGNS = {d for ds in figure1.DESIGNS.values() for d in ds}


def written(tmp_path, monkeypatch, designs: set[str]) -> None:
    """`figures.main` into ``tmp_path`` with the bounds against overhead planned for ``designs``, and what it wrote: each
    figure's JSON under its schema with its PNG and SVG, the break-even as its search gave it with its table, the bounds
    against overhead for ``designs`` and no other, and Figure 1's four renders."""
    monkeypatch.setattr(total_cost, "BETAS", (1e-2, 1.0))
    monkeypatch.setattr(total_cost, "run", functools.partial(total_cost.run, scope_names=(figure1.PRIMARY,)))
    monkeypatch.setattr(figure1, "run", functools.partial(figure1.run, scope_names=(figure1.PRIMARY,),
                                                          designs=tuple(sorted(designs))))
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
                         ("figure2", figure2.SCHEMA), ("figure3", figure3.SCHEMA), ("figure4", figure4.SCHEMA)):
        assert json.loads((tmp_path / f"{name}.json").read_text())["schema"] == schema
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (tmp_path / f"{name}.svg").exists()
    over = json.loads((tmp_path / "figure1_overhead.json").read_text())
    for rows in ([s for p in over["panels"].values() for s in p.get("series", ())], over["at_overhead"]):
        assert {r["design"] for r in rows} & DESIGNS == designs
    for pattern in ("figure1_overhead_appendix*", "figure1_skipped_work*"):
        assert len(list(tmp_path.glob(f"{pattern}.png"))) == 1, pattern
    assert len(list(tmp_path.glob("figure1*.png"))) == 4


@pytest.mark.slow
def test_figures_writes_a_figure_of_each_kind(tmp_path, monkeypatch):
    """One of each kind of figure `figures.py` writes, with its JSON, and the break-even's table. The only run of
    `figures.main` in every check: it holds each planner's output to its writer and its renders, and Figure 4 to
    Figure 1's JSON, which the figures' own tests hold one module at a time. The bounds against overhead are planned
    without `SLOWEST`, so the main figure has no capped line and the appendix six of its eight exfiltration lines. What
    `test_figures_writes_every_figure` catches that this does not: those two designs at the primary scope, their rows
    in the table at the vertical line, and all eight lines plotted on one appendix (`test_figure1`'s capped curves
    plan and render the two at a smaller scope, beside the uncapped design alone)."""
    written(tmp_path, monkeypatch, DESIGNS - set(SLOWEST))


@pytest.mark.slow
@pytest.mark.skipif(not settings.SLOW, reason="about 7 minutes on a check node, most of it planning two designs' bounds "
                                              "against overhead: VERITY_SPOT_FIGURES_SLOW=1")
def test_figures_writes_every_figure(tmp_path, monkeypatch):
    """Every figure with every design of the bounds against overhead, `SLOWEST` among them."""
    written(tmp_path, monkeypatch, DESIGNS)
