"""The frontier renderer's point schema and its frontier rule: x ≥ x* = (1 − τ)/μ_budget, references excluded."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frontier as F  # noqa: E402


def point(**kw):
    p = {"schema": F.SCHEMA, "lane": "t", "b": 521, "k": 1, "arithmetic": "a", "batch": 1, "y_total": 1.6,
         "y_decode": 0.3, "y_tag": 0.2, "y_commit": 0.1, "x": 230.0, "tau": 0.6}
    p.update(kw)
    return p


def test_check_fills_base_and_measured():
    (p,) = F.check([point()])
    assert p["y_base"] == pytest.approx(1.0) and p["measured"] is True


def test_check_refuses_other_schemas_and_missing_fields():
    with pytest.raises(ValueError):
        F.check([point(schema="other")])
    bad = point()
    del bad["tau"]
    with pytest.raises(ValueError):
        F.check([bad])


def test_frontier_takes_the_lowest_y_at_x_at_least_x_star():
    budget = 0.005
    assert F.x_star(0.6, budget) == pytest.approx(80.0) and F.x_star(0.6, budget, 1.6) == pytest.approx(200.0)
    pts = F.check([point(y_total=1.5, x=150.0), point(y_total=1.7), point(y_total=1.6), point(batch=32, y_total=1.9),
                   point(batch=32, y_total=2.0, x=300.0), point(y_total=1.0, reference=True)])
    best = F.frontier(pts, budget)
    assert best[1]["y_total"] == 1.6 and best[32]["y_total"] == 2.0
    assert F.frontier(pts, budget, d=1.0)[1]["y_total"] == 1.5
    assert "| 1 | 521 | 1 |" in F.table(pts, budget) and "none at x ≥ x*" in F.table(pts, budget, batches=(8,))


def test_best_per_config_keeps_the_lowest_y_of_each_configuration():
    pts = F.check([point(y_total=1.7), point(y_total=1.6, arithmetic="b"), point(k=2, y_total=1.9),
                   point(lane="u", y_total=1.8), point(batch=32, y_total=2.0), point(y_total=1.0, measured=False)])
    best = {(p["lane"], p["k"], p["batch"], p["measured"]): p["y_total"] for p in F.best_per_config(pts)}
    assert best == {("t", 1, 1, True): 1.6, ("t", 2, 1, True): 1.9, ("u", 1, 1, True): 1.8, ("t", 1, 32, True): 2.0,
                    ("t", 1, 1, False): 1.0}
