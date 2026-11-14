"""The dashboard's runner (`benchmarks/dashboard/run.py`): `running` counts a top goal only when its own number, the first
metric the dashboard lists under it, has a target, a fresh value and a mark, and names each other goal with the first of
those it lacks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "run.py"


def _runner():
    spec = importlib.util.spec_from_file_location("dashboard_run", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(id: str, target=1.0, mark="measured", stale=False, latest=True) -> dict:
    value = {"value": 1.0, "target": target, "mark": mark, "time": "t"} if latest else None
    return {"id": id, "latest": value, "mark": mark if latest else None, "stale": stale}


def test_each_point_a_number_can_miss():
    report = {
        "metrics": [_row("a/ok"), _row("b/no-target", target=None), _row("c/stale", stale=True),
                    _row("d/never", target=None, latest=False), _row("e/no-mark", mark=None), _row("f/second")],
        "top": [{"id": "g/runs", "metrics": ["a/ok"]}, {"id": "g/none", "metrics": []},
                {"id": "g/undeclared", "metrics": ["z/gone"]}, {"id": "g/target", "metrics": ["b/no-target", "a/ok"]},
                {"id": "g/stale", "metrics": ["c/stale"]}, {"id": "g/never", "metrics": ["d/never"]},
                {"id": "g/mark", "metrics": ["e/no-mark"]}, {"id": "g/listed-second", "metrics": ["z/gone", "f/second"]}],
    }
    assert _runner().running(report) == (2, 8, [
        ("g/none", "no number"), ("g/undeclared", "no number"), ("g/target", "no target"),
        ("g/stale", "stale"), ("g/never", "never measured"), ("g/mark", "no mark")])


def test_the_value_then_the_row():
    run = _runner()
    row = {"id": "a/x", "latest": {"value": 1.0, "target": None, "mark": None}, "target": 3.0, "mark": "modeled",
           "stale": False}
    assert run.running({"metrics": [row], "top": [{"id": "g", "metrics": ["a/x"]}]}) == (1, 1, [])
    never = {"id": "a/x", "latest": None, "target": 3.0, "mark": None, "stale": False}
    assert run.running({"metrics": [never], "top": [{"id": "g", "metrics": ["a/x"]}]}) == (0, 1,
                                                                                         [("g", "never measured")])
