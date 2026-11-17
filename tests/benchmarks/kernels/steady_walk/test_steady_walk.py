"""steady_walk.py's result: a statement's GPU-seconds with the walk on the card over on the host, only for two rows that
differ in the walk alone, its member the model and the card; and its walk spec is the base's point with the walk on."""
from __future__ import annotations

import importlib.util
import json
import tomllib
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
DIR, SPECS = ROOT / "benchmarks" / "kernels", ROOT / "benchmarks" / "harness" / "specs"
PIN = "ab" * 64


def _row(walk: bool, rate: float) -> dict:
    return {"type": "census:T", "pin": PIN, "provers": 1, "gpus": 1, "cores": {"count": 10, "list": "72-81"},
            "statements_per_s": rate, "unit_walk": [walk]}


@pytest.fixture
def sw(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    from verity_bench import kit
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": "NVIDIA H200", "gpus": [], "node": "n", "commit": None})
    spec = importlib.util.spec_from_file_location("steady_walk_under_test", DIR / "steady_walk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spec(name: str) -> dict:
    return tomllib.loads((SPECS / f"{name}.toml").read_text())


def test_two_rows_differing_in_the_walk_record_the_ratio_and_its_member(sw: types.ModuleType, tmp_path: Path) -> None:
    sw.record(tmp_path, _row(False, 0.4), _row(True, 0.5), {"base": "r1", "walk": "r2"}, "NVIDIA H200", "glm-5.3")
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["validation"]["status"] == "passed" and doc["validation"]["evidence"] == ["rows.json"]
    [m] = doc["measurements"]
    assert (m["name"], m["unit"], m["member"], m["mark"]) == (sw.METRIC, sw.UNIT, "glm-5.3-h200", "measured")
    assert m["value"] == pytest.approx(0.8)
    assert (m["walk_gpu_seconds"], m["host_gpu_seconds"], m["walk_run"], m["host_run"]) == (2.0, 2.5, "r2", "r1")
    assert json.loads((tmp_path / "rows.json").read_text())["runs"] == {"base": "r1", "walk": "r2"}


@pytest.mark.parametrize(("base", "walk", "says"), [
    (_row(False, 0.4), _row(False, 0.5), "not the card's walk"),
    (_row(True, 0.4), _row(True, 0.5), "not the host's walk"),
    (_row(False, 0.4), {**_row(True, 0.5), "unit_walk": [True, False]}, "not the card's walk"),
    (_row(False, 0.4), {**_row(True, 0.5), "pin": "cd" * 64}, "pin differ"),
    (_row(False, 0.4), {**_row(True, 0.5), "cores": {"count": 24}}, "cores differ"),
])
def test_rows_differing_in_more_than_the_walk_are_refused(sw: types.ModuleType, tmp_path: Path, base: dict, walk: dict,
                                                          says: str) -> None:
    with pytest.raises(SystemExit, match=says):
        sw.record(tmp_path, base, walk, {"base": "r1", "walk": "r2"}, "NVIDIA H200", "glm-5.3")
    assert not (tmp_path / "result.json").exists()


def test_the_metric_is_the_one_metrics_toml_declares(sw: types.ModuleType) -> None:
    rows = [row for row in tomllib.loads((DIR / "metrics.toml").read_text())["metric"] if row["id"] == sw.METRIC]
    assert len(rows) == 1 and rows[0]["unit"] == sw.UNIT


def test_the_walk_spec_is_the_base_point_with_the_walk_on(sw: types.ModuleType) -> None:
    base, walk = _spec(sw.BASE), _spec(sw.WALK)
    [point] = [row for row in base["row"] if row["type"] == sw.TYPE]
    assert walk["row"] == [{**point, "provers": [sw.PROVERS]}] and sw.PROVERS in point["provers"]
    assert walk["env"] == {**base["env"], "FC_UNIT_WALK": "1"}
    assert ({k: v for k, v in walk.items() if k not in ("id", "env", "row")}
            == {k: v for k, v in base.items() if k not in ("id", "env", "row")})
