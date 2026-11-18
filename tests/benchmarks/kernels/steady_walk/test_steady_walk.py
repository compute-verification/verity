"""steady_walk.py's result: a statement's GPU-seconds with the walk on the card over on the host, one member a pair, only
for two rows that differ in the walk alone, else the member without a value and why; and each walk spec is its base's
point with the walk on."""
from __future__ import annotations

import importlib.util
import json
import sys
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


def _result(out: Path) -> dict:
    return json.loads((out / "result.json").read_text())


def test_two_rows_differing_in_the_walk_record_the_ratio_and_its_member(sw: types.ModuleType, tmp_path: Path) -> None:
    glm, llama = sw.PAIRS
    entries = [sw.entry(glm, "glm-5.3", "h200", _row(False, 0.4), _row(True, 0.5), {"base": "r1", "walk": "r2"}),
               sw.entry(llama, "llama-3.2-1b", "h200", why="no passing run")]
    sw.record(tmp_path, entries, {"glm-5.3-h200": {"runs": {"base": "r1", "walk": "r2"}}})
    doc = _result(tmp_path)
    assert doc["validation"]["status"] == "passed" and doc["validation"]["evidence"] == ["rows.json"]
    assert doc["validation"]["detail"].startswith("glm-5.3-h200:")
    m, n = doc["measurements"]
    assert (m["name"], m["unit"], m["member"], m["mark"]) == (sw.METRIC, sw.UNIT, "glm-5.3-h200", "measured")
    assert m["value"] == pytest.approx(0.8)
    assert (m["walk_gpu_seconds"], m["host_gpu_seconds"], m["walk_run"], m["host_run"]) == (2.0, 2.5, "r2", "r1")
    assert (n["member"], n["value"], n["why"]) == ("llama-3.2-1b-h200", None, "no passing run")
    assert json.loads((tmp_path / "rows.json").read_text())["glm-5.3-h200"]["runs"] == {"base": "r1", "walk": "r2"}


@pytest.mark.parametrize(("base", "walk", "says"), [
    (_row(False, 0.4), _row(False, 0.5), "not the card's walk"),
    (_row(True, 0.4), _row(True, 0.5), "not the host's walk"),
    (_row(False, 0.4), {**_row(True, 0.5), "unit_walk": [True, False]}, "not the card's walk"),
    (_row(False, 0.4), {**_row(True, 0.5), "pin": "cd" * 64}, "pin differ"),
    (_row(False, 0.4), {**_row(True, 0.5), "cores": {"count": 24}}, "cores differ"),
])
def test_rows_differing_in_more_than_the_walk_give_no_value(sw: types.ModuleType, base: dict, walk: dict,
                                                            says: str) -> None:
    e = sw.entry(sw.PAIRS[1], "llama-3.2-1b", "h200", base, walk, {"base": "r1", "walk": "r2"})
    assert e["value"] is None and says in e["why"] and e["member"] == "llama-3.2-1b-h200"


@pytest.mark.parametrize(("reported", "member", "value"), [
    ("NVIDIA H200", "glm-5.3-h200", 0.8),
    ("NVIDIA H100 80GB HBM3", "glm-5.3-h100", 0.8),
    ("NVIDIA A100-SXM4-80GB", "glm-5.3-h200", None),
])
def test_the_member_takes_the_card_record_short_of_the_reported_name(
        sw: types.ModuleType, monkeypatch: pytest.MonkeyPatch, reported: str, member: str, value: float | None) -> None:
    import verity_bench
    workloads = types.SimpleNamespace(WORKLOADS={"w": types.SimpleNamespace(model="GLM-5.3")})
    monkeypatch.setattr(verity_bench, "prove_common", workloads, raising=False)
    monkeypatch.setitem(sys.modules, "verity_bench.prove_common", workloads)
    spec = types.SimpleNamespace(workload="w", card="h200")
    monkeypatch.setattr(sw, "passing", lambda name, block: (name, f"run-{name}", None, spec))
    walked = {sw.PAIRS[0].base: _row(False, 0.4), sw.PAIRS[0].walk: _row(True, 0.5)}
    monkeypatch.setattr(sw, "row_and_card", lambda store, run, point: (walked[point], reported))
    e, rows = sw.measure(None, sw.PAIRS[0])
    assert (e["member"], e["value"]) == (member, pytest.approx(0.8) if value else None)
    if value is None:
        assert e["why"].startswith("0 card records under ") and "'NVIDIA A100-SXM4-80GB'" in e["why"] and rows == {}


def test_a_run_where_no_member_has_a_value_is_refused(sw: types.ModuleType, tmp_path: Path) -> None:
    entries = [sw.entry(pair, "m", "h200", why="no passing run") for pair in sw.PAIRS]
    with pytest.raises(SystemExit, match="no passing run"):
        sw.record(tmp_path, entries, {})
    assert not (tmp_path / "result.json").exists()


def test_the_metric_is_the_one_metrics_toml_declares(sw: types.ModuleType) -> None:
    rows = [row for row in tomllib.loads((DIR / "metrics.toml").read_text())["metric"] if row["id"] == sw.METRIC]
    assert len(rows) == 1 and rows[0]["unit"] == sw.UNIT


@pytest.mark.parametrize("index", [0, 1])
def test_each_walk_spec_is_its_base_point_with_the_walk_on(sw: types.ModuleType, index: int) -> None:
    pair = sw.PAIRS[index]
    base, walk = _spec(pair.base), _spec(pair.walk)
    [point] = [row for row in base["row"] if row["type"] == pair.type]
    assert walk["row"] == [{**point, "provers": [sw.PROVERS]}] and sw.PROVERS in point["provers"]
    assert walk["env"] == {**base["env"], "FC_UNIT_WALK": "1"} and "FC_UNIT_WALK" not in base["env"]
    assert ({k: v for k, v in walk.items() if k not in ("id", "env", "row")}
            == {k: v for k, v in base.items() if k not in ("id", "env", "row")})


def test_the_llama_point_is_the_statement_wide_leaf_times(sw: types.ModuleType) -> None:
    statement = tomllib.loads((DIR / "wide_leaf" / "statement.toml").read_text())
    [row] = _spec(sw.PAIRS[1].base)["row"]
    assert (row["type"], row["n"], row["pin"]) == (statement["unit"], statement["n"], statement["pin"])
    assert row["staged"] == f"{statement['art']}/{statement['path']}"
