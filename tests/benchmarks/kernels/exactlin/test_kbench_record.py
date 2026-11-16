"""kbench.py's result: the gate's verdict as its validation, and the metric only where the gate passed and the metric's
model was timed in both arms. CPU only: torch, int8_fused and kgate are stand-ins."""
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
import types
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "kernels" / "exactlin"
STEP = {"llama-3.2-1b": {"exact": {"us_per_layer_m32": 75.0, "ms_per_step_m32": 1.2},
                         "stock": {"us_per_layer_m32": 62.5, "ms_per_step_m32": 1.0}}}


@pytest.fixture
def kbench(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    engine = types.ModuleType("verity_vllm.engine")
    engine.int8_fused = types.ModuleType("int8_fused")
    package = types.ModuleType("verity_vllm")
    package.engine = engine
    for name, mod in (("torch", torch), ("verity_vllm", package), ("verity_vllm.engine", engine),
                      ("kgate", types.ModuleType("kgate"))):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    spec = importlib.util.spec_from_file_location("kbench_under_test", DIR / "kbench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gate(ok: bool) -> dict:
    return {"ok": ok, "shapes": 12, "models": ["llama-3.2-1b"]}


def test_a_passed_gate_records_the_ratio_and_both_arms(kbench: types.ModuleType, tmp_path: Path) -> None:
    kbench.record(tmp_path, _gate(True), STEP)
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["validation"]["status"] == "passed" and doc["validation"]["evidence"] == ["gate.json"]
    [m] = doc["measurements"]
    assert (m["name"], m["value"], m["unit"], m["mark"]) == (kbench.METRIC, pytest.approx(1.2), "times stock", "measured")
    assert (m["exact_ms_per_step"], m["stock_ms_per_step"]) == (1.2, 1.0)


def test_a_failed_gate_records_no_measurement(kbench: types.ModuleType, tmp_path: Path) -> None:
    kbench.record(tmp_path, _gate(False), STEP)
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["validation"]["status"] == "failed" and doc["measurements"] == []


def test_no_measurement_without_the_metrics_model(kbench: types.ModuleType, tmp_path: Path) -> None:
    kbench.record(tmp_path, _gate(True), {"llama-3.1-8b": STEP["llama-3.2-1b"]})
    assert json.loads((tmp_path / "result.json").read_text())["measurements"] == []


def test_the_metric_is_the_one_metrics_toml_declares(kbench: types.ModuleType) -> None:
    [row] = tomllib.loads((DIR / "metrics.toml").read_text())["metric"]
    assert row["id"] == kbench.METRIC and f"--models {kbench.METRIC_MODEL}" in row["command"]
