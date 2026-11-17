"""wleaf.py's result: every step's ten checks as its validation, and the two ratios only where every step passed and both
leaves were timed, each the wide leaf's median over today's, its member the model and the card. CPU only."""
from __future__ import annotations

import importlib.util
import json
import tomllib
import types
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "kernels" / "wide_leaf"
HOST = {"card": "NVIDIA H200", "gpus": [], "node": "n", "commit": None, "cpus": list(range(10))}
STAGE = {"definition": "Gemm_v2", "n": 256, "k_log": 27, "vus_per_block": 1}


@pytest.fixture
def wleaf(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    from verity_bench import kit
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": "no card here", "gpus": [], "node": "n", "commit": None})
    spec = importlib.util.spec_from_file_location("wleaf_under_test", DIR / "wleaf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(tag: str, wide: bool, gpu_s: float, size: int, status: str = "pass") -> dict:
    return {"step": tag, "wide": wide, "status": status, "why": None if status == "pass" else "flat: drifted",
            "gpu_s": gpu_s, "proof_bytes": [size]}


STEPS = [_step("a-own", False, 3.2, 2_400_000), _step("b-wide", True, 2.4, 8_800_000),
         _step("c-wide", True, 2.5, 8_800_000), _step("d-own", False, 3.1, 2_400_000),
         _step("e-own", False, 3.3, 2_400_000), _step("f-wide", True, 2.3, 8_800_000)]


def _result(out: Path) -> dict:
    return json.loads((out / "result.json").read_text())


def test_passed_steps_record_both_ratios_under_the_model_and_card(wleaf: types.ModuleType, tmp_path: Path) -> None:
    wleaf.record(tmp_path, STEPS, HOST, STAGE, "llama-3.2-1b")
    doc = _result(tmp_path)
    assert doc["validation"]["status"] == "passed" and doc["validation"]["evidence"] == ["out/summary.json"]
    time, size = doc["measurements"]
    assert (time["name"], time["value"], time["unit"], time["mark"]) == (
        wleaf.TIME, pytest.approx(2.4 / 3.2), "times today's leaf", "measured")
    assert (size["name"], size["value"], size["unit"]) == (wleaf.SIZE, pytest.approx(8.8 / 2.4), "times today's proof")
    for m in (time, size):
        assert (m["member"], m["m"], m["provers"], m["cores"]) == ("llama-3.2-1b-h200", 35, 1, 10)
        assert "monitoring period" in m["not_said"] and "unit cover" in m["not_said"]
    assert time["own_gpu_s"] == [3.2, 3.1, 3.3] and time["wide_gpu_s"] == [2.4, 2.5, 2.3]
    assert (size["own_proof_bytes"], size["wide_proof_bytes"]) == (2_400_000, 8_800_000)


def test_a_step_that_failed_its_checks_records_no_measurement(wleaf: types.ModuleType, tmp_path: Path) -> None:
    wleaf.record(tmp_path, [*STEPS[:5], _step("f-wide", True, 2.3, 8_800_000, status="fail")], HOST, STAGE,
                 "llama-3.2-1b")
    doc = _result(tmp_path)
    assert doc["validation"]["status"] == "failed" and "f-wide: fail (flat: drifted)" in doc["validation"]["detail"]
    assert doc["measurements"] == []


def test_one_leaf_untimed_records_no_measurement(wleaf: types.ModuleType, tmp_path: Path) -> None:
    wleaf.record(tmp_path, [s for s in STEPS if not s["wide"]], HOST, STAGE, "llama-3.2-1b")
    assert _result(tmp_path)["measurements"] == []


def test_a_step_whose_proofs_vary_in_size_records_no_measurement(wleaf: types.ModuleType, tmp_path: Path) -> None:
    wleaf.record(tmp_path, [{**STEPS[0], "proof_bytes": [2_400_000, 2_400_064]}, *STEPS[1:]], HOST, STAGE, "llama-3.2-1b")
    assert _result(tmp_path)["measurements"] == []


def test_the_metrics_are_the_ones_metrics_toml_declares(wleaf: types.ModuleType) -> None:
    rows = tomllib.loads((DIR / "metrics.toml").read_text())["metric"]
    assert [row["id"] for row in rows] == [wleaf.TIME, wleaf.SIZE]
