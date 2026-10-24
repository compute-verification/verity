"""Every pinned preset under ``benchmarks/specs`` resolves as an engine spec (`verity_vllm.engine.spec`) to its
checkpoint pin (`verity_vllm.config.EnginePin`)."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from verity_vllm import EnginePin
from verity_vllm.engine import spec as ES

SPECS = Path(__file__).resolve().parents[1]


def _presets() -> list[dict]:
    out = []
    for p in sorted(SPECS.glob("*.toml")):
        doc = tomllib.loads(p.read_text())
        if doc.get("schema") == "verity-bench/vllm-spec/v0" and doc["preset"].get("revision"):
            out.append(doc["preset"])
    return out


def test_the_specs_hold_pinned_presets():
    assert _presets()


@pytest.mark.parametrize("preset", _presets(), ids=lambda p: p["id"])
def test_every_pinned_benchmark_preset_resolves_as_an_engine_spec(preset):
    spec = ES.from_mapping({"schema": ES.SCHEMA, "model": preset["model"], "revision": preset["revision"],
                            "engine": {"tensor_parallel_size": preset["gpu_count"],
                                       "max_model_len": preset["engine_settings"]["max_model_len"]}})
    pin = EnginePin.resolve(spec.model, spec.revision, world=spec.engine.tensor_parallel_size)
    assert pin.case == preset["case"]
