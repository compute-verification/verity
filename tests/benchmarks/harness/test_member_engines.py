from pathlib import Path
from typing import Any

import pytest

from verity_vllm.engine.build import execution_label_of
from verity_vllm.engine.served import execution

from verity_bench.vllm.arms import resolve_arms
from verity_bench.vllm.engine import arm_execution, engine_args
from verity_bench.vllm.presets import resolve_spec, served

R6, BIG, SMALL = ("llama31-8b-fp8-1gpu-bandwidth-served-r6", "llama31-8b-fp8-1gpu-bandwidth-served-h200",
                  "llama32-1b-fp8-1gpu-bandwidth-served-h200")


def _engines(name: str) -> dict[str, dict[str, Any]]:
    spec = resolve_spec(name)
    preset, eager = served(spec.preset, spec.eager)
    out = {}
    for arm in resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings):
        built, arm_eager = arm_execution(arm, preset, eager)
        engine = engine_args(built, arm.precision or built.precision, resident=False, eager=arm_eager)
        if built.model_digest is None:
            label = engine.get("execution") or execution_label_of(engine["enforce_eager"])
            invariant = bool(execution(label)[0].get("VLLM_BATCH_INVARIANT"))
        else:
            label, invariant = None, False
        out[arm.id] = {"engine": {k: v for k, v in engine.items() if k != "execution"}, "label": label,
                       "batch_invariant": invariant}
    return out


def test_the_bandwidth_audits_h200_members_build_every_arm_as_the_rtx_record_does_but_for_the_model(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    r6, big, small = _engines(R6), _engines(BIG), _engines(SMALL)
    assert {arm: e["engine"] for arm, e in big.items()} == {arm: e["engine"] for arm, e in small.items()}
    assert {arm: e["engine"] for arm, e in r6.items()} == {arm: e["engine"] for arm, e in big.items()}
    for engines in (r6, big, small):
        assert not any(e["batch_invariant"] for e in engines.values())
    assert {e["label"] for e in r6.values()} == {e["label"] for e in big.values()} == {None}
    assert {e["label"] for e in small.values()} == {"compiled"}
    preset, eager = served(resolve_spec(R6).preset, False)
    assert "execution" not in engine_args(preset, "fp8", resident=False, eager=eager)
