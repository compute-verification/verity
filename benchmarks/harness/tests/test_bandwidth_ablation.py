"""The bandwidth audit's breakdown arms (`verity_bench.vllm.bandwidth_ablation`) and its spec: the top-up-off spans arms are
the bandwidth audit's `BandwidthArm` built through its supported settings, uncredited and token-controlled by
bandwidth-plain, and the spec runs them beside the credited and spans arms, plain first. No engine."""
from __future__ import annotations

import ast
import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from verity_bench.vllm.arms import resolve_arms
from verity_bench.vllm.presets import resolve_spec

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = ("VERITY_VLLM_BENCH_TOPUP", "VERITY_VLLM_BENCH_SPANS", "VERITY_VLLM_BENCH_KV_RECEIPTS",
            "VERITY_VLLM_BENCH_CODEC_ROUNDS")
SPEC = "llama31-8b-fp8-1gpu-bandwidth-ablation"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    from benchmarks.vllm_encoded import bench_arm

    from verity_bench.vllm import bandwidth_ablation

    return bench_arm, bandwidth_ablation


def test_the_ablations_are_bandwidth_arms_with_the_top_up_off_and_spans_on(modules):
    bench_arm, ablation = modules
    want = {"PLAIN_SPANS_NO_TOPUP": ("bandwidth-plain-spans-no-topup", False),
            "AUDIT_SPANS_NO_TOPUP": ("bandwidth-audit-spans-no-topup", True)}
    assert set(ablation.__all__) == set(want)
    for name, (arm_id, encoded) in want.items():
        arm = getattr(ablation, name)
        assert isinstance(arm, bench_arm.BandwidthArm)
        assert (arm.id, arm.encoded, arm.control, arm.resident) == (arm_id, encoded, bench_arm.PLAIN.id, False)
        assert arm.settings() == {**bench_arm.PLAIN.settings(), "topup": False, "spans": True}
        assert arm_id not in bench_arm.CREDITED
        judged = bench_arm.credit(arm_id, topup=False, spans=True, empty_runs=0)
        assert judged["passes"] and not judged["credited"]


def test_the_ablations_use_only_the_arms_supported_names(modules):
    """A rename inside `benchmarks/vllm_encoded` breaks this test, not an install on a GPU node: the module imports
    only `bench_arm`'s exported names and sets no knob, pauses no top-up and reads no arm's state itself."""

    bench_arm, ablation = modules
    source = Path(ablation.__file__).read_text()
    tree = ast.parse(source)
    imported = [(node.module, alias.name) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names if node.module != "__future__"]
    assert imported and all(module == "benchmarks.vllm_encoded.bench_arm" and name in bench_arm.__all__
                            for module, name in imported)
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    code = ast.unparse(tree)
    for private in ("VERITY_VLLM_BENCH_", "environ", "tu_pause", ".so", ".state", "topup.", "_Installed"):
        assert private not in code, private


def test_the_ablation_spec_runs_the_credited_spans_and_top_up_off_arms_against_plain_first(modules):
    bench_arm, ablation = modules
    spec, bandwidth = resolve_spec(SPEC), resolve_spec("llama31-8b-fp8-1gpu-bandwidth")
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [a.id for a in arms] == ["bandwidth-plain", "bandwidth-audit", "bandwidth-plain-spans", "bandwidth-audit-spans",
                                    "bandwidth-plain-spans-no-topup", "bandwidth-audit-spans-no-topup"]
    assert [a.id for a in arms] == list(spec.arms)
    assert all(a.control == "bandwidth-plain" and not a.resident for a in arms[1:])
    objects = [bench_arm.PLAIN, bench_arm.AUDIT, bench_arm.PLAIN_SPANS, bench_arm.AUDIT_SPANS,
               ablation.PLAIN_SPANS_NO_TOPUP, ablation.AUDIT_SPANS_NO_TOPUP]
    for arm, obj in zip(arms, objects, strict=True):
        module, name = arm.module_spec.split(":")
        assert getattr(importlib.import_module(module), name) is obj, arm.id
    assert [(o.settings()["topup"], o.settings()["spans"]) for o in objects] == [
        (True, False), (True, False), (True, True), (True, True), (False, True), (False, True)]
    assert not {"VERITY_VLLM_BENCH_TOPUP", "VERITY_VLLM_BENCH_SPANS"} & set(spec.env)
    assert spec.reps % 2 == 0
    assert replace(bandwidth.preset, id=spec.preset.id) == spec.preset
    same = dict(id=spec.id, arms=spec.arms, arm_modules=spec.arm_modules, reps=spec.reps, preset=spec.preset)
    assert replace(bandwidth, **same) == spec
