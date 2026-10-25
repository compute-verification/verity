"""The bandwidth audit's ablation arms (`verity_bench.vllm.bandwidth_ablation`) against fake bandwidth arms: each sets its
knobs for the install only, pauses its filler once the engine is ready, forwards the arm's gate, spans and extras, and
is token-controlled by bandwidth-plain; the ablation spec runs them beside plain and audit, plain first. No engine."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from verity_bench.vllm import bandwidth_ablation as BA
from verity_bench.vllm.arms import resolve_arms
from verity_bench.vllm.presets import resolve_spec

KNOBS = ("VERITY_VLLM_BENCH_KV_RECEIPTS", "VERITY_VLLM_BENCH_CODEC_ROUNDS")
ROOT = Path(__file__).resolve().parents[3]


class FakeTopup:
    def __init__(self, events):
        self.so = self
        self.events = events

    def tu_pause(self, on):
        self.events.append(("pause", on))


class FakeState:
    def __init__(self, events):
        self.events, self.topup = events, FakeTopup(events)

    def ready(self, target):
        self.events.append("ready")

    def close(self):
        self.events.append("close")


class FakeArm:
    def __init__(self):
        self.events, self.seen = [], {}

    def install(self, target):
        self.seen = {name: os.environ.get(name) for name in KNOBS}
        return FakeState(self.events)

    def gate(self, target):
        return {"passed": True, "layers": 225}

    def spans(self, target):
        return {"filler_bytes": 0.0}

    def extras(self, target):
        return {"filler_share": 0.0}

    def kernel_digest(self, target):
        return "d" * 64


@pytest.fixture
def fake(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(BA.Ablation, "base", lambda self: arm)
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    return arm


def test_an_ablation_sets_its_knobs_for_the_install_only(fake, monkeypatch):
    monkeypatch.setenv("VERITY_VLLM_BENCH_KV_RECEIPTS", "on")
    state = BA.AUDIT_NO_KV_RECEIPTS.install(None)
    assert fake.seen == {"VERITY_VLLM_BENCH_KV_RECEIPTS": "off", "VERITY_VLLM_BENCH_CODEC_ROUNDS": None}
    assert os.environ["VERITY_VLLM_BENCH_KV_RECEIPTS"] == "on"
    BA.AUDIT_K1.install(None)
    assert fake.seen == {"VERITY_VLLM_BENCH_KV_RECEIPTS": "on", "VERITY_VLLM_BENCH_CODEC_ROUNDS": "1"}
    assert "VERITY_VLLM_BENCH_CODEC_ROUNDS" not in os.environ
    state.ready(None)
    assert fake.events == ["ready"]


def test_a_no_filler_ablation_pauses_the_filler_once_the_engine_is_ready(fake):
    for arm in (BA.AUDIT_NO_FILLER, BA.PLAIN_NO_FILLER):
        fake.events.clear()
        state = arm.install(None)
        state.ready(None)
        state.close()
        assert fake.events == ["ready", ("pause", 1), "close"]


def test_an_ablation_forwards_the_arms_gate_spans_extras_and_digest(fake):
    arm = BA.AUDIT_NO_FILLER
    assert arm.gate(None) == {"passed": True, "layers": 225, "ablation": {"env": {}, "filler": False}}
    assert arm.spans(None) == {"filler_bytes": 0.0} and arm.extras(None) == {"filler_share": 0.0}
    assert arm.kernel_digest(None) == "d" * 64


def test_the_ablations_wrap_memory_accountings_arms(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.vllm_encoded import bench_arm

    assert BA.AUDIT_K1.base() is bench_arm.AUDIT and BA.PLAIN_NO_FILLER.base() is bench_arm.PLAIN
    for knob in KNOBS:
        assert f'"{knob}"' in Path(bench_arm.__file__).read_text(), knob


def test_the_ablation_spec_times_every_ablation_against_plain_first(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec, bandwidth = resolve_spec("llama31-8b-fp8-1gpu-bandwidth-ablation"), resolve_spec("llama31-8b-fp8-1gpu-bandwidth")
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [a.id for a in arms[:2]] == ["bandwidth-plain", "bandwidth-audit"]
    assert all(a.control == "bandwidth-plain" and not a.resident for a in arms[1:])
    assert {a.id for a in arms[2:]} == {a.id for a in (BA.AUDIT_NO_KV_RECEIPTS, BA.AUDIT_K1, BA.AUDIT_NO_FILLER,
                                                       BA.PLAIN_NO_FILLER)}
    assert spec.reps % 2 == 0
    assert replace(bandwidth.preset, id=spec.preset.id) == spec.preset
    same = dict(id=spec.id, arms=spec.arms, arm_modules=spec.arm_modules, reps=spec.reps, preset=spec.preset)
    assert replace(bandwidth, **same) == spec
