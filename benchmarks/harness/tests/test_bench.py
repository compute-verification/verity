from __future__ import annotations

from dataclasses import replace

import pytest

from verity_bench.cli import _parser
from verity_bench.vllm.arms import Arm, resolve_arms
from verity_bench.vllm.presets import GPUS, SweepPoint, expand_sweep, get_preset, local_checkpoint
from verity_bench.vllm.result import IdentityMismatch, judge, validate
from verity_bench.vllm.runner import ResidentCapacityError, run_benchmark
from verity_bench.vllm.table import render


class FakeEngine:
    def __init__(self, arm, events, *, gate_ok=True, bad_tokens=False):
        self.arm, self.events, self.gate_ok, self.bad_tokens = arm, events, gate_ok, bad_tokens
        self.identity = {"vllm_version": "0.fake+gd9105ea80", "kernel_digest": f"digest-{arm.id}"}

    def gate(self, point):
        self.events.append(("gate", self.arm.id, point.id))
        token = 8 if self.bad_tokens else 7
        return {f"{point.id}-0": (token, 9)}

    def custom_gate(self):
        self.events.append(("custom-gate", self.arm.id))
        return {"passed": self.gate_ok, "provided": True}

    def warmup(self, point):
        self.events.append(("warmup", self.arm.id, point.id))

    def measure(self, point):
        rep = sum(e[:2] == ("measure", self.arm.id) for e in self.events)
        self.events.append(("measure", self.arm.id, point.id))
        scale = 1.0 if self.arm.id == "stock-fp8" else 2.0
        return {
            "metrics": {
                "time_per_output_token_s": scale * (rep + 1),
                "time_to_first_token_s": scale * 2,
                "tokens_per_s": 100 / scale,
                "peak_gpu_memory_bytes": 1000 * scale,
            },
            "spans": {"hashing": scale},
            "extras": {"share": scale / 10},
        }

    def close(self):
        self.events.append(("close", self.arm.id))


class FakeFactory:
    def __init__(self, *, resident_failure=None, gate_ok=True, bad_tokens=None):
        self.events = []
        self.resident_failure = resident_failure
        self.failed = False
        self.gate_ok = gate_ok
        self.bad_tokens = bad_tokens

    def create(self, arm, preset, *, resident):
        self.events.append(("create", arm.id, resident))
        if resident and arm.id == self.resident_failure and not self.failed:
            self.failed = True
            raise ResidentCapacityError(f"{arm.id} does not fit")
        return FakeEngine(arm, self.events, gate_ok=self.gate_ok,
                          bad_tokens=arm.id == self.bad_tokens)


def inputs():
    preset = get_preset("llama31-8b-fp8-1gpu")
    point = SweepPoint(32, 512, "mixed", 64)
    arms = (Arm("stock-fp8", precision="fp8"), Arm("candidate", control="stock-fp8"))
    return preset, (point,), arms


def test_sweep_expansion_and_staggered_mixed_requests():
    quick, full = expand_sweep("quick"), expand_sweep("full")
    assert [(p.batch, p.prompt_tokens, p.traffic) for p in quick] == [(8, 512, "mixed"), (32, 512, "mixed")]
    assert len(full) == 4 * 2 * 3
    assert {p.prompt_tokens for p in full} == {512, 8192}
    assert {p.traffic for p in full} == {"decode-heavy", "prefill-heavy", "mixed"}
    assert [r.arrive_step for r in quick[0].requests()] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_cli_and_presets_use_the_benchmark_package_and_stable_gpu_ids(tmp_path):
    args = _parser().parse_args([
        "vllm", "--preset", "llama31-8b-fp8-1gpu", "--arms", "stock,stock-fp8", "--out", str(tmp_path),
    ])
    assert args.command == "vllm"
    assert get_preset(args.preset).gpu_id == "rtx-pro-6000"
    assert set(GPUS) == {"rtx-pro-6000", "h200"}


def test_stock_precisions_are_independent_reference_arms():
    arms = resolve_arms("stock,stock-fp8")
    assert [(arm.id, arm.precision, arm.control) for arm in arms] == [
        ("stock", "bfloat16", None),
        ("stock-fp8", "fp8", None),
    ]


def test_local_checkpoint_is_content_bound(tmp_path):
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    preset = local_checkpoint(get_preset("llama31-8b-fp8-1gpu"), tmp_path)
    assert preset.model == str(tmp_path)
    assert preset.revision is preset.case is None
    assert len(preset.model_digest) == 64
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    assert local_checkpoint(get_preset("llama31-8b-fp8-1gpu"), tmp_path).model_digest != preset.model_digest


def test_resident_engines_interleave_arms_rep_by_rep_and_discard_warmups():
    preset, points, arms = inputs()
    factory = FakeFactory()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    measured = [e[1] for e in factory.events if e[0] == "measure"]
    assert measured == ["stock-fp8", "candidate", "candidate", "stock-fp8"]
    assert doc["execution"]["interleave"] == "resident"
    assert all(doc["points"][0]["arms"][a.id]["metrics"]["time_per_output_token_s"]["n"] == 2 for a in arms)
    assert doc["points"][0]["arms"]["candidate"]["metrics"]["time_per_output_token_s"]["ratio_to_stock"]["mean"] == 2


def test_resident_failure_falls_back_to_alternating_blocks():
    preset, points, arms = inputs()
    factory = FakeFactory(resident_failure="candidate")
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=3, factory=factory, git_commit="abc")
    assert doc["execution"]["interleave"] == "blocks"
    assert "does not fit" in doc["execution"]["resident_failure"]
    measured = [e[1] for e in factory.events if e[0] == "measure"]
    assert measured == ["stock-fp8", "stock-fp8", "candidate", "candidate", "stock-fp8", "candidate"]
    assert all(doc["points"][0]["arms"][a.id]["metrics"]["time_per_output_token_s"]["n"] == 3 for a in arms)


@pytest.mark.parametrize("factory", [FakeFactory(gate_ok=False), FakeFactory(bad_tokens="candidate")])
def test_a_failed_gate_records_the_failure_and_stops_before_timing(factory):
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    assert any(not g["passed"] for g in doc["gates"])
    assert doc["points"] == []
    assert not any(e[0] == "measure" for e in factory.events)
    validate(doc)


def test_result_schema_and_identity_strict_judge():
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    validate(doc)
    other = {**doc, "identity": {**doc["identity"], "git_commit": "def"}}
    with pytest.raises(IdentityMismatch, match="git_commit"):
        judge(doc, other)
    assert judge(doc, other, ("git_commit",))["accepted"]


def test_ratio_table_puts_the_headline_cell_first():
    preset, points, arms = inputs()
    points = (replace(points[0], batch=8), points[0])
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    lines = [line for line in render(doc).splitlines() if line.startswith("| b")]
    assert lines[0].startswith("| b32-p512-mixed **headline** | time / output token |")
    assert "2.000×" in lines[0]
