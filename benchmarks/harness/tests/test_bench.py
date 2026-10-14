from __future__ import annotations

import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verity_config import Admission

from verity_bench import jobs
from verity_bench.cli import _parser
from verity_bench.vllm.arms import Arm, resolve_arms
from verity_bench.vllm.cli import effective_eager
from verity_bench.vllm.presets import (
    GPUS,
    SPEC_ROOT,
    SweepPoint,
    expand_sweep,
    get_preset,
    load_spec,
    load_sweep,
    local_checkpoint,
    resolve_spec,
)
from verity_bench.vllm.result import IdentityMismatch, judge, validate
from verity_bench.vllm.runner import ResidentCapacityError, run_benchmark
from verity_bench.vllm.table import render


class FakeEngine:
    def __init__(self, arm, events, *, gate_ok=True, bad_tokens=False, kv_tokens=None):
        self.arm, self.events, self.gate_ok, self.bad_tokens = arm, events, gate_ok, bad_tokens
        self.identity = {
            "vllm_version": "0.fake+gd9105ea80",
            "kernel_digest": f"digest-{arm.id}",
            "engine_overrides": {"quantization": "fp8"} if arm.id == "stock-fp8" else {},
            **({"kv_cache_tokens": kv_tokens} if kv_tokens is not None else {}),
        }

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
    def __init__(self, *, resident_failure=None, gate_ok=True, bad_tokens=None, kv_tokens=None):
        self.events = []
        self.resident_failure = resident_failure
        self.failed = False
        self.gate_ok = gate_ok
        self.bad_tokens = bad_tokens
        self.kv_tokens = kv_tokens or {}

    def create(self, arm, preset, *, resident):
        self.events.append(("create", arm.id, resident))
        if resident and arm.id == self.resident_failure and not self.failed:
            self.failed = True
            raise ResidentCapacityError(f"{arm.id} does not fit")
        return FakeEngine(arm, self.events, gate_ok=self.gate_ok,
                          bad_tokens=arm.id == self.bad_tokens,
                          kv_tokens=self.kv_tokens.get(arm.id))


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
    preset = get_preset(args.preset)
    assert preset.gpu_id == "rtx-pro-6000"
    assert preset.engine_settings["resident_gpu_memory_utilization"] == 0.40
    assert preset.engine_settings["block_gpu_memory_utilization"] == 0.80
    assert set(GPUS) == {"rtx-pro-6000", "h200"}


def test_every_toml_spec_strictly_loads():
    loaded = []
    for path in sorted(SPEC_ROOT.glob("*.toml")):
        first = path.read_text().splitlines()[0]
        loaded.append(load_sweep(path.stem) if "vllm-sweep/v0" in first else load_spec(path))
    assert len(loaded) == 5


def test_plan_identity_binds_resolved_spec_commit_lock_pin_and_point_but_not_admission():
    root = Path(__file__).resolve().parents[3]
    spec = resolve_spec("llama31-8b-fp8-1gpu")
    other = replace(spec, admission=Admission(999.0, "2099-01-01T00:00:00Z", 1))
    assert spec.config_sha256 == other.config_sha256
    pins = jobs.toolchain(root, spec, commit="a" * 40)
    point = expand_sweep("quick")[0]
    identity = jobs.point_identity(spec.config_sha256, pins, point)
    assert identity != jobs.point_identity(spec.config_sha256, replace(pins, commit="b" * 40), point)
    assert identity != jobs.point_identity(spec.config_sha256, replace(pins, uv_lock_sha256="c" * 64), point)
    assert identity != jobs.point_identity(spec.config_sha256, pins, expand_sweep("quick")[1])


def test_plan_marks_passing_points_and_dispatches_one_recorded_run_for_each_missing_point():
    root = Path(__file__).resolve().parents[3]
    seen = []

    def lookup(identity, commit):
        seen.append((identity, commit))
        return "r-passed" if len(seen) == 1 else None

    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lookup)
    assert [point.passing_run for point in plan.points] == ["r-passed", None]
    commands = []
    assert jobs.dispatch(plan, "llama31-8b-fp8-1gpu", root=root,
                         execute=lambda command, cwd: commands.append((list(command), cwd)) or 0) == 0
    [(command, cwd)] = commands
    assert cwd == root
    assert command[:3] == ["research", "run", "--queue"]
    assert command[command.index("--kind") + 1] == "vllm-bench-1gpu"
    assert command[command.index("--gpus") + 1] == "1"
    assert command[command.index("--gpu-model") + 1] == "rtx-pro-6000"
    assert command[command.index("--") + 1:][:4] == ["python", "-m", "verity_bench", "_vllm-point"]
    assert command[command.index("--point") + 1] == "b32-p512-mixed"


def test_every_spec_dispatches_as_a_registered_queue_kind_that_fits_it():
    """The queue refuses an unregistered kind's job without --mem-gb, so each spec's points need a kind whose shape has
    the spec's GPUs and whose wall time covers its admission."""

    kinds_toml = Path(__file__).resolve().parents[3] / "infra" / "cluster" / "kinds" / "benchmarks.toml"
    kinds = tomllib.loads(kinds_toml.read_text())["kinds"]
    phase_gpus = {"gpu1": 1, "gpu8-timed": 8}
    specs = [load_spec(p) for p in sorted(SPEC_ROOT.glob("*.toml")) if "vllm-spec/v0" in p.read_text().splitlines()[0]]
    assert specs
    for spec in specs:
        kind = kinds[jobs.queue_kind(spec.preset.gpu_count)]
        assert phase_gpus[kind["phase"]] == spec.preset.gpu_count, spec.id
        assert spec.admission.max_minutes is None or spec.admission.max_minutes <= kind["max_wall_min"], spec.id
    with pytest.raises(ValueError, match="register one in infra/cluster/kinds/benchmarks.toml"):
        jobs.queue_kind(4)


def test_plan_run_check_and_compare_are_top_level_commands(tmp_path):
    parser = _parser()
    assert parser.parse_args(["plan", "llama31-8b-fp8-1gpu"]).command == "plan"
    assert parser.parse_args(["run", "llama31-8b-fp8-1gpu"]).command == "run"
    assert parser.parse_args(["check", "llama31-8b-fp8-1gpu"]).command == "check"
    assert parser.parse_args(["compare", str(tmp_path / "a"), str(tmp_path / "b")]).command == "compare"


def test_glm53_keeps_the_verified_vllm_scheduler_default_and_runs_eager():
    preset = get_preset("glm53-fp8-8gpu")
    assert preset.revision == "aca966e4e02791568aa6a4ced368624b3d897f42"
    assert "max_num_batched_tokens" not in preset.engine_settings
    assert effective_eager(preset, requested=False) is True
    assert get_preset("llama31-8b-fp8-1gpu").engine_settings["max_num_batched_tokens"] == 16_384


def test_stock_precisions_are_independent_reference_arms():
    arms = resolve_arms("stock,stock-fp8")
    assert [(arm.id, arm.precision, arm.control) for arm in arms] == [
        ("stock", "bfloat16", None),
        ("stock-fp8", "fp8", None),
    ]


def test_experimental_arm_can_require_an_isolated_engine(monkeypatch):
    module = ModuleType("bench_isolated_arm")
    module.ARM = SimpleNamespace(id="isolated", install=lambda target: None, resident=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    arm, = resolve_arms("isolated", (f"{module.__name__}:ARM",))
    assert arm.experimental and not arm.resident


def test_graphs_are_refused_for_arms_bound_to_the_build():
    from verity_bench.vllm.arms import build_bound
    from verity_bench.vllm.runner import ProcessEngineFactory

    bound_arm, _control = resolve_arms("sampled-proofs:hm96,stock-fp8")
    assert build_bound(bound_arm) == ("sampled-proofs:hm96",)
    assert build_bound(resolve_arms("stock")[0]) == ()
    with pytest.raises(ValueError, match="Build"):
        ProcessEngineFactory(eager=False).create(bound_arm, get_preset("llama31-8b-fp8-1gpu"), resident=True)


def test_a_dead_engine_process_reports_its_own_error_not_a_connection_reset():
    from verity_bench.vllm.runner import _ProcessEngine

    class DeadConn:
        def __init__(self, final=None):
            self.final = final

        def send(self, _):
            raise BrokenPipeError

        def poll(self, _timeout):
            return self.final is not None

        def recv(self):
            return self.final

    class DeadProcess:
        exitcode = -9

        def join(self):
            pass

    point = SweepPoint(batch=8, prompt_tokens=512, traffic="mixed", output_tokens=64)
    drained = _ProcessEngine(DeadProcess(), DeadConn({"error": "RuntimeError: CUDA out of memory", "trace": ""}), {})
    with pytest.raises(RuntimeError, match="died during gate: RuntimeError: CUDA out of memory"):
        drained.gate(point)
    silent = _ProcessEngine(DeadProcess(), DeadConn(), {})
    with pytest.raises(RuntimeError, match=r"exit code -9 \(SIGKILL: likely out of memory"):
        silent.gate(point)


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


def test_a_failed_greedy_gate_records_the_first_divergent_token():
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2,
                        factory=FakeFactory(bad_tokens="candidate"), git_commit="abc")
    failed, = [g for g in doc["gates"] if g["kind"] == "greedy-tokens" and not g["passed"]]
    assert failed["first_divergence"] == {
        "request": "b32-p512-mixed-0", "position": 0, "token": 8, "control_token": 7,
    }
    assert failed["divergent_requests"] == 1


def test_unequal_kv_pools_fail_the_gate_and_name_the_pin():
    """MEM-15: at b128-p8192 the audit arm's engine got a smaller KV pool than its plain twin's (its encoded store and
    scratch shrink what vLLM's profiling leaves), vLLM scheduled the two arms differently once the point overran the
    pool, and the greedy tokens diverged. Equal pools are a precondition of the token gate, so unequal pools must fail
    loudly on their own, before any timing."""

    preset, points, arms = inputs()
    factory = FakeFactory(kv_tokens={"stock-fp8": 494_064, "candidate": 491_376})
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    gate, = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert not gate["passed"]
    assert (gate["tokens"], gate["control_tokens"]) == (491_376, 494_064)
    assert "block_kv_cache_memory_bytes" in gate["detail"]
    assert doc["points"] == []
    assert not any(e[0] == "measure" for e in factory.events)
    validate(doc)


def test_equal_kv_pools_pass_and_are_recorded_in_the_identity():
    preset, points, arms = inputs()
    factory = FakeFactory(kv_tokens={"stock-fp8": 458_752, "candidate": 458_752})
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    gate, = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert gate["passed"] and doc["points"]
    assert {a["id"]: a["kv_cache_tokens"] for a in doc["identity"]["arms"]} == {
        "stock-fp8": 458_752, "candidate": 458_752,
    }


def test_block_mode_pins_one_kv_pool_for_every_arm_of_the_llama_preset():
    settings = get_preset("llama31-8b-fp8-1gpu").engine_settings
    assert settings["block_kv_cache_memory_bytes"] == 56 * 2**30
    assert "block_kv_cache_memory_bytes" not in get_preset("glm53-fp8-8gpu").engine_settings


def test_result_schema_and_identity_strict_judge():
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    validate(doc)
    assert doc["identity"]["arms"][0]["engine_overrides"] == {"quantization": "fp8"}
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
