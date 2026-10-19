from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verity_config import Admission

from verity_bench import jobs
from verity_bench.cli import _parser, main
from verity_bench.vllm.arms import Arm, resolve_arms
from verity_bench.vllm.presets import (
    GPUS,
    SPEC_ROOT,
    Sweep,
    SweepPoint,
    expand_sweep,
    load_spec,
    load_sweep,
    local_checkpoint,
    resolve_spec,
    served,
)
from verity_bench.vllm.result import IdentityMismatch, identity_mismatches, judge, validate
from verity_bench.vllm.runner import ResidentCapacityError, run_benchmark
from verity_bench.vllm.table import render


class FakeEngine:
    def __init__(self, arm, events, *, gate_ok=True, bad_tokens=False, kv_tokens=458_752, timed_digest=None,
                 drift=0.0):
        self.arm, self.events, self.gate_ok, self.bad_tokens = arm, events, gate_ok, bad_tokens
        self.timed_digest, self.drift = timed_digest, drift
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
        # GPU-wide drift: each timed run, of any arm, is `drift` slower than the one before it.
        slower = 1 + self.drift * sum(e[0] == "measure" for e in self.events)
        self.events.append(("measure", self.arm.id, point.id))
        scale = 1.0 if self.arm.id == "stock-fp8" else 2.0
        return {
            "token_digest": self.timed_digest or ("bad" if self.bad_tokens else "good"),
            "metrics": {
                "time_per_output_token_s": scale * slower if self.drift else scale * (rep + 1),
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
    def __init__(self, *, resident_failure=None, gate_ok=True, bad_tokens=None, kv_tokens=None, timed_digest=None,
                 drift=0.0, kv_after_gate=None):
        self.events = []
        self.resident_failure = resident_failure
        self.failed = False
        self.gate_ok = gate_ok
        self.bad_tokens = bad_tokens
        self.kv_tokens = kv_tokens or {}
        self.timed_digest = timed_digest or {}
        self.drift = drift
        self.kv_after_gate = kv_after_gate or {}

    def create(self, arm, preset, *, resident):
        self.events.append(("create", arm.id, resident))
        if resident and arm.id == self.resident_failure and not self.failed:
            self.failed = True
            raise ResidentCapacityError(f"{arm.id} does not fit")
        gated = any(e[:2] == ("gate", arm.id) for e in self.events)
        kv = self.kv_after_gate.get(arm.id) if gated and arm.id in self.kv_after_gate else self.kv_tokens.get(arm.id, 458_752)
        return FakeEngine(arm, self.events, gate_ok=self.gate_ok,
                          bad_tokens=arm.id == self.bad_tokens,
                          kv_tokens=kv, timed_digest=self.timed_digest.get(arm.id), drift=self.drift)


def inputs():
    preset = load_spec("llama31-8b-fp8-1gpu").preset
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
    preset = load_spec(args.preset).preset
    assert preset.gpu_id == "rtx-pro-6000"
    assert preset.engine_settings["resident_gpu_memory_utilization"] == 0.40
    assert preset.engine_settings["block_gpu_memory_utilization"] == 0.80
    assert set(GPUS) == {"rtx-pro-6000", "h200"}


def test_every_toml_spec_strictly_loads():
    loaded = []
    for path in sorted(SPEC_ROOT.glob("*.toml")):
        first = path.read_text().splitlines()[0]
        loaded.append(load_sweep(path.stem) if "vllm-sweep/v0" in first else load_spec(path))
    sweeps = {d.id for d in loaded if isinstance(d, Sweep)}
    specs = [d for d in loaded if not isinstance(d, Sweep)]
    assert specs and {d.sweep for d in specs} <= sweeps


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
    assert "--preemptible" in command[:command.index("--")]
    assert command[command.index("--gpus") + 1] == "1"
    assert command[command.index("--gpu-model") + 1] == "rtx-pro-6000"
    assert command[command.index("--disk-gb") + 1] == str(jobs.POINT_DISK_GB)
    assert command[command.index("--") + 1:][:2] == ["bash", "benchmarks/harness/run_point.sh"]
    assert (root / "benchmarks/harness/run_point.sh").is_file()
    assert command[command.index("--point") + 1] == "b32-p512-mixed"


def test_dispatch_forwards_the_point_venv_and_the_launcher_runs_the_point_in_it(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    monkeypatch.setenv("VERITY_BENCH_VENV", str(tmp_path / "venv"))
    command = jobs.research_command(plan, "llama31-8b-fp8-1gpu", plan.points[0], root)
    assert f"VERITY_BENCH_VENV={tmp_path / 'venv'}" == command[command.index("--env") + 1]

    launcher = root / "benchmarks/harness/run_point.sh"
    env = {"PATH": "/usr/bin:/bin", "WORK": str(tmp_path), "CUDA_HOME": str(tmp_path)}
    missing = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root, env=env, capture_output=True, text=True)
    assert missing.returncode == 2 and "pod_bootstrap.sh" in missing.stderr
    python = tmp_path / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$PYTHONPATH"\necho "$@"\n')
    python.chmod(0o755)
    ran = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root, env=env, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    pythonpath, argv = ran.stdout.splitlines()
    assert pythonpath.split(":")[:2] == [f"{root}/benchmarks/harness", f"{root}/integrations/vllm"]
    assert argv == "-m verity_bench _vllm-point --point p"


def _spec_with(tmp_path, extra: str) -> Path:
    base = (SPEC_ROOT / "llama31-8b-fp8-1gpu.toml").read_text()
    head, sep, tail = base.partition("\n[preset]\n")
    path = tmp_path / "spec.toml"
    path.write_text(head + "\n" + extra + "\n" + sep + tail)
    return path


def test_a_spec_env_and_setup_load_strictly_and_bind_the_identity(tmp_path):
    spec = resolve_spec(_spec_with(tmp_path, 'setup = "benchmarks/harness/run_point.sh"\n[env]\nARM_KNOB = "on"'))
    assert spec.env == {"ARM_KNOB": "on"} and spec.setup == "benchmarks/harness/run_point.sh"
    assert spec.config_sha256 != replace(spec, env={"ARM_KNOB": "off"}).config_sha256
    assert spec.config_sha256 != replace(spec, setup=None).config_sha256
    for bad in ('[env]\nVERITY_BENCH_VENV = "/x"', '[env]\nCUDA_VISIBLE_DEVICES = "1"', '[env]\nlower = "x"'):
        with pytest.raises(ValueError, match="env names"):
            load_spec(_spec_with(tmp_path, bad))
    with pytest.raises(ValueError, match="env"):
        load_spec(_spec_with(tmp_path, '[env]\nARM_KNOB = 1'))
    for bad in ("/etc/profile", "../outside.sh", "benchmarks/harness/no-such.sh"):
        with pytest.raises(ValueError, match="must name a script in the source tree"):
            load_spec(_spec_with(tmp_path, f'setup = "{bad}"'))


def test_dispatch_forwards_the_spec_env_and_setup_and_the_point_refuses_any_other(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    spec = resolve_spec(_spec_with(tmp_path, 'setup = "benchmarks/harness/setup/vllm_encoded.sh"\n[env]\nARM_KNOB = "on"'))
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    plan = jobs.Plan(spec, spec.config_sha256, jobs.toolchain(root, spec, commit="a" * 40),
                     tuple(jobs.PlannedPoint(p, "f" * 64, None) for p in expand_sweep(spec.sweep.id)))
    command = jobs.research_command(plan, "llama31-8b-fp8-1gpu", plan.points[0], root)
    flags = command[:command.index("--")]
    forwarded = [flags[i + 1] for i, token in enumerate(flags) if token == "--env"]
    assert forwarded == ["ARM_KNOB=on", "VERITY_BENCH_SETUP=benchmarks/harness/setup/vllm_encoded.sh"]

    environ = dict(item.split("=", 1) for item in forwarded)
    jobs.check_point_environment(spec, environ)
    for wrong in ({**environ, "ARM_KNOB": "off"}, {"ARM_KNOB": "on"}):
        with pytest.raises(ValueError, match="verity-bench run SPEC"):
            jobs.check_point_environment(spec, wrong)
    stock = resolve_spec("llama31-8b-fp8-1gpu")
    jobs.check_point_environment(stock, {})
    with pytest.raises(ValueError, match=r"\['VERITY_BENCH_SETUP'\]"):
        jobs.check_point_environment(stock, {"VERITY_BENCH_SETUP": "benchmarks/harness/setup/vllm_encoded.sh"})


def test_a_point_refuses_a_local_checkpoint_whose_files_differ_from_the_spec_digest(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    digest = local_checkpoint(load_spec("llama31-8b-fp8-1gpu").preset, checkpoint).model_digest
    path = _spec_with(tmp_path, "")
    text = path.read_text().replace('model = "unsloth/Llama-3.1-8B"', f'model = "{checkpoint}"\nmodel_digest = "{digest}"')
    path.write_text(re.sub(r'(?m)^(revision|case) = .*\n', "", text))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    item = jobs.make_plan(path, lookup=lambda identity, commit: None).points[0]
    (checkpoint / "config.json").write_text('{"changed": true}')
    with pytest.raises(SystemExit):
        main(["_vllm-point", "--spec", str(path), "--point", item.point.id, "--identity", item.identity])
    assert f"not the spec's model_digest {digest}" in capsys.readouterr().err


def test_the_launcher_sources_the_setup_before_the_point_runs(tmp_path):
    root = Path(__file__).resolve().parents[3]
    setup = tmp_path / "setup.sh"
    python = tmp_path / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$FROM_SETUP $SETUP_SAW_PYTHONPATH"\n')
    python.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "WORK": str(tmp_path), "CUDA_HOME": str(tmp_path),
           "VERITY_BENCH_SETUP": os.path.relpath(setup, root)}
    setup.write_text('export FROM_SETUP="built in $PWD" SETUP_SAW_PYTHONPATH=${PYTHONPATH%%:*}\n')
    ran = subprocess.run(["bash", "benchmarks/harness/run_point.sh"], cwd=root, env=env, capture_output=True, text=True)
    setup.write_text("false\n")
    failed = subprocess.run(["bash", "benchmarks/harness/run_point.sh"], cwd=root, env=env, capture_output=True,
                            text=True)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == f"built in {root} {root}/benchmarks/harness"
    assert failed.returncode != 0 and failed.stdout == ""


def test_the_bandwidth_spec_runs_memory_accountings_arms_with_receipts_on_and_their_libraries_built():
    spec = resolve_spec("llama31-8b-fp8-1gpu-bandwidth")
    assert spec.arms == ("stock-fp8", "bandwidth-plain", "bandwidth-audit")
    assert spec.arm_modules == ("benchmarks.vllm_encoded.bench_arm:PLAIN", "benchmarks.vllm_encoded.bench_arm:AUDIT")
    assert spec.env["VERITY_VLLM_BENCH_KV_RECEIPTS"] == "on"
    stock = resolve_spec("llama31-8b-fp8-1gpu")
    bound = dict(id=stock.preset.id, model=stock.preset.model, revision=stock.preset.revision, case=stock.preset.case,
                 model_digest=None)
    assert spec.preset.model_digest and replace(spec.preset, **bound) == stock.preset
    root = Path(__file__).resolve().parents[3]
    setup = (root / spec.setup).read_text()
    bench_arm = (root / "benchmarks/vllm_encoded/bench_arm.py").read_text()
    for name in re.findall(r'"(VERITY_VLLM_BENCH_\w+_LIB)"', bench_arm):
        assert f"{name}=" in setup, name
    for source in re.findall(r"benchmarks/[\w/]+\.cu", setup):
        assert (root / source).is_file(), source


def test_a_queued_point_records_its_identity_as_the_key_param_resume_matches():
    from verity_bench.tool import VLLM_BENCH_POINT

    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    item = plan.points[0]
    workload = jobs.research_command(plan, "llama31-8b-fp8-1gpu", item, root)
    workload = workload[workload.index("--") + 1:]
    wrapped = ["/workspace/pouw/infra/bin/gpu-lease", "1", "--wait", "--max-min", "120", "--mem-gb", "64", "--", *workload]
    for argv in (workload, wrapped):
        params, key = VLLM_BENCH_POINT.resolve_params(argv, root)
        assert key == {"identity": item.identity}
        assert params["point"] == item.point.id and params["spec"] == "benchmarks/specs/llama31-8b-fp8-1gpu.toml"


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
    preset = load_spec("glm53-fp8-8gpu").preset
    assert preset.revision == "aca966e4e02791568aa6a4ced368624b3d897f42"
    assert "max_num_batched_tokens" not in preset.engine_settings
    graphs_off, eager = served(preset, eager=False)
    assert eager is True and graphs_off.engine_settings["cuda_graphs"] is False
    assert load_spec("llama31-8b-fp8-1gpu").preset.engine_settings["max_num_batched_tokens"] == 16_384


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
        ProcessEngineFactory(eager=False).create(bound_arm, load_spec("llama31-8b-fp8-1gpu").preset, resident=True)


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


def test_closing_an_engine_whose_worker_already_sent_its_error_keeps_that_error():
    """A gate that fails in the worker (an arm's refusal) ends the worker's loop; the runner's cleanup then closes the
    engine, and that close must not replace the gate's error with a connection error (r20261018-205649-abfc)."""

    from verity_bench.vllm.runner import _ProcessEngine

    class ExitingConn:
        def send(self, _):
            pass

        def recv(self):
            raise EOFError

        def close(self):
            pass

    class ExitingProcess:
        def __init__(self, code):
            self.exitcode = code

        def is_alive(self):
            return True

        def join(self):
            pass

    _ProcessEngine(ExitingProcess(0), ExitingConn(), {}).close()
    with pytest.raises(RuntimeError, match="exited with code 1 while closing"):
        _ProcessEngine(ExitingProcess(1), ExitingConn(), {}).close()


def test_local_checkpoint_is_content_bound(tmp_path):
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    preset = local_checkpoint(load_spec("llama31-8b-fp8-1gpu").preset, tmp_path)
    assert preset.model == str(tmp_path)
    assert preset.revision is preset.case is None
    assert len(preset.model_digest) == 64
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    assert local_checkpoint(load_spec("llama31-8b-fp8-1gpu").preset, tmp_path).model_digest != preset.model_digest


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
    assert measured == ["stock-fp8", "stock-fp8", "candidate", "candidate", "candidate", "stock-fp8"]
    assert all(doc["points"][0]["arms"][a.id]["metrics"]["time_per_output_token_s"]["n"] == 3 for a in arms)


def test_an_engine_that_fails_to_start_closes_the_engines_already_up():
    preset, points, arms = inputs()
    factory = FakeFactory()
    create = factory.create

    def refuse(arm, preset, *, resident):
        if arm.id == "candidate":
            raise RuntimeError("Refused: protocol options refused")
        return create(arm, preset, resident=resident)

    factory.create = refuse
    with pytest.raises(RuntimeError, match="Refused"):
        run_benchmark(preset, arms, points, sweep="quick", reps=1, factory=factory, git_commit="abc")
    assert factory.events == [("create", "stock-fp8", True), ("close", "stock-fp8")]


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
    settings = load_spec("llama31-8b-fp8-1gpu").preset.engine_settings
    assert settings["block_kv_cache_memory_bytes"] == 56 * 2**30
    assert "block_kv_cache_memory_bytes" not in load_spec("glm53-fp8-8gpu").preset.engine_settings


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


def test_decode_time_per_token_counts_only_decode_only_forwards():
    from verity_bench.vllm.runner import decode_time_per_token

    steps = [
        {"forward": True, "s": 0.9, "prefill_tokens": 512, "decode_tokens": 0},
        {"forward": True, "s": 0.5, "prefill_tokens": 64, "decode_tokens": 7},
        {"forward": True, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": True, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": False, "s": 0.01, "prefill_tokens": 0, "decode_tokens": 0},
    ]
    assert decode_time_per_token(steps) == pytest.approx(0.4 / 16)
    assert decode_time_per_token([{**s, "prefill_tokens": None, "decode_tokens": None} for s in steps]) is None
    assert decode_time_per_token([]) is None
    # 23 decode tokens across all steps (7 + 8 + 8); a run that generated fewer had prompt tokens counted as decode
    assert decode_time_per_token(steps, generated=23) == pytest.approx(0.4 / 16)
    assert decode_time_per_token(steps, generated=22) is None


def test_the_decode_metric_is_reported_only_where_every_rep_has_it():
    from verity_bench.vllm.runner import _aggregate

    arms = (Arm("stock"), Arm("candidate", control="stock"))
    point = expand_sweep("quick")[0]

    def rep(i, tpot, decode):
        metrics = {"time_per_output_token_s": tpot, "time_to_first_token_s": 0.1, "tokens_per_s": 1 / tpot,
                   "peak_gpu_memory_bytes": 1.0}
        return {"rep": i, "metrics": metrics if decode is None else {**metrics, "decode_time_per_token_s": decode}}

    full = {("stock", point.id): [rep(0, 1.0, 0.5), rep(1, 1.0, 0.5)],
            ("candidate", point.id): [rep(0, 2.0, 1.5), rep(1, 2.0, 1.5)]}
    [row] = _aggregate(arms, (point,), "stock", full)
    assert row["arms"]["candidate"]["metrics"]["decode_time_per_token_s"]["ratio_to_stock"]["mean"] == 3
    doc = {"baseline": "stock", "identity": {"arms": [{"id": "stock"}, {"id": "candidate"}]}, "points": [row]}
    assert "decode time / token" in render(doc)

    partial = {**full, ("candidate", point.id): [rep(0, 2.0, 1.5), rep(1, 2.0, None)]}
    [row] = _aggregate(arms, (point,), "stock", partial)
    assert "decode_time_per_token_s" not in row["arms"]["candidate"]["metrics"]
    doc["points"] = [row]
    assert "decode time / token" not in render(doc)


THREE_ARMS = (Arm("stock-fp8", precision="fp8"), Arm("a", control="stock-fp8"), Arm("b", control="stock-fp8"))


def bench(factory=None, *, arms=None, reps=2, points=None):
    preset, default_points, default_arms = inputs()
    return run_benchmark(preset, arms or default_arms, points or default_points, sweep="quick", reps=reps,
                         factory=factory or FakeFactory(), git_commit="abc")


@pytest.mark.parametrize("resident_failure", [None, "a"])
def test_drift_over_the_run_cancels_in_every_arms_mean_over_even_reps(resident_failure):
    """Red team, attack 1: each timed run is 1% slower than the one before (clocks, heat). Every arm's mean timed
    position is the same, so the means keep the arms' true 2x in either mode, and the paired ratios stay within the
    drift's second-order residue; blocks used to repeat its first cycle's order in the second, so the baseline always
    ran first and the last arm read 2.08x."""

    doc = bench(FakeFactory(drift=0.01, resident_failure=resident_failure), arms=THREE_ARMS, reps=4)
    assert doc["execution"]["interleave"] == ("blocks" if resident_failure else "resident")
    tpot = {a: doc["points"][0]["arms"][a]["metrics"]["time_per_output_token_s"] for a in ("stock-fp8", "a", "b")}
    for arm in ("a", "b"):
        assert tpot[arm]["mean"] / tpot["stock-fp8"]["mean"] == pytest.approx(2)
        assert tpot[arm]["ratio_to_stock"]["mean"] == pytest.approx(2, rel=0.005)


def test_a_block_engine_unlike_the_gated_one_fails_engine_identity_and_times_nothing_more():
    """Red team, attacks 1 and 3: in blocks the gate's engine and the timed engines are different lifetimes, and the
    identity recorded the gated one's KV pool, so a timed engine with another pool went unnoticed."""

    factory = FakeFactory(resident_failure="candidate", kv_after_gate={"candidate": 491_376})
    doc = bench(factory)
    [gate] = [g for g in doc["gates"] if g["kind"] == "engine-identity" and not g["passed"]]
    assert gate["arm"] == "candidate" and gate["differs"] == {"kv_cache_tokens": [458_752, 491_376]}
    assert doc["points"] == [] and not any(e[:2] == ("measure", "candidate") for e in factory.events)


def test_decode_time_charges_a_step_to_the_forward_it_waited_on_and_refuses_unseen_device_work():
    """Red team, attack 2: under async scheduling (output lag 1) a step's host time holds the previous forward's device
    time, so a decode-only step after a prefill carried the prefill: 1.1 s over 16 tokens instead of 0.4 s. Device
    work still running after the last step means the host steps missed it, and the metric is dropped."""

    from verity_bench.vllm.runner import decode_time_per_token

    steps = [
        {"forward": True, "s": 0.05, "prefill_tokens": 512, "decode_tokens": 0},
        {"forward": True, "s": 0.9, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": True, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": False, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 0},
    ]
    assert decode_time_per_token(steps) == pytest.approx(1.1 / 16)
    assert decode_time_per_token(steps, output_lag=1) == pytest.approx(0.4 / 16)
    assert decode_time_per_token(steps, output_lag=1, device_tail_s=0.001) == pytest.approx(0.4 / 16)
    assert decode_time_per_token(steps, output_lag=1, device_tail_s=0.01) is None


def test_validate_refuses_dropped_reordered_or_edited_reps():
    """Red team, attack 3: a result's points are recomputed from its reps, so a caller can't drop a slow rep, reorder,
    edit a mean or change `reps` and still have `judge` accept it."""

    doc = bench(reps=3)
    validate(doc)
    row = doc["points"][0]["arms"]["candidate"]
    reps = row["reps"]
    for forged in (
        {**row, "reps": reps[:2]},
        {**row, "reps": [reps[1], reps[0], reps[2]]},
        {**row, "metrics": {**row["metrics"], "tokens_per_s": {**row["metrics"]["tokens_per_s"], "mean": 99.0}}},
    ):
        bad = {**doc, "points": [{**doc["points"][0], "arms": {**doc["points"][0]["arms"], "candidate": forged}}]}
        with pytest.raises(ValueError):
            validate(bad)
    with pytest.raises(ValueError, match="reps"):
        validate({**doc, "execution": {**doc["execution"], "reps": 2}})
    with pytest.raises(IdentityMismatch, match="reps"):
        judge(doc, bench(reps=2))
    with pytest.raises(IdentityMismatch, match="interleave"):
        judge(doc, bench(FakeFactory(resident_failure="candidate"), reps=3))


def test_an_allow_hides_only_the_paths_it_names_and_judge_lists_what_it_hid():
    """Red team, attack 3: `--allow precision` used to match every leaf named precision (each arm's as well as the
    preset's), and `--allow id` every arm id."""

    doc = bench()
    arms = [dict(a) for a in doc["identity"]["arms"]]
    arms[1].update(id="other", precision="int4", kernel_digest="digest-x")
    other = {**doc, "identity": {**doc["identity"], "arms": arms}}
    for allow in (("id",), ("precision",), ("kernel_digest",)):
        assert identity_mismatches(doc, other, allow)
    digest_only = {**doc, "identity": {**doc["identity"], "arms": [arms[0], {**doc["identity"]["arms"][1],
                                                                              "kernel_digest": "digest-x"}]}}
    verdict = judge(doc, digest_only, ("arms[*].kernel_digest",))
    assert verdict["differences"] == {"arms[1].kernel_digest": ["digest-candidate", "digest-x"]}


def test_table_merges_one_specs_point_results_only_when_their_identities_agree(tmp_path):
    """Red team, attack 3: a dispatched spec runs each point as its own run, so a sweep's table is assembled from
    several results; `verity-bench table` refuses results whose identities differ anywhere but the point's own."""

    from verity_bench.cli import main
    from verity_bench.vllm.result import merge

    preset, (p32,), arms = inputs()
    p8 = replace(p32, batch=8)

    def point(p, identity, **kw):
        return run_benchmark(preset, arms, (p,), sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc",
                             config_identity={"point_identity": identity, **kw})

    merged = merge([point(p8, "i8"), point(p32, "i32")])
    assert [p["point"]["id"] for p in merged["points"]] == ["b8-p512-mixed", "b32-p512-mixed"]
    assert merged["identity"]["config"]["point_identity"] == ["i8", "i32"]
    with pytest.raises(IdentityMismatch, match="uv_lock_sha256"):
        merge([point(p8, "i8", uv_lock_sha256="a"), point(p32, "i32", uv_lock_sha256="b")])
    with pytest.raises(ValueError, match="two results"):
        merge([point(p32, "i32"), point(p32, "i32")])
    files = []
    for i, doc in enumerate((point(p8, "i8"), point(p32, "i32"))):
        files.append(tmp_path / f"{i}.json")
        files[-1].write_text(json.dumps(doc))
    assert main(["table", *map(str, files), "--out", str(tmp_path / "merged.json")]) == 0
    validate(json.loads((tmp_path / "merged.json").read_text()))


def test_an_arm_cannot_be_its_own_control():
    """Red team, attack 4: its greedy gate would compare it with itself and always pass."""

    with pytest.raises(ValueError, match="own control"):
        bench(arms=(Arm("stock-fp8", precision="fp8"), Arm("candidate", control="candidate")))


@pytest.mark.parametrize("kv_tokens", [{"candidate": None}, {"stock-fp8": None}, {"stock-fp8": None, "candidate": None}])
def test_an_unreported_kv_pool_fails_the_gate(kv_tokens):
    """Red team, attack 4: the kv-pool gate used to be skipped when either engine didn't report its pool."""

    doc = bench(FakeFactory(kv_tokens=kv_tokens))
    [gate] = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert not gate["passed"] and "unchecked" in gate["detail"] and doc["points"] == []


def test_an_arm_whose_timed_tokens_differ_from_its_controls_keeps_no_points():
    """Red team, attack 4: the greedy gate ran before timing only, so an arm could pass it and compute something else
    in its timed reps."""

    doc = bench(FakeFactory(timed_digest={"candidate": "other"}))
    [gate] = [g for g in doc["gates"] if g["kind"] == "timed-tokens"]
    assert not gate["passed"] and gate["reps"] == [0, 1] and doc["points"] == []
    validate(doc)
    good = bench()
    with pytest.raises(ValueError, match="lacks gates"):
        validate({**good, "gates": [g for g in good["gates"] if g["kind"] != "greedy-tokens"]})


def test_intervals_use_the_t_quantile_for_n_and_ratios_refuse_a_zero_or_missing_baseline_rep():
    """Red team, attack 5: from n = 12 the interval used 1.96 instead of t (11 df: 2.201), too narrow; a zero baseline
    rep was dropped from the ratio, which then averaged the other reps; n = 1 rendered like a measured interval."""

    from verity_bench.vllm.runner import _aggregate, _summary

    assert _summary([1.0])["n"] == 1 and "ci95" not in _summary([1.0])
    assert _summary([1.0, 3.0])["ci95"] == pytest.approx([2 - 12.706, 2 + 12.706])
    assert _summary([1.0, 2.0, 3.0])["ci95"] == pytest.approx([2 - 4.303 / 3**0.5, 2 + 4.303 / 3**0.5])
    values = [float(i % 2) for i in range(12)]
    half = 2.201 * statistics.stdev(values) / 12**0.5
    assert _summary(values)["ci95"] == pytest.approx([0.5 - half, 0.5 + half])

    arms = (Arm("stock"), Arm("candidate", control="stock"))
    point = expand_sweep("quick")[0]

    def rep(i, tpot):
        return {"rep": i, "metrics": {"time_per_output_token_s": tpot, "time_to_first_token_s": 0.1,
                                      "tokens_per_s": 1.0, "peak_gpu_memory_bytes": 1.0}}

    samples = {("stock", point.id): [rep(0, 0.0), rep(1, 1.0)], ("candidate", point.id): [rep(0, 2.0), rep(1, 9.0)]}
    [row] = _aggregate(arms, (point,), "stock", samples)
    assert row["arms"]["candidate"]["metrics"]["time_per_output_token_s"]["ratio_to_stock"] == {"mean": None, "n": 0}
    doc = {"baseline": "stock", "identity": {"arms": [{"id": "stock"}, {"id": "candidate", "control": "stock"}]},
           "points": [row]}
    assert "| time / output token | — | — |" in render(doc)
    with pytest.raises(ValueError, match="same reps"):
        _aggregate(arms, (point,), "stock", {**samples, ("stock", point.id): [rep(0, 1.0)]})
    assert "2.000× (n=1)" in render(bench(reps=1))


def test_resume_counts_only_a_clean_exited_passing_run_of_the_same_identity_and_commit(tmp_path):
    """Red team, attacks 6 and 7: validation comes from result.json alone, so a run killed after writing it (rc 143,
    preempted) read as passed, and so did a run from a dirty tree, whose code its commit doesn't name."""

    import sqlite3

    from research.store.index import SCHEMA

    db = sqlite3.connect(tmp_path / "index.sqlite")
    db.executescript(SCHEMA)
    ident, commit = "f" * 64, "a" * 40
    rows = [("passed-clean", "vllm_bench_point", "passed", 0, commit, 0, ident, "2026-10-18T01"),
            ("preempted", "vllm_bench_point", "passed", 143, commit, 0, ident, "2026-10-18T02"),
            ("dirty", "vllm_bench_point", "passed", 0, commit, 1, ident, "2026-10-18T03"),
            ("failed", "vllm_bench_point", "failed", 3, commit, 0, ident, "2026-10-18T04"),
            ("other-commit", "vllm_bench_point", "passed", 0, "b" * 40, 0, ident, "2026-10-18T05"),
            ("other-point", "vllm_bench_point", "passed", 0, commit, 0, "e" * 64, "2026-10-18T06"),
            ("other-tool", "something_else", "passed", 0, commit, 0, ident, "2026-10-18T07")]
    for rid, tool, validation, rc, src, dirty, key, start in rows:
        db.execute("INSERT INTO attempts (id, tool, validation, rc, source_commit, source_dirty, start_utc) "
                   "VALUES (?,?,?,?,?,?,?)", (rid, tool, validation, rc, src, dirty, start))
        db.execute("INSERT INTO attempt_params VALUES (?, 'identity', ?)", (rid, key))
    assert db.execute(jobs.PASS_QUERY.format(identity=ident, commit=commit)).fetchall() == [("passed-clean",)]

    root = Path(__file__).resolve().parents[3]
    spec = resolve_spec("llama31-8b-fp8-1gpu")
    pins = jobs.toolchain(root, spec, commit=commit)
    point = expand_sweep("quick")[0]
    identity = jobs.point_identity(spec.config_sha256, pins, point)
    assert identity != jobs.point_identity(spec.config_sha256, replace(pins, vllm_pin="0" * 40), point)
    assert identity != jobs.point_identity("0" * 64, pins, point)


def test_a_requeued_point_deletes_its_earlier_attempts_verdict_before_it_runs(tmp_path, monkeypatch):
    """Red team, attack 7: a preempted point reruns whole; if it reran in its run directory and died before writing a
    new result.json, the earlier attempt's verdict would stand beside the new attempt's half-written files."""

    import verity_bench.cli as cli

    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    item = plan.points[0]
    (tmp_path / "result.json").write_text('{"validation": {"status": "passed"}}')
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.chdir(root)

    def preempted(*args, **kwargs):
        raise RuntimeError("preempted")

    monkeypatch.setattr(cli, "run_benchmark", preempted)
    with pytest.raises(RuntimeError, match="preempted"):
        cli.main(["_vllm-point", "--spec", "llama31-8b-fp8-1gpu", "--point", item.point.id, "--identity", item.identity])
    assert not (tmp_path / "result.json").exists()


def test_the_readme_states_the_contract_the_code_keeps():
    """The README is the harness's contract (BENCH-3): every metric, gate kind, mode, schema and kind it names is the
    code's, and the code has none it doesn't name."""

    from verity_bench.vllm import result as result_format
    from verity_bench.vllm.runner import METRICS, OPTIONAL_METRICS

    root = Path(__file__).resolve().parents[3]
    readme = (root / "benchmarks/harness/README.md").read_text()
    runner_source = (root / "benchmarks/harness/verity_bench/vllm/runner.py").read_text()
    for metric in METRICS + OPTIONAL_METRICS:
        assert f"`{metric}`" in readme, metric
    for kind in re.findall(r'"kind": "([a-z-]+)"', runner_source):
        assert f"`{kind}`" in readme, kind
    for mode in ("resident", "blocks"):
        assert f"`{mode}`" in readme and f'"{mode}"' in runner_source
    assert f"`{result_format.SCHEMA}`" in readme
    for kind in jobs.QUEUE_KINDS.values():
        assert f"`{kind}`" in readme, kind
    [subparsers] = [a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction)]
    public = {name for name in subparsers.choices if not name.startswith("_")}
    assert public == set(re.findall(r"^verity-bench ([a-z]+) ", readme, re.M))
