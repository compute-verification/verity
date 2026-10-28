from __future__ import annotations

import argparse
import calendar
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
from typing import Any

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
from verity_bench.vllm.runner import ResidentCapacityError, _token_digest, run_benchmark
from verity_bench.vllm.table import render


class FakeEngine:
    def __init__(self, arm, events, *, gate_ok=True, bad_tokens=False, kv_tokens=458_752, timed_digest=None,
                 drift=0.0, timed_ok=True):
        self.arm, self.events, self.gate_ok, self.bad_tokens = arm, events, gate_ok, bad_tokens
        self.timed_digest, self.drift, self.timed_ok = timed_digest, drift, timed_ok
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
        timed = any(e[:2] == ("measure", self.arm.id) for e in self.events)
        return {"passed": self.gate_ok and (self.timed_ok or not timed), "provided": True}

    def warmup(self, point):
        self.events.append(("warmup", self.arm.id, point.id))
        return {"wall_s": 0.5}

    def measure(self, point):
        rep = sum(e[:2] == ("measure", self.arm.id) for e in self.events)
        # GPU-wide drift: each timed run, of any arm, is `drift` slower than the one before it.
        slower = 1 + self.drift * sum(e[0] == "measure" for e in self.events)
        self.events.append(("measure", self.arm.id, point.id))
        scale = 1.0 if self.arm.id == "stock-fp8" else 2.0
        return {
            "token_digest": self.timed_digest or _token_digest({f"{point.id}-0": (8 if self.bad_tokens else 7, 9)}),
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
                 drift=0.0, kv_after_gate=None, timed_ok=True):
        self.events = []
        self.resident_failure = resident_failure
        self.failed = False
        self.gate_ok = gate_ok
        self.bad_tokens = bad_tokens
        self.kv_tokens = kv_tokens or {}
        self.timed_digest = timed_digest or {}
        self.drift = drift
        self.kv_after_gate = kv_after_gate or {}
        self.timed_ok = timed_ok

    def create(self, arm, preset, *, resident):
        self.events.append(("create", arm.id, resident))
        if resident and arm.id == self.resident_failure and not self.failed:
            self.failed = True
            raise ResidentCapacityError(f"{arm.id} does not fit")
        gated = any(e[:2] == ("gate", arm.id) for e in self.events)
        kv = self.kv_after_gate.get(arm.id) if gated and arm.id in self.kv_after_gate else self.kv_tokens.get(arm.id, 458_752)
        return FakeEngine(arm, self.events, gate_ok=self.gate_ok,
                          bad_tokens=arm.id == self.bad_tokens,
                          kv_tokens=kv, timed_digest=self.timed_digest.get(arm.id), drift=self.drift,
                          timed_ok=self.timed_ok)


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
    assert set(GPUS) == {"rtx-pro-6000", "h200", "b200"}


def test_every_toml_spec_strictly_loads():
    loaded = []
    for path in sorted(SPEC_ROOT.glob("*.toml")):
        if path.name == "pyproject.toml":   # the specs' own suite
            continue
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


def test_direct_dispatch_runs_each_missing_point_on_the_nodes_gpu_slots_as_the_queued_point(monkeypatch):
    """`run --direct NODE`: `research run --on NODE` through one of the node's GPU slots and its gpu-lease, the slot-only
    launcher's form, with the queued point's tool, point command, identity and environment, so plan and check find it."""

    from research.pods.nebius.slot_only import through_slot

    from verity_bench.tool import VLLM_BENCH_POINT

    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("VERITY_BENCH_VENV", "/workspace/jobs/venv312")
    seen = []
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40,
                          lookup=lambda identity, commit: seen.append(identity) or ("r-passed" if len(seen) == 1 else None))
    assert [point.passing_run for point in plan.points] == ["r-passed", None]
    commands = []
    assert jobs.dispatch(plan, "llama31-8b-fp8-1gpu", root=root, direct="vy-nebius-3",
                         execute=lambda command, cwd: commands.append((list(command), cwd)) or 0) == 0
    [(command, cwd)] = commands
    item = plan.points[1]
    queued = jobs.research_command(plan, "llama31-8b-fp8-1gpu", item, root)
    flags, workload = command[:command.index("--")], command[command.index("--") + 1:]
    assert cwd == root and flags[:4] == ["research", "run", "--on", "vy-nebius-3"]
    assert "--queue" not in flags and "--gpus" not in flags
    assert flags[flags.index("--source") + 1] == str(root) and flags[flags.index("--cwd") + 1] == "source"
    assert flags[flags.index("--timeout") + 1] == "120m" and "--require-result" in flags
    assert flags[flags.index("--tool") + 1] == queued[queued.index("--tool") + 1]
    queued_flags = queued[:queued.index("--")]
    assert [flags[i + 1] for i, t in enumerate(flags) if t == "--env"] == [
        queued_flags[i + 1] for i, t in enumerate(queued_flags) if t == "--env"] == ["VERITY_BENCH_VENV=/workspace/jobs/venv312"]
    assert through_slot(workload)
    assert workload[:4] == ["env", f"CHECK_SLOTS={jobs.GPU_SLOTS}", "python3", "tools/verity/check/slot.py"]
    assert workload[workload.index("--run-max") + 1] == "7200"
    lease = workload[workload.index("--", 4) + 1:]
    assert lease[:3] == ["gpu-lease", "1", "--wait"] and lease[lease.index("--max-min") + 1] == "120"
    assert lease[lease.index("--") + 1:] == queued[queued.index("--") + 1:]
    assert VLLM_BENCH_POINT.parse(command)["identity"] == VLLM_BENCH_POINT.parse(queued)["identity"] == item.identity


def test_direct_dispatch_refuses_a_cpu_only_node_and_a_spec_without_max_minutes():
    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: "r-passed")
    launched = []
    for node in ("vy-nebius-cpu-1", "vy-nebius-cpu-4"):
        with pytest.raises(ValueError, match=f"--direct {node}: not a GPU node with gpu slots"):
            jobs.dispatch(plan, "llama31-8b-fp8-1gpu", root=root, direct=node, execute=lambda c, cwd: launched.append(c) or 0)
    unbounded = replace(plan, spec=replace(plan.spec, admission=Admission(None, None, None)))
    with pytest.raises(ValueError, match="admission max_minutes"):
        jobs.dispatch(unbounded, "llama31-8b-fp8-1gpu", root=root, direct="vy-nebius-2",
                      execute=lambda c, cwd: launched.append(c) or 0)
    assert launched == []
    assert _parser().parse_args(["run", "x", "--direct", "vy-nebius-3"]).direct == "vy-nebius-3"
    assert _parser().parse_args(["run", "x"]).direct is None


@pytest.mark.parametrize("node, gpu", [("vy-dc-h200", "h200"), ("vy-dc-b200", "b200")])
def test_the_served_commits_width_sweep_runs_direct_on_a_dc_vms_one_gpu_as_on_the_rtx(node, gpu):
    """The Commit's width spec on an H200 and a B200 is the RTX PRO 6000's spec (arms, sweep, reps, model, engine
    settings) on another GPU, the B200's on FlashAttention 2 as its Build's row declares, and `--direct` takes the DC VM's
    one GPU through its slot and gpu-lease, the wait for other leases on the node inside the spec's max_minutes."""

    from research.pods.nebius.slot_only import through_slot

    root = Path(__file__).resolve().parents[3]
    name = f"llama31-8b-fp8-1gpu-commit-served-width-{gpu}"
    spec, rtx = resolve_spec(name), resolve_spec("llama31-8b-fp8-1gpu-commit-served-width")
    assert (spec.arms, spec.arm_modules, spec.arm_settings, spec.sweep, spec.reps, spec.eager) == (
        rtx.arms, rtx.arm_modules, rtx.arm_settings, rtx.sweep, rtx.reps, rtx.eager)
    assert (spec.preset.gpu_id, spec.preset.model, spec.preset.revision, spec.preset.precision) == (
        gpu, rtx.preset.model, rtx.preset.revision, rtx.preset.precision)
    attention = {"attention_backend": "FLASH_ATTN", "attention_config": {"flash_attn_version": 2}} if gpu == "b200" else {}
    assert dict(spec.preset.engine_settings) == {**rtx.preset.engine_settings, **attention}
    from verity_bench.vllm.runner import engine_args
    assert {key: value for key, value in engine_args(spec.preset, "fp8", resident=False, eager=False).items()
            if key.startswith("attention")} == attention
    assert not {key for key in engine_args(rtx.preset, "fp8", resident=False, eager=False) if key.startswith("attention")}
    assert [point.id for point in expand_sweep(spec.sweep.id)] == ["b1-p512-mixed", "b8-p512-mixed", "b32-p512-mixed"]

    assert node in jobs.DIRECT_NODES
    plan = jobs.make_plan(name, root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    commands = []
    assert jobs.dispatch(plan, name, root=root, direct=node,
                         execute=lambda command, cwd: commands.append(list(command)) or 0) == 0
    assert len(commands) == 3
    for command in commands:
        flags, workload = command[:command.index("--")], command[command.index("--") + 1:]
        assert flags[:4] == ["research", "run", "--on", node] and flags[flags.index("--timeout") + 1] == "120m"
        assert through_slot(workload)
        lease = workload[workload.index("--", 4) + 1:]
        assert lease[:3] == ["gpu-lease", "1", "--wait"] and "--preemptible" in lease[:lease.index("--")]


def test_the_launcher_finds_a_dc_vms_environment_after_the_others(tmp_path):
    """The DC VMs' pinned-vLLM Python is $WORK/cp/venv312: the launcher takes it when none before it exists."""

    root = Path(__file__).resolve().parents[3]
    launcher = root / "benchmarks/harness/run_point.sh"
    env = {"PATH": "/usr/bin:/bin", "WORK": str(tmp_path), "CUDA_HOME": str(tmp_path)}
    python = tmp_path / "cp" / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$0"\n')
    python.chmod(0o755)
    ran = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root, env=env, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == str(python)


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


def test_the_launcher_finds_a_nodes_jobs_environment_without_verity_bench_venv(tmp_path):
    """A jobs node's pinned-vLLM Python is $WORK/jobs/venv312, not pod_bootstrap.sh's $WORK/venv312: with no
    VERITY_BENCH_VENV the launcher takes the first that exists, and refuses naming every path it tried."""

    root = Path(__file__).resolve().parents[3]
    launcher = root / "benchmarks/harness/run_point.sh"
    env = {"PATH": "/usr/bin:/bin", "WORK": str(tmp_path), "CUDA_HOME": str(tmp_path)}
    missing = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root, env=env, capture_output=True, text=True)
    assert missing.returncode == 2 and f"{tmp_path}/jobs/venv312" in missing.stderr
    python = tmp_path / "jobs" / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$0"\n')
    python.chmod(0o755)
    ran = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root, env=env, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == str(python)
    unset = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root,
                           env={**env, "VERITY_BENCH_VENV": str(tmp_path / "nope")}, capture_output=True, text=True)
    assert unset.returncode == 2 and f"{tmp_path}/nope" in unset.stderr


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


def test_a_spec_warmups_count_loads_strictly_and_binds_the_identity(tmp_path):
    assert resolve_spec("llama31-8b-fp8-1gpu").warmups == 1
    spec = resolve_spec(_spec_with(tmp_path, "warmups = 3"))
    assert spec.warmups == 3
    assert spec.config_sha256 != replace(spec, warmups=1).config_sha256
    with pytest.raises(ValueError, match="warmups must be positive"):
        load_spec(_spec_with(tmp_path, "warmups = 0"))
    capped = resolve_spec(_spec_with(tmp_path, "warmups = 2\nmax_warmups = 8"))
    assert capped.max_warmups == 8 and capped.config_sha256 != replace(capped, max_warmups=None).config_sha256
    with pytest.raises(ValueError, match="max_warmups must be at least warmups"):
        load_spec(_spec_with(tmp_path, "warmups = 3\nmax_warmups = 2"))


def test_an_engine_setting_the_runner_doesnt_read_refuses_by_name(tmp_path):
    from verity_bench.vllm.presets import ENGINE_SETTINGS

    path = tmp_path / "spec.toml"
    path.write_text((SPEC_ROOT / "llama31-8b-fp8-1gpu.toml").read_text().replace(
        "[preset.engine_settings]\n", '[preset.engine_settings]\nmoe_backend = "triton"\n'))
    refusal = f"unknown engine_settings ['moe_backend']; known: {list(ENGINE_SETTINGS)}"
    with pytest.raises(ValueError, match=re.escape(refusal)):
        load_spec(path)


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


def test_a_local_checkpoint_is_bound_by_the_files_a_loader_reads_not_the_hub_snapshots_metadata(tmp_path):
    preset = load_spec("llama31-8b-fp8-1gpu").preset
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    digest = local_checkpoint(preset, checkpoint).model_digest
    (checkpoint / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (checkpoint / "README.md").write_text("# a model card\n")
    assert local_checkpoint(preset, checkpoint).model_digest == digest
    (checkpoint / "generation_config.json").write_text("{}")
    assert local_checkpoint(preset, checkpoint).model_digest != digest


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


def test_the_served_bandwidth_spec_is_the_bandwidth_spec_with_compiled_stock_beside_its_graph_path_at_six_reps():
    from verity_bench.vllm import runner

    graph, served_spec = resolve_spec("llama31-8b-fp8-1gpu-bandwidth"), resolve_spec("llama31-8b-fp8-1gpu-bandwidth-served-r6")
    assert served_spec.arms == (*graph.arms, "stock-fp8@compiled") and served_spec.reps == 6
    assert served_spec.arm_settings["stock-fp8@compiled"].control == "none"
    assert replace(graph.preset, id=served_spec.preset.id) == served_spec.preset
    same = dict(id=served_spec.id, arms=served_spec.arms, arm_settings=served_spec.arm_settings, preset=served_spec.preset,
                reps=served_spec.reps)
    assert replace(graph, **same) == served_spec

    stock, compiled = resolve_arms("stock-fp8,stock-fp8@compiled", (), served_spec.arm_settings)
    assert compiled.control is None and compiled.execution == "compiled"
    built = []
    for arm in (stock, compiled):
        preset, eager = runner.arm_execution(arm, served_spec.preset, served_spec.eager)
        built.append(runner.engine_args(preset, arm.precision, resident=False, eager=eager))
    assert built[0]["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                              "cudagraph_capture_sizes": [1, 8, 32, 128]}
    assert "compilation_config" not in built[1] and built[1]["execution"] == "compiled"
    assert not built[0]["enforce_eager"] and not built[1]["enforce_eager"]
    assert {b["kv_cache_memory_bytes"] for b in built} == {60129542144}


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
    phase_gpus = {"gpu1": 1}
    specs = [load_spec(p) for p in sorted(SPEC_ROOT.glob("*.toml")) if "vllm-spec/v0" in p.read_text().splitlines()[0]]
    assert specs
    for spec in specs:
        kind = kinds[jobs.queue_kind(spec.preset.gpu_count)]
        assert phase_gpus[kind["phase"]] == spec.preset.gpu_count, spec.id
        assert spec.admission.max_minutes is None or spec.admission.max_minutes <= kind["max_wall_min"], spec.id
    with pytest.raises(ValueError, match="register one in infra/cluster/kinds/benchmarks.toml"):
        jobs.queue_kind(8)


def test_plan_run_check_and_compare_are_top_level_commands(tmp_path):
    parser = _parser()
    assert parser.parse_args(["plan", "llama31-8b-fp8-1gpu"]).command == "plan"
    assert parser.parse_args(["run", "llama31-8b-fp8-1gpu"]).command == "run"
    assert parser.parse_args(["check", "llama31-8b-fp8-1gpu"]).command == "check"
    assert parser.parse_args(["compare", str(tmp_path / "a"), str(tmp_path / "b")]).command == "compare"


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
    engine, and that close must not replace the gate's error with a connection error."""

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


@pytest.mark.parametrize("resident_failure", [None, "candidate"])
def test_every_engine_runs_the_warmups_before_its_first_timed_rep_and_the_schedule_keeps_them(resident_failure):
    """A `blocks` engine's first timed rep after one warm-up was slow on the bandwidth arms:
    the count is a spec knob in the identity, each engine runs all of them before it times anything, and each warm-up's
    own summary stays in the schedule."""

    preset, points, arms = inputs()
    factory = FakeFactory(resident_failure=resident_failure)
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=3, warmups=3, factory=factory, git_commit="abc")
    assert doc["identity"]["warmups"] == doc["execution"]["warmups"] == 3
    lifetimes: dict[str, list[list[str]]] = {}
    for event in factory.events:
        if event[0] == "create":
            lifetimes.setdefault(event[1], []).append([])
        elif event[0] in ("warmup", "measure"):
            lifetimes[event[1]][-1].append(event[0])
    timed = [runs for arm in lifetimes.values() for runs in arm if "measure" in runs]
    assert timed and all(runs[:3] == ["warmup"] * 3 and "warmup" not in runs[3:] for runs in timed)
    warm = [s for s in doc["execution"]["schedule"] if s["phase"] == "warmup"]
    assert len(warm) == 3 * len(timed) and [s["warmup"] for s in warm[:3]] == [0, 1, 2]
    assert all(s["run"]["wall_s"] == 0.5 for s in warm)
    with pytest.raises(ValueError, match="warm-up"):
        run_benchmark(preset, arms, points, sweep="quick", reps=1, warmups=0, factory=FakeFactory(), git_commit="abc")
    with pytest.raises(ValueError, match="max_warmups"):
        run_benchmark(preset, arms, points, sweep="quick", reps=1, warmups=3, max_warmups=2, factory=FakeFactory(),
                      git_commit="abc")


def test_an_engine_warms_past_its_minimum_until_two_runs_agree_and_says_when_it_never_did():
    """A run's plain engines warmed in 7.6 s, 2.3 s, 1.26 s and then ran at 1.26 s: a fixed count that
    happens to cover one host's stall is luck, so past `warmups` an engine warms until two consecutive runs agree."""

    from verity_bench.vllm.runner import _warm

    class Stalls:
        def __init__(self, walls):
            self.walls = list(walls)

        def warmup(self, point):
            return {"wall_s": self.walls.pop(0)}

    point = expand_sweep("quick")[1]
    schedule = []
    _warm(Stalls([7.6, 2.27, 1.26, 1.262, 9.9]), point, 2, 8, schedule, {"arm": "plain"})
    assert [s["run"]["wall_s"] for s in schedule] == [7.6, 2.27, 1.26, 1.262]
    assert "settled" not in schedule[-1]
    schedule = []
    _warm(Stalls([1.0, 1.01, 9.9]), point, 2, 8, schedule, {"arm": "stock"})
    assert len(schedule) == 2
    schedule = []
    _warm(Stalls([5.0, 4.0, 3.0, 2.0, 1.0]), point, 1, 3, schedule, {"arm": "never"})
    assert len(schedule) == 3 and schedule[-1]["settled"] is False
    schedule = []
    _warm(Stalls([5.0, 4.0]), point, 2, 2, schedule, {"arm": "fixed"})
    assert len(schedule) == 2 and "settled" not in schedule[-1]


def test_a_timed_rep_keeps_its_step_times_request_times_and_host_counters():
    """A slow rep must say where its time went: which steps were slow, which requests waited, and whether the host was
    busy, ran slower code or waited on the device."""

    from verity_bench.vllm.runner import HostProbe, run_trace

    timing = {
        "steps": [{"s": 0.0123456789, "prefill_tokens": 512, "decode_tokens": 0},
                  {"s": 0.002, "prefill_tokens": 0, "decode_tokens": 8}],
        "requests": {"r-0": {"arrival_s": 0.0, "first_output_s": 0.0141},
                     "r-1": {"arrival_s": 0.0005, "first_output_s": None}},
    }
    assert run_trace(timing, start=12.5, ready=10.0) == {
        "since_ready_s": 2.5,
        "step_s": [0.012346, 0.002],
        "step_prefill_tokens": [512, 0],
        "step_decode_tokens": [0, 8],
        "arrival_s": [0.0, 0.0005],
        "first_output_s": [0.0141, None],
    }
    import gc

    allocations = iter([{"num_device_alloc": 10, "num_alloc_retries": 0}, {"num_device_alloc": 13, "num_alloc_retries": 1}])
    probe = HostProbe(lambda: next(allocations))
    gc.collect()
    host = probe.close()
    assert host.pop("device") == {"num_device_alloc": 3, "num_alloc_retries": 1}
    assert set(host) == {"loadavg_1m", "process_cpu_s", "thread_cpu_s", "involuntary_switches", "voluntary_switches",
                         "minor_faults", "major_faults", "gc_s"}
    assert host["gc_s"] > 0 and all(v >= 0 for v in host.values())
    assert probe._on_gc not in gc.callbacks
    assert "device" not in HostProbe().close()


def test_resident_failure_falls_back_to_blocks():
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


@pytest.mark.parametrize("resident_failure", [None, "candidate"])
def test_an_arm_gate_is_read_again_after_its_timed_runs_and_its_failure_drops_the_points(resident_failure):
    """An arm judges its warm-up and timed runs only once they have run: served_commit's gate names a rootless warm-up
    in its ``warmups`` and fails a measured run without a root, so the verdict must hold what the gate says after them."""

    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    closing = [g for g in doc["gates"] if g["kind"] == "arm-closing"]
    assert all(g["passed"] for g in closing) and doc["points"]
    assert [g["arm"] for g in closing] == ["stock-fp8", "candidate"]

    factory = FakeFactory(timed_ok=False, resident_failure=resident_failure)
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    assert all(g["passed"] for g in doc["gates"] if g["kind"] == "arm")
    closing = [g for g in doc["gates"] if g["kind"] == "arm-closing"]
    assert closing and not any(g["passed"] for g in closing)
    if resident_failure:
        # blocks mode stops opening engines at the first failed gate (its points are dropped either way)
        assert (closing[0]["arm"], closing[0]["cycle"]) == (arms[0].id, 0) and len(closing) == 1
    assert doc["points"] == []
    validate(doc)


def test_a_failed_token_gate_records_the_first_divergent_token():
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2,
                        factory=FakeFactory(bad_tokens="candidate"), git_commit="abc")
    failed, = [g for g in doc["gates"] if g["kind"] == "tokens" and not g["passed"]]
    assert failed["first_divergence"] == {
        "request": "b32-p512-mixed-0", "position": 0, "token": 8, "control_token": 7,
    }
    assert failed["divergent_requests"] == 1


def test_unequal_kv_pools_fail_the_gate_and_name_the_pin():
    """At b128-p8192 the audit arm's engine got a smaller KV pool than its plain twin's (its encoded store and
    scratch shrink what vLLM's profiling leaves), vLLM scheduled the two arms differently once the point overran the
    pool, and the greedy tokens diverged. Equal pools are a precondition of the token gate, so unequal pools must fail
    loudly on their own, before any timing."""

    preset, points, arms = inputs()
    factory = FakeFactory(kv_tokens={"stock-fp8": 494_064, "candidate": 491_376})
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    gate, = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert not gate["passed"]
    assert (gate["tokens"], gate["control_tokens"]) == (491_376, 494_064)
    assert "resident_kv_cache_memory_bytes" in gate["detail"] and "block_kv_cache_memory_bytes" in gate["detail"]
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


def test_blocks_mode_pins_one_kv_pool_for_every_arm_of_the_llama_preset():
    settings = load_spec("llama31-8b-fp8-1gpu").preset.engine_settings
    assert settings["block_kv_cache_memory_bytes"] == 56 * 2**30


def _serving_defaults_preset():
    preset = load_spec("llama32-1b-bf16-rtx-served-arm").preset
    return replace(preset, engine_settings={**preset.engine_settings, "serving_defaults": True})


def test_engine_args_pin_the_pool_in_either_mode_and_serving_defaults_leave_compilation_to_vllm():
    from verity_bench.vllm.runner import engine_args

    served_preset = _serving_defaults_preset()
    for resident in (True, False):
        args = engine_args(served_preset, "bfloat16", resident=resident, eager=False)
        assert args["kv_cache_memory_bytes"] == 4 * 2**30
        assert "compilation_config" not in args and args["enforce_eager"] is False and args["execution"] == "compiled"
    assert engine_args(served_preset, "bfloat16", resident=True, eager=True)["execution"] == "eager"
    pinned = load_spec("llama31-8b-fp8-1gpu").preset
    resident = engine_args(pinned, "fp8", resident=True, eager=False)
    assert "kv_cache_memory_bytes" not in resident and resident["quantization"] == "fp8"
    assert resident["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                              "cudagraph_capture_sizes": [1, 8, 32, 128]}
    assert engine_args(pinned, "fp8", resident=False, eager=False)["kv_cache_memory_bytes"] == 56 * 2**30
    assert "compilation_config" not in engine_args(pinned, "fp8", resident=False, eager=True)


def test_serving_defaults_give_every_arm_process_one_inductor_cache(monkeypatch):
    from verity_bench.vllm import runner

    spawned = []

    class Context:
        def Pipe(self):
            class End:
                def recv(self):
                    return {"ready": {}}

                def close(self):
                    pass
            return End(), End()

        def Process(self, target, args):
            spawned.append(dict(runner.os.environ))
            return SimpleNamespace(start=lambda: None, join=lambda: None)

    monkeypatch.setenv(runner.INDUCTOR_DIR_MARK, "")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/elsewhere")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda prefix: f"/tmp/{prefix}x")
    factory = runner.ProcessEngineFactory()
    factory.context = Context()
    preset = _serving_defaults_preset()
    for arm in resolve_arms("stock"):
        factory.create(arm, preset, resident=True)
        factory.create(arm, preset, resident=True)
    assert [e["TORCHINDUCTOR_CACHE_DIR"] for e in spawned] == ["/tmp/verity_bench_inductor_x"] * 2
    assert all(e[runner.INDUCTOR_DIR_MARK] == "1" for e in spawned)


def test_a_served_variant_times_a_builtin_engine_in_its_own_execution_under_an_eager_spec(monkeypatch):
    """Every protocol row times eager against eager while vLLM serves compiled: `llama31-8b-fp8-1gpu-compiled-full` times
    the eager rows' stock FP8 engine against itself at vLLM's serving defaults, eager and compiled, in one spec."""

    from verity_bench.vllm import runner

    spec = resolve_spec("llama31-8b-fp8-1gpu-compiled-full")
    assert spec.eager and "serving_defaults" not in spec.preset.engine_settings
    base, served_eager, compiled = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution) for a in (base, served_eager, compiled)] == [
        ("stock-fp8", "fp8", None, None),
        ("stock-fp8@eager", "fp8", None, "eager"),
        ("stock-fp8@compiled", "fp8", None, "compiled"),
    ]
    assert compiled.to_json()["execution"] == "compiled"
    assert resolve_arms("stock-fp8,stock-fp8@compiled")[1].control == "stock-fp8"
    for bad in ("stock-fp8@bi-eager", "nope@compiled"):
        with pytest.raises(ValueError, match="BUILTIN@LABEL"):
            resolve_arms(f"stock-fp8,{bad}")

    spawned = []

    class Context:
        def Pipe(self):
            class End:
                def recv(self):
                    return {"ready": {}}

                def close(self):
                    pass
            return End(), End()

        def Process(self, target, args):
            spawned.append(args)
            return SimpleNamespace(start=lambda: None, join=lambda: None)

    monkeypatch.setenv(runner.INDUCTOR_DIR_MARK, "")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda prefix: f"/tmp/{prefix}x")
    factory = runner.ProcessEngineFactory(eager=True)
    factory.context = Context()
    preset, _ = served(spec.preset, spec.eager)
    for arm in (base, served_eager, compiled):
        factory.create(arm, preset, resident=True)
    built = [runner.engine_args(p, a.precision, resident=True, eager=e) for _, a, p, _, e in spawned]
    assert [(b["enforce_eager"], b.get("execution"), "compilation_config" in b) for b in built] == [
        (True, None, False), (True, "eager", False), (False, "compiled", False)]
    assert {b["kv_cache_memory_bytes"] for b in built} == {8 << 30}
    assert {b["max_num_batched_tokens"] for b in built} == {4096}
    assert runner.os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/verity_bench_inductor_x"


def test_a_served_variants_arm_settings_take_only_its_control(tmp_path):
    source = (SPEC_ROOT / "llama31-8b-fp8-1gpu-compiled-full.toml").read_text()
    path = tmp_path / "spec.toml"
    section = '[arm_settings."stock-fp8@compiled"]\ncontrol = "none"'
    assert section in source
    path.write_text(source.replace(section, '[arm_settings."stock-fp8@compiled"]\ncontrol = "stock-fp8@eager"'))
    spec = load_spec(path)
    assert resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)[2].control == "stock-fp8@eager"
    for old, new, why in ((section, '[arm_settings."stock-fp8@compiled"]\nprecision = "bfloat16"', "takes only its control"),
                          (section, '[arm_settings."stock-fp8@compiled"]\ncontrol = "stock"', "control must be another arm"),
                          ('[arm_settings."stock-fp8@compiled"]', '[arm_settings."stock-fp8"]', "must name a protocol arm")):
        assert old in source
        path.write_text(source.replace(old, new))
        with pytest.raises(ValueError, match=why):
            load_spec(path)


def test_a_served_variant_loads_its_own_checkpoint_where_the_presets_is_another_encoding(tmp_path):
    """`llama31-8b-1gpu-pouw-h1-served` times H1 on the int8 pin over stock FP8 as vLLM serves it, which vLLM can't load
    from the pin (`verity_int8`): its served variant names the pin's BF16 base, and only its engine loads it."""

    from verity_bench.vllm import runner

    spec = resolve_spec("llama31-8b-1gpu-pouw-h1-served")
    compiled, int8, h1 = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution, a.model) for a in (compiled, int8, h1)] == [
        ("stock-fp8@compiled", "fp8", None, "compiled", "unsloth/Llama-3.1-8B"),
        ("stock-int8", "int8_exact", None, None, None),
        ("pouw:eps1-sm120-h1fx1", None, "stock-int8", None, None),
    ]
    assert compiled.to_json()["revision"] == "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
    preset, eager = served(spec.preset, spec.eager)
    built = [runner.arm_execution(a, preset, eager) for a in (compiled, int8, h1)]
    assert [(p.model, p.revision, p.case, e) for p, e in built] == [
        ("unsloth/Llama-3.1-8B", "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814", None, False),
        ("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7", "LLAMA31_8B_INT8", True),
        ("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7", "LLAMA31_8B_INT8", True),
    ]
    args = runner.engine_args(built[0][0], compiled.precision, resident=True, eager=False)
    assert (args["quantization"], args["execution"], args["max_num_batched_tokens"]) == ("fp8", "compiled", 64)

    source = (SPEC_ROOT / "llama31-8b-1gpu-pouw-h1-served.toml").read_text()
    path = tmp_path / "spec.toml"
    pin = 'revision = "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"\n'
    for old, new, why in ((pin, "", "set both"),
                          ('control = "stock-int8"\n', f'control = "stock-int8"\nmodel = "x/y"\n{pin}',
                           "only a served variant names its own checkpoint")):
        assert old in source
        path.write_text(source.replace(old, new))
        with pytest.raises(ValueError, match=why):
            load_spec(path)


def test_a_protocol_arm_takes_its_precision_control_and_pouw_settings_from_the_spec(monkeypatch):
    """`pouw:pearl-c-sm120-v1-h3` was refused before any engine started: the harness
    gave every protocol arm the preset's FP8 and no pouw settings. Its spec's `arm_settings` give it the BF16 checkpoint,
    stock FP8 as its control, and the device, kernel and setup-built cubin, with the harness's challenger as its coins,
    which the adapter's refusals accept."""

    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw.coins import CoinCommitment
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.interface import Context

    from verity_bench.vllm.arms import protocols_config

    spec = resolve_spec("llama31-8b-1gpu-pouw-protocol-served")
    control, arm, _ = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert (control.id, control.precision) == ("stock-fp8", "fp8")
    assert (arm.protocols, arm.precision, arm.control) == (("pouw:pearl-c-sm120-v1-h3",), "bfloat16", None)
    assert arm.protocols_config == {"pouw": {"device": "cuda", "kernel": "benchmarks/pouw/pearl_c_sm120",
                                             "cubin": "$PEARLC_CUBIN", "schedule": "deferred"}}
    assert arm.to_json()["protocols_config"] == arm.protocols_config
    settings = spec.arm_settings["pouw:pearl-c-sm120-v1-h3"]
    other = replace(settings, protocols_config={"pouw": {**settings.protocols_config["pouw"], "schedule": "serial"}})
    assert spec.config_sha256 != replace(spec, arm_settings={arm.id: other}).config_sha256
    assert resolve_arms(spec.arms)[1].control == "stock-fp8" and resolve_arms(spec.arms)[1].precision is None

    with pytest.raises(ValueError, match=r"\$PEARLC_CUBIN"):
        protocols_config(arm, {})
    monkeypatch.setattr(verifier, "coins", lambda n: b"\1" * n)
    config = protocols_config(arm, {"PEARLC_CUBIN": "/run/out/setup/pearl_c_sm120.cubin"})
    assert config["pouw"]["cubin"] == "/run/out/setup/pearl_c_sm120.cubin"
    assert isinstance(config["pouw"]["coins"].commitment, CoinCommitment)
    ctx = Context(stage="serve", enabled=arm.protocols, settings=config)
    assert pouw.refusals("pearl-c-sm120-v1-h3", ctx) == ()
    assert "device only" in pouw.refusals("pearl-c-sm120-v1-h3", Context(stage="serve", enabled=arm.protocols))[0]


def test_arm_settings_name_a_protocol_arm_of_the_spec_and_a_control_it_runs(tmp_path):
    source = (SPEC_ROOT / "llama31-8b-1gpu-pouw-protocol-served.toml").read_text()
    path = tmp_path / "spec.toml"
    for control, want in (("stock-fp8", "stock-fp8"), ("none", None)):
        path.write_text(source.replace('control = "none"', f'control = "{control}"', 1))
        spec = load_spec(path)
        assert resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)[1].control == want
    for old, new, why in (('arms = ["stock-fp8", "pouw:pearl-c-sm120-v1-h3", "stock-fp8@compiled"]',
                           'arms = ["stock-fp8", "pouw:pearl-c-h100-v1-h3", "stock-fp8@compiled"]', "must name a protocol arm"),
                          ('[arm_settings."pouw:pearl-c-sm120-v1-h3"]\n', '[arm_settings.stock]\n', "must name a protocol arm"),
                          ('precision = "bfloat16"', 'precision = "int4"', "precision is one of"),
                          ('control = "none"', 'control = "stock"', "control must be another arm"),
                          ('control = "none"', 'control = "pouw:pearl-c-sm120-v1-h3"', "control must be another arm")):
        assert old in source
        path.write_text(source.replace(old, new, 1))
        with pytest.raises(ValueError, match=why):
            load_spec(path)
    with pytest.raises(ValueError, match="not in --arms"):
        resolve_arms("stock-fp8", (), load_spec("llama31-8b-1gpu-pouw-protocol-served").arm_settings)


def test_a_pearl_c_protocol_arms_passes_are_the_adapters_and_the_harness_only_reads_them():
    """The adapter ends a Pearl-C pass at each ``execute_model`` (`pouw.install`); the harness wraps nothing, finds the
    device run through the option's ``end_pass`` and reports the running totals of the passes it ended."""

    from verity_vllm.protocols.pouw import pearl_c as PPD

    from verity_bench.vllm import runner

    class Run(PPD.PearlCDeviceRun):
        def __init__(self):
            self.details = {"passes": []}

        def end_pass(self):
            self.details["passes"].append({"units": 224, "tiles": 5000, "screened_rows": 64, "failing_rows": 0,
                                           "unsure_rows": 1})

    run = Run()
    option = SimpleNamespace(end_pass=run.end_pass, release=lambda: None)
    llm = SimpleNamespace(protocols=SimpleNamespace(get=lambda name: option if name == "pouw" else None))
    assert runner._pearl_c_run(llm) is run
    assert runner._pass_totals(run) == dict.fromkeys(("passes", *runner.PASS_COUNTS), 0)
    run.end_pass()
    run.end_pass()
    assert runner._pass_totals(run) == {"passes": 2, "units": 448, "tiles": 10000, "screened_rows": 128,
                                        "failing_rows": 0, "unsure_rows": 2}
    other = SimpleNamespace(protocols=SimpleNamespace(get=lambda name: SimpleNamespace(release=lambda: None)))
    assert runner._pearl_c_run(other) is None
    assert runner._pearl_c_run(SimpleNamespace(protocols=None)) is None


def test_a_pearl_c_protocol_arm_is_gated_by_the_experimental_arms_call_check(monkeypatch):
    """Pearl-C's tokens carry the epoch's live salt, so no stock arm is a protocol arm's token control; its own gate is
    the experimental arm's call check on the installed run: y against x·Wᵀ, the commitment twice, the negative control.
    A live run's coins name its challenge phase, its salt epoch and that epoch's root over the weights' root."""

    import torch
    from verity.core.protocols.pouw.salt import epoch_open_root

    from verity_bench.vllm import runner

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))

    class Run:
        def __init__(self, y, **commits):
            self.y, self.omit, self.details = y, frozenset(), {"passes": [{}]}
            self.scheme, self.schedule = SimpleNamespace(name="pearl-c-sm120-v1-h3"), SimpleNamespace(name="deferred")
            self.commits = {"weights_root": "ab" * 64, **commits}
            g = torch.Generator().manual_seed(0)
            self.weights = {w: SimpleNamespace(tensor=torch.randn(n, 32, generator=g).to(torch.bfloat16))
                            for w, n in ((0, 64), (1, 64), (2, 128))}

        def begin(self, stepped=False, at=None):
            self.stepped, self.ats = stepped, [*getattr(self, "ats", []), at]

        def epoch_calls(self):
            return 7

        def poison(self):
            pass

        def gemm(self, w, x, W, bias):
            return self.y(x, W)

        def end(self):
            return {"transcript_root": "omitted" if self.omit else "root", "activation_roots": "a", "units": 1}

    def exact_y(x, W):
        return (x.float() @ W.float().T).to(torch.bfloat16)

    run = Run(exact_y)
    exact = runner._pearl_c_gate(run)
    assert exact["passed"] and exact["provided"] and exact["schedule"] == "deferred"
    assert exact["coins"] == {"source": "in-process"}
    assert run.stepped and set(run.ats[:-1]) == {7} and run.ats[-1] is None    # the compared passes share one index
    assert sorted({(c["n"], c["m"]) for c in exact["checks"]}) == [(64, 16), (64, 64), (128, 16), (128, 64)]
    wrong = runner._pearl_c_gate(Run(lambda x, W: torch.zeros(x.shape[0], W.shape[0], dtype=torch.bfloat16)))
    assert not wrong["passed"] and wrong["provided"]
    session = {"session": "s1", "root": "ab" * 64, "commitment": "cd" * 64}
    live = runner._pearl_c_gate(Run(exact_y, coin_source="challenger", session=session, pass_index=0))
    after = epoch_open_root(bytes.fromhex("ab" * 64), 0).hex()
    assert live["passed"] and live["coins"] == {"source": "challenger", "session": session, "pass_index": 0,
                                                "salt_after": after}


def test_pearl_cs_served_row_is_its_protocol_row_beside_the_compiled_engine():
    """`llama31-8b-1gpu-pouw-protocol-served` measures Pearl-C over the compiled engine in one run, so every arm's engine
    shares its step budget, sequence bound and KV pool."""

    from verity_bench.vllm import runner

    spec = resolve_spec("llama31-8b-1gpu-pouw-protocol-served")
    pearl = "pouw:pearl-c-sm120-v1-h3"
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution) for a in arms] == [
        ("stock-fp8", "fp8", None, None),
        (pearl, "bfloat16", None, None),
        ("stock-fp8@compiled", "fp8", None, "compiled"),
    ]
    preset, eager = served(spec.preset, spec.eager)
    for resident in (True, False):
        built = [runner.engine_args(p, a.precision or p.precision, resident=resident, eager=e)
                 for a in arms for p, e in [runner.arm_execution(a, preset, eager)]]
        assert [(b["enforce_eager"], b.get("execution")) for b in built] == [(True, None), (True, None), (False, "compiled")]
        shared = ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "gpu_memory_utilization",
                  "kv_cache_memory_bytes")
        assert len({tuple(b[k] for k in shared) for b in built}) == 1
        assert built[0]["kv_cache_memory_bytes"] == 8 << 30 and built[0]["max_num_batched_tokens"] == 4096

    b32 = resolve_spec("llama31-8b-1gpu-pouw-protocol-served-b32")
    assert (b32.arms, b32.arm_settings, b32.setup, b32.eager, b32.preset) == (
        spec.arms, spec.arm_settings, spec.setup, spec.eager, replace(spec.preset, id=b32.id))
    assert b32.reps == 6 and b32.sweep == replace(spec.sweep, id="b32", batches=(32,))


def test_an_h1_arm_is_gated_against_exact_int8_and_each_run_gets_a_window_the_llm_accepts():
    """`pouw:eps1-sm120-h1fx1` serves the int8_exact arm's own y, so its token control is that engine without PoUW
    (`stock-int8`). The harness hands `verity_vllm.LLM` only H1's public parameters, T the engine's step budget and S
    from the run's step bound, and the LLM fills the model runner and in-process coins after the build; a spec can't
    supply them, nor S."""

    from verity_vllm import protocols as PO
    from verity_vllm.llm import protocol_context, resolve_options
    from verity_vllm.protocols.pouw import h1 as PH

    from verity_bench.vllm.arms import H1, h1_config, protocols_config, window_steps
    from verity_bench.vllm.runner import H1_OPENING_STEPS, engine_args

    spec = resolve_spec("llama31-8b-1gpu-pouw-h1")
    control, arm = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert (control.id, control.precision, control.control) == ("stock-int8", "int8_exact", None)
    assert (arm.id, arm.protocols, arm.precision, arm.control) == (H1, (H1,), None, "stock-int8")
    assert H1 == f"pouw:{PH.SCHEME}"
    preset, eager = served(spec.preset, spec.eager)
    args = engine_args(preset, "int8_exact", resident=True, eager=eager)
    assert (args["quantization"], args["attention_backend"], args["enforce_eager"]) == ("verity_int8", "CUSTOM", True)
    assert args["max_num_batched_tokens"] == spec.arm_settings[H1].protocols_config["pouw"]["rows"] == 64
    assert args["max_num_seqs"] == 64 and engine_args(load_spec("llama31-8b-fp8-1gpu").preset, "fp8", resident=True,
                                                      eager=True)["max_num_seqs"] == 128
    with pytest.raises(ValueError, match="eagerly only"):
        engine_args(preset, "int8_exact", resident=True, eager=False)

    b8, b32 = expand_sweep("quick")
    assert (window_steps(b8, 64), window_steps(b32, 64)) == (130, 294)
    config = protocols_config(arm, {})
    assert config == {"pouw": {"rows": 64, "admission": "per-round"}}
    assert h1_config(config, H1_OPENING_STEPS) == {"pouw": {"rows": 64, "admission": "per-round", "steps": 1}}
    for steps in (H1_OPENING_STEPS, 294):
        ctx = protocol_context(resolve_options({"protocols": H1, "protocols_config": h1_config(config, steps)}))
        assert ctx.settings_of("pouw")["steps"] == steps and ctx.filled == {"pouw": ("runner", "coins", "rounds")}
    with pytest.raises(PO.Refused, match=r"'rows' \(T"):
        protocol_context(resolve_options({"protocols": H1, "protocols_config": config}))
    for named in ({"steps": 10}, {"coins": object()}, {"runner": object()}):
        with pytest.raises(ValueError, match="each window's"):
            h1_config({"pouw": {"rows": 64, **named}}, 294)


def test_each_h1_run_opens_its_own_window_through_the_llm_and_closes_it_after_serving(monkeypatch):
    """Per run, the H1 engine calls `verity_vllm.LLM.reinstall` with the run's S before serving, reads that window
    (`pouw.h1_run`) and closes it with `LLM.close` after serving, even when serving fails; each window's summary is the
    arm's gate, and a timed rep's is its `extras.h1_window`."""

    from verity_vllm.protocols import pouw

    from verity_bench.vllm import runner
    from verity_bench.vllm.arms import H1

    calls: list[tuple[Any, ...]] = []
    window = SimpleNamespace(verdicts={0: {"credit": True}}, refused=[], order_ok=True, steps=130, rows=64, calls=[0],
                             requests=[0], refused_requests=[], coin_source="in-process",
                             details={"steps_served": 106, "static": True, "u": "1/2"}, request_gaps=lambda: [])

    class LLM:
        def reinstall(self, config: dict[str, Any]) -> None:
            calls.append(("reinstall", config))

        def close(self) -> None:
            calls.append(("close",))

    def serve(point: SweepPoint, phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(("serve", phase))
        if phase == "warmup":
            raise RuntimeError("serving failed")
        return {}, {"extras": {}}

    monkeypatch.setattr(pouw, "h1_run", lambda entry: calls.append(("h1_run", entry)) or window)
    engine = object.__new__(runner._VllmEngine)
    engine.llm, engine.h1 = LLM(), {"config": {"pouw": {"rows": 64, "admission": "per-round"}}, "rows": 64,
                                    "windows": []}
    monkeypatch.setattr(engine, "_serve", serve)
    b8 = expand_sweep("quick")[0]
    engine._run(b8, "gate")
    with pytest.raises(RuntimeError, match="serving failed"):
        engine._run(b8, "warmup")
    out = engine._run(b8, "measure")[1]
    opened = ("reinstall", {"pouw": {"rows": 64, "admission": "per-round", "steps": 130}})
    assert calls == [opened, ("h1_run", H1), ("serve", "gate"), ("close",)] + [
        opened, ("h1_run", H1), ("serve", "warmup"), ("close",), opened, ("h1_run", H1), ("serve", "measure"), ("close",)]
    assert [w["phase"] for w in engine.h1["windows"]] == ["gate", "warmup", "measure"]
    assert out["extras"]["h1_window"] is engine.h1["windows"][-1] and out["extras"]["h1_window"]["steps_served"] == 106
    assert engine.custom_gate()["passed"]


def test_an_h1_window_passes_its_gate_only_static_credited_and_entered():
    from verity_bench.vllm.runner import h1_gate, h1_summary

    def run(**over):
        base = dict(verdicts={0: {"credit": True}, 1: {"credit": True}}, refused=[], order_ok=True, steps=130, rows=64,
                    calls=[0, 1], requests=[0], refused_requests=[], coin_source="in-process",
                    details={"steps_served": 120, "static": True, "u": "1/2"}, request_gaps=lambda: [])
        return SimpleNamespace(**{**base, **over})

    good = h1_summary(run())
    assert good["ok"] and good["served_calls"] == good["credit_gate_passed"] == 2 and good["steps_served"] == 120
    bad = [h1_summary(run(refused=[{"step": 130, "linear": 0, "m": 8}])),
           h1_summary(run(details={"static": False})),
           h1_summary(run(verdicts={0: {"credit": False, "reason": "credit: served-output-not-credited-kernel"}})),
           h1_summary(run(request_gaps=lambda: ["step 3: 1 row of a request with no entry"])),
           h1_summary(run(details={"static": True, "unopened": "refused"})),
           h1_summary(run(verdicts={}))]
    assert not any(b["ok"] for b in bad)
    assert bad[2]["credit_gate_refusals"] == ["credit: served-output-not-credited-kernel"]
    assert h1_gate([good, good])["passed"] and not h1_gate([]).get("passed")
    gate = h1_gate([good, bad[0]])
    assert not gate["passed"] and gate["provided"] and gate["failed"] == [bad[0]] and "credits nothing" in gate["coins"]


def test_pearl_cs_whole_step_arm_builds_vllms_full_decode_graphs_beside_the_served_protocol_arm(monkeypatch):
    """`llama31-8b-1gpu-pearl-c-whole-served` times the whole-step arm beside the protocol arm as served (its settings the
    protocol row's) and the compiled engine. Under the eager spec, the arm's overrides build vLLM's FULL decode graphs at
    the preset's capture sizes without torch.compile, at the `compiled` label, which `verity_vllm.LLM` accepts, and its
    run takes the graph path's options and serial schedule."""

    from verity_bench.vllm import runner
    from verity_bench.vllm.arms import plugin
    from verity_vllm.llm import resolve_options

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    spec = resolve_spec("llama31-8b-1gpu-pearl-c-whole-served")
    pearl = "pouw:pearl-c-sm120-v1-h3"
    assert spec.arm_settings[pearl] == resolve_spec("llama31-8b-1gpu-pouw-protocol-served").arm_settings[pearl]
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution, a.experimental) for a in arms] == [
        ("stock-fp8", "fp8", None, None, False), (pearl, "bfloat16", None, None, False),
        ("stock-fp8@compiled", "fp8", None, "compiled", False), ("pouw-pearl-c-whole", "bfloat16", None, None, True)]
    assert arms[-1].engine_label == "graphs-mode0" and arms[-1].to_json()["engine_label"] == "graphs-mode0"
    whole = plugin(arms[-1])
    preset, eager = served(spec.preset, spec.eager)
    preset, eager = runner.arm_execution(arms[-1], preset, eager)
    built = {**runner.engine_args(preset, "bfloat16", resident=False, eager=eager), **whole.engine_overrides(None)}
    assert built["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                           "cudagraph_capture_sizes": list(spec.preset.engine_settings["capture_sizes"])}
    opts = resolve_options(built)
    assert opts["enforce_eager"] is False and opts["execution"] == "compiled"
    assert whole.run_options == {"graphs": True, "whole_defer": True, "whole_defer_a": True, "whole_fork_stats": True,
                                 "lag_ends": True, "whole_tree_sm": True, "aside_ends": True, "whole_prefetch_b": True,
                                 "whole_eager": "deferred-a"} and whole.schedule == "serial"
    monkeypatch.setenv("PEARLC_WHOLE_PREFETCH_B", "0")
    assert "whole_prefetch_b" not in type(whole)().run_options
    monkeypatch.setenv("PEARLC_WHOLE_EAGER", "deferred")
    assert type(whole)().run_options["whole_eager"] == "deferred"
    monkeypatch.setenv("PEARLC_WHOLE_EAGER", "0")
    assert "whole_eager" not in type(whole)().run_options


def test_pearl_cs_whole_step_arm_on_an_h200_runs_pearl_c_sm90s_kernels_under_the_h100_record(monkeypatch):
    """`llama31-8b-1gpu-pearl-c-whole-served-h200` is the whole-served spec's preset on one H200, the whole-step arm beside
    compiled stock, its setup building pearl_c_sm90's cubin; with PEARLC_KERNEL that directory, the arm takes its record
    (`pearl-c-h100-v1-h3`, a scheme the device path runs) and its build.sh cubin, and builds the same engine."""
    from verity_bench.vllm import runner
    from verity_vllm.protocols.pouw import pearl_c as PPD

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root))
    spec, rtx = resolve_spec("llama31-8b-1gpu-pearl-c-whole-served-h200"), resolve_spec("llama31-8b-1gpu-pearl-c-whole-served")
    assert (spec.preset.gpu_id, spec.preset.engine_settings, spec.preset.model) == ("h200", rtx.preset.engine_settings,
                                                                                     rtx.preset.model)
    assert (root / spec.setup).is_file() and "pearl_c_sm90/build.sh" in (root / spec.setup).read_text()
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.execution, a.experimental) for a in arms] == [("stock-fp8@compiled", "compiled", False),
                                                                   ("pouw-pearl-c-whole", None, True)]
    from benchmarks.pouw.pearl_c_vllm import bench_arm
    assert set(bench_arm.KERNEL_SCHEMES.values()) <= set(PPD.SCHEMES)
    assert all((root / "benchmarks/pouw" / d).is_dir() for d in bench_arm.KERNEL_SCHEMES)
    monkeypatch.setenv("PEARLC_KERNEL", str(root / "benchmarks/pouw/pearl_c_sm90"))
    monkeypatch.delenv("PEARLC_SCHEME", raising=False)
    monkeypatch.delenv("PEARLC_CUBIN", raising=False)
    whole = bench_arm.PearlCWholeArm()
    assert whole.scheme == "pearl-c-h100-v1-h3" and whole.kernel_dir.name == "pearl_c_sm90"
    assert whole.kernel_digest(None) is None or (whole.kernel_dir / "pearl_c_sm90.cubin").is_file()
    preset, eager = served(spec.preset, spec.eager)
    preset, eager = runner.arm_execution(arms[-1], preset, eager)
    built = {**runner.engine_args(preset, "bfloat16", resident=False, eager=eager), **whole.engine_overrides(None)}
    assert built["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert whole.run_options == {"graphs": True, "whole_defer": True, "whole_defer_a": True, "whole_fork_stats": True,
                                 "lag_ends": True, "whole_tree_sm": True, "aside_ends": True} and whole.schedule == "serial"


def test_a_sweep_crosses_samplings_and_greedy_points_keep_their_ids():
    served_points = expand_sweep("served")
    assert [(p.batch, p.sampling) for p in served_points] == [
        (b, s) for b in (1, 8, 32, 64) for s in ("greedy", "stochastic")
    ]
    assert [p.id for p in served_points[4:6]] == ["b32-p512-mixed", "b32-p512-mixed-stochastic"]
    assert [p.headline for p in served_points[4:6]] == [True, False]
    assert served_points[5].to_json()["sampling"] == "stochastic"
    assert {p.sampling for p in expand_sweep("quick")} == {"greedy"}


def test_an_unknown_sampling_refuses(tmp_path, monkeypatch):
    from verity_bench.vllm import presets

    (tmp_path / "odd.toml").write_text(
        (SPEC_ROOT / "served.toml").read_text().replace('id = "served"', 'id = "odd"').replace('"stochastic"', '"beam"'))
    monkeypatch.setattr(presets, "SPEC_ROOT", tmp_path)
    with pytest.raises(presets.ConfigError, match="unknown sampling"):
        presets.load_sweep("odd")


def test_inter_token_latency_is_per_token_after_the_first_and_shares_a_multi_token_return():
    from verity_bench.vllm.runner import inter_token_latencies, percentile

    returns = [
        (1.0, {"a": 1}),
        (1.5, {"a": 2, "b": 1}),
        (2.0, {}),
        (2.5, {"a": 4, "b": 2}),
        (3.0, {"a": 4}),
    ]
    assert inter_token_latencies(returns) == [0.5, 0.5, 0.5, 1.0]
    assert inter_token_latencies([(1.0, {"a": 1})]) == []
    assert percentile([0.5, 0.5, 0.5, 1.0], 50) == 0.5
    assert percentile([0.5, 0.5, 0.5, 1.0], 99) == pytest.approx(0.985)
    assert percentile([2.0], 99) == 2.0 and percentile([], 50) is None


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
    """Each timed run is 1% slower than the one before (clocks, heat). Every arm's mean timed
    position is the same, so the means keep the arms' true 2x in either mode, and the paired ratios stay within the
    drift's second-order residue; blocks used to repeat its first cycle's order in the second, so the baseline always
    ran first and the last arm read 2.08x."""

    doc = bench(FakeFactory(drift=0.01, resident_failure=resident_failure), arms=THREE_ARMS, reps=4)
    assert doc["execution"]["interleave"] == ("blocks" if resident_failure else "resident")
    tpot = {a: doc["points"][0]["arms"][a]["metrics"]["time_per_output_token_s"] for a in ("stock-fp8", "a", "b")}
    for arm in ("a", "b"):
        assert tpot[arm]["mean"] / tpot["stock-fp8"]["mean"] == pytest.approx(2)
        assert tpot[arm]["ratio_to_stock"]["mean"] == pytest.approx(2, rel=0.005)


def test_a_blocks_engine_unlike_the_gated_one_fails_engine_identity_and_times_nothing_more():
    """In `blocks` the gate's engine and the timed engines are different lifetimes, and the
    identity recorded the gated one's KV pool, so a timed engine with another pool went unnoticed."""

    factory = FakeFactory(resident_failure="candidate", kv_after_gate={"candidate": 491_376})
    doc = bench(factory)
    [gate] = [g for g in doc["gates"] if g["kind"] == "engine-identity" and not g["passed"]]
    assert gate["arm"] == "candidate" and gate["differs"] == {"kv_cache_tokens": [458_752, 491_376]}
    assert doc["points"] == [] and not any(e[:2] == ("measure", "candidate") for e in factory.events)


def test_decode_time_charges_a_step_to_the_forward_it_waited_on_and_refuses_unseen_device_work():
    """Under async scheduling (output lag 1) a step's host time holds the previous forward's device
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
    """A result's points are recomputed from its reps, so a caller can't drop a slow rep, reorder,
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
    """`--allow precision` used to match every leaf named precision (each arm's as well as the
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
    """A dispatched spec runs each point as its own run, so a sweep's table is assembled from
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
    """An arm that controlled itself would have its greedy gate compare it with itself and always pass."""

    with pytest.raises(ValueError, match="own control"):
        bench(arms=(Arm("stock-fp8", precision="fp8"), Arm("candidate", control="candidate")))


@pytest.mark.parametrize("kv_tokens", [{"candidate": None}, {"stock-fp8": None}, {"stock-fp8": None, "candidate": None}])
def test_an_unreported_kv_pool_fails_the_gate(kv_tokens):
    """The kv-pool gate used to be skipped when either engine didn't report its pool."""

    doc = bench(FakeFactory(kv_tokens=kv_tokens))
    [gate] = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert not gate["passed"] and "unchecked" in gate["detail"] and doc["points"] == []


def test_an_arm_whose_timed_tokens_differ_from_its_controls_keeps_no_points():
    """The greedy gate ran before timing only, so an arm could pass it and compute something else
    in its timed reps."""

    doc = bench(FakeFactory(timed_digest={"candidate": "other"}))
    [gate] = [g for g in doc["gates"] if g["kind"] == "timed-tokens" and g["control"]]
    assert not gate["passed"] and gate["reps"] == [0, 1] and doc["points"] == []
    validate(doc)


def test_an_arm_without_a_control_is_timed_on_its_own_gated_tokens_in_resident_mode():
    """An uncontrolled arm (a Pearl-C protocol arm, whose tokens no stock arm reproduces) must still produce in every
    timed rep the tokens its gate saw from the same engine; blocks mode times other engines, so it has no such gate."""

    doc = bench(FakeFactory(timed_digest={"stock-fp8": "other"}))
    [gate] = [g for g in doc["gates"] if g["kind"] == "timed-tokens" and g["arm"] == "stock-fp8"]
    assert (gate["against"], gate["passed"], gate["reps"]) == ("gate", False, [0, 1]) and doc["points"] == []
    good = bench()
    assert all(g["passed"] for g in good["gates"] if g["kind"] == "timed-tokens") and good["points"]
    blocks = bench(FakeFactory(resident_failure="candidate"))
    assert blocks["execution"]["interleave"] == "blocks"
    assert not [g for g in blocks["gates"] if g["kind"] == "timed-tokens" and g["control"] is None]
    good = bench()
    with pytest.raises(ValueError, match="lacks gates"):
        validate({**good, "gates": [g for g in good["gates"] if g["kind"] != "tokens"]})


def test_intervals_use_the_t_quantile_for_n_and_ratios_refuse_a_zero_or_missing_baseline_rep():
    """From n = 12 the interval used 1.96 instead of t (11 df: 2.201), too narrow; a zero baseline
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


COMMITTED = calendar.timegm((2026, 10, 21, 12, 0, 0))


def _repo(tmp_path):
    """A workspace checkout whose one commit, made at COMMITTED, is the commit the lookup reads its tree from."""

    (repo := tmp_path / "repo").mkdir()
    (repo / "uv.lock").write_text("lock\n")
    (repo / "pyproject.toml").write_text("")
    dates = f"GIT_AUTHOR_DATE='@{COMMITTED} +0000' GIT_COMMITTER_DATE='@{COMMITTED} +0000'"
    subprocess.run(f"git init -q && git add . && {dates} git -c user.name=t -c user.email=t@example.com "
                   "-c commit.gpgsign=false commit -qm points", shell=True, cwd=repo, check=True)
    return repo, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _attempt(run, identity, repo, *, validation="passed", rc=0, dirty=False, tree=None, tool="vllm_bench_point"):
    """An attempt as `research run` records it from `repo` (from another commit's `tree`, or another `tool`)."""

    from research import env
    from research.store import derivation

    source = {**env.source_identity(env.source_conditions(repo)), **({"tree": tree} if tree else {}), "dirty": dirty}
    drv = derivation.derivation_id({"name": tool, "version": "0"}, {}, {"identity": identity}, {}, ["hw.gpu.family"])
    return {"schema": "research/attempt/v0.1", "id": run, "tool": {"name": tool, "version": "0"}, "derivation": drv,
            "reuse_key": derivation.reuse_key(drv, source), "argv": [], "params": {"identity": identity}, "inputs": {},
            "key_params": ["identity"], "conditions": {}, "key_conditions": [], "source": source, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"},
            "cost": None, "status": {"state": "done", "rc": rc, "validation": validation}, "outputs": {}, "telemetry": None}


def test_resume_counts_only_a_clean_exited_passing_run_of_the_same_identity_and_commit(tmp_path):
    """Validation comes from result.json alone, so a run killed after writing it (rc 143,
    preempted) read as passed, and so did a run from a dirty tree, whose code its commit doesn't name."""

    from research.store.local import LocalStore

    (repo, commit), ident = _repo(tmp_path), "f" * 64
    store, config = _stores(tmp_path, [_attempt("r20261021-130000-0001", ident, repo),
                                       _attempt("r20261021-140000-0002", ident, repo),
                                       _attempt("r20261021-150000-0003", ident, repo, rc=143),
                                       _attempt("r20261021-160000-0004", ident, repo, dirty=True),
                                       _attempt("r20261021-170000-0005", ident, repo, validation="failed", rc=3),
                                       _attempt("r20261021-180000-0006", ident, repo, tree="b" * 40),
                                       _attempt("r20261021-190000-0007", "e" * 64, repo),
                                       _attempt("r20261021-200000-0008", ident, repo, tool="something_else")])
    assert jobs.StoreCompletion(LocalStore(store, config_path=config), root=repo)(ident, commit) == "r20261021-140000-0002"

    root = Path(__file__).resolve().parents[3]
    spec = resolve_spec("llama31-8b-fp8-1gpu")
    pins = jobs.toolchain(root, spec, commit=commit)
    point = expand_sweep("quick")[0]
    identity = jobs.point_identity(spec.config_sha256, pins, point)
    assert identity != jobs.point_identity(spec.config_sha256, replace(pins, vllm_pin="0" * 40), point)
    assert identity != jobs.point_identity("0" * 64, pins, point)


def _stores(tmp_path, remote_attempts):
    """A fresh local store whose remote is a directory holding `remote_attempts`, so nothing reaches the network."""

    from research.store import canon

    remote = tmp_path / "remote"
    (remote / "attempts").mkdir(parents=True)
    for doc in remote_attempts:
        (remote / "attempts" / f"{doc['id']}.json").write_bytes(canon.dumps(doc))
    config = tmp_path / "store.toml"
    config.write_text(f'[remote]\ntype = "fs"\npath = "{remote}"\n')
    return tmp_path / "local", config


def test_a_point_another_machine_ran_counts_and_an_absent_point_stays_missing(tmp_path):
    """The lookup once read only this machine's catalog, so a fresh VM saw every point as missing (and `run` would
    have launched them again). It pulls the remote's attempts since the commit, once, and nothing older."""

    from research.store.local import LocalStore

    (repo, commit), found, absent, failed = _repo(tmp_path), "1" * 64, "2" * 64, "3" * 64
    remote_attempts = [
        _attempt("r20261021-130000-0001", found, repo),
        _attempt("r20261021-140000-0002", failed, repo, validation="failed", rc=3),
        _attempt("r20261018-120000-0003", absent, repo),  # before the commit's margin: never fetched
        _attempt("r20261021-150000-0004", absent, repo, tree="b" * 40),
    ]
    root, config = _stores(tmp_path, remote_attempts)
    store = LocalStore(root, config_path=config)
    logged = []
    lookup = jobs.StoreCompletion(store, root=repo, log=logged.append)

    assert jobs.StoreCompletion(LocalStore(root, config_path=config), root=repo, local_only=True)(found, commit) is None
    assert lookup(found, commit) == "r20261021-130000-0001"
    assert lookup(absent, commit) is None
    assert lookup(failed, commit) is None
    assert sorted(store.attempt_ids()) == ["r20261021-130000-0001", "r20261021-140000-0002", "r20261021-150000-0004"]
    assert logged == [f"pulled 3 attempts started since r20261020 from {store.remote.describe()} (0 of 3 were here)"]

    later = _attempt("r20261021-160000-0005", absent, repo)
    (tmp_path / "remote" / "attempts" / f"{later['id']}.json").write_bytes(json.dumps(later).encode())
    assert lookup(absent, commit) is None, "one pull per commit"
    again = jobs.StoreCompletion(store, root=repo, log=logged.append)
    assert again(found, commit) == "r20261021-130000-0001" and len(logged) == 1, "a local hit reads no remote"
    assert again(absent, commit) == "r20261021-160000-0005"
    assert logged[-1].startswith("pulled 1 attempts") and logged[-1].endswith("(3 of 4 were here)")


def test_without_a_remote_the_lookup_refuses_unless_told_to_read_the_catalog_alone(tmp_path):
    from research.store.local import LocalStore

    repo, commit = _repo(tmp_path)
    config = tmp_path / "store.toml"
    config.write_text("")
    store = LocalStore(tmp_path / "local", config_path=config)
    with pytest.raises(RuntimeError, match="--local-only"):
        jobs.StoreCompletion(store, root=repo)("1" * 64, commit)
    assert jobs.StoreCompletion(store, root=repo, local_only=True)("1" * 64, commit) is None


def test_check_on_a_fresh_store_counts_the_points_the_remote_holds(tmp_path, monkeypatch, capsys):
    repo, commit = _repo(tmp_path)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", commit)
    monkeypatch.chdir(repo)
    points = jobs.make_plan("llama31-8b-fp8-1gpu", root=repo, lookup=lambda identity, commit: None).points
    ran = [_attempt(f"r20261021-13000{i}-000{i}", item.identity, repo) for i, item in enumerate(points[1:])]
    store, config = _stores(tmp_path, ran)
    monkeypatch.setenv("RESEARCH_STORE", str(store))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(config))

    assert main(["check", "llama31-8b-fp8-1gpu", "--local-only"]) == 1
    assert f"0/{len(points)} points" in capsys.readouterr().out
    assert main(["check", "llama31-8b-fp8-1gpu"]) == 1
    out = capsys.readouterr().out
    assert f"{len(points) - 1}/{len(points)} points" in out
    assert f"{points[0].point.id}\t{points[0].identity}\tmissing" in out
    assert all(f"{item.identity}\t{run['id']}" in out for item, run in zip(points[1:], ran))


def test_a_requeued_point_deletes_its_earlier_attempts_verdict_before_it_runs(tmp_path, monkeypatch):
    """A preempted point reruns whole; if it reran in its run directory and died before writing a
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
    """The README is the harness's contract: every metric, gate kind, mode, schema and kind it names is the
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
    assert public == set(re.findall(r"^verity-bench ([a-z-]+) ", readme, re.M))


def test_a_stopped_bandwidth_setup_removes_its_partial_build_even_when_its_builds_already_exited(tmp_path):
    """The setup's TERM trap runs under `set -e`: a `kill` of an nvcc that already exited returns nonzero, and without a
    guard the trap stopped there, exiting 1 and leaving its partial build in the shared cache."""

    root = Path(__file__).resolve().parents[3]
    script = (root / "benchmarks/harness/setup/vllm_encoded.sh").read_text()
    [body] = re.findall(r"trap '([^']*)' TERM INT", script)
    partial = tmp_path / "partial"
    partial.mkdir()
    run = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; declare -A VE_PIDS=(); true & VE_PIDS[x]=$!; wait "${{VE_PIDS[x]}}"; '
                       f'VE_TMP={partial}; {body}'],
        capture_output=True, text=True,
    )
    assert run.returncode == 143, run.stderr
    assert not partial.exists()
