from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_config import Admission

from verity_bench import jobs, served_table, steady_spec
from verity_bench.cli import _parser, main
from verity_bench.tool import VLLM_BENCH_POINT
from verity_bench.vllm import presets
from verity_bench.vllm import result as result_format
from verity_bench.vllm.arms import (
    build_bound,
    resolve_arms,
)
from verity_bench.vllm.engine import (
    ProcessEngineFactory,
    arm_execution,
    engine_args,
)
from verity_bench.vllm.presets import (
    ENGINE_SETTINGS,
    FORMER_SPECS,
    GPUS,
    SOURCE_ROOT,
    SPEC_ROOT,
    Sweep,
    expand_sweep,
    load_spec,
    load_sweep,
    resolve_spec,
)
from verity_bench.vllm.runner import (
    METRICS,
    OPTIONAL_METRICS,
    _pearl_c_gate,
)


def test_sweep_expansion_and_staggered_mixed_requests():
    quick, full = expand_sweep("quick"), expand_sweep("full")
    assert [(p.batch, p.prompt_tokens, p.traffic) for p in quick] == [(8, 512, "mixed"), (32, 512, "mixed")]
    assert len(full) == 4 * 2 * 3
    assert {p.prompt_tokens for p in full} == {512, 8192}
    assert {p.traffic for p in full} == {"decode-heavy", "prefill-heavy", "mixed"}
    assert [r.arrive_step for r in quick[0].requests()] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [(p.id, p.to_json()["sampling"]) for p in quick] == [("b8-p512-mixed", "greedy"), ("b32-p512-mixed", "greedy")]


def test_cli_and_presets_use_the_benchmark_package_and_stable_gpu_ids(tmp_path):
    args = _parser().parse_args([
        "vllm", "--preset", "llama31-8b-fp8-1gpu", "--arms", "stock,stock-fp8", "--out", str(tmp_path),
    ])
    assert args.command == "vllm"
    preset = resolve_spec(args.preset).preset
    assert preset.gpu_id == "rtx-pro-6000"
    assert preset.engine_settings["resident_gpu_memory_utilization"] == 0.40
    assert preset.engine_settings["block_gpu_memory_utilization"] == 0.80
    assert set(GPUS) == {"rtx-pro-6000", "h200", "h100", "b200"}


def test_every_toml_spec_strictly_loads():
    loaded = []
    for path in sorted(SPEC_ROOT.glob("*.toml")):
        if path.name == "pyproject.toml":   # the specs' own suite
            continue
        if steady_spec.is_steady(path):
            assert steady_spec.load_spec(path).id == path.stem
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


def _lease_host(monkeypatch, held: bool) -> None:
    monkeypatch.setattr(jobs, "lease_host", lambda: held)
    for name in ("GPU_LEASE_UUID", "GPU_LEASE_ON", "GPU_LEASE_CORES"):
        monkeypatch.delenv(name, raising=False)


def test_run_runs_the_named_point_here_and_a_passed_one_only_again(monkeypatch, capsys):
    """`run SPEC --point P` is the point's own command in this process's run, and no launch of another run: a point a
    passing run stands for runs only when --again names it."""

    root = Path(__file__).resolve().parents[3]
    _lease_host(monkeypatch, False)
    seen = []
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40,
                          lookup=lambda identity, commit: seen.append(identity) or ("r-passed" if len(seen) == 1 else None))
    assert [point.passing_run for point in plan.points] == ["r-passed", None]
    commands = []

    def execute(command, cwd):
        commands.append((list(command), cwd))
        return 0

    assert jobs.dispatch(plan, "llama31-8b-fp8-1gpu", point="b32-p512-mixed", root=root, execute=execute) == 0
    assert commands == [(["env", "bash", "benchmarks/harness/run_point.sh", "--spec",
                          "benchmarks/harness/specs/llama31-8b-fp8-1gpu.toml", "--point", "b32-p512-mixed",
                          "--identity", plan.points[1].identity], root)], "the whole command: a changed token is a changed run"
    assert jobs.dispatch(plan, "llama31-8b-fp8-1gpu", point="b8-p512-mixed", root=root, execute=execute) == 0
    assert len(commands) == 1 and "has its passing run r-passed, so nothing runs" in capsys.readouterr().err
    again = jobs.again(plan, ["b8-p512-mixed"])
    assert jobs.dispatch(again, "llama31-8b-fp8-1gpu", root=root, execute=execute) == 0
    assert commands[-1][0][commands[-1][0].index("--point") + 1] == "b8-p512-mixed"
    for name in ("b8-p512-mixed", "b32-p512-mixed"):
        assert jobs.to_run(plan, name).point.id == name
    with pytest.raises(ValueError, match="has 2 points .* and a run runs one: name it with --point"):
        jobs.dispatch(plan, "llama31-8b-fp8-1gpu", root=root, execute=execute)
    with pytest.raises(ValueError, match="--point b9: not a point of spec"):
        jobs.dispatch(plan, "llama31-8b-fp8-1gpu", point="b9", root=root, execute=execute)
    assert len(commands) == 2


def test_on_a_lease_host_a_point_waits_for_its_cards_in_its_own_lease_and_in_a_leased_job_takes_none(monkeypatch):
    """Where gpu-lease hands out every GPU a run sees none until it takes a lease: the point takes one of the spec's
    GPUs for its max_minutes, preemptible, of GPU_LEASE_ON's cards on GPU_LEASE_CORES' CPUs; a job holding its lease
    (GPU_LEASE_UUID) takes no second one, and a spec with no max_minutes is refused before anything runs."""

    root = Path(__file__).resolve().parents[3]
    _lease_host(monkeypatch, True)
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    item = plan.points[1]
    point = ["bash", "benchmarks/harness/run_point.sh", "--spec", "benchmarks/harness/specs/llama31-8b-fp8-1gpu.toml",
             "--point", item.point.id, "--identity", item.identity]
    assert jobs.point_command(plan, "llama31-8b-fp8-1gpu", item, root) == [
        "gpu-lease", "1", "--wait", "--max-min", "120", "--preemptible", "--", "env", *point]
    monkeypatch.setenv("GPU_LEASE_ON", "4")
    monkeypatch.setenv("GPU_LEASE_CORES", "48-59")
    assert jobs.point_command(plan, "llama31-8b-fp8-1gpu", item, root) == [
        "gpu-lease", "1", "--wait", "--on", "4", "--cores", "48-59", "--max-min", "120", "--preemptible", "--", "env",
        *point]
    monkeypatch.setenv("GPU_LEASE_UUID", "GPU-1")
    assert jobs.point_command(plan, "llama31-8b-fp8-1gpu", item, root) == ["env", *point]
    monkeypatch.delenv("GPU_LEASE_UUID")
    unbounded = replace(plan, spec=replace(plan.spec, admission=Admission(None, None, None)))
    launched = []
    with pytest.raises(ValueError, match="admission max_minutes"):
        jobs.dispatch(unbounded, "llama31-8b-fp8-1gpu", point=item.point.id, root=root,
                      execute=lambda c, cwd: launched.append(c) or 0)
    assert launched == []


def test_run_has_one_launch_and_no_flag_of_the_launches_it_replaced():
    """The one launch is `research run ... -- verity-bench run SPEC`: no flag of a node, a slot, a booked card, cores or
    end, and no queue of the harness's own; a booked period's are `research run`'s (`jobs.booked`)."""

    for flag in ("--direct", "--here", "--card", "--cores", "--end"):
        with pytest.raises(SystemExit):
            _parser().parse_args(["run", "x", flag, "1"])
    assert _parser().parse_args(["run", "x", "--point", "b8"]).point == "b8"
    for gone in ("DIRECT_NODES", "GPU_SLOTS", "DIRECT_WAIT_S", "SLOT_GRACE_S", "slot_run", "direct_command",
                 "direct_minutes", "research_command", "QUEUE_KINDS", "queue_kind", "here_command"):
        assert not hasattr(jobs, gone), gone


def test_the_served_commits_width_sweep_on_a_dc_vms_one_gpu_is_the_rtxs():
    """The Commit's width spec on an H200 is the RTX PRO 6000's spec (arms, sweep, reps, model, engine settings) on
    another GPU, and a run of it runs the point it names."""

    root = Path(__file__).resolve().parents[3]
    name = "llama31-8b-fp8-1gpu-commit-served-width@h200"
    spec, rtx = resolve_spec(name), resolve_spec("llama31-8b-fp8-1gpu-commit-served-width@rtx-pro-6000")
    assert (spec.arms, spec.arm_modules, spec.arm_settings, spec.sweep, spec.reps, spec.eager) == (
        rtx.arms, rtx.arm_modules, rtx.arm_settings, rtx.sweep, rtx.reps, rtx.eager)
    assert (spec.preset.gpu_id, spec.preset.model, spec.preset.revision, spec.preset.precision) == (
        "h200", rtx.preset.model, rtx.preset.revision, rtx.preset.precision)
    assert dict(spec.preset.engine_settings) == dict(rtx.preset.engine_settings)
    assert (rtx.admission.max_minutes, spec.admission.max_minutes) == (40, 120)
    assert [point.id for point in expand_sweep(spec.sweep.id)] == ["b1-p512-mixed", "b8-p512-mixed", "b32-p512-mixed"]
    plan = jobs.make_plan(name, root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    assert [jobs.to_run(plan, point.id).point.id for point in expand_sweep(spec.sweep.id)] == [
        "b1-p512-mixed", "b8-p512-mixed", "b32-p512-mixed"]


def test_the_launcher_runs_the_point_in_the_venv_the_launch_gives_and_refuses_without_one(tmp_path):
    """The point's Python is VERITY_BENCH_VENV, which the launch gives: no node's directory stands in for it."""

    root = Path(__file__).resolve().parents[3]
    launcher = root / "benchmarks/harness/run_point.sh"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "CUDA_HOME": str(tmp_path)}
    python = tmp_path / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$PYTHONPATH"\necho "$@"\n')
    python.chmod(0o755)
    missing = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root, env=env, capture_output=True, text=True)
    assert missing.returncode == 2 and "else the path recorded in" in missing.stderr and "(none of them set)" in missing.stderr
    nope = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root,
                          env={**env, "VERITY_BENCH_VENV": str(tmp_path / "nope")}, capture_output=True, text=True)
    assert nope.returncode == 2 and f"{tmp_path}/nope" in nope.stderr
    ran = subprocess.run(["bash", str(launcher), "--point", "p"], cwd=root,
                         env={**env, "VERITY_BENCH_VENV": str(tmp_path / "venv312")}, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    pythonpath, argv = ran.stdout.splitlines()
    assert {f"{root}/benchmarks/harness", f"{root}/adapters/vllm"} <= set(pythonpath.split(":"))
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
    monkeypatch.setattr(jobs, "lease_host", lambda: False)
    command = jobs.point_command(plan, "llama31-8b-fp8-1gpu", plan.points[0], root)
    forwarded = command[1:command.index("bash")]
    assert command[0] == "env" and forwarded == ["ARM_KNOB=on", "VERITY_BENCH_SETUP=benchmarks/harness/setup/vllm_encoded.sh"]

    environ = dict(item.split("=", 1) for item in forwarded)
    jobs.check_point_environment(spec, environ)
    for wrong in ({**environ, "ARM_KNOB": "off"}, {"ARM_KNOB": "on"}):
        with pytest.raises(ValueError, match="verity-bench run SPEC"):
            jobs.check_point_environment(spec, wrong)
    stock = resolve_spec("llama31-8b-fp8-1gpu")
    jobs.check_point_environment(stock, {})
    with pytest.raises(ValueError, match=r"\['VERITY_BENCH_SETUP'\]"):
        jobs.check_point_environment(stock, {"VERITY_BENCH_SETUP": "benchmarks/harness/setup/vllm_encoded.sh"})


@pytest.mark.parametrize("card", ["NVIDIA H100 80GB HBM3", "NVIDIA H200"])
def test_a_point_refuses_a_card_that_is_not_its_specs_before_anything_is_timed(tmp_path, monkeypatch, capsys, card):
    path = _spec_with(tmp_path, "")
    path.write_text(path.read_text().replace('cards = ["rtx-pro-6000"]', 'cards = ["h200"]'))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    item = jobs.make_plan(path, lookup=lambda identity, commit: None).points[0]
    monkeypatch.setattr(result_format.kit, "host_stamp", lambda: {"card": card})
    monkeypatch.setattr("verity_bench.cli.run_benchmark", lambda *args, **kwargs: sys.exit("timed"))
    with pytest.raises(SystemExit) as stop:
        main(["_vllm-point", "--spec", str(path), "--point", item.point.id, "--identity", item.identity])
    if card == "NVIDIA H200":
        assert stop.value.code == "timed"
    else:
        assert stop.value.code == 2 and (
            f"the card is a {card}, not the h200 of spec 'llama31-8b-fp8-1gpu' (NVIDIA H200): run it on a node whose "
            "card is a NVIDIA H200, or name this card where the spec lists it (`SPEC@CARD`)") in capsys.readouterr().err


def test_the_launcher_sources_the_setup_before_the_point_runs(tmp_path):
    root = Path(__file__).resolve().parents[3]
    setup = tmp_path / "setup.sh"
    python = tmp_path / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$FROM_SETUP $SETUP_SAW_PYTHONPATH"\n')
    python.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "VERITY_BENCH_VENV": str(tmp_path / "venv312"), "CUDA_HOME": str(tmp_path),
           "VERITY_BENCH_SETUP": os.path.relpath(setup, root)}
    setup.write_text('export FROM_SETUP="built in $PWD" SETUP_SAW_PYTHONPATH=${PYTHONPATH##*:}\n')
    ran = subprocess.run(["bash", "benchmarks/harness/run_point.sh"], cwd=root, env=env, capture_output=True, text=True)
    setup.write_text("false\n")
    failed = subprocess.run(["bash", "benchmarks/harness/run_point.sh"], cwd=root, env=env, capture_output=True,
                            text=True)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == f"built in {root} {root}/benchmarks"
    assert failed.returncode != 0 and failed.stdout == ""


def test_the_bandwidth_spec_runs_memory_accountings_arms_with_receipts_on_and_their_libraries_built():
    spec = resolve_spec("llama31-8b-fp8-1gpu-bandwidth-served-reps6@rtx-pro-6000")
    assert spec.arms == ("stock-fp8", "bandwidth-plain", "bandwidth-audit", "stock-fp8@compiled")
    assert spec.arm_modules == ("benchmarks.vllm_encoded.bench_arm:PLAIN", "benchmarks.vllm_encoded.bench_arm:AUDIT")
    assert spec.env["VERITY_VLLM_BENCH_KV_RECEIPTS"] == "on"
    stock = resolve_spec("llama31-8b-fp8-1gpu")
    bound = dict(id=stock.preset.id, model=stock.preset.model, revision=stock.preset.revision, case=stock.preset.case,
                 engine_settings={k: v for k, v in spec.preset.engine_settings.items() if k != "batch_invariant"})
    assert spec.preset.revision.startswith("sha256:") and replace(spec.preset, **bound) == stock.preset
    root = Path(__file__).resolve().parents[3]
    setup = (root / spec.setup).read_text()
    bench_arm = (root / "benchmarks/vllm_encoded/bench_arm.py").read_text()
    for name in re.findall(r'"(VERITY_VLLM_BENCH_\w+_LIB)"', bench_arm):
        assert f"{name}=" in setup, name
    for source in re.findall(r"benchmarks/[\w/]+\.cu", setup):
        assert (root / source).is_file(), source


def test_the_served_bandwidth_spec_times_compiled_stock_uncontrolled_beside_its_graph_path_at_six_reps():
    served_spec = resolve_spec("llama31-8b-fp8-1gpu-bandwidth-served-reps6@rtx-pro-6000")
    assert served_spec.reps == 6 and served_spec.arm_settings["stock-fp8@compiled"].control == "none"

    stock, compiled = resolve_arms("stock-fp8,stock-fp8@compiled", (), served_spec.arm_settings)
    assert compiled.control is None and compiled.execution == "compiled"
    built = []
    for arm in (stock, compiled):
        preset, eager = arm_execution(arm, served_spec.preset, served_spec.eager)
        built.append(engine_args(preset, arm.precision, resident=False, eager=eager))
    assert built[0]["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                              "cudagraph_capture_sizes": [1, 8, 32, 128]}
    assert "compilation_config" not in built[1] and built[1]["execution"] == "compiled"
    assert not built[0]["enforce_eager"] and not built[1]["enforce_eager"]
    assert {b["kv_cache_memory_bytes"] for b in built} == {60129542144}


def test_a_points_run_records_its_identity_as_the_key_param_resume_matches(monkeypatch):
    """The launch `research run ... -- uv run verity-bench run SPEC --point P` records the identity of the point the run
    runs, planned in the launching checkout as the run plans it; so does the point's own command, leased or not."""

    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(jobs, "lease_host", lambda: False)
    monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, lookup=lambda identity, commit: None)
    item = plan.points[1]
    workload = jobs.point_command(plan, "llama31-8b-fp8-1gpu", item, root)
    wrapped = ["/workspace/pouw/infra/bin/gpu-lease", "1", "--wait", "--max-min", "120", "--mem-gb", "64", "--", *workload]
    launch = ["uv", "run", "verity-bench", "run", "llama31-8b-fp8-1gpu", "--point", item.point.id]
    for argv in (workload, wrapped, launch):
        params, key = VLLM_BENCH_POINT.resolve_params(argv, root)
        assert key == {"identity": item.identity}, argv
        assert params["point"] == item.point.id
    assert VLLM_BENCH_POINT.resolve_params(["uv", "run", "verity-bench", "run", "llama31-8b-fp8-1gpu"], root)[1] == {}


def test_a_spec_named_by_a_path_under_the_specs_former_directory_is_the_spec_of_that_name_here(tmp_path):
    """A command written while the specs had a directory of their own names a file there. It loads as the spec of that
    name, a steady-state spec's path resolves the same way, and its points record the path the spec has now."""

    name = "llama31-8b-fp8-1gpu"
    former = SOURCE_ROOT / FORMER_SPECS / f"{name}.toml"
    assert presets.former_name(former) == presets.former_name(f"{FORMER_SPECS}/{name}.toml") == name
    assert presets.former_name(name) is None and presets.former_name(tmp_path / f"{name}.toml") is None
    assert presets.former_name(f"not-{FORMER_SPECS}/{name}.toml") is None
    assert load_spec(former) == load_spec(name) and resolve_spec(former) == resolve_spec(name)
    assert steady_spec.spec_path(former).samefile(SPEC_ROOT / f"{name}.toml")
    assert jobs._relative_spec(former, SOURCE_ROOT) == served_table.spec_path(name)
    assert (SOURCE_ROOT / served_table.spec_path(name)).samefile(SPEC_ROOT / f"{name}.toml")
    with pytest.raises(Exception, match="no benchmark spec"):
        load_spec(SOURCE_ROOT / FORMER_SPECS / "no-such-spec.toml")


def test_every_spec_fits_the_queued_points_kind():
    """A point run through the queue is a job of the kind the README names, `vllm-bench-1gpu`: each spec's GPUs are its
    shape's, and its wall time covers the spec's admission."""

    kinds_toml = Path(__file__).resolve().parents[3] / "infra" / "cluster" / "kinds" / "benchmarks.toml"
    kind = tomllib.loads(kinds_toml.read_text())["kinds"]["vllm-bench-1gpu"]
    specs = [load_spec(p) for p in sorted(SPEC_ROOT.glob("*.toml")) if "vllm-spec/v0" in p.read_text().splitlines()[0]]
    assert specs and kind["phase"] == "gpu1"
    for spec in specs:
        assert spec.preset.gpu_count == 1, spec.id
        assert spec.admission.max_minutes is None or spec.admission.max_minutes <= kind["max_wall_min"], spec.id


def test_plan_run_check_and_compare_are_top_level_commands(tmp_path):
    parser = _parser()
    assert parser.parse_args(["plan", "llama31-8b-fp8-1gpu"]).command == "plan"
    assert parser.parse_args(["run", "llama31-8b-fp8-1gpu"]).command == "run"
    assert parser.parse_args(["check", "llama31-8b-fp8-1gpu"]).command == "check"
    assert parser.parse_args(["compare", str(tmp_path / "a"), str(tmp_path / "b")]).command == "compare"


def test_graphs_are_refused_for_arms_bound_to_the_build():
    bound_arm, _control = resolve_arms("sampled-proofs:hm96,stock-fp8")
    assert build_bound(bound_arm) == ("sampled-proofs:hm96",)
    assert build_bound(resolve_arms("stock")[0]) == ()
    with pytest.raises(ValueError, match="Build"):
        ProcessEngineFactory(eager=False).create(bound_arm, resolve_spec("llama31-8b-fp8-1gpu").preset, resident=True)


def test_a_pearl_c_protocol_arm_is_gated_by_the_experimental_arms_call_check(monkeypatch):
    """Pearl-C's tokens carry the epoch's live salt, so no stock arm is a protocol arm's token control; its own gate is
    the experimental arm's call check on the installed run: y against x·Wᵀ, the commitment twice, the negative control.
    A live run's coins name its challenge phase, its salt epoch and that epoch's root over the weights' root."""

    import torch
    from verity.protocols.resource_accounting.compute.pouw.verifier.salt import epoch_open_root

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))

    class Run:
        def __init__(self, y, **commits):
            self.y, self.omit, self.details = y, frozenset(), {"passes": [{}]}
            self.scheme, self.schedule = SimpleNamespace(name="pearl-c-h100-v1-h3"), SimpleNamespace(name="deferred")
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
    exact = _pearl_c_gate(run)
    assert exact["passed"] and exact["provided"] and exact["schedule"] == "deferred"
    assert exact["coins"] == {"source": "in-process"}
    assert run.stepped and set(run.ats[:-1]) == {7} and run.ats[-1] is None    # the compared passes share one index
    assert sorted({(c["n"], c["m"]) for c in exact["checks"]}) == [(64, 16), (64, 64), (128, 16), (128, 64)]
    wrong = _pearl_c_gate(Run(lambda x, W: torch.zeros(x.shape[0], W.shape[0], dtype=torch.bfloat16)))
    assert not wrong["passed"] and wrong["provided"]
    session = {"session": "s1", "root": "ab" * 64, "commitment": "cd" * 64}
    live = _pearl_c_gate(Run(exact_y, coin_source="challenger", session=session, pass_index=0))
    after = epoch_open_root(bytes.fromhex("ab" * 64), 0).hex()
    assert live["passed"] and live["coins"] == {"source": "challenger", "session": session, "pass_index": 0,
                                                "salt_after": after}


def test_the_readme_states_the_contract_the_code_keeps():
    """The README is the harness's contract: every metric, gate kind, mode, schema and kind it names is the
    code's, and the code has none it doesn't name."""

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
    assert "`vllm-bench-1gpu`" in readme
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
        ["bash", "-c", (f'set -euo pipefail; declare -A VE_PIDS=(); true & VE_PIDS[x]=$!; wait "${{VE_PIDS[x]}}"; '
                        f'VE_TMP={partial}; {body}')],
        capture_output=True, text=True,
    )
    assert run.returncode == 143, run.stderr
    assert not partial.exists()
