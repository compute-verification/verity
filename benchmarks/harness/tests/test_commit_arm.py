"""The Commit arms (`verity_bench.vllm.commit_arm`) against a fake committer: their runs open and close one Commit each, the gate
and warm-up runs are learn-only, a measured run without a root or with a collector or source error fails the point, the custom
gate reads the runs, and each arm commits the deployed scheme or refuses. ``commit@compiled`` builds its engine compiled. Both
give the committer an engine of the LLM's public names, a local checkpoint's included. No engine."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from verity_bench.vllm.arms import Target, resolve_arms
from verity_bench.vllm.commit_arm import COMMIT, COMMIT_COMPILED, CommitArm, CompiledCommitArm
from verity_bench.vllm.presets import resolve_spec


class FakeCommitter:
    name = "native_collect_v2_auto_nw+rows-sha512+retain_none"
    commit_scheme = "rows-sha512"

    def __init__(self, *, root=b"\1" * 32, learned=(), errors=(), committed=None, retain_levels=True):
        self.root, self.learned, self.errs, self.committed = root, list(learned), list(errors), committed
        self.events, self.learn_only, self.retain_levels = [], False, retain_levels

    def reset(self):
        self.events.append("reset")

    def install(self):
        self.events.append(("install", self.learn_only))

    def uninstall(self):
        self.events.append("uninstall")

    def finalize(self):
        self.events.append("finalize")
        learned = self.learned if self.learn_only else []
        return SimpleNamespace(
            run_root=b"" if learned else self.root, steps=[0, 1, 2], root_ready_delay_s=0.01, t_finalize_s=0.02,
            extra={"errors": self.errs, "expected_tensors": 10, "committed_tensors": 10 if self.committed is None else self.committed,
                   "values_retained_bytes": 4096, "bounded_staging": {"learned_only_steps": learned, "plan_mismatches": 0}})


def _arm(committer):
    arm = CommitArm("commit")
    arm.committer = committer
    return arm, Target(resolve_arms("stock-fp8")[0], None, None, None)


def _run(arm, target, phase):
    target.phase, target.point = phase, "b8-p512-mixed"
    arm.begin_run(target)
    arm.end_run(target)


def test_the_spec_resolves_the_commit_arm_against_stock_fp8():
    stock, commit = resolve_arms(["stock-fp8", "commit"], ("verity_bench.vllm.commit_arm:COMMIT",))
    assert commit.id == "commit" and commit.experimental and commit.control == "stock-fp8" and commit.protocols == ()
    assert commit.precision is None and commit.resident and stock.id == "stock-fp8"
    assert isinstance(COMMIT, CommitArm)


def test_the_served_spec_times_both_commits_against_their_own_stock_engines():
    spec = resolve_spec("llama31-8b-fp8-1gpu-commit-served")
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.control, a.execution, a.experimental) for a in arms] == [
        ("stock-fp8", None, None, False),
        ("commit", "stock-fp8", None, True),
        ("stock-fp8@compiled", None, "compiled", False),
        ("commit@compiled", "stock-fp8@compiled", "compiled", True),
    ]
    assert spec.eager and isinstance(COMMIT_COMPILED, CompiledCommitArm)


def test_the_width_spec_times_only_the_served_commit_against_served_stock_in_blocks():
    served, width = resolve_spec("llama31-8b-fp8-1gpu-commit-served"), resolve_spec("llama31-8b-fp8-1gpu-commit-served-width")
    arms = resolve_arms(width.arms, width.arm_modules, width.arm_settings)
    assert [(a.id, a.control, a.execution) for a in arms] == [
        ("stock-fp8@compiled", None, "compiled"), ("commit@compiled", "stock-fp8@compiled", "compiled")]
    settings, served_settings = width.preset.engine_settings, served.preset.engine_settings
    assert {k: v for k, v in settings.items() if k != "resident_gpu_memory_utilization"} == {
        k: v for k, v in served_settings.items() if k != "resident_gpu_memory_utilization"}
    assert 2 * settings["resident_gpu_memory_utilization"] > 1 and width.reps % 2 == 0


def test_an_experimental_arms_execution_is_one_of_the_served_executions(monkeypatch):
    mod = ModuleType("fake_exec_arm")
    mod.ARM = SimpleNamespace(id="x", install=lambda target: None, execution="bi-compiled")
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    with pytest.raises(ValueError, match="execution 'bi-compiled'"):
        resolve_arms(["x"], ("fake_exec_arm:ARM",))


class _Collector:
    """`NativeCollectCommitter` as ``ready`` drives it, committing under its ``scheme``."""

    scheme = "rows-sha512"

    def __init__(self, engine, **kw):
        self.engine, self.kw = engine, kw
        self.windowed = self.bounded = True
        self.retain, self.commit_scheme, self.name = "none", _Collector.scheme, "fake"

    def size_arena(self, envelope):
        return {"per_token_bytes": 7}

    def enable_windows(self, sizing):
        return {"slots": 8}


def _engine():
    vc = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16384, max_num_seqs=128),
                         model_config=SimpleNamespace(max_model_len=8384))
    return SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vc)), model="model")


@pytest.mark.parametrize("scheme", ["rows-sha512", "vllm-v1"])
def test_each_arm_commits_the_deployed_scheme_or_refuses(monkeypatch, scheme):
    native_collect = ModuleType("verity_vllm.commit.committer.native_collect")
    native_collect.NativeCollectCommitter = _Collector
    monkeypatch.setitem(sys.modules, native_collect.__name__, native_collect)
    monkeypatch.setattr(_Collector, "scheme", scheme)
    for k in ("VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED", "VERITY_RETAIN"):
        monkeypatch.delenv(k, raising=False)
    eager, compiled = CommitArm("commit"), CompiledCommitArm("commit@compiled")
    attached = []
    compiled.source = SimpleNamespace(note_engine=attached.append, attach=attached.append, summary=dict)
    monkeypatch.setattr(CommitArm, "engine", lambda self, target: _engine())
    for arm in (eager, compiled):
        if scheme != "rows-sha512":
            with pytest.raises(RuntimeError, match="commits 'vllm-v1', not the deployed 'rows-sha512'"):
                arm.ready(None)
            continue
        arm.ready(None)
        assert arm.committer.kw["native_hooks"] and arm.per_token_bytes == 7
        assert arm.gate(None)["commit_scheme"] == "rows-sha512"
    if scheme == "rows-sha512":
        assert attached == ["model", compiled.committer]


def _vllm(path: str):
    """A ``vllm.LLM`` as far as `committer_engine` reads it: the in-process runner (`handle.model_runner`'s path) and the model
    config; the LLM and its runner."""
    runner = SimpleNamespace(model=object())
    runner.get_model = lambda: runner.model
    core = SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner)))
    vc = SimpleNamespace(model_config=SimpleNamespace(model=path, revision=None))
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core, vllm_config=vc)), runner


def test_both_arms_take_the_committers_engine_from_the_llms_public_names():
    from verity_bench.vllm.runner import _DirectLLM

    vllm, runner = _vllm("/hf/snapshots/4e20de3624")
    pinned = SimpleNamespace(vllm=vllm, model=runner.model, pin=SimpleNamespace(model="meta-llama/Llama-3.2-1B", revision="4e20de3"))
    local_vllm, local_runner = _vllm("/models/local")
    local = _DirectLLM(local_vllm, {})
    assert not hasattr(local, "_engine")
    for arm in (CommitArm("commit"), CompiledCommitArm("commit@compiled")):
        got = arm.engine(SimpleNamespace(engine=pinned))
        assert (got.llm, got.runner, got.model) == (vllm, runner, runner.model)
        assert got.cp == {"repo": "meta-llama/Llama-3.2-1B", "revision": "4e20de3"}
        got = arm.engine(SimpleNamespace(engine=local))
        assert (got.llm, got.runner, got.model) == (local_vllm, local_runner, local_runner.model)
        assert got.cp == {"repo": "/models/local", "revision": None}


class _Source:
    def __init__(self):
        self.errors, self.stats, self.runs = [], {"replayed_steps": 3, "eager_steps": 0, "drain_s": 0.1}, 0

    def begin_run(self):
        self.runs += 1

    def summary(self):
        return {"errors": list(self.errors)}


def test_the_compiled_arms_source_errors_fail_its_runs_capture_time_ones_included():
    arm = CompiledCommitArm("commit@compiled")
    arm.committer, arm.source = FakeCommitter(committed=0), _Source()
    target = Target(resolve_arms("stock-fp8")[0], None, None, None)
    _run(arm, target, "gate")
    _run(arm, target, "measure")
    assert arm.gate(target)["passed"] and arm.source.runs == 2
    assert arm.extras(target)["replayed_steps"] == 3
    arm.source.errors.append("capture FULL:8: 2 captured copies never flushed")
    with pytest.raises(RuntimeError, match="never flushed"):
        _run(arm, target, "measure")
    assert not arm.gate(target)["passed"]


def test_each_run_is_one_commit_and_only_gate_and_warmup_are_learn_only():
    com = FakeCommitter(learned=[0])
    arm, target = _arm(com)
    for phase in ("gate", "warmup", "measure"):
        _run(arm, target, phase)
    assert com.events == ["reset", ("install", True), "finalize", "uninstall"] * 2 + ["reset", ("install", False), "finalize", "uninstall"]
    gate = arm.gate(target)
    assert gate["passed"] and [r["learned_only_steps"] for r in gate["runs"]] == [1, 1, 0]
    assert arm.extras(target)["committed_bytes"] == 4096


def test_a_measured_run_without_a_root_fails_the_point():
    arm, target = _arm(FakeCommitter(root=b""))
    _run(arm, target, "gate")
    with pytest.raises(RuntimeError, match="no run root"):
        _run(arm, target, "measure")


def test_host_memory_reads_the_tightest_limit_on_the_cgroup_path_and_its_anonymous_bytes(tmp_path):
    from verity_bench.vllm.commit_arm import host_memory

    root = tmp_path / "cgroup"
    job, engine = root / "app.slice" / "job.scope", root / "app.slice" / "job.scope" / "engine"
    engine.mkdir(parents=True)
    (root / "app.slice" / "memory.max").write_text(f"{128 << 30}\n")
    (root / "app.slice" / "memory.stat").write_text("anon 1\nfile 2\n")
    (job / "memory.max").write_text(f"{64 << 30}\n")
    (job / "memory.stat").write_text(f"anon {20 << 30}\nfile {30 << 30}\n")
    (engine / "memory.max").write_text("max\n")
    self_cgroup = tmp_path / "self"
    self_cgroup.write_text("0::/app.slice/job.scope/engine\n")
    assert host_memory(root, self_cgroup) == (64 << 30, 20 << 30)
    self_cgroup.write_text("1:name=systemd:/x\n")
    assert host_memory(root, self_cgroup) is None


def test_a_run_whose_levels_cannot_fit_the_jobs_memory_is_refused_by_name(monkeypatch):
    import verity_bench.vllm.commit_arm as commit_arm

    arm, target = _arm(FakeCommitter())
    arm.per_token_bytes = 27_600_000
    monkeypatch.setattr(commit_arm, "host_memory", lambda: (64 << 30, 20 << 30))
    target.tokens = 18_432
    _run(arm, target, "gate")
    target.tokens = 66_560
    with pytest.raises(RuntimeError, match=r"refused before the run: 66560 tokens .* 53\.5 GiB of Merkle levels"):
        _run(arm, target, "gate")
    monkeypatch.setattr(commit_arm, "host_memory", lambda: None)
    _run(arm, target, "gate")


def test_a_committer_that_keeps_no_levels_on_host_is_never_refused(monkeypatch):
    import verity_bench.vllm.commit_arm as commit_arm

    arm, target = _arm(FakeCommitter(retain_levels=False))
    arm.per_token_bytes = 27_600_000
    monkeypatch.setattr(commit_arm, "host_memory", lambda: (64 << 30, 20 << 30))
    target.tokens = 66_560
    _run(arm, target, "gate")


def test_collector_errors_and_missing_tensors_fail_the_gate_and_a_measured_run():
    arm, target = _arm(FakeCommitter(errors=["windowed staging: plan mismatch"]))
    _run(arm, target, "gate")
    assert not arm.gate(target)["passed"]
    arm, target = _arm(FakeCommitter(committed=9))
    with pytest.raises(RuntimeError, match="9 of 10 expected tensors committed"):
        _run(arm, target, "measure")
