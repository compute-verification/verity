"""The Commit arm (`verity_bench.vllm.commit_arm`) against a fake committer: its runs open and close one Commit each, the gate
and warm-up runs are learn-only, a measured run without a root or with a collector error fails the point, and the custom gate
reads the gate runs. No engine."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_bench.vllm.arms import Target, resolve_arms
from verity_bench.vllm.commit_arm import COMMIT, CommitArm


class FakeCommitter:
    name = "native_collect_v2_auto_nw+retain_none"

    def __init__(self, *, root=b"\1" * 32, learned=(), errors=(), committed=None):
        self.root, self.learned, self.errs, self.committed = root, list(learned), list(errors), committed
        self.events, self.learn_only = [], False

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
    arm.per_token_bytes = 3_450_000
    monkeypatch.setattr(commit_arm, "host_memory", lambda: (64 << 30, 20 << 30))
    target.tokens = 18_432
    _run(arm, target, "gate")
    target.tokens = 66_560
    with pytest.raises(RuntimeError, match=r"refused before the run: 66560 tokens .* 53\.5 GiB of Merkle levels"):
        _run(arm, target, "gate")
    monkeypatch.setattr(commit_arm, "host_memory", lambda: None)
    _run(arm, target, "gate")


def test_collector_errors_and_missing_tensors_fail_the_gate_and_a_measured_run():
    arm, target = _arm(FakeCommitter(errors=["windowed staging: plan mismatch"]))
    _run(arm, target, "gate")
    assert not arm.gate(target)["passed"]
    arm, target = _arm(FakeCommitter(committed=9))
    with pytest.raises(RuntimeError, match="9 of 10 expected tensors committed"):
        _run(arm, target, "measure")
