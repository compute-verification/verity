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


def test_collector_errors_and_missing_tensors_fail_the_gate_and_a_measured_run():
    arm, target = _arm(FakeCommitter(errors=["windowed staging: plan mismatch"]))
    _run(arm, target, "gate")
    assert not arm.gate(target)["passed"]
    arm, target = _arm(FakeCommitter(committed=9))
    with pytest.raises(RuntimeError, match="9 of 10 expected tensors committed"):
        _run(arm, target, "measure")
