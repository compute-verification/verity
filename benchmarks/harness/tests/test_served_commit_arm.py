"""served_commit's arm (`benchmarks/served_commit/arm.py`) gates on the harness's run phases: under bounded staging its gate and
warm-up runs are learn-only, a run of theirs left without a root is a declared warm-up named in the custom gate's verdict, and a
measured run without a root fails the gate. The committer and the compiled source are fakes; the roots are the GPU suites'."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ARM = Path(__file__).resolve().parents[2] / "served_commit" / "arm.py"


def _arm_module():
    spec = importlib.util.spec_from_file_location("served_commit_arm", ARM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Committer:
    def __init__(self, bounded: bool) -> None:
        self.bounded, self.learn_only, self._layouts, self.result = bounded, False, {}, None

    def reset(self) -> None:
        pass

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def finalize(self):
        return self.result


class _Source:
    stats = {"replayed_steps": 0, "eager_steps": 0, "drain_s": 0.0}
    errors: list = []

    def begin_run(self) -> None:
        pass

    def summary(self) -> dict:
        return {}


def _sweep(arm, com, runs) -> dict:
    """Runs (phase, point, run root, learned-only steps) through the arm as the harness would; the custom gate's verdict."""
    learn_only = []
    for phase, point, root, learned in runs:
        target = SimpleNamespace(phase=phase, point=point)
        arm.begin_run(target)
        learn_only.append(com.learn_only)
        com.result = SimpleNamespace(run_root=root, extra={"bounded_staging": {"learned_only_steps": list(range(learned))}})
        arm.end_run(target)
    return {"learn_only": learn_only, **arm.gate(None)}


def test_a_rootless_gate_or_warmup_run_is_a_declared_warmup():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-noretain", retain="none"), _Committer(bounded=True)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B8", b"r" * 32, 0), ("gate", "B32", b"", 4), ("warmup", "B32", b"", 4),
                            ("measure", "B32", b"r" * 32, 0)])
    assert got["learn_only"] == [True, True, True, False]
    assert got["passed"] and not got["failed"]
    assert got["warmups"] == [{"run": "gate:B32", "learned_only_steps": 4}, {"run": "warmup:B32", "learned_only_steps": 4}]


def test_a_measured_run_without_a_root_fails_the_gate():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-noretain", retain="none"), _Committer(bounded=True)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B32", b"r" * 32, 0), ("measure", "B32", b"", 4)])
    assert not got["passed"] and not got["warmups"]
    assert got["failed"][0]["run"] == "measure:B32" and "not a declared warm-up" in got["failed"][0]["errors"][0]


def test_unbounded_staging_runs_no_learn_only_pass():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-noretain", retain="none"), _Committer(bounded=False)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B32", b"r" * 32, 0), ("warmup", "B32", b"r" * 32, 0)])
    assert got["learn_only"] == [False, False] and got["passed"] and not got["warmups"]
