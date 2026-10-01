"""The TP Commit's warm-up instrumented run is the bounded-staging learn-only pass on every rank (`rank_worker.tp2_commit_install`), as the
single-GPU Commit's is: a learning step over the transient budget records its plan instead of failing the warm-up."""
from __future__ import annotations

from types import SimpleNamespace

from verity_vllm.engine import rank_worker as RW


class _Com:
    def __init__(self, bounded: bool) -> None:
        self.bounded, self.learn_only, self.calls = bounded, False, []

    def reset(self) -> None:
        self.calls.append("reset")

    def install(self) -> None:
        self.calls.append(("install", self.learn_only))

    def finalize(self):
        return SimpleNamespace(steps=[SimpleNamespace(n_leaves=3)], extra={"errors": [], "bounded_staging": {"learned_only_steps": [0]}})

    def uninstall(self) -> None:
        self.calls.append("uninstall")


def _worker(com):
    w = RW.TP2CaptureWorkerExtension.__new__(RW.TP2CaptureWorkerExtension)
    w._tp2_commit = {"com": com, "rank": 1, "eng": None, "base": None}
    return w


def test_the_warm_up_learns_and_the_record_run_does_not(monkeypatch):
    import verity_vllm.pipeline.commit as C

    monkeypatch.setattr(C, "assert_pristine", lambda eng, base: None)
    com = _Com(bounded=True)
    w = _worker(com)
    assert w.tp2_commit_install(learn_only=True)["learn_only"] is True and com.calls[-1] == ("install", True)
    fin = w.tp2_commit_finalize()
    assert fin["learn_only"] is True and fin["leaves"] == 3 and com.learn_only is False
    assert w.tp2_commit_install()["learn_only"] is False and com.calls[-1] == ("install", False)
    unbounded = _Com(bounded=False)
    assert _worker(unbounded).tp2_commit_install(learn_only=True)["learn_only"] is False, "learn-only is a bounded-staging pass"
