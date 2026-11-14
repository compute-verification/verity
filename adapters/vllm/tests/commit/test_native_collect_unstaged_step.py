"""A step the collector's worker is handed without its having been staged through the window ring is refused by name, and nothing of
it is committed (`verity_vllm.commit.committer.native_collect.NativeCollectCommitter._commit_step`).

The committers commit rows-sha512 only: `verity_vllm.commit.committer.native_host.NativeHostCommitter.__init__` sets `rows` on every
path, and the collector hashes a step's rows window by window.  A step that reaches the worker with no `windowed` mark has no window
to hash, whatever else is set on the collector or on the step.  The collector here is bare (no engine, no extension build), so the
tests need no device."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.commit import step_rows
from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.commit.committer.native_host import NativeHostCommitter

REFUSAL = "rows-sha512: step 7 was not staged through the window ring (enable_windows before the run)"


def _host(retain: str) -> NativeHostCommitter:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VERITY_RETAIN", retain)
        return NativeHostCommitter(None, run_id="unstaged", gpu_tree=True)


def _bare(**attrs: Any) -> NC.NativeCollectCommitter:
    """A collector with only what the worker's step reads before it refuses, `rows` as a constructed committer has it."""
    com = NC.NativeCollectCommitter.__new__(NC.NativeCollectCommitter)
    com.consumer_delay_s = 0.0
    com.rows = _host("none").rows
    com.fault, com.native_worker = "", False
    for name, value in attrs.items():
        setattr(com, name, value)
    com._done, com._layouts, com._gpu_blocks, com._gpu_levels = {}, {}, {}, {}
    return com


class _Job(SimpleNamespace):
    """A worker's step with a recorder in the arena's place: the refusal comes before anything of the step is read."""


@pytest.mark.parametrize("retain", ["device", "host", "none", "sample"])
def test_a_committer_commits_rows_under_every_retention(retain: str) -> None:
    com = _host(retain)
    assert com.rows is True
    assert com.commit_scheme == step_rows.NAME == "rows-sha512"


@pytest.mark.parametrize("job", [
    pytest.param({}, id="no-mark"),
    pytest.param({"windowed": False}, id="marked-not-windowed"),
    pytest.param({"windowed": None}, id="mark-null"),
    pytest.param({"windowed": 0}, id="mark-zero"),
])
@pytest.mark.parametrize("collector", [
    pytest.param({}, id="plain"),
    pytest.param({"native_worker": True}, id="native-worker"),
    pytest.param({"fault": "release_before_keep"}, id="fault-release-before-keep"),
    pytest.param({"fault": "release_before_keep", "native_worker": True}, id="fault-and-native-worker"),
])
def test_a_step_not_staged_through_the_window_ring_is_refused_by_name(job: dict[str, Any], collector: dict[str, Any]) -> None:
    com = _bare(**collector)
    step = _Job(step=7, seq=0, arena=object(), overflow=False, cursor=4096, **job)
    before = dict(vars(step))
    with pytest.raises(RuntimeError) as refused:
        com._commit_step(step)
    assert str(refused.value) == REFUSAL
    assert vars(step) == before                                     # nothing was bound on the step, no layout among it
    assert not (com._done or com._layouts or com._gpu_blocks or com._gpu_levels)


def test_the_refusal_names_the_scheme_and_the_step() -> None:
    com = _bare()
    with pytest.raises(RuntimeError, match=rf"^{step_rows.NAME}: step 123 was not staged through the window ring"):
        com._commit_step(_Job(step=123))
