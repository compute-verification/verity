"""The bytes a run retains on the host, as `verity_vllm.commit.committer.native_host.NativeHostCommitter.finalize` reports them
(`host_retained_bytes`): every level a step keeps on the host, and each step's retained copy at what the copy says it retains, or
at its size where it says nothing.  A copy that says zero counts zero; a released copy, one on the device and one a device pool
holds count nothing.  Offline (no CUDA)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.commit import test_openings_after_release as OR


def _copy(**fields: Any) -> SimpleNamespace:
    """A retained copy of 1,000 bytes on the host, with `fields` over that."""
    return SimpleNamespace(**{"is_cuda": False, "numel": lambda: 1000, **fields})


def _levels(com: Any) -> int:
    return sum(int(lv.nbytes) for lvls in com._gpu_levels.values() for lv in lvls)


def test_a_host_copy_counts_its_size_and_every_level_counts() -> None:
    com = OR._committer("host")
    for s in range(3):
        metas, block = OR._metas(s, 1)
        com.commit_block_offline(s, block, metas)
    copies = sum(int(b.numel()) for b, _used in com._gpu_blocks.values())
    assert copies > 0 and _levels(com) > 0
    assert com.finalize().extra["host_retained_bytes"] == _levels(com) + copies


def test_a_run_that_retains_no_copy_counts_its_levels_alone() -> None:
    com = OR._committer("none")
    run = OR._run(com)
    assert [b for b, _used in com._gpu_blocks.values()] == [None, None, None]
    assert run.extra["host_retained_bytes"] == _levels(com) > 0


@pytest.mark.parametrize(("copy", "counted"), [
    (_copy(), 1000),                                            # says nothing: its size
    (_copy(retained_bytes=None), 1000),                         # says null: its size
    (_copy(retained_bytes=4096), 4096),                         # says what it retains
    (_copy(retained_bytes=0), 0),                               # says zero, and zero is counted, not its size
    (_copy(retained_bytes=4096, on_device=True), 0),            # a device pool's: counted with the device's bytes
    (_copy(retained_bytes=4096, on_device=False), 4096),
    (_copy(is_cuda=True), 0),                                   # on the device
    (None, 0),                                                  # released
], ids=["no word", "null", "a count", "zero", "a device pool's", "a host pool's", "on the device", "released"])
def test_a_retained_copy_counts_what_it_says_it_retains(copy: Any, counted: int) -> None:
    com = OR._committer("none")
    OR._run(com, steps=2)
    com._gpu_blocks[1] = (copy, 0)
    assert com.finalize().extra["host_retained_bytes"] == _levels(com) + counted
