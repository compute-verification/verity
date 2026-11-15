"""Range openings (`NativeHostCommitter.leaf_span` / `open_range` / `verify_range`): a contiguous run of leaves opened with the boundary
siblings of its tree and verified against the ORIGINAL run commitment -- the primitive `oracle_compare.OpenedReader` reads every compared
value through.  Both committer layouts: the GPU-tree step block (`commit_block_offline`: 256 B leaves spanning tensors; needs torch) and
the per-tensor host path (`commit_offline`: a range lies inside one tensor's leaves)."""
from __future__ import annotations

import dataclasses
import random

import pytest

from tests.check.opened import commit_host_steps, commit_steps, meta


def _world(seed: int = 3) -> dict[int, list[tuple[str, bytes]]]:
    rnd = random.Random(seed)
    return {0: [(f"t0.{i}", rnd.randbytes(n)) for i, n in enumerate((1, 255, 256, 700, 3000))],
            1: [(f"t1.{i}", rnd.randbytes(n)) for i, n in enumerate((9, 513, 64))]}


def _commit(layout: str, steps: dict, **kw):
    if layout == "block":
        pytest.importorskip("torch")
        return commit_steps(steps, **kw)
    return commit_host_steps(steps, **kw)


def _ranges(com, run, step: int, layout: str):
    """Every contiguous leaf range a reader can ask for: any [a, b) of a block step; any [a, b) inside one tensor on the per-tensor path."""
    if layout == "block":
        n = run.steps[step].n_leaves
        return [(a, b) for a in range(n) for b in range(a + 1, n + 1)]
    return [(m.first_leaf + a, m.first_leaf + b) for m in com._layouts[step] for a in range(m.n_leaves) for b in range(a + 1, m.n_leaves + 1)]


@pytest.mark.parametrize("layout", ["block", "host"])
def test_every_leaf_range_opens_and_verifies_and_every_member_byte_range_reads_back(layout):
    steps = _world()
    com, run = _commit(layout, steps)
    for s in steps:
        for a, b in _ranges(com, run, s, layout):
            ro = com.open_range(s, a, b)
            assert ro is not None and (ro.lo, ro.hi) == (a, b) and com.verify_range(run, ro), (layout, s, a, b)
    roots: dict[int, bytes] = {}
    for s, ts in steps.items():
        for name, raw in ts:
            m, n = meta(com, s, name), len(raw)
            for lo, hi in ((0, n), (0, 1), (n - 1, n), (n // 3, 2 * n // 3 + 1)):
                a, b, off = com.leaf_span(s, m, lo, hi)
                ro = com.open_range(s, a, b)
                assert com.verify_range(run, ro, roots) and ro.value[off:off + hi - lo] == raw[lo:hi], (layout, s, name, lo, hi)
    assert sorted(roots) == [0, 1]                             # one authenticated bare root per step, kept by the caller


@pytest.mark.parametrize("layout", ["block", "host"])
def test_a_changed_byte_a_replaced_sibling_a_shifted_range_or_another_runs_commitment_is_refused(layout):
    steps = _world()
    com, run = _commit(layout, steps)
    m = meta(com, 0, "t0.4")
    a, b, off = com.leaf_span(0, m, 1000, 1100)
    ro = com.open_range(0, a, b)
    roots: dict[int, bytes] = {}
    assert com.verify_range(run, ro, roots) and 0 in roots
    flipped = bytearray(ro.value)
    flipped[off] ^= 1
    assert not com.verify_range(run, dataclasses.replace(ro, value=bytes(flipped)), roots)          # also with the step's root cached
    assert not com.verify_range(run, dataclasses.replace(ro, value=ro.value[:-1]))
    k = next(i for i, (left, right) in enumerate(ro.path) if left is not None or right is not None)
    left, right = ro.path[k]
    bad = (bytes(32), right) if left is not None else (left, bytes(32))
    assert not com.verify_range(run, dataclasses.replace(ro, path=ro.path[:k] + (bad,) + ro.path[k + 1:]))
    assert not com.verify_range(run, dataclasses.replace(ro, path=ro.path[:-1]))                    # wrong path shape
    assert not com.verify_range(run, dataclasses.replace(ro, lo=ro.lo + 1, hi=ro.hi + 1))           # the same bytes claimed one leaf later
    assert not com.verify_range(run, dataclasses.replace(ro, step=1))
    assert not com.verify_range(run, dataclasses.replace(ro, step=len(run.steps)))
    _other, run2 = _commit(layout, steps, run_id="another-run")                                     # same tensors, another run
    assert not com.verify_range(run2, ro)


def test_a_failed_verification_does_not_cache_the_step_root():
    com, run = commit_host_steps(_world())
    ro = com.open_range(0, 0, 1)
    roots: dict[int, bytes] = {}
    assert not com.verify_range(run, dataclasses.replace(ro, value=bytes([ro.value[0] ^ 1])), roots) and roots == {}
    assert com.verify_range(run, ro, roots) and list(roots) == [0]


def test_bytes_the_committer_does_not_retain_open_as_none_never_zeros():
    pytest.importorskip("torch")
    com, _run = commit_steps(_world(), retain="none")
    m = meta(com, 0, "t0.4")
    a, b, _off = com.leaf_span(0, m, 0, m.nbytes)
    assert com.open_range(0, a, b) is None
