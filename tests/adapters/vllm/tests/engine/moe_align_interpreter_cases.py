"""The interpreter cases of `test_moe_align.py`, which runs them in a pytest of their own: the op's Triton kernels, in Triton's
interpreter on CPU tensors, write the reference on both paths and at 0 pairs.  Never collected with the suite: Triton builds its
own library's kernels (`tl.sum`, `tl.cumsum`) for the interpreter only when `TRITON_INTERPRET` is set at its first import, and
`triton.jit` reads it when it decorates a kernel, while the suite's `conftest.py` loads Triton before any test (it imports
`torch._inductor.config`) and the variable is set for no other test.  This process starts with it."""
from __future__ import annotations

import os
import random

import pytest
import torch

from verity_vllm.engine import moe_align
from verity_vllm.engine.moe_align import reference

from .test_moe_align import E, routing

assert os.environ.get("TRITON_INTERPRET") == "1", "test_moe_align.py runs these, in a pytest that starts with TRITON_INTERPRET=1"


def run_kernels(align, ids, num_experts, block, expert_map=None, dtype=torch.int32):
    t = torch.tensor(ids, dtype=dtype).reshape(-1, 1)
    L = moe_align.sorted_len(len(ids), num_experts, block)
    s = torch.full((L,), -7, dtype=torch.int32)
    b = torch.full((moe_align.cdiv(L, block),), -7, dtype=torch.int32)
    n = torch.full((1,), -7, dtype=torch.int32)
    m = None if expert_map is None else torch.tensor(expert_map, dtype=torch.int32)
    align(t, num_experts, block, s, b, n, m)
    return s.tolist(), b.tolist(), int(n[0])


@pytest.mark.parametrize("path", ["one", "chunked"])
def test_kernels_write_the_reference_in_interpreter(path):
    small = 1 << 20 if path == "one" else 0
    align = moe_align.Align(small_max=small, chunk=64, tile=16, long_tile=32, cols=16)
    rng = random.Random(path)
    cases = [([2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3], 5, 4, None), ([1, 3, 1, 1, 1], 4, 2, None), ([5, 0], 8, 4, None),
             ([-1, 2, 3, 7, -9, 1], 3, 2, None), ([0, 1, 2, 1], 3, 2, [-1, 0, 2]), ([2, 0, 2, 2, 0], 3, 1, None)]
    cases += [([rng.randrange(-1, 9) for _ in range(rng.randrange(1, 70))], 8, bm, None) for bm in (1, 2, 3, 8, 16)]
    cases += [([rng.randrange(0, 2) for _ in range(100)], 3, 4, None), ([2] * 50, 3, 8, [0, 1, 2])]
    cases += [([rng.randrange(0, 5) for _ in range(90)], 5, 37, None)]
    cases += [(routing(rng, rows, how), E, 8, None) for rows, how in ((1, "uniform"), (7, "skew"), (11, "same6"))]
    for ids, ne, bm, mp in cases:
        assert run_kernels(align, ids, ne, bm, mp) == reference(ids, ne, bm, mp), (ids, ne, bm, mp)
    ids = [1 << 32 | 1, 2, -1, 1, 0, 1 << 33]
    assert run_kernels(align, ids, 3, 2, dtype=torch.int64) == reference(ids, 3, 2)
    assert align.calls[path] == len(cases) + 1


def test_kernels_write_the_total_at_zero_pairs_in_interpreter():
    align = moe_align.Align()
    for ne, bm, mp in ((E, 8, None), (5, 4, None), (3, 2, [-1, 0, 2])):
        assert run_kernels(align, [], ne, bm, mp) == reference([], ne, bm, mp) == ([], [], 0)
    assert align.calls == {"one": 3, "chunked": 0}
