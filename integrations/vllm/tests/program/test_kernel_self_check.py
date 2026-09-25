"""Every (Definition, kernel) pair registered through `verity_vllm.program.kernels.kernel_registry` — the replay row
kernels (`rows`) and the twins — against the reference by core `verity.evaluation.self_check`, on each of the kernel's
check specialisations: no mismatch, and not every instance declined."""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from verity.evaluation import kernels, self_check
from verity_vllm.program.kernels import kernel_registry as KR
from verity_vllm.program.kernels import rows, twins

N = 6
PAIRS = [(r.key, r.kernel) for r in KR.registered()]


def test_every_row_family_and_twin_is_registered_with_core():
    assert {k for k, kern in PAIRS if kern == rows.KERNEL} == set(rows.ROWS)
    assert {k for k, kern in PAIRS if kern == twins.KERNEL} == set(twins.FAMILIES)
    for key, kernel in PAIRS:
        assert kernels(key)[kernel] is next(r.core for r in KR.registered() if (r.key, r.kernel) == (key, kernel))


@pytest.mark.parametrize("key,kernel", PAIRS, ids=[f"{k}-{kern}" for k, kern in PAIRS])
def test_kernel_matches_the_reference(key, kernel):
    reg = next(r for r in KR.registered() if (r.key, r.kernel) == (key, kernel))
    targets = reg.check_targets(np.random.default_rng(zlib.crc32(f"{key}/{kernel}".encode())))
    assert targets, f"{key}: no check specialisations"
    for t in targets:
        assert KR.covers(t, kernel) is None, (t.id, KR.covers(t, kernel))
        sc = self_check(t, kernel=kernel, n=N, seed=0)
        assert sc.ok, str(sc)
        assert sc.declined < sc.n, f"{t.id}: {kernel} declined all {sc.n} instances"
