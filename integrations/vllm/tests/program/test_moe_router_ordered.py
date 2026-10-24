"""The MoE router in its selection order (`registry.moe.MoeRouterTopKOrdered_v1` / `...OrderedNorm_v1`): its outputs (weights and ids)
equal the kernel-order Definitions' bit for bit (ties, NaN, +-inf, +-0); it is the default construction and the kernel-order routers stay
declarable (`moe_construction = "indexed-read"`), their digests unchanged.  The interior
words it commits are `topkGating`'s own through `MoeRouterProbs` (the `fmaxf` row max, NaN exponentials and reciprocal as 0x7FFFFFFF),
which the kernel-order statement's interior is not: `tests/properties/test_router_tap_exactness.py` and the router-tap exactness record.
Its partition (the softmax committed once, nothing computed twice) is checked with the no-recompute rule, `tests/query/test_word.py`."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.codec import encode_program
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import moe

NAN, PINF, NINF, PZ, NZ = 0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000


def _bf16(a) -> list[int]:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.int64).tolist()


def _rows(E: int, n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    rows = [[PZ] * E, [NZ] * E, [NINF] * E, [NAN] * E, [PINF] * E, _bf16([2.0] * E), [0x7F7F] * E,
            _bf16(np.repeat(rng.standard_normal(E // 2 + 1), 2)[:E]), _bf16(10.0 + rng.standard_normal(E) * 1e-3),
            _bf16([-200.0] * (E - 1) + [100.0]), [int(x) for x in rng.integers(1, 0x7F, E)]]
    for r in range(n):
        row = _bf16(rng.choice(rng.standard_normal(3) * 3, E) if r % 2 else rng.standard_normal(E) * 4)
        for i in rng.choice(E, int(rng.integers(0, 4)), replace=False):
            row[int(i)] = int(rng.choice([NAN, PINF, NINF, PZ, NZ, 0x7F7F, 0xFF7F, 0x0001]))
        rows.append(row)
    return rows


@pytest.mark.parametrize("old,new", [(moe.MoeRouterTopK, moe.MoeRouterTopKOrdered), (moe.MoeRouterTopKNorm, moe.MoeRouterTopKOrderedNorm)])
@pytest.mark.parametrize("E,TOPK,VPT,n", [(16, 4, 4, 60), (64, 8, 8, 6)])
def test_the_ordered_router_is_the_kernel_order_router(old, new, E, TOPK, VPT, n):
    a, b = bind(old, E=E, TOPK=TOPK, VPT=VPT), bind(new, E=E, TOPK=TOPK, VPT=VPT)
    for row in _rows(E, n, seed=E + TOPK):
        assert evaluate(a, row) == evaluate(b, row), row


def test_ties_go_to_the_lower_expert_in_both():
    row = [PZ] * 16
    row[9] = row[3] = _bf16([5.0])[0]
    ids = evaluate(bind(moe.MoeRouterTopKOrdered, E=16, TOPK=4, VPT=4), row)[4:]
    assert list(ids[:2]) == [3, 9] and list(ids[2:]) == [0, 1]


def _digest(fn, **st) -> str:
    return hashlib.sha256(json.dumps(encode_program(Program(bind(fn, **st))), sort_keys=True).encode()).hexdigest()[:16]


def test_the_kernel_order_routers_of_record_are_unchanged():
    assert _digest(moe.MoeRouterTopK, E=64, TOPK=8, VPT=8) == "581a4ad647391b7e"
    assert _digest(moe.MoeRouterTopK, E=128, TOPK=8, VPT=8) == "fb04cb4e755fd571"
    assert _digest(moe.MoeRouterTopKNorm, E=128, TOPK=8, VPT=8) == "ef2f47ef03606d54"


def test_the_statement_is_the_default_and_the_kernel_order_router_stays_declarable():
    base = TargetProfile.default()
    assert "moe_construction" not in base.to_json() and base.effective_moe_construction() == "indexed-read-ordered"
    assert base.moe_construction_declaration()["definitions"][0] == "MoeRouterTopKOrdered_v1{E,TOPK,VPT}"
    kernel_order = base.with_overrides(moe_construction="indexed-read")
    assert kernel_order.digest() != base.digest()
    assert kernel_order.moe_construction_declaration()["definitions"][0] == "MoeRouterTopK_v1{E,TOPK,VPT}"


def test_the_fold_states_the_declared_router():
    from verity_vllm.engine.profiles.gen_ov_moe_patterns import MOE_CONSTRUCTION_PATTERNS
    assert MOE_CONSTRUCTION_PATTERNS["indexed-read"]({}).router is None
    assert MOE_CONSTRUCTION_PATTERNS["indexed-read-ordered"]({}).router == "ordered"
