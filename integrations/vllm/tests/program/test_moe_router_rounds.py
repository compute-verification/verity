"""The MoE router stated round by round (`registry.moe.MoeRouterTopKRounds_v1` / `...RoundsNorm_v1`): the same function as the kernel-order
Definitions bit for bit (ties, NaN, +-inf, +-0), a `Q_word_v1` partition with no committed interior value, and opt-in only (no digest of
record moves)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.codec import encode_program
from verity.ir.defs import bind
from verity.ir.program import Program
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import moe
from verity_vllm.query import word as W

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


@pytest.mark.parametrize("old,new", [(moe.MoeRouterTopK, moe.MoeRouterTopKRounds), (moe.MoeRouterTopKNorm, moe.MoeRouterTopKRoundsNorm)])
@pytest.mark.parametrize("E,TOPK,VPT,n", [(16, 4, 4, 60), (64, 8, 8, 6)])
def test_the_rounds_router_is_the_kernel_order_router(old, new, E, TOPK, VPT, n):
    a, b = bind(old, E=E, TOPK=TOPK, VPT=VPT), bind(new, E=E, TOPK=TOPK, VPT=VPT)
    for row in _rows(E, n, seed=E + TOPK):
        assert evaluate(a, row) == evaluate(b, row), row


def test_ties_go_to_the_lower_expert_in_both():
    row = [PZ] * 16
    row[9] = row[3] = _bf16([5.0])[0]
    ids = evaluate(bind(moe.MoeRouterTopKRounds, E=16, TOPK=4, VPT=4), row)[4:]
    assert list(ids[:2]) == [3, 9] and list(ids[2:]) == [0, 1]


@pytest.mark.parametrize("fn", [moe.MoeRouterTopKRounds, moe.MoeRouterTopKRoundsNorm])
def test_every_unit_is_one_32_bit_output_and_nothing_interior_is_committed(fn):
    r = W.unit_rule(bind(fn, E=64, TOPK=8, VPT=8))
    assert r["by_kind"] == {"output": 16} and r["committed_interior_words"] == 0 and not r["violations"]
    assert {(c["out_bits"], c["out_gates"]) for c in r["classes"]} == {(32, 1)}
    assert W.unit_rule(bind(moe.MoeRouterTopK, E=64, TOPK=8, VPT=8))["committed_interior_words"] == 1074


def _digest(fn, **st) -> str:
    return hashlib.sha256(json.dumps(encode_program(Program(bind(fn, **st))), sort_keys=True).encode()).hexdigest()[:16]


def test_the_kernel_order_routers_of_record_are_unchanged():
    assert _digest(moe.MoeRouterTopK, E=64, TOPK=8, VPT=8) == "581a4ad647391b7e"
    assert _digest(moe.MoeRouterTopK, E=128, TOPK=8, VPT=8) == "fb04cb4e755fd571"
    assert _digest(moe.MoeRouterTopKNorm, E=128, TOPK=8, VPT=8) == "ef2f47ef03606d54"


def test_the_statement_is_opt_in():
    base = TargetProfile.default()
    assert "moe_construction" not in base.to_json() and base.effective_moe_construction() == "indexed-read"
    rounds = base.with_overrides(moe_construction="indexed-read-rounds")
    assert rounds.digest() != base.digest()
    assert rounds.moe_construction_declaration()["definitions"][0] == "MoeRouterTopKRounds_v1{E,TOPK,VPT}"


def test_the_fold_states_the_declared_router():
    from verity_vllm.engine.profiles.gen_ov_moe_patterns import MOE_CONSTRUCTION_PATTERNS
    assert MOE_CONSTRUCTION_PATTERNS["indexed-read"]({}).router is None
    assert MOE_CONSTRUCTION_PATTERNS["indexed-read-rounds"]({}).router == "rounds"
