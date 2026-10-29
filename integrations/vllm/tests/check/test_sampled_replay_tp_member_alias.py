"""CPU tests for MOE-TP-01: a rank Program's TP member (`part_rank<r>` / `shard_rank<r>`) compares against the relation's one output
when the evaluator did not name it -- `0` / `out` (TP-08d, the row-parallel / lm_head Gemm) and now MoeSum's `sum` (the rank's MoE-block
partial before the all-reduce: a Qwen3-30B-A3B TP2 B2 Commit left every MoeSum_v1 stratum on both ranks
'evaluator produced no member part_rank<r>' -- 336 strata per rank not evaluated, grade partial).  Never a guess between two outputs;
never for a member that is not a TP member."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from verity_vllm.check.replay.compare import _tp_member_alias


def test_moesum_sum_is_the_ranks_partial():
    y = np.arange(4, dtype=np.uint16)
    assert _tp_member_alias({"sum": y}, "part_rank0") is y
    assert _tp_member_alias({"sum": y}, "part_rank1") is y


def test_gemm_zero_and_out_still_win_and_shard_members_too():
    a, b = np.zeros(2, dtype=np.uint16), np.ones(2, dtype=np.uint16)
    assert _tp_member_alias({"0": a}, "part_rank0") is a
    assert _tp_member_alias({"out": b}, "shard_rank1") is b
    assert _tp_member_alias({"0": a, "sum": b}, "part_rank0") is a  # the named output wins over the one-output rule


def test_never_a_guess_between_two_outputs_and_never_a_non_tp_member():
    a, b = np.zeros(2, dtype=np.uint16), np.ones(2, dtype=np.uint16)
    assert _tp_member_alias({"up": a, "down": b}, "part_rank0") is None
    assert _tp_member_alias({"sum": a}, "out") is None
    assert _tp_member_alias({"sum": a}, "down") is None
    assert _tp_member_alias({"part_rank0": a, "sum": b}, "part_rank0") is None  # already produced: nothing to alias
