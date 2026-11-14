"""`verity_vllm.engine.profiles.layer_classes.SHARDED_ENDINGS`: which of the class names the module's tables hold the
endings pick, and that each ending picks one."""
from __future__ import annotations

import pytest

from verity_vllm.engine.profiles import layer_classes as LC

#: every class name the module's tables hold
NAMES = {*LC.TAPPED, *LC.LINEAR, *LC.NORMS, *LC.GATED_ACTS, LC.LM_HEAD, LC.SAMPLER}


def test_the_endings_pick_the_parallel_linears_the_embeddings_and_the_lm_head() -> None:
    """Of the tables' names the endings pick the four parallel linears, the two embeddings and the LM head. They leave
    the linear base and the replicated linear, and ``LOGITS``, which a reader picks by its whole name."""
    assert {n for n in NAMES if n.endswith(LC.SHARDED_ENDINGS)} == {
        LC.COLUMN_LINEAR, LC.ROW_LINEAR, "QKVParallelLinear", "MergedColumnParallelLinear", LC.EMBEDDING, LC.ROTARY, LC.LM_HEAD}
    assert not LC.LOGITS.endswith(LC.SHARDED_ENDINGS) and not LC.LINEAR_BASE.endswith(LC.SHARDED_ENDINGS)


@pytest.mark.parametrize("ending", LC.SHARDED_ENDINGS)
def test_each_ending_picks_a_name_of_the_tables(ending: str) -> None:
    assert any(n.endswith(ending) for n in NAMES)
