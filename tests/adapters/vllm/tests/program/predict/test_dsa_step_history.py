"""A step Program on the sparse-attention decoder at the key count its rows had when served: its rows at pos0.. read their
sequence's first pos0 positions as a history (`build.build_program`'s step at C = pos0), so each indexer row scores pos0 + i + 1
keys, and a served step's (M, pos0, C) is its advice row's (`segment.step_of`). GLM-5.3's checkpoint is cut to its first 4
layers (three full indexers, then one sharing the third's top-k), as `test_dsa_long_rows` cuts it."""

from __future__ import annotations

import pytest

from verity.protocols.resource_accounting.network.verifier import advice as I1
from verity_vllm.program.predict.build import build_program, decoder
from verity_vllm.program.predict.common import Unexpressible
from verity_vllm.program.predict.segment import step_of
from tests.program.predict.test_dsa_long_rows import phi  # noqa: F401  (the cut checkpoint's φ)


def _logits_keys(r) -> list[int]:
    return [n.fn.bindings["N"] for n in r.program.fn.body.nodes if n.fn.id.startswith("DsaMqaLogits_v3{")]


def test_a_step_at_its_history_scores_the_keys_its_rows_had(phi):
    """Two rows at positions 2 and 3 after a history of 2: each full indexer layer's logits read 3 and 4 keys, where the same step
    without its history reads 1 and 2; the history's rows (`Net.prefix_params(2)`) follow its token rows (a rank's `prompt`), and
    its SRC config holds the history."""
    bare = build_program(phi, "step", M=2, pos0=2, with_desc=False)
    held = build_program(phi, "step", M=2, pos0=2, C=2, with_desc=False)
    assert _logits_keys(bare) == [1, 2] * 3 and _logits_keys(held) == [3, 4] * 3
    names, history = [n for n, _ in held.program.fn.signature()[0]], [n for n, _ in decoder(phi, 2, 4).prefix_params(2)]
    assert names[:1 + len(history)] == ["prompt", *history] and names[-1] == "weights"
    assert held.program.fn.bindings["SRC"]["config_sha256"] != bare.program.fn.bindings["SRC"]["config_sha256"]


def test_a_step_history_other_than_its_rows_positions_before_pos0_is_refused(phi):
    with pytest.raises(Unexpressible, match="a history of 1 on a greedy step at pos0 2"):
        build_program(phi, "step", M=2, pos0=2, C=1, with_desc=False)


def test_a_served_step_takes_its_advice_rows_positions_and_history():
    """The capture's decode row (request 0 at position 256 after its 256-row prompt) is a step of one row at pos0 256 after a
    history of 256; a step of two requests' rows is no one sequence's step."""
    assert step_of(I1.StepAdvice((I1.Row(0, "decode", 256, 1),))) == (1, 256, 256)
    with pytest.raises(Unexpressible, match="a step of 2 rows"):
        step_of(I1.StepAdvice((I1.Row(0, "decode", 256, 1), I1.Row(1, "decode", 40, 1))))
