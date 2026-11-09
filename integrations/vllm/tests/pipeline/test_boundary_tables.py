"""A two-stage window's history tables (`pipeline.boundary_tables`): their names from public items alone."""

from __future__ import annotations

import pytest

from verity_vllm.commit import history_tables as H
from verity_vllm.pipeline import boundary_tables as BT

RUN = "a" * 128


def test_a_tables_name_binds_the_run_the_class_the_slot_and_the_layer_and_nothing_else():
    base = BT.name(RUN, 4, 8, 2, 7)
    assert base == H.name(f"{RUN}/p4-o8/s2", 7) == f"history/{RUN}/p4-o8/s2/model.layers.7.self_attn.latent"
    others = {BT.name("b" * 128, 4, 8, 2, 7), BT.name(RUN, 8, 8, 2, 7), BT.name(RUN, 4, 16, 2, 7), BT.name(RUN, 4, 8, 3, 7),
              BT.name(RUN, 4, 8, 2, 8)}
    assert len(others) == 5 and base not in others


@pytest.mark.parametrize("args", [("", 4, 8, 0), (RUN, -1, 8, 0), (RUN, 4, 8, -1), (RUN, 4, 8, 1.0), (RUN, 4, True, 0)])
def test_a_key_is_refused_without_a_run_or_with_a_class_or_slot_that_isnt_a_natural_number(args):
    with pytest.raises(ValueError):
        BT.key(*args)
