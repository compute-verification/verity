"""A host core-second's weight on each kind of node, from its node file and cards: the overhead door's GPUs take theirs
from their node files, and the H100 node's host is the H200 node's, with the same SM rate, so its weight is theirs."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity.device import node_kinds
from verity_bench import kit, overhead
from verity_bench.host_weight import cpu_sm_s_per_core_s


@pytest.mark.parametrize("kind, gpu", [("h200-sxm-141gb-x1", "h200"), ("rtx-pro-6000-server-x8", "rtx-pro-6000")])
def test_a_nodes_weight_is_the_one_the_overhead_reads_for_its_card(kind: str, gpu: str) -> None:
    assert overhead.card(gpu).cpu_sm_s_per_core_s == float(cpu_sm_s_per_core_s(kind))


def test_the_weights_exactly() -> None:
    assert {kind: cpu_sm_s_per_core_s(kind) for kind in node_kinds(directory=kit.nodes_directory())} == {
        "h100-sxm-80gb-x1": Fraction(95, 198), "h200-sxm-141gb-x1": Fraction(95, 198),
        "rtx-pro-6000-server-x8": Fraction(1595, 972)}
