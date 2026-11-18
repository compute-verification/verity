"""Every word primitive the catalog registers: the gate table's price (`versions.gate_table`) is the count of its body in a
circuit that calls it (`verity.circuits.count`), and one with no Boolean body is unpriced by the one and refused by the other."""
from __future__ import annotations

import pytest

from verity.circuits import boolean as BL
from verity.circuits import count as K
from verity.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import NoBooleanBody, Program, is_input

import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)
from verity_vllm.definitions.registry.catalog import load


def _word_primitives() -> list[PrimitiveDefinition]:
    _, errors = load()
    assert not errors, errors
    V.load()
    return sorted((d for d in list(REGISTRY.defs.values()) if isinstance(d, PrimitiveDefinition) and not is_input(d)
                   and not BL.is_gate(d) and not BL.is_constant_bit(d)), key=lambda d: d.id)


def _calling(p: PrimitiveDefinition) -> Program:
    params, ret = p.signature()
    return Program(bind(CompositeDefinition(f"CountOf[{p.id}]", 1, (), lambda S: (params, ret),
                                            lambda B, S, *a: B.call(p, *a), register=False)))


@pytest.mark.slow
def test_the_gate_tables_price_of_every_word_primitive_is_its_bodys_count():
    """About 35 s: every body the catalog registers is built and counted, which no smaller set covers."""
    agree, unpriced, disagree = 0, [], []
    for p in _word_primitives():
        table, none = V.gate_table([p])
        try:
            got = K.count(_calling(p))[p.id].gates
        except NoBooleanBody:
            assert none == [p.id], p.id
            unpriced.append(p.id)
            continue
        if table.get(p.id) == got:
            agree += 1
        else:
            disagree.append((p.id, table.get(p.id), got))
    assert not disagree, disagree
    assert agree > 150 and len(unpriced) < agree
