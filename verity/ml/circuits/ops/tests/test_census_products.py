"""census/v1's `PRODUCTS` names every tensor-core step these Definitions register, word primitive or Boolean Definition, with its k."""
from __future__ import annotations

import re

import verity.ml.circuits.ops.gemm  # noqa: F401  (registers the steps)
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.census import PRODUCTS
from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits.ops.boolean import gemm as BG

DOT = re.compile(r"Dot(\d+)_v\d+$")


def test_every_tensor_core_step_has_its_products():
    steps = {fid: int(m[1]) for fid in REGISTRY.defs if (m := DOT.search(fid))}
    assert len(steps) >= 16
    assert {fid: PRODUCTS.get(fid) for fid in steps} == steps


def test_a_boolean_step_performs_its_word_steps_products():
    pairs = {d.id: word_of(d).id for d in (getattr(BG, n) for n in BG.__all__) if DOT.search(getattr(d, "id", ""))}
    assert len(pairs) == 8
    assert all(PRODUCTS[b] == PRODUCTS[w] for b, w in pairs.items()), pairs
