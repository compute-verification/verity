"""census/v1's `PRODUCTS` names every tensor-core step these Definitions register, word primitive or Boolean Definition, with its k,
and `PRECISION` each one's operand format."""
from __future__ import annotations

import re

import verity.ml.circuits.ops.gemm  # noqa: F401  (registers the steps)
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.census import PRECISION, PRODUCTS
from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits.ops.boolean import gemm as BG

DOT = re.compile(r"Dot(\d+)_v\d+$")


def test_every_tensor_core_step_has_its_products():
    steps = {fid: int(m[1]) for fid in REGISTRY.defs if (m := DOT.search(fid))}
    assert len(steps) >= 16
    assert {fid: PRODUCTS.get(fid) for fid in steps} == steps


def test_a_boolean_step_performs_its_word_steps_products():
    pairs = {d.id: word_of(d).id for d in (getattr(BG, n) for n in BG.__all__) if DOT.search(getattr(d, "id", ""))}
    assert len(pairs) == 9
    assert all(PRODUCTS[b] == PRODUCTS[w] and PRECISION[b] == PRECISION[w] for b, w in pairs.items()), pairs


def test_every_tensor_core_step_has_its_precision():
    assert set(PRECISION) == set(PRODUCTS) and set(PRECISION.values()) == {"bf16", "tf32", "fp8", "fp4", "int8"}
    assert {f: PRECISION[f] for f in PRODUCTS if "E5m2" in f or "Mxf4" in f or "U8S8" in f} == {
        "BlackwellE5m2QmmaDot32_v1": "fp8", "BlackwellE5m2QmmaDot32_v2": "fp8", "BlackwellMxf4OmmaDot64_v1": "fp4",
        "BlackwellMxf4OmmaDot64_v2": "fp4", "PouwDot16U8S8_v1": "int8", "BlackwellU8S8ImmaDot32_v1": "int8",
        "BlackwellU8S8ImmaDot32_v2": "int8"}
