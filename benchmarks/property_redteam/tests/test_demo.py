"""`demo.py`'s modular arithmetic: every operation a run of the reference evaluator on a registry primitive or on the u8
IMMA step, a product mod p1024 right in both carry forms, at its pinned census."""

from __future__ import annotations

import demo as D
import pytest

CENSUS = {
    "idiv": {"I32Add_v1": 896, "I32Div_v1": 896, "I32Rem_v1": 896, "IMMA.u8.u8.SAT (silicon model)": 768,
             "SelectI32_v1": 128, "U32Ge_v1": 1},
    "cmp": {"I32Add_v1": 15232, "I32Sub_v1": 14336, "IMMA.u8.u8.SAT (silicon model)": 768, "SelectI32_v1": 28800,
            "U32Ge_v1": 14337},
}


@pytest.fixture(scope="module")
def prims():
    return D.primitives()


@pytest.mark.parametrize("carry", sorted(CENSUS))
def test_a_product_mod_p1024_is_right_at_its_census(prims, carry):
    p = D.C.DEFAULT.field.p
    x, y = 0x0123456789ABCDEF ** 16 % p, p - 2
    arith = D.ModArith(prims, carry)
    assert D.undigits(arith.mul(D.digits(x), D.digits(y))) == x * y % p
    assert dict(arith.n) == CENSUS[carry]


def test_the_u8_imma_step_is_the_silicon_models():
    a, b = list(range(200, 232)), list(range(255, 223, -1))
    assert D.evaluate(D.IMMA_U8_STEP, 7, a, b) == (D.IMMA_U8.step(7, a, b),)
    assert D.evaluate(D.IMMA_U8_STEP, D.S32_MAX - 5, a, b) == (D.S32_MAX,)
