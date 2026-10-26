"""The naive-versus-Strassen cost model (`verity_machine_model.strassen`): its pinned numbers."""
from fractions import Fraction as F

from verity_machine_model.strassen.costs import Costs, inst
from verity_machine_model.strassen.depth import STRASSEN, WINOGRAD, kappa


def test_int8_only_reproduces_the_coordinators_table():
    C = Costs(inst("int8-only"))
    assert C.closure == (256,)
    got = {r: kappa(C, STRASSEN, r, C.lower) for r in (1, 8, 16, 17)}
    assert got == {1: F(5, 2), 8: F(5764545, 4194304), 16: F(287661412107657, 281474976710656),
                   17: F(64265881099529, 70368744177664)}
    assert kappa(C, WINOGRAD, 18, C.lower) == F(17532748091928053, 18014398509481984)


def test_fp16_makes_depth_one_cheaper_on_both_cards():
    for name in ("h200-sheet", "rtx6000-sheet"):
        C = Costs(inst(name))
        up = lambda a, b: C.exact(a, b)[0]
        assert kappa(C, STRASSEN, 1, up) == kappa(C, STRASSEN, 1, C.lower) == F(7, 4)
        assert kappa(C, WINOGRAD, 1, up) == kappa(C, WINOGRAD, 1, C.lower) == F(3, 2)


def test_a_nine_bit_square_is_one_fp16_pair():
    C = Costs(inst("h200-sheet"))
    assert C.c(511, 511) == 2 and C.c(256, 256) == 1 and C.c(4097, 4097) == 4
