"""The naive-versus-Strassen cost model (`verity_machine_model.strassen`): its pinned numbers."""
from fractions import Fraction as F

from verity_machine_model.strassen.costs import Costs, inst
from verity_machine_model.strassen.depth import STRASSEN, WINOGRAD, kappa


def test_int8_only_reproduces_its_published_table():
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


def test_every_tree_with_a_scheme_loses_at_small_depth():
    from verity_machine_model.strassen.sched import STRASSEN as S, TRIVIAL, WINOGRAD as W, Tables, at

    T = Tables("h200-sheet", 3, [S, W, TRIVIAL])
    tabs = T.level(None, None, None, None, 0)
    for l in (1, 2, 3):
        tabs = T.level(*tabs, l)
        lo, up, hlo, hup = tabs
        assert at(T, lo, 256, 256) == at(T, up, 256, 256) == 8 ** (l + 1)
        assert at(T, hlo, 256, 256) == at(T, hup, 256, 256) == 8 ** (l + 1) + 32


def test_output_count_bound_against_enumeration():
    import numpy as np

    from verity_machine_model.strassen.crt import outputs_lower, outputs_upper

    P = np.zeros(32641)
    for x in range(-128, 128):
        P[np.arange(-128, 128) * x + 16256] = 1
    S = P.copy()
    for K in (2, 3):
        L = len(S) + len(P) - 1
        S = (np.fft.irfft(np.fft.rfft(S, L) * np.fft.rfft(P, L), L) > 0.5).astype(float)
        assert outputs_lower(K) <= int(S.sum()) <= outputs_upper(K) == len(S)


def test_residues_mod_2_and_256_cost_strassen_at_every_level():
    from verity_machine_model.strassen.crt import residue_costs

    for l, (lo, up, _) in enumerate(residue_costs("h200-sheet", 2, 4)):
        assert lo == up == 7 ** l  # b1 at 1/8 a MAC, Strassen at every level
    for l, (lo, up, _) in enumerate(residue_costs("h200-sheet", 256, 4)):
        assert lo == up == 8 * 7 ** l


def test_crt_knapsack_is_below_the_explicit_set():
    from verity_machine_model.strassen.crt import best_coprime, knapsack_lower

    costs = {m: 1 for m in range(2, 257)}  # every modulus up to 256 at one int8 product; 256 * 255 * 253 < 16711681
    ub, S = best_coprime(costs, 16711681)
    assert ub == 4 and len(S) == 4
    lo_at = lambda m: costs.get(m, 10 ** 6)
    assert knapsack_lower(lo_at, 16662912, [], min_items=2) == 4  # 256 * 255 * 254 < 16662912
    assert knapsack_lower(lo_at, 16000000, [], min_items=2) == 3
