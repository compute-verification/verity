"""The WP2 negative patterns (``benchmarks/numerical/tests/checker/test_negative.py``) as exported mutated witnesses.

Each entry keeps the public (x, y) and changes hints exactly as the test does; the manifest records the WP2
predicate that rejects it so the Rust verifier's rejection can be cross-referenced.  Structural mutations
(missing groups, a ``bool`` hint) have no fixed-layout witness and are not exportable; ``y`` mutations are."""
from __future__ import annotations

import copy
from collections.abc import Callable

from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot

from verity_numerical.checker import REAL, Reject, check_step, gen_step
from verity_numerical.checker.gadgets import Bf16Decode
from verity_numerical.checker.tables import align_table, norm_table

__all__ = ["negative_cases"]

P = REAL


def bf16(sign: int, t: int, f: int) -> int:
    return (sign << 15) | (t << 7) | f


def f32(sign: int, t: int, f: int) -> int:
    return (sign << 31) | (t << 23) | f


ONE = bf16(0, 127, 0)


def case(c: int, pairs: list[tuple[int, int]]) -> tuple[int, list[int], list[int], int, dict]:
    pairs = pairs + [(0, 0)] * (16 - len(pairs))
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    y = tc_dot(AMPERE_BF16_M16N8K16, c, a, b)
    return c, a, b, y, gen_step(P, c, a, b)


NONZERO = case(f32(0, 130, 0x123456), [(bf16(0, 128, 3), bf16(1, 126, 5))] * 16)
WITH_ZERO_PRODUCT = case(f32(0, 130, 0), [(ONE, ONE), (0, ONE)] + [(ONE, ONE)] * 14)
SUBNORMAL_STATE = case(f32(0, 0, 0x2A), [(ONE, bf16(0, 20, 0))])
CANCEL = case(0, [(ONE, ONE), (ONE, bf16(1, 127, 0))] * 8)
SHIFTED = case(f32(0, 140, 0), [(bf16(0, 121, 0x55), bf16(0, 121, 0x33))] * 16)
WIDE = case(f32(0, 130, 0x7FFFFF), [(bf16(0, 128, 0x7F), bf16(0, 129, 0x7F))] * 16)
FLUSH_STATE = case(f32(0, 10, 0x7), [(bf16(0, 200, 1), bf16(0, 200, 1))])
OVERFLOW = case(0, [(bf16(0, 254, 0x7F), bf16(0, 254, 0x7F))] * 8)

Mutation = Callable[[dict], None]


def _m(name: str, x: tuple, mutate: Mutation, predicate: str) -> tuple[str, tuple, Mutation | None, int | None, str]:
    return name, x, mutate, None, predicate


def _y(name: str, x: tuple, y_bad: int, predicate: str) -> tuple[str, tuple, Mutation | None, int | None, str]:
    return name, x, None, y_bad, predicate


def _set(path: list, value):
    def f(h):
        cur = h
        for p in path[:-1]:
            cur = cur[p]
        cur[path[-1]] = value
    return f


def _add(path: list, delta: int):
    def f(h):
        cur = h
        for p in path[:-1]:
            cur = cur[p]
        cur[path[-1]] += delta
    return f


def _xor(path: list, mask: int = 1):
    def f(h):
        cur = h
        for p in path[:-1]:
            cur = cur[p]
        cur[path[-1]] ^= mask
    return f


def _many(*fs: Mutation) -> Mutation:
    def f(h):
        for g in fs:
            g(h)
    return f


def _raw_cases() -> list[tuple[str, tuple, Mutation | None, int | None, str]]:
    sr = align_table(P).row(SHIFTED[4]["groups"][0]["max"]["E"] - (2 * (121 - 127)))[1]
    hw = WIDE[4]["groups"][0]
    k = hw["max"]["E"] - max(hw["max"]["E"] + hw["lead"]["l"] - P.width, P.e_min) + P.acc_sig - P.width
    div = norm_table(P).row(k)[1]
    g = NONZERO[4]["groups"][1]["max"]
    floor_sel = [0] * (len(g["sel"]) - 1) + [1]
    sel_prod = [0, 1] + [0] * (len(SHIFTED[4]["groups"][0]["max"]["sel"]) - 2)

    def pack_wrap(h):
        h["pack"]["hidden"] ^= 1
        h["pack"]["low"] += (1 << 23) if h["pack"]["hidden"] == 0 else -(1 << 23)

    def bf16_nonfinite(word):
        def f(h):
            h["a"][0] = Bf16Decode.gen(P, word)
        return f

    cases = [
        _m("bf16.wrapped_reconstruction", NONZERO, _many(_add(["a", 0, "f"], 128), _add(["a", 0, "t"], -1)), "a[0].f"),
        _m("bf16.sign_flip", NONZERO, _xor(["b", 0, "s"]), "b[0].word"),
        _m("bf16.nz_cleared", NONZERO, _set(["a", 0, "nz"], 0), "a[0].nz0"),
        _m("bf16.nz_set_on_zero", WITH_ZERO_PRODUCT, _set(["a", 1, "nz"], 1), "a[1].nz1"),
        _m("fp32.wrapped_reconstruction", NONZERO, _many(_add(["acc", "f"], 1 << 23), _add(["acc", "t"], -1)), "acc.f"),
        _m("fp32.sign_flip", NONZERO, _xor(["acc", "s"]), "acc.word"),
        _m("fp32.false_zero", NONZERO, _set(["acc", "z"], 1), "acc.z0"),
        _m("fp32.false_nonzero", CANCEL, _set(["acc", "z"], 0), "acc.z1"),
        _m("fp32.subnormal_hidden_bit", SUBNORMAL_STATE, _set(["acc", "nz"], 1), "acc.nz1"),
        _m("product.claim_zero", NONZERO, _set(["prod", 0, "z"], 1), "prod[0].z0"),
        _m("product.claim_nonzero", WITH_ZERO_PRODUCT, _set(["prod", 1, "z"], 0), "prod[1].z1"),
        _m("group_max.excessive", NONZERO, _add(["groups", 0, "max", "E"], 1), "g0.max.attained"),
        _m("group_max.excessive_floor_selector", NONZERO,
           _many(_add(["groups", 1, "max", "E"], 3), _set(["groups", 1, "max", "sel"], floor_sel)), "g1.max.attained"),
        _m("group_max.false_selector", SHIFTED,
           _many(_set(["groups", 0, "max", "sel"], sel_prod), _add(["groups", 0, "max", "E"], -25)), "g0.max.d[0]"),
        _m("group_max.two_selectors", SHIFTED, _set(["groups", 0, "max", "sel", 1], 1), "g0.max.onehot"),
        _m("group_max.below_floor", SHIFTED, _set(["groups", 0, "max", "E"], P.floor - 1), "g0.max.E"),
        _m("align_state.r_plus_one", SHIFTED, _add(["groups", 0, "acc_align", "r"], 1), "g0.align_state.div"),
        _m("align_state.wrapped_remainder", SHIFTED,
           _many(_add(["groups", 0, "acc_align", "q"], -1), _add(["groups", 0, "acc_align", "r"], 1)), "g0.align_state.rem"),
        _m("align_state.flushed_claims_quotient", FLUSH_STATE,
           _many(_set(["groups", 0, "acc_align", "q"], 1), _add(["groups", 0, "acc_align", "r"], -(2**P.acc_bits))),
           "g0.align_state.r"),
        _m("align_state.flushed_claims_quotient_keep_r", FLUSH_STATE, _set(["groups", 0, "acc_align", "q"], 1),
           "g0.align_state.div"),
        _m("align_product.wrong_r", SHIFTED, _add(["groups", 0, "prod_align", 0, "r"], 1), "g0.align_prod[0].div"),
        _m("align_product.wrapped", SHIFTED,
           _many(_add(["groups", 0, "prod_align", 0, "q"], 1), _add(["groups", 0, "prod_align", 0, "r"], -sr)),
           "g0.align_prod[0].r"),
        _m("align_product.under_quotient", SHIFTED,
           _many(_add(["groups", 0, "prod_align", 0, "q"], -1), _add(["groups", 0, "prod_align", 0, "r"], sr)),
           "g0.align_prod[0].rem"),
        _m("align_product.steal_unshifted", NONZERO,
           _many(_add(["groups", 0, "prod_align", 0, "q"], -1), _add(["groups", 0, "prod_align", 0, "r"], 1)),
           "g0.align_prod[0].rem"),
        _m("signed_sum.sign_flip", NONZERO, _xor(["groups", 0, "sum", "sgn"]), "g0.sum.v"),
        _m("signed_sum.claim_zero", NONZERO, _set(["groups", 0, "sum", "z"], 1), "g0.sum.z0"),
        _m("signed_sum.claim_nonzero", CANCEL, _set(["groups", 0, "sum", "z"], 0), "g0.sum.z1"),
        _m("signed_sum.negative_zero", CANCEL, _set(["groups", 0, "sum", "sgn"], 1), "g0.sum.zero_sign"),
        _m("leading_bit.minus_one", NONZERO, _add(["groups", 0, "lead", "l"], -1), "g0.lead.window"),
        _m("leading_bit.plus_one", NONZERO, _add(["groups", 0, "lead", "l"], 1), "g0.lead.window"),
        _m("leading_bit.zero_claims_bit", CANCEL, _set(["groups", 0, "lead", "l"], 1), "g0.lead.window"),
        _m("normalise.flip_clamp", NONZERO, _xor(["groups", 0, "norm", "clamp"]), "g0.norm.clamp_proof"),
        _m("normalise.flip_clamp_subnormal", SUBNORMAL_STATE, _xor(["groups", 0, "norm", "clamp"]), "g0.norm.clamp_proof"),
        _m("normalise.m_plus_one_wide", WIDE, _add(["groups", 0, "norm", "M"], 1), "g0.norm.div"),
        _m("normalise.m_plus_one", NONZERO, _add(["groups", 0, "norm", "M"], 1), "g0.norm.div"),
        _m("normalise.wrapped", WIDE,
           _many(_add(["groups", 0, "norm", "M"], -1), _add(["groups", 0, "norm", "r"], div)), "g0.norm.rem"),
        _m("normalise.left_shift_steals_remainder", NONZERO,
           _many(_add(["groups", 0, "norm", "M"], -1), _add(["groups", 0, "norm", "r"], 1)), "g0.norm.rem"),
        _m("state_out.claim_zero", NONZERO, _set(["groups", 1, "out", "z"], 1), "g1.out.z0"),
        _m("state_out.claim_nonzero", CANCEL, _set(["groups", 1, "out", "z"], 0), "g1.out.z1"),
        _m("state_out.claim_overflow", NONZERO, _set(["groups", 1, "out", "ovf"], 1), "g1.out.ovf_proof"),
        _m("state_out.deny_overflow", OVERFLOW, _set(["groups", 0, "out", "ovf"], 0), "g0.out.ovf_proof"),
        _m("pack.wrapped", NONZERO, pack_wrap, "pack.low"),
        _m("pack.hidden_only", NONZERO, _xor(["pack", "hidden"]), "pack.M"),
        _y("output.sign_flip", NONZERO, NONZERO[3] ^ (1 << 31), "y"),
        _y("output.plus_one", NONZERO, NONZERO[3] + 1, "y"),
        _y("overflow.max_finite", OVERFLOW, 0x7F7FFFFF, "y"),
        _y("overflow.neg_inf", OVERFLOW, 0xFF800000, "y"),
        _y("overflow.zero", OVERFLOW, 0, "y"),
    ]
    for word in (0x7F80, 0xFF80, 0x7FC0, 0x7F81):
        c, a, b, y, h = NONZERO
        cases.append((f"bf16.non_finite_{word:04x}", (c, [word] + a[1:], b, y, h), bf16_nonfinite(word), None, "a[0].t"))
    return cases


def negative_cases() -> list[tuple[str, int, list[int], list[int], int, dict, str]]:
    """``(name, c, a, b, y, mutated hints, WP2 predicate)`` for every exportable negative; each is verified to be
    rejected by ``check_step`` at the named predicate before it is returned."""
    out = []
    for name, (c, a, b, y, h), mutate, y_bad, predicate in _raw_cases():
        bad = copy.deepcopy(h)
        if mutate is not None:
            mutate(bad)
        yy = y if y_bad is None else y_bad
        try:
            check_step(P, c, a, b, yy, bad)
        except Reject as e:
            assert predicate in str(e), (name, str(e))
        else:
            raise AssertionError(f"{name}: WP2 accepts the mutation")
        out.append((name, c, a, b, yy, bad, predicate))
    return out
