"""The v2 transition unit and proof block, same signatures as ``checker.step`` (v1) plus the state-chained
forms.

    check_step_expr(ctx, p, c, a, b, h)         word in, word out: the backend transition unit (arbitrary finite c)
    check_unit_expr(ctx, p, state, a, b, h)     state in, (state, ovfs) out: the unit as the VU chains it
    check_vu_expr(ctx, p, a, b, h)              K/16 state-chained units from the constant zero state, one Pack, the
                                                epilogue; every non-final unit asserts its two saturation flags = 0
                                                (silicon: an infinite intermediate accumulator is outside the domain)

The VU's relation ((a, b) -> y16) is v1's; passing the state (s, e_hat, M, z) between units instead of the FP32
word is a proof-internal choice (the intermediate words were never public).  ``check_vu_words_expr`` keeps the
word-chained composition for comparison.

Hints (``gen_step``): ``{"acc": {hdr, f, z} | None, "groups": [{"E", "acc_split": {M_hi, M_lo} | None,
"sum": {sgn, v_hi, v_lo}}]}``: 3 + 6 per group = 15 hint columns per unit at REAL (v1: 225).
"""
from __future__ import annotations

from collections.abc import Sequence

from ..ctx import IntCtx
from ..gadgets import Epilogue
from ..params import REAL, Params, Reject
from .gadgets import (
    GroupMax,
    Normalise,
    OpDecode,
    Pack,
    Product,
    ProductShift,
    SignedSum,
    StateDecode,
    StateShift,
)

__all__ = ["accepts_step", "accepts_vu", "check_step", "check_step_expr", "check_unit_expr", "check_vu",
           "check_vu_expr", "check_vu_words_expr", "contracts", "gen_step", "gen_unit", "gen_vu", "step_output",
           "vu_output"]

def _k(p: Params) -> int:
    return sum(p.groups)


def zero_state(p: Params) -> tuple[int, int, int, int]:
    return 0, p.floor, 0, 1


# ---------------------------------------------------------------------------------------------------------------------
# One unit on a state
# ---------------------------------------------------------------------------------------------------------------------

def _group(ctx, p: Params, state, prods: Sequence, hg: dict, tag: str, static: bool = False):
    """One group: returns the next state (s, e_hat, M, z) and (ovf_eff, sgn).  ``static``: the incoming state is
    the constant zero state (the VU's first unit), whose halves need no hint."""
    s_c, e_c, M_c, _z_c = state
    cands = [e_c] + [e for _, e, _ in prods]
    s4s = [s_c] + [s4 for _, _, s4 in prods]
    E = hg["E"]
    rows = [GroupMax.align(ctx, p, E, c, s, f"{tag}.max[{i}]") for i, (c, s) in enumerate(zip(cands, s4s))]
    dts, sps, iszs = zip(*rows)
    h_split = hg.get("acc_split")
    if h_split is None:
        if not (static and isinstance(M_c, int)):
            raise Reject(f"{tag}.sshift: the state split hint is missing")
        h_split = StateShift.gen(p, M_c)
    t_acc = StateShift.check(ctx, p, M_c, dts[0], sps[0], h_split, f"{tag}.sshift")
    terms = [t_acc] + [ProductShift.check(ctx, p, u, dt, sp, None, f"{tag}.pshift[{i}]")
                       for i, ((u, _, _), dt, sp) in enumerate(zip(prods, dts[1:], sps[1:]))]
    sgn, v_hi, v_lo, l = SignedSum.check(ctx, p, terms, hg["sum"], f"{tag}.sum")
    e_hat, M, z, ovf, isfloor = Normalise.check(ctx, p, E, l, v_hi, v_lo, None, f"{tag}.norm")
    GroupMax.attained(ctx, p, iszs, isfloor, f"{tag}.max")
    s_out = sgn - sgn * z
    return (s_out, e_hat, M, z), (ovf, sgn)


def check_unit_expr(ctx, p: Params, state, a: Sequence, b: Sequence, h: dict, static: bool = False):
    """Every predicate of one unit on the incoming state; returns ``(state_out, [(ovf_eff, sgn) per group])``.
    ``static``: the state is the constant zero state (no split hint for its first group)."""
    k = _k(p)
    if len(a) != k or len(b) != k:
        raise Reject(f"{k} operand pairs expected")
    if len(h["groups"]) != len(p.groups):
        raise Reject(f"{len(p.groups)} groups expected")
    dec_a = [OpDecode.check(ctx, p, a[i], None, f"a[{i}]") for i in range(k)]
    dec_b = [OpDecode.check(ctx, p, b[i], None, f"b[{i}]") for i in range(k)]
    prods = [Product.check(ctx, p, dec_a[i], dec_b[i]) for i in range(k)]
    ovfs = []
    start = 0
    for g, size in enumerate(p.groups):
        state, ov = _group(ctx, p, state, prods[start:start + size], h["groups"][g], f"g{g}", static and g == 0)
        ovfs.append(ov)
        start += size
    return state, ovfs


def check_step_expr(ctx, p: Params, c, a: Sequence, b: Sequence, h: dict):
    """The word-in, word-out transition unit: returns the output word as an expression."""
    state, ovfs = check_unit_expr(ctx, p, StateDecode.check(ctx, p, c, h["acc"]), a, b, h)
    return Pack.check(ctx, p, state, ovfs)


def gen_unit(p: Params, state: tuple[int, int, int, int], a: Sequence[int], b: Sequence[int]) -> tuple[dict, tuple, list]:
    """Honest hints for one unit from a concrete state; returns ``(h, state_out, ovfs)``."""
    k = _k(p)
    if len(a) != k or len(b) != k:
        raise Reject(f"{k} operand pairs expected")
    ctx = IntCtx()
    dec_a = [OpDecode.check(ctx, p, w) for w in a]
    dec_b = [OpDecode.check(ctx, p, w) for w in b]
    prods = [Product.check(ctx, p, da, db) for da, db in zip(dec_a, dec_b)]
    groups, ovfs = [], []
    start = 0
    for size in p.groups:
        group = prods[start:start + size]
        start += size
        s_c, e_c, M_c, _ = state
        cands = [e_c] + [e for _, e, _ in group]
        hg: dict = {"E": GroupMax.gen(p, cands)["E"], "acc_split": StateShift.gen(p, M_c)}
        E = hg["E"]        # the sum's hints need the aligned terms: run the alignment on ints
        s4s = [s_c] + [s4 for _, _, s4 in group]
        rows = [GroupMax.align(ctx, p, E, c, s, "gen") for c, s in zip(cands, s4s)]
        dts, sps, _ = zip(*rows)
        t_acc = StateShift.check(ctx, p, M_c, dts[0], sps[0], hg["acc_split"])
        terms = [t_acc] + [ProductShift.check(ctx, p, u, dt, sp) for (u, _, _), dt, sp in zip(group, dts[1:], sps[1:])]
        hg["sum"] = SignedSum.gen(p, terms)
        state, ov = _group(ctx, p, state, group, hg, "gen")
        groups.append(hg)
        ovfs.append(ov)
    return {"groups": groups}, state, ovfs


def gen_step(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> dict:
    """Honest hints for the word-in transition unit."""
    h_acc = StateDecode.gen(p, c)
    state = StateDecode.check(IntCtx(), p, c, h_acc)
    h, _state, _ovfs = gen_unit(p, state, a, b)
    h["acc"] = h_acc
    return h


def check_step(p: Params, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> None:
    ctx = IntCtx()
    ctx.eq("y", check_step_expr(ctx, p, c, a, b, h), y)


def accepts_step(p: Params, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> bool:
    try:
        check_step(p, c, a, b, y, h)
    except Reject:
        return False
    return True


def step_output(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> int:
    return check_step_expr(IntCtx(), p, c, a, b, gen_step(p, c, a, b))


# ---------------------------------------------------------------------------------------------------------------------
# The proof block
# ---------------------------------------------------------------------------------------------------------------------

def gen_vu(p: Params, a: Sequence[int], b: Sequence[int]) -> dict:
    """Hints of the state-chained VU: one unit witness per k-chunk and the epilogue's."""
    k = _k(p)
    if len(a) != len(b) or not a or len(a) % k:
        raise Reject(f"operand count must be a positive multiple of {k}")
    state = zero_state(p)
    steps = []
    n = len(a) // k
    for i, start in enumerate(range(0, len(a), k)):
        h, state, ovfs = gen_unit(p, state, a[start:start + k], b[start:start + k])
        if i == 0:
            h["groups"][0]["acc_split"] = None       # the constant zero state is split statically: no hint
        steps.append(h)
        if i < n - 1 and any(o for o, _ in ovfs):
            raise Reject(f"unit {i}: saturation before the last unit is outside the domain")
    y32 = Pack.check(IntCtx(), p, state, ovfs)
    return {"steps": steps, "epilogue": Epilogue.gen(p, y32)}


def check_vu_expr(ctx, p: Params, a: Sequence, b: Sequence, h: dict):
    k = _k(p)
    if len(a) != len(b) or not a or len(a) % k or len(h["steps"]) != len(a) // k:
        raise Reject(f"operand count must be a positive multiple of {k} with one unit witness per chunk")
    n = len(a) // k
    state = zero_state(p)
    for i, start in enumerate(range(0, len(a), k)):
        state, ovfs = check_unit_expr(ctx, p, state, a[start:start + k], b[start:start + k], h["steps"][i], static=(i == 0))
        if i < n - 1:
            for g, (ovf, _) in enumerate(ovfs):
                ctx.eq(f"u{i}.g{g}.finite", ovf, 0)
    y32 = Pack.check(ctx, p, state, ovfs)
    return Epilogue.check(ctx, p, y32, h["epilogue"])


def check_vu_words_expr(ctx, p: Params, a: Sequence, b: Sequence, h: dict):
    """The word-chained composition (v1's shape) on the v2 unit, for comparison; ``h`` as ``gen_vu_words``."""
    k = _k(p)
    c = 0
    for i, start in enumerate(range(0, len(a), k)):
        c = check_step_expr(ctx, p, c, a[start:start + k], b[start:start + k], h["steps"][i])
    return Epilogue.check(ctx, p, c, h["epilogue"])


def gen_vu_words(p: Params, a: Sequence[int], b: Sequence[int]) -> dict:
    k = _k(p)
    steps, c = [], 0
    ctx = IntCtx()
    for start in range(0, len(a), k):
        hs = gen_step(p, c, a[start:start + k], b[start:start + k])
        steps.append(hs)
        c = check_step_expr(ctx, p, c, a[start:start + k], b[start:start + k], hs)
    return {"steps": steps, "epilogue": Epilogue.gen(p, c)}


def check_vu(p: Params, a: Sequence[int], b: Sequence[int], y16: int, h: dict) -> None:
    ctx = IntCtx()
    ctx.eq("y16", check_vu_expr(ctx, p, a, b, h), y16)


def accepts_vu(p: Params, a: Sequence[int], b: Sequence[int], y16: int, h: dict) -> bool:
    try:
        check_vu(p, a, b, y16, h)
    except Reject:
        return False
    return True


def vu_output(p: Params, a: Sequence[int], b: Sequence[int]) -> int:
    return check_vu_expr(IntCtx(), p, a, b, gen_vu(p, a, b))


# ---------------------------------------------------------------------------------------------------------------------
# Contracts and the hint layout (for the census and the red-team column lists)
# ---------------------------------------------------------------------------------------------------------------------

def contracts(p: Params = REAL) -> list:
    return [StateDecode.contract(p), OpDecode.contract(p), Product.contract(p), GroupMax.contract(p),
            StateShift.contract(p), ProductShift.contract(p), SignedSum.contract(p), Normalise.contract(p),
            Pack.contract(p), Epilogue.contract(p)]
