"""The transition unit Check(x, y, h) and, by chaining, the proof block.

    x = (c, a[0..k), b[0..k))     one FP32 word and 2k operand words
    y                             the FP32 word tc_dot(c, a, b) (an infinity when a group saturates)
    h                             the hints of every gadget, in data-flow order (see ``gen_step``)

The VU is ``check_vu``: ``K/16`` units chained through their FP32 words from ``c_0 = 0``, then the epilogue cast.
An infinite intermediate word is rejected by the next unit's ``Fp32Decode`` (its exponent field is off the
finite range) -- the obligation is outside the relation, exactly as ``verity_sp1.proofs.families`` rejects it;
an infinite *final* word is cast to the BF16 infinity, as the gate ``f32_to_bf16`` does.
"""
from __future__ import annotations

from collections.abc import Sequence

from .ctx import IntCtx
from .gadgets import (
    AlignProduct,
    AlignState,
    Bf16Decode,
    Epilogue,
    Fp32Decode,
    GroupMax,
    LeadingBit,
    Normalise,
    Pack,
    Product,
    RescaleState,
    SignedSum,
    StateOut,
)
from .params import REAL, Params, Reject

__all__ = ["accepts_step", "accepts_vu", "check_step", "check_step_expr", "check_vu", "check_vu_expr", "gen_step",
           "gen_vu", "step_output", "vu_output"]


def _k(p: Params) -> int:
    return sum(p.groups)


# ---------------------------------------------------------------------------------------------------------------------
# The transition unit
# ---------------------------------------------------------------------------------------------------------------------

def gen_step(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> dict:
    """The honest hints for one unit: every gadget's ``gen`` applied to the values the checker will form."""
    k = _k(p)
    if len(a) != k or len(b) != k:
        raise Reject(f"{k} operand pairs expected")
    h: dict = {"acc": Fp32Decode.gen(p, c), "a": [Bf16Decode.gen(p, w) for w in a],
               "b": [Bf16Decode.gen(p, w) for w in b], "prod": [], "groups": []}
    ctx = IntCtx()
    s_c, e_c, M_c = Fp32Decode.check(ctx, p, c, h["acc"])
    dec_a = [Bf16Decode.check(ctx, p, w, hw) for w, hw in zip(a, h["a"])]
    dec_b = [Bf16Decode.check(ctx, p, w, hw) for w, hw in zip(b, h["b"])]
    prods = []
    for da, db in zip(dec_a, dec_b):
        hp = Product.gen(p, da[1], db[1])
        h["prod"].append(hp)
        prods.append(Product.check(ctx, p, da, db, hp))
    state = (s_c, e_c, M_c)
    start = 0
    for size in p.groups:
        group = prods[start:start + size]
        start += size
        hg: dict = {}
        cands = [state[1]] + [e for _, _, e in group]
        hg["max"] = GroupMax.gen(p, cands)
        E, ds = GroupMax.check(ctx, p, cands, hg["max"])
        P = state[2]
        if p.lossy:                                  # the FP8 adders: truncate the state to W bits first
            hg["acc_rescale"] = RescaleState.gen(p, state[2])
            P = RescaleState.check(ctx, p, state[2], hg["acc_rescale"])
        hg["acc_align"] = AlignState.gen(p, P, ds[0])
        q_acc = AlignState.check(ctx, p, P, ds[0], hg["acc_align"])
        hg["prod_align"] = [AlignProduct.gen(p, u, d) for (u, _, _), d in zip(group, ds[1:])]
        q_prods = [AlignProduct.check(ctx, p, u, d, hq) for (u, _, _), d, hq in zip(group, ds[1:], hg["prod_align"])]
        terms = [(state[0], q_acc)] + [(s, q) for (_, s, _), q in zip(group, q_prods)]
        hg["sum"] = SignedSum.gen(p, terms)
        sgn, v = SignedSum.check(ctx, p, terms, hg["sum"])
        hg["lead"] = LeadingBit.gen(p, v)
        l = LeadingBit.check(ctx, p, v, hg["lead"])
        hg["norm"] = Normalise.gen(p, E, l, v)
        e1, M = Normalise.check(ctx, p, E, l, v, hg["norm"])
        hg["out"] = StateOut.gen(p, e1, M)
        s_out, e_out, M_out, _ovf, _z = StateOut.check(ctx, p, sgn, e1, M, hg["out"])
        state = (s_out, e_out, M_out)
        h["groups"].append(hg)
    h["pack"] = Pack.gen(p, state[2])
    return h


def check_step_expr(ctx, p: Params, c, a: Sequence, b: Sequence, h: dict):
    """Every predicate of one unit on ``ctx``; returns the output word as an expression of ``x`` and ``h``."""
    k = _k(p)
    if len(a) != k or len(b) != k or len(h["a"]) != k or len(h["b"]) != k or len(h["prod"]) != k:
        raise Reject(f"{k} operand pairs expected")
    if len(h["groups"]) != len(p.groups):
        raise Reject(f"{len(p.groups)} groups expected")
    ctx.rng("x.c", c, 0, 2**p.acc_word_bits)
    for i in range(k):
        ctx.rng(f"x.a[{i}]", a[i], 0, 2**p.word_bits)
        ctx.rng(f"x.b[{i}]", b[i], 0, 2**p.word_bits)
    state = Fp32Decode.check(ctx, p, c, h["acc"])
    dec_a = [Bf16Decode.check(ctx, p, a[i], h["a"][i], f"a[{i}]") for i in range(k)]
    dec_b = [Bf16Decode.check(ctx, p, b[i], h["b"][i], f"b[{i}]") for i in range(k)]
    prods = [Product.check(ctx, p, dec_a[i], dec_b[i], h["prod"][i], f"prod[{i}]") for i in range(k)]
    ovfs = []
    start = 0
    z = 0
    for g, size in enumerate(p.groups):
        group = prods[start:start + size]
        start += size
        hg = h["groups"][g]
        if len(hg["prod_align"]) != size:
            raise Reject(f"group {g}: {size} product alignments expected")
        tag = f"g{g}"
        cands = [state[1]] + [e for _, _, e in group]
        E, ds = GroupMax.check(ctx, p, cands, hg["max"], f"{tag}.max")
        P = state[2]
        if p.lossy:
            if "acc_rescale" not in hg:
                raise Reject(f"{tag}.rescale_state: the lossy rescale hints are missing")
            P = RescaleState.check(ctx, p, state[2], hg["acc_rescale"], f"{tag}.rescale_state")
        q_acc = AlignState.check(ctx, p, P, ds[0], hg["acc_align"], f"{tag}.align_state")
        q_prods = [
            AlignProduct.check(ctx, p, u, d, hq, f"{tag}.align_prod[{i}]")
            for i, ((u, _, _), d, hq) in enumerate(zip(group, ds[1:], hg["prod_align"]))
        ]
        terms = [(state[0], q_acc)] + [(s, q) for (_, s, _), q in zip(group, q_prods)]
        sgn, v = SignedSum.check(ctx, p, terms, hg["sum"], f"{tag}.sum")
        l = LeadingBit.check(ctx, p, v, hg["lead"], f"{tag}.lead")
        e1, M = Normalise.check(ctx, p, E, l, v, hg["norm"], f"{tag}.norm")
        s_out, e_out, M_out, ovf_eff, z = StateOut.check(ctx, p, sgn, e1, M, hg["out"], f"{tag}.out")
        ovfs.append((ovf_eff, sgn))
        state = (s_out, e_out, M_out)
    return Pack.check(ctx, p, (*state, z), ovfs, h["pack"])


def check_step(p: Params, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> None:
    """Check(x, y, h) for the transition unit on Python ints; raises ``Reject`` with the first violated predicate."""
    ctx = IntCtx()
    got = check_step_expr(ctx, p, c, a, b, h)
    ctx.eq("y", got, y)


def accepts_step(p: Params, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> bool:
    try:
        check_step(p, c, a, b, y, h)
    except Reject:
        return False
    return True


def step_output(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> int:
    """The output word the checker's own relation assigns to ``x`` (through the honest hints)."""
    return check_step_expr(IntCtx(), p, c, a, b, gen_step(p, c, a, b))


# ---------------------------------------------------------------------------------------------------------------------
# The proof block
# ---------------------------------------------------------------------------------------------------------------------

def gen_vu(p: Params, a: Sequence[int], b: Sequence[int]) -> dict:
    """Hints of the whole chain from ``c_0 = 0``: one unit per k-chunk, then the epilogue's."""
    k = _k(p)
    if len(a) != len(b) or not a or len(a) % k:
        raise Reject(f"operand count must be a positive multiple of {k}")
    steps = []
    c = 0
    ctx = IntCtx()
    for start in range(0, len(a), k):
        hs = gen_step(p, c, a[start:start + k], b[start:start + k])
        steps.append(hs)
        c = check_step_expr(ctx, p, c, a[start:start + k], b[start:start + k], hs)
    return {"steps": steps, "epilogue": Epilogue.gen(p, c)}


def check_vu_expr(ctx, p: Params, a: Sequence, b: Sequence, h: dict):
    k = _k(p)
    if len(a) != len(b) or not a or len(a) % k or len(h["steps"]) != len(a) // k:
        raise Reject(f"operand count must be a positive multiple of {k} with one unit witness per chunk")
    c = 0
    for s, start in enumerate(range(0, len(a), k)):
        c = check_step_expr(ctx, p, c, a[start:start + k], b[start:start + k], h["steps"][s])
    return Epilogue.check(ctx, p, c, h["epilogue"])


def check_vu(p: Params, a: Sequence[int], b: Sequence[int], y16: int, h: dict) -> None:
    """Check for GemmCoordinate<K>: the chain from +0 and the RN-even cast; raises ``Reject``."""
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


def contracts(p: Params = REAL) -> list:
    """Every gadget's contract at ``p``, in data-flow order (``RescaleState`` only when the adder is narrower than
    the accumulator; ``Epilogue`` only when the operand format shares the accumulator's exponent width)."""
    return [
        Fp32Decode.contract(p), Bf16Decode.contract(p), Product.contract(p), GroupMax.contract(p),
        *([RescaleState.contract(p)] if p.lossy else []),
        AlignState.contract(p), AlignProduct.contract(p), SignedSum.contract(p), LeadingBit.contract(p),
        Normalise.contract(p), StateOut.contract(p), Pack.contract(p),
        *([Epilogue.contract(p)] if p.acc_exp == p.exp else []),
    ]
