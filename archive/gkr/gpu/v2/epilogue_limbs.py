"""The v2 epilogue on 16-bit limbs (BabyBear-sound Pack + cast), the F4-class repair of ``checker_min.md`` 3.2.

``checker.v2.gadgets.Pack.check`` forms the FP32 word ``s 2^31 + fld 2^23 + M`` as one field element and
``Epilogue.check`` splits it as ``hi 2^16 + lo``; over a 31-bit field a second ``(hi, lo)`` satisfies the congruence
(z3: SAT on M31/BabyBear).  ``pack_limbs`` / ``epilogue_limbs`` below compute the *same function over Z* on two
16-bit limbs -- ``M = M_hi8 2^16 + M_lo16`` (two range-checked hints), the finite high limb as the expression
``s 2^15 + fld 2^7 + M_hi8`` (``< 2^16`` on every admissible state, no wrap), the infinity blend per limb, and v1's
epilogue on the limb pair (PROTOCOL.md 13, rule LIMB16).  ``y_hi 2^16 + y_lo`` equals ``Pack.word`` identically
(``test_pack_limbs_matches_pack`` in ``tests/gkr_export/test_gpu_v2_epilogue.py``), so the relation is unchanged;
what changes is the embedding.

The two gadgets are those of ``lane/b-air-v2`` (``verity_numerical.redteam.z3_air_v2.pack_limbs`` /
``epilogue_limbs``, ``V2_AIR.md`` 4.5), whose z3 report ``b_air_v2_z3.json`` records ``pack_epilogue_limbs``:
integer UNSAT, wrap-free UNSAT and direct UNSAT over BabyBear at REAL widths (hints symbolic; premises = the state
is admissible, which the unit body's LEADNORM outputs and range checks establish and the chain links carry).
They are copied here verbatim until that branch is merged; the import switches then.

Epilogue columns: ``one, s, e, M, z, ovf[g], sgn[g] (per group), y16, h.M_hi8, h.M_lo16, h.hi2, h.lsb, h.carry,
h.rem`` + derived (the same 9 leading columns as the a-v2 epilogue, so ``chain.txt`` is unchanged).
"""
from __future__ import annotations

from collections.abc import Sequence

from verity_numerical.checker import REAL, Params
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.v2.gadgets import Pack
from verity_numerical.gkr_export.circuit import CircuitCtx
from verity_numerical.gkr_export.v2 import derive

__all__ = ["build_epilogue_circuit_limbs", "epilogue_limbs", "epilogue_witness_limbs", "pack_limbs", "pack_word_limbs"]

HINTS = ("M_hi8", "M_lo16", "hi2", "lsb", "carry", "rem")


# --- lane/b-air-v2, redteam/z3_air_v2.py (verbatim) -------------------------------------------------------------------

def pack_limbs(ctx, p: Params, state, ovfs: Sequence, h: dict, tag="pack"):
    """(s, e_hat, M, z) and the groups' (ovf, sgn) -> (y_hi, y_lo): M = M_hi8 2^16 + M_lo16; the finite high limb
    is the expression s 2^15 + fld 2^7 + M_hi8."""
    s, e_hat, M, z = state
    M_hi8, M_lo16 = h["M_hi8"], h["M_lo16"]
    ctx.rng(f"{tag}.M_hi8", M_hi8, 0, 2**8)
    ctx.rng(f"{tag}.M_lo16", M_lo16, 0, 2**16)
    ctx.eq(f"{tag}.M", M_hi8 * 2**16 + M_lo16, M)
    fld = e_hat + (p.bias - 1) + z * (p.e_min - p.floor)
    y_hi = s * 2**15 + fld * 2**7 + M_hi8
    y_lo = M_lo16
    any_ovf, inf_sign, none_yet = 0, 0, 1
    for ovf_g, sgn_g in ovfs:
        first = none_yet * ovf_g
        any_ovf = any_ovf + first
        inf_sign = inf_sign + first * sgn_g
        none_yet = none_yet - first
    inf_hi = inf_sign * 2**15 + p.t_finite * 2**7
    return y_hi + any_ovf * (inf_hi - y_hi), y_lo - any_ovf * y_lo


def pack_gen(p: Params, M: int) -> dict:
    return {"M_hi8": M >> 16, "M_lo16": M & 0xFFFF}


def epilogue_limbs(ctx, p: Params, y_hi, y_lo, h: dict, tag="cast"):
    hi2, lsb, carry, rem = h["hi2"], h["lsb"], h["carry"], h["rem"]
    ctx.rng(f"{tag}.hi", y_hi, 0, 2**16)
    ctx.rng(f"{tag}.lo", y_lo, 0, 2**16)
    ctx.rng(f"{tag}.hi2", hi2, 0, 2**15)
    ctx.boolean(f"{tag}.lsb", lsb)
    ctx.boolean(f"{tag}.carry", carry)
    ctx.rng(f"{tag}.rem", rem, 0, 2**16)
    ctx.eq(f"{tag}.lsb_split", 2 * hi2 + lsb, y_hi)
    ctx.eq(f"{tag}.round", carry * 2**16 + rem, y_lo + lsb + (2**15 - 1))
    y16 = y_hi + carry
    ctx.rng(f"{tag}.y16", y16, 0, 2**16)
    return y16


def epilogue_gen(p: Params, y_hi: int, y_lo: int) -> dict:
    lsb = y_hi & 1
    t = y_lo + lsb + (2**15 - 1)
    return {"hi2": y_hi >> 1, "lsb": lsb, "carry": t >> 16, "rem": t & 0xFFFF}


# --- the circuit and its witness ----------------------------------------------------------------------------------------

def pack_word_limbs(p: Params, state, flags) -> tuple[int, int, int]:
    """``(y_hi, y_lo, y16)`` of the limb gadgets on integers (the witness generator's reference)."""
    ctx = IntCtx()
    M = int(state[2])
    h = pack_gen(p, M)
    y_hi, y_lo = pack_limbs(ctx, p, tuple(int(v) for v in state), [(int(o), int(s)) for o, s in flags], h)
    h.update(epilogue_gen(p, int(y_hi), int(y_lo)))
    y16 = epilogue_limbs(ctx, p, y_hi, y_lo, h)
    return int(y_hi), int(y_lo), int(y16)


def build_epilogue_circuit_limbs(p: Params = REAL) -> tuple[CircuitCtx, list[str]]:
    """Columns ``one, s, e, M, z, ovf[g], sgn[g], y16, h.*`` (``HINTS`` order) + derived; ``y16`` asserted equal to
    the cast of the limb pair."""
    ctx = CircuitCtx()
    G = len(p.groups)
    state = tuple(ctx.column(n) for n in ("s", "e", "M", "z"))
    flags = [(ctx.column(f"ovf[{g}]"), ctx.column(f"sgn[{g}]")) for g in range(G)]
    y16 = ctx.column("y16")
    h = {k: ctx.column(f"h.{k}") for k in HINTS}
    y_hi, y_lo = pack_limbs(ctx, p, state, flags, h)
    got = epilogue_limbs(ctx, p, y_hi, y_lo, h)
    ctx.eq("y16", got, y16)
    return ctx, [f"h.{k}" for k in HINTS]


def epilogue_witness_limbs(ctx: CircuitCtx, paths: list[str], state: Sequence[int], flags: Sequence[tuple[int, int]],
                           y16: int, p: Params = REAL) -> list[int]:
    """The epilogue row from the last state and flags; ``y16`` is the claimed word (honest: ``pack_word_limbs``)."""
    M = int(state[2])
    h = pack_gen(p, M)
    y_hi, y_lo, _ = pack_word_limbs(p, state, flags)
    h.update(epilogue_gen(p, y_hi, y_lo))
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    for i in range(4):
        vals[1 + i] = int(state[i])
    for g, (ovf, sgn) in enumerate(flags):
        vals[5 + 2 * g] = int(ovf)
        vals[6 + 2 * g] = int(sgn)
    off = 5 + 2 * len(flags)
    vals[off] = int(y16)
    for i, path in enumerate(paths):
        vals[off + 1 + i] = int(h[path[2:]])
    return derive(ctx, vals)


def pack_check_word(p: Params, state, flags) -> int:
    """``Pack.check`` on integers (the a-v2 epilogue's word), for the equivalence test."""
    return int(Pack.check(IntCtx(), p, tuple(int(v) for v in state), [(int(o), int(s)) for o, s in flags]))
