"""A-GKR at the E4M3 relations (``fp8-ada``: RTX 4090 ``mma.sync.m16n8k32.e4m3``, two groups of 16; ``fp8-hopper``: H100
``wgmma.m64n8k32.e4m3``, one group of 32; both a 14-bit truncating adder with exponent floor -139): the checker-v2 unit
on the pipeline's *adder-side* view and a chain-end epilogue that proves the packed FP32 word.

Adder-side view.  ``Params(sig=4, exp=4, acc_sig=14, width=14, acc_exp=8, op_nan_rule="e4m3")``: E4M3 operands decoded
by T_OP (the NaN words ``0x7f`` / ``0xff`` have no row), products ``u 2^7`` on the 14-bit adder, the FP32 exponent range
for the state (``acc_e_min = -126``, ``acc_e_max = 127``).  The chained state ``(s, e_hat, M, z)`` is the adder's 14-bit
magnitude, the FP32 word being ``M << 10``.  Silicon truncates the incoming FP32 significand to the adder
(``M24 >> 10``) and takes its zero flag from all 24 bits; on a chained state the low 10 bits are zero, so that rescale is
the identity and the unit needs no rescale gadget (``checker.v2.params.check_widths`` still refuses a lossy view).  The
VU starts from the constant zero state (``c_0 = +0``), every non-final unit asserts its saturation flags zero (an
infinite intermediate accumulator is outside the relation's domain, as in B-Ligero's chain) and the units are
``sum(groups) = 32`` products each: 48 per K = 1536 VU.

Epilogue.  One row per VU: ``one, s, e, M, z, ovf[g], sgn[g], y16`` with ``y16`` asserted equal to ``Pack.check`` of the
last state and flags at the adder-side params, i.e. the 22-bit word ``s 2^21 + t 2^13 + (f >> 10)`` that
``archive/direct/ligero/fp8/relation.pack_public`` makes of the claimed FP32 word (the verifier rejects a claim with
nonzero low 10 bits before packing).  The column keeps the name ``y16`` because the chain binds the public word to the
epilogue column of that name (``gpu/run.read_chain``, ``verifier/src/main.rs``); it holds the relation's public word.
Every linked component is fixed by the last unit's lookups (``z``, ``ovf``, ``e_hat`` LEADNORM outputs, ``sgn`` a
Boolean-checked hint, ``M = v_hi A + TNORM(v_lo, k)`` with every factor a table output), so the packed word is the
integer ``Pack`` of an admissible state, below ``2^22 < p``, and the field equality is the integer one.
"""
from __future__ import annotations

from collections.abc import Sequence

from verity_numerical.checker import Params, Reject
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.v2.gadgets import Pack
from verity_numerical.checker.v2.step import gen_unit, zero_state
from verity_numerical.gkr_export import v2 as GV2
from verity_numerical.gkr_export.circuit import CircuitCtx

P = 2**31 - 2**27 + 1
ACC_FRAC = 23                  # FP32 fraction bits
ACC_EXP = 8                    # FP32 exponent field bits
PACK_DROP = 10                 # low FP32 fraction bits every 14-bit-adder word has zero (pack_public drops them)


def adder_params(model) -> Params:
    """The adder-side checker ``Params`` of an E4M3 :class:`verity.ml.circuits.silicon.models.GroupSum` with a 14-bit adder."""
    from verity.ml.circuits.silicon.models import GroupSum
    from verity.ml.circuits.silicon.term import FORMATS

    if not isinstance(model, GroupSum):
        raise ValueError(f"{getattr(model, 'name', model)!r} is not a GroupSum")
    fmt = FORMATS[model.dtype]
    if fmt.nan_rule != "e4m3":
        raise ValueError(f"{model.name}: the FP8 relations decode E4M3 operands, not {fmt.name}")
    if ACC_FRAC + 1 - model.width != PACK_DROP:
        raise ValueError(f"{model.name}: the 22-bit public packing needs the 14-bit adder, got width {model.width}")
    return Params(sig=fmt.sig_bits, exp=fmt.exp_bits, acc_sig=model.width, width=model.width, floor=model.zero_exponent,
                  groups=tuple(model.groups), acc_exp=ACC_EXP, op_nan_rule="e4m3")


def is_fp8(p: Params) -> bool:
    return p.op_nan_rule == "e4m3"


def relation_params(rel) -> tuple[Params, str]:
    """``(checker Params, epilogue)`` A-GKR proves a ``archive/direct/ligero/relations.py`` relation at: the adder-side
    view and the ``fp8`` epilogue for an E4M3 relation (whose ``rel.params`` is B-Ligero's own ``FP8Params``, a different
    reading of the same fields), else the relation's ``Params`` and the limb epilogue."""
    from verity.ml.circuits.silicon.term import FORMATS

    if FORMATS[rel.model.dtype].nan_rule == "e4m3":
        return adder_params(rel.model), "fp8"
    return rel.params, "limbs"


def unit_k(p: Params) -> int:
    return sum(p.groups)


def build_epilogue_circuit_fp8(p: Params) -> tuple[CircuitCtx, list[str]]:
    """Columns ``one, s, e, M, z, ovf[g], sgn[g] (per group), y16``; ``y16`` asserted equal to the packed word."""
    ctx = CircuitCtx()
    G = len(p.groups)
    state = tuple(ctx.column(n) for n in ("s", "e", "M", "z"))
    flags = [(ctx.column(f"ovf[{g}]"), ctx.column(f"sgn[{g}]")) for g in range(G)]
    y = ctx.column("y16")
    ctx.eq("y16", Pack.check(ctx, p, state, flags), y)
    return ctx, []


def epilogue_witness_fp8(ctx: CircuitCtx, state: Sequence[int], flags: Sequence[tuple[int, int]], y: int) -> list[int]:
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    for i in range(4):
        vals[1 + i] = int(state[i])
    for g, (ovf, sgn) in enumerate(flags):
        vals[5 + 2 * g] = int(ovf)
        vals[6 + 2 * g] = int(sgn)
    vals[5 + 2 * len(flags)] = int(y)
    return GV2.derive(ctx, vals)


def packed_word(p: Params, state, flags) -> int:
    return int(Pack.check(IntCtx(), p, state, flags))


def vu_rows_fp8(unit, epi, a: Sequence[int], b: Sequence[int], y: int, accs: Sequence[int] | None = None,
                p: Params | None = None, cross_check: bool = True) -> tuple[list[list[int]], list[int]]:
    """The unit rows (state-chained from the zero state) and the epilogue row of one VU.  ``accs[i]``: the FP32 word the
    silicon model recorded after unit ``i``; every unit's packed state must equal ``accs[i] >> 10`` with the dropped bits
    zero, so a chain the model computes differently is rejected here, never repaired.  ``cross_check``: the packed final
    state must equal ``y`` (off for negatives, whose wrong claim must reach the proof system) and every row must satisfy
    its circuit mod p."""
    uctx, upaths = unit
    ectx, _ = epi
    k = unit_k(p)
    if len(a) != len(b) or not a or len(a) % k:
        raise Reject(f"operand count must be a positive multiple of {k}")
    steps = len(a) // k
    if accs is not None and len(accs) != steps:
        raise Reject("one recorded accumulator per unit expected")
    rows = []
    state = zero_state(p)
    flags: list[tuple[int, int]] = []
    for i in range(steps):
        ai, bi = list(a[i * k:(i + 1) * k]), list(b[i * k:(i + 1) * k])
        h, state_out, flags = gen_unit(p, state, ai, bi)
        if i < steps - 1 and any(o for o, _ in flags):
            raise Reject(f"unit {i}: saturation before the last unit is outside the domain")
        if accs is not None:
            acc = int(accs[i])
            got = packed_word(p, state_out, flags)
            if acc & (2**PACK_DROP - 1) or got != acc >> PACK_DROP:
                raise Reject(f"unit {i}: v2 packed word {got:#x} differs from the recorded accumulator {acc:#x} >> 10")
        rows.append(GV2.unit_witness_v2(uctx, upaths, state, ai, bi, state_out, h))
        state = state_out
    if cross_check and packed_word(p, state, flags) != int(y):
        raise Reject(f"packed final word {packed_word(p, state, flags):#x} differs from the claim {int(y):#x}")
    erow = epilogue_witness_fp8(ectx, state, flags, y)
    if cross_check:
        for r in rows:
            bad = GV2.verify_row(uctx, r, P)
            if bad is not None:
                raise Reject(f"circuit rejects an honest unit: {bad}")
        bad = GV2.verify_row(ectx, erow, P)
        if bad is not None:
            raise Reject(f"circuit rejects the honest epilogue: {bad}")
    return rows, erow
