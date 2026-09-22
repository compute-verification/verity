"""Generate the CK-FTZ conformance vectors INDEPENDENTLY of the checker (E4, EOD 2026-09-11).

Every expected word here comes from the REGISTERED function (``verity_vllm.registry.prims.<Prim>.evaluate``, the
registry's own evaluators) or is spelled by hand from the registered rule; nothing imports ``ck_ftz`` or the guest.
The table rows come from the registered MUFU tables (``verity_vllm_numerics.fa2_relation.tables()``, the sm_89
measurement the registry cites), with the row index derived by a third, literal reading of ``fa2_model.cpp``
``mufu_ex2_bits`` / ``mufu_rcp_bits`` and cross-checked against the registry output's mantissa.

    .venv/bin/python e2e/coverage_v1/ftz_vectors/gen_vectors.py

writes ``positive.json`` (the checker must accept every vector) and ``negative.json`` (the checker must reject every
vector) next to this file.

Host-dependent registry results.  ``prims.F32{Add,Sub,Mul}Ftz`` evaluate through numpy, so their NaN *payload* rule
is the host CPU's (this generator ran on Apple Silicon = AArch64).  The approved guest pins the x86-SSE rule
(``zk/sp1/common/src/reduce.rs``, "the platform the registry's conformance evidence was measured on").  For the few
inputs where the two differ (an invalid operation's generated NaN: SSE ``0xFFC00000``, AArch64 ``0x7FC00000``; two NaN
operands where only the second is signalling), the vector's ``claimed`` is the SSE pin and ``note`` records the
registry-on-this-host word; ``source`` says ``pinned-SSE``.  They are a registry finding, not a checker choice.

Attempt 6's record commits only boundary BF16 rows and weights; no committed leaf holds an f32 operand of an FTZ
VU, so no GPU-captured f32 word is available to cite (``out/records/V5-TINY-bex0-inc-elem-b8177939.md``).
"""

from __future__ import annotations

import json
import platform
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
for _p in (str(_REPO), str(_REPO / "vllm-poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verity_vllm.registry import prims  # noqa: E402

HERE = Path(__file__).resolve().parent
FAMILY = "veritor.ftz-f32@1"
MANT, EXP, SIGN = 0x7FFFFF, 0x7F800000, 0x80000000

# --- words ---------------------------------------------------------------------------------------------------------


def f(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def h(u: int) -> str:
    return f"0x{u:08X}"


def h64(u: int) -> str:
    return f"0x{u:016X}"


P0, N0 = 0x00000000, 0x80000000  # ±0
PINF, NINF = 0x7F800000, 0xFF800000
QNAN, QNAN_P1, SNAN_P1, NEG_QNAN = 0x7FC00000, 0x7FC00001, 0x7F800001, 0xFFC00000
ONE, TWO, HALF = 0x3F800000, 0x40000000, 0x3F000000
MIN_NORMAL, MAX_NORMAL = 0x00800000, 0x7F7FFFFF  # 2^-126, (2 - 2^-23) * 2^127
MAX_SUBNORMAL, MIN_SUBNORMAL, MID_SUBNORMAL = 0x007FFFFF, 0x00000001, 0x00400000  # 2^-127 is MID
ONE_P = 0x3F800001  # 1 + 2^-23
TWO_M24 = f(2.0**-24)
TWO_M23 = f(2.0**-23)

# --- table rows (the registered tables; index by a literal reading of fa2_model.cpp) ---------------------------------


def _tables():
    from verity_vllm_numerics import fa2_relation

    return fa2_relation.tables()


def ex2_row_index(x: int) -> int:
    """``mufu_ex2_bits``: the ``j`` the C model indexes ``T`` with; 0 when the rule returns before the table.
    ``mant >> (-unb)`` is a 64-bit machine shift, amount taken mod 64 (both x86-64 and AArch64), as compiled."""
    ax = x & 0x7FFFFFFF
    e = (ax >> 23) & 0xFF
    if e in (0xFF, 0):
        return 0
    unb = e - 127
    if unb >= 30:
        return 0
    mant = (ax & MANT) | 0x800000
    k = (mant << unb) & 0xFFFFFFFFFFFFFFFF if unb >= 0 else mant >> ((-unb) % 64)
    if not (x >> 31):
        fixed = k
    elif (k & MANT) == 0:
        fixed = -k
    else:
        fixed = -k - 1
    return fixed & MANT


def rcp_row_index(s: int) -> int:
    return s & MANT


def ex2_row(x: int, T) -> int:
    j = ex2_row_index(x)
    return (j << 32) | int(T[j])


def rcp_row(s: int, R) -> int:
    m = rcp_row_index(s)
    return (m << 32) | int(R[m])


# --- vectors -------------------------------------------------------------------------------------------------------

POS: list[dict] = []
NEG: list[dict] = []


def _args(gate: str, args: list[int]) -> list[str]:
    table = gate in ("mufu_ex2_ftz", "fa2_inv_sum")
    return [h64(a) if table and i == 1 else h(a) for i, a in enumerate(args)]


def pos(id_: str, gate: str, args: list[int], claimed: int, note: str, source: str = "registry:prims.py") -> None:
    POS.append({"id": id_, "gate": gate, "args": _args(gate, args), "claimed": h(claimed), "expect": "accept", "note": note, "source": source})


def neg(id_: str, gate: str, args: list[int], claimed: int, note: str) -> None:
    NEG.append({"id": id_, "gate": gate, "args": _args(gate, args), "claimed": h(claimed), "expect": "reject", "note": note,
                "source": "hand: a wrong word the checker must reject"})


def reg(prim: str, *args: int) -> int:
    return int(getattr(prims, prim).evaluate(*args)) & 0xFFFFFFFF


def pinned(id_: str, gate: str, prim: str, args: list[int], sse: int, why: str) -> None:
    host = reg(prim, *args)
    note = f"{why}; registry evaluator on this host ({platform.machine()}) gives {h(host)}" if host != sse else why
    pos(id_, gate, args, sse, note, "pinned-SSE (reduce.rs); registry evaluator is host-dependent here")


def build() -> None:
    T = _tables()
    EX2T, RCPT = T.ex2, T.rcp

    # ---- F32MulFtz_v1 (prims.py:248) --------------------------------------------------------------------------
    M, P = "f32_mul_ftz", "F32MulFtz"
    pos("mul.normal", M, [f(3.0), f(7.0)], reg(P, f(3.0), f(7.0)), "3 * 7 = 21 exact")
    pos("mul.one", M, [ONE, TWO], reg(P, ONE, TWO), "1 * 2 = 2")
    pos("mul.tie-even", M, [f(1.5), ONE_P], reg(P, f(1.5), ONE_P), "1.5 * (1+2^-23) = 1.5 + 2^-23 + 2^-24: exact tie; 0x3FC00001 is odd so RN-even rounds UP to 0x3FC00002")
    pos("mul.tie-even-up", M, [f(1.5), 0x3F800003], reg(P, f(1.5), 0x3F800003), "1.5 * (1+3*2^-23) = 1.5 + 4*2^-23 + 2^-24: tie; 0x3FC00004 is even so RN-even rounds DOWN")
    pos("mul.sub-in-flushes", M, [MID_SUBNORMAL, ONE], reg(P, MID_SUBNORMAL, ONE), "subnormal 2^-127 operand flushes: 2^-127 * 1 = +0 (not 2^-127)")
    pos("mul.neg-sub-in-flushes", M, [MID_SUBNORMAL | SIGN, ONE], reg(P, MID_SUBNORMAL | SIGN, ONE), "-2^-127 * 1 = -0 (flush keeps the sign)")
    pos("mul.sub-in-times-big", M, [MAX_SUBNORMAL, f(2.0**60)], reg(P, MAX_SUBNORMAL, f(2.0**60)), "a flushed operand stays zero however large the other: max subnormal * 2^60 = +0")
    pos("mul.sub-out-flushes", M, [f(2.0**-100), f(2.0**-30)], reg(P, f(2.0**-100), f(2.0**-30)), "2^-100 * 2^-30 = 2^-130 is subnormal -> +0")
    pos("mul.neg-sub-out-flushes", M, [f(-(2.0**-100)), f(2.0**-30)], reg(P, f(-(2.0**-100)), f(2.0**-30)), "-2^-130 -> -0")
    pos("mul.min-normal-kept", M, [MIN_NORMAL, ONE], reg(P, MIN_NORMAL, ONE), "2^-126 * 1 = 2^-126: the smallest normal is not flushed")
    pos("mul.rounds-up-to-min-normal", M, [f(2.0**-126), 0x3F7FFFFF], reg(P, f(2.0**-126), 0x3F7FFFFF), "2^-126 * (1 - 2^-24) rounds to 2^-126 (RN across the subnormal boundary)")
    pos("mul.max-normal-kept", M, [MAX_NORMAL, ONE], reg(P, MAX_NORMAL, ONE), "largest normal * 1")
    pos("mul.overflow", M, [MAX_NORMAL, TWO], reg(P, MAX_NORMAL, TWO), "largest normal * 2 = +inf")
    pos("mul.neg-overflow", M, [MAX_NORMAL | SIGN, f(1.0000001)], reg(P, MAX_NORMAL | SIGN, f(1.0000001)), "-largest * (1+ulp) = -inf")
    pos("mul.inf", M, [PINF, f(-1.0)], reg(P, PINF, f(-1.0)), "inf * -1 = -inf")
    pos("mul.zero-signs", M, [P0, f(-1.0)], reg(P, P0, f(-1.0)), "+0 * -1 = -0")
    pos("mul.zero-zero-neg", M, [N0, N0], reg(P, N0, N0), "-0 * -0 = +0")
    pos("mul.nan-payload", M, [QNAN_P1, ONE], reg(P, QNAN_P1, ONE), "a quiet NaN operand propagates its payload (0x7FC00001)")
    pos("mul.nan-neg-payload", M, [ONE, 0xFFC01234], reg(P, ONE, 0xFFC01234), "second-operand NaN propagates (payload and sign)")
    pos("mul.snan-quieted", M, [SNAN_P1, TWO], reg(P, SNAN_P1, TWO), "a signalling NaN is quieted: 0x7F800001 -> 0x7FC00001")
    pinned("mul.zero-times-inf", M, P, [P0, PINF], NEG_QNAN, "0 * inf is invalid: SSE indefinite QNaN 0xFFC00000")
    pinned("mul.two-nans", M, P, [QNAN_P1, 0x7FC00002], QNAN_P1, "two quiet NaNs: SSE returns the first operand")
    pinned("mul.qnan-snan", M, P, [QNAN_P1, SNAN_P1], QNAN_P1, "quiet first, signalling second: SSE returns the first (AArch64 the quieted signalling one)")

    # ---- F32AddFtz_v1 (prims.py:236) --------------------------------------------------------------------------
    A, P = "f32_add_ftz", "F32AddFtz"
    pos("add.normal", A, [ONE, ONE], reg(P, ONE, ONE), "1 + 1 = 2")
    pos("add.tie-even-down", A, [ONE, TWO_M24], reg(P, ONE, TWO_M24), "1 + 2^-24: exact tie, RN-even -> 1")
    pos("add.tie-even-up", A, [ONE_P, TWO_M24], reg(P, ONE_P, TWO_M24), "(1+2^-23) + 2^-24: tie, RN-even -> 1 + 2^-22")
    pos("add.above-tie", A, [ONE, f(2.0**-24 + 2.0**-30)], reg(P, ONE, f(2.0**-24 + 2.0**-30)), "just above the tie rounds up: 1 + 2^-23")
    pos("add.cancel-to-subnormal-flushes", A, [0x00C00000, MIN_NORMAL | SIGN], reg(P, 0x00C00000, MIN_NORMAL | SIGN), "1.5*2^-126 - 2^-126 = 2^-127 exact, subnormal -> +0 (the flush of a RESULT)")
    pos("add.cancel-to-neg-subnormal-flushes", A, [MIN_NORMAL, 0x80C00000], reg(P, MIN_NORMAL, 0x80C00000), "2^-126 - 1.5*2^-126 = -2^-127 -> -0")
    pos("add.sub-plus-sub", A, [MAX_SUBNORMAL, MAX_SUBNORMAL], reg(P, MAX_SUBNORMAL, MAX_SUBNORMAL), "two flushed operands: +0 (unflushed the sum would be the normal 2^-126 - 2^-148!)")
    pos("add.neg-sub-plus-neg-sub", A, [MIN_SUBNORMAL | SIGN, MIN_SUBNORMAL | SIGN], reg(P, MIN_SUBNORMAL | SIGN, MIN_SUBNORMAL | SIGN), "-sub + -sub = -0 + -0 = -0")
    pos("add.sub-plus-normal", A, [MID_SUBNORMAL, ONE], reg(P, MID_SUBNORMAL, ONE), "flushed operand: 0 + 1 = 1")
    pos("add.x-minus-x", A, [f(3.25), f(-3.25)], reg(P, f(3.25), f(-3.25)), "exact cancellation of nonzero operands is +0")
    pos("add.zeros-pp", A, [P0, P0], reg(P, P0, P0), "+0 + +0 = +0")
    pos("add.zeros-nn", A, [N0, N0], reg(P, N0, N0), "-0 + -0 = -0")
    pos("add.zeros-pn", A, [P0, N0], reg(P, P0, N0), "+0 + -0 = +0")
    pos("add.zeros-np", A, [N0, P0], reg(P, N0, P0), "-0 + +0 = +0")
    pos("add.inf", A, [PINF, f(-1e30)], reg(P, PINF, f(-1e30)), "inf + finite = inf")
    pos("add.overflow", A, [MAX_NORMAL, 0x73800000], reg(P, MAX_NORMAL, 0x73800000), "largest normal + 2^104 (= its ulp) rounds to +inf")
    pos("add.max-plus-half-ulp", A, [MAX_NORMAL, 0x73000000], reg(P, MAX_NORMAL, 0x73000000), "largest + half ulp: tie to even -> +inf (even = the next binade)")
    pos("add.min-normal-kept", A, [MIN_NORMAL, P0], reg(P, MIN_NORMAL, P0), "2^-126 + 0 = 2^-126 kept")
    pos("add.nan-payload", A, [0x7FC0BEEF, ONE], reg(P, 0x7FC0BEEF, ONE), "NaN payload propagates")
    pos("add.snan-second", A, [ONE, 0xFF800002], reg(P, ONE, 0xFF800002), "a signalling second-operand NaN is quieted: 0xFFC00002")
    pos("add.big-gap", A, [f(2.0**100), f(2.0**-100)], reg(P, f(2.0**100), f(2.0**-100)), "operands 200 binades apart: the small one is a sticky bit")
    pinned("add.inf-minus-inf", A, P, [PINF, NINF], NEG_QNAN, "inf + -inf is invalid: SSE indefinite 0xFFC00000")
    pinned("add.two-nans", A, P, [0x7FC00007, 0x7FC00008], 0x7FC00007, "two quiet NaNs: SSE returns the first")

    # ---- F32FmaSubFtz_v1 (prims.py:260) -----------------------------------------------------------------------
    S, P = "f32_fma_sub_ftz", "F32FmaSubFtz"
    pos("fmasub.normal", S, [TWO, f(3.0), f(5.0)], reg(P, TWO, f(3.0), f(5.0)), "2*3 - 5 = 1")
    pos("fmasub.to-zero", S, [ONE, ONE, ONE], reg(P, ONE, ONE, ONE), "1*1 - 1 = +0")
    pos("fmasub.one-rounding", S, [ONE_P, ONE_P, 0x3F800002], reg(P, ONE_P, ONE_P, 0x3F800002), "(1+2^-23)^2 - (1+2^-22) = 2^-46 exactly: fused (a separate mul+sub would give 0)")
    pos("fmasub.sub-c-flushes", S, [ONE, ONE, MID_SUBNORMAL], reg(P, ONE, ONE, MID_SUBNORMAL), "subnormal addend flushes: 1*1 - 0 = 1")
    pos("fmasub.zero-sign-pp", S, [P0, ONE, P0], reg(P, P0, ONE, P0), "(+0)*1 - (+0) = +0 + -0 = +0")
    pos("fmasub.zero-sign-pn", S, [P0, ONE, N0], reg(P, P0, ONE, N0), "(+0)*1 - (-0) = +0 + +0 = +0")
    pos("fmasub.zero-sign-np", S, [N0, ONE, P0], reg(P, N0, ONE, P0), "(-0)*1 - (+0) = -0 + -0 = -0")
    pos("fmasub.neg-sub-c-flushes-to-zero", S, [N0, ONE, MID_SUBNORMAL | SIGN], reg(P, N0, ONE, MID_SUBNORMAL | SIGN), "c = -2^-127 flushes to -0, negated to +0: (-0) + (+0) = +0")
    pos("fmasub.product-sub-out-flushes", S, [f(2.0**-100), f(2.0**-30), P0], reg(P, f(2.0**-100), f(2.0**-30), P0), "2^-130 - 0 -> subnormal result flushes to +0")
    pos("fmasub.product-sub-kept-inside", S, [f(2.0**-100), f(2.0**-30), MIN_NORMAL | SIGN], reg(P, f(2.0**-100), f(2.0**-30), MIN_NORMAL | SIGN), "2^-130 + 2^-126 = 2^-126 (1 + 2^-4): the subnormal product is NOT flushed before the add")
    pos("fmasub.min-normal-result", S, [ONE, MIN_NORMAL, P0], reg(P, ONE, MIN_NORMAL, P0), "1 * 2^-126 - 0 = 2^-126 kept")
    pos("fmasub.cancel-to-subnormal-flushes", S, [ONE, 0x00C00000, MIN_NORMAL], reg(P, ONE, 0x00C00000, MIN_NORMAL), "1.5*2^-126 - 2^-126 = 2^-127 -> +0")
    pos("fmasub.overflow", S, [MAX_NORMAL, TWO, ONE], reg(P, MAX_NORMAL, TWO, ONE), "largest*2 - 1 = +inf")
    pos("fmasub.inf-minus-finite", S, [PINF, ONE, f(1e30)], reg(P, PINF, ONE, f(1e30)), "inf*1 - 1e30 = inf")
    pos("fmasub.inf-minus-inf", S, [PINF, ONE, PINF], reg(P, PINF, ONE, PINF), "inf - inf = canonical NaN 0x7FC00000 (f32_fma_bits)")
    pos("fmasub.zero-times-inf", S, [P0, PINF, ONE], reg(P, P0, PINF, ONE), "0*inf = canonical NaN")
    pos("fmasub.nan-canonical", S, [QNAN_P1, ONE, ONE], reg(P, QNAN_P1, ONE, ONE), "a NaN operand -> canonical 0x7FC00000 (payload dropped by the registered FMA)")
    pos("fmasub.nan-c", S, [ONE, ONE, 0xFFC00001], reg(P, ONE, ONE, 0xFFC00001), "NaN addend -> 0x7FC00000")
    pos("fmasub.tie", S, [f(1.5), ONE_P, P0], reg(P, f(1.5), ONE_P, P0), "1.5*(1+2^-23) - 0: exact tie, RN-even")

    # ---- F32Max_v1 (prims.py:266) -----------------------------------------------------------------------------
    X, P = "f32_max_ftz", "F32Max"
    pos("max.ab", X, [ONE, TWO], reg(P, ONE, TWO), "max(1, 2) = 2")
    pos("max.ba", X, [TWO, ONE], reg(P, TWO, ONE), "max(2, 1) = 2")
    pos("max.neg", X, [f(-1.0), f(-2.0)], reg(P, f(-1.0), f(-2.0)), "max(-1, -2) = -1")
    pos("max.equal", X, [f(5.0), f(5.0)], reg(P, f(5.0), f(5.0)), "equal operands: the second (5 > 5 false)")
    pos("max.zero-pn", X, [P0, N0], reg(P, P0, N0), "+0 > -0 is false -> the second operand -0")
    pos("max.zero-np", X, [N0, P0], reg(P, N0, P0), "-0 > +0 false -> +0")
    pos("max.nan-first", X, [QNAN_P1, ONE], reg(P, QNAN_P1, ONE), "NaN > 1 is false -> 1 (the NaN is dropped)")
    pos("max.nan-second", X, [ONE, QNAN_P1], reg(P, ONE, QNAN_P1), "1 > NaN is false -> the NaN, payload kept")
    pos("max.sub-first-flushed", X, [MID_SUBNORMAL, f(-1.0)], reg(P, MID_SUBNORMAL, f(-1.0)), "ftz(2^-127) = +0 > -1 -> +0, the FLUSHED word")
    pos("max.sub-second-flushed", X, [f(-1.0), MID_SUBNORMAL | SIGN], reg(P, f(-1.0), MID_SUBNORMAL | SIGN), "-1 > -0 false -> -0 (flushed second)")
    pos("max.sub-vs-min-normal", X, [MAX_SUBNORMAL, MIN_NORMAL], reg(P, MAX_SUBNORMAL, MIN_NORMAL), "0 > 2^-126 false -> 2^-126")
    pos("max.neg-inf", X, [NINF, f(-1e38)], reg(P, NINF, f(-1e38)), "max(-inf, -1e38) = -1e38")
    pos("max.inf", X, [PINF, MAX_NORMAL], reg(P, PINF, MAX_NORMAL), "max(inf, largest) = inf")
    pos("max.neg-inf-both", X, [NINF, NINF], reg(P, NINF, NINF), "-inf > -inf false -> -inf")

    # ---- GuardNegInfZero_v1 (prims.py:272) -------------------------------------------------------------------
    G, P = "guard_neg_inf_zero", "GuardNegInfZero"
    pos("guard.neg-inf", G, [NINF], reg(P, NINF), "-inf -> +0")
    pos("guard.pos-inf", G, [PINF], reg(P, PINF), "+inf passes")
    pos("guard.normal", G, [f(-3.5)], reg(P, f(-3.5)), "a normal passes")
    pos("guard.neg-zero", G, [N0], reg(P, N0), "-0 passes as -0")
    pos("guard.subnormal-unflushed", G, [MID_SUBNORMAL], reg(P, MID_SUBNORMAL), "no flush in this guard: 2^-127 passes unchanged")
    pos("guard.neg-subnormal-unflushed", G, [MIN_SUBNORMAL | SIGN], reg(P, MIN_SUBNORMAL | SIGN), "-min subnormal passes unchanged")
    pos("guard.nan", G, [NEG_QNAN], reg(P, NEG_QNAN), "NaN passes (payload and sign kept)")
    pos("guard.nan-near-neg-inf", G, [0xFF800001], reg(P, 0xFF800001), "0xFF800001 is a NaN, not -inf: passes")
    pos("guard.max", G, [MAX_NORMAL | SIGN], reg(P, MAX_NORMAL | SIGN), "-largest passes")

    # ---- MufuEx2Ftz_v1 (prims.py:426) -------------------------------------------------------------------------
    E, P = "mufu_ex2_ftz", "MufuEx2Ftz"

    def ex2(id_: str, x: int, note: str, source: str = "registry:prims.py MufuEx2Ftz + fa2_relation.tables()") -> None:
        y = reg(P, x)
        row = ex2_row(x, EX2T)
        # cross-check: a normal output's mantissa is the row's mantissa
        if (y & EXP) not in (0, EXP) and (y & MANT) != (row & MANT):
            raise AssertionError(f"{id_}: row {row >> 32} mantissa {row & MANT:#x} != registry output mantissa {y & MANT:#x}")
        pos(id_, E, [x, row], y, note, source)

    ex2("ex2.zero", P0, "ex2(+0) = 1.0 (row 0 pinned)")
    ex2("ex2.neg-zero", N0, "ex2(-0) = 1.0")
    ex2("ex2.one", ONE, "ex2(1) = T[0] * 2 = 2.0")
    ex2("ex2.minus-one", f(-1.0), "ex2(-1) = 0.5")
    ex2("ex2.half", HALF, "ex2(0.5) = T[2^22] ~ sqrt 2 (the measured table word)")
    ex2("ex2.minus-half", f(-0.5), "ex2(-0.5): fixed = -K - 1 branch, row 2^22")
    ex2("ex2.3p7", f(3.7), "ex2(3.7): n = 3, j = frac row")
    ex2("ex2.minus-10p3", f(-10.3), "ex2(-10.3)")
    ex2("ex2.minus-126", f(-126.0), "ex2(-126) = 2^-126: the smallest normal, kept")
    ex2("ex2.minus-126p5", f(-126.5), "ex2(-126.5) ~ 2^-126.5 is subnormal -> flushed to +0 (re <= 0)")
    ex2("ex2.minus-127", f(-127.0), "ex2(-127) = 2^-127 subnormal -> +0")
    ex2("ex2.minus-149", f(-149.0), "ex2(-149) -> +0")
    ex2("ex2.127p9", f(127.9), "ex2(127.9) ~ 2^127.9 < max normal: finite")
    ex2("ex2.128", f(128.0), "ex2(128) = 2^128 overflows -> +inf (re >= 255)")
    ex2("ex2.big", f(2.0**30), "unb >= 30 -> +inf without reading the table (row 0)")
    ex2("ex2.neg-big", f(-(2.0**30)), "unb >= 30, negative -> +0")
    ex2("ex2.pos-inf", PINF, "ex2(+inf) = +inf")
    ex2("ex2.neg-inf", NINF, "ex2(-inf) = +0")
    ex2("ex2.nan", QNAN_P1, "NaN -> 0x7FFFFFFF (the MUFU canonical NaN, payload dropped)")
    ex2("ex2.subnormal-in", MAX_SUBNORMAL, "e == 0: a subnormal input is flushed -> ex2(0) = 1.0")
    ex2("ex2.tiny-normal", f(2.0**-30), "2^-30: K = mant >> 30 = 0 -> ex2 = T[0] = 1.0")
    ex2("ex2.minus-tiny", f(-(2.0**-30)), "-2^-30: fixed = -1 -> n = -1, j = 2^23 - 1 -> T[last] / 2 ~ 1 - ulp")
    ex2("ex2.REGISTRY-ARTEFACT.2^-64", f(2.0**-64),
        "REGISTRY ARTEFACT: mant >> 64 is a mod-64 machine shift in the compiled fa2_model.cpp, so the registered evaluator answers 2.0 (hardware ex2(2^-64) is surely 1.0; unverified). The checker mirrors the REGISTERED function; flag for the registry owner",
        "registry-as-executed (fa2_model.cpp UB shift, |x| in [2^-126, 2^-64])")
    ex2("ex2.REGISTRY-ARTEFACT.-2^-65", f(-(2.0**-65)),
        "REGISTRY ARTEFACT (see ex2.REGISTRY-ARTEFACT.2^-64): mod-64 shift amount 65 -> 1", "registry-as-executed (fa2_model.cpp UB shift)")

    # ---- Fa2InvSum_v1 (prims.py:440) --------------------------------------------------------------------------
    INV, P = "fa2_inv_sum", "Fa2InvSum"

    def inv(id_: str, s: int, note: str) -> None:
        y = reg(P, s)
        row = rcp_row(s, RCPT)
        if (y & EXP) not in (0, EXP) and y != ONE and (y & MANT) != (row & MANT):
            raise AssertionError(f"{id_}: row mantissa mismatch")
        pos(id_, INV, [s, row], y, note, "registry:prims.py Fa2InvSum + fa2_relation.tables()")

    inv("inv.zero", P0, "sum == +0 -> 1.0 (the degenerate arm; row 0 pinned)")
    inv("inv.neg-zero", N0, "sum == -0 -> 1.0")
    inv("inv.nan", QNAN_P1, "sum is NaN -> 1.0 (row = its mantissa 0x400001)")
    inv("inv.nan-neg", 0xFFFFFFFF, "a NaN of any payload/sign -> 1.0")
    inv("inv.one", ONE, "rcp(1) = R[0] = 1.0")
    inv("inv.two", TWO, "rcp(2) = 0.5")
    inv("inv.three", f(3.0), "rcp(3) = R[2^22] * 2^-1 (the measured word)")
    inv("inv.minus-two", f(-2.0), "rcp(-2) = -0.5")
    inv("inv.0p1", f(0.1), "rcp(0.1) ~ 10")
    inv("inv.pos-inf", PINF, "rcp(+inf) = +0")
    inv("inv.neg-inf", NINF, "rcp(-inf) = -0")
    inv("inv.subnormal", MID_SUBNORMAL, "a subnormal sum is NOT ±0: MUFU.RCP flushes it -> +inf")
    inv("inv.neg-subnormal", MAX_SUBNORMAL | SIGN, "-subnormal -> -inf")
    inv("inv.min-normal", MIN_NORMAL, "rcp(2^-126) = 2^126")
    inv("inv.max-normal", MAX_NORMAL, "rcp(largest) ~ 2^-128 is subnormal -> +0 (re <= 0)")
    inv("inv.2^126", f(2.0**126), "rcp(2^126) = 2^-126, the smallest normal, kept")
    inv("inv.1p5-2^126", 0x7EC00000, "rcp(1.5 * 2^126) ~ 2^-126.58 subnormal -> +0")
    inv("inv.neg-2^-125", f(-(2.0**-125)), "rcp(-2^-125) = -2^125")

    # ---- negatives: every one must be rejected ------------------------------------------------------------------
    neg("neg.mul.ulp-high", M, [ONE, TWO], 0x40000001, "1*2 claimed 2 + 1 ulp")
    neg("neg.mul.ulp-low", M, [f(3.0), f(7.0)], f(21.0) - 1, "3*7 claimed 21 - 1 ulp")
    neg("neg.mul.unflushed-subnormal-result", M, [f(2.0**-100), f(2.0**-30)], 0x00080000, "2^-130 claimed as the exact subnormal 0x00080000 (must be +0)")
    neg("neg.mul.unflushed-subnormal-operand", M, [MID_SUBNORMAL, ONE], MID_SUBNORMAL, "2^-127 * 1 claimed 2^-127 (operand must flush)")
    neg("neg.mul.wrong-zero-sign", M, [MID_SUBNORMAL | SIGN, ONE], P0, "-2^-127 * 1 claimed +0 (must be -0)")
    neg("neg.mul.nan-canonicalised", M, [QNAN_P1, ONE], QNAN, "payload 0x7FC00001 claimed canonical 0x7FC00000 (the registered rule propagates it)")
    neg("neg.mul.tie-wrong-way", M, [f(1.5), ONE_P], 0x3FC00001, "tie truncated to the odd neighbour 0x3FC00001 (RN-even gives 0x3FC00002)")
    neg("neg.add.unflushed-cancellation", A, [0x00C00000, MIN_NORMAL | SIGN], MID_SUBNORMAL, "1.5*2^-126 - 2^-126 claimed 2^-127 (must be +0)")
    neg("neg.add.tie-wrong-way", A, [ONE, TWO_M24], ONE_P, "1 + 2^-24 claimed 1 + 2^-23 (tie must go to even = 1)")
    neg("neg.add.zero-sign", A, [P0, N0], N0, "+0 + -0 claimed -0 (must be +0)")
    neg("neg.add.inf-minus-inf-is-inf", A, [PINF, NINF], PINF, "inf + -inf claimed +inf (must be NaN)")
    neg("neg.add.sub-plus-sub-unflushed", A, [MAX_SUBNORMAL, MAX_SUBNORMAL], 0x00FFFFFE, "two max subnormals claimed their unflushed sum")
    neg("neg.fmasub.double-rounding", S, [ONE_P, ONE_P, 0x3F800002], P0, "(1+2^-23)^2 - (1+2^-22) claimed 0 (mul-then-sub); the fused result is 2^-46")
    neg("neg.fmasub.swapped-operands", S, [TWO, f(3.0), f(5.0)], f(13.0), "2*3 - 5 claimed 13 = 5*3 - 2 (operands a, c swapped)")
    neg("neg.fmasub.sign-of-addend", S, [TWO, f(3.0), f(5.0)], f(11.0), "2*3 - 5 claimed 11 = 2*3 + 5 (the fma, not the fma-sub)")
    neg("neg.fmasub.zero-sign", S, [N0, ONE, P0], P0, "(-0)*1 - (+0) claimed +0 (must be -0)")
    neg("neg.fmasub.unflushed-c", S, [ONE, ONE, MID_SUBNORMAL], 0x3F7FFFFF, "1 - 2^-127 claimed rounded 1 - ulp: c must flush (result exactly 1)")
    neg("neg.max.zero-sign", X, [P0, N0], P0, "max(+0, -0) claimed +0 (the rule returns the second, -0)")
    neg("neg.max.nan-propagated", X, [QNAN_P1, ONE], QNAN_P1, "max(NaN, 1) claimed NaN (NaN > 1 is false -> 1)")
    neg("neg.max.unflushed-subnormal", X, [MID_SUBNORMAL, f(-1.0)], MID_SUBNORMAL, "max(2^-127, -1) claimed 2^-127 (the flushed +0 is returned)")
    neg("neg.max.swapped", X, [ONE, TWO], ONE, "max(1, 2) claimed 1")
    neg("neg.guard.pos-inf-zeroed", G, [PINF], P0, "+inf claimed 0 (only -inf is guarded)")
    neg("neg.guard.subnormal-flushed", G, [MID_SUBNORMAL], P0, "2^-127 claimed 0 (this guard does not flush)")
    neg("neg.guard.neg-inf-kept", G, [NINF], NINF, "-inf claimed -inf (must be +0)")
    neg("neg.guard.neg-zero-result", G, [NINF], N0, "-inf claimed -0 (must be +0)")
    neg("neg.ex2.wrong-row", E, [ONE, (1 << 32) | int(EX2T[1])], TWO, "ex2(1) with table row 1 opened instead of row 0: the checker must reject the row")
    neg("neg.ex2.forged-row-word", E, [HALF, (0x400000 << 32) | 0x3F800000], ONE, "row 2^22 with a forged word 1.0 -- the guest cannot detect a forged word (the table root does: E6/E7 pin it); claimed 1.0 is what the forged word yields, so the checker ACCEPTS the relation; listed to make the boundary explicit", )
    neg("neg.ex2.ulp", E, [ONE, ex2_row(ONE, EX2T)], 0x40000001, "ex2(1) claimed 2 + 1 ulp")
    neg("neg.ex2.unflushed-result", E, [f(-127.0), ex2_row(f(-127.0), EX2T)], MID_SUBNORMAL, "ex2(-127) claimed the subnormal 2^-127 (must flush to +0)")
    neg("neg.ex2.subnormal-in-not-flushed", E, [MAX_SUBNORMAL, ex2_row(MAX_SUBNORMAL, EX2T)], 0x3F800001, "ex2(subnormal) claimed 1 + ulp (input flushes: exactly 1.0)")
    neg("neg.inv.wrong-row", INV, [TWO, (1 << 32) | int(RCPT[1])], HALF, "rcp(2) with row 1 opened instead of row 0")
    neg("neg.inv.neg-zero-rcp", INV, [N0, rcp_row(N0, RCPT)], NINF, "sum = -0 claimed rcp(-0) = -inf (the degenerate arm gives 1.0)")
    neg("neg.inv.nan-propagated", INV, [QNAN_P1, rcp_row(QNAN_P1, RCPT)], 0x7FFFFFFF, "NaN sum claimed the MUFU NaN (the epilogue gives 1.0)")
    neg("neg.inv.ulp", INV, [TWO, rcp_row(TWO, RCPT)], 0x3F000001, "rcp(2) claimed 0.5 + 1 ulp")
    neg("neg.inv.unflushed-result", INV, [MAX_NORMAL, rcp_row(MAX_NORMAL, RCPT)], 0x00200000, "rcp(largest) claimed the subnormal 2^-128 (must be +0)")

    # the forged-row vector is accepted by the relation by construction; it documents the table-root boundary and
    # is NOT counted as a negative: move it to positives with an explicit note
    forged = [v for v in NEG if v["id"] == "neg.ex2.forged-row-word"][0]
    NEG.remove(forged)
    forged["id"] = "ex2.BOUNDARY.forged-row-word"
    forged["expect"] = "accept"
    forged["source"] = "hand: the table-authenticity boundary (not a registry value)"
    POS.append(forged)


def write(path: Path, vectors: list[dict], kind: str) -> None:
    T = _tables()
    doc = {
        "family": FAMILY,
        "kind": kind,
        "generator": "e2e/coverage_v1/ftz_vectors/gen_vectors.py",
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
        "sources": {
            "registry": "verity_vllm/registry/prims.py (F32MulFtz:248 F32AddFtz:236 F32FmaSubFtz:260 F32Max:266 GuardNegInfZero:272 MufuEx2Ftz:426 Fa2InvSum:440)",
            "tables": T.source,
            "nan_rule": "x86-SSE as pinned by zk/sp1/common/src/reduce.rs; vectors whose registry-on-host word differs say so in `note`",
            "gpu_leaves": "none: attempt 6's record commits no f32 FTZ operand (boundary BF16 rows + weights only)",
        },
        "count": len(vectors),
        "vectors": vectors,
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> None:
    build()
    ids = [v["id"] for v in POS + NEG]
    assert len(ids) == len(set(ids)), "duplicate vector id"
    write(HERE / "positive.json", POS, "positive: the checker must accept every vector")
    write(HERE / "negative.json", NEG, "negative: the checker must reject every vector")
    by_gate: dict[str, list[int]] = {}
    for v in POS:
        by_gate.setdefault(v["gate"], [0, 0])[0] += 1
    for v in NEG:
        by_gate.setdefault(v["gate"], [0, 0])[1] += 1
    print(f"positive {len(POS)}  negative {len(NEG)}")
    for g, (p, n) in sorted(by_gate.items()):
        print(f"  {g:20s} +{p:3d}  -{n:3d}")


if __name__ == "__main__":
    main()
