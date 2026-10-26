"""D6 v2's size measure (CMP-158; Daniel, 26 Oct 11:46 AM and 12:11 PM PDT: Boolean circuits throughout, size in place of
work): each instruction of 𝓜 (#4662's `Accel` at 08bfdc3a07, `Security/Definitions/Core/Accel/`) as the canonical
Boolean circuit of its semantics, Φ per card, s_×, and the gate prices D5's resources take.

**Gate content.** An instruction's gate content is the size of its circuit traced by `verity.ml.circuits.boolean.trace`:
the And2, Xor2 and Not gates of the builder's circuit (the canonical format's two-input members; constants free). The
tensor-core dots are the repository's exact silicon steps (`silicon.integer.step_s8_bits`, `silicon.fp`'s total dots,
`silicon.fp4.step_total`), one accumulator coordinate of one MMA step; the lane functions and 𝓜's over-approximated
units (`Isa.lean`: DMMA and DFMA as exact integer MACs within ±2^53, HFMA2 within ±2048) are `forms` builders of
`Isa.lean`'s `Lane.eval` and `dmmaEval`. Formats 𝓜 leaves `Open.dot` (f16, tf32, e2m1 on sm_120, every 2:4 sparse dot)
take a stand-in: the format's exact products through the card's pinned FP pipeline, and for a sparse dot the dense dot
of its kept pairs after the selectors' 4:1 multiplexers. A gate count of a builder is an upper bound on the least circuit
for the function; nothing here proves a lower bound.

**Φ** (`phi`): the most gates a card completes a second, max Σ_i x_i g_i over instruction rates x_i ≥ 0 under 𝓜's units
a tick (`H200.lean`'s and `Sm120.lean`'s ρ, `Card.lean`'s `opUse`): tensor and tensorLong, issue, alu, imad, shfl, f64,
smem, l2 and dram. Copies, broadcasts and shuffles compute nothing (0 gates); a reduction into L2 is its adds at L2's
read rate. 𝓜 overlaps every pool in full, so Φ is generous where the card's pools don't overlap (`H200.lean`).

**s_×** (`s_x`): the fewest gates of a counted operation, a MAC of the workloads' formats (s8, E4M3, E5M2) in its dot
step, gates a step over its k. It is read as the least size of any circuit for a counted operation: an assumption (a
builder bounds that least size from above only).
"""
from __future__ import annotations

import functools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from verity.ml.circuits import boolean as BL  # noqa: E402
from verity.ml.circuits.boolean import forms as FM  # noqa: E402
from verity.ml.circuits.boolean.forms import NOT, ONE, ZERO  # noqa: E402
from verity.ml.circuits.boolean.trace import trace  # noqa: E402
from verity.ml.circuits.silicon import fp, fp4, tc_step  # noqa: E402
from verity.ml.circuits.silicon.integer import (BLACKWELL_SM120_S8_M16N8K32,  # noqa: E402
                                                BLACKWELL_SM120_S8_M16N8K32_WRAP, step_s8_bits)
from verity.ml.circuits.types import Array  # noqa: E402

ACCEL_REF = "#4662 at 08bfdc3a07 (Security/Definitions/Core/Accel/{H200,Sm120,Card,Isa}.lean)"


def size(build, widths, out: int) -> dict:
    """The gates of `build` traced on one element a width in `widths` (bits), `out` output bits."""
    tr = trace(build, [Array(w, BL.BIT) for w in widths], Array(out, BL.BIT))
    return dict(tr.counts, total=sum(tr.counts.values()))


# ---------------------------------------------------------------------------------------------------------------------
# Tensor-core dots: one accumulator coordinate of one step, k products.

F16 = tc_step.Fmt("f16", 5, 10, "ieee")
FLOAT = {"e4m3": tc_step.E4M3, "e5m2": tc_step.E5M2, "bf16": tc_step.BF16, "f16": F16, "tf32": tc_step.TF32,
         "e2m1": tc_step.E2M1}
PIPE = {   # (groups' sizes, adder width, floor) a card's FP dot runs at (H200.lean's and Sm120.lean's `captured`)
    "H200": {"e4m3": ((32,), 14, -139), "e5m2": ((32,), 14, -139), "bf16": ((16,), 26, -133)},
    "RTX PRO 6000": {"e4m3": ((32,), 26, -133), "e5m2": ((32,), 26, -133), "bf16": ((16,), 26, -133)},
}
K_STEP = {"s8": 32, "e4m3": 32, "e5m2": 32, "e2m1": 32, "bf16": 16, "f16": 16, "tf32": 8, "nvf4": 64, "mxf4": 64,
          "b1": 256}
OPEN_DOT = {"f16": "bf16", "tf32": "bf16", "e2m1": "e4m3"}   # an Open.dot format's stand-in pipeline


def _float_dot(card: str, f: str, k: int):
    groups, w, floor = PIPE[card][OPEN_DOT.get(f, f)]
    groups = (k,) if sum(groups) != k else groups
    return lambda C, acc, a, b: fp._tc_total(C, acc, a, b, FLOAT[f], groups, w, floor)


def _s8_dot(k: int, sat: bool = True):
    model = replace(BLACKWELL_SM120_S8_M16N8K32 if sat else BLACKWELL_SM120_S8_M16N8K32_WRAP, k_per_instruction=k)
    return lambda C, acc, a, b: step_s8_bits(C, model, acc, a, b)


def _b1_dot(C, acc, a, b):
    """`.and.popc`: acc + popcount(a AND b), modulo 2^32."""
    cols = [[x] for x in acc]
    cols[0] += [C.AND(x[0], y[0]) for x, y in zip(a, b)]
    return FM.compress_columns(C, cols)[:32]


def dot_fn(card: str, f: str, k: int):
    """(C, acc, a, b) -> 32 bits: one coordinate of format f's dot of k products on `card`."""
    if f == "s8":
        return _s8_dot(k)
    if f == "b1":
        return _b1_dot
    return _float_dot(card, f, k)


def bits(f: str) -> int:
    return {"s8": 8, "e4m3": 8, "e5m2": 8, "e2m1": 4, "bf16": 16, "f16": 16, "tf32": 19, "b1": 1}[f]


@functools.cache
def dot_gates(card: str, f: str, sparse: bool = False) -> dict:
    """Gates of one coordinate of format f's MMA step on `card` (k = K_STEP[f]), and its gates a MAC (a real product).
    sparse: the 2:4 form at logical 2k (PTX's sparse k, twice the dense), its k kept pairs each meeting the B element
    its 2-bit selector names (`Isa.lean`'s `mmaSpEval`), the dense dot of those pairs (`Open.dotSp`'s stand-in)."""
    k = K_STEP[f]
    if f in ("nvf4", "mxf4"):
        ns = 4 if f == "nvf4" else 2
        g = size(lambda C, acc, *x: fp4.step_total(C, acc, x[:64], x[64:128], x[128:128 + ns], x[128 + ns:],
                                                   ue8m0=f == "mxf4"),
                 [32] + [4] * 128 + [8] * (2 * ns), 32)
        return dict(g, k=k, per_mac=g["total"] / k)
    w, dot = bits(f), dot_fn(card, f, k)
    if not sparse:
        g = size(lambda C, acc, *ab: dot(C, acc, list(ab[:k]), list(ab[k:])), [32] + [w] * (2 * k), 32)
        return dict(g, k=k, per_mac=g["total"] / k)
    h, k = k, 2 * k

    def build(C, acc, *x):
        a, sel, b = x[:h], x[h:2 * h], x[2 * h:]
        picked = []
        for l in range(h):
            g4 = b[4 * (l // 2):4 * (l // 2) + 4]
            s0, s1 = sel[l]
            picked.append(FM.mux(C, s1, FM.mux(C, s0, g4[3], g4[2]), FM.mux(C, s0, g4[1], g4[0])))
        return dot(C, acc, list(a), picked)

    g = size(build, [32] + [w] * h + [2] * h + [w] * k, 32)
    return dict(g, k=k, per_mac=g["total"] / h)


# ---------------------------------------------------------------------------------------------------------------------
# DMMA and DFMA, over-approximated (`Isa.lean`): exact integer MACs within FP64's ±2^53, low 64 bits.

def _int53(C, v):
    """`Lane.int53`: the 64-bit two's complement v if |v| ≤ 2^53, else 0; as 55 bits sign-extended to 64."""
    fits = NOT(FM.or_reduce(C, [x ^ v[63] for x in v[53:63]]))           # v in [-2^53, 2^53)
    top = FM.and_reduce(C, [v[53]] + [NOT(x) for x in v[54:]] + [NOT(x) for x in v[:53]])   # v = 2^53
    ok = fits ^ top ^ C.AND(fits, top)
    w = [C.AND(ok, x) for x in v[:55]]
    return w + [w[54]] * 9


def _imac64(C, pairs, acc):
    """(acc + Σ a·b) mod 2^64 on 64-bit two's-complement words (sign-extended forms; equal ANDs are shared)."""
    cols = [[x] for x in acc]
    for a, b in pairs:
        for i in range(64):
            for j in range(64 - i):
                cols[i + j].append(C.AND(a[i], b[j]))
    return FM.compress_columns(C, cols)[:64]


@functools.cache
def int53_gates() -> int:
    return size(lambda C, v: _int53(C, v), [64], 64)["total"]


@functools.cache
def dmma_coord_gates(k: int) -> dict:
    """One DMMA coordinate (`dmmaEval`): C + Σ_l A_il B_lj over k, every element decoded by `Lane.int53`."""
    def build(C, acc, *x):
        d = [_int53(C, v) for v in x]
        return _imac64(C, list(zip(d[:k], d[k:])), _int53(C, acc))
    g = size(build, [64] * (2 * k + 1), 64)
    return dict(g, decode=(2 * k + 1) * int53_gates())


def dmma_gates(m: int, n: int, k: int) -> dict:
    """A whole DMMA m×n×k: m·n coordinates sharing the decodes of A's m·k and B's k·n elements."""
    c = dmma_coord_gates(k)
    total = m * n * (c["total"] - c["decode"]) + (m * k + k * n + m * n) * int53_gates()
    return dict(total=total, per_mac=total / (m * n * k))


# ---------------------------------------------------------------------------------------------------------------------
# Lane functions (`Isa.lean`'s `Lane.eval`), 32-bit words LSB first.

def _sext(x, w):
    return list(x) + [x[-1]] * (w - len(x))


def _mul_cols(C, a, b, w):
    cols = [[] for _ in range(w)]
    for i in range(min(len(a), w)):
        for j in range(min(len(b), w - i)):
            cols[i + j].append(C.AND(a[i], b[j]))
    return cols


def _add_rows(C, rows, w):
    cols = [[] for _ in range(w)]
    for r in rows:
        for i, x in enumerate(r[:w]):
            cols[i].append(x)
    return FM.compress_columns(C, cols)[:w]


def _shf(left: bool, clamp: bool):
    def build(C, a, b, c):
        w = list(a) + list(b)
        if left:
            w = w[::-1]
        amt = list(c[:5])
        if clamp:
            big = FM.or_reduce(C, list(c[5:]))
            amt = [x ^ C.AND(big, NOT(x)) for x in amt]    # c ≥ 32: shift 31, then once more below
        for s, sb in enumerate(amt):
            sh = 1 << s
            w = [x ^ C.AND(sb, x ^ (w[i + sh] if i + sh < 64 else ZERO)) for i, x in enumerate(w)]
        if clamp:
            w = [x ^ C.AND(big, x ^ (w[i + 1] if i + 1 < 64 else ZERO)) for i, x in enumerate(w)]
        out = w[:32]
        return out[::-1] if left else out
    return build


def _prmt(C, a, b, c):
    x = list(a) + list(b)
    bytes_ = [x[8 * i:8 * i + 8] for i in range(8)]
    out = []
    for i in range(4):
        s = c[4 * i:4 * i + 4]
        lvl = bytes_
        for sb in s[:3]:
            lvl = [FM.mux(C, sb, lvl[2 * j + 1], lvl[2 * j]) for j in range(len(lvl) // 2)]
        v = lvl[0]
        out += FM.mux(C, s[3], [v[7]] * 8, v)
    return out


def _anf_lop3(lut: int):
    """LOP3 with truth table `lut` by its algebraic normal form, or its complement's and a NOT, whichever is smaller."""
    def anf(t):
        coef = [(t >> i) & 1 for i in range(8)]
        for s in range(3):
            for i in range(8):
                if i >> s & 1:
                    coef[i] ^= coef[i ^ (1 << s)]
        return coef

    def build_for(coef, flip):
        def build(C, a, b, c):
            out = []
            for i in range(32):
                x, y, z = c[i], b[i], a[i]     # Isa.lean: bit t of lut at (a, b, c) = (t/4, t/2, t) mod 2
                ab = C.AND(z, y) if coef[6] or coef[7] else None
                terms = {1: x, 2: y, 4: z, 3: lambda: C.AND(y, x), 5: lambda: C.AND(z, x), 6: lambda: ab,
                         7: lambda: C.AND(ab, x)}
                v = ONE if coef[0] else ZERO
                for m in (1, 2, 4, 3, 5, 6, 7):
                    if coef[m]:
                        t = terms[m]
                        v = v ^ (t() if callable(t) else t)
                out.append(NOT(v) if flip else v)
            return out
        return build

    def gates(t, flip):
        return size(build_for(anf(t), flip), [32] * 3, 32)["total"]

    return min(gates(lut, False), gates(255 - lut, True))


def _cmp_lt(C, a, b, signed):
    if signed:
        a, b = list(a[:31]) + [NOT(a[31])], list(b[:31]) + [NOT(b[31])]
    _, lt = FM.sub(C, a, b, 32)
    return lt


@functools.cache
def lane_gates() -> dict:
    """Gates of each lane function a lane computes (`Isa.lean`'s `Lane`), the largest of its variants."""
    s = {}
    s["iadd3"] = size(lambda C, a, b, c: _add_rows(C, [a, b, c], 32), [32] * 3, 32)
    s["mad"] = size(lambda C, a, b, c: FM.compress_columns(C, [col + [c[i]] for i, col in
                                                               enumerate(_mul_cols(C, a, b, 32))])[:32], [32] * 3, 32)
    s["madWide"] = size(lambda C, a, b, c0, c1: FM.compress_columns(
        C, [col + [(list(c0) + list(c1))[i]] for i, col in enumerate(_mul_cols(C, a, b, 64))])[:64], [32] * 4, 64)
    s["madHi"] = max((size(lambda C, a, b, c, sg=sg: FM.add(C, FM.compress_columns(
        C, _mul_cols(C, _sext(a, 64) if sg else a, _sext(b, 64) if sg else b, 64))[32:64], c, width=32),
        [32] * 3, 32) for sg in (False, True)), key=lambda g: g["total"])
    s["lop3"] = dict(total=max(_anf_lop3(t) for t in range(256)))
    s["shf"] = max((size(_shf(lf, cl), [32] * 3, 32) for lf in (False, True) for cl in (False, True)),
                   key=lambda g: g["total"])
    s["prmt"] = size(_prmt, [32] * 3, 32)
    s["popc"] = size(lambda C, a: (FM.compress_columns(C, [list(a)])[:6] + [ZERO] * 26), [32], 32)
    s["flo"] = size(lambda C, a: (lambda z: [FM.OR(C, NOT(x), z[0]) for x in z[1][:5]] + [z[0]] * 27)(
        FM.lzc(C, list(a))), [32], 32)
    s["setp"] = size(lambda C, a, b: [_cmp_lt(C, a, b, True)] + [ZERO] * 31, [32] * 2, 32)
    s["sel"] = size(lambda C, a, b, c: FM.mux(C, FM.or_reduce(C, list(c)), a, b), [32] * 3, 32)
    s["min"] = size(lambda C, a, b: FM.mux(C, _cmp_lt(C, a, b, True), a, b), [32] * 2, 32)
    s["lea"] = size(lambda C, a, b: FM.add(C, ([ZERO] * 16 + list(a))[:32], b, width=32), [32] * 2, 32)
    s["leaHi"] = size(lambda C, a, b, c: FM.add(C, (list(a) + list(c))[16:48], b, width=32), [32] * 3, 32)
    s["dp4a"] = size(lambda C, a, b, c: _s8_dot(4, sat=False)(C, c, [a[8 * i:8 * i + 8] for i in range(4)],
                                                              [b[8 * i:8 * i + 8] for i in range(4)]), [32] * 3, 32)
    s["ffma"] = size(lambda C, a, b, c: fp.f32_fma(C, a, b, c, nan_word=fp.F32_NAN_PTX), [32] * 3, 32)
    s["fadd"] = size(lambda C, a, b: fp.f32_add(C, a, b, nan_word=fp.F32_NAN_PTX), [32] * 2, 32)
    s["fmul"] = size(lambda C, a, b: fp.f32_mul(C, a, b, nan_word=fp.F32_NAN_PTX), [32] * 2, 32)
    s["dfma"] = dmma_coord_gates(1)

    def hfma2(C, a, b, c):
        out = []
        for h in (0, 1):
            x, y, z = (v[16 * h:16 * h + 16] for v in (a, b, c))
            dec = [_half11(C, v) for v in (x, y, z)]
            cols = [col + [dec[2][i]] for i, col in enumerate(_mul_cols(C, dec[0], dec[1], 16))]
            out += FM.compress_columns(C, cols)[:16]
        return out
    s["hfma2"] = size(hfma2, [32] * 3, 32)
    s["add2"] = size(lambda C, a, b: FM.add(C, a[:16], b[:16], width=16) + FM.add(C, a[16:], b[16:], width=16),
                     [32] * 2, 32)
    s["shfl"] = dict(total=0)
    return {k: v["total"] for k, v in s.items()}


def _half11(C, v):
    """`Lane.half11`: the 16-bit two's complement v if |v| ≤ 2048, else 0."""
    fits = NOT(FM.or_reduce(C, [x ^ v[15] for x in v[11:15]]))
    top = FM.and_reduce(C, [v[11]] + [NOT(x) for x in v[12:]] + [NOT(x) for x in v[:11]])
    ok = fits ^ top ^ C.AND(fits, top)
    w = [C.AND(ok, x) for x in v[:13]]
    return w + [w[12]] * 3


@functools.cache
def red_gates() -> int:
    """A reduction's function of the word it replaces and the new word (`Lane.reduces`: iadd3 with c = 0, min, max,
    LOP3 with c = 0, add2), the largest."""
    two = [size(lambda C, a, b: FM.add(C, a, b, width=32), [32] * 2, 32)["total"],
           size(lambda C, a, b: FM.mux(C, _cmp_lt(C, a, b, True), a, b), [32] * 2, 32)["total"],
           size(lambda C, a, b: [x ^ y ^ C.AND(x, y) for x, y in zip(a, b)], [32] * 2, 32)["total"],
           lane_gates()["add2"]]
    return max(two)


# ---------------------------------------------------------------------------------------------------------------------
# 𝓜's cards: units a tick per SM (ρ over the SMs), the formats' tensor weights, the MMA forms.

ACCEL = {
    "H200": dict(sms=132, mhz=1980, rho=dict(tensor=4096, issue=4, alu=64, imad=64, shfl=32, f64=64, smem=32,
                                             l2=512, dram=20000 / 33 / 132),
                 weight={"s8": 1, "e4m3": 1, "e5m2": 1, "bf16": 2, "f16": 2, "tf32": 4, "b1": 1 / 8},
                 wgmma=True, sparse=("s8", "e4m3", "e5m2", "bf16", "f16", "tf32"), dmma=(16, 8, 16)),
    "RTX PRO 6000": dict(sms=188, mhz=2430, rho=dict(tensor=1024, issue=4, alu=64, imad=64, shfl=32, f64=2, smem=32,
                                                     l2=5, dram=39925 / 243 / 188),
                         weight={"s8": 1, "e4m3": 1, "e5m2": 1, "e2m1": 1, "nvf4": 0.5, "mxf4": 0.5, "bf16": 2,
                                 "f16": 2, "tf32": 4},
                         wgmma=False, sparse=("s8", "e4m3", "e5m2", "e2m1", "bf16", "f16", "tf32"), dmma=(16, 8, 16)),
}
#: b1's tensor weight on the H200 is `Open.b1`, floored at 1/8 (θ's `b1 ≥ 1/8`, CALIB); sm_120 has no b1 MMA.
PIPES = {"iadd3": "alu", "lop3": "alu", "shf": "alu", "prmt": "alu", "popc": "alu", "flo": "alu", "setp": "alu",
         "sel": "alu", "min": "alu", "lea": "alu", "leaHi": "alu", "mad": "imad", "madWide": "imad", "madHi": "imad",
         "dp4a": "imad", "ffma": "fma", "fadd": "fma", "fmul": "fma", "hfma2": "fma", "add2": "fma", "shfl": "shfl",
         "dfma": "f64"}
IMAD_UNITS = {"madWide": 2, "madHi": 2}   # `opUse`: IMAD.WIDE 2, IMAD.HI `Open.madHi` at θ's floor 2


def instructions(card: str) -> list[dict]:
    """Every instruction class of 𝓜's card with its gates and its units an instruction (`opUse`)."""
    a, out = ACCEL[card], []
    for f, wt in a["weight"].items():
        g = dot_gates(card, f)
        m, n = (64, 256) if a["wgmma"] else (16, 8)
        k = K_STEP[f]
        smem = n * k * (bits(f) if f not in ("nvf4", "mxf4") else 4) / 32 if a["wgmma"] else 0.0   # wgmma's B
        out.append(dict(name=f"mma {f}", gates=m * n * g["total"], use=dict(tensor=m * n * k * wt, issue=1, smem=smem),
                        per_unit=g["total"] / (k * wt)))
        if f in a["sparse"]:
            gs = dot_gates(card, f, sparse=True)
            out.append(dict(name=f"mma.sp {f}", gates=m * n * gs["total"],
                            use=dict(tensor=m * n * k * wt, issue=1, smem=2 * smem),
                            per_unit=gs["total"] / (k * wt)))
    m, n, k = a["dmma"]
    d = dmma_gates(m, n, k)
    out.append(dict(name="dmma", gates=d["total"], use=dict(tensor=m * n * k * 32, issue=1), per_unit=d["per_mac"] / 32))
    for g, x in lane_gates().items():
        pipe = PIPES[g]
        use = dict(issue=1 / 32)
        if pipe != "fma":
            use[pipe] = IMAD_UNITS.get(g, 1)
        out.append(dict(name=f"lane {g}", gates=x, use=use, per_unit=x / use.get(pipe, 1 / 32)))
    out.append(dict(name="red.v8 reg→L2", gates=8 * red_gates(), use=dict(issue=1 / 32, l2=8), per_unit=red_gates()))
    return out


@functools.cache
def phi(card: str) -> dict:
    """Φ: max Σ x_i gates_i a SM-clock over 𝓜's units (one LP), then gates a second at 𝓜's tick, with the mix."""
    a, ins = ACCEL[card], instructions(card)
    res = list(a["rho"])
    A = np.array([[i["use"].get(r, 0.0) for i in ins] for r in res])
    sol = linprog(-np.array([i["gates"] for i in ins], dtype=float), A_ub=A, b_ub=[a["rho"][r] for r in res],
                  bounds=[(0, None)] * len(ins), method="highs")
    assert sol.status == 0, sol.message
    per_sm = -sol.fun
    mix = {i["name"]: x * i["gates"] / per_sm for i, x in zip(ins, sol.x) if x * i["gates"] > 1e-9 * per_sm}
    tensor = max((i for i in ins if "tensor" in i["use"]), key=lambda i: i["per_unit"])
    on_tensor = {i["name"] for i in ins if "tensor" in i["use"]}
    return dict(per_sm_clock=per_sm, per_s=per_sm * a["sms"] * a["mhz"] * 1e6, mix=mix, tensor_best=tensor["name"],
                tensor_per_unit=tensor["per_unit"], tensor_share=sum(v for k, v in mix.items() if k in on_tensor),
                s8_mac_s=a["rho"]["tensor"] * a["sms"] * a["mhz"] * 1e6,
                per_s8_mac=per_sm / a["rho"]["tensor"], binding=[r for r, row in zip(res, A)
                                                                if row @ sol.x >= a["rho"][r] * (1 - 1e-9)])


COUNTED = ("s8", "e4m3", "e5m2")


@functools.cache
def s_x() -> dict:
    """s_×: the fewest gates of a counted operation (a MAC of a counted format in its card's dot step), with each."""
    each = {(card, f): dot_gates(card, f)["per_mac"] for card in ACCEL for f in COUNTED}
    (card, f), g = min(each.items(), key=lambda kv: kv[1])
    fp4_ = min(dot_gates("RTX PRO 6000", f)["per_mac"] for f in ("nvf4", "mxf4"))
    return dict(s_x=g, at=(card, f), each={f"{c} {f}": v for (c, f), v in each.items()}, fp4=fp4_)


# ---------------------------------------------------------------------------------------------------------------------
# D5's resources priced in gates (`d6.size_lps`): what the cheapest computing instruction on each costs a unit.

def d5_sigma(card: str, pb_card, bound: str = "lo") -> dict:
    """Gates a unit of each of D5's resources (`public_best.RES`), for the job LP's least size. tensor: SM-clocks at
    θ's s8 rate, so the s8 MMA's gates a SM-clock (f16, 2:4 and FP4 MACs on the same row are priced as s8 MAC-time,
    below their own gates: `tensor_checks`; b1's term is rescaled to its own gates, `bit_size`). alu: IADD3 (hi) or
    LOP3's cheapest two-input function (lo: 32 gates, the least ALU instruction that computes); fma, in D5's slots
    (FFMA 1, IMAD and DP4A 2): the largest (hi) or least (lo) of FFMA, IMAD/2 and DP4A/2; f64 DFMA; xu POPC. issue,
    rf, vec and smem carry no gates (scheduling counts and moves)."""
    lg = lane_gates()
    fma = (lg["ffma"], lg["mad"] / 2, lg["dp4a"] / 2)
    alu = lg["iadd3"] if bound == "hi" else size(lambda C, a, b: [C.AND(x, y) for x, y in zip(a, b)],
                                                 [32] * 2, 32)["total"]
    return dict(tensor=pb_card.s8 * dot_gates(card, "s8")["per_mac"], alu=alu, fma=max(fma) if bound == "hi" else min(fma),
                f64=lg["dfma"], xu=lg["popc"])


def bit_size(card: str) -> float:
    """b c^T's gates a naive MAC over an s8 MAC's (D5's `bit` in s8 MAC-times, priced at the tensor row's gates):
    the H200's b1 wgmma, one b1 MAC; sm_120's FP4 Strassen depth 2 (ALGO-COST A7), (7/8)² NVFP4 MACs at scale 1."""
    s8 = dot_gates(card, "s8")["per_mac"]
    if card == "H200":
        return dot_gates(card, "b1")["per_mac"] / s8
    return (7 / 8) ** 2 * dot_gates(card, "nvf4")["per_mac"] / s8


def tensor_checks(card: str, pb_card) -> dict:
    """The tensor row's other formats in D5, their gates a MAC against the price an s8 MAC-time puts on them (< 1:
    D5's size of a route on them is below the circuit's, so the least size errs low)."""
    s8 = dot_gates(card, "s8")["per_mac"]
    out = {"f16": (2 * s8) / dot_gates(card, "f16")["per_mac"],
           "2:4 s8": (s8 / 2) / (dot_gates(card, "s8", sparse=True)["per_mac"] / 2)}
    if getattr(pb_card, "f4", 0.0):
        out["FP4"] = (s8 * pb_card.s8 / pb_card.f4) / dot_gates(card, "nvf4")["per_mac"]
    return out


def lookup_gates(table_entries: int, width: int, macs: float) -> float:
    """An indexed read's least canonical circuit in this model: 𝓜 has no data-dependent load (`Isa.lean`'s `Op`:
    copies between fixed locations), so a table read is its multiplexer tree, (entries − 1)·width ANDs and twice
    that in XORs (`forms.mux`), over the MACs it stands for."""
    return 3 * (table_entries - 1) * width / macs


def report() -> str:
    L = []
    sx = s_x()
    for card in ACCEL:
        p = phi(card)
        L.append(f"{card}: Φ = {p['per_s']:.4g} gates/s ({p['per_sm_clock']:.4g} a SM-clock; {p['per_s8_mac']:.0f} "
                 f"a tensor s8 MAC-time), tensor best {p['tensor_best']} at {p['tensor_per_unit']:.1f} gates a unit, "
                 f"tensor {p['tensor_share']:.1%} of Φ; binding {p['binding']}")
        L.append("  mix: " + ", ".join(f"{k} {v:.1%}" for k, v in sorted(p["mix"].items(), key=lambda kv: -kv[1])))
    L.append(f"s_× = {sx['s_x']:.1f} gates ({sx['at']}); each: " +
             ", ".join(f"{k} {v:.1f}" for k, v in sx["each"].items()) + f"; FP4 (not counted) {sx['fp4']:.1f}")
    L.append("lanes: " + ", ".join(f"{k} {v}" for k, v in lane_gates().items()))
    return "\n".join(L)


if __name__ == "__main__":
    print(report())
