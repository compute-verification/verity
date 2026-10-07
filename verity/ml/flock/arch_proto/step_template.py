"""Prototype (lane proofs-arch; not a production path): the sm_120 BF16 k16 step as a slot template over pre-decoded operands.

Today a GemmCoordinate{K} unit is one expanded circuit per coordinate (K = 2048: 1,103,522 ANDs, 1,169,281 rows in a 2^21 slot, 56%
full; M0 #20). Its 128 k16 steps are identical, and each step decodes all 32 of its operand words again, although in a P x C tile
every x word is read by C coordinates and every w word by P. This splits the step into

* ``decode_word``: the per-word part (classes, the finite stand-in, the hidden bit, the effective exponent), done once per
  operand word on the row side, and
* ``step_decoded``: the rest of ``verity_flock.fp.tc_dot16`` (groups (16,), W = 26, F = -133: HopperBF16WgmmaDot16_v1, which the
  sm_120 step uses) reading those fields,

and checks that decode + step equals ``fp.tc_dot16`` bit for bit (bit-sliced, on random words with NaN, Inf, zero and subnormal
operands) and the core reference ``verity.ml.circuits.silicon.total.tc_dot_total`` on HOPPER_BF16_WGMMA_K16. The census says whether one step fits a
2^13-row slot (inputs, ANDs, output copies and the constant row), which halves the GEMM half of every GEMM statement's bits.

    python step_template.py [--T 8192] [--ref 400] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import random
import time

from verity_flock import fp
from verity_flock import unit as U
from verity_flock.gf2 import ONE, ZERO, Circuit, NOT, OR, add, const_bits, multiply, mux, or_reduce

FIELDS = (("s", 1), ("e", 8), ("sig", 8), ("nz", 1), ("nan", 1), ("inf", 1), ("zero", 1))
DECODED_BITS = sum(n for _, n in FIELDS)


def decode_word(C: Circuit, x: list) -> dict:
    """A bf16 word's fields as ``fp._tc_total`` derives them: its classes (nan, inf, zero), and the finite stand-in's sign,
    effective exponent, significand with hidden bit and nonzero flag."""
    fmt = U.BF16
    nan, inf, zero = fp._classes(C, x, fmt.man_bits, fmt.exp_bits)
    s, e, sig, nz = fp._tc_operand(C, fmt, fp._finite_or_zero(C, x, fmt.man_bits, fmt.exp_bits))
    return {"s": [s], "e": e, "sig": sig, "nz": [nz], "nan": [nan], "inf": [inf], "zero": [zero]}


def decoded_input(C: Circuit, name: str) -> dict:
    return {f: C.input(f"{name}.{f}", n) for f, n in FIELDS}


def step_decoded(C: Circuit, acc: list, A: list[dict], B: list[dict], groups=(16,), W: int = 26, F: int = -133) -> list:
    """``fp._tc_total`` on decoded operands: the same gates after each operand's decode, in the same order."""
    fmt = U.BF16
    rescale = W - 24
    one = lambda d, f: d[f][0]  # noqa: E731
    p_nan = [OR(C, OR(C, one(a, "nan"), one(b, "nan")), OR(C, C.AND(one(a, "inf"), one(b, "zero")), C.AND(one(b, "inf"), one(a, "zero"))))
             for a, b in zip(A, B)]
    p_inf = [C.AND(OR(C, one(a, "inf"), one(b, "inf")), NOT(n)) for (a, b), n in zip(zip(A, B), p_nan)]
    p_sign = [one(a, "s") ^ one(b, "s") for a, b in zip(A, B)]
    c_nan, c_inf, _ = fp._classes(C, acc, 23, 8)
    C.phase = "products"
    prods = []
    for a, b in zip(A, B):
        S = add(C, a["e"], b["e"], width=fmt.exp_bits + 1)
        S = add(C, S, const_bits((fmt.offset + U.S_BIAS) % 1024, 10), width=10)
        mag = multiply(C, a["sig"], b["sig"])[: 2 * fmt.sig_bits]
        sh = fmt.product_shift + rescale
        prods.append((one(a, "s") ^ one(b, "s"), S, C.AND(one(a, "nz"), one(b, "nz")), ([ZERO] * sh + mag) if sh >= 0 else mag[-sh:]))
    C.phase = "accumulator + specials"
    c = fp._finite_or_zero(C, acc, 23, 8)
    cm, cf, cs = c[:23], c[23:31], c[31]
    hc = or_reduce(C, cf)
    cf_eff = [cf[0] ^ C.AND(NOT(cf[0]), NOT(hc))] + cf[1:]
    mag24 = cm + [hc]
    state = (cs, add(C, cf_eff, const_bits(127, 10), width=10), OR(C, hc, or_reduce(C, cm)),
             ([ZERO] * rescale + mag24) if rescale >= 0 else mag24[-rescale:])
    nan, pos, neg = c_nan, C.AND(c_inf, NOT(acc[31])), C.AND(c_inf, acc[31])
    sats, start, m24 = [], 0, None
    for size in groups:
        for i in range(start, start + size):
            nan = OR(C, nan, p_nan[i])
            pos = OR(C, pos, C.AND(p_inf[i], NOT(p_sign[i])))
            neg = OR(C, neg, C.AND(p_inf[i], p_sign[i]))
        nan = OR(C, nan, C.AND(pos, neg))
        s_o, S_o, nz_o, m24 = U.group_sum(C, [state] + prods[start:start + size], W, F)
        C.phase = "pack + specials"
        sat = U.saturated(C, S_o, nz_o)
        sats.append((sat, s_o))
        fin_sat = C.AND(sat, NOT(OR(C, pos, neg)))
        pos, neg = OR(C, pos, C.AND(fin_sat, NOT(s_o))), OR(C, neg, C.AND(fin_sat, s_o))
        state = (s_o, S_o, nz_o, ([ZERO] * rescale + m24) if rescale >= 0 else m24[-rescale:])
        start += size
    s_o, S_o, nz_o = state[0], state[1], state[2]
    field = add(C, S_o[:8], const_bits((-128) % 256, 8), cin=m24[23], width=8)
    sign_n = C.AND(s_o, nz_o)
    field_n = [C.AND(nz_o, x) for x in field]
    sat_any, sat_sign = sats[-1]
    for sat_g, sgn in reversed(sats[:-1]):
        sat_sign = mux(C, sat_g, [sgn], [sat_sign])[0]
        sat_any = OR(C, sat_g, sat_any)
    finite = [C.AND(NOT(sat_any), x) for x in m24[:23]] + [OR(C, sat_any, x) for x in field_n] + [mux(C, sat_any, [sat_sign], [sign_n])[0]]
    inf = OR(C, pos, neg)
    special = fp.select(C, nan, const_bits(fp.TC_NAN, 32), const_bits(fp.F32_INF, 31) + [neg])
    return fp.select(C, OR(C, nan, inf), special, finite)


def rows(counts: dict) -> int:
    """Committed rows of a slot holding the circuit: inputs, ANDs, output copies and the constant row."""
    return counts["inputs"] + counts["ands"] + counts["output_copies"] + 1


# --- test vectors -----------------------------------------------------------------------------------------------------------------

def rand_bf16(rng: random.Random, finite: bool = False) -> int:
    r = rng.random()
    if finite and r < 0.05:
        r = 0.05 + 0.95 * rng.random()
    if r < 0.03:
        return rng.choice([0x7F80, 0xFF80])                              # +-inf
    if r < 0.05:
        return (rng.getrandbits(1) << 15) | 0x7F80 | rng.randrange(1, 128)  # NaN
    if r < 0.12:
        return rng.getrandbits(1) << 15                                   # +-0
    if r < 0.2:
        return (rng.getrandbits(1) << 15) | rng.randrange(1, 128)         # subnormal
    if r < 0.6:
        return (rng.getrandbits(1) << 15) | (rng.randrange(118, 138) << 7) | rng.getrandbits(7)  # near 1, many carries
    return rng.getrandbits(16)


def rand_f32(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.02:
        return rng.choice([0x7F800000, 0xFF800000])
    if r < 0.04:
        return 0x7F800000 | rng.randrange(1, 1 << 23) | (rng.getrandbits(1) << 31)
    if r < 0.1:
        return rng.getrandbits(1) << 31
    if r < 0.6:
        return (rng.getrandbits(1) << 31) | (rng.randrange(115, 140) << 23) | rng.getrandbits(23)
    return rng.getrandbits(32)


def pack_bits(words: list[int], width: int) -> list[int]:
    """T words -> ``width`` T-bit ints (bit t of int j = bit j of word t)."""
    out = [0] * width
    for t, w in enumerate(words):
        for j in range(width):
            if (w >> j) & 1:
                out[j] |= 1 << t
    return out


def unpack(bits: list[int], T: int) -> list[int]:
    return [sum(((bits[j] >> t) & 1) << j for j in range(len(bits))) for t in range(T)]


def build_reference():
    C = Circuit()
    acc, a, b = C.input("c", 32), C.input("a", 256), C.input("b", 256)
    C.output("o", fp.tc_dot16(C, acc, [a[16 * i:16 * i + 16] for i in range(16)], [b[16 * i:16 * i + 16] for i in range(16)],
                              groups=(16,), W=26, F=-133))
    return C


def build_split():
    """decode + step in one circuit on raw words (for the equality check)."""
    C = Circuit()
    acc, a, b = C.input("c", 32), C.input("a", 256), C.input("b", 256)
    C.phase = "decode"
    A = [decode_word(C, a[16 * i:16 * i + 16]) for i in range(16)]
    B = [decode_word(C, b[16 * i:16 * i + 16]) for i in range(16)]
    C.output("o", step_decoded(C, acc, A, B))
    return C


def build_step():
    """The step template alone: its inputs are the decoded operands and the accumulator word."""
    C = Circuit()
    acc = C.input("c", 32)
    A = [decoded_input(C, f"a{i}") for i in range(16)]
    B = [decoded_input(C, f"b{i}") for i in range(16)]
    C.output("o", step_decoded(C, acc, A, B))
    return C


def build_decode():
    C = Circuit()
    x = C.input("x", 16)
    d = decode_word(C, x)
    for f, _ in FIELDS:
        C.output(f, d[f])
    return C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8192, help="bit-sliced random vectors")
    ap.add_argument("--ref", type=int, default=400, help="vectors also checked against verity.ml.circuits.silicon.total")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    t0 = time.time()
    R, S, T1, D = build_reference(), build_split(), build_step(), build_decode()
    fin = [rng.random() < 0.75 for _ in range(a.T)]
    accs = [rand_f32(rng) if not f else rand_f32(rng) & 0xBFFFFFFF for f in fin]
    xs = [[rand_bf16(rng, f) for _ in range(16)] for f in fin]
    ws = [[rand_bf16(rng, f) for _ in range(16)] for f in fin]
    inp = {"c": pack_bits(accs, 32), "a": pack_bits([sum(v << (16 * i) for i, v in enumerate(r)) for r in xs], 256),
           "b": pack_bits([sum(v << (16 * i) for i, v in enumerate(r)) for r in ws], 256)}
    (ro, rf), (so, sf) = R.evaluate(inp, a.T), S.evaluate(inp, a.T)
    ref_out, split_out = unpack(ro["o"], a.T), unpack(so["o"], a.T)
    mism = sum(1 for x, y in zip(ref_out, split_out) if x != y)
    from verity.ml.circuits.silicon.models import HOPPER_BF16_WGMMA_K16
    from verity.ml.circuits.silicon.total import tc_dot_total
    core_mism = sum(1 for t in range(min(a.ref, a.T)) if tc_dot_total(HOPPER_BF16_WGMMA_K16, accs[t], xs[t], ws[t]) != split_out[t])
    n_special = sum(1 for t in range(a.T) if (split_out[t] & 0x7F800000) == 0x7F800000)
    cr, cs, cd = R.counts(), T1.counts(), D.counts()
    rec = {
        "question": "does the sm_120 BF16 k16 step over row-decoded operands fit a 2^13-row slot, bit-exact?",
        "vectors": a.T, "mismatches_vs_fp_tc_dot16": mism, "core_checked": min(a.ref, a.T), "mismatches_vs_core_tc_dot_total": core_mism,
        "outputs_nan_or_inf": n_special,
        "reference_step": {"ands": cr["ands"], "rows": rows(cr), "nnz_AB": cr["nnz_AB"]},
        "step_template": {"ands": cs["ands"], "inputs": cs["inputs"], "output_copies": cs["output_copies"], "rows": rows(cs),
                          "nnz_AB": cs["nnz_AB"], "slot_2^13_fill": rows(cs) / 8192, "phases": T1.phase_counts()},
        "decode_per_word": {"ands": cd["ands"], "outputs": DECODED_BITS},
        "current_unit_rows_per_step_K2048": 1169281 / 128, "current_unit_ands_per_step_K2048": 1103522 / 128,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(rec, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1)
    return 0 if mism == 0 and core_mism == 0 and rows(cs) <= 8192 else 1


if __name__ == "__main__":
    raise SystemExit(main())
