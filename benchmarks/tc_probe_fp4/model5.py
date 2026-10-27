"""A minimal Python reference of autoproof's *model5* for one Blackwell block-scaled FP4 MMA
(``mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3``), kept as ``probe.py``'s
control.  The probe refuted it on sm_120 (discrepancy-log entries D16 and D17: a -0 accumulator over all-zero products, and an
opposite-sign accumulator at the alignment window's edge); the pinned model is ``BLACKWELL_SM120_NVF4``.

Written from the read-only autoproof sources ``fp4-ref/src/native.rs`` (the model), ``fp4-ref/src/bitref.rs`` (the independent
bit-level datapath) and ``spec/SEMANTICS.lock.v1.json`` (the adopted parameters: W1=42, W2=36, RZ, hot-group clip at 2^-19).
No code is shared with ``verity.ml.circuits.silicon`` or with autoproof; everything below is re-derived from the description.

The model, per instruction (K = 64, four groups of 16):
  1. four EXACT integer group dots  gd_g = sum_{k in group g} num(a_k) * num(b_k)   (E2M1 numerators in units of 2^-1)
  2. lossless scale application     term_g = gd_g * ma * mb * 2^(-2 + ea + eb)      (UE4M3 code -> (m, e); UE8M0 -> (1, e))
  3. one five-way aligned add of the four terms and the incoming FP32 accumulator: every nonzero addend is truncated toward
     zero at the common lsb = max(E1 - 41, acc != 0 ? ilog2(acc) - 35 : -inf, hot ? -19 : -inf), E1 = max exponent over the
     nonzero addends (accumulator included); the truncated addends are summed exactly and normalized to binary32 with RZ.
     All-zero addends leave the accumulator untouched (a -0 accumulator stays -0); an exact zero sum is +0.
  The hot-group clip: any group with some a_k*b_k != 0 AND both UE4M3 scale codes in 0x78..0x7E (exponent field 15) makes
  the whole firing lose its 2^-20 bit.  UE8M0 scales never fire it (the trigger reads the UE4M3 exponent field).

``check_autoproof`` replays autoproof's recorded ``sample_cases`` (probe3: K in {64,128,256}, chained ascending-K with a zero
accumulator; probe2: unscaled ``kind::f8f6f4`` m16n8k32, exact regime) through the model and counts agreements, offline.
"""
from __future__ import annotations

import json
import sys

import numpy as np

# E2M1 low-nibble code -> numerator in units of 2^-1 (s e1 e0 m): 0, .5, 1, 1.5, 2, 3, 4, 6.
E2M1_NUM = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int64)

W1 = 42
W2 = 36
HOT_CLIP_LSB = -19
GROUP = 16


def ue4m3(code: int) -> tuple[int, int]:
    """UE4M3 scale code -> (mantissa, exponent); value = m * 2^e.  0x7F is NaN and refused."""
    assert 0 <= code < 0x80 and code != 0x7F, f"invalid UE4M3 code {code:#x}"
    e = (code >> 3) & 0xF
    m = code & 7
    return (m, -9) if e == 0 else (8 + m, e - 10)


def ue8m0(code: int) -> tuple[int, int]:
    assert 0 <= code < 0xFF, f"invalid UE8M0 code {code:#x}"
    return (1, code - 127)


SCALE_DECODE = {"ue4m3": ue4m3, "ue8m0": ue8m0, "none": lambda c: (1, 0)}


def transducer(acc_bits: int, terms: list[tuple[int, int]], hot: bool, w1: int = W1, w2: int = W2) -> int:
    """One firing: FP32 accumulator bits + exact terms (mant, exp) -> FP32 result bits.  Raises OverflowError past binary32."""
    e = (acc_bits >> 23) & 0xFF
    f = acc_bits & 0x7FFFFF
    assert e != 0xFF, "non-finite accumulator"
    acc_mag, acc_exp = (f, -149) if e == 0 else ((1 << 23) | f, e - 150)
    acc_neg = bool(acc_bits >> 31)
    addends = [(m, ex) for m, ex in terms if m != 0]
    acc_msb = None
    if acc_mag:
        acc_msb = acc_mag.bit_length() - 1 + acc_exp
        addends.append((-acc_mag if acc_neg else acc_mag, acc_exp))
    if not addends:
        return acc_bits
    e1 = max(abs(m).bit_length() - 1 + ex for m, ex in addends)
    lsb = e1 - (w1 - 1)
    if acc_msb is not None:
        lsb = max(lsb, acc_msb - (w2 - 1))
    if hot:
        lsb = max(lsb, HOT_CLIP_LSB)
    total = 0
    for m, ex in addends:
        mag = abs(m)
        mag = mag << (ex - lsb) if ex >= lsb else mag >> (lsb - ex)
        total += -mag if m < 0 else mag
    if total == 0:
        return 0
    neg = total < 0
    mag = abs(total)
    n = mag.bit_length()
    e_val = n - 1 + lsb
    if e_val >= -126:
        if e_val > 127:
            raise OverflowError("binary32 overflow")
        sh = n - 24
        frac = mag >> sh if sh >= 0 else mag << -sh
        bits = ((e_val + 127) << 23) | (frac & 0x7FFFFF)
    else:
        sh = -149 - lsb
        bits = mag >> sh if sh >= 0 else mag << -sh
    return bits | (0x80000000 if neg else 0)


def group_terms(a_codes, b_codes, sa_codes, sb_codes, scale_kind: str, group: int = GROUP):
    """The exact scaled terms of one dot of length K and the hot flag.  ``sa_codes``/``sb_codes``: one per scale block; the
    block size is K / len(codes) (16 for UE4M3 .4X, 32 for UE8M0 .2X); groups are ``group`` wide (16 = the 4X hypothesis)."""
    a = E2M1_NUM[np.asarray(a_codes, dtype=np.int64) & 0xF]
    b = E2M1_NUM[np.asarray(b_codes, dtype=np.int64) & 0xF]
    k = a.shape[0]
    dec = SCALE_DECODE[scale_kind]
    nblk = max(1, len(sa_codes))
    blk = k // nblk
    terms = []
    hot = False
    for g0 in range(0, k, group):
        gd = int(np.dot(a[g0:g0 + group], b[g0:g0 + group]))
        live = bool(np.any((a[g0:g0 + group] != 0) & (b[g0:g0 + group] != 0)))
        ca = int(sa_codes[g0 // blk]) if sa_codes else 0
        cb = int(sb_codes[g0 // blk]) if sb_codes else 0
        ma, ea = dec(ca)
        mb, eb = dec(cb)
        terms.append((gd * ma * mb, -2 + ea + eb))
        if scale_kind == "ue4m3" and live and (ca & 0x78) == 0x78 and (cb & 0x78) == 0x78:
            hot = True
    return terms, hot


def dot_chain(acc_bits: int, a_codes, b_codes, sa_codes, sb_codes, scale_kind: str, k_per_mma: int = 64, group: int = GROUP) -> int:
    """Ascending-K chain of MMAs over a K-vector (K a multiple of ``k_per_mma``), one firing per instruction."""
    k = len(a_codes)
    nblk = max(1, len(sa_codes))
    blk = k // nblk
    for base in range(0, k, k_per_mma):
        sl = slice(base, base + k_per_mma)
        sb = slice(base // blk, (base + k_per_mma) // blk) if sa_codes else slice(0, 0)
        terms, hot = group_terms(a_codes[sl], b_codes[sl], list(sa_codes)[sb], list(sb_codes)[sb], scale_kind, group)
        acc_bits = transducer(acc_bits, terms, hot)
    return acc_bits


def tile_eval(A: np.ndarray, Bt: np.ndarray, SFA: np.ndarray, SFB: np.ndarray, C: np.ndarray, scale_kind: str, group: int = GROUP):
    """Model of one MMA tile: ``A`` (16,K) codes, ``Bt`` (8,K) codes, ``SFA`` (16,nblk), ``SFB`` (8,nblk), ``C`` (16,8) FP32 bits ->
    ``D`` (16,8) bits and an error mask (binary32 overflow)."""
    k = A.shape[1]
    ng = k // group
    a = E2M1_NUM[A.astype(np.int64) & 0xF].reshape(16, ng, group)
    b = E2M1_NUM[Bt.astype(np.int64) & 0xF].reshape(8, ng, group)
    gd = np.einsum("igk,jgk->ijg", a, b)  # exact: |gd| <= 16*144
    live = np.einsum("igk,jgk->ijg", (a != 0).astype(np.int64), (b != 0).astype(np.int64)) > 0
    def dec(code: int):
        try:
            return SCALE_DECODE[scale_kind](code)
        except AssertionError:  # an invalid scale code (UE8M0 0xFF): outside the model, for the elements that read it
            return None

    nblk = SFA.shape[1] if SFA.ndim == 2 and SFA.shape[1] else 1
    gpb = ng // nblk  # groups per scale block
    sa = [[dec(int(SFA[i, g // gpb])) if SFA.size else (1, 0) for g in range(ng)] for i in range(16)]
    sb = [[dec(int(SFB[j, g // gpb])) if SFB.size else (1, 0) for g in range(ng)] for j in range(8)]
    hot_a = (SFA.astype(np.int64) & 0x78) == 0x78 if (scale_kind == "ue4m3" and SFA.size) else np.zeros((16, nblk), bool)
    hot_b = (SFB.astype(np.int64) & 0x78) == 0x78 if (scale_kind == "ue4m3" and SFB.size) else np.zeros((8, nblk), bool)
    D = np.zeros((16, 8), np.uint32)
    err = np.zeros((16, 8), np.bool_)
    for i in range(16):
        for j in range(8):
            if any(s is None for s in sa[i] + sb[j]):
                err[i, j] = True
                continue
            terms = []
            hot = False
            for g in range(ng):
                (ma, ea), (mb, eb) = sa[i][g], sb[j][g]
                terms.append((int(gd[i, j, g]) * ma * mb, -2 + ea + eb))
                if live[i, j, g] and hot_a[i, g // gpb] and hot_b[j, g // gpb]:
                    hot = True
            try:
                D[i, j] = transducer(int(C[i, j]), terms, hot)
            except (OverflowError, AssertionError):  # binary32 overflow / non-finite accumulator: outside the model
                err[i, j] = True
    return D, err


def check_autoproof(path: str) -> dict:
    """Replay a probes/results/probe{2,3}_*.json ``sample_cases`` list through the model (zero-init accumulator, ascending K)."""
    data = json.load(open(path))
    probe = data["probe"]
    kind = "ue4m3" if probe == 3 else "none"
    k_per = 64 if probe == 3 else 32
    agree = 0
    first = []
    for idx, c in enumerate(data["sample_cases"]):
        got = dot_chain(0, c["a_codes"], c["b_codes"], c["a_scales"], c["b_scales"], kind, k_per)
        want = int(c["silicon_c_bits"]) & 0xFFFFFFFF
        if got == want:
            agree += 1
        elif len(first) < 10:
            first.append({"index": idx, "k": len(c["a_codes"]), "silicon": f"{want:08x}", "model": f"{got:08x}"})
    return {"file": path, "probe": probe, "target": data.get("target"), "cases": len(data["sample_cases"]), "agree": agree,
            "first_mismatches": first}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(check_autoproof(p)))
