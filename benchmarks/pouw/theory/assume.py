"""ASSUME: what any premise can credit (the zero-data ceiling), and the checks behind the XOR fold.

**The zero-data ceiling.** A premise only lower-bounds an adversary's cost, so under any true premise the credited work is
at most C₀, the cost of one explicit adversary: the prover that chooses zero data (model M0) and computes only what the
check reads. T1 ≤ 1% therefore needs cost(H) − C₀ ≤ 1% of stock, a property of the design alone. This script prices C₀
for the classes a sparse or outputs-only commitment can take:

- *Served* (the noise stays in the served output), a word every k k32 steps, q of each step's 32 s8 lanes noise: the
  zero-data words are the noise lanes' partial sums, packed into depth-32 s8 MMAs, so C₀ = 32·⌈qk/32⌉ a word against
  the honest 32k. T1 then needs the data MACs a word, D = (32 − q)k, to stay below 32 at every (q, k): a stride-k served
  design never commits fewer words a data MAC than the lead's 31 + 1 lanes do, under any premise.
- *Outputs only* is the served case at k = K/32.
- *Exact* (the noise cancels in the GEMM): the final word is 0 at zero data, so the last segment is skipped,
  32k/K of stock (N-SAKU's `skip_adv`). T1's budget, MAC-weighted over a model's linears, caps k per distinct K; the rest
  of the credit needs SAK on the cancelling structure (rank 4b), which this script assumes in that table's favour.

**The XOR fold.** σ ← LOP3(σ, S_t, S_{t+1}): every k32 running word folded by a 3-input XOR, one instruction per two
words, sealed by one SHA-512 per cell (or per row of cells): main's `Accounting.TS.thinFold` with every step XOR3. Its
own work is ½ a node a word only at three sources; `Accounting.Assumptions.TS.FanIn` counts guards, carry-ins and kept
destinations, up to 11 sources on sm_90, so `FoldFanIn` credits about ⅒ a word and the rest is a FOLD-OPT premise.
The checks:

1. A linear fold folds (H12): Σ r_t S_t = Σ_u δ_u R_u with R_u = Σ_{t≥u} r_t, so no word need be written.
2. The XOR fold is affine over GF(2) in the increments only at bit 0, and isn't additive over ℤ/2³² at all.
3. (Inj) for one skipped word: σ is a bijection of each word given the others (fibre 1).
4. (Inj) for a persistent skip (Lemma B′: a skipped product shifts every later prefix word by X): the fibre
   #{X : fold(S + X) = fold(S)}, exhaustive over X at 16 bits, for the XOR fold and for li (`fold_routes.li_half`).
   At 32 bits X = 2³¹ is in the XOR fold's fibre on every even run (a sign flip XORs out in pairs).
5. Superoptimizer windows (`fold_routes.single` and `two`): no single modelled sm_90 instruction computes
   x ⊕ (x + y) or x ⊕ (x + y·z) (an add fused into an XOR), and no two-instruction program computes the XOR fold's pair
   step σ ⊕ S_t ⊕ S_{t+1} from (σ, S_{t−1}, δ_t, δ_{t+1}); the honest step is three (two adds and the LOP3), 1.5 a word
   with the words. The same at zero data, one s8 coin lane a step, from (σ, S_{t−1}, e_t, f_t, e_{t+1}, f_{t+1}).
6. The only LOP3 steps that flip on every single input bit are 0x96 and 0x69: an injective bitwise fold is the 3-XOR,
   so it is affine over GF(2) bit by bit, and acts lane-wise on packed fields.
7. Packing (SWAR) at zero data: two cells of one row share e_t, so one IMAD on P = (S_A + 2^14) + 2^16 (S_B + 2^14)
   with F_t = f_A + 2^16 f_B advances both, one predicate-only LOP3 on P & 0x80008000 flags the first step either word
   leaves [−2^14, 2^14), and the XOR of the packed words decodes exactly to both cells' folds. It pays while the walks
   stay small; the gain is priced at two detector prices and two overhead counts. The thin fold's IADD3 steps break the
   bias that makes the decode exact (a biased XOR is not a biased sum).

**The alternating fold** (main's `thinFold`, XOR3 and IADD3 steps alternating):

8. Affine over GF(2) only at bit 0, never additive over ℤ/2³²; (Inj) for one word; persistent-skip fibres exhaustive
   at 16 bits by run length and first step; X = 2³¹ is in the fibre of every step-aligned run and of no mid-step one.
9. The IADD3 step's windows (`fold_routes.single` and `two`): no two-instruction program computes σ + S_t + S_{t+1}
   from (σ, S_{t−1}, δ_t, δ_{t+1}), nor at zero data from (σ, S_{t−1}, e_t, f_t, e_{t+1}, f_{t+1}); and none writes both
   it and S_{t+1} in two (one would have to be a single instruction of the pool and the other of the pool and it).
10. Packing two cells against it at zero data with the borrows tracked (`swar_alt`): a bias of 2^15 on the low field
    keeps the high field borrow-free; each cell's σ is its low half, packed, and its high half, which an XOR3 step
    flips on a sign parity and an IADD3 step moves by the low half's signed carry. Bit-exact, and it costs more than
    honest at every step, with the detector, F_t and the exit decode free.

Run: `python assume.py [--out DIR] [--trials N]`. numpy, and `fold_routes` beside this file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_routes as FR

T1 = 0.01
M32 = 0xFFFFFFFF

#: transformer-block linears (name, K, n, multiplicity), MACs a token = K·n·multiplicity; public configs
MODELS = {
    "Llama-3.2-1B": [("q", 2048, 2048, 1), ("kv", 2048, 512, 2), ("o", 2048, 2048, 1),
                     ("gate_up", 2048, 8192, 2), ("down", 8192, 2048, 1)],
    "Llama-3.1-8B": [("q", 4096, 4096, 1), ("kv", 4096, 1024, 2), ("o", 4096, 4096, 1),
                     ("gate_up", 4096, 14336, 2), ("down", 14336, 4096, 1)],
    "Llama-3.1-70B": [("q", 8192, 8192, 1), ("kv", 8192, 1024, 2), ("o", 8192, 8192, 1),
                      ("gate_up", 8192, 28672, 2), ("down", 28672, 8192, 1)],
}


# ---- the zero-data ceiling ---------------------------------------------------------------------------------------

def served_ceiling(k_max: int = 256) -> dict:
    """Every (q, k): the zero-data gap a word, and whether T1 holds; the most data MACs a word any passing design has."""
    best = {"D": 0, "q": None, "k": None}
    passing = 0
    for q in range(0, 33):
        for k in range(1, k_max + 1):
            honest = 32 * k
            c0 = 32 * math.ceil(q * k / 32)
            d = (32 - q) * k
            if d == 0:
                continue
            if honest - c0 <= T1 * d:
                passing += 1
                if d > best["D"]:
                    best = {"D": d, "q": q, "k": k}
    return {"k_max": k_max, "passing_designs": passing, "max_data_macs_a_word": best,
            "lead_data_macs_a_word": 31,
            "h_ratio_floor": 31 / best["D"] if best["D"] else None}


def outputs_only(K: int) -> dict:
    k = K // 32
    rows = []
    for q in range(0, 32):
        honest, c0, d = 32 * k, 32 * math.ceil(q * k / 32), (32 - q) * k
        rows.append({"q": q, "gap_over_stock": (honest - c0) / d})
    return {"K": K, "min_gap_over_stock": min(r["gap_over_stock"] for r in rows)}


def exact_caps(model: str, k_cap: int = 64) -> dict:
    """The integer stride k for each distinct K that minimises words a MAC subject to the MAC-weighted skip ≤ T1."""
    lin = MODELS[model]
    macs = {}
    for _, K, n, mult in lin:
        macs[K] = macs.get(K, 0) + K * n * mult
    tot = sum(macs.values())
    w = {K: m / tot for K, m in macs.items()}
    Ks = sorted(w)
    best = None
    for ks in np.ndindex(*([k_cap] * len(Ks))):
        kk = [k + 1 for k in ks]
        skip = sum(w[K] * 32 * k / K for K, k in zip(Ks, kk))
        if skip > T1:
            continue
        words = sum(w[K] / k for K, k in zip(Ks, kk))
        if best is None or words < best["words"]:
            best = {"k": dict(zip(map(str, Ks), kk)), "skip": skip, "words": words}
    dense_skip = sum(w[K] * 32 / K for K in Ks)
    return {"model": model, "mac_share": {str(K): w[K] for K in Ks}, "dense_skip": dense_skip,
            "best": best, "h_divisor": (1 / best["words"]) if best else None}


# ---- the XOR fold ------------------------------------------------------------------------------------------------

def prefix(d: np.ndarray, mask: int) -> np.ndarray:
    return np.cumsum(d.astype(np.uint64), axis=-1) & np.uint64(mask)


def xor_fold(S: np.ndarray) -> np.ndarray:
    return np.bitwise_xor.reduce(S, axis=-1)


def check_linear_folds(rng: np.random.Generator, trials: int, T: int = 64) -> dict:
    ok = 0
    for _ in range(trials):
        d = rng.integers(0, 2**32, T, dtype=np.uint64)
        r = rng.integers(0, 2**32, T, dtype=np.uint64)
        S = prefix(d, M32)
        lhs = int(sum(int(a) * int(b) for a, b in zip(r, S)) & M32)
        R = np.cumsum(r[::-1].astype(object))[::-1]
        rhs = int(sum(int(a) * int(b) for a, b in zip(d, R)) & M32)
        ok += lhs == rhs
    return {"trials": trials, "identity_holds": ok}


def check_xor_affinity(rng: np.random.Generator, trials: int, T: int = 64) -> dict:
    """GF(2) additivity of σ in the increments, bit by bit, and ℤ/2³² additivity."""
    gf2 = np.zeros(32)
    zadd = 0
    zero = int(xor_fold(prefix(np.zeros(T, dtype=np.uint64), M32)))
    for _ in range(trials):
        a = rng.integers(0, 2**32, T, dtype=np.uint64)
        b = rng.integers(0, 2**32, T, dtype=np.uint64)
        fa, fb = int(xor_fold(prefix(a, M32))), int(xor_fold(prefix(b, M32)))
        fab = int(xor_fold(prefix(a ^ b, M32)))
        diff = fab ^ fa ^ fb ^ zero
        for bit in range(32):
            gf2[bit] += ((diff >> bit) & 1) == 0
        fsum = int(xor_fold(prefix((a + b) & np.uint64(M32), M32)))
        zadd += fsum == ((fa + fb - zero) & M32)
    return {"trials": trials, "gf2_additive_rate_by_bit": (gf2 / trials).round(4).tolist(),
            "z_additive_rate": zadd / trials}


def check_single_word_inj(rng: np.random.Generator, trials: int, T: int = 64) -> dict:
    changed = 0
    for _ in range(trials):
        S = rng.integers(0, 2**32, T, dtype=np.uint64)
        t = int(rng.integers(0, T))
        c = int(rng.integers(1, 2**32))
        S2 = S.copy()
        S2[t] = (int(S2[t]) + c) & M32
        changed += int(xor_fold(S)) != int(xor_fold(S2))
    return {"trials": trials, "fold_changed": changed}


def persistent_fibre(rng: np.random.Generator, trials: int, lengths=(1, 2, 3, 4, 6, 8, 16, 32, 64),
                     width: int = 16) -> dict:
    """#{X ∈ ℤ/2^w : fold(S_t + X for every t) = fold(S)}, X = 0 included, for the XOR fold and for li."""
    mask = (1 << width) - 1
    X = np.arange(1 << width, dtype=np.uint64)
    out = {}
    for ell in lengths:
        zx, zl = [], []
        for _ in range(trials):
            S = rng.integers(0, 1 << width, ell, dtype=np.uint64)
            acc = np.zeros_like(X)
            for s in S:
                acc ^= ((s + X) & np.uint64(mask)) ^ s
            zx.append(int(np.count_nonzero(acc == 0)))
            k = int(rng.integers(0, 1 << width))
            a = int(rng.integers(0, 1 << width)) | 1
            words = list(S) + ([np.uint64(0)] if ell % 2 else [])
            y_ref = 0
            y = np.zeros_like(X)
            for i in range(0, len(words), 2):
                d0, d1 = int(words[i]), int(words[i + 1])
                pad = ell % 2 == 1 and i + 1 == len(words) - 1
                y_ref = (((y_ref ^ d0 ^ k) * a) + d1) & mask
                e0 = (np.uint64(d0) + X) & np.uint64(mask)
                e1 = np.uint64(d1) if pad else (np.uint64(d1) + X) & np.uint64(mask)
                y = (((y ^ e0 ^ np.uint64(k)) * np.uint64(a)) + e1) & np.uint64(mask)
            zl.append(int(np.count_nonzero(y == y_ref)))
        out[str(ell)] = {
            "xor": {"mean": float(np.mean(zx)), "max": int(np.max(zx)), "p_ge2": float(np.mean(np.array(zx) >= 2))},
            "li": {"mean": float(np.mean(zl)), "max": int(np.max(zl)), "p_ge2": float(np.mean(np.array(zl) >= 2))},
        }
    return {"width": width, "trials": trials, "by_length": out}


def sign_flip_32(rng: np.random.Generator, trials: int) -> dict:
    """At 32 bits, X = 2³¹ on an even run: in the XOR fold's fibre always; li's rate for comparison."""
    xor_hits = li_hits = 0
    for _ in range(trials):
        ell = 2 * int(rng.integers(1, 33))
        S = [int(v) for v in rng.integers(0, 2**32, ell, dtype=np.uint64)]
        Sx = [(s + 2**31) & M32 for s in S]
        fx = 0
        for s, sx in zip(S, Sx):
            fx ^= s ^ sx
        xor_hits += fx == 0
        k, a = int(rng.integers(0, 2**32)), int(rng.integers(0, 2**32)) | 1
        y0 = y1 = 0
        for i in range(0, ell, 2):
            y0 = FR.li_half(y0, S[i], S[i + 1], k, a)
            y1 = FR.li_half(y1, Sx[i], Sx[i + 1], k, a)
        li_hits += y0 == y1
    return {"trials": trials, "xor_in_fibre": xor_hits, "li_in_fibre": li_hits}


def superopt_windows(seed: int) -> dict:
    rng = random.Random(seed)
    x, y, z, s0 = FR.vecs(rng), FR.vecs(rng), FR.vecs(rng), FR.vecs(rng)
    out = {}
    t1 = [FR.u32(a ^ FR.u32(a + b)) for a, b in zip(x, y)]
    out["x^(x+y): one instruction"] = FR.single(t1, {"x": x, "y": y, "0": FR.const(0)})
    t2 = [FR.u32(a ^ FR.u32(a + b * c)) for a, b, c in zip(x, y, z)]
    out["x^(x+y*z): one instruction"] = FR.single(t2, {"x": x, "y": y, "z": z, "0": FR.const(0)})
    d1, d2 = y, z
    St = [FR.u32(a + b) for a, b in zip(x, d1)]
    St1 = [FR.u32(a + b) for a, b in zip(St, d2)]
    step = [FR.u32(a ^ b ^ c) for a, b, c in zip(s0, St, St1)]
    pool = {"sig": s0, "Sprev": x, "d1": d1, "d2": d2}
    out["pair step sig^S_t^S_t+1 from (sig,Sprev,d1,d2): two instructions"] = FR.two(step, pool)
    ext = dict(pool)
    ext["St"] = St
    out["sanity: pair step given S_t: two instructions"] = FR.two(step, ext)
    return out


def s8(rng: random.Random, n: int = FR.NV) -> list[int]:
    return [FR.u32(rng.randint(-128, 127)) for _ in range(n)]


def superopt_zero_data(seed: int) -> dict:
    """The zero-data pair step: one s8 coin lane a k32 step, so S_t = S_{t-1} + e_t·f_t is one IMAD, and the step
    σ ⊕ S_t ⊕ S_{t+1} reads (σ, S_{t-1}, e_t, f_t, e_{t+1}, f_{t+1}). Honest-shaped is three instructions."""
    rng = random.Random(seed ^ 0x2E40)
    sig, sp = FR.vecs(rng), FR.vecs(rng)
    e1, f1, e2, f2 = s8(rng), s8(rng), s8(rng), s8(rng)
    St = [FR.u32(a + b * c) for a, b, c in zip(sp, e1, f1)]
    St1 = [FR.u32(a + b * c) for a, b, c in zip(St, e2, f2)]
    step = [FR.u32(a ^ b ^ c) for a, b, c in zip(sig, St, St1)]
    pool = {"sig": sig, "Sprev": sp, "e1": e1, "f1": f1, "e2": e2, "f2": f2}
    out = {"S_t given its operands: one instruction": FR.single(St, pool),
           "sig^S_t from (sig,Sprev,e1,f1): one instruction": FR.single(
               [FR.u32(a ^ b) for a, b in zip(sig, St)], {k: pool[k] for k in ("sig", "Sprev", "e1", "f1")}),
           "pair step from the six: two instructions": FR.two(step, pool)}
    ext = dict(pool)
    ext["St"] = St
    out["sanity: pair step given S_t: two instructions"] = FR.two(step, ext)
    return out


# ---- packing two cells' small zero-data words in one register (SWAR) ------------------------------------------------

def injective_luts() -> list[str]:
    """The LOP3 steps f(σ, a, b) that flip their output bit whenever any one input bit flips: the only bitwise steps
    that keep every bit of the state and of both words."""
    out = []
    for lut in range(256):
        def f(s, a, b, lut=lut):
            return (lut >> (s | a << 1 | b << 2)) & 1
        bits = (0, 1)
        if all(f(0, a, b) != f(1, a, b) and f(a, 0, b) != f(a, 1, b) and f(a, b, 0) != f(a, b, 1)
               for a in bits for b in bits):
            out.append(f"{lut:#04x}")
    return out


B14 = 1 << 14


def swar_decode(field: int, n: int) -> int:
    """σ over n words from the XOR of their 16-bit fields S + 2^14, each S in [-2^14, 2^14): bits 0..13 are S's low
    bits, bit 14 is S's sign inverted, and S's bits 14..31 all equal its sign."""
    neg_odd = (n & 1) ^ ((field >> 14) & 1)
    return (field & (B14 - 1)) | (0xFFFFC000 if neg_odd else 0)


def swar_xor(rng: np.random.Generator, trials: int, T: int) -> dict:
    """Two cells of one row at zero data (row coin e_t, column coins f_A, f_B): one IMAD on P = (S_A + 2^14) +
    2^16 (S_B + 2^14) with F_t = f_A + 2^16 f_B advances both, a LOP3 to a predicate on P & 0x80008000 flags the first
    step either word leaves [-2^14, 2^14), and the XOR of the packed words decodes to both cells' folds."""
    taus, exact, packed_runs = [], 0, 0
    for _ in range(trials):
        e = rng.integers(-128, 128, T)
        fa, fb = rng.integers(-128, 128, T), rng.integers(-128, 128, T)
        sa, sb = np.cumsum(e * fa), np.cumsum(e * fb)
        P, X, tau = B14 + (B14 << 16), 0, 0
        for t in range(T):
            P = (P + int(e[t]) * (int(fa[t]) + (int(fb[t]) << 16))) & M32
            if P & 0x80008000:
                break
            X ^= P
            tau += 1
        taus.append(tau)
        if tau:
            packed_runs += 1
            ta = tb = 0
            for t in range(tau):
                ta ^= int(sa[t]) & M32
                tb ^= int(sb[t]) & M32
            exact += swar_decode(X & 0xFFFF, tau) == ta and swar_decode(X >> 16, tau) == tb
        assert tau == T or max(abs(int(sa[tau])), abs(int(sb[tau]))) >= B14
    taus = np.array(taus)
    honest, packed = 96.0, 48.5
    gain = {f"detector {d}, overhead {o}": float(max(0.0, np.mean(taus) * (honest - packed - d) - 32 * o)
                                                 / (2 * 31 * T))
            for d in (0, 32) for o in (8, 12)}
    return {"T": T, "trials": trials, "mean_packed_words": float(taus.mean()), "max_packed_words": int(taus.max()),
            "p_ge_10": float(np.mean(taus >= 10)), "decode_exact": exact, "packed_runs": packed_runs,
            "gap_over_stock": gain,
            "accounting": "a pair-step: honest-shaped 2 IMAD + 2 half-LOP3 = 96; packed IMAD 32 + half-LOP3 16 + "
                          "F_t 0.5 (shared by 64 rows) + detector d (a predicate-only LOP3: 0 if W1 charges only "
                          "registers written, 32 if it charges the issue); unpack and decode o instructions a pair"}


# ---- the alternating fold (main's `Accounting.TS.thinFold`) -------------------------------------------------------

def thin_fold(words, xor: bool = True, s: int = 0, mask: int = M32) -> int:
    """`thinFold xor s`: the words two a step, the steps alternating XOR3 and IADD3 from `xor`."""
    for i in range(0, len(words) - 1, 2):
        s = (s ^ words[i] ^ words[i + 1]) if xor else (s + words[i] + words[i + 1]) & mask
        xor = not xor
    return s & mask


def thin_fold_np(W: np.ndarray, xor: bool, s, mask: int) -> np.ndarray:
    """The same over the last axis of `W` (uint64), every row at once."""
    s = np.asarray(s, dtype=np.uint64) & np.uint64(mask)
    for i in range(0, W.shape[-1] - 1, 2):
        s = (s ^ W[..., i] ^ W[..., i + 1]) if xor else (s + W[..., i] + W[..., i + 1]) & np.uint64(mask)
        xor = not xor
    return s


def check_alt_affinity(rng: np.random.Generator, trials: int, T: int = 64) -> dict:
    """GF(2) additivity of the alternating fold in the increments, bit by bit, and ℤ/2³² additivity."""
    gf2 = np.zeros(32)
    zadd = 0
    f = lambda d: thin_fold([int(v) for v in prefix(d, M32)])
    zero = f(np.zeros(T, dtype=np.uint64))
    for _ in range(trials):
        a = rng.integers(0, 2**32, T, dtype=np.uint64)
        b = rng.integers(0, 2**32, T, dtype=np.uint64)
        fa, fb, fab = f(a), f(b), f(a ^ b)
        diff = fab ^ fa ^ fb ^ zero
        for bit in range(32):
            gf2[bit] += ((diff >> bit) & 1) == 0
        zadd += f((a + b) & np.uint64(M32)) == ((fa + fb - zero) & M32)
    return {"trials": trials, "gf2_additive_rate_by_bit": (gf2 / trials).round(4).tolist(),
            "z_additive_rate": zadd / trials}


def check_alt_single_word_inj(rng: np.random.Generator, trials: int, T: int = 64) -> dict:
    changed = 0
    for _ in range(trials):
        S = [int(v) for v in rng.integers(0, 2**32, T, dtype=np.uint64)]
        t = int(rng.integers(0, T))
        S2 = list(S)
        S2[t] = (S2[t] + int(rng.integers(1, 2**32))) & M32
        changed += thin_fold(S) != thin_fold(S2)
    return {"trials": trials, "fold_changed": changed}


def alt_persistent_fibre(rng: np.random.Generator, trials: int, lengths=(1, 2, 3, 4, 6, 8, 16, 32, 64),
                         width: int = 16) -> dict:
    """#{X ∈ ℤ/2^w : thinFold(S + X on the run) = thinFold(S)}, X = 0 included, exhaustive at w bits, for a run of ℓ
    shifted words whose first full step is XOR3 or IADD3 (an odd run starts mid-step, beside one unshifted word).
    With the cell's final word O′ sealed beside σ, every X ≠ 0 shifts O′, so the sealed pair's fibre is 1."""
    mask = (1 << width) - 1
    X = np.arange(1 << width, dtype=np.uint64)
    out = {}
    for ell in lengths:
        for first in ("xor", "add"):
            z = []
            for _ in range(trials):
                lead_in = ell % 2
                S = rng.integers(0, 1 << width, ell + lead_in, dtype=np.uint64)
                s0 = int(rng.integers(0, 1 << width))
                shift = np.zeros(ell + lead_in, dtype=np.uint64)
                shift[lead_in:] = 1
                W = (S[None, :] + X[:, None] * shift[None, :]) & np.uint64(mask)
                ref = thin_fold([int(v) for v in S], first == "xor", s0, mask)
                got = thin_fold_np(W, first == "xor", s0, mask)
                z.append(int(np.count_nonzero(got == np.uint64(ref))))
            z = np.array(z)
            out[f"{ell} {first}"] = {"mean": float(z.mean()), "max": int(z.max()), "p_ge2": float(np.mean(z >= 2))}
    return {"width": width, "trials": trials, "by_length_and_first_step": out}


def alt_sign_flip_32(rng: np.random.Generator, trials: int) -> dict:
    """X = 2³¹ on a step-aligned run (each step's two words both shifted) is in the alternating fold's fibre always:
    XOR3 cancels the two top-bit flips and IADD3 adds 2³²; a run that starts mid-step flips σ's top bit."""
    aligned = mid = 0
    for _ in range(trials):
        n = 2 * int(rng.integers(1, 33))
        S = [int(v) for v in rng.integers(0, 2**32, n, dtype=np.uint64)]
        p = 2 * int(rng.integers(0, n // 2))
        Sa = S[:p] + [(v + 2**31) & M32 for v in S[p:]]
        aligned += thin_fold(Sa) == thin_fold(S)
        q = p + 1
        Sm = S[:q] + [(v + 2**31) & M32 for v in S[q:]]
        mid += thin_fold(Sm) == thin_fold(S)
    return {"trials": trials, "aligned_in_fibre": aligned, "mid_step_in_fibre": mid}


def superopt_iadd3(seed: int) -> dict:
    """The IADD3 step's windows: σ + S_t + S_{t+1} = σ + 2·S_{t−1} + 2·δ_t + δ_{t+1} from (σ, S_{t−1}, δ_t, δ_{t+1}),
    and at zero data from (σ, S_{t−1}, e_t, f_t, e_{t+1}, f_{t+1}). The chain also needs S_{t+1}: a two-instruction
    program for both outputs writes one with each instruction, so it exists only if one output is a single instruction
    of the pool and the other a single instruction of the pool and the first."""
    rng = random.Random(seed ^ 0x1AD3)
    sig, sp, d1, d2 = FR.vecs(rng), FR.vecs(rng), FR.vecs(rng), FR.vecs(rng)
    St = [FR.u32(a + b) for a, b in zip(sp, d1)]
    St1 = [FR.u32(a + b) for a, b in zip(St, d2)]
    step = [FR.u32(a + b + c) for a, b, c in zip(sig, St, St1)]
    pool = {"sig": sig, "Sprev": sp, "d1": d1, "d2": d2, "0": FR.const(0)}
    out = {"pair step from (sig,Sprev,d1,d2): two instructions": FR.two(step, pool)}
    out["pair step from the pool: one instruction"] = FR.single(step, pool)
    out["S_t+1 from the pool: one instruction"] = FR.single(St1, pool)
    out["pair step given S_t+1: one instruction"] = FR.single(step, {**pool, "St1": St1})
    out["S_t+1 given the pair step: one instruction"] = FR.single(St1, {**pool, "sig2": step})
    out["sanity: pair step given S_t and S_t+1: one instruction"] = FR.single(step, {**pool, "St": St, "St1": St1})
    out["sanity: pair step given S_t+1: two instructions"] = FR.two(step, {**pool, "St1": St1})

    e1, f1, e2, f2 = s8(rng), s8(rng), s8(rng), s8(rng)
    Zt = [FR.u32(a + b * c) for a, b, c in zip(sp, e1, f1)]
    Zt1 = [FR.u32(a + b * c) for a, b, c in zip(Zt, e2, f2)]
    zstep = [FR.u32(a + b + c) for a, b, c in zip(sig, Zt, Zt1)]
    zpool = {"sig": sig, "Sprev": sp, "e1": e1, "f1": f1, "e2": e2, "f2": f2}
    out["zero data: pair step from the six: two instructions"] = FR.two(zstep, zpool)
    out["zero data: S_t+1 from the six: one instruction"] = FR.single(Zt1, zpool)
    out["zero data: pair step given S_t+1: one instruction"] = FR.single(zstep, {**zpool, "St1": Zt1})
    out["zero data: S_t+1 given the pair step: one instruction"] = FR.single(Zt1, {**zpool, "sig2": zstep})
    out["zero data sanity: S_t is one IMAD"] = FR.single(Zt, zpool)
    out["zero data sanity: pair step given S_t: two instructions"] = FR.two(zstep, {**zpool, "St": Zt})
    return out


def swar_alt(rng: np.random.Generator, trials: int, T: int) -> dict:
    """Packing two cells of one row against the alternating fold at zero data, with the borrows tracked, counted
    instruction by instruction and checked bit-exact against both cells' 32-bit folds.

    P = (S_A + 2^15) + 2^16 S_B: the bias keeps field A in [0, 2^16), so field B never borrows and one IMAD on
    F_t = f_A + 2^16 f_B advances both while |S_A|, |S_B| < 2^15 (a free detector stops before a step whose words
    could leave: |S| ≥ 2^14 before it). σ = L + 2^16 H a cell: Q = L_A + 2^16 L_B packed, H_A and H_B in the high
    halves of their own registers.
    - XOR3 step (6): X = a ⊕ b; Q ^= X (field A's biases cancel in pairs); M_A, M_B = PRMT sign-replicates of X's
      bits 15 and 31 (the pair's sign parities); H_A ^= M_A; H_B ^= M_B.
    - IADD3 step (13): Y = (Q & 0xFFFF) + (a | 0xFFFF0000) + (b & 0xFFFF) = L_A + Σ_A, signed; L_A' = Y & 0xFFFF;
      H_A += Y − L_A' (the signed carry, in the high half); U = (Q >> 16) + (a >>s 16) + (b >>s 16) = L_B + Σ_B;
      L_B' = U & 0xFFFF; H_B += U − L_B'; Q = (L_B' << 16) + L_A'.
    Honest is 2 IMAD and 1 fold instruction a cell a step pair: 6 a step pair for the two cells."""
    counts = {"xor": 6, "add": 13}
    honest_pair = 6
    taus, exact, runs = [], 0, 0
    for _ in range(trials):
        e = rng.integers(-128, 128, T)
        fa, fb = rng.integers(-128, 128, T), rng.integers(-128, 128, T)
        sa, sb = np.cumsum(e * fa), np.cumsum(e * fb)
        P, Q, HA, HB, xor, tau = 1 << 15, 0, 0, 0, True, 0
        words = []
        for t in range(0, T - 1, 2):
            pre = (abs(int(sa[t - 1])) if t else 0, abs(int(sb[t - 1])) if t else 0)
            if max(pre) >= B14 or max(abs(int(sa[t])), abs(int(sb[t]))) >= B14:
                break
            a = (P + int(e[t]) * (int(fa[t]) + (int(fb[t]) << 16))) & M32
            b = (a + int(e[t + 1]) * (int(fa[t + 1]) + (int(fb[t + 1]) << 16))) & M32
            P = b
            if xor:
                X = a ^ b
                Q ^= X
                MA = FR.prmt(X, 0, 0x9999)
                MB = FR.prmt(X, 0, 0xBBBB)
                HA ^= MA
                HB ^= MB
            else:
                Y = ((Q & 0xFFFF) + (a | 0xFFFF0000) + (b & 0xFFFF)) & M32
                LA = Y & 0xFFFF
                HA = (HA + Y - LA) & M32
                sgn = lambda v: (v >> 16) - (1 << 16) if v >> 31 else v >> 16
                U = ((Q >> 16) + sgn(a) + sgn(b)) & M32
                LB = U & 0xFFFF
                HB = (HB + U - LB) & M32
                Q = ((LB << 16) + LA) & M32
            xor = not xor
            tau += 2
            words.append((t, t + 1))
        taus.append(tau)
        if tau:
            runs += 1
            ref_a = thin_fold([int(v) & M32 for v in sa[:tau]])
            ref_b = thin_fold([int(v) & M32 for v in sb[:tau]])
            got_a = (HA & 0xFFFF0000) | (Q & 0xFFFF)
            got_b = (HB & 0xFFFF0000) | (Q >> 16)
            exact += got_a == ref_a and got_b == ref_b
    taus = np.array(taus)
    pair_steps = taus / 2
    packed_cost = {"xor": 2 + counts["xor"], "add": 2 + counts["add"]}
    gain_a_pair = {k: honest_pair - v for k, v in packed_cost.items()}
    return {"T": T, "trials": trials, "mean_packed_words": float(taus.mean()), "max_packed_words": int(taus.max()),
            "decode_exact": exact, "packed_runs": runs,
            "instructions_a_step_pair_two_cells": {"honest": honest_pair, "packed": packed_cost},
            "gain_a_step_pair_instructions": gain_a_pair,
            "gain_over_stock_free_detector_no_decode": float(max(0.0, np.mean(
                [sum(gain_a_pair["xor" if i % 2 == 0 else "add"] for i in range(int(k))) for k in pair_steps]))
                * 32 / (2 * 28 * T)),
            "accounting": "a step pair = two k32 steps of two cells; packed = 2 IMAD on P plus the step's count; the "
                          "detector, F_t's formation and the exit decode are left out, in the route's favour"}


# ---- pricing ----------------------------------------------------------------------------------------------------

def price(h_lead=(120.0, 230.0)) -> dict:
    """h for the XOR fold beside the lead's, in the lead's W1 units (estimate on the measured lead and LOP3 price)."""
    rows = {}
    for K in (2048, 8192):
        for seal, cells in (("cell", 1), ("row of 8 cells", 8)):
            fold = 16 / 31
            seal_h = tuple(h * 32 / (K * cells) for h in h_lead)
            tot = tuple(fold + s for s in seal_h)
            rows[f"K {K}, seal a {seal}"] = {
                "fold": round(fold, 3), "seal": [round(v, 3) for v in seal_h], "h": [round(v, 3) for v in tot],
                "reduction": [round(h_lead[0] / tot[0], 1), round(h_lead[1] / tot[1], 1)]}
    return {"h_lead": list(h_lead), "rows": rows,
            "time_reading": "one LOP3 a word over wgmma s8 at L 32: +2.13 units a stock MAC (SW-GPU, measured); "
                            "the 3-input fold is half a LOP3 a word"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0xA55E)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {
        "served_ceiling": served_ceiling(),
        "outputs_only": [outputs_only(K) for K in (2048, 8192)],
        "exact": [exact_caps(m) for m in MODELS],
        "linear_fold": check_linear_folds(rng, args.trials),
        "xor_affinity": check_xor_affinity(rng, args.trials),
        "single_word_inj": check_single_word_inj(rng, args.trials),
        "persistent_fibre": persistent_fibre(rng, max(20, args.trials // 4)),
        "sign_flip_32": sign_flip_32(rng, args.trials),
        "superopt": superopt_windows(args.seed),
        "superopt_zero_data": superopt_zero_data(args.seed),
        "injective_luts": injective_luts(),
        "swar_xor": [swar_xor(rng, args.trials * 10, T) for T in (64, 256)],
        "alt_affinity": check_alt_affinity(rng, args.trials),
        "alt_single_word_inj": check_alt_single_word_inj(rng, args.trials),
        "alt_persistent_fibre": alt_persistent_fibre(rng, max(20, args.trials // 4)),
        "alt_sign_flip_32": alt_sign_flip_32(rng, args.trials),
        "superopt_iadd3": superopt_iadd3(args.seed),
        "swar_alt": [swar_alt(rng, args.trials * 10, T) for T in (64, 256)],
        "price": price(),
    }
    fails = []
    sc = res["served_ceiling"]["max_data_macs_a_word"]["D"]
    if not sc < 32:
        fails.append(f"served ceiling: a passing design has {sc} data MACs a word")
    if any(o["min_gap_over_stock"] <= T1 for o in res["outputs_only"]):
        fails.append("outputs-only passes T1 somewhere")
    if res["linear_fold"]["identity_holds"] != args.trials:
        fails.append("linear fold identity failed")
    if res["single_word_inj"]["fold_changed"] != args.trials:
        fails.append("XOR fold not injective in a word")
    if res["superopt"]["x^(x+y): one instruction"] or res["superopt"]["x^(x+y*z): one instruction"]:
        fails.append("a single instruction fuses an add into an XOR")
    if res["superopt"]["pair step sig^S_t^S_t+1 from (sig,Sprev,d1,d2): two instructions"]:
        fails.append("the pair step has a two-instruction program")
    if not res["superopt"]["sanity: pair step given S_t: two instructions"]:
        fails.append("sanity: the pair step given S_t should take two instructions")
    zd = res["superopt_zero_data"]
    if not zd["S_t given its operands: one instruction"] or not zd["sanity: pair step given S_t: two instructions"]:
        fails.append("sanity: the zero-data step should be one IMAD, and the pair step given S_t two instructions")
    if zd["sig^S_t from (sig,Sprev,e1,f1): one instruction"] or zd["pair step from the six: two instructions"]:
        fails.append("the zero-data pair step has a shorter program than three instructions")
    if res["injective_luts"] != ["0x69", "0x96"]:
        fails.append(f"injective LOP3 steps: {res['injective_luts']}, expected only the 3-XOR and its complement")
    for s in res["swar_xor"]:
        if s["decode_exact"] != s["packed_runs"]:
            fails.append(f"SWAR decode not exact at T {s['T']}")
    if res["alt_single_word_inj"]["fold_changed"] != args.trials:
        fails.append("the alternating fold is not injective in a word")
    sf = res["alt_sign_flip_32"]
    if sf["aligned_in_fibre"] != args.trials or sf["mid_step_in_fibre"]:
        fails.append(f"X = 2^31: aligned {sf['aligned_in_fibre']}, mid-step {sf['mid_step_in_fibre']} of {args.trials}")
    si = res["superopt_iadd3"]
    for k in ("pair step from (sig,Sprev,d1,d2): two instructions", "pair step from the pool: one instruction",
              "pair step given S_t+1: one instruction", "zero data: pair step from the six: two instructions",
              "zero data: pair step given S_t+1: one instruction"):
        if si[k]:
            fails.append(f"IADD3 window: {k}: {si[k]}")
    for k in ("sanity: pair step given S_t and S_t+1: one instruction", "sanity: pair step given S_t+1: two instructions",
              "zero data sanity: S_t is one IMAD", "zero data sanity: pair step given S_t: two instructions"):
        if not si[k]:
            fails.append(f"IADD3 window sanity failed: {k}")
    for s in res["swar_alt"]:
        if s["decode_exact"] != s["packed_runs"]:
            fails.append(f"alternating-fold packed route not exact at T {s['T']}: {s['decode_exact']}/{s['packed_runs']}")
    res["fails"] = fails
    text = json.dumps(res, indent=1, default=str)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "assume.json"), "w") as f:
            f.write(text)
    print(text)
    print("FAIL" if fails else "OK", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
