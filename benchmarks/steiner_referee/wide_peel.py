#!/usr/bin/env python3
"""A wide alphabet on the peeled side of the low-rank edit: Theorem E's projection and the format routes, exact
(RS-D2F round 3, CMP-13).

The served product is (A + U Wᵀ)(B + F) − U (Wᵀ (B + F)) = A (B + F): attention (A = P, B = V) or a linear layer
(A n×k, B k×m, both committed). U (n×L) is uniform on a wide alphabet S, and peeled, so its magnitude reaches only A's
int8 headroom. F (k×m) is ±1 and served. W (k×L) is the stacked identities unless stated.

Part 1 (exact over Z/2^32). S is a union of pairs {c, c + 1}, so U = c + β with β a fair bit given the pair. For
every A and B (wrapping values included, attention or linear shape), the second difference of every stride-L prefix
target at (β_{i'a}, β'_{jl'}), with F = −1 + 2β', is 2·W_ja·[i = i'][l = l'][j < (t + 1)L]. Read mod 4, the reduced
targets are W mod 2, as in round 2. Mutation: round 2's ±1 value 4·W mismatches.
Part 2. For each candidate S: δ (the least out-of-E2M1 fraction over shifts and power-of-2 scales), int4's δ, FP6
holdability, the pairing, the headroom 127 − max|S|, and the hybrid's DLV floor one-sided (F ±1) and two-sided.
Part 3 (bit-exact through BLACKWELL_SM120_MXF4). The one-sided hybrid for S = {±1..±8}: a 2·E2M1 core through the
two-window split with an accumulator bias, plus the ±1 correction bit-sliced as popc(m) − 2·popc(m & (σ ^ f)), against
the exact prefixes. Mutations: no correction; no bias; the unsplit P̃, which E2M1 can't encode.
Part 4 (exact). The pure bit-slice: four planes, 12 lane-ops a committed word.
Part 5 (bit-exact through MXF4). The 2-digit split U = 4h + lo as one firing a window ([H | Lo] against [F; F], block
scales 4 : 1), chained on the accumulator: 64 FP4 K for 32 products, the int8 rate.
Part 6. The three-piece 2:4 decomposition of P̃ (each piece at most 2 nonzeros in every group of 4, 96 logical K a
window): exact group feasibility per alphabet and piece scales, the best of a few scale triples chosen per window
(seeded sampling), and the arithmetic bit-exact through the dense BLACKWELL_SM120_NVF4 model standing in for the
sparse instruction (UE4M3 scale 1.25, two-window fields in the normal range).
Part 7. Prices per route: B_1/5 (ρ = 1/32), a ρ = 1/4 vector, and a real-time pipe model at RS-MODEL's rates.

Usage: wide_peel.py SEED [PART ...]
"""

import itertools
import json
import struct
import sys
from fractions import Fraction as Fr
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_MXF4 as MXF4  # noqa: E402
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4 as NVF4  # noqa: E402

M32 = 1 << 32
E2M1_MAG = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
WIDE = [v for v in range(-8, 9) if v]


def alphabet(n):
    return [v for v in range(-n, n + 1) if v]


def pair_bases(S):
    """Bases c of a partition of S into pairs {c, c + 1}, or None."""
    left, bases = sorted(S), []
    while left:
        c = left[0]
        if c + 1 not in left:
            return None
        bases.append(c)
        left.remove(c)
        left.remove(c + 1)
    return bases


# ---------------------------------------------------------------- part 1: the projection, attention and linear shape
def targets(A, B, U, W, F, L):
    At = (A + U.dot(W.T)) % M32
    Bt = (B + F) % M32
    k = A.shape[1]
    out = np.zeros((k // L, A.shape[0], B.shape[1]), dtype=object)
    acc = np.zeros((A.shape[0], B.shape[1]), dtype=object)
    for t in range(k // L):
        blk = slice(t * L, (t + 1) * L)
        acc = (acc + At[:, blk].dot(Bt[blk, :])) % M32
        out[t] = acc
    return out


def rank2_rows(rows):
    basis = []
    for v in rows:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def projection_check(rng, n, k, m, r, L, w_mode, inst, S, factor):
    if inst == "zero":
        A, B = np.zeros((n, k), dtype=object), np.zeros((k, m), dtype=object)
    elif inst == "wrap":
        A = np.array(rng.integers(0, M32, size=(n, k)), dtype=object)
        B = np.array(rng.integers(0, M32, size=(k, m)), dtype=object)
    else:
        A = np.array(rng.integers(-128, 128, size=(n, k)), dtype=object)
        B = np.array(rng.integers(-128, 128, size=(k, m)), dtype=object)
    if w_mode == "eye":
        W = np.array([[int(j % L == a) for a in range(r)] for j in range(k)], dtype=object)
    elif w_mode == "01":
        W = np.array(rng.integers(0, 2, size=(k, r)), dtype=object)
    else:
        W = np.array(2 * rng.integers(0, 2, size=(k, r)) - 1, dtype=object)
    bases = pair_bases(S)
    Ubg = np.array(rng.choice(S, size=(n, r)), dtype=object)
    Fbg = np.array(2 * rng.integers(0, 2, size=(k, m)) - 1, dtype=object)
    mism = checks = 0
    reduced = {}
    for i2, a in itertools.product(range(n), range(r)):
        c = int(rng.choice(bases))
        U0, U1 = Ubg.copy(), Ubg.copy()
        U0[i2, a], U1[i2, a] = c, c + 1
        for j, l2 in itertools.product(range(k), range(m)):
            F0, F1 = Fbg.copy(), Fbg.copy()
            F0[j, l2], F1[j, l2] = -1, 1
            diff = (targets(A, B, U1, W, F1, L) - targets(A, B, U1, W, F0, L)
                    - targets(A, B, U0, W, F1, L) + targets(A, B, U0, W, F0, L)) % M32
            for t, i, l in itertools.product(range(k // L), range(n), range(m)):
                want = (factor * W[j, a] * (i == i2) * (l == l2) * (j < (t + 1) * L)) % M32
                checks += 1
                mism += int(diff[t, i, l] != want)
                if i == i2 == 0 and l == l2 == 0:
                    reduced[(t, a, j)] = (int(diff[t, i, l]) // 2) % 2
    ranks = [rank2_rows([sum(reduced[(t, a, j)] << j for j in range(k)) for a in range(r)]) for t in range(k // L)]
    wmod2 = [rank2_rows([sum((int(W[j, a]) % 2) << j for j in range(min(k, (t + 1) * L))) for a in range(r)])
             for t in range(k // L)]
    return {"part": 1, "shape": [n, k, m], "r": r, "L": L, "W": w_mode, "instance": inst, "factor": factor,
            "checks": checks, "mismatches": mism, "cell00_prefix_ranks_F2": ranks, "W_mod2_prefix_ranks": wmod2}


def part1(rng):
    out = []
    for shape, label in (((4, 4, 2), "attention"), ((3, 4, 5), "linear")):
        for w_mode in ("eye", "01", "pm"):
            for inst in ("random", "zero", "wrap"):
                res = projection_check(rng, *shape, 2, 2, w_mode, inst, WIDE, 2)
                res["label"] = label
                out.append(res)
    mut = projection_check(rng, 3, 4, 5, 2, 2, "eye", "random", WIDE, 4)
    mut["label"] = "mutation: round 2's ±1 value 4·W"
    out.append(mut)
    ok = all(r["mismatches"] == 0 and r["cell00_prefix_ranks_F2"] == r["W_mod2_prefix_ranks"] for r in out[:-1])
    out.append({"part": 1, "verdict": ok and mut["mismatches"] > 0,
                "claim": "second difference 2·W·[…] for wide U, both shapes, every instance; reduced targets = W mod 2; "
                         "the ±1 value 4·W is refuted"})
    return out


# ---------------------------------------------------------------- part 2: alphabets
def held_ints(scale, mags=E2M1_MAG):
    out = set()
    for mg in mags:
        v = Fr(mg).limit_denominator(64) * Fr(scale).limit_denominator(64)
        if v.denominator == 1:
            out |= {int(v), -int(v)}
    return out


E2M3_MAG = tuple([Fr(i, 8) for i in range(8)] + [Fr(8 + i, 8) * 2 ** e for e in range(3) for i in range(8)])


def delta_e2m1(S):
    best, cfg = Fr(1), None
    for s in (Fr(1, 2), 1, 2, 4, 8):
        held = held_ints(s)
        for am in range(-24, 25):
            out = Fr(sum((am + e) not in held for e in S), len(S))
            if out < best:
                best, cfg = out, (am, str(s))
    return best, cfg


def delta_int4(S):
    return min(Fr(sum(not -8 <= c + e <= 7 for e in S), len(S)) for c in range(-24, 25))


def fp6_held(S):
    return any(all((am + e) in held_ints(s, E2M3_MAG) for e in S) for s in (1, 2, 4) for am in range(-8, 9))


CORE, MU = Fr(9, 16), Fr(3, 2) * Fr(1, 32)


def dlv_one_sided(d):
    kappa = min(Fr(15, 2) / 8 + Fr(4, 5) / 4, Fr(4, 5) / 4 / d) / Fr(4, 5) if d else Fr(0)
    return CORE + kappa * d + MU


def dlv_two_sided(d):
    kappa = min(Fr(15, 2) / 8 + Fr(4, 5) / 4, Fr(4, 5) / 4 / d) / Fr(4, 5) if d else Fr(0)
    return CORE + kappa * (2 * d + d * d) + MU


def part2():
    rows = []
    cands = {"{±1..±4}": alphabet(4), "{±1..±6}": alphabet(6), "{±1..±8}": alphabet(8),
             "u8": list(range(-8, 9)), "{±1..±10}": alphabet(10), "{±1..±12}": alphabet(12),
             "{±1..±16}": alphabet(16), "{±1,±2,±4,±5}": [-5, -4, -2, -1, 1, 2, 4, 5]}
    for name, S in cands.items():
        d, cfg = delta_e2m1(S)
        rows.append({"part": 2, "alphabet": name, "size": len(S), "delta_e2m1": str(d), "best_shift_scale": cfg,
                     "delta_int4": str(delta_int4(S)), "fp6_holds": fp6_held(S),
                     "pairs_at_v0": pair_bases(S) is not None, "headroom": 127 - max(abs(v) for v in S),
                     "G_use": [-(127 - max(abs(v) for v in S)), 127 - max(abs(v) for v in S)],
                     "X_dlv_one_sided": round(float(dlv_one_sided(d)), 4),
                     "X_dlv_two_sided": round(float(dlv_two_sided(d)), 4)})
    w = {r["alphabet"]: r for r in rows}
    ok = (w["{±1..±8}"]["delta_e2m1"] == "1/4" and w["{±1..±4}"]["delta_e2m1"] == "0"
          and w["u8"]["delta_e2m1"] == "4/17" and w["{±1..±8}"]["delta_int4"] == "1/16"
          and w["{±1..±8}"]["pairs_at_v0"] and not w["u8"]["pairs_at_v0"]
          and all(r["X_dlv_one_sided"] < 1 for r in rows)
          and w["{±1..±8}"]["X_dlv_two_sided"] > 1)
    rows.append({"part": 2, "verdict": ok,
                 "claim": "δ({±1..±8}) = 1/4, δ_int4 = 1/16, pairs at v = 0 (u8 has none: 17 values); DLV's one-sided "
                          "floor is under honest for every alphabet, its two-sided floor over it at {±1..±8}"})
    return rows


# ---------------------------------------------------------------- part 3: the one-sided hybrid through MXF4
def e2m1_code(v):
    m = abs(v)
    return E2M1_MAG.index(m) | (8 if v < 0 else 0)


H2 = held_ints(2)


def split_wide(v):
    core = v if v in H2 else (4 if abs(v) == 5 else 8) * (1 if v > 0 else -1)
    return core, v - core


def masks(vals):
    return sum(1 << q for q, x in enumerate(vals) if x)


def hybrid_row(p, f, bias=256, correct=True):
    """Prefixes of the row p (P̃, K = S·32) against the column f (±1) through the hybrid; returns the list of T_t."""
    core = [split_wide(int(v))[0] for v in p]
    corr = [split_wide(int(v))[1] for v in p]
    out, T = [], 0
    for w in range(0, len(p), 64):
        q = MXF4.step_scaled(bias * 4096 + bias if bias else 0, [e2m1_code(x / 2) for x in core[w:w + 64]],
                             [e2m1_code(int(x)) for x in f[w:w + 64]], [53, 59], [53, 59])
        for half in (0, 1):
            dcore = ((q >> (12 * half)) & 0xFFF) - bias
            sl = slice(w + 32 * half, w + 32 * half + 32)
            mk = masks([c != 0 for c in corr[sl]])
            sg = masks([c < 0 for c in corr[sl]])
            fs = masks([x < 0 for x in f[sl]])
            dcorr = mk.bit_count() - 2 * (mk & (sg ^ fs)).bit_count()
            T += dcore + (dcorr if correct else 0)
            out.append(T)
    return out


def exact_prefixes(p, f, L=32):
    return [int(v) for v in np.cumsum([int(np.dot(p[w:w + L], f[w:w + L])) for w in range(0, len(p), L)])]


def rows_wide(rng, K, nrand):
    rows = [(rng.choice(WIDE, K), 2 * rng.integers(0, 2, K) - 1) for _ in range(nrand)]
    for v in (8, -8, 7, -7, 5, -5):
        for fv in (1, -1):
            rows.append((np.full(K, v), np.full(K, fv)))
        rows.append((np.full(K, v), np.array([(-1) ** q for q in range(K)])))
    return rows


def part3(rng):
    K = 8 * 32
    rows = rows_wide(rng, K, 60)
    bad = sum(hybrid_row(p, f) != exact_prefixes(p, f) for p, f in rows)
    bad_nc = sum(hybrid_row(p, f, correct=False) != exact_prefixes(p, f) for p, f in rows)
    bad_nb = 0
    for p, f in rows:
        try:
            bad_nb += hybrid_row(p, f, bias=0) != exact_prefixes(p, f)
        except Exception:
            bad_nb += 1
    unenc = 0
    for v in WIDE:
        try:
            e2m1_code(v / 2)
        except ValueError:
            unenc += 1
    res = {"part": 3, "rows": len(rows), "windows": len(rows) * 8, "bad": bad, "mut_no_correction_bad": bad_nc,
           "mut_no_bias_bad": bad_nb, "unsplit_unencodable_fraction": str(Fr(unenc, len(WIDE))),
           "ops_per_word_correction": "LOP3 m&(σ^f), POPC, IADD3 (+1 merge): 3-4",
           "verdict": bad == 0 and bad_nc > 0 and bad_nb > 0 and Fr(unenc, len(WIDE)) == Fr(1, 4),
           "claim": "core 2·E2M1 through the MXF4 two-window split (bias 256) + bit-sliced ±1 correction = exact prefixes; "
                    "dropping the correction or the bias fails; P̃ itself is 1/4 unencodable"}
    return [res]


# ---------------------------------------------------------------- part 4: pure bit-slice
def bitslice_row(p, f):
    out, T = [], 0
    for w in range(0, len(p), 32):
        pw, fw = p[w:w + 32], f[w:w + 32]
        s = masks([x < 0 for x in pw])
        mag = [abs(int(x)) - 1 for x in pw]
        b = [masks([(x >> j) & 1 for x in mag]) for j in range(3)]
        g = s ^ masks([x < 0 for x in fw])
        c0 = 32 + sum((bj.bit_count() << j) for j, bj in enumerate(b))
        p0, p1, p2, p3 = g.bit_count(), (b[0] & g).bit_count(), (b[1] & g).bit_count(), (b[2] & g).bit_count()
        T += c0 - 2 * (p0 + p1 + 2 * p2 + 4 * p3)
        out.append(T)
    return out


def part4(rng):
    rows = rows_wide(rng, 8 * 32, 200)
    bad = sum(bitslice_row(p, f) != exact_prefixes(p, f) for p, f in rows)
    return [{"part": 4, "rows": len(rows), "bad": bad, "ops_per_word": 12,
             "ops": "XOR, POPC, 3×(LOP3, POPC), 2 LEA, IADD3, LEA (C_it per row and window amortized)",
             "pm1_ops_per_word": 3, "verdict": bad == 0,
             "claim": "U·F over 32 k = C − 2(popc γ + popc(b0&γ) + 2 popc(b1&γ) + 4 popc(b2&γ)), exact"}]


# ---------------------------------------------------------------- part 5: 2-digit split, one firing a window
DIGITS = {1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3), 8: (2, 0)}


def digits(v):
    h, lo = DIGITS[abs(v)]
    return (h, lo) if v > 0 else (-h, -lo)


def two_digit_row(p, f, bias=1 << 20):
    c, out = bias, []
    for w in range(0, len(p), 32):
        h = [digits(int(x))[0] for x in p[w:w + 32]]
        lo = [digits(int(x))[1] for x in p[w:w + 32]]
        fw = [e2m1_code(int(x)) for x in f[w:w + 32]]
        c = MXF4.step_scaled(c, [e2m1_code(x) for x in h] + [e2m1_code(x) for x in lo], fw + fw, [54, 52], [53, 53])
        out.append(c - bias)
    return out


def part5(rng):
    rows = rows_wide(rng, 8 * 32, 60)
    bad = sum(two_digit_row(p, f) != exact_prefixes(p, f) for p, f in rows)
    held = held_ints(1)
    ok_digits = all(h in held and lo in held and 4 * h + lo == v for v in WIDE for h, lo in [digits(v)])
    return [{"part": 5, "rows": len(rows), "bad": bad, "digits_held": ok_digits, "fp4_K_per_window": 64,
             "verdict": bad == 0 and ok_digits,
             "claim": "[H | Lo]·[F; F] at block scales 4:1, chained, gives every prefix exactly: one FP4 firing (64 K) "
                      "a 32-product window, the int8 rate, so it ties honest"}]


# ---------------------------------------------------------------- part 6: three 2:4-sparse pieces
SUBSETS = [s for r in (1, 2, 3) for s in itertools.combinations(range(3), r)]


def options(v, scales):
    """Subsets of the pieces that can carry v, each piece a nonzero value of its held set."""
    hs = [held_ints(s) - {0} for s in scales]
    opts = []
    for sub in SUBSETS:
        if len(sub) == 1:
            ok = v in hs[sub[0]]
        elif len(sub) == 2:
            ok = any((v - a) in hs[sub[1]] for a in hs[sub[0]])
        else:
            ok = any((v - a - b) in hs[sub[2]] for a in hs[sub[0]] for b in hs[sub[1]])
        if ok:
            opts.append(sub)
    return tuple(opts)


_memo = {}


def group_drops(key):
    """Least number of entries of a group (given as their option tuples) sent to CUDA so the rest fits 2 per piece."""
    if key in _memo:
        return _memo[key]
    best = len(key)
    for keep in range(len(key), -1, -1):
        for kept in itertools.combinations(range(len(key)), keep):
            for choice in itertools.product(*[key[q] for q in kept]):
                use = [0, 0, 0]
                for sub in choice:
                    for pc in sub:
                        use[pc] += 1
                if max(use) <= 2:
                    best = len(key) - keep
                    break
            if best == len(key) - keep:
                break
        if best == len(key) - keep:
            break
    _memo[key] = best
    return best


def exact_group_stats(S, scales):
    opt = {v: options(v, scales) for v in S}
    classes = {}
    for v in S:
        classes.setdefault(opt[v], 0)
        classes[opt[v]] += 1
    cl = list(classes.items())
    p_inf, e_drop = Fr(0), Fr(0)
    for combo in itertools.product(range(len(cl)), repeat=4):
        w = Fr(1)
        for c in combo:
            w *= Fr(cl[c][1], len(S))
        d = group_drops(tuple(sorted(cl[c][0] for c in combo)))
        p_inf += w * (d > 0)
        e_drop += w * d
    return p_inf, e_drop


TRIPLES = [(2, 2, Fr(5, 4)), (2, 2, Fr(5, 2)), (2, 2, Fr(7, 4)), (2, Fr(5, 2), Fr(7, 4)), (2, 4, Fr(5, 2)),
           (2, 2, Fr(7, 2)), (1, 2, Fr(5, 4)), (2, 2, 3), (2, 4, Fr(7, 2)), (2, 4, 3), (4, 4, Fr(5, 2)),
           (2, Fr(5, 2), Fr(7, 2)), (4, 4, Fr(7, 2)), (2, 4, 4)]


def window_drops(rng, S, nwin):
    opts = [{v: options(v, tr) for v in S} for tr in TRIPLES]
    tot = 0
    for _ in range(nwin):
        p = rng.choice(S, 32)
        tot += min(sum(group_drops(tuple(sorted(o[int(v)] for v in p[g:g + 4]))) for g in range(0, 32, 4))
                   for o in opts)
    return Fr(tot, nwin)


def ue4m3_code(x):
    x = Fr(x)
    for code in range(1, 127):
        e, m = code >> 3, code & 7
        if (Fr(m, 8) * Fr(1, 64) if e == 0 else Fr(8 + m, 8) * Fr(2) ** (e - 7)) == x:
            return code
    raise ValueError(x)


def f32(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def f32val(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def decompose_group(vals):
    """x1, x2 at scale 2 and z at scale 5/4 for one group of 4, and the CUDA remainder."""
    opt = [options(int(v), (2, 2, Fr(5, 4))) for v in vals]
    for keep in range(4, -1, -1):
        for kept in itertools.combinations(range(4), keep):
            for choice in itertools.product(*[opt[q] for q in kept]):
                use = [0, 0, 0]
                for sub in choice:
                    for pc in sub:
                        use[pc] += 1
                if max(use) > 2:
                    continue
                pieces = [[0] * 4 for _ in range(3)]
                cuda = [int(v) for v in vals]
                for q, sub in zip(kept, choice):
                    v = int(vals[q])
                    hs = [sorted(held_ints(s) - {0}, key=abs) for s in (2, 2, Fr(5, 4))]
                    for parts in itertools.product(*[hs[pc] for pc in sub]):
                        if sum(parts) == v:
                            for pc, x in zip(sub, parts):
                                pieces[pc][q] = x
                            cuda[q] = 0
                            break
                return pieces, cuda
    raise AssertionError


def three_piece_row(p, f, B0=512):
    """Prefixes through 3 dense NVF4 firings per window pair (x1|x2 of w1, x1|x2 of w2, z of w1|w2), two fields."""
    out, T = [], 0
    bias = B0 + 4096 * B0
    for w in range(0, len(p), 64):
        pcs, cud = [[], [], []], []
        for g in range(w, w + 64, 4):
            pieces, cuda = decompose_group(p[g:g + 4])
            assert all(sum(x != 0 for x in pc) <= 2 for pc in pieces)
            assert all(sum(pc[q] for pc in pieces) + cuda[q] == int(p[g + q]) for q in range(4))
            for j in range(3):
                pcs[j] += pieces[j]
            cud += cuda
        fw = [e2m1_code(int(x)) for x in f[w:w + 64]]
        x1 = [e2m1_code(Fr(x, 2)) for x in pcs[0]]
        x2 = [e2m1_code(Fr(x, 2)) for x in pcs[1]]
        z = [e2m1_code(Fr(x) / Fr(5, 4)) for x in pcs[2]]
        c = NVF4.step_scaled(f32(bias), x1[:32] + x2[:32], fw[:32] + fw[:32], [64] * 4, [56] * 4)
        c = NVF4.step_scaled(c, x1[32:] + x2[32:], fw[32:] + fw[32:], [112] * 4, [104] * 4)
        c = NVF4.step_scaled(c, z, fw, [58, 58, 106, 106], [56, 56, 104, 104])
        v = int(f32val(c)) - bias
        lo, hi = (v + B0) % 4096 - B0, None
        hi = (v - lo) // 4096
        for d, sl in ((lo, slice(w, w + 32)), (hi, slice(w + 32, w + 64))):
            T += d + int(np.dot(cud[sl.start - w:sl.stop - w], f[sl]))
            out.append(T)
    return out


def part6(rng):
    rows = []
    for name, n in (("{±1..±8}", 8), ("{±1..±10}", 10), ("{±1..±12}", 12), ("{±1..±16}", 16), ("{±1..±24}", 24),
                    ("{±1..±32}", 32)):
        S = alphabet(n)
        for tr in TRIPLES[:2]:
            pinf, ed = exact_group_stats(S, tr)
            rows.append({"part": 6, "alphabet": name, "scales": [str(s) for s in tr], "P_group_overflow": str(pinf),
                         "P_group_overflow_f": round(float(pinf), 5), "E_drops_per_window": round(8 * float(ed), 4)})
        wd = window_drops(rng, S, 3000)
        rows.append({"part": 6, "alphabet": name, "max_abs": n,
                     "scales": f"best of {len(TRIPLES)} triples per window (3000 sampled windows)",
                     "E_drops_per_window": round(float(wd), 4)})
    assert ue4m3_code(Fr(5, 4)) == 58 and ue4m3_code(2) == 64 and ue4m3_code(1) == 56
    assert ue4m3_code(128) == 112 and ue4m3_code(64) == 104 and ue4m3_code(80) == 106
    K = 8 * 32
    data = rows_wide(rng, K, 40)
    bad = sum(three_piece_row(p, f) != exact_prefixes(p, f) for p, f in data)
    rows.append({"part": 6, "rows": len(data), "bad": bad, "logical_K_per_window": 96,
                 "verdict": bad == 0,
                 "claim": "x1 + x2 (scale 2) + z (UE4M3 5/4) with ≤ 2 nonzeros per group of 4 each, plus CUDA for the "
                          "overflow, through the dense NVF4 model (stand-in for OMMA.SF.SP), two fields in the normal "
                          "range: every prefix exact"})
    return rows


# ---------------------------------------------------------------- part 7: prices
FP32_RATE, ALU_RATE, XU_RATE = 3.885 / 4, 2.0 / 4, 0.5 / 4


def real_time(mma, fp32=0.0, alu=0.0, xu=0.0, lsu=0.0, stores=0.0):
    """SMSP cycles per 256 committed words over honest's (2 int8 MMAs, plus the same stores), each pipe and issue."""
    t = max(16 * mma, fp32 / FP32_RATE, alu / ALU_RATE, xu / XU_RATE, mma + fp32 + alu + xu + lsu + stores)
    return t / max(32, 2 + stores)


def part7(drops):
    rho, rho4 = Fr(1, 32), Fr(1, 4)
    kc = Fr(15, 2) * Fr(5, 4) / 8 / Fr(4, 5) + 1
    routes = []

    def add(name, tensor, ops_per_word, rt, note):
        rt_args = dict(rt)
        routes.append({"part": 7, "route": name, "B_1/5": round(float(tensor + ops_per_word * rho), 4),
                       "rho_quarter": round(float(tensor + ops_per_word * rho4), 4),
                       "real_time": round(real_time(**rt_args), 3),
                       "real_time_with_stores": round(real_time(**rt_args, stores=8), 3), "note": note})

    add("honest int8", Fr(1), 0, dict(mma=2), "")
    add("FP4 two-window split, F and U held (round 2 / {±1})", Fr(1, 2), 2, dict(mma=1, fp32=16),
        "closed for wide U")
    add("2-digit split, two split products", Fr(1), 4, dict(mma=2, fp32=32), "ties or loses")
    add("2-digit split, one firing a window", Fr(1), 1, dict(mma=2, alu=8), "ties")
    add("hybrid, sparse adds at DLV's κ = 1", Fr(1, 2), 2 + 8 + Fr(3, 2), dict(mma=1, fp32=16, alu=64),
        "DLV's floor one-sided, μ = 1.5ρ included")
    add("hybrid, sparse adds constructed (κ = 2.46)", Fr(1, 2), 2 + 8 * kc + Fr(3, 2),
        dict(mma=1, fp32=16, alu=64, lsu=10), "P2-FF r3 nibble gather, μ included")
    add("hybrid, bit-sliced ±1 correction", Fr(1, 2), 2 + 3, dict(mma=1, fp32=16, alu=16, xu=8),
        "POPC of a mixed LOP3 word: W4 q ≤ 36·L (head-valued)")
    add("pure bit-slice, 4 planes", Fr(0), 12, dict(mma=0, alu=64, xu=32), "head-valued POPCs")
    add("{±1} XNOR-popcount (reference)", Fr(0), 3, dict(mma=0, alu=16, xu=8), "D4-ATT measured 1.99× int8")
    tail_over = Fr(7, 16)
    add("2:4 tail on its own accumulator (est.)", Fr(3, 4), 2 + Fr(5, 2) + tail_over * kc,
        dict(mma=1.5, fp32=36, alu=8 * float(tail_over), lsu=1), "estimate: sparse semantics unmodelled")
    for name, d in drops.items():
        for ext, tag in ((Fr(13, 4), "straddled chain, 13 FP32 ops per 4 windows"), (Fr(2), "2 ops a word")):
            add(f"three 2:4 pieces at {name}, {tag} (est.)", Fr(3, 4), ext + d * kc,
                dict(mma=1.5, fp32=8 * float(ext), alu=8 * float(d), lsu=1 + 1.3 * float(d)),
                f"estimate: 96 logical K a window, {float(d):.3f} CUDA adds a word; sparse semantics unmodelled")
    return routes


def main():
    rng = np.random.default_rng(int(sys.argv[1]))
    parts = set(sys.argv[2:]) or {"1", "2", "3", "4", "5", "6", "7"}
    drops = {}
    for name, fn in (("1", part1), ("2", lambda r: part2()), ("3", part3), ("4", part4), ("5", part5),
                     ("6", part6)):
        if name in parts:
            for row in fn(rng):
                if name == "6" and "best" in str(row.get("scales")):
                    drops[row["alphabet"]] = Fr(row["E_drops_per_window"]).limit_denominator(10000)
                print(json.dumps(row, default=str), flush=True)
    if "7" in parts:
        for row in part7(drops or {"{±1..±8}": Fr(0)}):
            print(json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
