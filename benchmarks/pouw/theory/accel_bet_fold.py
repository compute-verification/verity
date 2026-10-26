"""ACCEL BET-FOLD (CMP-128): which receipt folds can have no cheap part (FOLD-LUCK's NCP), what they cost on the H200's
data sheet (DS-90), and what any fold's term of ε is when the cheater's fold is priced by what the model can prove.

    python benchmarks/pouw/theory/accel_bet_fold.py [--json PATH] [--quick]

**The folds.** One chain step each, on 32-bit words (`spec_*` below, exact at any width W):

- XC: `y ← y ⊕ w ⊕ w′` (one `LOP3.LUT 0x96`, ½ ALU a word); a dp4a word by `y ← y·K + d` (one IMAD).
- li (FOLD-LUCK §2, `segment.py` `_li_absorb`): `y ← ((y ⊕ w₀ ⊕ K)·A + w₁)`, A odd (½ ALU + ½ IMAD a word).
- li-down (FOLD-LUCK §8's candidate): li, then `y ← y + (y >> 16)` (one `IMAD.HI`) before the checkpoint.
- RA, rotate-add: `y ← rotl(y, 1) + w`, then two steps `y ← rotl(y, 1) + c` with constants (`ra_finish`) a chain
  before the checkpoint, so the last word's top bits reach the low ones. ptxas 12.9 at `-arch=sm_90` emits each step
  as one `LEA.HI R, y, w, y, s` (rotl then add; the note has the SASS), an integer instruction off IMAD, so DS-90's
  ALU class (64 a clock): 1 ALU a word.
- MW, multiply-wide: a two-register state (a, b) takes the pair (w_e, w_o) by `V = a·K + w_e` (64 bits) and
  `(a, b) ← (hi V + b + w_o, lo V)`: one `IMAD.WIDE.U32` with a 64-bit addend (w_e, b + w_o) and one IADD3 a pair,
  then one step with no words before the checkpoint. DS-90 has no IMAD.WIDE line (UNLISTED); `MW_SLOTS` prices it at
  2 IMAD slots, its two 32-bit results, and the table shows 1 beside it.
- FE, the IMAD pipe alone: `r ← r·K + w` (IMAD), then the Feistel step `(l, r) ← (r, l + hi(r·K′))` (IMAD.HI),
  2 IMAD a word; RA's design offers it as a second route, for the IMAD pipe RA leaves idle.

**Lemma T** (paper proof in the note; checked below). Call a step triangular (a T-step) if output bit j reads only input
bits ≤ j, as LOP3, IADD3, IMAD by a constant and XOR all are. A T-step bijective in an input passes a flip of that
input's top bit to a flip of exactly the output's top bit. So (1) if a word reaches the checkpoint through T-steps
alone, then C mod 2^(j+1) ignores the word's bits above j, and 2^(j+1)·e_w is in K₊(C_j); (2) if two words meet through
T-steps before either passes a non-T step, their top-bit pair is in K_⊕ of every component. NCP therefore needs a
non-T step, bijective in the state, after every word and between any two words of one chain. li, XC and li-down fail
it; MW and FE pass it at every context checked, and RA at every start state but those with rotl(y0, 1) ≡ 0 mod 2^(W−2)
(a 2^−30 fraction at 32 bits), exhaustively at small widths (`ncp`). `ncp` also counts NCP⁺'s failures (a component
whose derivative is constant, 0 or 1, as cheap as an invariant one) and prints the largest probability that a nonzero
change of the words leaves the checkpoint unchanged. NCP is necessary for FOLD-LUCK's slot, not sufficient.

**The cost floor at L = 32.** RA costs 1 ALU a word, and the ALU retires 64 a clock against the tensor core's 128
words, so RA alone gives γ ≥ 2 there. MW costs ½ ALU + `MW_SLOTS`/2 IMAD. Among the non-T forms on DS-90's classes
(`NCP_FORMS`), the most words a clock an NCP fold can absorb is 96 at two IMAD slots for IMAD.WIDE (γ ≥ 4/3), and 127
at one. So FOLD-LUCK's budget of 0.54 ALU + 0.54 IMAD a word at L = 32 is met only if IMAD.WIDE with a 64-bit addend
costs one IMAD slot. At L = 64 RA fits on the ALU alone.

**Theorem F, the kill** (`barrier`; the proof is in the note). Let T_F(ε) say that every program outputting the receipt
R_F(A, B, σ) costs at least (1 − ε)·H, H honest's cost, and φ = 1 − 1/γ_plain honest's fold share. (a) Composing any
program for the words with honest's fold gives one for R_F, so T_F(ε) proves, from θ alone, that every program for the
words (the exact segment products) costs at least (1 − ε − φ)·H: SAK's content, which R-ACCEL-6 takes as the one
premise because no proof of it is known. (b) A reduction from SAK that uses a program for R_F only through its outputs
computes the words from q·32 bits a set of 4096 words and its own work, so it refutes SAK given that much advice. (c)
Standard crypto binds a receipt only if the checkpoint is a cryptographic primitive, which makes the fold a hash
(Consequence 2). With sparse checkpoints, T_F uniform in the shape is also a superlinear circuit lower bound for R_F.

**The fold's term of ε** (`table`). Each LP is over one SM-clock at DS-90's rates (`machine_model/data/devices/sm_90/
sheet.toml`: tensor 4096 s8 MACs, issue 128 lane slots, ALU 64, IMAD 64, SHFL 32, the uniform pipe
3200000000/49027679), in s8 MACs a clock, and is solved by an exact two-phase simplex whose optimum is certified by
strong duality. Honest: credited words on honest's routes, the most MACs a clock. Cheater: at least honest's credited
MACs, plus other work (worst-case int8 MACs: the tensor core, dp4a, FFMA, or bit-sliced on the ALU at `B_SLICE` lane
instructions a MAC), with every credited word folded by the cheapest routes the premise allows. ε = cheater/honest − 1.
Every premise reads the words as the fold program's inputs (NO-BYPASS). By Theorem B nothing is provable from
(A, B, σ), and under R-ACCEL-6 no premise but SAK(matmul) is available, so these figures are the words-as-inputs part
only:

- FREE: no fold premise, so the cheater's fold costs nothing. This caps ε_fold over the same honest kernel.
- CENSUS: main's proved `FanInCount` with the measured `FanIn` census (`accounting-ts-fan-in`, sm_90's merges a class).
- SEM: the bit-conservation count, Σ_live (bits read − live bits written) ≥ 32(N − |O|) (an identity, checked on random
  dataflows by `conservation`), with three named lemmas. GL: guards, predicates and kept destinations add no merge. NL:
  an instruction whose only live output is a predicate kills no word. MS: a load, a store or a shuffle moves words; an
  atomic or bulk XOR merges one word a word. Then a three-input lane form merges 2, REDUX 31 a warp instruction,
  IMMA.16832 192, and STS.128 plus a bulk XOR reduction 1 a word at a quarter issue slot.
- FOLD-OPT: main's premise, which prices the cheater's fold at honest's (with FILL-OPT for li's filler).

Exits 1 if a check fails. Note: compute-research/20261026T0150Z-draft-accel-bet-fold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from fractions import Fraction as F

import numpy as np

FAIL: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


# --------------------------------------------------------------------------------------------------------------------
# The folds' chain steps, exact at width W (ints or numpy uint64 arrays; W ≤ 32 keeps a·K inside 64 bits)

LI_A, LI_K = 0x9E3779B1, 0x5A5A5A5A
MW_K = 0x9E3779B1
FE_K = 0x85EBCA6B
RA_S = 1
RA_C = (0x7F4A7C15, 0xF39CC060)   # the finalizer's two constants


def top(c, W):
    """A 32-bit constant at width W: its top W bits (so a multiplier stays odd and spread at small widths)."""
    return c >> (32 - W)


def rotl(y, s, W):
    m = (1 << W) - 1
    return ((y << s) | (y >> (W - s))) & m


def spec_xc(ws, ctx, W):
    m = (1 << W) - 1
    (y,) = ctx
    for i in range(0, len(ws), 2):
        y = y ^ ws[i] ^ (ws[i + 1] if i + 1 < len(ws) else 0)
    return y & m


def spec_li(ws, ctx, W):
    m = (1 << W) - 1
    A, K = top(LI_A, W) | 1, top(LI_K, W)
    (y,) = ctx
    for i in range(0, len(ws), 2):
        y = (((y ^ ws[i] ^ K) * A) + (ws[i + 1] if i + 1 < len(ws) else 0)) & m
    return y


def spec_li_down(ws, ctx, W):
    m = (1 << W) - 1
    y = spec_li(ws, ctx, W)
    return (y + (y >> (W // 2))) & m


def ra_finish(y, W):
    """Two more `LEA.HI` steps a chain, with constants for words: the last word's bits reach every bit of the state."""
    m = (1 << W) - 1
    for c in RA_C:
        y = (rotl(y, RA_S, W) + top(c, W)) & m
    return y


def spec_ra(ws, ctx, W):
    m = (1 << W) - 1
    (y,) = ctx
    for w in ws:
        y = (rotl(y, RA_S, W) + w) & m
    return ra_finish(y, W)


def ra_bad_context(y0, W):
    """The start states at which RA's first addition carries nothing out of its low bits, so a change of the first word
    in bit W − 2 passes as an XOR and the second word's top bit cancels it: rotl(y0, s) ≡ 0 mod 2^(W−2)."""
    return rotl(y0, RA_S, W) % (1 << (W - 2)) == 0


def spec_mw(ws, ctx, W):
    m = (1 << W) - 1
    K = top(MW_K, W) | 1
    a, b = ctx
    for i in range(0, len(ws), 2):
        we, wo = ws[i], (ws[i + 1] if i + 1 < len(ws) else 0)
        V = a * K + we
        a, b = ((V >> W) + b + wo) & m, V & m
    V = a * K
    a, b = ((V >> W) + b) & m, V & m
    return a ^ b


def spec_fe(ws, ctx, W):
    """FE, the IMAD pipe alone: r ← r·K + w (IMAD), then the Feistel step (l, r) ← (r, l + hi(r·K′)) (IMAD.HI)."""
    m = (1 << W) - 1
    K, K2 = top(MW_K, W) | 1, top(FE_K, W) | 1
    l, r = ctx
    for w in list(ws) + [0]:
        r = (r * K + w) & m
        l, r = r, (l + ((r * K2) >> W)) & m
    return l ^ r


def spec_fa(ws, ctx, W):
    """FA: FE with the word added, not multiplied in: r ← r + w (IADD3), then the same Feistel step (IMAD.HI)."""
    m = (1 << W) - 1
    K2 = top(FE_K, W) | 1
    l, r = ctx
    for w in list(ws) + [0]:
        r = (r + w) & m
        l, r = r, (l + ((r * K2) >> W)) & m
    return l ^ r


SPECS = {"XC": (spec_xc, 1), "li": (spec_li, 1), "li-down": (spec_li_down, 1), "RA": (spec_ra, 1), "MW": (spec_mw, 2),
         "FE": (spec_fe, 2), "FA": (spec_fa, 2)}


# --------------------------------------------------------------------------------------------------------------------
# NCP, exhaustively at small widths (FOLD-LUCK §8)

def _odd_masks(W):
    """For each nonzero S, the 2^W-bit mask of the x with S·x = 1 (W ≤ 6, so a mask fits a uint64)."""
    assert W <= 6
    return [np.uint64(sum(1 << x for x in range(1 << W) if (S & x).bit_count() & 1)) for S in range(1, 1 << W)]


def _span_full(presence, masks):
    """The values a presence mask (bit x set when x occurs) holds span F₂^W: no nonzero S is 0 on all of them."""
    ok = np.ones(presence.shape, dtype=bool)
    for M in masks:
        ok &= (presence & M) != 0
    return ok


def ncp(name, W, n, contexts):
    """For one checkpoint of n words at width W, over the contexts given and every nonzero d, XOR and additive: the d
    that leave some component of C invariant for every w at some context (NCP fails there), the contexts where that
    happens, and the largest probability over w that a d leaves the whole checkpoint unchanged, at the contexts where NCP
    holds (the cheater knows the context, so the most over them)."""
    fold, _ = SPECS[name]
    m = (1 << W) - 1
    idx = np.arange(1 << (W * n), dtype=np.uint64)
    words = [(idx >> np.uint64(W * k)) & np.uint64(m) for k in range(n)]
    C = np.stack([fold(words, tuple(np.uint64(c) for c in ctx), W) for ctx in contexts])
    masks = _odd_masks(W)
    one = np.uint64(1)
    out, bad_ctx, bad_aff, P = {}, set(), set(), {}
    for law in ("xor", "add"):
        invariant = affine = 0
        P[law] = []
        for d in range(1, 1 << (W * n)):
            if law == "xor":
                j = idx ^ np.uint64(d)
            else:
                j = np.zeros_like(idx)
                for k in range(n):
                    j |= ((words[k] + np.uint64((d >> (W * k)) & m)) & np.uint64(m)) << np.uint64(W * k)
            X = C[:, j.astype(np.int64)] ^ C
            ok = _span_full(np.bitwise_or.reduce(one << X, axis=1), masks)
            if not ok.all():
                invariant += 1
                bad_ctx.update(i for i in range(len(contexts)) if not ok[i])
            ok_aff = _span_full(np.bitwise_or.reduce(one << (X ^ X[:, :1]), axis=1), masks)
            if not ok_aff.all():
                affine += 1
                bad_aff.update(i for i in range(len(contexts)) if not ok_aff[i])
            P[law].append((X == 0).mean(axis=1))
        out[law] = {"invariant_d": invariant, "constant_derivative_d": affine}
    out["bad_contexts_constant_derivative"] = [list(map(int, contexts[i])) for i in sorted(bad_aff)]
    good = [i for i in range(len(contexts)) if i not in bad_ctx]
    for law in ("xor", "add"):
        arr = np.stack(P[law])
        out[law]["worst_p"] = float(arr[:, good].max()) if good else None
        out[law]["worst_d"] = int(arr[:, good].max(axis=1).argmax()) + 1 if good else None
    out["bad_contexts"] = [list(map(int, contexts[i])) for i in sorted(bad_ctx)]
    out["contexts"] = len(contexts)
    return out


def ncp_all(quick):
    """NCP at (W, n): every context for a one-register state, sixteen seeded ones for a two-register state."""
    cases = [(4, 3)] if quick else [(4, 3), (5, 2), (6, 2)]
    res = {}
    print()
    print("NCP at small widths: the nonzero d (XOR, additive) leaving a component of C invariant for every w at some "
          "context; the contexts where one does; and the largest Pr_w[C(w ⋆ d) = C(w)] at the other contexts")
    print("(NCP⁺ columns: the d giving some component a constant derivative, 0 or 1, and the contexts where one does)")
    print(f"{'fold':8} {'W':>2} {'n':>2} {'ctx':>4}  {'K⊕ d':>6} {'K₊ d':>6} {'bad ctx':>7}  {'max p ⊕':>8} {'max p +':>8}"
          f"  {'⁺⊕ d':>6} {'⁺₊ d':>6} {'⁺ ctx':>6}")
    for W, n in cases:
        for name, (_, regs) in SPECS.items():
            if regs == 1:
                ctxs = [(c,) for c in range(1 << W)]
            else:
                rng = random.Random(f"{name}/W{W}/n{n}")
                ctxs = [(rng.randrange(1 << W), rng.randrange(1 << W)) for _ in range(16)]
            r = ncp(name, W, n, ctxs)
            res[f"{name}/W{W}/n{n}"] = r
            fmt = lambda v: "   n/a" if v is None else f"{v:8.4f}"
            print(f"{name:8} {W:2} {n:2} {len(ctxs):4}  {r['xor']['invariant_d']:6} {r['add']['invariant_d']:6} "
                  f"{len(r['bad_contexts']):7}  {fmt(r['xor']['worst_p'])} {fmt(r['add']['worst_p'])}  "
                  f"{r['xor']['constant_derivative_d']:6} {r['add']['constant_derivative_d']:6} "
                  f"{len(r['bad_contexts_constant_derivative']):6}")
    clean = lambda r: not r["bad_contexts"] and not r["bad_contexts_constant_derivative"]
    for W, n in cases:
        g = lambda nm, law, W=W, n=n: res[f"{nm}/W{W}/n{n}"][law]["invariant_d"]
        check(all(g(nm, "xor") > 0 for nm in ("XC", "li", "li-down")),
              f"W={W}, n={n}: XC, li and li-down fail NCP under XOR (the top-bit pair, Lemma T (2))")
        check(all(g(nm, "add") > 0 for nm in ("XC", "li", "li-down")),
              f"W={W}, n={n}: XC, li and li-down fail NCP under addition (Lemma T (1))")
        ra = res[f"RA/W{W}/n{n}"]
        if W == 4:
            check(all(ra_bad_context(c[0], W) for c in ra["bad_contexts"]),
                  f"W={W}, n={n}: RA fails NCP only at the start states with rotl(y0, 1) ≡ 0 mod 2^(W−2) "
                  f"({len(ra['bad_contexts'])} of {ra['contexts']})")
        else:
            check(len(ra["bad_contexts"]) == ra["contexts"],
                  f"W={W}, n={n}: RA fails NCP at every start state (rotl by 1 lifts a difference's lowest set bit one "
                  "place a step, so the low bits miss a word until it wraps)")
        check(len(ra["bad_contexts_constant_derivative"]) == ra["contexts"],
              f"W={W}, n={n}: RA fails NCP⁺ at every start state (2^(W−1) on the last word flips bits 0 and 1 together)")
        check(all(clean(res[f"{nm}/W{W}/n{n}"]) for nm in ("FE", "FA")),
              f"W={W}, n={n}: FE and FA meet NCP and NCP⁺ at every context checked")
    if len(cases) > 1:
        check(any(not clean(res[f"MW/W{W}/n{n}"]) for W, n in cases),
              "MW fails NCP or NCP⁺ at some context checked (" +
              ", ".join(f"W={W}: {len(res[f'MW/W{W}/n{n}']['bad_contexts'])} and "
                        f"{len(res[f'MW/W{W}/n{n}']['bad_contexts_constant_derivative'])}" for W, n in cases) + ")")
    return res


def facts_32(trials=2000):
    """Lemma T's two failures, RA's best two-word change, and li-down's step, at 32 bits on seeded words."""
    rng = random.Random(0x32B17)
    W, m = 32, (1 << 32) - 1
    out = {}
    li_pair = xc_pair = ra_hit = 0
    for _ in range(trials):
        ws = [rng.randrange(1 << W) for _ in range(8)]
        ctx = (rng.randrange(1 << W),)
        i, j = rng.sample(range(8), 2)
        flip = list(ws)
        flip[i] ^= 1 << 31
        flip[j] ^= 1 << 31
        li_pair += spec_li(flip, ctx, W) == spec_li(ws, ctx, W)
        bit = rng.randrange(W)
        a = 2 * rng.randrange(4)
        f2 = list(ws)
        f2[a] ^= 1 << bit
        f2[a + 1] ^= 1 << bit
        xc_pair += spec_xc(f2, ctx, W) == spec_xc(ws, ctx, W)
        k = rng.randrange(7)
        f3 = list(ws)
        f3[k] = (f3[k] + (1 << 31)) & m
        f3[k + 1] = (f3[k + 1] - (1 << (RA_S - 1))) & m
        ra_hit += spec_ra(f3, ctx, W) == spec_ra(ws, ctx, W)
    check(li_pair == trials, f"li at 32 bits: flipping bit 31 of any two words of a checkpoint keeps it, {li_pair}/{trials}")
    check(xc_pair == trials, f"XC at 32 bits: one bit flipped in both words of an XOR3 keeps it, {xc_pair}/{trials}")
    frac = ra_hit / trials
    check(0.4 < frac < 0.6, f"RA at 32 bits: (+2^31 on w_k, −2^(s−1) on w_(k+1)) keeps the checkpoint with probability "
                            f"about 1/2 ({ra_hit}/{trials}): bit 31 of the state after w_k decides the carry")
    y1, y2 = (1 << 32) - 1, 65534
    down = lambda y: (y + (y >> 16)) & m
    check(down(y1) == down(y2) and y1 != y2, "li-down's step y + (y >> 16) is no bijection at 32 bits: "
                                             f"{y1:#x} and {y2:#x} both map to {down(y1):#x}")
    out.update(li_pair=li_pair, xc_pair=xc_pair, ra_half=ra_hit, trials=trials)
    return out


# --------------------------------------------------------------------------------------------------------------------
# XC and RA as laid out on a warp: two chains a lane, one word an entry a set, the checkpoint XORed across lanes

def receipt(design, words, dps, keys, width=32, K=None):
    """`words[t][l]`: lane l's tensor-core words in set t (an even count, half to each chain); `dps[t][l]`: its dp4a
    word or None; `keys[l]`: its two chain starts. XC: XOR3 two at a time, the dp4a word by y·K + d. RA: `LEA.HI` a word,
    the dp4a word on chain 0, and `ra_finish`'s two steps a chain before each checkpoint."""
    m = (1 << width) - 1
    K = top(0x9E3779B1, width) | 1 if K is None else K
    s = RA_S
    y = [list(k) for k in keys]
    cps = []
    for t, row in enumerate(words):
        for lane, ws in enumerate(row):
            h = len(ws) // 2
            for c, part in enumerate((ws[:h], ws[h:])):
                if design == "XC":
                    for i in range(0, len(part), 2):
                        y[lane][c] ^= part[i] ^ part[i + 1]
                else:
                    for w in part:
                        y[lane][c] = (rotl(y[lane][c], s, width) + w) & m
            d = dps[t][lane]
            if d is not None:
                y[lane][0] = (y[lane][0] * K + d) & m if design == "XC" else (rotl(y[lane][0], s, width) + d) & m
            if design == "RA":
                y[lane] = [ra_finish(v, width) for v in y[lane]]
        r = 0
        for lane in range(len(row)):
            r ^= y[lane][0] ^ y[lane][1]
        cps.append(r)
    return tuple(cps)


def prefix_words(chunks, width):
    m = (1 << width) - 1
    out, acc = [], None
    for row in chunks:
        acc = [list(r) for r in row] if acc is None else [[(a + c) & m for a, c in zip(ar, cr)] for ar, cr in zip(acc, row)]
        out.append([list(r) for r in acc])
    return out


def random_layout(rng, width, sets=3, lanes=4, per_lane=8, dp_lanes=(0, 2)):
    m = (1 << width) - 1
    chunks = [[[rng.randrange(m + 1) for _ in range(per_lane)] for _ in range(lanes)] for _ in range(sets)]
    dch = [[rng.randrange(m + 1) if lane in dp_lanes else None for lane in range(lanes)] for _ in range(sets)]
    keys = [(rng.randrange(m + 1), rng.randrange(m + 1)) for _ in range(lanes)]
    words = prefix_words(chunks, width)
    dps, acc = [], [0] * lanes
    for row in dch:
        acc = [(a + (d or 0)) & m for a, d in zip(acc, row)]
        dps.append([acc[lane] if row[lane] is not None else None for lane in range(lanes)])
    return words, dps, keys


def positions(words, dps):
    for t, row in enumerate(words):
        for lane, ws in enumerate(row):
            for k in range(len(ws)):
                yield ("w", t, lane, k)
            if dps[t][lane] is not None:
                yield ("d", t, lane, None)


def with_value(words, dps, pos, v):
    w2 = [[list(ws) for ws in row] for row in words]
    d2 = [list(row) for row in dps]
    kind, t, lane, k = pos
    if kind == "w":
        w2[t][lane][k] = v
    else:
        d2[t][lane] = v
    return w2, d2


def value_at(words, dps, pos):
    kind, t, lane, k = pos
    return words[t][lane][k] if kind == "w" else dps[t][lane]


def persist(words, dps, pos, X, width):
    """X added to one entry's committed words from set t₀ on (a skipped chunk carried by the prefix sums)."""
    m = (1 << width) - 1
    w2 = [[list(ws) for ws in row] for row in words]
    d2 = [list(row) for row in dps]
    kind, t0, lane, k = pos
    for t in range(t0, len(words)):
        if kind == "w":
            w2[t][lane][k] = (w2[t][lane][k] + X) & m
        else:
            d2[t][lane] = (d2[t][lane] + X) & m
    return w2, d2


def binding(quick=False) -> dict:
    """Injectivity of the warp receipts: in any one word and in any one entry's persistent pattern (fibre 1, no key
    premise), exhaustively at width 8 and on seeded 32-bit trials; XC's multi-entry class; the Latin LUTs."""
    out = {}
    for design in ("XC", "RA"):
        rng = random.Random(0xB3F01D)
        W = 8
        words, dps, keys = random_layout(rng, W)
        pos = list(positions(words, dps))
        bad = 0
        for p in pos:
            seen = set()
            for v in range(1 << W):
                seen.add(receipt(design, *with_value(words, dps, p, v), keys, W))
            bad += len(seen) != (1 << W)
        check(bad == 0, f"{design}: one word at width {W}: the checkpoints are injective in it at all {len(pos)} "
                        f"positions (fibre 1)")
        bad = 0
        for p in pos:
            t0 = p[1]
            first = {receipt(design, *persist(words, dps, p, X, W), keys, W)[t0] for X in range(1 << W)}
            bad += len(first) != (1 << W)
        check(bad == 0, f"{design}: one entry's persistent pattern at width {W}: set t₀'s checkpoint is injective in X "
                        f"at every entry and t₀ (fibre 1)")
        rng32 = random.Random(0x5EED32)
        w32, d32, k32 = random_layout(rng32, 32, sets=4, lanes=8, per_lane=16, dp_lanes=(1, 4, 6))
        b32 = receipt(design, w32, d32, k32)
        pos32 = list(positions(w32, d32))
        ok, trials = True, 0
        for _ in range(500 if quick else 3000):
            p = rng32.choice(pos32)
            v = rng32.randrange(1 << 32)
            if v == value_at(w32, d32, p):
                continue
            trials += 1
            ok &= receipt(design, *with_value(w32, d32, p, v), k32) != b32
            X = rng32.randrange(1, 1 << 32)
            ok &= receipt(design, *persist(w32, d32, p, X, 32), k32)[p[1]] != b32[p[1]]
        check(ok, f"{design}: at 32 bits ({trials} seeded trials over {len(pos32)} positions) one wrong word, and one "
                  f"entry's persistent X, always move the checkpoints")
        out[design] = {"positions_w8": len(pos), "trials_32": trials}

    rng = random.Random(0xB3F01D)
    W = 8
    words, dps, keys = random_layout(rng, W)
    base = receipt("XC", words, dps, keys, W)
    p, q = ("w", 1, 1, 0), ("w", 1, 1, 1)
    delta = 0x5A
    w2, d2 = with_value(words, dps, p, value_at(words, dps, p) ^ delta)
    w2, d2 = with_value(w2, d2, q, value_at(w2, d2, q) ^ delta)
    check(receipt("XC", w2, d2, keys, W) == base, "XC control: two entries of one XOR3 with errors (w ⊕ δ, w′ ⊕ δ) "
                                                  "cancel (the multi-entry class)")
    a, b = 0x37, 0x37 ^ (1 << (W - 1))
    X = 0x2C
    m = (1 << W) - 1
    check(((a + X) & m) ^ ((b + X) & m) == a ^ b, "XC control: (p + X) ⊕ (p′ + X) = p ⊕ p′ when p′ − p = 2^(W−1), so a "
                                                  "shared skipped X cancels without being known")
    lat = latin_luts()
    check(lat == [0x69, 0x96], f"of the 256 three-input LUTs only {[hex(x) for x in lat]} (XNOR3, XOR3) are bijective "
                               f"in each input for all values of the others, so a LOP3 that absorbs two words is XOR3")
    out["latin_luts"] = lat
    return out


def latin_luts() -> list[int]:
    good = []
    for lut in range(256):
        f = lambda a, b, c, lut=lut: (lut >> (a << 2 | b << 1 | c)) & 1
        if all(f(0, b, c) != f(1, b, c) for b in (0, 1) for c in (0, 1)) and \
           all(f(a, 0, c) != f(a, 1, c) for a in (0, 1) for c in (0, 1)) and \
           all(f(a, b, 0) != f(a, b, 1) for a in (0, 1) for b in (0, 1)):
            good.append(lut)
    return good


def tri_steps(W=4) -> None:
    """Lemma T's step: a T-function bijective in an input maps a top-bit flip of it to exactly a top-bit flip, for the
    steps the folds use, exhaustively at width W."""
    m = (1 << W) - 1
    tb = 1 << (W - 1)
    A, K = top(LI_A, W) | 1, top(LI_K, W)
    steps = {
        "XOR3 (y)": lambda y, w: y ^ w ^ 5, "IADD3 (y)": lambda y, w: (y + w + 3) & m,
        "IMAD y·A + w (y)": lambda y, w: (y * A + w) & m, "IMAD y·A + w (w)": lambda w, y: (y * A + w) & m,
        "li half step (y)": lambda y, w: ((y ^ w ^ K) * A + 7) & m,
    }
    bad = [nm for nm, f in steps.items() for y in range(1 << W) for w in range(1 << W) if f(y ^ tb, w) != f(y, w) ^ tb]
    check(not bad, f"Lemma T's step at width {W}: XOR3, IADD3, IMAD by a constant and li's half step each turn a top-bit "
                   f"flip of the input into exactly a top-bit flip of the output, at every value")
    moved = sum(rotl(y ^ tb, RA_S, W) ^ rotl(y, RA_S, W) == 1 << (RA_S - 1) for y in range(1 << W))
    check(moved == 1 << W, f"RA's rotation at width {W} moves the top-bit flip to bit s − 1 at every value (a non-T step)")


# --------------------------------------------------------------------------------------------------------------------
# The bit-conservation identity on random dataflows

def conservation(seed=0xC0DE, trials=3000) -> None:
    """Σ over live instructions of (bits read − live bits written) ≥ bits of live inputs − bits of O, for random
    straight-line programs with multi-output instructions of mixed widths (32-bit words, 64-bit pairs, 1-bit
    predicates). Live = feeds O. Equality is not required; the slack is reads past a value's first."""
    rng = random.Random(seed)
    worst = None
    for _ in range(trials):
        n_in = rng.randint(1, 12)
        width = {("i", j): rng.choice((32, 32, 32, 64, 1)) for j in range(n_in)}
        vals = list(width)
        instrs = []
        for i in range(rng.randint(1, 30)):
            src = rng.sample(vals, rng.randint(1, min(5, len(vals))))
            outs = [("o", i, r) for r in range(rng.randint(1, 3))]
            for o in outs:
                width[o] = rng.choice((32, 32, 64, 1))
            instrs.append((src, outs))
            vals += outs
        O = set(rng.sample(vals, rng.randint(1, min(3, len(vals)))))
        live = set(O)
        for src, outs in reversed(instrs):
            if any(o in live for o in outs):
                live.update(src)
        lhs = sum(sum(width[s] for s in src) - sum(width[o] for o in outs if o in live)
                  for src, outs in instrs if any(o in live for o in outs))
        rhs = sum(width[v] for v in live if v[0] == "i") - sum(width[v] for v in O)
        if lhs < rhs:
            check(False, f"conservation: {lhs} < {rhs}")
            return
        worst = lhs - rhs if worst is None else min(worst, lhs - rhs)
    check(True, f"conservation identity on {trials} random dataflows (mixed widths, multi-output): never violated, "
                f"least slack {worst}")


# --------------------------------------------------------------------------------------------------------------------
# An exact two-phase simplex, certified by duality

def simplex(c, A, b):
    """max c·x subject to A x ≤ b, x ≥ 0, b of any sign, in Fractions (two phases, Bland's rule). Returns (value, x, y)
    with y the duals, after checking primal feasibility, dual feasibility and a zero duality gap."""
    m, n = len(A), len(c)
    neg = [i for i in range(m) if F(b[i]) < 0]
    na = len(neg)
    N = n + m + na
    T, basis = [], []
    for i in range(m):
        s = -1 if i in neg else 1
        row = [s * F(v) for v in A[i]] + [F(s if i == j else 0) for j in range(m)] + [F(0)] * na + [s * F(b[i])]
        if i in neg:
            row[n + m + neg.index(i)] = F(1)
            basis.append(n + m + neg.index(i))
        else:
            basis.append(n + i)
        T.append(row)

    def zrow(cv):
        return [sum(cv[basis[i]] * T[i][j] for i in range(m)) - cv[j] for j in range(N)] + \
               [sum(cv[basis[i]] * T[i][-1] for i in range(m))]

    def run(z, allowed):
        while True:
            e = next((j for j in allowed if z[j] < 0), None)
            if e is None:
                return z
            rows = [(T[i][-1] / T[i][e], basis[i], i) for i in range(m) if T[i][e] > 0]
            if not rows:
                raise ValueError("unbounded")
            r = min(rows)[2]
            piv = T[r][e]
            T[r] = [v / piv for v in T[r]]
            for i in range(m):
                if i != r and T[i][e] != 0:
                    f = T[i][e]
                    T[i] = [a - f * p for a, p in zip(T[i], T[r])]
            f = z[e]
            z = [a - f * p for a, p in zip(z, T[r])]
            basis[r] = e

    if na:
        z = run(zrow([F(0)] * (n + m) + [F(-1)] * na), range(N))
        if z[-1] < 0:
            raise ValueError("infeasible")
        for i in range(m):
            if basis[i] >= n + m:
                e = next((j for j in range(n + m) if T[i][j] != 0), None)
                if e is None:
                    raise ValueError("redundant row")
                piv = T[i][e]
                T[i] = [v / piv for v in T[i]]
                for k in range(m):
                    if k != i and T[k][e] != 0:
                        f = T[k][e]
                        T[k] = [a - f * p for a, p in zip(T[k], T[i])]
                basis[i] = e
    z = run(zrow([F(v) for v in c] + [F(0)] * (m + na)), range(n + m))
    x = [F(0)] * n
    for i, j in enumerate(basis):
        if j < n:
            x[j] = T[i][-1]
    y = [z[n + i] for i in range(m)]
    val = sum(F(ci) * xi for ci, xi in zip(c, x))
    assert all(xi >= 0 for xi in x), "primal sign"
    assert all(sum(F(A[i][j]) * x[j] for j in range(n)) <= F(b[i]) for i in range(m)), "primal infeasible"
    assert all(yi >= 0 for yi in y), "dual sign"
    assert all(sum(y[i] * F(A[i][j]) for i in range(m)) >= F(c[j]) for j in range(n)), "dual infeasible"
    assert sum(y[i] * F(b[i]) for i in range(m)) == val, "duality gap"
    return val, x, y


# --------------------------------------------------------------------------------------------------------------------
# DS-90 and the routes, everything per s8 MAC or per word of L MACs

RATES = {"tensor": F(4096), "issue": F(128), "alu": F(64), "imad": F(64), "shfl": F(32),
         "uniform": F(3200000000, 49027679)}
B_SLICE = 3          # bit-sliced int8 MAC on LOP3s: lane-instructions a MAC (≈ 80 bit-ops a multiply, brief 4(e))
MW_SLOTS = 2         # IMAD slots an IMAD.WIDE takes (UNLISTED on DS-90; its two 32-bit results)
SETW = 4096          # words a warp a set (wgmma m64n256's accumulators: 4096 a warp), so 128 a lane
WORLD = {"week": 1.8e29, "month": 9.4e30 / 12, "year": 9.4e30}   # FP8 FLOP, brief §0
SHA = {"mac_per_s": 9.9e14, "compressions_per_s": 4.79e9, "words_per_compression": 32}   # FOLD-LUCK §5


def add(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, F(0)) + F(v)
    return out


def scale(d, s):
    return {k: F(v) * s for k, v in d.items()}


def tc_word(L):
    return {"tensor": L, "issue": F(L, 4096)}           # IGMMA m64n256k32: 4 warp instructions a 524288 MACs


def dp_word(L):
    return {"imad": F(L, 4), "issue": F(L, 4)}


OTHER = {   # one MAC of other work, worst case for the accountant (no Strassen: brief 4(a))
    "tensor": {"tensor": 1, "issue": F(1, 4096)},
    "dp4a": {"imad": F(1, 4), "issue": F(1, 4)},
    "FFMA": {"issue": 1},
    "bit-sliced ALU": {"alu": B_SLICE, "issue": B_SLICE},
}

CP_XC = {"issue": F(4 * 32, SETW), "uniform": F(32, SETW)}   # 2 REDUX + ULOP3 + STG a warp a set
LANE_SET = F(32, SETW)                                       # one lane instruction a lane a set, a word


def folds(design):
    """Honest's fold a credited word, by the word's kind: {kind: [(route name, use)]}."""
    if design == "XC":
        tc = add({"alu": F(1, 2), "issue": F(1, 2)}, CP_XC)
        im = {"imad": 1, "issue": 1}
        return {"tensor": [("XOR3", tc), ("IMAD", add(im, CP_XC))], "dp4a": [("IMAD", im), ("XOR3", tc)]}
    if design == "li":
        li = {"alu": F(1, 2), "imad": F(1, 2), "issue": 1}
        return {"tensor": [("li", li)], "dp4a": [("li", li)]}
    if design == "RA":
        ra = add({"alu": 1, "issue": 1}, CP_XC, scale({"alu": 4, "issue": 4}, LANE_SET))   # ra_finish: 2 a chain, 2 chains
        fe = add({"imad": 2, "issue": 2}, CP_XC, scale({"imad": 2, "issue": 2}, LANE_SET))  # FE's no-word step
        return {"tensor": [("LEA.HI", ra), ("FE", fe)], "dp4a": [("LEA.HI", ra), ("FE", fe)]}
    if design == "FA":
        fin = scale({"imad": 2, "issue": 2}, LANE_SET)                    # the no-word Feistel step, 2 chains
        fa = add({"alu": 1, "imad": 1, "issue": 2}, CP_XC, fin)
        fe = add({"imad": 2, "issue": 2}, CP_XC, fin)
        return {"tensor": [("FA", fa), ("FE", fe)], "dp4a": [("FA", fa), ("FE", fe)]}
    if design.startswith("MW"):
        slots = int(design[2:]) if design[2:] else MW_SLOTS
        fin = scale({"imad": slots, "alu": 1, "issue": 2}, LANE_SET)
        mw = add({"imad": F(slots, 2), "alu": F(1, 2), "issue": 1}, CP_XC, fin)
        return {"tensor": [("IMAD.WIDE", mw)], "dp4a": [("IMAD.WIDE", mw)]}
    raise ValueError(design)


LI_FILL = {"issue": F(2, 100)}    # 0.02 FFMA a word (dither-1s-li-l32's [filler]); FILL-OPT is its premise


def per_merge(use, m):
    return {k: F(v) / m for k, v in use.items()}


LANE = lambda pipe: {pipe: 1, "issue": 1} if pipe else {"issue": 1}
WARP = lambda pipe: {pipe: 32, "issue": 32} if pipe else {"issue": 32}

CENSUS = {   # accounting-ts-fan-in, sm_90's merges a class (Assumptions/TS.lean), priced on DS-90
    "ALU (IADD3.X, 9)": per_merge(LANE("alu"), 9),
    "IMAD (7) / IMAD.WIDE (9)": per_merge(LANE("imad"), 9),
    "FFMA (5)": per_merge(LANE(None), 5),
    "NOROW P2R (10)": per_merge(LANE(None), 10),
    "shared load/store (7)": per_merge(LANE(None), 7),
    "global load (15)": per_merge(LANE(None), 15),
    "global store (13)": per_merge(LANE(None), 13),
    "shared atomic (15)": per_merge(LANE(None), 15),
    "global atomic (18)": per_merge(LANE(None), 18),
    "texture/surface (20)": per_merge(LANE(None), 20),
    "SHFL (191 a warp)": per_merge(WARP("shfl"), 191),
    "R2UR (96 a warp)": per_merge(WARP(None), 96),
    "IMMA.16832 (480 a warp)": per_merge({"tensor": 4096, "issue": 32}, 480),
}

SEM = {
    "3-input ALU form (2)": per_merge(LANE("alu"), 2),
    "3-input IMAD form (2)": per_merge(LANE("imad"), 2),
    "FFMA (2)": per_merge(LANE(None), 2),
    "REDUX.XOR (31 a warp)": per_merge(WARP(None), 31),
    "IMMA.16832 (192 a warp)": per_merge({"tensor": 4096, "issue": 32}, 192),
    "shared atomic XOR (1)": LANE(None),
    "global atomic XOR (1)": LANE(None),
    "STS.128 + bulk XOR (1)": {"issue": F(1, 4)},
}


def lp(acts, fold_routes, need_fold, floor=None):
    """acts: {name: (use, MACs)}; fold_routes: {name: use} a word folded; need_fold: {act: words a unit}, the words each
    unit of that act brings that must be folded; floor: (acts, MACs), credited MACs the solution must reach."""
    res = list(RATES)
    names = list(acts) + [f"fold: {k}" for k in fold_routes]
    cols = [acts[k][0] for k in acts] + [fold_routes[k] for k in fold_routes]
    c = [acts[k][1] for k in acts] + [0] * len(fold_routes)
    A = [[F(col.get(r, 0)) for col in cols] for r in res]
    b = [RATES[r] for r in res]
    if fold_routes:
        A.append([F(need_fold.get(nm, 0)) for nm in acts] + [F(-1)] * len(fold_routes))
        b.append(F(0))
        res = res + ["fold"]
    if floor:
        credited, macs = floor
        A.append([-F(acts[nm][1]) if nm in credited else F(0) for nm in acts] + [F(0)] * len(fold_routes))
        b.append(-F(macs))
        res = res + ["credit floor"]
    val, x, y = simplex(c, A, b)
    mix = {nm: xi for nm, xi in zip(names, x) if xi}
    duals = {r: yi for r, yi in zip(res, y) if yi}
    return val, mix, duals


def honest(design, L):
    acts = {}
    for kind, word in (("tensor", tc_word(L)), ("dp4a", dp_word(L))):
        for route, use in folds(design)[kind]:
            fill = LI_FILL if design == "li" else {}
            acts[f"{kind} word, {route}"] = (add(word, use, fill), L)
    return lp(acts, {}, {})


def cheater(design, level, L, h):
    """At least honest's credited MACs h, every credited word folded by the routes `level` allows, plus other work. li's
    filler is charged only under FOLD-OPT (with FILL-OPT): no other premise gives a filler over public inputs a floor."""
    acts = {f"other: {k}": (use, 1) for k, use in OTHER.items()}
    fill = LI_FILL if design == "li" and level == "FOLD-OPT" else {}
    acts["credited: tensor word"] = (add(tc_word(L), fill), L)
    acts["credited: dp4a word"] = (add(dp_word(L), fill), L)
    if level == "FREE":
        routes = {"free": {}}
    elif level == "CENSUS":
        routes = dict(CENSUS)
    elif level == "SEM":
        routes = dict(SEM)
    elif level == "FOLD-OPT":
        routes = {}
    else:
        raise ValueError(level)
    for kind in ("tensor", "dp4a"):
        for route, use in folds(design)[kind]:
            routes[f"honest's {route}"] = use
    need = {"credited: tensor word": 1, "credited: dp4a word": 1}
    return lp(acts, routes, need, (set(need), h))


def plain():
    """The plain kernel: other work alone, with dp4a and FFMA beside the tensor core (no bit-sliced MACs)."""
    return lp({f"other: {k}": (use, 1) for k, use in OTHER.items() if k != "bit-sliced ALU"}, {}, {})


def r_prices(design, level):
    """r_H and r_min a word in MACs of other work forgone, at the plain kernel's dual prices (MACs a unit of each
    resource): honest's cheapest fold route, and the cheapest route the premise allows."""
    _, _, duals = plain()
    price = lambda use: sum(duals.get(r, 0) * F(v) for r, v in use.items())
    rh = min(price(use) for kind in ("tensor", "dp4a") for _, use in folds(design)[kind])
    routes = {"FREE": [{}], "CENSUS": list(CENSUS.values()), "SEM": list(SEM.values()), "FOLD-OPT": []}[level]
    return rh, min([price(u) for u in routes] + [rh])


DESIGNS = [("XC", 32), ("XC", 64), ("li", 32), ("RA", 32), ("RA", 64), ("MW2", 32), ("MW1", 32), ("FA", 32),
           ("FA", 64)]
LEVELS = ("FREE", "CENSUS", "SEM", "FOLD-OPT")


def table(out: dict) -> None:
    print()
    print("The fold's term of ε on DS-90 under NO-BYPASS (MACs a SM-clock; ε = cheater/honest − 1; r in MACs a word):")
    print(f"{'design':8} {'L':>3} {'premise':9} {'honest':>10} {'cheater':>10} {'ε':>9}  {'r_H':>8} {'r_min':>8}  "
          f"binds (cheater)")
    for design, L in DESIGNS:
        hv, hmix, hd = honest(design, L)
        for level in LEVELS:
            cv, cmix, cd = cheater(design, level, L, hv)
            eps = cv / hv - 1
            rh, rm = r_prices(design, level)
            binds = ",".join(sorted(cd))
            print(f"{design:8} {L:3} {level:9} {float(hv):10.2f} {float(cv):10.2f} {float(eps):9.4%}  {float(rh):8.3f} "
                  f"{float(rm):8.3f}  {binds}")
            out.setdefault("eps", {})[f"{design}/L{L}/{level}"] = {
                "honest": str(hv), "cheater": str(cv), "eps": str(eps), "eps_float": float(eps),
                "r_H": str(rh), "r_min": str(rm), "honest_mix": {k: str(v) for k, v in hmix.items()},
                "cheater_mix": {k: str(v) for k, v in cmix.items()}, "cheater_duals": {k: str(v) for k, v in cd.items()},
                "honest_duals": {k: str(v) for k, v in hd.items()},
                "per_period_flop": {p: float(eps) * w for p, w in WORLD.items()},
            }


def gamma(out: dict) -> None:
    peak = RATES["tensor"]
    pv, _, _ = plain()
    print()
    print(f"γ on DS-90 (honest's MACs a clock against the tensor core alone, {float(peak):.0f}, and against the plain "
          f"kernel with dp4a and FFMA beside it, {float(pv):.2f}); hashing every word with SHA-512 beside it:")
    for design, L in DESIGNS:
        hv, hmix, hd = honest(design, L)
        g1, g2 = peak / hv, pv / hv
        hash_g = SHA["mac_per_s"] / (SHA["compressions_per_s"] * SHA["words_per_compression"]) / L
        print(f"  {design:4} L={L:3}: honest {float(hv):8.2f} ({', '.join(f'{k} {float(v):.2f}' for k, v in hmix.items())});"
              f" γ {float(g1):.4f} (plain {float(g2):.4f}); binds {', '.join(sorted(hd))}; SHA-512 a word {hash_g:.0f}")
        out.setdefault("gamma", {})[f"{design}/L{L}"] = {
            "honest": str(hv), "gamma_tensor": float(g1), "gamma_plain": float(g2), "honest_binds": sorted(hd),
            "gamma_sha512_word": hash_g}
    c = SHA["mac_per_s"] / (SHA["compressions_per_s"] * SHA["words_per_compression"])
    print(f"  SHA-512 a word costs {c:.0f}/L of the MAC time: γ ≤ 10 from L ≥ {c / 10:.0f} if it overlaps the tensor core "
          f"(γ = max(1, {c:.0f}/L)), from L ≥ {c / 9:.0f} if not (γ = 1 + {c:.0f}/L); at L = 8192 the term is {c / 8192:.3f}")
    out["sha512_min_L_gamma10"] = {"overlapped": c / 10, "serial": c / 9}


def barrier(out: dict, eps=F(1, 100)) -> None:
    """Theorem F (a) in numbers: a proved floor (1 − ε)·H on every program for R_F, composed with honest's fold (φ·H,
    φ = 1 − 1/γ_plain), is a floor (1 − ε − φ)·H on every program for the words, the exact segment products: SAK's
    content at c_mm = 1 − ε − φ. And the circuit size a floor on R_F implies when its checkpoints are sparse."""
    print()
    print(f"Theorem F (a) at ε = {float(eps):.0%}: a proved fold minimum proves, from θ alone, a floor on every program for "
          "the segment products (SAK's content):")
    pv, _, _ = plain()
    res = {}
    for design, L in DESIGNS:
        hv, _, _ = honest(design, L)
        phi = 1 - hv / pv
        res[f"{design}/L{L}"] = {"phi": str(phi), "c_mm_implied": str(1 - eps - phi)}
        print(f"  {design:4} L={L:3}: φ {float(phi):.4%}, so T_F(ε) would prove c_mm ≥ {float(1 - eps - phi):.4f} for "
              "the words from θ alone")
    # Sparse checkpoints, uniform in n: one LOP3 a gate with its two operands and result in HBM (12 bytes, DS-90's 4.8
    # TB/s over 132 SMs at 1830 MHz) runs any circuit, so s ≥ ((1 − ε)·H − κ_io·(m + out))/κ_gate.
    hbm = F(48 * 10**11) / (132 * 1830 * 10**6)
    k_gate, k_io = F(12) / hbm, F(4) / hbm
    rows = []
    for n in (8192, 65536, 1 << 20):
        H = F(n**3) / pv
        m = 16 * n * n
        for cp_words in (SETW, n * n):
            outbits = 32 * F(n**3, 32 * cp_words)
            s = ((1 - eps) * H - k_io * (m + outbits)) / k_gate
            rows.append({"n": n, "words_a_checkpoint": cp_words, "s_over_m_plus_out": float(s / (m + outbits))})
    for r in rows:
        print(f"  n = {r['n']:7}, a checkpoint every {r['words_a_checkpoint']:>12} words (L = 32): the circuit it implies "
              f"for R_F has s/(m + out) ≥ {r['s_over_m_plus_out']:9.3f}  (best known explicit: 3.1)")
    out["theorem_f"] = {"eps": str(eps), "designs": res, "circuit": rows, "kappa_gate_sm_clocks": float(k_gate)}
    dense = [r for r in rows if r["words_a_checkpoint"] == SETW]
    sparse = [r for r in rows if r["words_a_checkpoint"] != SETW]
    check(all(r["s_over_m_plus_out"] < 3.1 for r in dense),
          "dense checkpoints (one a set): the circuit bound a fold minimum implies is below 3.1 (m + out), so no circuit "
          "barrier there; Theorem F (a) is the kill")
    check(sparse[-1]["s_over_m_plus_out"] > 3.1,
          "sparse checkpoints (one a K-segment of the whole product) at n = 2^20: beyond every known explicit circuit lower bound")


NCP_FORMS = {   # the forms that met NCP and NCP⁺ at every context `ncp` checked, a word on DS-90's classes
    "FA (IADD3 + IMAD.HI Feistel step)": {"alu": 1, "imad": 1, "issue": 2},
    "FE (IMAD + IMAD.HI Feistel step)": {"imad": 2, "issue": 2},
}
FAILED_FORMS = {   # cheaper forms that fail NCP or NCP⁺ at small widths
    "RA (LEA.HI)": {"alu": 1, "issue": 1},
    "MW (IMAD.WIDE at 1)": {"imad": F(1, 2), "alu": F(1, 2), "issue": 1},
    "MW (IMAD.WIDE at 2)": {"imad": 1, "alu": F(1, 2), "issue": 1},
}


def _floor(forms):
    acts = {k: (add(tc_word(32), u), 1) for k, u in forms.items()}
    v, mix, _ = lp(acts, {}, {})
    per = {r: sum(F(acts[k][0].get(r, 0)) * x for k, x in mix.items()) / v for r in ("alu", "imad")}
    names = list(forms)
    cap, _, _ = simplex([F(1)] * len(names), [[F(forms[k].get(r, 0)) for k in names] for r in ("alu", "imad")],
                        [F(54, 100), F(54, 100)])
    return {"words": str(v), "gamma": float(128 / v), "mix": {k: str(x) for k, x in mix.items()},
            "alu_per_word": float(per["alu"]), "imad_per_word": float(per["imad"]), "budget_weight": str(cap)}


def ncp_floor(out: dict) -> None:
    """The most words of L = 32 a clock the NCP⁺ forms absorb, and the most weight of them FOLD-LUCK's budget of 0.54
    ALU and 0.54 IMAD a word holds (1 would fit); beside them, the cheaper forms that fail."""
    print()
    print("Folds at L = 32 on DS-90: the most words a clock (the tensor core gives 128), and the most weight of a word "
          "within 0.54 ALU + 0.54 IMAD:")
    res = {}
    for label, forms in (("NCP⁺ forms", NCP_FORMS), ("with the failing forms too", {**NCP_FORMS, **FAILED_FORMS})):
        r = res[label] = _floor(forms)
        print(f"  {label}: {float(F(r['words'])):.2f} words (γ ≥ {r['gamma']:.4f}): "
              f"{', '.join(f'{k} {float(F(x)):.2f}' for k, x in r['mix'].items())}; a word ALU {r['alu_per_word']:.3f}, "
              f"IMAD {r['imad_per_word']:.3f}; budget weight {float(F(r['budget_weight'])):.4f}")
    out["ncp_floor"] = res
    a = res["NCP⁺ forms"]
    check(F(a["words"]) <= 64 and F(a["budget_weight"]) < 1,
          "the NCP⁺ forms absorb at most 64 words a clock at L = 32 (γ ≥ 2) and no mix of them fits 0.54 ALU + 0.54 IMAD "
          "a word")


def checks_lp(out: dict) -> None:
    e = {k: F(v["eps"]) for k, v in out["eps"].items()}
    for design, L in DESIGNS:
        k = f"{design}/L{L}"
        check(e[f"{k}/FOLD-OPT"] <= e[f"{k}/SEM"] <= e[f"{k}/CENSUS"] <= e[f"{k}/FREE"],
              f"{k}: ε falls as the premise strengthens (FREE ≥ CENSUS ≥ SEM ≥ FOLD-OPT)")
    check(e["li/L32/FOLD-OPT"] == 0, "li under FOLD-OPT with FILL-OPT: ε = 0 (main's premise prices the cheater at honest)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", help="write every figure here")
    ap.add_argument("--quick", action="store_true", help="the small NCP case and fewer 32-bit trials only")
    args = ap.parse_args()
    out = {"binding": binding(args.quick)}
    tri_steps()
    out["facts_32"] = facts_32(500 if args.quick else 2000)
    conservation(trials=500 if args.quick else 3000)
    out["ncp"] = ncp_all(args.quick)
    ncp_floor(out)
    table(out)
    gamma(out)
    barrier(out)
    checks_lp(out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    print(f"\n{len(FAIL)} failed" if FAIL else "\nall checks pass")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
