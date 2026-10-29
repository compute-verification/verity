"""SI-NOISE: a seeded noise generator and a sketch unit inside the MMA, checked on exact small instances and priced.

The design: on every accumulator write of the credited MMA form (one full k32 step), the unit draws a fresh b-bit word
n = PRF(seed, tile, i, j, t) from the verifier's live seed and absorbs w_t = acc_t + n into a sketch register,
σ ← rotl(σ, 1) ⊕ (acc_t + n). The noise never enters the served accumulator, and no instruction reads n or σ except σ's
final value. What this script establishes (each a `Checks` line):

- **COUNTER's strategies on PRF words** (`hash_counter.py`, COUNTER's, copied unchanged from
  `cursor/compute-research-hash-counter-7b4e` at 3d0281bea7, `benchmarks/pouw/theory/`): on the lead's zero-data words
  they break the nh2, nh, xor, multiply–xor and thin folds; on per-element PRF words (written as COUNTER's coins with
  a rank-n identity column operand) every strategy but `honest` either doesn't apply or must first compute the noise,
  M = c_sw·b writes an element.
- **Rank-r decomp:** COUNTER's `decomp` generalized to words x + Σ_l R_t(i,l)·C_t(j,l) of rank r a step, exact bit for
  bit against the nh2 sketch, priced in W1; the largest r at which it still breaks T1, beside the per-step rank of the
  lead's words and of PRF words.
- **Separation:** whether σ(acc^H + n) determines σ(acc^D + n) without n (the double-duty route) for each sketch.
- **Fibres:** each sketch is injective in a skipped step's noise word, so a skipped step is guessed at 2^-b.
- **Designs:** h, the cheapest prover a cell and the gap for the lead, in-accumulator noise under four commitments, and
  the sketch unit; the egress seed rate; Δarea from `synth.json` (`synth.py`); the proof's AND gates a MAC.

Run (from this directory): `uv run --no-project --with numpy --with-editable ../exact_kit python -B silicon_noise.py
OUT.json [--synth=synth.json] [--quick] [--mutations]`.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

import hash_counter as hc
from exact_kit.checks import Checks

M32 = hc.M32
T1 = hc.T1
WB = hc.WB
UNITS_A_WRITE = 32
CARD = {"h_lead": 127.50, "h_words": 127.23, "h_rows": 0.203, "h_bound": 0.069, "useful_a_word": 27.75}
AND_PER_MAC = hc.AND_PER_MAC
SHA512_AND = 57947
TOKEN_MACS = 16 * 60.82e6
SEED_BITS = 128
CAPS = (32, 320)
LEAD_COIN_BYTES = (2.24e6, 13.2e6)
ROW = 64
KS = (2048, 8192)
SI_ALG = {"s8": 817.9, "fp8_k64": 841.7, "fp8_fma": 2285.5}
SI_ALG_ANDS = 136.8

# Software cost of a PRF on the GPU, in W1 writes an output bit (estimate: instruction counts a block / its bits).
C_SW = {
    "trivium": (10 / 32, "bitsliced, ~10 LOP3/ALU a 32-instance update (3 AND, 8 XOR fused)"),
    "chacha8": ((8 * 48 + 16) / 512, "8 rounds x 4 QRs x 12 ops + 16 adds a 512-bit block"),
    "chacha12": ((12 * 48 + 16) / 512, "12 rounds"),
    "chacha20": ((20 * 48 + 16) / 512, "20 rounds"),
    "keccak24": (4.4, "Keccak-f[1600] on 32-bit lanes, ~200 ops a round, rate 1088"),
    "keccak12": (2.2, "12 rounds"),
    "sha512": (hc.SHA_REGS / 512, "SHA_REGS a compression, CTR mode, 512 output bits"),
}


def synth_ge_per_bit(blocks: dict) -> dict:
    """Hardware GE an output bit a cycle, from the synthesized blocks: pipelined rounds times the round count."""
    b = {k: v["ge_per_unit"] for k, v in blocks.items()}
    return {"trivium": b["trivium_x64"],
            "chacha8": 8 * b["chacha_round"] + b["chacha_ff"],
            "chacha12": 12 * b["chacha_round"] + b["chacha_ff"],
            "chacha20": 20 * b["chacha_round"] + b["chacha_ff"],
            "keccak24": 24 * b["keccak_round"],
            "keccak12": 12 * b["keccak_round"],
            "sha512": 80 * b["sha512_round"]}


def prf_words(seed: int, shape: tuple, b: int) -> np.ndarray:
    """n(seed, index) for every index of `shape` in C order: b-bit words cut from SHA-512(seed ‖ block) in CTR mode."""
    count = int(np.prod(shape))
    blocks = -(-count * b // 512)
    key = seed.to_bytes(16, "little")
    stream = b"".join(hashlib.sha512(key + i.to_bytes(8, "little")).digest() for i in range(blocks))
    bits = np.unpackbits(np.frombuffer(stream, np.uint8), bitorder="little")[: count * b].reshape(count, b)
    return (bits.astype(np.int64) << np.arange(b, dtype=np.int64)).sum(axis=1).reshape(shape)


def rotl(x, k=1):
    return ((x << k) | (x >> (32 - k))) & M32


def gf_mul(a, r):
    """a·r in GF(2^32) mod x^32 + x^7 + x^3 + x^2 + 1 (`sk_gf32`'s polynomial), elementwise."""
    a = np.asarray(a, np.int64).copy()
    out = np.zeros_like(a)
    for bit in range(32):
        out ^= np.where((r >> bit) & 1, a, 0)
        a = ((a << 1) & M32) ^ np.where(a >> 31, 0x8D, 0)
    return out


def crc_step(s, w):
    """CRC-32 (reflected, 0xEDB88320) of the 32-bit word w into state s, bit by bit."""
    s = s ^ w
    for _ in range(32):
        s = np.where(s & 1, (s >> 1) ^ 0xEDB88320, s >> 1)
    return s


SKETCHES = ("xorrot", "xorrot_x", "crc32", "gf32", "lin", "nh")


def sketch(kind: str, acc, z, r):
    """σ over a (cells, T) word stream: acc the accumulator words, z the noise, r one public odd word a step. `nh` pairs
    cell 2c with 2c + 1 (σ is a pair's)."""
    cells, T = acc.shape
    s = np.zeros(cells if kind != "nh" else cells // 2, np.int64)
    for t in range(T):
        a, n = acc[:, t], z[:, t]
        w = (a + n) & M32
        if kind == "xorrot":
            s = rotl(s) ^ w
        elif kind == "xorrot_x":
            s = rotl(s) ^ a ^ n
        elif kind == "crc32":
            s = crc_step(s, w)
        elif kind == "gf32":
            s = gf_mul(s, r[t]) ^ w
        elif kind == "lin":
            s = (s + hc.mulmod(w, r[t])) & M32
        else:
            s = (s + hc.mulmod(w[0::2], w[1::2])) & M32
    return s


def run_strategies(rule, inst, d, names, stock, noise_cost=0.0):
    """COUNTER's `evaluate` loop over the named strategies; `noise_cost` writes a cell are added to every strategy but
    `honest`, which takes its noise from the MMA."""
    _, ref, _ = hc.honest(rule, inst, d)
    rows = {}
    for name in names:
        res = hc.STRATEGIES[name](rule, inst, d)
        if res is None:
            rows[name] = None
            continue
        cost, out, how, *_ = res
        cost += 0.0 if name == "honest" else noise_cost
        rows[name] = {"writes": round(cost, 3), "exact": bool(np.array_equal(out, ref)), "how": how}
    exact = {k: v for k, v in rows.items() if v and v["exact"]}
    best = min(exact, key=lambda k: exact[k]["writes"])
    return {"rule": str(rule), "stock": stock, "strategies": rows, "cheapest": best,
            "gap_share": round(max(0.0, (stock - exact[best]["writes"]) / stock), 4)}


def prf_draw(m, n, S, rule, b, walk, seed):
    """A COUNTER `Draw` whose coins are per-element PRF words: U[i, t, j] = n(i, j, t) against an identity column
    operand, q = n lanes, zero data. `walk`: the word is the accumulator with the noise in it (w = Σ_τ n_τ); else the
    sketched word acc + n with acc 0."""
    d = hc.Draw.__new__(hc.Draw)
    N = prf_words(seed, (m, n, S), b)
    inc = N if walk else np.diff(N, axis=2, prepend=0)
    d.U = inc.transpose(0, 2, 1).copy()
    d.V = np.broadcast_to(np.eye(n, dtype=np.int64)[:, None, :], (n, S, n)).copy()
    d.A, d.B = np.zeros((m, S, 0), np.int64), np.zeros((n, S, 0), np.int64)
    d.w0 = np.zeros((m, n), np.int64)
    d.rho = np.ones(S, np.int64)
    d.P = rule.positions(S)
    g = np.random.default_rng(seed)
    d.r = g.integers(0, 1 << 32, len(d.P), dtype=np.int64)
    d.sbox = g.permutation(256).astype(np.int64)
    d.c = inc & M32
    d.w = np.cumsum(inc, axis=2) & M32
    return hc.Inst(m, n, S, n, "zero", "add", seed, "prf"), d


LEAD_RULES = ("all/32/nh2:2", "all/32/nh:4", "all/32/xor", "all/32/mulxor", "all/32/thin")
PRF_RULES = LEAD_RULES + ("all/32/lin", "all/32/sq", "every:8/32/hash", "all/low:1/hash")
PRF_STRATEGIES = ("honest", "skip", "fold", "bitslice", "four_russians", "pack", "pack_det", "decomp", "edge", "expand",
                  "copy", "table")


def rule_of(s: str):
    return hc.Rule(*s.split("/"))


def counter_section(checks, quick, M):
    today = []
    for coins in ("lead:e7",) if quick else ("lead:e7", "lead:rot"):
        inst = hc.Inst(64, 48, 67, 5, "zero", "add", 0, coins)
        for rs in LEAD_RULES:
            r = hc.evaluate(rule_of(rs), inst)
            cheat = r["cheapest_writes"]
            today.append({"coins": coins, "rule": rs, "cheapest": r["cheapest"], "cheapest_writes": cheat,
                          "stock": r["stock"], "cheater_over_stock": round(cheat / r["stock"], 3),
                          "gap_hw_sketch": r["gap_share"], "gap_sw_sketch": r["gap_claimed"]})
    nh = [t for t in today if ":" in t["rule"] and t["rule"].split("/")[2].startswith("nh")]
    folds = [t for t in today if t not in nh]
    checks.check("a hardware sketch on the lead's zero-data words (honest's sketch free, credit = stock): decomp still "
                 "breaks NH (nh2, nh) by the same share of stock", all(t["gap_hw_sketch"] > 0.5 for t in nh),
                 [(t["coins"], t["rule"], t["gap_hw_sketch"]) for t in nh])
    checks.check("… while pack (xor, thin) and edge (mulxor) must still fold in software and pay ≥ stock, so those "
                 "hardware folds survive COUNTER's strategies on today's words (the margin is the fold's software cost)",
                 all(t["gap_hw_sketch"] == 0 for t in folds),
                 [(t["coins"], t["rule"], t["cheapest"], t["cheater_over_stock"], t["gap_sw_sketch"]) for t in folds])
    prf = []
    for walk in (True, False):
        for rs in PRF_RULES:
            rule = rule_of(rs)
            inst, d = prf_draw(64, 48, 67, rule, 8, walk, 11)
            res = run_strategies(rule, inst, d, PRF_STRATEGIES, stock=inst.S, noise_cost=M * inst.S)
            res["model"] = "noise in the accumulator (walk)" if walk else "noise in the sketched word"
            prf.append(res)
    applies = sorted({k for r in prf for k, v in r["strategies"].items() if v is not None})
    checks.check("on PRF words, pack, pack_det, decomp, edge, expand, copy and table don't apply (no separable form, "
                 "no repeated entry, no small table)",
                 all(r["strategies"][k] is None for r in prf for k in ("pack", "pack_det", "decomp", "edge", "expand",
                                                                        "copy", "table")),
                 f"applying: {applies}")
    checks.check("every strategy that applies on PRF words is exact", all(v["exact"] for r in prf
                                                                           for v in r["strategies"].values() if v))
    checks.check(f"on PRF words honest is cheapest for every rule once the noise is charged (M = {M:.2f} writes an "
                 "element), so no gap", all(r["cheapest"] == "honest" and r["gap_share"] == 0 for r in prf),
                 [(r["model"][:12], r["rule"], {k: v["writes"] for k, v in r["strategies"].items() if v}) for r in prf][:4])
    free = []
    for r in prf:
        for k, v in r["strategies"].items():
            if v and k != "honest":
                free.append((r["rule"], k, round(v["writes"] - M * 67, 2)))
    wide = [f for f in free if "/32/" in f[0]]
    checks.check("even with the noise free (factored, as raw coins), the non-honest strategies pay ≥ stock on PRF words "
                 "for every 32-bit field", all(w >= 67 * (1 - T1) for _, _, w in wide),
                 sorted(wide, key=lambda x: x[2])[:3])
    narrow = sorted((f for f in free if "/32/" not in f[0]), key=lambda x: x[2])
    return {"today": today, "prf": prf, "noise_free": free, "narrow_noise_free": narrow}


def digit_chains(nP, nR=2):
    pairs = [sum(1 for x in range(nR) if 0 <= w - x < 4) for w in range(4)]
    return sum(math.ceil(p * nP / 32) for p in pairs) + 3


def decomp_rank_price(r, T, p, n_call=hc.CALL_N, m_call=hc.CALL_M):
    """Writes a value of nh2:p by decomp over rank-r words, and honest's and stock's."""
    e = 2 * p
    chains = digit_chains(T * r * r)
    gemm = e * math.ceil(r * T / 32)
    alu = 6 * p + 3
    shared = r * r * T * (e / n_call + p / m_call)
    return {"decomp": chains + gemm + alu + shared, "honest": e * T * 1.5, "stock": e * T}


def decomp_rank(x, R, C, p):
    """nh2:p over w = x + Σ_l R_l C_l, taken apart: Σ_t w_a w_b = T·x_a x_b + x_a·D(b) + x_b·D(a) + Σ_t Σ_{l,l'} R_l R_l'
    C_a,l C_b,l' with D = Σ_{t,l} R C (a u8 IMMA GEMM) and the cross term as u8-digit IMMA chains (`digit_dot`)."""
    m, T, r = R.shape
    n = C.shape[0]
    G = n // (2 * p)
    Q = np.einsum("itl,itk->itlk", R, R).reshape(m, T * r * r)
    Ca, Cb = C[0::2], C[1::2]
    Chat = np.einsum("ptl,ptk->ptlk", Ca, Cb).reshape(G, p, T * r * r).sum(axis=1)
    cross, chains = hc.digit_dot(Q, Chat[None], np.zeros(m, np.int64), 2)
    D = np.einsum("itl,jtl->ij", R, C) & M32
    xa, xb = x[:, 0::2], x[:, 1::2]
    Da, Db = D[:, 0::2], D[:, 1::2]
    per = (hc.mulmod(hc.mulmod(xa, xb), T) + hc.mulmod(xa, Db) + hc.mulmod(xb, Da)) & M32
    return (per.reshape(m, G, p).sum(axis=2) + cross) & M32, chains


def rank_section(checks, quick):
    g = np.random.default_rng(5)
    m, n, T = 64, 64, 67
    exact, prices = [], []
    for r in (1, 2, 3, 4) if quick else (1, 2, 3, 4, 6, 8):
        R = g.integers(0, 256, (m, T, r), dtype=np.int64)
        C = g.integers(0, 256, (n, T, r), dtype=np.int64)
        x = g.integers(0, 1 << 32, (m, n), dtype=np.int64)
        w = (x[:, :, None] + np.einsum("itl,jtl->ijt", R, C)) & M32
        for p in (2, 8, 32):
            ref = hc.nh_sketch(hc.Rule("all", "32", f"nh2:{p}"), w)
            out, chains = decomp_rank(x, R, C, p)
            exact.append((r, p, bool(np.array_equal(out, ref)), chains == digit_chains(T * r * r)))
    checks.check("rank-r decomp equals the nh2:p sketch bit for bit, and its chain count is the priced one",
                 all(e and c for _, _, e, c in exact), [(r, p) for r, p, e, c in exact if not (e and c)] or len(exact))
    cross = {}
    for p in (2, 8, 32):
        row = []
        for r in range(1, 65):
            pr = decomp_rank_price(r, T, p)
            row.append({"r": r, **{k: round(v, 1) for k, v in pr.items()},
                        "gap_share": round(max(0.0, 1 - pr["decomp"] / pr["stock"]), 4)})
        breaking = [e["r"] for e in row if e["gap_share"] > T1]
        cross[p] = {"largest_breaking_r": max(breaking) if breaking else 0, "rows": row[:8] + [row[47], row[63]]}
        prices.append((p, cross[p]["largest_breaking_r"], row[63]["decomp"], row[63]["stock"]))
    checks.check("rank-r decomp breaks T1 only at small r (r ≤ 16 for values up to a 64-cell row), and at full rank "
                 "(64) costs many times stock", all(c["largest_breaking_r"] <= 16 for c in cross.values())
                 and all(d > 10 * s for _, _, d, s in prices), prices)
    ranks = {}
    for coins in ("lead:e7", "lead:rot"):
        inst = hc.Inst(64, 48, 67, 5, "zero", "add", 0, coins)
        d = hc.Draw(inst, hc.Rule("all", "32", "hash"))
        c = np.einsum("itl,jtl->ijt", d.U, d.V)
        W = np.cumsum(c, axis=2)
        cr = max(np.linalg.matrix_rank(c[:, :, t].astype(float)) for t in range(0, 67, 11))
        wr = max(np.linalg.matrix_rank(W[:, :, t].astype(float)) for t in range(0, 67, 11))
        ranks[coins] = {"step": int(cr), "word_minus_offset": int(wr)}
    N = prf_words(3, (64, 48, 67), 8)
    ranks["prf b8"] = {"step": int(max(np.linalg.matrix_rank(N[:, :, t].astype(float)) for t in range(0, 67, 11))),
                       "word": int(max(np.linalg.matrix_rank(np.cumsum(N, axis=2)[:, :, t].astype(float))
                                       for t in range(0, 67, 11)))}
    checks.check("the lead's per-step words have rank ≤ 3 (beside the offset); PRF words have full rank 48",
                 all(v["word_minus_offset"] <= 3 for k, v in ranks.items() if k.startswith("lead"))
                 and ranks["prf b8"]["step"] == 48 and ranks["prf b8"]["word"] == 48, ranks)
    return {"exact": exact, "crossover": cross, "ranks": ranks}


def distinct_diffs(kind, hidden, b=8, draws=64, seed=9):
    """Over `draws` noise draws, the distinct differences between σ(acc^H + n) and σ(acc^D + n) at zero declared data
    (the fewer of XOR and subtraction). One means σ^D follows from σ^H and the accumulators alone."""
    g = np.random.default_rng(seed)
    T = hidden.shape[1]
    D = np.zeros_like(hidden)
    r = g.integers(0, 1 << 31, T, dtype=np.int64) * 2 + 1
    xs, ss = set(), set()
    for _ in range(draws):
        z = g.integers(0, 1 << b, hidden.shape, dtype=np.int64)
        sh, sd = sketch(kind, hidden, z, r), sketch(kind, D, z, r)
        xs.add(tuple(sh ^ sd))
        ss.add(tuple((sh - sd) & M32))
    return min(len(xs), len(ss))


def separation_section(checks, b=8, sketches=SKETCHES):
    H = np.random.default_rng(10).integers(0, 1 << 32, (16, 24), dtype=np.int64)
    H_low0 = H & ~((1 << b) - 1) & M32
    out = {}
    for kind in sketches:
        out[f"{kind} / random hidden words"] = distinct_diffs(kind, H, b)
        out[f"{kind} / hidden words ≡ 0 mod 2^{b}"] = distinct_diffs(kind, H_low0, b)
    linear = ("xorrot_x", "lin")
    checks.check("σ(acc^H + n) gives σ(acc^D + n) without n for the linear sketches (xorrot_x, lin), on any words",
                 all(out[f"{k} / random hidden words"] == 1 for k in linear), {k: out[f"{k} / random hidden words"]
                                                                              for k in linear})
    checks.check("it doesn't for the adder sketches on random hidden words (xorrot, crc32, gf32, nh)",
                 all(out[f"{k} / random hidden words"] > 1 for k in sketches if k not in linear),
                 {k: out[f"{k} / random hidden words"] for k in sketches if k not in linear})
    checks.check(f"but on hidden words ≡ 0 mod 2^{b} (no carry out of the noise bits) xorrot, crc32 and gf32 separate "
                 "too, and nh doesn't", all(out[f"{k} / hidden words ≡ 0 mod 2^{b}"] == 1 for k in ("xorrot", "crc32",
                                                                                                     "gf32"))
                 and out[f"nh / hidden words ≡ 0 mod 2^{b}"] > 1,
                 {k: out[f"{k} / hidden words ≡ 0 mod 2^{b}"] for k in sketches})
    return out


def fibre_section(checks, b=8, contexts=16, sketches=SKETCHES):
    """Distinct final σ as one step's noise word runs over its 2^b values, the other words fixed: 2^b means a skipped
    step is guessed with probability 2^-b. `nh`'s σ is a pair's, so the varied word is the pair's first."""
    g = np.random.default_rng(13)
    T, V = 24, 1 << b
    out = {}
    for kind in sketches:
        worst = V
        for _ in range(contexts):
            acc = g.integers(0, 1 << 32, (2, T), dtype=np.int64)
            z = g.integers(0, V, (2, T), dtype=np.int64)
            r = g.integers(0, 1 << 31, T, dtype=np.int64) * 2 + 1
            for t in (0, T // 2, T - 1):
                accs, zs = np.tile(acc, (V, 1)), np.tile(z, (V, 1))
                zs[0::2, t] = np.arange(V)
                s = sketch(kind, accs, zs, r)
                worst = min(worst, len(set((s if kind == "nh" else s[0::2]).tolist())))
        out[kind] = worst
    checks.check(f"every sketch is injective in a skipped step's {b}-bit noise word (2^{b} distinct σ, any step)",
                 all(v == 1 << b for v in out.values()), out)
    return out


def designs(h_rest=None):
    """h (W1 units a stock MAC), the cheapest prover a cell (writes) and the gap for each design at K, all lanes data
    (the noise comes from the unit). T = K/32 writes a cell; `seal16` is 16 bits a tile row of 64 cells."""
    rest = CARD["h_rows"] + CARD["h_bound"] if h_rest is None else h_rest
    out = []
    for K in KS:
        T = K // 32
        seal16 = 16 * WB * UNITS_A_WRITE / (ROW * K)
        out += [
            {"K": K, "design": "lead (card)", "h": CARD["h_lead"], "cheapest": "stock (ShaWrites)", "gap": "≤ 1%",
             "premise": "ShaWrites, 4a, to 1%/127.5"},
            {"K": K, "design": "noise in acc + SHA-512 every word", "h": round(32 * WB + rest, 3),
             "cheapest": f"{T} (each word needs its noise)", "gap": 0.0, "premise": "PRF margin 4a + ShaWrites 4a"},
            {"K": K, "design": "noise in acc + software nh2:8", "h": round(0.5 + 32 * WB * UNITS_A_WRITE / (16 * K)
                                                                            + rest, 3),
             "cheapest": f"{T} + 0.5·{T} (honest; decomp needs rank ≤ 16)", "gap": 0.0,
             "premise": "PRF margin 4a (the fold's optimality moves only h)"},
            {"K": K, "design": "noise in acc + software thin fold", "h": round(0.5 + 64 * WB * UNITS_A_WRITE / K + rest, 3),
             "cheapest": f"{T} + 0.5·{T} (pack needs separable words)", "gap": 0.0,
             "premise": "PRF margin 4a (the fold's optimality moves only h)"},
            {"K": K, "design": "noise in acc, exact denoise (nacc readable), outputs only",
             "h": round(32 * WB * UNITS_A_WRITE / K + rest, 3),
             "cheapest": "2 a declared zero-data cell beside hidden work (read nacc, commit)",
             "gap": round(1 - 2 / T, 4), "premise": "F (double duty)"},
            {"K": K, "design": "sketch unit, noise only in the sketched word, σ_row 16 bits",
             "h": round(seal16 + rest, 4), "cheapest": f"{T} (a zero-data run) or {T}·M (software PRF)", "gap": 0.0,
             "premise": "P1-P2 vendor circuit (3), PRF margin 4a to a factor M, seal ShaWrites 4a to 1%/h"},
        ]
    return out


def egress(h):
    """N-EGRESS's obstruction, coin bits a token ≥ 100·W_mac·cap, at W_mac = 1 + h, as 128-bit seeds delivered raw, beside
    the lead's stream (N-EGRESS: 2.24 MB a token at 64 rows, 13.2 MB at decode with one row)."""
    out = []
    for cap in CAPS:
        bits = 100 * (1 + h) * cap
        seeds = math.ceil(bits / SEED_BITS)
        out.append({"cap": cap, "bits_a_token": round(bits), "seeds_a_token": seeds, "bytes": seeds * SEED_BITS // 8,
                    "macs_a_seed": round(TOKEN_MACS / seeds), "vs_lead_64rows": round(LEAD_COIN_BYTES[0] / (seeds * 16)),
                    "vs_lead_decode": round(LEAD_COIN_BYTES[1] / (seeds * 16))})
    return out


def area(blocks: dict):
    """Δarea a MAC: the PRF's b/32 bits a MAC and the sketch's logic once a 32-MAC write, beside this lane's s8 and FP8
    dp4 MACs and SI-ALG's baseline (`SI_ALG`: its s8 4-wide unit, its 4-wide FP8 into a Kulisch k64 accumulator and its
    1-wide FP8 FMA, GE a MAC, report-silicon-baseline)."""
    ge_bit = synth_ge_per_bit(blocks)
    s8, fp8 = blocks["mac_s8_dp4"]["ge_per_unit"], blocks["mac_fp8_dp4"]["ge_per_unit"]
    sk = {k: blocks[f"sk_{k}"]["ge_comb"] / (64 if k == "nh" else 32) for k in SKETCHES}
    prf = []
    for name, ge in ge_bit.items():
        c, _ = C_SW[name]
        for b in (1, 2, 4, 8, 16):
            M = c * b
            if M < 1 or M > 64:
                continue
            ge_mac = ge * b / 32
            tot = ge_mac + sk["xorrot"]
            prf.append({"prf": name, "b": b, "M": round(M, 2), "ge_a_mac": round(ge_mac, 2),
                        "with_sketch": round(tot, 2), "pct_s8": round(100 * tot / s8, 2),
                        "pct_fp8": round(100 * tot / fp8, 2),
                        "pct_si_alg": {k: round(100 * tot / v, 2) for k, v in SI_ALG.items()}})
    merit = {k: round(ge_bit[k] / C_SW[k][0], 1) for k in ge_bit}
    return {"mac_ge": {"s8": s8, "fp8": fp8}, "prf_ge_a_bit": {k: round(v, 2) for k, v in ge_bit.items()},
            "c_sw": {k: round(v[0], 3) for k, v in C_SW.items()}, "ge_a_bit_per_write_a_bit": merit,
            "sketch_ge_a_mac": {k: round(v, 2) for k, v in sk.items()},
            "sketch_pct": {k: (round(100 * v / s8, 2), round(100 * v / fp8, 2)) for k, v in sk.items()},
            "sketch_pct_si_alg": {k: {b: round(100 * v / g, 2) for b, g in SI_ALG.items()} for k, v in sk.items()},
            "prf": prf}


def proof_cost():
    """AND gates a MAC the proof adds: the sketch with a public noise word (a 32-bit add of a public constant is ≤ 31
    ANDs, XOR free) once a 32-MAC write; the lead's SHA-512 over 32 bits a word of 27.75 useful MACs."""
    sketch = 31 / 32
    lead = SHA512_AND * 32 / 1024 / CARD["useful_a_word"]
    seal = {K: SHA512_AND / (ROW * 64 * K) for K in KS}
    return {"basis": AND_PER_MAC, "si_alg_s8": SI_ALG_ANDS, "sketch": round(sketch, 3),
            "sketch_pct": round(100 * sketch / AND_PER_MAC, 2), "sketch_pct_si_alg": round(100 * sketch / SI_ALG_ANDS, 2),
            "lead_sha": round(lead, 1), "lead_pct": round(100 * lead / AND_PER_MAC, 1),
            "seal": {K: round(v, 5) for K, v in seal.items()}}


def mutations(checks, quick):
    with checks.mutant("PRF noise replaced by the lead's low-rank coins (decomp applies again)"):
        inst = hc.Inst(64, 48, 67, 5, "zero", "add", 0, "lead:rot")
        rule = rule_of("all/32/nh2:2")
        d = hc.Draw(inst, rule)
        checks.check("decomp doesn't apply", hc.decomp(rule, inst, d) is None)
    with checks.mutant("rank-r decomp with one cell's offset off by one"):
        g = np.random.default_rng(1)
        R, C = g.integers(0, 256, (64, 67, 2)), g.integers(0, 256, (64, 67, 2))
        x = g.integers(0, 1 << 32, (64, 64))
        w = (x[:, :, None] + np.einsum("itl,jtl->ijt", R, C)) & M32
        x2 = x.copy()
        x2[3, 5] += 1
        checks.check("exact", np.array_equal(decomp_rank(x2, R, C, 8)[0], hc.nh_sketch(rule_of("all/32/nh2:8"), w)))
    with checks.mutant("rank-r decomp dropping a rank lane"):
        checks.check("exact", np.array_equal(decomp_rank(x, R[:, :, :1], C[:, :, :1], 8)[0],
                                             hc.nh_sketch(rule_of("all/32/nh2:8"), w)))
    with checks.mutant("xorrot with its adder replaced by XOR (separates on random words)"):
        H = np.random.default_rng(10).integers(0, 1 << 32, (16, 24), dtype=np.int64)
        checks.check("the sketch doesn't separate", distinct_diffs("xorrot_x", H) > 1)
    with checks.mutant("a sketch that drops the noise's top bit (fibres halve)"):
        g = np.random.default_rng(2)
        acc = g.integers(0, 1 << 32, (256, 8))
        z = np.zeros((256, 8), np.int64)
        z[:, 3] = np.arange(256)
        s = sketch("xorrot", acc[:1].repeat(256, 0), z & 0x7F, np.ones(8, np.int64))
        checks.check("injective", len(set(s.tolist())) == 256)


def main() -> int:
    checks = Checks()
    args = [a for a in checks.args if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in checks.args if a.startswith("--") and "=" in a)
    quick = "--quick" in checks.args
    out_path = Path(args[0] if args else "silicon_noise.json")
    synth_path = Path(opts.get("synth", "synth.json"))
    if not synth_path.is_file():
        raise SystemExit(f"no {synth_path}: run `python3 synth.py {synth_path}` first (yosys on PATH), or pass "
                         "--synth=PATH")
    synth = json.loads(synth_path.read_text())
    blocks = synth["blocks"]
    M_triv = C_SW["trivium"][0] * 8
    res = {"schema": "silicon-noise/v1", "yosys": synth.get("yosys"), "library": synth.get("library")}
    res["counter"] = counter_section(checks, quick, M_triv)
    res["rank"] = rank_section(checks, quick)
    res["separation"] = separation_section(checks)
    res["fibres"] = fibre_section(checks, contexts=4 if quick else 16)
    res["designs"] = designs()
    h_unit = [d for d in res["designs"] if d["design"].startswith("sketch unit")]
    checks.check("the sketch unit's h is under 0.3 at K 2,048 and 8,192 (no per-word hashing), ≥ 400× under the lead",
                 all(d["h"] < 0.3 and CARD["h_lead"] / d["h"] >= 400 for d in h_unit),
                 [(d["K"], d["h"], round(CARD["h_lead"] / d["h"])) for d in h_unit])
    res["egress"] = egress(h_unit[0]["h"])
    res["area"] = area(blocks)
    res["proof"] = proof_cost()
    best = min((p for p in res["area"]["prf"] if p["M"] >= 2), key=lambda p: p["with_sketch"])
    checks.check("a PRF at margin M ≥ 2 with the xorrot sketch costs under 3% of an s8 MAC and 2% of an FP8 MAC",
                 best["pct_s8"] < 3 and best["pct_fp8"] < 2, best)
    if checks.mutations:
        mutations(checks, quick)
    out_path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    for d in res["designs"]:
        print(f"K {d['K']:5d}  h {d['h']:>9}  gap {d['gap']!s:>7}  {d['design']}  [{d['premise']}]")
    for e in res["egress"]:
        print(f"cap {e['cap']:3d}: {e['bits_a_token']} bits, {e['seeds_a_token']} seeds ({e['bytes']} B) a token, "
              f"one per {e['macs_a_seed']:.3g} MACs; {e['vs_lead_64rows']}x / {e['vs_lead_decode']}x under the lead")
    for p in res["area"]["prf"]:
        print(f"{p['prf']:9s} b {p['b']:2d} M {p['M']:6.2f}  {p['ge_a_mac']:7.2f} GE/MAC  +sketch {p['with_sketch']:7.2f}"
              f"  {p['pct_s8']:5.2f}% s8  {p['pct_fp8']:5.2f}% fp8  SI-ALG {p['pct_si_alg']}")
    print("sketch GE a MAC:", res["area"]["sketch_ge_a_mac"], "merit:", res["area"]["ge_a_bit_per_write_a_bit"])
    print("proof:", res["proof"])
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
