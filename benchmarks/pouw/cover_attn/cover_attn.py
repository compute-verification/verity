"""COVER-ATTN: coverage of all matmul MACs and γ when attention (QKᵀ, PV) is credited, at the world model's mix.

CPU, under a second: `python benchmarks/pouw/cover_attn/cover_attn.py`. Conventions are PROTO-DESIGN's no-fold receipt
(`benchmarks/pouw/accel_score/nofold.py` on cursor/compute-research-accel-proto-design-ea24 at d205c68f8f): H200 INT8
dense 9.895e14 MAC/s, SHA-512 4.79e9 compressions/s, plain at η = 0.85, γ in plain MAC-times a MAC, a w-byte word
packed 128/w to a leaf compression, one tree node a leaf (binary tree, ×2) or ×1.5 (16-ary).

The request is the world model's: 4,500 tokens processed causally from position 1 (2 FLOP an active parameter a token,
DeepSeek-V4-Pro, 49e9 active), so a dense model's mean attended context is about N/2.
"""
from __future__ import annotations

import math

RHO8, SHA, ETA = 9.895e14, 4.79e9, 0.85
MT_COMP = RHO8 / SHA
BIT = MT_COMP * ETA / 512
ALU_MT = 4096 / 64


def c_word(nbytes: float, tree: float = 2.0) -> float:
    return MT_COMP * nbytes / 128 * tree * ETA


def dense(name, layers, hidden, hq, hkv, d, ffn, vocab):
    static = layers * hidden * (hq * d + 2 * hkv * d + hq * d + 2 * ffn) + layers * ffn * hidden + hidden * vocab
    return {"name": name, "static": static, "layers": layers, "hq": hq, "hkv": hkv, "dqk": d, "dv": d,
            "att": lambda t: [(layers, t)], "indexer": lambda t: 0}


def dsv4():
    def att(t):
        w = min(t, 128)
        return [(30, w + min(1024, t // 4)), (31, w + t // 128)]

    return {"name": "DeepSeek-V4-Pro", "static": 49e9, "layers": 61, "hq": 128, "hkv": 1, "dqk": 512, "dv": 512,
            "att": att, "indexer": lambda t: 30 * 64 * 128 * (t // 4)}


MODELS = [dense("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256),
          dense("Llama-3.1-70B", 80, 8192, 64, 8, 128, 28672, 128256), dsv4()]


def macs(m, n_tokens=4500, offset=0):
    """Per-request MACs: static, QKᵀ, PV, indexer, and the MAC-weighted attended context of the QKᵀ/PV rows."""
    st = m["static"] * n_tokens
    qk = pv = ix = 0.0
    for t in range(offset + 1, offset + n_tokens + 1):
        for nl, n in m["att"](t):
            qk += nl * m["hq"] * m["dqk"] * n
            pv += nl * m["hq"] * m["dv"] * n
        ix += m["indexer"](t)
    return st, qk, pv, ix


def gamma_seg(K, C, e=None, base=0.0, rho=0.5):
    """γ of a matmul class with inner length K a word chain: words every L, L = eγK under the serving skip at budget e,
    or L = K with the skip designed away (e None). base: per-MAC extras besides the MMA (coins, operand commits)."""
    if e is None:
        return 1 + base + C / K + rho, K
    g = 10.0
    for _ in range(200):
        L = min(K, e * g * K)
        g = 1 + base + C / L + rho * L / K
    return g, min(K, e * g * K)


def rows():
    out = []
    for m in MODELS:
        for off in (0, 1750, 5750, 29750):
            st, qk, pv, ix = macs(m, offset=off)
            tot = st + qk + pv + ix
            out.append((m["name"], off, st / tot, qk / tot, pv / tot, ix / tot))
    return out


if __name__ == "__main__":
    print("| model | mean attended (dense) | static | QKᵀ | PV | indexer | cov static | cov +PV | cov +PV+QKᵀ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, off, s, q, p, i in rows():
        print(f"| {name} | {off + 2250} | {s:.2%} | {q:.2%} | {p:.2%} | {i:.2%} | {s:.1%} | {s + p:.1%} | {s + p + q:.1%} |")
    for nb in (4, 2):
        for tree in (2.0, 1.5):
            print(f"C(word {nb} B, tree x{tree}) = {c_word(nb, tree):,.0f}")
    print(f"coin bit = {BIT:.0f} plain MAC-times; ALU op = {ALU_MT:.0f} MAC-times")
