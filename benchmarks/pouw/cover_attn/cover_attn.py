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


STATIC_SHAPES = {
    "Llama-3.1-8B": [(4096, 6144, 32), (4096, 4096, 32), (4096, 28672, 32), (14336, 4096, 32), (4096, 128256, 1)],
    "Llama-3.1-70B": [(8192, 10240, 80), (8192, 8192, 80), (8192, 57344, 80), (28672, 8192, 80), (8192, 128256, 1)],
    # Approximate, from config.json: 7 experts' gate/up and down, q_a, q_b, kv_a, grouped o_a and o_b, the head.
    "DeepSeek-V4-Pro": [(7168, 7 * 6144, 61), (3072, 7 * 7168, 61), (7168, 1536, 61), (1536, 65536, 61),
                        (7168, 576, 61), (4096, 16 * 1024, 61), (16384, 7168, 61), (7168, 129280, 1)],
}


def static_gamma(name, C, e, rho=0.5):
    """MAC-weighted static γ: operand rows committed (one byte an entry, c_word(1)/N) and their coin bits (BIT/N)."""
    w = s = 0.0
    for k, n, reps in STATIC_SHAPES[name]:
        g, _ = gamma_seg(k, C, e, c_word(1) / n + BIT / n, rho)
        w += g * k * n * reps
        s += k * n * reps
    return w / s


def attn_gamma(m, C, e, design, n_tokens=4500, offset=0, rho=1.0):
    """MAC-weighted γ of PV and of QKᵀ, each row priced at its own inner length.

    PV: inner = the row's attended keys; fresh bits on P (one a P entry, BIT/d_v a MAC) and on V (BIT/(g·K)).
    QKᵀ: `head` = a word per score at L = d (the whole output is one segment, so only with the skip designed away);
    `gqa` = one query head from each KV group summed along the inner dimension, L up to H_kv·d (distinct K per head,
    so nothing factors), plus one IADD3 a cell per two heads; `mqa` = all query heads summed, with fresh bits on K per
    query head (BIT/K a MAC) so the shared K doesn't factor out.
    """
    g_heads = m["hq"] // m["hkv"]
    pv_w = pv_m = qk_w = qk_m = 0.0
    for t in range(offset + 1, offset + n_tokens + 1):
        for nl, n in m["att"](t):
            mac_pv = nl * m["hq"] * m["dv"] * n
            if design == "mqa":
                gpv, _ = gamma_seg(m["hq"] * n, C, e, BIT / m["dv"] + BIT / n + ALU_MT / 2 / n, rho)
            else:
                gpv, _ = gamma_seg(n, C, e, BIT / m["dv"] + BIT / (g_heads * n), rho)
            pv_w += gpv * mac_pv
            pv_m += mac_pv
            mac_qk = nl * m["hq"] * m["dqk"] * n
            coins = BIT / n + BIT / (g_heads * n)
            if design == "head":
                gqk = math.inf if e is not None else gamma_seg(m["dqk"], C, None, coins, rho)[0]
            elif design == "gqa":
                gqk = gamma_seg(m["hkv"] * m["dqk"], C, e, coins + ALU_MT / 2 / m["dqk"], rho)[0]
            else:
                gqk = gamma_seg(m["hq"] * m["dqk"], C, e, BIT / n + BIT / n + ALU_MT / 2 / m["dqk"], rho)[0]
            qk_w += gqk * mac_qk
            qk_m += mac_qk
    return pv_w / pv_m, qk_w / qk_m


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
    print()
    print("| model | mean attended (dense) | word | skip | γ static | γ PV | γ PV batched | γ QKᵀ head | γ QKᵀ gqa | "
          "γ QKᵀ mqa |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for off in (0, 5750):
            for nb in (4, 2):
                for e in (0.01, None):
                    C = c_word(nb)
                    gs = static_gamma(m["name"], C, e)
                    pv, qh = attn_gamma(m, C, e, "head", offset=off)
                    _, qg = attn_gamma(m, C, e, "gqa", offset=off)
                    pvb, qm = attn_gamma(m, C, e, "mqa", offset=off)
                    print(f"| {m['name']} | {off + 2250} | {nb} B | {'e 1%' if e else 'fixed'} | {gs:.1f} | {pv:.1f} | "
                          f"{pvb:.1f} | {qh:.1f} | {qg:.1f} | {qm:.1f} |")
