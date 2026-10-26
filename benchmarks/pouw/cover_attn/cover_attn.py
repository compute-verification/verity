"""COVER-ATTN: coverage of all matmul MACs and γ when attention (QKᵀ, PV) is credited, at the world model's mix.

CPU, under a minute: `python benchmarks/pouw/cover_attn/cover_attn.py [e]` (e = the static serving-skip budget, v1's
0.0075 by default). Conventions are protocol v1's (`benchmarks/pouw/accel_score/nofold.py` `v1()` at 1bfece4240 on
cursor/compute-research-accel-proto-design-ea24): data-sheet INT8 rate, measured SHA-512 rate, plain at η = 0.85, γ in
plain MAC-times a MAC, 4-byte words 32 to a leaf compression, a binary tree (one node a leaf, ×2; 16-ary ×1.5), HYB's
wraps on the ALU pipe, static peel of five products of inner L under the skip.

ATT's attention rounds are skip-free by construction: the served attention output comes from a plain second stream and
is never a word of the round (RT-PROTO P1), so a PV cell's words run to the end of its attended keys, capped by the
proof unit's size, and QKᵀ's one word a score is a drawable leaf.

The request is the world model's: 4,500 tokens processed causally from position 1 (DeepSeek-V4-Pro, 49e9 active).
"""
from __future__ import annotations

import math
import sys

E = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0075
ETA, HYB_P, HYB_RHO = 0.85, 0.0126, 5
AND_PER_MAC, UNIT_ANDS = 370, 1.34e8

CARDS = {
    # INT8 dense MAC/s, SHA-512 compressions/s, tensor MACs an ALU lane-op, HBM B/s, tensor MACs an SM-clock
    "H200": (1978.9e12 / 2, 4.79e9, 64, 4.8e12, 4096),
    "RTX PRO 6000": (1.0e15 / 2, 7.13e9, 8, 1.792e12, 1024),
}


class Card:
    def __init__(self, name, nbytes=4, tree=2.0):
        self.name = name
        self.mac, self.sha, self.c_op, self.hbm, self.tc_sm = CARDS[name]
        self.mt_comp = self.mac / self.sha
        self.C = tree * ETA * self.mt_comp * nbytes / 128
        self.byte = tree * ETA * self.mt_comp / 128
        self.bit = ETA * self.mt_comp / 512
        self.wrap = self.c_op * HYB_P / 2


def dense(name, layers, hidden, hq, hkv, d, ffn, vocab):
    static = layers * hidden * (hq * d + 2 * hkv * d + hq * d + 2 * ffn) + layers * ffn * hidden + hidden * vocab
    return {"name": name, "static": static, "hq": hq, "hkv": hkv, "dqk": d, "dv": d, "kv_bytes": d,
            "att": lambda t: [(layers, t)], "indexer": lambda t: 0}


def dsv4():
    def att(t):
        w = min(t, 128)
        return [(30, w + min(1024, t // 4)), (31, w + t // 128)]

    return {"name": "DeepSeek-V4-Pro", "static": 49e9, "hq": 128, "hkv": 1, "dqk": 512, "dv": 512, "kv_bytes": 584,
            "att": att, "indexer": lambda t: 30 * 64 * 128 * (t // 4)}


MODELS = [dense("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256),
          dense("Llama-3.1-70B", 80, 8192, 64, 8, 128, 28672, 128256), dsv4()]

STATIC_SHAPES = {
    "Llama-3.1-8B": [(4096, 6144, 32), (4096, 4096, 32), (4096, 28672, 32), (14336, 4096, 32), (4096, 128256, 1)],
    "Llama-3.1-70B": [(8192, 10240, 80), (8192, 8192, 80), (8192, 57344, 80), (28672, 8192, 80), (8192, 128256, 1)],
    # Approximate, from config.json: 7 experts' gate/up and down, q_a, q_b, kv_a, grouped o_a and o_b, the head.
    "DeepSeek-V4-Pro": [(7168, 7 * 6144, 61), (3072, 7 * 7168, 61), (7168, 1536, 61), (1536, 65536, 61),
                        (7168, 576, 61), (4096, 16 * 1024, 61), (16384, 7168, 61), (7168, 129280, 1)],
}


def rows_of(m, n_tokens=4500, offset=0):
    """{attended n: number of (token, layer) rows}, plus the indexer's MACs."""
    hist, ix = {}, 0.0
    for t in range(offset + 1, offset + n_tokens + 1):
        for nl, n in m["att"](t):
            hist[n] = hist.get(n, 0) + nl
        ix += m["indexer"](t)
    return hist, ix


def macs(m, offset=0):
    hist, ix = rows_of(m, offset=offset)
    qk = sum(c * m["hq"] * m["dqk"] * n for n, c in hist.items())
    pv = sum(c * m["hq"] * m["dv"] * n for n, c in hist.items())
    return m["static"] * 4500, qk, pv, ix


def static_gamma(m, card, e):
    """v1's static γ per served shape: a = 1 + operand commit + wraps; under the skip L = eγK and a peel of five products
    of inner L; with the skip designed away (e None) L = K and the served product from a second stream."""
    w = s = 0.0
    for k, n, reps in STATIC_SHAPES[m["name"]]:
        a = 1 + card.byte / n + card.wrap
        if e is None:
            g = a + card.C / k + 1
        else:
            g = 10.0
            for _ in range(100):
                L = min(k, e * g * k)
                g = a + card.C / L + HYB_RHO * L / k
        w += g * k * n * reps
        s += k * n * reps
    return w / s


def pv_gamma(m, card, n, h_p, batched=False):
    """PV row of n attended keys: h_p fresh bits a P entry (343/d_v a bit on the H200), full-width noise on V (8 bits an
    entry, reused by the g query heads and every later query), words to the end of the row capped by the unit, served
    from a second stream. Batched: heads summed along the inner dimension over V's per-head noise (prefill only)."""
    g_heads = m["hq"] // m["hkv"]
    coins = h_p * card.bit / m["dv"] + 8 * card.bit / (g_heads * n)
    L_unit = UNIT_ANDS / (AND_PER_MAC * (m["dqk"] + 1))
    if batched:
        coins = h_p * card.bit / m["dv"] + 8 * card.bit / n
        L = min(m["hq"] * n, L_unit)
        return 1 + card.wrap + coins + card.C / L + card.c_op / 2 / n + 1
    return 1 + card.wrap + coins + card.C / min(n, L_unit) + 1


def qk_gamma(m, card, n, form):
    """QKᵀ row: full-rank wide noise on Q and K (each entry reused by n keys or n queries), one word a score (head), one
    query head from each KV group summed along the inner dimension (gqa), or all heads over per-head noise on the shared
    K (mqa, prefill only)."""
    g_heads = m["hq"] // m["hkv"]
    coins = 8 * card.bit / n + 8 * card.bit / (g_heads * n)
    if form == "head":
        return 1 + card.wrap + coins + card.C / m["dqk"] + 1
    if form == "gqa":
        return 1 + card.wrap + coins + card.C / (m["hkv"] * m["dqk"]) + card.c_op / 2 / m["dqk"] + 1
    coins = 8 * card.bit / n + 8 * card.bit / n
    return 1 + card.wrap + coins + card.C / (m["hq"] * m["dqk"]) + card.c_op / 2 / m["dqk"] + 1


def decode_gamma(m, card, g_cb, noise_stream=2.0):
    """Roofline at decode for a per-head form: plain = max(compute, KV stream); protocol = max(compute γ_cb, KV stream with
    its cached full-width noise, noise_stream ×)."""
    g_heads = m["hq"] // m["hkv"]
    intensity = g_heads * m["dv"] * 2 * 1 / (m["kv_bytes"] * (2 if m["hkv"] > 1 else 1))
    t_mem = card.mac / (intensity * card.hbm)
    plain = max(1 / ETA, t_mem)
    return max(g_cb / ETA, noise_stream * t_mem) / plain


def weighted(m, card, fn, offset=0):
    hist, _ = rows_of(m, offset=offset)
    w = sum(c * n * fn(n) for n, c in hist.items())
    return w / sum(c * n for n, c in hist.items())


DESIGNS = [  # name, static skip (E or None), PV form, QKᵀ form
    ("static only", "e", None, None),
    ("ATT-PV", "e", "head", None),
    ("ATT-PV + QKᵀ per head", "e", "head", "head"),
    ("ATT-PV + QKᵀ batched", "e", "head", "batched"),
    ("skip fixed: ATT-PV + QKᵀ per head", None, "head", "head"),
    ("skip fixed: ATT-PV + QKᵀ batched", None, "head", "batched"),
]


def totals(m, card, h_p, offset=0):
    st, qk, pv, ix = macs(m, offset)
    tot = st + qk + pv + ix
    out = []
    for name, skip, pvf, qkf in DESIGNS:
        e = E if skip == "e" else None
        gs = static_gamma(m, card, e)
        gp = weighted(m, card, lambda n: pv_gamma(m, card, n, h_p)) if pvf else 1.0
        form = ("gqa" if m["hkv"] > 1 else "mqa") if qkf == "batched" else qkf
        gq = weighted(m, card, lambda n: qk_gamma(m, card, n, form)) if qkf else 1.0
        g = (st * gs + pv * gp + qk * gq + ix) / tot
        cov = (st + (pv if pvf else 0) + (qk if qkf else 0)) / tot
        out.append((name, g, cov, (pv * (gp - 1) + qk * (gq - 1)) / tot, gp, gq))
    return out


def lookup_gain(m, card, n, h_p):
    """First estimate of O-IO's lookup route on PV at the prover's own workload (P = 0, V = 0): P̃ is h_p fresh bits, so a
    word is Σ b·Ṽ, computed by Four-Russians lookups of g-bit chunks of b (h_p/g lookups a MAC) into tables of Ṽ's chunks
    shared by the n queries after them, on rows a full-rate GEMM leaves idle, while the tensor does other work.
    Returns y, the credited MACs gained per tensor MAC of capacity (an upper bound on ε against an honest prover with no
    lookup route on real P), maximized over g; the binding rows are ALU, issue and the shared-memory port (128 B an
    SM-clock, of which the GEMM uses tc_sm/64)."""
    hash_mt = card.C / ETA / min(n, UNIT_ANDS / (AND_PER_MAC * (m["dqk"] + 1)))
    coin_mt = h_p * card.bit / ETA / m["dv"]
    alu_lanes = card.tc_sm / card.c_op
    smem_free = 128 - card.tc_sm / 64
    best = 0.0
    for g in range(2, 14):
        look = h_p / g
        build = (2 ** g / max(n / 2, 1)) * look
        alu = (look + build) * card.tc_sm / alu_lanes + hash_mt + coin_mt
        issue = (2 * look + build) * card.tc_sm / 128 + hash_mt + coin_mt
        smem = 4 * look * card.tc_sm / smem_free
        best = max(best, min(1 / alu, 1 / issue, 1 / smem))
    return best


if __name__ == "__main__":
    print(f"static skip budget e = {E:.2%}")
    for name in CARDS:
        for nb, tree in ((4, 2.0), (4, 1.5), (2, 2.0)):
            c = Card(name, nb, tree)
            print(f"{name}: C(word {nb} B, tree ×{tree}) = {c.C:,.0f}; coin bit {c.bit:.0f}; wraps {c.wrap:.2f}")
    print()
    print("| model | mean attended (dense) | static | QKᵀ | PV | indexer | cov static | cov +PV | cov +PV+QKᵀ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for off in (0, 1750, 5750, 29750):
            st, qk, pv, ix = macs(m, off)
            t = st + qk + pv + ix
            print(f"| {m['name']} | {off + 2250} | {st/t:.2%} | {qk/t:.2%} | {pv/t:.2%} | {ix/t:.2%} | {st/t:.1%} | "
                  f"{(st+pv)/t:.1%} | {(st+pv+qk)/t:.1%} |")
    for h_p in (1, 3, 8):
        print()
        print(f"P's fresh noise: {h_p} bit(s) an entry")
        print("| card | word, tree | model | design | γ total | coverage | attention's part | γ PV | γ QKᵀ |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name in CARDS:
            for nb, tree in ((4, 2.0), (4, 1.5), (2, 2.0)):
                card = Card(name, nb, tree)
                for m in MODELS:
                    for dn, g, cov, d, gp, gq in totals(m, card, h_p):
                        print(f"| {name} | {nb} B ×{tree} | {m['name']} | {dn} | {g:.2f} | {cov:.1%} | +{d:.2f} | "
                              f"{gp:.1f} | {gq:.1f} |")
    print()
    print("Decode by roofline, per-head forms, 4 B ×2, cached full-width noise doubling the KV stream (h_p = 3)")
    print("| card | model | γ PV prefill | γ PV decode | γ QKᵀ head prefill | γ QKᵀ head decode |")
    print("|---|---|---|---|---|---|")
    for name in CARDS:
        card = Card(name)
        for m in MODELS:
            gp = weighted(m, card, lambda n: pv_gamma(m, card, n, 3))
            gq = weighted(m, card, lambda n: qk_gamma(m, card, n, "head"))
            print(f"| {name} | {m['name']} | {gp:.1f} | {decode_gamma(m, card, gp):.2f} | {gq:.1f} | "
                  f"{decode_gamma(m, card, gq):.2f} |")
    print()
    print("Lookup route on PV at P = V = 0 (first estimate, upper bound y; 4 B ×2), at the row's attended n")
    print("| card | model | n | y at h_p 1 | h_p 3 | h_p 8 |")
    print("|---|---|---|---|---|---|")
    for name in CARDS:
        card = Card(name)
        for m in MODELS:
            for n in ((2250, 32768) if m["hkv"] > 1 else (690, 1152)):
                ys = [lookup_gain(m, card, n, h) for h in (1, 3, 8)]
                print(f"| {name} | {m['name']} | {n} | {ys[0]:.2%} | {ys[1]:.2%} | {ys[2]:.2%} |")
