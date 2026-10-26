"""COVER-ATTN: coverage of all matmul MACs and γ when attention (QKᵀ, PV) is credited, at the world model's mix.

CPU, under a minute: `python benchmarks/pouw/cover_attn/cover_attn.py [e]` (e = the static serving-skip budget, v1's
0.0075 by default). Conventions are protocol v1's (`benchmarks/pouw/accel_score/nofold.py` `v1()` at 1bfece4240 on
cursor/compute-research-accel-proto-design-ea24): data-sheet INT8 rate, measured SHA-512 rate, plain at η = 0.85, γ in
plain MAC-times a MAC, 4-byte words 32 to a leaf compression, a binary tree (one node a leaf, ×2; 16-ary ×1.5), HYB's
wraps on the ALU pipe, static peel of five products of inner L under the skip.

Static is v1's skip or SKIP-FIX's SEG-COINS; hashing at the measured ALU-only build and RT-UNITS' paper schedules. γ is
v1's additive convention (tensor, then ALU) and, beside it, the period's two pipes overlapped (max). ε_alg is O-IO's
lookup route on PV's round against the honest kernel's own deadline, serial or overlapped.

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
SOFTMAX_PASS = 1.0  # two-pass softmax: S once more on the plain stream, a tensor MAC per QKᵀ MAC
PSYM_PRG, PSYM_N = 6460, 2 ** 16  # O-IO's twin: PRG at 6460/n plain MAC-times a MAC (H200, 4.79 G/s), twin inner n
G_MAX = 8  # Four-Russians chunk: 2^g·d_v·4 B tables must fit shared memory

CARDS = {
    # INT8 dense MAC/s, SHA-512 compressions/s, tensor MACs an ALU lane-op, HBM B/s, tensor MACs an SM-clock
    "H200": (1978.9e12 / 2, 4.79e9, 64, 4.8e12, 4096),
    "RTX PRO 6000": (1.0e15 / 2, 7.13e9, 8, 1.792e12, 1024),
}
# RT-UNITS' paper SHA-512 schedules (IMAD for adds and rotations), v2's honest rates, beside the measured ALU-only build.
SHA_RATES = {"H200": (4.79e9, 6.80e9), "RTX PRO 6000": (7.13e9, 9.82e9)}
# c_bit: public-best price of a uniform 0/1 product per MAC (SKIP-FIX, CALIB): b1 at 8× on the H200, FP4 0/1 at 2× on sm_120.
C_BIT = {"H200": 1 / 8, "RTX PRO 6000": 1 / 2}


class Card:
    def __init__(self, name, nbytes=4, tree=2.0, sha=None):
        self.name = name
        self.mac, self.sha, self.c_op, self.hbm, self.tc_sm = CARDS[name]
        self.sha = sha or self.sha
        self.mt_comp = self.mac / self.sha
        self.C = tree * ETA * self.mt_comp * nbytes / 128
        self.byte = tree * ETA * self.mt_comp / 128
        self.bit = ETA * self.mt_comp / 512
        self.wrap = self.c_op * HYB_P / 2


def dense(name, layers, hidden, hq, hkv, d, ffn, vocab):
    static = layers * hidden * (hq * d + 2 * hkv * d + hq * d + 2 * ffn) + layers * ffn * hidden + hidden * vocab
    return {"name": name, "static": static, "hq": hq, "hkv": hkv, "dqk": d, "dv": d, "kv_bytes": d,
            "noise_bytes": 2 * hkv * d * layers, "att": lambda t: [(layers, t)], "indexer": lambda t: 0}


def dsv4():
    def att(t):
        w = min(t, 128)
        return [(30, w + min(1024, t // 4)), (31, w + t // 128)]

    # One 512-wide latent serves as both K and V, so its noise is streamed once: 512 B a layer a key.
    return {"name": "DeepSeek-V4-Pro", "static": 49e9, "hq": 128, "hkv": 1, "dqk": 512, "dv": 512, "kv_bytes": 584,
            "noise_bytes": 512 * 61, "att": att, "indexer": lambda t: 30 * 64 * 128 * (t // 4)}


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


def static_parts(m, card, form):
    """Static (tensor, ALU) plain MAC-times a MAC, MAC-weighted over the served shapes. 'v1': v1's skip, L = eγK and a
    peel of five products of inner L. 'seg': SKIP-FIX's SEG-COINS, one long segment of (1 − c_bit)K with its coins up
    front and a final segment of c_bit·K whose dither the challenger draws after the long segments' root: two words a
    cell, a peel of one product of the long segment, the served O from the words (no second stream), ε_skip = 0. Both
    carry v1's operand commit and HYB's wraps on the ALU."""
    t = a = s = 0.0
    for k, n, reps in STATIC_SHAPES[m["name"]]:
        alu = card.byte / n + card.wrap
        if form == "seg":
            L = (1 - C_BIT[card.name]) * k
            tk, ak = 1 + L / k, alu + 2 * card.C / k
        else:
            g = 10.0
            for _ in range(100):
                L = min(k, E * g * k)
                g = 1 + alu + card.C / L + HYB_RHO * L / k
            tk, ak = 1 + HYB_RHO * L / k, alu + card.C / L
        wgt = k * n * reps
        t, a, s = t + tk * wgt, a + ak * wgt, s + wgt
    return t / s, a / s


def pv_parts(m, card, n, h_p):
    """PV row of n attended keys: h_p fresh bits a P̃ entry, full-width noise on Ṽ (8 bits an entry, streamed by the
    network certifier and reused by the g query heads and every later query), words to the end of the row capped by the
    unit, the served O from a plain second stream."""
    g_heads = m["hq"] // m["hkv"]
    coins = h_p * card.bit / m["dv"] + 8 * card.bit / (g_heads * n)
    L_unit = UNIT_ANDS / (AND_PER_MAC * (m["dqk"] + 1))
    return 2.0, card.wrap + coins + card.C / min(n, L_unit)


def qk_parts(m, card, n, form):
    """QKᵀ row: full-width noise on Q̃ and K̃ (each entry reused by n keys or n queries), one word a score (head), or one
    query head from each KV group summed along the inner dimension (gqa); the served S from the plain stream."""
    g_heads = m["hq"] // m["hkv"]
    coins = 8 * card.bit / n + 8 * card.bit / (g_heads * n)
    if form == "head":
        return 2.0, card.wrap + coins + card.C / m["dqk"]
    return 2.0, card.wrap + coins + card.C / (m["hkv"] * m["dqk"]) + card.c_op / 2 / m["dqk"]


def psym_parts(card):
    """O-IO's P-SYM (formula until its spec is out): the useful product untouched, plus a PRF-uniform twin of inner n
    credited 1:1, its PRG on SHA-512 and its words at SEG-COINS's two a cell."""
    prg = PSYM_PRG * card.mt_comp / (CARDS["H200"][0] / 4.79e9)
    return 2.0, card.wrap + (prg + 2 * card.C) / PSYM_N


def decode_gamma(m, card, g_cb, noise_stream=2.0):
    """Roofline at decode for a per-head form: plain = max(compute, KV stream); protocol = max(compute γ_cb, KV stream with
    its cached full-width noise, noise_stream ×)."""
    g_heads = m["hq"] // m["hkv"]
    intensity = g_heads * m["dv"] * 2 * 1 / (m["kv_bytes"] * (2 if m["hkv"] > 1 else 1))
    t_mem = card.mac / (intensity * card.hbm)
    plain = max(1 / ETA, t_mem)
    return max(g_cb / ETA, noise_stream * t_mem) / plain


def weighted(m, fn, offset=0):
    hist, _ = rows_of(m, offset=offset)
    w = sum(c * n for n, c in hist.items())
    parts = [(c * n, fn(n)) for n, c in hist.items()]
    return tuple(sum(x * p[i] for x, p in parts) / w for i in range(2))


def mean_n(m):
    hist, _ = rows_of(m)
    return sum(c * n * n for n, c in hist.items()) / sum(c * n for n, c in hist.items())


def lookup_alu(m, card):
    """ALU cost, in tensor-MAC-times, of one PV MAC moved to Four-Russians lookups on 8-bit P̃: 8/g adds a MAC over the
    bit-planes, plus the 2^g-entry tables of Ṽ chunks built once for the n/2 queries after them, g ≤ G_MAX."""
    n = mean_n(m)
    return min(card.c_op * 8 / g * (1 + 2 ** g / (n / 2)) for g in range(2, G_MAX + 1))


DESIGNS = [  # name, static form, attention: None, "pv", "att" (PV and QKᵀ per head), "gqa", "psym"
    ("v1 skip, static only", "v1", None),
    ("v1 skip + ATT", "v1", "att"),
    ("SEG-COINS, static only", "seg", None),
    ("SEG-COINS + ATT-PV", "seg", "pv"),
    ("SEG-COINS + ATT", "seg", "att"),
    ("SEG-COINS + ATT, QKᵀ GQA-batched", "seg", "gqa"),
    ("SEG-COINS + P-SYM twins for attention", "seg", "psym"),
]


def totals(m, card, h_p=8):
    """Per design: γ additive (v1's convention, tensor then ALU), γ overlapped (max of the two pipes over the period),
    coverage, attention's part of additive γ, and ε_alg of the lookup route on PV against a serial and an overlapped
    honest kernel (the deadline is the honest kernel's time; the route moves PV round MACs into the ALU's slack)."""
    st, qk, pv, ix = macs(m)
    tot = st + qk + pv + ix
    ell = lookup_alu(m, card)
    out = []
    for name, sform, att in DESIGNS:
        ts, as_ = static_parts(m, card, sform)
        T, A, cov = st * ts, st * as_, st
        if att == "psym":
            tp, ap = psym_parts(card)
            T, A, cov = T + (qk + pv + ix) * tp, A + (qk + pv + ix) * ap, cov + qk + pv + ix
        else:
            T += ix
            if att:
                tp, ap = weighted(m, lambda n: pv_parts(m, card, n, h_p))
                T, A, cov = T + pv * tp + qk * SOFTMAX_PASS, A + pv * ap, cov + pv
            else:
                T += pv
            if att in ("att", "gqa"):
                form = "gqa" if att == "gqa" and m["hkv"] > 1 else "head"
                tq, aq = weighted(m, lambda n: qk_parts(m, card, n, form))
                T, A, cov = T + qk * tq, A + qk * aq, cov + qk
            else:
                T += qk
        g_add, g_ov = (T + A) / tot, max(T, A) / tot
        att_part = g_add - (st * (ts + as_) + qk + pv + ix) / tot
        routable = pv if att in ("pv", "att", "gqa") else 0.0
        e_serial = min(routable, T / ell) / cov
        e_ov = min(routable, max(0.0, T - A) / ell) / cov
        out.append((name, g_add, g_ov, cov / tot, att_part, e_serial, e_ov, T / tot, A / tot))
    return out


def stream(m, card, g):
    """Network-certifier bulk noise for ATT (K̃ and Ṽ once a key), bytes a second a GPU, at γ g; tokens a second a GPU."""
    st, qk, pv, ix = macs(m)
    tok = card.mac * ETA / ((st + qk + pv + ix) / 4500 * g)
    return tok, tok * m["noise_bytes"], tok * (qk + pv + ix) / 4500 / PSYM_N


if __name__ == "__main__":
    print(f"v1's static skip budget e = {E:.2%}; P̃'s fresh noise 8 bits; two-pass softmax +{SOFTMAX_PASS:.0f} QKᵀ pass")
    for name in CARDS:
        for sha in SHA_RATES[name]:
            for nb, tree in ((4, 2.0), (4, 1.5), (2, 2.0)):
                c = Card(name, nb, tree, sha)
                print(f"{name} SHA {sha/1e9:.2f} G/s: C(word {nb} B, tree ×{tree}) = {c.C:,.0f}; coin bit {c.bit:.0f}; "
                      f"wraps {c.wrap:.2f}; c_bit {C_BIT[name]}")
    print()
    print("| model | mean attended (dense) | static | QKᵀ | PV | indexer | cov static | cov +PV | cov +PV+QKᵀ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for off in (0, 1750, 5750, 29750):
            st, qk, pv, ix = macs(m, off)
            t = st + qk + pv + ix
            print(f"| {m['name']} | {off + 2250} | {st/t:.2%} | {qk/t:.2%} | {pv/t:.2%} | {ix/t:.2%} | {st/t:.1%} | "
                  f"{(st+pv)/t:.1%} | {(st+pv+qk)/t:.1%} |")
    print()
    print("Totals at the world model (4 B words, binary tree unless stated)")
    print("| card | SHA G/s | word, tree | model | design | γ additive | γ overlapped | coverage | attention's part | "
          "ε_alg serial | ε_alg overlapped | tensor | ALU |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in CARDS:
        for sha in SHA_RATES[name]:
            for nb, tree in ((4, 2.0), (4, 1.5), (2, 2.0)):
                card = Card(name, nb, tree, sha)
                for m in MODELS:
                    for dn, ga, go, cov, d, es, eo, T, A in totals(m, card):
                        if (nb, tree) != (4, 2.0) and dn != "SEG-COINS + ATT":
                            continue
                        print(f"| {name} | {sha/1e9:.2f} | {nb} B ×{tree} | {m['name']} | {dn} | {ga:.2f} | {go:.2f} | "
                              f"{cov:.1%} | {d:+.2f} | {es:.2%} | {eo:.2%} | {T:.2f} | {A:.2f} |")
    print()
    print("Decode by roofline, per-head forms, 4 B ×2, cached full-width noise doubling the KV stream (h_p = 8)")
    print("| card | SHA G/s | model | γ PV prefill | γ PV decode | γ QKᵀ head prefill | γ QKᵀ head decode |")
    print("|---|---|---|---|---|---|---|")
    for name in CARDS:
        for sha in SHA_RATES[name]:
            card = Card(name, sha=sha)
            for m in MODELS:
                gp = sum(weighted(m, lambda n: pv_parts(m, card, n, 8)))
                gq = sum(weighted(m, lambda n: qk_parts(m, card, n, "head")))
                print(f"| {name} | {sha/1e9:.2f} | {m['name']} | {gp:.1f} | {decode_gamma(m, card, gp):.2f} | {gq:.1f} | "
                      f"{decode_gamma(m, card, gq):.2f} |")
    print()
    print("Network-certifier bulk stream a GPU (SEG-COINS + ATT, 4 B ×2, paper SHA): ATT's K̃ and Ṽ noise once a key; "
          "P-SYM's streamed twin side (8/n bits a MAC)")
    print("| card | model | noise B a token | γ additive: tokens/s, GB/s | γ overlapped: tokens/s, GB/s | "
          "P-SYM attention twins GB/s |")
    print("|---|---|---|---|---|---|")
    for name in CARDS:
        card = Card(name, sha=SHA_RATES[name][1])
        for m in MODELS:
            r = {dn: (ga, go) for dn, ga, go, *_ in totals(m, card)}
            ga, go = r["SEG-COINS + ATT"]
            ta, ba, _ = stream(m, card, ga)
            to, bo, _ = stream(m, card, go)
            _, _, bp = stream(m, card, r["SEG-COINS + P-SYM twins for attention"][1])
            print(f"| {name} | {m['name']} | {m['noise_bytes']:,} | {ta:,.0f}, {ba/1e9:.2f} | {to:,.0f}, {bo/1e9:.2f} | "
                  f"{bp/1e9:.3f} |")
    print()
    print("Lookup route: ALU tensor-MAC-times per PV MAC moved (8-bit P̃, g ≤ 8), at the PV-weighted mean attended n")
    print("| card | model | mean n | ℓ |")
    print("|---|---|---|---|")
    for name in CARDS:
        for m in MODELS:
            print(f"| {name} | {m['name']} | {mean_n(m):,.0f} | {lookup_alu(m, Card(name)):.1f} |")
