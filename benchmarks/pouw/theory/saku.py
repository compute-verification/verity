"""N-SAKU: designs that rest on SAK on literally uniform operands (SAK-U), and the zero-data test every design must pass.

Checks (exact integer arithmetic, small shapes, seeded):
  ncp_zero   NCP (X = [A+R | A | R], Y = [B+S | -S | -(B+S)]) is exact, and at zero data an adversary that computes
             only block 1's prefix sums outputs every committed word at density L: a third of the honest MACs.
  ssn_zero   ssn_1l (31 data lanes + 1 coin lane a k32 step, step 1 28 + 4 against a column code) at zero data: every
             committed word is the code term plus the row's coin prefix sum, computed without the GEMM.
  ortho      an exact design's zero-data operands (NCP, ncp3) have orthogonal row spans, dims summing to at most the
             stack width, so a uniform instance embedded in the noise is at most half the width.
  exact_skip the middle-segment adversary on ncp3 (forward, then backward from the known total) outputs every committed
             word without computing one segment, at zero data and with A.B as advice.
  dch_linear prover-chosen data (zero, random, rank 1, zero halves, repeated rows, constant, Gram, one side zero) never
             gives ncp3's segment products a linear relation beyond the total.
Pricing (estimates, W1 units, moves free): the best SAK-U-only design, a uniform side product with final words committed;
the zero-data gap of NCP, ssn_1l and served designs with q noise lanes a step at commit density L; the skip, L/K; and
ncp3 at sparse L in M0, for Llama-3.2-1B's block linears and a V4-like dense layer.

usage: saku.py [OUT.json]
"""
import json
import math
import sys

import numpy as np

H32 = 32
HASH_B = {"floor": 908.0, "bar": 1775.0}
COIN_B = {"sw-noise": 2101.0, "sw-price": 3550.0}

LLAMA = [("qkv", 2048, 3072), ("o", 2048, 2048), ("gate_up", 2048, 16384), ("down", 8192, 2048)]
LLAMA_LAYERS = 16
V4 = [("v4 dense", 7168, 7168)]


def prefix_words(x, y, L):
    """Every L-th prefix sum of x @ y.T along k: shape (K'/L, m, n)."""
    k = x.shape[1]
    assert k % L == 0
    out, acc = [], np.zeros((x.shape[0], y.shape[0]), dtype=np.int64)
    for j in range(0, k, L):
        acc = acc + x[:, j:j + L] @ y[:, j:j + L].T
        out.append(acc.copy())
    return np.stack(out)


def ncp_stack(a, b, r, s):
    return np.concatenate([a + r, a, r], 1), np.concatenate([b + s, -s, -(b + s)], 1)


def ncp_zero(rng, m=24, n=20, k=128, L=64):
    a = rng.integers(-64, 64, (m, k))
    b = rng.integers(-64, 64, (n, k))
    r = rng.integers(-64, 64, (m, k))
    s = rng.integers(-64, 64, (n, k))
    x, y = ncp_stack(a, b, r, s)
    exact = bool((x @ y.T == a @ b.T).all())
    z = np.zeros_like
    x0, y0 = ncp_stack(z(a), z(b), r, s)
    honest = prefix_words(x0, y0, L)
    p = prefix_words(r, s, L)
    adv = np.concatenate([p, np.repeat(p[-1:], k // L, 0), p[-1] - p], 0)
    macs_adv, macs_honest = k * m * n, 3 * k * m * n
    return {"m": m, "n": n, "k": k, "L": L, "exact": exact, "zero_data_words_match": bool((adv == honest).all()),
            "words": int(honest.size), "macs_adv": macs_adv, "macs_honest": macs_honest,
            "adv_share": macs_adv / macs_honest}


def ssn_code(n, lanes=4, base=13):
    c = np.arange(1, n + 1, dtype=np.int64)
    centre = (base ** lanes - 1) // 2
    c = np.where(c < centre, c, c + 1)
    return np.stack([(c // base ** e) % base - 6 for e in range(lanes)], 1)


def ssn_stack(a, b, coin, code):
    """a (m, 28 + 31 (S-1)), b likewise; coin (m, 4 + S - 1) bits; code (n, 4). Lanes: step 1 is 28 data + 4 coin lanes
    against the column code, each later step 31 data + 1 coin lane against weight 1."""
    m, n = a.shape[0], b.shape[0]
    s_steps = 1 + (a.shape[1] - 28) // 31
    xs = [np.concatenate([a[:, :28], coin[:, :4]], 1)]
    ys = [np.concatenate([b[:, :28], code], 1)]
    for t in range(1, s_steps):
        lo = 28 + 31 * (t - 1)
        xs.append(np.concatenate([a[:, lo:lo + 31], coin[:, 3 + t:4 + t]], 1))
        ys.append(np.concatenate([b[:, lo:lo + 31], np.ones((n, 1), dtype=np.int64)], 1))
    return np.concatenate(xs, 1), np.concatenate(ys, 1)


def ssn_zero(rng, m=24, n=20, s_steps=8, L=64):
    kd = 28 + 31 * (s_steps - 1)
    code = ssn_code(n)
    coin = rng.integers(0, 2, (m, 4 + s_steps - 1))
    a0, b0 = np.zeros((m, kd), dtype=np.int64), np.zeros((n, kd), dtype=np.int64)
    x0, y0 = ssn_stack(a0, b0, coin, code)
    honest = prefix_words(x0, y0, L)
    per = L // 32
    code_term = coin[:, :4] @ code.T
    run = np.cumsum(coin[:, 4:], 1)
    adv = []
    for w in range(s_steps // per):
        last = (w + 1) * per - 1
        adv.append(code_term + (run[:, last - 1][:, None] if last >= 1 else 0))
    adv = np.stack(adv)
    a = rng.integers(-64, 64, (m, kd))
    b = rng.integers(-64, 64, (n, kd))
    x, y = ssn_stack(a, b, coin, code)
    served = x @ y.T - a @ b.T
    return {"m": m, "n": n, "steps": s_steps, "L": L, "zero_data_words_match": bool((adv == honest).all()),
            "words": int(honest.size), "adv_ops": int(4 * m * n + m * (s_steps - 1) + honest.size),
            "honest_macs": int(32 * s_steps * m * n),
            "served_noise_is_coins_only": bool((served == code_term + run[:, -1][:, None]).all())}


def ncp3_stack(a, b, e1, f1, r):
    """SW-NOISE's ncp3 (noise_sweep.py): X = [A+E1 | A+E1R | -(A+E1(I+R))], Y = [B+F1; B-RF1; B+(I-R)F1], as row-major
    operands of X @ Y.T."""
    e3 = e1 + e1 @ r
    f3 = f1 - r @ f1
    x = np.concatenate([a + e1, a + e1 @ r, -(a + e3)], 1)
    y = np.concatenate([b.T + f1, b.T - r @ f1, b.T + f3], 0)
    return x, y.T


def block_perm(rng, k):
    r = np.zeros((k, k), dtype=np.int64)
    for i, p in enumerate(rng.permutation(k // 4)):
        r[4 * i:4 * i + 4, 4 * p:4 * p + 4] = np.eye(4, dtype=np.int64)
    return r


def skip_adversary(x, y, total, L, skip):
    """Every L-th prefix sum of x @ y.T without computing segment `skip`: forward up to it, backward from the known
    total after it. Returns the words and the MACs spent."""
    k = x.shape[1]
    s = k // L
    seg = lambda t: x[:, t * L:(t + 1) * L] @ y[:, t * L:(t + 1) * L].T
    words = [None] * s
    acc = np.zeros_like(total)
    for t in range(skip):
        acc = acc + seg(t)
        words[t] = acc.copy()
    words[s - 1] = total.copy()
    acc = total.copy()
    for t in range(s - 1, skip, -1):
        acc = acc - seg(t)
        words[t - 1] = acc.copy()
    return np.stack(words), (s - 1) * L * x.shape[0] * y.shape[0]


def exact_skip(rng, m=24, n=20, k=128, L=64, zero=True):
    """The middle-segment adversary on ncp3: the final word is A.B (0 at zero data, a move), so one segment of width L
    is never computed; the backward segments accumulate into the known total at the forward ones' price. Checked
    against the honest words at zero data and, with A.B as advice, at random data."""
    a = np.zeros((m, k), dtype=np.int64) if zero else rng.integers(-63, 64, (m, k))
    b = np.zeros((n, k), dtype=np.int64) if zero else rng.integers(-63, 64, (n, k))
    e1, f1 = rng.integers(-31, 33, (m, k)), rng.integers(-31, 33, (k, n))
    x, y = ncp3_stack(a, b, e1, f1, block_perm(rng, k))
    honest = prefix_words(x, y, L)
    exact = bool((honest[-1] == a @ b.T).all())
    s = x.shape[1] // L
    adv, macs = skip_adversary(x, y, a @ b.T, L, s // 2)
    return {"m": m, "n": n, "k": k, "L": L, "zero_data": zero, "exact": exact,
            "words_match": bool((adv == honest).all()), "macs_adv": macs, "macs_honest": s * L * m * n,
            "saved_share_of_gemm": 1 - macs / (s * L * m * n)}


def ncp3_zero_stack(rng, m, n, k, r):
    e1, f1 = rng.integers(-31, 33, (m, k)), rng.integers(-31, 33, (k, n))
    return ncp3_stack(np.zeros((m, k), dtype=np.int64), np.zeros((n, k), dtype=np.int64), e1, f1, r)


def ortho(rng, design="ncp", m=24, n=20, k=48, draws=6):
    """Zero-data row spans of an exact stack at fixed public structure (ncp3's permutation R drawn once). A uniform
    instance embedded in the noise coordinates is at most min(dim_x, dim_y) <= width / 2 wide."""
    rows_x, rows_y, zero = [], [], True
    r = block_perm(rng, k)
    for _ in range(draws):
        if design == "ncp":
            r = rng.integers(-64, 64, (m, k))
            s = rng.integers(-64, 64, (n, k))
            x0, y0 = ncp_stack(np.zeros_like(r), np.zeros_like(s), r, s)
        else:
            x0, y0 = ncp3_zero_stack(rng, m, n, k, r)
        zero &= bool((x0 @ y0.T == 0).all())
        rows_x.append(x0)
        rows_y.append(y0)
    dx = int(np.linalg.matrix_rank(np.concatenate(rows_x).astype(float)))
    dy = int(np.linalg.matrix_rank(np.concatenate(rows_y).astype(float)))
    cross = bool((np.concatenate(rows_x) @ np.concatenate(rows_y).T == 0).all())
    return {"design": design, "width": 3 * k, "dim_x": dx, "dim_y": dy, "final_zero": zero,
            "spans_orthogonal": cross, "dims_le_width": dx + dy <= 3 * k, "embed_max": min(dx, dy)}


def rank_mod_p(a, p=2_147_483_647):
    """Rank over GF(p), a lower bound on the rank over Q."""
    a = [[int(v) % p for v in row] for row in a]
    rank, cols = 0, len(a[0]) if a else 0
    for c in range(cols):
        piv = next((i for i in range(rank, len(a)) if a[i][c]), None)
        if piv is None:
            continue
        a[rank], a[piv] = a[piv], a[rank]
        inv = pow(a[rank][c], p - 2, p)
        a[rank] = [v * inv % p for v in a[rank]]
        for i in range(len(a)):
            if i != rank and a[i][c]:
                f = a[i][c]
                a[i] = [(v - f * w) % p for v, w in zip(a[i], a[rank])]
        rank += 1
    return rank


def ncp3_data(rng, kind, m, n, k):
    """Prover-chosen data for the DataCantHelp search, int7 like ncp3's activations."""
    full = lambda r, c: rng.integers(-63, 64, (r, c))
    if kind == "zero":
        return np.zeros((m, k), dtype=np.int64), np.zeros((n, k), dtype=np.int64)
    if kind == "random":
        return full(m, k), full(n, k)
    if kind == "rank1":
        return np.outer(rng.integers(-7, 8, m), rng.integers(-7, 8, k)), np.outer(rng.integers(-7, 8, n), rng.integers(-7, 8, k))
    if kind == "zero_half":
        a, b = full(m, k), full(n, k)
        a[:, : k // 2] = 0
        b[:, k // 2:] = 0
        return a, b
    if kind == "repeated_rows":
        return np.repeat(full(1, k), m, 0), np.repeat(full(1, k), n, 0)
    if kind == "constant":
        return np.full((m, k), 63, dtype=np.int64), np.full((n, k), -63, dtype=np.int64)
    if kind == "gram":
        a = full(m, k)
        return a, a[:n].copy()
    if kind == "a_only":
        return full(m, k), np.zeros((n, k), dtype=np.int64)
    raise ValueError(kind)


def segment_relations(rng, kind, m=6, n=5, k=64, L=8, draws=8):
    """Linear relations sum_t c_t S_t = const(data) among ncp3's segment products S_t, holding for every noise draw at
    fixed prover-chosen data: the null space of the products' differences across draws, its dimension by an exact
    rank (GF(p) lower bound, plus the known all-ones relation). One relation, the total, is the middle-segment skip;
    more would be data helping a linear skip. The bilinear noise part of S_t doesn't depend on the data, so data can
    only remove relations zero data has."""
    a, b = ncp3_data(rng, kind, m, n, k)
    s = 3 * k // L
    prods = []
    for _ in range(draws):
        e1, f1 = rng.integers(-31, 33, (m, k)), rng.integers(-31, 33, (k, n))
        x, y = ncp3_stack(a, b, e1, f1, block_perm(rng, k))
        prods.append(np.stack([(x[:, t * L:(t + 1) * L] @ y[:, t * L:(t + 1) * L].T).ravel() for t in range(s)], 1))
    diffs = np.concatenate([p - prods[0] for p in prods[1:]])
    rank = rank_mod_p(diffs.tolist())
    total_ok = all(bool((p.sum(1) == (a @ b.T).ravel()).all()) for p in prods)
    return {"data": kind, "segments": s, "rank": rank, "relations": s - rank, "total_relation": total_ok}


def usp_row(name, m, k, n, h, c):
    """Uniform side product, shape-matched: honest = stock + U V^T (U, V from coins) with V's final words committed."""
    stock = m * n * k
    uniform = m * n * k
    hash_words = h * 4 * m * n
    hash_a = h * m * k
    coins = c * (m + n) * k
    t2 = (stock + uniform + hash_words + hash_a + coins) / stock
    return {"linear": name, "m": m, "k": k, "n": n, "T1_credited_coins": 1.0, "T1_uncredited_coins": 1.0 + coins / stock,
            "T2_w1": t2, "hash_share": (hash_words + hash_a) / stock, "coin_share": coins / stock}


def zero_gap(design, L, alpha_ssn=1.044):
    """Zero-data gap in units of stock at a committed word every L stacked lanes, every distinct word written at H32.
    NCP: the honest GEMM is 3K wide; the adversary runs block 1's K-wide GEMM, writes block 2's one repeated value
    once and block 3's K/L distinct words. ssn_1l: the adversary writes its alpha K / L words and runs no GEMM."""
    if design == "ncp":
        honest, adv = 3.0, 1.0 + H32 / L
    else:
        honest, adv = alpha_ssn, alpha_ssn * H32 / L
    return max(0.0, honest - adv)


def served_zero_gap(L, q, depth=32, alpha=1.0):
    """Zero-data gap, in units of stock, of a served design whose noise fills q of a k32 step's 32 lanes on both
    operands: at zero data a segment's L lanes hold q L / 32 noise lanes, packed into the narrowest MMA the noise's
    alphabet allows (depth 32 for s8, 64 for s4, 256 for b1) at H32 a register written."""
    return max(0.0, alpha * (1.0 - H32 * math.ceil(q * L / 32 / depth) / L))


def ncp3_sparse(L, m, shapes, c1):
    """ncp3 at a committed word every L stacked lanes, M0 (prover-chosen data), MAC-weighted over `shapes`: the skip
    L/K (it replaces SW-NOISE's final word, 32/K at L 32), forming 32/n + 32/m (8/n + 8/m left once C1 forces 24),
    and the coins, 6 bits an entry of E1 and F1 at SW-NOISE's 2,101 units a byte, when uncredited."""
    f = 8.0 if c1 else 32.0
    w = [m * k * n for _, k, n in shapes]
    terms = {"skip": [L / k for _, k, n in shapes], "forming": [f / n + f / m for _, k, n in shapes],
             "coins": [COIN_B["sw-noise"] * 6 / 8 * (1 / n + 1 / m) for _, k, n in shapes]}
    agg = {key: sum(v * wi for v, wi in zip(vals, w)) / sum(w) for key, vals in terms.items()}
    agg["T1_credited_coins"] = agg["skip"] + agg["forming"]
    agg["T1_uncredited_coins"] = agg["T1_credited_coins"] + agg["coins"]
    return agg


def main():
    rng = np.random.default_rng(20261029)
    checks = {"ncp_zero": [ncp_zero(rng, L=L) for L in (32, 64, 128)],
              "ssn_zero": [ssn_zero(rng, L=L) for L in (32, 64, 128)],
              "ortho": [ortho(rng), ortho(rng, "ncp3")],
              "exact_skip": [exact_skip(rng, L=L, zero=z) for L in (32, 64, 128) for z in (True, False)],
              "dch_linear": [segment_relations(rng, kind, L=L)
                             for kind in ("zero", "random", "rank1", "zero_half", "repeated_rows", "constant", "gram",
                                          "a_only") for L in (4, 8, 16)]}
    ok = all(r["exact"] and r["zero_data_words_match"] for r in checks["ncp_zero"])
    ok &= all(r["exact"] and r["words_match"] for r in checks["exact_skip"])
    ok &= all(r["zero_data_words_match"] and r["served_noise_is_coins_only"] for r in checks["ssn_zero"])
    ok &= all(o["final_zero"] and o["spans_orthogonal"] and o["dims_le_width"] for o in checks["ortho"])
    ok &= all(r["total_relation"] and r["relations"] == 1 for r in checks["dch_linear"])

    pricing = {}
    for t in (64, 512, 8192):
        for hk, h in HASH_B.items():
            for ck, c in COIN_B.items():
                rows = [usp_row(nm, t, k, n, h, c) for nm, k, n in LLAMA]
                w = [t * k * n for _, k, n in LLAMA]
                agg = {key: sum(r[key] * wi for r, wi in zip(rows, w)) / sum(w)
                       for key in ("T1_credited_coins", "T1_uncredited_coins", "T2_w1", "hash_share", "coin_share")}
                pricing[f"llama T{t} hash-{hk} coin-{ck}"] = {"aggregate": agg, "rows": rows}
    for t in (1, 8192):
        for hk, h in HASH_B.items():
            r = usp_row("v4 dense", t, 7168, 7168, h, COIN_B["sw-noise"])
            pricing[f"v4 dense m{t} hash-{hk} coin-sw-noise"] = {"aggregate": r, "rows": [r]}

    zero = {f"{d} L{L}": zero_gap(d, L) for d in ("ncp", "ssn") for L in (32, 64, 128, 256, 1024)}
    served = {f"{fmt} q{q} L{L}": served_zero_gap(L, q, depth)
              for fmt, depth in (("s8", 32), ("s4", 64), ("b1", 256)) for q in (1, 16, 17, 24, 25, 32) for L in (64, 128)}
    llama_macs = sum(k * n for _, k, n in LLAMA)
    skip = {}
    for L in (32, 64, 96, 128, 256, 1024):
        skip[f"llama linears L{L}"] = sum(L / k * k * n for _, k, n in LLAMA) / llama_macs
        skip[f"v4 dense L{L}"] = L / 7168
    skip_max_L = {f"K {k}": 0.01 * k for k in (2048, 7168, 8192, 16384)}
    sparse = {f"{wl} m{m} L{L}{' C1' if c1 else ''}": ncp3_sparse(L, m, shapes, c1)
              for wl, shapes in (("llama", LLAMA), ("v4 dense", V4)) for m in (64, 512, 8192)
              for L in (32, 64, 128) for c1 in (False, True)}

    for name, rs in checks.items():
        print(name, json.dumps(rs))
    for key, v in pricing.items():
        a = v["aggregate"]
        print(f"usp {key:42s} T1 {100 * a['T1_credited_coins']:.1f}% (coins uncredited {100 * a['T1_uncredited_coins']:.1f}%) "
              f"T2(W1) {a['T2_w1']:.2f}x hash {a['hash_share']:.3f} coins {a['coin_share']:.3f}")
    for key, g in zero.items():
        print(f"zero-data gap {key:18s} {100 * g:.1f}% of stock")
    for key, g in served.items():
        print(f"served design, zero-data gap {key:10s} {100 * g:.1f}% of stock (alpha 1)")
    for key, g in skip.items():
        print(f"exact design, middle-segment skip {key:20s} {100 * g:.3f}% of stock")
    for key, v in skip_max_L.items():
        print(f"exact design: largest L with skip <= 1% at {key}: {v:.1f} lanes")
    for key, a in sparse.items():
        print(f"ncp3 M0 {key:24s} T1 {100 * a['T1_credited_coins']:.3f}% (skip {100 * a['skip']:.3f}, forming "
              f"{100 * a['forming']:.3f}; coins uncredited {100 * a['T1_uncredited_coins']:.2f}%)")
    print("PASS" if ok else "FAIL", "checks")
    if len(sys.argv) > 1:
        json.dump({"checks": checks, "pricing": pricing, "zero_data_gap": zero, "served_zero_gap": served,
                   "exact_skip_gap": skip, "exact_skip_max_L": skip_max_L, "ncp3_sparse_m0": sparse, "pass": ok},
                  open(sys.argv[1], "w"), indent=1, default=float)
    sys.exit(0 if ok else 1)


main()
