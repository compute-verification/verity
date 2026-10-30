"""R80X's word entry at 32 bits against every two-instruction program without LOP3.

Every program of two of fold_window's IADD3 (+-a +-b +-c), IMAD (+-a*b +-c) and LEA ((a << s) +- b), on four row,
four column and four constant sources and a zero, leaves the double difference
D(i, j) = g(i, j) - g(i, 0) - g(0, j) + g(0, 0) of its value g with every 3x3 minor zero, as an identity of integer
polynomials, so zero mod 2^32. Each half's word entry
((R + C) ^ c) + k has a 3x3 minor of D nonzero mod 2^32 (`witnesses`), so no such program computes N + g: at N = 0 its
value would be g. A hi half's word entry is the same question at its (c, k), on lo coins and an h lo half chosen so
neither carry into the hi half occurs.

LEA's 2^s is a free coefficient, so every s is covered. Each program is evaluated on a 4x4 grid at random points mod a
prime, once a seed; a nonzero minor polynomial (degree at most 12) vanishes at a random point with probability at most
12/p. Controls: three products have a nonzero minor, IMAD(v, v, z) after an IMAD has D of rank exactly 2, and a
mutation whose first instruction is row*col + row*col, outside the ISA, must show nonzero minors.

`witnesses` also gives each half four cells with entries nonzero mod 2^31 and D nonzero mod 2^31. That rules out IADD3
then LOP3 when the IADD3 reads N once: flipping N's bit i flips the target's, so the LOP3 passes the IADD3's bit i, and
g is s or -s - 1 (mod 2^31), with s the IADD3's separable other terms, except on cells where g = 0 (mod 2^31). With N
read two or three times, `word_entry_mux` covers it.

Run: `uv run --no-project --with numpy python -B word_entry_rank.py PAIRS.json OUT.json [SEEDS]` (PAIRS:
`r80x_batch.py --pairs-out`), about a minute a seed."""
import itertools
import json
import random
import sys
import time

import numpy as np

P = 2147483647  # 2^31 - 1: products of two residues fit in int64
NR = NC = 4


def sources(rng):
    """Rows 0..3 vary with the grid row, cols with the column, consts not; index 0 is the zero source (N = 0)."""
    i, j = np.meshgrid(np.arange(NR), np.arange(NC), indexing="ij")
    out = [np.zeros(NR * NC, dtype=np.int64)]
    for _ in range(4):
        out.append(rng.integers(0, P, NR)[i].ravel())
    for _ in range(4):
        out.append(rng.integers(0, P, NC)[j].ravel())
    for _ in range(4):
        out.append(np.full(NR * NC, rng.integers(0, P), dtype=np.int64))
    return np.stack(out)


def programs(n):
    """(kind, a, b, c, signs) over n sources, with fold_window's operand symmetries."""
    out = []
    for a, b, c in itertools.combinations_with_replacement(range(n), 3):
        for s in itertools.product((1, -1), repeat=3):
            out.append(("IADD3", a, b, c, s))
    for a, b in itertools.combinations_with_replacement(range(n), 2):
        for c in range(n):
            for s in itertools.product((1, -1), repeat=2):
                out.append(("IMAD", a, b, c, s))
    for a in range(n):
        for b in range(n):
            for s in (1, -1):
                out.append(("LEA", a, b, None, (s,)))
    return out


def table(progs):
    k = np.array([("IADD3", "IMAD", "LEA").index(p[0]) for p in progs])
    a = np.array([p[1] for p in progs])
    b = np.array([p[2] for p in progs])
    c = np.array([p[3] if p[3] is not None else 0 for p in progs])
    s0 = np.array([p[4][0] for p in progs])
    s1 = np.array([p[4][1] if len(p[4]) > 1 else 1 for p in progs])
    s2 = np.array([p[4][2] if len(p[4]) > 2 else 1 for p in progs])
    return k, a, b, c, s0, s1, s2


def run(tab, S, t):
    """Values of every program in tab on sources S (n x cells), LEA's coefficient t."""
    k, a, b, c, s0, s1, s2 = tab
    A, B, C = S[a], S[b], S[c]
    add = (s0[:, None] * A + s1[:, None] * B + s2[:, None] * C) % P
    mad = (s0[:, None] * (A * B % P) + s1[:, None] * C) % P
    lea = (t * A + s0[:, None] * B) % P
    return np.where((k == 0)[:, None], add, np.where((k == 1)[:, None], mad, lea))


def mul(x, y):
    return x * y % P


def det3(D):
    """3x3 determinants of D (m x 3 x 3) mod P, products reduced at each step."""
    a, b, c = D[:, 0], D[:, 1], D[:, 2]
    t0 = mul(a[:, 0], (mul(b[:, 1], c[:, 2]) - mul(b[:, 2], c[:, 1])) % P)
    t1 = mul(a[:, 1], (mul(b[:, 0], c[:, 2]) - mul(b[:, 2], c[:, 0])) % P)
    t2 = mul(a[:, 2], (mul(b[:, 0], c[:, 1]) - mul(b[:, 1], c[:, 0])) % P)
    return (t0 - t1 + t2) % P


def dd(g):
    """Double differences of values g (m x cells) on the grid: m x 3 x 3."""
    g = g.reshape(-1, NR, NC)
    return (g[:, 1:, 1:] - g[:, 1:, :1] - g[:, :1, 1:] + g[:, :1, :1]) % P


def rank1(D):
    return np.all((mul(D[:, 0, 0], D[:, 1, 1]) - mul(D[:, 0, 1], D[:, 1, 0])) % P == 0)


def check(seed, mutate=False):
    rng = np.random.default_rng(seed)
    S = sources(rng)
    t0, t1 = rng.integers(1, P, 2)
    n = len(S)
    first, second = programs(n), programs(n + 1)
    tab1, tab2 = table(first), table(second)
    V0 = run(tab1, S, t0)
    if mutate:  # a first instruction outside the ISA, row*col + row*col: the enumeration must find rank 3 after it
        first = [("MAD2", a, b, c, d) for a in (1, 2) for b in (5, 6) for c in (3, 4) for d in (7, 8)]
        V0 = np.stack([(mul(S[a], S[b]) + mul(S[c], S[d])) % P for _, a, b, c, d in first])
    rank3, count = [], 0
    for i in range(len(first)):
        V1 = run(tab2, np.vstack([S, V0[i]]), t1)
        d = det3(dd(V1))
        bad = np.nonzero(d)[0]
        count += len(bad)
        rank3 += [(first[i], second[j]) for j in bad[:4]]
        if i % 2000 == 0:
            print(f"seed {seed}: {i}/{len(first)} first instructions, rank 3 so far {len(rank3)}", flush=True)
    # controls: the enumeration's IMAD(v, v, c) after IMAD(row, col, col) has rank exactly 2; three products rank 3
    r, c_ = S[1], S[5]
    v = (r * c_ % P + S[6]) % P
    sq = mul(v, v)
    ctl2 = dd(sq[None])
    three = (mul(S[1], S[5]) + mul(S[2], S[6]) + mul(S[3], S[7])) % P
    ctl3 = det3(dd(three[None]))[0]
    return {"seed": seed, "first": len(first), "second": len(second), "programs": len(first) * len(second),
            "rank3": [[list(map(str, x)), list(map(str, y))] for x, y in rank3[:20]], "rank3_count": count,
            "control_square_rank2": bool(det3(ctl2)[0] == 0 and not rank1(ctl2)),
            "control_three_products_rank3": bool(ctl3 != 0)}


M32 = 1 << 32


def entry(c, k):
    return lambda a, b: ((((a + b) % M32) ^ c) + k) % M32


def val(x, w=32):
    x %= 1 << w
    return w if x == 0 else (x & -x).bit_length() - 1


def witnesses(c, k, tries=4000, seed=0):
    """A half's 32-bit witnesses: a 3x3 minor of D nonzero mod 2^32 (no program of rank <= 2), and a D nonzero mod 2^31
    on four cells whose entries are nonzero mod 2^31 (IADD3 then LOP3 with N read once). Either is None if none was
    found in `tries` draws."""
    rng = random.Random(f"{seed}-{c}-{k}")
    f = entry(c, k)
    d3 = s10 = None
    for _ in range(tries):
        a0, b0 = rng.randrange(M32), rng.randrange(M32)
        rs = [rng.randrange(M32) for _ in range(3)]
        cs = [rng.randrange(M32) for _ in range(3)]
        D = [[f(a, b) - f(a, b0) - f(a0, b) + f(a0, b0) for b in cs] for a in rs]
        a, b, cc = D
        det = (a[0] * (b[1] * cc[2] - b[2] * cc[1]) - a[1] * (b[0] * cc[2] - b[2] * cc[0])
               + a[2] * (b[0] * cc[1] - b[1] * cc[0]))
        if det % M32 and (d3 is None or val(det) < d3["valuation"]):
            d3 = {"a0": a0, "b0": b0, "rows": rs, "cols": cs, "valuation": val(det)}
        cells = [f(rs[0], cs[0]), f(rs[0], b0), f(a0, cs[0]), f(a0, b0)]
        if s10 is None and all(x % (1 << 31) for x in cells) and D[0][0] % (1 << 31):
            s10 = {"a": rs[0], "a0": a0, "b": cs[0], "b0": b0, "valuation": val(D[0][0], 31)}
        if d3 and d3["valuation"] < 20 and s10:
            break
    return d3, s10


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        pairs = json.load(fh)
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2]
    t = time.time()
    res = [check(s) for s in seeds]
    mut = check(seeds[0], mutate=True)
    halves = []
    for p in pairs:
        d3, s10 = witnesses(p["c"], p["k"])
        halves.append({"covers": p["covers"][0], "c": p["c"], "k": p["k"], "d3": d3, "s10": s10})
    ctl = {"c=1 (D rank 1)": witnesses(1, 3, 300)[0] is None, "c=0 (D = 0)": witnesses(0, 5, 300) == (None, None)}
    ok = (all(r["rank3_count"] == 0 and r["control_square_rank2"] and r["control_three_products_rank3"] for r in res)
          and mut["rank3_count"] > 0 and all(h["d3"] and h["s10"] for h in halves) and all(ctl.values()))
    out = {"results": res, "mutation": {k: mut[k] for k in ("programs", "rank3_count", "rank3")},
           "halves": halves, "witness_controls": ctl, "seconds": round(time.time() - t, 1),
           "verdict": "no half has an IADD3/IMAD/LEA program of length 2, and each has the IADD3-LOP3 witness"
           if ok else "FAIL"}
    with open(sys.argv[2], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("seconds", "verdict")}), "mutation rank 3:", mut["rank3_count"],
          mut["rank3"][:2],
          [{k: r[k] for k in ("seed", "programs", "rank3_count", "control_square_rank2",
                              "control_three_products_rank3")} for r in res])
    print("halves:", len(halves), "d3 valuations", sorted({h["d3"]["valuation"] for h in halves if h["d3"]}),
          "s10 valuations", sorted({h["s10"]["valuation"] for h in halves if h["s10"]}),
          "missing", [h["covers"] for h in halves if not (h["d3"] and h["s10"])], "controls", ctl)
