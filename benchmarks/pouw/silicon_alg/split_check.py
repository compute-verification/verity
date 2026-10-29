"""Which sketches split on zero-data words, by the accumulator's ring against the sketch's ring (exact).

At zero data a step's words are w_t(i, j) = R_t(i) + C_t(j) over one parity class's rows i and the call's columns j,
with + the accumulator's addition (COUNTER's `decomp`, x + R_t(i) + C_t(class(i), j) with x chosen 0). A sketch S of a
cell's words splits when S(w) = S(R) . S(C) . E: S(R) is shared by the columns, S(C) by the class's rows, and E is the
residual a prover must still compute for each cell. For every pairing below this checks the identity bit for bit, checks
that dropping E (or perturbing one word, where E = 0) is caught, and measures E: the rank of each bit plane of a step's E
over the class tile, rows x columns. A plane of rank r is a sum of r products of a row bit and a column bit, which an
MMA over 0/1 digits batches (IMMA or BMMA); a plane at full rank, min(rows, columns), doesn't factor.

Rank doesn't see one structure: every carry is a threshold, [R mod 2^b >= 2^b - C mod 2^b]. For zg and zp this also
computes a value's residual over a row's columns by sorting (the column side sorts C mod 2^b and keeps suffix sums of
the value's coefficients, shared by the class's rows; each row bisects once a bit) and checks it equals the direct one.

Pairings, accumulator / sketch:
  zz  Z/2^32 / Z/2^32 linear      E = 0
  pp  F_p / F_p linear            E = 0
  ppn F_p / F_p NH over column pairs   E = 0; the cross term sum_t R_t(i)(C_t(j) + C_t(j')) is one F_p matmul
  gg  GF(2) xor / GF(2^64) fold   E = 0
  zp  Z/2^32 / F_p linear         E = 2^32 sum_t r_t b_t, b = [R + C >= 2^32]: one bit a word
  pz  F_p / Z/2^32 linear         E = p sum_t r_t b_t, b = [R + C >= p]: one bit a word
  gz  GF(2) xor / Z/2^32 linear   E = 2 sum_t r_t (R_t & C_t): bit planes of rank 1, one 0/1 matmul over 32 T digits
  zg  Z/2^32 / GF(2^64) fold      E = fold of the carry word K_t = (R + C) ^ R ^ C
  pg  F_p / GF(2^64) fold         E = fold of K_t = ((R + C) mod p) ^ R ^ C

Regimes: `lead` has the lead's magnitudes (R signed below 2^21, C = 0x4B400000 plus a signed term below 2^21: uniform
stand-ins for the walks, as COUNTER's `lead:*`); `salt` has full-width uniform words (row plus column salts).
p is 2^31 - 1; GF(2^64) is rtl/formats.v's mac_gf64 field, x^64 + x^4 + x^3 + x + 1.

    python3 split_check.py --out OUT [--rows 32] [--cols 64] [--steps 74] [--seeds 0,1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import refs

HERE = Path(__file__).resolve().parent
M32 = (1 << 32) - 1
P = (1 << 31) - 1
GF_LOW = 0x1B
LEAD_C = 0x4B400000
PAIRINGS = ["zz", "pp", "ppn", "gg", "zp", "pz", "gz", "zg", "pg"]


def gf_mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 64:
            a ^= (1 << 64) | GF_LOW
    return r


def gf_tables(c: int) -> list[list[int]]:
    """Byte tables for x -> c * x on 32-bit x, by linearity from the basis."""
    tabs = []
    for k in range(4):
        t = [0] * 256
        for m in range(8):
            t[1 << m] = gf_mul(c, 1 << (8 * k + m))
        for x in range(1, 256):
            low = x & -x
            if x != low:
                t[x] = t[x ^ low] ^ t[low]
        tabs.append(t)
    return tabs


def gf_fold(ws: list[int], tabs: list[list[list[int]]]) -> int:
    s = 0
    for w, (t0, t1, t2, t3) in zip(ws, tabs):
        s ^= t0[w & 255] ^ t1[(w >> 8) & 255] ^ t2[(w >> 16) & 255] ^ t3[w >> 24]
    return s


def rank_gf2(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for v in rows:
        while v:
            top = v.bit_length() - 1
            if top not in basis:
                basis[top] = v
                break
            v ^= basis[top]
    return len(basis)


def rank_mod(mat: list[list[int]], p: int) -> int:
    m = [row[:] for row in mat]
    rank, ncol = 0, len(m[0]) if m else 0
    for c in range(ncol):
        piv = next((r for r in range(rank, len(m)) if m[r][c] % p), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][c], p - 2, p)
        m[rank] = [x * inv % p for x in m[rank]]
        for r in range(len(m)):
            if r != rank and m[r][c] % p:
                f = m[r][c]
                m[r] = [(x - f * y) % p for x, y in zip(m[r], m[rank])]
        rank += 1
    return rank


def draw(rng: random.Random, regime: str, acc: str, n: int, side: str) -> list[int]:
    mod = P if acc == "p" else 1 << 32
    if regime == "salt":
        return [rng.randrange(mod) for _ in range(n)]
    base = LEAD_C if side == "C" else 0
    return [(base + rng.randrange(-(1 << 21), 1 << 21)) % mod for _ in range(n)]


def add(acc: str, a: int, b: int) -> int:
    if acc == "z":
        return (a + b) & M32
    if acc == "p":
        return (a + b) % P
    return a ^ b


def dominance_zg(R: list[list[int]], C: list[list[int]], rng: random.Random) -> bool:
    """The carry residual by sorting, for a value spanning a row's columns at a step: X_t(i) = xor_j c_tj K_t(i, j).

    Bit b of K is [C mod 2^b >= 2^b - R mod 2^b], so for each b the column side sorts C mod 2^b and keeps suffix xors of
    c_tj x^b (shared by the class's rows), and each row reads one suffix by bisection: 32 lookups a row and step in
    place of a fold of every column's word.
    """
    import bisect

    T, I, J = len(R), len(R[0]), len(C[0])
    for t in range(T):
        c = [rng.getrandbits(64) for _ in range(J)]
        tabs = []
        for b in range(1, 32):
            order = sorted(range(J), key=lambda j: C[t][j] & ((1 << b) - 1))
            keys = [C[t][j] & ((1 << b) - 1) for j in order]
            suf = [0] * (J + 1)
            for k in range(J - 1, -1, -1):
                suf[k] = suf[k + 1] ^ gf_mul(c[order[k]], 1 << b)
            tabs.append((b, keys, suf))
        for i in range(I):
            direct = 0
            for j in range(J):
                direct ^= gf_mul(c[j], ((R[t][i] + C[t][j]) & M32) ^ R[t][i] ^ C[t][j])
            fast = 0
            for b, keys, suf in tabs:
                fast ^= suf[bisect.bisect_left(keys, (1 << b) - (R[t][i] & ((1 << b) - 1)))]
            if fast != direct:
                return False
    return True


def dominance_zp(R: list[list[int]], C: list[list[int]], rng: random.Random) -> bool:
    """The wrap residual by sorting: sum_j r_tj [R_t(i) + C_t(j) >= 2^32] mod p, one bisection a row and step."""
    import bisect

    T, I, J = len(R), len(R[0]), len(C[0])
    for t in range(T):
        r = [rng.randrange(P) for _ in range(J)]
        order = sorted(range(J), key=lambda j: C[t][j])
        keys = [C[t][j] for j in order]
        suf = [0] * (J + 1)
        for k in range(J - 1, -1, -1):
            suf[k] = (suf[k + 1] + r[order[k]]) % P
        for i in range(I):
            direct = sum(r[j] for j in range(J) if R[t][i] + C[t][j] >= 1 << 32) % P
            if suf[bisect.bisect_left(keys, (1 << 32) - R[t][i])] != direct:
                return False
    return True


def planes(E: list[list[list[int]]], width: int) -> list[list[int]]:
    """GF(2) rank of bit plane b of E_t over the tile, for every step t: out[b] = [rank at t for t ...]."""
    out = []
    for b in range(width):
        out.append([rank_gf2([sum(((row[j] >> b) & 1) << j for j in range(len(row))) for row in Et]) for Et in E])
    return out


def run_pairing(pair: str, regime: str, I: int, J: int, T: int, seed: int) -> dict:
    acc, sk = pair[0], pair[1]
    rng = random.Random(f"{seed}/{regime}/{pair}")
    R = [draw(rng, regime, acc, I, "R") for _ in range(T)]
    C = [draw(rng, regime, acc, J, "C") for _ in range(T)]
    w = [[[add(acc, R[t][i], C[t][j]) for j in range(J)] for i in range(I)] for t in range(T)]
    res: dict = {"pairing": pair, "regime": regime, "seed": seed, "rows": I, "cols": J, "steps": T}

    def cell(t_list, i, j):
        return [t_list[t][i][j] for t in range(T)]

    if sk == "g":
        tabs = [gf_tables(rng.getrandbits(64) | 1) for _ in range(T)]
        SR = [gf_fold([R[t][i] for t in range(T)], tabs) for i in range(I)]
        SC = [gf_fold([C[t][j] for t in range(T)], tabs) for j in range(J)]
        if acc == "g":
            Ew = None
        else:
            Ew = [[[w[t][i][j] ^ R[t][i] ^ C[t][j] for j in range(J)] for i in range(I)] for t in range(T)]
        ok, dropped, nz = True, True, 0
        for i in range(I):
            for j in range(J):
                s = gf_fold(cell(w, i, j), tabs)
                e = 0 if Ew is None else gf_fold(cell(Ew, i, j), tabs)
                ok &= s == SR[i] ^ SC[j] ^ e
                dropped &= s == SR[i] ^ SC[j]
                nz += e != 0
        if Ew is None:
            ws = cell(w, I // 2, J // 3)
            ws[T // 2] ^= 1
            caught = gf_fold(ws, tabs) != SR[I // 2] ^ SC[J // 3]
        else:
            caught = not dropped
        res.update(identity=ok, mutation_caught=caught, residual_cells_nonzero=nz / (I * J))
        res["plane_ranks"] = planes(Ew, 32) if Ew is not None else []
        if pair == "zg":
            res["dominance_exact"] = dominance_zg(R, C, rng)
        return res

    if pair in ("zz", "pp", "zp", "pz", "gz"):
        mod = (1 << 32) if sk == "z" else P
        r = [rng.randrange(mod) for _ in range(T)]
        SR = [sum(r[t] * R[t][i] for t in range(T)) % mod for i in range(I)]
        SC = [sum(r[t] * C[t][j] for t in range(T)) % mod for j in range(J)]
        bits = None
        if pair == "zp":
            bits = [[[(R[t][i] + C[t][j]) >> 32 for j in range(J)] for i in range(I)] for t in range(T)]
            E = [[(1 << 32) * sum(r[t] * bits[t][i][j] for t in range(T)) % mod for j in range(J)] for i in range(I)]
        elif pair == "pz":
            bits = [[[int(R[t][i] + C[t][j] >= P) for j in range(J)] for i in range(I)] for t in range(T)]
            E = [[P * sum(r[t] * bits[t][i][j] for t in range(T)) % mod for j in range(J)] for i in range(I)]
        elif pair == "gz":
            # The cross term as one matmul over 0/1 digits: U[i][(t, b)] = bit b of R_t(i), V[(t, b)][j] = r_t 2^b bit b of C_t(j).
            U = [[(R[t][i] >> b) & 1 for t in range(T) for b in range(32)] for i in range(I)]
            V = [[(r[t] << b) * ((C[t][j] >> b) & 1) for j in range(J)] for t in range(T) for b in range(32)]
            E = [[2 * sum(U[i][k] * V[k][j] for k in range(32 * T) if U[i][k]) % mod for j in range(J)] for i in range(I)]
            bits = [[[R[t][i] & C[t][j] for j in range(J)] for i in range(I)] for t in range(T)]
        else:
            E = [[0] * J for _ in range(I)]
        ok, dropped, nz = True, True, 0
        for i in range(I):
            for j in range(J):
                s = sum(r[t] * w[t][i][j] for t in range(T)) % mod
                ok &= s == (SR[i] + SC[j] - E[i][j]) % mod
                dropped &= s == (SR[i] + SC[j]) % mod
                nz += E[i][j] != 0
        if pair in ("zz", "pp"):
            i0, j0, t0 = I // 2, J // 3, T // 2
            s = sum(r[t] * (w[t][i0][j0] + (t == t0)) for t in range(T)) % mod
            caught = s != (SR[i0] + SC[j0]) % mod
        else:
            caught = not dropped
        res.update(identity=ok, mutation_caught=caught, residual_cells_nonzero=nz / (I * J))
        if bits is not None:
            width = 32 if pair == "gz" else 1
            res["plane_ranks"] = planes(bits, width)
            if pair in ("zp", "pz"):
                res["plane_ranks_mod_p"] = [rank_mod(bits[t], P) for t in range(T)]
            if pair == "zp":
                res["dominance_exact"] = dominance_zp(R, C, rng)
        else:
            res["plane_ranks"] = []
        return res

    if pair == "ppn":
        r = [rng.randrange(P) for _ in range(T)]
        pairs = [(j, j + 1) for j in range(0, J - 1, 2)]
        RR = [sum(r[t] * R[t][i] * R[t][i] for t in range(T)) % P for i in range(I)]
        CC = [sum(r[t] * C[t][a] * C[t][b] for t in range(T)) % P for a, b in pairs]
        U = [[r[t] * R[t][i] % P for t in range(T)] for i in range(I)]
        V = [[(C[t][a] + C[t][b]) % P for a, b in pairs] for t in range(T)]
        X = [[sum(U[i][t] * V[t][q] for t in range(T)) % P for q in range(len(pairs))] for i in range(I)]
        ok = all(
            sum(r[t] * w[t][i][a] * w[t][i][b] for t in range(T)) % P == (RR[i] + X[i][q] + CC[q]) % P
            for i in range(I)
            for q, (a, b) in enumerate(pairs)
        )
        i0, (a0, b0) = I // 2, pairs[len(pairs) // 3]
        s = sum(r[t] * (w[t][i0][a0] + (t == T // 2)) * w[t][i0][b0] for t in range(T)) % P
        caught = s != (RR[i0] + X[i0][len(pairs) // 3] + CC[len(pairs) // 3]) % P
        res.update(identity=ok, mutation_caught=caught, residual_cells_nonzero=0.0, plane_ranks=[],
                   cross_term_inner_dim=T, value_pairs=len(pairs))
        return res
    raise ValueError(pair)


def summarize(res: dict) -> dict:
    half = min(res["rows"], res["cols"]) / 2
    pr = res["plane_ranks"]
    mx = [max(p) for p in pr]
    out = {
        "planes": len(pr),
        "planes_rank_gt2": sum(m > 2 for m in mx),
        "planes_rank_ge_half": sum(m >= half for m in mx),
        "max_rank_by_bit": mx,
        "mean_rank": round(sum(sum(p) / len(p) for p in pr) / len(pr), 2) if pr else 0,
    }
    if "plane_ranks_mod_p" in res:
        mp = res["plane_ranks_mod_p"]
        out["rank_mod_p_max_mean"] = [max(mp), round(sum(mp) / len(mp), 2)]
    if "dominance_exact" in res:
        out["dominance_exact"] = res["dominance_exact"]
    return out


def gf_crosscheck(n: int) -> bool:
    rng = random.Random(7)
    return all(
        gf_mul(a, b) == refs.gf_reduce(refs.clmul(a, b), 64, GF_LOW)
        for a, b in ((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(n))
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--steps", type=int, default=74)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    gf_ok = gf_crosscheck(2000)
    if not gf_ok:
        failures.append("gf_mul disagrees with refs.clmul + refs.gf_reduce")
    for regime in ("lead", "salt"):
        for pair in PAIRINGS:
            for seed in (int(s) for s in a.seeds.split(",")):
                res = run_pairing(pair, regime, a.rows, a.cols, a.steps, seed)
                res["summary"] = summarize(res)
                rows.append(res)
                tag = f"{regime}/{pair}/seed {seed}"
                if not res["identity"]:
                    failures.append(f"{tag}: identity fails")
                if not res["mutation_caught"] and (res["residual_cells_nonzero"] > 0 or pair in ("zz", "pp", "ppn", "gg")):
                    failures.append(f"{tag}: mutation not caught")
                if res.get("dominance_exact") is False:
                    failures.append(f"{tag}: the sorted residual disagrees with the direct one")
                print(f"{tag}: identity {res['identity']} caught {res['mutation_caught']} "
                      f"E!=0 {res['residual_cells_nonzero']:.3f} {res['summary']}", flush=True)
    src = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "split_check.py", HERE / "refs.py")}
    doc = {"rows": rows, "gf_crosscheck": gf_ok, "failures": failures, "sources": src, "args": vars(a)}
    (out / "split_check.json").write_text(json.dumps(doc, indent=1))
    lines = ["| regime | pairing | identity | mutation caught | planes | rank > 2 | rank >= rows/2 | sorted residual exact "
             "| max GF(2) rank by bit (over seeds) |",
             "|---|---|---|---|---:|---:|---:|---|---|"]
    for regime in ("lead", "salt"):
        for pair in PAIRINGS:
            rs = [x for x in rows if x["regime"] == regime and x["pairing"] == pair]
            pr = [max(max(x["plane_ranks"][b]) for x in rs) for b in range(len(rs[0]["plane_ranks"]))]
            half = min(a.rows, a.cols) / 2
            dom = {x.get("dominance_exact") for x in rs} - {None}
            lines.append(f"| {regime} | {pair} | {all(x['identity'] for x in rs)} | {all(x['mutation_caught'] for x in rs)} "
                         f"| {len(pr)} | {sum(m > 2 for m in pr)} | {sum(m >= half for m in pr)} "
                         f"| {all(dom) if dom else '-'} | {','.join(map(str, pr)) if pr else '-'} |")
    (out / "split_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"failures: {len(failures)}", *failures, sep="\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
