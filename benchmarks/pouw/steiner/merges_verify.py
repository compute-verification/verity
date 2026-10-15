"""Independent check of a merge/copy computation (merges_sat.py's JSON steps), sharing no code with merges_sat or steiner.

A form is a set of monomials (('x', r, k), ('y', k, j)); a step is replayed on those sets, the rank of every edge
increment is computed by Gaussian elimination on its own matrix, and the terminals are rebuilt from the definition
T[(r, j), s] = sum_{k < s L} x[r][k] y[k][j]. Also `lift`: a single-level computation for <a,b,c> (L = b) becomes one
for <a,bm,c> (L = bm) by replacing each inner index k with k + i b for every block i = 0..m-1: an edge u (x) w becomes a
chain of m edges (u (x) w in each block; each has rank 1), and a merge or copy stays one step, since the lift of a sum is
the sum of the lifts. Same merges and copies, m times the edges.

`python3 merges_verify.py check FILE.jsonl` re-checks every sat line; `lift FILE.jsonl --m 2,3,...` lifts every
single-level sat line and checks the result.
"""

from __future__ import annotations

import argparse
import json


def decode_form(bits: int, a: int, b: int, c: int) -> frozenset:
    R, C = a * b, b * c
    out = set()
    for i in range(R):
        for j in range(C):
            if (bits >> (i * C + j)) & 1:
                out.add((("x",) + divmod(i, b), ("y",) + divmod(j, c)))
    return frozenset(out)


def vec_x(u: int, a: int, b: int) -> list:
    return [("x",) + divmod(i, b) for i in range(a * b) if (u >> i) & 1]


def vec_y(w: int, b: int, c: int) -> list:
    return [("y",) + divmod(j, c) for j in range(b * c) if (w >> j) & 1]


def sym_diff_all(forms) -> frozenset:
    acc: set = set()
    for f in forms:
        acc ^= set(f)
    return frozenset(acc)


def rank_of(form: frozenset) -> int:
    rows = sorted({m[0] for m in form})
    cols = sorted({m[1] for m in form})
    ci = {cc: n for n, cc in enumerate(cols)}
    mat = []
    for r in rows:
        v = 0
        for m in form:
            if m[0] == r:
                v ^= 1 << ci[m[1]]
        mat.append(v)
    rank = 0
    for bit in range(len(cols)):
        piv = next((i for i in range(rank, len(mat)) if (mat[i] >> bit) & 1), None)
        if piv is None:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        for i in range(len(mat)):
            if i != rank and (mat[i] >> bit) & 1:
                mat[i] ^= mat[rank]
        rank += 1
    return rank


def terminals(a: int, b: int, c: int, L: int) -> dict:
    out = {}
    for r in range(a):
        for j in range(c):
            for s in range(1, b // L + 1):
                out[(r, j, s)] = frozenset((("x", r, k), ("y", k, j)) for k in range(s * L))
    return out


def check(a, b, c, L, steps, arity=3) -> dict:
    """Replays steps given as edges {'from', 'gx': [x vars], 'gy': [y vars]} or heads {'type', 'of'}."""
    vals = [frozenset()]
    edges = merges = copies = 0
    errs = []
    for t, st in enumerate(steps, 1):
        if st["type"] == "edge":
            inc = frozenset((xv, yv) for xv in st["gx"] for yv in st["gy"])
            if rank_of(inc) != 1:
                errs.append(f"step {t}: increment rank {rank_of(inc)}")
            if not 0 <= st["from"] < t:
                errs.append(f"step {t}: parent")
            vals.append(sym_diff_all([vals[st["from"]], inc]))
            edges += 1
        else:
            of = st["of"]
            lim = arity if st["type"] == "merge" else 2
            if not (2 <= len(of) <= lim and len(set(of)) == len(of) and all(1 <= s < t for s in of)):
                errs.append(f"step {t}: operands {of}")
            vals.append(sym_diff_all(vals[s] for s in of))
            if st["type"] == "merge":
                merges += 1
            else:
                copies += 1
    have = set(vals)
    missing = [k for k, T in terminals(a, b, c, L).items() if T not in have]
    if missing:
        errs.append(f"missing terminals {missing}")
    return {"ok": not errs, "errors": errs, "edges": edges, "merges": merges, "copies": copies,
            "naive": a * b * c, "savings": a * b * c - edges - copies}


def from_sat(res: dict) -> list:
    a, b, c, L = res["shape"]
    out = []
    for st in res["steps"]:
        if st["type"] == "edge":
            out.append({"type": "edge", "from": st["from"], "gx": vec_x(st["u"], a, b), "gy": vec_y(st["w"], b, c)})
        else:
            out.append({"type": st["type"], "of": list(st["of"])})
    return out


def lift(steps: list, b: int, m: int) -> list:
    """Each edge becomes a chain of m edges (one per inner block); node indices are renumbered."""
    out, where = [], [0]
    for st in steps:
        if st["type"] == "edge":
            prev = where[st["from"]]
            for i in range(m):
                gx = [("x", v[1], v[2] + i * b) for v in st["gx"]]
                gy = [("y", v[1] + i * b, v[2]) for v in st["gy"]]
                out.append({"type": "edge", "from": prev, "gx": gx, "gy": gy})
                prev = len(out)
            where.append(prev)
        else:
            out.append({"type": st["type"], "of": [where[s] for s in st["of"]]})
            where.append(len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("file")
    p = sub.add_parser("lift")
    p.add_argument("file")
    p.add_argument("--m", default="2,3,4")
    args = ap.parse_args()
    for line in open(args.file):
        res = json.loads(line)
        if res.get("status") != "sat" or "steps" not in res:  # a cube group's summary line has no steps
            continue
        a, b, c, L = res["shape"]
        steps = from_sat(res)
        tag = f"<{a},{b},{c}> L={L} E={res['E']} M={res['M']} K={res['K']} arity={res['arity']}"
        if args.cmd == "check":
            print(tag, json.dumps(check(a, b, c, L, steps, res["arity"])))
        elif L == b:
            for m in [int(x) for x in args.m.split(",")]:
                r = check(a, b * m, c, L * m, lift(steps, b, m), res["arity"])
                print(f"{tag} lifted x{m} -> <{a},{b * m},{c}> L={L * m}", json.dumps(r))


if __name__ == "__main__":
    main()
