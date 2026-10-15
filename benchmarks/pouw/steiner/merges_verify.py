"""Independent check of a merge/copy computation (merges_sat.py's JSON steps), sharing no code with merges_sat or steiner.

A form is a set of monomials (('x', r, k), ('y', k, j)); a step is replayed on those sets, the rank of every edge
increment is computed by Gaussian elimination on its own matrix, and the terminals are rebuilt from the definition
T[(r, j), s] = sum_{k < s L} x[r][k] y[k][j]. Also `lift`: a computation for <a,b,c> at L becomes one for <a,bm,c> at
L m by replacing each inner index k with m k + i for i = 0..m-1 (the prefix k < sL becomes k < smL): an edge u (x) w
becomes a chain of m edges (u (x) w on each copy; each has rank 1), and a merge or copy stays one step, since the lift of
a sum is the sum of the lifts. Same merges and copies, m times the edges.

`python3 merges_verify.py check FILE.jsonl` re-checks every sat line; `lift FILE.jsonl --m 2,3,...` lifts every sat
line and checks the result; `prefix --levels 1,2,... --m 1,2,...` builds `prefix_222` (<2,2n,2> at L = 2 with n
levels, 7n edges, 4n - 3 merges), lifts it and checks it.
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


def lift(steps: list, m: int) -> list:
    """Each edge becomes a chain of m edges, inner index k -> m k + i for i < m (so the prefix k < sL becomes
    k < smL); node indices are renumbered."""
    out, where = [], [0]
    for st in steps:
        if st["type"] == "edge":
            prev = where[st["from"]]
            for i in range(m):
                gx = [("x", v[1], m * v[2] + i) for v in st["gx"]]
                gy = [("y", m * v[1] + i, v[2]) for v in st["gy"]]
                out.append({"type": "edge", "from": prev, "gx": gx, "gy": gy})
                prev = len(out)
            where.append(prev)
        else:
            out.append({"type": st["type"], "of": [where[s] for s in st["of"]]})
            where.append(len(out))
    return out


def _e(frm, xs, ys):
    return {"type": "edge", "from": frm, "gx": [("x", r, k) for r, k in xs], "gy": [("y", k, j) for k, j in ys]}


#: <2,2,2> at L = 2 in 7 edges and one 3-summand merge; node 2 = T00, 5 = T10, 6 = T11, 8 = T01
ONE_MERGE_222 = [
    _e(0, [(0, 0)], [(0, 0)]),
    _e(1, [(0, 1)], [(1, 0)]),
    _e(1, [(0, 0), (1, 0)], [(0, 1), (1, 1)]),
    _e(3, [(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)]),
    _e(4, [(1, 1)], [(0, 0), (0, 1), (1, 0), (1, 1)]),
    _e(4, [(1, 0), (1, 1)], [(0, 0), (0, 1)]),
    {"type": "merge", "of": [1, 3, 6]},
    _e(7, [(0, 0), (0, 1), (1, 0), (1, 1)], [(1, 1)]),
]
ONE_MERGE_222_AT = {(0, 0): 2, (1, 0): 5, (1, 1): 6, (0, 1): 8}


def prefix_222(levels: int) -> list:
    """<2,2n,2> at L = 2 with n levels: level 1 is ONE_MERGE_222; level s >= 2 is ONE_MERGE_222 on inner indices
    2s-2, 2s-1 rooted at T[(0,0), s-1], which reaches T[(0,0), s] itself and R + D_p for the other positions p
    (R = T[(0,0), s-1], D_p level s's part of T[p, s]), and the merge (R + D_p) + R + T[p, s-1] = T[p, s] finishes
    each. 7n edges and 4n - 3 merges against 8n."""
    out: list = []
    at = {}
    for s in range(1, levels + 1):
        sh = 2 * (s - 1)
        root = 0 if s == 1 else at[(0, 0)]
        base = len(out)
        idx = lambda t: root if t == 0 else base + t  # noqa: E731
        for st in ONE_MERGE_222:
            if st["type"] == "edge":
                out.append({"type": "edge", "from": idx(st["from"]),
                            "gx": [("x", v[1], v[2] + sh) for v in st["gx"]],
                            "gy": [("y", v[1] + sh, v[2]) for v in st["gy"]]})
            else:
                out.append({"type": st["type"], "of": [idx(t) for t in st["of"]]})
        new = {p: base + t for p, t in ONE_MERGE_222_AT.items()}
        if s > 1:
            for p in ((1, 0), (1, 1), (0, 1)):
                out.append({"type": "merge", "of": [new[p], root, at[p]]})
                new[p] = len(out)
        at = new
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("file")
    p = sub.add_parser("lift")
    p.add_argument("file")
    p.add_argument("--m", default="2,3,4")
    p = sub.add_parser("prefix")
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--m", default="1,2,4,8")
    args = ap.parse_args()
    if args.cmd == "prefix":
        for n in [int(x) for x in args.levels.split(",")]:
            steps = prefix_222(n)
            for m in [int(x) for x in args.m.split(",")]:
                r = check(2, 2 * n * m, 2, 2 * m, lift(steps, m) if m > 1 else steps)
                print(f"prefix <2,{2 * n * m},2> L={2 * m} levels={n}", json.dumps(r))
        return
    for line in open(args.file):
        res = json.loads(line)
        if res.get("status") != "sat" or "steps" not in res:  # a cube group's summary line has no steps
            continue
        a, b, c, L = res["shape"]
        steps = from_sat(res)
        tag = f"<{a},{b},{c}> L={L} E={res['E']} M={res['M']} K={res['K']} arity={res['arity']}"
        if args.cmd == "check":
            print(tag, json.dumps(check(a, b, c, L, steps, res["arity"])))
        else:
            for m in [int(x) for x in args.m.split(",")]:
                r = check(a, b * m, c, L * m, lift(steps, m), res["arity"])
                print(f"{tag} lifted x{m} -> <{a},{b * m},{c}> L={L * m}", json.dumps(r))


if __name__ == "__main__":
    main()
