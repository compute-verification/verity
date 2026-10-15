"""Searches for counterexamples to Conjecture A (Steiner rigidity of matmul prefix targets), driving dw_f2 / dw_f3.

A bilinear form over X (a x b) and Y (b x c) is an R x C matrix, R = a*b X-variables by C = b*c Y-variables, with
x[r][k] at row r*b + k and y[k][j] at column k*c + j; over F2 it is an int whose bit i*C + j is entry (i, j). The targets
with prefix stride L are T[(r, j), s] = sum_{k < s*L} x[r][k] y[k][j], s = 1..b/L; the naive tree walks each position's
chain from 0 and has length a*b*c. dw_f2/dw_f3 compute exact Dreyfus-Wagner Steiner lengths (on the whole space, or
inside a subspace S, an upper bound); this script builds their specs, compares every terminal subset's length with its
naive length, and checks each returned tree independently: every edge has rank 1, the edges connect 0 to every
terminal, and their number is the reported length.

Pure stdlib. `python3 steiner.py --help` lists the subcommands; each writes JSON lines to --out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- F2 linear algebra on ints


def rows_of(m: int, R: int, C: int) -> list[int]:
    mask = (1 << C) - 1
    return [(m >> (i * C)) & mask for i in range(R)]


def rank_rows(rows: list[int]) -> int:
    piv: list[int] = []
    for x in rows:
        for p in piv:
            x = min(x, x ^ p)
        if x:
            piv.append(x)
            piv.sort(reverse=True)
    return len(piv)


def rank_f2(m: int, R: int, C: int) -> int:
    return rank_rows(rows_of(m, R, C))


def col_space(m: int, R: int, C: int) -> list[int]:
    """Basis of the column space (vectors in F2^R) as ints."""
    cols = []
    for j in range(C):
        v = 0
        for i in range(R):
            if (m >> (i * C + j)) & 1:
                v |= 1 << i
        cols.append(v)
    return reduce_basis(cols)


def row_space(m: int, R: int, C: int) -> list[int]:
    return reduce_basis(rows_of(m, R, C))


def reduce_basis(vs: list[int]) -> list[int]:
    piv: list[int] = []
    for x in vs:
        for p in piv:
            x = min(x, x ^ p)
        if x:
            piv.append(x)
            piv.sort(reverse=True)
    return piv


def trivially_intersect(A: list[int], B: list[int]) -> bool:
    return len(reduce_basis(A + B)) == len(A) + len(B)


def outer(u: int, w: int, R: int, C: int) -> int:
    m = 0
    for i in range(R):
        if (u >> i) & 1:
            m |= w << (i * C)
    return m


# ---------------------------------------------------------------- matmul targets


def matmul_targets(a: int, b: int, c: int, L: int) -> tuple[int, int, list[dict]]:
    assert b % L == 0
    R, C = a * b, b * c
    out = []
    for r in range(a):
        for j in range(c):
            m = 0
            for s in range(1, b // L + 1):
                for k in range((s - 1) * L, s * L):
                    m |= 1 << ((r * b + k) * C + (k * c + j))
                out.append({"m": m, "label": f"r{r}c{j}s{s}", "chain": f"r{r}c{j}", "len": s * L})
    return R, C, out


def naive_by_mask(terms: list[dict]) -> list[int]:
    """Length of the naive tree for {0} + subset: per chain, the length to its deepest member in the subset."""
    n = len(terms)
    res = []
    for S in range(1 << n):
        best: dict[str, int] = {}
        for i in range(n):
            if (S >> i) & 1:
                t = terms[i]
                best[t["chain"]] = max(best.get(t["chain"], 0), t["len"])
        res.append(sum(best.values()))
    return res


# ---------------------------------------------------------------- specs and the solver


def spec_problem(name: str, R: int, C: int, terms: list[dict], basis: list[int] | None = None) -> str:
    lines = [f"P {name} {R} {C} {len(basis) if basis else 0} {len(terms)}"]
    for v in basis or []:
        lines.append(f"B {v:x}")
    for t in terms:
        lines.append(f"T {t['m']:x} {t['label']}")
    return "\n".join(lines) + "\n"


def build(binary: str) -> str:
    exe = os.path.join(os.environ.get("STEINER_BIN", HERE), binary)
    if not os.path.exists(exe):
        src = os.path.join(HERE, binary + ".c")
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", exe, src], check=True)
    return exe


def run_solver(spec: str, batch: bool = False, binary: str = "dw_f2") -> list[dict]:
    exe = build(binary)
    with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        args = [exe, path] + (["--batch"] if batch else [])
        out = subprocess.run(args, check=True, capture_output=True, text=True)
    finally:
        os.unlink(path)
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def verify_tree_f2(res: dict, R: int, C: int) -> dict:
    """Independent check of a returned tree: edge ranks, connectivity from 0, terminals covered, edge count."""
    edges = {tuple(sorted((int(a, 16), int(b, 16)))) for a, b in res["tree"]}
    bad = [e for e in edges if rank_f2(e[0] ^ e[1], R, C) != 1]
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, todo = {0}, [0]
    while todo:
        v = todo.pop()
        for w in adj.get(v, []):
            if w not in seen:
                seen.add(w)
                todo.append(w)
    terms = [int(t[0], 16) for t in res["terminals"]]
    missing = [hex(t) for t in terms if t not in seen]
    ok = not bad and not missing and res["tree_err"] == 0 and len(edges) == res["answer"]
    return {"ok": ok, "edges": len(edges), "bad_edges": len(bad), "missing": missing, "nodes": len(seen)}


def analyze(res: dict, terms: list[dict], R: int, C: int) -> dict:
    naive = naive_by_mask(terms)
    n = len(terms)
    below = [S for S in range(1, 1 << n) if res["root_dp"][S] < naive[S]]
    by_size: dict[int, list[int]] = {}
    for S in range(1, 1 << n):
        by_size.setdefault(bin(S).count("1"), []).append(res["root_dp"][S] - naive[S])
    v = verify_tree_f2(res, R, C) if res.get("field", 2) == 2 else None
    return {
        "naive": naive[-1],
        "answer": res["answer"],
        "beats_naive": res["answer"] < naive[-1],
        "subsets_below_naive": [[S, res["root_dp"][S], naive[S]] for S in below],
        "min_excess_by_size": {k: min(v) for k, v in sorted(by_size.items())},
        "tree_check": v,
    }


def emit(out, rec: dict) -> None:
    line = json.dumps(rec, separators=(",", ":"))
    print(line[:600] + ("..." if len(line) > 600 else ""), flush=True)
    if out:
        out.write(line + "\n")
        out.flush()


# ---------------------------------------------------------------- subcommands


def cmd_validate(args, out) -> None:
    R, C, terms = matmul_targets(2, 2, 2, 2)
    cases = [("222_L2_all", terms), ("222_L2_same_row", [terms[0], terms[1]]), ("222_L2_disjoint", [terms[0], terms[3]])]
    expect = {"222_L2_all": 8, "222_L2_same_row": 4, "222_L2_disjoint": 4}
    for name, ts in cases:
        (res,) = run_solver(spec_problem(name, R, C, ts))
        a = analyze(res, ts, R, C)
        emit(out, {"case": name, "expect": expect[name], "pass": res["answer"] == expect[name] and a["tree_check"]["ok"],
                   "rank1": res["rank1_in_S"], "wall_s": res["wall_s"], **a})
    for shape, want in (((2, 2, 2), 225), ((2, 2, 3), 945)):
        R, C, ts = matmul_targets(*shape, shape[1])
        n_r1 = sum(1 for u in range(1, 1 << R) for w in range(1, 1 << C))
        emit(out, {"case": f"rank1_count_{shape}", "R": R, "C": C, "count": n_r1, "expect": want, "pass": n_r1 == want})


def span_elements(basis: list[int]) -> list[int]:
    els = [0]
    for v in basis:
        els += [e ^ v for e in els]
    return els


def brute_steiner(R: int, C: int, terms: list[int], basis: list[int] | None) -> int:
    """Exact Steiner length by brute force: every set of at most n-2 Steiner points, MST in the metric closure."""
    states = span_elements(basis) if basis else list(range(1 << (R * C)))
    sset = set(states)
    gens = [g for g in states if g and rank_f2(g, R, C) == 1]
    index = {s: i for i, s in enumerate(states)}

    def bfs(src: int) -> list[int]:
        dist = [99] * len(states)
        dist[index[src]] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for v in frontier:
                for g in gens:
                    w = v ^ g
                    if w in sset and dist[index[w]] == 99:
                        dist[index[w]] = dist[index[v]] + 1
                        nxt.append(w)
            frontier = nxt
        return dist

    nodes = [0] + terms
    D = {t: bfs(t) for t in nodes}

    def mst(pts: list[int]) -> int:
        inside, best, total = [pts[0]], {p: D[pts[0]][index[p]] for p in pts[1:]}, 0
        while best:
            p = min(best, key=best.get)
            total += best.pop(p)
            for q in best:
                best[q] = min(best[q], D[p][index[q]] if p in D else bfs(p)[index[q]])
        return total

    others = [s for s in states if s not in nodes]
    best = mst(nodes)
    for npts in range(1, len(nodes) - 1):
        for P in itertools.combinations(others, npts):
            for p in P:
                if p not in D:
                    D[p] = bfs(p)
            best = min(best, mst(nodes + list(P)))
    return best


def cmd_selftest(args, out) -> None:
    """dw_f2 against brute force on random tiny instances, on the whole space and in random subspaces."""
    rng = random.Random(args.seed)
    specs, cases = [], []
    for i in range(args.cases):
        R, C = rng.choice([(2, 2), (2, 3), (3, 2)])
        full = i % 2 == 0
        basis = None
        if not full:
            k = rng.randint(3, R * C - 1)
            while True:
                basis = reduce_basis([rng.randrange(1, 1 << (R * C)) for _ in range(k)])
                if len(basis) == k:
                    break
        pool = span_elements(basis) if basis else list(range(1 << (R * C)))
        n = rng.randint(2, 4)
        terms = rng.sample([p for p in pool if p], n)
        cases.append((R, C, terms, basis))
        specs.append(spec_problem(f"st{i}", R, C, [{"m": t, "label": f"t{j}"} for j, t in enumerate(terms)], basis))
    results = run_solver("".join(specs), batch=True)
    by_name = {r["name"]: r for r in results}
    fails = 0
    for i, (R, C, terms, basis) in enumerate(cases):
        r = by_name[f"st{i}"]
        want = min(brute_steiner(R, C, terms, basis), 99)
        got = r["answer"] if r["answer"] < 120 else 99
        v = verify_tree_f2(r, R, C) if got < 99 else {"ok": True}
        ok = got == want and v["ok"]
        fails += not ok
        if not ok:
            emit(out, {"case": f"st{i}", "R": R, "C": C, "terms": terms, "basis": basis, "dw": got, "brute": want, "tree": v})
    finite = sum(1 for r in results if r["answer"] < 120)
    emit(out, {"case": "selftest", "cases": args.cases, "finite": finite, "fails": fails, "pass": fails == 0})


def cmd_matmul(args, out) -> None:
    R, C, terms = matmul_targets(args.a, args.b, args.c, args.L)
    name = f"mm{args.a}{args.b}{args.c}_L{args.L}_F2"
    (res,) = run_solver(spec_problem(name, R, C, terms))
    a = analyze(res, terms, R, C)
    res_small = {k: v for k, v in res.items() if k not in ("root_dp",)}
    emit(out, {"case": name, "shape": [args.a, args.b, args.c], "L": args.L, **a, "solver": res_small,
               "root_dp": res["root_dp"], "labels": [t["label"] for t in terms]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="JSON lines file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("selftest")
    p.add_argument("--cases", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("matmul")
    for k in ("a", "b", "c", "L"):
        p.add_argument(f"--{k}", type=int, required=True)
    args = ap.parse_args()
    out = open(args.out, "a") if args.out else None
    {"validate": cmd_validate, "selftest": cmd_selftest, "matmul": cmd_matmul}[args.cmd](args, out)


if __name__ == "__main__":
    main()
