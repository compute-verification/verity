"""Searches for counterexamples to Conjecture A (Steiner rigidity of matmul prefix targets), driving dw_f2 / dw_f3 /
dw_full / sa.

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


def prefix_closed(S: int, terms: list[dict]) -> bool:
    """Whether S holds, for each chain, its shallowest members only (listed shallow to deep): then a tree for S extends
    along the chains to every terminal at exactly the naive extra cost, so beating naive on S beats it on all."""
    seen_gap: dict[str, bool] = {}
    for i, t in enumerate(terms):
        inside = bool((S >> i) & 1)
        if inside and seen_gap.get(t["chain"]):
            return False
        if not inside:
            seen_gap[t["chain"]] = True
    return True


def naive_full(terms: list[dict]) -> int:
    best: dict[str, int] = {}
    for t in terms:
        best[t["chain"]] = max(best.get(t["chain"], 0), t["len"])
    return sum(best.values())


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
        out = subprocess.run(args, check=True, stdout=subprocess.PIPE, text=True)
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
        "subsets_checked": (1 << n) - 1,
        "prefix_closed_subsets": sum(prefix_closed(S, terms) for S in range(1, 1 << n)),
        "subsets_below_naive": [[S, res["root_dp"][S], naive[S], prefix_closed(S, terms)] for S in below],
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


# ---------------------------------------------------------------- F3 (matrices as tuples of trits, entry (i,j) at i*C+j)


def rank_f3(t: tuple[int, ...], R: int, C: int) -> int:
    rows = [[t[i * C + j] % 3 for j in range(C)] for i in range(R)]
    rank, col = 0, 0
    for col in range(C):
        piv = next((r for r in range(rank, R) if rows[r][col]), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = rows[rank][col]  # 1 and 2 are their own inverses mod 3
        rows[rank] = [(x * inv) % 3 for x in rows[rank]]
        for r in range(R):
            if r != rank and rows[r][col]:
                f = rows[r][col]
                rows[r] = [(x - f * y) % 3 for x, y in zip(rows[r], rows[rank])]
        rank += 1
    return rank


def tsub(a: str, b: str) -> tuple[int, ...]:
    return tuple((int(x) - int(y)) % 3 for x, y in zip(a, b))


def matmul_targets_f3(a: int, b: int, c: int, L: int) -> tuple[int, int, list[dict]]:
    R, C, terms = matmul_targets(a, b, c, L)
    for t in terms:
        t["t"] = "".join("1" if (t["m"] >> p) & 1 else "0" for p in range(R * C))
    return R, C, terms


def spec_problem_f3(name: str, R: int, C: int, terms: list[dict]) -> str:
    return f"P {name} {R} {C} {len(terms)}\n" + "".join(f"T {t['t']} {t['label']}\n" for t in terms)


def verify_tree_f3(res: dict, R: int, C: int) -> dict:
    edges = {tuple(sorted(e)) for e in res["tree"]}
    bad = [e for e in edges if rank_f3(tsub(e[0], e[1]), R, C) != 1]
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    zero = "0" * (R * C)
    seen, todo = {zero}, [zero]
    while todo:
        v = todo.pop()
        for w in adj.get(v, []):
            if w not in seen:
                seen.add(w)
                todo.append(w)
    missing = [t[0] for t in res["terminals"] if t[0] not in seen]
    ok = not bad and not missing and res["tree_err"] == 0 and len(edges) == res["answer"]
    return {"ok": ok, "edges": len(edges), "bad_edges": len(bad), "missing": missing, "nodes": len(seen)}


def brute_steiner_f3(R: int, C: int, terms: list[str]) -> int:
    n = R * C
    states = ["".join(str((x // 3**p) % 3) for p in range(n)) for x in range(3**n)]
    dist = {}

    def d(a: str, b: str) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in dist:
            dist[key] = rank_f3(tsub(a, b), R, C)
        return dist[key]

    nodes = ["0" * n] + terms

    def mst(pts: list[str]) -> int:
        best, total = {p: d(pts[0], p) for p in pts[1:]}, 0
        while best:
            p = min(best, key=best.get)
            total += best.pop(p)
            for q in best:
                best[q] = min(best[q], d(p, q))
        return total

    others = [s for s in states if s not in nodes]
    best = mst(nodes)
    for k in range(1, len(nodes) - 1):
        for P in itertools.combinations(others, k):
            best = min(best, mst(nodes + list(P)))
    return best


def cmd_selftest3(args, out) -> None:
    rng = random.Random(args.seed)
    fails = 0
    for i in range(args.cases):
        R, C = rng.choice([(2, 2), (2, 2), (1, 3), (2, 3)])
        n = rng.randint(2, 3 if R * C == 6 else 4)
        pool = ["".join(str((x // 3**p) % 3) for p in range(R * C)) for x in range(1, 3 ** (R * C))]
        terms = rng.sample(pool, n)
        (r,) = run_solver(spec_problem_f3(f"s3_{i}", R, C, [{"t": t, "label": f"t{j}"} for j, t in enumerate(terms)]),
                          binary="dw_f3")
        want = brute_steiner_f3(R, C, terms)
        v = verify_tree_f3(r, R, C)
        ok = r["answer"] == want and v["ok"]
        fails += not ok
        if not ok:
            emit(out, {"case": f"s3_{i}", "R": R, "C": C, "terms": terms, "dw": r["answer"], "brute": want, "tree": v})
    emit(out, {"case": "selftest3", "cases": args.cases, "fails": fails, "pass": fails == 0})


def cmd_matmul3(args, out) -> None:
    R, C, terms = matmul_targets_f3(args.a, args.b, args.c, args.L)
    name = f"mm{args.a}{args.b}{args.c}_L{args.L}_F3"
    (res,) = run_solver(spec_problem_f3(name, R, C, terms), binary="dw_f3")
    naive = naive_by_mask(terms)
    n = len(terms)
    below = [[S, res["root_dp"][S], naive[S]] for S in range(1, 1 << n) if res["root_dp"][S] < naive[S]]
    res_small = {k: v for k, v in res.items() if k != "root_dp"}
    emit(out, {"case": name, "shape": [args.a, args.b, args.c], "L": args.L, "naive": naive[-1], "answer": res["answer"],
               "beats_naive": res["answer"] < naive[-1], "subsets_below_naive": below,
               "tree_check": verify_tree_f3(res, R, C), "solver": res_small, "root_dp": res["root_dp"],
               "labels": [t["label"] for t in terms]})


# ---------------------------------------------------------------- task 4: the general version


def random_independent(rng: random.Random, dim: int, r: int, within: list[int] | None = None) -> list[int]:
    """r random independent vectors of F2^dim, inside span(within) when given."""
    while True:
        if within:
            vs = [0] * r
            for i in range(r):
                for w in within:
                    if rng.getrandbits(1):
                        vs[i] ^= w
        else:
            vs = [rng.randrange(1, 1 << dim) for _ in range(r)]
        if len(reduce_basis(vs)) == r:
            return vs


def chain_terms(name: str, a: list[int], b: list[int], cuts: list[int], R: int, C: int) -> list[dict]:
    """Geodesic chain sum_{i < cut} a_i b_i^T for each cut (a, b independent, so each step adds rank exactly)."""
    out, m, i = [], 0, 0
    for s, cut in enumerate(cuts):
        while i < cut:
            m ^= outer(a[i], b[i], R, C)
            i += 1
        out.append({"m": m, "label": f"{name}s{s + 1}", "chain": name, "len": cut})
    return out


def condition_i(terms: list[dict], R: int, C: int) -> bool:
    last: dict[str, int] = {}
    for t in terms:
        last[t["chain"]] = t["m"]  # chains are listed shallow to deep
    ms = list(last.values())
    for p, q in itertools.combinations(ms, 2):
        if not (trivially_intersect(col_space(p, R, C), col_space(q, R, C))
                or trivially_intersect(row_space(p, R, C), row_space(q, R, C))):
            return False
    return True


def independence(terms: list[dict], R: int, C: int) -> dict:
    """Matmul's extra structure: positions whose row (column) spaces meet trivially have independent ones jointly."""
    last: dict[str, int] = {}
    for t in terms:
        last[t["chain"]] = t["m"]
    ms = list(last.values())
    res = {}
    for side, sp in (("rows", row_space), ("cols", col_space)):
        spaces = [sp(m, R, C) for m in ms]
        worst = None
        for k in range(3, len(ms) + 1):
            for Q in itertools.combinations(range(len(ms)), k):
                if all(trivially_intersect(spaces[i], spaces[j]) for i, j in itertools.combinations(Q, 2)):
                    tot = sum(len(spaces[i]) for i in Q)
                    got = len(reduce_basis([v for i in Q for v in spaces[i]]))
                    if got < tot and worst is None:
                        worst = {"positions": list(Q), "dim_sum": got, "sum_dims": tot}
        res[side] = worst
    return res


def jointly_independent(terms: list[dict], R: int, C: int) -> bool:
    ind = independence(terms, R, C)
    return ind["rows"] is None and ind["cols"] is None


def draw_family(rng: random.Random, R: int, C: int, max_positions: int, max_rank: int) -> tuple[list[dict], dict]:
    """3..max_positions positions, each a geodesic chain (its full form and, half the time, one prefix) in random bases,
    kept only while the family has distinct terminals and (i); (ii) holds by construction. Returns the terminals and
    each placed chain's factors (a, b), the form being sum a_i b_i^T."""
    npos = rng.randint(3, max_positions)
    terms: list[dict] = []
    factors: dict[str, tuple[list[int], list[int]]] = {}
    for p in range(npos):
        for _ in range(50):
            top = rng.randint(1, min(R, C, max_rank))
            cuts = sorted(set([top] + ([rng.randint(1, top - 1)] if top > 1 and rng.random() < 0.5 else [])))
            a, b = random_independent(rng, R, top), random_independent(rng, C, top)
            cand = terms + chain_terms(f"p{p}", a, b, cuts, R, C)
            ms = [t["m"] for t in cand]
            if len(set(ms)) == len(ms) and condition_i(cand, R, C):
                terms = cand
                factors[f"p{p}"] = (a, b)
                break
    return terms, factors


def hub_example() -> tuple[int, int, list[dict]]:
    """Three rank-2 forms on X = F2^2, Y = F2^4, with pairwise trivially intersecting row spaces, each at distance 1 from
    the hub m = e1 f1^T + e2 f2^T: the tree 0 -> m -> each has length 5, under the sum 6."""
    R, C = 2, 4
    f = [1, 2, 4, 8]

    def form(r0: int, r1: int) -> int:
        return r0 | (r1 << C)

    T = [form(f[0] ^ f[2], f[1]), form(f[0], f[1] ^ f[3]), form(f[0] ^ f[2] ^ f[3], f[1] ^ f[2] ^ f[3])]
    return R, C, [{"m": m, "label": f"p{i}", "chain": f"p{i}", "len": 2} for i, m in enumerate(T)]


def inv2(M: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c, d = M
    assert (a * d + b * c) % 2 == 1
    return (d, b, c, a)


def mul2(M, N):
    a, b, c, d = M
    e, f, g, h = N
    return ((a * e + b * g) % 2, (a * f + b * h) % 2, (c * e + d * g) % 2, (c * f + d * h) % 2)


def holonomy_class(blocks: dict) -> str:
    """Conjugacy class in GL2(F2) of T00^-1 T01 T11^-1 T10, the invariant of a 2x2 grid of invertible 2x2 blocks."""
    H = mul2(mul2(inv2(blocks[0, 0]), blocks[0, 1]), mul2(inv2(blocks[1, 1]), blocks[1, 0]))
    if H == (1, 0, 0, 1):
        return "identity"
    return "order2" if mul2(H, H) == (1, 0, 0, 1) else "order3"


def grid_terms(blocks: dict, rdims: list[int], cdims: list[int], chains: dict | None = None) -> tuple[int, int, list[dict]]:
    """Positions (r, c) of a grid X = (+) X_r, Y = (+) Y_c (coordinate blocks); block (r, c) holds a geodesic chain inside
    X_r (x) Y_c, given as (a vectors, b vectors, cuts) in block coordinates, or one full-rank form (2x2 tuple)."""
    R, C = sum(rdims), sum(cdims)
    roff = [sum(rdims[:i]) for i in range(len(rdims))]
    coff = [sum(cdims[:i]) for i in range(len(cdims))]
    terms = []
    for (r, c), spec in sorted((chains or blocks).items()):
        if chains:
            a, b, cuts = spec
            a = [x << roff[r] for x in a]
            b = [y << coff[c] for y in b]
            terms += chain_terms(f"r{r}c{c}", a, b, cuts, R, C)
        else:
            m = 0
            for i in range(2):
                for j in range(2):
                    if spec[2 * i + j]:
                        m |= 1 << ((roff[r] + i) * C + coff[c] + j)
            terms.append({"m": m, "label": f"r{r}c{c}", "chain": f"r{r}c{c}", "len": 2})
    return R, C, terms


def solve_batch(problems: list[tuple[str, int, int, list[dict], dict]], out, kind: str, batch: bool = True) -> dict:
    """Solve every (name, R, C, terms, meta) on the whole space; emit each one below its chains' sum, and a summary."""
    spec = "".join(spec_problem(name, R, C, terms) for name, R, C, terms, _ in problems)
    t0 = time.time()
    results = {r["name"]: r for r in run_solver(spec, batch=batch)}
    below = joint = joint_below = 0
    for name, R, C, terms, meta in problems:
        r = results[name]
        naive = naive_by_mask(terms)
        sub_below = [S for S in range(1, 1 << len(terms)) if r["root_dp"][S] < naive[S]]
        ind = independence(terms, R, C)
        ji = ind["rows"] is None and ind["cols"] is None
        joint += ji
        if r["answer"] < naive[-1] or sub_below:
            below += 1
            joint_below += ji
            best = min(sub_below, key=lambda S: (r["root_dp"][S] - naive[S], bin(S).count("1")))
            emit(out, {"kind": kind, "case": name, "R": R, "C": C, "naive": naive[-1], "answer": r["answer"],
                       "subsets_below": len(sub_below),
                       "prefix_closed_below": sum(prefix_closed(S, terms) for S in sub_below),
                       "best_subset": [best, r["root_dp"][best], naive[best]],
                       "terms": [[hex(t["m"]), t["label"]] for t in terms],
                       "tree": r["tree"], "tree_check": verify_tree_f2(r, R, C), **meta, "independence": ind})
    summary = {"kind": kind, "summary": True, "families": len(problems), "below_naive": below,
               "jointly_independent": joint, "jointly_independent_below": joint_below,
               "wall_s": round(time.time() - t0, 2), "states": 1 << (problems[0][1] * problems[0][2])}
    emit(out, summary)
    return summary


def cmd_general(args, out) -> None:
    rng = random.Random(args.seed)
    # the hand-built family: refutes the general version with three terminals
    R, C, terms = hub_example()
    assert condition_i(terms, R, C)
    solve_batch([("hub_example", R, C, terms, {"note": "three rank-2 forms at distance 1 from a rank-2 hub"})], out,
                "hub_example")
    # random families with (i) and (ii)
    for R, C in [tuple(map(int, s.split("x"))) for s in args.sizes]:
        probs = []
        while len(probs) < args.families:
            terms, factors = draw_family(rng, R, C, args.max_positions, args.max_rank)
            if len(terms) <= args.max_terminals and len(factors) >= 3:
                if args.joint_only and not jointly_independent(terms, R, C):
                    continue
                probs.append((f"rand_{R}x{C}_{len(probs)}", R, C, terms, {}))
        solve_batch(probs, out, f"random_{R}x{C}" + ("_joint" if args.joint_only else ""), batch=R * C <= 20)
    if args.joint_only:
        return
    # block grids: matmul's direct-sum structure with arbitrary block contents (2x2 grid of 2x2 blocks, exhaustive)
    inv = [m for m in itertools.product((0, 1), repeat=4) if (m[0] * m[3] + m[1] * m[2]) % 2]
    probs = []
    for combo in itertools.product(inv, repeat=4):
        blocks = dict(zip([(0, 0), (0, 1), (1, 0), (1, 1)], combo))
        R, C, terms = grid_terms(blocks, [2, 2], [2, 2])
        probs.append((f"grid22_{len(probs)}", R, C, terms, {"holonomy": holonomy_class(blocks)}))
    s = solve_batch(probs, out, "grid2x2_blocks2x2_full")
    # block grids with chains (rank 1 then 2 in each 2x2 block), sampled
    probs = []
    bases = [(a1, a2) for a1 in (1, 2, 3) for a2 in (1, 2, 3) if a1 != a2]
    for i in range(args.grid_chains):
        chains = {(r, c): (list(rng.choice(bases)), list(rng.choice(bases)), [1, 2]) for r in (0, 1) for c in (0, 1)}
        R, C, terms = grid_terms({}, [2, 2], [2, 2], chains)
        probs.append((f"gridch22_{i}", R, C, terms, {}))
    solve_batch(probs, out, "grid2x2_blocks2x2_chains")


def cmd_jointsub(args, out) -> None:
    """Task 4 past the whole space's reach: random jointly independent (i)+(ii) families on larger R x C, each solved by
    DW inside subspaces spanned by its chains' rank-1 pieces a_i b_i^T plus random cross pieces a_i b'_j^T (an X factor
    of one position times a Y factor of another or the same, the shape of the hub's saving). A tree found in a subspace
    is valid in the whole space, so a length under the chains' sum is a counterexample; none found proves nothing.
    --calibrate draws families without joint independence instead, on sizes small enough for the whole space too, and
    counts how many of the whole space's savings the subspaces find."""
    rng = random.Random(args.seed)
    for R, C in [tuple(map(int, s.split("x"))) for s in args.sizes]:
        fams = []
        while len(fams) < args.families:
            terms, factors = draw_family(rng, R, C, args.max_positions, args.max_rank)
            if len(terms) <= args.max_terminals and len(factors) >= 3 and \
                    jointly_independent(terms, R, C) != args.calibrate:
                fams.append((terms, factors))
        probs = []
        for f, (terms, factors) in enumerate(fams):
            own = [outer(x, y, R, C) for a, b in factors.values() for x, y in zip(a, b)]
            xs = [x for a, _ in factors.values() for x in a]
            ys = [y for _, b in factors.values() for y in b]
            cross = list({outer(x, y, R, C) for x in xs for y in ys} - set(own))
            for s in range(args.subspaces):
                rng.shuffle(cross)
                basis = reduce_basis([t["m"] for t in terms] + own)
                for g in cross:
                    if len(basis) >= args.k:
                        break
                    basis = reduce_basis(basis + [g])
                probs.append((f"js_{R}x{C}_{f}_{s}", f, len(basis),
                              spec_problem(f"js_{R}x{C}_{f}_{s}", R, C, terms, basis)))
        t0 = time.time()
        found: dict[int, dict] = {}
        for i in range(0, len(probs), 256):
            chunk = probs[i:i + 256]
            info = {p[0]: p for p in chunk}
            for r in run_solver("".join(p[3] for p in chunk), batch=True):
                _, f, k, _ = info[r["name"]]
                terms = fams[f][0]
                naive = naive_by_mask(terms)
                below = [S for S in range(1, 1 << len(terms)) if r["root_dp"][S] < naive[S]]
                if below and f not in found:
                    best = min(below, key=lambda S: (r["root_dp"][S] - naive[S], bin(S).count("1")))
                    found[f] = {"case": r["name"], "k": k, "answer": r["answer"], "naive": naive[-1],
                                "best_subset": [best, r["root_dp"][best], naive[best]],
                                "terms": [[hex(t["m"]), t["label"]] for t in terms], "tree": r["tree"],
                                "tree_check": verify_tree_f2(r, R, C)}
        kind = f"jointsub_{R}x{C}" + ("_calibrate" if args.calibrate else "")
        for f, rec in sorted(found.items()):
            emit(out, {"kind": kind, **rec, "independence": independence(fams[f][0], R, C)})
        summary = {"kind": kind, "summary": True, "families": len(fams), "subspaces": len(probs),
                   "k_min": min(p[2] for p in probs), "k_max": max(p[2] for p in probs),
                   "subspace_below": len(found), "wall_s": round(time.time() - t0, 2)}
        if args.calibrate:
            whole = run_solver("".join(spec_problem(f"w{f}", R, C, t) for f, (t, _) in enumerate(fams)), batch=True)
            wb = {int(r["name"][1:]) for r in whole
                  if any(r["root_dp"][S] < m for S, m in enumerate(naive_by_mask(fams[int(r["name"][1:])][0])) if S)}
            assert set(found) <= wb, "a subspace tree beat the whole space's minimum"
            summary.update({"whole_below": len(wb), "found_of_whole_below": len(set(found) & wb)})
        emit(out, summary)


def cmd_aligned(args, out) -> None:
    """Families with (i) and (ii) whose positions' column and row spaces are coordinate subspaces (spanned by sets of
    X- and Y-coordinates; the chain inside uses random bases of them). Then column (row) spaces that meet pairwise
    trivially are jointly independent, as in matmul, but the supports need not form a grid."""
    rng = random.Random(args.seed)
    for R, C in [tuple(map(int, s.split("x"))) for s in args.sizes]:
        probs = []
        tries = 0
        while len(probs) < args.families and tries < 100 * args.families:
            tries += 1
            npos = rng.randint(3, args.max_positions)
            terms: list[dict] = []
            supports = []
            for p in range(npos):
                for _ in range(50):
                    top = rng.randint(1, min(R, C, 3))
                    S = rng.sample(range(R), top)
                    T = rng.sample(range(C), top)
                    a = random_independent(rng, R, top, [1 << i for i in S])
                    b = random_independent(rng, C, top, [1 << j for j in T])
                    cuts = sorted(set([top] + ([rng.randint(1, top - 1)] if top > 1 and rng.random() < 0.5 else [])))
                    cand = terms + chain_terms(f"p{p}", a, b, cuts, R, C)
                    last = cand[-1]["m"]
                    cs, rs = col_space(last, R, C), row_space(last, R, C)
                    sup_c = {i for v in cs for i in range(R) if v >> i & 1}
                    sup_r = {j for v in rs for j in range(C) if v >> j & 1}
                    ok = all(not (sup_c & qc) or not (sup_r & qr) for qc, qr in supports)
                    if ok and len({t["m"] for t in cand}) == len(cand):
                        terms = cand
                        supports.append((sup_c, sup_r))
                        break
            if len(terms) <= args.max_terminals and len({t["chain"] for t in terms}) >= 3:
                grid = is_grid(supports)
                probs.append((f"al_{R}x{C}_{len(probs)}", R, C, terms, {"grid": grid}))
        s = solve_batch(probs, out, f"aligned_{R}x{C}", batch=R * C <= 20)
        emit(out, {"kind": f"aligned_{R}x{C}", "grid_families": sum(p[4]["grid"] for p in probs)})


def is_grid(supports: list[tuple[set, set]]) -> bool:
    """Whether the X supports are pairwise equal or disjoint, and the Y supports too (positions on a grid)."""
    for i, j in itertools.combinations(range(len(supports)), 2):
        for side in (0, 1):
            A, B = supports[i][side], supports[j][side]
            if A & B and A != B:
                return False
    return True


def cmd_grid(args, out) -> None:
    """Block grids too large for the local machine: 2x3 grids of 2x2 blocks (R=4, C=6), sampled."""
    rng = random.Random(args.seed)
    inv = [m for m in itertools.product((0, 1), repeat=4) if (m[0] * m[3] + m[1] * m[2]) % 2]
    probs = []
    seen = set()
    for i in range(args.families):
        blocks = {(r, c): rng.choice(inv) for r in (0, 1) for c in (0, 1, 2)}
        if i == 0:
            blocks = {k: (1, 0, 0, 1) for k in blocks}  # matmul <2,2,3> itself
        key = tuple(sorted(blocks.items()))
        if key in seen:
            continue
        seen.add(key)
        R, C, terms = grid_terms(blocks, [2, 2], [2, 2, 2])
        hol = [holonomy_class({(0, 0): blocks[0, a], (0, 1): blocks[0, b], (1, 0): blocks[1, a], (1, 1): blocks[1, b]})
               for a, b in ((0, 1), (0, 2), (1, 2))]
        probs.append((f"grid23_{i}", R, C, terms, {"holonomies": hol}))
    solve_batch(probs, out, "grid2x3_blocks2x2_full", batch=False)


# ---------------------------------------------------------------- task 3: subspace-restricted search on <a,b,c> with stride L

STRASSEN = [  # (U, V) over 2x2 A and B as (11, 12, 21, 22); over F2 every sign is +
    ((1, 0, 0, 1), (1, 0, 0, 1)), ((0, 0, 1, 1), (1, 0, 0, 0)), ((1, 0, 0, 0), (0, 1, 0, 1)), ((0, 0, 0, 1), (1, 0, 1, 0)),
    ((1, 1, 0, 0), (0, 0, 0, 1)), ((1, 0, 1, 0), (1, 1, 0, 0)), ((0, 1, 0, 1), (0, 0, 1, 1)),
]
GL2 = [m for m in itertools.product((0, 1), repeat=4) if (m[0] * m[3] + m[1] * m[2]) % 2]


def tr2(M):
    return (M[0], M[2], M[1], M[3])


def strassen_orbit() -> list[list[tuple]]:
    """Strassen's products under (A, B, C) in GL2(F2)^3 (X -> AXB, Y -> B^-1 Y C): each a rank-7 scheme for <2,2,2>."""
    seen, out = set(), []
    for A in GL2:
        for B in GL2:
            for Cm in GL2:
                Bi = inv2(B)
                prods = frozenset((mul2(mul2(tr2(A), U), tr2(B)), mul2(mul2(tr2(Bi), V), tr2(Cm))) for U, V in STRASSEN)
                if prods not in seen:
                    seen.add(prods)
                    out.append(sorted(prods))
    return out


def block_products(scheme, a: int, b: int, c: int, rows: tuple, ks: tuple, cols: tuple) -> list[int]:
    """A <2,2,2> scheme on rows (r0, r1) of X, inner indices (k0, k1) and columns (c0, c1) of Y, as R x C forms."""
    R, C = a * b, b * c
    ms = []
    for U, V in scheme:
        u = sum(1 << (rows[i] * b + ks[j]) for i in range(2) for j in range(2) if U[2 * i + j])
        v = sum(1 << (ks[i] * c + cols[j]) for i in range(2) for j in range(2) if V[2 * i + j])
        ms.append(outer(u, v, R, C))
    return ms


def x_factor(m: int, R: int, C: int) -> int:
    """u of a rank-1 form u v^T over F2 (the rows where it is nonzero)."""
    return sum(1 << i for i, x in enumerate(rows_of(m, R, C)) if x)


def y_factor(m: int, R: int, C: int) -> int:
    return next(x for x in rows_of(m, R, C) if x)


def standard_pieces(a: int, b: int, c: int) -> list[int]:
    R, C = a * b, b * c
    return [1 << ((r * b + k) * C + (k * c + j)) for r in range(a) for j in range(c) for k in range(b)]


def random_rank1(rng: random.Random, R: int, C: int, a: int, b: int, c: int, how: str) -> int:
    if how == "dense":
        return outer(rng.randrange(1, 1 << R), rng.randrange(1, 1 << C), R, C)
    r, j = rng.randrange(a), rng.randrange(c)
    rowmask = sum(1 << (r * b + k) for k in range(b))
    colmask = sum(1 << (k * c + j) for k in range(b))
    while True:
        u = rng.randrange(1, 1 << R) & (rowmask if how in ("aligned", "row") else (1 << R) - 1)
        v = rng.randrange(1, 1 << C) & (colmask if how in ("aligned", "col") else (1 << C) - 1)
        if u and v:
            return outer(u, v, R, C)


def subspace_problem(name: str, R: int, C: int, terms: list[dict], gens: list[int]) -> tuple[str, int] | None:
    basis = reduce_basis([t["m"] for t in terms] + gens)
    return spec_problem(name, R, C, terms, basis), len(basis)


def cmd_subspace(args, out) -> None:
    """Exact DW inside many subspaces S containing the targets: each tree found is valid in the whole space, so a length
    under naive anywhere is a counterexample. A counterexample of length l lies in the span of its l rank-1 edges."""
    a, b, c, L = args.a, args.b, args.c, args.L
    R, C, terms = matmul_targets(a, b, c, L)
    naive_mask = naive_by_mask(terms)
    naive = naive_mask[-1]
    rng = random.Random(args.seed)
    std = standard_pieces(a, b, c)
    orbit = strassen_orbit()
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    emit(out, {"kind": "strassen_orbit", "schemes": len(orbit)})
    if b == 4:
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    elif b == 2:
        pairings = [((0, 1),)]
    else:
        pairings = [((0, 1),), ((0, 2),), ((1, 2),)]
    rowpairs = list(itertools.combinations(range(a), 2))
    colpairs = list(itertools.combinations(range(c), 2))
    deadline = time.time() + args.budget_s
    best: dict = {"answer": 999}
    stats: dict[str, dict] = {}

    def n_for(k: int) -> int:
        return max(4, int(args.samples * 2 ** min(0, args.batch_k - k)))

    def run_family(family: str, items: list[tuple[str, list[int]]]) -> None:
        nonlocal best
        probs = []
        for nm, gens in items:
            sp, k = subspace_problem(nm, R, C, terms, gens)
            if k <= args.max_k:
                probs.append((nm, sp, k, gens))
        st = stats.setdefault(family, {"family": family, "subspaces": 0, "best": 999, "unreachable": 0, "k_min": 99,
                                       "k_max": 0, "wall_s": 0.0, "hist": {}, "subset_counterexamples": 0})
        small = [p for p in probs if p[2] <= args.batch_k]
        large = [p for p in probs if p[2] > args.batch_k]
        chunks = [(small[i:i + 4 * threads], True) for i in range(0, len(small), 4 * threads)]
        chunks += [(large[i:i + 2], False) for i in range(0, len(large), 2)]
        for chunk, batch in chunks:
            if time.time() > deadline:
                st["cut_by_deadline"] = True
                break
            t0 = time.time()
            res = run_solver("".join(p[1] for p in chunk), batch=batch)
            st["wall_s"] = round(st["wall_s"] + time.time() - t0, 1)
            info = {p[0]: p for p in chunk}
            for r in res:
                _, _, k, gens = info[r["name"]]
                st["subspaces"] += 1
                st["k_min"], st["k_max"] = min(st["k_min"], k), max(st["k_max"], k)
                ans = r["answer"]
                st["unreachable"] += ans >= 120
                st["hist"][str(ans)] = st["hist"].get(str(ans), 0) + 1
                st["best"] = min(st["best"], ans)
                below = [S for S in range(1, 1 << len(terms)) if r["root_dp"][S] < naive_mask[S]]
                st["subset_counterexamples"] += bool(below)
                if ans < best["answer"] or ans < naive or below:
                    rec = {"kind": "subspace_best", "family": family, "case": r["name"], "k": k, "answer": ans,
                           "naive": naive, "beats_naive": ans < naive, "tree_check": verify_tree_f2(r, R, C),
                           "subsets_below": [[S, r["root_dp"][S], naive_mask[S]] for S in below][:20],
                           "tree": r["tree"], "gens": [hex(g) for g in gens]}
                    if ans < best["answer"]:
                        best = rec
                    emit(out, rec)
        emit(out, {"kind": "subspace_family", **st})

    def scheme_on(scheme, ks, rows=(0, 1), cols=(0, 1)) -> list[int]:
        return block_products(scheme, a, b, c, rows, ks, cols)

    def random_scheme_gens(nblocks: int) -> list[int]:
        pairing = rng.choice(pairings)
        g = []
        for ks in pairing[:nblocks]:
            g += scheme_on(rng.choice(orbit), ks, rng.choice(rowpairs), rng.choice(colpairs))
        return g

    # (a) Strassen schemes (36 over F2) on the 2x2x2 blocks of the inner indices: all pairs, k = 14 for <2,4,2>
    for pi, pairing in enumerate(pairings if "a" in args.only else []):
        if len(pairing) == 2:
            items = [(f"a_p{pi}_{i}_{j}", scheme_on(s0, pairing[0]) + scheme_on(s1, pairing[1]))
                     for i, s0 in enumerate(orbit) for j, s1 in enumerate(orbit)]
        else:
            items = [(f"a_p{pi}_{i}", scheme_on(s0, pairing[0])) for i, s0 in enumerate(orbit)]
        run_family(f"a_strassen_pairing{pi}", items)
    if "a" in args.only:
        # the standard pieces x_rk y_kc (so the naive tree is inside S) plus one scheme on one block, every one
        items = [(f"a1_p{pi}_h{h}_{i}", std + scheme_on(s0, pairing[h]))
                 for pi, pairing in enumerate(pairings) for h in range(len(pairing)) for i, s0 in enumerate(orbit)]
        run_family("a_std_plus_one_scheme", items)
        # the standard pieces plus a scheme on each block, sampled
        if b == 4:
            items = [(f"a2_{t}", std + random_scheme_gens(2)) for t in range(n_for(len(std) + 6))]
            run_family("a_std_plus_two_schemes", items)
        # one scheme per block plus m cross products (the X factor of one product times the Y factor of another)
        for m in (2, 4, 6):
            items = []
            for t in range(n_for(14 + m)):
                g = random_scheme_gens(2)
                cross = [outer(x_factor(p, R, C), y_factor(q, R, C), R, C)
                         for p, q in (rng.sample(g, 2) for _ in range(m))]
                items.append((f"a3_m{m}_{t}", g + cross))
            run_family(f"a_schemes_plus_cross_m{m}", items)
    # (b) the standard pieces plus m random dense rank-1 forms; (c) plus m aligned ones: u in one row's X block and v
    #     in one column's Y block, or only u aligned, or only v
    for m in args.extra:
        for how, fam in (("dense", "b_random_dense"), ("aligned", "c_aligned"), ("row", "c_row_aligned"),
                         ("col", "c_col_aligned")):
            if fam[0] not in args.only:
                continue
            items = [(f"{fam}{m}_{t}", std + [random_rank1(rng, R, C, a, b, c, how) for _ in range(m)])
                     for t in range(n_for(len(std) + m))]
            run_family(f"{fam}_m{m}", items)
    # (d) greedy growth from the best tree found (with (d) alone, the naive tree in the standard pieces' span): add
    #     rank-1 forms built from its edges' factors
    if "d" in args.only and not best.get("tree"):
        run_family("d_seed_standard_pieces", [("d_seed", std)])
    for rnd in range(args.greedy_rounds if "d" in args.only else 0):
        if not best.get("tree") or time.time() > deadline:
            break
        gens = [int(g, 16) for g in best["gens"]]
        edges = [int(x, 16) ^ int(y, 16) for x, y in best["tree"]]
        us = [x_factor(e, R, C) for e in edges]
        vs = [y_factor(e, R, C) for e in edges]
        items = []
        for t in range(n_for(len(reduce_basis(gens + [x["m"] for x in terms])) + 2)):
            extra = []
            for _ in range(rng.randint(1, 4)):
                u = rng.choice(us) ^ (rng.choice(us) if rng.random() < 0.5 else 0)
                v = rng.choice(vs) ^ (rng.choice(vs) if rng.random() < 0.5 else 0)
                if u and v:
                    extra.append(outer(u, v, R, C))
            items.append((f"d_greedy{rnd}_{t}", gens + extra))
        run_family(f"d_greedy_round{rnd}", items)
    emit(out, {"kind": "subspace_summary", "shape": [a, b, c], "L": L, "naive": naive, "best": best.get("answer"),
               "subspaces": sum(s["subspaces"] for s in stats.values()), "families": list(stats.values())})


# ---------------------------------------------------------------- annealing over Steiner points (sa.c), whole space


def geodesic_path(A: int, B: int, R: int, C: int) -> list[int]:
    """Nodes A = P0, ..., Pr = B with every step rank 1 (a rank decomposition of B - A from row echelon form)."""
    D = A ^ B
    rows = rows_of(D, R, C)
    basis: list[int] = []  # fully reduced, pivot = lowest bit
    for x in rows:
        for v in basis:
            if x & v & -v:
                x ^= v
        if x:
            p = x & -x
            basis = [v ^ x if v & p else v for v in basis] + [x]
    path, cur = [A], A
    for v in basis:
        p = v & -v
        u = sum(1 << i for i, x in enumerate(rows) if x & p)
        cur ^= outer(u, v, R, C)
        path.append(cur)
    assert cur == B, "rank decomposition failed"
    return path


def sa_tree(points: list[int], terms: list[int], R: int, C: int) -> list[tuple[int, int]]:
    nodes = [0] + terms + points
    inside, best = {0}, {i: (rank_f2(nodes[i], R, C), 0) for i in range(1, len(nodes))}
    edges = []
    while best:
        i = min(best, key=lambda j: best[j][0])
        _, j = best.pop(i)
        inside.add(i)
        p = geodesic_path(nodes[j], nodes[i], R, C)
        edges += list(zip(p, p[1:]))
        for q in best:
            d = rank_f2(nodes[i] ^ nodes[q], R, C)
            if d < best[q][0]:
                best[q] = (d, i)
    return edges


def run_sa(R: int, C: int, terms: list[int], naive: int, iters: int, restarts: int, seed: int) -> dict:
    exe = build("sa")
    lines = [f"{R} {C} {len(terms)} {naive} {iters} {restarts} {seed}"]
    lines += [" ".join(f"{x:x}" for x in rows_of(t, R, C)) for t in terms]
    res = json.loads(subprocess.run([exe], input="\n".join(lines) + "\n", check=True, stdout=subprocess.PIPE,
                                    text=True).stdout)
    pts = [sum(int(x, 16) << (i * C) for i, x in enumerate(p)) for p in res.pop("points")]
    if res["best"] < naive:
        edges = sa_tree(pts, terms, R, C)
        uniq = {tuple(sorted(e)) for e in edges}
        fake = {"tree": [[f"{a:x}", f"{b:x}"] for a, b in uniq], "terminals": [[f"{t:x}", ""] for t in terms],
                "tree_err": 0, "answer": len(uniq)}
        res["tree_check"] = verify_tree_f2(fake, R, C)
        res["tree"] = fake["tree"]
    res["steiner_points"] = [f"{p:x}" for p in pts]
    return res


def cmd_sa(args, out) -> None:
    R, C, ts = matmul_targets(args.a, args.b, args.c, args.L)
    if args.positions:
        keep = set(args.positions.split(","))
        ts = [t for t in ts if t["chain"] in keep]
    naive = naive_full(ts)
    res = run_sa(R, C, [t["m"] for t in ts], naive, args.iters, args.restarts, args.seed)
    emit(out, {"kind": "sa", "shape": [args.a, args.b, args.c], "L": args.L, "terminals": len(ts),
               "positions": args.positions or "all", "beats_naive": res["best"] < naive, **res})


def cmd_sa_calibrate(args, out) -> None:
    """The annealer against exact DW on families that do beat their naive sum (general-version counterexamples):
    how often it finds the exact optimum, at the iteration budget used on matmul."""
    rng = random.Random(args.seed)
    found = []
    R, C, terms = hub_example()
    found.append(("hub_example", R, C, [t["m"] for t in terms], 5, 6))
    for R, C in ((2, 6), (3, 5), (4, 4), (3, 6)):
        tries = 0
        while sum(1 for f in found if f[1:3] == (R, C)) < args.per_size and tries < 400:
            tries += 1
            terms = []
            for p in range(rng.randint(3, 6)):
                for _ in range(50):
                    top = rng.randint(1, min(R, C, 3))
                    a, b = random_independent(rng, R, top), random_independent(rng, C, top)
                    cand = terms + chain_terms(f"p{p}", a, b, [top], R, C)
                    if len({t["m"] for t in cand}) == len(cand) and condition_i(cand, R, C):
                        terms = cand
                        break
            if len(terms) < 3:
                continue
            (r,) = run_solver(spec_problem(f"cal{tries}", R, C, terms))
            naive = naive_by_mask(terms)[-1]
            if r["answer"] < naive:
                found.append((f"cal_{R}x{C}_{tries}", R, C, [t["m"] for t in terms], r["answer"], naive))
    hits = 0
    for name, R, C, ts, opt, naive in found:
        res = run_sa(R, C, ts, naive, args.iters, args.restarts, args.seed)
        hit = res["best"] == opt
        hits += hit
        emit(out, {"kind": "sa_calibrate", "case": name, "R": R, "C": C, "terminals": len(ts), "naive": naive,
                   "exact": opt, "sa_best": res["best"], "hit": hit, "hist": res["hist"], "wall_s": res["wall_s"],
                   "tree_check": res.get("tree_check")})
    emit(out, {"kind": "sa_calibrate_summary", "instances": len(found), "hits": hits})


def cmd_matmul(args, out) -> None:
    R, C, terms = matmul_targets(args.a, args.b, args.c, args.L)
    name = f"mm{args.a}{args.b}{args.c}_L{args.L}_F2"
    (res,) = run_solver(spec_problem(name, R, C, terms))
    a = analyze(res, terms, R, C)
    res_small = {k: v for k, v in res.items() if k not in ("root_dp",)}
    emit(out, {"case": name, "shape": [args.a, args.b, args.c], "L": args.L, **a, "solver": res_small,
               "root_dp": res["root_dp"], "labels": [t["label"] for t in terms]})


def run_full(*args: str) -> dict:
    out = subprocess.run([build("dw_full"), *args], check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(out.stdout)


def cmd_full(args, out) -> None:
    """<a,b,c> with L = b by dw_full (orbit-compressed DW, whole space). First its checks: the in-place relaxation against
    brute force; on small shapes every subset's orbit image against its directly computed array at every state, and
    every subset's root value against dw_f2; random terminal sets (with Steiner savings) against dw_f2, with the tree."""
    rel = subprocess.run([build("dw_full"), "--test-relax", str(args.seed)], stdout=subprocess.PIPE, text=True)
    emit(out, {"case": "dw_full_test_relax", "pass": rel.returncode == 0, "lines": rel.stdout.splitlines()[-1]})
    for a, b, c in [tuple(map(int, x.split(","))) for x in args.check_shapes]:
        R, C, terms = matmul_targets(a, b, c, b)
        got = run_full(str(a), str(b), str(c), "--check")
        (ref,) = run_solver(spec_problem(f"ref{a}{b}{c}", R, C, terms))
        same = got["root_dp"] == ref["root_dp"]
        tree = verify_tree_f2(got, R, C)
        emit(out, {"case": f"dw_full_check_{a}{b}{c}", "pass": same and got["check_mismatch"] == 0 and tree["ok"],
                   "answer": got["answer"], "dw_f2": ref["answer"], "arrays": got["arrays"],
                   "symmetry_order": got["symmetry_order"], "check_mismatch": got["check_mismatch"],
                   "root_dp_equal": same, "tree_check": tree})
    rng = random.Random(args.seed)
    fails = 0
    for i in range(args.cross):
        R, C = rng.choice([(3, 3), (4, 4), (3, 4), (2, 6)])
        nt = rng.randint(3, 6)
        ms: list[int] = []
        while len(ms) < nt:
            m = 0
            for _ in range(rng.randint(1, 3)):
                m ^= outer(rng.randrange(1, 1 << R), rng.randrange(1, 1 << C), R, C)
            if m and m not in ms:
                ms.append(m)
        terms = [{"m": m, "label": f"t{q}"} for q, m in enumerate(ms)]
        (ref,) = run_solver(spec_problem(f"cross{i}", R, C, terms))
        got = run_full("--terms", str(R), str(C), *[f"{m:x}" for m in ms])
        tree = verify_tree_f2(got, R, C)
        ok = got["root_dp"] == ref["root_dp"] and tree["ok"]
        fails += not ok
        if not ok:
            emit(out, {"case": f"dw_full_cross{i}", "pass": False, "answer": got["answer"], "dw_f2": ref["answer"],
                       "tree_check": tree})
    emit(out, {"case": "dw_full_cross", "instances": args.cross, "fails": fails, "pass": fails == 0})
    for a, b, c in [tuple(map(int, x.split(","))) for x in args.shapes]:
        R, C, terms = matmul_targets(a, b, c, b)
        res = run_full(str(a), str(b), str(c))
        if [int(t[0], 16) for t in res["terminals"]] != [t["m"] for t in terms]:
            raise SystemExit(f"dw_full {a},{b},{c}: its terminals are not matmul_targets({a}, {b}, {c}, {b})")
        a_ = analyze(res, terms, R, C)
        emit(out, {"case": res["name"], "shape": [a, b, c], "L": b, **a_,
                   "solver": {k: v for k, v in res.items() if k != "root_dp"}, "root_dp": res["root_dp"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="JSON lines file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("selftest")
    p.add_argument("--cases", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("selftest3")
    p.add_argument("--cases", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    for name in ("matmul", "matmul3"):
        p = sub.add_parser(name)
        for k in ("a", "b", "c", "L"):
            p.add_argument(f"--{k}", type=int, required=True)
    p = sub.add_parser("general")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sizes", nargs="+", default=["4x4", "3x5", "2x6"])
    p.add_argument("--families", type=int, default=300)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--max-terminals", type=int, default=8)
    p.add_argument("--grid-chains", type=int, default=300)
    p.add_argument("--max-rank", type=int, default=3, help="the largest rank of a position's chain")
    p.add_argument("--joint-only", action="store_true",
                   help="keep only random families whose pairwise-trivial row and column spaces are jointly independent "
                        "(the refined general version), and skip the grids")
    p = sub.add_parser("jointsub")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sizes", nargs="+", default=["4x8", "5x6", "6x6"])
    p.add_argument("--families", type=int, default=200)
    p.add_argument("--subspaces", type=int, default=4, help="subspaces per family")
    p.add_argument("--k", type=int, default=18, help="each subspace's dimension (fewer when the pieces span less)")
    p.add_argument("--max-positions", type=int, default=6)
    p.add_argument("--max-terminals", type=int, default=8)
    p.add_argument("--max-rank", type=int, default=3)
    p.add_argument("--calibrate", action="store_true",
                   help="families without joint independence, also solved on the whole space")
    p = sub.add_parser("aligned")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sizes", nargs="+", default=["4x4", "3x5", "2x6"])
    p.add_argument("--families", type=int, default=300)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--max-terminals", type=int, default=8)
    p = sub.add_parser("grid")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--families", type=int, default=40)
    p = sub.add_parser("subspace")
    for k, v in (("a", 2), ("b", 4), ("c", 2), ("L", 2)):
        p.add_argument(f"--{k}", type=int, default=v)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--samples", type=int, default=200, help="subspaces per sampled family")
    p.add_argument("--extra", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--max-k", type=int, default=24)
    p.add_argument("--batch-k", type=int, default=18,
                   help="solve subspaces up to this dimension in parallel, one each; above it, sample counts halve per dim")
    p.add_argument("--greedy-rounds", type=int, default=5)
    p.add_argument("--budget-s", type=float, default=3600)
    p.add_argument("--only", nargs="+", default=["a", "b", "c", "d"], choices=["a", "b", "c", "d"],
                   help="the families to run: (a) Strassen schemes, (b) random dense, (c) aligned, (d) greedy")
    p = sub.add_parser("sa")
    for k in ("a", "b", "c", "L"):
        p.add_argument(f"--{k}", type=int, required=True)
    p.add_argument("--positions", help="comma-separated chains to keep, e.g. r0c0,r0c1,r1c0")
    p.add_argument("--iters", type=int, default=200000)
    p.add_argument("--restarts", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("sa-calibrate")
    p.add_argument("--per-size", type=int, default=5)
    p.add_argument("--iters", type=int, default=200000)
    p.add_argument("--restarts", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("full")
    p.add_argument("--shapes", nargs="+", default=["2,2,4", "2,3,2"], help="a,b,c items, solved with L = b")
    p.add_argument("--check-shapes", nargs="+", default=["2,2,2", "2,1,3", "3,1,2", "2,1,4", "3,1,3", "2,2,3"])
    p.add_argument("--cross", type=int, default=300, help="random terminal sets checked against dw_f2")
    p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = open(args.out, "a") if args.out else None
    cmds = {"validate": cmd_validate, "selftest": cmd_selftest, "selftest3": cmd_selftest3, "matmul": cmd_matmul,
            "matmul3": cmd_matmul3, "general": cmd_general, "grid": cmd_grid, "subspace": cmd_subspace, "sa": cmd_sa,
            "sa-calibrate": cmd_sa_calibrate, "aligned": cmd_aligned, "jointsub": cmd_jointsub, "full": cmd_full}
    cmds[args.cmd](args, out)


if __name__ == "__main__":
    main()
