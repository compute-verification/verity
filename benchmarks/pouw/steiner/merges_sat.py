"""Exact minimum edge counts with merges and copies (Conjecture B and Question C), by SAT over F2.

A computation over F2 on R x C matrices (steiner.py's layout: x[r][k] at row r*b + k, y[k][j] at column k*c + j, bit
i*C + j) is a sequence of nodes from node 0 = 0. Each step adds one node:

  edge  (cost 1)                : u + g, u any existing node, g of rank 1;
  merge (cost 0, counted in M)  : the sum of 2 or 3 (--arity) distinct existing nodes other than 0;
  copy  (cost 1, counted in K)  : u + w for two distinct existing nodes other than 0 (w of any rank).

Nodes persist, so any node may be extended or reused any number of times. The terminals are the matmul prefix targets
T[(r, j), s] = sum_{k < sL} x[r][k] y[k][j]; a computation covers them when each is some node's value. For a budget
(at most E edges + copies, at most M merges, at most K copies) `encode` builds a CNF that is satisfiable iff such a
computation exists, and CaDiCaL decides it. Every model is replayed by `replay`, which recomputes each node from the
step's operands alone (rank 1 by steiner.rank_f2, sums, coverage, the counts) and shares no code with the encoding.

The CNF restricts to computations that some optimal one can be put into (so UNSAT is exact):
- node values are nonzero and pairwise distinct (a repeated value can use the earlier node);
- every node is a terminal or an operand of a later step (an unused node can be dropped);
- idle steps (budget not used) come last;
- if step t+1 does not read node t, then value(t) < value(t+1) as integers: list the steps by the greedy linear
  extension that always takes the smallest available value; a step that does not read its predecessor was available
  when its predecessor was taken.

Rank bound: every value is a sum of edge increments, so #edges >= the tensor rank of the target set, whatever M and K.

Subcommands: `solve` (one budget), `table` (min E for each M or K, in parallel), `replay` (re-check a JSON line).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from multiprocessing.connection import wait as mp_wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steiner import matmul_targets, rank_f2  # noqa: E402

from pysat.card import CardEnc, EncType  # noqa: E402
from pysat.formula import IDPool  # noqa: E402
from pysat.solvers import Solver  # noqa: E402


def outer(u: int, w: int, R: int, C: int) -> int:
    m = 0
    for i in range(R):
        if (u >> i) & 1:
            m |= w << (i * C)
    return m


def shape_terms(a: int, b: int, c: int, L: int) -> tuple[int, int, list[dict]]:
    return matmul_targets(a, b, c, L)


# ---------------------------------------------------------------- the encoding


def encode(R: int, C: int, terms: list[int], E: int, M: int, K: int, arity: int, extra_steps: int = 0):
    """CNF for: a computation with <= E edges + copies, <= M merges, <= K copies covering terms. Returns (clauses,
    decode) where decode(model) gives the steps."""
    P = R * C
    S = E + M + extra_steps
    pool = IDPool()
    cl: list[list[int]] = []
    V = lambda *k: pool.id(k)  # noqa: E731

    ed = [None] + [V("ed", t) for t in range(1, S + 1)]
    me = [None] + [V("me", t) for t in range(1, S + 1)]
    co = [None] + [V("co", t) for t in range(1, S + 1)]
    idl = [None] + [V("id", t) for t in range(1, S + 1)]

    def n(t, p):
        return V("n", t, p)

    def exactly_one(lits):
        cl.append(list(lits))
        enc = CardEnc.atmost(lits=list(lits), bound=1, vpool=pool, encoding=EncType.pairwise if len(lits) <= 6 else EncType.seqcounter)
        cl.extend(enc.clauses)

    for t in range(1, S + 1):
        exactly_one([ed[t], me[t], co[t], idl[t]])
        if M == 0:
            cl.append([-me[t]])
        if K == 0:
            cl.append([-co[t]])
        if t < S:
            cl.append([-idl[t], idl[t + 1]])
        for slot in "ABC":
            exactly_one([V(slot, t, s) for s in range(t)])
            for s in range(t):
                x = V(slot, t, s)
                for p in range(P):
                    v = V("v" + slot, t, p)
                    if s == 0:
                        cl.append([-x, -v])
                    else:
                        cl.append([-x, -v, n(s, p)])
                        cl.append([-x, v, -n(s, p)])
        u = [V("u", t, i) for i in range(R)]
        w = [V("w", t, j) for j in range(C)]
        for i in range(R):
            for j in range(C):
                g = V("g", t, i, j)
                cl.append([-g, u[i]])
                cl.append([-g, w[j]])
                cl.append([g, -u[i], -w[j]])
        cl.append([-ed[t]] + u)
        cl.append([-ed[t]] + w)
        for x in u + w:
            cl.append([ed[t], -x])
        # n = vA ^ vB ^ vC ^ g
        for i in range(R):
            for j in range(C):
                p = i * C + j
                xs = [n(t, p), V("vA", t, p), V("vB", t, p), V("vC", t, p), V("g", t, i, j)]
                for mask in range(32):
                    if bin(mask).count("1") % 2 == 1:
                        cl.append([xs[k] if (mask >> k) & 1 == 0 else -xs[k] for k in range(5)])
        A0, B0, C0 = V("A", t, 0), V("B", t, 0), V("C", t, 0)
        cl.append([-ed[t], B0])
        cl.append([-ed[t], C0])
        cl.append([-idl[t], A0])
        cl.append([-idl[t], B0])
        cl.append([-idl[t], C0])
        for typ in (me[t], co[t]):
            cl.append([-typ, -A0])
            cl.append([-typ, -B0])
        cl.append([-co[t], C0])
        if arity == 2:
            cl.append([-me[t], C0])
        # operand order: index(A) < index(B) < index(C) whenever B, C are not 0
        for s in range(1, t):
            cl.append([-V("B", t, s)] + [V("A", t, s2) for s2 in range(s)])
            cl.append([-V("C", t, s)] + [V("B", t, s2) for s2 in range(1, s)])
        # nonzero
        cl.append([idl[t]] + [n(t, p) for p in range(P)])

    # distinct values
    for t in range(1, S + 1):
        for t2 in range(t + 1, S + 1):
            ds = []
            for p in range(P):
                d = V("d", t, t2, p)
                ds.append(d)
                cl.append([-d, n(t, p), n(t2, p)])
                cl.append([-d, -n(t, p), -n(t2, p)])
            cl.append([idl[t2]] + ds)

    # terminals
    h = {}
    for j, T in enumerate(terms):
        lits = []
        for t in range(1, S + 1):
            x = V("h", j, t)
            h[j, t] = x
            lits.append(x)
            cl.append([-x, -idl[t]])
            for p in range(P):
                cl.append([-x, n(t, p) if (T >> p) & 1 else -n(t, p)])
        cl.append(lits)

    # every node is a terminal or read later
    for t in range(1, S + 1):
        reads = [V(slot, t2, t) for t2 in range(t + 1, S + 1) for slot in "ABC"]
        cl.append([idl[t]] + [h[j, t] for j in range(len(terms))] + reads)

    # adjacent independent steps in increasing value order (bit P-1 most significant)
    for t in range(1, S):
        dep = [V(slot, t + 1, t) for slot in "ABC"]
        lt = [V("lt", t, k) for k in range(P)]  # lt[k]: value(t) < value(t+1) on bits k..0 (from the top: P-1-k)
        bits = list(range(P - 1, -1, -1))
        for k, p in enumerate(bits):
            a, b_ = n(t, p), n(t + 1, p)
            cl.append([-lt[k], -a, b_])
            if k + 1 < P:
                cl.append([-lt[k], -a, lt[k + 1]])
                cl.append([-lt[k], b_, lt[k + 1]])
            else:
                cl.append([-lt[k], -a])
                cl.append([-lt[k], b_])
        cl.append([idl[t + 1]] + dep + [lt[0]])

    # budgets
    for lits, bound in (
        ([ed[t] for t in range(1, S + 1)] + [co[t] for t in range(1, S + 1)], E),
        ([me[t] for t in range(1, S + 1)], M),
        ([co[t] for t in range(1, S + 1)], K),
    ):
        if bound < len(lits):
            cl.extend(CardEnc.atmost(lits=lits, bound=bound, vpool=pool, encoding=EncType.seqcounter).clauses)

    def decode(model):
        val = set(x for x in model if x > 0)
        steps = []
        for t in range(1, S + 1):
            if V("id", t) in val:
                break
            typ = "edge" if V("ed", t) in val else "merge" if V("me", t) in val else "copy"
            ops = [next(s for s in range(t) if V(slot, t, s) in val) for slot in "ABC"]
            st = {"type": typ}
            if typ == "edge":
                st["from"] = ops[0]
                st["u"] = sum(1 << i for i in range(R) if V("u", t, i) in val)
                st["w"] = sum(1 << j for j in range(C) if V("w", t, j) in val)
            else:
                st["of"] = [s for s in ops if s]
            st["value"] = sum(1 << p for p in range(P) if V("n", t, p) in val)
            steps.append(st)
        return steps

    return cl, decode, pool.top


# ---------------------------------------------------------------- independent replay


def replay(R: int, C: int, terms: list[int], steps: list[dict], E: int, M: int, K: int, arity: int) -> dict:
    vals = [0]
    ne = nm = nk = 0
    errs = []
    for t, st in enumerate(steps, 1):
        if st["type"] == "edge":
            a = st["from"]
            if not (0 <= a < t):
                errs.append(f"step {t}: parent {a}")
            g = outer(st["u"], st["w"], R, C)
            if rank_f2(g, R, C) != 1:
                errs.append(f"step {t}: increment rank {rank_f2(g, R, C)}")
            v = vals[a] ^ g
            ne += 1
        else:
            of = st["of"]
            lim = arity if st["type"] == "merge" else 2
            if not (2 <= len(of) <= lim) or len(set(of)) != len(of) or any(not (1 <= s < t) for s in of):
                errs.append(f"step {t}: operands {of}")
            v = 0
            for s in of:
                v ^= vals[s]
            if st["type"] == "merge":
                nm += 1
            else:
                nk += 1
        if v != st["value"]:
            errs.append(f"step {t}: value {v:x} != claimed {st['value']:x}")
        vals.append(v)
    have = set(vals)
    missing = [j for j, T in enumerate(terms) if T not in have]
    if missing:
        errs.append(f"terminals not covered: {missing}")
    if ne + nk > E or nm > M or nk > K:
        errs.append(f"over budget: edges {ne} copies {nk} merges {nm}")
    return {"ok": not errs, "errors": errs, "edges": ne, "merges": nm, "copies": nk, "cost": ne + nk}


# ---------------------------------------------------------------- driver


def solve(job: dict) -> dict:
    a, b, c, L = job["shape"]
    R, C, tl = shape_terms(a, b, c, L)
    terms = [t["m"] for t in tl]
    E, M, K, arity = job["E"], job["M"], job["K"], job["arity"]
    t0 = time.time()
    clauses, decode, nvars = encode(R, C, terms, E, M, K, arity)
    t1 = time.time()
    res = dict(job, R=R, C=C, naive=a * b * c, vars=nvars, clauses=len(clauses), encode_s=round(t1 - t0, 2))
    with Solver(name=job.get("solver", "cadical195"), bootstrap_with=clauses) as s:
        r = s.solve()
        res["solve_s"] = round(time.time() - t1, 2)
        res["status"] = "sat" if r else "unsat"
        if r:
            steps = decode(s.get_model())
            res["steps"] = steps
            res["replay"] = replay(R, C, terms, steps, E, M, K, arity)
    return res


def _child(job: dict, conn) -> None:
    conn.send(solve(job))
    conn.close()


def run_jobs(jobs: list[dict], procs: int, emit) -> None:
    """Each job in its own process (CaDiCaL ignores pysat's interrupt), killed at its timeout (status unknown)."""
    pending = list(jobs)
    running = {}
    while pending or running:
        while pending and len(running) < procs:
            j = pending.pop(0)
            r, w = mp.Pipe(duplex=False)
            p = mp.Process(target=_child, args=(j, w), daemon=True)
            p.start()
            w.close()
            running[r] = (p, j, time.monotonic() + j["timeout"] if j.get("timeout") else None)
        deadlines = [d for (_, _, d) in running.values() if d is not None]
        ready = mp_wait(list(running), timeout=max(0.0, min(deadlines) - time.monotonic()) if deadlines else None)
        for r in ready:
            p, j, _ = running.pop(r)
            try:
                res = r.recv()
            except EOFError:
                res = dict(j, status="crashed", solve_s=None)
            p.join()
            emit(res)
        now = time.monotonic()
        for r, (p, j, d) in list(running.items()):
            if d is not None and now >= d:
                p.kill()
                p.join()
                running.pop(r)
                emit(dict(j, status="unknown", solve_s=j["timeout"]))


def jobs_for_table(shape, model: str, Ms: list[int], arity: int, lo: int, timeout: float) -> list[dict]:
    """Every budget E in [lo, naive - 1] for each M (model merge) or K (model copy)."""
    a, b, c, L = shape
    naive = a * b * c
    out = []
    for x in Ms:
        for E in range(lo, naive):
            M, K = (x, 0) if model == "merge" else (0, x)
            out.append({"shape": shape, "model": model, "E": E, "M": M, "K": K, "arity": arity, "timeout": timeout})
    return out


def job(shape, model, E, x, arity=3, timeout=0.0) -> dict:
    M, K = (x, 0) if model == "merge" else (0, x)
    return {"shape": shape, "model": model, "E": E, "M": M, "K": K, "arity": arity, "timeout": timeout}


def plan(name: str, timeout: float) -> list[dict]:
    """Named job lists. `small`: the requested shapes (<2,2,2> and <2,2,3>, L = 2 and 1), every M and K up to 4 at
    the budgets between the rank bound and naive - 1, plus E = rank - 1 with M = 8 (the rank bound inside the model).
    `kappa`: shapes whose gap naive - rank exceeds 1, where one merge could save more than L."""
    J = []
    if name == "small":
        for shape, E in (((2, 2, 2, 2), 7), ((2, 2, 3, 2), 11)):
            J += [job(shape, "merge", E, m, 3, timeout) for m in (0, 1, 2, 3)]
            J += [job(shape, "merge", E, m, 2, timeout) for m in (1, 2, 3, 4)]
            J += [job(shape, "copy", E, k, 3, timeout) for k in (1, 2, 3, 4)]
        J.append(job((2, 2, 2, 2), "merge", 6, 8, 3, timeout))
        for shape, E in (((2, 2, 2, 1), 7), ((2, 2, 3, 1), 11)):
            J += [job(shape, "merge", E, m, 3, timeout) for m in (1, 2, 3)]
            J += [job(shape, "copy", E, k, 3, timeout) for k in (1, 2)]
    elif name == "kappa":
        # (shape, rank of the target set, M values, include copies)
        for shape, rank, Ms, copies in (
            ((2, 2, 4, 2), 14, (0, 1, 2, 3, 4), True),
            ((2, 3, 2, 3), 11, (0, 1, 2), False),
            ((3, 2, 3, 2), 15, (0, 1, 2, 3), True),
            ((2, 4, 2, 4), 14, (1, 2, 3), False),
            ((2, 4, 2, 2), 14, (1, 2, 3), False),
            ((2, 3, 3, 3), 15, (1, 2, 3), False),
            ((2, 2, 6, 2), 21, (1, 2), False),
        ):
            naive = shape[0] * shape[1] * shape[2]
            for E in range(naive - 1, rank - 1, -1):
                J += [job(shape, "merge", E, m, 3, timeout) for m in Ms]
                if shape == (2, 2, 4, 2):
                    J += [job(shape, "merge", E, m, 2, timeout) for m in Ms if m]
                if copies:
                    J += [job(shape, "copy", E, k, 3, timeout) for k in (1, 2, 3)]
    else:
        raise SystemExit(f"unknown plan {name}")
    return J


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="-")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("batch")
    p.add_argument("--plan", required=True, help="small | kappa")
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p = sub.add_parser("solve")
    p.add_argument("--shape", required=True, help="a,b,c,L")
    p.add_argument("--E", type=int, required=True)
    p.add_argument("--M", type=int, default=0)
    p.add_argument("--K", type=int, default=0)
    p.add_argument("--arity", type=int, default=3)
    p.add_argument("--timeout", type=float, default=0)
    p = sub.add_parser("table")
    p.add_argument("--shape", required=True, help="a,b,c,L")
    p.add_argument("--model", choices=["merge", "copy"], default="merge")
    p.add_argument("--values", required=True, help="M (or K) values, comma separated")
    p.add_argument("--arity", type=int, default=3)
    p.add_argument("--lo", type=int, required=True, help="smallest E tried (the rank bound)")
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    out = sys.stdout if args.out == "-" else open(args.out, "a")
    shape = tuple(int(x) for x in args.shape.split(",")) if args.cmd != "batch" else None
    if args.cmd == "batch":
        jobs = [j for p in args.plan.split(",") for j in plan(p, args.timeout)]
    elif args.cmd == "solve":
        jobs = [{"shape": shape, "model": "merge" if args.K == 0 else "copy", "E": args.E, "M": args.M, "K": args.K,
                 "arity": args.arity, "timeout": args.timeout}]
    else:
        jobs = jobs_for_table(shape, args.model, [int(x) for x in args.values.split(",")], args.arity, args.lo,
                              args.timeout)
    def emit(res):
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"{res['shape']} {res['model']} E={res['E']} M={res['M']} K={res['K']} ar={res['arity']}: "
              f"{res['status']} {res['solve_s']}s" + (f" replay={res['replay']['ok']}" if "replay" in res else ""),
              file=sys.stderr, flush=True)

    run_jobs(jobs, 1 if args.cmd == "solve" else args.procs, emit)


if __name__ == "__main__":
    main()
