"""duo / duo2 differential trails as SAT (CaDiCaL through pysat): `trails.c`'s model, exact, with no list or form limit.

    python3 sat_trails.py check                                   # the model against trails.c's known trails
    python3 sat_trails.py run --kind duo2 --t0 0-255 --len 4-12 --bound 47 --jobs 40 --budget 300 > out.jsonl

The model is `trails.py`'s (round 2), per lane and word t, with the key known (the prover computes φ):

- the lane's state difference is additive, (P, Q) for (y, z);
- y's XOR difference is the support of any signed-digit form δ of P (pos − neg ≡ P mod 2^32, disjoint), at cost
  wt(δ) without bit 31;
- u = y ⊕ p ⊕ k_t has XOR difference X_y ⊕ Δp_t (Δp_t, the word's XOR difference, is shared by both lanes), and
  signed digits σ on that support with free signs, at cost wt(σ);
- P' = a_t·val(σ) + Q and Q' = val(rotl(σ, r_t)) + P', exactly (a_t known, so the multiply is a constant one).

A trail starts from equal states with Δp_{t0} ≠ 0 and ends, after `len` words, either with every lane's state equal
(collision) or, when it reaches the block's end (t0 + len = 256), with the receipt's XOR difference zero (fold). Its
probability is 2^-cost. UNSAT at a bound proves, within this model, that no trail of that shape costs that little.

`floors` uses the open end (no condition after the k-th word): every trail from t0, of any length and either end,
costs at least its first k words, so UNSAT there bounds every trail from t0 at once.

    python3 sat_trails.py floors --kind duo2 --t0 0-255 --k 8 --bounds 13,16,20,24,28,32,36,40,44,47 --jobs 8

`long` raises the bound for a set of starts step by step, with no conflict budget: each start climbs 18, 20, 22, …,
the lowest pending bound always first, and every finished step is a line of OUT/steps.jsonl as it finishes. A
cheap open prefix (SAT, not a trail) moves that start to a longer window; SAT at the fold end is a trail. OUT/
levels.jsonl gains a line each time every start of the set is UNSAT at a new bound, and OUT/proved.json holds
the current state, so a run stopped at any point has proved its last level. `proved` reads the steps of any runs
together:

    python3 sat_trails.py long --t0 0-255:10 --bounds 18,20,22,24,26,28 --jobs 8 --out OUT
    python3 sat_trails.py proved run1/out/steps.jsonl run2/out/steps.jsonl ...
"""

from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

try:
    from pysat.card import ITotalizer
    from pysat.solvers import Solver
except ImportError:  # `proved` only reads outputs
    ITotalizer = Solver = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trails  # noqa: E402

W = 32
B2 = 256


class Cnf:
    """A clause list with a variable counter, constant folding on the literal 0 = false and its negation."""

    def __init__(self) -> None:
        self.n = 1
        self.clauses: list[list[int]] = [[1]]
        self.T, self.F = 1, -1

    def var(self) -> int:
        self.n += 1
        return self.n

    def vec(self) -> list[int]:
        return [self.var() for _ in range(W)]

    def const(self, x: int) -> list[int]:
        return [self.T if (x >> i) & 1 else self.F for i in range(W)]

    def xor(self, a: int, b: int) -> int:
        if a in (self.T, self.F):
            return b if a == self.F else -b
        if b in (self.T, self.F):
            return a if b == self.F else -a
        if a == b:
            return self.F
        if a == -b:
            return self.T
        c = self.var()
        self.clauses += [[-a, -b, -c], [a, b, -c], [a, -b, c], [-a, b, c]]
        return c

    def and_(self, a: int, b: int) -> int:
        if self.F in (a, b):
            return self.F
        if a == self.T:
            return b
        if b == self.T:
            return a
        c = self.var()
        self.clauses += [[-c, a], [-c, b], [c, -a, -b]]
        return c

    def or_(self, a: int, b: int) -> int:
        return -self.and_(-a, -b)

    def maj(self, a: int, b: int, c: int) -> int:
        lits = (a, b, c)
        t = sum(x == self.T for x in lits)
        f = sum(x == self.F for x in lits)
        if t >= 2:
            return self.T
        if f >= 2:
            return self.F
        if t == 1 and f == 1:
            return next(x for x in lits if x not in (self.T, self.F))
        if t == 1:
            x, y = [v for v in lits if v != self.T]
            return self.or_(x, y)
        if f == 1:
            x, y = [v for v in lits if v != self.F]
            return self.and_(x, y)
        m = self.var()
        self.clauses += [[-a, -b, m], [-a, -c, m], [-b, -c, m], [a, b, -m], [a, c, -m], [b, c, -m]]
        return m

    def add(self, x: list[int], y: list[int], cin: int | None = None) -> list[int]:
        c = self.F if cin is None else cin
        out = []
        for i in range(W):
            out.append(self.xor(self.xor(x[i], y[i]), c))
            if i < W - 1:
                c = self.maj(x[i], y[i], c)
        return out

    def sub(self, x: list[int], y: list[int]) -> list[int]:
        return self.add(x, [-v for v in y], self.T)

    def mul_const(self, v: list[int], a: int) -> list[int]:
        """a·v mod 2^32 through a's non-adjacent form: one add or subtract of v << i per nonzero digit."""
        acc = self.const(0)
        i, x = 0, a & 0xFFFFFFFF
        while x and i < W:
            if x & 1:
                d = 2 - (x & 3) if i < W - 1 else 1
                sh = [self.F] * i + v[: W - i]
                acc = self.add(acc, sh) if d == 1 else self.sub(acc, sh)
                x -= d
            x >>= 1
            i += 1
        return acc

    def equal(self, x: list[int], y: list[int]) -> None:
        for a, b in zip(x, y):
            self.clauses += [[-a, b], [a, -b]]

    def nonzero(self, x: list[int]) -> None:
        self.clauses.append(list(x))


def rotl(v: list[int], r: int) -> list[int]:
    r %= W
    return v[W - r:] + v[: W - r]


def signed_digits(c: Cnf) -> tuple[list[int], list[int]]:
    pos, neg = c.vec(), c.vec()
    for a, b in zip(pos, neg):
        c.clauses.append([-a, -b])
    return pos, neg


def build(lanes, t0: int, n: int, end: str) -> tuple[Cnf, list[int], dict]:
    """The CNF of every trail over words t0..t0+n-1 of `lanes` ((a, r) schedules), its cost literals, and the vectors
    to read a model back."""
    c = Cnf()
    st = [(c.const(0), c.const(0)) for _ in lanes]
    cost: list[int] = []
    rec: dict = {"dp": [], "lanes": [[] for _ in lanes]}
    for t in range(t0, t0 + n):
        dp = c.vec()
        if t == t0:
            c.nonzero(dp)
        rec["dp"].append(dp)
        new = []
        for li, ((a, r), (P, Q)) in enumerate(zip(lanes, st)):
            pos, neg = signed_digits(c)
            c.equal(c.sub(pos, neg), P)
            xy = [c.or_(p, q) for p, q in zip(pos, neg)]
            cost += xy[: W - 1]
            xu = [c.xor(x, d) for x, d in zip(xy, dp)]
            sp, sn = signed_digits(c)
            for x, p, q in zip(xu, sp, sn):
                c.clauses += [[-x, p, q], [x, -p], [x, -q]]
            cost += xu
            v = c.sub(sp, sn)
            P2 = c.add(c.mul_const(v, a[t]), Q)
            Q2 = c.add(c.sub(rotl(sp, r[t]), rotl(sn, r[t])), P2)
            new.append((P2, Q2))
            rec["lanes"][li].append({"P": P, "xy": xy, "xu": xu, "sp": sp, "sn": sn})
        st = new
    if end == "collision":
        for P, Q in st:
            c.equal(P, c.const(0))
            c.equal(Q, c.const(0))
    elif end == "fold":
        fold = c.const(0)
        for li, (P, Q) in enumerate(st):
            xs = []
            for X in (P, Q):
                pos, neg = signed_digits(c)
                c.equal(c.sub(pos, neg), X)
                x = [c.or_(p, q) for p, q in zip(pos, neg)]
                cost += x[: W - 1]
                xs.append(x)
            lane = [c.xor(a, b) for a, b in zip(xs[0], rotl(xs[1], 16))]
            fold = [c.xor(a, b) for a, b in zip(fold, rotl(lane, 8 * li))]
        c.equal(fold, c.const(0))
    return c, cost, rec


def holds(model: set[int], x: int) -> bool:
    return (x in model) if x > 0 else (-x not in model)


def word(model: set[int], v: list[int]) -> int:
    return sum(1 << i for i, x in enumerate(v) if holds(model, x))


def solve(kind: str, kseed: int, t0: int, n: int, bound: int, budget: float, end: str = "collision",
          conflicts: int = 2_000_000) -> dict:
    """The cheapest trail of `kind` over words t0..t0+n-1 costing at most `bound`, or the proof that none does
    (`unsat`), or `unknown` when a solve call exhausts `conflicts` or the calls together pass `budget` seconds.
    (pysat's CaDiCaL takes a conflict budget, not an interrupt.)"""
    lanes = [(a, r) for a, _, r in trails.lanes_of(kind, kseed)]
    t_build = time.time()
    c, cost, rec = build(lanes, t0, n, end)
    tot = ITotalizer(lits=cost, ubound=bound + 1, top_id=c.n)
    out = {"kind": kind, "kseed": hex(kseed), "t0": t0, "len": n, "end": end, "bound": bound, "vars": tot.top_id,
           "clauses": len(c.clauses) + len(tot.cnf.clauses), "build_s": round(time.time() - t_build, 2)}
    best, status, b = None, "unsat", bound
    t_solve = time.time()
    with Solver(name="cadical195", bootstrap_with=c.clauses + tot.cnf.clauses) as s:
        while b >= 0:
            if time.time() - t_solve > budget:
                status = "unknown" if best is None else "unknown-below"
                break
            s.conf_budget(conflicts)
            r = s.solve_limited(assumptions=[-tot.rhs[b]])
            if r is None:
                status = "unknown" if best is None else "unknown-below"
                break
            if r is False:
                status = "unsat" if best is None else "optimal"
                break
            m = set(x for x in s.get_model() if x > 0)
            w = sum(1 for x in cost if holds(m, x))
            best = {"cost": w, "dp": [hex(word(m, v)) for v in rec["dp"]],
                    "sigma": [[[hex(word(m, st["sp"])), hex(word(m, st["sn"]))] for st in lane] for lane in rec["lanes"]]}
            b = w - 1
    out.update(status=status, solve_s=round(time.time() - t_solve, 2), trail=best)
    return out


def floor(kind: str, kseed: int, t0: int, k: int, bounds: list[int], conflicts: int) -> dict:
    """The highest of `bounds` (ascending) below which no trail from t0 costs, through the open end: any trail
    starting at t0 (collision or fold, any length) costs at least the cheapest k-word prefix, so UNSAT at b proves
    every trail from t0 costs more than b. A start whose k words reach the block's end takes the fold end over the
    words left instead, which covers its collisions too (a collided state stays equal to the end at no cost).
    Stops at the first bound that is SAT (reporting that prefix) or undecided."""
    n = min(k, B2 - t0)
    end = "fold" if t0 + k >= B2 else "open"
    lanes = [(a, r) for a, _, r in trails.lanes_of(kind, kseed)]
    c, cost, rec = build(lanes, t0, n, end)
    tot = ITotalizer(lits=cost, ubound=bounds[-1] + 1, top_id=c.n)
    out = {"kind": kind, "kseed": hex(kseed), "t0": t0, "k": n, "end": end, "conflicts": conflicts, "floor": None,
           "stopped": None, "prefix": None, "steps": []}
    with Solver(name="cadical195", bootstrap_with=c.clauses + tot.cnf.clauses) as s:
        for b in bounds:
            t = time.time()
            s.conf_budget(conflicts)
            r = s.solve_limited(assumptions=[-tot.rhs[b]])
            out["steps"].append([b, {None: "unknown", False: "unsat", True: "sat"}[r], round(time.time() - t, 2)])
            if r is False:
                out["floor"] = b
                continue
            out["stopped"] = b
            if r:
                m = set(x for x in s.get_model() if x > 0)
                out["prefix"] = {"cost": sum(1 for x in cost if holds(m, x)),
                                 "dp": [hex(word(m, v)) for v in rec["dp"]]}
            break
    return out


def _job(args: tuple) -> dict:
    return solve(*args)


def _floor_job(args: tuple) -> dict:
    return floor(*args)


def step(kind: str, kseed: int, t0: int, k: int, b: int) -> dict:
    """One step of `long`, solved to the end: UNSAT proves every trail from t0 costs more than b; SAT gives the
    cheapest k-word prefix (open end) or a trail (fold end, when t0's k words reach the block's end) within b, with
    its word differences measured on the function (`sim_log2p`, None when they don't collide)."""
    n = min(k, B2 - t0)
    end = "fold" if t0 + k >= B2 else "open"
    lanes = [(a, r) for a, _, r in trails.lanes_of(kind, kseed)]
    c, cost, rec = build(lanes, t0, n, end)
    tot = ITotalizer(lits=cost, ubound=b + 1, top_id=c.n)
    t = time.time()
    with Solver(name="cadical195", bootstrap_with=c.clauses + tot.cnf.clauses) as s:
        r = s.solve(assumptions=[-tot.rhs[b]])
        out = {"t0": t0, "k": k, "n": n, "end": end, "bound": b, "status": "sat" if r else "unsat",
               "s": round(time.time() - t, 1)}
        if r:
            m = set(x for x in s.get_model() if x > 0)
            dp = [word(m, v) for v in rec["dp"]]
            sim = trails.simulate_lanes(kind, kseed, [(t0 + i, d) for i, d in enumerate(dp)], 1 << 16, 1, False,
                                        "fold" if end == "fold" else "collision")
            out.update(cost=sum(1 for x in cost if holds(m, x)), dp=[hex(d) for d in dp], sim_log2p=sim["log2p"])
    return out


def _step_job(args: tuple) -> dict:
    return step(*args)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def long_run(kind: str, kseed: int, starts: list[int], bounds: list[int], k0: int, kmax: int, kstep: int,
             jobs: int, out: Path) -> int:
    """`long`: per start a chain of steps up `bounds`, scheduled lowest bound first over `jobs` processes."""
    out.mkdir(parents=True, exist_ok=True)
    order = {t0: i for i, t0 in enumerate(starts)}
    best: dict[int, int | None] = {t0: None for t0 in starts}
    stopped: dict[int, dict] = {}
    heap = [(bounds[0], order[t0], t0, k0) for t0 in starts]
    heapq.heapify(heap)
    inflight: dict[int, tuple[int, int, float]] = {}
    done_q: queue.Queue = queue.Queue()
    level, t_start = None, time.time()
    meta = {"kind": kind, "kseed": hex(kseed), "starts": starts, "bounds": bounds, "k0": k0, "kmax": kmax}
    with open(out / "steps.jsonl", "a") as steps, open(out / "levels.jsonl", "a") as levels, mp.Pool(jobs) as pool:
        def submit() -> None:
            while heap and len(inflight) < jobs:
                b, _, t0, k = heapq.heappop(heap)
                inflight[t0] = (k, b, time.time())
                pool.apply_async(_step_job, ((kind, kseed, t0, k, b),), callback=done_q.put,
                                 error_callback=lambda e, t0=t0, k=k, b=b: done_q.put(
                                     {"t0": t0, "k": k, "bound": b, "status": "error", "error": repr(e)}))

        submit()
        while inflight:
            r = done_q.get()
            t0 = r["t0"]
            k, b, _ = inflight.pop(t0)
            r["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            steps.write(json.dumps(r) + "\n")
            steps.flush()
            if r["status"] == "unsat":
                best[t0] = b
                nxt = [x for x in bounds if x > b]
                if nxt:
                    heapq.heappush(heap, (nxt[0], order[t0], t0, k))
            elif r["status"] == "sat" and r["end"] == "open" and r["sim_log2p"] is None and k + kstep <= kmax:
                heapq.heappush(heap, (b, order[t0], t0, k + kstep))
            else:
                stopped[t0] = r
            low = min((x if x is not None else -1) for x in best.values())
            if low >= 0 and low != level:
                level = low
                levels.write(json.dumps({"bound": level, "starts": len(starts), "utc": r["utc"],
                                         "elapsed_s": round(time.time() - t_start)}) + "\n")
                levels.flush()
            _write_json(out / "proved.json", meta | {
                "subset_bound": level, "meaning": "every start of `starts` is UNSAT at subset_bound: no trail from "
                "any of them costs subset_bound or less (probability above 2^-subset_bound-1), in the model",
                "per_start": best, "stopped": stopped,
                "inflight": {t: {"k": kk, "bound": bb, "since_s": round(time.time() - ts)}
                             for t, (kk, bb, ts) in inflight.items()},
                "updated_utc": r["utc"]})
            submit()
    return 0


def proved(paths: list[str], base: int) -> dict:
    """The all-starts bound from the steps of any `long` runs together: per start, its highest UNSAT bound (at least
    `base`, what earlier runs proved for every start), and the least of those over the 256 starts."""
    best = {t0: base for t0 in range(B2)}
    found = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["status"] == "unsat":
                best[r["t0"]] = max(best[r["t0"]], r["bound"])
            elif r["status"] == "sat" and (r["end"] == "fold" or r.get("sim_log2p") is not None):
                found.append(r)
    low = min(best.values())
    behind = sorted(t for t, v in best.items() if v == low)
    levels = {b: sum(v >= b for v in best.values()) for b in sorted(set(best.values()))}
    return {"all_starts_bound": low, "log2p_at_most": -(low + 1), "starts_at_bound": len(behind),
            "first_starts_at_bound": behind[:16], "starts_unsat_at_least": levels, "trails": found}


def span(s: str) -> list[int]:
    """"lo-hi" or "lo-hi:step", inclusive, or a comma list of those."""
    if "," in s:
        return sorted({t for part in s.split(",") for t in span(part)})
    s, _, step = s.partition(":")
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1, int(step or 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", default="duo2", choices=trails.SCHED_KINDS)
    r.add_argument("--kseed", type=lambda x: int(x, 0), default=0xC0FFEE11)
    r.add_argument("--t0", default="0-255")
    r.add_argument("--len", default="4-12")
    r.add_argument("--bound", type=int, default=47)
    r.add_argument("--budget", type=float, default=300)
    r.add_argument("--conflicts", type=int, default=2_000_000)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--end", default="collision", choices=("collision", "fold"))
    f = sub.add_parser("floors")
    f.add_argument("--kind", default="duo2", choices=trails.SCHED_KINDS)
    f.add_argument("--kseed", type=lambda x: int(x, 0), default=0xC0FFEE11)
    f.add_argument("--t0", default="0-255")
    f.add_argument("--k", type=int, default=8)
    f.add_argument("--bounds", default="13,16,20,24,28,32,36,40,44,47")
    f.add_argument("--conflicts", type=int, default=2_000_000)
    f.add_argument("--jobs", type=int, default=1)
    g = sub.add_parser("long")
    g.add_argument("--kind", default="duo2", choices=trails.SCHED_KINDS)
    g.add_argument("--kseed", type=lambda x: int(x, 0), default=0xC0FFEE11)
    g.add_argument("--t0", default="0-255")
    g.add_argument("--bounds", default="18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47")
    g.add_argument("--k0", type=int, default=8)
    g.add_argument("--kmax", type=int, default=16)
    g.add_argument("--kstep", type=int, default=4)
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("proved")
    p.add_argument("steps", nargs="+")
    p.add_argument("--base", type=int, default=16, help="the bound earlier runs proved for every start")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "long":
        return long_run(a.kind, a.kseed, span(a.t0), [int(x) for x in a.bounds.split(",")], a.k0, a.kmax, a.kstep,
                        a.jobs, a.out)
    if a.cmd == "proved":
        print(json.dumps(proved(a.steps, a.base), indent=1))
        return 0
    if a.cmd == "floors":
        bounds = [int(x) for x in a.bounds.split(",")]
        tasks = [(a.kind, a.kseed, t0, a.k, bounds, a.conflicts) for t0 in span(a.t0)]
        with mp.Pool(a.jobs) as pool:
            for o in pool.imap_unordered(_floor_job, tasks):
                print(json.dumps(o), flush=True)
        return 0
    if a.cmd == "check":
        cases = [("duo", 87, 3, 4), ("duo2-lane2", 41, 3, 3), ("duo2-lane1", 20, 4, 5)]
        ok = True
        for kind, t0, n, want in cases:
            o = solve(kind, 0xC0FFEE11, t0, n, want + 2, 120)
            got = o["trail"]["cost"] if o["trail"] else None
            diffs = [(t0 + i, int(d, 16)) for i, d in enumerate(o["trail"]["dp"])] if got is not None else []
            sim = trails.simulate_lanes(kind, 0xC0FFEE11, diffs, 1 << 18, 1, False, "collision") if diffs else {}
            print(json.dumps({"case": [kind, t0, n], "want": want, "got": got, "status": o["status"],
                              "sim_log2p": sim.get("log2p"), "solve_s": o["solve_s"]}), flush=True)
            ok &= got == want and o["status"] == "optimal" and sim["log2p"] is not None and sim["log2p"] >= -got - 0.3
        return 0 if ok else 1
    tasks = [(a.kind, a.kseed, t0, n, a.bound, a.budget, a.end, a.conflicts) for n in span(a.len) for t0 in span(a.t0)
             if t0 + n <= B2 and (a.end == "collision" or t0 + n == B2)]
    with mp.Pool(a.jobs) as pool:
        for o in pool.imap_unordered(_job, tasks):
            print(json.dumps(o), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
