"""Critical-path planner over the declaration graph.

Inputs: deps.tsv(.gz) from Extract.lean (every in-tree constant and the in-tree constants it uses), module times (measured
wall seconds per module), per-declaration times (parse_trace.py's JSON), and the import graph (graph.py).

It evaluates, as the critical path of the module DAG with each module weighing its measured time:
  current     the import graph as it is;
  shaken      each module importing only the modules whose constants its constants use (what a perfect shake leaves);
  split       the shaken graph after greedy splits: a module on the critical path is cut in two, the part its importer on
              the path needs (closed under uses inside the module) and the rest, when that shortens the path.
A part weighs the floor (import and fixed cost) plus its share of the module's measured time above the floor, by its
declarations' traced times (by count when the module has none traced).

    python3 plan.py --repo R --deps deps.tsv.gz --times times.json --decls trace.json [--splits 20] [--out plan.json]"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

AUX = re.compile(r"^(_.*|proof_\d+|match_\d+|eq_\d+|eq_def|induct|induct_unfolding|fun_cases|mutual_induct|rec|recOn|casesOn|"
                 r"noConfusion|noConfusionType|below|brecOn|binductionOn|ibelow|mk|sizeOf_spec|inj|injEq|congr_simp|"
                 r"ext|ext_iff|spec_\d+|unfold|hcongr_\d+|splitter|else_eq|eq_iff|sizeOf|ctorIdx|toCtorIdx|ofNat|"
                 r"proof_\d+_\d+|inj_\d+)$")


def owner_full(n: str) -> str:
    """``n`` without its trailing auxiliary components (an equation lemma, a matcher, a proof_N), private prefix kept."""
    parts = n.split(".")
    while len(parts) > 1 and AUX.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def owner_name(n: str) -> str:
    if n.startswith("_private."):
        parts = n.split(".")
        i = next((k for k, p in enumerate(parts) if p.isdigit()), None)
        n = ".".join(parts[i + 1:]) if i is not None else n
    parts = n.split(".")
    while len(parts) > 1 and AUX.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def load_deps(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    mods, consts = {}, []
    with op(path, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "M":
                mods[int(p[1])] = p[2]
            elif p[0] == "C":
                consts.append((int(p[1]), int(p[2]), p[3], p[4], [int(x) for x in p[5].split(",") if x]))
    return mods, consts


def longest(root, deps, weight, merge=None):
    """The longest path from ``root`` by ``weight``; with ``merge`` = (A, B, w), as if A were merged into B weighing w."""
    if merge is not None:
        A, B, wB = merge
        merged = (set(deps.get(A, ())) | set(deps.get(B, ()))) - {A, B}

        def ds(n):
            raw = merged if n == B else deps.get(n, ())
            return {B if d == A else d for d in raw} - {n}

        def wt(n):
            return wB if n == B else weight.get(n, 0.0)
    else:
        def ds(n):
            return deps.get(n, ())

        def wt(n):
            return weight.get(n, 0.0)
    memo = {}
    order, stack = [], [(root, False)]
    while stack:
        n, done = stack.pop()
        if done:
            best, nxt = 0.0, None
            for d in ds(n):
                if memo[d][0] > best:
                    best, nxt = memo[d][0], d
            memo[n] = (best + wt(n), nxt)
            continue
        if n in memo:
            continue
        memo[n] = (0.0, None)  # placeholder, overwritten on completion
        stack.append((n, True))
        for d in ds(n):
            if d not in memo:
                stack.append((d, False))
    path, n = [], root
    while n is not None:
        path.append(n)
        n = memo[n][1]
    return memo[root][0], path


class Plan:
    def __init__(self, mods, consts, mtime, dtime, floor, imports, closure):
        self.floor, self.mtime, self.dtime = floor, mtime, dtime
        node_of, self.unit = {}, {}
        mod_of_name = {name: mods[mi] for cid, mi, kind, name, deps in consts}
        for cid, mi, kind, name, deps in consts:
            # a realized constant (an equation lemma) is recorded where it was first realized, which need not be where
            # its declaration is: it belongs to its declaration's node
            base, plain = owner_full(name), owner_name(name)
            mod = mod_of_name.get(base) or mod_of_name.get(plain) or mods[mi]
            key = (mod, plain)
            node_of[cid] = key
            self.unit[key] = mod
        clo = {}

        def closure_of(m):
            if m not in clo:
                s = set()
                for i in imports(m):
                    s.add(i)
                    s |= closure_of(i)
                clo[m] = s
            return clo[m]

        sys.setrecursionlimit(100000)
        self.uses, self.used_by = defaultdict(set), defaultdict(set)
        self.dropped = 0
        for cid, mi, kind, name, deps in consts:
            a = node_of[cid]
            for d in deps:
                b = node_of.get(d)
                if b and b != a:
                    if b[0] != a[0] and b[0] not in closure_of(a[0]):
                        self.dropped += 1  # a name two modules share (a private one, a realized core one)
                        continue
                    self.uses[a].add(b)
                    self.used_by[b].add(a)
        self.members = defaultdict(set)
        for k, u in self.unit.items():
            self.members[u].add(k)
        self.base = {u: u for u in self.members}
        self.weight = {}
        for m in closure:
            self.weight[m] = mtime.get(m, floor)
        self.deps = defaultdict(set)
        for m in closure:
            if m in self.members:
                self.deps[m] = self.unit_deps(self.members[m], m)
            else:
                self.deps[m] = set(imports(m))  # an aggregator re-exports its imports

    def unit_deps(self, decls, me):
        out = set()
        for k in decls:
            for b in self.uses[k]:
                u = self.unit[b]
                if u != me:
                    out.add(u)
        return out

    def dt(self, key):
        return self.dtime.get((key[0], key[1].split(".")[-1]), 0.0)

    def part_time(self, mod, part):
        """The floor plus, for each original module whose declarations ``part`` holds, its share of that module's time
        above the floor (by traced time, or by count when none of its declarations is traced)."""
        by = defaultdict(list)
        for k in part:
            by[k[0]].append(k)
        t = self.floor
        for m, ks in by.items():
            above = max(0.0, self.mtime.get(m, self.floor) - self.floor)
            allk = self._all[m]
            traced = sum(self.dt(k) for k in allk)
            share = sum(self.dt(k) for k in ks) / traced if traced > 0 else len(ks) / max(1, len(allk))
            t += above * share
        return t

    def reaches(self, starts, target):
        seen, stack = set(), list(starts)
        while stack:
            m = stack.pop()
            if m == target:
                return True
            if m in seen:
                continue
            seen.add(m)
            stack.extend(self.deps.get(m, ()))
        return False

    def merge(self, A, B):
        for k in self.members[A]:
            self.unit[k] = B
        self.members[B] = self.members[B] | self.members.pop(A)
        self.deps[B] = (set(self.deps[A]) | set(self.deps[B])) - {A, B}
        for n, ds in self.deps.items():
            if A in ds:
                ds.discard(A)
                if n != B:
                    ds.add(B)
        del self.deps[A]
        self.weight.pop(A, None)
        self.weight[B] = self.part_time(None, self.members[B])

    def try_split(self, unit, need):
        """The deps and weights after cutting `need` out of `unit` into a new unit; returns (changes, new_name)."""
        rest = self.members[unit] - need
        lo = f"{self.base[unit]}#{len([u for u in self.base if self.base[u] == self.base[unit]])}"
        deps_lo = set()
        for k in need:
            for b in self.uses[k]:
                u = self.unit[b]
                if u != unit:
                    deps_lo.add(u)
        deps_rest = set()
        for k in rest:
            for b in self.uses[k]:
                u = self.unit[b]
                if u != unit:
                    deps_rest.add(u)
                elif b in need:
                    deps_rest.add(lo)
        importers = defaultdict(lambda: [False, False])
        for k in self.members[unit]:
            for user in self.used_by[k]:
                u = self.unit[user]
                if u == unit:
                    continue
                importers[u][0 if k in need else 1] = True
        return lo, rest, deps_lo, deps_rest, importers

    def apply(self, unit, need, lo, rest, deps_lo, deps_rest, importers):
        for k in need:
            self.unit[k] = lo
        self.members[lo], self.members[unit] = set(need), set(rest)
        self.base[lo] = self.base[unit]
        self.deps[lo], self.deps[unit] = deps_lo, deps_rest
        for u, (a, b) in importers.items():
            d = self.deps[u]
            d.discard(unit)
            if a:
                d.add(lo)
            if b:
                d.add(unit)
        self.weight[lo] = self.part_time(None, need)
        self.weight[unit] = self.part_time(None, rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--deps", required=True)
    ap.add_argument("--times", required=True, help="JSON {module: seconds}")
    ap.add_argument("--decls", required=True, help="parse_trace.py's JSON")
    ap.add_argument("--root", default="Proofs.Flock")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--splits", type=int, default=20, help="at most this many operations")
    ap.add_argument("--merges", action="store_true", help="also consider merging a module on the path into its importer")
    ap.add_argument("--max-elab", type=float, default=60.0, help="a merged module's time above the floor, at most")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.argv = ["x", a.repo]
    import graph
    mods, consts = load_deps(Path(a.deps))
    mtime = json.loads(Path(a.times).read_text())
    tr = json.loads(Path(a.decls).read_text()) if Path(a.decls).exists() else {}
    dtime = {(m, n): sum(v.values()) for m, r in tr.items() for n, v in r["decls"].items()}
    floor = a.floor if a.floor is not None else sorted(mtime.values())[len(mtime) // 20]
    cl = graph.closure(a.root)
    imp = {m: set(graph.imports(m)) for m in cl}
    w = {m: mtime.get(m, floor) for m in cl}
    cur_t, cur_p = longest(a.root, imp, w)
    print(f"floor {floor:.1f} s; current: {cur_t / 60:.1f} min over {len(cur_p)} modules")
    P = Plan(mods, consts, mtime, dtime, floor, graph.imports, cl)
    P._all = defaultdict(list)
    for k in P.unit:
        P._all[k[0]].append(k)
    sh_t, sh_p = longest(a.root, P.deps, P.weight)
    print(f"shaken: {sh_t / 60:.1f} min over {len(sh_p)} modules")
    splits = []
    for it in range(a.splits):
        t, p = longest(a.root, P.deps, P.weight)
        best = None
        for i in range(len(p) - 1):
            up, unit = p[i], p[i + 1]
            if len(P.members.get(unit, ())) < 2 or up not in P.members:
                continue
            need = {b for k in P.members[up] for b in P.uses[k] if P.unit.get(b) == unit}
            stack = list(need)
            while stack:
                k = stack.pop()
                for b in P.uses[k]:
                    if P.unit.get(b) == unit and b not in need:
                        need.add(b)
                        stack.append(b)
            if not need or need == P.members[unit]:
                continue
            lo, rest, dl, dr, imps = P.try_split(unit, need)
            saved = {x: (set(P.deps[x]) if x in P.deps else None) for x in [lo, unit, *imps]}
            savedw = {lo: P.weight.get(lo), unit: P.weight[unit]}
            savedm = {k: P.unit[k] for k in need}
            P.apply(unit, need, lo, rest, dl, dr, imps)
            nt, _ = longest(a.root, P.deps, P.weight)
            # undo
            for k, u in savedm.items():
                P.unit[k] = u
            P.members[unit] = set(P.members[unit]) | set(need)
            del P.members[lo]
            del P.base[lo]
            for x, d in saved.items():
                if d is None:
                    P.deps.pop(x, None)
                else:
                    P.deps[x] = d
            P.weight.pop(lo, None)
            P.weight[unit] = savedw[unit]
            if best is None or nt < best[0]:
                best = (nt, "split", unit, set(need), up)
        if a.merges:
            for i in range(len(p) - 1):
                B, A = p[i], p[i + 1]
                if A == a.root or not P.members.get(A) or not P.members.get(B):
                    continue
                wB = P.part_time(None, P.members[A] | P.members[B])
                if wB - floor > a.max_elab or P.reaches(set(P.deps[B]) - {A}, A):
                    continue
                nt, _ = longest(a.root, P.deps, P.weight, merge=(A, B, wB))
                if best is None or nt < best[0]:
                    best = (nt, "merge", A, B, None)
        if best is None or best[0] >= t - 1:
            break
        if best[1] == "merge":
            nt, _, A, B, _ = best
            names = sorted(n for _, n in P.members[A])
            origin = sorted({k[0] for k in P.members[A]})
            P.merge(A, B)
            splits.append({"op": "merge", "module": A, "into": B, "from_modules": origin, "moves": names,
                           "saves_s": round(t - nt, 1), "path_after_min": round(nt / 60, 2), "merged_s": round(P.weight[B], 1)})
            print(f"merge {it}: {A} ({len(names)} decls) into {B}: path {t / 60:.1f} -> {nt / 60:.1f} min")
            continue
        nt, _, unit, need, up = best
        lo, rest, dl, dr, imps = P.try_split(unit, need)
        P.apply(unit, need, lo, rest, dl, dr, imps)
        splits.append({"op": "split", "module": unit, "new_part": lo, "for_importer": up, "moves": sorted(n for _, n in need),
                       "from_modules": sorted({k[0] for k in need}),
                       "kept": len(rest), "saves_s": round(t - nt, 1), "path_after_min": round(nt / 60, 2),
                       "part_s": round(P.weight[lo], 1), "rest_s": round(P.weight[unit], 1)})
        print(f"split {it}: {unit} ({len(need)} decls to {lo}, {len(rest)} kept) for {up}: path {t / 60:.1f} -> {nt / 60:.1f} min")
    fin_t, fin_p = longest(a.root, P.deps, P.weight)
    if a.out:
        json.dump({"floor": floor, "current_min": cur_t / 60, "current_path": cur_p, "shaken_min": sh_t / 60,
                   "shaken_path": sh_p, "splits": splits, "final_min": fin_t / 60, "final_path": fin_p},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
