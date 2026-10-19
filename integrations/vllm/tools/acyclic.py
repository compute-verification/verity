"""ACYCLIC: whether each Call's Q_word units (`verity/partition/v1`, X = 16, W = 32) read each other in a cycle.

A unit reads another when one of its computing gates reads a gate the other owns (through structure, as `cut.CallGraph` lists
the reads).  `check_cut` holds a cut to the invariant and the width rule but not to an acyclic unit graph.  Core's partition
theorems need none (a unit is checked gate by gate in gate order, `compose_cone`), but POP1's population theorems take an
acyclic unit graph as a hypothesis, which a Q_word cut need not meet (`qword_not_acyclic`).  Per distinct Definition this tool
cuts once, as POPREAD does (core's `cut_word` on POPREAD's `LeanGraph`; a separable body is its members' cuts, which read no
other member), builds the units' read graph, finds its strongly connected components with more than one unit (Tarjan), and
names one shortest cycle with a gate read per step.  Per row it counts the Calls whose Definition's cut has such a component
and their share of the Program's units and gates.

A Program's units read units of their own Call and outputs of earlier Calls: Q_word applies only to a topological root
(`partition_object.order_violation`, checked here on descriptor rows), so every cycle of the whole Program's unit graph lies in
one Call, and the Program's components are its Calls' components, one set per activation.

    python tools/acyclic.py descriptor DESCRIPTOR.json.gz [--jobs N] [--log L.jsonl] [--rules L.jsonl] --name NAME --out OUT.json
    python tools/acyclic.py served BUILD.json [...] --out OUT.json      (a served-compiled Build's P_served, Calls as POPREAD counts them)
    python tools/acyclic.py table OUT.json [...] [--md OUT.md]
    python tools/acyclic.py instances DESCRIPTOR.json.gz --id ID [...] --out OUT.json   (each key of ID and the root Calls naming it)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import popread as PR  # noqa: E402

SCHEMA = "acyclic/v1"
#: the most gates a cyclic Definition may have for its example's gates to be named (one more pass over its operands)
NAME_GATES_BELOW = 8_000_000


def tarjan(U: int, start: list[int], nbr: list[int]) -> list[list[int]]:
    """The strongly connected components of more than one node of the graph on 0..U-1 whose node v's successors are
    `nbr[start[v]:start[v + 1]]`, iteratively."""
    index, low, on = [-1] * U, [0] * U, bytearray(U)
    stack: list[int] = []
    comps: list[list[int]] = []
    nxt = 0
    for s in range(U):
        if index[s] != -1:
            continue
        index[s] = low[s] = nxt
        nxt += 1
        stack.append(s)
        on[s] = 1
        work = [[s, start[s]]]
        while work:
            top = work[-1]
            v, i = top
            if i < start[v + 1]:
                top[1] = i + 1
                w = nbr[i]
                if index[w] == -1:
                    index[w] = low[w] = nxt
                    nxt += 1
                    stack.append(w)
                    on[w] = 1
                    work.append([w, start[w]])
                elif on[w] and index[w] < low[v]:
                    low[v] = index[w]
                continue
            work.pop()
            if work and low[v] < low[work[-1][0]]:
                low[work[-1][0]] = low[v]
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = 0
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    comps.append(sorted(comp))
    return comps


def shortest_cycle(comp: list[int], start: list[int], nbr: list[int]) -> list[int]:
    """A shortest cycle through `comp[0]` inside the component: [u0, u1, ..., uk] with an edge u_i -> u_{i+1} and uk -> u0."""
    inside, u0 = set(comp), comp[0]
    prev = {u0: -1}
    q = deque([u0])
    while q:
        v = q.popleft()
        for w in nbr[start[v]:start[v + 1]]:
            if w == u0:
                path = [v]
                while prev[path[-1]] != -1:
                    path.append(prev[path[-1]])
                return path[::-1]
            if w in inside and w not in prev:
                prev[w] = v
                q.append(w)
    raise AssertionError("a strongly connected component without a cycle through its first node")


def kept_gates(fn: Any, jobs: int = 1) -> list[int]:
    """The Call's gate index of each computing gate of `LeanGraph(fn)`, in its order: the gates its pass doesn't fold into
    structure (a constant, wiring, or a non-external gate whose operands are all constants)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from array import array
    from verity.ml.circuits.cut import WIRING, call_scope
    c, call = call_scope(fn)
    n = fn.gates
    spans = [(a, min(a + PR.CHUNK_GATES, n)) for a in range(0, n, PR.CHUNK_GATES)]
    PR._GATES[:] = [c, call, n, WIRING]
    try:
        if jobs > 1 and len(spans) > 1:
            gc.freeze()
            with ProcessPoolExecutor(min(jobs, len(spans)), mp_context=multiprocessing.get_context("fork")) as ex:
                chunks = list(ex.map(PR._gate_chunk, *zip(*spans)))
        else:
            chunks = [PR._gate_chunk(a, b) for a, b in spans]
    finally:
        PR._GATES.clear()
        gc.unfreeze()
    const, keep = bytearray(n), []
    for (a, _b), (_wb, flags, eb, ob) in zip(spans, chunks):
        end, ops = array("q"), array("q")
        end.frombytes(eb), ops.frombytes(ob)
        start = 0
        for i, fl in enumerate(flags):
            j = a + i
            mine = ops[start:end[i]]
            start = end[i]
            cj, fj = bool(fl & 1), bool(fl & 3)
            if not fj and not fl & 4 and len(mine) and all(const[o] for o in mine):
                cj = fj = True
            const[j] = cj
            if not fj:
                keep.append(j)
    return keep


def _gate_name(c: Any, call: Any, j: int) -> dict[str, Any]:
    g = c.gate(call.offset + j)
    return {"gate": j, "prim": g.prim.id, "scope": g.scope.spec.id[:160]}


def flat_cycles(fn: Any, jobs: int = 1) -> dict[str, Any]:
    """`cut_word` of `fn`'s whole body on its `LeanGraph`, its units' read graph and that graph's cyclic components."""
    import numpy as np
    from verity.ml.circuits.cut import cut_word
    g = PR.LeanGraph(fn, jobs=jobs)
    own, kinds, _head = cut_word(g, PR.UNIT_BITS)
    U = len(kinds)
    out: dict[str, Any] = {"units": U, "computing_gates": g.n, "edges": len(g.src)}
    if not len(g.src):
        return dict(out, unit_edges=0, sccs=0, scc_units=0, max_scc=0, sizes={}, example=None)
    o = np.asarray(own, dtype=np.int64)
    src, dst = np.frombuffer(g.src, dtype=np.int64), np.frombuffer(g.dst, dtype=np.int64)
    a, b = o[src], o[dst]
    cross = a != b
    pairs = np.unique(a[cross] * U + b[cross])                      # producer unit -> the unit that reads it, sorted by producer
    pa, pb = pairs // U, pairs % U
    start = np.searchsorted(pa, np.arange(U + 1)).tolist()
    nbr = pb.tolist()
    comps = tarjan(U, start, nbr)
    sizes = Counter(len(c) for c in comps)
    out.update(unit_edges=int(len(pairs)), sccs=len(comps), scc_units=sum(len(c) for c in comps),
               max_scc=max(sizes, default=0), sizes={str(k): v for k, v in sorted(sizes.items())}, example=None)
    if comps:
        comp = min(comps, key=len)
        cyc = shortest_cycle(comp, start, nbr)
        reads = []
        cand = np.flatnonzero(cross)
        ca, cb = a[cand], b[cand]
        for x, y in zip(cyc, cyc[1:] + cyc[:1]):
            e = int(cand[np.flatnonzero((ca == x) & (cb == y))[0]])
            reads.append({"from_unit": x, "to_unit": y, "producer": int(src[e]), "reader": int(dst[e])})
        first: dict[int, int] = {}
        for i, k in enumerate(g.outputs):
            first.setdefault(k, i)
        ex = {"component_size": len(comp), "cycle": cyc, "kinds": [kinds[u] for u in cyc], "reads": reads,
              "returned_at": {str(u): sorted(i for k, i in first.items() if own[k] == u)[:4] for u in cyc}}
        if fn.gates < NAME_GATES_BELOW:
            from verity.ml.circuits.cut import call_scope
            keep = kept_gates(fn, jobs)
            if len(keep) != g.n:
                raise AssertionError(f"{fn.id}: {len(keep)} kept gates against the LeanGraph's {g.n}")
            c, call = call_scope(fn)
            for r in reads:
                r["producer"] = _gate_name(c, call, keep[r["producer"]]) | {"computing": r["producer"]}
                r["reader"] = _gate_name(c, call, keep[r["reader"]]) | {"computing": r["reader"]}
        out["example"] = ex
    return out


def cycles(fn: Any, memo: dict[str, dict[str, Any]], jobs: int = 1) -> dict[str, Any]:
    """Per activation of `fn`: units, cyclic components (count, units in them, the largest, sizes) and one example, as Q_word
    evaluates it: a primitive is a unit or none, a separable body its members' cuts, any other body `flat_cycles`."""
    from verity.ml.circuits.cut import WIRING, separable
    got = memo.get(fn.id)
    if got is not None:
        return got
    if not hasattr(fn, "body"):
        got = {"units": 0 if (not fn.params or fn.id in WIRING) else 1, "sccs": 0, "scc_units": 0, "max_scc": 0, "sizes": {},
               "example": None, "how": "primitive"}
    elif (sep := separable(fn)) is not None:
        got = {"units": 0, "sccs": 0, "scc_units": 0, "max_scc": 0, "sizes": Counter(), "example": None, "how": "separable"}
        for k, (callee, m) in enumerate(sep):
            sub = cycles(callee, memo, jobs)
            got["units"] += m * sub["units"]
            got["sccs"] += m * sub["sccs"]
            got["scc_units"] += m * sub["scc_units"]
            got["max_scc"] = max(got["max_scc"], sub["max_scc"])
            for s, c in sub["sizes"].items():
                got["sizes"][s] += m * c
            if got["example"] is None and sub["example"] is not None:
                got["example"] = dict(sub["example"], via={"node": k, "definition": callee.id[:160]})
        got["sizes"] = dict(got["sizes"])
    else:
        got = dict(flat_cycles(fn, jobs), how="flat")
    memo[fn.id] = got
    return got


_PENDING: list[Any] = []


def _analyze(i: int, jobs: int = 1) -> dict[str, Any]:
    fn = _PENDING[i]
    t = time.monotonic()
    r = cycles(fn, {}, jobs)
    return {"definition": fn.id, "gates": fn.gates, **r, "seconds": round(time.monotonic() - t, 1)}


class Cycles:
    """Per distinct Definition (by `definition_digest`) its `cycles`, cached from earlier `--log`s (`rules`), else computed in
    `jobs` forked processes as POPREAD's `Units` cuts them: Definitions of `PR.BIG_GATES` or more one at a time, their graph
    resolved by all the jobs, the rest one per job.  Each result is appended to `log` as it lands."""

    def __init__(self, rules: Iterable[str | Path] = (), log: str | Path | None = None, jobs: int = 1,
                 gates_below: int | None = None, gates_from: int = 0) -> None:
        self.cache = load_logs(rules)
        self.log, self.jobs, self.gates_below, self.gates_from = log, jobs, gates_below, gates_from
        self.counts = Counter()

    def of(self, fns: dict[str, Any]) -> dict[str, str]:
        """name -> the `definition_digest` of `fns[name]`, each digest's result in `cache` unless its Definition's gates are
        outside [`gates_from`, `gates_below`) (pending)."""
        from verity_vllm.query.word import definition_digest
        digest = {fn.id: definition_digest(fn) for fn in fns.values()}
        todo = {fn.id: fn for fn in fns.values() if digest[fn.id] not in self.cache}
        self.counts["cached"] += len({fn.id for fn in fns.values()}) - len(todo)
        if self.gates_below is not None or self.gates_from:
            held = {i for i, fn in todo.items()
                    if fn.gates < self.gates_from or (self.gates_below is not None and fn.gates >= self.gates_below)}
            self.counts["pending"] += len(held)
            todo = {i: fn for i, fn in todo.items() if i not in held}
        for r in self._run(sorted(todo.values(), key=lambda fn: -fn.gates)):
            r["digest"] = digest[r["definition"]]
            self.cache[r["digest"]] = r
            self.counts["computed"] += 1
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps(r) + "\n")
            print(f"acyclic: {r['units']:>10,} units {r['sccs']:>8,} cyclic components {r['gates']:>12,} gates {r['seconds']:>8}s "
                  f"{r['definition'][:110]}", file=sys.stderr, flush=True)
        return {name: digest[fn.id] for name, fn in fns.items()}

    def _run(self, todo: list[Any]) -> Iterator[dict[str, Any]]:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        big = [fn for fn in todo if fn.gates >= PR.BIG_GATES]
        small = [fn for fn in todo if fn.gates < PR.BIG_GATES]
        try:
            for fn in big:
                _PENDING[:] = [fn]
                yield _analyze(0, self.jobs)
            _PENDING[:] = small
            if self.jobs <= 1 or len(small) <= 1:
                for i in range(len(small)):
                    yield _analyze(i)
                return
            gc.freeze()
            with ProcessPoolExecutor(min(self.jobs, len(small)), mp_context=multiprocessing.get_context("fork")) as ex:
                for f in as_completed([ex.submit(_analyze, i) for i in range(len(small))]):
                    yield f.result()
        finally:
            _PENDING.clear()
            gc.unfreeze()


def load_logs(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """digest -> `cycles` result, from `--log` JSONL files (the first result per digest)."""
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.setdefault(d["digest"], d)
    return out


def entries(apps: Counter, defs: dict[str, Any], digest: dict[str, str]) -> list[dict[str, Any]]:
    """Per distinct Definition of a row: its id, digest, Calls and gates per activation."""
    return [{"definition": defs[s].id[:240], "digest": digest[s], "calls": n, "gates": defs[s].gates} for s, n in apps.items()]


def summarize(name: str, defs: list[dict[str, Any]], per: dict[str, dict[str, Any]], extra: dict[str, Any]) -> dict:
    """A row's report from its `entries` and the `cycles` result per digest; a Definition without one is pending (its units
    unknown, its Calls and gates counted under `pending`)."""
    fams: dict[str, dict[str, Any]] = {}
    T = Counter()
    for e in defs:
        n, gates, r = e["calls"], e["gates"], per.get(e["digest"])
        fam = PR.base(e["definition"])
        f = fams.setdefault(fam, {"family": fam, "calls": 0, "gates": 0, "units": 0, "cyclic_calls": 0, "cyclic_gates": 0,
                                  "cyclic_units": 0, "sccs": 0, "scc_units": 0, "max_scc": 0, "sizes": Counter(), "definitions": 0,
                                  "cyclic_definitions": 0, "pending_definitions": 0, "pending_calls": 0, "example": None})
        f["calls"] += n
        f["gates"] += n * gates
        f["definitions"] += 1
        T.update(calls=n, gates=n * gates)
        if r is None:
            f["pending_definitions"] += 1
            f["pending_calls"] += n
            T.update(pending_definitions=1, pending_calls=n, pending_gates=n * gates)
            continue
        f["units"] += n * r["units"]
        T.update(units=n * r["units"])
        if r["sccs"]:
            f["cyclic_definitions"] += 1
            f["cyclic_calls"] += n
            f["cyclic_gates"] += n * gates
            f["cyclic_units"] += n * r["units"]
            f["sccs"] += n * r["sccs"]
            f["scc_units"] += n * r["scc_units"]
            f["max_scc"] = max(f["max_scc"], r["max_scc"])
            for s, c in r["sizes"].items():
                f["sizes"][s] += n * c
            T.update(cyclic_calls=n, cyclic_gates=n * gates, cyclic_units=n * r["units"], sccs=n * r["sccs"],
                     scc_units=n * r["scc_units"])
            if f["example"] is None or gates < f["example"]["gates"]:
                f["example"] = {"definition": e["definition"], "gates": gates, "units": r["units"], **r["example"]}
    rows = sorted(fams.values(), key=lambda f: (-f["cyclic_units"], -f["units"]))
    for f in rows:
        f["sizes"] = dict(sorted(f["sizes"].items(), key=lambda kv: int(kv[0])))
    tot = {k: T[k] for k in ("calls", "gates", "units", "cyclic_calls", "cyclic_gates", "cyclic_units", "sccs", "scc_units",
                             "pending_definitions", "pending_calls", "pending_gates")}
    tot.update(share_calls=T["cyclic_calls"] / T["calls"] if T["calls"] else 0.0,
               share_units=T["cyclic_units"] / T["units"] if T["units"] else 0.0,
               share_gates=T["cyclic_gates"] / T["gates"] if T["gates"] else 0.0,
               share_scc_units=T["scc_units"] / T["units"] if T["units"] else 0.0)
    keep = ("units", "sccs", "scc_units", "max_scc", "sizes", "example")
    out_defs = [dict(e, result=None if (r := per.get(e["digest"])) is None else {k: r[k] for k in keep}) for e in defs]
    return {"schema": SCHEMA, "name": name, **extra, "totals": tot, "families": rows,
            "cyclic_families": [f["family"] for f in rows if f["cyclic_calls"]], "definitions": out_defs}


def analyze_descriptor(path: str | Path, helper: Cycles, name: str) -> dict[str, Any]:
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.cut import QueryInapplicable
    from verity_vllm.program.boolean import load_program
    program, _digest = load_program(Path(path))
    order = PO.order_violation(program)
    calls = PO.calls(program)
    for k, fn in calls:
        if program.root.body.nodes[k].form not in ("call", "primitive"):
            raise QueryInapplicable(f"root node {k} is not a call or primitive node", {"call": k, "definition": fn.id})
    defs = {fn.id: fn for _k, fn in calls}
    per = helper.of(defs)
    return summarize(name, entries(Counter(fn.id for _k, fn in calls), defs, per), helper.cache,
                     {"adapter": "descriptor/v1", "topological": order is None, "order_violation": order,
                      "source": dict(helper.counts)})


def served_apps(build: dict[str, Any]) -> tuple[Counter, dict[str, Any]]:
    """A P_served's Calls per Definition, counted as POPREAD's `analyze_served` counts them, and the Definitions."""
    from verity_vllm.query.call_scope import specialization
    P = build["program"]
    width = {k: int(v["width"]) for k, v in P["values"].items()}
    specs: dict[str, dict[str, Any]] = {P["sampled"]["definition"]: P["sampled"]["statics"]}
    for plan in build["steps_plan"]:
        specs.update(dict([PR._call_spec(c, plan) for c in P["calls"]]))
        specs.update({a["definition"]: a["statics"] for r in plan["requests"] for a in r["attention"]})
        specs[plan["logits"]["definition"]] = plan["logits"]["statics"]
    defs = {spec: specialization(PR.base(spec), statics) for spec, statics in specs.items()}
    ret = {spec: d.signature()[1].leaves for spec, d in defs.items()}
    apps: Counter = Counter()
    for plan in build["steps_plan"]:
        tokens = int(plan["tokens"])
        for c in P["calls"]:
            spec, _statics = PR._call_spec(c, plan)
            apps[spec] += tokens * max(width[c["writes"][0]] // ret[spec], 1)
        for r in plan["requests"]:
            for a in r["attention"]:
                for att in P.get("attention") or ():
                    apps[a["definition"]] += max(width[att["q"]["value"]] // ret[a["definition"]], 1)
            apps[plan["logits"]["definition"]] += 1
            apps[P["sampled"]["definition"]] += 1
    return +apps, defs


def analyze_served(path: str | Path, helper: Cycles, name: str) -> dict[str, Any]:
    build = json.loads(Path(path).read_text())
    apps, defs = served_apps(build)
    per = helper.of(defs)
    return summarize(name, entries(apps, defs, per), helper.cache, {"adapter": "P_served", "program_digest": build.get("program_digest"),
                                                                     "topological": None, "source": dict(helper.counts)})


def merge(report: dict[str, Any], logs: Iterable[str | Path]) -> dict[str, Any]:
    """`report` resummarized with the results of `logs` (other runs' `--log`s) for its pending Definitions."""
    per = {e["digest"]: e["result"] for e in report["definitions"] if e["result"] is not None}
    for d, r in load_logs(logs).items():
        per.setdefault(d, r)
    extra = {k: v for k, v in report.items() if k not in ("schema", "name", "totals", "families", "cyclic_families", "definitions")}
    extra["merged_logs"] = [str(p) for p in logs]
    return summarize(report["name"], [{k: e[k] for k in ("definition", "digest", "calls", "gates")} for e in report["definitions"]],
                     per, extra)


def instances(path: str | Path, ids: Iterable[str]) -> list[dict[str, Any]]:
    """Per descriptor key whose Definition id is in `ids`: the root Calls naming it, its binding record, its body's node count
    and the SHA-256 of its entry, read off the descriptor without decoding it.  `codec.decode_program` keys a composite by its
    id and binding record and builds one Definition per key from that entry and its callees', so the Calls naming one key are
    activations of one Definition, which Q_word cuts once (step 1) and `Cycles` measures once."""
    import gzip
    import hashlib
    from verity.ml.circuits.codec import canonical_json
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        desc = json.load(f)
    defs, ids = desc["definitions"], set(ids)
    callee = (lambda i: desc["callees"][i]) if "callees" in desc else (lambda fid: fid)
    calls = Counter(callee(nd["fn"]) for nd in defs[desc["root"]]["body"]["nodes"])
    return [{"key": k, "id": d["id"], "bindings": d.get("bindings") or {}, "nodes": len(d["body"]["nodes"]) if "body" in d else 0,
             "root_calls": calls[k], "sha256": hashlib.sha256(canonical_json(d)).hexdigest()}
            for k, d in defs.items() if d["id"] in ids]


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render(reports: Iterable[dict[str, Any]]) -> str:
    out = ["| row | Calls | cyclic Calls | share of Calls / units / gates | cyclic components (units in them) | cyclic families |",
           "|---|---:|---:|---|---|---|"]
    for r in reports:
        t = r["totals"]
        fams = ", ".join(f"{f['family']} ({f['cyclic_calls']:,})" for f in r["families"] if f["cyclic_calls"]) or "-"
        pend = (f" ({t['pending_definitions']:,} Definitions, {t['pending_calls']:,} Calls pending)"
                if t.get("pending_definitions") else "")
        out.append(f"| {r['name']} | {t['calls']:,} | {t['cyclic_calls']:,}{pend} | {_pct(t['share_calls'])} / "
                   f"{_pct(t['share_units'])} / {_pct(t['share_gates'])} | {t['sccs']:,} ({t['scc_units']:,}) | {fams} |")
    out.append("")
    for r in reports:
        for f in r["families"]:
            if not f["cyclic_calls"]:
                continue
            ex = f["example"]
            reads = "; ".join(
                f"u{x['to_unit']} reads u{x['from_unit']} ("
                + (f"{x['reader']['prim']} <- {x['producer']['prim']}" if isinstance(x["reader"], dict) else f"g{x['reader']} <- g{x['producer']}")
                + ")" for x in ex["reads"])
            out.append(f"- {r['name']}: {f['family']}: {f['cyclic_definitions']}/{f['definitions']} Definitions, "
                       f"{f['cyclic_calls']:,}/{f['calls']:,} Calls, component sizes {f['sizes']}; e.g. {ex['definition'][:100]} "
                       f"({ex['units']} units): cycle {ex['cycle']} ({'/'.join(ex['kinds'])}): {reads}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("descriptor", "served"):
        x = sub.add_parser(cmd)
        x.add_argument("path")
        x.add_argument("--name", required=True)
        x.add_argument("--out", required=True)
        x.add_argument("--jobs", type=int, default=1)
        x.add_argument("--log", default=None, help="append each Definition's result here (JSONL) as it lands")
        x.add_argument("--rules", action="append", default=[], help="an earlier --log to reuse results from (repeatable)")
        x.add_argument("--gates-below", type=int, default=None, metavar="G",
                       help="compute only Definitions of fewer than G gates; the rest stay pending (see `merge`)")
        x.add_argument("--gates-from", type=int, default=0, metavar="G", help="compute only Definitions of at least G gates")
    m = sub.add_parser("merge", help="fill a report's pending Definitions from other runs' --logs, without loading the Program")
    m.add_argument("report")
    m.add_argument("--rules", action="append", default=[], required=True)
    m.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("reports", nargs="+")
    t.add_argument("--md", default=None)
    i = sub.add_parser("instances", help="each descriptor key of the given Definition ids and the root Calls naming it")
    i.add_argument("path")
    i.add_argument("--id", action="append", required=True, dest="ids")
    i.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "instances":
        got = instances(a.path, a.ids)
        Path(a.out).write_text(json.dumps(got, indent=1))
        for e in got:
            print(f"acyclic: {e['root_calls']:>10,} root Calls {e['nodes']:>6,} nodes {e['key'][:160]}", file=sys.stderr)
        return 0
    if a.cmd == "table":
        md = render(json.loads(Path(x).read_text()) for x in a.reports)
        if a.md:
            Path(a.md).write_text(md + "\n")
        print(md)
        return 0
    if a.cmd == "merge":
        Path(a.out).write_text(json.dumps(merge(json.loads(Path(a.report).read_text()), a.rules), indent=1))
        return 0
    helper = Cycles(a.rules, a.log, a.jobs, a.gates_below, a.gates_from)
    r = (analyze_descriptor if a.cmd == "descriptor" else analyze_served)(a.path, helper, a.name)
    Path(a.out).write_text(json.dumps(r, indent=1))
    t = r["totals"]
    print(f"acyclic: {r['name']}: {t['cyclic_calls']:,} of {t['calls']:,} Calls cyclic ({_pct(t['share_units'])} of units, "
          f"{_pct(t['share_gates'])} of gates), {t['sccs']:,} cyclic components", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
