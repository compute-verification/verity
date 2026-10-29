"""POPREAD: whether each unit's cross-unit reads are fixed by public structure or depend on a committed value.

A Q_word unit (`verity/partition/v1`, X = 16) lies inside one Call, a root node of the Program.  It reads other units through
its Call's operands and through its Definition's own wiring, and descriptor/v1 references are all static (affine, strided,
explicit, concat), so within one Program no read position depends on a value.  A value can still steer what a unit reads in
two ways, and this tool names both, per Call family:

* **VALUE-DEPENDENT**: the unit reads a Program input whose binding to other units' outputs is planned from committed values.
  Today that is a warm request's `prefix` input, bound to its producers' KV rows by prompt-block equality (`prefix_plan`), and
  `plan` tests a planner for it by changing committed values with the public launch context held fixed.
* **PUBLIC, selecting in the unit**: the unit reads every candidate at fixed positions and a selection primitive inside it
  (`Gather<T>x<V>`, `Select<T>`) picks one by a value.  The read set is public; the selection's index is named (the Call
  family or input that produces it).

`--workload-program` adds the activity the stop rule keys on committed tokens: a request's steps from 1 + LAG on run only if
no earlier token was an EOS, so their units are programmed (their reads stay public) but their activity is value-keyed.  Under
a stop rule with ignore_eos no request stops early, and the counts say what an EOS stop would key.

Each report also carries `egress`: E(u) = |O ∩ W(down{u})|, the declared output slots O in unit u's downstream cone, which
bounds what wrong units can put on the egress (Reach_inf(B) <= sum of E(u) over B <= |B| E_max).  It is a wiring property,
bounded per unit by its Call's cone, and given in slots and bits (a sampled id's bits are ceil(log2 V), its word's 32).

    python tools/popread.py descriptor DESCRIPTOR.json.gz [--units] [--workload-program workload_program.json] --out OUT.json
    python tools/popread.py served BUILD.json [--units] [--ignore-eos] [--lag N] --out OUT.json   (a served-compiled P_served)
    python tools/popread.py plan WORKLOAD.json [...] --planner prefix_plan.py --out OUT.json
    python tools/popread.py table OUT.json [...] [--md OUT.md] [--egress]

`--units` counts Q_word units per Definition (`Units`): from `--rules` (a manifest's unit-rule directory, or an earlier
`--units-log`) by content digest, else cut on a `LeanGraph` by core's `cut_word` in `--jobs` forked processes, with
`--verify-below G` also cutting small Definitions by the reference and `--recut-cached` checking cached rules.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCHEMA = "popread/v1"
UNIT_BITS = 16
INPUT = re.compile(r"^(Registered)?Input\d+_v\d+$")
SELECT = re.compile(r"^(?:Gather(?P<t>[A-Za-z0-9]*?)x(?P<v>\d+)(?:u\d+)?|Select(?P<s>[A-Za-z0-9]+))_v\d+$")
INDEX_PARAMS = ("token", "idx", "index", "c", "cond", "pred", "sel")
CONST = re.compile(r"^Const\d+\[")
#: input groups whose binding to other units' outputs is planned from committed values, and the value
VALUE_BOUND = {"prefix": "prompt_token_ids: the cached length C and each block's producer come from prompt-block equality "
                         "(prefix_plan.plan), so which KV rows the warm request reads depends on the prompts"}


def base(fid: str) -> str:
    return fid.split("{", 1)[0]


def group(name: str) -> str:
    """An input's group: its name without a request prefix (`r3.`) or an index (`peers_12`, `weights[4]`)."""
    name = re.sub(r"^r\d+\.", "", name)
    return re.sub(r"(_\d+|\[\d+\])+$", "", name.split(".")[0]) or name


def load(path: str | Path) -> dict[str, Any]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def members(node: dict[str, Any]) -> int:
    return int(node.get("n", 1)) if node["form"] in ("batch", "scan") else 1


def ref_parts(refs: dict[str, Any], nodes: list[dict[str, Any]]) -> Iterator[tuple[str, int]]:
    """The (space, index) of every part a reference sequence reads, aliases resolved."""
    stack = [refs]
    while stack:
        j = stack.pop()
        if "concat" in j:
            stack.extend(j["concat"])
        elif "alias" in j:
            k, a = j["alias"]
            stack.append(nodes[k]["args"][a]["refs"])
        elif "affine" in j:
            if j["affine"][3]:
                yield j["affine"][0], j["affine"][1]
        elif "strided" in j:
            if j["strided"][6]:
                yield j["strided"][0], j["strided"][1]
        elif "explicit" in j:
            yield from {(s, i) for s, i, _ in j["explicit"]}


def ref_runs(refs: dict[str, Any], nodes: list[dict[str, Any]]) -> Iterator[tuple[str, int, int]]:
    """(space, index, leaves) of each run a reference sequence reads, aliases resolved."""
    stack = [refs]
    while stack:
        j = stack.pop()
        if "concat" in j:
            stack.extend(j["concat"])
        elif "alias" in j:
            k, a = j["alias"]
            stack.append(nodes[k]["args"][a]["refs"])
        elif "affine" in j:
            yield j["affine"][0], j["affine"][1], j["affine"][3]
        elif "strided" in j:
            yield j["strided"][0], j["strided"][1], j["strided"][6]
        elif "explicit" in j:
            for s, i, _ in j["explicit"]:
                yield s, i, 1


class Walk:
    """A raw descriptor/v1's Definitions: gates per activation and the selection primitives inside, with what indexes them."""

    def __init__(self, desc: dict[str, Any]) -> None:
        self.defs, self.callees = desc["definitions"], desc["callees"]
        self._gates: dict[str, int] = {}
        self._sel: dict[str, Counter] = {}

    def callee(self, node: dict[str, Any]) -> str:
        return self.callees[node["fn"]]

    def gates(self, fid: str) -> int:
        got = self._gates.get(fid)
        if got is None:
            d = self.defs[fid]
            got = 1 if d["form"] == "primitive" else sum(members(n) * self.gates(self.callee(n)) for n in d["body"]["nodes"])
            self._gates[fid] = got
        return got

    def _label(self, nodes: list[dict[str, Any]], k: int) -> tuple[str, str]:
        fid = self.callee(nodes[k])
        return ("v", "const") if CONST.match(fid) else ("v", base(fid))

    def selects(self, fid: str) -> Counter:
        """{(selection primitive, frozenset of index sources): count per activation}, a source ("p", i) for the Definition's
        parameter i or ("v", family) for the body node that computes the index."""
        got = self._sel.get(fid)
        if got is not None:
            return got
        d, got = self.defs[fid], Counter()
        if d["form"] == "primitive":
            if SELECT.match(fid):
                names = [p[0] for p in d["params"]]
                i = next((names.index(n) for n in INDEX_PARAMS if n in names), 0)
                got[(fid, frozenset({("p", i)}))] = 1
        else:
            nodes = d["body"]["nodes"]
            for n in nodes:
                sub = self.selects(self.callee(n))
                if not sub:
                    continue
                m = members(n)
                for (prim, srcs), c in sub.items():
                    new: set = set()
                    for s in srcs:
                        if s[0] != "p":
                            new.add(s)
                            continue
                        for space, i in ref_parts(n["args"][s[1]]["refs"], nodes):
                            new.add(("p", i) if space == "p" else self._label(nodes, i))
                    got[(prim, frozenset(new))] += m * c
        self._sel[fid] = got
        return got


def root_selects(desc: dict[str, Any]) -> Counter:
    """A one-Definition Program's selections, an index source ("p", i) for the Definition's parameter i (its i-th Input node)
    or ("v", family) for the body node that computes it."""
    W = Walk(desc)
    nodes = W.defs[desc["root"]]["body"]["nodes"]
    param = {}
    for k, n in enumerate(nodes):
        if INPUT.match(W.callee(n)):
            param[k] = len(param)
    got: Counter = Counter()
    for k, n in enumerate(nodes):
        if k in param:
            continue
        for (prim, isrc), c in W.selects(W.callee(n)).items():
            new: set = set()
            for s in isrc:
                if s[0] != "p":
                    new.add(s)
                    continue
                for _space, i in ref_parts(n["args"][s[1]]["refs"], nodes):
                    new.add(("p", param[i]) if i in param else W._label(nodes, i))
            got[(prim, frozenset(new))] += members(n) * c
    return got


def select_kind(prim: str) -> tuple[str, int]:
    m = SELECT.match(prim)
    return (f"Gather{m['t']}x{m['v']}", int(m["v"])) if m["v"] else (f"Select{m['s']}", 2)


def input_names(desc: dict[str, Any], n_inputs: int) -> list[str]:
    """The root's input batches' names, in order: the correspondence's run inputs, then its weights."""
    corr = (desc.get("annotations") or {}).get("correspondence") or {}
    names = list(corr.get("inputs") or []) + [f"weight:{w}" for w in corr.get("weights") or []]
    return names if len(names) == n_inputs else [f"input_{k}" for k in range(n_inputs)]


def input_group(name: str) -> str:
    return "weight" if name.startswith("weight:") else group(name)


def classify(families: dict[str, dict[str, Any]], value_bound: dict[str, str]) -> None:
    for f in families.values():
        bound = sorted(g for g in f["reads"]["inputs"] if g in value_bound)
        f["verdict"] = "VALUE-DEPENDENT" if bound else "PUBLIC"
        f["value"] = "; ".join(f"{g}: {value_bound[g]}" for g in bound) or None


def _family(name: str) -> dict[str, Any]:
    return {"family": name, "calls": 0, "gates": 0, "units": 0, "eos_keyed": {"calls": 0, "gates": 0, "units": 0},
            "eos_step_keyed": {"calls": 0, "gates": 0, "units": 0}, "reads": {"inputs": set(), "calls": set()}, "selects": Counter()}


def _finish(out: dict[str, Any], families: dict[str, dict[str, Any]], value_bound: dict[str, str], have_units: bool) -> dict:
    classify(families, value_bound)
    T = {"calls": sum(f["calls"] for f in families.values()), "gates": sum(f["gates"] for f in families.values()),
         "units": sum(f["units"] for f in families.values()) if have_units else None}
    rows = []
    for f in sorted(families.values(), key=lambda f: -(f["units"] if have_units else f["gates"])):
        sel = defaultdict(lambda: {"per_family": 0, "index": set()})
        for (kind, V, idx), c in f["selects"].items():
            sel[(kind, V)]["per_family"] += c
            sel[(kind, V)]["index"].update(idx)
        rows.append({**{k: f[k] for k in ("family", "calls", "gates", "verdict", "value", "eos_keyed", "eos_step_keyed")},
                     "units": f["units"] if have_units else None,
                     "share_gates": f["gates"] / T["gates"] if T["gates"] else 0.0,
                     "share_units": (f["units"] / T["units"] if T["units"] else 0.0) if have_units else None,
                     "reads": {"inputs": sorted(f["reads"]["inputs"]), "calls": sorted(f["reads"]["calls"])},
                     "selects": [{"select": k, "V": V, "count": s["per_family"], "index": sorted(s["index"])}
                                 for (k, V), s in sorted(sel.items())]})

    def total(pred: Callable[[dict], bool], key: str = "") -> dict[str, Any]:
        sub = [r for r in rows if pred(r)]
        get = (lambda r, k: r[key][k]) if key else (lambda r, k: r[k])
        o = {k: sum(get(r, k) or 0 for r in sub) for k in ("calls", "gates", "units")}
        o["share_gates"] = o["gates"] / T["gates"] if T["gates"] else 0.0
        o["share_units"] = (o["units"] / T["units"] if T["units"] else 0.0) if have_units else None
        return o

    out.update(totals=T, families=rows, summary={
        "value_dependent": total(lambda r: r["verdict"] == "VALUE-DEPENDENT"),
        "in_unit_select": total(lambda r: any(s["V"] > 2 for s in r["selects"])),
        "eos_keyed": total(lambda r: True, "eos_keyed"),
        "eos_step_keyed": total(lambda r: True, "eos_step_keyed"),
    })
    return out


def step_of_calls(wp: dict[str, Any], n_calls: int) -> tuple[list[int] | None, int, str | None]:
    """Call ordinal -> its request-relative step, from a workload Program's chronology, whose node ranges index the request
    body's nodes after its Input batches (`program.workload`), and LAG; None when the ranges don't tile this Program's Calls."""
    ch, lag = wp.get("chronology") or {}, int((wp.get("lag") or {}).get("value") or 0)
    spans = [(m["nodes"][0], m["nodes"][1], int(m["s"])) for e in ch.get("engine_steps") or () for m in e["members"]]
    if len({m["request_id"] for e in ch.get("engine_steps") or () for m in e["members"]}) != 1:
        return None, lag, "the chronology has several requests: per-request steps need the component's own ranges"
    end = max((hi for _lo, hi, _s in spans), default=0)
    if end != n_calls:
        return None, lag, f"the chronology's node ranges end at {end}, not this Program's {n_calls} Calls"
    step = [0] * n_calls
    for lo, hi, s in spans:
        step[lo:hi] = [s] * (hi - lo)
    return step, lag, None


#: (circuit, the Call's activation, its gate count, wiring) for `_gate_chunk`'s workers, set before they fork
_GATES: list[Any] = []
#: gates per `_gate_chunk`, and the fewest gates `LeanGraph` resolves in forked workers
CHUNK_GATES = 1 << 16


def _gate_chunk(a: int, b: int) -> tuple[bytes, bytes, bytes, bytes]:
    """Gates a..b-1 of the Call: per gate its declared width, flags (1: a constant, no parameters; 2: wiring; 4: reads outside
    the Call), and the in-Call operands it reads (CSR: end offsets, operands), as `CallGraph` enumerates them."""
    from array import array
    from verity.ml.circuits.cut import WIDE_FAN_IN, wide_operands
    c, call, n, wiring = _GATES
    lo = call.offset
    hi = lo + n
    width, flags, end, ops = array("q"), bytearray(), array("q"), array("q")
    for j in range(a, b):
        g = c.gate(lo + j)
        p, o = g.prim, g.operands
        if len(o) > WIDE_FAN_IN:
            mine, ext = [x - lo for x in wide_operands(g, call) if lo <= x < hi], True
        else:
            mine = [x - lo for x in o if lo <= x < hi]
            ext = len(mine) != len(o)
        width.append(p.ret.w)
        flags.append((not p.params) | (p.id in wiring) << 1 | ext << 2)
        ops.extend(mine)
        end.append(len(ops))
    return width.tobytes(), bytes(flags), end.tobytes(), ops.tobytes()


class LeanGraph:
    """What core's `cut.cut_word` reads of a `cut.CallGraph` (`n`, `src`, `dst`, `outputs`, `width`), built without the
    reference's per-gate operand lists, Call-input reads and recompute keys: a 384-way expert gather makes those tens of KB a
    gate, hundreds of GB for one MoE expert Gemm.  The operands are resolved in `jobs` forked workers over ranges of gates
    (`_gate_chunk`), then one pass in order does what `CallGraph` does with them.  `same_as` holds it to the reference."""

    def __init__(self, fn: Any, wiring: frozenset[str] | None = None, jobs: int = 1) -> None:
        import multiprocessing
        from array import array
        from concurrent.futures import ProcessPoolExecutor
        from verity.ml.circuits.cut import WIRING, QueryInapplicable, call_scope
        from verity.ml.circuits.layout import resolve
        c, call = call_scope(fn)
        lo, n = call.offset, fn.gates
        spans = [(a, min(a + CHUNK_GATES, n)) for a in range(0, n, CHUNK_GATES)]
        _GATES[:] = [c, call, n, WIRING if wiring is None else wiring]
        try:
            if jobs > 1 and len(spans) > 1:
                gc.freeze()                                         # else a collection in each worker copies the shared heap
                with ProcessPoolExecutor(min(jobs, len(spans)), mp_context=multiprocessing.get_context("fork")) as ex:
                    chunks = list(ex.map(_gate_chunk, *zip(*spans)))
            else:
                chunks = [_gate_chunk(a, b) for a, b in spans]
        finally:
            _GATES.clear()
            gc.unfreeze()
        free, const = bytearray(n), bytearray(n)
        idx = array("q", [-1]) * n
        srcs: dict[int, tuple[int, ...]] = {}
        self.width, self.src, self.dst = array("q"), array("q"), array("q")
        k = 0
        for (a, _b), (wb, flags, eb, ob) in zip(spans, chunks):
            width, end, ops = array("q"), array("q"), array("q")
            width.frombytes(wb), end.frombytes(eb), ops.frombytes(ob)
            start = 0
            for i, fl in enumerate(flags):
                j = a + i
                mine = ops[start:end[i]]
                start = end[i]
                if mine and max(mine) >= j:
                    raise QueryInapplicable(f"{fn.id}: gate {j} reads gate {max(mine)}, not before it (the layout is not topological)",
                                            {"definition": fn.id, "gate": j, "reads": max(mine), "why": "not topological"})
                cj, fj = bool(fl & 1), bool(fl & 3)
                if not fj and not fl & 4 and mine and all(const[o] for o in mine):
                    cj = fj = True
                const[j], free[j] = cj, fj
                if fj:
                    s: set[int] = set()
                    for o in mine:
                        s.update(srcs.get(o, ()) if free[o] else (o,))
                    if s:
                        srcs[j] = tuple(sorted(s))
                    continue
                idx[j] = k
                self.width.append(width[i])
                seen: set[int] = set()
                for o in mine:
                    for u in (srcs.get(o, ()) if free[o] else (o,)):
                        if u not in seen:
                            seen.add(u)
                            self.src.append(idx[u])
                            self.dst.append(k)
                k += 1
        self.n = k
        self.outputs: list[int] = []
        for g in ([0] if not hasattr(fn, "body") else (resolve(call, r) - lo for r in fn.body.ret)):
            if 0 <= g < n:
                self.outputs.extend(idx[u] for u in (srcs.get(g, ()) if free[g] else (g,)))

    def same_as(self, ref: Any) -> bool:
        return (self.n, list(self.src), list(self.dst), list(self.outputs), list(self.width)) == (
            ref.n, list(ref.src), list(ref.dst), list(ref.outputs), list(ref.width))


def lean_units(fn: Any, memo: dict[str, int], jobs: int = 1) -> int:
    """`cut.evaluate_definition(fn, 16).units`, each body cut as a whole cut by core's `cut_word` on its `LeanGraph`."""
    from verity.ml.circuits.cut import WIRING, cut_word, separable
    got = memo.get(fn.id)
    if got is not None:
        return got
    if not hasattr(fn, "body"):
        u = 0 if (not fn.params or fn.id in WIRING) else 1
    elif (sep := separable(fn)) is not None:
        u = sum(lean_units(callee, memo, jobs) * m for callee, m in sep)
    else:
        u = len(cut_word(LeanGraph(fn, jobs=jobs), UNIT_BITS)[1])
    memo[fn.id] = u
    return u


#: the Definitions a `Units` pool cuts, set before its workers fork (a Definition need not pickle)
_PENDING: list[Any] = []


#: the fewest gates a `Units` cuts alone, its gates resolved by all its jobs; smaller Definitions are cut one per job
BIG_GATES = 1_000_000


def _cut_pending(i: int, verify_below: int, jobs: int = 1) -> dict[str, Any]:
    import time
    from verity.ml.circuits.cut import CallGraph, evaluate_definition, separable
    fn = _PENDING[i]
    t = time.monotonic()
    out = {"definition": fn.id, "gates": fn.gates, "units": lean_units(fn, {}, jobs), "seconds": round(time.monotonic() - t, 1)}
    if fn.gates <= verify_below:
        out["reference_units"] = evaluate_definition(fn, UNIT_BITS, {}).units
        if hasattr(fn, "body") and separable(fn) is None:
            out["graph_same"] = LeanGraph(fn).same_as(CallGraph(fn))
    return out


class Units:
    """Q_word{X=16} units per activation of each Definition: a unit rule cached under the Definition's content digest
    (`verity.ml.definitions.word`'s rule directories, whose cut code is core's `cut_word`, or this tool's JSONL `log`), else cut by
    `lean_units` in `jobs` forked processes, largest first, each result appended to `log` as it lands.  A Definition of at most
    `verify_below` gates is also cut by the reference (`evaluate_definition`, `CallGraph`); `recut` recuts cached ones too.
    Any disagreement is listed in `summary()["mismatches"]`."""

    def __init__(self, rules: Iterable[str | Path] = (), log: str | Path | None = None, jobs: int = 1, verify_below: int = 0,
                 recut: bool = False) -> None:
        self.cache: dict[str, tuple[int, str]] = {}
        for r in rules:
            self._load(Path(r))
        self.log, self.jobs, self.verify_below, self.recut = log, jobs, verify_below, recut
        self.counts = Counter()
        self.mismatches: list[dict[str, Any]] = []

    def _load(self, p: Path) -> None:
        for f in (sorted(p.glob("*/*.json")) if p.is_dir() else [p]):
            text = f.read_text()
            for d in ([json.loads(x) for x in text.splitlines() if x.strip()] if f.suffix == ".jsonl" else [json.loads(text)]):
                if "key" in d:
                    k, rule = d["key"], d["rule"]
                    too_large = any(v.get("class") == "too-large" for v in rule.get("violations") or ())
                    if k.get("query") != [UNIT_BITS, 32, 0] or too_large:
                        continue
                    self.cache.setdefault(k["digest"], (int(rule["units_per_call"]), str(p)))
                elif "digest" in d:
                    self.cache.setdefault(d["digest"], (int(d["units"]), str(p)))

    def of(self, fns: dict[str, Any]) -> dict[str, int]:
        """name -> units per activation of `fns[name]`."""
        from verity.ml.definitions.word import definition_digest
        digest = {fn.id: definition_digest(fn) for fn in fns.values()}
        by_id: dict[str, int] = {}
        todo: dict[str, Any] = {}
        for fn in fns.values():
            hit = self.cache.get(digest[fn.id])
            if hit is not None:
                by_id[fn.id] = hit[0]
                self.counts["cached"] += 1
            if hit is None or self.recut:
                todo[fn.id] = fn
        for r in self._cut(sorted(todo.values(), key=lambda fn: -fn.gates)):
            i = r["definition"]
            r["digest"] = digest[i]
            self.counts["cut"] += 1
            if "reference_units" in r:
                self.counts["verified_units"] += 1
                if r["reference_units"] != r["units"]:
                    self.mismatches.append({"definition": i, "lean": r["units"], "reference": r["reference_units"]})
            if "graph_same" in r:
                self.counts["verified_graphs"] += 1
                if not r["graph_same"]:
                    self.mismatches.append({"definition": i, "graph": "LeanGraph differs from CallGraph"})
            if i in by_id:
                self.counts["recut_cached"] += 1
                if by_id[i] != r["units"]:
                    self.mismatches.append({"definition": i, "lean": r["units"], "cached": by_id[i]})
            by_id[i] = r["units"]
            self.cache[r["digest"]] = (r["units"], "cut")
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps(r) + "\n")
            print(f"popread units: {r['units']:>10,} {r['gates']:>12,} gates {r['seconds']:>8}s {i[:120]}", file=sys.stderr,
                  flush=True)
        return {name: by_id[fn.id] for name, fn in fns.items()}

    def _cut(self, todo: list[Any]) -> Iterator[dict[str, Any]]:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        big = [fn for fn in todo if fn.gates >= BIG_GATES]
        small = [fn for fn in todo if fn.gates < BIG_GATES]
        try:
            for fn in big:
                _PENDING[:] = [fn]
                yield _cut_pending(0, self.verify_below, self.jobs)
            _PENDING[:] = small
            if self.jobs <= 1 or len(small) <= 1:
                for i in range(len(small)):
                    yield _cut_pending(i, self.verify_below)
                return
            gc.freeze()                                             # else a collection in each worker copies the shared heap
            with ProcessPoolExecutor(min(self.jobs, len(small)), mp_context=multiprocessing.get_context("fork")) as ex:
                for f in as_completed([ex.submit(_cut_pending, i, self.verify_below) for i in range(len(small))]):
                    yield f.result()
        finally:
            _PENDING.clear()
            gc.unfreeze()

    def summary(self) -> dict[str, Any]:
        return {"cached": self.counts["cached"], "cut": self.counts["cut"], "recut_cached": self.counts["recut_cached"],
                "verified_units": self.counts["verified_units"], "verified_graphs": self.counts["verified_graphs"],
                "verify_below_gates": self.verify_below, "rules": sorted({src for _u, src in self.cache.values() if src != "cut"}),
                "mismatches": self.mismatches}


def q_word_units(path: str | Path, units: Units | None = None) -> tuple[dict[int, int], int]:
    """Q_word{X=16, W=32} v1's units per Call (root node index), and the population (`partition_object.Partition`'s)."""
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.cut import QueryInapplicable
    from verity.ml.definitions.boolean import load_program
    program, _digest = load_program(Path(path))
    calls = PO.calls(program)
    for k, fn in calls:
        if program.root.body.nodes[k].form not in ("call", "primitive"):
            raise QueryInapplicable(f"root node {k} is not a call or primitive node", {"call": k, "definition": fn.id})
    per = (units or Units()).of({fn.id: fn for _k, fn in calls})
    out = {k: per[fn.id] for k, fn in calls}
    return out, sum(out.values())


def id_bits(fid: str, bindings: dict[str, Any], width: int) -> int:
    """The bits an egress leaf can carry: a sampled id's ceil(log2 V), else its declared width."""
    v = bindings.get("V")
    return (int(v) - 1).bit_length() if base(fid).startswith("TokenSelect") and v else width


class Egress:
    """E(u) over a Program's units, as a histogram over E = (bits, slots, declared bits) weighted by activations, gates and units,
    with the families at each E and each component's (a request's) share and largest E."""

    def __init__(self) -> None:
        self.hist: dict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
        self.fams: dict[tuple[int, int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        self.comps: dict[Any, dict[str, Any]] = {}

    def add(self, e: tuple[int, int, int], family: str, comp: Any, calls: int, gates: int, units: int) -> None:
        if calls <= 0:
            return
        family = "Const[...]" if CONST.match(family) else family
        for t in (self.hist[e], self.fams[e][family]):
            t[0], t[1], t[2] = t[0] + calls, t[1] + gates, t[2] + units
        c = self.comps.setdefault(comp, {"calls": 0, "gates": 0, "units": 0, "E_max": (0, 0, 0)})
        c["calls"], c["gates"], c["units"], c["E_max"] = c["calls"] + calls, c["gates"] + gates, c["units"] + units, max(c["E_max"], e)

    def report(self, have_units: bool, slots: dict[str, int], comp_slots: dict[Any, int], cone: str) -> dict[str, Any]:
        w = 2 if have_units else 1
        keys = sorted(self.hist)
        total = sum(self.hist[k][w] for k in keys) or 1
        E = lambda k: {"bits": k[0], "slots": k[1], "declared_bits": k[2]}

        def quantile(p: float) -> dict[str, int]:
            cum = 0
            for k in keys:
                cum += self.hist[k][w]
                if cum >= p * total:
                    return E(k)
            return E(keys[-1] if keys else (0, 0, 0))

        top = keys[-1] if keys else (0, 0, 0)
        at = sorted(self.fams[top].items(), key=lambda kv: (-kv[1][w], -kv[1][1], kv[0]))
        zero = self.hist.get((0, 0, 0), [0, 0, 0])
        comps = sorted(self.comps.items(), key=lambda kv: (-kv[1]["E_max"][0], -kv[1][("calls", "gates", "units")[w]]))
        live = [(c, v) for c, v in comps if comp_slots.get(c)]
        return {"weight": "units" if have_units else "gates", "cone": cone, **slots,
                "E_max": E(top), "median": quantile(0.5), "p99": quantile(0.99),
                "at_max": {"share": self.hist[top][w] / total if keys else 0.0, "calls": self.hist[top][0] if keys else 0,
                           "families": [{"family": f, "calls": v[0], "gates": v[1], "units": v[2] if have_units else None,
                                         "share": v[w] / total} for f, v in at]},
                "zero": {"share": zero[w] / total, "calls": zero[0], "gates": zero[1], "units": zero[2] if have_units else None},
                "components": {"with_egress": len(live), "without_egress": len(comps) - len(live),
                               "without_egress_calls": sum(v["calls"] for c, v in comps if not comp_slots.get(c)),
                               "per": [{"component": c, "slots": comp_slots[c], "E_max": E(v["E_max"]), "calls": v["calls"],
                                        "gates": v["gates"], "units": v["units"] if have_units else None,
                                        "share": v[("calls", "gates", "units")[w]] / total} for c, v in live[:64]]},
                "hist": [[*k, *self.hist[k]] for k in keys]}


def descriptor_egress(desc: dict[str, Any], W: Walk, units: dict[int, int] | None) -> dict[str, Any]:
    """E per root Call: the declared output leaves (the root's `ret`) written by it or any Call downstream of it, one reverse
    sweep over the topologically ordered root body.  Components join Calls that read one another, so two share only inputs."""
    from verity.ml.circuits.types import type_from_json
    root = W.defs[desc["root"]]
    nodes = root["body"]["nodes"]
    n = len(nodes)
    is_in = [bool(INPUT.match(W.callee(x))) for x in nodes]
    group: dict[int, int] = {}
    groups: list[list[int]] = []                                    # per egress node: leaves, bits, declared bits
    for space, k, count in ref_runs(root["body"]["ret"], nodes):
        if space != "n" or not count:
            continue
        fid = W.callee(nodes[k])
        widths = type_from_json(desc["types"][nodes[k]["out"]]).leaf_widths()
        width = max(widths) if len(set(widths)) == 1 else None
        if width is None:
            raise ValueError(f"declared output node {k} ({fid[:80]}) has leaves of several widths")
        if k not in group:
            group[k] = len(groups)
            groups.append([0, 0, 0])
        g = groups[group[k]]
        g[0], g[1], g[2] = g[0] + count, g[1] + count * id_bits(fid, W.defs[fid].get("bindings") or {}, width), g[2] + count * width
    acc = [0] * n
    for k, gi in group.items():
        acc[k] |= 1 << gi
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in range(n - 1, -1, -1):
        if is_in[k]:
            continue
        m = acc[k]
        for a in nodes[k]["args"]:
            for space, i in ref_parts(a["refs"], nodes):
                if space != "n" or is_in[i]:
                    continue
                acc[i] |= m
                ri, rk = find(i), find(k)
                if ri != rk:
                    parent[ri] = rk
    memo: dict[int, tuple[int, int, int]] = {}

    def value(m: int) -> tuple[int, int, int]:
        got = memo.get(m)
        if got is None:
            sel = [groups[g] for g in range(len(groups)) if m >> g & 1]
            got = memo[m] = (sum(g[1] for g in sel), sum(g[0] for g in sel), sum(g[2] for g in sel))
        return got

    eg = Egress()
    for k in range(n):
        if not is_in[k]:
            fid = W.callee(nodes[k])
            eg.add(value(acc[k]), base(fid), find(k), 1, W.gates(fid) * members(nodes[k]), (units or {}).get(k, 0))
    comp_slots: dict[int, int] = defaultdict(int)
    for k, gi in group.items():
        comp_slots[find(k)] += groups[gi][0]
    bits = sorted({(g[1] // g[0], g[2] // g[0]) for g in groups if g[0]})
    return eg.report(units is not None, {"slots": sum(g[0] for g in groups), "bits": sum(g[1] for g in groups),
                                         "declared_bits": sum(g[2] for g in groups), "slot_bits": [list(b) for b in bits]},
                     dict(comp_slots), "Call: a unit's E(u) is at most its Call's, the egress any Call downstream of it writes")


def analyze_descriptor(desc: dict[str, Any], *, units: dict[int, int] | None = None, workload_program: dict | None = None,
                       value_bound: dict[str, str] = VALUE_BOUND) -> dict[str, Any]:
    W = Walk(desc)
    nodes = W.defs[desc["root"]]["body"]["nodes"]
    is_in = [bool(INPUT.match(W.callee(n))) for n in nodes]
    names = input_names(desc, sum(is_in))
    label: list[str] = []
    it = iter(names)
    for k, n in enumerate(nodes):
        label.append(f"input:{input_group(next(it))}" if is_in[k] else f"call:{base(W.callee(n))}")
    n_calls = len(nodes) - sum(is_in)
    step, lag, why = step_of_calls(workload_program, n_calls) if workload_program else (None, 0, "no workload Program given")
    fams: dict[str, dict[str, Any]] = {}
    by_step: dict[int, dict[str, Any]] = {}
    ordinal = -1
    for k, n in enumerate(nodes):
        if is_in[k]:
            continue
        ordinal += 1
        fid = W.callee(n)
        f = fams.setdefault(base(fid), _family(base(fid)))
        g, u = W.gates(fid) * members(n), (units or {}).get(k, 0)
        f["calls"] += 1
        f["gates"] += g
        f["units"] += u
        if step is not None:
            b = by_step.setdefault(step[ordinal], {"s": step[ordinal], "calls": 0, "gates": 0, "units": 0, "families": {}})
            b["calls"], b["gates"], b["units"] = b["calls"] + 1, b["gates"] + g, b["units"] + u
            bf = b["families"].setdefault(base(fid), {"calls": 0, "gates": 0, "units": 0})
            bf["calls"], bf["gates"], bf["units"] = bf["calls"] + 1, bf["gates"] + g, bf["units"] + u
        if step is not None and step[ordinal] >= 1 + lag:
            for key, v in (("calls", 1), ("gates", g), ("units", u)):
                f["eos_keyed"][key] += v
                f["eos_step_keyed"][key] += v
        srcs = [{label[i] for _s, i in ref_parts(a["refs"], nodes)} for a in n["args"]]
        for s in srcs:
            for lab in s:
                kind, _, what = lab.partition(":")
                f["reads"]["inputs" if kind == "input" else "calls"].add(what)
        for (prim, isrc), c in W.selects(fid).items():
            idx = set()
            for s in isrc:
                idx |= srcs[s[1]] if s[0] == "p" else {f"in-call:{s[1]}"}
            kind, V = select_kind(prim)
            f["selects"][(kind, V, frozenset(idx))] += c * members(n)
    ignore = bool(((workload_program or {}).get("stop_rule") or {}).get("ignore_eos"))
    out = {"schema": SCHEMA, "adapter": "descriptor/v1", "root": desc["root"][:160],
           "inputs": dict(Counter(input_group(x) for x in names)), "value_bound": value_bound,
           "eos": _eos(lag, step is not None, ignore, why),
           "by_step": [b for _s, b in sorted(by_step.items())],
           "egress": descriptor_egress(desc, W, units)}
    return _finish(out, fams, value_bound, units is not None)


IGNORE_EOS = "the row serves with ignore_eos, so every request runs to max_tokens"


def _eos(lag: int, counted: bool, ignore: bool, why: str | None) -> dict[str, Any]:
    """`counted`: the keyed counts are filled (what an EOS stop would key); `applied`: the workload's stop rule honors EOS, so they are."""
    return {"lag": lag, "keyed_from_step": 1 + lag, "counted": counted, "applied": counted and not ignore,
            "why_not": why if not counted else (IGNORE_EOS if ignore else None)}


def _value_label(r: dict[str, Any], producer: dict[str, str]) -> str:
    if "weight" in r:
        return "input:weight"
    v = r.get("value", "")
    return f"input:{v.split('/', 1)[1]}" if v.startswith("c00.input/") else f"call:{producer.get(v, '?')}"


def _call_spec(c: dict[str, Any], plan: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """A P_served Call's Definition at one step: its own, or the step plan's for its Gemm shape or its MoE key."""
    if "definition" in c:
        return c["definition"], c["statics"]
    g = plan["moe"][c["moe"]] if "moe" in c else plan["gemms"][",".join(str(x) for x in c["gemm"])]
    return g["definition"], g["statics"]


def served_live(P: dict[str, Any]) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """The template's Calls ("c", i) and attention layers ("a", l) whose writes reach an attention's k or v, which a request's later
    rows read (live at every row), and those that reach k, v or the logits' x (live at the row a request samples)."""
    writes = {("c", i): tuple(c.get("writes") or ()) for i, c in enumerate(P["calls"])}
    writes.update({("a", l): (a["name"],) for l, a in enumerate(P.get("attention") or ())})
    readers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for i, c in enumerate(P["calls"]):
        for r in c["reads"]:
            if "value" in r:
                readers[r["value"]].append(("c", i))
    for l, a in enumerate(P.get("attention") or ()):
        for s in ("q", "k", "v"):
            readers[a[s]["value"]].append(("a", l))
    kv = {a[s]["value"] for a in P.get("attention") or () for s in ("k", "v")}

    def reach(targets: set[str]) -> set[tuple[str, int]]:
        live: set[tuple[str, int]] = set()
        grew = True
        while grew:
            grew = False
            for node in reversed(list(writes)):
                if node not in live and any(w in targets or any(x in live for x in readers[w]) for w in writes[node]):
                    live.add(node)
                    grew = True
        return live

    return reach(kv), reach(kv | {P["logits"]["x"]["value"]})


SERVED_CONE = ("row: a unit of request r at step s has E = r's sampled ids at steps >= s if it reaches a k/v value or (at r's sampled "
               "row) the logits, else 0; a sampled id feeds r's next step's input ids, an edge the engine binds")


def analyze_served(build: dict[str, Any], *, units: "Units | bool" = False, ignore_eos: bool = False, lag: int = 0) -> dict[str, Any]:
    """A served-compiled Build's P_served: per step, each Call's Definition applied written width / its return leaves times per
    token row, attention once per query row and layer, the logits and the sampled id once per request.  The plan runs every
    request to max_tokens, so a request's rows from its own step 1 + LAG on are the activity an EOS stop would key
    (`eos_keyed`), and every row of an engine step holding one of them is in a batch whose shape that stop would change
    (`eos_step_keyed`: the step's tokens, padding, capture mode, Gemm M and attention splits).  `ignore_eos`: the workload serves
    with it (the Build doesn't record the sampling), so no stop happens and both counts are what one would key."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.program import Program
    from verity.ml.definitions.call_scope import specialization

    P = build["program"]
    width = {k: int(v["width"]) for k, v in P["values"].items()}
    producer = {w: c.get("base") or "Gemm" for c in P["calls"] for w in c.get("writes") or ()}
    for a in P.get("attention") or ():
        producer[a["name"]] = "Attention"
    producer["logits"] = base(P["logits"].get("name") or "logits")
    arrive = {r["request_id"]: int(r["arrive"]) for r in (build.get("workload") or {}).get("schedule") or ()}
    specs: dict[str, dict[str, Any]] = {P["sampled"]["definition"]: P["sampled"]["statics"]}
    for plan in build["steps_plan"]:
        specs.update(dict([_call_spec(c, plan) for c in P["calls"]]))
        specs.update({a["definition"]: a["statics"] for r in plan["requests"] for a in r["attention"]})
        specs[plan["logits"]["definition"]] = plan["logits"]["statics"]
    defs = {spec: specialization(base(spec), statics) for spec, statics in specs.items()}
    helper = (units if isinstance(units, Units) else Units()) if units else None
    unit_of = helper.of(defs) if helper else {}
    memo: dict[str, Any] = {}

    def info(spec: str, statics: dict[str, Any]) -> dict[str, Any]:
        got = memo.get(spec)
        if got is None:
            d = defs[spec]
            got = memo[spec] = {"gates": d.gates, "ret": d.signature()[1].leaves, "selects": root_selects(encode_program(Program(d))),
                                "units": unit_of.get(spec, 0)}
        return got

    fams: dict[str, dict[str, Any]] = {}

    def add(fam: str, spec: str, statics: dict[str, Any], apps: int, reads: list[dict[str, Any]], eos_apps: int,
            step_keyed: bool) -> None:
        if apps <= 0:
            return
        i, f = info(spec, statics), fams.setdefault(fam, _family(fam))
        f["calls"] += apps
        f["gates"] += apps * i["gates"]
        f["units"] += apps * i["units"]
        for field, n in (("eos_keyed", eos_apps), ("eos_step_keyed", apps if step_keyed else 0)):
            for key, v in (("calls", n), ("gates", n * i["gates"]), ("units", n * i["units"])):
                f[field][key] += v
        labs = [_value_label(r, producer) for r in reads]
        for lab in labs:
            kind, _, what = lab.partition(":")
            f["reads"]["inputs" if kind == "input" else "calls"].add(what)
        for (prim, isrc), c in i["selects"].items():
            idx = {labs[s[1]] if s[0] == "p" and s[1] < len(labs) else f"in-call:{s[1]}" for s in isrc}
            kind, V = select_kind(prim)
            f["selects"][(kind, V, frozenset(idx))] += apps * c

    live_kv, live_x = served_live(P)
    steps_of: dict[str, list[int]] = defaultdict(list)
    for plan in build["steps_plan"]:
        for r in plan["requests"]:
            steps_of[r["request_id"]].append(int(plan["step"]))
    sm = P["sampled"]
    w_id = defs[sm["definition"]].signature()[1].leaf_widths()[0]
    b_id = id_bits(sm["definition"], sm["statics"], w_id)
    eg = Egress()

    def egress(fam: str, spec: str, statics: dict[str, Any], rid: str, s: int, live: int, dead: int) -> None:
        i, n = info(spec, statics), sum(t >= s for t in steps_of[rid])
        eg.add((n * b_id, n, n * w_id), fam, rid, live, live * i["gates"], live * i["units"])
        eg.add((0, 0, 0), fam, rid, dead, dead * i["gates"], dead * i["units"])

    for plan in build["steps_plan"]:
        s, tokens = int(plan["step"]), int(plan["tokens"])
        keyed = {r["request_id"]: s - arrive.get(r["request_id"], 0) >= 1 + lag for r in plan["requests"]}
        eos_rows = sum(int(r["tokens"]) for r in plan["requests"] if keyed[r["request_id"]])
        any_keyed = any(keyed.values())
        pad = tokens - sum(int(r["tokens"]) for r in plan["requests"])
        for ci, c in enumerate(P["calls"]):
            spec, statics = _call_spec(c, plan)
            per_row = max(width[c["writes"][0]] // info(spec, statics)["ret"], 1)
            add(base(spec), spec, statics, tokens * per_row, c["reads"], eos_rows * per_row, any_keyed)
            for r in plan["requests"]:
                rows = int(r["tokens"])
                live = rows if ("c", ci) in live_kv else int(("c", ci) in live_x)
                egress(base(spec), spec, statics, r["request_id"], s, live * per_row, (rows - live) * per_row)
            egress(base(spec), spec, statics, "padding", s, 0, pad * per_row)
        for r in plan["requests"]:
            k, rid, rows = keyed[r["request_id"]], r["request_id"], int(r["tokens"])
            for row, a in enumerate(r["attention"]):
                for l, att in enumerate(P.get("attention") or ()):
                    per = max(width[att["q"]["value"]] // info(a["definition"], a["statics"])["ret"], 1)
                    add(base(a["definition"]), a["definition"], a["statics"], per, [att["q"], att["k"], att["v"]], per * k, any_keyed)
                    live = ("a", l) in live_kv or (row == rows - 1 and ("a", l) in live_x)
                    egress(base(a["definition"]), a["definition"], a["statics"], rid, s, per * live, per * (not live))
            lg = plan["logits"]
            add(base(lg["definition"]), lg["definition"], lg["statics"], 1, [P["logits"]["x"], P["logits"]["w"]], int(k), any_keyed)
            add(sm["base"], sm["definition"], sm["statics"], 1, [{"value": "logits"}], int(k), any_keyed)
            egress(base(lg["definition"]), lg["definition"], lg["statics"], rid, s, 1, 0)
            egress(sm["base"], sm["definition"], sm["statics"], rid, s, 1, 0)
    n_ids = sum(len(v) for v in steps_of.values())
    out = {"schema": SCHEMA, "adapter": "P_served", "program_digest": build.get("program_digest"),
           "schedule": build.get("workload"), "value_bound": VALUE_BOUND,
           "inputs": {"ids": 1, "positions": 1, "weight": len({r["weight"] for c in P["calls"] for r in c["reads"] if "weight" in r})},
           "plan_inputs": "steps_plan reads the declared schedule only (arrive, prompt_tokens, max_tokens) and the dispatch "
                          "(capture sizes, max_num_seqs, split_cap, sms): no token value",
           "eos": _eos(lag, True, ignore_eos, None),
           "egress": eg.report(bool(units), {"slots": n_ids, "bits": n_ids * b_id, "declared_bits": n_ids * w_id,
                                             "slot_bits": [[b_id, w_id]]}, {r: len(v) for r, v in steps_of.items()}, SERVED_CONE)}
    if helper:
        out["units_source"] = helper.summary()
    return _finish(out, fams, VALUE_BOUND, bool(units))


def load_planner(path: str | Path, name: str = "plan") -> Callable[[dict], Any]:
    spec = importlib.util.spec_from_file_location(f"popread_planner_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def _variants(w: dict[str, Any], seed: int) -> Iterator[tuple[str, dict[str, Any]]]:
    """The workload with its committed values changed and its public launch context (lengths, arrivals, engine) held: every
    prompt's ids replaced by fresh ones, and every prompt made a copy of the first's prefix."""
    rng = random.Random(seed)
    reqs = w.get("requests") or []
    if not any(isinstance(r.get("prompt_token_ids"), list) for r in reqs):
        return
    fresh = [dict(r, prompt_token_ids=[rng.randrange(1, 1 << 15) for _ in r["prompt_token_ids"]]) for r in reqs]
    yield "distinct prompts", dict(w, requests=fresh)
    first = max(reqs, key=lambda r: len(r["prompt_token_ids"]))["prompt_token_ids"]
    same = [dict(r, prompt_token_ids=list(first[:len(r["prompt_token_ids"])])) for r in reqs]
    yield "one shared prompt", dict(w, requests=same)


def plan_check(w: dict[str, Any], planner: Callable[[dict], Any], seed: int = 0) -> dict[str, Any]:
    """Whether `planner(workload)` (the Program shapes and input bindings it plans) changes when only committed values change."""
    got = planner(w)
    changed = [{"variant": name, "plan": p} for name, v in _variants(w, seed) if (p := planner(v)) != got]
    return {"plan": got, "verdict": "VALUE-DEPENDENT" if changed else "PUBLIC",
            "value": "prompt_token_ids" if changed else None, "changed_by": changed,
            "held": ["prompt_len", "max_tokens", "arrive_step", "engine"]}


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def render(reports: Iterable[dict[str, Any]]) -> str:
    out = []
    for r in reports:
        if "families" not in r:
            out.append(f"### {r.get('name', '?')}: plan\n\n{r.get('verdict')} (value: {r.get('value')}); "
                       f"changed by: {', '.join(c['variant'] for c in r.get('changed_by') or []) or 'nothing'}\n")
            continue
        t, s = r["totals"], r["summary"]
        out.append(f"### {r.get('name', '?')}\n\n{t['calls']:,} Calls, {t['gates']:,} gates"
                   + (f", {t['units']:,} Q_word units" if t["units"] is not None else "") + "\n")
        if u := r.get("units_source"):
            out.append(f"Units: {u['cached']} Definitions from cached rules, {u['cut']} cut ({u['recut_cached']} of them recut "
                       f"against their rule); {u['verified_units']} also cut by the reference, {len(u['mismatches'])} disagree.\n")
        out.append("| family | calls | units | share units | gates | share gates | cross-unit reads | in-unit selection (index) |")
        out.append("|---|---:|---:|---:|---:|---:|---|---|")
        for f in _fold_consts(r["families"]):
            sel = "; ".join(f"{x['select']} x{x['count']:,} by {', '.join(x['index'])}" for x in f["selects"]) or "-"
            reads = f["verdict"] + (f" ({f['value']})" if f["value"] else "")
            out.append(f"| {f['family']} | {f['calls']:,} | {'-' if f['units'] is None else format(f['units'], ',')} | "
                       f"{_fmt(f['share_units'])} | {f['gates']:,} | {_fmt(f['share_gates'])} | {reads} | {sel} |")
        vd, sel = s["value_dependent"], s["in_unit_select"]
        out.append(f"\nVALUE-DEPENDENT: {_fmt(vd['share_units'])} of units, {_fmt(vd['share_gates'])} of gates. "
                   f"PUBLIC with an in-unit gather (V > 2): {_fmt(sel['share_units'])} of units, {_fmt(sel['share_gates'])} of gates. "
                   + _render_eos(r["eos"], s) + "\n")
        if e := r.get("egress"):
            out.append(_render_egress_line(e) + "\n")
    return "\n".join(out)


def _families_at_max(e: dict[str, Any], top: int = 3) -> str:
    fams = e["at_max"]["families"]
    names = ", ".join(f["family"] for f in fams[:top]) + (f", +{len(fams) - top}" if len(fams) > top else "")
    return f"{_fmt(e['at_max']['share'])} in {len(fams)} families ({names})"


def _render_egress_line(e: dict[str, Any]) -> str:
    per = ", ".join(f"{c['component']}: {c['E_max']['bits']}" for c in e["components"]["per"][:8])
    return (f"Egress: {e['slots']:,} declared slots, {e['bits']:,} bits ({e['declared_bits']:,} as declared words). "
            f"E_max = {e['E_max']['slots']:,} slots = {e['E_max']['bits']:,} bits; by {e['weight']}, median "
            f"{e['median']['bits']:,} bits, p99 {e['p99']['bits']:,}; at E_max {_families_at_max(e)}; E = 0: {_fmt(e['zero']['share'])}. "
            f"{e['components']['with_egress']} component(s) with egress, E_max bits each: {per}.")


def render_egress(reports: Iterable[dict[str, Any]]) -> str:
    out = ["| Program | egress slots (bits/slot) | E_max slots / bits | median / p99 bits | at E_max (share of weight, families) | "
           "E = 0 | E_max bits per component |", "|---|---:|---:|---:|---|---:|---|"]
    for r in reports:
        if not (e := r.get("egress")):
            continue
        per = ", ".join(str(c["E_max"]["bits"]) for c in e["components"]["per"][:8])
        bits = "/".join(str(b) for b, _w in e["slot_bits"])
        out.append(f"| {r.get('name', '?')} | {e['slots']:,} ({bits}) | {e['E_max']['slots']:,} / {e['E_max']['bits']:,} | "
                   f"{e['median']['bits']:,} / {e['p99']['bits']:,} | {_families_at_max(e)} | {_fmt(e['zero']['share'])} | {per} |")
    return "\n".join(out)


def _fold_consts(fams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The families with every `Const<W>[...]` Definition summed into one row (a Program holds one per literal it uses)."""
    consts = [f for f in fams if CONST.match(f["family"])]
    if len(consts) < 2:
        return fams

    def total(k: str) -> Any:
        return None if any(f[k] is None for f in consts) else sum(f[k] for f in consts)

    row = {k: total(k) for k in ("calls", "units", "share_units", "gates", "share_gates")}
    verdicts = sorted({f["verdict"] for f in consts}, reverse=True)
    row.update(family=f"Const[...] ({len(consts)} Definitions)", verdict="/".join(verdicts),
               value="; ".join(sorted({f["value"] for f in consts if f["value"]})) or None, selects=[])
    return [f for f in fams if not CONST.match(f["family"])] + [row]


def _render_eos(e: dict[str, Any], s: dict[str, Any]) -> str:
    if not e.get("counted", e.get("applied")):
        return f"EOS: {e['why_not']}."
    own, step = s["eos_keyed"], s.get("eos_step_keyed") or s["eos_keyed"]
    keyed = (f"{_fmt(own['share_units'])} of units ({_fmt(own['share_gates'])} of gates) are a request's steps >= "
             f"{e['keyed_from_step']}, and {_fmt(step['share_units'])} of units ({_fmt(step['share_gates'])} of gates) are in "
             f"an engine step holding one")
    return f"Activity keyed on EOS: {keyed}." if e["applied"] else f"EOS: {e['why_not']}; were EOS honored, {keyed}."


def _bound(items: list[str]) -> dict[str, str]:
    vb = dict(VALUE_BOUND)
    for it in items:
        g, _, why = it.partition("=")
        vb[g] = why
    return vb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("descriptor")
    d.add_argument("descriptor")
    d.add_argument("--units", action="store_true", help="Q_word units per Call (decodes the Program through the registry)")
    d.add_argument("--workload-program", help="workload_program.json: its chronology attributes root nodes to steps")
    d.add_argument("--value-bound", action="append", default=[], metavar="GROUP=WHY")
    d.add_argument("--name", default=None)
    d.add_argument("--out", required=True)
    s = sub.add_parser("served")
    s.add_argument("build")
    s.add_argument("--units", action="store_true")
    s.add_argument("--ignore-eos", action="store_true", help="the row serves with ignore_eos (served_match does)")
    s.add_argument("--lag", type=int, default=0, help="the engine's LAG: steps a request runs past its EOS")
    s.add_argument("--name", default=None)
    s.add_argument("--out", required=True)
    for x in (d, s):
        x.add_argument("--rules", action="append", default=[], metavar="PATH",
                       help="cached unit rules: a `verity.ml.definitions.word` rule directory or a --units-log JSONL (repeatable)")
        x.add_argument("--units-log", default=None, help="append each Definition's cut here (JSONL) as it lands")
        x.add_argument("--jobs", type=int, default=1, help="forked processes cutting Definitions, largest first")
        x.add_argument("--verify-below", type=int, default=0, metavar="GATES",
                       help="also cut Definitions of at most GATES gates by the reference (evaluate_definition, CallGraph)")
        x.add_argument("--recut-cached", action="store_true", help="recut cached Definitions too and compare")
    p = sub.add_parser("plan")
    p.add_argument("workload", nargs="+")
    p.add_argument("--planner", required=True, help="a planner module's path, PATH[:FUNC] (default FUNC plan)")
    p.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("reports", nargs="+")
    t.add_argument("--md", default=None)
    t.add_argument("--egress", action="store_true", help="one row per Program: E_max, E's median and p99, families, components")
    a = ap.parse_args(argv)
    helper = Units(a.rules, a.units_log, a.jobs, a.verify_below, a.recut_cached) if a.cmd in ("descriptor", "served") else None
    if a.cmd == "descriptor":
        desc = load(a.descriptor)
        units = None
        if a.units:
            units, population = q_word_units(a.descriptor, helper)
        r = analyze_descriptor(desc, units=units, value_bound=_bound(a.value_bound),
                               workload_program=json.loads(Path(a.workload_program).read_text()) if a.workload_program else None)
        r["name"] = a.name or a.descriptor
        if units is not None:
            r["population"], r["units_source"] = population, helper.summary()
        Path(a.out).write_text(json.dumps(r, indent=1, default=sorted))
    elif a.cmd == "served":
        r = analyze_served(json.loads(Path(a.build).read_text()), units=helper if a.units else False, ignore_eos=a.ignore_eos,
                           lag=a.lag)
        r["name"] = a.name or a.build
        Path(a.out).write_text(json.dumps(r, indent=1, default=sorted))
    elif a.cmd == "plan":
        path, _, fn = a.planner.partition(":")
        planner = load_planner(path, fn or "plan")
        reports = [dict(plan_check(json.loads(Path(w).read_text()), planner), name=Path(w).stem) for w in a.workload]
        Path(a.out).write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=1))
    else:
        reps: list[dict[str, Any]] = []
        for x in a.reports:
            j = json.loads(Path(x).read_text())
            reps.extend(j if isinstance(j, list) else [j])
        md = render_egress(reps) if a.egress else render(reps)
        if a.md:
            Path(a.md).write_text(md)
        print(md)
    if helper is not None and helper.mismatches:
        print(f"popread: {len(helper.mismatches)} unit count(s) disagree: {helper.mismatches[:4]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
