"""A structural lowering's counts (PRF-133): each specialized Definition lowered once, its calls, batch members and scan
iterations kept as calls, instead of :func:`ir_lower.lower`'s one walk over the unrolled gates.

A specialization is lowered once per pattern of its argument bits: which are constants (and their values), and which
are one caller form or its complement (an alias), so the constant folding and ``x·x`` folding the unrolled lowering does
across a call boundary are kept. Every other argument bit is one of the callee's input columns, and a caller form that is
not a single committed bit is committed first (one row of the caller); a returned form that is not a single committed bit,
a constant or an argument is committed by the callee. Primitive gates are lowered by :data:`ir_lower.PIECES` and a Boolean
Definition by its builder, as :func:`ir_lower.lower_gates` lowers them, so a template made this way is O(its
Definitions) and differs from the unrolled unit only by those commits and the folds a pattern doesn't carry.

:func:`count` returns the unrolled totals (multiplied through the calls, not unrolled) and the template's size: its
Definitions and their own rows and references.

    python -m verity_flock.struct_lower K [R C]   # the H1 tile unit (benchmarks/pouw/h1_prove/h1unit.Shape)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from functools import partial

from verity.ml.circuits.boolean.forms import ONE, Circuit, L
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import FunctionCall, Region

from . import ir_lower as IL


@dataclass
class Spec:
    """One lowered specialization: its own rows (ANDs, commits, lookup bits) and references, its returns as
    ``("c", v)`` | ``("a", label, c)`` | ``("col",)``, and its calls' keys."""

    key: tuple
    rows: int = 0
    commits: int = 0
    lookups: int = 0
    asserts: int = 0
    refs: int = 0
    rets: list = field(default_factory=list)
    calls: list = field(default_factory=list)


def _pattern(words: list[list[L]]) -> tuple[tuple, list[L]]:
    """The argument bits' pattern and, per label, the caller's form it stands for."""
    labels: dict[frozenset, int] = {}
    forms: list[L] = []
    pat = []
    for w in words:
        for b in w:
            if b.is_const():
                pat.append(("c", b.c))
            else:
                i = labels.get(b.s)
                if i is None:
                    i = labels[b.s] = len(forms)
                    forms.append(L(b.s, 0))
                pat.append((i, b.c))
    return tuple(pat), forms


class _Lowerer:
    def __init__(self, circuit):
        self.circuit = circuit
        self.specs: dict[tuple, Spec] = {}
        self.totals: dict[tuple, dict] = {}

    def call(self, C: Circuit, fc: FunctionCall, wires: dict) -> None:
        """Lower call ``fc`` of the Definition being lowered on C: commit its arguments' forms, lower its specialization
        (once per pattern), and set its returns' words."""
        for a in dict.fromkeys(fc.args):
            w = wires[a]
            if any(not b.is_const() and not (len(b.s) == 1 and b.c == 0) for b in w):
                wires[a] = C.commit(w)
        words = [wires[a] for a in fc.args]
        pat, forms = _pattern(words)
        key = (fc.fn.id, pat)
        sp = self.specs.get(key) or self.lower(fc, key, pat)
        bits = iter(sp.rets)
        for r in fc.returns:
            if r in wires and not (fc.lo <= r < fc.hi):
                for _ in wires[r]:
                    next(bits)
                continue
            out = []
            for _ in range(IL._width(self.circuit.gate(r).prim)):
                t = next(bits)
                out.append(L(frozenset(), t[1]) if t[0] == "c" else (forms[t[1]].inv() if t[2] else forms[t[1]]) if t[0] == "a"
                           else C._new("call", None))
            wires.setdefault(r, out)
        C.used.extend(b for w in words for b in w)
        self.last = key

    def lower(self, fc: FunctionCall, key: tuple, pat: tuple) -> Spec:
        C = Circuit()
        C.used = []
        fresh: dict[int, L] = {}
        sc = fc.scope()
        it = iter(pat)
        wires: dict[int, list] = {}
        for a in fc.args:
            if a in wires:
                for _ in wires[a]:
                    next(it)
                continue
            w = []
            for _ in range(len(self._arg_width(sc, a))):
                t = next(it)
                if t[0] == "c":
                    w.append(L(frozenset(), t[1]))
                else:
                    if t[0] not in fresh:
                        fresh[t[0]] = C._new("arg", t[0])
                    w.append(fresh[t[0]].inv() if t[1] else fresh[t[0]])
            wires[a] = w
        d = getattr(sc.spec, "defn", sc.spec)
        build = getattr(d, "build", None)
        sp = Spec(key)
        if callable(build):
            if getattr(d, "statics", ()):
                build = partial(lambda S, C, *e: d.build(C, S, *e), sc.spec.S())
            if not IL._lower_activation(C, sc, build, wires, IL.PIECES):
                raise KeyError(f"{fc.id}: a parameter leaf has no wire")
        else:
            body = sc.spec.body
            for k, node in enumerate(body.nodes):
                lo = sc.offset + body.offsets[k]
                if isinstance(node.fn, PrimitiveDefinition):
                    IL.lower_gates(C, Region(self.circuit, lo, lo + node.gates), wires)
                    continue
                for m in range(1 if node.form == "call" else node.n):
                    ch = FunctionCall(self.circuit, sc, k, m)
                    self.call(C, ch, wires)
                    sp.calls.append(self.last)
        rets = []
        argbit = {f.s: i for i, f in fresh.items()}
        for r in fc.returns:
            w = wires[r]
            for b in w:
                if b.is_const():
                    rets.append(("c", b.c))
                elif b.s in argbit:
                    rets.append(("a", argbit[b.s], b.c))
                elif len(b.s) == 1 and b.c == 0:
                    rets.append(("col",))
                    C.used.append(b)
                else:
                    C.used.extend(C.commit([b]))
                    rets.append(("col",))
        C.output("used", C.used)
        live = C.live()
        for v in live:
            kind = C.kind[v]
            if kind == "and":
                a, b = C.data[v]
                sp.rows += 1
                sp.refs += len(a.s) + len(b.s)
                sp.commits += b is ONE
            elif kind == "lookup":
                sp.lookups += 1
        sp.asserts = len(C.asserts)
        sp.rets = rets
        self.specs[key] = sp
        return sp

    def _arg_width(self, sc, a: int) -> range:
        return range(IL._width(self.circuit.gate(a).prim))

    def total(self, key: tuple) -> dict:
        t = self.totals.get(key)
        if t is None:
            sp = self.specs[key]
            t = {"rows": sp.rows, "commits": sp.commits, "lookups": sp.lookups, "asserts": sp.asserts, "refs": sp.refs}
            for c in sp.calls:
                ct = self.total(c)
                for x in t:
                    t[x] += ct[x]
            self.totals[key] = t
        return t


def count(target) -> dict:
    """The structural lowering of ``target`` (a bound composite): the unrolled totals and the template's size."""
    sys.setrecursionlimit(100000)
    _prog, root = IL.standalone(target)
    lw = _Lowerer(root.circuit)
    C = Circuit()
    C.used = []
    wires = {a: C.input(f"in{a}", IL._width(root.circuit.gate(a).prim)) for a in dict.fromkeys(root.args)}
    lw.call(C, root, wires)
    top = lw.last
    tot = lw.total(top)
    own = {"rows": sum(s.rows for s in lw.specs.values()), "lookups": sum(s.lookups for s in lw.specs.values()),
           "refs": sum(s.refs for s in lw.specs.values()), "calls": sum(len(s.calls) for s in lw.specs.values())}
    by_fn: dict[str, int] = {}
    for k in lw.specs:
        by_fn[k[0]] = by_fn.get(k[0], 0) + 1
    return {"unrolled": tot, "template": own | {"specializations": len(lw.specs), "definitions": len(by_fn),
                                                "patterns_max": max(by_fn.values())},
            "patterns": {f: n for f, n in sorted(by_fn.items(), key=lambda kv: -kv[1])[:8]}}


def main(argv: list[str]) -> int:
    import resource
    from pathlib import Path

    h1 = Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "h1_prove"
    sys.path[:0] = [str(h1), str(h1.parent / "hidden_zk")]
    import h1unit as U

    k = int(argv[0])
    shape = U.Shape(k, *map(int, argv[1:3])) if len(argv) >= 3 else U.Shape(k)
    t0 = time.time()
    out = count(shape.unit())
    out |= {"k": k, "tile": [shape.r, shape.c], "seconds": round(time.time() - t0, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print("STRUCT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
