"""C-Flock over an arbitrary partition: every unit class (`verity.ml.circuits.units`) lowered once to AND/XOR/NOT with C-Flock's pieces,
and checked against the reference evaluator on synthetic inputs lifted through the unit's flat Definition.

    uv run python -m verity_flock.partition_units --model tiny [--partition q-word|blocks:64|scatter:8]
    uv run python -m verity_flock.partition_units --definition 'RoPE_v1{NHEADS=32,D=64}' --vectors 32

* **Lowering.**  `lower_class` lays one unit of a class out as a `gf2.Circuit` (inputs ``in0 ..``, outputs ``out0 ..``, one word per
  port) through `ir_lower.lower_gates`, so a primitive without a piece names itself.
* **Synthetic inputs.**  Drawn where they are a flat Definition's parameters, never at the unit: a unit's inputs are interior values
  (a softmax's max, sum and probabilities are correlated).  `synthetic_words` draws every parameter leaf uniformly from a counter
  stream; since circuits are total, every word is a valid input.  The flat Definition is evaluated on each sample (`Activation.evaluate`)
  and every unit's boundary values are sliced out of the transcript.
* **Partitions.**  Any query `partition_object` evaluates, or an owner function over each Call's `CallGraph`.  `blocks` and `scatter`
  are deliberately odd ones, for testing that nothing downstream depends on `Q_word`: contiguous runs of computing gates in
  canonical order, and a seeded hash of each computing gate into a few units (units that read each other's values both ways).
  They are not partition proposals and are not queries: no record can name them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import CallGraph, check_cut
from verity.ml.circuits.program import input_id
from verity.ml.circuits.units import Activation, PartitionUnits, UnitCircuit
from verity.ml.randomness import derive

from . import gf2
from . import ir_lower as IL

SOURCE = b"verity partition-units synthetic"
DOMAIN = "verity/flock/partition-units/v0"


# --- deliberately odd partitions (owner functions) ------------------------------------------------------------------------------

def blocks(size: int) -> Callable[[Any, CallGraph], list[int]]:
    """Units of ``size`` consecutive computing gates, in canonical order."""
    return lambda fn, graph: [c // size for c in range(graph.n)]


def scatter(k: int, seed: int = 0) -> Callable[[Any, CallGraph], list[int]]:
    """Each computing gate into one of ``k`` units by a seeded hash of (Definition, gate): units interleave and read each other's
    committed values in both directions."""
    def owner(fn: Any, graph: CallGraph) -> list[int]:
        units = min(k, graph.n)
        own = [int.from_bytes(hashlib.sha256(f"{seed}/{fn.id}/{c}".encode()).digest()[:8], "little") % units for c in range(graph.n)]
        used = {u: i for i, u in enumerate(sorted(set(own)))}          # number the units that got a gate 0 .. U-1
        return [used[u] for u in own]
    return owner


PARTITIONS: dict[str, Callable[[str], Mapping[str, Any]]] = {
    "q-word": lambda arg: {"query": PO.query(*(int(x) for x in arg.split("/")) if arg else (16, 32))},
    "blocks": lambda arg: {"owners": blocks(int(arg or 64))},
    "scatter": lambda arg: {"owners": scatter(*(int(x) for x in (arg or "8").split("/")))},
}


def partition_units(program: Any, spec: str) -> PartitionUnits:
    """``q-word[:X/W]``, ``q-template-instance[:TEMPLATE]``, ``blocks[:B]`` or ``scatter[:k[/seed]]`` over ``program``.
    ``q-template-instance`` is ``Q_template_instance`` v0 of TEMPLATE (a descriptor id), by default of the one Definition the
    program's root calls (a population program's template: every instance one unit)."""
    name, _, arg = spec.partition(":")
    if name == "q-template-instance":
        if not arg:
            called = {PO.template_instance_query(fn)["params"]["template"] for _k, fn in PO.calls(program)}
            if len(called) != 1:
                raise ValueError(f"q-template-instance: the root calls {len(called)} Definitions; name the template")
            arg = called.pop()
        return PartitionUnits(program, query=PO.template_instance_query(arg))
    return PartitionUnits(program, **PARTITIONS[name](arg))


def check_owners(units: PartitionUnits) -> list[dict]:
    """The partition invariant on every Call's cut of an owner-function partition (`cut.check_cut`, no width rule): refusals."""
    bad = []
    for _k, _m, fn, owner in units._calls():
        if isinstance(owner, list):
            v = check_cut(units.activation(fn).graph, owner, limits=None)
            if not v.ok:
                bad.append({"definition": fn.id, "codes": list(v.codes)})
    return bad


# --- lowering --------------------------------------------------------------------------------------------------------------------

@dataclass
class ClassCircuit:
    shape: str
    circuit: gf2.Circuit | None
    counts: dict | None
    missing: tuple[str, ...] = ()
    reason: str = ""


def lower_class(act: Activation, uc: UnitCircuit, max_gates: int = 1 << 20) -> ClassCircuit:
    """C-Flock's Boolean lowering of one unit (inputs ``in0 ..``, outputs ``out0 ..``, one word per port), or why not."""
    prims = {act.gate(j).prim.id for j in uc.gates}
    missing = tuple(sorted(p for p in prims if p not in IL.PIECES and not p.startswith("Const") and input_id(p) is None))
    if missing:
        return ClassCircuit(uc.shape, None, None, missing, "no piece")
    if len(uc.gates) > max_gates:
        return ClassCircuit(uc.shape, None, None, (), f"more than {max_gates} word gates")
    C = gf2.Circuit()
    wires = {act.absolute(k): C.input(f"in{t}", w) for t, (k, w) in enumerate(zip(uc.inputs, uc.in_widths))}
    try:
        IL.lower_gates(C, [act.gate(j) for j in uc.gates], wires, reads=[act.absolute(j) for j in uc.outputs])
    except (IL.NoPiece, NotImplementedError) as e:
        return ClassCircuit(uc.shape, None, None, (), f"{type(e).__name__}: {e}"[:200])
    for t, j in enumerate(uc.outputs):
        C.output(f"out{t}", wires[act.absolute(j)])
    return ClassCircuit(uc.shape, C, C.counts())


# --- synthetic inputs, lifted through the flat Definition ---------------------------------------------------------------------

def synthetic_words(key: Any, fn_id: str, widths: Sequence[int], T: int) -> list[list[int]]:
    """T draws of a flat Definition's parameter leaves: uniform words of each leaf's width, from the stream at (Definition, draw)."""
    nbytes = [(w + 7) // 8 for w in widths]
    out = []
    for t in range(T):
        raw = key.stream(("definition", fn_id, t), sum(nbytes))
        words, at = [], 0
        for w, nb in zip(widths, nbytes):
            words.append(int.from_bytes(raw[at:at + nb], "little") & ((1 << w) - 1))
            at += nb
        out.append(words)
    return out


def evaluate(act: Activation, words: Sequence[int]) -> dict[int, int]:
    """Every gate's value (local keys; parameter leaves as ``-1 - p``) on the parameter leaves ``words``, by each primitive's own
    evaluator in canonical order."""
    t: dict[int, int] = {-1 - p: int(w) for p, w in enumerate(words)}
    for j in range(act.fn.gates):
        g = act.gate(j)
        ops = [act.local(o) for o in g.operands]
        lazy = getattr(g.prim, "evaluate_lazy", None)
        t[j] = lazy(lambda q, _o=ops: t[_o[q]]) if lazy is not None else g.prim.evaluate(*(t[o] for o in ops))
    return t


def lanes(act: Activation, units: Sequence[UnitCircuit], key: Any, T: int) -> dict[tuple[str, int], tuple[list, list]]:
    """Per unit of one flat Definition: its input and output words over T synthetic draws (per port, one word per draw)."""
    widths = [act.width(-1 - p) for p in range(act.params)]
    runs = [evaluate(act, w) for w in synthetic_words(key, act.fn.id, widths, T)]
    return {(uc.definition, uc.unit): ([[r[x] for r in runs] for x in uc.inputs], [[r[j] for r in runs] for j in uc.outputs])
            for uc in units}


def class_mismatches(C: gf2.Circuit, uc: UnitCircuit, ins: Sequence[Sequence[int]], want: Sequence[Sequence[int]]) -> int:
    """Output words of the class circuit C that differ from ``want`` on the lanes ``ins`` (per port, one word per lane)."""
    n = len(want[0]) if want else 0
    got, _fail = C.evaluate({f"in{i}": IL.slice_words(v, w) for i, (v, w) in enumerate(zip(ins, uc.in_widths))}, n)
    return sum(int((IL.unslice_words(got[f"out{i}"], n) != np.asarray(v, dtype=np.uint64)).sum()) for i, v in enumerate(want))


def check(units: PartitionUnits, T: int = 16, seed: int = 0, max_gates: int = 1 << 20, log: Callable[[str], None] | None = None) -> dict:
    """Lower one unit per class, then check every distinct unit circuit of every lowered class (each flat Definition once) against
    the reference on T synthetic draws lifted through its flat Definition.  A summary with per-class rows and the mismatches
    (expected 0)."""
    t0 = time.time()
    classes = units.classes()
    extract_s = time.time() - t0
    t1 = time.time()
    lowered: dict[str, ClassCircuit] = {}
    for s, cl in sorted(classes.items(), key=lambda kv: len(kv[1].representative.gates)):
        lowered[s] = lower_class(units.activation(cl.definition), cl.representative, max_gates)
        if log:
            c = lowered[s]
            log(f"  {cl.representative.definition} u{cl.representative.unit}: {len(cl.representative.gates)} word gates, "
                f"{c.counts['ands'] if c.counts else c.reason} {','.join(c.missing)}")
    lower_s = time.time() - t1
    key = derive(SOURCE, DOMAIN, {"program": program_digest(units.program), "seed": int(seed)})
    t2 = time.time()
    mismatches, checked, lanes_checked = Counter(), 0, 0
    for fn, circuits in list(units._flat.values()):
        todo = [uc for uc in circuits if lowered.get(uc.shape) and lowered[uc.shape].circuit is not None]
        if not todo:
            continue
        got = lanes(units.activation(fn), todo, key, T)
        for uc in todo:
            ins, want = got[(uc.definition, uc.unit)]
            mismatches[uc.shape] += class_mismatches(lowered[uc.shape].circuit, uc, ins, want)
            checked += 1
            lanes_checked += T
    check_s = time.time() - t2
    ok = [s for s, c in lowered.items() if c.circuit is not None]
    missing = Counter()
    for s, c in lowered.items():
        for p in c.missing:
            missing[p] += classes[s].units
    rows = [{"shape": s[:16], "definition": cl.representative.definition, "unit": cl.representative.unit, "units": cl.units,
             "word_gates": len(cl.representative.gates), "in_bits": sum(cl.representative.in_widths),
             "out_bits": sum(cl.representative.out_widths),
             **({"ands": lowered[s].counts["ands"], "xors_nots_free": True, "mismatches": mismatches[s]} if s in ok else
                {"not_lowered": lowered[s].reason, "missing": list(lowered[s].missing)})}
            for s, cl in sorted(classes.items(), key=lambda kv: -kv[1].units)]
    return {
        "query": units.query, "owners": None if units.query is not None else "owner function (not a query)",
        "partition": PO.digest(PO.build(units.program, units.query)) if units.query is not None else None,
        "units": sum(cl.units for cl in classes.values()),
        "shapes": len(classes), "shapes_lowered": len(ok), "units_lowered": sum(classes[s].units for s in ok),
        "distinct_circuits_checked": checked, "lanes_checked": lanes_checked, "vectors": T,
        "mismatches": sum(mismatches.values()),
        "ands_all_lowered_units": sum(lowered[s].counts["ands"] * classes[s].units for s in ok),
        "largest_class_ands": max((lowered[s].counts["ands"] for s in ok), default=0),
        "not_lowered": {"no piece (units per primitive)": dict(missing.most_common()),
                        "other": dict(Counter(c.reason for c in lowered.values() if c.circuit is None and not c.missing))},
        "seconds": {"classes": round(extract_s, 3), "lower": round(lower_s, 3), "check": round(check_s, 3)},
        "classes": rows,
    }


# --- programs for the CLI ------------------------------------------------------------------------------------------------------

def one_call(fn: Any) -> Any:
    """A Program whose one Call is ``fn`` over Input gates."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.program import Program

    params, ret = fn.signature()
    shell = CompositeDefinition(f"OneCall[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *x: B.call(fn, *x),
                                register=False)
    return Program(shell)


def _program(a: argparse.Namespace) -> Any:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.program import Program

    from circuit_check import targets as TG

    TG.load_registries()
    if a.model == "tiny":
        from verity.ml.definitions.registry import b1
        return Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2))
    return one_call(TG.definition(a.definition))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=["tiny"], help="circuit-check's tiny Serve_v1 Program (38 Calls)")
    src.add_argument("--definition", help="a registered Definition id, as a one-Call Program (circuit-check's parser)")
    ap.add_argument("--partition", default="q-word", help="q-word[:X/W], q-template-instance[:TEMPLATE], blocks[:B] or scatter[:k[/seed]]")
    ap.add_argument("--vectors", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gates", type=int, default=1 << 20, help="lower a class only up to this many word gates")
    ap.add_argument("--classes", action="store_true", help="print every class's row")
    a = ap.parse_args(argv)
    units = partition_units(_program(a), a.partition)
    r = check(units, a.vectors, a.seed, a.max_gates)
    if units.query is None:
        r["invariant_refusals"] = check_owners(units)
    if not a.classes:
        r["classes"] = r["classes"][:12]
    json.dump(r, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
