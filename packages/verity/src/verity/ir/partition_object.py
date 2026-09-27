"""`verity/partition/v1`: a Program's unit partition, named as a query the verifier evaluates, P = Q(C).

    {"format": "verity/partition/v1",
     "program": "<SHA-512 of the program>",
     "query": {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon(object))`, canon = `codec.canonical_json` (keys sorted, no whitespace, ASCII).
* **Program.** `program_sha512`: SHA-512 over the canonical descriptor (`codec.encode_program`, annotations excluded) -- the bytes
  the evaluator's input is defined over.
* **Query.** Nothing about the partition is stored: owners, units and the committed set are what `evaluate(program, query)` computes,
  deterministically, from the program (`QUERIES`).  Any change to a query's behaviour bumps its version; `tests/ir/test_partition_object.py`
  pins a program, its object digest and the digest of its evaluated owners (`owners_digest`).
* **Q_word v1** (`cut.evaluate_definition`, the algorithm in `verity.ir.cut`'s docstring).  The Calls are the root body's nodes other
  than its Input nodes, in canonical order; each is cut as its Definition's activation (parameters as inputs).  Units are numbered by
  Call, then within the Call's cut; the input unit is not a unit (always checked, never drawn).  The width rule's X and W are the
  query's parameters (EXTRAS = 0).
* **Q_template_instance v0**, `params = {"template": <descriptor id>}` (`TemplatePartition`).  The root body's nodes other than its
  Input nodes must each be a `call` or a `batch` whose callee's descriptor id (`codec.descriptor_id`, the `callees` entry) is the
  template.  Each call is one unit and each batch member one unit, numbered in canonical order (node order, then member); the input
  unit is not a unit; no width rule.  A unit owns every gate of its instance (`gates(i)`, one interval); the committed set is derived
  as for Q_word: a gate is committed when a unit other than its own reads it (Input gates as the input unit's), or the root returns
  it (`committed`).  The query refuses any other program: a node of another function, a `scan`, a primitive at the root, or no
  instance of the template (a unit, not a node: a batch of 0 members is no instance).
* **Q_template_instances v0**, `params = {"templates": [<descriptor id>, ...]}` (non-empty, distinct, order as given).  The same, a node
  an instance of any listed template, every listed template with at least one instance; the unit order is node order, not list order.
  `tests/ir/test_template_instance.py` pins both (`template_instance_vectors.json`).
* `validate` checks the object's form (exact keys; a known query with exactly its parameters); `verify` evaluates the query and checks
  every Call's cut on its derived graph (the invariant, the width rule, and what serving commits against the derived committed set),
  or, for the template queries, that the query applies to the program.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .codec import canonical_json, descriptor_id, encode_program
from .cut import WIRING, CallGraph, DefinitionCut, check_cut, derived_committed, evaluate_definition, member_refs
from .layout import resolve
from .partition import CutVerdict

FORMAT = "verity/partition/v1"
_KEYS = {"format", "program", "query"}
_QUERY_KEYS = {"name", "version", "params"}
_SHA512 = re.compile(r"^[0-9a-f]{128}$")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _templates_ok(t: Any) -> bool:
    return isinstance(t, list) and len(t) > 0 and all(isinstance(x, str) for x in t) and len(set(t)) == len(t)


Q_TEMPLATE_INSTANCE = ("Q_template_instance", 0)
Q_TEMPLATE_INSTANCES = ("Q_template_instances", 0)

#: the queries a partition may name: (name, version) -> (its parameter names, the check its parameters pass)
QUERIES: dict[tuple[str, int], tuple[tuple[str, ...], Callable[[Mapping[str, Any]], bool]]] = {
    ("Q_word", 1): (("X", "W"), lambda p: all(_is_int(v) and v > 0 for v in p.values())),
    Q_TEMPLATE_INSTANCE: (("template",), lambda p: isinstance(p["template"], str)),
    Q_TEMPLATE_INSTANCES: (("templates",), lambda p: _templates_ok(p["templates"])),
}


class QueryRefused(ValueError):
    """The query does not apply to the program."""


def program_sha512(program_or_desc: Any) -> str:
    """The Program's SHA-512: over the canonical JSON of its descriptor (`codec.encode_program`) without annotations."""
    desc = program_or_desc if isinstance(program_or_desc, dict) else encode_program(program_or_desc)
    return hashlib.sha512(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()


def query(X: int = 16, W: int = 32) -> dict[str, Any]:
    return {"name": "Q_word", "version": 1, "params": {"X": int(X), "W": int(W)}}


def _template_id(t: Any) -> str:
    return t if isinstance(t, str) else descriptor_id(t)


def template_instance_query(template: Any) -> dict[str, Any]:
    """`Q_template_instance` v0 over `template` (a Definition, written as its descriptor id, or that id)."""
    return {"name": Q_TEMPLATE_INSTANCE[0], "version": Q_TEMPLATE_INSTANCE[1], "params": {"template": _template_id(template)}}


def template_instances_query(templates: Sequence[Any]) -> dict[str, Any]:
    """`Q_template_instances` v0 over `templates`, in the order given."""
    return {"name": Q_TEMPLATE_INSTANCES[0], "version": Q_TEMPLATE_INSTANCES[1], "params": {"templates": [_template_id(t) for t in templates]}}


def build(program: Any, q: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The object naming `program`'s partition as the query `q` (default `Q_word{X=16, W=32}` v1)."""
    return {"format": FORMAT, "program": program_sha512(program), "query": dict(q or query())}


def canonical(obj: Mapping[str, Any]) -> bytes:
    return canonical_json(obj)


def digest(obj: Mapping[str, Any]) -> str:
    """`SHA-512("verity/partition/v1\\0" || canon(object))`, hex."""
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical(obj)).hexdigest()


def validate(obj: Any) -> CutVerdict:
    """The object's own form: exactly `format`, `program`, `query` (nothing derived is stored); a SHA-512 program; a query of `QUERIES`
    with exactly its parameters, each a positive int."""
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    if not isinstance(obj, Mapping):
        return CutVerdict(False, ("not-an-object",), {"not-an-object": {"n": 1, "gates": []}})
    if set(obj) != _KEYS:
        fail("keys", sorted(set(obj) ^ _KEYS))
    if obj.get("format") != FORMAT:
        fail("format", obj.get("format"))
    if not isinstance(obj.get("program"), str) or not _SHA512.match(obj["program"]):
        fail("program-digest", obj.get("program"))
    q = obj.get("query")
    if not isinstance(q, Mapping) or set(q) != _QUERY_KEYS:
        fail("query-keys", sorted(set(q) ^ _QUERY_KEYS) if isinstance(q, Mapping) else q)
    else:
        known = QUERIES.get((q["name"], q["version"])) if isinstance(q["name"], str) and _is_int(q["version"]) else None
        if known is None:
            fail("query-unknown", [q["name"], q["version"]])
        elif not isinstance(q["params"], Mapping) or set(q["params"]) != set(known[0]) or not known[1](q["params"]):
            fail("query-params", q["params"])
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(v), "gates": v[:64]} for k, v in bad.items()})


def calls(program: Any) -> list[tuple[int, Any]]:
    """The Program's Calls: (index in the root body, the node's Definition) for every root body node that is not an Input."""
    return [(k, n.fn) for k, n in enumerate(program.root.body.nodes) if getattr(n.fn, "family", None) != "Input"]


class Partition:
    """`evaluate(program, query)`: per Call its `DefinitionCut` (activations of one Definition share it), the units numbered by Call
    then within the Call's cut (`population`, `locate`), and on demand a Call's graph, owners and derived committed set."""

    def __init__(self, program: Any, q: Mapping[str, Any], memo: dict | None = None, wiring: frozenset[str] = WIRING) -> None:
        if (q.get("name"), q.get("version")) not in QUERIES:
            raise ValueError(f"unknown query {q.get('name')!r} v{q.get('version')!r}")
        self.query = dict(q)
        self.X, self.W = int(q["params"]["X"]), int(q["params"]["W"])
        self._memo = {} if memo is None else memo
        self._wiring = wiring
        self.calls: list[tuple[int, DefinitionCut]] = []
        for k, fn in calls(program):
            if program.root.body.nodes[k].form not in ("call", "primitive"):
                raise NotImplementedError(f"root node {k} has form {program.root.body.nodes[k].form!r}: a Call is a call or primitive node")
            self.calls.append((k, evaluate_definition(fn, self.X, self._memo, wiring)))
        self._start: list[int] = []
        t = 0
        for _k, c in self.calls:
            self._start.append(t)
            t += c.units
        self.population = t
        self._graphs: dict[str, CallGraph] = {}

    def locate(self, i: int) -> tuple[int, int, int, list[tuple[int, int, int]]]:
        """Unit `i` -> (position in `calls`, the Call's root-body index, unit within its cut, path within a separable cut), by
        bisection over the Calls' prefix sums, then a descent through the cut."""
        if not 0 <= i < self.population:
            raise IndexError(f"unit {i} outside 0..{self.population - 1}")
        p = bisect.bisect_right(self._start, i) - 1
        k, c = self.calls[p]
        u = i - self._start[p]
        return p, k, u, c.locate(u)

    def graph(self, p: int) -> CallGraph:
        fn = self.calls[p][1].fn
        g = self._graphs.get(fn.id)
        if g is None:
            g = self._graphs[fn.id] = CallGraph(fn, self._wiring)
        return g

    def owner(self, p: int) -> list[int]:
        return self.calls[p][1].owner()

    def committed(self, p: int) -> list[int]:
        return derived_committed(self.graph(p), self.owner(p))


class TemplatePartition:
    """`Q_template_instance` v0 or `Q_template_instances` v0 applied to a program: every instance of a listed template is one unit.
    `ranges`: per root node, (root-body index, template, first unit, units); `population`; `locate(i)` -> (root-body index, member);
    `gates(i)`, the instance's gates [lo, hi); `inputs(i)` / `outputs(i)`, the gates its parameter / result leaves resolve to; the
    derived `committed()` set."""

    def __init__(self, program: Any, q: Mapping[str, Any]) -> None:
        kind = (q.get("name"), q.get("version"))
        if kind not in (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCES) or not isinstance(q.get("params"), Mapping) or \
                set(q["params"]) != set(QUERIES[kind][0]) or not QUERIES[kind][1](q["params"]):
            raise QueryRefused(f"not a template-instance query with its parameters: {dict(q)!r}")
        self.query = dict(q)
        self.templates = [q["params"]["template"]] if kind == Q_TEMPLATE_INSTANCE else list(q["params"]["templates"])
        self.program = program
        self.ranges: list[tuple[int, str, int, int]] = []
        nodes = program.root.body.nodes
        at = 0
        for k, fn in calls(program):
            tid = descriptor_id(fn)
            if nodes[k].form not in ("call", "batch") or tid not in self.templates:
                raise QueryRefused(f"root node {k} is a {nodes[k].form} of {tid!r}: an instance is a call or batch of one of {self.templates!r}")
            n = nodes[k].n if nodes[k].form == "batch" else 1
            self.ranges.append((k, tid, at, n))
            at += n
        missing = [t for t in self.templates if not any(tid == t and n for _k, tid, _a, n in self.ranges)]
        if missing:
            raise QueryRefused(f"the program has no instance of {missing!r}")
        self.population = at
        self._start = [a for _k, _t, a, _n in self.ranges]

    def locate(self, i: int) -> tuple[int, int]:
        if not 0 <= i < self.population:
            raise IndexError(f"unit {i} outside 0..{self.population - 1}")
        k, _t, a, _n = self.ranges[bisect.bisect_right(self._start, i) - 1]
        return k, i - a

    def gates(self, i: int) -> tuple[int, int]:
        k, m = self.locate(i)
        fn = self.program.root.body.nodes[k].fn
        lo = self.program.root.body.offsets[k] + m * fn.gates
        return lo, lo + fn.gates

    def inputs(self, i: int) -> list[int]:
        k, m = self.locate(i)
        node, rs = self.program.root.body.nodes[k], self.program.circuit.root_scope
        axes = node.axes if node.form == "batch" else (None,) * len(node.args)
        return [resolve(rs, r) for a, ax in zip(node.args, axes) for r in (member_refs(a, ax, m) if node.form == "batch" else a.refs)]

    def outputs(self, i: int) -> list[int]:
        k, m = self.locate(i)
        L = self.program.root.body.nodes[k].fn.out_leaves
        rs = self.program.circuit.root_scope
        return [resolve(rs, ("n", k, m * L + j)) for j in range(L)]

    def committed(self) -> list[int]:
        """The gates a unit other than their own reads (an instance's inputs never lie in the instance), and the root's outputs."""
        out = set(self.program.circuit.root_outputs())
        for i in range(self.population):
            out.update(self.inputs(i))
        return sorted(out)


def evaluate(program: Any, q: Mapping[str, Any] | None = None, memo: dict | None = None) -> Partition | TemplatePartition:
    q = dict(q or query())
    if (q.get("name"), q.get("version")) in (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCES):
        return TemplatePartition(program, q)
    return Partition(program, q, memo)


def ranges_digest(part: TemplatePartition) -> str:
    """SHA-512 over a template partition's units: `"verity/partition/v1/ranges\\0" || canon([[root-body index, template, first, count], ...])`."""
    return hashlib.sha512(f"{FORMAT}/ranges".encode() + b"\0" + canonical_json([list(r) for r in part.ranges])).hexdigest()


def owners_digest(part: Partition) -> str:
    """SHA-512 over the evaluated owners: `"verity/partition/v1/owners\\0" || canon([[root-body index, owner list], ...])`."""
    return hashlib.sha512(f"{FORMAT}/owners".encode() + b"\0" + canonical_json([[k, part.owner(p)] for p, (k, _c) in enumerate(part.calls)])).hexdigest()


def verify(obj: Mapping[str, Any], program: Any, served: Mapping[int, Iterable[int]] | None = None) -> CutVerdict:
    """The object against the program: `validate`, `program` = `program_sha512(program)`, then the query evaluated here and every Call's
    cut checked on its derived graph (`cut.check_cut`: the invariant with the derived committed set, and the width rule with the query's
    X and W), once per Definition.  `served` (root-body index -> the computing gates serving commits for that Call): held to the derived
    set (`committed-unread`, `read-uncommitted`, `output-uncommitted`).  Failures by Call: {"call", "codes", "detail"}.  A template
    query's units are whole instances, which tile the program's non-Input gates by construction: it is checked that the query applies
    (`query-refused`); `served` is Q_word's."""
    v = validate(obj)
    if not v.ok:
        return v
    bad: dict[str, list] = {}
    if obj["program"] != program_sha512(program):
        bad["program"] = [obj["program"]]
    if (obj["query"]["name"], obj["query"]["version"]) in (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCES):
        if served is not None:
            raise ValueError("served commits are checked for Q_word; a template query commits whole instances' inputs and results")
        try:
            evaluate(program, obj["query"])
        except QueryRefused as e:
            bad["query-refused"] = [str(e)]
        codes = tuple(bad)
        return CutVerdict(not codes, codes, {k: {"n": len(x), "gates": x[:64]} for k, x in bad.items()})
    part = evaluate(program, obj["query"])
    checked: dict[str, CutVerdict] = {}
    for p, (k, c) in enumerate(part.calls):
        if served is not None and k in served:
            r = check_cut(part.graph(p), part.owner(p), served[k], (part.X, part.W, 0))
        else:
            r = checked.get(c.fn.id)
            if r is None:
                r = checked[c.fn.id] = check_cut(part.graph(p), part.owner(p), None, (part.X, part.W, 0))
        if not r.ok:
            bad.setdefault("cut", []).append({"call": k, "codes": list(r.codes), "detail": r.detail})
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(x), "gates": x[:64]} for k, x in bad.items()})


__all__ = ["FORMAT", "QUERIES", "Q_TEMPLATE_INSTANCE", "Q_TEMPLATE_INSTANCES", "QueryRefused", "program_sha512", "query",
           "template_instance_query", "template_instances_query", "build", "canonical", "digest", "validate", "calls", "Partition",
           "TemplatePartition", "evaluate", "owners_digest", "ranges_digest", "verify"]
