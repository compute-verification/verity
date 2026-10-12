"""`verity/partition/v1`: a Program's unit partition, named as a query the verifier evaluates, P = Q(C).

    {"format": "verity/partition/v1",
     "program": "<SHA-512 of the program>",
     "query": {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon(object))`, canon = `codec.canonical_json` (keys sorted, no whitespace, ASCII).
* **Program.** `codec.program_digest`: SHA-512 over the canonical descriptor (`codec.encode_program`, schema `verity-ir/descriptor/v1`,
  annotations excluded) -- the bytes the evaluator's input is defined over; a descriptor of another schema is refused.  It carries no
  domain prefix (the object and owners digests do).
* **Query.** Nothing about the partition is stored: owners, units and the committed set are what `evaluate(program, query)` computes,
  deterministically, from the program (`QUERIES`).  Any change to a query's behaviour bumps its version; `verity/ml/circuits/tests/test_partition_object.py`
  pins a program, its object digest and the digest of its evaluated owners (`owners_digest`).
* **Q_word v1** (`cut.evaluate_definition`, the algorithm in `verity.ml.circuits.cut`'s docstring).  The Calls are the root body's nodes other
  than its Input nodes, in canonical order; each is cut as its Definition's activation (parameters as inputs).  Units are numbered by
  Call, then within the Call's cut; the input unit is not a unit (always checked, never drawn).  The width rule's X and W are the
  query's parameters (EXTRAS = 0), integers in 1..2^16.  It applies to a program whose root Calls are `call` or `primitive` nodes
  (`cut`'s step 0); otherwise `evaluate` raises `QueryInapplicable` and `verify` refuses (`query-inapplicable`).
* **Q_word v2**, the same parameters: v1's evaluation exactly (the same Calls, units, owners and committed sets, so the same
  `owners_digest`; the object's digest differs by its version), and `verify` checks every Call's cut as v1 does but for one rule: a
  value computed in two units is reported (`recomputed_across`), not refused (`gate-recomputed`) (`cut`'s "Q_word v2").  v2's units
  may compute a common value, so a count of them bounds wrong work but is not distinct work; a consumer that credits work takes v1.
* **Q_call v1**, `params = {"X": <bits>}` (`call_query`, 32 the agreed value; `CallPartition`; the algorithm in `cut`'s "Q_call v1",
  `PROTOCOL.md` §11).  The same Calls, each cut in isolation; a unit is a gate interval of the Call (the Call, a body node, a member or
  an iteration), cut down from the Call until its outputs fit in X bits; Input gates and constants are in no unit and every other gate
  is in exactly one.  Units are numbered by Call, then layout order (`locate`, `gates`); `owners_digest` digests their owners.  The
  query is inapplicable (`query-inapplicable`) to a root Call not of form `call` or `primitive` (`why: "root-form"`) and to a computing
  gate wider than X bits (`why: "gate-too-wide"`).  `verify` checks every Call's cut (`cut.check_call_cut`: the invariant, w_out <= X)
  and reports a value computed in two units as Q_word v2 does (`recomputed_across`), never refusing it; `served` is numbered by
  `CallGates`' compact computing gates.
* **Q_template_instance v0**, `params = {"template": <descriptor id>}` (`TemplatePartition`).  The root body's nodes other than its
  Input nodes must each be a `call` or a `batch` whose callee's descriptor id (`codec.descriptor_id`, the `callees` entry) is the
  template.  Each call is one unit and each batch member one unit, numbered in canonical order (node order, then member); the input
  unit is not a unit; no width rule.  A unit owns every gate of its instance (`gates(i)`, one interval); the committed set is derived
  as for Q_word: a gate is committed when a unit other than its own reads it (Input gates as the input unit's), or the root returns
  it (`committed`).  The query is inapplicable to any other program (`query-inapplicable`): a node of another function, a `scan`, a primitive at the root, or no
  instance of the template (a unit, not a node: a batch of 0 members is no instance).
* **Q_template_instances v0**, `params = {"templates": [<descriptor id>, ...]}` (non-empty, distinct, order as given).  The same, a node
  an instance of any listed template, every listed template with at least one instance; the unit order is node order, not list order.
  `verity/ml/circuits/tests/test_template_instance.py` pins both (`template_instance_vectors.json`).  Like every query, both apply only to a
  topological program (below), so an instance never reads its own gates.
* **Every query** applies only to a topological program: every node of every body names only its body's parameters and earlier
  nodes (a body's `ret` any node).  `evaluate` checks it once, before any query (`order_violation`), so every query and `verify`
  refuse the same programs (`query-inapplicable`, with the definition, node, argument and reference).
* `validate` checks the object's form (exact keys; a known query with exactly its parameters); `digest` digests only a valid object.
  `verify` evaluates the query and checks every Call's cut on its derived graph (the invariant, the width rule), and, given what serving
  commits, holds it to the derived committed set: every Call that commits must be given and no other key (`served-missing`,
  `served-unknown`).  `verity/ml/circuits/tests/qword_vectors.json` pins every refusal code of both.  For the template queries `verify` checks that the
  query applies to the program; `served` is Q_word's.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .codec import canonical_json, descriptor_id, encode_program, program_digest
from .codec import SCHEMA_V1
from .cut import (CALL_X, RECOMPUTE, WIRING, CallCut, CallGates, CallGraph, DefinitionCut, QueryInapplicable, call_committed, call_gates,
                  check_call_cut, check_cut, derived_committed, evaluate_call, evaluate_definition, member_refs)
from .layout import resolve
from .partition import CutVerdict
from .program import is_input
from .refs import Concat, Explicit

FORMAT = "verity/partition/v1"
_KEYS = {"format", "program", "query"}
_QUERY_KEYS = {"name", "version", "params"}
_SHA512 = re.compile(r"^[0-9a-f]{128}$")

PARAM_MAX = 1 << 16


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _templates_ok(t: Any) -> bool:
    return isinstance(t, list) and len(t) > 0 and all(isinstance(x, str) for x in t) and len(set(t)) == len(t)


Q_TEMPLATE_INSTANCE = ("Q_template_instance", 0)
Q_TEMPLATE_INSTANCES = ("Q_template_instances", 0)
Q_CALL = ("Q_call", 1)

def _word_params_ok(p: Mapping[str, Any]) -> bool:
    return all(_is_int(v) and 0 < v <= PARAM_MAX for v in p.values())


#: the queries a partition may name: (name, version) -> (its parameter names, the check its parameters pass)
QUERIES: dict[tuple[str, int], tuple[tuple[str, ...], Callable[[Mapping[str, Any]], bool]]] = {
    ("Q_word", 1): (("X", "W"), _word_params_ok),
    ("Q_word", 2): (("X", "W"), _word_params_ok),
    Q_CALL: (("X",), _word_params_ok),
    Q_TEMPLATE_INSTANCE: (("template",), lambda p: isinstance(p["template"], str)),
    Q_TEMPLATE_INSTANCES: (("templates",), lambda p: _templates_ok(p["templates"])),
}

#: one-stage's name for the template queries' refusal
QueryRefused = QueryInapplicable


def _program(program_or_desc: Any) -> str:
    """The `program` field: `codec.program_digest` of a v1 descriptor."""
    desc = program_or_desc if isinstance(program_or_desc, dict) else encode_program(program_or_desc)
    if desc.get("schema") != SCHEMA_V1:
        raise ValueError(f"the program digest is over a {SCHEMA_V1} descriptor, not {desc.get('schema')!r}")
    return program_digest(desc)


#: the program field under the name META's `program_sha512` and its callers use; equal to `codec.program_digest`
program_sha512 = _program


def query(X: int = 16, W: int = 32, version: int = 1) -> dict[str, Any]:
    """`Q_word{X, W}`, v1 unless `version` names v2 (`cut.RECOMPUTE`)."""
    if not all(_is_int(v) and 0 < v <= PARAM_MAX for v in (X, W)):
        raise ValueError(f"Q_word's X and W are integers in 1..{PARAM_MAX}, not {X!r}, {W!r}")
    if ("Q_word", version) not in QUERIES:
        raise ValueError(f"Q_word has versions 1 and 2, not {version!r}")
    return {"name": "Q_word", "version": version, "params": {"X": X, "W": W}}


def call_query(X: int = CALL_X) -> dict[str, Any]:
    """`Q_call{X}` v1 (`cut`'s "Q_call v1"): units of at most X bits out, cut along Calls."""
    if not (_is_int(X) and 0 < X <= PARAM_MAX):
        raise ValueError(f"Q_call's X is an integer in 1..{PARAM_MAX}, not {X!r}")
    return {"name": Q_CALL[0], "version": Q_CALL[1], "params": {"X": X}}


def template_id(t: Any) -> str:
    """A template as the template queries name it: a Definition's descriptor id (`codec.descriptor_id`), or that id."""
    return t if isinstance(t, str) else descriptor_id(t)


def template_instance_query(template: Any) -> dict[str, Any]:
    """`Q_template_instance` v0 over `template` (a Definition, written as its descriptor id, or that id)."""
    return {"name": Q_TEMPLATE_INSTANCE[0], "version": Q_TEMPLATE_INSTANCE[1], "params": {"template": template_id(template)}}


def template_instances_query(templates: Sequence[Any]) -> dict[str, Any]:
    """`Q_template_instances` v0 over `templates`, in the order given."""
    return {"name": Q_TEMPLATE_INSTANCES[0], "version": Q_TEMPLATE_INSTANCES[1], "params": {"templates": [template_id(t) for t in templates]}}


def build(program: Any, q: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The object naming `program`'s partition as the query `q` (default `Q_word{X=16, W=32}` v1)."""
    return {"format": FORMAT, "program": _program(program), "query": dict(q or query())}


def canonical(obj: Mapping[str, Any]) -> bytes:
    return canonical_json(obj)


def digest(obj: Mapping[str, Any]) -> str:
    """`SHA-512("verity/partition/v1\\0" || canon(object))`, hex, of a valid object (`validate`)."""
    v = validate(obj)
    if not v.ok:
        raise ValueError(f"not a valid {FORMAT} object: {v.codes}")
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical(obj)).hexdigest()


def validate(obj: Any) -> CutVerdict:
    """The object's own form: exactly `format`, `program`, `query` (nothing derived is stored); a SHA-512 program; a query of `QUERIES`
    with exactly its parameters, each an integer in 1..PARAM_MAX."""
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    if not isinstance(obj, Mapping):
        return CutVerdict(False, ("not-an-object",), {"not-an-object": {"n": 1, "gates": []}})
    if set(obj) != _KEYS:
        fail("keys", sorted(set(obj) ^ _KEYS, key=repr))
    if obj.get("format") != FORMAT:
        fail("format", obj.get("format"))
    if not isinstance(obj.get("program"), str) or not _SHA512.fullmatch(obj["program"]):
        fail("program-digest", obj.get("program"))
    q = obj.get("query")
    if not isinstance(q, Mapping) or set(q) != _QUERY_KEYS:
        fail("query-keys", sorted(set(q) ^ _QUERY_KEYS, key=repr) if isinstance(q, Mapping) else q)
    else:
        known = QUERIES.get((q["name"], q["version"])) if isinstance(q["name"], str) and _is_int(q["version"]) else None
        if known is None:
            fail("query-unknown", [q["name"], q["version"]])
        elif not isinstance(q["params"], Mapping) or set(q["params"]) != set(known[0]) or not known[1](q["params"]):
            fail("query-params", q["params"])
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(v), "gates": v[:64]} for k, v in bad.items()})


def calls(program: Any) -> list[tuple[int, Any]]:
    """The Program's Calls: (index in the root body, the node's Definition) for every root body node that is not an Input node
    (`program.is_input`: the callee's id is `Input<w>_v1` or `RegisteredInput<w>_v1`)."""
    return [(k, n.fn) for k, n in enumerate(program.root.body.nodes) if not is_input(n.fn)]


class Partition:
    """`evaluate(program, query)`: per Call its `DefinitionCut` (activations of one Definition share it), the units numbered by Call
    then within the Call's cut (`population`, `locate`), and on demand a Call's graph, owners and derived committed set."""

    def __init__(self, program: Any, q: Mapping[str, Any], memo: dict | None = None, wiring: frozenset[str] = WIRING) -> None:
        if (q.get("name"), q.get("version")) not in QUERIES:
            raise ValueError(f"unknown query {q.get('name')!r} v{q.get('version')!r}")
        self.query = dict(q)
        self.X, self.W = int(q["params"]["X"]), int(q["params"]["W"])
        self.recompute = RECOMPUTE[q["version"]]
        self._memo = {} if memo is None else memo
        self._wiring = wiring
        self.calls: list[tuple[int, DefinitionCut]] = []
        for k, fn in calls(program):
            form = program.root.body.nodes[k].form
            if form not in ("call", "primitive"):
                raise QueryInapplicable(f"root node {k} has form {form!r}: Q_word v{q['version']}'s Calls are call or primitive nodes",
                                        {"call": k, "form": form, "definition": fn.id})
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

    def check(self, p: int, committed: Iterable[int] | None = None) -> CutVerdict:
        """Call `p`'s cut on its graph (`cut.check_cut`), held to `committed` when given."""
        return check_cut(self.graph(p), self.owner(p), committed, (self.X, self.W, 0), self.recompute)


class CallPartition:
    """`Q_call` v1 applied to a program (`cut`'s "Q_call v1"): per Call its `CallCut` (activations of one Definition share it), the
    units numbered by Call then layout order (`population`, `locate`, `gates`), and per Call its gates, owners and derived committed
    set, in `CallGates`' compact numbering."""

    recompute = "report"

    def __init__(self, program: Any, q: Mapping[str, Any], memo: dict | None = None) -> None:
        if (q.get("name"), q.get("version")) != Q_CALL:
            raise ValueError(f"not Q_call v1: {q.get('name')!r} v{q.get('version')!r}")
        self.query = dict(q)
        self.X = int(q["params"]["X"])
        self._memo = {} if memo is None else memo
        self.program = program
        self.calls: list[tuple[int, CallCut]] = []
        for k, fn in calls(program):
            form = program.root.body.nodes[k].form
            if form not in ("call", "primitive"):
                raise QueryInapplicable(f"root node {k} has form {form!r}: Q_call v1's Calls are call or primitive nodes",
                                        {"call": k, "form": form, "definition": fn.id, "why": "root-form"})
            try:
                self.calls.append((k, evaluate_call(fn, self.X, self._memo)))
            except QueryInapplicable as e:
                raise QueryInapplicable(str(e), {"call": k, **e.detail}) from None
        self._start: list[int] = []
        t = 0
        for _k, c in self.calls:
            self._start.append(t)
            t += len(c.units)
        self.population = t

    def locate(self, i: int) -> tuple[int, int, int, list[list[int]]]:
        """Unit `i` -> (position in `calls`, the Call's root-body index, unit within the Call, its path of `[k]` / `[k, m]` steps)."""
        if not 0 <= i < self.population:
            raise IndexError(f"unit {i} outside 0..{self.population - 1}")
        p = bisect.bisect_right(self._start, i) - 1
        k, c = self.calls[p]
        u = i - self._start[p]
        return p, k, u, [list(s) for s in c.units[u][2]]

    def gates(self, i: int) -> tuple[int, int]:
        """Unit `i`'s root gate interval [lo, hi): its computing gates are its gates; Input gates and constants in it are in no unit."""
        p, k, u, _path = self.locate(i)
        lo, hi, _ = self.calls[p][1].units[u]
        base = self.program.root.body.offsets[k]
        return base + lo, base + hi

    def graph(self, p: int) -> CallGates:
        return call_gates(self.calls[p][1].fn, self._memo)

    def owner(self, p: int) -> list[int]:
        return self.calls[p][1].owner()

    def committed(self, p: int) -> list[int]:
        return call_committed(self.graph(p), self.calls[p][1])

    def check(self, p: int, committed: Iterable[int] | None = None) -> CutVerdict:
        return check_call_cut(self.graph(p), self.owner(p), self.X, committed)


class TemplatePartition:
    """`Q_template_instance` v0 or `Q_template_instances` v0 applied to a program: every instance of a listed template is one unit.
    `ranges`: per root node, (root-body index, template, first unit, units); `population`; `locate(i)` -> (root-body index, member);
    `gates(i)`, the instance's gates [lo, hi); `inputs(i)` / `outputs(i)`, the gates its parameter / result leaves resolve to; the
    derived `committed()` set."""

    def __init__(self, program: Any, q: Mapping[str, Any]) -> None:
        kind = (q.get("name"), q.get("version"))
        if kind not in (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCES) or not isinstance(q.get("params"), Mapping) or \
                set(q["params"]) != set(QUERIES[kind][0]) or not QUERIES[kind][1](q["params"]):
            raise QueryInapplicable(f"not a template-instance query with its parameters: {dict(q)!r}", {"query": dict(q)})
        self.query = dict(q)
        self.templates = [q["params"]["template"]] if kind == Q_TEMPLATE_INSTANCE else list(q["params"]["templates"])
        self.program = program
        self.ranges: list[tuple[int, str, int, int]] = []
        nodes = program.root.body.nodes
        at = 0
        for k, fn in calls(program):
            tid = descriptor_id(fn)
            if nodes[k].form not in ("call", "batch") or tid not in self.templates:
                raise QueryInapplicable(f"root node {k} is a {nodes[k].form} of {tid!r}: an instance is a call or batch of one of {self.templates!r}",
                                        {"call": k, "form": nodes[k].form, "definition": tid})
            n = nodes[k].n if nodes[k].form == "batch" else 1
            self.ranges.append((k, tid, at, n))
            at += n
        missing = [t for t in self.templates if not any(tid == t and n for _k, tid, _a, n in self.ranges)]
        if missing:
            raise QueryInapplicable(f"the program has no instance of {missing!r}", {"missing": missing})
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


def order_violation(program: Any) -> dict | None:
    """The first reference of the program that names neither a parameter of its body nor an earlier node of it (a body's `ret` may
    name any of its nodes), over every body the root reaches, root first and callees in node order, each Definition once; None when the
    program is topological.  A program with such a reference is no computation (a gate would read itself or a later gate): every query
    is inapplicable to it."""
    seen: set[int] = set()
    todo = [program.root]
    while todo:
        fn = todo.pop(0)
        if id(fn) in seen or not hasattr(fn, "body"):
            continue
        seen.add(id(fn))
        body = fn.body
        for k, node in enumerate(body.nodes):
            for i, a in enumerate(node.args):
                bad = _out_of_order(a.refs, len(body.params), k)
                if bad is not None:
                    return {"definition": fn.id, "node": k, "arg": i, "reference": bad, "why": "not topological"}
            todo.append(node.fn)
        bad = _out_of_order(body.ret, len(body.params), len(body.nodes))
        if bad is not None:
            return {"definition": fn.id, "node": "ret", "reference": bad, "why": "not topological"}
    return None


def _out_of_order(refs: Any, params: int, nodes: int) -> list | None:
    """The first (space, index) of `refs` outside the body's parameters `0..params-1` and nodes `0..nodes-1`, or None."""
    parts = [refs]
    while parts:
        r = parts.pop(0)
        if isinstance(r, Concat):
            parts[:0] = r.parts
            continue
        names = [(x[0], x[1]) for x in r.items] if isinstance(r, Explicit) else ([(r.space, r.idx)] if r.count else [])
        for space, idx in names:
            if not (_is_int(idx) and 0 <= idx < (params if space == "p" else nodes if space == "n" else 0)):
                return [space, idx]
    return None


def evaluate(program: Any, q: Mapping[str, Any] | None = None, memo: dict | None = None) -> Partition | CallPartition | TemplatePartition:
    """The query `q` (default `Q_word{16, 32}` v1) applied to `program`: `QueryInapplicable` when the program is not topological
    (`order_violation`), checked once here for every query, or when the query does not apply to it."""
    bad = order_violation(program)
    if bad is not None:
        raise QueryInapplicable(f"{bad['definition']} node {bad['node']} names {bad['reference']}: not a parameter or an earlier node", bad)
    q = dict(q or query())
    if (q.get("name"), q.get("version")) in (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCES):
        return TemplatePartition(program, q)
    if (q.get("name"), q.get("version")) == Q_CALL:
        return CallPartition(program, q, memo)
    return Partition(program, q, memo)


def ranges_digest(part: TemplatePartition) -> str:
    """SHA-512 over a template partition's units: `"verity/partition/v1/ranges\\0" || canon([[root-body index, template, first, count], ...])`."""
    return hashlib.sha512(f"{FORMAT}/ranges".encode() + b"\0" + canonical_json([list(r) for r in part.ranges])).hexdigest()


def owners_digest(part: Partition | CallPartition) -> str:
    """SHA-512 over the evaluated owners: `"verity/partition/v1/owners\\0" || canon([[root-body index, owner list], ...])`."""
    return hashlib.sha512(f"{FORMAT}/owners".encode() + b"\0" + canonical_json([[k, part.owner(p)] for p, (k, _c) in enumerate(part.calls)])).hexdigest()


def verify(obj: Mapping[str, Any], program: Any, served: Mapping[int, Iterable[int]] | None = None) -> CutVerdict:
    """The object against the program: `validate`, `program` = `codec.program_digest(program)`, then the query evaluated here
    (`query-inapplicable` when it does not apply) and every Call's cut checked on its derived graph (`cut.check_cut`: the invariant with
    the derived committed set, and the width rule with the query's X and W), once per Definition; failures by Call under `cut`
    ({"call", "codes", "detail"}).

    `served`, when given: root-body index -> the gates serving commits for that Call, numbered as `CallGraph` numbers the Call's
    computing gates (compacted, in layout order; not the Call's raw gate indices).  Its keys must be exactly Calls, including every
    Call whose derived committed set is not empty (`served-missing`: such a Call not given; `served-unknown`: a key that is no Call;
    `served-malformed`: a Call's entry that is not a list of distinct non-negative integers);
    each given Call is held to its derived set (`committed-unread`, `read-uncommitted`, `output-uncommitted`, under `cut`).  A template
    query commits whole instances' inputs and results; `served` with one is refused (`served-unsupported`).

    Under `Q_word` v2 a value computed in two units is no failure; the verdict's detail reports it, never as a code, under
    `recomputed_across`: {"n": the pairs over every Call, "gates": per Call with any, {"call", "n", "gates": its first pairs}}."""
    v = validate(obj)
    if not v.ok:
        return v
    bad: dict[str, list] = {}
    report: list[dict] = []
    if obj["program"] != _program(program):
        bad["program"] = [obj["program"]]
    template = (obj["query"]["name"], obj["query"]["version"]) in (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCES)
    if template and served is not None:
        bad["served-unsupported"] = [obj["query"]["name"]]
    try:
        part = evaluate(program, obj["query"])
        if not template:
            _verify_calls(part, served, bad, report)
    except QueryInapplicable as e:
        bad["query-inapplicable"] = [e.detail]
    verdict = _verdict(bad)
    kind = (obj["query"]["name"], obj["query"]["version"])
    if kind == Q_CALL or (kind[0] == "Q_word" and RECOMPUTE.get(kind[1]) == "report"):
        verdict.detail["recomputed_across"] = {"n": sum(r["n"] for r in report), "gates": report[:64]}
    return verdict


def _verify_calls(part: Partition | CallPartition, served: Mapping[int, Iterable[int]] | None, bad: dict[str, list], report: list[dict]) -> None:
    if served is not None:
        known = {k for k, _c in part.calls}
        unknown = sorted((k for k in served if k not in known), key=repr)
        if unknown:
            bad["served-unknown"] = unknown
        malformed = sorted((k for k, v in served.items() if k in known and not _gate_list(v)), key=repr)
        if malformed:
            bad["served-malformed"] = malformed
            return
    checked: dict[str, CutVerdict] = {}
    for p, (k, c) in enumerate(part.calls):
        if served is not None and k in served:
            r = part.check(p, served[k])
        else:
            if served is not None and part.committed(p):
                bad.setdefault("served-missing", []).append(k)
            r = checked.get(c.fn.id)
            if r is None:
                r = checked[c.fn.id] = part.check(p)
        if not r.ok:
            bad.setdefault("cut", []).append({"call": k, "codes": list(r.codes), "detail": r.detail})
        across = r.detail.get("recomputed_across")
        if across and across["n"]:
            report.append({"call": k, **across})


def _gate_list(v: Any) -> bool:
    """A Call's served commits: a list or tuple of distinct non-negative integers (never coerced)."""
    return isinstance(v, (list, tuple)) and all(_is_int(g) and g >= 0 for g in v) and len(set(v)) == len(v)


def _verdict(bad: dict[str, list]) -> CutVerdict:
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(x), "gates": x[:64]} for k, x in bad.items()})


__all__ = ["FORMAT", "QUERIES", "PARAM_MAX", "QueryInapplicable", "QueryRefused", "Q_TEMPLATE_INSTANCE", "Q_TEMPLATE_INSTANCES", "Q_CALL",
           "call_query", "CallPartition", "template_id", "template_instance_query", "template_instances_query", "TemplatePartition", "ranges_digest", "query", "build", "canonical", "digest", "validate", "calls", "Partition", "evaluate",
           "owners_digest", "program_sha512", "verify"]
