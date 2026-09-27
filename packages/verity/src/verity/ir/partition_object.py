"""`verity/partition/v1`: a Program's unit partition, named as a query the verifier evaluates, P = Q(C).

    {"format": "verity/partition/v1",
     "program": "<SHA-512 of the program>",
     "query": {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon(object))`, canon = `codec.canonical_json` (keys sorted, no whitespace, ASCII).
* **Program.** `program_sha512`: SHA-512 over the canonical descriptor (`codec.encode_program`, schema `verity-ir/descriptor/v1`,
  annotations excluded) -- the bytes the evaluator's input is defined over; a descriptor of another schema is refused.  It carries no
  domain prefix (the object and owners digests do): adding one is deferred to the re-baseline, where the IR's identity digests move
  anyway, because tonight's A2 roots already bind the current digest.
* **Query.** Nothing about the partition is stored: owners, units and the committed set are what `evaluate(program, query)` computes,
  deterministically, from the program (`QUERIES`).  Any change to a query's behaviour bumps its version; `tests/ir/test_partition_object.py`
  pins a program, its object digest and the digest of its evaluated owners (`owners_digest`).
* **Q_word v1** (`cut.evaluate_definition`, the algorithm in `verity.ir.cut`'s docstring).  The Calls are the root body's nodes other
  than its Input nodes, in canonical order; each is cut as its Definition's activation (parameters as inputs).  Units are numbered by
  Call, then within the Call's cut; the input unit is not a unit (always checked, never drawn).  The width rule's X and W are the
  query's parameters (EXTRAS = 0), integers in 1..2^16.  It applies to a program whose root Calls are `call` or `primitive` nodes
  (`cut`'s step 0); otherwise `evaluate` raises `QueryInapplicable` and `verify` refuses (`query-inapplicable`).
* `validate` checks the object's form (exact keys; a known query with exactly its parameters); `digest` digests only a valid object.
  `verify` evaluates the query and checks every Call's cut on its derived graph (the invariant, the width rule), and, given what serving
  commits, holds it to the derived committed set: every Call that commits must be given and no other key (`served-missing`,
  `served-unknown`).  `tests/ir/qword_vectors.json` pins every refusal code of both.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from typing import Any, Iterable, Mapping

from .codec import canonical_json, encode_program
from .codec import SCHEMA_V1
from .cut import WIRING, CallGraph, DefinitionCut, QueryInapplicable, check_cut, derived_committed, evaluate_definition
from .partition import CutVerdict

FORMAT = "verity/partition/v1"
_KEYS = {"format", "program", "query"}
_QUERY_KEYS = {"name", "version", "params"}
_SHA512 = re.compile(r"^[0-9a-f]{128}$")

#: the queries a partition may name: (name, version) -> the parameter names, each an integer in 1..PARAM_MAX
QUERIES: dict[tuple[str, int], tuple[str, ...]] = {("Q_word", 1): ("X", "W")}
PARAM_MAX = 1 << 16


def program_sha512(program_or_desc: Any) -> str:
    """The Program's SHA-512: over the canonical JSON of its descriptor (`codec.encode_program`) without annotations."""
    desc = program_or_desc if isinstance(program_or_desc, dict) else encode_program(program_or_desc)
    if desc.get("schema") != SCHEMA_V1:
        raise ValueError(f"the program digest is over a {SCHEMA_V1} descriptor, not {desc.get('schema')!r}")
    return hashlib.sha512(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()


def query(X: int = 16, W: int = 32) -> dict[str, Any]:
    if not all(_is_int(v) and 0 < v <= PARAM_MAX for v in (X, W)):
        raise ValueError(f"Q_word's X and W are integers in 1..{PARAM_MAX}, not {X!r}, {W!r}")
    return {"name": "Q_word", "version": 1, "params": {"X": X, "W": W}}


def build(program: Any, q: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The object naming `program`'s partition as the query `q` (default `Q_word{X=16, W=32}` v1)."""
    return {"format": FORMAT, "program": program_sha512(program), "query": dict(q or query())}


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
    if not isinstance(obj.get("program"), str) or not _SHA512.match(obj["program"]):
        fail("program-digest", obj.get("program"))
    q = obj.get("query")
    if not isinstance(q, Mapping) or set(q) != _QUERY_KEYS:
        fail("query-keys", sorted(set(q) ^ _QUERY_KEYS, key=repr) if isinstance(q, Mapping) else q)
    else:
        names = QUERIES.get((q["name"], q["version"])) if isinstance(q["name"], str) and _is_int(q["version"]) else None
        if names is None:
            fail("query-unknown", [q["name"], q["version"]])
        elif not isinstance(q["params"], Mapping) or set(q["params"]) != set(names) or \
                not all(_is_int(v) and 0 < v <= PARAM_MAX for v in q["params"].values()):
            fail("query-params", q["params"])
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(v), "gates": v[:64]} for k, v in bad.items()})


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


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
            form = program.root.body.nodes[k].form
            if form not in ("call", "primitive"):
                raise QueryInapplicable(f"root node {k} has form {form!r}: Q_word v1's Calls are call or primitive nodes",
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


def evaluate(program: Any, q: Mapping[str, Any] | None = None, memo: dict | None = None) -> Partition:
    return Partition(program, dict(q or query()), memo)


def owners_digest(part: Partition) -> str:
    """SHA-512 over the evaluated owners: `"verity/partition/v1/owners\\0" || canon([[root-body index, owner list], ...])`."""
    return hashlib.sha512(f"{FORMAT}/owners".encode() + b"\0" + canonical_json([[k, part.owner(p)] for p, (k, _c) in enumerate(part.calls)])).hexdigest()


def verify(obj: Mapping[str, Any], program: Any, served: Mapping[int, Iterable[int]] | None = None) -> CutVerdict:
    """The object against the program: `validate`, `program` = `program_sha512(program)`, then the query evaluated here
    (`query-inapplicable` when it does not apply) and every Call's cut checked on its derived graph (`cut.check_cut`: the invariant with
    the derived committed set, and the width rule with the query's X and W), once per Definition; failures by Call under `cut`
    ({"call", "codes", "detail"}).

    `served`, when given: root-body index -> the gates serving commits for that Call, numbered as `CallGraph` numbers the Call's
    computing gates (compacted, in layout order; not the Call's raw gate indices).  Its keys must be exactly Calls, including every
    Call whose derived committed set is not empty (`served-missing`: such a Call not given; `served-unknown`: a key that is no Call);
    each given Call is held to its derived set (`committed-unread`, `read-uncommitted`, `output-uncommitted`, under `cut`)."""
    v = validate(obj)
    if not v.ok:
        return v
    bad: dict[str, list] = {}
    if obj["program"] != program_sha512(program):
        bad["program"] = [obj["program"]]
    try:
        part = evaluate(program, obj["query"])
    except QueryInapplicable as e:
        bad["query-inapplicable"] = [e.detail]
        return _verdict(bad)
    if served is not None:
        known = {k for k, _c in part.calls}
        unknown = sorted((k for k in served if k not in known), key=repr)
        if unknown:
            bad["served-unknown"] = unknown
    checked: dict[str, CutVerdict] = {}
    for p, (k, c) in enumerate(part.calls):
        if served is not None and k in served:
            r = check_cut(part.graph(p), part.owner(p), served[k], (part.X, part.W, 0))
        else:
            if served is not None and part.committed(p):
                bad.setdefault("served-missing", []).append(k)
            r = checked.get(c.fn.id)
            if r is None:
                r = checked[c.fn.id] = check_cut(part.graph(p), part.owner(p), None, (part.X, part.W, 0))
        if not r.ok:
            bad.setdefault("cut", []).append({"call": k, "codes": list(r.codes), "detail": r.detail})
    return _verdict(bad)


def _verdict(bad: dict[str, list]) -> CutVerdict:
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(x), "gates": x[:64]} for k, x in bad.items()})


__all__ = ["FORMAT", "QUERIES", "PARAM_MAX", "QueryInapplicable", "program_sha512", "query", "build", "canonical", "digest", "validate", "calls", "Partition", "evaluate",
           "owners_digest", "verify"]
