"""The partition as a named query over a program, P = Q(C): core's `verity/partition/v1` (`verity.ml.circuits.partition_object`),
applied here with the template queries `Q_template_instance` v0 and `Q_template_instances` v0, and the programs this
protocol builds for them.

    {"format": "verity/partition/v1", "program": "<SHA-512 of the program>",
     "query": {"name": "Q_template_instance", "version": 0, "params": {"template": "<Definition id>"}}}

* **Object, digest, program digest** are core's (`partition`, `digest`, `program_digest`, also named `program_sha512`). A
  record made before the canonical envelope names `{"program", "query"}`; `digest` gives it the envelope's rule,
  `SHA-512("verity/partition/v1\\0" || canon(object))`, which the roots registered with it bind.
* **Units** are core's evaluation (`partition_object.evaluate`) of a query of valid form (`partition_object.validate`):
  `QueryRefused` for a program that is not topological, or one the query does not apply to. The template queries (the
  served ones: a served window registers a population program under one) are the only ones applied here, and a template
  query reads the root body only. `Q_word` and `Q_call` are retired from the verifier of record: their objects keep
  core's form and digest, and `evaluate` refuses them. Every unit's committed bits fit a public size class
  (`UNIT_CLASSES`, powers of two from 16 to `UNIT_CLASS_MAX` bits; ASK-61, Daniel 19 Oct 2:59 PM PDT, provisional), its
  class the least that holds them (`unit_class`), a public property of its Definition: a program with a unit wider than
  the widest class is refused (a template instance's bits are its distinct returned gates', `_template_bits`), as is one
  with a template of no gate. `ranges`, `units` and `strata` read their units in canonical order (node order, then
  member), and `unit_gates` gives each unit's root gates.
* **Programs**: `population_program` (instances of one template, `TemplateInstances_v1{template, N}`) and
  `mixed_population_program` (of several, `TemplatePopulations_v1{templates, counts}`), whose binding records name the
  templates by descriptor id. v0 of each, which names only the counts, is kept for the records made with it
  (`POPULATION_VERSIONS`).

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
unit index on the prover's word.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.cut import call_scope
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.partition_object import (FORMAT, QueryRefused, program_sha512, template_id,
                                        template_instance_query, template_instances_query)
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Tuple

_PRE_CANONICAL = {"program", "query"}
Q_WORD = ("Q_word", 1)
#: The widest public size class of a unit's committed bits (the claims' `Claims.Sampling.unitClassMax`). 16 sets the
#: classes back to the former literal limit alone.
UNIT_CLASS_MAX = 262_144
#: The public size classes, ascending: the powers of two from 16 to `UNIT_CLASS_MAX` (the claims' `unitClasses`).
UNIT_CLASSES = tuple(16 << i for i in range((UNIT_CLASS_MAX // 16).bit_length()))


def unit_class(bits: int) -> int | None:
    """The size class of a unit that commits `bits` bits: the least of `UNIT_CLASSES` that holds them, None when none
    does."""
    return next((w for w in UNIT_CLASSES if bits <= w), None)


#: The population programs' versions, the builders' default first. v1 names its templates (each one's descriptor id) in its
#: binding record beside the counts, so the Definition's identity determines its body. v0 names the counts alone, so two
#: populations of equal counts over different templates share its id; it stays buildable for the records made with it.
POPULATION_VERSIONS = (1, 0)


def _population_version(version: int) -> int:
    if version not in POPULATION_VERSIONS:
        raise ValueError(f"no population program v{version}: the versions are {POPULATION_VERSIONS}")
    return version


def mixed_population_program(members: Sequence[tuple[SpecializedDefinition, int]], *, version: int = 1) -> Program:
    """`TemplatePopulations_v1{templates, counts}`: one batch per member `(template, n)`, in order; member j's parameters are
    `j.<name>`, arrays over its instances, and the result is the tuple of the batches' results. `templates` are the members'
    descriptor ids, in order. `version=0` builds `TemplatePopulations_v0{counts}` (`POPULATION_VERSIONS`)."""
    if not members or any(n < 1 for _, n in members):
        raise ValueError("every member has at least one instance")
    _population_version(version)
    sigs = [t.signature() for t, _ in members]

    def signature(S):
        params = tuple((f"{j}.{name}", Array(n, t)) for j, ((ps, _), n) in enumerate(zip(sigs, S.counts)) for name, t in ps)
        return params, Tuple(*(Array(n, ret) for (_, ret), n in zip(sigs, S.counts)))

    def body(B, S, *args):
        outs, at = [], 0
        for (t, _), (ps, _) in zip(members, sigs):
            outs.append(B.batch(t, *args[at:at + len(ps)], axes=(0,) * len(ps)))
            at += len(ps)
        return coll_concat(signature(S)[1], outs)

    named = {} if version == 0 else {"templates": tuple(descriptor_id(t) for t, _ in members)}
    defn = CompositeDefinition("TemplatePopulations", version, (*named, "counts"), signature, body,
                               doc="independent instances of " + ", ".join(f"{n} x {t.id}" for t, n in members), register=False)
    return Program(bind(defn, **named, counts=tuple(n for _, n in members)))


def population_program(template: SpecializedDefinition, n: int, *, version: int = 1) -> Program:
    """`TemplateInstances_v1{template, N=n}`: `n` independent instances of `template` as one batch, every parameter an array
    over instances; `template` is its descriptor id. Instance i of the batch is instance i of the input set it is built for.
    `version=0` builds `TemplateInstances_v0{N=n}` (`POPULATION_VERSIONS`)."""
    if n < 1:
        raise ValueError("a population has at least one instance")
    _population_version(version)
    params, ret = template.signature()

    def signature(S):
        return tuple((name, Array(S.N, t)) for name, t in params), Array(S.N, ret)

    def body(B, S, *args):
        return B.batch(template, *args, axes=(0,) * len(args))

    named = {} if version == 0 else {"template": descriptor_id(template)}
    defn = CompositeDefinition("TemplateInstances", version, (*named, "N"), signature, body,
                               doc=f"{n} independent instances of {template.id}", register=False)
    return Program(bind(defn, **named, N=n))


def partition(program: Program, query: Mapping[str, Any]) -> dict[str, Any]:
    return PO.build(program, query)


def digest(obj: Mapping[str, Any]) -> str:
    """Core's digest of a valid `verity/partition/v1` object (`ValueError` otherwise), or of a pre-canonical `{program, query}`."""
    if isinstance(obj, Mapping) and set(obj) == _PRE_CANONICAL:
        return hashlib.sha512(FORMAT.encode() + b"\0" + PO.canonical(dict(obj))).hexdigest()
    return PO.digest(obj)


def _template_bits(fn: Any) -> int:
    """The bits one instance of template `fn` commits: the summed widths, as its return type gives them, of the distinct
    gates of its activation that it returns. A template has no Input gate below the root, and nothing outside an instance
    reads its gates but through its returns, so this bounds the claims' `Prog.Boundary` count of the whole activation, read
    off the return's references alone rather than the activation's gates."""
    widths = fn.signature()[1].leaf_widths()
    if not hasattr(fn, "body"):
        return sum(widths)
    _c, call = call_scope(fn)
    lo, seen, bits = call.offset, set(), 0
    for r, w in zip(fn.body.ret, widths):
        g = resolve(call, r)
        if lo <= g < lo + fn.gates and g not in seen:
            seen.add(g)
            bits += w
    return bits


def _check_units(program: Program, part: PO.TemplatePartition) -> None:
    """Refuses a template of no gate, and a template instance whose committed bits fit no public class (`unit_class`,
    `_template_bits`), each template once, as its instances commit alike."""
    templates = {t: program.root.body.nodes[k].fn for k, t, _first, n in part.ranges if n}
    for t, fn in templates.items():
        if fn.gates == 0:
            raise QueryRefused(f"template {t} has no gate", {"template": t})
        bits = _template_bits(fn)
        if unit_class(bits) is None:
            raise QueryRefused(f"an instance of {t} commits {bits} bits, more than the widest class, {UNIT_CLASS_MAX}",
                               {"template": t, "bits": bits})


def evaluate(program: Program, query: Mapping[str, Any]) -> PO.TemplatePartition:
    """`query` applied to `program` by core, once core accepts the query's form; `QueryRefused` when the query is malformed,
    is not a template query (`Q_word` and `Q_call` are retired from the verifier of record), does not apply to the program
    (core's refusals, a program that is not topological among them), or has a unit wider than the widest public class
    (`_check_units`) or a template of no gate."""
    obj = PO.build(program, query)
    v, q = PO.validate(obj), obj["query"]
    if not v.ok:
        raise QueryRefused(f"not a {FORMAT} query ({', '.join(v.codes)}): {q!r}", {"query": q, "codes": list(v.codes)})
    if (q["name"], q["version"]) not in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES):
        raise QueryRefused(f"one-stage applies the template queries only, not {q['name']} v{q['version']}: Q_word and "
                           f"Q_call are retired from the verifier of record", {"query": q})
    part = PO.evaluate(program, q)
    _check_units(program, part)
    return part


def ranges(program: Program, query: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """`query` applied to `program`: (template id, first unit index, unit count) per root node computing units, in
    canonical order."""
    return [(tid, lo, n) for _k, tid, lo, n in evaluate(program, query).ranges]


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of units, indexed 0.. in canonical order."""
    return evaluate(program, query).population


def strata(program: Program, query: Mapping[str, Any]) -> list[tuple[str, list[list[int]]]]:
    """`query` applied to `program`, grouped by template: one stratum per descriptor id, its units as
    ascending, disjoint half-open ranges `[lo, hi)` with adjacent ones merged, the strata in the order of their first
    unit. These are the stratified law's strata, which each verifier derives from its own copies."""
    groups: dict[str, list[list[int]]] = {}
    for tid, lo, n in ranges(program, query):
        rs = groups.setdefault(tid, [])
        if rs and rs[-1][1] == lo:
            rs[-1][1] = lo + n
        else:
            rs.append([lo, lo + n])
    return list(groups.items())


def unit_gates(program: Program, query: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Each unit's root gates `[lo, hi)`, by unit: an instance's (`TemplatePartition.gates`)."""
    part = evaluate(program, query)
    return [part.gates(u) for u in range(part.population)]


__all__ = ["template_id", "FORMAT", "QueryRefused", "digest", "evaluate", "mixed_population_program", "partition",
           "POPULATION_VERSIONS", "population_program", "program_digest", "program_sha512", "ranges", "strata", "template_instance_query",
           "template_instances_query", "unit_class", "UNIT_CLASS_MAX", "UNIT_CLASSES", "unit_gates", "units"]
