"""The partition as a named query over a program, P = Q(C): core's `verity/partition/v1` (`verity.circuits.partition_object`),
applied here with the template queries `Q_template_instance` v0 and `Q_template_instances` v0, `Q_word` v1 at `X = 16`
and `Q_call` v1 at `X <= 16`, and the programs this protocol builds for the template queries.

    {"format": "verity/partition/v1", "program": "<SHA-512 of the program>",
     "query": {"name": "Q_template_instance", "version": 0, "params": {"template": "<Definition id>"}}}

* **Object, digest, program digest** are core's (`partition`, `digest`, `program_digest`, also named `program_sha512`). A
  record made before the canonical envelope names `{"program", "query"}`; `digest` gives it the envelope's rule,
  `SHA-512("verity/partition/v1\\0" || canon(object))`, which the roots registered with it bind.
* **Proof blocks** are core's evaluation (`partition_object.evaluate`) of a query of valid form (`partition_object.validate`):
  `QueryRefused` for a program that is not topological, or one the query does not apply to. The template queries (the
  served ones: a served window registers a population program under one), `Q_word` v1 at `X = 16` and `Q_call` v1 at
  `X <= 16` are the ones applied here. A template query reads the root body only; `Q_word` and `Q_call` lay out one
  activation of each distinct Call's Definition gate by gate, so they cost its unrolled size
  (`benchmarks/verifier_cost/`'s `partition`). Every proof block's committed bits fit a public size class
  (`verity.circuits.partition_v2`'s `UNIT_CLASSES`, powers of two from 16 to `UNIT_CLASS_MAX` bits; provisional), its
  class the least that holds them (`unit_class`), a public property of its Definition: a program with a proof block wider
  than the widest class is refused (a template instance's bits are its distinct returned gates', `_template_bits`; a
  `Q_call` proof block's count the constants and Input gates of its Definition's activation, `_committed_bits`, which `Q_call`'s width rule leaves
  out; a `Q_word` proof block's are its cut's), as is one with a template of no gate. `ranges`, `units` and `strata` read their
  proof blocks in canonical order (node order, then member, for a template query; Call order, then the Call's own, for
  `Q_word` and `Q_call`), and `unit_gates` gives
  each proof block's root gates. `in_order` holds when the proof blocks read stratum by stratum are the population in order.
  `fanout` gives a template query's fan-out F, at least the most proof blocks one group joined by reads of each other's
  results holds, and `shared_inputs` the run Input nodes its proof blocks don't each read alone, both a root node at a
  time (R8, `registration.check`).
* **Programs**: `population_program` (instances of one template, `TemplateInstances_v1{template, N}`) and
  `mixed_population_program` (of several, `TemplatePopulations_v1{templates, counts}`), whose binding records name the
  templates by descriptor id, are `verity.circuits.population`'s, named here too: a serving stack builds its window's
  program with the same functions. v0 of each, which names only the counts, is kept for the records made with it
  (`POPULATION_VERSIONS`).

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
proof block index on the prover's word.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

from verity.circuits import partition_object as PO
from verity.circuits.cut import boundary_widths, call_scope
from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.layout import resolve
from verity.circuits.partition_object import (FORMAT, QueryRefused, program_sha512, template_id,
                                        template_instance_query, template_instances_query)
from verity.circuits.partition_v2 import UNIT_CLASS_MAX, unit_class
from verity.circuits.population import POPULATION_VERSIONS, mixed_population_program, population_program
from verity.circuits.program import Program, is_input, is_registered
from verity.circuits.refs import Affine, Concat, Strided
from verity.circuits.types import Value

_PRE_CANONICAL = {"program", "query"}
Q_WORD = ("Q_word", 1)
#: `Q_word` v1's `X`, and the most `Q_call` v1's `X` may be: the width rule of the queries applied here.
QUERY_X = 16


def partition(program: Program, query: Mapping[str, Any]) -> dict[str, Any]:
    return PO.build(program, query)


def digest(obj: Mapping[str, Any]) -> str:
    """Core's digest of a valid `verity/partition/v1` object (`ValueError` otherwise), or of a pre-canonical `{program, query}`."""
    if isinstance(obj, Mapping) and set(obj) == _PRE_CANONICAL:
        return hashlib.sha512(FORMAT.encode() + b"\0" + PO.canonical(dict(obj))).hexdigest()
    return PO.digest(obj)


def _check_word(part: PO.Partition) -> None:
    """Each `Q_word` proof block's committed bits fit a public class (`unit_class`): each Definition's cut, once, on its Call's
    graph."""
    seen: set[str] = set()
    for p, (_k, c) in enumerate(part.calls):
        fid = descriptor_id(c.fn)
        if fid in seen:
            continue
        seen.add(fid)
        g, owner = part.graph(p), c.owner()
        if len(owner) != g.n:
            raise QueryRefused(f"{fid}'s cut leaves a computing gate out", {"definition": fid})
        ob, _ = boundary_widths(g, owner)
        wide = [u for u, b in enumerate(ob) if unit_class(b) is None]
        if wide:
            raise QueryRefused(f"unit {wide[0]} of {fid} commits {ob[wide[0]]} bits, more than the widest class, "
                               f"{UNIT_CLASS_MAX}", {"definition": fid, "unit": wide[0], "bits": ob[wide[0]]})


def _committed_bits(fn: Any, units: Sequence[tuple[int, int]]) -> list[int]:
    """Per proof block `[lo, hi)` of one activation of `fn` (its local gates, as `Q_call` numbers them), the bits it commits: the
    summed declared widths of its gates that are Input gates, or that a later gate of the activation reads or the
    activation returns, constants included. Nothing outside an activation reads its gates but through its returns, so in
    any program this bounds the claims' `Prog.Boundary` count, as `Q_call`'s width rule bounds its computing gates'."""
    c, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    width, inp, last = [0] * n, [False] * n, [-1] * n
    for j in range(n):
        g = c.gate(lo + j)
        width[j], inp[j] = (g.prim.ret.w if isinstance(g.prim.ret, Value) else 0), is_input(g.prim)
        for o in g.operands:
            if lo <= o < lo + j:
                last[o - lo] = j
    for r in ([0] if not hasattr(fn, "body") else (resolve(call, r) - lo for r in fn.body.ret)):
        if 0 <= r < n:
            last[r] = n
    return [sum(width[g] for g in range(a, b) if inp[g] or last[g] >= b) for a, b in units]


def _template_bits(fn: Any) -> int:
    """The bits one instance of template `fn` commits: the summed widths, as its return type gives them, of the distinct
    gates of its activation that it returns. A template has no Input gate below the root, and nothing outside an instance
    reads its gates but through its returns, so this is `_committed_bits` of the whole activation, read off the return's
    references alone rather than the activation's gates."""
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


def _check_units(program: Program, part: PO.TemplatePartition | PO.CallPartition) -> None:
    """Refuses a template of no gate, and a template instance or `Q_call` proof block whose committed bits fit no public class
    (`unit_class`; `_template_bits`, and `_committed_bits` with constants counted, which `Q_call`'s width rule leaves
    out), each Definition once, as activations of one Definition commit alike."""
    if isinstance(part, PO.TemplatePartition):
        templates = {t: program.root.body.nodes[k].fn for k, t, _first, n in part.ranges if n}
        for t, fn in templates.items():
            if fn.gates == 0:
                raise QueryRefused(f"template {t} has no gate", {"template": t})
            bits = _template_bits(fn)
            if unit_class(bits) is None:
                raise QueryRefused(f"an instance of {t} commits {bits} bits, more than the widest class, {UNIT_CLASS_MAX}",
                                   {"template": t, "bits": bits})
        return
    for cut in {cut.fn.id: cut for _k, cut in part.calls}.values():
        fn = cut.fn
        bits = _committed_bits(fn, [(lo, hi) for lo, hi, _path in cut.units])
        wide = next((u for u, b in enumerate(bits) if unit_class(b) is None), None)
        if wide is not None:
            raise QueryRefused(f"unit {wide} of {fn.id} commits {bits[wide]} bits, more than the widest class, "
                               f"{UNIT_CLASS_MAX}", {"definition": fn.id, "unit": wide, "bits": bits[wide]})


def evaluate(program: Program, query: Mapping[str, Any]) -> PO.TemplatePartition | PO.Partition | PO.CallPartition:
    """`query` applied to `program` by core, once core accepts the query's form; `QueryRefused` when the query is malformed,
    is not one applied here (a template query, `Q_word` v1 at `X = 16` or `Q_call` v1 at `X <= 16`), does not apply to the
    program (core's refusals, a program that is not topological among them), or has a proof block wider than the widest public
    class (`_check_word`, `_check_units`) or a template of no gate."""
    obj = PO.build(program, query)
    v, q = PO.validate(obj), obj["query"]
    if not v.ok:
        raise QueryRefused(f"not a {FORMAT} query ({', '.join(v.codes)}): {q!r}", {"query": q, "codes": list(v.codes)})
    kind, x = (q["name"], q["version"]), q["params"].get("X")
    if kind in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES):
        part = PO.evaluate(program, q)
        _check_units(program, part)
        return part
    if kind == Q_WORD and x == QUERY_X:
        part = PO.evaluate(program, q)
        _check_word(part)
        return part
    if kind == PO.Q_CALL and x <= QUERY_X:
        part = PO.evaluate(program, q)
        _check_units(program, part)
        return part
    raise QueryRefused(f"one-stage applies the template queries, Q_word v1 at X = {QUERY_X} and Q_call v1 at "
                       f"X <= {QUERY_X}, not {q['name']} v{q['version']} at X = {x}", {"query": q})


def is_template_query(query: Mapping[str, Any]) -> bool:
    """Whether `query` names a template query, whose proof blocks are the root nodes' instances (`fanout`,
    `shared_inputs`)."""
    return isinstance(query, Mapping) and (query.get("name"), query.get("version")) in (PO.Q_TEMPLATE_INSTANCE,
                                                                                         PO.Q_TEMPLATE_INSTANCES)


def ranges(program: Program, query: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """`query` applied to `program`: (template or Definition id, first proof block index, proof block count) per root node computing
    proof blocks (a template query's) or per Call with proof blocks (`Q_word`'s and `Q_call`'s), in canonical order."""
    part = evaluate(program, query)
    if isinstance(part, PO.TemplatePartition):
        return [(tid, lo, n) for _k, tid, lo, n in part.ranges]
    out, at = [], 0
    for _k, c in part.calls:
        n = c.units if isinstance(part, PO.Partition) else len(c.units)
        if n:
            out.append((descriptor_id(c.fn), at, n))
            at += n
    return out


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of proof blocks, indexed 0.. in canonical order."""
    return evaluate(program, query).population


def strata(program: Program, query: Mapping[str, Any]) -> list[tuple[str, list[list[int]]]]:
    """`query` applied to `program`, grouped by template or Definition: one stratum per descriptor id, its proof blocks as
    ascending, disjoint half-open ranges `[lo, hi)` with adjacent ones merged, the strata in the order of their first
    proof block. These are the stratified sampling design's strata, which each verifier derives from its own copies."""
    groups: dict[str, list[list[int]]] = {}
    for tid, lo, n in ranges(program, query):
        rs = groups.setdefault(tid, [])
        if rs and rs[-1][1] == lo:
            rs[-1][1] = lo + n
        else:
            rs.append([lo, lo + n])
    return list(groups.items())


def unit_gates(program: Program, query: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Each proof block's root gates `[lo, hi)`, by proof block: an instance's (`TemplatePartition.gates`), a `Q_call` proof block's interval
    (`CallPartition.gates`), or the span of the computing gates a `Q_word` proof block owns, from the least to past the greatest,
    which another proof block's gates may interleave."""
    part = evaluate(program, query)
    if not isinstance(part, PO.Partition):
        return [part.gates(u) for u in range(part.population)]
    out: list[tuple[int, int]] = []
    offsets = program.root.body.offsets
    for p, (k, c) in enumerate(part.calls):
        keep, span = part.graph(p).keep, [None] * c.units
        for i, u in enumerate(c.owner()):
            g = offsets[k] + keep[i]
            span[u] = (g, g + 1) if span[u] is None else (min(span[u][0], g), max(span[u][1], g + 1))
        if None in span:
            raise QueryRefused(f"a unit of {descriptor_id(c.fn)} owns no gate", {"definition": descriptor_id(c.fn)})
        out += span
    return out


def in_order(program: Program, query: Mapping[str, Any], n: int) -> bool:
    """`Slots.popOk`: `query` applies to `program`, every proof block has its gates (`unit_gates`), and the proof blocks read stratum by
    stratum (`strata`) are `0, ..., n - 1` in order, so slot `u` is proof block `u`. The run refuses a registration whose program
    and query fail it (`population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's proof block `u`."""
    try:
        ss, gates = strata(program, query), unit_gates(program, query)
    except (QueryRefused, ValueError):
        return False
    us = [u for _t, rs in ss for lo, hi in rs for u in range(lo, hi)]
    return us == list(range(n)) and all(u < len(gates) for u in us)


def _once(inner: int, outer_stride: int, inner_stride: int, count: int) -> bool:
    """Whether a strided run of `count` references names each of its leaves at most once: its rows (or its columns) lie
    apart. A run whose rows and columns interleave counts as naming a leaf twice."""
    rows, cols = -(-count // inner), min(inner, count)
    if rows == 1 or cols == 1:
        return (cols == 1 or inner_stride != 0) and (rows == 1 or outer_stride != 0)
    return (inner_stride != 0 and abs(inner_stride) * (cols - 1) < abs(outer_stride)
            or outer_stride != 0 and abs(outer_stride) * (rows - 1) < abs(inner_stride))


def _runs(refs: Any) -> Iterator[tuple[int, int, int, bool]]:
    """`(node, lo, hi, once)` per run of root references in `refs`: each leaf of the run is one of leaves `[lo, hi)` of root
    node `node`'s result, and with `once` no two of the run's references name the same leaf. A strided run is taken at its
    span; an explicit one is a run a reference, which the description lists."""
    if not refs.count:
        return
    if isinstance(refs, Concat):
        for p in refs.parts:
            yield from _runs(p)
    elif isinstance(refs, Affine):
        if refs.space == "n":
            yield refs.idx, refs.base, refs.base + refs.count, True
    elif isinstance(refs, Strided):
        if refs.space == "n":
            q, r = divmod(refs.count - 1, refs.inner)
            corners = [(0, 0), (0, min(refs.inner, refs.count) - 1), (q, 0), (q, r), *([(q - 1, refs.inner - 1)] if q else [])]
            at = [refs.base + a * refs.outer_stride + b * refs.inner_stride for a, b in corners]
            yield (refs.idx, min(at), max(at) + 1,
                   _once(refs.inner, refs.outer_stride, refs.inner_stride, refs.count))
    else:
        yield from ((i, j, j + 1, True) for s, i, j in refs if s == "n")


def _template(program: Program, query: Mapping[str, Any], what: str) -> PO.TemplatePartition:
    part = evaluate(program, query)
    if not isinstance(part, PO.TemplatePartition):
        raise QueryRefused(f"{what} is a template query's, not {query.get('name')}'s", {"query": dict(query)})
    return part


def fanout(program: Program, query: Mapping[str, Any]) -> int:
    """F, the fan-out of a template query's partition of `program`, as the verifier summarizes it a root node at a time:
    at least the most proof blocks one group holds, two proof blocks joined when one reads part of the other's result and
    the groups closed under joining. F is 1 when no proof block reads another's result, as in every population program,
    whose instances each read only their own share of each Input node; the requests a group's proof blocks serve then
    read in no other group's, so a bound of S served bits per proof block holds each group's served bits to F S (the
    capped form at N = F, `plancheck.two`'s `fanout`). The summary joins root nodes, not proof blocks: a node reading
    any part of another's result joins the two, and F is the most proof blocks the nodes of one joined set hold, so it
    is the group's size exactly where every joined node is a call, and at least it where a joined batch's members read
    other proof blocks' results in parts. It reads each node's references once, never its members or its gates. A
    read of another proof block's values counts only as a reference to that block's result: a value another proof block
    writes that the program takes as an Input node joins nothing, which is why `shared_inputs` refuses a run Input two
    proof blocks read. `QueryRefused` for a query other than a template query."""
    part = _template(program, query, "fan-out")
    nodes = program.root.body.nodes
    members = {k: n for k, _t, _lo, n in part.ranges}
    parent = {k: k for k in members}

    def group(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    joined: set[int] = set()
    for k in members:
        for a in nodes[k].args:
            for i, _lo, _hi, _once_ in _runs(a.refs):
                if i in members:
                    parent[group(i)] = group(k)
                    joined.update((i, k))
    held = Counter()
    for k in joined:
        held[group(k)] += members[k]
    return max([1, *held.values()])


def shared_inputs(program: Program, query: Mapping[str, Any], held: Iterable[range] = ()) -> list[int]:
    """The run Input nodes of `program` (`program.INPUT_ID`; a registered input, and an Input node whose gates lie in one
    of `held`, the gates of the parameters the verifier expects registered, are registered values, held by R7) whose
    leaves a template query's proof blocks don't each read alone, in node order: a batch of more than one member reading
    one whole (an argument it broadcasts), or two proof blocks reading one of its leaves, as the summary sees them a root
    node at a time. A batch's members read disjoint elements of each argument it maps (`cut.member_refs`), so a batch
    reads an Input node's leaves alone when no two of its references to them name one leaf; a call is one proof block,
    whatever it reads twice. A strided run is taken at its span, and one that names a leaf twice, or two of a batch's
    runs that overlap, count as shared. A served port's rows reach a proof block either as its own leaves or through a
    registered value: a run Input two proof blocks read could hold rows one of them computed, which `fanout` can't see.
    `QueryRefused` for a query other than a template query."""
    part = _template(program, query, "an Input's readers")
    nodes, offsets, held = program.root.body.nodes, program.root.body.offsets, list(held)
    run_input = {k for k, node in enumerate(nodes) if is_input(node.fn) and not is_registered(node.fn)
                 and not any(h.start <= offsets[k] and offsets[k + 1] <= h.stop for h in held)}
    shared: set[int] = set()
    spans: dict[int, list[tuple[int, int]]] = {}
    for k, _t, _lo, n in part.ranges:
        node = nodes[k]
        axes = node.axes if node.form == "batch" else (None,) * len(node.args)
        mine: dict[int, list[tuple[int, int]]] = {}
        for a, ax in zip(node.args, axes):
            for i, lo, hi, once in _runs(a.refs):
                if i not in run_input:
                    continue
                if n > 1 and (ax is None or not once):
                    shared.add(i)
                mine.setdefault(i, []).append((lo, hi))
        for i, rs in mine.items():
            rs.sort()
            if n > 1 and any(b[0] < a[1] for a, b in zip(rs, rs[1:])):
                shared.add(i)
            spans.setdefault(i, []).extend(_merged(rs))
    for i, rs in spans.items():
        rs.sort()
        top = None
        for lo, hi in rs:
            if top is not None and lo < top:
                shared.add(i)
            top = hi if top is None else max(top, hi)
    return sorted(shared)


def within(program: Program, gates: range, k: int) -> bool:
    """Whether root node `k`'s gates lie in `gates`. The Lean's `Partition.within` is this."""
    offsets = program.root.body.offsets
    return gates.start <= offsets[k] and offsets[k + 1] <= gates.stop


def run_inputs(program: Program, gates: range) -> bool:
    """Whether the root's nodes whose gates lie in `gates` are run Input nodes, and one does: a parameter R8 may let
    through as tied is one the run binds. The Lean's `Partition.runInputs` is this."""
    nodes = program.root.body.nodes
    inside = [k for k in range(len(nodes)) if within(program, gates, k)]
    return bool(inside) and all(is_input(nodes[k].fn) and not is_registered(nodes[k].fn) for k in inside)


def _merged(rs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted spans `rs` with the overlapping ones merged."""
    out: list[tuple[int, int]] = []
    for lo, hi in rs:
        if out and lo < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


__all__ = ["template_id", "FORMAT", "QueryRefused", "digest", "evaluate", "fanout", "in_order", "is_template_query", "mixed_population_program", "partition",
           "POPULATION_VERSIONS", "population_program", "program_digest", "program_sha512", "ranges", "run_inputs", "shared_inputs", "strata", "template_instance_query",
           "template_instances_query", "unit_gates", "units", "within", "QUERY_X"]
