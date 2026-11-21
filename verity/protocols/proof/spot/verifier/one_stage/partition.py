"""The partition as a named query over a program, P = Q(C): core's `verity/partition/v1` (`verity.circuits.partition_object`),
applied here with the template queries `Q_template_instance` v0 and `Q_template_instances` v0, `Q_word` v1 at `X = 16`
and `Q_call` v1 at `X <= 16`, and the programs this protocol builds for the template queries.

    {"format": "verity/partition/v1", "program": "<SHA-512 of the program>",
     "query": {"name": "Q_template_instance", "version": 0, "params": {"template": "<Definition id>"}}}

* **Object, digest, program digest** are core's (`partition`, `digest`, `program_digest`, also named `program_sha512`). A
  record made before the canonical envelope names `{"program", "query"}`; `digest` gives it the envelope's rule,
  `SHA-512("verity/partition/v1\\0" || canon(object))`, which the roots registered with it bind.
* **Proof blocks** are `verity.circuits.partition_query`'s: core's evaluation (`partition_object.evaluate`) of a query of valid form (`partition_object.validate`):
  `QueryRefused` for a program that is not topological, or one the query does not apply to. The template queries (the
  served ones: a served window registers a population program under one), `Q_word` v1 at `X = 16` and `Q_call` v1 at
  `X <= 16` are the ones applied here. A template query reads the root body only; `Q_word` and `Q_call` lay out one
  activation of each distinct Call's Definition gate by gate, so they cost its unrolled size
  (`benchmarks/verifier_cost/`'s `partition`). Every proof block's committed bits fit a public size class
  (`verity.circuits.partition_v2`'s `SIZE_CLASSES`, powers of two from 16 to `SIZE_CLASS_MAX` bits; provisional), its
  class the least that holds them (`size_class`), a public property of its Definition: a program with a proof block wider
  than the widest class is refused (a template instance's bits are its distinct returned gates', `_template_bits`; a
  `Q_call` proof block's count the constants and Input gates of its Definition's activation, `_committed_bits`, which `Q_call`'s width rule leaves
  out; a `Q_word` proof block's are its cut's), as is one with a template of no gate. `ranges`, `units` and `strata` read their
  proof blocks in canonical order (node order, then member, for a template query; Call order, then the Call's own, for
  `Q_word` and `Q_call`), and `unit_gates` gives
  each proof block's root gates. `in_order` holds when the proof blocks read stratum by stratum are the population in order.
  `fanout` gives a template query's fan-out F, at least the most proof blocks one group joined by reads of each other's
  results holds, a root node at a time (`registration.check`).
* **Programs**: a partition's program is built by `verity.circuits.population`.

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
proof block index on the prover's word.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.partition_object import (FORMAT, QueryRefused, program_sha512, template_id,
                                        template_instance_query, template_instances_query)
from verity.circuits.partition_query import QUERY_X, evaluate, is_template_query, ranges, strata
from verity.circuits.program import RUN, Program, input_id, is_input, is_registered
from verity.circuits.refs import Affine, Concat, Grid, Strided
from verity.circuits.types import axis_runs

_PRE_CANONICAL = {"program", "query"}


def partition(program: Program, query: Mapping[str, Any]) -> dict[str, Any]:
    return PO.build(program, query)


def digest(obj: Mapping[str, Any]) -> str:
    """Core's digest of a valid `verity/partition/v1` object (`ValueError` otherwise), or of a pre-canonical `{program, query}`."""
    if isinstance(obj, Mapping) and set(obj) == _PRE_CANONICAL:
        return hashlib.sha512(FORMAT.encode() + b"\0" + PO.canonical(dict(obj))).hexdigest()
    return PO.digest(obj)


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of proof blocks, indexed 0.. in canonical order."""
    return evaluate(program, query).population


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
    stratum (`strata`) are `0, ..., n - 1` in order, so slot `u` is proof block `u`. The run refuses a docket whose program
    and query fail it (`population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's proof block `u`."""
    try:
        ss, gates = strata(program, query), unit_gates(program, query)
    except (QueryRefused, ValueError):
        return False
    us = [u for _t, rs in ss for lo, hi in rs for u in range(lo, hi)]
    return us == list(range(n)) and all(u < len(gates) for u in us)


def cover_ranges(rs: Iterable[Sequence[int]]) -> int | None:
    """`FilePart.coverRanges`: N where the ranges ``[lo, hi)``, in any order, are a disjoint cover of ``0, ..., N - 1``
    (a range with ``hi <= lo`` holding no unit), else None."""
    at = 0
    for lo, hi in sorted(((lo, hi) for lo, hi in rs if lo < hi), key=lambda r: r[0]):
        if lo != at:
            return None
        at = hi
    return at


def cover(strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> int | None:
    """`FilePart.cover`: `cover_ranges` of every stratum's ranges together, None too where a range ends before it
    starts."""
    rs = [(lo, hi) for _t, s in strata for lo, hi in s]
    return None if any(hi < lo for lo, hi in rs) else cover_ranges(rs)


def covers(program: Program, query: Mapping[str, Any], n: int) -> bool:
    """`Slots.popCovers` (run as `Slots.popCoverOk`), `in_order`'s cover form: `query` applies to `program`, the strata's ranges are a disjoint cover
    of ``0, ..., n - 1`` (`cover`) and every one of those proof blocks has its gates. What a subset design's draw needs:
    it reads no stratum, and slot `u` is proof block `u`."""
    try:
        ss, gates = strata(program, query), unit_gates(program, query)
    except (QueryRefused, ValueError):
        return False
    return cover_ranges(r for _t, s in ss for r in s) == n and n <= len(gates)


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
    if isinstance(refs, (Concat, Grid)):
        for p in refs.parts if isinstance(refs, Concat) else refs.pieces():
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
    writes that the program takes as an Input node joins nothing: a run Input's rows are one tree the members' check
    holds. `QueryRefused` for a query other than a template query."""
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
    The variant with no members refuses such a node (`R8-shared-input`); the per-member variant holds each reader to the
    node's one tree instead. `QueryRefused` for a query other than a template query."""
    part = _template(program, query, "an Input's readers")
    nodes, offsets, held = program.root.body.nodes, program.root.body.offsets, list(held)
    run_input = {k for k, node in enumerate(nodes) if _run_bound(node.fn)
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


def input_reads(program: Program, query: Mapping[str, Any], held: Iterable[range]) -> list[tuple[str, str, range]]:
    """Each template parameter whose argument reads an Input node inside one of `held`, as ``(template, parameter, the
    range of `held` it reads)``, a root node at a time, never its members or its gates. `QueryRefused` for a query other
    than a template query."""
    part = _template(program, query, "an Input's readers")
    nodes, held = program.root.body.nodes, list(held)
    inside = {k: h for k, node in enumerate(nodes) if is_input(node.fn) for h in held if within(program, h, k)}
    out: list[tuple[str, str, range]] = []
    for k, t, _lo, _n in part.ranges:
        params = [p for p, _ in nodes[k].fn.signature()[0]]
        for p, a in zip(params, nodes[k].args):
            out.extend((t, p, inside[i]) for i in dict.fromkeys(i for i, *_ in _runs(a.refs) if i in inside))
    return out


def held_value(program: Program, held: Sequence[tuple[str, range]], i: int) -> str | None:
    """The value of `held` whose gates hold root node `i`, an Input node (the first that holds it); None for another node
    or one outside them. The Lean's `Partition.heldValue` is this."""
    if not is_input(program.root.body.nodes[i].fn):
        return None
    return next((v for v, g in held if within(program, g, i)), None)


def member_nodes(program: Program, k: int, m0: int, m1: int) -> list[tuple[str, int]]:
    """Each ``(parameter, root node)`` whose argument root node `k`'s members ``[m0, m1)`` read names, in parameter and
    part order, once each. A part counts where it meets the members' own leaves of a mapped argument (a batch's mapped
    axis, a scan's xs), modulo the period the axis repeats at, and wherever an unmapped argument has it. The Lean's
    `Partition.memberNodes` is this."""
    node = program.root.body.nodes[k]
    out: list[tuple[str, int]] = []
    for pi, (p, a) in enumerate(zip([p for p, _ in node.fn.signature()[0]], node.args)):
        dims = _dims(node, pi, a.type)
        parts = a.refs.parts if isinstance(a.refs, Concat) else [a.refs]
        starts = a.refs.starts if isinstance(a.refs, Concat) else [0]
        for q, s in zip(parts, starts):
            if q.count and _meets(dims, m0, m1, s, q.count):
                out.extend(x for i in _part_nodes(q) if (x := (p, i)) not in out)
    return out


def held_of(program: Program, held: Sequence[tuple[str, range]], ns: Iterable[tuple[str, int]]) -> list[tuple[str, str]]:
    """The ``(parameter, value)``s of `ns` whose node is an Input inside the gates of a value of `held` (`held_value`),
    once each. The Lean's `Partition.heldOf` is this."""
    out: list[tuple[str, str]] = []
    for p, i in ns:
        if (v := held_value(program, held, i)) is not None and (p, v) not in out:
            out.append((p, v))
    return out


def member_reads(program: Program, held: Sequence[tuple[str, range]], k: int, m0: int, m1: int) -> list[tuple[str, str]]:
    """The ``(parameter, value)``s of `held` root node `k`'s members ``[m0, m1)`` read (`member_nodes`, `held_of`). The
    Lean's `Partition.memberReads` is this."""
    return held_of(program, held, member_nodes(program, k, m0, m1))


def _dims(node: Any, pi: int, t: Any) -> tuple[int, int] | None:
    """``(d, L)`` of argument `pi` of `node` where its members each read a run of `L` leaves at m of `d` (the Lean's
    `Arg.dims`); None where every member reads all of it."""
    if node.form == "batch" and node.axes[pi] is not None:
        _outer, d, leaves = axis_runs(t, node.axes[pi])
        return d, leaves
    if node.form == "scan" and 1 <= pi <= node.n_xs:
        return t.n, t.elem.leaves
    return None


def _meets(dims: tuple[int, int] | None, m0: int, m1: int, s: int, n: int) -> bool:
    if dims is None:
        return True
    d, leaves = dims
    per = d * leaves
    if per == 0 or n >= per:
        return True
    o = s % per
    return (o < m1 * leaves and m0 * leaves < o + n) or m0 * leaves + per < o + n


def _part_nodes(q: Any) -> list[int]:
    """The root nodes part `q` of an argument names (the Lean's `Registered.partNodes`)."""
    if isinstance(q, (Affine, Strided)):
        return [q.idx] if q.space == "n" else []
    if isinstance(q, Grid):
        return [i for s, i, *_ in q.lanes if s == "n"]
    return [i for s, i, _j in q if s == "n"]


def within(program: Program, gates: range, k: int) -> bool:
    """Whether root node `k`'s gates lie in `gates`. The Lean's `Partition.within` is this."""
    offsets = program.root.body.offsets
    return gates.start <= offsets[k] and offsets[k + 1] <= gates.stop


def any_run_input(program: Program, gates: range) -> bool:
    """Whether a root node whose gates lie in `gates` is a run Input node: a registered or supplied value's gates hold
    none (`R7-run-input`). The Lean's `Partition.anyRunInput` is this."""
    nodes = program.root.body.nodes
    return any(within(program, gates, k) and input_id(nodes[k].fn.id) is not None and input_id(nodes[k].fn.id)[0] == RUN
               for k in range(len(nodes)) if is_input(nodes[k].fn))


def run_inputs(program: Program, gates: range) -> bool:
    """Whether the root's nodes whose gates lie in `gates` are run Input nodes, and one does: a parameter R8 may let
    through as tied is one the run binds. The Lean's `Partition.runInputs` is this."""
    nodes = program.root.body.nodes
    inside = [k for k in range(len(nodes)) if within(program, gates, k)]
    return bool(inside) and all(_run_bound(nodes[k].fn) for k in inside)


def _run_bound(fn: Any) -> bool:
    """Whether `fn` is an input the run binds, neither registered nor supplied (Lean's `binding == RUN`)."""
    return is_input(fn) and (i := input_id(fn.id)) is not None and i[0] == RUN


def _merged(rs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted spans `rs` with the overlapping ones merged."""
    out: list[tuple[int, int]] = []
    for lo, hi in rs:
        if out and lo < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


__all__ = ["template_id", "FORMAT", "QueryRefused", "cover", "cover_ranges", "covers", "digest", "evaluate", "fanout", "in_order", "is_template_query", "partition",
           "program_digest", "program_sha512", "held_of", "held_value", "member_nodes", "member_reads", "ranges", "any_run_input", "run_inputs", "shared_inputs", "strata", "template_instance_query",
           "template_instances_query", "unit_gates", "units", "within", "QUERY_X"]
