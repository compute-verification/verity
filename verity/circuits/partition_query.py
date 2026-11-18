"""The partition query a verifier applies: core's evaluation (`partition_object.evaluate`) of the queries applied here,
refusing a proof block wider than the widest public class, and its proof blocks read by template or Definition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_id
from verity.circuits.cut import boundary_widths, call_scope
from verity.circuits.layout import resolve
from verity.circuits.partition_object import FORMAT, QueryRefused
from verity.circuits.partition_v2 import SIZE_CLASS_MAX, size_class
from verity.circuits.program import Program, is_input
from verity.circuits.types import Value

#: `Q_word` v1's `X`, and the most `Q_call` v1's `X` may be: the width rule of the queries applied here.
QUERY_X = 16


def _check_word(part: PO.Partition) -> None:
    """Each `Q_word` proof block's committed bits fit a public class (`size_class`): each Definition's cut, once, on its Call's
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
        wide = [u for u, b in enumerate(ob) if size_class(b) is None]
        if wide:
            raise QueryRefused(f"unit {wide[0]} of {fid} commits {ob[wide[0]]} bits, more than the widest class, "
                               f"{SIZE_CLASS_MAX}", {"definition": fid, "unit": wide[0], "bits": ob[wide[0]]})


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
    (`size_class`; `_template_bits`, and `_committed_bits` with constants counted, which `Q_call`'s width rule leaves
    out), each Definition once, as activations of one Definition commit alike."""
    if isinstance(part, PO.TemplatePartition):
        templates = {t: program.root.body.nodes[k].fn for k, t, _first, n in part.ranges if n}
        for t, fn in templates.items():
            if fn.gates == 0:
                raise QueryRefused(f"template {t} has no gate", {"template": t})
            bits = _template_bits(fn)
            if size_class(bits) is None:
                raise QueryRefused(f"an instance of {t} commits {bits} bits, more than the widest class, {SIZE_CLASS_MAX}",
                                   {"template": t, "bits": bits})
        return
    for cut in {cut.fn.id: cut for _k, cut in part.calls}.values():
        fn = cut.fn
        bits = _committed_bits(fn, [(lo, hi) for lo, hi, _path in cut.units])
        wide = next((u for u, b in enumerate(bits) if size_class(b) is None), None)
        if wide is not None:
            raise QueryRefused(f"unit {wide} of {fn.id} commits {bits[wide]} bits, more than the widest class, "
                               f"{SIZE_CLASS_MAX}", {"definition": fn.id, "unit": wide, "bits": bits[wide]})


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
    if kind == PO.Q_WORD and x == QUERY_X:
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
