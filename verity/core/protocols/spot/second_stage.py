"""Round 2's statement check, as the verifier's `Spot.SecondStage` defines it, which the difftest's
`second_statements` holds this to: a two-stage run's second-stage sessions held, each by its statement, to the interior
registration and to the second draw's cells. `service.Stream` adds it to the interior registration's audit record as
`checks.statements` (`attach`). The parts and their cells, a statement, its codes and the form's switch are the Lean's
(its module docstring).

A statement is one second-stage session's: ``{"template", "header", "draw", "registered"}``, the template its circuit is
an instance of (its stratum's name under the check's query), its public file's header, its session record's
``unit_draw``, and the registered reads it was verified against (``--registered``), None for none.

The design's links are the verifier's own, derived from the replay block's Definition (`replay_links`, which the
difftest's ``replay_links`` holds to `SecondStage.replayLinks`) and never supplied.

`FORM` is `SecondStage.FORM`, which the run of record reads: off, every two-stage run is refused last
(``two-stage-form``), and the Python flipped alone disagrees with the run of record."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.codec import canonical_json, definitions, descriptor_id
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.refs import Concat, Explicit
from verity.ml.circuits.types import RecordType, Tuple

from .one_stage.verdict import Verdict, _Collect

FORM = False
FORM_WHY = ("refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the "
            "soundness proof does not cover")
CELLS_WHY = "the interior program's strata are not one range each, of a positive multiple of the replay sample's size"
STATEMENTS_WHY = ("a two-stage outcome names its second-stage sessions' statements: each one's template, header, draw and "
                  "registered reads")
REPORTED_WHY = ("the registered reads the verifiers of record report are not the statements' registered reads, value by "
                "value")
LINKS_WHY = ("the replay block's Definition reads a parameter, or returns, other than through templates of the "
             "interior's parts, each called once and reading one parameter alone or returned alone")
STMT_KEYS = ["draw", "header", "registered", "template"]


@dataclass(frozen=True)
class Part:
    """A template's part of the interior population: ``q·n`` slots from `lo`, `n` in each sampled replay block."""

    template: str
    lo: int
    n: int

    def holds(self, q: int, u: int) -> bool:
        return self.lo <= u < self.lo + q * self.n

    def span(self, u: int) -> int:
        """Slot `u`'s place in the sample, `u` one of the part's: the index of its cell's replay block."""
        return (u - self.lo) // self.n


@dataclass(frozen=True)
class Stmt:
    template: str
    header: dict[str, Any]
    draw: dict[str, Any]
    registered: dict[str, Any] | None


@dataclass(frozen=True)
class Link:
    """A link of the two-stage design: template `template`'s session port `port` reads the boundary registration's port
    `value`."""

    template: str
    port: str
    value: str


@dataclass(frozen=True)
class Own:
    """What the verifier holds of the run: the replay sample, the parts (`parts`), the interior and boundary
    registrations' records, the second draw's units, the design's links (`replay_links`, None when the replay block's
    Definition makes none), and the ``{value: root}`` the verifiers of record report the sessions read
    (`audit.session_registered`)."""

    sample: tuple[int, ...]
    parts: tuple[Part, ...] | None
    interior: Any
    boundary: Any
    drawn: tuple[int, ...]
    links: tuple[Link, ...] | None = ()
    reported: Mapping[str, Any] | None = None


def parts(q: int, strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> tuple[Part, ...] | None:
    """One part per stratum of the interior program (`partition.strata`), its one range of a positive multiple of the
    sample's `q` slots; None when a stratum is not such a range."""
    out = []
    for t, rs in strata:
        if len(rs) != 1 or q <= 0:
            return None
        lo, hi = rs[0]
        if not (lo < hi and (hi - lo) % q == 0):
            return None
        out.append(Part(t, lo, (hi - lo) // q))
    return tuple(out)


def _get(j: Any, k: str) -> Any:
    return j.get(k) if isinstance(j, dict) else None


def _nat(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _entries(j: Any) -> list[tuple[str, Any]]:
    return list(j.items()) if isinstance(j, dict) else []


def _jeq(a: Any, b: Any) -> bool:
    """`jeq`: the same canonical JSON, so `true` is not `1`."""
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def _by_port(rec: Any) -> dict[str, Any]:
    """`Record.byPort`: the record's last root of each port."""
    roots = _get(rec, "roots")
    out: dict[str, Any] = {}
    for r in roots if isinstance(roots, list) else ():
        if isinstance(_get(r, "port"), str):
            out.pop(r["port"], None)
            out[r["port"]] = r
    return out


def _stmt(j: Any) -> Stmt | None:
    if not (isinstance(j, dict) and sorted(j) == STMT_KEYS):
        return None
    t, h, d, r = j["template"], j["header"], j["draw"], j["registered"]
    if isinstance(t, str) and isinstance(h, dict) and isinstance(d, dict) and (r is None or isinstance(r, dict)):
        return Stmt(t, h, d, r)
    return None


def _port_of(named: bool, t: str, p: str) -> str:
    return f"{t}/{p}" if named else p


def _by_template(rec: Any) -> bool:
    """`SecondStage.byTemplate`: the record names its ports ``<template>/<port>``."""
    return any("/" in x for x in _by_port(rec))


def _outputs(t: Any) -> int:
    """`SecondStage.outputs`: the output ports ``o<j>`` of a circuit returning `t`, one per field: a tuple's or a
    record's fields, else one."""
    return len(t.fields) if isinstance(t, (Tuple, RecordType)) else 1


def _sources(r: Any) -> list[tuple[bool, int]]:
    """`SecondStage.sources`: what a body's reference `r` reads, parameters (True) and nodes (False), by index."""
    out: list[tuple[bool, int]] = []
    todo = [r]
    while todo:
        q = todo.pop()
        if isinstance(q, Concat):
            todo.extend(reversed(q.parts))
        elif isinstance(q, Explicit):
            out += [(s == "p", i) for s, i, _j in q.items]
        else:
            out.append((q.space == "p", q.idx))
    return out


def _callers(defs: Mapping[str, Any]) -> dict[str, int]:
    """`SecondStage.callers`: how many nodes of the bodies of `defs` call each definition, by descriptor id."""
    out: dict[str, int] = {}
    for f in defs.values():
        for node in () if isinstance(f, PrimitiveDefinition) else f.body.nodes:
            out[descriptor_id(node.fn)] = out.get(descriptor_id(node.fn), 0) + 1
    return out


def _template(callers: Mapping[str, int], node: Any) -> Any:
    """`SecondStage.template?`: the composite definition `node` calls or batches, when no other node calls it."""
    c = node.fn
    once = callers.get(descriptor_id(c)) == 1
    return c if node.form in ("call", "batch") and not isinstance(c, PrimitiveDefinition) and once else None


def _node_links(callers: Mapping[str, int], fn: Any, node: Any) -> list[Link] | None:
    """`SecondStage.nodeLinks`: for each argument of `node` that reads one of replay block `fn`'s parameters, its
    template's parameter there reads that one; None when the argument reads anything else too, or `node` is no
    template's."""
    params, out = [name for name, _t in fn.signature()[0]], []
    for a, arg in enumerate(node.args):
        src = _sources(arg.refs)
        first = next((x for x in src if x[0]), None)
        if first is None:
            continue
        c = _template(callers, node) if all(x == first for x in src) else None
        ports = [] if c is None else [name for name, _t in c.signature()[0]]
        if c is None or a >= len(ports) or first[1] >= len(params):
            return None
        out.append(Link(descriptor_id(c), ports[a], params[first[1]]))
    return out


def _return_links(callers: Mapping[str, int], fn: Any) -> list[Link] | None:
    """`SecondStage.returnLinks`: when replay block `fn` returns one node's output alone, that node a template's of one
    output and `fn` of one, that template's ``o0`` is `fn`'s; None when it returns anything else."""
    src, nodes = _sources(fn.body.ret), fn.body.nodes
    if not src:
        return []
    p, k = src[0]
    if p or any(x != (False, k) for x in src) or _outputs(fn.signature()[1]) != 1 or k >= len(nodes):
        return None
    c = _template(callers, nodes[k])
    return None if c is None or _outputs(c.signature()[1]) != 1 else [Link(descriptor_id(c), "o0", "o0")]


def links_of(defs: Mapping[str, Any], key: str) -> tuple[Link, ...] | None:
    """`SecondStage.linksOf`: the design's links, from replay block ``defs[key]``'s registered Definition, its
    parameters and its body: those its nodes make (`_node_links`) and its return (`_return_links`), each value its own
    port's name. None when its body reads a parameter, or returns, any other way."""
    fn = defs.get(key)
    if fn is None or isinstance(fn, PrimitiveDefinition):
        return None
    callers, out = _callers(defs), []
    for node in fn.body.nodes:
        links = _node_links(callers, fn, node)
        if links is None:
            return None
        out += links
    ret = _return_links(callers, fn)
    return None if ret is None else tuple(out + ret)


def replay_links(program: Any, key: str, rec: Any) -> tuple[Link, ...] | None:
    """`SecondStage.replayLinks`: the links of the replay block `program` defines under descriptor id `key` (`links_of`
    over the definitions its descriptor records, `codec.definitions`), each value the boundary registration's port for
    it, as record `rec` names its ports."""
    links, named = links_of(definitions(program), key), _by_template(rec)
    return None if links is None else tuple(Link(x.template, x.port, _port_of(named, key, x.value)) for x in links)


def _links_ok(ps: Sequence[Part], links: Sequence[Link] | None) -> bool:
    """`SecondStage.linksOk`: links are derived, each for a template one of the parts is of."""
    return links is not None and all(any(p.template == x.template for p in ps) for x in links)


def _link_of(own: Own, t: str, port: str) -> Link | None:
    return next((x for x in own.links or () if x.template == t and x.port == port), None)


def _population_ok(q: int, p: Part, h: Any) -> bool:
    indices, m = _get(_get(h, "units"), "indices"), _get(h, "instances")
    return (isinstance(indices, list) and _nat(m) and m == q * p.n and len(indices) == q * p.n
            and all(_nat(x) and x == p.lo + i for i, x in enumerate(indices)))


def _roots_ok(named: bool, rec: Any, t: str, h: Any, reads: Sequence[str]) -> bool:
    fv = _get(h, "frame_v3")
    roots = [(p, r) for p, r in _entries(_get(fv, "roots")) if p not in reads]
    reg = _by_port(rec)

    def held(p: str, r: Any) -> bool:
        e = reg.get(_port_of(named, t, p))
        return e is not None and _jeq(r, _get(e, "root")) and _jeq(_get(_get(fv, "domain_ids"), p), _get(e, "domain"))
    mine = [x for x in reg if not named or x.startswith(t + "/")]
    return (_jeq(_get(h, "scheme"), _get(rec, "scheme")) and all(held(p, r) for p, r in roots)
            and all(any(_port_of(named, t, p) == x for p, _ in roots) or any(_port_of(named, t, p) == x for p in reads)
                    for x in mine))


def own_draw(q: int, p: Part, drawn: Sequence[int]) -> dict[str, Any]:
    """The verifier's own sample for part `p`'s session: the second draw's units among its slots, counted from its first,
    as ``unit_draw`` writes a subset of its population."""
    units = [u - p.lo for u in drawn if p.holds(q, u)]
    return {"law": "subset", "population": q * p.n, "k": len(units), "units": units}


def _registered_entry(rec: Any, v: str) -> Any:
    es = _get(rec, "registered")
    return next((e for e in (es if isinstance(es, list) else ()) if _get(e, "value") == v
                 and isinstance(_get(e, "value"), str)), None)


def _same_at(e: Any, x: Any) -> bool:
    return all(_jeq(_get(e, k), _get(x, k)) for k in ("root", "domain", "leaves"))


def _rows_ok(q: int, p: Part, e: Any, units: int, unit: Callable[[int], int | None]) -> bool:
    """`SecondStage.rowsOk`: every read of registered-reads entry `e` (slot ``reads_from + k`` reads row ``reads[k]``,
    None for none) is by one of part `p`'s slots `u`, of a row of the unit ``unit(u)`` names, in a tree of `units` units'
    rows, ``leaves // units`` each."""
    first = _get(e, "reads_from")
    first = 0 if first is None else first
    leaves, rs = _get(e, "leaves"), _get(e, "reads")
    if not (_nat(leaves) and isinstance(rs, list) and _nat(first)) or units <= 0 or leaves % units:
        return False
    r = leaves // units

    def read_ok(u: int, x: Any) -> bool:
        if x is None:
            return True
        b = unit(u)
        return _nat(x) and b is not None and p.holds(q, u) and b * r <= x < (b + 1) * r
    return all(read_ok(first + k, x) for k, x in enumerate(rs))


def _read_ok(own: Own, named: bool, p: Part, port: str, e: Any) -> bool:
    """`SecondStage.readOk`: template ``p.template``'s session port `port` may read registered-reads entry `e`: a port a
    link names, only the boundary port it names, each slot only its cell's replay block's rows; any other, its own port
    of the interior record, each slot only its own rows, or a value either registration registers, never a boundary
    port. Each at its root, domain and leaves."""
    v, q = _get(e, "value"), len(own.sample)
    if not isinstance(v, str):
        return False
    link = _link_of(own, p.template, port)
    if link is not None:
        b, m = _by_port(own.boundary).get(v), _get(own.boundary, "population")

        def block(u: int) -> int | None:
            i = p.span(u)
            return own.sample[i] if 0 <= i < q else None
        return v == link.value and b is not None and _nat(m) and _same_at(e, b) and _rows_ok(q, p, e, m, block)
    x = _by_port(own.interior).get(v)
    if x is not None:
        return (v == _port_of(named, p.template, port) and _same_at(e, x)
                and _rows_ok(q, p, e, q * p.n, lambda u: u - p.lo))
    if v in _by_port(own.boundary):
        return False
    x = _registered_entry(own.boundary, v)
    if x is None:
        x = _registered_entry(own.interior, v)
    return x is not None and _same_at(e, x)


def _linked_read(own: Own, t: str, reads: Sequence[tuple[str, Any]]) -> bool:
    """`SecondStage.linkedRead`: every port a link names for template `t` is one of `reads`."""
    return all(x.template != t or any(port == x.port for port, _ in reads) for x in own.links or ())


def _stmt_codes(own: Own, named: bool, p: Part, s: Stmt) -> list[str]:
    q, reads = len(own.sample), _entries(_get(s.registered, "ports"))
    checks = (("second-population", _population_ok(q, p, s.header)),
              ("second-roots", _roots_ok(named, own.interior, s.template, s.header, [name for name, _ in reads])),
              ("second-sample", _jeq(s.draw, own_draw(q, p, own.drawn))),
              ("second-read", all(_read_ok(own, named, p, port, e) for port, e in reads)
               and _linked_read(own, p.template, reads)))
    return [code for code, ok in checks if not ok]


def _templates(ss: Sequence[Any]) -> list[str]:
    return [s.template for s in map(_stmt, ss) if s is not None]


def statement_reads(ss: Sequence[Any]) -> dict[str, Any]:
    """`SecondStage.statementReads`: the ``{value: root}`` statements `ss` read, merged as `audit.session_registered`
    merges the verdicts': a value read under two roots maps to None."""
    out: dict[str, Any] = {}
    for s in map(_stmt, ss):
        for _, e in _entries(_get(s.registered, "ports")) if s is not None else ():
            v = _get(e, "value")
            if isinstance(v, str):
                root = _get(e, "root")
                out[v] = (out[v] if _jeq(out[v], root) else None) if v in out else root
    return out


def _same_dict(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return len(a) == len(b) and all(k in b and _jeq(v, b[k]) for k, v in a.items())


def _failures(allow: bool, own: Own, ps: Sequence[Part], ss: Sequence[Any]) -> list[tuple[str, Any]]:
    named, q = _by_template(own.interior), len(own.sample)
    found: list[tuple[str, Any]] = []
    for i, j in enumerate(ss):
        s = _stmt(j)
        if s is None:
            found.append(("second-statement", {"statement": i, "template": None}))
            continue
        p = next((p for p in ps if p.template == s.template), None)
        detail = {"statement": i, "template": s.template}
        if p is None or s.template in _templates(ss[:i]):
            found.append(("second-template", detail))
        else:
            found += [(c, detail) for c in _stmt_codes(own, named, p, s)]
    covered = _templates(ss)
    found += [("second-cover", {"template": p.template}) for p in ps
              if p.template not in covered and any(p.holds(q, u) for u in own.drawn)]
    if not _same_dict(own.reported or {}, statement_reads(ss)):
        found.append(("second-reported", REPORTED_WHY))
    return found if found or allow else [("two-stage-form", FORM_WHY)]


def check(allow: bool, own: Own, statements: Sequence[Any] | None) -> Verdict:
    """Round 2's statement check of `statements`, the two-stage form allowed when `allow` (`FORM` in a run): each code
    once, with the detail it first failed with (`SecondStage.check`)."""
    if own.parts is None:
        found = [("second-cells", CELLS_WHY)]
    elif not _links_ok(own.parts, own.links):
        found = [("second-links", LINKS_WHY)]
    elif statements is None:
        found = [("second-statements", STATEMENTS_WHY)]
    else:
        found = _failures(allow, own, own.parts, list(statements))
    c = _Collect()
    for code, what in found:
        c.fail(code, what)
    return c.verdict()


def attach(rec: Mapping[str, Any], v: Verdict) -> dict[str, Any]:
    """The interior registration's audit record `rec` with round 2's check `v` as ``checks.statements``, failing where `v`
    does as a failed check of its own fails it: not accepted, and no profile, integrity profile or wrong units."""
    out = {**rec, "checks": {**rec["checks"], "statements": v.to_json()}}
    return out if v.ok else {**out, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None}


__all__ = ["FORM", "Link", "Own", "Part", "attach", "check", "links_of", "own_draw", "parts", "replay_links",
           "statement_reads"]
