"""Round 2's statement check, as the verifier's `Spot.SecondStage` defines it, which the difftest's
`second_statements` holds this to: a two-stage run's second-stage sessions held, each by its statement, to the interior
registration and to the second draw's cells. `service.Stream` adds it to the interior registration's audit record as
`checks.statements` (`attach`). The parts and their cells, a statement, its codes and the form's switch are the Lean's
(its module docstring).

A statement is one second-stage session's: ``{"template", "header", "draw", "registered"}``, the template its circuit is
an instance of (its stratum's name under the check's query), its public file's header, its session record's
``unit_draw``, and the registered reads it was verified against (``--registered``), None for none. Round 2 takes it only
where an accepted verdict of record names its header, its registered reads and its sample (``second-verified``,
`verified_of`): ``public_header_sha512`` and ``registered_sha512``, the SHA-512 of the header's and the
``--registered`` file's canonical JSON, and ``unit_draw``, which `flock-verify verify` writes of the session it checked;
and only where every one that does checked its session as one table (``session_tables``, ``second-tables``): a session
split across Flock tables is refused, as one split across statements is (``second-template``).

The design's links are the verifier's own, derived from the replay block's Definition (`replay_links`, which the
difftest's ``replay_links`` holds to `SecondStage.replayLinks`) and never supplied; so are its wires (`replay_wires`,
`SecondStage.replayWires`): a template's parameter that is another template's whole output row reads that writer's
output port, one value the interior commits once, so ``second-read`` holds a wired port to its writer's port at its
writer's rows and `at_run` binds it by that port (`_held_port`).

Each statement's header is held to the reference header the auditor's params give its template (`Own.refs`, from
``proofs.templates``: `service.template_circuits`), the registered reference statement's, fixed before the run, placed
in the run (`at_run`): its ``set`` the session's input set, ``served/<run>/m<k>`` (`set_name`, ``<run>`` the interior
record's ``window.run`` and ``k`` its template's position among the parts, its member's in the interior window), and its
``frame_v3.bindings`` each of its ``frame_v3.schemas`` ports' bindings in the run (`port_binding`, the window-domain rule
over the interior record's program, partition and that port's leaves), both of which the verifier computes, so the
prover chooses neither. The two are the same with the run's fields blanked (`blank_run`, `RUN_FIELDS`): ``instances``
and ``units.indices``, which ``second-population`` holds, ``units.blocks``, the indices cut into blocks, which the
session's verifier holds to them, ``frame_v3.roots`` and ``frame_v3.domain_ids``, which ``second-roots`` holds, and
``registered``, whose reads ``second-read`` holds (``second-header``).

A header that shares rows (``shared_rows``: tables whose rows its instances read by the refs in its public file's body)
shares them only at ports its statement reads registered (``second-refs``, `_refs_ok`): there the session's verifier
holds each instance's ref to the row at the position the read gives its unit (`Flock.Registered.check`), and
``second-read`` holds the read to the slot's own rows or its cell's replay block's; elsewhere it only bounds a ref by its
table, so a ref could point a drawn unit at another unit's rows.

Each statement's registered reads are its template's reference statement's (`Own.ref_reads`, the registered reads the
auditor's params give beside its header, fixed before the run and taken as given: round 2 never verifies the reference
statement against them), the two the same with each port's run fields blanked (`blank_reads`) (``second-values``,
`_values_ok`). The run's fields are those ``second-read`` holds
to the run: every port's ``root``, ``domain`` and ``leaves`` (`READ_RUN_FIELDS`), and the rows (``reads``,
``reads_from``: `READ_ROW_FIELDS`) of a port a link names or one reading a port of the interior record (its own, or at
a port a wire names its writer's), whose rows the run places (`_run_places_rows`). So the ports a session reads
registered, the value each reads, and the rows at which it reads a value a registration registers are fixed before the
run: ``second-read`` alone lets a port no link or wire names read any value either registration registers, at any rows.

The interior record names the replay sample it lays out (``window.replay_sample``, which the proof server writes), and
``second-interior-order`` fails unless it is the verifier's own, in its order (`interior_order_ok`), checked before any
statement is read.

`FORM` is `SecondStage.FORM`, which the run of record reads: off, every two-stage run is refused last
(``two-stage-form``), and the Python flipped alone disagrees with the run of record."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.codec import canonical_json, definitions, descriptor_id
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.refs import Affine, Concat, Explicit
from verity.ml.circuits.types import RecordType, Tuple
from verity.ml.commitments.window import window_binding

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
             "interior's parts, each called once and reading one parameter alone, returned alone, or returned whole as "
             "one field each")
WIRES_WHY = ("the replay block's Definition reads a node other than as a template's parameter that is one row of another "
             "template's one output, each called once, or a wire names a template of no part")
INTERIOR_ORDER_WHY = "the interior record's replay sample (window.replay_sample) is not the verifier's own, in its order"
STMT_KEYS = ["draw", "header", "registered", "template"]
#: The run's fields of a statement's header, each a path of keys: the only ones a header may differ from its template's
#: reference placed in the run in, which other checks hold instead: ``second-population`` the first two, the session's
#: verifier ``units.blocks``, the indices cut into blocks of ``vus_per_block`` (its header's ``units`` must be
#: `Flock.HmRow.unitsEntry` of its indices), ``second-roots`` the next two, and ``second-read`` ``registered``, each
#: read's root, domain and rows, with ``second-values`` the value each port reads, and the rows at which it reads a
#: value a registration registers, through the ``--registered`` file the session's verifier holds it to
#: (`SecondStage.RUN_FIELDS`). A reference staged at other slots differs from the header only in the indices and their
#: blocks.
RUN_FIELDS = (("instances",), ("units", "indices"), ("units", "blocks"), ("frame_v3", "roots"), ("frame_v3", "domain_ids"),
              ("registered",))


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
class Checked:
    """What an accepted verdict of record names of the session it checked: the digests of the public file's header and
    of the ``--registered`` file it checked the session against (``public_header_sha512``, ``registered_sha512``, None
    for none), the sample it checked it at (``unit_draw``) and the number of tables it set the session up as
    (``session_tables``)."""

    header: str
    registered: Any
    draw: Any
    tables: Any


#: `SecondStage.ONE_TABLE`: the ``session_tables`` a verdict of record names of a session round 2 reads
ONE_TABLE = 1


def verified_of(verdicts: Mapping[str, Any]) -> tuple[Checked, ...]:
    """`SecondStage.verifiedOf`: what each accepted verdict of record among `verdicts` that names a header names."""
    return tuple(Checked(v["public_header_sha512"], v.get("registered_sha512"), v.get("unit_draw"),
                         v.get("session_tables"))
                 for v in verdicts.values() if isinstance(v, Mapping) and isinstance(v.get("public_header_sha512"), str)
                 and v.get("of_record") and v.get("accepted") is True)


def _integral(j: Any) -> bool:
    """Whether JSON value `j` holds no non-integer number, the one value Lean's `canon` refuses."""
    if isinstance(j, float):
        return False
    if isinstance(j, Mapping):
        return all(_integral(v) for v in j.values())
    return not isinstance(j, (list, tuple)) or all(_integral(v) for v in j)


def _sha512(j: Any) -> str | None:
    """`canonSha512`: the SHA-512 of JSON value `j`'s canonical JSON, None where it holds a non-integer number."""
    return hashlib.sha512(canonical_json(j)).hexdigest() if _integral(j) else None


def _checked(s: Stmt) -> Checked | None:
    """`Stmt.checked`: what a verdict of record that checked statement `s`'s session, as round 2 reads one, names of it
    (one table), None when its header or registered reads hold a non-integer number."""
    h = _sha512(s.header)
    r = None if s.registered is None else _sha512(s.registered)
    return None if h is None or (s.registered is not None and r is None) else Checked(h, r, s.draw, ONE_TABLE)


@dataclass(frozen=True)
class Link:
    """A link of the two-stage design: template `template`'s session port `port` reads the boundary registration's port
    `value`."""

    template: str
    port: str
    value: str


@dataclass(frozen=True)
class Wire:
    """A wire of the two-stage design: template `template`'s session port `port` reads template `writer`'s output port
    `wport`, one value the interior commits once, at the writer's port."""

    template: str
    port: str
    writer: str
    wport: str


@dataclass(frozen=True)
class Own:
    """What the verifier holds of the run: the replay sample, the parts (`parts`), the interior and boundary
    registrations' records, the second draw's units, the design's links (`replay_links`, None when the replay block's
    Definition makes none), the ``{value: root}`` the verifiers of record report the sessions read
    (`audit.session_registered`), what they name of each session they accepted (`verified_of`), the reference header
    of each template's statement, ``(template, header)``, and the registered reads it reads,
    ``(template, registered)``, None for none (`ref_reads`), as the auditor's params give them
    (`service.template_circuits`), and the design's wires (`replay_wires`, None when the Definition makes none)."""

    sample: tuple[int, ...]
    parts: tuple[Part, ...] | None
    interior: Any
    boundary: Any
    drawn: tuple[int, ...]
    links: tuple[Link, ...] | None = ()
    reported: Mapping[str, Any] | None = None
    verified: tuple[Checked, ...] = ()
    refs: tuple[tuple[str, Any], ...] = ()
    ref_reads: tuple[tuple[str, Any], ...] = ()
    wires: tuple[Wire, ...] | None = ()


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


def _fields(t: Any) -> list[Any]:
    """`SecondStage.fields`: the fields of a circuit returning `t`, one per output port: a tuple's or a record's, else
    `t`."""
    if isinstance(t, Tuple):
        return list(t.fields)
    return [f for _name, f in t.fields] if isinstance(t, RecordType) else [t]


def _parts(r: Any) -> list[Any]:
    """A body reference's parts, its concatenations spliced in, as the descriptor's ``Seq`` holds them."""
    return [x for q in r.parts for x in _parts(q)] if isinstance(r, Concat) else [r]


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


def _field_link(callers: Mapping[str, int], fn: Any, j: int, f: Any, r: Any) -> tuple[int, Link] | None:
    """`SecondStage.fieldLink`: the link field `j` of replay block `fn`'s return, of type `f`, makes when it is part `r`
    of the return: when `r` is a node's whole output, the node a template's of one output, that template's ``o0`` is
    `fn`'s ``o<j>``; with it, the node's index. None when it is anything else."""
    nodes = fn.body.nodes
    if not (isinstance(r, Affine) and r.space == "n" and r.base == 0 and r.idx < len(nodes)):
        return None
    c = _template(callers, nodes[r.idx])
    if r.count != f.leaves or r.count != nodes[r.idx].out.leaves or c is None or _outputs(c.signature()[1]) != 1:
        return None
    return r.idx, Link(descriptor_id(c), "o0", f"o{j}")


def _return_links(callers: Mapping[str, int], fn: Any) -> list[Link] | None:
    """`SecondStage.returnLinks`: when replay block `fn` returns one node's output alone, that node a template's of one
    output and `fn` of one, that template's ``o0`` is `fn`'s; when `fn` returns several fields, each part of its return
    one field, each one node's whole output and no node's twice (`_field_link`), each one's template's ``o0`` is its
    field's. None when it returns anything else."""
    src, nodes, ret = _sources(fn.body.ret), fn.body.nodes, fn.signature()[1]
    if not src:
        return []
    if _outputs(ret) == 1:
        p, k = src[0]
        if p or any(x != (False, k) for x in src) or k >= len(nodes):
            return None
        c = _template(callers, nodes[k])
        return None if c is None or _outputs(c.signature()[1]) != 1 else [Link(descriptor_id(c), "o0", "o0")]
    fs, parts = _fields(ret), _parts(fn.body.ret)
    got = [_field_link(callers, fn, j, f, r) for j, (f, r) in enumerate(zip(fs, parts))]
    links = [x for x in got if x is not None]
    if len(parts) != len(fs) or len(links) != len(got) or len({k for k, _ in links}) != len(links):
        return None
    return [x for _, x in links]


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


def _node_wires(callers: Mapping[str, int], fn: Any, node: Any) -> list[Wire] | None:
    """`SecondStage.nodeWires`: for each argument of `node` that reads a node, its template's parameter there reads that
    node's whole output, one row of the node's template, of one output (``o0``) and so of one instance, and one row of
    the parameter. None when such an argument reads anything else too, or part of a row, or a node no template's, or
    `node` is no template's. An argument that reads only `fn`'s parameters is a link's (`_node_links`)."""
    nodes, out = fn.body.nodes, []
    for a, arg in enumerate(node.args):
        if all(p for p, _i in _sources(arg.refs)):
            continue
        rs = _parts(arg.refs)
        r = rs[0] if len(rs) == 1 else None
        if not (isinstance(r, Affine) and r.space == "n" and r.base == 0 and r.idx < len(nodes)):
            return None
        tw, t = _template(callers, nodes[r.idx]), _template(callers, node)
        params = [] if t is None else t.signature()[0]
        if tw is None or a >= len(params) or _outputs(tw.signature()[1]) != 1 or not (
                r.count == nodes[r.idx].out.leaves == tw.signature()[1].leaves == params[a][1].leaves):
            return None
        out.append(Wire(descriptor_id(t), params[a][0], descriptor_id(tw), "o0"))
    return out


def wires_of(defs: Mapping[str, Any], key: str) -> tuple[Wire, ...] | None:
    """`SecondStage.wiresOf`: the design's wires, from replay block ``defs[key]``'s registered Definition: those its
    nodes make (`_node_wires`). None when its body reads a node any other way."""
    fn = defs.get(key)
    if fn is None or isinstance(fn, PrimitiveDefinition):
        return None
    callers, out = _callers(defs), []
    for node in fn.body.nodes:
        wires = _node_wires(callers, fn, node)
        if wires is None:
            return None
        out += wires
    return tuple(out)


def replay_wires(program: Any, key: str) -> tuple[Wire, ...] | None:
    """`SecondStage.replayWires`: the wires of the replay block `program` defines under descriptor id `key` (`wires_of`
    over the definitions its descriptor records, `codec.definitions`)."""
    return wires_of(definitions(program), key)


def _wires_ok(ps: Sequence[Part], wires: Sequence[Wire] | None) -> bool:
    """`SecondStage.wiresOk`: wires are derived, each between templates the parts are of."""
    return wires is not None and all(any(p.template == w.template for p in ps) and any(p.template == w.writer for p in ps)
                                     for w in wires)


def _wire_of(own: Own, t: str, port: str) -> Wire | None:
    return next((w for w in own.wires or () if w.template == t and w.port == port), None)


def _writer_of(own: Own, w: Wire) -> Part | None:
    """`SecondStage.writerOf`: wire `w`'s writer's part."""
    return next((p for p in own.parts or () if p.template == w.writer), None)


def _held_port(own: Own, named: bool, t: str, p: str) -> str:
    """`SecondStage.heldPort`: the interior record's port that holds template `t`'s session port `p`: at a port a wire
    names, its writer's output port, the one tree the interior commits that value in; else its own."""
    w = _wire_of(own, t, p)
    return _port_of(named, t, p) if w is None else _port_of(named, w.writer, w.wport)


def _put_at(v: Any, j: Any, path: Sequence[str]) -> Any:
    """`SecondStage.putAt`: `j` with the field at key path `path` set to `v` where its parent is an object, each key
    before the last followed only to an object; anything else unchanged."""
    if not path or not isinstance(j, dict):
        return j
    k, rest = path[0], path[1:]
    if not rest:
        return {**j, k: v}
    x = j.get(k)
    return {**j, k: _put_at(v, x, rest)} if isinstance(x, dict) else j


def blank_run(h: Any) -> Any:
    """`SecondStage.blankRun`: header `h` with the run's fields (`RUN_FIELDS`) blanked, None."""
    for path in RUN_FIELDS:
        h = _put_at(None, h, path)
    return h


def _run_of(rec: Any) -> str | None:
    """`SecondStage.runOf`: the run the interior record names (``window.run``)."""
    run = _get(_get(rec, "window"), "run")
    return run if isinstance(run, str) else None


def set_name(run: str, k: int) -> str:
    """`SecondStage.setName`: the input set of the `k`-th part's session in run `run`, as the served path names it."""
    return f"served/{run}/m{k}"


def port_binding(rec: Any, port: str, schema: str, run: str) -> str | None:
    """`SecondStage.portBinding`: the binding, in run `run`, of record `rec`'s port `port` of row schema `schema`, under the
    window-domain rule (`window_binding`) over the record's program and partition and the leaves it gives the port; None
    when the record names no such port, or no program or partition."""
    program, partition, e = _get(rec, "program"), _get(rec, "partition"), _by_port(rec).get(port)
    leaves = _get(e, "leaves")
    if not (isinstance(program, str) and isinstance(partition, str) and _nat(leaves)):
        return None
    return window_binding(program, partition, port, schema, leaves, run).hex()


def at_run(own: Own, named: bool, k: int, t: str, r: Any) -> Any:
    """`SecondStage.atRun`: template `t`'s reference header `r` placed in the run, for the session of the `k`-th part:
    its ``set`` the run's input set (`set_name`) and its ``frame_v3.bindings`` the binding of each port of
    its ``frame_v3.schemas`` (`port_binding`, the interior record's port that holds it, `_held_port`: a wired port's its
    writer's), which the verifier computes. None when the interior record names no run, or a port the verifier can't
    bind."""
    run = _run_of(own.interior)
    if run is None:
        return None
    bindings = {}
    for p, s in _entries(_get(_get(r, "frame_v3"), "schemas")):
        b = port_binding(own.interior, _held_port(own, named, t, p), s, run) if isinstance(s, str) else None
        if b is None:
            return None
        bindings[p] = b
    return _put_at(bindings, _put_at(set_name(run, k), r, ("set",)), ("frame_v3", "bindings"))


def _header_ok(own: Own, named: bool, k: int, t: str, h: Any) -> bool:
    """`SecondStage.headerOk`: header `h` of a statement for the `k`-th part is template `t`'s reference header
    (`Own.refs`) placed in the run (`at_run`) but for the run's fields, the two the same with those blanked; False for a
    template the params give no reference, or one the verifier can't place in the run."""
    ref = next(((r,) for x, r in own.refs if x == t), None)
    placed = None if ref is None else at_run(own, named, k, t, ref[0])
    return placed is not None and _jeq(blank_run(h), blank_run(placed))


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
    link names, only the boundary port it names, each slot only its cell's replay block's rows; a port a wire names,
    only its writer's output port of the interior record, each slot only its writer's first instance's rows in its
    cell's replay block (the writer's slot ``p.span(u)·n_w`` of its ``Q·n_w``); any other, its own port of the interior
    record, each slot only its own rows, or a value either registration registers, never a boundary port. Each at its
    root, domain and leaves."""
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
    wire = _wire_of(own, p.template, port)
    if wire is not None:
        x, pw = _by_port(own.interior).get(v), _writer_of(own, wire)
        return (v == _port_of(named, wire.writer, wire.wport) and x is not None and pw is not None and _same_at(e, x)
                and _rows_ok(q, p, e, q * pw.n, lambda u: p.span(u) * pw.n))
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


def _wired_read(own: Own, t: str, reads: Sequence[tuple[str, Any]]) -> bool:
    """`SecondStage.wiredRead`: every port a wire names for template `t` is one of `reads`."""
    return all(w.template != t or any(port == w.port for port, _ in reads) for w in own.wires or ())


def _verified_ok(own: Own, s: Stmt) -> bool:
    """`SecondStage.verifiedOk`: an accepted verdict of record names statement `s`'s header, registered reads and
    sample."""
    c = _checked(s)
    return c is not None and any(v.header == c.header and _jeq(v.registered, c.registered) and _jeq(v.draw, c.draw)
                                 for v in own.verified)


def _tables_ok(own: Own, s: Stmt) -> bool:
    """`SecondStage.tablesOk`: every accepted verdict of record that names statement `s`'s header, registered reads and
    sample set its session up as one table."""
    c = _checked(s)
    return c is None or all(not (v.header == c.header and _jeq(v.registered, c.registered) and _jeq(v.draw, c.draw))
                            or _jeq(v.tables, ONE_TABLE) for v in own.verified)


def _refs_ok(s: Stmt) -> bool:
    """`SecondStage.refsOk`: statement `s` reads registered every port its header shares rows at (``shared_rows``, the
    tables whose rows its instances read by the refs in its public file's body)."""
    reads = _entries(_get(s.registered, "ports"))
    return all(any(name == port for name, _ in reads) for port, _ in _entries(_get(s.header, "shared_rows")))


#: `SecondStage.READ_RUN_FIELDS`: the run's fields of every port of a registered-reads file, which ``second-read`` holds
#: to the run (`_same_at`)
READ_RUN_FIELDS = ("root", "domain", "leaves")

#: `SecondStage.READ_ROW_FIELDS`: the rows a port of a registered-reads file reads, slot ``reads_from + k`` its row
#: ``reads[k]``, run fields only where the run places them (`_run_places_rows`)
READ_ROW_FIELDS = ("reads", "reads_from")


def _run_places_rows(own: Own, t: str, port: str, e: Any) -> bool:
    """`SecondStage.runPlacesRows`: template `t`'s session port `port` reads registered-reads entry `e` at rows the run
    places, which ``second-read`` holds to the run (`_rows_ok`): a port a link names, each slot at its cell's replay
    block's rows, or one reading a port of the interior record, its own port each slot at its own rows or, at a port a
    wire names, its writer's at its writer's rows (`_read_ok`'s first three cases). Any other port reads a value a
    registration registers, which ``second-read`` holds only to the value's root,
    domain and leaves (`_same_at`), so ``second-values`` holds its rows to the reference statement's, the same rows in
    every run. A workload whose rows there depend on the sampled block (a positional table) would need a rule of its
    own."""
    v = _get(e, "value")
    return _link_of(own, t, port) is not None or (isinstance(v, str) and v in _by_port(own.interior))


def blank_reads(own: Own, t: str, r: Any) -> Any:
    """`SecondStage.blankReads`: registered-reads file `r` of a session of template `t` with each port's run fields
    None: `READ_RUN_FIELDS`, and `READ_ROW_FIELDS` where the run places the port's rows (`_run_places_rows`); anything
    else unchanged."""
    ports = _get(r, "ports")
    if not isinstance(ports, dict):
        return r

    def blank(p: str, e: Any) -> Any:
        if not isinstance(e, dict):
            return e
        return {**e, **dict.fromkeys(READ_RUN_FIELDS + (READ_ROW_FIELDS if _run_places_rows(own, t, p, e) else ()))}
    return {**r, "ports": {p: blank(p, e) for p, e in ports.items()}}


def _values_ok(own: Own, s: Stmt) -> bool:
    """`SecondStage.valuesOk`: statement `s`'s registered reads are its template's reference statement's
    (`Own.ref_reads`) but for the run's fields, the two the same with those blanked (`blank_reads`), so None only where
    the reference's is, and each value a registration registers read at the reference's rows; False for a template the
    params give no reference reads."""
    ref = next(((r,) for t, r in own.ref_reads if t == s.template), None)
    return ref is not None and _jeq(blank_reads(own, s.template, s.registered), blank_reads(own, s.template, ref[0]))


def _stmt_codes(own: Own, named: bool, k: int, p: Part, s: Stmt) -> list[str]:
    q, reads = len(own.sample), _entries(_get(s.registered, "ports"))
    checks = (("second-verified", _verified_ok(own, s)),
              ("second-tables", _tables_ok(own, s)),
              ("second-header", _header_ok(own, named, k, s.template, s.header)),
              ("second-population", _population_ok(q, p, s.header)),
              ("second-roots", _roots_ok(named, own.interior, s.template, s.header, [name for name, _ in reads])),
              ("second-sample", _jeq(s.draw, own_draw(q, p, own.drawn))),
              ("second-read", all(_read_ok(own, named, p, port, e) for port, e in reads)
               and _linked_read(own, p.template, reads) and _wired_read(own, p.template, reads)),
              ("second-refs", _refs_ok(s)),
              ("second-values", _values_ok(own, s)))
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


def _order_failures(ps: Sequence[Part], ss: Sequence[Any]) -> list[tuple[str, Any]]:
    """`SecondStage.orderFailures`: ``second-order`` for each statement for a part, no earlier statement's, other than
    the part at its position, round 2 reading the statements by position, the i-th the i-th part's (the parts' order,
    the interior program's strata's)."""
    found = []
    for i, s in enumerate(map(_stmt, ss)):
        if (s is not None and any(p.template == s.template for p in ps) and s.template not in _templates(ss[:i])
                and (i >= len(ps) or ps[i].template != s.template)):
            found.append(("second-order", {"statement": i, "template": s.template}))
    return found


def _same_dict(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return len(a) == len(b) and all(k in b and _jeq(v, b[k]) for k, v in a.items())


def interior_order_ok(own: Own) -> bool:
    """`SecondStage.interiorOrderOk`: the replay sample the interior record names (``window.replay_sample``, which the
    proof server writes) is the verifier's own, ``own.sample``, in its order, each entry a natural number, never a
    boolean."""
    bs = _get(_get(own.interior, "window"), "replay_sample")
    return (isinstance(bs, list) and len(bs) == len(own.sample)
            and all(_nat(b) and b == u for b, u in zip(bs, own.sample)))


def _interior_failures(own: Own) -> list[tuple[str, Any]]:
    """`SecondStage.interiorFailures`: ``second-interior-order`` unless `interior_order_ok`."""
    return [] if interior_order_ok(own) else [("second-interior-order", INTERIOR_ORDER_WHY)]


def _pos_of(ps: Sequence[Part], t: str) -> int:
    """`SecondStage.posOf`: template `t`'s position among the parts `ps`, its first part's (its member's index in the
    interior window); ``len(ps)``, no member's, for none."""
    return next((k for k, p in enumerate(ps) if p.template == t), len(ps))


def _failures(allow: bool, own: Own, ps: Sequence[Part], ss: Sequence[Any]) -> list[tuple[str, Any]]:
    named = _by_template(own.interior)
    found = _interior_failures(own)
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
            found += [(c, detail) for c in _stmt_codes(own, named, _pos_of(ps, s.template), p, s)]
    found += _order_failures(ps, ss)
    covered = _templates(ss)
    found += [("second-cover", {"template": p.template}) for p in ps if p.template not in covered]
    if not _same_dict(own.reported or {}, statement_reads(ss)):
        found.append(("second-reported", REPORTED_WHY))
    return found if found or allow else [("two-stage-form", FORM_WHY)]


def check(allow: bool, own: Own, statements: Sequence[Any] | None) -> Verdict:
    """Round 2's statement check of `statements`, the two-stage form allowed when `allow` (`FORM` in a run): each code
    once, with the detail it first failed with (`SecondStage.check`), the interior record's order first."""
    if own.parts is None:
        found = [*_interior_failures(own), ("second-cells", CELLS_WHY)]
    elif not (_links_ok(own.parts, own.links) and _wires_ok(own.parts, own.wires)):
        found = [*_interior_failures(own), *([] if _links_ok(own.parts, own.links) else [("second-links", LINKS_WHY)]),
                 *([] if _wires_ok(own.parts, own.wires) else [("second-wires", WIRES_WHY)])]
    elif statements is None:
        found = [*_interior_failures(own), ("second-statements", STATEMENTS_WHY)]
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


__all__ = ["FORM", "READ_ROW_FIELDS", "READ_RUN_FIELDS", "Checked", "Link", "Own", "Part", "Wire", "at_run", "attach",
           "blank_reads", "blank_run", "check", "interior_order_ok", "links_of", "own_draw", "parts", "port_binding",
           "replay_links", "replay_wires", "set_name", "statement_reads", "verified_of", "wires_of"]
