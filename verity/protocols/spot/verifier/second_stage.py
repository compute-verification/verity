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
`SecondStage.replayWires`): a template's parameter that is, at each of its callers, one field of one instance of
another template reads that writer's output port for the field, one value the interior commits once, so
``second-read`` holds a wired port to its writer's port at its writer unit's row and `at_run` binds it by that port
(`_held_port`). No wire reads an output the replay block returns, a writer's port a link names: the boundary commits
that output, and the interior's port of it is another tree, which nothing holds to the first. And each wire's writer's
part is before its reader's (``second-wires``, `_wires_ok`).
A template may have several callers in the body, a batch's members among them, its units theirs in
body order; each port a link or a wire names reads, unit by unit, the row its caller names (`replay_rows`,
`SecondStage.replayRows`), summarized once per Definition as runs (`Run`) and never unrolled. A port a link names reads
the boundary port the link names or, where the interior registration registers a value of the link's name (a value its
verifier supplies, say), that value, at its unit's row of its cell's sampled block (`_read_ok`). Each name is one
registration's: an interior registration that registers a value of a name the boundary registration's record has, a
port or a registered value, is refused (``second-interior-names``, `names_ok`), so the interior's value of a name never
stands in for one round 1 committed. What a replay block returns is committed before the first stage's sample: an
interior registration that registers a value of a name the block's return has in the boundary registration's record,
one of its output ports ``o<j>`` (`replay_returns`), is refused (``second-return``, `returns_ok`), so a port a link
names for the return is read only at the port round 1's record has for it.

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
statement is read. With it, ``second-interior-names`` fails where the interior record registers a value of a name the
boundary registration's record has (`names_ok`), ``second-return`` where it registers one of a name the replay
block's return has there (`returns_ok`), and ``second-schedule`` where the boundary registration's record
commits a schedule (``window.schedule``) that is no schedule of its replay blocks (`schedule.failure`); a record that
commits none passes.

Every input of a replay block that the prover gives is a port of round 1's record (a link), another template's output
in the block (a wire), a value the verifier holds without the prover, or the opened row of a read against a root round
1 committed: the interior registration, made after the first stage's sample, registers nothing else
(``second-inputs``, `inputs_failure`). A value it registers passes where the verifier pinned it, holds it to an earlier
record, or fills it from its own bytes (`Own.fixed`, `Own.supplied`). Any other is the prover's, and passes only as a
read's row, salt or path, where the verifier fills the read's root and domain with the root and the domain id round 1's
record registers for one table (`_one_table`, by their bytes: `Own.supplied` holds what the verifier computed for a
table in place of its name, as the interior registration's check read it), the read's key is a port of round 1's
record or a value the verifier holds, and the read's verdict port reads a value the verifier fills with 1
(`_opened_row`): the row is then the table's at that key, short of a SHA-512
collision. A key the block computes, a wire, is refused, with a reason of its own (`KEY_WHY`): no statement covers it.
A read is a template the auditor's params record as one (`Own.reads`, `service.template_reads`): the template
check (``flock-read-tmpl check V H``, `Flock.ReadNet.check`) accepted the circuit file they name for it as the read of
a table of V rows, where the params were made; round 2 runs no template check. Which table a read opens is the one
whose root and domain id the verifier fills, a table of V leaves in round 1's record, and which row its key. A design
whose key is the prover's after the sample, whose root and domain are not one table's of round 1's record, or whose
table has another leaf count than the read's rows, is refused here. The code is reported beside the others, in every
branch of `check`, so a run that has it still reaches its other stops.

`FORM` is `SecondStage.FORM`, which the run of record reads: off, every two-stage run is refused last
(``two-stage-form``), and the Python flipped alone disagrees with the run of record."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.codec import canonical_json, definitions, descriptor_id
from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.refs import Affine, Concat, Explicit
from verity.circuits.types import Array, RecordType, Tuple
from verity.commitments.window import window_binding

from .one_stage.verdict import Verdict, _Collect
from .schedule import failure as schedule_failure

FORM = False
FORM_WHY = ("refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the "
            "soundness proof does not cover")
CELLS_WHY = "the interior program's strata are not one range each, of a positive multiple of the replay sample's size"
STATEMENTS_WHY = ("a two-stage outcome names its second-stage sessions' statements: each one's template, header, draw and "
                  "registered reads")
REPORTED_WHY = ("the registered reads the verifiers of record report are not the statements' registered reads, value by "
                "value")
LINKS_WHY = ("the replay block's Definition reads a parameter, or returns, other than through templates of the "
             "interior's parts, each caller reading one aligned row of one parameter alone, or returned whole within "
             "one field, alike at every caller, or a link's rows don't place its part's units")
WIRES_WHY = ("the replay block's Definition reads a node other than as a template's parameter that is one field of one "
             "instance of another template, alike at every caller, or a wire names a template of no part, or its rows "
             "don't place its part's units or its writer's, or it reads an output the replay block returns, or its "
             "writer's part is not before its reader's")
INTERIOR_ORDER_WHY = "the interior record's replay sample (window.replay_sample) is not the verifier's own, in its order"
NAMES_WHY = ("the interior registration registers a value of a name the boundary registration's record has, a port or a "
             "registered value")
RETURN_WHY = ("the interior registration registers a value of a name the replay block's return has in the boundary "
              "registration's record: what a replay block returns is read only at a port that record commits")
INPUTS_WHY = ("the interior registration registers a value that is neither one the verifier holds without the prover (one "
              "it pinned, one it holds to an earlier record, or one it fills from its own bytes) nor the opened row of a "
              "read the auditor's params record, whose root and domain it fills from one table of the read's rows that "
              "round 1's record registers, whose key is a port of round 1's record or a value it holds, and whose "
              "verdict it holds to 1")
KEY_WHY = ("the interior registration registers the opened row of a read at a key the block computes (the read's key is "
           "wired to another template's output): no statement covers it")
#: A read's ports (`gates.embedding.EmbeddingRead`, `Flock.ReadNet.readTmpl`): its key, the table's root and domain
#: id, the opened row with its salt and its path, and its verdict (`SecondStage.OPENING_KEY` and the names after it).
OPENING_KEY, OPENING_TABLE, OPENING_ROW, OPENING_VERDICT = "tok", ("root", "domain"), ("row", "salt", "path"), "o1"
OPENING_ROOT, OPENING_DOMAIN = OPENING_TABLE
#: The sources of a value the verifier fills from what it holds before the first stage's sample: its own bytes, among
#: them the root and the domain id it computed for a table, which `Own.supplied` holds in place of the table's name
#: (`history.resolve`; `SecondStage.OWN_SOURCES`). A ``port`` or a ``value`` source reads a root of the interior record,
#: the prover's after the sample, and a ``position`` or a ``window`` source the interior record's population or window.
#: A ``history`` source left unresolved is none of the verifier's bytes.
OWN_SOURCES = ("bytes",)
#: The row of a value that holds a read's verdict to 1: the verifier's bytes of the number 1
#: (`SecondStage.VERDICT_ONE`).
VERDICT_ONE = (("bytes", "00000001"),)
STMT_KEYS = ["draw", "header", "registered", "template"]
#: The most leaves a tree a session reads may have: its positions are 64-bit (`SecondStage.MAX_LEAVES`).
MAX_LEAVES = 2 ** 64
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

    def unit(self, u: int) -> int:
        """`Part.unit`: slot `u`'s unit in its cell, its place among the `n` slots there."""
        return (u - self.lo) % self.n


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
class Run:
    """`SecondStage.Run`: a run of units' rows, `count` units from unit `unit`, the first reading row `row`, each next
    `step` rows on."""

    unit: int
    count: int
    row: int
    step: int


@dataclass(frozen=True)
class Rows:
    """`SecondStage.Rows`: the rows template `template`'s units read at its session port `port`, one a link or a wire
    names: of the value's `per` rows a replay block, the boundary port's (a link) or its writer's units' (a wire), unit
    `j` reads the one its `runs` give it (`row_at`)."""

    template: str
    port: str
    per: int
    runs: tuple[Run, ...]

    def to_json(self) -> dict[str, Any]:
        return {"per": self.per, "port": self.port, "runs": [[r.unit, r.count, r.row, r.step] for r in self.runs],
                "template": self.template}


def rows_of(j: Any) -> tuple[Rows, ...] | None:
    """`SecondStage.rowsOf?`: rows from their JSON, a list of objects of exactly ``template``, ``port``, ``per`` and
    ``runs``, two strings, a natural number and a list of ``[unit, count, row, step]``, each a natural number; None for
    anything else."""
    if not isinstance(j, list):
        return None
    out = []
    for r in j:
        if not (isinstance(r, dict) and sorted(r) == ["per", "port", "runs", "template"]):
            return None
        t, port, per, runs = r["template"], r["port"], r["per"], r["runs"]
        if not (isinstance(t, str) and isinstance(port, str) and _nat(per) and isinstance(runs, list)
                and all(isinstance(x, list) and len(x) == 4 and all(map(_nat, x)) for x in runs)):
            return None
        out.append(Rows(t, port, per, tuple(Run(*x) for x in runs)))
    return tuple(out)


@dataclass(frozen=True)
class Own:
    """What the verifier holds of the run: the replay sample, the parts (`parts`), the interior and boundary
    registrations' records, the second draw's units, the design's links (`replay_links`, None when the replay block's
    Definition makes none), the ``{value: root}`` the verifiers of record report the sessions read
    (`audit.session_registered`), what they name of each session they accepted (`verified_of`), the reference header
    of each template's statement, ``(template, header)``, and the registered reads it reads,
    ``(template, registered)``, None for none (`ref_reads`), as the auditor's params give them
    (`service.template_circuits`), the design's wires (`replay_wires`, None when the Definition makes none), the
    rows of the ports they and the links name (`replay_rows`), the names the replay block's return has in the boundary
    registration's record (`replay_returns`), and, of the values the interior registration registers, the verifier's
    own copies of those it holds without the prover: each one it fills itself, with its rows of sources
    (`supplied_of`), and the names of those it pinned or holds to an earlier record (``fixed``); and the templates the
    auditor's params record as reads, each with its table's rows, ``(template, V)`` (`service.template_reads`)."""

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
    rows: tuple[Rows, ...] = ()
    returns: tuple[str, ...] = ()
    supplied: tuple[tuple[str, tuple[tuple[tuple[str, str], ...], ...]], ...] = ()
    fixed: tuple[str, ...] = ()
    reads: tuple[tuple[str, int], ...] = ()


def supplied_of(supplied: Any) -> tuple[tuple[str, tuple[tuple[tuple[str, str], ...], ...]], ...]:
    """`SecondStage.suppliedOf`: the verifier's supplied values ``{value: [[source, ...], ...]}``, a row a list of
    sources, each ``{kind: name}`` of two strings, as ``(value, rows)`` each, a row its ``(kind, name)`` sources; None is
    none. ValueError on any other form."""
    if supplied is None:
        return ()
    if not isinstance(supplied, Mapping):
        raise ValueError(f"supplied values are an object of rows, not {supplied!r}")
    out = []
    for v, rows in supplied.items():
        if not (isinstance(v, str) and isinstance(rows, (list, tuple))
                and all(isinstance(r, (list, tuple)) and all(isinstance(x, Mapping) and len(x) == 1 and all(
                    isinstance(k, str) and isinstance(n, str) for k, n in x.items()) for x in r) for r in rows)):
            raise ValueError(f"supplied value {v!r}: its rows are not lists of sources, each a kind naming a string")
        out.append((v, tuple(tuple(next(iter(x.items())) for x in r) for r in rows)))
    return tuple(out)


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
    """`SecondStage.callCounts`: how many nodes of the bodies of `defs` call each definition, by descriptor id."""
    out: dict[str, int] = {}
    for f in defs.values():
        for node in () if isinstance(f, PrimitiveDefinition) else f.body.nodes:
            out[descriptor_id(node.fn)] = out.get(descriptor_id(node.fn), 0) + 1
    return out


def _members(node: Any) -> int:
    """`Node.members`: a batch's members or a scan's iterations, else one."""
    return node.n if node.form in ("batch", "scan") else 1


def _axis_dims(t: Any, ax: int) -> tuple[int, int] | None:
    """`Ty.axisDims`: the length of array axis `ax` of `t` and the leaves of its elements; None for none."""
    while isinstance(t, Array):
        if ax == 0:
            return t.n, t.elem.leaves
        t, ax = t.elem, ax - 1
    return None


def _dims(node: Any, a: int) -> tuple[int, int] | None:
    """`Arg.dims`: for a batch's argument `a` mapped on an axis, or a scan's sequence, its length and its elements'
    leaves; None for any other argument."""
    t = node.args[a].type
    if node.form == "batch":
        ax = node.axes[a] if a < len(node.axes) else None
        return None if ax is None else _axis_dims(t, ax)
    if node.form == "scan" and 1 <= a <= node.n_xs and isinstance(t, Array):
        return t.n, t.elem.leaves
    return None


@dataclass(frozen=True)
class _Callers:
    """`SecondStage.Callers`: a replay block's body's templates (each node's, None for a node no template's), each
    node's first unit among its definition's, and each definition's units in the body, by descriptor id."""

    template: tuple[Any, ...]
    first: tuple[int, ...]
    units: Mapping[str, int]


def _callers_of(defs: Mapping[str, Any], fn: Any) -> _Callers:
    """`SecondStage.callersOf`: replay block `fn`'s body's templates and their units, once over `defs` and its body:
    each node's template, the composite definition it calls or batches when every node of `defs` that calls it is one
    of `fn`'s body's; each node's first unit among its definition's in a replay block, its callers before it in body
    order holding the units before it, a batch its members' (`_members`), never walked; and each definition's units."""
    every, here = _callers(defs), _callers({"": fn})
    units: dict[str, int] = {}
    template, first = [], []
    for node in fn.body.nodes:
        d = descriptor_id(node.fn)
        first.append(units.get(d, 0))
        units[d] = units.get(d, 0) + _members(node)
        c = node.fn
        template.append(c if node.form in ("call", "batch") and not isinstance(c, PrimitiveDefinition)
                        and every.get(d) == here.get(d) else None)
    return _Callers(tuple(template), tuple(first), units)


def _push_run(rs: list[Run], r: Run) -> None:
    """`SecondStage.pushRun`: run `r` after runs `rs`, merged into the last when `r` is one unit right after it
    continuing its rows (the last one unit, at a row no later than `r`'s, or `r`'s row the last's next), else
    appended."""
    last = rs[-1] if rs else None
    if last is not None and r.count == 1 and last.unit + last.count == r.unit:
        if last.count == 1 and last.row <= r.row:
            rs[-1] = Run(last.unit, 2, last.row, r.row - last.row)
            return
        if last.count > 1 and r.row == last.row + last.count * last.step:
            rs[-1] = Run(last.unit, last.count + 1, last.row, last.step)
            return
    rs.append(r)


def row_at(rs: Sequence[Run], j: int) -> int | None:
    """`SecondStage.rowAt`: unit `j`'s row under runs `rs`, in order of their first units: the run holding it, found by
    bisection over their first units, gives ``row + (j - unit)·step``; None for a unit no run holds."""
    lo, hi = 0, len(rs)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rs[mid].unit <= j:
            lo = mid
        else:
            hi = mid
    if not rs:
        return None
    r = rs[lo]
    return r.row + (j - r.unit) * r.step if r.unit <= j < r.unit + r.count else None


def _fits(rs: Sequence[Run], n: int) -> bool:
    """`SecondStage.fits`: runs `rs` hold units 0 to ``n - 1``, each once, in order: the first from unit 0, each next
    from the last's end, none empty, the last ending at `n`."""
    e = 0
    for r in rs:
        if r.unit != e or r.count <= 0:
            return False
        e += r.count
    return e == n


@dataclass(frozen=True)
class _Share:
    """`SecondStage.Share`: one caller's share of a link or a wire `edge`: its template's descriptor id `fn`, the
    value's rows a replay block `per`, and its units' rows `run`."""

    edge: Any
    fn: str
    per: int
    run: Run


def _gather(port: Callable[[Any], tuple[str, str]], same: Callable[[Any, Any], bool], units: Mapping[str, int],
            xs: Sequence[_Share]) -> list[tuple[Any, int, tuple[Run, ...]]] | None:
    """`SecondStage.gather`: shares `xs` gathered one per template port (`port`), in order of first appearance, each
    port's runs in order (`_push_run`), with its value's rows a block; None when two of a port name different values
    (`same`) or rows a block, or a port's runs don't hold each of its template's `units` once, in order (`_fits`)."""
    g: list[tuple[_Share, list[Run]]] = []
    for x in xs:
        j = next((i for i, (y, _rs) in enumerate(g) if port(y.edge) == port(x.edge)), None)
        if j is None:
            g.append((x, [x.run]))
            continue
        y, rs = g[j]
        if not (same(y.edge, x.edge) and y.per == x.per):
            return None
        _push_run(rs, x.run)
    if not all(_fits(rs, units.get(y.fn, 0)) for y, rs in g):
        return None
    return [(y.edge, y.per, tuple(rs)) for y, rs in g]


def _slice(node: Any, a: int, w: int) -> tuple[int, int] | None:
    """`SecondStage.slice`: the row of its collection's rows node `node`'s argument `a` reads, `w` leaves a member, and
    each next member's step, when the argument is one aligned slice of `w` leaves a member of one collection, its one
    affine part: the same slice for every member (a call's argument, or a batch's shared one, step 0), or a member's
    from the first, outer axis mapped (step 1). None for any other argument."""
    rs = _parts(node.args[a].refs)
    r = rs[0] if len(rs) == 1 else None
    if w <= 0 or not isinstance(r, Affine) or r.base % w:
        return None
    dims = _dims(node, a)
    if dims is None:
        return (r.base // w, 0) if r.count == w else None
    return (r.base // w, 1) if node.form == "batch" and dims[1] == w and r.count == _members(node) * w else None


def _node_links(cs: _Callers, fn: Any, k: int, node: Any) -> list[_Share] | None:
    """`SecondStage.nodeLinks`: the links node `k` of replay block `fn`'s body makes, each its share: for each argument
    that reads one of `fn`'s parameters, its template's parameter there reads that one, the parameter's rows a block
    those of the template parameter's width, its units the rows its slice is (`_slice`). None when the argument reads
    anything else too, or anything but such a slice, or `node` is no template's."""
    params, out = fn.signature()[0], []
    for a, arg in enumerate(node.args):
        src = _sources(arg.refs)
        first = next((x for x in src if x[0]), None)
        if first is None:
            continue
        c = cs.template[k]
        ports = [] if c is None else c.signature()[0]
        if not all(x == first for x in src) or c is None or a >= len(ports) or first[1] >= len(params):
            return None
        (port, ty), (name, pty) = ports[a], params[first[1]]
        at = _slice(node, a, ty.leaves)
        if at is None or pty.leaves % ty.leaves:
            return None
        out.append(_Share(Link(descriptor_id(c), port, name), descriptor_id(c), pty.leaves // ty.leaves,
                          Run(cs.first[k], _members(node), *at)))
    return out


def _field_at(t: Any, off: int, c: int) -> int | None:
    """`SecondStage.fieldAt`: the field of a value of type `t` from leaf `off`, of `c` leaves: its index among
    `_fields`; None for none."""
    s = 0
    for j, f in enumerate(_fields(t)):
        if s == off and f.leaves == c:
            return j
        s += f.leaves
    return None


def _field_of(t: Any, s: int, c: int) -> tuple[int, int] | None:
    """`SecondStage.fieldOf`: the field of a value of type `t` holding its leaves ``[s, s + c)``: its index and first
    leaf; None for none."""
    o = 0
    for j, f in enumerate(_fields(t)):
        if o <= s and s + c <= o + f.leaves:
            return j, o
        o += f.leaves
    return None


def _return_links(cs: _Callers, fn: Any) -> list[_Share] | None:
    """`SecondStage.returnLinks`: the links replay block `fn`'s return makes, each its share: each part of its return,
    which holds all of its leaves, an output of node `k`, a template's: one whole field `f` of its output where `k` is
    one instance (`_field_at`), the field's leaves its width, or else `k`'s whole output, the template's of one output
    (`f` = 0), its leaves its width. Within one field `j` of the return at a multiple of that width from the field's
    first, it makes the template's ``o<f>`` `fn`'s ``o<j>``, the field's rows a block those of that width, and `k`'s
    units the field's rows from there on, one each. None when a part is anything else."""
    nodes, rty, parts = fn.body.nodes, fn.signature()[1], _parts(fn.body.ret)
    if sum(r.count for r in parts) != rty.leaves:
        return None
    out, s = [], 0
    for r in parts:
        start, s = s, s + r.count
        if not (isinstance(r, Affine) and r.space == "n" and r.idx < len(nodes)):
            return None
        x, t = nodes[r.idx], cs.template[r.idx]
        if t is None:
            return None
        tty = t.signature()[1]
        if _members(x) == 1:
            f, w = _field_at(tty, r.base, r.count) if x.out.leaves == tty.leaves else None, r.count
        else:
            f, w = 0 if r.base == 0 and _outputs(tty) == 1 and r.count == x.out.leaves else None, tty.leaves
        if f is None or w <= 0 or r.count != _members(x) * w:
            return None
        at = _field_of(rty, start, r.count)
        if at is None:
            return None
        j, off = at
        g = _fields(rty)[j]
        if (start - off) % w or g.leaves % w:
            return None
        out.append(_Share(Link(descriptor_id(t), f"o{f}", f"o{j}"), descriptor_id(t), g.leaves // w,
                          Run(cs.first[r.idx], _members(x), (start - off) // w, 0 if _members(x) == 1 else 1)))
    return out


def link_runs(defs: Mapping[str, Any], key: str) -> list[tuple[Link, int, tuple[Run, ...]]] | None:
    """`SecondStage.linkRuns`: the design's links with their rows a block and runs, from replay block ``defs[key]``'s
    registered Definition, its parameters and its body: those its nodes make (`_node_links`) and its return
    (`_return_links`), one per template port (`_gather`), each value its own port's name. None when its body reads a
    parameter, or returns, any other way, or a template's callers read a linked port unlike each other."""
    fn = defs.get(key)
    if fn is None or isinstance(fn, PrimitiveDefinition):
        return None
    cs, shares = _callers_of(defs, fn), []
    for k, node in enumerate(fn.body.nodes):
        got = _node_links(cs, fn, k, node)
        if got is None:
            return None
        shares += got
    ret = _return_links(cs, fn)
    if ret is None:
        return None
    return _gather(lambda x: (x.template, x.port), lambda a, b: a.value == b.value, cs.units, shares + ret)


def links_of(defs: Mapping[str, Any], key: str) -> tuple[Link, ...] | None:
    """`SecondStage.linksOf`: the design's links (`link_runs`)."""
    runs = link_runs(defs, key)
    return None if runs is None else tuple(x for x, _per, _rs in runs)


def replay_links(program: Any, key: str, rec: Any) -> tuple[Link, ...] | None:
    """`SecondStage.replayLinks`: the links of the replay block `program` defines under descriptor id `key` (`links_of`
    over the definitions its descriptor records, `codec.definitions`), each value the boundary registration's port for
    it, as record `rec` names its ports."""
    links, named = links_of(definitions(program), key), _by_template(rec)
    return None if links is None else tuple(Link(x.template, x.port, _port_of(named, key, x.value)) for x in links)


def replay_returns(program: Any, key: str, rec: Any) -> tuple[str, ...]:
    """`SecondStage.replayReturns`: the names the return of the replay block `program` defines under descriptor id `key`
    has in the boundary registration, as record `rec` names its ports: its output ports ``o<j>``, one per field of what
    it returns (`_outputs`), among them the value of each link its return makes (`_return_links`). Empty for a `key`
    it doesn't define."""
    fn, named = definitions(program).get(key), _by_template(rec)
    return () if fn is None else tuple(_port_of(named, key, f"o{j}") for j in range(_outputs(fn.signature()[1])))


def _words(t: Any) -> int:
    """`Spot.Bits.wordsOf`: the 16-bit words of a value of type `t`, its bits packed sixteen to a word."""
    return (sum(t.leaf_widths()) + 15) // 16


def returned_bits(defs: Mapping[str, Any], key: str) -> int | None:
    """`SecondStage.returnedBits`: the bits one replay block of ``defs[key]`` returns to the boundary: each part of its
    return (`_return_links`) is one output field of a template's units, and gives its units times sixteen bits for each
    word of the field (`_words`). The second stage's files lay a returned field on words of its own, every bit of them
    a computed gate (`Spot.Bits.retWords`, `Spot.Bits.check`), so these are the gates a replay block's request is
    served; a served cap below them charges it fewer bits than it returns (`plancheck.two`'s `returned`). None when
    the design's links can't be derived (`link_runs`)."""
    if link_runs(defs, key) is None:
        return None
    fn = defs[key]
    return sum(s.run.count * 16 * _words(_fields(defs[s.fn].signature()[1])[int(s.edge.port[1:])])
               for s in _return_links(_callers_of(defs, fn), fn) or ())


def replay_returned(program: Any, key: str) -> int | None:
    """`SecondStage.replayReturned`: the bits one replay block returns (`returned_bits`), the replay block the one
    `program` defines under descriptor id `key`."""
    return returned_bits(definitions(program), key)


def _link_of(own: Own, t: str, port: str) -> Link | None:
    return next((x for x in own.links or () if x.template == t and x.port == port), None)


def _node_wires(cs: _Callers, fn: Any, k: int, node: Any) -> list[_Share] | None:
    """`SecondStage.nodeWires`: the wires node `k` of replay block `fn`'s body makes, each its share: for each argument
    that reads a node, its template's parameter there reads that node's output field `f`, one of its template's
    (`_field_at`), the node one instance and the argument the field whole, the same for each of `node`'s members: the
    writer's units a block its rows, `node`'s units reading the node's. None when such an argument reads anything else
    too, or part of a field, several nodes, a node no template's, or `node` is no template's. An argument that reads
    only `fn`'s parameters is a link's (`_node_links`)."""
    nodes, out = fn.body.nodes, []
    for a, arg in enumerate(node.args):
        if all(p for p, _i in _sources(arg.refs)):
            continue
        rs = _parts(arg.refs)
        r = rs[0] if len(rs) == 1 else None
        if not (isinstance(r, Affine) and r.space == "n" and r.idx < len(nodes)):
            return None
        w, tw, t = nodes[r.idx], cs.template[r.idx], cs.template[k]
        params = [] if t is None else t.signature()[0]
        if tw is None or a >= len(params):
            return None
        port, ty = params[a]
        if not (_dims(node, a) is None and _members(w) == 1 and 0 < r.count == ty.leaves
                and w.out.leaves == tw.signature()[1].leaves):
            return None
        f = _field_at(tw.signature()[1], r.base, r.count)
        if f is None:
            return None
        out.append(_Share(Wire(descriptor_id(t), port, descriptor_id(tw), f"o{f}"), descriptor_id(t),
                          cs.units.get(descriptor_id(tw), 0), Run(cs.first[k], _members(node), cs.first[r.idx], 0)))
    return out


def wire_runs(defs: Mapping[str, Any], key: str) -> list[tuple[Wire, int, tuple[Run, ...]]] | None:
    """`SecondStage.wireRuns`: the design's wires with their rows a block and runs, from replay block ``defs[key]``'s
    registered Definition: those its nodes make (`_node_wires`), one per template port (`_gather`). None when its body
    reads a node any other way, or a template's callers read a wired port unlike each other."""
    fn = defs.get(key)
    if fn is None or isinstance(fn, PrimitiveDefinition):
        return None
    cs, shares = _callers_of(defs, fn), []
    for k, node in enumerate(fn.body.nodes):
        got = _node_wires(cs, fn, k, node)
        if got is None:
            return None
        shares += got
    return _gather(lambda w: (w.template, w.port), lambda a, b: a.writer == b.writer and a.wport == b.wport, cs.units,
                   shares)


def wires_of(defs: Mapping[str, Any], key: str) -> tuple[Wire, ...] | None:
    """`SecondStage.wiresOf`: the design's wires (`wire_runs`)."""
    runs = wire_runs(defs, key)
    return None if runs is None else tuple(x for x, _per, _rs in runs)


def replay_wires(program: Any, key: str) -> tuple[Wire, ...] | None:
    """`SecondStage.replayWires`: the wires of the replay block `program` defines under descriptor id `key` (`wires_of`
    over the definitions its descriptor records, `codec.definitions`)."""
    return wires_of(definitions(program), key)


def replay_rows(program: Any, key: str) -> tuple[Rows, ...]:
    """`SecondStage.replayRows`: the design's rows, those of the ports the links and the wires of the replay block
    `program` defines under descriptor id `key` name (`link_runs`, `wire_runs`), none for either it makes none of."""
    defs = definitions(program)
    if key not in defs:
        return ()
    runs = [*(link_runs(defs, key) or ()), *(wire_runs(defs, key) or ())]
    return tuple(Rows(x.template, x.port, per, rs) for x, per, rs in runs)


def _rows_of(own: Own, t: str, port: str) -> Rows | None:
    """`SecondStage.rowsOf`: the rows of template `t`'s session port `port` (`Own.rows`)."""
    return next((r for r in own.rows if r.template == t and r.port == port), None)


def _placed(own: Own, p: Part, port: str) -> bool:
    """`SecondStage.placed`: part `p`'s units are placed at its template's session port `port`: its rows' runs hold the
    `n` units a block of `p` (`_fits`)."""
    r = _rows_of(own, p.template, port)
    return r is not None and _fits(r.runs, p.n)


def _links_ok(ps: Sequence[Part], own: Own) -> bool:
    """`SecondStage.linksOk`: the links are derived, each for a template one of the parts is of, whose units its rows
    place (`_placed`)."""
    return own.links is not None and all(any(p.template == x.template and _placed(own, p, x.port) for p in ps)
                                         for x in own.links)


def _wires_ok(ps: Sequence[Part], own: Own) -> bool:
    """`SecondStage.wiresOk`: the wires are derived, each between templates the parts are of, its reader's units placed
    by its rows (`_placed`), whose rows a block are its writer's units a block, whose writer's port no link names, and
    whose writer's part is before its reader's (`_pos_of`)."""
    def writer_ok(w: Wire) -> bool:
        r = _rows_of(own, w.template, w.port)
        return (r is not None and any(p.template == w.writer and r.per == p.n for p in ps)
                and _link_of(own, w.writer, w.wport) is None and _pos_of(ps, w.writer) < _pos_of(ps, w.template))
    return own.wires is not None and all(any(p.template == w.template and _placed(own, p, w.port) for p in ps)
                                         and writer_ok(w) for w in own.wires)


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


INTERIOR_ROOT = "verity-vllm/interior-run/v1"


def interior_root(boundary: str, sample: Sequence[int]) -> str:
    """`SecondStage.interiorRoot`: the interior window's open root for the boundary registration of digest `boundary`
    and the replay sample `sample`, in order. The verifier computes it, so a second stage sampled on a session of its
    own is sampled on the one window its boundary registration and replay sample name."""
    message = canonical_json({"boundary": boundary, "replay_sample": [int(u) for u in sample]})
    return hashlib.sha512(INTERIOR_ROOT.encode() + b"\x00" + message).hexdigest()


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
    return (isinstance(indices, list) and _nat(m) and m == q * p.n and m <= MAX_LEAVES and len(indices) == q * p.n
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
    """`SecondStage.sameAt`: entries `e` and `x` name one root, domain and leaf count, at most `MAX_LEAVES`."""
    leaves = _get(e, "leaves")
    return (all(_jeq(_get(e, k), _get(x, k)) for k in ("root", "domain", "leaves"))
            and _nat(leaves) and leaves <= MAX_LEAVES)


def _rows_ok(q: int, p: Part, e: Any, units: int, unit: Callable[[int], int | None]) -> bool:
    """`SecondStage.rowsOk`: every read of registered-reads entry `e` (slot ``reads_from + k`` reads row ``reads[k]``,
    None for none) is by one of part `p`'s slots `u`, of the first row of the unit ``unit(u)`` names, in a tree of
    `units` units' rows, ``leaves // units`` each."""
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
        return _nat(x) and b is not None and p.holds(q, u) and x == b * r
    return all(read_ok(first + k, x) for k, x in enumerate(rs))


def _placed_ok(q: int, p: Part, e: Any, leaves: int, row: Callable[[int], int | None]) -> bool:
    """`SecondStage.placedOk`: registered-reads entry `e` is of `leaves` leaves, and its every read (slot
    ``reads_from + k`` reads row ``reads[k]``, None for none) is by one of part `p`'s slots `u`, of row ``row(u)``."""
    first = _get(e, "reads_from")
    first = 0 if first is None else first
    n, rs = _get(e, "leaves"), _get(e, "reads")
    if not (_nat(n) and isinstance(rs, list) and _nat(first)) or n != leaves:
        return False

    def read_ok(u: int, x: Any) -> bool:
        if x is None:
            return True
        r = row(u)
        return _nat(x) and r is not None and p.holds(q, u) and x == r
    return all(read_ok(first + k, x) for k, x in enumerate(rs))


def _linked_row(own: Own, p: Part, r: Rows, u: int) -> int | None:
    """`SecondStage.linkedRow`: the row slot `u` of part `p` reads at a port a link names, of rows `r`: its unit's
    (`row_at`) of the `per` its cell's replay block holds in the boundary port."""
    i, x = p.span(u), row_at(r.runs, p.unit(u))
    return own.sample[i] * r.per + x if 0 <= i < len(own.sample) and x is not None and x < r.per else None


def _wired_row(p: Part, r: Rows, u: int) -> int | None:
    """`SecondStage.wiredRow`: the row slot `u` of part `p` reads at a port a wire names, of rows `r`: its unit's writer
    unit's (`row_at`), of the `per` its cell's replay block holds in the writer's output port."""
    x = row_at(r.runs, p.unit(u))
    return p.span(u) * r.per + x if x is not None and x < r.per else None


def _read_ok(own: Own, named: bool, p: Part, port: str, e: Any) -> bool:
    """`SecondStage.readOk`: template ``p.template``'s session port `port` may read registered-reads entry `e`: a port a
    link names, only the boundary port it names, of its rows' `per` a replay block, each slot only its unit's row of its
    cell's replay block's (`_linked_row`), or, where the interior registration registers a value of the link's name,
    one the boundary registration's record has no port or value of (`names_ok`), only that value, of its rows' `per` a
    sampled block, each slot only its unit's row of its cell's (`_wired_row`); a port a wire names, only its writer's output port of the interior record, of
    the writer's units a block, each slot only its unit's writer unit's row in its cell's replay block (`_wired_row`);
    any other, its own port of the interior record, each slot only its own first row, or a value either registration
    registers, never a boundary port. Each at its root, domain and leaves, at most `MAX_LEAVES` of them."""
    v, q = _get(e, "value"), len(own.sample)
    if not isinstance(v, str):
        return False
    link = _link_of(own, p.template, port)
    if link is not None:
        if v != link.value:
            return False
        x, r = _registered_entry(own.interior, v), _rows_of(own, p.template, port)
        if x is not None:
            ir = r
            return ir is not None and _same_at(e, x) and _placed_ok(q, p, e, q * ir.per, lambda u: _wired_row(p, ir, u))
        b, m = _by_port(own.boundary).get(v), _get(own.boundary, "population")
        if not (b is not None and _nat(m) and r is not None):
            return False
        lr = r
        return _same_at(e, b) and _placed_ok(q, p, e, m * lr.per, lambda u: _linked_row(own, p, lr, u))
    wire = _wire_of(own, p.template, port)
    if wire is not None:
        x, pw, r = _by_port(own.interior).get(v), _writer_of(own, wire), _rows_of(own, p.template, port)
        if not (v == _port_of(named, wire.writer, wire.wport) and x is not None and pw is not None and r is not None):
            return False
        wr = r
        return (_same_at(e, x) and pw.n == wr.per
                and _placed_ok(q, p, e, q * wr.per, lambda u: _wired_row(p, wr, u)))
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
    places, which ``second-read`` holds to the run (`_placed_ok`, `_rows_ok`): a port a link names, each slot at its
    unit's row of its cell's replay block's, or one reading a port of the interior record, its own port each slot at its
    own rows or, at a port a wire names, its writer's at its writer unit's row (`_read_ok`'s first three cases). Any
    other port reads a value a registration registers, which ``second-read`` holds only to the value's root, domain and
    leaves (`_same_at`), so ``second-values`` holds its rows to the reference statement's, the same rows in every run.
    A workload whose rows there depend on the sampled block (a positional table) would need a rule of its own."""
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


def interior_reads(own: Own, ss: Sequence[Any]) -> list[dict[str, str]]:
    """`SecondStage.interiorReads`: the reads of statements `ss` that `_read_ok` takes at a value only the interior
    registration registers: at a port a link names, of the link's value where the interior registers it (never the
    boundary port then); or at a port no link or wire names, of a value that is no port of either record and that the
    boundary doesn't register. Each is ``{"template", "port", "value"}``, in statement and port order. These reads rest
    on the interior's own values (a history table the interior registers, or a value its verifier supplies, say), and
    the attached record names them (`attach`)."""
    out = []
    for s in map(_stmt, ss):
        for port, e in sorted(_entries(_get(s.registered, "ports")), key=lambda pe: pe[0]) if s is not None else ():
            v = _get(e, "value")
            if not isinstance(v, str):
                continue
            link = _link_of(own, s.template, port)
            if link is not None:
                on_interior = link.value == v and _registered_entry(own.interior, v) is not None
            else:
                on_interior = (_wire_of(own, s.template, port) is None
                               and v not in _by_port(own.interior) and v not in _by_port(own.boundary)
                               and _registered_entry(own.boundary, v) is None
                               and _registered_entry(own.interior, v) is not None)
            if on_interior:
                out.append({"template": s.template, "port": port, "value": v})
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


def names_ok(own: Own) -> bool:
    """`SecondStage.namesOk`: no value the interior registration registers (its ``registered`` entries' values) is of a
    name the boundary registration's record has, a port (`_by_port`) or a registered value (`_registered_entry`), so
    each name a session reads is one registration's."""
    ports, es = _by_port(own.boundary), _get(own.interior, "registered")
    return not any(isinstance(v := _get(e, "value"), str)
                   and (v in ports or _registered_entry(own.boundary, v) is not None)
                   for e in (es if isinstance(es, list) else ()))


def returns_ok(own: Own) -> bool:
    """`SecondStage.returnsOk`: no value the interior registration registers (its ``registered`` entries' values) is of
    a name the replay block's return has in the boundary registration's record (`Own.returns`, `replay_returns`), so a
    port a link names for the return reads no value of the interior's (`_read_ok`): only the port the boundary
    registration's record has of that name, committed before the first stage's sample."""
    es = _get(own.interior, "registered")
    return not any(isinstance(v := _get(e, "value"), str) and v in own.returns
                   for e in (es if isinstance(es, list) else ()))


def _own_values(own: Own) -> dict[str, tuple[tuple[tuple[str, str], ...], ...]]:
    """`SecondStage.ownValues`: the supplied values each of whose sources is the verifier's own (`OWN_SOURCES`), with
    their rows: a name's last."""
    return {v: rows for v, rows in own.supplied if all(k in OWN_SOURCES for r in rows for k, _n in r)}


def _bytes_of(row: Sequence[tuple[str, str]]) -> str | None:
    """`SecondStage.bytesOf`: the bytes a row is, in hex: ``h`` where the row is the one source ``{"bytes": h}``, None
    for any other row."""
    return row[0][1] if len(row) == 1 and row[0][0] == "bytes" else None


def _read_size(own: Own, t: str) -> int | None:
    """`SecondStage.readSize`: the rows V of the table template `t` reads, where the auditor's params record `t` as a
    read (``own.reads``, a template's first); None for a template they record none for."""
    return next((v for u, v in own.reads if u == t), None)


def _one_table(own: Own, mine: Mapping[str, Any], t: str, root: str, domain: str) -> bool:
    """`SecondStage.oneTable`: values `root` and `domain`, which template `t`'s root and domain ports read, are one
    table's of round 1's record at every unit, a table of as many leaves as the read has rows. `t` is a read of V rows
    (`_read_size`). Each value has at least one row and as many as the other; row ``i`` of each is one ``bytes`` source
    (`_bytes_of`), and some port of round 1's record (`_by_port`) registers row ``i`` of `root` as its root under row
    ``i`` of `domain` as its domain id, at V leaves; and the two ports' rows place each unit alike (``own.rows``: one
    ``per`` and the same runs). So a unit reads the root and the domain id of one table of V leaves that round 1's
    record registers."""
    a, b = ([_bytes_of(r) for r in mine[v]] for v in (root, domain))
    ports, size = list(_by_port(own.boundary).values()), _read_size(own, t)
    at = [None if (r := _rows_of(own, t, p)) is None else (r.per, r.runs) for p in OPENING_TABLE]
    return (size is not None and bool(a) and len(a) == len(b)
            and all(r is not None and d is not None
                    and any(type(_get(e, "root")) is str and type(_get(e, "domain")) is str
                            and _get(e, "root") == r and _get(e, "domain") == d
                            and _nat(n := _get(e, "leaves")) and n == size for e in ports)
                    for r, d in zip(a, b))
            and at[0] is not None and at[0] == at[1])


def _opened_row(own: Own, registered: set[str], mine: Mapping[str, Any], x: Link) -> bool:
    """`SecondStage.openedRow`: link `x` names the opened row of a read against a root round 1 committed. Its template
    is a read (`_read_size`) and its port the row, its salt or its path (`OPENING_ROW`); the read's root and domain
    (`OPENING_TABLE`) are each linked, at values the verifier fills from its own bytes (`mine`), which are one table's
    of round 1's record at every unit, of the read's rows (`_one_table`); its key (`OPENING_KEY`) is no wire, and is
    linked at values the interior registration doesn't register (`registered`: a port of round 1's record then, by
    ``second-links``) or the verifier holds (``own.fixed``, `mine`); and its verdict (`OPENING_VERDICT`) reads, in its
    template's reference statement's reads (`Own.ref_reads`, the template's first), a value the verifier fills, of at
    least one row, each row `VERDICT_ONE`. A key the block computes, a wire, is refused: no statement covers it
    (`_inputs_why`)."""
    t = x.template
    at = lambda p: [y.value for y in own.links or () if y.template == t and y.port == p]  # noqa: E731
    held = lambda v: v not in registered or v in own.fixed or v in mine  # noqa: E731
    ref = next((r for u, r in own.ref_reads if u == t), None)
    verdict = _get(_get(_get(ref, "ports"), OPENING_VERDICT), "value")
    roots, domains = at(OPENING_ROOT), at(OPENING_DOMAIN)
    return (_read_size(own, t) is not None and x.port in OPENING_ROW
            and all(at(p) and all(v in mine for v in at(p)) for p in OPENING_TABLE)
            and all(_one_table(own, mine, t, r, d) for r in roots for d in domains)
            and _wire_of(own, t, OPENING_KEY) is None and bool(at(OPENING_KEY)) and all(map(held, at(OPENING_KEY)))
            and isinstance(verdict, str) and bool(mine.get(verdict)) and all(r == VERDICT_ONE for r in mine[verdict]))


def _read_unlinked(own: Own, v: str) -> bool:
    """`SecondStage.readUnlinked`: whether some template's reference statement (``own.ref_reads``) reads value ``v`` at
    a port no link names it at: a port `second-values` then lets every session of the template read ``v`` at, with no
    link's check of it."""
    at = {(x.template, x.port) for x in own.links or () if x.value == v}
    reads = [(t, ports) for t, ref in own.ref_reads if isinstance(ports := _get(ref, "ports"), Mapping)]
    return any((t, p) not in at for t, ports in reads for p, e in ports.items() if _get(e, "value") == v)


def inputs_failure(own: Own) -> str | None:
    """`SecondStage.inputsFailure`: the first value the interior registration registers (its ``registered`` entries'
    values, in the record's order) that a replay block's input may not read, None for none. A value passes where the
    verifier holds it without the prover: one it pinned or holds to an earlier record (``own.fixed``), or one it fills
    from its own bytes (`_own_values`). Any other passes only as the opened row of a read against a root round 1
    committed: at least one link names it, each link that does is such a row's (`_opened_row`), and no template's
    reference statement reads it at a port no link names it at (`_read_unlinked`), so nothing reads the row but its
    opening. So every input of a replay block the prover gives after the first stage's sample is a row a read opens
    under the root of a table round 1's record registers, at a key round 1's record or the verifier fixes, and the
    prover's choice of it is SHA-512's collision: the links at round 1's ports, the wires and the values the verifier
    holds are the block's other inputs, none the interior registration's to choose."""
    es = _get(own.interior, "registered")
    names = [v for e in (es if isinstance(es, list) else ()) if isinstance(v := _get(e, "value"), str)]
    registered, mine = set(names), _own_values(own)
    for v in names:
        if v in own.fixed or v in mine:
            continue
        links = [x for x in own.links or () if x.value == v]
        if not (links and all(_opened_row(own, registered, mine, x) for x in links)) or _read_unlinked(own, v):
            return v
    return None


def _inputs_why(own: Own, v: str) -> str:
    """`SecondStage.inputsWhy`: why value `v` is refused (`inputs_failure`): `KEY_WHY` where a link names it as the
    opened row of a read whose key is wired, a key the block computes, else `INPUTS_WHY`."""
    wired = any(x.value == v and _read_size(own, x.template) is not None and x.port in OPENING_ROW
                and _wire_of(own, x.template, OPENING_KEY) is not None for x in own.links or ())
    return KEY_WHY if wired else INPUTS_WHY


def _interior_failures(own: Own) -> list[tuple[str, Any]]:
    """`SecondStage.interiorFailures`: ``second-interior-order`` unless `interior_order_ok`, then
    ``second-interior-names`` unless `names_ok`, then ``second-return`` unless `returns_ok`, then ``second-schedule``
    where the schedule the boundary registration's record commits is not one of its replay blocks
    (`schedule_failure`), then ``second-inputs`` where the interior registration registers a value a replay block's
    input may not read (`inputs_failure`), naming the first and why (`_inputs_why`)."""
    why, value = schedule_failure(own.boundary), inputs_failure(own)
    return (([] if interior_order_ok(own) else [("second-interior-order", INTERIOR_ORDER_WHY)])
            + ([] if names_ok(own) else [("second-interior-names", NAMES_WHY)])
            + ([] if returns_ok(own) else [("second-return", RETURN_WHY)])
            + ([] if why is None else [("second-schedule", why)])
            + ([] if value is None else [("second-inputs", {"value": value, "why": _inputs_why(own, value)})]))


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
    elif not (_links_ok(own.parts, own) and _wires_ok(own.parts, own)):
        found = [*_interior_failures(own), *([] if _links_ok(own.parts, own) else [("second-links", LINKS_WHY)]),
                 *([] if _wires_ok(own.parts, own) else [("second-wires", WIRES_WHY)])]
    elif statements is None:
        found = [*_interior_failures(own), ("second-statements", STATEMENTS_WHY)]
    else:
        found = _failures(allow, own, own.parts, list(statements))
    c = _Collect()
    for code, what in found:
        c.fail(code, what)
    return c.verdict()


def attach(rec: Mapping[str, Any], v: Verdict, reads: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """The interior registration's audit record `rec` with round 2's check `v` as ``checks.statements``, and the reads
    `reads` that rest on the interior's own values (`interior_reads`) as ``interior_reads`` where there are any. It
    fails where `v` does, as a failed check of its own fails it: not accepted, and no profile, integrity profile or
    wrong units."""
    out = {**rec, "checks": {**rec["checks"], "statements": v.to_json()},
           **({"interior_reads": [dict(r) for r in reads]} if reads else {})}
    return out if v.ok else {**out, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None}


__all__ = ["FORM", "READ_ROW_FIELDS", "READ_RUN_FIELDS", "RETURN_WHY", "Checked", "Link", "Own", "Part", "Rows", "Run",
           "Wire",
           "at_run", "attach", "blank_reads", "blank_run", "check", "inputs_failure", "interior_order_ok",
           "interior_reads", "interior_root", "link_runs", "links_of",
           "own_draw", "parts", "port_binding", "replay_links", "replay_returns", "replay_rows", "replay_wires",
           "returns_ok", "row_at", "rows_of",
           "set_name", "statement_reads", "supplied_of", "verified_of", "wire_runs", "wires_of"]
