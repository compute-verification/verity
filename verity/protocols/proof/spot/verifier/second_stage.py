"""Round 2's statement check, as the verifier's `Spot.SecondStage` defines it, which the difftest's
`second_statements` holds this to: a two-stage run's second-stage sessions held, each by its statement, to the interior
docket and to the second sample's cells. Its verdict of record is the Lean's (`SecondStage.checkAny`, inside
`Run.run`), which no interface here reaches on its own: `check` is the Python twin's verdict held to the codes the
Lean's run gave (`DISAGREE` where they differ), `complete` raises (`verifier.not_written`), and `check_twin` and
`complete_twin` are the Python twins, the difftest's references. `service.Stream` adds the twin's check to the
interior docket's audit record as `checks.statements` (`attach`), the Python side `Run.run` is held to, and the
outcome is `Run.run`'s. The parts and their cells, a statement and its codes are the Lean's
(its module docstring).

A statement is one second-stage session's: ``{"template", "header", "draw", "registered"}``, the template its circuit is
an instance of (its stratum's name under the check's query), its public file's header, its session record's
``unit_draw``, and the registered reads it was verified against (``--registered``), None for none. Round 2 takes it only
where an accepted verdict of record names its header, its registered reads and its sample (``second-verified``,
`verified_of`): ``public_header_sha512`` and ``registered_sha512``, the SHA-512 of the header's and the
``--registered`` file's canonical JSON, and ``unit_draw``, which `verity-verify-proof verify` writes of the session it checked;
and only where every one that does checked its session as one table (``session_tables``, ``second-tables``): a session
split across Flock tables is refused, as one split across statements is (``second-template``).

The design's links are the verifier's own, derived from the replay block's Definition (`replay_links`, which the
difftest's ``replay_links`` holds to `SecondStage.replayLinks`) and never supplied; so are its wires (`replay_wires`,
`SecondStage.replayWires`): a template's parameter that is, at each of its callers, one field of one instance of
another template reads that writer's output port for the field, one value the interior commits once, so
``second-read`` holds a wired port to its writer's port at its writer unit's row and `at_run` binds it by that port
(`placings`). No wire reads an output the replay block returns, a writer's port a link names: the boundary commits
that output, and the interior's port of it is another tree, which nothing holds to the first. No link names a wire's
reader's port either: a session reads a port a link names at the link's entry (`_read_ok`), so a wire naming it too
would hold nothing of what is read. And each wire's writer's part is before its reader's (``second-wires``,
`_wires_ok`).
A template may have several callers in the body, a batch's members among them, its units theirs in
body order; each port a link or a wire names reads, unit by unit, the row its caller names (`replay_rows`,
`SecondStage.replayRows`), summarized once per Definition as runs (`Run`) and never unrolled. A port a link names reads
the boundary port the link names or, where the interior docket registers a value of the link's name (a value its
verifier supplies, say), that value, at its unit's row of its cell's sampled block (`_read_ok`). Each name is one
docket's: an interior docket that registers a value of a name the boundary docket's record has, a
port or a registered value, is refused (``second-interior-names``, `names_ok`), so the interior's value of a name never
stands in for one round 1 committed. What a replay block returns is committed before the first stage's sample: an
interior docket that registers a value of a name the block's return has in the boundary docket's record,
one of its output ports ``o<j>`` (`replay_returns`), is refused (``second-return``, `returns_ok`), so a port a link
names for the return is read only at the port round 1's record has for it.

Each statement's header is held to the reference header the auditor's params give its template (`Own.refs`, from
``proofs.templates``: `service.template_circuits`), the registered reference statement's, fixed before the run, placed
in the run (`at_run`): its ``set`` the session's input set, ``served/<run>/m<k>`` (`set_name`, ``<run>`` the interior
record's ``window.run`` and ``k`` its template's position among the parts, its member's in the interior window), and its
``frame_v3.bindings`` each of its ``frame_v3.schemas`` ports' bindings at its one placing (`placings`): its own tree's or
its writer's tree's under the window-domain rule over the interior record's program, partition and that tree's leaves
(`port_binding`), or the binding the registered value a link names it to read was committed under, recomputed from the
held registration (`registered_binding`); a port none or two rules place is refused (``second-place-none``,
``second-place-two``, `placing_failures`). The verifier computes both, so the prover chooses neither. The two are the
same with the run's fields blanked (`blank_run`, `RUN_FIELDS`): ``instances`` and ``units.indices``, which
``second-population`` holds, ``units.blocks``, the indices cut into blocks, which the session's verifier holds to them,
``frame_v3.roots`` and ``frame_v3.domain_ids``, which ``second-roots`` holds, and ``registered``, whose reads ``second-read`` holds (``second-header``).

A header that shares rows (``shared_rows``: tables whose rows its instances read by the refs in its public file's body)
shares them only at ports its statement reads registered (``second-refs``, `_refs_ok`): there the session's verifier
holds each instance's ref to the row at the position the read gives its unit (`Flock.Registered.check`), and
``second-read`` holds the read to the slot's own rows or its cell's replay block's; elsewhere it only bounds a ref by its
table, so a ref could point a sampled proof block at another unit's rows.

Each statement's registered reads are its template's reference statement's (`Own.ref_reads`, the registered reads the
auditor's params give beside its header, fixed before the run and taken as given: round 2 never verifies the reference
statement against them), the two the same with each port's run fields blanked (`blank_reads`) (``second-values``,
`_values_ok`). The run's fields are those ``second-read`` holds
to the run: every port's ``root``, ``domain`` and ``leaves`` (`READ_RUN_FIELDS`), and the rows (``reads``,
``reads_from``: `READ_ROW_FIELDS`) of a port a link names or one reading a port of the interior record (its own, or at
a port a wire names its writer's), whose rows the run places (`_run_places_rows`). So the ports a session reads
registered, the value each reads, and the rows at which it reads a value a docket registers are fixed before the
run: ``second-read`` alone lets a port no link or wire names read any value either docket registers, at any rows.

The interior record names the replay sample it lays out (``window.replay_sample``, which the proof server writes), and
``second-interior-order`` fails unless it is the verifier's own, in its order (`interior_order_ok`), checked before any
statement is read. With it, ``second-interior-names`` fails where the interior record registers a value of a name the
boundary docket's record has (`names_ok`), ``second-return`` where it registers one of a name the replay
block's return has there (`returns_ok`), and ``second-schedule`` where the boundary docket's record
commits a schedule (``window.schedule``) that is no schedule of its replay blocks (`schedule.failure`); a record that
commits none passes.

Every input of a replay block that the prover gives is a port of round 1's record (a link), another template's output
in the block (a wire), a value the verifier holds without the prover, or the opened row of a read against a root round
1 committed: the interior docket, made after the first stage's sample, registers nothing else
(``second-inputs``, `inputs_failure`). A value it registers passes where the verifier pinned it, holds it to an earlier
record, or fills it from its own bytes (`Own.fixed`, `Own.supplied`). Any other is the prover's, and passes only as a
read's row, salt or path, where the verifier fills the read's root and domain with the root and the domain id round 1's
record registers for one table (`_one_table`, by their bytes: `Own.supplied` holds what the verifier computed for a
table in place of its name, as the interior docket's check read it), the read's key is a port of round 1's
record or a value the verifier holds, and the read's verdict port reads a value the verifier fills with 1
(`_opened_row`): the row is then the table's at that key, short of a SHA-512
collision. A key the block computes, a wire, is refused, with a reason of its own (`KEY_WHY`): no statement covers it.
A read is a template the auditor's params record as one (`Own.reads`, `service.template_reads`): the template
check (``flock-read-tmpl check V H``, `Flock.ReadNet.check`) accepted the circuit file they name for it as the read of
a table of V rows, where the params were made; round 2 runs no template check. Which table a read opens is the one
whose root and domain id the verifier fills, a table of V leaves in round 1's record, and which row its key. A design
whose key is the prover's after the sample, whose root and domain are not one table's of round 1's record, or whose
table has another leaf count than the read's rows, is refused here. The code is reported beside the others, in every
branch of `check`, so a run that has it still reaches its other stops."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from verity.circuits import partition_object as PO
from verity.circuits.codec import canonical_json, definitions
from verity.circuits.cut import member_refs
from verity.circuits.program import RUN, input_id
from verity.circuits.refs import runs as ref_runs
from verity.circuits.replay_links import (
    Link, Part, Rows, Run, Wire, _by_port, _by_template, _callers_of, _fields, _fits, _get, _outputs, _port_of,
    _return_links, link_runs, links_of, parts, replay_links, replay_rows, row_at, wire_runs,
)
from verity.commitments import registered_value as RV
from verity.commitments.window import window_binding

from .one_stage import verifier as V
from .one_stage.verdict import Verdict, _Collect
from .schedule import failure as schedule_failure

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
             "don't place its part's units or its writer's, or it reads an output the replay block returns, or a link "
             "names the port that reads it, or its writer's part is not before its reader's")
INTERIOR_ORDER_WHY = "the interior record's replay sample (window.replay_sample) is not the verifier's own, in its order"
NAMES_WHY = ("the interior registration registers a value of a name the boundary registration's record has, a port or a "
             "registered value")
RETURN_WHY = ("the interior registration registers a value of a name the replay block's return has in the boundary "
              "registration's record: what a replay block returns is read only at a port that record commits")
INPUTS_WHY = ("the interior registration registers a value that is neither one the verifier holds without the prover (one "
              "it pinned, one it holds to an earlier record, or one it fills from its own bytes) nor the opened row of a "
              "read the auditor's params record, whose root and domain it fills from one table of the read's rows that "
              "round 1's record registers or its params pin, whose key is a port of round 1's record or a value it "
              "holds, and whose verdict it holds to 1")
KEY_WHY = ("the interior registration registers the opened row of a read at a key the block computes (the read's key is "
           "wired to another template's output): no statement covers it")
#: `SecondStage.WIRED_KEY`, off: whether round 2 takes a read whose key a wire fills (`_opened_row`, `_key_failure`).
#: Off, it refuses each one with `KEY_WHY`. A constant of the code: `Own.wired_key` defaults to it, and nothing the
#: service reads sets the field. Only development plays set it on, never an accept of record. It turns on only once
#: the statement that a sampled reader unit's key is its writer unit's committed output row is proved.
WIRED_KEY = False
#: The four rules that place a session port's binding (`placings`): its own tree of the interior record, its writer's
#: tree there, a value the interior registration registers, round 1's tree of a boundary port.
OWN_TREE, WRITER_TREE, REGISTERED, ROUND1_TREE = "own", "writer", "registered", "round1"
PLACE_NONE_WHY = ("a port of a template's reference header is placed by none of the four rules: its own tree of the "
                  "interior record, its writer's tree where a wire names it, a value the interior registration "
                  "registers where a link names it, round 1's tree of the boundary port a link names it to read")
PLACE_TWO_WHY = ("a port of a template's reference header is placed by more than one of the three rules: its own tree of "
                 "the interior record, its writer's tree where a wire names it, a value the interior registration "
                 "registers where a link names it")
KEY_WIRES_WHY = "a read's key is wired, but not by exactly one wire"
KEY_LINKED_WHY = "a read's key is both wired and linked: it is its writer's row or a value the verifier holds, never both"
KEY_WIDTH_WHY = "a read's key is wired to a writer's output row of other u16 words than the key's row"
KEY_ORDER_WHY = "a read's key is wired to a writer whose part is not before the reader's in the interior window"
KEY_BOUNDS_WHY = ("a read's key is wired, and its rows give a unit a writer row outside its writer's units a replay "
                  "block")
#: A read's ports (`gates.embedding.EmbeddingRead`, `Flock.ReadNet.readTmpl`): its key, the table's root and domain
#: id, the opened row with its salt and its path, and its verdict (`SecondStage.OPENING_KEY` and the names after it).
OPENING_KEY, OPENING_TABLE, OPENING_ROW, OPENING_VERDICT = "tok", ("root", "domain"), ("row", "salt", "path"), "o1"
#: A read's ports a key reads at: its key, its row's ports and its verdict (`SecondStage.KEYED_PORTS`).
KEYED_PORTS = (OPENING_KEY, *OPENING_ROW, OPENING_VERDICT)
KEY_ROWS_WHY = ("a read whose key is wired lists table_rows at a port of its opening: the rows it reads are a function "
                "of a committed value, never public positions, so it lists none")
TABLE_ROWS_WHY = "a registered read lists table_rows, which round 2 does not read yet: a read lists every row"
OPENING_ROOT, OPENING_DOMAIN = OPENING_TABLE
#: The sources of a value the verifier fills from what it holds before the first stage's sample: its own bytes, among
#: them the root and the domain id it computed for a table, which `Own.supplied` holds in place of the table's name
#: (`history.resolve`; `SecondStage.OWN_SOURCES`). A ``port`` or a ``value`` source reads a root of the interior record,
#: the prover's after the sample, and a ``position`` or a ``window`` source the interior record's population or window.
#: A ``history`` source left unresolved is none of the verifier's bytes.
OWN_SOURCES = ("bytes",)
#: The row of a value that holds a read's verdict to 1: the verifier's bytes of the number 1 as one 16-bit half, one leaf
#: of the verdict's one word (`registered.half_leaves`; `SecondStage.VERDICT_ONE_LEAF`).
VERDICT_ONE_LEAF = (("bytes", "0001"),)
STMT_KEYS = ["draw", "header", "registered", "template"]
#: The most leaves a tree a session reads may have: its positions are 64-bit (`SecondStage.MAX_LEAVES`).
MAX_LEAVES = 2 ** 64
#: The run's fields of a statement's header, each a path of keys: the only ones a header may differ from its template's
#: reference placed in the run in, which other checks hold instead: ``second-population`` the first two, the session's
#: verifier ``units.blocks``, the indices cut into blocks of ``vus_per_block`` (its header's ``units`` must be
#: `Flock.HmRow.unitsEntry` of its indices), ``second-roots`` the next two, and ``second-read`` ``registered``, each
#: read's root, domain and rows, with ``second-values`` the value each port reads, and the rows at which it reads a
#: value a docket registers, through the ``--registered`` file the session's verifier holds it to
#: (`SecondStage.RUN_FIELDS`). A reference staged at other slots differs from the header only in the indices and their
#: blocks.
RUN_FIELDS = (("instances",), ("units", "indices"), ("units", "blocks"), ("frame_v3", "roots"), ("frame_v3", "domain_ids"),
              ("registered",))


def complete_twin(request_definition: bool, records: Sequence[Mapping[str, Any] | None]) -> bool:
    """Whether a two-stage outcome is complete, in the sense one-stage's accept publishes (`one_stage.audit.COMPLETE`:
    every verdict ran). It is only when both hold. The replay block is the registered request Definition
    (`request_definition`): a sample of a stand-in audits the stand-in, never the request. Each stage's audit record
    (`records`, at least one, none missing) accepted and is itself complete. So it is false of every two-stage outcome
    whose replay block is a stand-in, whatever its audit records say."""
    return (request_definition is True and len(records) > 0
            and all(r is not None and r.get("accepted") is True and r.get("complete") is True for r in records))


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
class Width:
    """`SecondStage.Width`: the u16 words of the rows a wire joins: template `template`'s session port `port` reads rows
    of `reads` words, and its writer's output port writes rows of `writes` (`replay_widths`)."""

    template: str
    port: str
    reads: int
    writes: int

    def to_json(self) -> dict[str, Any]:
        return {"port": self.port, "reads": self.reads, "template": self.template, "writes": self.writes}


def widths_of(j: Any) -> tuple[Width, ...] | None:
    """`SecondStage.widthsOf?`: widths from their JSON, a list of objects of exactly ``template``, ``port``, ``reads``
    and ``writes``, two strings and two natural numbers; None for anything else."""
    if not isinstance(j, list):
        return None
    out = []
    for w in j:
        if not (isinstance(w, dict) and sorted(w) == ["port", "reads", "template", "writes"]):
            return None
        if not (isinstance(w["template"], str) and isinstance(w["port"], str) and _nat(w["reads"])
                and _nat(w["writes"])):
            return None
        out.append(Width(w["template"], w["port"], w["reads"], w["writes"]))
    return tuple(out)


@dataclass(frozen=True)
class Held:
    """`SecondStage.Held`: the terms the interior registration's own entries are computed under (`Docket.ownEntries`,
    the docket check's ``R7-registered``): its Program's digest, its window's open root, and the values held across
    windows, whose domain leaves that root out (`Docket.fixedOf`)."""
    program: str
    window: str | None
    fixed: tuple[str, ...]


def held_of(program: str, window_root: str | None, held: Sequence[str] | None = None,
            pinned: Mapping[str, Any] | None = None, supplied: Mapping[str, Any] | None = None,
            outputs: Mapping[str, Any] | None = None) -> Held:
    """`Held` of the interior registration the docket check took with these arguments: `program` its Program's digest,
    and the values held across windows its ``held`` list, then its pinned, supplied and output values (`docket._window`)."""
    return Held(program, window_root, (*(held or ()), *(pinned or {}), *(supplied or {}), *(outputs or {})))


@dataclass(frozen=True)
class Pos:
    """`SecondStage.Pos`: template `template`'s session port `port` reads row ``rel[k]`` at unit ``first + k`` (None
    for none), as C gives it (`Flock.OneCircuit.pos`)."""

    template: str
    port: str
    rel: tuple[int | None, ...]
    first: int


def _pos(program: Any, tp: Any, indices: Sequence[int], port: str, e: Any) -> Pos | None:
    """`Flock.OneCircuit.pos` of registered-reads entry `e` at the statement's units `indices`, None where it refuses:
    each unit's parameter (``param``, else `port`, ``name[r]`` its row ``r``) one row of one input node's leaves, the
    same node for every unit, whose rows are the entry's ``leaves``."""
    body, offs = program.root.body, list(program.root.body.offsets)
    param = _get(e, "param") if isinstance(_get(e, "param"), str) and _get(e, "param") else port
    name, _, row = param.partition("[")
    value = _get(e, "value") if isinstance(_get(e, "value"), str) else ""
    first, node, rel = min(indices), None, {}
    for u in indices:
        try:
            k, m = tp.locate(u)
        except IndexError:
            return None
        x = body.nodes[k]
        names = [p for p, _t in x.fn.signature()[0]]
        if name not in names:
            return None
        i = names.index(name)
        refs = member_refs(x.args[i], x.axes[i] if x.form == "batch" else None, m)
        n = len(refs)
        if row:
            if not (row.endswith("]") and row[:-1].isdigit()):
                return None
            cut, at = n // x.fn.signature()[0][i][1].n, int(row[:-1])
            if at * cut >= n:
                return None
            refs, n = refs.slice(at * cut, (at + 1) * cut), cut
        got = ref_runs(refs)
        if len(got) != 1 or got[0][0] != "n" or n == 0 or got[0][2] % n:
            return None
        w, lo = got[0][1], got[0][2]
        kind = input_id(getattr(body.nodes[w].fn, "id", ""))
        if kind is None or (node is not None and node != w):
            return None
        if node is None:
            fresh = f"input/n{w}"
            if (kind[0] != RUN and value.startswith("input/")) or (kind[0] == RUN and value != fresh) \
                    or (offs[w + 1] - offs[w]) // n != _get(e, "leaves"):
                return None
        node, rel[u] = w, lo // n
    return Pos("", port, tuple(rel.get(u) for u in range(first, max(indices) + 1)), first)


def positions_of(program: Any, query: Mapping[str, Any], statements: Sequence[Any] | None) -> tuple[Pos, ...]:
    """`SecondStage.positionsOf`: for each port of each statement's registered reads, the rows C, the interior
    `program` under `query`, gives its header's ``units.indices`` (`_pos`); a port C gives none has none, which round 2
    refuses but at a link's output, read at round 1's rows (`_at_pos`)."""
    tp, out = PO.TemplatePartition(program, query), []
    for j in statements or ():
        s = _stmt(j)
        idx = _get(_get(s.header, "units") if s else None, "indices")
        if s is None or not isinstance(idx, list) or not idx or not all(_nat(i) for i in idx):
            continue
        for port, e in _entries(_get(s.registered, "ports")):
            p = _pos(program, tp, idx, port, e)
            if p is not None:
                out.append(replace(p, template=s.template))
    return tuple(out)


@dataclass(frozen=True)
class Own:
    """What the verifier holds of the run: the replay sample, the parts (`parts`), the interior and boundary
    dockets' records, the second sample's units, the design's links (`replay_links`, None when the replay block's
    Definition makes none), the ``{value: root}`` the verifiers report the sessions read
    (`audit.session_registered`), what they name of each session they accepted (`verified_of`), the reference header
    of each template's statement, ``(template, header)``, and the registered reads it reads,
    ``(template, registered)``, None for none (`ref_reads`), as the auditor's params give them
    (`service.template_circuits`), the design's wires (`replay_wires`, None when the Definition makes none), the
    rows of the ports they and the links name (`replay_rows`), the names the replay block's return has in the boundary
    docket's record (`replay_returns`), and, of the values the interior docket registers, the verifier's
    own copies of those it holds without the prover: each one it fills itself, with its rows of sources
    (`supplied_of`), and the names of those it pinned or holds to an earlier record (``fixed``); the templates the
    auditor's params record as reads, each with its table's rows, ``(template, V)`` (`service.template_reads`); the
    registered tables the     auditor's params pin before any coin, ``(parameter, entry)`` each, the entry the tables'
    docket names (``tables``); the words of the rows each wire joins (`replay_widths`); the wired key's
    switch, by default `WIRED_KEY`, which the service never sets otherwise; and the terms the interior registration's
    own entries were held to (`Held`), None for none, which binds no registered value."""

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
    tables: tuple[tuple[str, Any], ...] = ()
    widths: tuple[Width, ...] = ()
    wired_key: bool = WIRED_KEY
    held: Held | None = None
    #: the rows each statement's port reads, as the verifier derives them from C (`SecondStage.positionsOf`); None: as
    #: the file states them (``reads``, ``reads_from``), which only tests take. Lean's run of record always sets it.
    positions: tuple[Pos, ...] | None = None


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


def _stmt(j: Any) -> Stmt | None:
    if not (isinstance(j, dict) and sorted(j) == STMT_KEYS):
        return None
    t, h, d, r = j["template"], j["header"], j["draw"], j["registered"]
    if isinstance(t, str) and isinstance(h, dict) and isinstance(d, dict) and (r is None or isinstance(r, dict)):
        return Stmt(t, h, d, r)
    return None


def round_one_held(links: Sequence[Link], boundary: Any) -> list[dict[str, Any]]:
    """`SecondStage.roundOneHeld`: each link whose value round 1's record `boundary` holds, with that record's entry of
    it, as `Docket.Linked`'s JSON."""
    ports = _by_port(boundary)
    return [{"template": l.template, "port": l.port, "value": l.value, "entry": dict(ports[l.value])}
            for l in links if l.value in ports]


def replay_returns(program: Any, key: str, rec: Any) -> tuple[str, ...]:
    """`SecondStage.replayReturns`: the names the return of the replay block `program` defines under descriptor id `key`
    has in the boundary docket, as record `rec` names its ports: its output ports ``o<j>``, one per field of what
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


def wires_of(defs: Mapping[str, Any], key: str) -> tuple[Wire, ...] | None:
    """`SecondStage.wiresOf`: the design's wires (`wire_runs`)."""
    runs = wire_runs(defs, key)
    return None if runs is None else tuple(x for x, _per, _rs in runs)


def replay_wires(program: Any, key: str) -> tuple[Wire, ...] | None:
    """`SecondStage.replayWires`: the wires of the replay block `program` defines under descriptor id `key` (`wires_of`
    over the definitions its descriptor records, `codec.definitions`)."""
    return wires_of(definitions(program), key)


def replay_widths(program: Any, key: str) -> tuple[Width, ...]:
    """`SecondStage.replayWidths`: the design's widths: for each wire of the replay block `program` defines under
    descriptor id `key` (`replay_wires`), the u16 words of a row its reader's port reads, its parameter's, and of one
    its writer writes, the writer's output field the wire names (`_words`). A wire whose templates or field the
    program doesn't define gives none."""
    defs, out = definitions(program), []
    for w in wires_of(defs, key) or ():
        t, tw = defs.get(w.template), defs.get(w.writer)
        if t is None or tw is None:
            continue
        ty = next((ty for p, ty in t.signature()[0] if p == w.port), None)
        fs = _fields(tw.signature()[1])
        g = next((g for f, g in enumerate(fs) if f"o{f}" == w.wport), None)
        if ty is not None and g is not None:
            out.append(Width(w.template, w.port, _words(ty), _words(g)))
    return tuple(out)


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
    by its rows (`_placed`), whose rows a block are its writer's units a block, whose writer's port and whose own port
    no link names, and whose writer's part is before its reader's (`_pos_of`)."""
    def writer_ok(w: Wire) -> bool:
        r = _rows_of(own, w.template, w.port)
        return (r is not None and any(p.template == w.writer and r.per == p.n for p in ps)
                and _link_of(own, w.writer, w.wport) is None and _link_of(own, w.template, w.port) is None
                and _pos_of(ps, w.writer) < _pos_of(ps, w.template))
    return own.wires is not None and all(any(p.template == w.template and _placed(own, p, w.port) for p in ps)
                                         and writer_ok(w) for w in own.wires)


def _wire_of(own: Own, t: str, port: str) -> Wire | None:
    return next((w for w in own.wires or () if w.template == t and w.port == port), None)


def _writer_of(own: Own, w: Wire) -> Part | None:
    """`SecondStage.writerOf`: wire `w`'s writer's part."""
    return next((p for p in own.parts or () if p.template == w.writer), None)


def _key_failure(own: Own, t: str) -> str | None:
    """`SecondStage.keyFailure`: why template `t`'s wired key is not whole, None when it is: exactly one wire names its
    key port (`OPENING_KEY`, else `KEY_WIRES_WHY`), no link does (`KEY_LINKED_WHY`), the writer's output field the wire
    names is a row of the key row's u16 words (`Own.widths`, `KEY_WIDTH_WHY`), the writer's part is before the
    reader's in the interior window (`_pos_of`, `KEY_ORDER_WHY`), and the key's rows give each unit a writer row among
    the writer's units a replay block (`KEY_BOUNDS_WHY`). That a sampled proof block's key is that row's committed value is
    ``second-read``'s (`_read_ok`)."""
    ws = [w for w in own.wires or () if w.template == t and w.port == OPENING_KEY]
    if len(ws) != 1:
        return KEY_WIRES_WHY
    w, ps = ws[0], own.parts or ()
    if _link_of(own, t, OPENING_KEY) is not None:
        return KEY_LINKED_WHY
    width = next((x for x in own.widths if x.template == t and x.port == OPENING_KEY), None)
    if width is None or width.reads != width.writes:
        return KEY_WIDTH_WHY
    if not _pos_of(ps, w.writer) < _pos_of(ps, t):
        return KEY_ORDER_WHY
    p, r = _writer_of(own, w), _rows_of(own, t, OPENING_KEY)
    if p is None or r is None or not all(x.row + max(x.count - 1, 0) * x.step < p.n for x in r.runs):
        return KEY_BOUNDS_WHY
    return None


def placings(own: Own, named: bool, t: str, p: str) -> list[tuple[str, Any]]:
    """`SecondStage.placings`: every rule that places template `t`'s session port `p`, by the interior record and the
    design alone: `OWN_TREE` at its own port of the record, `WRITER_TREE` at its writer's output port where a wire names
    it, and `REGISTERED` at the interior registration's entry of the value a link names it to read; where none of
    those does, `ROUND1_TREE` at the boundary record's port a link names it to read. A port is placed where exactly one
    holds (`placing_failures`)."""
    w, x = _wire_of(own, t, p), _link_of(own, t, p)
    e = None if x is None else _registered_entry(own.interior, x.value)
    three = ([(OWN_TREE, _port_of(named, t, p))] if _port_of(named, t, p) in _by_port(own.interior) else []) + (
        [] if w is None else [(WRITER_TREE, _port_of(named, w.writer, w.wport))]) + (
        [] if e is None else [(REGISTERED, e)])
    if three or x is None:
        return three
    return [(ROUND1_TREE, x.value)] if x.value in _by_port(own.boundary) else []


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
    """`SecondStage.interiorRoot`: the interior window's open root for the boundary docket of digest `boundary`
    and the replay sample `sample`, in order. The verifier computes it, so a second stage sampled on a session of its
    own is sampled on the one window its boundary docket and replay sample name."""
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


_HEX_DIGITS = frozenset("0123456789abcdef")


def registered_binding(held: Held | None, e: Any) -> str | None:
    """`SecondStage.registeredBinding`: the binding the interior registration's entry `e` was committed under, from the
    held registration `held`: `registered_value.domain` over its Program's digest, the value's name, words, leaves and
    word bits, and its window's open root but for a value held across windows; None unless that domain's id is the
    entry's ``domain``, or where `held` is None or the entry names no such domain."""
    v, words, leaves, b = _get(e, "value"), _get(e, "words"), _get(e, "leaves"), _get(e, RV.WIDTH_KEY)
    b = RV.WORD_BITS if b is None else b
    if held is None or not (isinstance(v, str) and _nat(words) and _nat(leaves) and _nat(b) and b in RV.WIDTHS):
        return None
    window = None if v in held.fixed else held.window
    if window is not None and not (isinstance(window, str) and len(window) == 128 and set(window) <= _HEX_DIGITS):
        return None
    d = RV.domain(held.program, v, words, leaves, b, window)
    return d.binding.hex() if _jeq(d.domain_id.hex(), _get(e, "domain")) else None


def _placed_binding(own: Own, named: bool, t: str, p: str, schema: str, run: str) -> str | None:
    """`SecondStage.placedBinding`: the binding of template `t`'s session port `p` of row schema `schema` in run `run`,
    at its one placing (`placings`): an interior record's port's (`port_binding`) or a registered value's
    (`registered_binding`); None where the port has none or two."""
    ps = placings(own, named, t, p)
    if len(ps) != 1:
        return None
    (rule, at), = ps
    if rule == ROUND1_TREE:
        r1 = _run_of(own.boundary)
        return None if r1 is None else port_binding(own.boundary, at, schema, r1)
    return registered_binding(own.held, at) if rule == REGISTERED else port_binding(own.interior, at, schema, run)


def at_run(own: Own, named: bool, k: int, t: str, r: Any) -> Any:
    """`SecondStage.atRun`: template `t`'s reference header `r` placed in the run, for the session of the `k`-th part:
    its ``set`` the run's input set (`set_name`) and its ``frame_v3.bindings`` the binding of each port of
    its ``frame_v3.schemas`` at its one placing (`_placed_binding`): its own tree's, its writer's tree's, or the
    binding the registered value it reads was committed under, which the verifier computes. None when the interior
    record names no run, or a port the verifier can't bind."""
    run = _run_of(own.interior)
    if run is None:
        return None
    bindings = {}
    for p, s in _entries(_get(_get(r, "frame_v3"), "schemas")):
        b = _placed_binding(own, named, t, p, s, run) if isinstance(s, str) else None
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
    """The verifier's own sample for part `p`'s session: the second sample's units among its slots, counted from its first,
    as ``unit_draw`` writes a subset of its population."""
    units = [u - p.lo for u in drawn if p.holds(q, u)]
    return {"law": "subset", "population": q * p.n, "k": len(units), "units": units}


def sampled_slots(q: int, p: Part, us: Sequence[int]) -> list[int]:
    """`SecondStage.sampledSlots`: part `p`'s slots among the second sample's units `us`, in its order."""
    return [u for u in us if p.holds(q, u)]


def _sampled_ok(q: int, p: Part, us: Sequence[int], h: Any) -> bool:
    """`SecondStage.sampledOk`: the header's ``units.indices`` are part `p`'s slots the second stage samples and no
    other, at least one, and its ``instances`` their count: the header of a statement of only its part's sampled proof
    blocks."""
    slots, indices, m = sampled_slots(q, p, us), _get(_get(h, "units"), "indices"), _get(h, "instances")
    return (isinstance(indices, list) and _nat(m) and 0 < m == len(slots) and len(indices) == len(slots)
            and all(_nat(x) and x == u for x, u in zip(indices, slots)))


def sampled_only(q: int, p: Part, us: Sequence[int], h: Any) -> bool:
    """`SecondStage.sampledOnly`: header `h` is of only part `p`'s sampled proof blocks (`_sampled_ok`), and not of
    every slot of the part (`_population_ok`): where the second stage samples every slot the two are one header, held
    as the part's whole."""
    return _sampled_ok(q, p, us, h) and not _population_ok(q, p, h)


def _rows_read(e: Any, slots: Sequence[int]) -> int:
    """`SecondStage.rowsRead`: the rows registered-reads entry `e` gives slots `slots`, each once."""
    f, rs = _get(e, "reads_from"), _get(e, "reads")
    f = f if _nat(f) else 0
    if not isinstance(rs, list):
        return 0
    return len({rs[u - f] for u in slots if f <= u < f + len(rs) and _nat(rs[u - f])})


def at_sampled(r: Any, slots: Sequence[int], reads: Sequence[tuple[str, Any]]) -> Any:
    """`SecondStage.atSampled`: reference header `r`, placed in the run, at the slots a statement of only its part's
    sampled proof blocks holds: its ``range`` those ``len(slots)`` instances', its ``units.classes`` its first
    ``len(slots)``, and each port of its ``shared_rows`` the rows the statement's registered reads `reads` give those
    slots (`_rows_read`), a port they don't read left as it is; a field the reference lacks stays absent."""
    k = len(slots)
    if isinstance(r, dict) and "range" in r:
        r = _put_at([0, k], r, ("range",))
    classes = _get(_get(r, "units"), "classes")
    if isinstance(classes, list):
        r = _put_at(classes[:k], r, ("units", "classes"))
    shared = _get(r, "shared_rows")
    if isinstance(shared, dict):
        by: dict[str, Any] = {}
        for port, e in reads:
            by.setdefault(port, e)
        r = _put_at({port: _rows_read(by[port], slots) if port in by else n for port, n in shared.items()}, r,
                    ("shared_rows",))
    return r


def _sampled_header_ok(own: Own, named: bool, k: int, t: str, h: Any, slots: Sequence[int],
                       reads: Sequence[tuple[str, Any]]) -> bool:
    """`SecondStage.sampledHeaderOk`: `_header_ok` for a statement of only its part's sampled proof blocks `slots`,
    reading `reads` registered: its header is its template's reference header placed in the run (`at_run`) and at those
    slots (`at_sampled`), but for the run's fields."""
    ref = next(((r,) for x, r in own.refs if x == t), None)
    placed = None if ref is None else at_run(own, named, k, t, ref[0])
    return placed is not None and _jeq(blank_run(h), blank_run(at_sampled(placed, slots, reads)))


def whole_sample(m: int) -> dict[str, Any]:
    """`SecondStage.wholeSample`: the sample of every unit of a population of `m`, as ``unit_draw`` writes a subset of
    it: the session's sample of a statement of only its part's sampled proof blocks."""
    return {"law": "subset", "population": m, "k": m, "units": list(range(m))}


def _registered_entry(rec: Any, v: str) -> Any:
    es = _get(rec, "registered")
    return next((e for e in (es if isinstance(es, list) else ()) if _get(e, "value") == v
                 and isinstance(_get(e, "value"), str)), None)


def _same_at(e: Any, x: Any) -> bool:
    """`SecondStage.sameAt`: entries `e` and `x` name one root, domain and leaf count, at most `MAX_LEAVES`."""
    leaves = _get(e, "leaves")
    return (all(_jeq(_get(e, k), _get(x, k)) for k in ("root", "domain", "leaves"))
            and _nat(leaves) and leaves <= MAX_LEAVES)


def _at_pos(own: Own, t: str, port: str, e: Any, or_else: Any = None) -> Any:
    """`SecondStage.atPos`: registered-reads entry `e` of template `t`'s session port `port` at the rows the verifier
    holds (`Own.positions`): its ``reads`` and ``reads_from`` the ones C gives, else ``or_else``, else None, so
    `_placed_ok` and `_rows_ok` refuse the read whatever the file states; `e` itself where the verifier holds no
    positions."""
    if own.positions is None or not isinstance(e, dict):
        return e
    x = next((x for x in own.positions if x.template == t and x.port == port), None)
    if x is None:
        return {**e, "reads": None, "reads_from": None} if or_else is None else or_else
    return {**e, "reads": list(x.rel), "reads_from": x.first}


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


def _linked_rows(own: Own, p: Part, r: Rows, e: Any) -> Any:
    """`SecondStage.linkedRows`: entry `e` at each slot's unit's row of its cell's replay block (`_linked_row`), from
    the part's first slot, `_at_pos`'s ``or_else`` for a link's output C gives no position."""
    if not isinstance(e, dict):
        return e
    return {**e, "reads": [_linked_row(own, p, r, p.lo + k) for k in range(len(own.sample) * p.n)], "reads_from": p.lo}


def _wired_row(p: Part, r: Rows, u: int) -> int | None:
    """`SecondStage.wiredRow`: the row slot `u` of part `p` reads at a port a wire names, of rows `r`: its unit's writer
    unit's (`row_at`), of the `per` its cell's replay block holds in the writer's output port."""
    x = row_at(r.runs, p.unit(u))
    return p.span(u) * r.per + x if x is not None and x < r.per else None


def _read_ok(own: Own, named: bool, p: Part, port: str, e: Any) -> bool:
    """`SecondStage.readOk`: template ``p.template``'s session port `port` may read registered-reads entry `e`: a port a
    link names, only the boundary port it names, of its rows' `per` a replay block, each slot only its unit's row of its
    cell's replay block's (`_linked_row`), or, where the interior docket registers a value of the link's name,
    one the boundary docket's record has no port or value of (`names_ok`), only that value, of its rows' `per` a
    sampled block, each slot only its unit's row of its cell's (`_wired_row`); a port a wire names, only its writer's output port of the interior record, of
    the writer's units a block, each slot only its unit's writer unit's row in its cell's replay block (`_wired_row`);
    any other, its own port of the interior record, each slot only its own first row, or a value either docket
    registers, never a boundary port. Each at its root, domain and leaves, at most `MAX_LEAVES` of them, and each
    slot's row the one the verifier holds (`_at_pos`: C's, in Lean's run of record)."""
    v, q, ep = _get(e, "value"), len(own.sample), _at_pos(own, p.template, port, e)
    if not isinstance(v, str):
        return False
    link = _link_of(own, p.template, port)
    if link is not None:
        if v != link.value:
            return False
        x, r = _registered_entry(own.interior, v), _rows_of(own, p.template, port)
        if x is not None:
            ir = r
            return (ir is not None and _same_at(e, x)
                    and _placed_ok(q, p, ep, q * ir.per, lambda u: _wired_row(p, ir, u)))
        b, m = _by_port(own.boundary).get(v), _get(own.boundary, "population")
        if not (b is not None and _nat(m) and r is not None):
            return False
        lr = r
        return _same_at(e, b) and _placed_ok(q, p, _at_pos(own, p.template, port, e, _linked_rows(own, p, lr, e)),
                                             m * lr.per, lambda u: _linked_row(own, p, lr, u))
    wire = _wire_of(own, p.template, port)
    if wire is not None:
        x, pw, r = _by_port(own.interior).get(v), _writer_of(own, wire), _rows_of(own, p.template, port)
        if not (v == _port_of(named, wire.writer, wire.wport) and x is not None and pw is not None and r is not None):
            return False
        wr = r
        return (_same_at(e, x) and pw.n == wr.per
                and _placed_ok(q, p, ep, q * wr.per, lambda u: _wired_row(p, wr, u)))
    x = _by_port(own.interior).get(v)
    if x is not None:
        return (v == _port_of(named, p.template, port) and _same_at(e, x)
                and _rows_ok(q, p, ep, q * p.n, lambda u: u - p.lo))
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
    other port reads a value a docket registers, which ``second-read`` holds only to the value's root, domain and
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
    the reference's is, and each value a docket registers read at the reference's rows; False for a template the
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


def _sampled_codes(own: Own, named: bool, k: int, p: Part, s: Stmt) -> list[str]:
    """`SecondStage.sampledCodes`: `_stmt_codes` for a statement of only its part `p`'s sampled proof blocks
    (`sampled_only`): its population is the part's slots the second stage samples (`_sampled_ok`), its header its
    template's reference's placed at them (`_sampled_header_ok`), and its session's sample every unit of it
    (`whole_sample`); every other check is `_stmt_codes`'s."""
    q, reads = len(own.sample), _entries(_get(s.registered, "ports"))
    slots = sampled_slots(q, p, own.drawn)
    checks = (("second-verified", _verified_ok(own, s)),
              ("second-tables", _tables_ok(own, s)),
              ("second-header", _sampled_header_ok(own, named, k, s.template, s.header, slots,
                                                   [(port, _at_pos(own, p.template, port, e)) for port, e in reads])),
              ("second-population", _sampled_ok(q, p, own.drawn, s.header)),
              ("second-roots", _roots_ok(named, own.interior, s.template, s.header, [name for name, _ in reads])),
              ("second-sample", _jeq(s.draw, whole_sample(len(slots)))),
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
    docket registers: at a port a link names, of the link's value where the interior registers it (never the
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
    """`SecondStage.namesOk`: no value the interior docket registers (its ``registered`` entries' values) is of a
    name the boundary docket's record has, a port (`_by_port`) or a registered value (`_registered_entry`), so
    each name a session reads is one docket's."""
    ports, es = _by_port(own.boundary), _get(own.interior, "registered")
    return not any(isinstance(v := _get(e, "value"), str)
                   and (v in ports or _registered_entry(own.boundary, v) is not None)
                   for e in (es if isinstance(es, list) else ()))


def returns_ok(own: Own) -> bool:
    """`SecondStage.returnsOk`: no value the interior docket registers (its ``registered`` entries' values) is of
    a name the replay block's return has in the boundary docket's record (`Own.returns`, `replay_returns`), so a
    port a link names for the return reads no value of the interior's (`_read_ok`): only the port the boundary
    docket's record has of that name, committed before the first stage's sample."""
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
    table's at every unit, of round 1's record or of the tables the auditor's params pin (``own.tables``), a table of as
    many leaves as the read has rows. `t` is a read of V rows (`_read_size`). Each value has at least one row and as
    many as the other; row ``i`` of each is one ``bytes`` source (`_bytes_of`), and some port of round 1's record
    (`_by_port`) or some pinned table's entry names row ``i`` of `root` as its root and row ``i`` of `domain` as its
    domain id, at V leaves; and the two ports' rows place each unit alike (``own.rows``: one ``per`` and the same runs).
    So a unit reads the root and the domain id of one table of V leaves that round 1's record registers or the auditor
    pinned before any coin."""
    a, b = ([_bytes_of(r) for r in mine[v]] for v in (root, domain))
    size = _read_size(own, t)
    tables = [*_by_port(own.boundary).values(), *(e for _, e in own.tables)]
    at = [None if (r := _rows_of(own, t, p)) is None else (r.per, r.runs) for p in OPENING_TABLE]
    return (size is not None and bool(a) and len(a) == len(b)
            and all(r is not None and d is not None
                    and any(type(_get(e, "root")) is str and type(_get(e, "domain")) is str
                            and _get(e, "root") == r and _get(e, "domain") == d
                            and _nat(n := _get(e, "leaves")) and n == size for e in tables)
                    for r, d in zip(a, b))
            and at[0] is not None and at[0] == at[1])


def _opened_row(own: Own, registered: set[str], mine: Mapping[str, Any], x: Link) -> bool:
    """`SecondStage.openedRow`: link `x` names the opened row of a read against a root round 1 committed. Its template
    is a read (`_read_size`) and its port the row, its salt or its path (`OPENING_ROW`); the read's root and domain
    (`OPENING_TABLE`) are each linked, at values the verifier fills from its own bytes (`mine`), which are one table's
    of round 1's record or of the tables its params pin at every unit, of the read's rows (`_one_table`); its key
    (`OPENING_KEY`) is either no wire,
    and linked at values the interior docket doesn't register (`registered`: a port of round 1's record then, by
    ``second-links``) or the verifier holds (``own.fixed``, `mine`), or, where the wired key's switch is on
    (`Own.wired_key`), one wire, whole (`_key_failure`), the key at each sampled proof block its writer unit's committed output
    row (``second-read``'s `_read_ok`); and its verdict (`OPENING_VERDICT`) reads, in its template's reference
    statement's reads (`Own.ref_reads`, the template's first), a value the verifier fills, of at least one row, each row
    `VERDICT_ONE_LEAF`. With the switch off, a key the block computes, a wire, is refused: no statement covers it
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
            and ((bool(at(OPENING_KEY)) and all(map(held, at(OPENING_KEY)))) if _wire_of(own, t, OPENING_KEY) is None
                 else own.wired_key and _key_failure(own, t) is None)
            and isinstance(verdict, str) and bool(mine.get(verdict)) and all(r == VERDICT_ONE_LEAF for r in mine[verdict]))


def _read_unlinked(own: Own, v: str) -> bool:
    """`SecondStage.readUnlinked`: whether some template's reference statement (``own.ref_reads``) reads value ``v`` at
    a port no link names it at: a port `second-values` then lets every session of the template read ``v`` at, with no
    link's check of it."""
    at = {(x.template, x.port) for x in own.links or () if x.value == v}
    reads = [(t, ports) for t, ref in own.ref_reads if isinstance(ports := _get(ref, "ports"), Mapping)]
    return any((t, p) not in at for t, ports in reads for p, e in ports.items() if _get(e, "value") == v)


def inputs_failure(own: Own) -> str | None:
    """`SecondStage.inputsFailure`: the first value the interior docket registers (its ``registered`` entries'
    values, in the record's order) that a replay block's input may not read, None for none. A value passes where the
    verifier holds it without the prover: one it pinned or holds to an earlier record (``own.fixed``), or one it fills
    from its own bytes (`_own_values`). Any other passes only as the opened row of a read against a root round 1
    committed: at least one link names it, each link that does is such a row's (`_opened_row`), and no template's
    reference statement reads it at a port no link names it at (`_read_unlinked`), so nothing reads the row but its
    opening. So every input of a replay block the prover gives after the first stage's sample is a row a read opens
    under the root of a table round 1's record registers or the auditor's params pin, at a key round 1's record or the verifier fixes, and the
    prover's choice of it is SHA-512's collision: the links at round 1's ports, the wires and the values the verifier
    holds are the block's other inputs, none the interior docket's to choose."""
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
    """`SecondStage.inputsWhy`: why value `v` is refused (`inputs_failure`): where a link names it as the opened row of
    a read whose key is wired, a key the block computes, `KEY_WHY` with the wired key's switch off (`Own.wired_key`),
    and with it on why the key is not whole (`_key_failure`); else `INPUTS_WHY`."""
    x = next((x for x in own.links or () if x.value == v and _read_size(own, x.template) is not None
              and x.port in OPENING_ROW and _wire_of(own, x.template, OPENING_KEY) is not None), None)
    if x is None:
        return INPUTS_WHY
    return (_key_failure(own, x.template) or INPUTS_WHY) if own.wired_key else KEY_WHY


def _interior_failures(own: Own) -> list[tuple[str, Any]]:
    """`SecondStage.interiorFailures`: ``second-interior-order`` unless `interior_order_ok`, then
    ``second-interior-names`` unless `names_ok`, then ``second-return`` unless `returns_ok`, then ``second-schedule``
    where the schedule the boundary docket's record commits is not one of its replay blocks
    (`schedule_failure`), then ``second-inputs`` where the interior docket registers a value a replay block's
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


def _failures(own: Own, ps: Sequence[Part], ss: Sequence[Any]) -> list[tuple[str, Any]]:
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
            codes = _sampled_codes if sampled_only(len(own.sample), p, own.drawn, s.header) else _stmt_codes
            found += [(c, detail) for c in codes(own, named, _pos_of(ps, s.template), p, s)]
    found += _order_failures(ps, ss)
    covered = _templates(ss)
    found += [("second-cover", {"template": p.template}) for p in ps if p.template not in covered]
    if not _same_dict(own.reported or {}, statement_reads(ss)):
        found.append(("second-reported", REPORTED_WHY))
    found += table_rows_failures(own, ss)
    found += link_root_failures(own, ss)
    return found + placing_failures(own, ps)


LINK_ROOT_WHY = ("a port a link names reads the rows round 1 committed, at round 1's root, domain and leaves: a tree "
                 "committed after the first stage's sample (a run input's input/n<w> among them) holds rows nothing "
                 "ties to round 1's")


def _linked_entry(own: Own, v: str) -> Any:
    """`SecondStage.linkedEntry`: the entry round 1 holds for value `v` a link names: the interior docket's registered
    value `v`, else the boundary docket's record's port `v`."""
    x = _registered_entry(own.interior, v)
    return x if x is not None else _by_port(own.boundary).get(v)


def link_root_failures(own: Own, ss: Sequence[Any]) -> list[tuple[str, Any]]:
    """`SecondStage.linkRootFailures`: ``second-link-root`` for each registered read of statements `ss` at a port a link
    names that is not at the root, domain and leaves round 1 holds for the link's value (`_linked_entry`, `_same_at`),
    whatever value the read names; naming the statement, its template and the port."""
    out: list[tuple[str, Any]] = []
    for i, j in enumerate(ss):
        s = _stmt(j)
        if s is None:
            continue
        for port, e in _entries(_get(s.registered, "ports")):
            link = _link_of(own, s.template, port)
            if link is None:
                continue
            x = _linked_entry(own, link.value)
            if x is None or not _same_at(e, x):
                out.append(("second-link-root", {"statement": i, "template": s.template, "port": port,
                                                 "why": LINK_ROOT_WHY}))
    return out


def placing_failures(own: Own, ps: Sequence[Part]) -> list[tuple[str, Any]]:
    """`SecondStage.placingFailures`: for each template of the parts `ps` in their order, once, and each port of its
    reference header's ``frame_v3.schemas`` in key order, ``second-place-none`` where no rule places it and
    ``second-place-two`` where more than one does (`placings`), naming the template and the port."""
    named, out, seen = _by_template(own.interior), [], []
    for p in ps:
        if p.template in seen:
            continue
        seen.append(p.template)
        ref = next(((r,) for x, r in own.refs if x == p.template), None)
        for port, _s in sorted(_entries(_get(_get(ref[0], "frame_v3"), "schemas")) if ref else (), key=lambda x: x[0]):
            n = len(placings(own, named, p.template, port))
            if n != 1:
                out.append(("second-place-none" if n == 0 else "second-place-two",
                            {"template": p.template, "port": port, "why": PLACE_NONE_WHY if n == 0 else PLACE_TWO_WHY}))
    return out


def table_rows_why(own: Own, t: str, port: str, e: Any) -> str | None:
    """`SecondStage.tableRowsWhy`: why template `t`'s registered read at `port`, entry `e`, is refused for the
    ``table_rows`` it lists; None where it lists none. A read whose key is wired lists none at a port of its opening
    (`KEYED_PORTS`): `KEY_ROWS_WHY`. Round 2 reads no ``table_rows`` yet, so every other one listed is refused too
    (`TABLE_ROWS_WHY`), until the change that reads them."""
    if _get(e, "table_rows") is None:
        return None
    return KEY_ROWS_WHY if port in KEYED_PORTS and _wire_of(own, t, OPENING_KEY) is not None else TABLE_ROWS_WHY


def table_rows_failures(own: Own, ss: Sequence[Any]) -> list[tuple[str, Any]]:
    """`SecondStage.tableRowsFailures`: ``second-table-rows`` for each registered read of statements `ss` that lists
    ``table_rows`` (`table_rows_why`), naming the statement, its template, the port and why, ports in key order."""
    out = []
    for i, j in enumerate(ss):
        s = _stmt(j)
        for port, e in sorted(_entries(_get(s.registered, "ports")), key=lambda x: x[0]) if s is not None else ():
            why = table_rows_why(own, s.template, port, e)
            if why is not None:
                out.append(("second-table-rows", {"statement": i, "template": s.template, "port": port, "why": why}))
    return out


def check_twin(own: Own, statements: Sequence[Any] | None) -> Verdict:
    """Round 2's statement check of `statements`: each code once, with the detail it first failed with, the interior record's order first (`SecondStage.checkAny`:
    `SecondStage.check` where every statement is of every slot of its part, and a statement of only its part's sampled
    proof blocks held to `_sampled_codes`)."""
    if own.parts is None:
        found = [*_interior_failures(own), ("second-cells", CELLS_WHY)]
    elif not (_links_ok(own.parts, own) and _wires_ok(own.parts, own)):
        found = [*_interior_failures(own), *([] if _links_ok(own.parts, own) else [("second-links", LINKS_WHY)]),
                 *([] if _wires_ok(own.parts, own) else [("second-wires", WIRES_WHY)])]
    elif statements is None:
        found = [*_interior_failures(own), ("second-statements", STATEMENTS_WHY)]
    else:
        found = _failures(own, own.parts, list(statements))
    c = _Collect()
    for code, what in found:
        c.fail(code, what)
    return c.verdict()


def held_to_lean(python: Sequence[str] | None, lean: Sequence[str] | None) -> list[str]:
    """The codes of round 2's verdict of record, given the Python twin's codes and the Lean's (`Run.run`'s
    `statements` check): the Lean's where the two are the same list, else `DISAGREE` alone; None for either, a run
    that gave no answer, is a disagreement too."""
    same = python is not None and lean is not None and list(python) == list(lean)
    return list(lean) if same and lean is not None else [V.DISAGREE]


def check(own: Own, statements: Sequence[Any] | None, lean: Sequence[str] | None) -> Verdict:
    """Round 2's verdict of record on `statements`: `check_twin`'s held to `lean`, the codes the Lean `Run.run` gave
    its `SecondStage.checkAny` of the same run, failing with `DISAGREE` alone where they differ."""
    v = check_twin(own, statements)
    if held_to_lean(v.codes, lean) == list(v.codes):
        return v
    c = _Collect()
    c.fail(V.DISAGREE, {"lean": None if lean is None else list(lean), "python": list(v.codes)})
    return c.verdict()


def complete(request_definition: bool, records: Sequence[Mapping[str, Any] | None]) -> bool:
    """Whether a two-stage outcome is complete, of record: no Lean says it yet, so this raises
    (`verifier.not_written`); `complete_twin` is its Python twin."""
    return V.not_written("complete", "Spot.SecondStage.complete")


def attach(rec: Mapping[str, Any], v: Verdict, reads: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """The interior docket's audit record `rec` with round 2's check `v` as ``checks.statements``, and the reads
    `reads` that rest on the interior's own values (`interior_reads`) as ``interior_reads`` where there are any. It
    fails where `v` does, as a failed check of its own fails it: not accepted, and no profile, integrity profile or
    incorrect proof blocks."""
    out = {**rec, "checks": {**rec["checks"], "statements": v.to_json()},
           **({"interior_reads": [dict(r) for r in reads]} if reads else {})}
    return out if v.ok else {**out, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None}


__all__ = ["OWN_TREE", "PLACE_NONE_WHY", "PLACE_TWO_WHY", "READ_ROW_FIELDS", "READ_RUN_FIELDS", "REGISTERED", "RETURN_WHY",
           "ROUND1_TREE", "WIRED_KEY", "WRITER_TREE", "Checked", "Held", "Link", "Own", "Part", "Pos",
           "Rows", "Run", "Width", "Wire",
           "at_run", "attach", "blank_reads", "blank_run", "check", "check_twin", "complete", "complete_twin", "held_to_lean",
           "inputs_failure", "interior_order_ok",
           "interior_reads", "interior_root", "link_runs", "links_of",
           "own_draw", "parts", "placing_failures", "placings", "port_binding", "registered_binding", "replay_links",
           "replay_returns", "replay_rows", "replay_widths", "replay_wires", "returns_ok", "row_at", "rows_of",
           "set_name", "statement_reads", "supplied_of", "table_rows_failures", "table_rows_why", "verified_of",
           "widths_of", "wire_runs", "wires_of"]
