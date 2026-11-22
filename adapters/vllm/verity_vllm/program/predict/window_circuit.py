"""I2's served window as one Circuit (`verity.commitments.periods.interface`): a Program a served request, every
unit's output a committed row and every input row wired or registered.

A window class (`WindowClass`, N1's scan class) is the network attester's envelope: ``lanes`` slots at each boundary, one
every ``every`` scan buckets from the window's start, ``slots`` in all, each of at most ``prompt`` prompt ids and ``outputs``
outputs. Slots are lane-major: slot 2j + b at the class of record is boundary b's lane j, anchored at bucket 16b. The circuit
is public, its size and shape included: a served window's circuit is a Program a request (`lower` with ``compiled``), the
Build's greedy request Program at one prompt row and n - 1 decode steps, n = its prompt + outputs - 1 positions, every
position computing the whole stack, the LM head and the token select, position p attending to keys 0..p; a lane with no
request has no Program. The class circuit (`class_circuit`), every slot's Program at the class's P = prompt + outputs - 1
positions, is the fixed load's, at which PoUW's tables are read (`FixedLoad`). Arrival steps are values, never structure.

A slot's request comes from the network attester's scan: boundary b's Programs read the scan's state row before it
(`scan_state.SCAN_STATE`, registered) through `boolean_io_shell.ScanSlots_v5` (`scan_state.bridge`), a template a lane,
which gives its lane its first position's token (its prompt's first id) and the slot it carries on, its occupancy word and
its head (its prompt's length and ``max_tokens``). A served window has a unit a lane of each boundary, empty lanes
included; a class circuit's slot holds a unit of every lane's template and reads its own lane's, so its slots lower the
same units. Every later position's token is `io_shell.TokenFeed_v1{prompt, FIRST=False}`'s below the class's
prompt limit (the prompt's id below its length, else the previous position's select), and past it the previous position's
select. Each select is collected in the unit that makes it (`io_shell.SelectCollect_v1`: the select, then
`io_shell.OutCollect_v1` over it). The last position's makes the slot's entry of the scan's Programs' row instead
(`io_shell.SelectEntry_v1`: the select, its collector, then `io_shell.SlotEntry_v1`), in the scan's layout: its read bit
(the occupancy word's bit 0), its output count (up to ``max_tokens`` and the model's first stop id, that one kept,
`stop_ids`) as a u32, then its ids cut to the scan's 17 bits, all zero for an empty slot (the entry reads the occupancy
word, so no select is masked on the way); a request the class can't hold (a prompt
past ``prompt`` ids, or outputs that don't end within ``outputs``) has ``outputs`` + 1 and no ids, which the scan refuses at
the anchor and the egress never renders. A served request's Program is built at the prompt length L its advice gives: each
position below L feeds its prompt's id (`io_shell.TokenFeed_v1{FIRST=True}`, no select), and a prompt position before the
last builds nothing a later token doesn't read (its LM head, select and final norm, and its last layer's query and that
layer from its attention on), its select's collector `io_shell.OutCollect_v1` alone. Its last select is collected like the
others, then ``outputs`` - O pads shift its O outputs to the front, and its entry is `io_shell.SlotEntry_v1{NP=1}`, which
fits only where the scan's prompt length is L: an advice of another prompt length gives ``outputs`` + 1. The entries are
one allocation after the walk, in slot order, whatever the Programs' order: entry j is lane j's, the slot the scan's
Programs' row reads it as. A Program past the
positions a key block holds at the model and grain is refused at the first position past them, naming that bound. A
collector's rows and an entry's are a few words at a small class, too narrow for a proof block of their own, so none is a
unit of its own. The state rows each lane's unit reads and the entries are the circuit's external ports.
A slot's Program
hands all its outputs to the scan at its anchor, which carries them across windows (`SC_GROUP`), so no Program runs into the
next window. `window_circuit(registered, compiled, cls, grain, placed=...)` refuses a window past the class's envelope and
emits its circuit. Neither C's shape nor the advice holds the bucket a request arrived in, so a served Program reads boundary
0's state row, and a served window with a request at another boundary, or compiled from the advice at a class of more than
one, is refused by name until a block checks the bucket as a committed value.

`lower` walks the longest Program's nodes in order and lowers each to units at ``grain``, once a slot that has it (the
Programs longest first, each a prefix of the longest): node j's units of slot 0, then of slot 1 and so on, each
allocation's slots side by side in its template (`window_builder`, which holds
the units, their feeds and weight reads, and wires them). Every unit reads its writers' rows as they hold them:

- `Embedding_v1{V, H}` is the registered read `EmbeddingRead_v4{V, H}` (`commitments.gates.embedding`): its ``root`` and
  ``domain`` are the embedding table's (a `Table` over the weights record, read by `TableRead`), its ``row``, ``salt`` and
  ``path`` a `TableOpening` at the hidden key ``tok``, pinned by its verdict ``o1``. Its ``tok`` is its position's feed.
- `Gemm_v2{K, N}` is ⌈N / n⌉ slices `Gemm_v2{K, n}`, each reading its weight rows from the weights record (`Read`) and
  its ``x`` as its writers' rows, n = ``grain.gemm_words`` // K columns, halved while an earlier node reads that slice
  template's ``x`` at another row count (a port is cut one way, and o_proj's ``x`` is a row a head).
- An int8 linear (`registry.int8_exact`) is its quantizer `Int8RowQuant_v1{K}`, one unit `Int8RowQuant_v3{K, R}` (v1 with
  ``x`` as R rows of K / R, R its writers' rows: one after a norm, a head's at o_proj; v5 at the K of H1's tables, below),
  and its product `ScaledMmS8_v1{K, N}`, ⌈N / n⌉ slices `ScaledMmS8_v3{K, n}` at the same weight words a slice (n =
  ``grain.gemm_words`` // (K / 2): an s8 weight row is K / 2 words), each reading its s8 weight rows and their f32 row scales
  as the weights record's u16 words, and the quantizer's codes ``x`` and scale ``sx`` whole. Words are laid out by each
  field's leaf width (`_spans`): a weight or node output whose leaves aren't one byte width filling whole u16 words, and a
  read that starts inside a word, are refused.
- `RoPE_v1{NHEADS, D}` is its heads (`RoPEHead_v1{D}`), each reading its slice of ``x``.
- `Attention_v18{T, NH, KVH, D, BN, ...}` is `Attention_v19`'s cut: each head's Calls (`AttentionHead_v19`'s
  body, read once a T) are its units, its first visited key block `AttnBlockFirst_v1{D, NVIS}` (keys (nb - 1)·BN ... T - 1,
  nb = ⌈T / BN⌉; `AttnBlockFirst_v2{..., VS}` where a key's value is rows of VS words, the value GEMM's slices), each full
  block below it `AttnBlock_v5` and its end `AttnHeadEnd_v1{D}`, one unit a head a Call, each
  block's running state its predecessor's rows. A block's keys and values a KV group are its keys' rows as their
  writers hold them (a key's RoPE head's row, a value's GEMM slices), key by key, which its group's heads read. A layer's
  cache is refused unless it is the keys its earlier positions' nodes read and one more.
- FA3's `Attention_v4` (sm_90) is `Attention_v20`'s cut the same way: its first visited block
  `AttnBlockFirst_v1` (FA3's first iteration is FA2's word for word), each full block below it
  `AttnBlock_v4{FIRST=False, CHECK = b >= MASKED_FROM}` and its end `AttnHeadEnd_v1{D, INV=Fa3InvSum_v1}`. `CUTS` names
  the versions cut by key block and the cut of each.
- Any other attention version (`Attention_v2`, as `fa3_construction = "guard-every-block"` binds) is its heads, a unit
  each of its own head Definition (`AttentionHead_v2{T}`), reading its KV group's every key's rows.
- `TokenSelect_v4{V}` is its function at groups of `GROUP` logits: V / C units `TokenSelectGroups_v1{C, GROUP}`, C =
  ``grain.select``, each reading its C logits as the LM head's rows and writing its groups' bests as a row and their indices
  as another, then `TokenSelectCombine_v2{V, GROUP}` over those rows and ``x0`` (``logits[0]``), from one `Gemm_v2{K, 1}`
  unit over the LM head's input and its first weight row, as the position's `SelectCollect_v1{V, GROUP, KO, FIRST}` (at the
  last, `SelectEntry_v1`). A V that isn't whole units of whole groups is refused.
- `SiluMul_v2{I}` is P units `SiluMul_v2{I / P}`, each reading its gate and up elements as the gate-up GEMM's rows, P the
  fewest that leaves a unit within ``grain.rows`` rows; `RMSNormTriton_v5` is one unit.
- `RMSNormFusedCuda_v2{N, EPS}` is one unit, or where an earlier node reads its template's ``x`` at another row count or one
  unit would pass ``grain.rows`` rows (down_proj's norm at Llama's grain), it is cut at its residual add: N / C units
  `RMSNormFusedCudaAdd_v2{N, C, AT, R}`, C = ``grain.norm``, part AT reading x's elements [AT·C, (AT+1)·C) as their writers'
  R rows (so two norms whose x are written in other row counts cut into templates of their own) and ``res`` whole, then `RMSNormFusedCudaTail_v1{N, EPS}` reading their sums ``r`` in order of AT (its row j part
  AT = j's, which the wiring check holds and the tail's equality to v2 rests on) and ``w``. The tail's second output is r,
  so v2's readers read it as they read v2's.

The weights record is never one registered parameter: each weight a port reads is its own at the port's row width
(`window_builder.weight_param`: ``weights/<label>@<words>``), its reads' bases and steps whole rows of it, so each is pinned
as a root over its own rows, by its own name, and the window's members take its rows as registered rows
(`interface.members`). An embedding's table is a parameter of its own (`window_builder.table_param`:
``weights/<label>@<words>/table``, its rows ``row-seg`` leaves), so a tied embedding is registered three times: as its
table, as the LM head's rows and as ``x0``'s rows at the table's width. Nothing in the circuit ties their roots to one
another or to the checkpoint: the verifier pins each by name, its root and domain id (`interface.registered_reads`), at the
committer's registration (`window_weights`).

Given H1's tables of the window (`Tables`, `window_class`), the circuit reads each where the model makes it, a read of an
output row (`interface.Read`) that holds the proof block's output row to the registered row: each quantizer at the
tables' K is `Int8RowQuant_v5{K, R, ROUND}`, its ``o2`` a row of its round's A table, and each position's token row a row of
its round's input-ids table, at a fixed load's rows (`Tables.at`). Each reader reads one table, and each table's readers are
templates of their own: round s's quantizers bind ROUND = s, round 0's token rows are its `ScanSlots_v5` and `TokenFeed_v1`
units' and a later round's the select before it, `io_shell.SelectCollect_v2{..., ROUND}`. So a sample stratified by
template (one stratum a template) has a reader of every table.

Given Pearl+'s tables (`PearlTables`, `window_class`), the circuit is cut at its linears, whose products PoUW credits
(`window_cut`): each linear but an LM head is a unit held to its call's A rows and a unit reading its call's y.

A reader port that reads m writer rows is cut m ways (`interface.Layout.cut`, `window_builder._Builder.settle`), a row of
it a writer row, so no unit copies rows together: a template's port is read at one row count, its rows its outer elements
cut that many ways. Every wire and read is stated per node as a pattern (a node's heads, KV groups and slices as digits of
one) and `interface.coalesce`d, and each allocation's slots sit side by side in its template, so the slots' copies of a
pattern are one and the circuit's description grows with the class Program's nodes, not its slots or units; so does the
lowering's own state (a node's output rows are runs). A template's units are read at every step, so its wires have cycles
between templates: the circuit states an order of its allocations, the allocations each wire reads in an allocation before
it (`interface.ordered`, the circuit's ``order``).

The circuit's blocks (`interface.Blocks`) carry its inference block: a unit's slot is its slot (its class Program; the blocks'
slots are the class's S) and its step its position, T the class's positions. An embedding opens its position, the stack, LM head
and token select with its collector after it are that position's (the last's with the entry), and a lane's `ScanSlots_v5`
unit its Program's first position's (an empty lane's, the slot's that holds its collector). An allocation's units are one entry, its slots' runs at one step side by side (the slot
digit, then the run), and like entries that follow on merge, at one step and then at consecutive steps
(`window_builder.inference`).

Refused by name (`Unexpressible`): a window past its class's slots (more requests than slots, more anchored at a boundary than
its lanes, an anchor that is no boundary or past boundary 0, a prompt past the limit, more outputs than the limit), with H1's
tables a window that isn't the fixed load, a class of more than one boundary or whose round 0 passes a table's rows a call, and a position whose
int8 linears of the tables' K aren't their calls a round, with Pearl+'s what `window_cut` refuses, a request whose Program
would read more than its prompt and the weights (a stochastic request's seed, a prefix-cache hit's prefix, a TP rank's
peers), a request a Commit window carries in or out, byte ingress, a family this lowering doesn't know, a weight whose
leaves aren't whole u16 words, a read that cuts a writer's row, and at `settle`, every one named at once, a template's port
read at two row counts or as rows that aren't its outer elements cut that many ways and a unit of more than ``grain.rows``
rows. A unit Flock's composer wouldn't prove is `check_provable`'s to refuse, after lowering; a served window's,
which the door proves, `check_served`'s. The circuit is completeness only:
that it computes what serving ran is held by the units' proofs on their committed rows, never by this lowering.

Its weight reads, the embedding's table reads and its openings count registered (`interface.input_count`) only where the
verifier pinned that weight's own parameter, ``weights/<label>@<words>``, before the run, at the root the window's
registration records for it; anywhere else they are free. The pins are the hiding roots the committer registered once per
model and key, never roots the verifier computes (`window_weights`). Its scan state reads are held otherwise: the window
commits the scan's state rows it reads as `scan_state.SCAN_STATE` (`ingress_scan.window_state`, `window_rows.state_value`)
and the scan's own state ports read that root under its domain, so the scan's chain and the window's reads are one tree.
No pin holds them, so `interface.input_count` counts them free.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.periods import interface as IF

from verity_vllm.definitions.registry import io_shell as IO
from verity_vllm.program.frontend.stop_rule import stop_ids as of_config

from . import scan_state as SS
from .build import Unexpressible
from .steps import RequestUnit, advice_of, compile_steps, request_program
from .window_builder import WEIGHTS, Grain, Row, _Builder, check_provable, check_served, inference
from .window_class import (
    CLASS_OF_RECORD,
    GREEDY,
    FixedLoad,
    PearlTables,
    Request,
    Tables,
    WindowClass,
    requests_of,
    takes,
)
from .window_cut import ROW_FORMAT, cut_values, linears
from .window_lowering import CUTS, FAMILIES, _Program

NAME = "verity-vllm/window"


_GRAIN = Grain()


def class_program(registered: Any, n: int) -> Any:
    """The request Program of ``n`` positions (a `build.BuildResult`): the Build's greedy request at one prompt row and
    ``n`` - 1 decode steps, in the registered compiler's form."""
    return request_program(registered, RequestUnit(-1, "", 1, n - 1, GREEDY, None, 0))


def stop_ids(registered: Any) -> tuple[int, ...]:
    """The registered model's stop ids (`frontend.stop_rule.stop_ids`): outputs end at the first, kept."""
    return of_config(registered.phi.hf_config)


@dataclass(frozen=True)
class Lowered:
    """A window's circuit with what its lowering knows (`window_rows` fills its units from it): the longest Program's
    function, each slot's lowered Program (node j's output as the rows that hold it, ``outs[j]``), each template's runs of
    units by their (slot, step), ``(at, count, slot, step)``, and in a served window's, each slot's request's index among
    the window's requests (`window_class.takes`), none in a class circuit's; each empty slot of a served window,
    ``(slot, the Program whose slot holds its units, its collector unit, its entry unit)`` (`_Program.place_empty`);
    each `ScanSlots_v5` unit, ``(anchor, lane, unit)``; and in a served window's, the templates `check_served` names."""
    circuit: IF.Circuit
    fn: Any
    programs: tuple[_Program, ...]
    spans: dict[str, list[tuple[int, int, int, int]]]
    requests: tuple[int, ...] | None = None
    empty: tuple[tuple[int, int, tuple[str, int], tuple[str, int]], ...] = ()
    scans: tuple[tuple[int, int, tuple[str, int]], ...] = ()
    unprovable: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def cut(self) -> frozenset[int]:
        """The class Program's nodes the circuit takes as registered, cut at its linears (`PearlTables`); none uncut."""
        return frozenset(self.programs[0].arows) if self.programs else frozenset()


def class_circuit(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN,
                  tables: FixedLoad | None = None, requests: Sequence[Request] | None = None) -> IF.Circuit:
    """``cls``'s circuit (module docstring): the class Program lowered once a slot, node by node, slot k reading its lane of the
    scan state row before its anchor (`WindowClass.anchor`), each allocation's slots side by side in its template; its blocks'
    slots are the slots. Its registered parameters are the weights (`window_builder.weight_param`) and ``SCAN_STATE``'s
    ``cls.buckets`` rows, row a the scan's state row before bucket a (`ingress_scan.window_state`), and with ``tables`` H1's
    tables (`Tables`) or Pearl+'s (`PearlTables`). A window's circuit is every slot's; ``programs`` is the first slots only, a probe's census
    (`class_size`). With ``requests``, the served window's circuit instead (`lower`)."""
    return lower(registered, cls, programs, grain, tables=tables, requests=requests).circuit


def lower(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN,
          tables: FixedLoad | None = None, requests: Sequence[Request] | None = None) -> Lowered:
    """`class_circuit`'s circuit with its lowering (`Lowered`). With ``tables``,
    refused unless the class is one boundary; with H1's, unless round 0's rows (each slot's prompt) fit a table's ``m`` rows
    a call and every position has ``tables.calls`` int8 linears of K = ``tables.k``; with Pearl+'s, unless every position's
    linears are the first's (`window_cut.linears`). With ``requests`` (`window_class.takes`, `window_class.requests_of`),
    the served window's circuit: a Program a request of the request's own positions, longest first, reading its lane at
    its anchor, so it holds exactly the requests' positions, and one `ScanSlots_v5` unit a lane of each boundary, one
    allocation each, and one entry a slot of the class in slot order, an empty lane's units in the slot of the first
    Program at its boundary (`_Program.place_empty`), or of the first Program at a boundary no request is anchored at;
    refused with ``tables`` or ``programs``, or for a window of no request."""
    if requests is not None and (tables is not None or programs is not None):
        raise Unexpressible("a served window's circuit is every request's Program, reading no tables")
    if requests is not None and not requests:
        raise Unexpressible("a window of no request has no circuit")
    order = sorted(range(len(requests)), key=lambda i: -requests[i].positions) if requests is not None else None
    programs = len(order) if order is not None else cls.slots if programs is None else programs
    if not 1 <= programs <= cls.slots:
        raise Unexpressible(f"{programs} Programs in a window of a class of {cls.slots} slots")
    if isinstance(tables, Tables) and (len(cls.boundaries) != 1 or cls.slots * cls.prompt > tables.m):
        raise Unexpressible(f"{cls}: H1's tables {tables} are read at a fixed load's rows, every slot at the window's one "
                            f"boundary and round 0 its {cls.slots} slots' prompts of {cls.prompt}, at most {tables.m} rows")
    if isinstance(tables, PearlTables) and len(cls.boundaries) != 1:
        raise Unexpressible(f"{cls}: Pearl+'s tables are read at a fixed load's rows, every slot at the window's one boundary")
    try:
        scans = SS.bridge(cls.scan, cls.prompt)
    except ValueError as e:
        raise Unexpressible(f"{cls}: its slots aren't a scan state row's lanes ({e})") from None
    entry = bind(IO.SlotEntryV2, KO=cls.outputs, IDB=cls.scan.IDB, NP=1 if requests is not None else cls.prompt,
                 STOP=stop_ids(registered))
    b = _Builder(grain, programs)
    if order is None:
        fn = class_program(registered, cls.positions).program.fn
        layout = linears(fn, tables.words, grain.rows) if isinstance(tables, PearlTables) else None
        progs = [_Program(b, fn, cls, f"class Program {k}", k, scans[cls.lane(k)], entry, tables, layout)
                 for k in range(programs)]
        fns = [fn] * programs
    else:
        assert requests is not None
        layout, built = None, {n: class_program(registered, n).program.fn for n in {r.positions for r in requests}}
        fns = [built[requests[i].positions] for i in order]
        fn = fns[0]
        progs = [_Program(b, f, cls, f"request {i}'s Program", k, scans[requests[i].lane], entry, at=requests[i])
                 for k, (i, f) in enumerate(zip(order, fns))]
        ids = [n.fn.id for n in fn.body.nodes]
        for i, f in zip(order, fns):
            if [n.fn.id for n in f.body.nodes] != ids[:len(f.body.nodes)]:
                raise Unexpressible(f"request {i}'s Program of {requests[i].positions} positions is no prefix of the "
                                    f"longest's, {requests[order[0]].positions}")
    by = {(r.anchor, r.lane): k for k, r in enumerate(progs)}
    host = {a: min([(r.lane, k) for k, r in enumerate(progs) if r.anchor == a], default=(0, 0))[1] for a in cls.boundaries}
    scanned: list[tuple[int, int, tuple[str, int]]] = []
    for j in range(cls.lanes):
        groups = ([[(cls.anchor(k), k) for k in range(programs)]] if order is None
                  else [[(a, by.get((a, j), host[a]))] for a in cls.boundaries])
        for group in groups:
            tid, u = b.allocation(scans[j], [(k, 0) for _, k in group])
            for i, (a, k) in enumerate(group):
                b.read(tid, "state", u + i, (1,), a * scans[j].bindings["NS"], (0,), SS.SCAN_STATE)
                scanned.append((a, j, (tid, u + i)))
                if order is None and cls.lane(k) == j:
                    progs[k].scan_unit = (tid, u + i)
    lanes = {(a, j): unit for a, j, unit in scanned}
    for r in progs if order is not None else ():
        r.scan_unit = lanes[(r.anchor, r.lane)]
    ends = [len(f.body.nodes) for f in fns]
    for j, node in enumerate(fn.body.nodes):
        active = sum(1 for e in ends if e > j)
        kinds = [r.kind(j, f.body.nodes[j]) for r, f in zip(progs[:active], fns)]
        lo = 0
        for k, (r, f) in enumerate(zip(progs[:active], fns)):
            if kinds[k] != kinds[lo]:
                lo = k
            hi = next((i for i in range(k, active) if kinds[i] != kinds[k]), active)
            b.among = (lo, hi)
            r.lower(j, f.body.nodes[j])
    b.among = None
    empty: list[tuple[int, int, tuple[str, int], tuple[str, int]]] = []
    if order is not None:
        filled = [by.get((cls.anchor(s), cls.lane(s))) for s in range(cls.slots)]
        gaps = [s for s, k in enumerate(filled) if k is None]
        if gaps:
            ctid, cu = b.allocation(bind(IO.OutCollectV2, KO=cls.outputs, FIRST=True),
                                    [progs[host[cls.anchor(s)]].entry_label for s in gaps])
        etid, eu = b.allocation(entry, [progs[k if k is not None else host[cls.anchor(s)]].entry_label
                                        for s, k in enumerate(filled)])
        for s, k in enumerate(filled):
            if k is not None:
                progs[k].place_entry(etid, eu + s)
        for i, s in enumerate(gaps):
            k = host[cls.anchor(s)]
            progs[k].place_empty(lanes[(cls.anchor(s), cls.lane(s))], (ctid, cu + i), (etid, eu + s))
            empty.append((s, k, (ctid, cu + i), (etid, eu + s)))
    for r in progs:
        if r.position != r.positions:
            raise Unexpressible(f"{r.name}: {r.position} positions, not its {r.positions}")
        if isinstance(tables, Tables) and set(r.quants) != {tables.calls}:
            raise Unexpressible(f"{r.name}: int8 linears of K = {tables.k} by position {r.quants}, not H1's {tables.calls} "
                                "calls a round")
        if layout is not None and set(r.linears) != {len(layout.calls)}:
            raise Unexpressible(f"{r.name}: linears cut by position {r.linears}, not the layout's {len(layout.calls)} calls")
    b.settle()
    roots, openings = b.merged()
    params = {**b.params, SS.SCAN_STATE: cls.buckets * scans[0].bindings["NS"]}
    if isinstance(tables, Tables):
        held = tuple(v for s in range(cls.outputs) for v in (tables.a(s), tables.ids(s)))
    elif isinstance(tables, PearlTables) and layout is not None:
        held = cut_values(tables, cls, layout)
    else:
        held = ()
    reg = IF.Registered(dict(sorted(params.items())), tuple(b.tables.values()), tuple(IF.coalesce(b.reads)), tuple(roots),
                        tuple(openings), values=(*held, SS.SCAN_STATE),
                        formats={v: ROW_FORMAT for v in held} if layout is not None else {})
    wires = tuple(IF.coalesce(b.wires if order is None else b.allocation_wires()))
    entry_units = {r.entry_unit[0] for r in progs if r.entry_unit is not None}
    if len(entry_units) != 1 or any(r.entry_unit is None for r in progs):
        raise Unexpressible(f"the Programs' entries are units of {sorted(entry_units)}, not each one of one template")
    circuit = IF.Circuit(NAME, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()),
                         external=(*(f"{descriptor_id(s)}/state" for s in scans if descriptor_id(s) in b.units),
                                   f"{entry_units.pop()}/o0"),
                         wires=wires, cuts=b.cuts, registered=reg, order=IF.ordered(b.units, wires, b.starts) if wires else None,
                         blocks=IF.Blocks(programs, max(r.positions for r in progs), tuple(inference(b.spans, programs))))
    return Lowered(circuit, fn, tuple(progs), b.spans, None if order is None else tuple(order), tuple(empty),
                   tuple(sorted(scanned)))


@dataclass(frozen=True)
class WindowCompiler:
    """The registered window compiler, ``compile(advice) -> C``: its one input is the advice, a bit string only it
    interprets (its code: `steps.advice_of`), of at most ℓ = ``registered.cap_bits`` bits, refused past it by name; the
    model (``registered``), the envelope (``cls``) and the ``grain`` are the compiler's own. C is a greedy Program a request
    of the advice's prompt and outputs (`window_class.requests_of`, `lower`); the certified bodies and the scan state are
    C's inputs, never the compiler's. A request's sampling is its body's, never the advice's, so no advice says a sampled
    request: `takes` refuses one where the bodies are read, and a Commit window's advice refuses one where it is made
    (`advice.from_trace`)."""
    registered: Any
    cls: WindowClass
    grain: Grain = _GRAIN

    def lower(self, advice: bytes) -> Lowered:
        """`__call__`'s circuit with its lowering (`Lowered`)."""
        return self._lower(advice)[1]

    def __call__(self, advice: bytes) -> IF.Circuit:
        return self.lower(advice).circuit

    def schedule(self, advice: bytes) -> dict[tuple[str, int], Scheduled]:
        """Each credited call of ``advice``'s circuit, a linear's run of units keyed by its template and first unit, at the
        engine step whose row computes its position and its place among that step's rows' positions, in the advice's
        rows by step (`Scheduled`); refused exactly where `__call__` is."""
        decoded, low = self._lower(advice)
        at: dict[tuple[int, int], tuple[int, int]] = {}
        for s, st in enumerate(decoded.steps):
            place = 0
            for r in st.rows:
                for p in range(r.start, r.start + r.n):
                    at.setdefault((r.request, p), (s, place))
                    place += 1
        assert low.requests is not None
        out = {}
        for k, prog in enumerate(low.programs):
            for p, tid, u, count in prog.calls:
                out[(tid, u)] = Scheduled(count, *at[(low.requests[k], p)])
        return dict(sorted(out.items()))

    def _lower(self, advice: bytes) -> tuple[Any, Lowered]:
        if type(advice) is not bytes:
            raise Unexpressible(f"the advice is a bit string, bytes, not a {type(advice).__name__}")
        decoded = advice_of(self.registered, advice)
        return decoded, lower(self.registered, self.cls, None, self.grain, requests=requests_of(decoded, self.cls))


class Scheduled(NamedTuple):
    """A credited call's run of ``units``, at engine step ``step``, the ``place``-th position its rows compute."""
    units: int
    step: int
    place: int


def served_lowering(registered: Any, cls: WindowClass, advice: Any, ingress: Any, grain: Grain = _GRAIN,
                    placed: Sequence[tuple[int, int | None]] | None = None, named: bool = False) -> tuple[Any, Lowered]:
    """A served window compiled and its circuit lowered, the one call the serve's close and the window commit make: I2's
    Programs of ``advice`` over the bodies ``ingress`` brought (`steps.compile_steps`, with each Program's names under
    ``named``), their requests within ``cls`` (`takes`), the circuit (`lower`), refused (`Unexpressible`) where the door
    wouldn't take a template, and with each template the door cannot prove named by reason in ``unprovable`` (`check_served`). ->
    the compile (its requests' ids and arrivals, which place the slots) and the lowering."""
    compiled = compile_steps(registered, advice, ingress, named=named)
    low = lower(registered, cls, None, grain, requests=takes(compiled, cls, placed))
    return compiled, replace(low, unprovable=check_served(low.circuit))


def window_circuit(registered: Any, compiled: Any, cls: WindowClass, grain: Grain = _GRAIN,
                   placed: Sequence[tuple[int, int | None]] | None = None,
                   tables: FixedLoad | None = None) -> IF.Circuit:
    """``compiled``'s window (`steps.Compiled`, of ``registered``) as its circuit, refused where it is past ``cls``'s envelope
    (`takes`): a Program a request, each of the request's own positions (`lower`). ``placed`` is each request's anchor bucket
    and the bucket of its last output, in its request order, as the network attester's scan placed it (the end None:
    unchecked), each anchor boundary 0 (`window_class._placed`); by default every request is there, its end unchecked. A
    last output past the window's end is the scan's to carry (module docstring). With ``tables``, H1's tables read
    (`Tables`) or the circuit cut at its linears (`PearlTables`), ``cls``'s class circuit, refused unless the window is the
    fixed load."""
    if tables is None:
        return class_circuit(registered, cls, None, grain, requests=takes(compiled, cls, placed))
    takes(compiled, cls, placed)
    tables.held(compiled, cls)
    return class_circuit(registered, cls, None, grain, tables)


__all__ = ["CLASS_OF_RECORD", "CUTS", "FAMILIES", "Grain", "Lowered", "NAME", "PearlTables", "Request", "Row", "Scheduled", "Tables",
           "WEIGHTS", "WindowClass", "WindowCompiler", "check_provable", "check_served", "class_circuit", "class_program", "lower",
           "requests_of", "served_lowering", "stop_ids",
           "takes", "window_circuit"]
