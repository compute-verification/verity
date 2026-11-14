"""I2's served window as one Circuit (`verity.commitments.windows.interface`): the window *class's* circuit, every
unit's output a committed row and every input row wired or registered.

A window class (`WindowClass`, N1's scan class) has a fixed number of slots a network timestep: ``lanes`` slots at each
boundary, one every ``every`` scan buckets from the window's start, ``slots`` in all, each of at most ``prompt`` prompt ids and
``outputs`` outputs. Slots are lane-major: slot 2j + b at the class of record is boundary b's lane j, anchored at bucket 16b.
Its circuit is one class Program a slot, whatever the traffic (`class_circuit`): the Build's greedy request Program at one
prompt row and P - 1 decode steps, P = prompt + outputs - 1 positions, every position computing the whole stack, the LM head and
the token select, position p attending to keys 0..p. So the circuit is a function of the class alone: which slots carry a
request, arrival steps, prompt lengths, output lengths and key counts are values, never structure.

A slot's request comes from the network certifier's scan: boundary b's Programs read the scan's state row before it
(`scan_state.SCAN_STATE`, registered) through `boolean_io_shell.ScanSlots_v2` (`scan_state.bridge`), which gives each lane
its first position's token (its prompt's first id) and the slot it carries on, its occupancy word and its head (its prompt's
length and ``max_tokens``). Every later position's token is `io_shell.TokenFeed_v1{prompt, FIRST=False}`'s below the class's
prompt limit (the prompt's id below its length, else the previous position's select), and past it the previous position's
select. Each select is collected in the unit that makes it (`io_shell.SelectCollect_v1`: the select, then
`io_shell.OutCollect_v1` over it). The last position's makes the slot's entry of the scan's Programs' row instead
(`io_shell.SelectEntry_v1`: the select, its collector, then `io_shell.SlotEntry_v1`), in the scan's layout: its read bit
(the occupancy word's bit 0), its output count (up to ``max_tokens`` and the model's first stop id, that one kept,
`stop_ids`) as a u32, then its ids cut to the scan's 17 bits, all zero for an empty slot (the entry reads the occupancy
word, so no select is masked on the way); a request the class can't hold (a prompt
past ``prompt`` ids, or outputs that don't end within ``outputs``) has ``outputs`` + 1 and no ids, which the scan refuses at
the anchor and the egress never renders. A collector's rows and an entry's are a few words at a small class, too narrow for
a proof block of their own, so none is a unit of its own. The state rows and the entries are the circuit's external ports.
A slot's Program
hands all its outputs to the scan at its anchor, which carries them across windows (`SC_GROUP`), so no Program runs into the
next window. `window_circuit(registered, compiled, cls, grain, placed=...)` refuses a window the class's slots don't take and
emits its circuit.

`class_circuit` walks the class Program's nodes in order and lowers each to units at ``grain``, once a slot: node j's units
of slot 0, then of slot 1 and so on, each allocation's slots side by side in its template (`window_builder`, which holds
the units, their feeds and weight reads, and wires them). Every unit reads its writers' rows as they hold them:

- `Embedding_v1{V, H}` is the registered read `EmbeddingRead_v1{V, H}` (`commitments.gates.embedding`): its ``root`` and
  ``domain`` are the embedding table's (a `Table` over the weights record, read by `TableRead`), its ``row``, ``salt`` and
  ``path`` a `TableOpening` at the hidden key ``tok``, pinned by its verdict ``o1``. Its ``tok`` is its position's feed.
- `Gemm_v2{K, N}` is ⌈N / n⌉ slices `Gemm_v2{K, n}`, each reading its weight rows from the weights record (`Read`) and
  its ``x`` as its writers' rows, n = ``grain.gemm_words`` // K columns, halved while an earlier node reads that slice
  template's ``x`` at another row count (a port is cut one way, and o_proj's ``x`` is a row a head).
- An int8 linear (`registry.int8_exact`) is its quantizer `Int8RowQuant_v1{K}`, one unit `Int8RowQuant_v3{K, R}` (v1 with
  ``x`` as R rows of K / R, R its writers' rows: one after a norm, a head's at o_proj; v5 at the K of H1's tables, below),
  and its product `ScaledMmS8_v1{K, N}`,
  ⌈N / n⌉ slices `ScaledMmS8_v1{K, n}` at the same weight words a slice (n = ``grain.gemm_words`` // (K / 2): an s8 weight row
  is K / 2 words), each reading its s8 weight rows and their f32 row scales from the weights record and the quantizer's codes
  ``x`` and scale ``sx`` whole. Words are laid out by each field's leaf width (`_spans`): a weight or node output whose leaves
  aren't one byte width filling whole u16 words, and a read that starts inside a word, are refused.
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
  `RMSNormFusedCudaAdd_v1{N, C, AT}`, C = ``grain.norm``, part AT reading x's elements [AT·C, (AT+1)·C) as their writers'
  rows and ``res`` whole, then `RMSNormFusedCudaTail_v1{N, EPS}` reading their sums ``r`` in order of AT (its row j part
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
templates of their own: round s's quantizers bind ROUND = s, round 0's token rows are its `ScanSlots_v2` and `TokenFeed_v1`
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
and token select with its collector after it are that position's (the last's with the entry), and a slot's `ScanSlots_v2`
unit its first position's. An allocation's units are one entry, its slots' runs at one step side by side (the slot
digit, then the run), and like entries that follow on merge, at one step and then at consecutive steps
(`window_builder.inference`).

Refused by name (`Unexpressible`): a window past its class's slots (more requests than slots, more anchored at a boundary than
its lanes, an anchor that is no boundary, a prompt past the limit, more outputs than the limit), with H1's tables a window that
isn't the fixed load, a class of more than one boundary or whose round 0 passes a table's rows a call, and a position whose
int8 linears of the tables' K aren't their calls a round, with Pearl+'s what `window_cut` refuses, a request whose Program
would read more than its prompt and the weights (a stochastic request's seed, a prefix-cache hit's prefix, a TP rank's
peers), a request a Commit window carries in or out, byte ingress, a family this lowering doesn't know, a weight whose
leaves aren't whole u16 words, a read that cuts a writer's row, and at `settle`, every one named at once, a template's port
read at two row counts or as rows that aren't its outer elements cut that many ways, a unit of more than ``grain.rows``
rows, and then a unit Flock's composer wouldn't prove (`interface.check_proof_block`). The circuit is completeness only:
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
from dataclasses import dataclass
from itertools import product
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.windows import interface as IF
from verity.definitions.registry import io_shell as IO

from verity_vllm.program.frontend.stop_rule import stop_ids as of_config

from . import scan_state as SS
from .build import Unexpressible
from .steps import RequestUnit, request_program
from .window_builder import WEIGHTS, Grain, Row, _Builder, inference
from .window_class import (
    CLASS_OF_RECORD,
    GREEDY,
    FixedLoad,
    PearlTables,
    Tables,
    WindowClass,
    takes,
)
from .window_cut import ROW_FORMAT, cut_values, linears
from .window_lowering import CUTS, FAMILIES, _Program

NAME = "verity-vllm/window"


_GRAIN = Grain()


def class_program(registered: Any, cls: WindowClass) -> Any:
    """The class's request Program (a `build.BuildResult`): the Build's greedy request at one prompt row and
    ``cls.positions`` - 1 decode steps, in the registered compiler's form."""
    return request_program(registered, RequestUnit(-1, "", 1, cls.positions - 1, GREEDY, None, 0))


def stop_ids(registered: Any) -> tuple[int, ...]:
    """The registered model's stop ids (`frontend.stop_rule.stop_ids`): outputs end at the first, kept."""
    return of_config(registered.phi.hf_config)


@dataclass(frozen=True)
class Lowered:
    """A class circuit with what its lowering knows (`window_rows` fills its units from it): the class Program's function, each
    slot's lowered Program (node j's output as the rows that hold it, ``outs[j]``) and each template's runs of units by their
    (slot, step), ``(at, count, slot, step)``."""
    circuit: IF.Circuit
    fn: Any
    programs: tuple[_Program, ...]
    spans: dict[str, list[tuple[int, int, int, int]]]

    @property
    def cut(self) -> frozenset[int]:
        """The class Program's nodes the circuit takes as registered, cut at its linears (`PearlTables`); none uncut."""
        return frozenset(self.programs[0].arows) if self.programs else frozenset()


def class_circuit(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN,
                  provable: bool = True, tables: FixedLoad | None = None) -> IF.Circuit:
    """``cls``'s circuit (module docstring): the class Program lowered once a slot, node by node, slot k reading its lane of the
    scan state row before its anchor (`WindowClass.anchor`), each allocation's slots side by side in its template; its blocks'
    slots are the slots. Its registered parameters are the weights (`window_builder.weight_param`) and ``SCAN_STATE``'s
    ``cls.buckets`` rows, row a the scan's state row before bucket a (`ingress_scan.window_state`), and with ``tables`` H1's
    tables (`Tables`) or Pearl+'s (`PearlTables`). A window's circuit is every slot's; ``programs`` is the first slots only, a probe's census
    (`class_size`). Each unit is a proof block Flock's composer proves (`_Builder.settle`) unless not ``provable`` (a probe's,
    or a test's of too small a class)."""
    return lower(registered, cls, programs, grain, provable=provable, tables=tables).circuit


def lower(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN,
          wired: bool = True, provable: bool = True, tables: FixedLoad | None = None) -> Lowered:
    """`class_circuit`'s circuit with its lowering (`Lowered`); unless ``wired``, its units and registered reads alone, no
    wire or cut, whatever `settle` would refuse (the weights a registration commits, `window_weights`). With ``tables``,
    refused unless the class is one boundary; with H1's, unless round 0's rows (each slot's prompt) fit a table's ``m`` rows
    a call and every position has ``tables.calls`` int8 linears of K = ``tables.k``; with Pearl+'s, unless every position's
    linears are the first's (`window_cut.linears`)."""
    programs = cls.slots if programs is None else programs
    if not 1 <= programs <= cls.slots:
        raise Unexpressible(f"{programs} Programs in a window of a class of {cls.slots} slots")
    if isinstance(tables, Tables) and (len(cls.boundaries) != 1 or cls.slots * cls.prompt > tables.m):
        raise Unexpressible(f"{cls}: H1's tables {tables} are read at a fixed load's rows, every slot at the window's one "
                            f"boundary and round 0 its {cls.slots} slots' prompts of {cls.prompt}, at most {tables.m} rows")
    if isinstance(tables, PearlTables) and len(cls.boundaries) != 1:
        raise Unexpressible(f"{cls}: Pearl+'s tables are read at a fixed load's rows, every slot at the window's one boundary")
    try:
        scan = SS.bridge(cls.scan, cls.prompt)
    except ValueError as e:
        raise Unexpressible(f"{cls}: its slots aren't a scan state row's lanes ({e})") from None
    entry = bind(IO.SlotEntry, KO=cls.outputs, IDB=cls.scan.IDB, NP=cls.prompt, STOP=stop_ids(registered))
    b, fn = _Builder(grain, programs, provable), class_program(registered, cls).program.fn
    layout = linears(fn, tables.words, grain.rows, provable) if isinstance(tables, PearlTables) else None
    progs = [_Program(b, fn, cls, f"class Program {k}", k, scan, entry, tables, layout) for k in range(programs)]
    for (j, node), r in product(enumerate(fn.body.nodes), progs):
        r.lower(j, node)
    for r in progs:
        if r.position != cls.positions:
            raise Unexpressible(f"{r.name}: {r.position} positions, not its class's {cls.positions}")
        if isinstance(tables, Tables) and set(r.quants) != {tables.calls}:
            raise Unexpressible(f"{r.name}: int8 linears of K = {tables.k} by position {r.quants}, not H1's {tables.calls} "
                                "calls a round")
        if layout is not None and set(r.linears) != {len(layout.calls)}:
            raise Unexpressible(f"{r.name}: linears cut by position {r.linears}, not the layout's {len(layout.calls)} calls")
    if wired:
        b.settle()
    roots, openings = b.merged()
    params = {**b.params, SS.SCAN_STATE: cls.buckets * scan.bindings["NS"]}
    if isinstance(tables, Tables):
        held = tuple(v for s in range(cls.outputs) for v in (tables.a(s), tables.ids(s)))
    elif isinstance(tables, PearlTables) and layout is not None:
        held = cut_values(tables, cls, layout)
    else:
        held = ()
    reg = IF.Registered(dict(sorted(params.items())), tuple(b.tables.values()), tuple(IF.coalesce(b.reads)), tuple(roots),
                        tuple(openings), values=held, formats={v: ROW_FORMAT for v in held} if layout is not None else {})
    wires = tuple(IF.coalesce(b.wires))
    starts = {tid: [at - slot * n for at, n, slot, _step in runs] for tid, runs in b.spans.items()}
    entry_unit = progs[0].entry_unit
    assert entry_unit is not None, f"{progs[0].name}: no select at its last position, whose unit makes the slot's entry"
    circuit = IF.Circuit(NAME, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()),
                         external=(f"{descriptor_id(scan)}/state", f"{entry_unit[0]}/o0"),
                         wires=wires, cuts=b.cuts, registered=reg, order=IF.ordered(b.units, wires, starts) if wires else None,
                         blocks=IF.Blocks(programs, cls.positions, tuple(inference(b.spans, programs))))
    return Lowered(circuit, fn, tuple(progs), b.spans)


def window_circuit(registered: Any, compiled: Any, cls: WindowClass, grain: Grain = _GRAIN,
                   placed: Sequence[tuple[int, int | None]] | None = None, provable: bool = True,
                   tables: FixedLoad | None = None) -> IF.Circuit:
    """``compiled``'s window (`steps.Compiled`, of ``registered``) as ``cls``'s circuit, every slot's class Program whatever
    its traffic, refused where the class's slots don't take the window: nothing of the window reaches the circuit.
    ``placed`` is each request's anchor bucket and the bucket of its last output, in its request order, as the network
    certifier's scan placed it (the end None: unchecked); by default every request is at the window's first boundary, its
    end unchecked. A last output past the window's end is the scan's to carry (module docstring). With ``tables``, H1's
    tables read (`Tables`) or the circuit cut at its linears (`PearlTables`), refused unless the window is the fixed load."""
    takes(compiled, cls, placed, tables)
    return class_circuit(registered, cls, None, grain, provable, tables)


__all__ = ["CLASS_OF_RECORD", "CUTS", "FAMILIES", "Grain", "Lowered", "NAME", "PearlTables", "Row", "Tables", "WEIGHTS",
           "WindowClass", "class_circuit", "class_program", "lower", "stop_ids", "takes", "window_circuit"]
