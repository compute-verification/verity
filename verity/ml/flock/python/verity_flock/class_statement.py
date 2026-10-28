"""A unit class of any partition (`verity.ml.circuits.units`) as a ``verity/flock-circuit`` statement (M0, :mod:`circuit`), proved by
``flock-circuit`` per class.

    python -m verity_flock.class_statement --model tiny --partition q-word --out DIR [--binary PATH] [--n 16] [--selftest]

M0's statement proves a template's identical units with every instance's input rows hashed in the circuit
(``hm96-sha512/row/v1`` under frame-v3-sha512 trees). A class becomes such a template here:

* one unit per instance: the class's circuit (`partition_units.lower_class`), its inputs packed into one u16 row port per source
  (`sources`: a Call parameter it reads, or a class of producing units whose committed outputs it reads; a word of w bits takes
  ceil(w / 16) consecutive u16 leaves, low half first, or under ``FLOCK_PACK_WORDS=1`` a word narrower than 16 bits takes the
  next w bits of the current leaf, low bits first, when they fit there and the next leaf's low bits when not, so four FP4 or two
  FP8 elements share a leaf and the row's little-endian bytes are the elements' packed memory layout; each row padded with
  unread zero words to a whole number of 64-word SHA-512 blocks, as frame-v3 rows are; under ``FLOCK_CORE_ROWS=1`` an FP GEMM
  coordinate's x and its scales are one packed row and its W column and scales another, each core's bit row (``sha512/row/v2``)
  of exactly its elements' bits, `element_rows`), its outputs into one u16 output port ``out`` (each output padded with zero bits to whole u16 words);
* each source's row is committed once and opened by every instance that reads it (M0's shared rows), rows being keyed by the
  identity of the values they hold, not by value (`class_lanes`);
* the instances are every unit of the class in its flat Definition (`class_lanes`: synthetic inputs lifted through the flat
  Definition, a Call's broadcast parameters drawn once per Call), the statement binds the partition's ``verity/partition/v1``
  digest and the program's SHA-512 (an owner-function partition, which no record can name, binds the SHA-512 of its description
  under the tag ``verity/flock/experimental-owners``).

This is a synthetic layout, not the scheme's layout rule over the partition (which is not specified yet). M0's statement commits
the outputs as hidden rows like the inputs (one ``out`` row per instance, its ``b || c`` public), so a partition's unit outputs
are committed values whose row a reading unit can open.  Every class is one plain
Boolean circuit, with no slot type other than M0's row slots: a unit's pinned-table reads (MUFU) are laid out inline as ordinary
rows by `ir_lower.layout` (the one-hot construction; 154-309 M matrix entries per read, about 20 B of prover memory each), and each
record carries the class's table reads and its unit's matrix entries. `UnitStager`'s ``reads="slots"`` instead puts a unit's
MUFU reads in C-Flock's lookup slots (`read_slots`), whose tables count once per statement.  `distinct_calls` builds a served row's workload from its program graph:
a Program with one Call of each distinct Call Definition, whose per-Call partitions have exactly the row's classes.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import resource
import subprocess
import threading
import time
import weakref
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.cut import DefinitionCut
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.units import Activation, PartitionUnits, UnitCircuit

from . import circuit as FC
from . import class_sweep
from . import gf2
from . import ir_lower as IL
from . import partition_units as PU
from . import read_slots as RS
from . import struct_lower as SL
from . import tmpl as TM

BLOCK_WORDS = 64
#: the selftest cases run on a read-bearing class (each re-parses its 1-3 GB circuit): the honest session, and the two witness tamperings
READ_CASES = ("honest", "unit_internal_bit_flipped", "unit_input_differs_from_its_message_bits")
#: and with --selftest-gpu, the GPU prover's paths against each other and against the CPU prover, and its sessions verified
#: beside the next ones' proving against one at a time (byte-identical proofs)
GPU_CASES = ("gpu_paths_agree", "gpu_proofs_match_cpu", "verify_ahead_matches_serial")
EXPERIMENTAL = "verity/flock/experimental-owners"


def _words(w: int) -> int:
    return -(-w // 16)


def _pad(n: int, to: int) -> int:
    return max(to, -(-n // to) * to)


#: FLOCK_PACK_WORDS=1: words narrower than a leaf share it (`_slots`); a different circuit and statement from the default layout's
PACK_WORDS = os.environ.get("FLOCK_PACK_WORDS") == "1"
#: FLOCK_CORE_ROWS=1: an FP GEMM coordinate's rows are core's bit rows (`element_rows`); packed, and a different circuit and
#: statement from both other layouts'
CORE_ROWS = os.environ.get("FLOCK_CORE_ROWS") == "1"
#: FLOCK_PLAIN_LEAVES=1: the proof's own Merkle leaves are plain SHA-512 (`circuit.LEAF_SCHEME_PLAIN`, a recursive session's
#: inner statement, ZK off); another statement from the default's
PLAIN_LEAVES = os.environ.get("FLOCK_PLAIN_LEAVES") == "1"
LEAF_SCHEME = FC.LEAF_SCHEME_PLAIN if PLAIN_LEAVES else None
#: FLOCK_OWN_ROWS=1: an unwired class is staged with every instance's own rows (`stage`'s ``share_rows=False``), the public
#: file a verifier without its own registered reads covers (``ProvedScope.sharedRowsOk``); rows a Call shares are committed
#: once per instance rather than once
OWN_ROWS = os.environ.get("FLOCK_OWN_ROWS") == "1"
#: Per FP GEMM coordinate family: the parameters each row holds, in order (x, then its scales; the W column, then its scales)
ELEMENT_ROWS = {"GemmCoordinateE4m3_v1": (("x",), ("wrow",)),
                "GemmCoordinateNvf4_v1": (("x", "sx"), ("wrow", "sw")),
                "GemmCoordinateMxf4_v1": (("x", "sx"), ("wrow", "sw"))}


def _slots(widths: Sequence[int], packed: bool = False) -> tuple[list[tuple[int, int]], int]:
    """Each word's (first leaf, first bit) in its row, and the leaves the row's words take: a word of 16 bits or more starts a
    leaf and takes ceil(w / 16); packed, a narrower one takes the current leaf's next w bits when they fit, else the next leaf's
    first w (a word never spans two leaves)."""
    at, leaf, bit = [], 0, 0
    for w in widths:
        if packed and w < 16:
            if bit + w > 16:
                leaf, bit = leaf + 1, 0
            at.append((leaf, bit))
            bit += w
        else:
            if bit:
                leaf, bit = leaf + 1, 0
            at.append((leaf, 0))
            leaf += _words(w)
    return at, leaf + (1 if bit else 0)


@dataclass
class ClassDefinition:
    """What M0's writers read of a Definition: one ``u16`` row parameter per input source and one ``u16`` result array, or
    with ``out_words`` a tuple of them (one output port each, their words ``w_out`` in all)."""

    id: str
    port_words: tuple[int, ...]
    w_out: int
    out_words: tuple[int, ...] = ()

    @property
    def w_in(self) -> int:
        return sum(self.port_words)

    def signature(self):
        params = tuple((f"p{i}", Array(w, Value(16))) for i, w in enumerate(self.port_words))
        if self.out_words:
            if sum(self.out_words) != self.w_out:
                raise ValueError(f"{self.id}: output ports of {self.out_words} words are not {self.w_out}")
            return params, Tuple(*(Array(w, Value(16)) for w in self.out_words))
        return params, Array(self.w_out, Value(16))


def sources(act: Activation, uc: UnitCircuit, owner_of: dict[int, str]) -> list[list[int]]:
    """The unit's inputs grouped by source, in first-read order: one group per Call parameter it reads leaves of, and one per class
    (shape) of producing units whose committed outputs it reads (``owner_of``: computing gate -> its unit's shape in the flat cut),
    as serving commits a tensor once, not per element. Each group is one row, and every instance reading the same values opens
    the same row."""
    leaf_param = []
    for i, (_n, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    groups: dict[tuple, list[int]] = {}
    for t, k in enumerate(uc.inputs):
        key = ("p", leaf_param[-1 - k]) if k < 0 else ("u", owner_of.get(k, -1))
        groups.setdefault(key, []).append(t)
    return list(groups.values())


def element_rows(act: Activation, uc: UnitCircuit, groups: list[list[int]]) -> tuple[list[list[int]], list[tuple[int, ...]], list[int]]:
    """FLOCK_CORE_ROWS's rows of an FP GEMM coordinate (`ELEMENT_ROWS`): x with its scales one row and the W column with its
    scales another, each core's bit row (``verity.ml.commitments.rowleaf.bit_row``: each element's bits low first, the codes, then
    the scales, nothing after). Returns the merged source groups, the `sources` groups each one joins, and each row's length in
    bits. Refuses a Definition that is not such a coordinate and a source that is not one whole parameter in element order. A
    row of any length is staged: the verifiers derive its fill bits and SHA-512 padding from the length."""
    family = act.fn.id.split("{")[0]
    if family not in ELEMENT_ROWS:
        raise ValueError(f"FLOCK_CORE_ROWS: {act.fn.id} is not an FP GEMM coordinate ({sorted(ELEMENT_ROWS)})")
    rows = ELEMENT_ROWS[family]
    sig = act.fn.signature()[0]
    first = [sum(t.leaves for _n, t in sig[:i]) for i in range(len(sig))]
    held = []
    for g in groups:
        leaves = [-1 - uc.inputs[t] if uc.inputs[t] < 0 else None for t in g]
        p = next((i for i in range(len(sig)) if leaves[0] is not None and first[i] <= leaves[0] < first[i] + sig[i][1].leaves), None)
        if p is None or leaves != list(range(first[p], first[p] + sig[p][1].leaves)):
            raise ValueError(f"FLOCK_CORE_ROWS: {act.fn.id}: a source that is not one whole parameter in element order")
        held.append(sig[p][0])
    merged, joins, bits = [], [], []
    for params in rows:
        if any(nm not in held for nm in params):
            raise ValueError(f"FLOCK_CORE_ROWS: {act.fn.id}: the unit reads no {params}")
        idx = tuple(held.index(nm) for nm in params)
        g = [t for i in idx for t in groups[i]]
        merged.append(g)
        joins.append(idx)
        bits.append(sum(uc.in_widths[t] for t in g))
    if sorted(i for j in joins for i in j) != list(range(len(groups))):
        raise ValueError(f"FLOCK_CORE_ROWS: {act.fn.id}: a source in no row")
    return merged, joins, bits


def _packs(packed: bool | Sequence[bool], groups: list[list[int]]) -> list[bool]:
    return [packed] * len(groups) if isinstance(packed, bool) else list(packed)


@dataclass
class Wiring:
    """A class statement of a template-instance partition (`Q_template_instance(s)` v0, the class one template's instances) whose
    units may read other units' outputs (PROTOCOL.md §16.13, Linking): the program, the query, and each staged instance's unit
    (ascending). Each such read is a link port: the reader's row of exactly the writer's output row's words (`link_ports`),
    committed under the writer's output row's salt, so the two rows' ``b || c`` are equal and the verifiers compare them
    (`link_triples`). This binds a public topology; it does not hide the topology: the verifiers derive the wiring from
    their own program, which is part of the circuit. ``strict`` False and ``writer_salts`` False stage what a cheating
    prover may commit instead (the negatives): a linked row that is not its writer's output row, or one under a salt of its
    own."""
    program: Any
    query: dict
    indices: Sequence[int]
    strict: bool = True
    writer_salts: bool = True


LINKS_FORMAT = "verity/flock-links/v1"


def cut_reads(tp: PO.TemplatePartition, units: Sequence[int]) -> list[list[tuple[int, int] | None]]:
    """Per unit of ``units`` (population indices), per template parameter leaf: the unit output its gate is (writer unit, output
    position), or None for an Input gate of the root (under a template query every other gate lies in a unit)."""
    import bisect

    owner = {g: (v, j) for v in range(tp.population) for j, g in enumerate(tp.outputs(v))}
    spans = sorted(tp.gates(v) for v in range(tp.population))
    lows = [lo for lo, _hi in spans]
    out = []
    for u in units:
        row: list[tuple[int, int] | None] = []
        for g in tp.inputs(u):
            if g in owner:
                row.append(owner[g])
                continue
            k = bisect.bisect_right(lows, g) - 1
            if k >= 0 and g < spans[k][1]:
                raise ValueError(f"unit {u} reads gate {g}, inside a unit but none of its outputs")
            row.append(None)
        out.append(row)
    return out


def _whole_row(reads: Sequence[tuple[int, int] | None], n_out) -> int | None:
    """The writer whose whole output row ``reads`` is, in order; None when every read is an Input gate; else a ValueError."""
    if all(x is None for x in reads):
        return None
    v = reads[0][0] if reads[0] is not None else None
    if v is None or any(x is None or x != (v, j) for j, x in enumerate(reads)) or len(reads) != n_out(v):
        raise ValueError(f"reads {list(reads)[:4]}...: not one unit's whole output row in order, nor all circuit inputs")
    return v


def link_ports(uc: UnitCircuit, groups: list[list[int]], reads: list[list], n_out) -> tuple[list[list[int]], list[int], list]:
    """Split every source group through which some unit reads another unit's outputs into segments, at each leaf where any unit's
    reads stop being one run (circuit inputs, or one writer's consecutive outputs). Each segment must then be, in every unit,
    circuit inputs only or exactly one writer's whole output row in order (``n_out(v)``: writer ``v``'s output count); a partial
    read is refused. Returns the groups (an unsplit group as it was), each one's original group, and each one's template leaf
    range ``(lo, hi)`` (a segment some unit links) or None."""
    if any(x >= 0 for x in uc.inputs):
        raise ValueError("a wired class reads template parameters only (its unit is one whole template instance)")
    leaf = [-1 - x for x in uc.inputs]
    cont = lambda a, b: (a is None and b is None) or (a is not None and b is not None and a[0] == b[0] and b[1] == a[1] + 1)  # noqa: E731
    out, origin, spans = [], [], []
    for gi, g in enumerate(groups):
        if all(r[leaf[t]] is None for r in reads for t in g):
            out.append(g)
            origin.append(gi)
            spans.append(None)
            continue
        g = sorted(g, key=lambda t: leaf[t])
        cuts = [0] + [k for k in range(1, len(g)) if any(not cont(r[leaf[g[k - 1]]], r[leaf[g[k]]]) for r in reads)] + [len(g)]
        for a, b in zip(cuts, cuts[1:]):
            seg = g[a:b]
            ls = [leaf[t] for t in seg]
            linked = [_whole_row([r[x] for x in ls], n_out) for r in reads]
            if any(w is not None for w in linked) and ls != list(range(ls[0], ls[0] + len(ls))):
                raise ValueError(f"a linked segment's leaves {ls[:4]}... are not consecutive")
            out.append(seg)
            origin.append(gi)
            spans.append((ls[0], ls[-1] + 1) if any(w is not None for w in linked) else None)
    return out, origin, spans


def link_triples(meta: dict, reads: list[list], indices: Sequence[int], n_out) -> list[tuple[int, str, int]]:
    """Every link the circuit's META ``links`` declares, as (reader instance, port, writer instance), from the units' reads
    (`cut_reads` of the statement's units ``indices``): a link port's leaves are, in the reader, circuit inputs (no link) or one
    writer's whole output row, the writer one of the statement's instances. Refuses a cut wire (a leaf reading another unit's
    output) that no link port carries."""
    links = meta.get("links") or []
    pos = {u: i for i, u in enumerate(indices)}
    covered = [(x["port"], *x["leaves"]) for x in links]
    out = []
    for i, r in enumerate(reads):
        for t, x in enumerate(r):
            if x is not None and not any(lo <= t < hi for _p, lo, hi in covered):
                raise ValueError(f"cut wire: unit {indices[i]}'s parameter leaf {t} reads unit {x[0]}'s output {x[1]}, and no link port carries it")
        for port, lo, hi in covered:
            try:
                v = _whole_row(r[lo:hi], n_out)
            except ValueError as e:
                raise ValueError(f"link port {port}: unit {indices[i]}'s leaves [{lo}, {hi}): {e}") from None
            if v is None:
                continue
            if v not in pos:
                raise ValueError(f"link port {port}: unit {indices[i]} reads unit {v}, which is not one of the statement's units")
            out.append((i, port, pos[v]))
    return out


def links_file(program: Any, query: dict, text: str, indices: Sequence[int]) -> dict:
    """The verifier's own ``--links`` file for the live verifier (``flock-circuit serve``): every link of the statement, derived
    from its own program and query (`link_triples`), bound to the circuit and the partition it checks them for."""
    meta = FC.meta_of(text)
    tp = PO.TemplatePartition(program, query)
    triples = link_triples(meta, cut_reads(tp, indices), indices, lambda v: len(tp.outputs(v)))
    return {"format": LINKS_FORMAT, "circuit_sha512": FC.digest(text), "partition": meta["partition"]["digest"],
            "ports": [x["port"] for x in meta.get("links") or []], "links": [list(x) for x in triples]}


def class_lowering(act: Activation, uc: UnitCircuit, name: str, groups: list[list[int]] | None = None,
                   packed: bool | Sequence[bool] = False, blocks: bool = True) -> IL.Lowering:
    """The class's unit on u16 ports: one row port per source group (``groups``: input positions; default one row for all), its
    words laid out by `_slots` (``packed``: narrow words share leaves; per group when a sequence), each padded to whole 64-word
    SHA-512 blocks (not ``blocks``: the words it uses, a bit row's ``ceil(bits / 16)``), and ``out0 .. out{W_out-1}``."""
    groups = groups or [list(range(len(uc.inputs)))]
    slots = [_slots([uc.in_widths[t] for t in g], pk) for g, pk in zip(groups, _packs(packed, groups), strict=True)]
    port_words = tuple(_pad(used, BLOCK_WORDS) if blocks else used for _at, used in slots)
    C = gf2.Circuit()
    leaves = []
    for i, w in enumerate(port_words):
        leaves += [C.input(f"in{len(leaves) + j}", 16) for j in range(w)]
    wires, base = {}, 0
    for g, (at, _used), w_port in zip(groups, slots, port_words):
        for t, (leaf, bit) in zip(g, at):
            w = uc.in_widths[t]
            wires[act.absolute(uc.inputs[t])] = [b for word in leaves[base + leaf:base + leaf + _words(bit + w)] for b in word][bit:bit + w]
        base += w_port
    IL.lower_gates(C, [act.gate(j) for j in uc.gates], wires, reads=[act.absolute(j) for j in uc.outputs])
    out_bits = []
    for j, w in zip(uc.outputs, uc.out_widths):
        out_bits += list(wires[act.absolute(j)]) + [gf2.ZERO] * (16 * _words(w) - w)
    w_out = max(1, len(out_bits) // 16)
    out_bits += [gf2.ZERO] * (16 * w_out - len(out_bits))
    for j in range(w_out):
        C.output(f"out{j}", out_bits[16 * j:16 * (j + 1)])
    w_in = sum(port_words)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(uc.shape,), gates=[], arg_gates=[], ret_gates=[])
    lay = IL.layout(C, [f"in{j}" for j in range(w_in)], [f"out{j}" for j in range(w_out)])
    return IL.Lowering(name, ClassDefinition(name, port_words, w_out), units, C, lay)


def pack(uc: UnitCircuit, ins: Sequence[Sequence[int]], want: Sequence[Sequence[int]], w_in: int, w_out: int,
         groups: list[list[int]] | None = None, port_words: Sequence[int] | None = None, packed: bool | Sequence[bool] = False):
    """The lanes as each port's rows ``{p{i}: (n, words)}``, all ports side by side ``(n, w_in)``, and outputs ``(n, w_out)``, u16, in
    the layout `class_lowering` reads (``packed`` as there)."""
    groups = groups or [list(range(len(uc.inputs)))]
    port_words = port_words or [w_in]
    n = len(ins[0]) if ins else len(want[0])
    rows = np.zeros((n, w_in), np.uint16)
    base = 0
    for g, wp, pk in zip(groups, port_words, _packs(packed, groups)):
        at, _used = _slots([uc.in_widths[t] for t in g], pk)
        for t, (leaf, bit) in zip(g, at):
            w = uc.in_widths[t]
            v = np.asarray(ins[t], dtype=np.uint64)
            if pk and w < 16:
                rows[:, base + leaf] |= ((v & np.uint64((1 << w) - 1)) << np.uint64(bit)).astype(np.uint16)
                continue
            for h in range(_words(w)):
                rows[:, base + leaf + h] = ((v >> np.uint64(16 * h)) & np.uint64(0xFFFF)).astype(np.uint16)
        base += wp
    outs = np.zeros((n, w_out), np.uint16)
    at = 0
    for vals, w in zip(want, uc.out_widths):
        v = np.asarray(vals, dtype=np.uint64)
        for h in range(_words(w)):
            outs[:, at + h] = ((v >> np.uint64(16 * h)) & np.uint64(0xFFFF)).astype(np.uint16)
        at += _words(w)
    per_port, base = {}, 0
    for i, wp in enumerate(port_words):
        per_port[f"p{i}"] = rows[:, base:base + wp]
        base += wp
    return rows, outs, per_port


def _shims(name: str, content: bytes, ports: int = 1, definition: Any = None):
    """The subcircuit and input set `circuit.compose` and `circuit.write` read for a class. With ``definition`` (the flat
    Definition the class's unit is one whole activation of), META ``program_digests`` names it as the circuit's template, as
    `rec_outer`'s statements name theirs, so that a verifier given the program finds the statement's units to be activations
    of the circuit's template; else the class itself (``unit-class/…``)."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    named = {} if definition is None else {"definition": lambda **_: definition}
    digests = ({f"unit-class/{name}": hashlib.sha256(content).hexdigest()} if definition is None else
               {definition.id: program_digest(Program(definition))})
    sub = SimpleNamespace(id=f"unit-class/{name}", template=SimpleNamespace(id="unit-class"), inputs=[port(f"p{i}") for i in range(ports)],
                          outputs=[port("out")], program_digests=lambda statics: digests, **named)
    return sub, SimpleNamespace(subcircuit=sub, name=f"unit-class-{name}", content_digest=hashlib.sha256(content).hexdigest())


_SLOTS: dict = {}


def cache_slot_circuits() -> None:
    """Build the SHA-512 and hm96 slot circuits, their layouts and netlists once: `circuit.compose` rebuilds them for every
    statement, identically, and they are about 95% of a small class's staging time."""
    if _SLOTS:
        return
    from . import sha512_circuit as S

    c3, hm = S.compression3(), S.hm96_row()
    layout, expanded = IL.layout, FC._expanded
    lays: dict = {}
    texts: dict = {}

    def cached_layout(C, *a, **k):
        if C is c3 or C is hm:
            key = (id(C), repr(a), repr(sorted(k.items())))
            if key not in lays:
                lays[key] = layout(C, *a, **k)
            return lays[key]
        return layout(C, *a, **k)

    def cached_expanded(lay, name):
        if any(lay is v for v in lays.values()):
            key = (id(lay), name)
            if key not in texts:
                texts[key] = expanded(lay, name)
            return texts[key]
        return expanded(lay, name)

    S.compression3, S.hm96_row = (lambda: c3), (lambda: hm)
    IL.layout, FC._expanded = cached_layout, cached_expanded
    _SLOTS.update(c3=c3, hm=hm)
    FC._expanded(IL.layout(c3, ["cv0", "msg0", "cv1", "msg1", "cv2", "msg2"], ["out0", "out1", "out2"]), FC.SHA_NET)   # as compose asks
    FC._expanded(IL.layout(hm, ["cv", "pad", "y"], ["bc"]), FC.HM_NET)


#: token-select Calls (the samplers): `--select` leaves them out
SAMPLER = class_sweep.SAMPLER


def distinct_calls(graph_path: str | None, modules: str | None = None, definitions: dict[str, int] | None = None,
                   exclude: re.Pattern | None = None) -> tuple[Any, dict[str, int], list[dict]]:
    """A served row's distinct Call Definitions, from its program graph (``vllm-program-graph/v1``): (a Program with one Call of
    each, the row's Call count of each by descriptor id, a provenance row per Definition). ``modules``: a module-path prefix
    (``model.model.layers.0.``) to keep. Each Definition is bound from the registry and its v0 encoding's SHA-256 is checked
    against the graph's ``definitions_index`` where the graph has it (every group's spec; attention's smallest and largest T).
    A per-Call partition (``Q_word``, an owner function) cuts this Program's Calls exactly as the row's, so its classes are the
    row's and the row's unit counts are the per-Call counts times the Call counts. ``definitions`` ({descriptor id: Call count})
    adds Definitions without a graph (bound from the registry, no digest to check); ``exclude`` drops the ids it matches."""
    from circuit_check import targets as TG
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.program import Program
    from verity.ml.definitions.encoding import encode_specialization, definition_digest

    TG.load_registries()
    g = json.loads(open(graph_path).read()) if graph_path else {"groups": []}
    index = g.get("definitions_index") or {}
    calls: dict[str, int] = dict(definitions or {})
    for fid, n in class_sweep.graph_calls(g, modules).items():
        calls[fid] = calls.get(fid, 0) + n
    if exclude is not None:
        calls = {fid: n for fid, n in calls.items() if not exclude.match(fid)}
    fns, prov = [], []
    for fid in calls:
        readable = re.sub(r'\{"f64":"([^"]+)"\}', lambda m: repr(float.fromhex(m[1])), fid)
        fn = TG.definition(readable)
        if descriptor_id(fn) != fid:
            raise ValueError(f"{fid}: the registry binds {descriptor_id(fn)}")
        want = index.get(fid)
        got = definition_digest(encode_specialization(fn)[1][fid]) if want else None
        if want and got != want:
            raise ValueError(f"{fid}: v0 digest {got} is not the row's {want}")
        fns.append(fn)
        prov.append({"definition": fid, "calls": calls[fid], "digest_checked": bool(want)})
    params = [(f"c{i}_{n}", t) for i, fn in enumerate(fns) for n, t in fn.signature()[0]]
    arity = [len(fn.signature()[0]) for fn in fns]

    def body(B, S, *xs):
        from verity.ml.circuits.refs import tuple_of
        outs, at = [], 0
        for fn, k in zip(fns, arity):
            outs.append(B.call(fn, *xs[at:at + k]))
            at += k
        return tuple_of(*outs)

    from verity.ml.circuits.types import Tuple as TupleT
    ret = TupleT(*[fn.signature()[1] for fn in fns])
    root = CompositeDefinition("DistinctCalls", 1, (), lambda S: (tuple(params), ret), body, register=False)
    return Program(root), calls, prov


#: keyed by the `PartitionUnits` object itself, never its id: a freed object's id is reused by a later one, which would then
#: read the freed partition's binding
_BINDINGS: weakref.WeakKeyDictionary[PartitionUnits, dict[str, tuple[str, str]]] = weakref.WeakKeyDictionary()


def partition_binding(units: PartitionUnits, spec: str) -> tuple[str, str]:
    """(partition digest, program SHA-512) the statement binds, computed once per partition."""
    cached = _BINDINGS.setdefault(units, {})
    if spec in cached:
        return cached[spec]
    program = program_digest(units.program)
    if units.query is not None:
        out = PO.digest(PO.build(units.program, units.query)), program
    else:
        out = hashlib.sha512(EXPERIMENTAL.encode() + b"\0" + json.dumps({"program": program, "owners": spec}, sort_keys=True).encode()).hexdigest(), program
    cached[spec] = out
    return out


#: one entry, keyed by the object as `_BINDINGS` is
_OWNERS: weakref.WeakKeyDictionary[PartitionUnits, tuple[str, dict[int, str]]] = weakref.WeakKeyDictionary()


def owners_of(units: PartitionUnits, fn_id: str) -> dict[int, str]:
    """Computing gate -> its unit's shape, in the flat Definition's cut (structure gates, copied into every reader, have none)."""
    hit = _OWNERS.get(units)
    if hit is None or hit[0] != fn_id:
        act = units.activation_of(fn_id)
        keep = set(act.graph.keep)
        _OWNERS.clear()
        hit = _OWNERS[units] = fn_id, {j: uc.shape for uc in units._flat[fn_id][1] for j in uc.gates if j in keep}
    return hit[1]


def broadcast_params(units: PartitionUnits) -> dict[str, tuple[set[int], int]]:
    """Per flat Definition that is a member of a separable Call: the parameters its batch node shares across members (axis None) and
    the member count, as the Call evaluates it (a GEMM coordinate's activation row, read by every coordinate of the Call)."""
    out: dict[str, tuple[set[int], int]] = {}

    def walk(cut):
        if not (isinstance(cut, DefinitionCut) and cut.parts is not None):
            return
        body = cut.fn.body
        for k, (child, m) in enumerate(cut.parts):
            if child is None:
                continue
            node = body.nodes[k]
            if node.form == "batch" and child.fn.id not in out:
                out[child.fn.id] = ({i for i, ax in enumerate(node.axes) if ax is None}, m)
            walk(child)

    for _k, _m, _fn, cut in units._calls():
        walk(cut)
    return out


#: evaluations of flat Definitions this large are kept for the next shape of the same Definition (the sampler's keep word is one flat
#: Definition of about 10^8 gates, and all of its ~86 shapes read the same evaluations)
KEEP_EVALUATIONS_ABOVE = 1 << 20
_EVALUATIONS: dict = {}


def _evaluation(act: Activation, key: tuple, words: list[int]) -> dict[int, int]:
    if act.fn.gates <= KEEP_EVALUATIONS_ABOVE:
        return PU.evaluate(act, words)
    if key not in _EVALUATIONS:
        if len(_EVALUATIONS) >= 2:
            _EVALUATIONS.pop(next(iter(_EVALUATIONS)))
        _EVALUATIONS[key] = PU.evaluate(act, words)
    return _EVALUATIONS[key]


def class_lanes(units: PartitionUnits, cl, key, n: int, shared: dict[str, tuple[set[int], int]]) -> tuple[list, list, list]:
    """``n`` instances of the class: every unit of that shape in its flat Definition, from as many evaluations as it takes, each
    evaluation's parameters uniform words, except those the Call shares across members (``shared``), which are drawn once per Call
    (group of members). Returns (inputs, outputs, rows): per input position every instance's word, and per source group
    (`sources` of the representative) every instance's row identity: the leaves it reads, and the Call they belong to when all
    are parameters the Call broadcasts, else the evaluation. Instances with one identity open one committed row."""
    act = units.activation(cl.definition)
    mine = [uc for uc in units._flat[cl.definition.id][1] if uc.shape == cl.shape]
    widths = [act.width(-1 - p) for p in range(act.params)]
    leaf_param = []
    for i, (_nm, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    bcast, members = shared.get(cl.definition.id, (set(), 1))
    groups = sources(act, cl.representative, owners_of(units, cl.definition.id))
    reads = [[(all(x < 0 and leaf_param[-1 - x] in bcast for x in sub), sub) for sub in (tuple(uc.inputs[t] for t in g) for g in groups)]
             for uc in mine]
    ins = [[] for _ in mine[0].inputs]
    want = [[] for _ in mine[0].outputs]
    rows = [[] for _ in groups]
    have, t = 0, 0
    while have < n:
        call = t // max(1, members)
        raw = PU.synthetic_words(key, f"{act.fn.id}#{t}", widths, 1)[0]
        grp = PU.synthetic_words(key, f"{act.fn.id}#call{call}", widths, 1)[0]
        words = [grp[p] if leaf_param[p] in bcast else raw[p] for p in range(len(raw))]
        run = _evaluation(act, (act.fn.id, t, call), words)
        for uc, per in zip(mine, reads):
            if have >= n:
                break
            for port, x in zip(ins, uc.inputs):
                port.append(run[x])
            for port, j in zip(want, uc.outputs):
                port.append(run[j])
            for ids, (b, sub) in zip(rows, per):
                ids.append((b, call if b else t, sub))
            have += 1
        t += 1
    return ins, want, rows


def stage(units: PartitionUnits, spec: str, shape: str, cl, lanes_: tuple, out_dir: Path, salt_key: bytes,
          packed: bool | None = None, core_rows: bool | None = None, wiring: Wiring | None = None, seg_ports: frozenset = frozenset(),
          seg_out_rows: tuple = (), v2_out_rows: tuple = (), pad: int | None = None, share_rows: bool = True) -> dict:
    """``circuit.txt``, the prover's ``inst-n.bin`` and the verifier's ``pub-n.bin`` for one class: one row port per input source,
    and each source's row committed once, shared by every instance that reads it (``lanes_`` from `class_lanes`; lanes without
    source identities share equal rows; ``packed``: narrow words share leaves, default FLOCK_PACK_WORDS; ``core_rows``: an FP
    GEMM coordinate's rows are core's bit rows, packed, one port per row (`element_rows`), default FLOCK_CORE_ROWS).
    ``seg_ports``, ``seg_out_rows`` and ``v2_out_rows`` are `circuit.compose`'s, over the shims' port names (``p0``, …, ``out``).
    Unwired, a class whose unit is one whole activation of its flat Definition names that Definition as the circuit's template
    (META ``program_digests``, `_shims`); a class of part of an activation, or of a primitive (which has no Program), names
    itself.

    ``wiring`` (a template-instance partition, lanes with source identities, the instances its ``indices``): META names the
    template as the unit class's ``definition`` (the class names itself in ``program_digests``), and every read of another
    unit's outputs is a link port (`link_ports`, META ``links``): its row the writer's output row's words (one word per output,
    as ``out`` holds them, never packed), one table row per reading instance, committed under the writer's ``out`` salt (and,
    ``strict``, checked here to be the writer's row).

    ``pad``: the statement padded to ``pad`` instances (a public cap, `rec_shape`), the lanes its first and the rest fillers:
    every input word zero, one source identity of their own per port (so the fillers open one committed row, zero, per port),
    and the output the class lowering's own on them, checked as every lane's is (`stage_tiled`'s ``real``).

    ``share_rows=False`` (an unwired class): every instance opens its own row per port, equal rows included, and the public file
    has no shared-row tables, the form a verifier without its own registered reads covers (``ProvedScope.sharedRowsOk``)."""
    if wiring is not None and not share_rows:
        raise ValueError("a wired class's link rows are its writers' shared rows (share_rows)")
    core_rows = CORE_ROWS if core_rows is None else core_rows
    packed = True if core_rows else PACK_WORDS if packed is None else packed
    act = units.activation(cl.definition)
    uc = cl.representative
    suffix = "-core-rows" if core_rows else "-packed" if packed else ""
    name = shape[:16] + suffix
    ins, want, idents = lanes_ if len(lanes_) == 3 else (*lanes_, None)
    real = None
    if pad is not None:
        real, fill = len(ins[0]), pad - len(ins[0])
        if wiring is not None or not 1 <= real <= pad:
            raise ValueError(f"class {name}: {real} lanes padded to {pad} instances (a wired statement is not padded)")
        ins, want = [[*p, *[0] * fill] for p in ins], [[*p, *[0] * fill] for p in want]
        idents = [[*ids, *[("filler",)] * fill] for ids in idents] if idents is not None else None
    partition, program = partition_binding(units, spec)
    groups = sources(act, uc, owners_of(units, cl.definition.id))
    bits = None
    if core_rows:
        groups, joins, bits = element_rows(act, uc, groups)
        idents = [list(zip(*(idents[i] for i in j))) for j in joins] if idents is not None else None
    packs: bool | list[bool] = packed
    spans: list = []
    writers: list = []
    definition = None
    if wiring is not None:
        from verity.ml.circuits.codec import descriptor_id

        if core_rows or idents is None:
            raise ValueError(f"class {name}: a wired class has u16 rows and lanes with source identities")
        tp = PO.TemplatePartition(wiring.program, wiring.query)
        definition = descriptor_id(act.fn)
        if definition not in tp.templates or len(wiring.indices) != len(ins[0]):
            raise ValueError(f"class {name}: {definition} is not one of the query's templates, or not one unit per instance")
        reads = cut_reads(tp, wiring.indices)
        n_out = lambda v: len(tp.outputs(v))  # noqa: E731
        pos = {u: i for i, u in enumerate(wiring.indices)}
        groups, origin, spans = link_ports(uc, groups, reads, n_out)
        packs = [packed and s is None for s in spans]
        for s in spans:
            ws = None
            if s is not None:
                ws = [_whole_row(r[s[0]:s[1]], n_out) for r in reads]
                if any(v is not None and v not in pos for v in ws):
                    raise NotImplementedError(f"class {name}: a unit reads a unit outside this statement (its salts are another statement's)")
                ws = [None if v is None else pos[v] for v in ws]
            writers.append(ws)
        idents = [idents[o] if s is None else [("link", w, k) if w is not None else ("in", k) for k, w in enumerate(writers[i])]
                  for i, (o, s) in enumerate(zip(origin, spans))]
    links = [{"port": f"p{i}", "leaves": list(s)} for i, s in enumerate(spans) if s is not None]
    low = class_lowering(act, uc, f"unit-class/{name}", groups, packs, blocks=not core_rows)
    whole = wiring is None and uc.computing == act.graph.n and not isinstance(act.fn, PrimitiveDefinition)
    sub, iset = _shims(name, uc.form.encode() + (b"/" + suffix[1:].encode() if suffix else b"") +
                       (b"/links:" + json.dumps(links).encode() if links else b""), len(groups), act.fn if whole else None)
    rows, outs, per_port = pack(uc, ins, want, low.definition.w_in, low.definition.w_out, groups, low.definition.port_words, packs)
    got, _cut, ok = low.evaluate(rows.astype(np.uint64))
    if real is not None:
        outs[real:] = got[real:].astype(np.uint16)
    text = FC.compose(low, sub, partition=partition, program=program, port_bits=bits, seg_ports=seg_ports, seg_out_rows=seg_out_rows,
                      v2_out_rows=v2_out_rows, links=links or None, definition=definition, leaf_scheme=LEAF_SCHEME)
    if not int(ok) == (1 << len(rows)) - 1 or not np.array_equal(got.astype(np.uint16), outs):
        raise AssertionError(f"class {name}: the laid-out unit rows disagree with the reference on its lanes")
    if bits is not None:
        _check_bit_rows(uc, ins, groups, bits, per_port, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.txt").write_text(text)
    n = len(rows)
    meta = FC.meta_of(text)
    per_instance = (1 << meta["k_log"]) // meta["vus_per_block"]
    cap_bits = _CTX.get("max_statement_bits", MAX_STATEMENT_BITS)
    if n * per_instance > cap_bits:                     # a unit whose rows are long pays for their hashing in every instance
        if wiring is not None or real is not None:
            raise ValueError(f"class {name}: a wired or padded statement of {n} instances is past --max-statement-bits")
        n = max(meta["vus_per_block"], 1 << ((cap_bits // per_instance).bit_length() - 1))
        rows, outs = rows[:n], outs[:n]
        per_port = {k: v[:n] for k, v in per_port.items()}
        idents = [ids[:n] for ids in idents] if idents is not None else None
    tables = refs = None
    firsts: dict[str, list[int]] = {}
    if idents is not None:
        if len(idents) != len(groups):
            raise AssertionError(f"class {name}: the lanes' source groups are not the statement's")
        tables, refs = {}, {}
        for i, ids in enumerate(idents):
            index: dict[tuple, int] = {}
            at: list[int] = []
            for k, ident in enumerate(ids):
                if ident not in index:
                    index[ident] = len(at)
                    at.append(k)
            ref = np.asarray([index[ident] for ident in ids])
            tables[f"p{i}"], refs[f"p{i}"] = per_port[f"p{i}"][at], ref
            firsts[f"p{i}"] = at
            if not np.array_equal(tables[f"p{i}"][ref], per_port[f"p{i}"]):
                raise AssertionError(f"class {name}: instances reading one source disagree on its row")
    salts = None
    if links:
        ports = FC._meta_ports(meta)
        salts = FC.stage_salts(iset, 0, n, ports, salt_key, {p.name: len(tables[p.name]) if p.name in tables else n for p in ports})
        out_name = ports[-1].name
        for i, ws in enumerate(writers):
            if ws is None:
                continue
            for r, k in enumerate(firsts[f"p{i}"]):
                w = ws[k]
                if w is None:
                    continue
                row = tables[f"p{i}"][r]
                if wiring.strict and (not np.array_equal(row[:outs.shape[1]], outs[w]) or row[outs.shape[1]:].any()):
                    raise AssertionError(f"class {name}: instance {k}'s link row p{i} is not instance {w}'s output row")
                if wiring.writer_salts:
                    salts[f"p{i}"][r] = salts[out_name][w]
    if not share_rows:
        tables = refs = None
    head = FC.write(text, iset, 0, n, out_dir / f"inst-{n}.bin", out_dir / f"pub-{n}.bin", salt_key, rows=per_port, outs={"out": outs},
                    share_rows=share_rows, tables=tables, refs=refs, salts=salts, indices=list(wiring.indices) if wiring else None)
    return ({} if real is None else {"real": real, "fillers": n - real}) | \
        {"n": n, "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "circuit_bytes": len(text), "w_in": low.definition.w_in,
            "w_out": low.definition.w_out, "unit_rows": low.layout.useful, "ports": list(low.definition.port_words),
            "shared_rows": head.get("shared_rows"), "ands": low.layout.counts["ands"], "k_log": meta["k_log"],
            "vus_per_block": meta["vus_per_block"], "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)} | \
        ({"packed": True} if packed else {}) | ({"row_schemas": meta["row_schemas"]} if bits is not None else {}) | \
        ({"links": links, "definition": definition} if wiring is not None else {})


def _check_bit_rows(uc: UnitCircuit, ins: Sequence[Sequence[int]], groups: list[list[int]], bits: list[int], per_port: dict,
                    name: str, lanes: int = 8) -> None:
    """The first ``lanes`` staged rows of each port are core's bit row of the lane's elements (``rowleaf.bit_row``), then zero."""
    from verity.ml.commitments.rowleaf import bit_row

    for i, (g, n_bits) in enumerate(zip(groups, bits)):
        rows = per_port[f"p{i}"]
        for lane in range(min(lanes, len(rows))):
            want, got_bits = bit_row(*(([int(ins[t][lane])], uc.in_widths[t]) for t in g))
            row = rows[lane].astype("<u2").tobytes()
            if got_bits != n_bits or rows.shape[1] != -(-n_bits // 16) or row[:len(want)] != want or any(row[len(want):]):
                raise AssertionError(f"class {name}: port p{i} lane {lane} is not core's bit row of its elements")


#: FLOCK_GEMM_TILE=PxC: a GemmCoordinate class whose tile fits a block (K = 2048 at 4x4) is staged as P x C tiles, one tile a VU:
#: C rows of the unit's port 0 and P of its port 1, and P * C units, unit (a, b) reading row b and row C + a, so each row is
#: hashed once per tile rather than once per coordinate
GEMM_TILE = tuple(int(x) for x in os.environ["FLOCK_GEMM_TILE"].split("x")) if os.environ.get("FLOCK_GEMM_TILE") else None


def tile_lowering(low1: IL.Lowering, P: int, C: int, name: str) -> IL.Lowering:
    """A two-port class lowering as a P x C tile: the same unit circuit and layout, P * C units over C + P row ports."""
    pw, w1 = low1.definition.port_words, low1.definition.w_out
    ports = (pw[0],) * C + (pw[1],) * P
    off = [sum(ports[:i]) for i in range(len(ports))]
    in_src = [[("leaf", off[b] + j) for j in range(pw[0])] + [("leaf", off[C + a] + j) for j in range(pw[1])] for a in range(P) for b in range(C)]
    out_src = [[("ret", u * w1 + j) for j in range(w1)] for u in range(P * C)]
    U1 = low1.units
    U = IL.Units(target=None, in_widths=U1.in_widths, out_widths=U1.out_widths, in_src=in_src, out_src=out_src, signature=U1.signature,
                 gates=[], arg_gates=[], ret_gates=[])
    return IL.Lowering(name, ClassDefinition(name, ports, w1 * P * C), U, low1.circuit, low1.layout)


def stage_tiled(units: PartitionUnits, spec: str, shape: str, cl, lanes_: tuple, out_dir: Path, salt_key: bytes, P: int, C: int,
                n: int, real: int | None = None) -> dict:
    """As `stage`, for ``n`` P x C tiles of a two-port class: every row of every tile a distinct lane's port-0 row (a valid word
    vector; a dot product's two ports are alike), each unit's output the class lowering's own on its pair, checked through the
    tile's leaf maps.

    ``real``: the statement padded to ``n`` instances (a public cap, `rec_shape`), its first ``real`` tiles the class's and the
    rest fillers: every row zero and the output the class lowering's own on them, checked as every tile's is. A filler's rows
    are equal, so the fillers open one committed row per port; the real tiles' rows and outputs are the unpadded statement's."""
    if PACK_WORDS or CORE_ROWS or OWN_ROWS:
        raise ValueError("FLOCK_PACK_WORDS, FLOCK_CORE_ROWS, FLOCK_OWN_ROWS: a tile is staged in the default layout only")
    if real is not None and not 1 <= real <= n:
        raise ValueError(f"--real {real}: between 1 and the statement's {n} instances")
    lanes_n = n if real is None else real
    act = units.activation(cl.definition)
    uc = cl.representative
    name = shape[:16]
    ins, want = lanes_[0], lanes_[1]
    partition, program = partition_binding(units, spec)
    groups = sources(act, uc, owners_of(units, cl.definition.id))
    low1 = class_lowering(act, uc, f"unit-class/{name}", groups)
    pw, w1 = low1.definition.port_words, low1.definition.w_out
    if len(pw) != 2 or pw[0] != pw[1]:
        raise ValueError(f"class {name}: a tile needs two row ports of one width, not {pw}")
    w = pw[0]
    _rows1, _outs1, per1 = pack(uc, ins, want, low1.definition.w_in, w1, groups, pw)
    if len(per1["p0"]) < lanes_n * (P + C):
        raise ValueError(f"class {name}: {lanes_n} tiles need {lanes_n * (P + C)} lanes, not {len(per1['p0'])}")
    R = np.ascontiguousarray(per1["p0"][:lanes_n * (P + C)]).reshape(lanes_n, P + C, w)
    if real is not None:
        R = np.concatenate([R, np.zeros((n - real, P + C, w), R.dtype)])
    R0, R1 = R[:, :C], R[:, C:]
    pairs = np.concatenate([np.broadcast_to(R0[:, None, :, :], (n, P, C, w)), np.broadcast_to(R1[:, :, None, :], (n, P, C, w))], axis=3)
    got, _cut, ok = low1.evaluate(pairs.reshape(n * P * C, 2 * w).astype(np.uint64))
    if not int(ok) == (1 << (n * P * C)) - 1:
        raise AssertionError(f"class {name}: a tile's unit is unsatisfied on its rows")
    outs = got.astype(np.uint16).reshape(n, P * C * w1)
    low = tile_lowering(low1, P, C, f"unit-class/{name}-tile{P}x{C}")
    per_port = {f"p{b}": np.ascontiguousarray(R0[:, b]) for b in range(C)} | {f"p{C + a}": np.ascontiguousarray(R1[:, a]) for a in range(P)}
    got_t, _c, ok_t = low.evaluate(np.concatenate([per_port[f"p{i}"] for i in range(P + C)], axis=1).astype(np.uint64))
    if not int(ok_t) == (1 << (n * P * C)) - 1 or not np.array_equal(got_t.astype(np.uint16), outs):
        raise AssertionError(f"class {name}: the tile's leaf maps disagree with its units")
    sub, iset = _shims(f"{name}-t{P}x{C}", uc.form.encode() + f"/tile{P}x{C}".encode(), P + C)
    text = FC.compose(low, sub, partition=partition, program=program, leaf_scheme=LEAF_SCHEME)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.txt").write_text(text)
    meta = FC.meta_of(text)
    per_instance = (1 << meta["k_log"]) // meta["vus_per_block"]
    cap_bits = _CTX.get("max_statement_bits", MAX_STATEMENT_BITS)
    if n * per_instance > cap_bits:
        if real is not None:
            raise ValueError(f"class {name}: a statement padded to {n} instances is past --max-statement-bits")
        n = max(meta["vus_per_block"], 1 << ((cap_bits // per_instance).bit_length() - 1))
        outs, per_port = outs[:n], {k: v[:n] for k, v in per_port.items()}
    FC.write(text, iset, 0, n, out_dir / f"inst-{n}.bin", out_dir / f"pub-{n}.bin", salt_key, rows=per_port, outs={"out": outs},
             share_rows=True)
    return ({} if real is None else {"real": real, "fillers": n - real}) | \
        {"n": n, "tile": [P, C], "coords_per_instance": P * C, "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"],
            "circuit_bytes": len(text), "w_in": low.definition.w_in, "w_out": low.definition.w_out, "unit_rows": low.layout.useful,
            "ports": list(low.definition.port_words), "ands": low.layout.counts["ands"] * P * C, "k_log": meta["k_log"],
            "vus_per_block": meta["vus_per_block"], "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)}


def _leaf_bits(t) -> list[int]:
    return list(t.leaf_widths())


def unit_lowering(definition, name: str, in_rows: Sequence[tuple[str, Sequence[str]]],
                  public: Sequence[str] = (), structural: bool = False, reads: str = "inline",
                  stage_ands: int | str | None = "auto") -> tuple[IL.Lowering, list[int]]:
    """A unit Definition on named u16 row ports: input row ``(row, params)`` is the little-endian bit concatenation of those
    parameters' leaves, in order (PoUW's rows, `leaves.words_*`; ``p[r]`` is row ``r`` of parameter ``p``, as a tile reads
    it), of exactly its words (its bits a multiple of 16); output
    port ``o{j}`` is return field ``j``'s leaves concatenated the same way, zero-padded to whole u16 words. Every
    parameter (or each of its rows) is in one row, or is one of ``public``: a public input port before the rows, each leaf in
    its own ceil(bits / 16) u16 words, low first (`verity.core.protocols.pouw.circuit.anchors.region_words`' layout, the unit's leaves as
    `circuit.compose`'s public input ports read them). Returns the lowering and each output port's words. ``structural``:
    the unit as `struct_lower` lowers it on these ports (`TmplLowering`, the flat unit never built), else `ir_lower`'s
    (`unit_circuit`), its circuit dropped once laid out (nothing reads it after `ir_lower.layout`). ``reads``: ``inline``, each
    table read laid out as rows of the unit (`ir_lower._Read`), or ``slots``, its reads of the pinned MUFU tables split out to
    lookup slots (`read_slots.split`, a `read_slots.ReadSlots`, its nets' ANDs capped by ``stage_ands``, by default the caps
    whose nets take the fewest cells; the flat unit's own when it reads none)."""
    if reads not in ("inline", "slots") or (structural and reads != "inline"):
        raise ValueError(f"{name}: reads is inline or slots, and a structural unit's are inline")
    ports, out_words, row_words = unit_ports(definition, name, in_rows, public)
    n_words, W = len(ports.ins), len(ports.outs)
    units = IL.Units(target=None, in_widths=[16] * n_words, out_widths=[16] * W, in_src=[[("leaf", p) for p in range(n_words)]],
                     out_src=[[("ret", o) for o in range(W)]], signature=(definition.id,), gates=[], arg_gates=[], ret_gates=[])
    sig = (tuple((r, Array(w, Value(16))) for r, w in zip([*public, *(r for r, _ps in in_rows)], row_words)),
           Tuple(tuple(Array(w, Value(16)) for w in out_words)))
    unit = SimpleNamespace(id=name, signature=lambda: sig)
    if structural:
        t, _lw, _top = SL.lower(definition, name=name, ports=ports)
        return TmplLowering(name, unit, units, None, _TmplLayout.of(t), t), out_words
    C = unit_circuit(definition, ports, _UnitCircuit)
    in_names, out_names = [f"in{j}" for j in range(n_words)], [f"out{o}" for o in range(W)]
    if reads == "slots" and (low := RS.split(C, in_names, out_names, name, unit, units, stage_ands=stage_ands)) is not None:
        return low, out_words
    lay = IL.layout(C, in_names, out_names)
    return IL.Lowering(name, unit, units, None, lay), out_words


def unit_ports(definition, name: str, in_rows: Sequence[tuple[str, Sequence[str]]],
               public: Sequence[str] = ()) -> tuple[SL.Ports, list[int], list[int]]:
    """`unit_lowering`'s ports: the u16 input ports, each argument leaf's bits among them, and the u16 output ports' bits
    (`struct_lower.Ports`); each output port's words; and each input port's (``public``'s, then the rows')."""
    params, ret = definition.signature()
    types = dict(params)

    def part(p):
        m = re.fullmatch(r"(\w+)\[(\d+)\]", p)
        return (m[1], int(m[2])) if m else (p, None)

    def widths(p):
        nm, r = part(p)
        return _leaf_bits(types[nm] if r is None else types[nm].elem)

    want = sorted((n, r) for n, t in params for r in ([None] if not any(part(p) == (n, 0) for _r, ps in in_rows for p in ps) else range(t.n)))
    if sorted([part(p) for _r, ps in in_rows for p in ps] + [(p, None) for p in public]) != want:
        raise ValueError(f"{name}: the rows {[list(ps) for _r, ps in in_rows]} are not the parameters {list(types)} (or their rows), once each")
    n_words, leaves = 0, {}
    for p in public:
        leaves[(p, None)] = []
        for w in widths(p):
            leaves[(p, None)].append(tuple(range(16 * n_words, 16 * n_words + w)))
            n_words += -(-w // 16)
    for row, ps in in_rows:
        bits = sum(sum(widths(p)) for p in ps)
        if bits % 16:
            raise ValueError(f"{name}: row {row} is {bits} bits, not whole u16 words")
        at = 16 * n_words
        n_words += bits // 16
        for p in ps:
            leaves[part(p)] = []
            for w in widths(p):
                leaves[part(p)].append(tuple(range(at, at + w)))
                at += w
    flat_in = [lf for n, _t in params for key in sorted(k for k in leaves if k[0] == n) for lf in leaves[key]]
    fields = [f[1] if isinstance(f, tuple) else f for f in ret.fields] if hasattr(ret, "fields") else (ret,)
    out_words, out_bits, at = [], [], 0
    for f in fields:
        bits = []
        for w in _leaf_bits(f):
            bits += range(at, at + w)
            at += w
        bits += [None] * (-len(bits) % 16)
        out_words.append(len(bits) // 16)
        out_bits += bits
    W = len(out_bits) // 16
    row_words = [sum(-(-w // 16) for w in widths(p)) for p in public] + [sum(sum(widths(p)) for p in ps) // 16 for _r, ps in in_rows]
    return SL.Ports((16,) * n_words, tuple(flat_in), (16,) * W, tuple(out_bits)), out_words, row_words


def unit_circuit(definition, ports: SL.Ports, cls: type[gf2.Circuit] = gf2.Circuit) -> gf2.Circuit:
    """The flat unit on ``ports`` (`unit_ports`), never laid out: ``definition`` lowered by `ir_lower.lower` on input ports
    ``in{j}``, its returned bits on output ports ``out{j}`` and the ports' other bits zero. ``cls``: the circuit's class
    (`unit_lowering`'s `_UnitCircuit`)."""
    C = cls()
    in_bits = [b for j, w in enumerate(ports.ins) for b in C.input(f"in{j}", w)]
    outs = [b for w in IL.lower(C, definition, [[in_bits[i] for i in lf] for lf in ports.leaves]) for b in w]
    bits = [gf2.ZERO if i is None else outs[i] for i in ports.bits]
    at = 0
    for j, w in enumerate(ports.outs):
        C.output(f"out{j}", bits[at:at + w])
        at += w
    return C


class _UnitCircuit(gf2.Circuit):
    """`unit_lowering`'s circuit: an AND keeps its operands' bits as a tuple, all `ir_lower.layout` reads of them (a frozenset of
    a dozen bits is four times the size: a GLM-5.3 norm unit's operand sets were most of its 51 GiB)."""

    def _new(self, kind, data):
        if kind == "and":
            a, b = data
            data = (gf2.L(tuple(a.s), a.c), gf2.L(tuple(b.s), b.c))
        return super()._new(kind, data)


@dataclass
class _TmplLayout:
    """What `circuit.compose` and `UnitStager` read of a unit's layout, for a unit given as its template: its port groups
    (`ir_lower.Layout`'s ``(word column, words, port widths)``) and ``counts``, the unrolled rows (`tmpl.totals`, the
    root's input and output copy rows not ANDs)."""

    in_groups: list
    out_groups: list
    counts: dict

    @classmethod
    def of(cls, t: TM.Tmpl) -> "_TmplLayout":
        g, at, groups = t.groups, 0, [[], []]
        for side in range(2):
            n = int(g[at])
            at += 1
            for _ in range(n):
                col, words, nb = int(g[at]), int(g[at + 1]), int(g[at + 2])
                groups[side].append((col, words, [int(x) for x in g[at + 3:at + 3 + nb]]))
                at += 3 + nb
        tot = TM.totals(t)
        own = len(t.defs[t.root].rows)
        return cls(groups[0], groups[1], {"ands": tot["rows"] - own, "rows": tot["rows"], "nnz_a": tot["nnz_a"],
                                          "nnz_b": tot["nnz_b"]})


class TmplLowering(IL.Lowering):
    """A unit lowered by `struct_lower` (`unit_lowering`'s ``structural``): ``template`` the unit, never unrolled; ``circuit``
    None and ``layout`` its ports (`_TmplLayout`). `evaluate` runs the template (`tmpl.evaluate`); `circuit.compose` writes
    the templated statement of ``template`` itself, its ``unit_sha256`` the template's text and the leaf pins."""

    def __init__(self, name, definition, units, circuit, layout, template: TM.Tmpl):
        super().__init__(name, definition, units, circuit, layout)
        self.template = template

    @property
    def domain(self) -> str:
        return "total"

    def evaluate(self, flat_inputs: np.ndarray, cuts: np.ndarray | None = None, flip: tuple[int, int] | None = None):
        if cuts is not None or flip is not None:
            raise NotImplementedError(f"{self.name}: a template's evaluation takes no cut words and flips no bit")
        x = np.asarray(flat_inputs, dtype=np.uint64)
        n = x.shape[0]
        bits = np.unpackbits(np.ascontiguousarray(x.astype("<u2")).view(np.uint8), axis=1, bitorder="little")
        got = TM.unslice_lanes(TM.evaluate(self.template, TM.slice_lanes(bits)), n)
        W = got.shape[1] // 16
        words = np.packbits(got.reshape(n, W * 16), axis=1, bitorder="little").view("<u2").astype(np.uint64)
        return words, np.zeros((n, 0), np.uint64), (1 << n) - 1


def _row_bits(words) -> np.ndarray:
    """Each instance's row ``(n, words)`` u16 as its bits ``(n, 16 words)``, little-endian."""
    return np.unpackbits(np.ascontiguousarray(words, dtype="<u2").view(np.uint8), axis=1, bitorder="little")


def _leaf_words(bits: np.ndarray, widths: Sequence[int], at: Sequence[int]) -> np.ndarray:
    """Leaf ``i`` of every instance: the ``widths[i]`` bits of ``bits`` from bit ``at[i]``, little-endian, as ``(n, leaves)``
    integers (Python ints when a leaf is wider than 64 bits)."""
    ws, at = np.asarray(widths, np.int64), np.asarray(at, np.int64)
    if ws.size and ws.max() > 64:
        return np.array([[int.from_bytes(np.packbits(b[a:a + w], bitorder="little").tobytes(), "little") for a, w in zip(at, ws)]
                         for b in bits], dtype=object).reshape(len(bits), len(ws))
    dt = next(d for top, d in ((8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)) if ws.max(initial=1) <= top)
    out = np.zeros((len(bits), len(ws)), dt)
    for w in np.unique(ws):
        k = np.flatnonzero(ws == w)
        out[:, k] = (bits[:, at[k, None] + np.arange(w)].astype(dt) << np.arange(w, dtype=dt)).sum(axis=2, dtype=dt)
    return out


class UnitStager:
    """One unit Definition (a PoUW unit: several output ports, rows a tile reads) as a bit-row statement, its circuit built once
    and each round's instances staged on it (`stage`): ``in_rows`` its input rows ``(row, schema, bits, params)``
    (`unit_lowering`; ``schema`` ``hm96-sha512/row/v2`` or ``…/row-seg/v1``, ``bits`` its META length), ``out_rows`` each output
    row ``(schema, output ports)``, a run of consecutive output ports of exactly their words (`circuit.compose`'s
    ``seg_out_rows`` / ``v2_out_rows``). ``public_inputs`` (``{param: {"seed", "derivation"}}``): parameters that are public
    input ports, not rows (`circuit.compose`; PoUW's anchors, which the verifier recomputes from public seeds), laid out as
    `unit_lowering`'s ``public``. ``circuit``: where circuit.txt is written, once; else `stage` writes it beside each round's
    files. ``pin`` and ``class_sha512`` (`circuit.digest`, `circuit.class_digest`) are computed on first read: `stage` reads
    them after writing circuit.txt, so a streamed circuit (``benchmarks/pouw/hidden_zk/fast_stage.py``) is hashed in the pass
    that writes it. ``bind`` (``{"partition", "program"}``, `circuit.compose`'s): the statement bound to a partition of another
    Program, a template query's that lists ``definition``, which META ``program_digests`` then names as the circuit's template
    (as `_shims`' does with a ``definition``); `stage` then takes each instance's global unit index (``indices``). The class
    (`circuit.class_digest`) is the unbound stager's. ``layout`` (a layout mode, ``iv/hybrid/4``): the templated statement
    (`circuit.compose`'s ``layout``), its unit the unit's template; else the flat statement. ``structural`` (with ``layout``):
    that template `struct_lower`'s (`unit_lowering`'s ``structural``), the flat unit never built, the outputs the template's
    (`TmplLowering.evaluate`); else `tmpl.emit`'s of the flat unit. ``reads`` (``inline`` or ``slots``, flat statements only):
    `unit_lowering`'s; under ``slots`` a unit reading a pinned MUFU table is the unit, tail stages and lookup slots of
    `read_slots`, the last stage returning the outputs, its nets' ANDs capped by ``stage_ands`` (`read_slots.split`'s). The
    proof's own leaves are `LEAF_SCHEME`'s (FLOCK_PLAIN_LEAVES=1: plain SHA-512, a recursive session's inner statement)."""

    def __init__(self, definition, name: str, in_rows: Sequence[tuple[str, str, int, Sequence[str]]],
                 out_rows: Sequence[tuple[str, Sequence[int]]], public_inputs: dict | None = None, circuit: Path | None = None,
                 bind: dict | None = None, layout: str | None = None, structural: bool = False, reads: str = "inline",
                 stage_ands: int | str | None = "auto"):
        from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG

        if bind is not None and set(bind) != {"partition", "program"}:
            raise ValueError(f"{name}: bind is exactly partition and program")

        public_inputs = dict(public_inputs or {})
        self.definition, self.name, self.in_rows, self.public_inputs = definition, name, list(in_rows), public_inputs
        if structural and layout is None:
            raise ValueError(f"{name}: a structural unit is written as its template: name its layout")
        if layout is not None and reads != "inline":
            raise ValueError(f"{name}: a templated statement's unit lays its reads out inline")
        self.low, self.out_words = unit_lowering(definition, name, [(r, ps) for r, _s, _b, ps in in_rows], tuple(public_inputs),
                                                 structural, reads, stage_ands)
        if sorted(p for _s, ps in out_rows for p in ps) != list(range(len(self.out_words))):
            raise ValueError(f"{name}: the output rows {[list(ps) for _s, ps in out_rows]} are not the outputs, once each")
        oname = lambda ps: tuple(f"o{p}" for p in ps)  # noqa: E731
        seg_out = tuple(oname(ps) for s, ps in out_rows if s == SCHEMA_HM96_SHA512_ROW_SEG)
        v2_out = tuple(oname(ps) for s, ps in out_rows if s != SCHEMA_HM96_SHA512_ROW_SEG)
        cache_slot_circuits()
        port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
        content = json.dumps([definition.id, [list(r[:3]) + [list(r[3])] for r in in_rows], [[s, list(ps)] for s, ps in out_rows]]
                             + ([public_inputs] if public_inputs else [])).encode()
        self.row_ports = [*public_inputs, *(r for r, *_ in in_rows)]
        programs = {definition.id: program_digest(Program(definition))} if bind else {name: hashlib.sha256(content).hexdigest()}
        self.sub = SimpleNamespace(id=name, template=SimpleNamespace(id="unit-class"), inputs=[port(r) for r in self.row_ports],
                                   outputs=[port(f"o{j}") for j in range(len(self.out_words))],
                                   program_digests=lambda statics: programs, **({"definition": lambda **_: definition} if bind else {}))
        self.iset = SimpleNamespace(subcircuit=self.sub, name=name.replace("/", "-"), content_digest=hashlib.sha256(content).hexdigest())
        self.text = FC.compose(self.low, self.sub, port_bits=[None] * len(public_inputs) + [b for _r, _s, b, _ps in in_rows],
                               seg_ports=frozenset(r for r, s, _b, _ps in in_rows if s == SCHEMA_HM96_SHA512_ROW_SEG),
                               seg_out_rows=seg_out, v2_out_rows=v2_out, public_inputs=public_inputs or None,
                               partition=bind and bind["partition"], program=bind and bind["program"], layout=layout,
                               tail=getattr(self.low, "tail", None), leaf_scheme=LEAF_SCHEME)
        self.bound = bind is not None
        self.meta = FC.meta_of(self.text)
        self.ports = FC._meta_ports(self.meta)
        self.circuit = circuit
        if circuit is not None:
            circuit.parent.mkdir(parents=True, exist_ok=True)
            circuit.write_text(self.text)

    @functools.cached_property
    def pin(self) -> str:
        return FC.digest(self.text)

    @functools.cached_property
    def class_sha512(self) -> str:
        return FC.class_digest(self.text)

    def reference_check(self, rows: dict[str, np.ndarray], outs: dict[str, np.ndarray]) -> None:
        """``outs`` (``{o{j}: (n, words)}``) are the IR reference's on every instance's ``rows``, leaf by leaf, from one call of
        `verity.ml.circuits.evaluation.evaluate_batch` (its registered kernel, or a Boolean Definition's word Definition's,
        where one is; else bit-sliced for a Boolean Definition and the reference for the rest), whose exceptions (rows outside
        the Definition's domain) propagate."""
        from verity.ml.circuits.evaluation import evaluate_batch

        params, ret = self.definition.signature()
        types = dict(params)
        leaves = {}
        for p in self.public_inputs:
            ws = _leaf_bits(types[p])
            leaves[(p, -1)] = _leaf_words(_row_bits(rows[p]), ws, 16 * np.cumsum([0] + [-(-w // 16) for w in ws[:-1]]))
        for r, _s, _b, ps in self.in_rows:
            bits, at = _row_bits(rows[r]), 0
            for q in ps:
                nm, _, rr = q.partition("[")
                ws = _leaf_bits(types[nm] if not rr else types[nm].elem)
                leaves[(nm, int(rr[:-1]) if rr else -1)] = _leaf_words(bits, ws, at + np.cumsum([0] + ws[:-1]))
                at += sum(ws)
        args = [np.concatenate([leaves[k] for k in sorted(k for k in leaves if k[0] == nm)], axis=1) for nm, _t in params]
        want = evaluate_batch(self.definition, *args).reshape(len(args[0]), ret.leaves)
        have = np.concatenate([_leaf_words(_row_bits(outs[f"o{j}"]), ws, np.cumsum([0] + ws[:-1]))
                               for j, ws in enumerate(_leaf_bits(f) for f in (ret.fields if hasattr(ret, "fields") else (ret,)))], axis=1)
        bad = np.flatnonzero((have != want).any(axis=1))
        if bad.size:
            raise AssertionError(f"{self.name}: instance {bad[0]}'s outputs are not the IR reference's")

    def stage(self, rows: dict[str, np.ndarray], out_dir: Path, salt_key: bytes, salts: dict | None = None,
              indices: Sequence[int] | None = None, set_name: str | None = None, tables: dict | None = None,
              refs: dict | None = None, registered: dict | None = None) -> dict:
        """A round's statement: ``rows`` each input row's words per instance ``(n, words)`` u16, and each public input's
        (``rows[param]``). The outputs are the lowering's, checked against the IR reference (`reference_check`); the salts are
        ``salts``' where given, else `circuit.stage_salts`'. Writes ``inst.bin`` and ``pub.bin`` (and ``circuit.txt`` without the
        stager's ``circuit``) under ``out_dir``; returns META, the outputs per port, every row's salts, and the unit's ANDs and
        matrix entries. ``indices``: each instance's global unit index, which a bound stager needs and an unbound one refuses
        (its units are ``0 .. n``). ``set_name``: the round's input set (the header's ``set``), else the stager's, which its
        frame domains and salts bind; a template's units split across statements (a statement's shape caps its instances) are
        one set per statement on the one circuit. ``tables``, ``refs`` and ``registered``: a shared-row file's, as
        `circuit.write` takes them (every input row's table and each instance's ref into it, whose rows must be ``rows``;
        the rows of registered values with their positions and paths), ``salts`` then one per table row."""
        if (indices is None) == self.bound:
            raise ValueError(f"{self.name}: a bound stager's round names its units' global indices, and only a bound one's")
        iset = self.iset if set_name is None else SimpleNamespace(**{**vars(self.iset), "name": set_name})
        n = len(rows[self.in_rows[0][0]])
        flat = np.concatenate([np.asarray(rows[r], dtype=np.uint16) for r in self.row_ports], axis=1)
        got, _cut, ok = self.low.evaluate(flat.astype(np.uint64))
        if int(ok) != (1 << n) - 1:
            raise AssertionError(f"{self.name}: a unit is unsatisfied on its rows")
        got = got.astype(np.uint16)
        at = np.cumsum([0] + self.out_words)
        outs = {f"o{j}": got[:, at[j]:at[j + 1]] for j in range(len(self.out_words))}
        self.reference_check(rows, outs)
        if tables is not None:
            for r, *_ in self.in_rows:
                if not np.array_equal(np.asarray(tables[r], dtype=np.uint16)[np.asarray(refs[r])], np.asarray(rows[r], dtype=np.uint16)):
                    raise ValueError(f"{self.name}: port {r}'s table rows at the instances' refs are not their rows")
        table_rows = None if tables is None else {p.name: len(tables[p.name]) if p.name in tables else n for p in self.ports}
        salts = {**FC.stage_salts(iset, 0, n, self.ports, salt_key, table_rows), **(salts or {})}
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.circuit is None:
            (out_dir / "circuit.txt").write_text(self.text)
        FC.write(self.text, iset, 0, n, out_dir / "inst.bin", out_dir / "pub.bin", salt_key,
                 rows={r: np.asarray(rows[r], dtype=np.uint16) for r, *_ in self.in_rows}, outs=outs, salts=salts,
                 public_words={p: np.asarray(rows[p], dtype=np.uint16) for p in self.public_inputs},
                 digests=(self.pin, self.class_sha512), indices=None if indices is None else [int(u) for u in indices],
                 share_rows=tables is not None, tables=tables, refs=refs, registered=registered, meta=self.meta)
        counts = getattr(self.low, "counts", None) or self.low.layout.counts
        return {"meta": FC.meta_of(self.text), "outs": outs, "salts": salts, "ands": counts["ands"],
                "unit_nnz": counts["nnz_a"] + counts["nnz_b"], "circuit_sha512": self.pin}


def stage_unit(definition, name: str, in_rows: Sequence[tuple[str, str, int, Sequence[str]]], out_rows: Sequence[tuple[str, Sequence[int]]],
               rows: dict[str, np.ndarray], out_dir: Path, salt_key: bytes, salts: dict | None = None,
               public_inputs: dict | None = None, bind: dict | None = None, indices: Sequence[int] | None = None) -> dict:
    """One round of a `UnitStager` (its arguments as there and in its `stage`): writes ``circuit.txt``, ``inst.bin`` and
    ``pub.bin``; returns META, the outputs per port, every row's salts, and the unit's ANDs and matrix entries."""
    return UnitStager(definition, name, in_rows, out_rows, public_inputs, bind=bind).stage(rows, out_dir, salt_key, salts, indices)


def statement(binary: str, d: Path, n: int, timeout: int = 3600) -> dict:
    """The statement as the verifier builds it from its public file (``flock-circuit statement``: its digest, the circuit's and
    the public file's SHA-512, k_log, blocks; nothing proved), or why it couldn't."""
    r = subprocess.run([binary, "statement", "--instances", str(d / f"pub-{n}.bin"), "--circuit", str(d / "circuit.txt")],
                       capture_output=True, text=True, timeout=timeout, check=False)
    got = next((json.loads(ln.split("\t", 1)[1]) for ln in r.stdout.splitlines() if ln.startswith("STATEMENT\t")), None)
    return got if r.returncode == 0 and got else {"why": f"exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}"}


def prove(binary: str, d: Path, n: int, port: int, selftest: bool | str = False, timeout: int = 3600, gpu: bool = False,
          warm: int = 0, runs: int = 1) -> dict:
    """A live session (``serve`` on 127.0.0.1, or a verifier pod's at FC_VERIFIERS, and ``prove``), or ``selftest``: the LIVE or
    SELFTEST record. The selftest reads the forged circuit and the tables `circuit.stage` writes beside the circuit, wherever
    it runs from; ``skipped`` lists the cases it skipped on this statement."""
    circ, inst, pub = d / "circuit.txt", d / f"inst-{n}.bin", d / f"pub-{n}.bin"
    if selftest:
        cmd = [binary, "selftest", "--instances", str(inst), "--circuit", str(circ), "--forged", str(d / "circuit-forged.txt"),
               "--tables", str(d / "tables")] + (["--gpu"] if gpu else [])
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd + (["--only", selftest] if isinstance(selftest, str) else []), capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"all_pass": False, "cases": 0, "failed": [], "finished": False, "timed_out": True, "exit_code": None,
                    "wall_s": round(time.monotonic() - t0, 3)}
        # all of it, also when the case finishes with a failed verdict: a session that fails inside it (a GPU allocation) says why
        # only here
        d.mkdir(parents=True, exist_ok=True)
        (d / f"selftest-{selftest if isinstance(selftest, str) else 'all'}.err").write_text(r.stderr)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("SELFTEST", "NEG"))]
        neg = [json.loads(ln.split("\t", 1)[1]) for ln in lines if ln.startswith("NEG")]
        head = next((json.loads(ln.split("\t", 1)[1]) for ln in lines if ln.startswith("SELFTEST")), None)
        failed = [c.get("case") for c in neg if c.get("pass") is not True]
        all_pass = head.get("all_pass") if head is not None else (bool(neg) and not failed)     # --only prints its one case, no summary
        # a process that ends without its verdict (killed, out of memory) is no verdict: its exit code and stderr say why
        finished = r.returncode == 0 and bool(lines)
        skipped = [json.loads(ln.split("\t", 1)[1])["case"] for ln in r.stdout.splitlines() if ln.startswith("SKIP\t")]
        return {"all_pass": all_pass, "cases": len(neg), "failed": failed, "skipped": skipped, "finished": finished, "timed_out": False,
                "exit_code": r.returncode, "wall_s": round(time.monotonic() - t0, 3)} | \
            ({} if finished else {"stderr_tail": r.stderr[-600:]}) | \
            ({"gpu_cases": gc} if (gc := [c for c in neg if c.get("case") in GPU_CASES]) else {})
    pin = FC.digest(circ.read_text())
    # FC_VERIFIERS=DIR: the verifier runs in another pod (`--serve-only DIR`), serving this statement's sessions at the addresses
    # it published; otherwise this pod serves its own on 127.0.0.1
    remote = None
    if os.environ.get("FC_VERIFIERS"):
        f = Path(os.environ["FC_VERIFIERS"]) / _served_name(pin, n, warm + runs)
        if not f.exists():
            return {"accepted": False, "why": f"FC_VERIFIERS: no verifier serving these {warm + runs} sessions ({f} is missing)"}
        remote = json.loads(f.read_text())
        srvs, addrs = [], remote["addrs"]
    else:
        srvs, addrs = _serve(binary, d, pub, circ, pin, warm + runs, "127.0.0.1", port, "loopback-stress")
    try:
        with open(d / "prove.out", "w") as fo, open(d / "prove.err", "w") as fe:
            # user and system CPU seconds, then the peak RSS in KB, last
            timer = ["/usr/bin/time", "-f", "%U %S %M", "-o", str(d / "prove.maxrss")] if os.path.exists("/usr/bin/time") else []
            pr = subprocess.Popen(timer + [binary, "prove", "--verifier", ",".join(addrs) or f"127.0.0.1:{port}", "--instances", str(inst),
                                           "--circuit", str(circ), "--runs", str(runs), "--warm", str(warm)] + (["--gpu"] if gpu else [])
                                  + _zk(), stdout=fo, stderr=fe)
            pr.wait()
        rusage = (d / "prove.maxrss").read_text().split() if timer else []
        prover_gb = int(rusage[-1]) / 2**20 if rusage else None   # the prover's own peak, KB
        prover_cpu_s = round(float(rusage[-3]) + float(rusage[-2]), 2) if len(rusage) >= 3 else None
        hwm = [_vm_hwm(s.pid) for s in srvs]
        verifier_gb = round(sum(hwm), 2) if srvs and None not in hwm else None    # the verifier processes' peaks, summed
    finally:
        for s in srvs:
            s.kill()
            s.wait()
    out, err = (d / "prove.out").read_text(), (d / "prove.err").read_text()
    live = [json.loads(ln[5:]) for ln in out.splitlines() if ln.startswith("LIVE")]
    if not live:
        return {"accepted": False, "why": (err or out)[-400:]}
    timed = [x for x in live if not x.get("warm")] or live
    v = sorted(timed, key=lambda x: x.get("prove_total_s") or 0)[len(timed) // 2]      # the median timed run
    # accepted: every session, warm and timed, all warm + runs of them, and prove exited 0; not the median run's verdict alone
    rejected = [x.get("run") for x in live if x.get("accepted") is not True]
    broke = (f"prove exited {pr.returncode}" if pr.returncode else
             f"{len(live)} LIVE records, not warm + runs = {warm + runs}" if len(live) != warm + runs else
             f"sessions {rejected} not accepted" if rejected else None)
    return {"accepted": broke is None, "why": broke or v.get("why")} | \
        {k: v.get(k) for k in ("prover", "instances", "vus_per_block", "k_log", "nbl", "dense_m", "e2e_s",
                               "witness_s", "witness_prebuilt_s", "pipeline_depth", "verify_ahead", "prove_s", "prove_total_s", "proof_bytes", "wire_sent",
                               "wire_recv", "statement_digest", "buckets", "calls", "wait_e2e_s", "open_rtt_ms", "open_rtt_min_ms")} | \
        {"prover_peak_gb": round(prover_gb, 2) if prover_gb else None, "prover_cpu_s": prover_cpu_s, "zk": bool(_zk()),
         "verifier_peak_gb": verifier_gb, "verify_servers": len(addrs),
         "verifier": f"pod {remote.get('host')}" if remote else "loopback", "coins": FC.coins(live),
         "verify_s": (v.get("verdict") or {}).get("verify_s"), "timed_runs": len(timed),
         "prove_total_s_runs": [x.get("prove_total_s") for x in timed],
         # per rep: the prover's phase timings (witness: units and row hashing; encoding and commitment; zerocheck, lincheck,
         # ring switch, Ligerito; t.total on the device), and host_s = the rep's wall time less t.total (host work and idle)
         "buckets": _buckets(v), "buckets_runs": [_buckets(x) for x in timed]}


def _zk() -> list[str]:
    return ["--zk"] if os.environ.get("FLOCK_ZK") == "1" else []


def _serve(binary: str, d: Path, pub: Path, circ: Path, pin: str, sessions: int, host: str, port: int, operator: str,
           bounded: bool = False, points: int = 1) -> tuple[list[subprocess.Popen], list[str]]:
    """The verifier processes for ``sessions`` sessions of one circuit, listening on ``host``, and their addresses. The prover's
    FC_VERIFY_AHEAD sessions awaiting their verdicts, and the one it proves, are served at once, by FC_VERIFY_SERVERS (default 1,
    at most ``sessions``) processes taking turns (session k the (k mod n)th's), the first on ``port`` and the rest on ports the
    system picks; ``bounded``: each exits once it has served its own sessions (otherwise the caller kills them), for ``points``
    provers one after another, each numbering its sessions from 0 and awaiting every verdict before it exits. FLOCK_DRAW_LAW (a
    law object's file, `flock-circuit serve --draw-law`): every session proves the units the verifier draws by that law at its
    Register, the staged statement their population. FLOCK_ZK=1: M1's masked sessions (`--zk`, server and prover alike)."""
    servers = max(1, min(sessions, int(os.environ.get("FC_VERIFY_SERVERS") or 1)))
    concurrency = -(-(1 + int(os.environ.get("FC_VERIFY_AHEAD") or 0)) // servers)
    logs = [d / ("serve.log" if i == 0 else f"serve-{i}.log") for i in range(servers)]
    law = ["--draw-law", os.environ["FLOCK_DRAW_LAW"]] if os.environ.get("FLOCK_DRAW_LAW") else []
    srvs = [subprocess.Popen([binary, "serve", "--listen", f"{host}:{port if i == 0 else 0}", "--out", str(d / "sessions"),
                              "--instances", str(pub), "--circuit", str(circ), "--pin", pin, "--operator", operator,
                              "--concurrency", str(concurrency)] + law + _zk()
                             + (["--sessions", str(points * len(range(i, sessions, servers)))] if bounded else []),
                             stdout=open(log, "w"), stderr=subprocess.STDOUT)
            for i, log in enumerate(logs)]
    addrs = []
    for log in logs:
        for _ in range(600):
            if (ln := next((x for x in log.read_text().splitlines() if x.startswith("SERVING ")), None)):
                addrs.append(ln.split()[1])
                break
            time.sleep(0.2)
    return srvs, addrs


def _served_name(pin: str, n: int, sessions: int) -> str:
    """The file a verifier pod publishes its addresses in: one circuit serves statements of several instance counts (a preflight
    check's and a step's), each for its own sessions."""
    return f"{pin[:32]}-n{n}-s{sessions}.json"


def serve_remote(binary: str, d: Path, n: int, where: Path, sessions: int, points: int = 1) -> dict:
    """``--serve-only``: serve one staged statement's ``sessions`` sessions on this host's address (FC_SERVE_HOST, else the
    address its hostname resolves to: a pod's IP), for a prover in another pod (``points`` of them one after another, the points
    of one GPU session, all on these processes), and publish the addresses under ``where`` (`_served_name`). The entry
    `serve_wait` takes."""
    circ, pub = d / "circuit.txt", d / f"pub-{n}.bin"
    pin = FC.digest(circ.read_text())
    import socket
    ip = os.environ.get("FC_SERVE_HOST") or socket.gethostbyname(socket.gethostname())
    srvs, bound = _serve(binary, d, pub, circ, pin, sessions, "0.0.0.0", 0, "verifier-pod", bounded=True, points=points)
    if len(bound) != len(srvs):
        for s in srvs:
            s.kill()
        raise RuntimeError(f"serve-only: {len(bound)} of {len(srvs)} verifier processes started serving")
    addrs = [f"{ip}:{a.rsplit(':', 1)[1]}" for a in bound]
    doc = {"pin": pin, "addrs": addrs, "sessions": sessions, "points": points, "host": socket.gethostname(), "run": os.environ.get("RESEARCH_RUN_ID"),
           "verify_ahead": int(os.environ.get("FC_VERIFY_AHEAD") or 0), "servers": len(srvs), "t": time.time()}
    where.mkdir(parents=True, exist_ok=True)
    f = where / _served_name(pin, n, sessions)
    tmp = f.with_name(f"{f.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, f)
    return {"file": f, "srvs": srvs, "doc": doc, "d": d, "t0": time.time()}


def serve_wait(entries: list[dict], limit_s: float) -> list[dict]:
    """Wait for every `serve_remote` entry's processes to serve their sessions (killed after ``limit_s``), withdraw the address
    files, and return each one's record: its sessions' verdicts and verify seconds (``sessions/index.jsonl``) and the
    processes' peak memory, summed."""
    peak = {id(s): 0.0 for e in entries for s in e["srvs"]}
    deadline = time.time() + limit_s
    while any(s.poll() is None for e in entries for s in e["srvs"]) and time.time() < deadline:
        for e in entries:
            for s in e["srvs"]:
                peak[id(s)] = max(peak[id(s)], _vm_hwm(s.pid) or 0.0)
        time.sleep(1)
    out = []
    for e in entries:
        timed_out = any(s.poll() is None for s in e["srvs"])
        for s in e["srvs"]:
            s.kill()
            s.wait()
        e["file"].unlink(missing_ok=True)
        idx = e["d"] / "sessions" / "index.jsonl"
        lines = [json.loads(ln) for ln in idx.read_text().splitlines() if ln.strip()] if idx.exists() else []
        vs = [x["verify_s"] for x in lines if x.get("verify_s") is not None]
        out.append(e["doc"] | {"served": len(lines), "accepted": bool(lines) and len(lines) == e["doc"]["sessions"] * e["doc"].get("points", 1)
                               and all(x.get("accepted") is True for x in lines),
                               "timed_out": timed_out, "wall_s": round(time.time() - e["t0"], 3),
                               "verify_s_median": sorted(vs)[len(vs) // 2] if vs else None,
                               "verifier_peak_gb": round(sum(peak[id(s)] for s in e["srvs"]), 2)})
    return out


def _buckets(live: dict) -> list[dict] | None:
    reps = live.get("buckets")
    if not reps:
        return None
    walls = live.get("prove_s") or []
    return [dict(b, host_s=round(walls[i] - b.get("t.total", 0), 6) if i < len(walls) else None) for i, b in enumerate(reps)]


def _vm_hwm(pid: int) -> float | None:
    """A live process's peak resident memory (VmHWM), GB."""
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmHWM:"):
                return round(int(ln.split()[1]) / 2**20, 2)
    except OSError:
        return None
    return None


#: what `_stage_job` reads: set once in `main` before the workers fork
_CTX: dict = {}
#: AND-instances per prove under ``--batch auto``
BATCH_ANDS = 1 << 29
#: witness bits per statement (instances x the block bits each takes, row hashing included); an 80 GB GPU proves 2^32
MAX_STATEMENT_BITS = 1 << 32


def auto_batch(ands: int, row_units: int) -> int:
    cap = min(1024, row_units, _CTX.get("batch_ands", BATCH_ANDS) // max(1, ands))
    return max(16, 1 << max(0, cap.bit_length() - 1))


def _tile_fits(cc, P: int, C: int) -> bool:
    """Whether a P x C tile of this class's unit fits the largest block (K = 2048 at 4 x 4; K = 8192 needs 2^27)."""
    return cc.counts["ands"] * P * C <= 1 << 25


def _source_digest() -> str:
    """The staging code's digest: every Python file of the packages a staged statement is a function of (with the program, the
    shape, the batch parameters and the seed).  A namespace package's portions are hashed as one tree, by path under the package,
    so sys.path order does not move it.  Every portion is hashed, since Python imports from one that holds only namespace
    subpackages; two portions that define one module make the import depend on that order, and raise ValueError.  A directory or
    file name holding a dot names no module, so a Lake package's `.lake/` (Mathlib's scripts, under `verity/lean/` once it is
    built) is neither walked nor hashed."""
    import importlib.util

    def py_files(root: Path):
        for d, dirs, files in root.walk():
            dirs[:] = [x for x in dirs if "." not in x]
            yield from (d / f for f in files if f.endswith(".py") and "." not in f[:-3])

    h = hashlib.sha256()
    for name in ("verity", "verity.ml.circuits", "verity.ml.kernels", "verity.core", "verity_flock", "circuit_check",
                 "verity_vllm"):
        spec = importlib.util.find_spec(name)
        roots = [Path(r) for r in (spec.submodule_search_locations or [])] if spec else []
        owner: dict[tuple[str, ...], Path] = {}
        for rel, f, root in sorted((f.relative_to(root), f, root) for root in roots for f in py_files(root)):
            mod = rel.with_suffix("").parts
            mod = mod[:-1] if mod[-1:] == ("__init__",) else mod
            if owner.setdefault(mod, root) != root:
                raise ValueError(f"{'.'.join((name, *mod))} is defined in two portions ({owner[mod]}, {root}): its import depends on sys.path order")
            h.update(f"{name}/{rel.as_posix()}\0".encode() + hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def _stage_cache(s: str) -> Path | None:
    """FLOCK_STAGE_CACHE: where a staged shape is kept for the next run of the same code, program, shape, batch and seed."""
    root, c = os.environ.get("FLOCK_STAGE_CACHE"), _CTX
    if not root:
        return None
    if "source_digest" not in c:
        c["source_digest"] = _source_digest()
    a = c["a"]
    ident = [c["source_digest"], c["program_digest"], s, a.batch, a.n, c.get("batch_ands"), c.get("max_statement_bits"), c["row_units"].get(s),
             a.seed, a.partition, a.max_ands, a.select, GEMM_TILE] + (["pack-words"] if PACK_WORDS else []) + (["core-rows"] if CORE_ROWS else []) \
        + (["plain-leaves"] if PLAIN_LEAVES else []) + (["own-rows"] if OWN_ROWS else []) \
        + ([f"real-{a.real}"] if a.real is not None else [])
    return Path(root) / hashlib.sha256(json.dumps(ident, default=str).encode()).hexdigest()[:32]


def _stage_job(s: str) -> dict:
    """Lower, lift the lanes of and stage one shape: its record, with ``stage`` when it staged, else ``broke`` (a shape staged
    before by the same code, program, batch and seed is taken from FLOCK_STAGE_CACHE)."""
    c = _CTX
    units, cl, a = c["units"], c["classes"][s], c["a"]
    cache = _stage_cache(s)
    if cache is not None and (cache / "rec.json").exists():
        t0 = time.time()
        d = c["out"] / s[:16]
        d.mkdir(parents=True, exist_ok=True)
        for f in cache.iterdir():
            if f.name != "rec.json":
                (d / f.name).unlink(missing_ok=True)
                try:
                    os.link(f, d / f.name)
                except OSError:
                    import shutil
                    shutil.copyfile(f, d / f.name)
        rec = json.loads((cache / "rec.json").read_text())
        rec["stage_cached"] = cache.name
        rec["stage_s"] = rec["stage_wall_s"] = round(time.time() - t0, 3)
        return rec
    rec = _stage_job_fresh(s)
    if cache is not None and "stage" in rec:
        import shutil
        tmp = cache.with_name(f"{cache.name}.tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for f in (c["out"] / s[:16]).iterdir():
            if f.is_file():
                shutil.copyfile(f, tmp / f.name)
        (tmp / "rec.json").write_text(json.dumps(rec, default=str))
        try:
            os.rename(tmp, cache)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    return rec


def _stage_job_fresh(s: str) -> dict:
    c = _CTX
    units, cl, a = c["units"], c["classes"][s], c["a"]
    uc = cl.representative
    rec: dict[str, Any] = {"shape": s, "definition": uc.definition, "unit": uc.unit, "units": cl.units, "row_units": c["row_units"].get(s),
                           "word_gates": len(uc.gates), "in_bits": sum(uc.in_widths), "out_bits": sum(uc.out_widths)}
    t0 = time.time()
    cc = PU.lower_class(units.activation(cl.definition), uc)
    if cc.circuit is None:
        rec["broke"] = {"at": "lowering", "why": cc.reason, "missing": list(cc.missing)}
    elif (a.select == "table-free" and cc.circuit.lookups) or (a.select == "reads" and not cc.circuit.lookups):
        rec["skipped"] = "reads" if cc.circuit.lookups else "table-free"
        rec["ands"] = cc.counts["ands"]
        rec["table_reads"] = dict(Counter(t for t, _i, _w in cc.circuit.lookups))
    elif cc.counts["ands"] > a.max_ands:
        rec["broke"] = {"at": "size", "why": f"{cc.counts['ands']} ANDs > --max-ands {a.max_ands}"}
        rec["ands"] = cc.counts["ands"]
    else:
        try:
            rec["ands"] = cc.counts["ands"]
            rec["domain"] = "finite-only" if cc.circuit.asserts else "total"
            rec["table_reads"] = dict(Counter(t for t, _i, _w in cc.circuit.lookups))
            tile = GEMM_TILE if GEMM_TILE and str(uc.definition).startswith("GemmCoordinate") and _tile_fits(cc, *GEMM_TILE) else None
            if tile:
                P, C = tile
                n = (auto_batch(P * C * cc.counts["ands"], max(1, (c["row_units"].get(s) or cl.units) // (P * C)))
                     if a.batch == "auto" else a.n)
                lanes_ = class_lanes(units, cl, c["key"], (a.real or n) * (P + C), c["shared"])
                rec["stage"] = stage_tiled(units, a.partition, s, cl, lanes_, c["out"] / s[:16], c["salt_key"], P, C, n, a.real)
            else:
                n = auto_batch(cc.counts["ands"], c["row_units"].get(s) or cl.units) if a.batch == "auto" else a.n
                lanes_ = class_lanes(units, cl, c["key"], a.real or n, c["shared"])
                rec["stage"] = stage(units, a.partition, s, cl, lanes_, c["out"] / s[:16], c["salt_key"],
                                     pad=n if a.real is not None else None, share_rows=not OWN_ROWS)
            rec["stage_py_maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        except Exception as e:  # noqa: BLE001  (a class that breaks is a result, recorded with where it broke)
            rec.pop("stage", None)
            rec["broke"] = {"at": "statement", "why": f"{type(e).__name__}: {e}"[:400]}
    rec["stage_s"] = rec["stage_wall_s"] = round(time.time() - t0, 3)
    return rec


def _staged(jobs: list[str], workers: int):
    """`_stage_job` over ``jobs``: in this process, in order, or in ``workers`` forked processes at most two shapes each ahead of
    the prover, each record as soon as its shape is staged (a long shape doesn't hold the GPU up). A worker that dies (the OOM
    killer) breaks the pool: the shapes in flight are staged again in a new one, and a shape lost twice is recorded as broken."""
    if workers <= 1:
        yield from map(_stage_job, jobs)
        return
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    queue, lost = list(reversed(jobs)), Counter()
    while queue:
        live: dict = {}
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            try:
                while queue or live:
                    while queue and len(live) < 2 * workers:
                        s = queue.pop()
                        live[ex.submit(_stage_job, s)] = s
                    done, _ = wait(live, return_when=FIRST_COMPLETED)      # the first staged shape proves first
                    for f in done:
                        rec = f.result()
                        del live[f]
                        yield rec
            except BrokenProcessPool:
                for _f, s in live.items():
                    lost[s] += 1
                    if lost[s] < 2:
                        queue.append(s)
                    else:
                        cl = _CTX["classes"][s]
                        yield {"shape": s, "definition": cl.representative.definition, "units": cl.units, "word_gates": len(cl.representative.gates),
                               "broke": {"at": "statement", "why": "its staging worker died twice (killed, likely out of memory)"}}


#: seconds between the device-memory samples of selftest cases run at once
SELFTEST_SAMPLE_S = 0.5


def _gpu_free_mib() -> int | None:
    """The least free memory of the GPUs this process sees (nvidia-smi), or None when it can't be read."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=60)
        return min(int(v) for v in r.stdout.split()) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _case_device_mib() -> dict[str, int] | None:
    """Each running selftest case process's device memory in MiB, by case (nvidia-smi's compute apps, whose pids the driver
    gives in this pod's pid namespace; a case process is named by its ``--only`` argument), or None when it can't be read."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out: dict[str, int] = {}
    for ln in r.stdout.splitlines():
        try:
            pid, mib = (int(v) for v in ln.split(","))
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (ValueError, OSError):
            continue
        if b"--only" in argv[:-1]:
            case = argv[argv.index(b"--only") + 1].decode()
            out[case] = out.get(case, 0) + mib
    return out


def _selftest_cases(binary: str, d: Path, n: int, port: int, cases: Sequence[str], gpu: bool, jobs: int,
                    case_mib: int | None) -> tuple[list[dict], dict | None]:
    """Each case in its own selftest process, up to ``jobs`` at once, the records in case order, and how they ran (None for
    ``jobs`` 1). On the GPU they run at once only while ``jobs`` x ``case_mib`` (one case process's device memory, measured)
    fits the GPU's free memory, else one at a time, as when either is unknown; each case process's device memory is sampled
    while they run."""
    def one(c: str) -> dict:
        return prove(binary, d, n, port, selftest=c, gpu=gpu)

    if jobs <= 1:
        return [one(c) for c in cases], None
    how: dict[str, Any] = {"jobs_asked": jobs, "jobs": min(jobs, len(cases))}
    if gpu and how["jobs"] > 1:
        free = _gpu_free_mib()
        how |= {"free_mib": free, "case_bound_mib": case_mib}
        if free is None or case_mib is None or how["jobs"] * case_mib > free:
            why = ("the GPU's free memory can't be read" if free is None else "no per-case device memory bound" if case_mib is None
                   else f"{how['jobs']} x {case_mib} MiB is more than the {free} MiB free")
            how |= {"jobs": 1, "fallback": why}
    peak: dict[str, int] = {}
    stop = threading.Event()

    def sample() -> None:
        while True:
            now = _case_device_mib()
            if now is None:
                how["sampler"] = "nvidia-smi's compute apps can't be read"
            else:
                how["samples"] = how.get("samples", 0) + 1
                for c, mib in now.items():
                    peak[c] = max(peak.get(c, 0), mib)
                how["sum_peak_mib"] = max(how.get("sum_peak_mib", 0), sum(now.values()))
            if stop.wait(SELFTEST_SAMPLE_S):
                return

    sampler = threading.Thread(target=sample, daemon=True) if gpu else None
    if sampler:
        how["sampled_every_s"] = SELFTEST_SAMPLE_S
        sampler.start()
    try:
        if how["jobs"] <= 1:
            runs = [one(c) for c in cases]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(how["jobs"]) as ex:
                runs = list(ex.map(one, cases))
    finally:
        stop.set()
        if sampler:
            sampler.join()
    for c, x in zip(cases, runs):
        x["device_peak_mib"] = peak.get(c)
    return runs, how


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=["tiny"])
    src.add_argument("--definition")
    src.add_argument("--program-module", help="MODULE:FUNC (FUNC() returns a Program); MODULE may be a .py path")
    src.add_argument("--program-graph", help="a served row's program graph: its distinct Call Definitions (distinct_calls)")
    src.add_argument("--definitions", help="a JSON file {descriptor id: Call count}: those Call Definitions, one Call of each (distinct_calls)")
    ap.add_argument("--modules", default=None, help="with --program-graph: keep groups whose module path starts with this")
    ap.add_argument("--partition", default="q-word")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default=os.environ.get("FLOCK_CIRCUIT", "flock-circuit"))
    ap.add_argument("--selftest-binary", default=None, help="the flock-circuit build with seed injection, for selftests (a proving "
                    "build never has it; default --binary)")
    ap.add_argument("--n", type=int, default=16, help="instances per class statement (synthetic lanes)")
    ap.add_argument("--real", type=int, default=int(os.environ["FLOCK_REAL"]) if os.environ.get("FLOCK_REAL") else None,
                    help="with --batch fixed: the statement's first REAL instances (tiles under FLOCK_GEMM_TILE) the class's, the "
                    "rest to --n fillers (zero inputs; stage's pad, stage_tiled's real), so its shape is --n's whatever REAL is "
                    "(default FLOCK_REAL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", type=int, default=0, help="also run flock-circuit's selftest (31 cases, negatives included) on every "
                    "Nth provable class and on each flat Definition's largest (0: never)")
    ap.add_argument("--selftest-gpu", action="store_true", help="run the selftests on the GPU (selftest --gpu; a read-bearing class's "
                    "subset adds GPU_CASES)")
    ap.add_argument("--selftest-cases", default=None, help="comma-separated selftest cases to run instead (one process each)")
    ap.add_argument("--selftest-jobs", type=int, default=1, help="case processes run at once (a read-bearing circuit's or "
                    "--selftest-cases'; each still its own process, the records in case order); each parses the circuit itself")
    ap.add_argument("--selftest-case-mib", type=int, default=None, help="with --selftest-jobs above 1 on the GPU: one case "
                    "process's device memory (measured); the cases run at once only while N of them fit the GPU's free memory")
    ap.add_argument("--no-selftest-largest", dest="selftest_largest", action="store_false",
                    help="selftest every Nth shape only, not also each flat Definition's largest (a full CPU selftest per attention key count)")
    ap.add_argument("--max-ands", type=int, default=1 << 24, help="skip classes above this many ANDs (reported)")
    ap.add_argument("--only", default=None, help="comma-separated shape prefixes")
    ap.add_argument("--selftest-only", action="store_true", help="stage and run the selftest cases only (no loopback session), into selftests.jsonl")
    ap.add_argument("--stage-only", action="store_true", help="stage the shapes (into FLOCK_STAGE_CACHE) and run nothing, into staged.jsonl: "
                    "a GPU job with the same settings takes them from the cache and starts at the prove")
    ap.add_argument("--statement-digest", action="store_true", help="with --stage-only: also record each staged statement's digest "
                    "as --binary builds it from the public file (flock-circuit statement; nothing proved)")
    ap.add_argument("--serve-only", default=None, metavar="DIR", help="stage the shapes (from FLOCK_STAGE_CACHE) and serve each one's "
                    "--warm + --runs sessions on this host's address, published as DIR/<pin>-n<N>-s<sessions>.json for a prover in "
                    "another pod (FC_VERIFIERS=DIR), then wait for them (FC_SERVE_WAIT_S, default 3600), into served.jsonl")
    ap.add_argument("--serve-points", type=int, default=1, help="with --serve-only: serve that many provers' --warm + --runs "
                    "sessions, one prover after another on the same verifier processes (the points of one GPU session)")
    ap.add_argument("--sample", type=int, default=0, help="prove only this many classes, drawn by a seeded hash of their shapes (0: all)")
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--gpu", action="store_true", help="prove on the GPU (flock-circuit prove --gpu)")
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1, help="timed runs per shape; the record is the median")
    ap.add_argument("--shapes", default=None, help="a file of shape digests (or prefixes), one per line: prove only these")
    ap.add_argument("--batch", default=None, help="'auto': instances per shape = the least of 1024, the row's units of it and 2^29 / its "
                    "ANDs, as a power of two >= 16 (overrides --n)")
    ap.add_argument("--batch-ands", type=int, default=BATCH_ANDS, help="--batch auto's AND-instances per prove (default 2^29)")
    ap.add_argument("--max-statement-bits", type=int, default=MAX_STATEMENT_BITS, help="witness bits per statement (default 2^32)")
    ap.add_argument("--jobs", type=int, default=1, help="shapes staged in parallel (forked workers); proving stays one at a time")
    ap.add_argument("--select", choices=["table-free", "reads", "units"], default=None,
                    help="leave out the token-select Calls, and prove only the shapes whose unit reads no pinned table (table-free), "
                    "reads one (reads), or either (units); the others are recorded as skipped, unstaged")
    ap.add_argument("--counts-only", action="store_true", help="write counts.json (every shape's row units and each Call Definition's "
                    "units per shape) and stage nothing: what `class_sweep rollup --counts` reads")
    ap.add_argument("--protocol", default="sweep-v0", help="the measurement protocol the result is labelled with (sweep-v0: never a "
                    "table cell)")
    a = ap.parse_args(argv)
    if a.real is not None and (a.batch not in (None, "fixed") or not 1 <= a.real <= a.n):
        ap.error(f"--real {a.real}: with --batch fixed, between 1 and --n {a.n}")
    calls: dict[str, int] | None = None
    if a.program_graph or a.definitions:
        defs = json.loads(Path(a.definitions).read_text()) if a.definitions else None
        program, calls, prov = distinct_calls(a.program_graph, a.modules, defs, SAMPLER if a.select else None)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "calls.json").write_text(json.dumps({"calls": calls, "provenance": prov}, indent=1))
    elif a.program_module:
        import importlib.util
        mod, fn = a.program_module.rsplit(":", 1)
        spec = importlib.util.spec_from_file_location("stress_program", mod) if mod.endswith(".py") else None
        if spec:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        else:
            m = importlib.import_module(mod)
        program = getattr(m, fn)()
    else:
        program = PU._program(a)
    units = PU.partition_units(program, a.partition)
    classes = units.classes()
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": program_digest(program), "seed": a.seed})
    salt_key = hashlib.sha256(b"verity/flock/class-statement/stress-salts\0" + str(a.seed).encode()).digest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / ("selftests.jsonl" if a.selftest_only else "staged.jsonl" if a.stage_only else "served.jsonl" if a.serve_only
                     else "results.jsonl")
    done = {json.loads(ln)["shape"] for ln in results.read_text().splitlines()} if results.exists() else set()
    row_units: dict[str, int] = {}
    per_definition: dict[str, dict[str, int]] = {}
    if calls is not None:
        from verity.ml.circuits.codec import descriptor_id
        for _k, _m, fn, cut in units._calls():
            per_definition[descriptor_id(fn)] = dict(units._count(cut, fn))
            for sh, c in per_definition[descriptor_id(fn)].items():
                row_units[sh] = row_units.get(sh, 0) + c * calls[descriptor_id(fn)]
    if a.counts_only:
        (out / "counts.json").write_text(json.dumps({
            "calls": calls, "per_definition": per_definition,
            "shapes": {s: {"definition": cl.representative.definition, "units": cl.units, "row_units": row_units.get(s),
                           "word_gates": len(cl.representative.gates)} for s, cl in classes.items()}}, sort_keys=True))
        print(f"counts: {len(classes)} shapes over {len(per_definition)} Call Definitions -> {out / 'counts.json'}")
        return 0
    order = sorted(classes.items(), key=lambda kv: len(kv[1].representative.gates))
    wanted = [ln.strip() for ln in Path(a.shapes).read_text().split() if ln.strip()] if a.shapes else None
    if a.sample:
        keep = set(sorted(classes, key=lambda sh: hashlib.sha256(f"{a.seed}/{sh}".encode()).digest())[:a.sample])
        order = [kv for kv in order if kv[0] in keep]
    largest = {}
    for sh, cl in order:
        largest[cl.definition.id if hasattr(cl.definition, "id") else cl.definition] = sh
    jobs = [s for s, _cl in order if not (s in done or (a.only and not any(s.startswith(p) for p in a.only.split(",")))
                                          or (wanted is not None and not any(s.startswith(p) for p in wanted)))]
    _CTX.update(units=units, classes=classes, row_units=row_units, shared=broadcast_params(units), key=key, salt_key=salt_key,
                out=out, a=a, batch_ands=a.batch_ands, max_statement_bits=a.max_statement_bits, program_digest=program_digest(program))
    partition_binding(units, a.partition)
    # a shape taken from FLOCK_STAGE_CACHE composes nothing; built before _staged, so forked stagers share them
    if any(not ((c := _stage_cache(s)) and (c / "rec.json").exists()) for s in jobs):
        cache_slot_circuits()
    serving: list[tuple[dict, dict]] = []
    for rec in _staged(jobs, a.jobs):
        s, cl = rec["shape"], classes[rec["shape"]]
        t0 = time.time() - rec.pop("stage_wall_s", 0)
        if "stage" in rec and a.serve_only:
            serving.append((rec, serve_remote(a.binary, out / s[:16], rec["stage"]["n"], Path(a.serve_only), a.warm + a.runs,
                                                a.serve_points)))
            print(json.dumps({"shape": s, "serving": serving[-1][1]["doc"]["addrs"]}), flush=True)
            continue
        if "stage" in rec:
            d = out / s[:16]
            try:
                if not (a.selftest_only or a.stage_only):
                    rec["live"] = prove(a.binary, d, rec["stage"]["n"], a.port, gpu=a.gpu, warm=a.warm, runs=a.runs)
                elif a.stage_only and a.statement_digest:
                    rec["statement"] = statement(a.binary, d, rec["stage"]["n"])
                if a.selftest and not a.stage_only and (len(done) % a.selftest == 0 or (a.selftest_largest and s == largest.get(cl.definition.id))):
                    sb, g = a.selftest_binary or a.binary, a.selftest_gpu
                    if rec["table_reads"] or a.selftest_cases:   # a read-bearing circuit re-parses per case: a few cases, one process each
                        cases = tuple(a.selftest_cases.split(",")) if a.selftest_cases else READ_CASES + (GPU_CASES if g else ())
                        runs, how = _selftest_cases(sb, d, rec["stage"]["n"], a.port, cases, g, a.selftest_jobs, a.selftest_case_mib)
                        rec["selftest"] = {"all_pass": all(x["all_pass"] for x in runs), "cases": len(runs),
                                           "failed": [c for c, x in zip(cases, runs) if not x["all_pass"]], "subset": list(cases),
                                           "did_not_finish": [c for c, x in zip(cases, runs) if not x["finished"]],
                                           "per_case": [{"case": c} | {k: x.get(k) for k in ("wall_s", "exit_code", "timed_out", "device_peak_mib",
                                                                                                "stderr_tail") if k in x}
                                                        for c, x in zip(cases, runs)]} | \
                            ({"run": how} if how else {}) | \
                            ({"gpu_cases": gc} if (gc := [c for x in runs for c in x.get("gpu_cases", [])]) else {})
                    else:
                        rec["selftest"] = prove(sb, d, rec["stage"]["n"], a.port, selftest=True, gpu=g)
                    rec["selftest"]["gpu"] = g
            except Exception as e:  # noqa: BLE001  (a class that breaks is a result, recorded with where it broke)
                rec["broke"] = {"at": "prover", "why": f"{type(e).__name__}: {e}"[:400]}
            for f in d.glob("*.bin"):
                f.unlink()
            if (d / "circuit.txt").exists():
                (d / "circuit.txt").unlink()                    # the record keeps its digest; restaging rebuilds it byte for byte, and
                                                                # the run's custody stays the proofs and records (a list's circuits are tens of GB)
        rec["seconds"] = round(time.time() - t0, 3)
        with results.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        done.add(s)
        print(json.dumps({k: rec.get(k) for k in ("shape", "definition", "units", "word_gates", "broke")} |
                         ({"accepted": rec["live"].get("accepted"), "prove_s": rec["live"].get("prove_total_s"),
                           "proof_bytes": rec["live"].get("proof_bytes"), "k_log": rec["live"].get("k_log")} if "live" in rec else {})), flush=True)
    for (rec, e), served in zip(serving, serve_wait([e for _r, e in serving], float(os.environ.get("FC_SERVE_WAIT_S") or 3600))):
        rec["serve"] = served
        for f in [*e["d"].glob("*.bin"), e["d"] / "circuit.txt"]:
            f.unlink(missing_ok=True)
        with results.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps({"shape": rec["shape"]} | {k: served[k] for k in ("served", "accepted", "timed_out", "verify_s_median")}), flush=True)
    write_result(results, a)
    return 0


def _device(gpu: bool) -> str | None:
    """The device that proved: the GPU's name (nvidia-smi), else the CPU model."""
    try:
        if gpu:
            return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=30).stdout.strip().splitlines()[0]
        return next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except (OSError, IndexError, StopIteration, subprocess.SubprocessError):
        return None


def write_result(results: Path, a) -> None:
    """``$RESEARCH_RUN_DIR/result.json`` (research/result/v0.1) when run as a recorded run: the settings as the fingerprint, the
    shapes' counts and summed timings as measurements, and every shape's record (keyed by shape digest) in ``results.jsonl``, copied
    beside it as the measurement file."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir or not results.exists():
        return
    import shutil

    recs = [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
    shutil.copy(results, Path(run_dir, "class-sweep.jsonl"))
    ok = [r for r in recs if (r.get("live") or {}).get("accepted")]
    rejected = [r for r in recs if r.get("live") and not r["live"].get("accepted")]
    selftests = [r["selftest"] for r in recs if r.get("selftest")]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"kind": "class-sweep/v1", "statement": FC.STATEMENT, "scheme": FC.SCHEME, "partition": a.partition,
                                    "gpu": a.gpu, "warm": a.warm, "runs": a.runs, "batch": a.batch or a.n, "seed": a.seed,
                                    "protocol": a.protocol, "select": a.select, "device": _device(a.gpu),
                                    "batch_ands": a.batch_ands, "max_statement_bits": a.max_statement_bits, "shapes": sorted(r["shape"] for r in recs)},
           "validation": {"status": "failed" if rejected or any(not x.get("all_pass") for x in selftests) else ("passed" if ok else "not_run"),
                          "checker": "flock-circuit loopback verifier (live coins) and selftest",
                          "detail": f"{len(ok)} of {len(recs)} shapes accepted, {len(rejected)} rejected, "
                                    f"{sum(1 for r in recs if r.get('broke'))} broke, {sum(1 for r in recs if r.get('skipped'))} skipped (--select); "
                                    f"selftests {sum(1 for x in selftests if x.get('all_pass'))}/{len(selftests)}",
                          "evidence": ["class-sweep.jsonl"]},
           "measurements": [{"name": "shapes", "value": len(recs), "unit": "count"},
                            {"name": "shapes_accepted", "value": len(ok), "unit": "count"},
                            {"name": "prove_total_s_sum", "value": round(sum(r["live"]["prove_total_s"] for r in ok), 3), "unit": "s"},
                            {"name": "proof_bytes_sum", "value": sum(sum(r["live"]["proof_bytes"]) for r in ok), "unit": "B"}],
           "measurement_files": ["class-sweep.jsonl"], "artifacts": []}
    tmp = Path(run_dir, "result.json.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True))
    os.replace(tmp, Path(run_dir, "result.json"))


if __name__ == "__main__":
    raise SystemExit(main())
