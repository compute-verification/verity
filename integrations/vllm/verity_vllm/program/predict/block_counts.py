"""A day's (c, k)-inference blocks, counted from its served windows' circuit records, as a public report
(`records.block_counts`): ``verity-vllm block-counts``, the data the inference-only check's per-day n is calibrated from.

**Blocks.** A window's inference blocks are B's (`inference_blocks`): one a slot, its class Program over every step, which is
one session, since no request is carried across windows. A block holds, of each template, the units of its slot's digit in
the template's inference entries, the template's units over the window's m slots. As a Definition the block is the
window's session (`session`): ``TemplatePopulations_v1`` of each template at its units a slot, one specialized Definition
a window circuit, and a window is m activations of it. A block class is that Definition, which the report names by its
SHA-512 (`partition_v2.definition_digest`).

**Gates** are counted as `NciCheck.Count` counts a block Definition's (``boolGates``,
`verity/core/protocols/network/nci/lean/NciCheck/Count.lean`): once per specialized Definition, callees first, a
primitive's from the gate table and a composite's the sum over its nodes of members times the callee's (`gates`). The gate
table (`gate_table`) gives each primitive its two-input Boolean gates as the census counts work (`census`: an n-input And
or Xor n - 1, a Not 1, a constant bit or an input 0) through its Boolean version (`definitions.boolean.boolean_version`).
A primitive with none, or whose Boolean version reaches a primitive that is no gate, is *unpriced*: its activations a
block are listed apart and its gates are not guessed. circuit-check pins no per-primitive table, so the table counted by is
published with the report.

**Fresh bits** are B's k (`inference_blocks.budgets`): |in(I) \\ θ|, the circuit's free and ingress input bits at the
verifier's pins (`interface.input_count`: θ the weights pinned before the run at the roots the window's registration
records) over its m slots, since no wire crosses slots. The wires are the record's declarations, which the session
Definition doesn't carry, so this is the record's count and not `NciCheck.Count`'s ``fresh`` on the session Definition,
which would count every wired read.

**Counts.** A window adds m blocks to its class. A class's blocks are published padded to the grid
(`partition_v2.grid`), the day's total is the classes' padded blocks summed, and c and k are the largest gates and fresh
bits of a class with a block. The work is per distinct circuit record (a class's windows share theirs, by digest): one pass
over its templates' Definitions and its description, never a unit.

**Public.** The report holds only per-class shapes, padded counts and the gate table (`records.block_counts`). The window
records, their registrations, the partition's entries, values, placement and times stay where the tool read them.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.ml.circuits import boolean as BL, census as CE, codec, partition_v2 as PV2
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.parts import n_members
from verity.ml.circuits.population import mixed_population_program
from verity.ml.circuits.program import is_input
from verity.ml.definitions.boolean import boolean_version
from verity.ml.windows import interface as IF
from verity_vllm.config import option
from verity_vllm.records import block_counts as RBC
from verity_vllm.records.weights_registration import WeightsRegistration


def session(doc: Mapping[str, Any]) -> tuple[Any, int, int]:
    """A window circuit record's session Definition (module docstring), its slots m and steps T. Refused (ValueError) where
    the record states no inference blocks or its entries don't tile its templates (`interface.Blocks.of`)."""
    if "blocks" not in doc:
        raise ValueError(f"{doc.get('name')}: the circuit record states no inference blocks")
    units = {t: int(n) for t, n in doc["templates"]}
    blocks = IF.Blocks.of(doc["blocks"], units, doc["partition"])
    m, per = blocks.slots, Counter()
    for e in blocks.inference:
        per[e.template] += e.units // m
    members = [(resolve(t), per[t]) for t, _ in doc["templates"]]
    return mixed_population_program(members).fn, m, blocks.steps


def resolve(tid: str) -> Any:
    """`interface.resolve` over the whole registry (`catalog.load`), a refusal naming any registry module that failed to
    import."""
    from verity.ml.definitions.registry.catalog import load
    _, errors = load()
    try:
        return IF.resolve(tid)
    except ValueError as e:
        if errors:
            raise ValueError(f"{e} (registry modules that failed to import: {'; '.join(errors)})") from e
        raise


def primitives(fns: Iterable[Any]) -> dict[str, Any]:
    """The primitives ``fns`` reach that are no input, by descriptor key, each Definition visited once."""
    out: dict[str, Any] = {}
    seen: set[int] = set()
    todo = list(fns)
    while todo:
        fn = todo.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        if isinstance(fn, PrimitiveDefinition):
            if not is_input(fn):
                out[codec.descriptor_id(fn)] = fn
        else:
            todo += [nd.fn for nd in fn.body.nodes]
    return out


def gate_table(prims: Mapping[str, Any]) -> tuple[dict[str, int], list[str]]:
    """Each primitive's two-input Boolean gates (module docstring), and the keys of those it can't price."""
    work, table, unpriced = CE.BooleanWork(), {}, []
    for key, p in sorted(prims.items()):
        try:
            got = work.of(p if BL.is_boolean(p) else boolean_version(p))
        except LookupError:
            got = key
        if isinstance(got, str):
            unpriced.append(key)
        else:
            table[key] = sum(g for g, _ in got.values())
    return table, unpriced


def gates(fn: Any, table: Mapping[str, int], memo: dict[int, tuple[Any, int, Counter]] | None = None) -> tuple[int, Counter]:
    """``fn``'s Boolean gates an activation by ``table`` (`NciCheck.Count`'s ``boolGates``), and its activations of each
    primitive the table lacks."""
    memo = {} if memo is None else memo
    if (hit := memo.get(id(fn))) is not None:
        return hit[1], hit[2]
    if isinstance(fn, PrimitiveDefinition):
        key = codec.descriptor_id(fn)
        got = (0, Counter()) if is_input(fn) else (table[key], Counter()) if key in table else (0, Counter({key: 1}))
    else:
        g, miss = 0, Counter()
        for nd in fn.body.nodes:
            cg, cm = gates(nd.fn, table, memo)
            g += n_members(nd) * cg
            for k, v in cm.items():
                miss[k] += n_members(nd) * v
        got = (g, miss)
    memo[id(fn)] = (fn, *got)
    return got


def fresh(doc: Mapping[str, Any], m: int, pins: Mapping[str, str] | None, roots: Mapping[str, str] | None) -> int:
    """A block's fresh-input bits (module docstring): the record's free and ingress bits at the verifier's ``pins`` and the
    window's ``roots`` (`interface.input_count`), over its m slots."""
    count = IF.input_count(doc, pins=pins, roots=roots)
    bits = count["free"] + count["ingress"]
    if bits % m:
        raise ValueError(f"{doc.get('name')}: {bits} fresh input bits over {m} slots, not a slot's share each")
    return bits // m


@dataclass
class Day:
    """A day's served windows, counted one at a time (`add`), as `report` publishes them."""
    day: str
    pins: Mapping[str, str] | None = None
    _sessions: dict[str, tuple[Any, int, int]] = field(default_factory=dict)
    _fresh: dict[tuple[str, frozenset], int] = field(default_factory=dict)
    _classes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, doc: Mapping[str, Any], roots: Mapping[str, str] | None = None) -> str:
        """One served window: its circuit record and the roots its registration records. -> its class's definition."""
        rec = hashlib.sha512(IF.circuit_bytes(IF.circuit_of(doc))).hexdigest()
        if rec not in self._sessions:
            self._sessions[rec] = session(doc)
        fn, m, steps = self._sessions[rec]
        held = (rec, IF.held_params(self.pins, roots))
        if held not in self._fresh:
            self._fresh[held] = fresh(doc, m, self.pins, roots)
        d = PV2.definition_digest(fn)
        c = self._classes.setdefault(d, {"fn": fn, "slots": m, "steps": steps, "fresh": 0, "count": 0})
        c["fresh"], c["count"] = max(c["fresh"], self._fresh[held]), c["count"] + m
        return d

    def report(self) -> RBC.BlockCounts:
        """The public report (`records.block_counts`) of the windows added."""
        table, _ = gate_table(primitives(c["fn"] for c in self._classes.values()))
        memo: dict = {}
        classes = []
        for d, c in sorted(self._classes.items()):
            g, miss = gates(c["fn"], table, memo)
            classes.append(RBC.BlockClass(definition=d, slots=c["slots"], steps=c["steps"], gates=g,
                                          unpriced=dict(sorted(miss.items())), fresh=c["fresh"],
                                          blocks=PV2.grid(c["count"])))
        return RBC.BlockCounts(day=self.day, table=table, classes=tuple(classes), **RBC.summary(classes))


def count(day: str, windows: Sequence[tuple[Mapping[str, Any], Mapping[str, str] | None]],
          pins: Mapping[str, str] | None = None) -> RBC.BlockCounts:
    """``day``'s report from its served ``windows``, each ``(circuit record, the roots its registration records)``, at the
    verifier's ``pins``."""
    d = Day(day, pins)
    for doc, roots in windows:
        d.add(doc, roots)
    return d.report()


@dataclass
class Options:
    """A day's (c, k)-inference blocks from its served windows' circuit records, as a public report: per block class its
    shape and padded count, the day's total, and the gate table."""
    day: str = option("--day", help="the UTC day served, YYYY-MM-DD")
    window: list = option("--window", nargs=2, action="append", metavar=("CIRCUIT", "REGISTRATION"),
                          help="one served window: its circuit record (circuit.json) and its registration (the "
                               "registered roots it records), or - for none; repeat for each of the day's windows")
    pins: Path | None = option("--pins", default=None, type=Path,
                               help="the committer's weights registration, the verifier's pins (none: every read is free)")
    out: Path | None = option("--out", default=None, type=Path, help="write the report here as well as to stdout")


def main(a: Options) -> int:
    pins = None
    if a.pins is not None:
        pins = dict(WeightsRegistration.of(json.loads(a.pins.read_text()), str(a.pins)).roots)
    d = Day(a.day, pins)
    for circuit, registration in a.window:
        roots = None if registration == "-" else IF.registered_roots(json.loads(Path(registration).read_text()))
        d.add(json.loads(Path(circuit).read_text()), roots)
    rep = RBC.BlockCounts.of(d.report().dump(), "block-counts")
    text = json.dumps(rep.dump(), indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    print(f"[block-counts] {a.day}: {rep.total} blocks in {len(rep.classes)} class(es), c {rep.c}, k {rep.k}", file=sys.stderr)
    return 0


__all__ = ["Day", "Options", "count", "fresh", "gate_table", "gates", "main", "primitives", "session"]
