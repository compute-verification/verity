"""A day's (c, k)-inference blocks, counted from its served windows' circuit records and judged against the day's declared
c, k and n (`NciCheck.Count`'s rule): ``verity-vllm block-counts``. The public output is the judgment alone
(`records.block_counts`); the exact counts go only to a private file.

**Blocks.** A window's inference blocks are B's (`inference_blocks`): one a slot, its class Program over every step, which is
one session, since no request is carried across windows. A block holds, of each template, the units of its slot's digit in
the template's inference entries, the template's units over the window's m slots. As a Definition the block is the
window's session (`session`): ``TemplatePopulations_v1`` of each template at its units a slot, one specialized Definition
a window circuit, and a window is m activations of it. A block class is that Definition, which the report names by its
SHA-512 (`partition_v2.id_digest`).

**Gates** are counted as `NciCheck.Count` counts a block Definition's (``boolGates``,
`verity/protocols/inference_only/verifier/nci/lean/NciCheck/Count.lean`): once per specialized Definition, callees first, a
primitive's from the gate table and a composite's the sum over its nodes of members times the callee's (`gates`). The gate
table is the one `definitions.boolean.gate_table` gives: each primitive's two-input Boolean gates as the census counts work
(`census`: an n-input And or Xor n - 1, a Not 1, a constant bit or an input 0) through its Boolean version
(`definitions.boolean.boolean_version`). A primitive with none, or whose Boolean version reaches a primitive that is no
gate, is *unpriced*: its activations a block are listed apart and its gates are not guessed. circuit-check pins no per-primitive table, so the table counted by is
published with the judgment.

**Fresh bits** are B's k (`inference_blocks.budgets`): |in(I) \\ θ|, the circuit's free and ingress input bits at the
verifier's pins (`interface.input_count`: θ the weights pinned before the run at the roots the window's registration
records) over its m slots, since no wire crosses slots. The wires are the record's declarations, which the session
Definition doesn't carry, so this is the record's count and not `NciCheck.Count`'s ``fresh`` on the session Definition,
which would count every wired read.

**The judgment** (`judge`). A window adds m blocks to its class, and the day's blocks are the classes' summed, exactly. The
declaration (``--declaration``, a JSON object with natural ``c``, ``k`` and ``n``, such as the inference-only policy) is
named by its SHA-512. Its rules, in `NciCheck.Count`'s order: every class's block at most c gates (undecided where a block
reaches an unpriced primitive, its gates then a lower bound, unless that bound already passes c) and k fresh bits, and the
day's blocks at most n. The day is accepted when every rule accepts, and refused otherwise, its reason the first rule that
doesn't, naming c, k or n and never a count. The work is per distinct circuit record (a class's windows share theirs, by
digest): one pass over its templates' Definitions and its description, never a unit.

**Public and private.** The public judgment holds the verdict, each rule's outcome, c, k, n, the declaration's digest,
per-class shapes (gates and fresh bits a block) and the gate table (`records.block_counts`): no count of blocks or windows,
padded or exact, since at m blocks a window any count of blocks tells the count of windows. Days whose counts differ but
fall under n publish the same bytes. The exact counts, each class's and the day's, go only to ``--private`` (written 0600).
The window records, their registrations, the partition's entries, values, placement and times stay where the tool read them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.circuits import codec, partition_v2 as PV2
from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.parts import n_members
from verity.circuits.population import mixed_population_program
from verity.circuits.program import is_input
from verity.definitions.boolean import gate_table
from verity.commitments.windows import interface as IF
from verity_vllm.config import option
from verity_vllm.records import block_counts as RBC
from verity_vllm.records.weights_registration import WeightsRegistration


def session(doc: Mapping[str, Any]) -> tuple[Any, int, int]:
    """A window circuit record's session Definition (module docstring), its slots m and steps T. Refused (ValueError) where
    the record states no inference blocks or its entries don't tile its templates (`interface.Blocks.of`)."""
    if "blocks" not in doc:
        raise ValueError(f"{doc.get('name')}: the circuit record states no inference blocks")
    units = {t: int(n) for t, n in doc["templates"]}
    blocks, _ = IF.Blocks.of(doc["blocks"], units, doc["partition"])
    m, per = blocks.slots, Counter()
    for e in blocks.inference:
        per[e.template] += e.units // m
    members = [(resolve(t), per[t]) for t, _ in doc["templates"]]
    return mixed_population_program(members).fn, m, blocks.steps


def resolve(tid: str) -> Any:
    """`interface.resolve` over the whole registry (`catalog.load`), a refusal naming any registry module that failed to
    import."""
    from verity.definitions.registry.catalog import load
    _, errors = load()
    try:
        return IF.resolve(tid)
    except ValueError as e:
        if errors:
            raise ValueError(f"{e} (registry modules that failed to import: {'; '.join(errors)})") from e
        raise


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


def declaration(doc: Any) -> tuple[str, int, int, int]:
    """A declaration's SHA-512 (its JSON, sorted keys, compact) and its natural ``c``, ``k`` and ``n``; refused
    (ValueError) where it isn't a JSON object holding them."""
    if not isinstance(doc, Mapping):
        raise ValueError(f"the declaration is a {type(doc).__name__}, not a JSON object")
    got = [doc.get(f) for f in ("c", "k", "n")]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in got):
        raise ValueError(f"the declaration's c, k and n are {got}, not natural numbers")
    return (hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), *got)


def judge(classes: Sequence[RBC.BlockClass], total: int, c: int, k: int, n: int) -> tuple[dict[str, str], str | None]:
    """`NciCheck.Count`'s rules on ``classes`` (each with a block) and the day's exact ``total``: each rule's outcome, and
    the first that doesn't accept as a sentence naming c, k or n, never a count (None when every rule accepts)."""
    checks = {**RBC.shape_checks(classes, c, k), "count": "accept" if total <= n else "refuse"}
    over = next((x for x in classes if x.gates > c), None)
    unpriced = sorted({p for x in classes for p in x.unpriced})
    fat = next((x for x in classes if x.fresh > k), None)
    reasons = {"gates": (f"a block of {over.gates} gates, more than c = {c}" if over else
                         f"undecided: a block reaches {', '.join(unpriced)}, which the gate table can't price, so its "
                         f"gates are a lower bound against c = {c}"),
               "fresh": f"a block of {fat.fresh} fresh bits, more than k = {k}" if fat else "",
               "count": f"the blocks' exact count is more than n = {n}"}
    return checks, next((reasons[r] for r in RBC.RULES if checks[r] != "accept"), None)


@dataclass
class Day:
    """A day's served windows, counted one at a time (`add`), as `judgment` judges them."""
    day: str
    pins: Mapping[str, str] | None = None
    _sessions: dict[str, tuple[Any, int, int]] = field(default_factory=dict)
    _fresh: dict[tuple[str, frozenset], int] = field(default_factory=dict)
    _classes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _windows: int = 0

    def add(self, doc: Mapping[str, Any], roots: Mapping[str, str] | None = None) -> str:
        """One served window: its circuit record and the roots its registration records. -> its class's definition."""
        rec = hashlib.sha512(IF.circuit_bytes(IF.circuit_of(doc))).hexdigest()
        if rec not in self._sessions:
            self._sessions[rec] = session(doc)
        fn, m, steps = self._sessions[rec]
        held = (rec, IF.held_params(self.pins, roots))
        if held not in self._fresh:
            self._fresh[held] = fresh(doc, m, self.pins, roots)
        d = PV2.id_digest(fn)
        c = self._classes.setdefault(d, {"fn": fn, "slots": m, "steps": steps, "fresh": 0, "count": 0})
        c["fresh"], c["count"] = max(c["fresh"], self._fresh[held]), c["count"] + m
        self._windows += 1
        return d

    def judgment(self, decl: Mapping[str, Any]) -> tuple[RBC.BlockJudgment, dict[str, Any]]:
        """The public judgment (`records.block_counts`) of the windows added against the declaration ``decl``, and the
        private one: the public one with each class's exact blocks, the day's and its windows."""
        digest, c, k, n = declaration(decl)
        table, _ = gate_table(x["fn"] for x in self._classes.values())
        memo: dict = {}
        classes, exact = [], {}
        for d, x in sorted(self._classes.items()):
            g, miss = gates(x["fn"], table, memo)
            classes.append(RBC.BlockClass(definition=d, slots=x["slots"], steps=x["steps"], gates=g,
                                          unpriced=dict(sorted(miss.items())), fresh=x["fresh"]))
            exact[d] = x["count"]
        total = sum(exact.values())
        checks, reason = judge(classes, total, c, k, n)
        pub = RBC.BlockJudgment(day=self.day, declaration=digest, c=c, k=k, n=n, table=table, classes=tuple(classes),
                                checks=checks, verdict="accept" if reason is None else "refuse", reason=reason)
        return pub, {**pub.dump(), "blocks": exact, "total": total, "windows": self._windows}


def judgment(day: str, windows: Sequence[tuple[Mapping[str, Any], Mapping[str, str] | None]], decl: Mapping[str, Any],
             pins: Mapping[str, str] | None = None) -> tuple[RBC.BlockJudgment, dict[str, Any]]:
    """``day``'s public and private judgment (`Day.judgment`) from its served ``windows``, each ``(circuit record, the
    roots its registration records)``, against ``decl`` at the verifier's ``pins``."""
    d = Day(day, pins)
    for doc, roots in windows:
        d.add(doc, roots)
    return d.judgment(decl)


def write_private(path: Path, doc: Mapping[str, Any]) -> None:
    """``doc`` as JSON at ``path``, readable by its owner alone (0600) from its creation."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")


@dataclass
class Options:
    """A day's (c, k)-inference blocks from its served windows' circuit records, judged against its declared c, k and n:
    the public judgment alone (no count), and the exact counts in a private file."""
    day: str = option("--day", help="the UTC day served, YYYY-MM-DD")
    declaration: Path = option("--declaration", type=Path,
                               help="the day's declaration: a JSON object with natural c, k and n (the inference-only policy)")
    private: Path = option("--private", type=Path,
                           help="write the private judgment, with the exact counts, here (0600), beside the run's private files")
    window: list = option("--window", nargs=2, action="append", metavar=("CIRCUIT", "REGISTRATION"), default=[],
                          help="one served window: its circuit record (circuit.json) and its registration (the "
                               "registered roots it records), or - for none; repeat for each of the day's windows")
    pins: Path | None = option("--pins", default=None, type=Path,
                               help="the committer's weights registration, the verifier's pins (none: every read is free)")
    out: Path | None = option("--out", default=None, type=Path, help="write the public judgment here as well as to stdout")


def main(a: Options) -> int:
    pins = None
    if a.pins is not None:
        pins = dict(WeightsRegistration.of(json.loads(a.pins.read_text()), str(a.pins)).roots)
    d = Day(a.day, pins)
    for circuit, registration in a.window:
        roots = None if registration == "-" else IF.registered_roots(json.loads(Path(registration).read_text()))
        d.add(json.loads(Path(circuit).read_text()), roots)
    pub, private = d.judgment(json.loads(a.declaration.read_text()))
    pub = RBC.BlockJudgment.of(pub.dump(), "block-counts")
    write_private(a.private, private)
    text = json.dumps(pub.dump(), indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    print(f"[block-counts] {a.day}: {pub.verdict}{'' if pub.reason is None else ': ' + pub.reason}, against c {pub.c}, "
          f"k {pub.k}, n {pub.n}", file=sys.stderr)
    return 0


__all__ = ["Day", "Options", "declaration", "fresh", "gates", "judge", "judgment", "main", "resolve", "session",
           "write_private"]
