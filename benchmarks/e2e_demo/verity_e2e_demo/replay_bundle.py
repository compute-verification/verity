"""The drawn-unit replay bundle (CMP-95 interface 5): everything the prover behind the firewall reads to prove the drawn units
of one window, one JSON document (``verity/drawn-unit-bundle/v2``) and one binary payload of the private bytes.

The bundle is the prover's: its payload holds values and salts, so it is written mode 0600, never published and never put in
the evidence store (a test fixture's salts are a test key's). The JSON names every private byte by its ``[offset, length]``
in the payload and holds only public structure, hiding commitments and paths beside it.

**The document.**

* ``window``: the window's run root, registration digest, circuit record SHA-512, partition and program digests, scheme,
  population and leaf layer (``registration.json`` / ``window.json``), and ``roots``: every port root a unit's opening
  reaches, ``{port: {root, leaves, domain, schema}}``, each the registration's root under the window-domain rule
  (`verity.ml.commitments.window`).
* ``target``: the card's binding of the templates' dot, ``{binding, card, dot}`` (`BINDINGS`: ``GEMM_SM90_S8`` for the
  H200's INT8 and ``GEMM_HOPPER`` for its BF16, ``GEMM_SM120_S8`` for the RTX PRO 6000). Every unit's template binds that
  dot, as circuits resolves the binding.
* ``draw``: the drawn units (global indices), the law, and ``source``: ``live`` (the proof service's draw of record, with
  ``receipt``, the challenger's live-coin receipt, which a live draw must carry) or ``fixture`` (chosen by hand). The law is
  the window's registered one, fixed before any coin, and a live draw's ``record`` (the verifier's draw object, when it
  carries one) is a draw under it over the registered population, of exactly the bundle's units (`check` against the
  registration).
* ``weights``: the registered values the units read (core's ``one_stage.registered``, as CMP-95 interface 1 registers the
  seeded weights: ``registered.domain(program, value, words, leaves, word_bits)``, rows in ``hm96-sha512/row/v2``):
  ``{program, values: [{value, root, leaves, words, word_bits, domain, row_schema}]}``.
* ``commitment``: the serving steps' commitment, CMP-95 interface 3 (kernels' commit call, `commit_slots`: step tree over
  slot trees over unit trees over rows, only each unit's outputs committed, one row ``hm96-sha512/row/v2`` under the step's
  salt key, C-Flock's ``hm96_salt`` with the slot as nonce). ``{format, reference, context, layout, layout_digest, slots,
  blocks: [{block, root, domain, steps: [{step, root, domain}]}]}``: the run's 64-byte context, the step layout (groups of
  units of rows of byte ranges, `layout_json`), and per server-time block every step's root (what went to the challenger
  before the step's live coin) under this bundle's block tree over them (`block_domain`).
* ``members``: per member with a drawn unit, what C-Flock's ``--zk`` session reads today (`examples/one-stage/a5.py`): its
  key, template, base and count, the drawn units' local indices and the session's subset draw over them, its header's
  ``frame_v3`` entry, its public body's SHA-512 and row schemas. The prover's input file is
  ``flock_inputs.flock_input_files(window, k, circuit, out, write=("inst",))`` under the circuit the verifier composed for
  the template (``compose_circuit.py``, bound to ``circuit.partition`` and ``circuit.program``).
* ``units``: per drawn unit ``{unit, coords, template, member, rows}``. ``coords`` is ``{block, step, slot, layer, op,
  unit}`` (``unit``: the coordinate within its op at that slot and step). ``template`` is ``{id, definition, binding,
  rows, outs}``: the descriptor id, the Definition's name, its binding statics as JSON (`interface._bindings`) and its bit
  rows. ``rows`` is each port's row in the member's port order, inputs then outputs: ``{port, kind: in|out, schema, words,
  row: [offset, length], salt: [offset, length], string, window: {port, leaf, path}, slot | null, registered | null}``.
  ``string`` is the row's public ``b || c`` (hex) and every row opens against its window port's registered root.
  Each row opens against exactly one commitment:

  - ``slot``: the step commitment. ``{block, step, slot, unit, rank, path, slot_path, step_path, producer}``: ``path`` the
    row's frame-v3 siblings in its unit's tree (leaf level first, hex), ``slot_path`` the unit root's opening in the slot
    tree and ``step_path`` the slot root's in the step tree (`row_table.RootOpening` JSON). An output row opens in the
    unit's own commitment unit (``producer`` null); an input row opens where its producer committed it as an output, and
    ``producer`` names that unit's coordinates ``{block, step, slot, layer, op, unit}`` (``unit`` its index in the layout).
    Every opening is in the unit's own slot and step: one request's replay opens a drawn unit.
  - ``registered``: a registered read (the weights), committed under the registered row's salt, opening against the
    value's root: ``{value, position, path}``. It is in no step commitment.

  An output row (``kind: out``) is the unit's committed output: a value, in the payload only.

**The payload** is every drawn unit's rows in bundle order, each row's words (u16 little-endian) then its 192-byte salt.

`check` is the consumer's check of a bundle from its bytes alone: the payload's digest, every string recomputed from its row
and salt, every path to its root, every root's domain recomputed from public data (the window's from the registration's
fields, a registered value's from its entry, the commitment's from its context, layout and coordinates), the draw's receipt
when live, the target's dot in every template, and each unit's Definition replayed on its input rows against its output rows.
`build` writes a bundle from a window ``Window.write`` wrote and the prover's step records (`Commitment`).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree, merkle_depth
from verity.ml.commitments.row_table import ROOTS_SCHEMA, RootOpening, RootTree, verify_roots
from verity.ml.commitments.window import window_domain
from verity.ml.windows import interface as IF, rows as CS
from verity_e2e_demo import commit_slots as S

FORMAT = "verity/drawn-unit-bundle/v2"
COMMITMENT = "verity/commit-slots/v1"
REFERENCE = {"path": "benchmarks/torch_gpu/commit_slots", "commit": "05038d5ab4d481218df7443d3ff9fe661a9f5256", "pr": 4469}
BLOCK_TAG = "verity/drawn-unit-bundle/block/v1"
PAYLOAD_LAYOUT = "each drawn unit in bundle order, each row in its member's port order: the row's u16 words little-endian, then its 192-byte salt"
INST = hm96.SHA512
SALT = INST.salt_bytes
COORDS = ("block", "step", "slot", "layer", "op", "unit")
#: a card's binding of the templates' dot, by name: (card, the binding's statics as circuits resolves them)
BINDINGS = {"GEMM_SM120_S8": ("sm_120", G.GEMM_SM120_S8), "GEMM_SM90_S8": ("sm_90", G.GEMM_SM90_S8),
            "GEMM_HOPPER": ("sm_90", G.GEMM_HOPPER)}

Coords = Callable[[Any, int], dict[str, Any]]
Reads = Callable[[Mapping[str, Any], str, str], "tuple[str, int] | None"]
#: a unit's port's row in the step commitment: (layout unit, rank, the producer's coordinates or None for the unit's own)
Locate = Callable[[Mapping[str, Any], str, str], "tuple[int, int, dict[str, Any] | None] | None"]


@dataclass(frozen=True)
class StepRecord:
    """What the prover keeps of one step's commitment: its salt key and every slot's rows, in layout order."""

    salt_key: bytes
    rows: tuple[tuple[bytes, ...], ...]


@dataclass(frozen=True)
class Commitment:
    """The run's step commitments, the prover's side: its context, step layout, slots, and each ``(block, step)``'s record."""

    context: bytes
    layout: S.Layout
    slots: int
    steps: Mapping[tuple[int, int], StepRecord]


def target(binding: str) -> dict[str, Any]:
    """The ``target`` of a bundle whose templates bind the dot of ``binding`` (`BINDINGS`)."""
    if binding not in BINDINGS:
        raise ValueError(f"{binding}: not one of {sorted(BINDINGS)}")
    card, statics = BINDINGS[binding]
    return {"binding": binding, "card": card, "dot": statics["DOT"].id}


def torch_loop_coords(label: Mapping[str, Any], offset: int, block: int = 0) -> dict[str, Any]:
    """`integrations/torch_loop`'s unit labels (``{request: "r<j>", step, layer, op}``, one per linear's run of units) as
    coordinates: slot j, and ``unit`` the output coordinate within the run."""
    return {"block": block, "step": int(label["step"]), "slot": int(str(label["request"]).lstrip("r")),
            "layer": int(label["layer"]), "op": str(label["op"]), "unit": int(offset)}


def block_domain(context: bytes, layout: S.Layout, block: int, steps: Sequence[int]) -> CommitmentDomain:
    """This bundle's tree over one server-time block's step roots, in ascending step order."""
    return CommitmentDomain(identity_digest_sha512(BLOCK_TAG, {"context": context.hex(), "layout": S.layout_digest(layout).hex(),
                                                               "block": block, "steps": list(steps), "schema": ROOTS_SCHEMA}),
                            -1, RangeIndexedDomain(0, len(steps)), hash="sha512")


def layout_json(layout: S.Layout) -> dict[str, Any]:
    return {"name": layout.name, "groups": [{"sources": list(g.sources),
                                             "units": [[[[p.source, p.offset, p.nbytes] for p in r.parts] for r in u] for u in g.units]}
                                            for g in layout.groups]}


def layout_of_json(doc: Mapping[str, Any]) -> S.Layout:
    return S.Layout(str(doc["name"]), tuple(S.Group(tuple(int(n) for n in g["sources"]),
                                                    tuple(tuple(S.Row(tuple(S.Part(*map(int, p)) for p in r)) for r in u)
                                                          for u in g["units"]))
                                            for g in doc["groups"]))


def _leaf_value(bc: bytes) -> bytes:
    return INST.tree_leaf(INST.default_key, bc)


def _path(tree: MerkleTree, rank: int) -> list[str]:
    return [p.hex() for p in tree.open(rank).path]


def _fold(domain: CommitmentDomain, rank: int, schema: str, value: bytes, path: Sequence[str]) -> bytes | None:
    """The root ``value`` at ``rank`` reaches through ``path`` under ``domain``, or None for a path of the wrong depth."""
    if not 0 <= rank < domain.count or len(path) != merkle_depth(domain.count):
        return None
    digest, i = domain.leaf(rank, rank, schema, value), rank
    for level, sib in enumerate(path):
        s = bytes.fromhex(sib)
        digest = domain.node(level, i >> 1, *((digest, s) if i % 2 == 0 else (s, digest)))
        i >>= 1
    return digest


def string_of(schema: str, row: bytes, salt: bytes) -> bytes:
    """A row's ``b || c`` from its u16 words' bytes and its salt, under its row schema (`rows._inner`)."""
    return INST.commit_string(INST.default_key, CS._inner(schema, len(row) // 2)(row), salt)


def _template(tid: str) -> dict[str, Any]:
    m = IF._ID.fullmatch(tid)
    if m is None:
        raise ValueError(f"{tid}: not a descriptor id")
    lay = IF.layout(IF.resolve(tid))
    return {"id": tid, "definition": m[1], "binding": IF._bindings(m[2]) if m[2] else {},
            "rows": [list(r) for r in lay.rows], "outs": [list(o) for o in lay.outs]}


def _dot_of(template: Mapping[str, Any]) -> str | None:
    dot = template["binding"].get("DOT")
    return dot.get("fn") if isinstance(dot, Mapping) else dot


def _draw(draw: Mapping[str, Any] | None, units: list[int]) -> dict[str, Any]:
    d = {"source": "fixture", **dict(draw or {}), "units": units}
    if d["source"] not in ("live", "fixture"):
        raise ValueError(f"draw source {d['source']!r}: live or fixture")
    if d["source"] == "live" and not d.get("receipt"):
        raise ValueError("a live draw carries the challenger's live-coin receipt (draw.receipt)")
    return d


class _Window:
    """A written window's files, read once: members, labels, bodies."""

    def __init__(self, d: Path) -> None:
        self.dir = d
        self.win = json.loads((d / "window.json").read_text())
        self.rec = json.loads((d / "registration.json").read_text())
        self.index = json.loads((d / "index.json").read_text())
        self.members = self.win["members"]
        for m in self.members:
            if m.get("shared_rows") or m.get("row_links") or m.get("wired"):
                raise ValueError(f"{m['name']}: a shared-row, row-table or wired member; the bundle reads members of whole rows")
        self._spans = _run_starts(self.index)
        self._lo = [a for a, _, _, _ in self._spans]
        self._pub: dict[int, bytes] = {}
        self._priv: dict[int, bytes] = {}

    def member_of(self, u: int) -> int:
        for k, m in enumerate(self.members):
            if m["base"] <= u < m["base"] + m["n"]:
                return k
        raise ValueError(f"unit {u} is in no member of the window's {self.win['population']}")

    def label(self, u: int) -> tuple[Any, int]:
        j = bisect.bisect_right(self._lo, u) - 1
        if j < 0 or not u < self._spans[j][1]:
            raise ValueError(f"unit {u} has no label in index.json")
        return self._spans[j][2], u - self._spans[j][3]

    def public(self, k: int) -> bytes:
        if k not in self._pub:
            self._pub[k] = (self.dir / f"m{k}-public.bin").read_bytes()
        return self._pub[k]

    def private(self, k: int) -> bytes:
        if k not in self._priv:
            self._priv[k] = (self.dir / f"m{k}-private.bin").read_bytes()
        return self._priv[k]

    def rows_of(self, k: int, i: int) -> list[tuple[str, str, int, bytes, bytes, bytes]]:
        """Member ``k``'s instance ``i``: each port's ``(port, schema, words, row, salt, string)``."""
        m = self.members[k]
        ports, n = m["ports"], m["n"]
        total = sum(w for _, w in ports)
        priv, pub = self.private(k), self.public(k)
        rows_at, salts_at = 2 * total * i, 2 * total * n + SALT * len(ports) * i
        out, wo = [], 0
        for j, (p, w) in enumerate(ports):
            row = priv[rows_at + 2 * wo:rows_at + 2 * (wo + w)]
            salt = priv[salts_at + SALT * j:salts_at + SALT * (j + 1)]
            bc = pub[CS.BC_BYTES * (len(ports) * i + j):CS.BC_BYTES * (len(ports) * i + j + 1)]
            out.append((p, m["frame_v3"]["schemas"][p], w, row, salt, bc))
            wo += w
        return out

    def strings(self, k: int, p: str) -> list[bytes]:
        m = self.members[k]
        j = [q for q, _ in m["ports"]].index(p)
        P, pub = len(m["ports"]), self.public(k)
        return [pub[CS.BC_BYTES * (P * i + j):CS.BC_BYTES * (P * i + j + 1)] for i in range(m["n"])]


def _run_starts(index: Mapping[str, Any]) -> list[tuple[int, int, Any, int]]:
    """Every labelled span of ``index.json`` as ``(lo, hi, label, start)``, ``start`` where its recorded run begins: members
    cut a run, so a span may start inside one, and a run's spans are the ones with its label (a label names one run)."""
    spans = sorted((a, b, x) for m in index["members"] for a, b, x in m["labels"])
    first: dict[str, int] = {}
    for a, _, x in spans:
        first.setdefault(json.dumps(x, sort_keys=True), a)
    return [(a, b, x, first[json.dumps(x, sort_keys=True)]) for a, b, x in spans]


class _Steps:
    """The step commitments a bundle opens, rebuilt from the prover's records: slot subtrees, step trees, block trees."""

    def __init__(self, com: Commitment) -> None:
        self.com = com
        self._slots: dict[tuple[int, int], list[S.SlotCommit]] = {}
        self._steps: dict[tuple[int, int], RootTree] = {}

    def slots(self, b: int, s: int) -> list[S.SlotCommit]:
        if (b, s) not in self._slots:
            rec = self.com.steps.get((b, s))
            if rec is None:
                raise ValueError(f"block {b} step {s}: no step record")
            if len(rec.rows) != self.com.slots:
                raise ValueError(f"block {b} step {s}: {len(rec.rows)} slots' rows for {self.com.slots} slots")
            self._slots[b, s] = [S.commit_slot(self.com.context, self.com.layout, rec.salt_key, j, rows)
                                 for j, rows in enumerate(rec.rows)]
        return self._slots[b, s]

    def step(self, b: int, s: int) -> RootTree:
        if (b, s) not in self._steps:
            self._steps[b, s] = S.step_tree(self.com.context, self.com.layout, s, [sc.root for sc in self.slots(b, s)])
        return self._steps[b, s]

    def blocks(self) -> list[dict[str, Any]]:
        out = []
        for b in sorted({b for b, _ in self.com.steps}):
            steps = sorted(s for bb, s in self.com.steps if bb == b)
            roots = [self.step(b, s).root for s in steps]
            dom = block_domain(self.com.context, self.com.layout, b, steps)
            out.append({"block": b, "root": RootTree(dom, roots).root.hex(), "domain": dom.domain_id.hex(),
                        "steps": [{"step": s, "root": r.hex(), "domain": self.step(b, s).domain.domain_id.hex()}
                                  for s, r in zip(steps, roots, strict=True)]})
        return out


def build(window_dir: str | Path, drawn: Sequence[int], *, commitment: Commitment, locate: Locate, binding: str,
          coords: Coords = torch_loop_coords, reads: Reads | None = None,
          weights: Mapping[str, REG.Registered] | None = None, draw: Mapping[str, Any] | None = None
          ) -> tuple[dict[str, Any], bytes]:
    """The bundle of ``drawn`` (global unit indices) from the window in ``window_dir`` (what ``Window.write`` wrote, prover's
    files included): ``(document, payload)``.

    ``commitment``: the run's step commitments, the prover's records. ``locate(coords, template, port)``: where a unit's port's
    row is in the step commitment, ``(layout unit, rank, producer)`` (``producer`` the producing unit's coordinates for an
    input, None for the unit's own output), or None for a registered read. ``binding``: the card's dot binding (`BINDINGS`),
    which every template must bind. ``coords(label, offset)``: a unit's coordinates from its label and its offset in its
    labelled run. ``reads(coords, template, port)``: the ``(value, position)`` a unit's port reads of a registered value;
    ``weights``: those values, the prover's side. ``draw``: the draw's ``law``, ``source`` and ``receipt`` (default
    ``{"source": "fixture"}``; a live draw without its receipt is refused).

    Refused (ValueError) when a row is neither in the step commitment nor a registered read, when a window string is not
    the step commitment's or the registered value's (a row committed under other salts has no path to that root), or when
    a template binds another dot than ``binding``'s."""
    w = _Window(Path(window_dir))
    run, part = w.win["run"], w.win["partition"]
    reads = reads or (lambda c, t, p: None)
    weights = dict(weights or {})
    drawn = [int(u) for u in drawn]
    if len(set(drawn)) != len(drawn):
        raise ValueError("a unit is drawn once")
    tgt = target(binding)
    doc_draw = _draw(draw, drawn)
    com, steps, layout = commitment, _Steps(commitment), commitment.layout

    roots = {r["port"]: r for r in w.rec["roots"]}
    port_trees: dict[tuple[int, str], MerkleTree] = {}
    payload = bytearray()
    units, members, used_roots, used_values = [], {}, {}, set()
    for u in drawn:
        k = w.member_of(u)
        m, i = w.members[k], u - w.members[k]["base"]
        c = coords(*w.label(u))
        tpl = _template(m["template"])
        if _dot_of(tpl) not in (None, tgt["dot"]):
            raise ValueError(f"unit {u}: its template binds {_dot_of(tpl)}, not {binding}'s {tgt['dot']}")
        rows = []
        for j, (p, schema, words, row, salt, bc) in enumerate(w.rows_of(k, i)):
            if string_of(schema, row, salt) != bc:
                raise ValueError(f"unit {u} {p}: its row and salt are not its committed string")
            name = f"{m['name']}/{p}"
            if (k, p) not in port_trees:
                ss = w.strings(k, p)
                dom = CommitmentDomain(bytes.fromhex(m["frame_v3"]["bindings"][p]), CS.OWNER, RangeIndexedDomain(0, len(ss)),
                                       hash="sha512")
                port_trees[k, p] = MerkleTree(dom, {r: _leaf_value(s) for r, s in enumerate(ss)}, lambda _r, schema=schema: schema)
                if port_trees[k, p].commitment.root.hex() != roots[name]["root"]:
                    raise ValueError(f"{name}: its strings are not the registered root")
            used_roots[name] = {"root": roots[name]["root"], "leaves": roots[name]["leaves"], "domain": roots[name]["domain"],
                                "schema": schema}
            entry: dict[str, Any] = {"port": p, "kind": "in" if j < m["input_ports"] else "out", "schema": schema, "words": words,
                                     "row": [len(payload), len(row)], "salt": [len(payload) + len(row), SALT], "string": bc.hex(),
                                     "window": {"port": name, "leaf": i, "path": _path(port_trees[k, p], i)},
                                     "slot": None, "registered": None}
            payload += row + salt
            at, hit = locate(c, m["template"], p), reads(c, m["template"], p)
            if (at is None) == (hit is None):
                raise ValueError(f"unit {u} {p}: in the step commitment or a registered read, exactly one")
            if hit is not None:
                value, pos = hit
                reg = weights[value]
                if reg.bcs[pos] != bc:
                    raise ValueError(f"unit {u} {p}: committed under its own salt, not {value} row {pos}'s registered one")
                entry["registered"] = {"value": value, "position": pos, "path": reg.path(pos).hex()}
                used_values.add(value)
            else:
                lu, rank, producer = at
                if (producer is None) != (entry["kind"] == "out"):
                    raise ValueError(f"unit {u} {p}: an output opens in its own unit, an input where its producer committed it")
                b, s, slot = c["block"], c["step"], c["slot"]
                sc = steps.slots(b, s)[slot]
                g = layout.row_base[lu] + rank
                if len(row) != layout.units[lu][rank].nbytes or sc.strings[g] != bc:
                    raise ValueError(f"unit {u} {p}: not the step commitment's row {rank} of unit {lu} (other bytes or salts)")
                entry["slot"] = {"block": b, "step": s, "slot": slot, "unit": lu, "rank": rank,
                                 "path": _path(sc.units[lu], rank), "slot_path": sc.tree.open([lu]).to_json(),
                                 "step_path": steps.step(b, s).open([slot]).to_json(),
                                 "producer": None if producer is None else {**{x: c[x] for x in ("block", "step", "slot")},
                                                                            **producer}}
            rows.append(entry)
        units.append({"unit": u, "coords": c, "template": tpl, "member": {"k": k, "local": i}, "rows": rows})
        mm = members.setdefault(k, {"k": k, "name": m["name"], "template": m["template"], "base": m["base"], "n": m["n"],
                                    "local": [], "frame_v3": m["frame_v3"], "public_sha512": m["public_sha512"],
                                    "row_schemas": m.get("row_schemas")})
        mm["local"].append(i)
    for mm in members.values():
        mm["local"].sort()
        mm["draw"] = {"law": "subset", "population": mm["n"], "k": len(mm["local"]), "units": mm["local"]}

    progs = {weights[v].program for v in used_values}
    if len(progs) > 1:
        raise ValueError(f"the drawn units read registered values of {len(progs)} programs; a bundle names one")
    pay = bytes(payload)
    doc = {"format": FORMAT,
           "payload": {"bytes": len(pay), "sha512": hashlib.sha512(pay).hexdigest(), "layout": PAYLOAD_LAYOUT},
           "window": {"run": run, "registration": w.win["registration_digest"], "circuit": w.rec["window"].get("circuit"),
                      "partition": w.win["partition_digest"], "program": part["program"], "scheme": w.win["scheme"],
                      "population": w.win["population"], "leaf_layer": w.win["leaf_layer"], "roots": used_roots},
           "target": tgt,
           "draw": doc_draw,
           "weights": {"program": next(iter(progs)) if progs else None,
                       "values": [{**weights[v].entry(), "row_schema": weights[v].schema} for v in sorted(used_values)]},
           "commitment": {"format": COMMITMENT, "reference": REFERENCE, "context": com.context.hex(), "layout": layout_json(layout),
                          "layout_digest": S.layout_digest(layout).hex(), "slots": com.slots, "blocks": steps.blocks()},
           "members": [members[k] for k in sorted(members)],
           "units": units}
    return doc, pay


def _elements(row: bytes, width: int, shape: tuple[int, ...]) -> np.ndarray:
    a = np.frombuffer(row, dtype=f"<u{width // 8}").astype(f"u{width // 8}")
    return a.reshape((1, *shape)) if shape else a.reshape(1)


def replay(unit: Mapping[str, Any], payload: bytes) -> bool:
    """Whether the unit's committed output rows are its Definition's outputs on its committed input rows."""
    fn = IF.resolve(unit["template"]["id"])
    params, ret = fn.signature()
    lay = IF.layout(fn)
    rows = {r["port"]: payload[r["row"][0]:r["row"][0] + r["row"][1]] for r in unit["rows"]}
    ins = []
    for (name, ty), width in zip(params, lay.widths):
        if ty.width_set() == {1}:
            raise ValueError(f"{name}: a Boolean-profile port; the bundle replays word ports")
        n = len(ty.leaf_widths())
        ins.append(_elements(rows[name], width, () if n == 1 and not getattr(ty, "shape", None) else (n,)))
    want = np.asarray(evaluate_batch(fn, *ins)).reshape(1, -1).astype(np.uint64)
    got = []
    for (o, words), f, width in zip(lay.outs, IF._fields(ret), lay.widths[len(params):]):
        leaves = len(f.leaf_widths())
        got.append(np.frombuffer(rows[o][:leaves * width // 8], dtype=f"<u{width // 8}").astype(np.uint64))
    got_a = np.concatenate(got).reshape(1, -1)
    return got_a.shape == want.shape and bool((got_a == want).all())


def _check_commitment(doc: Mapping[str, Any], bad: dict[str, str]) -> tuple[Any, ...] | None:
    """The commitment section's layout, context and block trees recomputed: ``(context, layout, slots, step roots by (block,
    step))``, or None (with ``bad`` filled) when its layout or context can't be read."""
    com = doc["commitment"]
    if com.get("format") != COMMITMENT:
        bad["commitment"] = f"not {COMMITMENT}"
        return None
    try:
        layout, context = layout_of_json(com["layout"]), bytes.fromhex(com["context"])
        S.layout_digest(layout)
        S.slot_domain(context, layout)
    except (ValueError, KeyError, TypeError) as e:
        bad["commitment"] = f"its layout or context is not one: {e}"
        return None
    if S.layout_digest(layout).hex() != com["layout_digest"]:
        bad["commitment"] = "its layout digest is not its layout's"
    step_roots = {}
    for b in com["blocks"]:
        steps = [s["step"] for s in b["steps"]]
        dom = block_domain(context, layout, b["block"], steps)
        roots = [bytes.fromhex(s["root"]) for s in b["steps"]]
        if dom.domain_id.hex() != b["domain"] or RootTree(dom, roots).root.hex() != b["root"]:
            bad[f"block:{b['block']}"] = "its root is not the block tree's over its steps' roots"
        for s, r in zip(b["steps"], roots, strict=True):
            if S.step_domain(context, layout, s["step"], com["slots"]).domain_id.hex() != s["domain"]:
                bad[f"step:{b['block']}/{s['step']}"] = "its domain is not the commit call's"
            step_roots[b["block"], s["step"]] = r
    return context, layout, com["slots"], step_roots


def _check_slot(r: Mapping[str, Any], unit: Mapping[str, Any], row: bytes, bc: bytes, com: tuple[Any, ...]) -> str | None:
    """Why a row's step-commitment opening fails, or None."""
    context, layout, slots, step_roots = com
    s, c = r["slot"], unit["coords"]
    if (s["block"], s["step"], s["slot"]) != (c["block"], c["step"], c["slot"]):
        return "it opens outside its unit's slot and step"
    if (s["producer"] is None) != (r["kind"] == "out"):
        return "an output opens in its own unit, an input names its producer"
    if s["producer"] is not None and ({x: s["producer"].get(x) for x in ("block", "step", "slot", "unit")}
                                      != {x: s[x] for x in ("block", "step", "slot", "unit")} or not {"layer", "op"} <= set(s["producer"])):
        return "its producer's coordinates are not where it opens"
    lu, rank = s["unit"], s["rank"]
    if not (0 <= lu < layout.unit_count and 0 <= rank < len(layout.units[lu])) or len(row) != layout.units[lu][rank].nbytes:
        return "not a row of the layout"
    step_root = step_roots.get((s["block"], s["step"]))
    if step_root is None or not 0 <= s["slot"] < slots:
        return "its step is not in the commitment"
    try:
        sp, tp = RootOpening.of_json(s["slot_path"]), RootOpening.of_json(s["step_path"])
    except (ValueError, KeyError, TypeError) as e:
        return f"its openings are not ones: {e}"
    unit_root = _fold(S.unit_domain(context, layout, lu), rank, S.ROW_SCHEMA, _leaf_value(bc), s["path"])
    sd, td = S.slot_domain(context, layout), S.step_domain(context, layout, s["step"], slots)
    if (unit_root is None or sp.indices != (lu,) or sp.roots[0] != unit_root or tp.indices != (s["slot"],)
            or not verify_roots(tp.roots[0], sd.domain_id, sd.count, sp) or not verify_roots(step_root, td.domain_id, td.count, tp)):
        return "its paths do not reach its unit, slot and step roots"
    return None


def check(doc: Mapping[str, Any], payload: bytes, *, registration: Mapping[str, Any] | None = None,
          replay_units: bool = True) -> dict[str, str]:
    """The consumer's check of a bundle (module docstring): ``{key: why}``, empty when it holds. ``registration``: the
    window's ``registration.json``, whose roots and law the bundle's must be."""
    bad: dict[str, str] = {}
    if doc.get("format") != FORMAT:
        return {"format": f"not {FORMAT}"}
    if len(payload) != doc["payload"]["bytes"] or hashlib.sha512(payload).hexdigest() != doc["payload"]["sha512"]:
        return {"payload": "its bytes are not the ones the bundle names"}
    d = doc["draw"]
    if d.get("source") not in ("live", "fixture") or (d["source"] == "live" and not d.get("receipt")):
        bad["draw"] = "a live draw carries the challenger's live-coin receipt, and a draw is live or fixture"
    try:
        tgt = target(doc["target"]["binding"])
    except ValueError as e:
        tgt = None
        bad["target"] = str(e)
    if tgt is not None and tgt != doc["target"]:
        bad["target"] = f"not {doc['target']['binding']}'s card and dot as circuits resolves it ({tgt})"
    win = doc["window"]
    for port, r in win["roots"].items():
        if window_domain(win["program"], win["partition"], port, r["schema"], r["leaves"], win["run"]) != r["domain"]:
            bad[f"root:{port}"] = "its domain is not the window-domain rule's over the registration's fields"
    if registration is not None:
        law = registration["law"]
        if d.get("law") != law or (law.get("law") == "subset" and len(d["units"]) != law.get("k")):
            bad["draw:law"] = f"the draw is not one under the registered law {law}"
        if "record" in d and not (D.check_draw(d["record"], registration, law).ok
                                  and d["record"].get("units") == sorted(d["units"])):
            bad["draw:record"] = "the draw's record is not a draw of its units under the registered law and population"
        reg_roots = {r["port"]: r for r in registration["roots"]}
        for port, r in win["roots"].items():
            if (reg_roots.get(port) or {}).get("root") != r["root"]:
                bad[f"root:{port}"] = "not the registration's root"
    values = {v["value"]: v for v in doc["weights"]["values"]}
    for name, v in values.items():
        dom = REG.domain(doc["weights"]["program"], name, v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
        if dom.domain_id.hex() != v["domain"]:
            bad[f"value:{name}"] = "its domain is not the registered-value rule's over its entry"
    com = _check_commitment(doc, bad)
    members = {m["k"]: m for m in doc["members"]}
    for unit in doc["units"]:
        u = unit["unit"]
        mm = members.get(unit["member"]["k"])
        if mm is None or mm["base"] + unit["member"]["local"] != u or unit["member"]["local"] not in mm["local"]:
            bad[f"unit:{u}"] = "its member does not hold it at its local index"
            continue
        if tgt is not None and _dot_of(unit["template"]) not in (None, tgt["dot"]):
            bad[f"unit:{u}"] = f"its template binds another dot than the target's {tgt['dot']}"
            continue
        outs = {r["slot"]["unit"] for r in unit["rows"] if r["kind"] == "out" and r["slot"] is not None}
        if len(outs) > 1:
            bad[f"unit:{u}"] = "its outputs open in more than one commitment unit"
        for r in unit["rows"]:
            key = f"unit:{u}/{r['port']}"
            row = payload[r["row"][0]:r["row"][0] + r["row"][1]]
            salt = payload[r["salt"][0]:r["salt"][0] + r["salt"][1]]
            bc = bytes.fromhex(r["string"])
            if len(row) != 2 * r["words"] or string_of(r["schema"], row, salt) != bc:
                bad[key] = "its row and salt are not its string"
                continue
            root = win["roots"].get(r["window"]["port"])
            if root is None:
                bad[key] = "its window port names no root"
                continue
            dom = CommitmentDomain(_binding(win, r["window"]["port"], r["schema"], root["leaves"]), CS.OWNER,
                                   RangeIndexedDomain(0, root["leaves"]), hash="sha512")
            if _fold(dom, r["window"]["leaf"], r["schema"], _leaf_value(bc), r["window"]["path"]) != bytes.fromhex(root["root"]):
                bad[key] = "its window path does not reach its port's registered root"
            if (r["slot"] is None) == (r["registered"] is None):
                bad[key] = "a row opens against the step commitment or a registered value, exactly one"
            elif r["registered"] is not None:
                v = values.get(r["registered"]["value"])
                d = v and REG.domain(doc["weights"]["program"], v["value"], v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
                if v is None or not REG.verify_read(d, bytes.fromhex(v["root"]), r["registered"]["position"], bc,
                                                    bytes.fromhex(r["registered"]["path"]), v.get("row_schema", REG.SCHEMA)):
                    bad[key] = "its registered path does not reach the value's root"
            elif com is None:
                bad[key] = "the commitment it opens against can't be read"
            elif r["schema"] != S.ROW_SCHEMA:
                bad[key] = f"the step commitment's rows are {S.ROW_SCHEMA}"
            elif (why := _check_slot(r, unit, row, bc, com)) is not None:
                bad[key] = why
        if replay_units and f"unit:{u}" not in bad and not any(k.startswith(f"unit:{u}/") for k in bad):
            try:
                if not replay(unit, payload):
                    bad[f"unit:{u}"] = "its committed outputs are not its Definition's on its committed inputs"
            except ValueError as e:
                bad[f"unit:{u}"] = f"not replayed: {e}"
    return bad


def _binding(win: Mapping[str, Any], port: str, schema: str, leaves: int) -> bytes:
    return identity_digest_sha512(CS.DOMAIN_TAG, {"program": win["program"], "partition": win["partition"], "port": port,
                                                  "schema": schema, "leaves": int(leaves), "run": win["run"]})


def write(doc: Mapping[str, Any], payload: bytes, out_dir: str | Path, name: str = "bundle") -> tuple[Path, Path]:
    """``<name>.json`` and ``<name>.bin`` (mode 0600: values and salts) in ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pj, pb = d / f"{name}.json", d / f"{name}.bin"
    pj.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    fd = os.open(pb, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return pj, pb


def read(path: str | Path) -> tuple[dict[str, Any], bytes]:
    """A bundle written by `write`: its document and its payload (``<name>.bin`` beside ``<name>.json``)."""
    p = Path(path)
    return json.loads(p.read_text()), p.with_suffix(".bin").read_bytes()


# --- the torch loop's fixture ------------------------------------------------------------------------------------------

LINEAR_NAMES = ("wup", "wdown")
FIXTURE_SALT_KEY = hashlib.sha256(b"verity/drawn-unit-bundle/fixture-salt-key (a test key)").digest()
CONTEXT_TAG = "verity/drawn-unit-bundle/torch-loop/context/v1"
RUN_ID = "drawn-unit-bundle-fixture"


def torch_loop_weights(model: Any, salt_key: bytes, program: str) -> dict[str, REG.Registered]:
    """The torch loop's linears as registered values, named as CMP-95 interface 1 names a decoder's (``l<i>.wup``,
    ``l<i>.wdown`` and their ``.scale``): a weight row is output channel c's s8 codes as 64 u16 words, a scale row its f32
    bits; row/v2 rows; row i's salt ChaCha20 under ``salt_key`` at the value's index as nonce (`rows.salts`)."""
    out: dict[str, REG.Registered] = {}
    for layer, lin in enumerate(model.layers):
        codes = lin.wq.contiguous().numpy().view(np.uint8)
        words = codes.reshape(codes.shape[0], -1).view("<u2").astype(np.uint16)
        scales = lin.sw.contiguous().numpy().view(np.uint32)
        for name, rows, bits in ((f"l{layer}.{LINEAR_NAMES[layer]}", [list(map(int, r)) for r in words], 16),
                                 (f"l{layer}.{LINEAR_NAMES[layer]}.scale", [[int(s)] for s in scales], 32)):
            ys = CS.salts(salt_key, len(out), len(rows))
            out[name] = REG.Registered.commit(program, name, rows, [ys[i * SALT:(i + 1) * SALT] for i in range(len(rows))],
                                              word_bits=bits, schema=REG.SCHEMA_V2)
    return out


def torch_loop_reads(c: Mapping[str, Any], template: str, port: str) -> tuple[str, int] | None:
    """The torch loop's registered reads: a unit's ``wrow`` and ``sw`` are its layer's weight and scale rows at its output
    coordinate."""
    if port == "wrow":
        return f"l{c['layer']}.{LINEAR_NAMES[c['layer']]}", c["unit"]
    if port == "sw":
        return f"l{c['layer']}.{LINEAR_NAMES[c['layer']]}.scale", c["unit"]
    return None


def torch_loop_layout(h: int, layers: int) -> S.Layout:
    """The torch loop's step layout: a group a layer, its tensors the linear's input codes (``h`` bytes), their f32 scale and
    the linear's bf16 output (``2h`` bytes). Its units: the input quantizer (the codes row, then the scale row), then one unit
    per output coordinate (its 2-byte row), the proof block `ScaledMmS8Coordinate_v1` proves."""
    quant = (S.Row((S.Part(0, 0, h),)), S.Row((S.Part(1, 0, 4),)))
    group = S.Group((h, 4, 2 * h), (quant, *((S.Row((S.Part(2, 2 * c, 2),)),) for c in range(h))))
    return S.Layout(f"torch-loop/mlp-w8a8-h{h}/every-linear/v1", (group,) * layers)


def torch_loop_locate(h: int) -> Locate:
    """The torch loop's rows in its step layout: a unit's ``x`` and ``sx`` are its layer's quantizer's rows 0 and 1 (their
    producer), its ``o0`` its own coordinate's unit; ``wrow`` and ``sw`` are registered reads."""
    def locate(c: Mapping[str, Any], template: str, port: str) -> tuple[int, int, dict[str, Any] | None] | None:
        base = c["layer"] * (h + 1)
        producer = {"layer": c["layer"], "op": f"{c['op']}.input_quant", "unit": base}
        return {"x": (base, 0, producer), "sx": (base, 1, producer), "o0": (base + 1 + c["unit"], 0, None)}.get(port)
    return locate


def _step_key(salt_key: bytes, block: int, step: int) -> bytes:
    return hashlib.sha256(salt_key + f"/block/{block}/step/{step}".encode()).digest()


def torch_loop_window(out: str | Path, requests: Sequence[int] = (3, 7), steps: int = 1, cap: int = 8, tamper: bool = False,
                      salt_key: bytes = FIXTURE_SALT_KEY, law: str | None = None
                      ) -> tuple[Any, dict[str, REG.Registered], Commitment]:
    """The torch loop's window written to ``out`` with its step commitments (a test salt key a step, from ``salt_key``): every
    row a unit commits or reads in the step commitment carries that commitment's salt, and every weight row its registered
    value's, so each window string is the one its opening reaches: -> (the loop, its registered weights, its commitment).
    ``law``: the law it registers (`draw.parse_law`; default the fixture's, ``subset:len(FIXTURE_DRAW)``). Needs torch and
    `integrations/torch_loop` on the path."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R
    from verity_torch_loop import loop as L

    loop = L.TorchLoop(L.Model(), list(requests), steps, tamper, RUN_ID)
    win = loop.window(D.parse_law(law or f"subset:{len(FIXTURE_DRAW)}"), cap)
    h, program = loop.model.h, f"torch-loop/mlp-w8a8-h{loop.model.h}/seed{loop.model.seed}"
    weights = torch_loop_weights(loop.model, salt_key, program)
    layout, locate = torch_loop_layout(h, len(loop.model.layers)), torch_loop_locate(h)
    context = identity_digest_sha512(CONTEXT_TAG, {"run": loop.run_root, "program": program})
    spans, base = _run_starts(win.labels()), win._bases()
    lo_of = [a for a, _, _, _ in spans]

    def coords_of(u: int) -> dict[str, Any]:
        a, _, lab, start = spans[bisect.bisect_right(lo_of, u) - 1]
        return torch_loop_coords(lab, u - start)

    data, grids = [], {}
    for m in win.window.members:
        lay = IF.layout(win.circuit.template(m.template))
        got = win._recorded[m.template]
        ports = [np.concatenate([b.words[g] for b in got]) for g in range(len(lay.rows) + len(lay.outs))]
        lo = m.base - base[m.template]
        d = {"rows": {p: ports[g][lo:lo + m.n] for g, (p, _) in enumerate(lay.rows)},
             "outs": {p: ports[len(lay.rows) + j][lo:lo + m.n] for j, (p, _) in enumerate(lay.outs)}}
        data.append(d)
        for i in range(m.n):
            c = coords_of(m.base + i)
            for p, arr in (*d["rows"].items(), *d["outs"].items()):
                at = locate(c, m.template, p)
                if at is None:
                    continue
                g = layout.row_base[at[0]] + at[1]
                grid = grids.setdefault((c["block"], c["step"]), [[None] * layout.row_count for _ in requests])
                row = np.ascontiguousarray(arr[i]).astype("<u2").tobytes()
                if grid[c["slot"]][g] not in (None, row):
                    raise ValueError(f"block {c['block']} step {c['step']} slot {c['slot']} row {g}: read as two values")
                grid[c["slot"]][g] = row
    if any(r is None for grid in grids.values() for rows in grid for r in rows):
        raise ValueError("a row of the step layout has no unit committing or reading it")
    com = Commitment(context, layout, len(requests),
                     {k: StepRecord(_step_key(salt_key, *k), tuple(tuple(rows) for rows in grid)) for k, grid in grids.items()})

    salts_of = {}
    for m in win.window.members:
        lay = IF.layout(win.circuit.template(m.template))
        for p, _ in (*lay.rows, *lay.outs):
            ys = b""
            for u in range(m.base, m.base + m.n):
                c = coords_of(u)
                at, hit = locate(c, m.template, p), torch_loop_reads(c, m.template, p)
                if at is not None:
                    rec = com.steps[c["block"], c["step"]]
                    ys += S.row_salts(rec.salt_key, c["slot"], layout.row_base[at[0]] + at[1], 1)
                else:
                    value, pos = hit
                    ys += weights[value].salts[pos]
            salts_of[m.port(p)] = ys
    committed = CS.commit(win.window, data, salt_key=salt_key, salts_of=salts_of)
    win.write(committed, out, {"loop": "torch-loop"}, registrar=R)
    return loop, weights, com


FIXTURE_DRAW = (5, 200, 300, 511)


def fixture_bundle(scratch: str | Path) -> tuple[dict[str, Any], bytes]:
    """The fixture: the torch loop (requests 3 and 7, one step, width 128: 512 units in members of 8) written to ``scratch``,
    and the bundle of `FIXTURE_DRAW` (slot 0 layer 0, slot 0 layer 1, slot 1 layer 0, slot 1 layer 1), its templates on the
    RTX PRO 6000's dot (``GEMM_SM120_S8``) and its draw a fixture's. It is built, not tracked; the suite pins its digests
    (`digests`)."""
    _, weights, com = torch_loop_window(scratch)
    return build(scratch, FIXTURE_DRAW, commitment=com, locate=torch_loop_locate(128), binding="GEMM_SM120_S8",
                 reads=torch_loop_reads, weights=weights,
                 draw={"source": "fixture", "law": {"law": "subset", "k": len(FIXTURE_DRAW)}})


def digests(doc: Mapping[str, Any], payload: bytes) -> dict[str, str]:
    """The SHA-512s of a bundle's document as `write` writes it and of its payload."""
    return {"json": hashlib.sha512((json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()).hexdigest(),
            "bin": hashlib.sha512(payload).hexdigest()}


def fixture(out_dir: str | Path, scratch: str | Path) -> tuple[Path, Path]:
    """`fixture_bundle` written to ``out_dir`` as ``drawn-unit-bundle.{json,bin}``."""
    return write(*fixture_bundle(scratch), out_dir, "drawn-unit-bundle")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The drawn-unit replay bundle: check one, or write the torch loop's fixture.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("bundle")
    c.add_argument("--registration")
    f = sub.add_parser("fixture")
    f.add_argument("--out", required=True)
    f.add_argument("--scratch", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        doc, pay = read(a.bundle)
        reg = json.loads(Path(a.registration).read_text()) if a.registration else None
        bad = check(doc, pay, registration=reg)
        print(json.dumps({"units": len(doc["units"]), "ok": not bad, "bad": bad}, sort_keys=True))
        return 0 if not bad else 1
    print(json.dumps([str(p) for p in fixture(a.out, a.scratch)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
