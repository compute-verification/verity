"""The drawn-unit replay bundle (CMP-95 interface 5): everything the prover behind the firewall reads to prove the drawn units
of one window, one JSON document (``verity/drawn-unit-bundle/v1``) and one binary payload of the private bytes.

The bundle is the prover's: its payload holds values and salts, so it is written mode 0600, never published and never put in
the evidence store (a test fixture's salts are a test key's). The JSON names every private byte by its ``[offset, length]``
in the payload and holds only public structure, hiding commitments and paths beside it.

**The document.**

* ``window``: the window's run root, registration digest, circuit record SHA-512, partition and program digests, scheme,
  population and leaf layer (``registration.json`` / ``window.json``), and ``roots``: every port root a unit's opening
  reaches, ``{port: {root, leaves, domain, schema}}``, each the registration's root under the window-domain rule
  (`verity.ml.commitments.window`).
* ``draw``: the drawn units (global indices), the law, and ``source`` (``live``: the proof service's draw of record, with
  its receipt; ``fixture``: chosen by hand).
* ``weights``: the registered values the units read (core's ``one_stage.registered``, as CMP-95 interface 1 registers the
  seeded weights: ``registered.domain(program, value, words, leaves, word_bits)``, rows in ``hm96-sha512/row/v2``):
  ``{program, values: [{value, root, leaves, words, word_bits, domain, row_schema}]}``.
* ``slot_trees``: the per-step commitment, one subtree per sequence slot (CMP-95 interface 3, kernels'). Until kernels posts
  its tree this is the fixture shape ``verity/slot-tree/fixture-v0`` (`slot_domain`, `step_domain`, `block_domain`): per
  server-time block, per step, per slot, a ``frame-v3-sha512`` tree over the slot's committed rows at that step (every row
  of its units in unit order, then port order, registered reads left out, each leaf the row's ``hm96-sha512`` tree leaf
  under its row schema); a step's tree over its slots' roots and a block's tree over its steps' roots (leaf schema
  ``frame-v3-sha512/root/v1``). ``blocks: [{block, root, domain, steps: [{step, root, domain, slots: [{slot, root, domain,
  leaves}]}]}]``.
* ``members``: per member with a drawn unit, what C-Flock's ``--zk`` session reads today (`benchmarks/one_stage/a5.py`): its
  key, template, base and count, the drawn units' local indices and the session's subset draw over them, its header's
  ``frame_v3`` entry, its public body's SHA-512 and row schemas. The prover's input file is
  ``flock_inputs.flock_input_files(window, k, circuit, out, write=("inst",))`` under the circuit the verifier composed for
  the template (``compose_circuit.py``, bound to ``circuit.partition`` and ``circuit.program``).
* ``units``: per drawn unit ``{unit, coords, template, member, rows}``. ``coords`` is ``{block, step, slot, layer, op,
  unit}`` (``unit``: the coordinate within its op at that slot and step). ``template`` is ``{id, definition, binding,
  rows, outs}``: the descriptor id, the Definition's name, its binding statics as JSON (`interface._bindings`) and its bit
  rows. ``rows`` is each port's row in the member's port order, inputs then outputs: ``{port, kind: in|out, schema, words,
  row: [offset, length], salt: [offset, length], string, window: {port, leaf, path}, slot: {block, step, slot, leaf, path,
  slot_path, step_path} | null, registered: {value, position, path} | null}``. ``string`` is the row's public ``b || c``
  (hex); every ``path`` is the frame-v3 siblings, leaf level first, hex. A registered read (the weights) is committed under
  the registered row's salt, so its string is the registered one and it opens against the value's root, and it is in no slot
  tree. An output row (``kind: out``) is the unit's committed output: a value, in the payload only.

**The payload** is every drawn unit's rows in bundle order, each row's words (u16 little-endian) then its 192-byte salt.

`check` is the consumer's check of a bundle from its bytes alone: the payload's digest, every string recomputed from its row
and salt, every path to its root, every root's domain recomputed from public data (the window's from the registration's
fields, a registered value's from its entry, a slot tree's from the run and its coordinates), and each unit's Definition
replayed on its input rows against its output rows. `build` writes a bundle from a window ``Window.write`` wrote.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree, merkle_depth
from verity.ml.commitments.row_table import ROOTS_SCHEMA
from verity.ml.commitments.window import window_domain
from verity.ml.windows import interface as IF, rows as CS

FORMAT = "verity/drawn-unit-bundle/v1"
SLOT_TREE = "verity/slot-tree/fixture-v0"
SUBTREE_TAG = "verity/slot-tree/subtree/fixture-v0"
STEP_TAG = "verity/slot-tree/step/fixture-v0"
BLOCK_TAG = "verity/slot-tree/block/fixture-v0"
PAYLOAD_LAYOUT = "each drawn unit in bundle order, each row in its member's port order: the row's u16 words little-endian, then its 192-byte salt"
INST = hm96.SHA512
SALT = INST.salt_bytes
COORDS = ("block", "step", "slot", "layer", "op", "unit")

Coords = Callable[[Any, int], dict[str, Any]]
Reads = Callable[[Mapping[str, Any], str, str], "tuple[str, int] | None"]


def torch_loop_coords(label: Mapping[str, Any], offset: int, block: int = 0) -> dict[str, Any]:
    """`integrations/torch_loop`'s unit labels (``{request: "r<j>", step, layer, op}``, one per linear's run of units) as
    coordinates: slot j, and ``unit`` the output coordinate within the run."""
    return {"block": block, "step": int(label["step"]), "slot": int(str(label["request"]).lstrip("r")),
            "layer": int(label["layer"]), "op": str(label["op"]), "unit": int(offset)}


def _domain(tag: str, manifest: dict[str, Any], leaves: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(tag, manifest), -1, RangeIndexedDomain(0, leaves), hash="sha512")


def slot_domain(run: str, block: int, step: int, slot: int, leaves: int) -> CommitmentDomain:
    """The fixture shape's subtree of slot ``slot`` at ``step`` of server-time block ``block``, over ``leaves`` rows."""
    return _domain(SUBTREE_TAG, {"run": run, "block": block, "step": step, "slot": slot, "leaves": leaves}, leaves)


def step_domain(run: str, block: int, step: int, slots: Sequence[int]) -> CommitmentDomain:
    """The fixture shape's tree over one step's slot roots, in ascending slot order."""
    return _domain(STEP_TAG, {"run": run, "block": block, "step": step, "slots": list(slots)}, len(slots))


def block_domain(run: str, block: int, steps: Sequence[int]) -> CommitmentDomain:
    """The fixture shape's tree over one server-time block's step roots, in ascending step order."""
    return _domain(BLOCK_TAG, {"run": run, "block": block, "steps": list(steps)}, len(steps))


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
                raise ValueError(f"{m['name']}: a shared-row, row-table or wired member; bundle v1 reads members of whole rows")
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


def build(window_dir: str | Path, drawn: Sequence[int], *, coords: Coords = torch_loop_coords, reads: Reads | None = None,
          weights: Mapping[str, REG.Registered] | None = None, draw: Mapping[str, Any] | None = None
          ) -> tuple[dict[str, Any], bytes]:
    """The bundle of ``drawn`` (global unit indices) from the window in ``window_dir`` (what ``Window.write`` wrote, prover's
    files included): ``(document, payload)``. ``coords(label, offset)``: a unit's coordinates from its label and its offset
    in its labelled run. ``reads(coords, template, port)``: the ``(value, position)`` a unit's port reads of a registered
    value, or None for a row of its own; ``weights``: those values, the prover's side (rows, salts, tree). ``draw``: the
    draw's ``law``, ``source`` and receipt (default ``{"source": "fixture"}``). Refused (ValueError) when a registered read's
    committed string is not the registered row's (a row committed under fresh salts has no path to the value's root)."""
    w = _Window(Path(window_dir))
    run, part = w.win["run"], w.win["partition"]
    reads = reads or (lambda c, t, p: None)
    weights = dict(weights or {})
    drawn = [int(u) for u in drawn]
    if len(set(drawn)) != len(drawn):
        raise ValueError("a unit is drawn once")

    unit_coords: dict[int, dict[str, Any]] = {}
    groups: dict[tuple[int, int, int], list[tuple[int, int, str, bytes, str]]] = {}
    for k, m in enumerate(w.members):
        schemas = m["frame_v3"]["schemas"]
        strings = {p: w.strings(k, p) for p, _ in m["ports"]}
        for i in range(m["n"]):
            u = m["base"] + i
            c = coords(*w.label(u))
            unit_coords[u] = c
            for p, _ in m["ports"]:
                if reads(c, m["template"], p) is None:
                    groups.setdefault((c["block"], c["step"], c["slot"]), []).append((u, k, p, strings[p][i], schemas[p]))
    subtrees: dict[tuple[int, int, int], MerkleTree] = {}
    where: dict[tuple[int, str], tuple[tuple[int, int, int], int]] = {}
    for key, leaves in sorted(groups.items()):
        dom = slot_domain(run, *key, len(leaves))
        subtrees[key] = MerkleTree(dom, {r: _leaf_value(bc) for r, (_, _, _, bc, _) in enumerate(leaves)},
                                   lambda r, leaves=leaves: leaves[r][4])
        for r, (u, _, p, _, _) in enumerate(leaves):
            where[u, p] = (key, r)
    step_trees: dict[tuple[int, int], tuple[list[int], MerkleTree]] = {}
    for b, s in sorted({(b, s) for b, s, _ in subtrees}):
        slots = sorted(j for bb, ss, j in subtrees if (bb, ss) == (b, s))
        step_trees[b, s] = (slots, MerkleTree(step_domain(run, b, s, slots),
                                              {i: subtrees[b, s, j].commitment.root for i, j in enumerate(slots)},
                                              lambda _i: ROOTS_SCHEMA))
    block_trees: dict[int, tuple[list[int], MerkleTree]] = {}
    for b in sorted({b for b, _ in step_trees}):
        steps = sorted(s for bb, s in step_trees if bb == b)
        block_trees[b] = (steps, MerkleTree(block_domain(run, b, steps),
                                            {i: step_trees[b, s][1].commitment.root for i, s in enumerate(steps)},
                                            lambda _i: ROOTS_SCHEMA))

    roots = {r["port"]: r for r in w.rec["roots"]}
    port_trees: dict[tuple[int, str], MerkleTree] = {}
    payload = bytearray()
    units, members, used_roots, used_values = [], {}, {}, set()
    for u in drawn:
        k = w.member_of(u)
        m, i = w.members[k], u - w.members[k]["base"]
        c = unit_coords[u]
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
            hit = reads(c, m["template"], p)
            if hit is not None:
                value, pos = hit
                reg = weights[value]
                if reg.bcs[pos] != bc:
                    raise ValueError(f"unit {u} {p}: committed under its own salt, not {value} row {pos}'s registered one")
                entry["registered"] = {"value": value, "position": pos, "path": reg.path(pos).hex()}
                used_values.add(value)
            else:
                key, r = where[u, p]
                slots, st = step_trees[key[0], key[1]]
                steps, bt = block_trees[key[0]]
                entry["slot"] = {"block": key[0], "step": key[1], "slot": key[2], "leaf": r, "path": _path(subtrees[key], r),
                                 "slot_path": _path(st, slots.index(key[2])), "step_path": _path(bt, steps.index(key[1]))}
            rows.append(entry)
        units.append({"unit": u, "coords": c, "template": _template(m["template"]),
                      "member": {"k": k, "local": i}, "rows": rows})
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
           "draw": {"source": "fixture", **dict(draw or {}), "units": drawn},
           "weights": {"program": next(iter(progs)) if progs else None,
                       "values": [{**weights[v].entry(), "row_schema": weights[v].schema} for v in sorted(used_values)]},
           "slot_trees": {"format": SLOT_TREE,
                          "blocks": [{"block": b, "root": bt.commitment.root.hex(), "domain": bt.domain.domain_id.hex(),
                                      "steps": [{"step": s, "root": step_trees[b, s][1].commitment.root.hex(),
                                                 "domain": step_trees[b, s][1].domain.domain_id.hex(),
                                                 "slots": [{"slot": j, "root": subtrees[b, s, j].commitment.root.hex(),
                                                            "domain": subtrees[b, s, j].domain.domain_id.hex(),
                                                            "leaves": subtrees[b, s, j].domain.count}
                                                           for j in step_trees[b, s][0]]}
                                                for s in steps]}
                                     for b, (steps, bt) in block_trees.items()]},
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
            raise ValueError(f"{name}: a Boolean-profile port; bundle v1 replays word ports")
        n = len(ty.leaf_widths())
        ins.append(_elements(rows[name], width, () if n == 1 and not getattr(ty, "shape", None) else (n,)))
    want = np.asarray(evaluate_batch(fn, *ins)).reshape(1, -1).astype(np.uint64)
    got = []
    for (o, words), f, width in zip(lay.outs, IF._fields(ret), lay.widths[len(params):]):
        leaves = len(f.leaf_widths())
        got.append(np.frombuffer(rows[o][:leaves * width // 8], dtype=f"<u{width // 8}").astype(np.uint64))
    got_a = np.concatenate(got).reshape(1, -1)
    return got_a.shape == want.shape and bool((got_a == want).all())


def check(doc: Mapping[str, Any], payload: bytes, *, registration: Mapping[str, Any] | None = None,
          replay_units: bool = True) -> dict[str, str]:
    """The consumer's check of a bundle (module docstring): ``{key: why}``, empty when it holds. ``registration``: the
    window's ``registration.json``, whose roots and digest the bundle's must be."""
    bad: dict[str, str] = {}
    if doc.get("format") != FORMAT:
        return {"format": f"not {FORMAT}"}
    if len(payload) != doc["payload"]["bytes"] or hashlib.sha512(payload).hexdigest() != doc["payload"]["sha512"]:
        return {"payload": "its bytes are not the ones the bundle names"}
    win = doc["window"]
    for port, r in win["roots"].items():
        if window_domain(win["program"], win["partition"], port, r["schema"], r["leaves"], win["run"]) != r["domain"]:
            bad[f"root:{port}"] = "its domain is not the window-domain rule's over the registration's fields"
    if registration is not None:
        reg_roots = {r["port"]: r for r in registration["roots"]}
        for port, r in win["roots"].items():
            if (reg_roots.get(port) or {}).get("root") != r["root"]:
                bad[f"root:{port}"] = "not the registration's root"
    values = {v["value"]: v for v in doc["weights"]["values"]}
    for name, v in values.items():
        dom = REG.domain(doc["weights"]["program"], name, v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
        if dom.domain_id.hex() != v["domain"]:
            bad[f"value:{name}"] = "its domain is not the registered-value rule's over its entry"
    trees = {}
    for b in doc["slot_trees"]["blocks"]:
        steps = [s["step"] for s in b["steps"]]
        bd = block_domain(win["run"], b["block"], steps)
        if bd.domain_id.hex() != b["domain"]:
            bad[f"block:{b['block']}"] = "its domain is not the fixture shape's"
        for s in b["steps"]:
            slots = [x["slot"] for x in s["slots"]]
            sd = step_domain(win["run"], b["block"], s["step"], slots)
            if sd.domain_id.hex() != s["domain"]:
                bad[f"step:{b['block']}/{s['step']}"] = "its domain is not the fixture shape's"
            for x in s["slots"]:
                d = slot_domain(win["run"], b["block"], s["step"], x["slot"], x["leaves"])
                if d.domain_id.hex() != x["domain"]:
                    bad[f"slot:{b['block']}/{s['step']}/{x['slot']}"] = "its domain is not the fixture shape's"
                trees[b["block"], s["step"], x["slot"]] = (d, bytes.fromhex(x["root"]), sd, slots.index(x["slot"]),
                                                          bytes.fromhex(s["root"]), bd, steps.index(s["step"]),
                                                          bytes.fromhex(b["root"]))
    members = {m["k"]: m for m in doc["members"]}
    for unit in doc["units"]:
        u = unit["unit"]
        mm = members.get(unit["member"]["k"])
        if mm is None or mm["base"] + unit["member"]["local"] != u or unit["member"]["local"] not in mm["local"]:
            bad[f"unit:{u}"] = "its member does not hold it at its local index"
            continue
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
                bad[key] = "a row opens against a slot tree or a registered value, exactly one"
            elif r["registered"] is not None:
                v = values.get(r["registered"]["value"])
                d = v and REG.domain(doc["weights"]["program"], v["value"], v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
                if v is None or not REG.verify_read(d, bytes.fromhex(v["root"]), r["registered"]["position"], bc,
                                                    bytes.fromhex(r["registered"]["path"]), v.get("row_schema", REG.SCHEMA)):
                    bad[key] = "its registered path does not reach the value's root"
            else:
                s = r["slot"]
                t = trees.get((s["block"], s["step"], s["slot"]))
                if t is None:
                    bad[key] = "its slot tree is not in the bundle"
                    continue
                d, sroot, sd, si, steproot, bd, bi, broot = t
                if (_fold(d, s["leaf"], r["schema"], _leaf_value(bc), s["path"]) != sroot
                        or _fold(sd, si, ROOTS_SCHEMA, sroot, s["slot_path"]) != steproot
                        or _fold(bd, bi, ROOTS_SCHEMA, steproot, s["step_path"]) != broot):
                    bad[key] = "its slot path does not reach its slot, step and block roots"
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


def torch_loop_window(out: str | Path, requests: Sequence[int] = (3, 7), steps: int = 1, cap: int = 8, tamper: bool = False,
                      salt_key: bytes = FIXTURE_SALT_KEY) -> tuple[Any, dict[str, REG.Registered]]:
    """The torch loop's window written to ``out``, its weight rows committed under their registered salts (so each opens
    against its value's root), every other row under the window's ChaCha20 salts from ``salt_key``: -> (the loop, its
    registered weights). Needs torch and `integrations/torch_loop` on the path."""
    from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
    from verity_torch_loop import loop as L

    loop = L.TorchLoop(L.Model(), list(requests), steps, tamper, "drawn-unit-bundle-fixture")
    win = loop.window(D.parse_law("subset:2"), cap)
    program = f"torch-loop/mlp-w8a8-h{loop.model.h}/seed{loop.model.seed}"
    weights = torch_loop_weights(loop.model, salt_key, program)
    labels = win.labels()
    spans = _run_starts(labels)
    base = win._bases()
    salts_of, data = {}, []
    for m in win.window.members:
        lay = IF.layout(win.circuit.template(m.template))
        got = win._recorded[m.template]
        ports = [np.concatenate([b.words[g] for b in got]) for g in range(len(lay.rows) + len(lay.outs))]
        lo = m.base - base[m.template]
        data.append({"rows": {p: ports[g][lo:lo + m.n] for g, (p, _) in enumerate(lay.rows)},
                     "outs": {p: ports[len(lay.rows) + j][lo:lo + m.n] for j, (p, _) in enumerate(lay.outs)}})
        for p in ("wrow", "sw"):
            ys = b""
            for u in range(m.base, m.base + m.n):
                lab, off = next((x, u - st) for a, b, x, st in spans if a <= u < b)
                value, pos = torch_loop_reads(torch_loop_coords(lab, off), m.template, p)
                ys += weights[value].salts[pos]
            salts_of[m.port(p)] = ys
    committed = CS.commit(win.window, data, salt_key=salt_key, salts_of=salts_of)
    win.write(committed, out, {"loop": "torch-loop"}, registrar=R)
    return loop, weights


FIXTURE_DRAW = (5, 200, 300, 511)


def fixture(out_dir: str | Path, scratch: str | Path) -> tuple[Path, Path]:
    """The checked-in fixture: the torch loop (requests 3 and 7, one step, width 128: 512 units in members of 8) written to
    ``scratch``, and the bundle of `FIXTURE_DRAW` (slot 0 layer 0, slot 0 layer 1, slot 1 layer 0, slot 1 layer 1) in
    ``out_dir``."""
    _, weights = torch_loop_window(scratch)
    doc, pay = build(scratch, FIXTURE_DRAW, reads=torch_loop_reads, weights=weights,
                     draw={"source": "fixture", "law": {"law": "subset", "k": len(FIXTURE_DRAW)}})
    return write(doc, pay, out_dir, "drawn-unit-bundle")


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
