"""The CORE record writer: the prover-side commitments of ``DemoCore_v1{N,R}`` in the engine's real tree construction
(``verity_vllm.commit.MerkleTree``: ``verity-vllm/leaf/v1`` leaf framing, ``node/lift/ids`` tags, ``commitments/v1`` schema) — mirroring what
``verity_vllm_adapter.selective_prover`` writes for a vLLM record, for this circuit.  Nothing here is a verifier.

Layout (brief, frozen interface 4; adapted to a program whose RU = one stage call):

* step 1 (phase 0): one boundary tree per RU ``boundary/<rid>`` — RU r's declared output tuple ``h_r`` as ONE leaf of N BF16 words in gate
  order (``op=stage.add/port=0``); RU 0's tree additionally holds the three circuit-input vectors ``a, b, c`` (``op=input.a|b|c``), i.e.
  ``v_in(C)`` — every input gate and every RU output gate has a committed position before r_q.  ``commitments_phase0.json``.
* step 3 (phase 2): one interior tree per RETAINED RU ``<INTERIOR_TREE_PREFIX>/<rid>/0`` (directory ``adapter/selective/<tree id>/{index.json,leaf-0.bin}``):
  leaves = chunks of <= 4096 words of the RU's interior VU output gates in gate order (here one chunk: ``g_r``, N words,
  ``op=stage.silu_mul/port=0``), ``origin: "interior"``; root under ``commitments.json#trees`` (kind ``ru``) AND ``#interior_roots``; the
  phase-0 trees are copied unchanged.  A gate with a boundary position is never emitted as interior (checked here).

Leaf ids parse under ``verity_vllm.identity`` (the tree construction requires it): ``p=<profile>/r=<rid>/t=0/ph=decode/l=<r>/op=<op>/port=0``.
``ph``/``t`` carry no meaning for this circuit (grammar constraints of the shared identity module); ``l`` = the stage index.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import hashing
from verity_vllm.commit import COMMITMENTS_SCHEMA, MerkleTree, Opening

from .circuit import Partition
from .producer import Values

__all__ = ["CoreRecord", "request_id", "profile_of", "CHUNK_WORDS", "INTERIOR_LAYOUT", "INTERIOR_TREE_PREFIX", "E2_LAYOUT_PIN", "INTERIOR_MODES", "interior_tree_id"]

CHUNK_WORDS = 4096
# Interface 4 spells the interior tree ``interior/<req>/<step>``; the SHIPPED builder (coverage_v1/ck_elem._owner_of) and the shipped
# verifier_v3.expected.committed_trees read ``ru/<req>/<step>`` today (E7 is adding the ``interior/`` owner mapping) -- until that lands the tree
# id AND its directory (adapter/selective/<tree id>/{index.json,leaf-*.bin}) use the ``ru`` prefix; flipping this constant is the whole change.
INTERIOR_TREE_PREFIX = "ru"
INTERIOR_LAYOUT = {"framing": "chunks-4096", "word_order": "gate", "tree": f"{INTERIOR_TREE_PREFIX}/<req>/<step>"}
DTYPE = "bfloat16"

# [E2, interior-layout/chunks-4096/v1 -- INTEGRATOR D2] the interior mode ``"e2"``: tree ``interior/<req>/<step:06d>`` (directory
# ``adapter/selective/interior/<req>/<step>/{index.json, gates.bin, leaf-<rank:06d>.bin}``), one chunk-leaf per (call, 4096 words, width run) of the
# RU's required interior gates in gate order, leaf id ``<prefix>call=<c>/chunk=<k>`` with ``c`` = the stage call's node NAME (``stage<r>``; E3 GenericLocator.calls()) and
# ``prefix = p=<profile>/r=<req>/``, dtype ``uint16`` (Value<16> words), ``commitments.json#interior_roots["<req>/<step>"]`` = E2's record dict.
# The pin (E6 froze it into the CORE instance): framing / word_order / tree / digest, the digest being E2's ``layout_digest()`` constant.
INTERIOR_MODES = ("ru", "e2")  # see also REQUEST_MODES (how RUs are addressed: one request per RU, or one request whose steps are the RUs)
E2_INDEX_SCHEMA = "verity-interior/index/v1"
def _e2_pin() -> dict[str, Any]:
    """E2's CURRENT rule pin (``e2e.interior.layout.layout_pin()``: ``interior-layout/chunks-4096/v2`` since D13 -- width groups within a call,
    leaf id ``<prefix>call=<c>/w=<w>/chunk=<k>``).  Taken from E2's module, never restated here, so the record follows E2's rule by construction;
    the v1 pin (digest ``2c56e6a0…``) is what the CORE-exhaustive-02/03, B1/X3, C1/X4 archives carry (historical, never edited)."""
    try:
        from e2e.interior import layout as IL
    except Exception:  # noqa: BLE001 -- without E2's module the e2 mode is refused when used (see _commit_interior_e2)
        return {"framing": "chunks-4096", "word_order": "gate", "tree": "interior/<req>/<step>", "digest": None, "rule": None}
    return dict(IL.layout_pin())


E2_LAYOUT_PIN = _e2_pin()
E2_LAYOUT_PIN_V1_DIGEST = "2c56e6a0c9a41a65f0caf904d4f9b588813c9653b1144796a3af4fe857c6c465"  # the rule of the -02/-03/B1/X3/C1/X4 archives


def interior_tree_id(rid: str, step: int, mode: str) -> str:
    return f"interior/{rid}/{int(step):06d}" if mode == "e2" else f"{INTERIOR_TREE_PREFIX}/{rid}/{int(step)}"


# [request spelling -- INTEGRATOR ruling pending, 21:18Z flag] how the R replay units are addressed in the record:
#   "per-ru"  (default; every archived run): one request per RU, ``core-ru<r>``; boundary tree ``boundary/core-ru<r>``; interior step 0.
#   "single"  (E6's frozen CORE instance: ``epoch.request_ids = ["core-0"]``, ``ru_count = R``, ``instances = 1``): ONE program instance
#             ``core-0`` whose RU r is its step r -- one boundary tree ``boundary/core-0`` (inputs + every RU's output tuple), interior tree
#             ``interior/core-0/<r:06d>`` (E2) or ``ru/core-0/<r>``; the tree id's step names the RU (E7's ``owner_of`` = step).
#             Needs the shipped ``steps.committed_rus`` to map (request, step) -> RU (today it maps the request only) -- E9.
REQUEST_MODES = ("per-ru", "single")
SINGLE_REQUEST_ID = "core-0"
# process-wide (the prover and the runner set it from --request; a verifier process that imports demo_core.adapter through E3's locator sets
# DEMO_CORE_REQUEST=single in its environment, or the adapter reads epoch.request_ids == ["core-0"] when the locator hands it the instance)
_REQUEST_MODE = [os.environ.get("DEMO_CORE_REQUEST", "per-ru")]
if _REQUEST_MODE[0] not in REQUEST_MODES:
    raise ValueError(f"DEMO_CORE_REQUEST={_REQUEST_MODE[0]!r} not in {REQUEST_MODES}")


def set_request_mode(mode: str) -> None:
    if mode not in REQUEST_MODES:
        raise ValueError(f"request mode {mode!r} not in {REQUEST_MODES}")
    _REQUEST_MODE[0] = mode


def request_mode() -> str:
    return _REQUEST_MODE[0]


# [E9 1becdd75, steps.ru_ids_of] on a registered epoch with one request and ru_indices 0..R-1 the driver's RU id is ``<request>-ru<r>``; the base
# is the epoch's request id (the runner's instances say ``core`` -> ``core-ru<r>``, E6's frozen CORE instance says ``core-0`` -> ``core-0-ru<r>``)
_RU_BASE = [os.environ.get("DEMO_CORE_REQUEST_BASE", "core")]


def set_ru_base(base: str) -> None:
    if not base or "/" in base:
        raise ValueError(f"bad RU-id base {base!r}")
    _RU_BASE[0] = str(base)


def ru_base() -> str:
    return _RU_BASE[0]


def request_id(r: int) -> str:
    """The record's id of RU ``r`` (its boundary tree ``boundary/<id>``, interior tree ``<prefix>/<id>/<step>``): ``<base>-ru<r>`` under per-ru
    (E9's ``ru_ids_of`` spelling), the one request id under single."""
    return SINGLE_REQUEST_ID if _REQUEST_MODE[0] == "single" else f"{_RU_BASE[0]}-ru{int(r)}"


def step_of(r: int) -> int:
    """The step of RU ``r`` within its request: 0 under per-ru (each request has one RU), ``r`` under single."""
    return int(r) if _REQUEST_MODE[0] == "single" else 0


def profile_of(part: Partition) -> str:
    """12-hex profile id = the pinned (program, partition) identity — what the vLLM capture derives from its manifest."""
    return hashing.profile_id({"schema": "veritor/demo-core/profile/v1", "program_digest": part.program_digest, "partition_digest": part.digest,
                               "program": part.to_json()["program"]})


def _vid(pid: str, rid: str, layer: int, op: str) -> str:
    return f"p={pid}/r={rid}/t=0/ph=decode/l={layer}/op={op}/port=0"


def _bytes(words: list[int]) -> bytes:
    return np.asarray(words, dtype="<u2").tobytes()


@dataclass
class Leaf:
    canonical_id: str
    tree: str
    words: list[int]
    gates: list[int]
    origin: str  # "boundary" | "input" | "interior"
    dtype: str = DTYPE  # boundary leaves: bfloat16 (the capture's spelling); E2 interior chunks: uint16 (Value<16> words)

    @property
    def value(self) -> bytes:
        return _bytes(self.words)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.words),)

    def leaf_hash(self) -> bytes:
        return hashing.leaf_hash(self.canonical_id, self.dtype, self.shape, self.value)


@dataclass
class CoreRecord:
    """Prover-side record of one run: values -> leaves -> trees -> commitments; ``open_leaf`` for the builders."""

    part: Partition
    vals: Values
    sel: Path
    interior: str = "ru"  # INTERIOR_MODES: "ru" (today's shipped ck_elem spelling) | "e2" (interior-layout/chunks-4096/v1, the D1 path)
    profile_id: str = ""
    leaves: dict[str, Leaf] = field(default_factory=dict)         # canonical id -> leaf
    gate_pos: dict[int, tuple[str, str, int]] = field(default_factory=dict)  # gate -> (tree, leaf id, flat word index)
    trees: dict[str, MerkleTree] = field(default_factory=dict)
    commitments: dict[str, Any] = field(default_factory=dict)
    phase0: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interior not in INTERIOR_MODES:
            raise ValueError(f"interior mode {self.interior!r} not in {INTERIOR_MODES}")
        self.profile_id = profile_of(self.part)

    @property
    def interior_layout(self) -> dict[str, Any]:
        return dict(E2_LAYOUT_PIN) if self.interior == "e2" else dict(INTERIOR_LAYOUT)

    def interior_tree(self, r: int) -> str:
        return interior_tree_id(request_id(r), step_of(r), self.interior)

    # -- step 1 -------------------------------------------------------------------------------------------------------------------
    def commit_phase0(self) -> dict[str, Any]:
        part, vals = self.part, self.vals
        pid = self.profile_id
        rid0 = request_id(0)
        for name in ("a", "b", "c"):
            gates = list(part.inputs[name])
            self._add(Leaf(_vid(pid, rid0, 0, f"input.{name}"), f"boundary/{rid0}", vals.words(gates), gates, "input"))
        for ru in part.rus:
            rid = request_id(ru.ru_index)
            gates = list(ru.returns)
            self._add(Leaf(_vid(pid, rid, ru.ru_index, "stage.add"), f"boundary/{rid}", vals.words(gates), gates, "boundary"))
        trees: dict[str, Any] = {}
        for ru in part.rus:
            tid = f"boundary/{request_id(ru.ru_index)}"
            if tid in trees:  # single spelling: the one boundary tree holds every RU's output tuple
                trees[tid]["ru_indices"].append(ru.ru_index)
                continue
            t = self._tree(tid)
            who = {"ru_index": ru.ru_index} if request_mode() == "per-ru" else {"ru_index": None, "ru_indices": [ru.ru_index]}
            trees[tid] = t.record("boundary", request_id=request_id(ru.ru_index), **who, origin="boundary",
                                  leaves=[{"canonical_id": lf.canonical_id, "shape": list(lf.shape), "dtype": DTYPE, "gates": [lf.gates[0], lf.gates[-1] + 1], "origin": lf.origin}
                                          for lf in self._leaves_of(tid)])
        # the weights tree: this circuit has no prescribed weights; the engine's step-2 check requires the tree to exist and to equal the
        # registered root, so it is the EMPTY tree (0 leaves, root = H("verity-vllm/empty/v1")) and the registration says so
        empty = MerkleTree({}, "weights")
        self.trees["weights"] = empty
        trees["weights"] = empty.record("weights", note="DemoCore has no prescribed weights: empty tree (0 leaves)")
        doc = {"schema": COMMITMENTS_SCHEMA, "profile_id": pid, "program_digest": part.program_digest, "partition_digest": part.digest,
               "hash": {"frame": hashing.FRAME_DESCRIPTION, "leaf": hashing.LEAF_TAG, "node": hashing.NODE_TAG, "lift": hashing.LIFT_TAG, "ids": hashing.IDS_TAG, "empty": hashing.EMPTY_TAG},
               "trees": trees, "label": "generic protocol demonstration (DemoCore_v1); NOT vLLM inference"}
        self.phase0 = doc
        self.commitments = json.loads(json.dumps(doc))
        self.sel.mkdir(parents=True, exist_ok=True)
        self._write_leaves("boundary")
        (self.sel / "commitments_phase0.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return doc

    # -- step 3 -------------------------------------------------------------------------------------------------------------------
    def commit_interior(self, retained: list[int]) -> dict[str, Any]:
        """Commit the interior VU outputs of every retained RU (chunks of <= CHUNK_WORDS words in gate order); phase-0 trees unchanged."""
        if not self.phase0:
            raise RuntimeError("commit_phase0 first")
        part, vals, pid = self.part, self.vals, self.profile_id
        trees = json.loads(json.dumps(self.phase0["trees"]))
        roots: dict[str, Any] = {}
        for r in sorted(int(x) for x in retained):
            ru = part.rus[r]
            rid = request_id(r)
            gates = [v.gate for v in ru.vus if not v.ru_output]
            dup = [g for g in gates if g in self.gate_pos]
            if dup:
                raise RuntimeError(f"INTERIOR_DUPLICATE_GATE: gates {dup[:4]} already have a boundary position")
            step = step_of(r)
            if self.interior == "e2":
                self._commit_interior_e2(ru, rid, step, gates, trees, roots)
                continue
            tid = f"{INTERIOR_TREE_PREFIX}/{rid}/{step}"
            for k in range(0, len(gates), CHUNK_WORDS):
                chunk = gates[k:k + CHUNK_WORDS]
                op = "stage.silu_mul" if k == 0 else f"stage.silu_mul.chunk{k // CHUNK_WORDS}"
                self._add(Leaf(_vid(pid, rid, r, op), tid, vals.words(chunk), chunk, "interior"))
            t = self._tree(tid)
            entries = [{"canonical_id": lf.canonical_id, "dtype": DTYPE, "shape": list(lf.shape), "gates": [lf.gates[0], lf.gates[-1] + 1], "origin": "interior",
                        "leaf_hash": lf.leaf_hash().hex(), "index": t.index_of(lf.canonical_id)} for lf in self._leaves_of(tid)]
            trees[tid] = t.record("ru", request_id=rid, step=step, ru_index=r, origin="interior", layout=INTERIOR_LAYOUT)
            roots[f"{rid}/{step}"] = t.root.hex()
            d = self.sel / tid
            d.mkdir(parents=True, exist_ok=True)
            (d / "index.json").write_text(json.dumps({"schema": "veritor/demo-core/interior-index/v1", "tree": tid, "request_id": rid, "step": step, "ru_index": r,
                                                      "layout": INTERIOR_LAYOUT, "root": t.root.hex(), "leaf_count": t.leaf_count, "ids_digest": t.ids_digest.hex(),
                                                      "entries": entries}, indent=1) + "\n")
            for i, lf in enumerate(self._leaves_of(tid)):
                (d / f"leaf-{i}.bin").write_bytes(lf.value)
        doc = dict(self.phase0)
        doc["trees"] = trees
        doc["interior_roots"] = roots
        doc["interior_layout"] = self.interior_layout
        self.commitments = doc
        (self.sel / "commitments.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return doc

    def _commit_interior_e2(self, ru, rid: str, step: int, gates: list[int], trees: dict[str, Any], roots: dict[str, Any]) -> None:
        """One RU's interior tree under E2's CURRENT rule (``layout_pin()["rule"]``; v2 since D13) THROUGH E2's module ``e2e.interior`` (layout derived by
        ``InteriorLayout.build`` over the RU's root call and its required gates; tree/index/files by ``InteriorTree`` / ``write_tree``).  Refuses
        by name when the module is not on this branch (no re-implementation here)."""
        try:
            from e2e.interior import layout as IL
            from e2e.interior import tree as IT
        except ImportError as e:
            raise RuntimeError(f"BLOCKED: interior mode 'e2' needs E2's e2e.interior (layout, tree) on this branch: {e}") from e
        r = ru.ru_index
        tid = IL.tree_id(rid, step)
        assert tid == interior_tree_id(rid, step, "e2"), (tid, interior_tree_id(rid, step, "e2"))
        if IL.check_pin(E2_LAYOUT_PIN):
            raise RuntimeError(f"E2_LAYOUT_PIN != e2e.interior.layout.layout_pin(): {IL.check_pin(E2_LAYOUT_PIN)}")
        prefix = f"p={self.profile_id}/r={rid}/"
        gates = sorted(int(g) for g in gates)
        if any(not (ru.lo <= g < ru.hi) for g in gates):
            raise RuntimeError(f"RU {r}: an interior gate lies outside its call [{ru.lo}, {ru.hi})")
        layout = IL.InteriorLayout.build(tid, [IL.CallSpan(str(ru.name), int(ru.lo), int(ru.hi))], gates, [16] * len(gates), prefix=prefix)
        values = [_bytes(self.vals.words([int(g) for g in lf.gates])) for lf in layout.leaves]
        t = IT.InteriorTree(layout, values)
        for rank, lf in enumerate(layout.leaves):
            self._add(Leaf(lf.canonical_id, tid, self.vals.words([int(g) for g in lf.gates]), [int(g) for g in lf.gates], "interior", dtype=lf.dtype))
        self.trees[tid] = _E2Tree(t)
        roots[f"{rid}/{step:06d}"] = t.record()
        # the phase-2 tree record the SHIPPED step-4 check reads (kind ``ru``, request_id -> RU); same root as interior_roots
        trees[tid] = {"kind": "ru", "root": t.root.hex(), "leaf_count": t.leaf_count, "ids_digest": t.ids_digest.hex(), "request_id": rid, "step": step,
                      "ru_index": r, "origin": "interior", "tree": tid, "layout": dict(E2_LAYOUT_PIN)}
        IT.write_tree(t, self.sel / tid)

    # -- openings (what a builder reads) --------------------------------------------------------------------------------------------
    def open_leaf(self, canonical_id: str) -> Opening:
        lf = self.leaves[canonical_id]
        t = self.trees[lf.tree]
        if isinstance(t, _E2Tree):
            return t.open(canonical_id)
        return Opening(tree=lf.tree, root=t.root, leaf_count=t.leaf_count, index=t.index_of(canonical_id), canonical_id=canonical_id, dtype=lf.dtype,
                       shape=lf.shape, value=lf.value, path=t.path(canonical_id))

    def position(self, g: int) -> tuple[str, str, int] | None:
        """(tree, leaf id, flat word) of gate ``g`` in the committed trees, or None (not committed)."""
        return self.gate_pos.get(int(g))

    def ids_of(self, tree_id: str) -> list[str]:
        return list(self.trees[tree_id].ids)

    # -- internals -------------------------------------------------------------------------------------------------------------------
    def _add(self, lf: Leaf) -> None:
        if lf.canonical_id in self.leaves:
            raise RuntimeError(f"duplicate leaf {lf.canonical_id}")
        for i, g in enumerate(lf.gates):
            if g in self.gate_pos:
                raise RuntimeError(f"gate {g} committed twice ({self.gate_pos[g]} and {lf.canonical_id})")
            self.gate_pos[g] = (lf.tree, lf.canonical_id, i)
        self.leaves[lf.canonical_id] = lf

    def _leaves_of(self, tid: str) -> list[Leaf]:
        return [self.leaves[c] for c in self.trees[tid].ids] if tid in self.trees else [lf for lf in self.leaves.values() if lf.tree == tid]

    def _tree(self, tid: str) -> MerkleTree:
        t = MerkleTree({lf.canonical_id: lf.leaf_hash() for lf in self.leaves.values() if lf.tree == tid}, tid)
        self.trees[tid] = t
        return t

    def _write_leaves(self, kind: str) -> None:
        for tid, t in self.trees.items():
            if not tid.startswith(kind + "/"):
                continue
            d = self.sel / tid
            d.mkdir(parents=True, exist_ok=True)
            entries = []
            for i, cid in enumerate(t.ids):
                lf = self.leaves[cid]
                (d / f"leaf-{i}.bin").write_bytes(lf.value)
                entries.append({"canonical_id": cid, "dtype": DTYPE, "shape": list(lf.shape), "gates": [lf.gates[0], lf.gates[-1] + 1], "origin": lf.origin,
                                "leaf_hash": lf.leaf_hash().hex(), "index": i})
            (d / "index.json").write_text(json.dumps({"schema": "veritor/demo-core/boundary-index/v1", "tree": tid, "root": t.root.hex(), "leaf_count": t.leaf_count,
                                                      "ids_digest": t.ids_digest.hex(), "entries": entries}, indent=1) + "\n")


class _E2Tree:
    """``MerkleTree``-shaped view of E2's ``InteriorTree`` for :class:`CoreRecord` (ids in rank order, ``index_of``, ``path``, ``root``...)."""

    def __init__(self, t) -> None:
        self.t = t
        self.ids = list(t.layout.ids)
        self._rank = {cid: i for i, cid in enumerate(self.ids)}

    @property
    def root(self) -> bytes:
        return self.t.root

    @property
    def leaf_count(self) -> int:
        return self.t.leaf_count

    @property
    def ids_digest(self) -> bytes:
        return self.t.ids_digest

    def index_of(self, cid: str) -> int:
        return self._rank[cid]

    def path(self, cid: str):
        return self.t.path(self._rank[cid])

    def open(self, cid: str) -> Opening:
        return self.t.open(self._rank[cid])

