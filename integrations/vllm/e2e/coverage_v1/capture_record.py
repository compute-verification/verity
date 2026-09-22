"""``coverage_v1.builders.Record`` over a real capture: the prover's view (values + trees) of a replayed record (A5/6).

Wraps ``verity_vllm.capture.CaptureReader`` / ``verity_vllm.commit.TreeCache`` / ``commitments.json`` the way A3's
``e2e.verifier_v3.expected.Deriver`` and ``scripts/rec_p1_obligations.py`` do, and answers the three builder questions:

* ``row(request_id, t, op, port)`` -- the committed operator row (canonical id from the RU trees' id lists, indexed once per
  ``(request, t, op, port)``; value bytes from the capture; rank = position in the tree's ``identity.tree_sorted`` order, which
  is the committed rank order the verifier reproduces with A2's ``rank_oracle(..., sort_key=identity.sort_key)``);
* ``tree(tree_id)`` -- ``CommittedTree`` (root, leaf count, ``ids_digest``, owner ordinal) from ``commitments.json``;
* ``opened(tree_id, rank)`` -- the ``PackedLeaf`` with path, from ``TreeCache.open``.

The id lists are read from each RU's ``index.json`` and their ``ids_digest`` is checked against ``commitments.json`` before use
(the same check as the verifier's ``committed_trees``): a builder never emits a position in a tree whose id list is not the one
committed before the challenge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .builders import Row
from .elem_builder import CommittedTree

__all__ = ["CaptureRecord"]


class CaptureRecord:
    def __init__(self, sel_dir: str | Path, commitments: dict[str, Any], *, config: dict[str, Any], session: bytes, compiled: bytes,
                 recompute_leaves: bool = True) -> None:
        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import TreeCache

        self.sel = Path(sel_dir)
        self.commitments = commitments
        self.config = dict(config)
        self.session, self.compiled = session, compiled
        self.reader = CaptureReader(self.sel)
        self.cache = TreeCache(self.reader, recompute_leaves=recompute_leaves)
        self._ids: dict[str, list[str]] = {}
        self._rank: dict[str, dict[str, int]] = {}
        self._rows: dict[tuple[str, int, str, int], tuple[str, str]] | None = None  # (rid, t, op, port) -> (tree_id, cid)
        self._owner = {tid: k for k, tid in enumerate(sorted(t for t in (commitments.get("trees") or {}) if t.startswith("ru/")))}

    # -- committed id lists (checked) ---------------------------------------------------------------------------------------

    def ids(self, tree_id: str) -> list[str]:
        if tree_id in self._ids:
            return self._ids[tree_id]
        from verity_vllm import hashing, identity

        c = (self.commitments.get("trees") or {}).get(tree_id)
        if c is None:
            raise KeyError(f"{tree_id}: not in commitments.json")
        if tree_id.startswith("ru/"):
            _, rid, step = tree_id.split("/")
            index = json.loads((self.sel / "ru" / rid / step / "index.json").read_text())
            ids = [e["canonical_id"] for e in index["entries"]]
        else:
            ids = list(self.cache.tree(tree_id).ids)
        if hashing.ids_digest(sorted(ids)).hex() != c.get("ids_digest") or len(ids) != int(c["leaf_count"]):
            raise ValueError(f"{tree_id}: id list does not match the committed ids_digest / leaf_count")
        ordered = identity.tree_sorted(ids)
        self._ids[tree_id] = ordered
        self._rank[tree_id] = {cid: i for i, cid in enumerate(ordered)}
        return ordered

    def rank_of(self, tree_id: str, canonical_id: str) -> int | None:
        if tree_id not in self._rank:
            try:
                self.ids(tree_id)
            except (KeyError, ValueError, FileNotFoundError):
                return None
        return self._rank[tree_id].get(canonical_id)

    def _index_rows(self) -> dict[tuple[str, int, str, int], tuple[str, str]]:
        if self._rows is None:
            from verity_vllm import identity

            rows: dict[tuple[str, int, str, int], tuple[str, str]] = {}
            for tid in (self.commitments.get("trees") or {}):
                if not tid.startswith("ru/"):
                    continue
                try:
                    ids = self.ids(tid)
                except (KeyError, ValueError, FileNotFoundError):
                    continue
                for cid_s in ids:
                    c = identity.parse(cid_s)
                    if c.kind == "value" and c.op and c.coordinate is None and c.request_id is not None and c.t is not None:
                        rows[(c.request_id, int(c.t), c.op, int(c.port or 0))] = (tid, cid_s)
            self._rows = rows
        return self._rows

    # -- Record protocol ---------------------------------------------------------------------------------------------------

    def row(self, request_id: str, t: int, op: str, port: int) -> Row | None:
        hit = self._index_rows().get((request_id, int(t), op, int(port)))
        if hit is None:
            return None
        tid, cid = hit
        v = self.reader.value(cid)
        words = np.frombuffer(v.data, dtype="<u4" if v.dtype == "float32" else "<u2")
        rank = self.rank_of(tid, cid)
        if rank is None:
            return None
        return Row(tid, cid, rank, v.dtype, tuple(int(d) for d in v.shape), np.asarray(words))

    def tree(self, tree_id: str) -> CommittedTree | None:
        c = (self.commitments.get("trees") or {}).get(tree_id)
        if c is None:
            return None
        owner = -2 if tree_id == "weights" else (-1 if tree_id.startswith("boundary/") else self._owner.get(tree_id, 0))
        return CommittedTree(tree_id, bytes.fromhex(str(c["root"])), int(c["leaf_count"]), bytes.fromhex(str(c["ids_digest"])), owner)

    def opened(self, tree_id: str, rank: int):
        from verity_vllm import hashing
        from veritor.protocol.proofs.packed import PackedLeaf

        cid = self.ids(tree_id)[rank]
        o = self.cache.open(cid)
        if int(o.index) != int(rank):
            raise ValueError(f"{tree_id}: opening of {cid} is at rank {o.index}, the committed id list says {rank}")
        return PackedLeaf(o.canonical_id, o.dtype, hashing.shape_text(o.shape), o.value, tuple(o.path))
