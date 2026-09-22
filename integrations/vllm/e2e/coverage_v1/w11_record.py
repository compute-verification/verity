"""A ``coverage_v1.builders.Record`` over the W11 natural fixture (B1 = Qwen2.5-1.5B served by vLLM; REAL GPU rows: q, K/V, out
of one attention call), committed under record-style canonical ids in one RU tree -- the stand-in for a TINY record until one with
values is available (Q-A56-6).  Used by ``tests/contract/a56_test_head_builder.py`` and ``zk/bench/a56_head_batches.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .builders import Row
from .elem_builder import CommittedTree

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "zk" / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "zk" / "bench"))

NATURAL = REPO / "fixtures" / "W11-40f0cebeb670-20260907T1800Z" / "natural"
NH, KVH, D = 12, 2, 128
RID = "11111111-2222-5333-8444-555555555555"


class W11Record:
    """A record-shaped view of one W11 prefill call: rope/gemm/attn rows per position in one RU tree ``ru/<rid>/000000``."""

    config = {"NH": NH, "KVH": KVH, "D": D}
    session = hashlib.sha256(b"a56/head/session").digest()
    compiled = hashlib.sha256(b"a56/head/compiled").digest()

    def __init__(self, layer: int = 0, T: int = 65) -> None:
        from pb_attn_batches import PackedTreeBuilder, u16_leaf

        idx = json.loads((NATURAL / "index.json").read_text())
        call = next(c for c in idx["calls"] if c["phase"] == "prefill" and c["layer"] == layer and c["Lq"] >= T)
        rows = call["rows"]  # the saved query rows (sparse after the first 66): q/out row of position t is at rows.index(t)
        assert all(t in rows for t in range(T)), f"the fixture saved rows {rows[:70]}"
        qf = np.load(NATURAL / call["q_file"])     # (saved rows, NH, D)
        of = np.load(NATURAL / call["out_file"])   # (saved rows, NH, D)
        q = np.stack([qf[rows.index(t)] for t in range(T)])
        out = np.stack([of[rows.index(t)] for t in range(T)])
        k = np.load(NATURAL / f"kv/r{call['req']}_L{layer}_k.npy")[:T]  # (T, KVH, D)
        v = np.load(NATURAL / f"kv/r{call['req']}_L{layer}_v.npy")[:T]
        self.tree_id = f"ru/{RID}/000000"
        self.tb = PackedTreeBuilder(self.tree_id)
        self.rows: dict[tuple[int, str, int], B.Row] = {}
        QKV = (NH + 2 * KVH) * D
        for t in range(T):
            pre = f"p=0000000000d1/r={RID}/t={t}/ph=prefill/l={layer}/op=layers.{layer}.self_attn"
            self._add(t, f"layers.{layer}.self_attn.rotary_emb.rope", 0, f"{pre}.rotary_emb.rope/port=0", (NH, D), q[t].reshape(-1), u16_leaf)
            self._add(t, f"layers.{layer}.self_attn.rotary_emb.rope", 1, f"{pre}.rotary_emb.rope/port=1", (KVH, D), k[t].reshape(-1), u16_leaf)
            row = np.zeros(QKV, dtype=np.uint16)
            row[(NH + KVH) * D:] = v[t].reshape(-1)
            self._add(t, f"layers.{layer}.self_attn.qkv_proj.gemm", 0, f"{pre}.qkv_proj.gemm/port=0", (QKV,), row, u16_leaf)
            self._add(t, f"layers.{layer}.self_attn.attn.attn", 0, f"{pre}.attn.attn/port=0", (NH, D), out[t].reshape(-1), u16_leaf)
        self.ids = [leaf.canonical_id for leaf in self.tb.leaves]

    def _add(self, t, op, port, cid, shape, words, enc) -> None:
        rank = self.tb.add(cid, "bfloat16", "[" + ", ".join(map(str, shape)) + "]", enc(words))
        self.rows[(t, op, port)] = Row(self.tree_id, cid, rank, "bfloat16", tuple(shape), np.asarray(words, dtype=np.uint16))

    def row(self, request_id, t, op, port):
        return self.rows.get((t, op, port)) if request_id == RID else None

    def tree(self, tree_id):
        if tree_id != self.tree_id:
            return None
        return CommittedTree(tree_id, self.tb.root, len(self.tb.leaves), self.tb.domain, 0)

    def opened(self, tree_id, rank):
        assert tree_id == self.tree_id
        return self.tb.opened(rank)

    # verifier-side oracles
    def tree_of_root(self):
        return {self.tb.root: self.tree_id}

    def id_at_rank(self, tree, rank):
        return self.ids[rank] if tree == self.tree_id and 0 <= rank < len(self.ids) else None
