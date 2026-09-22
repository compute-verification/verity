#!/usr/bin/env python3
"""CPROF: commitment cost across boundary layouts, on the SAME captured bytes (§32.11).

Variants measured (each with the exact production hash/fold functions, timers around the
read / hash / fold phases):

Weights tree (the amortizable tree; one per checkpoint):
  * v1 whole-tensor leaves (the ``commit`` store's layout: large-few-leaves regime)
  * rows-per-leaf sweep k in {64, 16, 4, 1} (v2 row-block leaves; k=1 is the ``commit-v2``
    layout the coordinate openings need: small-many-leaves regime)
  * word-per-leaf, ONE representative tensor measured fully, whole-store cost extrapolated
    per-leaf (the layout PB3's word-per-leaf batches rebuild; labelled extrapolated)

RU / activation trees (the recurring per-step trees):
  * as-is: one leaf per (position, op, port) row -- the v2 row-leaf layout every existing
    B0 record uses
  * per-layer CU: only interface rows committed (``--interface-regex``; default matches the
    4 residual-stream / norm-input rows per decoder layer, RC research-c-replay §2.5 plan 2)
  * per-layer trees: the same leaves split into one tree per ``l=`` value (tree-shape change)
  * one-leaf-per-RU: all row bytes of the unit concatenated into a single leaf
    (large-few-leaves extreme; destroys row addressability -- shape lever only)

Every row reports (leaves, bytes, read_s, hash_s, fold_s) so the cost model
T = L*alpha_leaf + beta*B + (L-1)*alpha_merkle can be validated against layout points.

Usage:
    python scripts/cprof_layouts.py --store <commit dir> \
        --out docs/data/commitment-prof/layouts.json [--reps 3] \
        [--word-leaf-tensor model.layers.0.self_attn.qkv_proj.weight]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
for p in (str(POC),):
    if p not in sys.path:
        sys.path.insert(0, p)

from verity_vllm import hashing, identity  # noqa: E402
from verity_vllm.capture import CaptureReader  # noqa: E402
from verity_vllm.commit import MerkleTree  # noqa: E402

# 4 interface rows per decoder layer (RC §2.5 plan 2 prices per-layer CUs at 4 rows/layer):
# attention block output, MLP block output (the residual stream after each block) and the two
# norm outputs feeding the blocks.
DEFAULT_INTERFACE_RE = (
    r"(input_layernorm|post_attention_layernorm|self_attn\.o_proj|mlp\.down_proj)"
)


def timed_tree(named_raws: list[tuple[str, str, tuple[int, ...], bytes]], tree_id: str,
               reps: int) -> dict:
    """hash + fold timings over already-read (id, dtype, shape, bytes) rows."""
    hash_t, fold_t = [], []
    leaves = {}
    for _ in range(reps):
        t0 = time.perf_counter()
        leaves = {cid: hashing.leaf_hash(cid, dtype, shape, raw)
                  for cid, dtype, shape, raw in named_raws}
        t1 = time.perf_counter()
        MerkleTree(leaves, tree_id)
        t2 = time.perf_counter()
        hash_t.append(t1 - t0)
        fold_t.append(t2 - t1)
    return {"leaves": len(leaves), "bytes": sum(len(r) for *_x, r in named_raws),
            "hash_s": statistics.median(hash_t), "fold_s": statistics.median(fold_t),
            "total_s": statistics.median(hash_t) + statistics.median(fold_t)}


def weights_variants(reader: CaptureReader, reps: int, word_tensor: str | None) -> list[dict]:
    w = reader.weights
    rows: list[dict] = []
    t0 = time.perf_counter()
    raws = [(e, w._read_whole(e)) for e in w.index.entries]  # noqa: SLF001 - the production read path
    read_s = time.perf_counter() - t0
    total_bytes = sum(len(r) for _, r in raws)

    # v1: whole-tensor leaves
    named = [(e.canonical_id, e.dtype, e.shape, raw) for e, raw in raws]
    rows.append({"variant": "weights/whole-tensor(v1)", **timed_tree(named, "weights", reps),
                 "read_s": read_s, "regime": "large-few-leaves"})

    # rows-per-leaf sweep (v2 layout, synthesised from the same bytes with the same leaf ids)
    for k in (64, 16, 4, 1):
        named = []
        for e, raw in raws:
            if len(e.shape) >= 2 and e.shape[0] > 1:  # row_eligible (weights_v2.row_eligible)
                rb = e.nbytes // e.shape[0]
                for start in range(0, e.shape[0], k):
                    end = min(start + k, e.shape[0])
                    cid = str(identity.weight_id(e.param, e.shard, start))
                    named.append((cid, e.dtype, (end - start, *e.shape[1:]),
                                  raw[start * rb:end * rb]))
            else:
                named.append((e.canonical_id, e.dtype, e.shape, raw))
        rows.append({"variant": f"weights/rows-per-leaf={k}(v2)",
                     **timed_tree(named, "weights", reps), "read_s": read_s,
                     "regime": "small-many-leaves" if k <= 4 else "mid"})

    # word-per-leaf: one tensor fully, extrapolated to the store
    if word_tensor:
        entry = next((e for e, _ in raws if e.param == word_tensor), None)
        if entry is not None:
            raw = next(r for e, r in raws if e is entry)
            word = 2 if entry.dtype == "bfloat16" else max(1, entry.nbytes // max(1, entry.shape[0] * (entry.shape[1] if len(entry.shape) > 1 else 1)))
            n_words = len(raw) // word
            named = [(f"w={entry.param}/c={i}", entry.dtype, (1,), raw[i * word:(i + 1) * word])
                     for i in range(n_words)]
            r = timed_tree(named, "weights", max(1, reps - 1))
            per_leaf_us = r["total_s"] / r["leaves"] * 1e6
            store_words = total_bytes // 2
            rows.append({"variant": f"weights/word-per-leaf[{word_tensor}]", **r,
                         "read_s": None, "regime": "word-per-leaf (extreme)",
                         "per_leaf_us": per_leaf_us,
                         "extrapolated_store_s": per_leaf_us * store_words / 1e6,
                         "extrapolated_store_leaves": store_words,
                         "label": "measured tensor; store total EXTRAPOLATED per-leaf"})
    return rows


def ru_variants(reader: CaptureReader, reps: int, interface_re: str) -> list[dict]:
    rows: list[dict] = []
    pat = re.compile(interface_re)
    for request_id, step in reader.ru_keys():
        ru = reader.ru(request_id, step)
        entries = ru.index.entries
        t0 = time.perf_counter()
        raws = [(e, ru.read(e.canonical_id)) for e in entries]
        read_s = time.perf_counter() - t0
        base = {"tree": ru.tree_id, "phase": ru.index.phase,
                "positions": len(ru.index.positions), "read_s": read_s}

        named = [(e.canonical_id, e.dtype, e.shape, raw) for e, raw in raws]
        rows.append({"variant": "ru/row-leaves(as-is)", **base, **timed_tree(named, ru.tree_id, reps)})

        # per-layer CU: interface rows only
        sel = [(e, raw) for e, raw in raws if pat.search(e.canonical_id)]
        if sel:
            named = [(e.canonical_id, e.dtype, e.shape, raw) for e, raw in sel]
            rows.append({"variant": "ru/per-layer-CU(interface-rows-only)", **base,
                         **timed_tree(named, ru.tree_id, reps),
                         "selected_of": len(raws)})

        # per-layer trees: same leaves, one tree per l=
        by_layer: dict[str, list] = defaultdict(list)
        for e, raw in raws:
            layer = identity.parse(e.canonical_id).layer
            by_layer[str(layer)].append((e.canonical_id, e.dtype, e.shape, raw))
        hash_t, fold_t = [], []
        for _ in range(reps):
            th = tf = 0.0
            for named in by_layer.values():
                t0 = time.perf_counter()
                leaves = {cid: hashing.leaf_hash(cid, d, s, r) for cid, d, s, r in named}
                t1 = time.perf_counter()
                MerkleTree(leaves, ru.tree_id)
                t2 = time.perf_counter()
                th += t1 - t0
                tf += t2 - t1
            hash_t.append(th)
            fold_t.append(tf)
        rows.append({"variant": "ru/per-layer-trees", **base,
                     "trees": len(by_layer), "leaves": len(raws),
                     "bytes": sum(len(r) for _, r in raws),
                     "hash_s": statistics.median(hash_t), "fold_s": statistics.median(fold_t),
                     "total_s": statistics.median(hash_t) + statistics.median(fold_t)})

        # one leaf per RU (concat in tree order)
        order = identity.tree_sorted([e.canonical_id for e, _ in raws])
        raw_by_id = {e.canonical_id: raw for e, raw in raws}
        blob = b"".join(raw_by_id[cid] for cid in order)
        named = [(entries[0].canonical_id, "uint8", (len(blob),), blob)]
        rows.append({"variant": "ru/one-leaf-concat", **base, **timed_tree(named, ru.tree_id, reps)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--interface-regex", default=DEFAULT_INTERFACE_RE)
    ap.add_argument("--word-leaf-tensor", default="model.layers.0.self_attn.qkv_proj.weight")
    args = ap.parse_args()

    reader = CaptureReader(Path(args.store))
    doc = {
        "schema": "veritor/cprof/layouts/v1",
        "store": args.store,
        "profile_id": reader.profile_id,
        "reps": args.reps,
        "interface_regex": args.interface_regex,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": ("same bytes, same hash/fold functions, different leaf grouping; read timed once "
                   "per store (page-cache warm after the first pass); hash/fold median of reps"),
        "weights": weights_variants(reader, args.reps, args.word_leaf_tensor),
        "ru": ru_variants(reader, args.reps, args.interface_regex),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    for r in doc["weights"]:
        print(f"{r['variant']:45s} L={r['leaves']:>9} B={r['bytes']:>11} hash={r['hash_s']:.4f}s fold={r['fold_s']:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
