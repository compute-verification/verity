#!/usr/bin/env python3
"""CPROF: attribute the non-hash overheads of the commitment path (follow-up to
cprof_store_profile.py).

Splits, on real store data:
  * ``identity.parse`` per real RU value id / weight row id (the MerkleTree constructor
    parses every id; the fold micro used short synthetic ids),
  * ``identity.tree_sorted`` for a real RU id set,
  * ``hashing.ids_digest`` for a real RU id set (``MerkleTree.record`` computes it per tree),
  * ``CaptureReader.ru()`` construction = ``index.json`` parse (the gap between
    build_commitments wall and the read+hash+fold phase sums),
  * ``MerkleTree.__init__`` on one real decode RU's precomputed leaves (fold with real ids).

Usage: python cprof_attrib.py --store <commit dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
for p in (str(POC),):
    if p not in sys.path:
        sys.path.insert(0, p)

from verity_vllm import hashing, identity  # noqa: E402
from verity_vllm.capture import CaptureReader  # noqa: E402
from verity_vllm.commit import MerkleTree  # noqa: E402


def bench(fn, reps: int = 7) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reader = CaptureReader(Path(args.store))
    keys = reader.ru_keys()
    decode_key = next((k for k in keys if reader.ru(*k).index.phase == "decode"), keys[-1])
    ru = reader.ru(*decode_key)
    ids = [e.canonical_id for e in ru.index.entries]
    leaves = ru.leaves(recompute=True)
    n = len(ids)

    doc: dict = {"schema": "veritor/cprof/attrib/v1", "store": args.store,
                 "decode_ru": ru.tree_id, "leaves": n,
                 "id_example": ids[0], "id_len_mean": sum(map(len, ids)) / n}

    # identity.parse on real ids
    t = bench(lambda: [identity.parse(c) for c in ids])
    doc["parse_us_per_id"] = t / n * 1e6
    wids = [f"w=layers.0.self_attn.qkv_proj.weight/c={i}" for i in range(n)]
    t = bench(lambda: [identity.parse(c) for c in wids])
    doc["parse_us_per_weight_row_id"] = t / n * 1e6

    # tree_sorted + ids_digest on real ids
    doc["tree_sorted_us_per_id"] = bench(lambda: identity.tree_sorted(ids)) / n * 1e6
    doc["ids_digest_us_per_id"] = bench(lambda: hashing.ids_digest(sorted(ids))) / n * 1e6

    # MerkleTree on real ids (precomputed leaves): fold incl. real-id parse
    doc["merkle_real_ids_us_per_leaf"] = bench(lambda: MerkleTree(leaves, ru.tree_id)) / n * 1e6

    # index.json parse: fresh RUReader construction for every RU
    def build_all() -> None:
        fresh = CaptureReader(Path(args.store))
        for k in fresh.ru_keys():
            fresh.ru(*k)

    doc["ru_index_parse_all_s"] = bench(build_all, reps=3)
    doc["ru_count"] = len(keys)

    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
