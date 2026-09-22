"""[R10 owner 7] TEST-SIDE cross-check: my hashlib-only L1 re-implementation (`review_l1_verify.py`) vs the R9 lanes' host reference
(`fa2_commit.hidden_stream.commit_stream` with native_host's duck-typed layout (HB=step, M=padded_words, NB=0), `fa2_commit.engine`
`fold_path` / `path_in_levels`, `verity_vllm.commit.path_shape`).  The lane code is imported HERE only, never in the verifier.

Run from the repo root with PYTHONPATH=.:vllm-poc.  Prints a JSON verdict; exit 0 iff every comparison agrees.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from verity_capture.bench import review_l1_verify as rv
from verity_capture.experimental.fa2_commit import engine as eng
from verity_capture.experimental.fa2_commit import hidden_stream as hs


def main() -> int:
    rnd = random.Random(11)
    out = {}
    agree = True
    for trial, nbytes in enumerate([256, 512, 700, 3 * 256 + 1, 17 * 256, 1000 * 256 + 5, 228790132 // 64]):
        step = trial
        stream = os.urandom(nbytes)
        padded = rv.pad_stream(stream)

        class _L:                                       # native_host._gpu_layout
            HB, M, NB = step, len(padded) // 4, 0

        # the lane's leaf rule exactly as native_host._commit_step_gpu / verify use it: chunk_header(duck layout) + chunk_leaf_digest, fasttree fold
        n = len(padded) // 256
        lane_leaves = [hs.chunk_leaf_digest(hs.chunk_header(_L, step, i, 64, 0), padded[i * 256:(i + 1) * 256]) for i in range(n)]  # type: ignore[arg-type]
        lane_levels = eng.ft.fold_levels(list(lane_leaves))

        class _C:
            leaf_count, merkle_root, levels = n, lane_levels[-1][0], lane_levels

        lane = _C
        mine_levels = rv.fold_levels(rv.l1_leaves(stream, step))
        mine_root = mine_levels[-1][0]
        r = {"nbytes": nbytes, "n_leaves": lane.leaf_count, "merkle_equal": lane.merkle_root == mine_root}
        # openings: lane's path (engine.path_in_levels) verified by MY fold; my path verified by the lane's fold
        ok_paths = True
        for k in sorted(set([0, lane.leaf_count - 1, rnd.randrange(lane.leaf_count)])):
            lane_path = list(eng.path_in_levels(lane.levels, k))
            my_path = rv.path_in_levels(mine_levels, k)
            chunk = padded[k * 256:(k + 1) * 256]
            ok_paths &= lane_path == my_path
            ok_paths &= rv.verify_l1_step_opening(merkle_root=lane.merkle_root, n_leaves=lane.leaf_count, step=step, rank=k, chunk=chunk, path=lane_path)[0]
            leaf = hs.chunk_leaf_digest(hs.chunk_header(_L, step, k, 64, 0), chunk)  # type: ignore[arg-type]
            ok_paths &= eng.fold_path(leaf, k, my_path) == mine_root
            ok_paths &= [p is not None for p in my_path] == eng.path_shape(lane.leaf_count, k)
        r["paths_and_shapes_equal"] = ok_paths
        out[f"trial{trial}"] = r
        agree &= r["merkle_equal"] and ok_paths
    # pos_leaf + bind_root + weights-root wrappers vs the engine's
    pd, ctx, geo, lay, mr = (os.urandom(32) for _ in range(5))
    agree &= eng.bind_root(pd, ctx, geo, lay, 1234, mr) == rv.l1_step_root.__wrapped__(pd, ctx, geo, lay, 1234, mr) if hasattr(rv.l1_step_root, "__wrapped__") else True
    import hashlib
    agree &= eng.bind_root(pd, hashlib.sha256(b"rid/step=3").digest(), geo, lay, 1234, mr) == rv.l1_step_root(pd, "rid", 3, geo, lay, 1234, mr)
    blob = os.urandom(300)
    agree &= eng.pos_leaf(blob) == rv.pos_leaf(blob)
    leaves = [os.urandom(32) for _ in range(9)]
    agree &= eng.ft.fold(list(leaves)) == rv.fold(leaves)
    out["bind_root_pos_leaf_fold_equal"] = bool(agree)
    out["ALL_AGREE"] = bool(agree)
    print(json.dumps(out, indent=1))
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
