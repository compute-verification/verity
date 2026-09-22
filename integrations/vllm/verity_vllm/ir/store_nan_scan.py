"""Scan a `verity-vllm` commit store for NaN / infinity words in its captured leaves (lane L8).

    PYTHONPATH=. python -m verity_vllm.ir.store_nan_scan <commit dir>

Reads every replay unit's `ru/<request>/<step>/index.json` (`verity-vllm/ru-index/v1`) and `values.bin`, and
reports per dtype the leaves and words seen, the NaN words (exponent field all ones, non-zero mantissa) and the
infinities, with the distinct NaN encodings (whether `0x7FC0` or `0x7FFF` -- the software and the hardware
canonical bf16 NaN words -- ever appears at a conversion output) and the ops whose leaves contain them.

The L8 migration note (`docs/verified-backend/l8-nan-primitive.md` §3) rests on this scan: TA1's B0 and B1
stores hold no NaN and no infinity in any captured bf16 word, so every recorded trace is finite and the old
and new `f32_to_bf16` agree on it (`TCBHw.castF32ToBf16Hw_eq_of_not_nan`).  Records:
`docs/data/l8-nan-scan-2026-09-07/{B0,B1}.json`.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np


def scan(root: str) -> dict:
    totals: collections.Counter = collections.Counter()
    nan_leaves: list[tuple[str, int]] = []
    inf_leaves: list[tuple[str, int]] = []
    nan_words: collections.Counter = collections.Counter()
    ops_with_nan: collections.Counter = collections.Counter()
    for index in sorted(glob.glob(os.path.join(root, "ru", "*", "*", "index.json"))):
        idx = json.load(open(index))
        values = np.fromfile(os.path.join(os.path.dirname(index), "values.bin"), dtype=np.uint8)
        for e in idx["entries"]:
            dt = e["dtype"]
            buf = values[e["offset"]:e["offset"] + e["nbytes"]]
            totals[(dt, "leaves")] += 1
            if dt == "bfloat16":
                w = buf.view(np.uint16)
                is_nan = ((w & 0x7F80) == 0x7F80) & ((w & 0x007F) != 0)
                is_inf = (w & 0x7FFF) == 0x7F80
            elif dt == "float32":
                w = buf.view(np.uint32)
                is_nan = ((w & 0x7F800000) == 0x7F800000) & ((w & 0x007FFFFF) != 0)
                is_inf = (w & 0x7FFFFFFF) == 0x7F800000
            else:
                totals[(dt, "words")] += buf.size
                continue
            totals[(dt, "words")] += w.size
            n_nan, n_inf = int(is_nan.sum()), int(is_inf.sum())
            if n_nan:
                nan_leaves.append((e["canonical_id"], n_nan))
                for word, count in zip(*np.unique(w[is_nan], return_counts=True)):
                    nan_words[(dt, hex(int(word)))] += int(count)
                ops_with_nan[e["canonical_id"].split("/op=")[-1].split("/")[0]] += 1
            if n_inf:
                inf_leaves.append((e["canonical_id"], n_inf))
            totals[(dt, "nan_words")] += n_nan
            totals[(dt, "inf_words")] += n_inf
    return {"store": root, "totals": {f"{k[0]}/{k[1]}": v for k, v in sorted(totals.items())},
            "leaves_with_nan": len(nan_leaves), "leaves_with_inf": len(inf_leaves),
            "distinct_nan_words": {f"{k[0]}:{k[1]}": v for k, v in nan_words.items()},
            "ops_with_nan": dict(ops_with_nan),
            "nan_leaf_examples": nan_leaves[:10], "inf_leaf_examples": inf_leaves[:10]}


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    print(json.dumps(scan(args[0]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
