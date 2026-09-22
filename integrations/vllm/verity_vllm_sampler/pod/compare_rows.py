"""Compare two replay logits dumps of the same trajectory row by row (bit-exact bf16), per request.

    python verity_vllm_sampler/pod/compare_rows.py <dump_a> <dump_b> <out.json> [label]

Used for: cap-64 vs cap-8/1 replays of one record (batch dependence, §15.5), and the earlier
(request 13 = 16-token prefix-cache hit) vs fresh-prefill replays (does the cache hit change the row bits?).
"""

import json
import os
import sys

import numpy as np

a, b, out = sys.argv[1:4]
label = sys.argv[4] if len(sys.argv) > 4 else ""
ia, ib = json.load(open(os.path.join(a, "index.json"))), json.load(open(os.path.join(b, "index.json")))
cache_a = {rid: v.get("replay") for rid, v in (ia.get("num_cached_tokens") or {}).items()}
cache_b = {rid: v.get("replay") for rid, v in (ib.get("num_cached_tokens") or {}).items()}
res = {"label": label, "a": a, "b": b, "vllm_a": ia.get("vllm_version"), "vllm_b": ib.get("vllm_version"),
       "requests": 0, "positions": 0, "identical_rows": 0, "rows_differing": 0, "max_abs_diff_bf16_as_f32": 0.0,
       "first_diff": None, "per_request": []}
for rid, infa in ia["requests"].items():
    infb = ib["requests"].get(rid)
    if infb is None:
        continue
    A, B = np.load(os.path.join(a, infa["file"])), np.load(os.path.join(b, infb["file"]))
    pa = {p: i for i, p in enumerate(infa["positions"])}
    res["requests"] += 1
    pr = {"request_id": rid, "prompt_len": infa["prompt_len"], "positions": 0, "identical_rows": 0, "rows_differing": 0,
          "max_abs_diff": 0.0, "max_scalars_differing_in_a_row": 0, "cached_a": cache_a.get(rid), "cached_b": cache_b.get(rid),
          "argmax_changes": 0, "first_diff_pos": None}
    for j, p in enumerate(infb["positions"]):
        if p not in pa:
            continue
        ra, rb = A[pa[p]], B[j]
        res["positions"] += 1
        pr["positions"] += 1
        if np.array_equal(ra, rb):
            res["identical_rows"] += 1
            pr["identical_rows"] += 1
        else:
            res["rows_differing"] += 1
            pr["rows_differing"] += 1
            fa = (ra.astype(np.uint32) << 16).view(np.float32)
            fb = (rb.astype(np.uint32) << 16).view(np.float32)
            d = float(np.max(np.abs(fa - fb)))
            n = int(np.sum(ra != rb))
            pr["max_abs_diff"] = max(pr["max_abs_diff"], d)
            pr["max_scalars_differing_in_a_row"] = max(pr["max_scalars_differing_in_a_row"], n)
            pr["argmax_changes"] += int(np.argmax(fa) != np.argmax(fb))
            res["max_abs_diff_bf16_as_f32"] = max(res["max_abs_diff_bf16_as_f32"], d)
            if pr["first_diff_pos"] is None:
                pr["first_diff_pos"] = int(p)
            if res["first_diff"] is None:
                i0 = int(np.flatnonzero(ra != rb)[0])
                res["first_diff"] = {"request_id": rid, "pos": int(p), "n_scalars_differing": n, "max_abs_diff": d,
                                     "first_coordinate": i0, "a_bits": f"0x{int(ra[i0]):04x}", "b_bits": f"0x{int(rb[i0]):04x}",
                                     "a_value": float(fa[i0]), "b_value": float(fb[i0])}
    res["per_request"].append(pr)
res["all_identical"] = res["rows_differing"] == 0
json.dump(res, open(out, "w"), indent=1)
print("ROWS_COMPARE", label, json.dumps({k: v for k, v in res.items() if k not in ("a", "b", "per_request")}))
for pr in res["per_request"]:
    if pr["rows_differing"]:
        print("  differing request", json.dumps(pr))
