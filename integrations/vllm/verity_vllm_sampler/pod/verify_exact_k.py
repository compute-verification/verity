"""Check: does 'exactly k tokens at the top-k boundary (surplus ties dropped in vocabulary order)' + nucleus
reproduce the Triton kernel's kept set on the disagreeing rows, and over every row of a dump?"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from verity_vllm_sampler import select as S  # noqa: E402
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton  # noqa: E402

R = os.environ["W3_RESULTS"]
K, P = 50, 0.9


def exact_k_masks(x32, k, p):
    x = x32.astype(np.float64)
    V = x.shape[0]
    kth = np.partition(x, V - k)[V - k]
    above = x > kth
    n_keep_ties = k - int(above.sum())
    ties = np.flatnonzero(x == kth)  # vocabulary order = Triton final-pass order
    keep = above.copy()
    keep[ties[:n_keep_ties]] = True
    xk = np.where(keep, x, -np.inf)
    e = np.exp(xk - xk.max())
    probs = e / e.sum()
    order = np.argsort(-xk, kind="stable")
    mass_above = np.empty(V)
    mass_above[order] = np.concatenate(([0.0], np.cumsum(probs[order])[:-1]))
    keep &= mass_above < p
    return keep, len(ties), n_keep_ties


def triton_keep(row):
    x = torch.from_numpy(row).cuda().unsqueeze(0)
    out = apply_top_k_top_p_triton(x.clone(), torch.tensor([K], dtype=torch.int32, device="cuda"), torch.tensor([P], device="cuda"))
    return torch.isfinite(out[0]).cpu().numpy()


for s in (1, 2, 3):
    name = f"topk_topp_s{s}_c64_r0"
    d = json.load(open(f"{R}/diag/topp_s{s}.json"))
    prod = json.load(open(f"{R}/prod/{name}.json"))
    idx = json.load(open(f"{R}/replay/logits_{name}/index.json"))
    recs = {r["index"]: r for r in prod["records"]}
    for r in d["rows"]:
        rec = recs[r["index"]]
        info = idx["requests"][rec["request_id"]]
        arr = np.load(f"{R}/replay/logits_{name}/{info['file']}")
        row = S.logits_from_bytes(arr[info["positions"].index(r["pos"])].tobytes(), idx["logits_dtype"])
        tri = triton_keep(row)
        keep, nties, nkeep = exact_k_masks(row, K, P)
        noise = S.gumbel_noise(rec["seed"], r["pos"], row.shape[0])
        score = np.where(keep, row.astype(np.float32) + noise.astype(np.float32), -np.inf)
        print(f"seed {s} idx {r['index']} k {r['k']}: ties_at_kth={nties} kept_of_ties={nkeep} exact_k==triton={np.array_equal(keep, tri)} "
              f"kept {int(keep.sum())}/{int(tri.sum())} argmax {int(np.argmax(score))} expected {r['expected']}")
    # whole dump: how often do the two top-k tie semantics give different kept sets / tokens?
    n = n_ties = n_diff_set = n_diff_tok = 0
    for rid, info in idx["requests"].items():
        arr = np.load(f"{R}/replay/logits_{name}/{info['file']}")
        rec = next(x for x in prod["records"] if x["request_id"] == rid)
        for i, pos in enumerate(info["positions"]):
            if pos - (info["prompt_len"] - 1) >= len(info["recorded"]):
                continue
            row = S.logits_from_bytes(arr[i].tobytes(), idx["logits_dtype"])
            keep_e, nt, _ = exact_k_masks(row, K, P)
            keep_t, _ = S.nucleus_masks(row, K, P)
            n += 1
            n_ties += nt > 1
            if not np.array_equal(keep_e, keep_t):
                n_diff_set += 1
                noise = S.gumbel_noise(rec["seed"], pos, row.shape[0]).astype(np.float32)
                sc = row.astype(np.float32) + noise
                if int(np.argmax(np.where(keep_e, sc, -np.inf))) != int(np.argmax(np.where(keep_t, sc, -np.inf))):
                    n_diff_tok += 1
    print(f"seed {s}: positions {n}, ties at the 50th logit {n_ties}, kept-set differs between tie semantics {n_diff_set}, "
          f"selected token differs {n_diff_tok}")
