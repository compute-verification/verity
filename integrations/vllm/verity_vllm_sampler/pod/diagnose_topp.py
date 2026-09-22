"""Explain independent-vs-same_runtime disagreements on top-k/top-p rows (contract §7 "find out WHY").

    python verity_vllm_sampler/pod/diagnose_topp.py --production prod.json --logits-dir logits_dir \
        --analysis select_x_auto.json --out diag.json

For every position where the independent prediction != recorded token (or is uncertified), compute on the
same row: (a) the numpy kept set (`select.nucleus_masks`), (b) vLLM's Triton `apply_top_k_top_p_triton`
kept set, (c) vLLM's torch-sort `apply_top_k_top_p_pytorch` kept set; report the symmetric differences,
each differing token's rank, prob and cumulative mass above it (float64), and the fp32 boundary picture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from verity_vllm_sampler import select as S  # noqa: E402
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch  # noqa: E402
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton  # noqa: E402


def kept_sets(row_f32: np.ndarray, k: int, p: float, batch_rows: int = 1) -> dict[str, np.ndarray]:
    keep_np, amb = S.nucleus_masks(row_f32, k, p)
    x = torch.from_numpy(row_f32.copy()).cuda().unsqueeze(0).repeat(batch_rows, 1)
    kt = torch.full((batch_rows,), k, dtype=torch.int32, device="cuda")
    pt = torch.full((batch_rows,), p, dtype=torch.float32, device="cuda")
    tri = apply_top_k_top_p_triton(x.clone(), kt, pt)[0]
    tor = apply_top_k_top_p_pytorch(x.clone(), kt, pt)[0]
    return {"numpy": keep_np, "ambiguous": amb, "triton": torch.isfinite(tri).cpu().numpy(), "torch": torch.isfinite(tor).cpu().numpy()}


def describe(row_f32: np.ndarray, k: int, p: float, tokens: list[int], sets: dict[str, np.ndarray]) -> list[dict]:
    x = row_f32.astype(np.float64)
    V = x.shape[0]
    kth = np.partition(x, V - k)[V - k]
    keep_k = x >= kth
    xk = np.where(keep_k, x, -np.inf)
    e = np.exp(xk - xk.max())
    probs = e / e.sum()
    order = np.argsort(-xk, kind="stable")
    rank = np.empty(V, dtype=np.int64)
    rank[order] = np.arange(V)
    cum_above = np.concatenate(([0.0], np.cumsum(probs[order])[:-1]))
    out = []
    for t in tokens:
        r = int(rank[t])
        out.append({"token": int(t), "logit": float(x[t]), "rank_desc": r, "prob": float(probs[t]),
                    "mass_above_f64": float(cum_above[r]), "mass_incl_f64": float(cum_above[r] + probs[t]),
                    "kept_numpy": bool(sets["numpy"][t]), "kept_triton": bool(sets["triton"][t]), "kept_torch": bool(sets["torch"][t]),
                    "ambiguous_numpy": bool(sets["ambiguous"][t])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", required=True)
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-uncertified", action="store_true", help="also diagnose uncertified-but-exact positions")
    args = ap.parse_args()
    prod = json.load(open(args.production))
    idx = json.load(open(os.path.join(args.logits_dir, "index.json")))
    ana = json.load(open(args.analysis))
    recs = {r["request_id"]: r for r in prod["records"]}
    results = []
    for row in ana["positions"]:
        wrong = row["independent"] != row["expected"]
        if not wrong and not (args.all_uncertified and not row["certified"]):
            continue
        rec = recs[row["request_id"]]
        info = idx["requests"][row["request_id"]]
        arr = np.load(os.path.join(args.logits_dir, info["file"]))
        i = info["positions"].index(row["pos"])
        raw = arr[i]
        logits = S.logits_from_bytes(raw.tobytes(), idx["logits_dtype"])
        sp = rec["sampling"]
        T, k, p = S.effective_params(sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0), logits.shape[0])
        sets = kept_sets(logits, k, p)
        sets16 = kept_sets(logits, k, p, batch_rows=16)  # the production batch shape (16 rows)
        diff_nt = np.flatnonzero(sets["numpy"] != sets["triton"]).tolist()
        diff_tt = np.flatnonzero(sets["torch"] != sets["triton"]).tolist()
        diff_n_torch = np.flatnonzero(sets["numpy"] != sets["torch"]).tolist()
        diff_tri_b = np.flatnonzero(sets["triton"] != sets16["triton"]).tolist()
        # the scores (logit + gumbel noise) of the tokens involved, from the numpy model
        noise = S.gumbel_noise(rec["seed"], row["pos"], logits.shape[0])
        score = logits.astype(np.float32) + noise.astype(np.float32)
        involved = sorted(set(diff_nt + diff_tt + diff_n_torch + [row["expected"], row["independent"]]))
        desc = describe(logits, k, p, involved, sets)
        for d in desc:
            d["score_logit_plus_noise"] = float(score[d["token"]])
        results.append({
            "index": rec["index"], "k": row["k"], "pos": row["pos"], "expected": row["expected"], "independent": row["independent"],
            "certified": row["certified"], "reason": row["reason"], "same_runtime": row.get("same_runtime"),
            "kept_counts": {"numpy": int(sets["numpy"].sum()), "triton": int(sets["triton"].sum()), "torch": int(sets["torch"].sum())},
            "numpy_vs_triton_diff_tokens": diff_nt, "torch_vs_triton_diff_tokens": diff_tt, "numpy_vs_torch_diff_tokens": diff_n_torch,
            "triton_batch1_vs_batch16_diff_tokens": diff_tri_b,
            "tokens": desc,
            "verdict": ("numpy == torch-sort != Triton" if not diff_n_torch and diff_tt else
                        "numpy != torch-sort" if diff_n_torch else "all three agree (noise/score issue)"),
        })
        print(json.dumps({k_: v for k_, v in results[-1].items() if k_ != "tokens"}))
    json.dump({"analysis": args.analysis, "n": len(results), "rows": results}, open(args.out, "w"), indent=1)
    print("DIAG_DONE", args.out, len(results))


if __name__ == "__main__":
    main()
