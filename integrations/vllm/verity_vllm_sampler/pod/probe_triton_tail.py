"""Is the Triton top-p under-keep caused by the row's tail (sigma-truncation outlier pre-pass)?

Take each disagreeing row and (a) keep only the top-50 logits, tail -> -inf; (b) tail -> -1e4; (c) tail
replaced by N(mean_tail, std_tail) noise; (d) tail unchanged but top-50 shifted +20; (e) the row in fp32
with the boundary token nudged by +/- half a bf16 ulp; report Triton vs torch kept counts and masses.
Also emulate the kernel's first pass: avg/std/outlier pivot and num_outliers vs k.
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


def run(row: np.ndarray, k: int, p: float):
    x = torch.from_numpy(row.astype(np.float32)).cuda().unsqueeze(0)
    kt = torch.tensor([k], dtype=torch.int32, device="cuda")
    pt = torch.tensor([p], dtype=torch.float32, device="cuda")
    tri = torch.isfinite(apply_top_k_top_p_triton(x.clone(), kt, pt)[0]).cpu().numpy()
    tor = torch.isfinite(apply_top_k_top_p_pytorch(x.clone(), kt, pt)[0]).cpu().numpy()
    return tri, tor


def mass(row: np.ndarray, k: int, keep: np.ndarray) -> float:
    x = row.astype(np.float64)
    fin = np.isfinite(x)
    V = x.shape[0]
    kth = np.partition(np.where(fin, x, -np.inf), V - k)[V - k]
    xk = np.where(x >= kth, x, -np.inf)
    e = np.exp(xk - xk.max())
    probs = e / e.sum()
    return float(probs[keep].sum())


def variants(row: np.ndarray, k: int):
    order = np.argsort(-row)
    top = order[:k]
    tail = order[k:]
    out = {}
    r = row.copy(); r[tail] = -np.inf; out["tail_-inf"] = r
    r = row.copy(); r[tail] = -1e4; out["tail_-1e4"] = r
    rng = np.random.default_rng(0)
    r = row.copy(); r[tail] = rng.normal(row[tail].mean(), row[tail].std(), tail.size).astype(np.float32); out["tail_gaussian_same_moments"] = r
    r = row.copy(); r[tail] = rng.normal(row[tail].mean(), 0.5, tail.size).astype(np.float32); out["tail_gaussian_narrow"] = r
    r = row.copy(); r[top] += 20.0; out["top50_+20"] = r
    r = row.copy(); r[tail] -= 20.0; out["tail_-20"] = r
    r = row.copy(); r[tail] = row[tail].mean(); out["tail_constant_mean"] = r
    return out


def first_pass_stats(row: np.ndarray, sigma: float = 3.0):
    x = row[np.isfinite(row)].astype(np.float64)
    avg, std = x.mean(), x.std()
    piv = avg + std * sigma
    return {"avg": float(avg), "std": float(std), "outlier_pivot_sigma3": float(piv), "num_outliers_sigma3": int((x > piv).sum()),
            "outlier_pivot_sigma2": float(avg + 2 * std), "num_outliers_sigma2": int((x > avg + 2 * std).sum()),
            "outlier_pivot_sigma4": float(avg + 4 * std), "num_outliers_sigma4": int((x > avg + 4 * std).sum()),
            "max": float(x.max()), "top50_min": float(np.sort(x)[-50])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", nargs="+", required=True)
    ap.add_argument("--logits-root", required=True)
    ap.add_argument("--prod-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = []
    for dpath in args.diag:
        d = json.load(open(dpath))
        name = os.path.basename(d["analysis"]).replace("select_", "").replace("_auto.json", "")
        prod = json.load(open(os.path.join(args.prod_root, name + ".json")))
        idx = json.load(open(os.path.join(args.logits_root, "logits_" + name, "index.json")))
        recs = {r["index"]: r for r in prod["records"]}
        for r in d["rows"]:
            rec = recs[r["index"]]
            info = idx["requests"][rec["request_id"]]
            arr = np.load(os.path.join(args.logits_root, "logits_" + name, info["file"]))
            row = S.logits_from_bytes(arr[info["positions"].index(r["pos"])].tobytes(), idx["logits_dtype"])
            k, p = 50, 0.9
            tri, tor = run(row, k, p)
            entry = {"run": name, "index": r["index"], "k": r["k"], "first_pass": first_pass_stats(row),
                     "original": {"triton": int(tri.sum()), "torch": int(tor.sum()), "triton_mass": mass(row, k, tri), "torch_mass": mass(row, k, tor)},
                     "variants": {}}
            for vn, vr in variants(row, k).items():
                t, o = run(vr, k, p)
                entry["variants"][vn] = {"triton": int(t.sum()), "torch": int(o.sum()), "triton_mass": mass(vr, k, t), "torch_mass": mass(vr, k, o),
                                        "agree": bool(np.array_equal(t, o)), "first_pass_num_outliers_sigma3": first_pass_stats(vr)["num_outliers_sigma3"]}
            out.append(entry)
            print(json.dumps({"run": name, "index": r["index"], "first_pass": entry["first_pass"], "original": entry["original"]}))
            for vn, v in entry["variants"].items():
                print(f"    {vn:28s} triton {v['triton']:>2} ({v['triton_mass']:.4f})  torch {v['torch']:>2} ({v['torch_mass']:.4f})  agree={v['agree']}  outliers3={v['first_pass_num_outliers_sigma3']}")
    json.dump(out, open(args.out, "w"), indent=1)
    print("TAIL_PROBE_DONE", args.out)


if __name__ == "__main__":
    main()
