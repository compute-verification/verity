"""Probe vLLM's Triton top-k/top-p kernel on the rows where it kept fewer tokens than the nucleus rule.

For each row: kept mass under Triton vs torch-sort vs numpy; how the Triton kept set changes with p in a
small neighbourhood; the same row with the boundary tie group broken (perturb each duplicate by one bf16
ulp) to see whether duplicates are what the kernel mishandles; and a synthetic bf16 row with a k-way tie
straddling p.
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
    npk, _ = S.nucleus_masks(row.astype(np.float32), k, p)
    return tri, tor, npk


def masses(row: np.ndarray, k: int, keep: np.ndarray) -> float:
    x = row.astype(np.float64)
    V = x.shape[0]
    kth = np.partition(x, V - k)[V - k]
    xk = np.where(x >= kth, x, -np.inf)
    e = np.exp(xk - xk.max())
    probs = e / e.sum()
    return float(probs[keep].sum())


def bf16_round(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", nargs="+", required=True, help="diag json files from diagnose_topp.py")
    ap.add_argument("--logits-root", required=True, help="replay dir root, e.g. $W3_RESULTS/replay")
    ap.add_argument("--prod-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out: dict = {"rows": [], "synthetic": []}
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
            sp = rec["sampling"]
            T, k, p = S.effective_params(sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0), row.shape[0])
            tri, tor, npk = run(row, k, p)
            entry = {"run": name, "index": r["index"], "k": r["k"], "pos": r["pos"],
                     "kept": {"triton": int(tri.sum()), "torch": int(tor.sum()), "numpy": int(npk.sum())},
                     "kept_mass": {"triton": masses(row, k, tri), "torch": masses(row, k, tor), "numpy": masses(row, k, npk)},
                     "p": p, "p_sweep": [], "tie_broken": None}
            for pp in (0.85, 0.88, 0.89, 0.895, 0.9, 0.905, 0.91, 0.92, 0.95):
                t2, o2, n2 = run(row, k, pp)
                entry["p_sweep"].append({"p": pp, "triton": int(t2.sum()), "torch": int(o2.sum()), "numpy": int(n2.sum()),
                                         "triton_mass": masses(row, k, t2), "numpy_mass": masses(row, k, n2)})
            # break the boundary tie: the differing tokens all share one logit; nudge them apart by bf16 ulps
            diff = sorted(set(r["numpy_vs_triton_diff_tokens"] + r["torch_vs_triton_diff_tokens"]))
            if diff:
                # spread the tied tokens by j * 2^-4 (one bf16 ulp at these magnitudes) downwards
                row3 = row.copy()
                base = row3[diff[0]]
                for j, t in enumerate(diff):
                    row3[t] = base - j * 0.0625
                t3, o3, n3 = run(row3, k, p)
                entry["tie_broken"] = {"tokens": diff, "spread_bf16_ulps_down": True,
                                       "kept": {"triton": int(t3.sum()), "torch": int(o3.sum()), "numpy": int(n3.sum())},
                                       "kept_mass": {"triton": masses(row3, k, t3), "torch": masses(row3, k, o3), "numpy": masses(row3, k, n3)},
                                       "triton_eq_torch": bool(np.array_equal(t3, o3)), "triton_eq_numpy": bool(np.array_equal(t3, n3))}
            out["rows"].append(entry)
            print(json.dumps({kk: v for kk, v in entry.items() if kk != "p_sweep"}))
    # synthetic: 30 distinct bf16 logits then a 4-way tie group straddling p
    rng = np.random.default_rng(0)
    V = 151936
    for trial in range(5):
        row = bf16_round(rng.normal(0, 2, V).astype(np.float32))
        top = np.argsort(-row)[:40]
        row[top[:30]] = bf16_round(np.linspace(20, 16, 30).astype(np.float32))
        tie_tokens = top[30:34]
        row[tie_tokens] = 15.0 + 0.0625 * trial  # equal logits
        # choose p so the boundary is inside the tie group
        x = row.astype(np.float64)
        kth = np.partition(x, V - 50)[V - 50]
        xk = np.where(x >= kth, x, -np.inf)
        e = np.exp(xk - xk.max()); probs = e / e.sum()
        order = np.argsort(-xk, kind="stable")
        cum = np.cumsum(probs[order])
        r0 = int(np.where(np.isin(order, tie_tokens))[0][0])
        p = float((cum[r0] + cum[r0 + 1]) / 2)  # boundary after ~1.5 of the 4 tie tokens
        tri, tor, npk = run(row, 50, p)
        out["synthetic"].append({"trial": trial, "p": p, "tie_logit": float(row[tie_tokens[0]]),
                                 "tie_group_mass_above": float(cum[r0 - 1]), "tie_group_mass_incl": float(cum[r0 + 3]),
                                 "kept": {"triton": int(tri.sum()), "torch": int(tor.sum()), "numpy": int(npk.sum())},
                                 "kept_mass": {"triton": masses(row, 50, tri), "torch": masses(row, 50, tor), "numpy": masses(row, 50, npk)},
                                 "tie_tokens_kept": {"triton": int(tri[tie_tokens].sum()), "torch": int(tor[tie_tokens].sum()), "numpy": int(npk[tie_tokens].sum())}})
        print("synthetic", json.dumps(out["synthetic"][-1]))
    json.dump(out, open(args.out, "w"), indent=1)
    print("PROBE_DONE", args.out)


if __name__ == "__main__":
    main()
