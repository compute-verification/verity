"""Validate the numpy re-derivation in `select.py` against the installed Triton/vLLM kernels.

Runs on the pod (GPU).  Three checks:
  1. `tl.randint(seed, pos)` (int64 pos) and `tl_rand32(gumbel_seed, keys)` bit-exact vs numpy.
  2. Gumbel noise -log(-log1p(-u)) : ulp-difference histogram (libdevice vs glibc).
  3. vLLM `gumbel_sample` on random logits rows vs `independent_select`: token agreement and
     certification statistics; and `same_runtime_select` vs `gumbel_sample` (must be identical).

Writes JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/veritor/vllm-poc")
from verity_vllm_sampler import select as S  # noqa: E402
from vllm.v1.worker.gpu.sample import gumbel as G  # noqa: E402


@triton.jit
def _probe_kernel(seed_ptr, pos_ptr, u_ptr, noise_ptr, gseed_ptr, V, BLOCK: tl.constexpr):
    seed = tl.load(seed_ptr)  # int64
    pos = tl.load(pos_ptr)  # int64
    gumbel_seed = tl.randint(seed, pos)
    if tl.program_id(0) == 0:
        tl.store(gseed_ptr, gumbel_seed.to(tl.int64))
    block = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = block < V
    u = G.tl_rand32(gumbel_seed, block, includes_zero=False)
    noise = -tl.log(-G.tldevice.log1p(-u))
    tl.store(u_ptr + block, u, mask=mask)
    tl.store(noise_ptr + block, noise, mask=mask)


def probe(seed: int, pos: int, V: int, device="cuda"):
    seed_t = torch.tensor([seed], dtype=torch.int64, device=device)
    pos_t = torch.tensor([pos], dtype=torch.int64, device=device)
    u = torch.empty(V, dtype=torch.float32, device=device)
    noise = torch.empty(V, dtype=torch.float32, device=device)
    gs = torch.empty(1, dtype=torch.int64, device=device)
    BLOCK = 1024
    _probe_kernel[(triton.cdiv(V, BLOCK),)](seed_t, pos_t, u, noise, gs, V, BLOCK=BLOCK)
    torch.cuda.synchronize()
    return u.cpu().numpy(), noise.cpu().numpy(), int(gs.item())


def ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    return np.abs(ai - bi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--n-rows", type=int, default=200)
    args = ap.parse_args()
    V = args.vocab
    report: dict = {"vocab_size": V, "triton": triton.__version__, "torch": torch.__version__}

    # ---- 1/2: RNG + noise
    cases = [(1, 31), (2, 127), (3, 2047), (1, 0), (12345678901234, 5), (-7, 9), (2**62 + 3, 100000)]
    rng_rows = []
    for seed, pos in cases:
        u_gpu, noise_gpu, gs_gpu = probe(seed, pos, V)
        gs_np = S.gumbel_seed_for(seed, pos)
        signed = False
        if gs_np != gs_gpu:
            # try the signed-int32 typing hypothesis
            gs_np2 = S.gumbel_seed_for(seed, pos, seed_is_signed_int32=True)
            if gs_np2 == gs_gpu:
                signed = True
        u_np = np.maximum(S.tl_rand32(S.gumbel_seed_for(seed, pos, signed), np.arange(V, dtype=np.int32)), S.TL_RAND_MIN)
        noise_np = S.gumbel_noise(seed, pos, V, seed_is_signed_int32=signed)
        d = ulp_diff(noise_gpu, noise_np)
        y_np = (-np.log1p(-u_np)).astype(np.float32)
        budget = S.noise_error_budget(noise_np, y_np)
        absdiff = np.abs(noise_gpu.astype(np.float64) - noise_np.astype(np.float64))
        viol = int(np.sum(absdiff > budget))
        rng_rows.append({
            "budget_violations": viol,
            "budget_min_slack": float(np.min(budget - absdiff)),
            "seed": seed, "pos": pos,
            "gumbel_seed_gpu": gs_gpu, "gumbel_seed_numpy": gs_np,
            "gumbel_seed_match": gs_gpu == gs_np or signed, "signed_int32_hypothesis": signed,
            "u_bit_exact_fraction": float(np.mean(u_gpu.view(np.int32) == u_np.view(np.int32))),
            "noise_ulp_hist": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(d, 8), return_counts=True))},
            "noise_max_ulp": int(d.max()),
            "noise_max_abs_diff": float(np.max(np.abs(noise_gpu.astype(np.float64) - noise_np))),
        })
        print(json.dumps(rng_rows[-1]))
    report["rng"] = rng_rows

    # ---- 3: token agreement on random logits
    torch.manual_seed(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    agree = {"temp": [0, 0, 0], "topk_topp": [0, 0, 0]}  # [attempted, exact, certified]
    uncert_wrong = {"temp": 0, "topk_topp": 0}
    same_rt_agree = {"temp": 0, "topk_topp": 0}
    margins = []
    t0 = time.time()
    for i in range(args.n_rows):
        # realistic-ish logits: bf16, peaky distribution
        base = torch.randn(V, generator=g, device="cuda") * 2.0
        peaks = torch.randint(0, V, (8,), generator=g, device="cuda")
        base[peaks] += torch.linspace(10, 22, 8, device="cuda")
        logits_bf16 = base.to(torch.bfloat16)
        row = logits_bf16.float().cpu().numpy()
        seed = int(i % 3 + 1)
        pos = int(31 + i)
        for variant, (k, p) in {"temp": (0, 1.0), "topk_topp": (50, 0.9)}.items():
            # the GPU sample under vLLM's kernels (as in a batch of 1: torch sort path)
            res_rt = S.same_runtime_select(row, 1.0, k, p, seed, pos, topk_topp_impl="auto")
            res_rt_tri = S.same_runtime_select(row, 1.0, k, p, seed, pos, topk_topp_impl="triton") if (k or p < 1) else res_rt
            res_ind = S.independent_select(row, 1.0, k, p, seed, pos)
            agree[variant][0] += 1
            if res_ind.predicted == res_rt.predicted:
                agree[variant][1] += 1
            elif res_ind.certified:
                uncert_wrong[variant] += 1
                print("CERTIFIED MISMATCH", variant, i, res_ind.to_json(), res_rt.to_json())
            if res_ind.certified:
                agree[variant][2] += 1
            if res_rt_tri.predicted == res_rt.predicted:
                same_rt_agree[variant] += 1
            margins.append(res_ind.margin)
    report["token_agreement"] = {
        v: {"attempted": a[0], "independent_exact": a[1], "independent_certified": a[2],
            "certified_but_wrong": uncert_wrong[v], "triton_vs_torch_topk_topp_agree": same_rt_agree[v]}
        for v, a in agree.items()
    }
    report["min_margin"] = float(np.min([m for m in margins if m is not None]))
    report["seconds"] = time.time() - t0
    print(json.dumps(report["token_agreement"], indent=1))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("VALIDATE_DONE")


if __name__ == "__main__":
    main()
