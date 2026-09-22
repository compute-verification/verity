"""[GEN sampling] POD-ONLY probe: is `verity_capture.bench.gen_sampling_rng` bit-exact against the real Triton / vLLM kernels?

Runs (in the pinned vLLM env, GPU) `tl.randint`, `tl.rand`, vLLM's `_gumbel_sample_kernel` chain (`tl_rand32` + the flipped
Gumbel transform) and `gumbel_sample` itself on a grid of seeds / positions, and compares element-wise with the numpy
recomputation.  Writes a JSON report; the laptop never imports this file (torch + triton).

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.bench.gen_sampling_triton_probe --out /vol/cp/sampling/rng_probe.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch
import triton
import triton.language as tl

from verity_capture.bench import gen_sampling_rng as R


@triton.jit
def _probe_kernel(seed, out_int_ptr, out_u_ptr, out_g_ptr, gs_ptr, pos, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = tl.randint(seed, offs.to(tl.int64))          # first level as vLLM uses it: int64 seed, int64 offset (pos)
    tl.store(out_int_ptr + offs, r, mask=mask)
    gumbel_seed = tl.randint(seed, pos)              # scalar: the per-(seed, pos) stream key
    tl.store(gs_ptr + offs, gumbel_seed, mask=offs == 0)
    u = tl.rand(gumbel_seed, offs)                   # second level: uint32 seed, int32 keys (vocab index)
    tl.store(out_u_ptr + offs, u, mask=mask)
    u = tl.maximum(u, 4.6566127342e-10)
    g = -tl.log(-tl.extra.cuda.libdevice.log1p(-u))
    tl.store(out_g_ptr + offs, g, mask=mask)


def probe(seed: int, pos: int, n: int) -> dict:
    dev = "cuda"
    out_int = torch.empty(n, dtype=torch.int32, device=dev)
    out_u = torch.empty(n, dtype=torch.float32, device=dev)
    out_g = torch.empty(n, dtype=torch.float32, device=dev)
    gs = torch.zeros(n, dtype=torch.int32, device=dev)
    BLOCK = 1024
    _probe_kernel[(triton.cdiv(n, BLOCK),)](seed, out_int, out_u, out_g, gs, pos, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    t_int = out_int.cpu().numpy().view(np.uint32)
    t_gs = int(gs[0].item()) & 0xFFFFFFFF
    t_u = out_u.cpu().numpy()
    t_g = out_g.cpu().numpy()
    n_int = R.tl_randint(seed, np.arange(n, dtype=np.int64))
    n_gs = R.gumbel_seed_of(seed, pos)
    n_u = R.tl_rand(n_gs, np.arange(n, dtype=np.int32), seed_bits=32)
    n_g = R.gumbel_noise(seed, pos, n)
    du = np.abs(t_u.astype(np.float64) - n_u.astype(np.float64))
    ulps_g = np.abs(t_g.view(np.int32).astype(np.int64) - n_g.view(np.int32).astype(np.int64))
    dg = np.abs(t_g.astype(np.float64) - n_g.astype(np.float64))
    rel = dg / np.maximum(np.abs(t_g.astype(np.float64)), 1e-30)
    big = np.abs(t_g) >= 1e-3      # away from the zero crossing of the Gumbel transform (u ~ 1 - 1/e), where ulps are tiny
    return {"seed": seed, "pos": pos, "n": n,
            "randint_exact": bool(np.array_equal(t_int, n_int)), "gumbel_seed_triton": t_gs, "gumbel_seed_numpy": n_gs,
            "gumbel_seed_exact": t_gs == n_gs, "rand_exact": bool(np.array_equal(t_u.view(np.uint32), n_u.view(np.uint32))),
            "rand_max_abs_diff": float(du.max()), "gumbel_max_ulps": int(ulps_g.max()), "gumbel_mean_ulps": float(ulps_g.mean()),
            "gumbel_frac_exact": float((ulps_g == 0).mean()), "gumbel_max_abs_diff": float(dg.max()), "gumbel_max_rel_diff": float(rel.max()),
            "gumbel_max_ulps_away_from_zero": int(ulps_g[big].max()) if big.any() else 0,
            "gumbel_ulps_hist": {str(k): int((ulps_g == k).sum()) for k in range(0, int(min(ulps_g.max(), 8)) + 1)}}


def probe_sampler(seed: int, pos: int, V: int, temperature: float, trials: int, apply_temperature: bool) -> dict:
    """vLLM's own `gumbel_sample` on random bf16-ish logits vs the numpy `sampled_token`: token agreement and margins."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(1234)
    agree = 0
    margins = []
    worst = None
    for t in range(trials):
        logits = (torch.randn(1, V, generator=g) * 4.0).to(torch.bfloat16).to(torch.float32).to(dev)
        if not apply_temperature and temperature not in (0.0, 1.0):
            logits = logits / temperature      # as `_temperature_kernel` does before the sampler
        mapping = torch.zeros(1, dtype=torch.int32, device=dev)
        temp = torch.full((1,), temperature, dtype=torch.float32, device=dev)
        seeds = torch.full((1,), seed, dtype=torch.int64, device=dev)
        posv = torch.full((1,), pos + t, dtype=torch.int64, device=dev)
        tok = int(gumbel_sample(logits, mapping, temp, seeds, posv, apply_temperature, False).item())
        rec = R.sampled_token(logits[0].cpu().numpy(), seed, pos + t, temperature=temperature, apply_temperature=apply_temperature)
        ok = rec["token"] == tok
        agree += ok
        margins.append(rec["margin_ulps"])
        if not ok and (worst is None or rec["margin_ulps"] < worst["margin_ulps"]):
            worst = {"trial": t, "triton": tok, "numpy": rec["token"], "margin_ulps": rec["margin_ulps"]}
    return {"seed": seed, "pos0": pos, "V": V, "temperature": temperature, "apply_temperature_in_kernel": apply_temperature,
            "trials": trials, "agree": agree, "min_margin_ulps": float(min(margins)), "median_margin_ulps": float(np.median(margins)),
            "worst_disagreement": worst}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {"schema": "verity-gen-sampling/rng-probe/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "triton": triton.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
           "probes": [], "sampler": []}
    try:
        import vllm
        rep["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        rep["vllm"] = f"unavailable: {e}"
    for seed, pos in [(0, 0), (0, 32), (0, 47), (1, 32), (1, 33), (2**40 + 7, 32), (-1, 32), (-(2**62), 5), (123456789, 2**33 + 1)]:
        rep["probes"].append(probe(seed, pos, 49152))
    try:
        for T, apply_t in [(0.7, False), (0.7, True), (1.0, False)]:
            rep["sampler"].append(probe_sampler(0, 32, 49152, T, 64, apply_t))
    except Exception as e:  # noqa: BLE001
        rep["sampler_error"] = repr(e)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    for p in rep["probes"]:
        print(f"seed={p['seed']} pos={p['pos']}: randint_exact={p['randint_exact']} gumbel_seed_exact={p['gumbel_seed_exact']} "
              f"rand_exact={p['rand_exact']} gumbel: frac_exact={p['gumbel_frac_exact']:.3f} max|d|={p['gumbel_max_abs_diff']:.3g} "
              f"max rel={p['gumbel_max_rel_diff']:.3g} max ulps(|g|>=1e-3)={p['gumbel_max_ulps_away_from_zero']}")
    for s in rep["sampler"]:
        print(f"sampler T={s['temperature']} apply_in_kernel={s['apply_temperature_in_kernel']}: agree {s['agree']}/{s['trials']} "
              f"min margin {s['min_margin_ulps']:.0f} ulp; worst {s['worst_disagreement']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
