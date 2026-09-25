"""Beyond GEMM: can an explicit IEEE-sequence model reproduce vLLM's element-wise / row-wise
kernels bit-exactly on Ampere?  (GPU host only; needs the vLLM wheel for its custom ops.)

Probes, each against the kernel vLLM 0.28 ships:

* ``silu_and_mul`` (``csrc/activation_kernels.cu``, packed BF16 path):
  ``out = bf16( float(bf16(silu_f32(gate))) * float(up) )``.  ``silu_f32`` is a function of one
  BF16 word, so the *whole* hardware function is a 65536-entry table read off the GPU once
  (``up = 1``).  We report how many table entries differ from a correctly-rounded IEEE
  ``x / (1 + exp(-x))`` (this is where ``expf``/division approximations would show) and the
  natural-input match of the table model and of the pure-IEEE model.
* ``rms_norm`` -- three kernels: vLLM's Triton ``_rms_norm_kernel`` (batch-invariant path when
  there is no residual), the CUDA ``rms_norm_kernel`` and the CUDA ``fused_add_rms_norm_kernel``
  (batch-invariant path *with* residual, i.e. every norm after the first).  All three end in a
  single per-row FP32 scalar ``s = rsqrt(mean(x^2) + eps)`` computed by an *approximate*
  instruction (``MUFU.RSQ`` via ``rsqrtf``; Triton: ``sqrt.approx`` + ``div.full``).  The model
  computes the IEEE correctly-rounded ``s`` and then asks for which ulp offset ``k`` (if any)
  ``s + k ulp`` reproduces the whole GPU row: ``k = 0`` everywhere would make the op modelled;
  ``|k| <= 2`` shows the boundary is exactly the approximate reciprocal square root.
* ``rotary_embedding`` (``csrc/pos_encoding_kernels.cu``, NeoX style, BF16 cos/sin cache):
  ``x' = x cos - y sin``, ``y' = y cos + x sin`` in FP32 then RNE to BF16.  With BF16 operands
  every product is exact in FP32, so FMA contraction cannot change the result; the model is
  exact arithmetic (float64) rounded once to FP32 and once to BF16.  The cache itself is a
  runtime-computed constant; we compare vLLM's cache with two independent recomputations.

Usage: python -m verity_vllm.program.kernels.beyond_gemm --acts /workspace/results/numerics/qwen_acts --out beyond_gemm.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HIDDEN, INTER, HEAD, NQ, NKV = 1536, 8960, 128, 12, 2
EPS = 1e-6
ROPE_BASE, MAX_POS = 1_000_000.0, 131072


def words(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def from_words(w: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).view(torch.bfloat16).cuda()


def bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (w.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = (u & 0x7F800000) == 0x7F800000
    nan &= (u & 0x007FFFFF) != 0
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    r[nan] = 0x7FC0
    return r


def f64_to_f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Exact value -> one RNE to FP32 -> one RNE to BF16 (what an FP32 kernel with exact
    intermediates does)."""
    return f32_to_bf16(np.asarray(x, dtype=np.float64).astype(np.float32))


def load_norm_weights() -> dict[str, np.ndarray]:
    from safetensors import safe_open

    from verity_vllm.config import checkpoints_hf_home

    snap = None
    root = os.path.join(os.environ.get("HF_HOME") or checkpoints_hf_home() or "", "hub", "models--Qwen--Qwen2.5-1.5B", "snapshots")
    for d in sorted(os.listdir(root)):
        if os.path.exists(os.path.join(root, d, "model.safetensors")):
            snap = os.path.join(root, d, "model.safetensors")
    out = {}
    with safe_open(snap, framework="pt") as f:
        for li in (0, 13, 27):
            for nm in ("input_layernorm", "post_attention_layernorm"):
                out[f"layers.{li}.{nm}"] = words(f.get_tensor(f"model.layers.{li}.{nm}.weight"))
        out["norm"] = words(f.get_tensor("model.norm.weight"))
    return out


# --- silu_and_mul ------------------------------------------------------------------------------


def silu_table_gpu(ops) -> np.ndarray:
    xs = np.arange(65536, dtype=np.uint16)
    d = 8192
    gate = xs.reshape(8, d)
    up = np.full((8, d), 0x3F80, np.uint16)  # 1.0
    inp = from_words(np.concatenate([gate, up], axis=1))
    out = torch.empty((8, d), dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul(out, inp)
    torch.cuda.synchronize()
    return words(out).reshape(-1)


def silu_table_ieee() -> np.ndarray:
    x = bf16_to_f32(np.arange(65536, dtype=np.uint16)).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        s = x / (1.0 + np.exp(-x))
    return f64_to_f32_to_bf16(s)


def silu_table_f32_numpy() -> np.ndarray:
    x = bf16_to_f32(np.arange(65536, dtype=np.uint16))
    with np.errstate(over="ignore", invalid="ignore"):
        s = (x / (np.float32(1.0) + np.exp(-x))).astype(np.float32)
    return f32_to_bf16(s)


def silu_mul_model(table: np.ndarray, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return f32_to_bf16(bf16_to_f32(table[gate]) * bf16_to_f32(up))


def probe_silu(ops, acts_dir: str, layers: list[int], Ms: list[int]) -> dict:
    t0 = time.time()
    gpu_tab = silu_table_gpu(ops)
    gpu_tab2 = silu_table_gpu(ops)
    ieee_tab = silu_table_ieee()
    f32_tab = silu_table_f32_numpy()
    xs = np.arange(65536, dtype=np.uint16)
    finite = np.isfinite(bf16_to_f32(xs))
    diff_ieee = np.nonzero((gpu_tab != ieee_tab) & finite)[0]
    diff_f32 = np.nonzero((gpu_tab != f32_tab) & finite)[0]
    res = {
        "kernel": "vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, silu_kernel, packed_silu_kernel, act_first=true, use_vec=true>",
        "model": "out = bf16(float(silu_bf16[gate]) * float(up)); silu_bf16 = 65536-entry table of bf16(silu_f32(x)) characterised once from the GPU (up = 1.0)",
        "table_deterministic_rerun": bool(np.array_equal(gpu_tab, gpu_tab2)),
        "table_entries_finite": int(finite.sum()),
        "table_vs_ieee_correctly_rounded": {
            "differing_entries": int(len(diff_ieee)),
            "examples": [
                {"x": f"{int(i):04x}", "x_value": float(bf16_to_f32(np.array([i], np.uint16))[0]), "gpu": f"{int(gpu_tab[i]):04x}", "ieee": f"{int(ieee_tab[i]):04x}"}
                for i in diff_ieee[:12]
            ],
        },
        "table_vs_numpy_float32": {"differing_entries": int(len(diff_f32))},
        "nan_inputs_map_to": sorted({f"{int(v):04x}" for v in gpu_tab[~finite & ((xs & 0x7F80) == 0x7F80) & ((xs & 0x7F) != 0)]}),
        "natural": [],
    }
    for li in layers:
        data = np.load(os.path.join(acts_dir, f"layer{li}.npz"))
        x_all, w = data["gate_up_proj.x"], data["gate_up_proj.w"]
        for M in Ms:
            if M > x_all.shape[0]:
                continue
            gu = F.linear(from_words(x_all[:M]), from_words(w))  # natural gate||up rows (any GEMM will do)
            out = torch.empty((M, INTER), dtype=torch.bfloat16, device="cuda")
            torch.ops._C.silu_and_mul(out, gu)
            torch.cuda.synchronize()
            gpu = words(out)
            guw = words(gu)
            gate, up = guw[:, :INTER], guw[:, INTER:]
            t1 = time.time()
            m_tab = silu_mul_model(gpu_tab, gate, up)
            dt = time.time() - t1
            m_ieee = silu_mul_model(ieee_tab, gate, up)
            mism = np.argwhere(m_tab != gpu)
            res["natural"].append({
                "label": f"layers.{li}.mlp.act_fn", "M": M, "attempted": int(gpu.size),
                "exact_table_model": int(np.sum(m_tab == gpu)),
                "exact_pure_ieee_model": int(np.sum(m_ieee == gpu)),
                "cpu_seconds_table_model": dt,
                "first_mismatch_table_model": None if len(mism) == 0 else {
                    "row": int(mism[0][0]), "col": int(mism[0][1]),
                    "gate": f"{int(gate[mism[0][0], mism[0][1]]):04x}", "up": f"{int(up[mism[0][0], mism[0][1]]):04x}",
                    "gpu": f"{int(gpu[mism[0][0], mism[0][1]]):04x}", "model": f"{int(m_tab[mism[0][0], mism[0][1]]):04x}"},
            })
    res["seconds"] = time.time() - t0
    return res


# --- rms_norm ----------------------------------------------------------------------------------


def ulp_offsets(s: np.ndarray, k: int) -> np.ndarray:
    """s (float32 array) moved by k ulps (k may be negative)."""
    u = s.view(np.int32)
    return (u + np.int32(k)).view(np.float32)


def rms_rows_model(x_bf16: np.ndarray, w_bf16: np.ndarray, s: np.ndarray) -> np.ndarray:
    """out = bf16((x * s) * w) with FP32 arithmetic, s per row (float32 [M])."""
    x = bf16_to_f32(x_bf16)
    w = bf16_to_f32(w_bf16)
    y = (x * s[:, None]).astype(np.float32)
    y = (y * w[None, :]).astype(np.float32)
    return f32_to_bf16(y)


def ieee_inv_rms(x_bf16: np.ndarray, eps: float, order: str) -> np.ndarray:
    x = bf16_to_f32(x_bf16)
    if order == "exact":
        var = (x.astype(np.float64) ** 2).sum(axis=1)
        var32 = var.astype(np.float32)
    elif order == "sequential_f32":
        var32 = np.zeros(x.shape[0], np.float32)
        for j in range(x.shape[1]):
            var32 = (var32 + x[:, j] * x[:, j]).astype(np.float32)
    else:
        raise ValueError(order)
    mean = (var32 / np.float32(x.shape[1])).astype(np.float32)
    arg = (mean + np.float32(eps)).astype(np.float32)
    # correctly rounded 1/sqrt in FP32: evaluate in float64, round once
    return (1.0 / np.sqrt(arg.astype(np.float64))).astype(np.float32)


def fit_rows(gpu: np.ndarray, x_bf16: np.ndarray, w_bf16: np.ndarray, s0: np.ndarray, kmax: int = 4) -> dict:
    """For each row, the smallest |k| with rows_model(s0 + k ulp) == gpu row, else None."""
    M = gpu.shape[0]
    best = np.full(M, 99, np.int32)
    for k in sorted(range(-kmax, kmax + 1), key=abs):
        m = rms_rows_model(x_bf16, w_bf16, ulp_offsets(s0, k))
        ok = np.all(m == gpu, axis=1) & (best == 99)
        best[ok] = k
    hist = {str(k): int(np.sum(best == k)) for k in range(-kmax, kmax + 1) if np.sum(best == k)}
    hist["none"] = int(np.sum(best == 99))
    m0 = rms_rows_model(x_bf16, w_bf16, s0)
    return {
        "rows": M,
        "rows_exact_with_ieee_rsqrt": int(np.sum(best == 0)),
        "rows_exact_with_some_k": int(np.sum(best != 99)),
        "k_histogram": hist,
        "elements_exact_with_ieee_rsqrt": int(np.sum(m0 == gpu)),
        "elements": int(gpu.size),
    }


def probe_rmsnorm(ops, acts_dir: str, layers: list[int], Ms: list[int]) -> dict:
    try:  # pinned nightly (0.28.1rc1.dev472) moved the module
        from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant
    except ImportError:  # stable 0.28.0
        from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

    norm_w = load_norm_weights()
    rng = np.random.default_rng(7)
    res = {"eps": EPS, "kernels": {
        "triton": "vllm.model_executor.layers.batch_invariant._rms_norm_kernel (BLOCK_SIZE=1024; tl.sqrt + fp32 division; batch-invariant path without residual)",
        "cuda_rms_norm": "vllm::rms_norm_kernel (cub BlockReduce, rsqrtf)",
        "cuda_fused_add_rms_norm": "vllm::fused_add_rms_norm_kernel<bf16, width=8> (bf16x2 residual add, FP32 sum of squares per 8-vector, cub BlockReduce<1024>, rsqrtf; batch-invariant path with residual)",
    }, "cases": []}

    def one(label, x_bf16, w_bf16, resid_bf16=None):
        M = x_bf16.shape[0]
        xt, wt = from_words(x_bf16), from_words(w_bf16)
        out = {"label": label, "M": M}
        # Triton
        y = rms_norm_batch_invariant(xt, wt, EPS)
        y2 = rms_norm_batch_invariant(xt, wt, EPS)
        torch.cuda.synchronize()
        gpu = words(y)
        s0 = ieee_inv_rms(x_bf16, EPS, "exact")
        out["triton"] = fit_rows(gpu, x_bf16, w_bf16, s0)
        out["triton"]["deterministic_rerun"] = bool(torch.equal(y, y2))
        # CUDA rms_norm
        o = torch.empty_like(xt)
        ops.rms_norm(o, xt, wt, EPS)
        torch.cuda.synchronize()
        out["cuda_rms_norm"] = fit_rows(words(o), x_bf16, w_bf16, s0)
        # CUDA fused_add_rms_norm (in place): residual' = bf16(x + residual); out = norm(residual')
        if resid_bf16 is None:
            resid_bf16 = f32_to_bf16(rng.standard_normal(x_bf16.shape).astype(np.float32) * 0.5)
        xi, ri = from_words(x_bf16), from_words(resid_bf16)
        ops.fused_add_rms_norm(xi, ri, wt, EPS)
        torch.cuda.synchronize()
        gpu_out, gpu_res = words(xi), words(ri)
        res_model = f32_to_bf16(bf16_to_f32(x_bf16) + bf16_to_f32(resid_bf16))
        s1 = ieee_inv_rms(res_model, EPS, "exact")
        fr = fit_rows(gpu_out, res_model, w_bf16, s1)
        fr["residual_out_exact_elements"] = int(np.sum(gpu_res == res_model))
        fr["residual_out_elements"] = int(gpu_res.size)
        out["cuda_fused_add_rms_norm"] = fr
        return out

    for li in layers:
        data = np.load(os.path.join(acts_dir, f"layer{li}.npz"))
        x_all = data["gate_up_proj.x"]  # post-norm hidden rows: natural magnitudes/outlier structure
        for M in Ms:
            if M > x_all.shape[0]:
                continue
            res["cases"].append(one(f"layers.{li}.post_attention_layernorm(natural-ish rows)", x_all[:M], norm_w[f"layers.{li}.post_attention_layernorm"]))
    # synthetic residual-stream-like rows: gaussian with a few outlier channels
    for M in (64, 512):
        x = rng.standard_normal((M, HIDDEN)).astype(np.float32) * 0.3
        x[:, rng.choice(HIDDEN, 8, replace=False)] *= 60.0
        res["cases"].append(one("synthetic residual stream (outlier channels)", f32_to_bf16(x), norm_w["layers.13.input_layernorm"]))
    return res


# --- rotary embedding --------------------------------------------------------------------------


def rope_cache_numpy(dtype_steps: str) -> np.ndarray:
    """cos||sin cache [MAX_POS, HEAD] as BF16 words.  ``exact``: float64 then RNE to BF16;
    ``torch_f32``: emulate torch's float32 steps (inv_freq = 1/base**(i/d) in f32, t*inv_freq in
    f32, cos/sin evaluated in float64 on the f32 angle, rounded to f32 then BF16)."""
    i = np.arange(0, HEAD, 2, dtype=np.float64)
    t = np.arange(MAX_POS, dtype=np.float64)
    if dtype_steps == "exact":
        inv = 1.0 / (ROPE_BASE ** (i / HEAD))
        ang = t[:, None] * inv[None, :]
        return np.concatenate([f64_to_f32_to_bf16(np.cos(ang)), f64_to_f32_to_bf16(np.sin(ang))], axis=1)
    inv = (np.float32(1.0) / (np.float32(ROPE_BASE) ** (i.astype(np.float32) / np.float32(HEAD)))).astype(np.float32)
    ang = (t.astype(np.float32)[:, None] * inv[None, :]).astype(np.float32)
    return np.concatenate([f64_to_f32_to_bf16(np.cos(ang.astype(np.float64))), f64_to_f32_to_bf16(np.sin(ang.astype(np.float64)))], axis=1)


def rope_model(x_bf16: np.ndarray, cache_bf16: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """x [M, H*HEAD] BF16 words, NeoX rotation with the BF16 cache; exact arithmetic, RNE to FP32,
    RNE to BF16."""
    M, D = x_bf16.shape
    nh = D // HEAD
    x = bf16_to_f32(x_bf16).astype(np.float64).reshape(M, nh, HEAD)
    cs = bf16_to_f32(cache_bf16[positions]).astype(np.float64)  # [M, HEAD]
    cos, sin = cs[:, None, : HEAD // 2], cs[:, None, HEAD // 2 :]
    a, b = x[:, :, : HEAD // 2], x[:, :, HEAD // 2 :]
    out = np.empty_like(x)
    out[:, :, : HEAD // 2] = a * cos - b * sin
    out[:, :, HEAD // 2 :] = b * cos + a * sin
    return f64_to_f32_to_bf16(out).reshape(M, D)


def probe_rope(ops, acts_dir: str, layers: list[int], Ms: list[int]) -> dict:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    # the engine builds the layer under a CUDA default device, so the cache is computed on the GPU
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        rope = get_rope(HEAD, max_position=MAX_POS, is_neox_style=True, rope_parameters={"rope_type": "default", "rope_theta": ROPE_BASE}, dtype=torch.bfloat16)
    cache = rope.cos_sin_cache
    assert cache.dtype == torch.bfloat16 and tuple(cache.shape) == (MAX_POS, HEAD), (cache.dtype, cache.shape)
    cache_w = words(cache)
    exact_w = rope_cache_numpy("exact")
    t32_w = rope_cache_numpy("torch_f32")
    res = {
        "kernel": "vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, IS_NEOX=true> (in-place on q and k)",
        "cache": {
            "dtype": str(cache.dtype), "shape": list(cache.shape), "computed_on": str(cache.device),
            "entries": int(cache_w.size),
            "differs_from_exact_float64_recompute": int(np.sum(cache_w != exact_w)),
            "differs_from_torch_float32_step_emulation": int(np.sum(cache_w != t32_w)),
            "note": "the checker must use the opened cache (derived leaf) unless these counts are 0",
        },
        "model": "NeoX pairs (i, i+64): x' = x*cos - y*sin, y' = y*cos + x*sin; BF16 operands -> products exact in FP32 -> one RNE to FP32 (FMA contraction irrelevant) -> RNE to BF16",
        "natural": [],
    }
    for li in layers:
        data = np.load(os.path.join(acts_dir, f"layer{li}.npz"))
        x_all, w = data["qkv_proj.x"], data["qkv_proj.w"]
        b = data["qkv_proj.b"]
        for M in Ms:
            if M > x_all.shape[0]:
                continue
            qkv = F.linear(from_words(x_all[:M]), from_words(w), from_words(b))
            q = qkv[:, : NQ * HEAD].contiguous()
            k = qkv[:, NQ * HEAD : (NQ + NKV) * HEAD].contiguous()
            q_in, k_in = words(q), words(k)
            pos_np = (np.arange(M) + (0 if li == 0 else 1000 * li)).astype(np.int64)
            positions = torch.from_numpy(pos_np).cuda()
            ops.rotary_embedding(positions, q, k, HEAD, cache, True)
            torch.cuda.synchronize()
            gq, gk = words(q), words(k)
            t1 = time.time()
            mq = rope_model(q_in, cache_w, pos_np)
            mk = rope_model(k_in, cache_w, pos_np)
            dt = time.time() - t1
            mq_exact = rope_model(q_in, exact_w, pos_np)
            mism = np.argwhere(mq != gq)
            res["natural"].append({
                "label": f"layers.{li}.self_attn.rotary_emb", "M": M, "positions": [int(pos_np[0]), int(pos_np[-1])],
                "q_attempted": int(gq.size), "q_exact_with_opened_cache": int(np.sum(mq == gq)),
                "k_attempted": int(gk.size), "k_exact_with_opened_cache": int(np.sum(mk == gk)),
                "q_exact_with_float64_recomputed_cache": int(np.sum(mq_exact == gq)),
                "cpu_seconds": dt,
                "first_mismatch_q": None if len(mism) == 0 else {"row": int(mism[0][0]), "col": int(mism[0][1]), "gpu": f"{int(gq[mism[0][0], mism[0][1]]):04x}", "model": f"{int(mq[mism[0][0], mism[0][1]]):04x}"},
            })
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,13,27")
    ap.add_argument("--ms", default="7,64,512")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    import vllm
    import vllm._custom_ops as ops

    layers = [int(v) for v in args.layers.split(",")]
    Ms = [int(v) for v in args.ms.split(",")]
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "vllm": vllm.__version__}
    skip = set(args.skip.split(",")) if args.skip else set()
    for name, fn in (("silu_and_mul", probe_silu), ("rms_norm", probe_rmsnorm), ("rotary_embedding", probe_rope)):
        if name in skip:
            continue
        t0 = time.time()
        try:
            out[name] = fn(ops, args.acts, layers, Ms)
        except Exception as e:  # keep the other probes' results
            import traceback

            out[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        out[name]["seconds"] = time.time() - t0
        print(name, json.dumps({k: v for k, v in out[name].items() if k not in ("natural", "cases", "traceback")}, indent=1)[:3000], flush=True)
        for c in out[name].get("natural", []) + out[name].get("cases", []):
            print("  ", json.dumps(c), flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
