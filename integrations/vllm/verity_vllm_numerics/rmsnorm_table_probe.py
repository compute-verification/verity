"""RMSNorm as an explicit sequence: does a fully specified model (reduction order + the characterised
approximate ``rsqrt`` tables from ``rsqrt_probe.py``) reproduce vLLM's RMSNorm kernels bit-exactly?

``beyond_gemm.py`` showed every row of the three kernels is reproduced by ``bf16((x * s) * w)`` for
*some* ``s`` within +-2 ulp of the IEEE ``rsqrt(mean(x^2) + eps)``; the residual ambiguity is the
FP32 reduction order of the sum of squares and the approximate ``rsqrt``.  This probe enumerates
candidate reduction orders (read off the kernel sources) x mean/eps arithmetic x rsqrt function
(IEEE | characterised table) x output rounding, and reports, per kernel, the candidates that make
every row exact (``k = 0`` for all rows) on the natural rows and the synthetic outlier rows.

Kernels (the replay profile launches the first two: ``fused_add_rms_norm`` 55x and the Triton
``_rms_norm_kernel`` 1x per forward; ``rms_norm_kernel`` is the unvectorised CUDA sibling):

* ``vllm::fused_add_rms_norm_kernel<bf16, 8>``: 192 threads, one 8-vector each; ``_f16Vec::sum_squares``
  (``result += z.x*z.x + z.y*z.y`` over 4 bf16x2 pairs); ``cub::BlockReduce<float,1024>`` (warp
  shfl_down tree, warp aggregates added sequentially); ``rsqrtf(variance / hidden_size + eps)``.
* Triton ``_rms_norm_kernel`` (BLOCK_SIZE=1024, 2 chunks for H=1536): per chunk ``tl.sum`` of a
  1024-vector (per-thread sequential over its elements, butterfly shuffles within the warp, then over
  the warps); ``sum_sq / n_cols`` (``div.full.f32``), ``1.0 / tl.sqrt(mean + eps)``.
* ``vllm::rms_norm_kernel``: 1024 threads, thread t sums x[t]^2 and x[t+1024]^2; cub BlockReduce; rsqrtf.

    PATH=/workspace/venv-nightly/bin:$PATH python -m verity_vllm_numerics.rmsnorm_table_probe --acts ACTS --tables rsqrt_tables.npz --out OUT.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch

from .beyond_gemm import EPS, HIDDEN, bf16_to_f32, f32_to_bf16, from_words, load_norm_weights, words
from .rsqrt_probe import rsqrt_from_table

H = HIDDEN


# --- reduction trees (all FP32, numpy RNE) ------------------------------------------------------------


def tree(P: np.ndarray, kind: str) -> np.ndarray:
    """Reduce the last axis of P [M, n] (n a power of two for the trees)."""
    P = P.astype(np.float32, copy=True)
    if kind == "seq":
        s = np.zeros(P.shape[0], np.float32)
        for j in range(P.shape[1]):
            s = (s + P[:, j]).astype(np.float32)
        return s
    if kind == "seq_from_first":  # cub ApplyWarpAggregates: agg = w0; agg += w1; ...
        s = P[:, 0].copy()
        for j in range(1, P.shape[1]):
            s = (s + P[:, j]).astype(np.float32)
        return s
    while P.shape[1] > 1:
        if kind == "adjacent":  # shfl_down 1,2,4,..: (0+1), (2+3), ... then pairs of pairs
            P = (P[:, 0::2] + P[:, 1::2]).astype(np.float32)
        elif kind == "bfly":  # shfl_bfly/xor 16,8,..: (i + i^16) first
            h = P.shape[1] // 2
            P = (P[:, :h] + P[:, h:]).astype(np.float32)
        else:
            raise ValueError(kind)
    return P[:, 0]


def warps_then_block(P: np.ndarray, warp_kind: str, block_kind: str) -> np.ndarray:
    """P [M, n_threads] thread partials -> per-warp reduce (32 lanes) -> across warps."""
    M, n = P.shape
    assert n % 32 == 0
    W = np.stack([tree(P[:, 32 * w : 32 * (w + 1)], warp_kind) for w in range(n // 32)], axis=1)
    if W.shape[1] == 1:
        return W[:, 0]
    if block_kind in ("adjacent", "bfly") and (W.shape[1] & (W.shape[1] - 1)):
        # pad to a power of two with zeros (inactive warps contribute 0)
        pad = 1 << (W.shape[1] - 1).bit_length()
        W = np.concatenate([W, np.zeros((M, pad - W.shape[1]), np.float32)], axis=1)
    return tree(W, block_kind)


# --- per-kernel thread partials -------------------------------------------------------------------------


def partials_fused(x: np.ndarray, inner: str) -> np.ndarray:
    """fused_add_rms_norm<bf16, 8>: 192 threads, thread t owns elements 8t..8t+7."""
    v = x.reshape(x.shape[0], H // 8, 8)
    sq = (v * v).astype(np.float32)  # exact for bf16 operands
    if inner == "pairs_seq":  # result += z.x*z.x + z.y*z.y
        pair = (sq[:, :, 0::2] + sq[:, :, 1::2]).astype(np.float32)
        return tree(pair.reshape(-1, 4), "seq").reshape(x.shape[0], H // 8)
    if inner == "elem_seq":  # fma chain over the 8 elements
        return tree(sq.reshape(-1, 8), "seq").reshape(x.shape[0], H // 8)
    if inner == "pairs_tree":
        pair = (sq[:, :, 0::2] + sq[:, :, 1::2]).astype(np.float32)
        return tree(pair.reshape(-1, 4), "adjacent").reshape(x.shape[0], H // 8)
    raise ValueError(inner)


def partials_rms1024(x: np.ndarray) -> np.ndarray:
    """rms_norm_kernel: 1024 threads, thread t: (x[t]^2) then + x[t+1024]^2 for t < 512."""
    sq = (x * x).astype(np.float32)
    P = sq[:, :1024].copy()
    P[:, :512] = (P[:, :512] + sq[:, 1024:1536]).astype(np.float32)
    return P


def triton_chunk_sum(chunk: np.ndarray, size_per_thread: int, warp_kind: str, block_kind: str, num_warps: int = 4) -> np.ndarray:
    """tl.sum over a 1024-vector [M, 1024] with a blocked layout: 128 threads, ``size_per_thread``
    contiguous elements per thread per repeat, repeats strided by 128*size_per_thread."""
    M = chunk.shape[0]
    n_threads = 32 * num_warps
    per_rep = n_threads * size_per_thread
    reps = 1024 // per_rep
    # element index owned by thread t, repeat r, slot j: r*per_rep + t*size_per_thread + j
    idx = np.arange(1024).reshape(reps, n_threads, size_per_thread)
    owned = chunk[:, idx]  # [M, reps, threads, spt]
    owned = np.transpose(owned, (0, 2, 1, 3)).reshape(M, n_threads, reps * size_per_thread)
    thread_partial = tree(owned.reshape(-1, reps * size_per_thread), "seq").reshape(M, n_threads)
    return warps_then_block(thread_partial, warp_kind, block_kind)


def triton_sum_sq(x: np.ndarray, size_per_thread: int, warp_kind: str, block_kind: str) -> np.ndarray:
    sq = (x * x).astype(np.float32)
    c0 = sq[:, :1024]
    c1 = np.concatenate([sq[:, 1024:1536], np.zeros((x.shape[0], 512), np.float32)], axis=1)
    s = np.zeros(x.shape[0], np.float32)
    for c in (c0, c1):
        s = (s + triton_chunk_sum(c, size_per_thread, warp_kind, block_kind)).astype(np.float32)
    return s


# --- scalar tail ------------------------------------------------------------------------------------------


def rsqrt_from_table_generic(table: np.ndarray, x: np.ndarray, exact_fn) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int64) - 127
    sig = (bits & 0x7FFFFF).astype(np.int64)
    parity = exp & 1
    delta = table[parity * (1 << 23) + sig].astype(np.int64)
    exact = exact_fn(x)
    return (exact.view(np.int32).astype(np.int64) + delta).astype(np.int32).view(np.float32)


def scalar_tail(var: np.ndarray, mean_kind: str, rsqrt_kind: str, tables: dict) -> np.ndarray:
    if mean_kind == "div_ieee":
        mean = (var / np.float32(H)).astype(np.float32)
    elif mean_kind == "mul_recip":
        mean = (var * np.float32(1.0 / H)).astype(np.float32)
    elif mean_kind == "div_full":
        mean = rsqrt_from_table_generic(tables["div_full_by_1536"], var, lambda v: (v.astype(np.float64) / H).astype(np.float32))
    else:
        raise ValueError(mean_kind)
    arg = (mean + np.float32(EPS)).astype(np.float32)
    if rsqrt_kind == "ieee":
        return (1.0 / np.sqrt(arg.astype(np.float64))).astype(np.float32)
    return rsqrt_from_table(tables[rsqrt_kind], arg)


def out_model(x: np.ndarray, w: np.ndarray, s: np.ndarray, kind: str) -> np.ndarray:
    if kind == "single":  # bf16((x*s)*w), FP32 throughout
        y = (x * s[:, None]).astype(np.float32)
        return f32_to_bf16((y * w[None, :]).astype(np.float32))
    if kind == "double":  # bf16(bf16(x*s) * w)  (_f16Vec *= scale, then *= weight)
        y = bf16_to_f32(f32_to_bf16((x * s[:, None]).astype(np.float32)))
        return f32_to_bf16((y * w[None, :]).astype(np.float32))
    raise ValueError(kind)


# --- main -------------------------------------------------------------------------------------------------


def characterise_div_full(divisor: float) -> np.ndarray:
    """Triton's fp32 `/` (div.full.f32) by a constant, as an int8 ulp-offset table over two binades."""
    import triton
    import triton.language as tl

    @triton.jit
    def _k(x_ptr, y_ptr, n, D: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m, other=1.0)
        tl.store(y_ptr + offs, v / D, mask=m)

    def fn(x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x).cuda()
        yt = torch.empty_like(xt)
        _k[(triton.cdiv(xt.numel(), 1024),)](xt, yt, xt.numel(), D=int(divisor), BLOCK=1024)
        return yt.cpu().numpy()

    parts, inv = [], {}
    for e in (0, 1):
        bits = ((e + 127) << 23) + np.arange(1 << 23, dtype=np.uint32)
        x = bits.view(np.float32)
        exact = (x.astype(np.float64) / divisor).astype(np.float32)
        d = fn(x).view(np.int32).astype(np.int64) - exact.view(np.int32).astype(np.int64)
        assert np.abs(d).max() <= 127
        parts.append(d.astype(np.int8))
    table = np.concatenate(parts)
    for e in (-20, -3, 2, 21):
        bits = ((e + 127) << 23) + np.arange(1 << 23, dtype=np.uint32)
        x = bits.view(np.float32)
        ref = rsqrt_from_table_generic(table, x, lambda v: (v.astype(np.float64) / divisor).astype(np.float32))
        inv[str(e)] = int(np.sum(fn(x) != ref))
    vals, counts = np.unique(table, return_counts=True)
    return table, {"ulp_error_histogram": {str(int(v)): int(c) for v, c in zip(vals, counts)}, "scaling_invariance_mismatches_per_binade": inv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,13,27")
    ap.add_argument("--m", type=int, default=512)
    args = ap.parse_args()
    import vllm
    import vllm._custom_ops as ops
    from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant

    t_all = time.time()
    tz = np.load(args.tables)
    tables = {"rsqrtf": tz["rsqrtf"], "triton_inv_sqrt": tz["triton_inv_sqrt"]}
    div_table, div_summary = characterise_div_full(float(H))
    tables["div_full_by_1536"] = div_table
    norm_w = load_norm_weights()
    rng = np.random.default_rng(7)

    # rows: natural post-attention-norm inputs (layers), synthetic outlier rows; the fused kernel's
    # input is (x + residual) in bf16, so give it x and a residual and model the bf16 add first
    cases = []
    for li in (int(v) for v in args.layers.split(",")):
        data = np.load(f"{args.acts}/layer{li}.npz")
        cases.append((f"layers.{li}.post_attention_layernorm", data["gate_up_proj.x"][: args.m], norm_w[f"layers.{li}.post_attention_layernorm"]))
    xs = rng.standard_normal((args.m, H)).astype(np.float32) * 0.3
    xs[:, rng.choice(H, 8, replace=False)] *= 60.0
    cases.append(("synthetic outlier rows", f32_to_bf16(xs), norm_w["layers.13.input_layernorm"]))

    # GPU outputs
    gpu = {"triton": [], "cuda_rms_norm": [], "cuda_fused_add_rms_norm": []}
    X, XF, Wt = [], [], []  # inputs for the two kinds (fused: the bf16 sum x + residual)
    for label, x_bf16, w_bf16 in cases:
        xt, wt = from_words(x_bf16), from_words(w_bf16)
        gpu["triton"].append(words(rms_norm_batch_invariant(xt, wt, EPS)))
        o = torch.empty_like(xt)
        ops.rms_norm(o, xt, wt, EPS)
        gpu["cuda_rms_norm"].append(words(o))
        resid = f32_to_bf16(rng.standard_normal(x_bf16.shape).astype(np.float32) * 0.5)
        xi, ri = from_words(x_bf16), from_words(resid)
        ops.fused_add_rms_norm(xi, ri, wt, EPS)
        torch.cuda.synchronize()
        xf = f32_to_bf16(bf16_to_f32(x_bf16) + bf16_to_f32(resid))
        assert np.array_equal(words(ri), xf), "residual-add model broke"
        gpu["cuda_fused_add_rms_norm"].append(words(xi))
        X.append(bf16_to_f32(x_bf16)); XF.append(bf16_to_f32(xf)); Wt.append(bf16_to_f32(w_bf16))

    def evaluate(kernel: str, var_fn, var_name: str, xin: list[np.ndarray]) -> list[dict]:
        out = []
        variances = [var_fn(x) for x in xin]
        for mean_kind, rsqrt_kind, out_kind in itertools.product(
            ("div_ieee", "mul_recip", "div_full"), ("ieee", "rsqrtf", "triton_inv_sqrt"), ("single", "double")
        ):
            rows_exact = elems_exact = rows = elems = 0
            first = None
            for ci, (label, _, _) in enumerate(cases):
                x, w, g = xin[ci], Wt[ci], gpu[kernel][ci]
                s = scalar_tail(variances[ci], mean_kind, rsqrt_kind, tables)
                m = out_model(x, w, s, out_kind)
                ok = np.all(m == g, axis=1)
                rows += x.shape[0]; rows_exact += int(ok.sum()); elems += g.size; elems_exact += int(np.sum(m == g))
                if first is None and not ok.all():
                    i = int(np.argmin(ok)); j = int(np.argmin(m[i] == g[i]))
                    first = {"case": label, "row": i, "col": j, "gpu": f"0x{int(g[i, j]):04x}", "model": f"0x{int(m[i, j]):04x}"}
            out.append({"variance": var_name, "mean": mean_kind, "rsqrt": rsqrt_kind, "output": out_kind, "rows_exact": rows_exact, "rows": rows, "elements_exact": elems_exact, "elements": elems, "first_mismatch": first})
        return out

    results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "vllm": vllm.__version__, "triton": __import__("triton").__version__,
               "eps": EPS, "hidden": H, "rows_per_case": args.m, "cases": [c[0] for c in cases], "div_full_by_1536": div_summary, "kernels": {}}

    # fused_add_rms_norm (192 threads x 8-vectors)
    cands = []
    for inner, wk, bk in itertools.product(("pairs_seq", "elem_seq", "pairs_tree"), ("adjacent", "bfly"), ("seq_from_first", "seq", "adjacent", "bfly")):
        cands += evaluate("cuda_fused_add_rms_norm", lambda x, inner=inner, wk=wk, bk=bk: warps_then_block(partials_fused(x, inner), wk, bk), f"fused192[{inner};warp={wk};block={bk}]", XF)
    cands += evaluate("cuda_fused_add_rms_norm", lambda x: (np.sum(x.astype(np.float64) ** 2, axis=1)).astype(np.float32), "exact_f64", XF)
    results["kernels"]["cuda_fused_add_rms_norm"] = cands
    # rms_norm_kernel (1024 threads)
    cands = []
    for wk, bk in itertools.product(("adjacent", "bfly"), ("seq_from_first", "seq", "adjacent", "bfly")):
        cands += evaluate("cuda_rms_norm", lambda x, wk=wk, bk=bk: warps_then_block(partials_rms1024(x), wk, bk), f"threads1024[warp={wk};block={bk}]", X)
    cands += evaluate("cuda_rms_norm", lambda x: (np.sum(x.astype(np.float64) ** 2, axis=1)).astype(np.float32), "exact_f64", X)
    results["kernels"]["cuda_rms_norm"] = cands
    # Triton _rms_norm_kernel
    cands = []
    for spt, wk, bk in itertools.product((1, 2, 4, 8), ("bfly", "adjacent"), ("bfly", "adjacent", "seq")):
        cands += evaluate("triton", lambda x, spt=spt, wk=wk, bk=bk: triton_sum_sq(x, spt, wk, bk), f"triton[spt={spt};warp={wk};warps={bk}]", X)
    cands += evaluate("triton", lambda x: (np.sum(x.astype(np.float64) ** 2, axis=1)).astype(np.float32), "exact_f64", X)
    results["kernels"]["triton"] = cands

    results["best"] = {}
    for k, cands in results["kernels"].items():
        best = max(cands, key=lambda c: (c["rows_exact"], c["elements_exact"]))
        exact_all = [c for c in cands if c["rows_exact"] == c["rows"]]
        results["best"][k] = {"best": best, "n_candidates": len(cands), "candidates_all_rows_exact": [f"{c['variance']} | mean={c['mean']} | rsqrt={c['rsqrt']} | out={c['output']}" for c in exact_all]}
        print(k, json.dumps(results["best"][k], indent=1), flush=True)
    results["seconds"] = time.time() - t_all
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
