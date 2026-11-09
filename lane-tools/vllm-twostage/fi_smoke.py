"""FlashInfer's FP8 block-scale GEMM on Hopper GPUs, as vLLM's FlashInferFp8DeepGEMMDynamicBlockScaledKernel runs it below 32
rows: fp8_blockscale_gemm_sm90 on a bf16 input (quantised inside, per 1x128) against an e4m3 weight with 128x128 f32 block scales.
At each of GLM-5.3's TP8 FP8 linear shapes (N, K per rank, as fi_select.py) and every M from 1 to 31, the module loads, the
kernels it compiles on first use build (TensorRT-LLM's DeepGEMM JIT: $CUDA_HOME/bin/nvcc, one cubin per configuration, cached in
$HOME/.tensorrt_llm/cache, where every later engine on the machine finds them), and the output is within TOL (0.08) relative
Frobenius error of the f32 product of the input with the dequantised weight: e4m3's rounding of the input is a few percent, and a
wrong kernel is ~1. The shapes are spread over GPUS (8) processes, one per GPU, so the compiles run side by side. Writes OUT (one
JSON object: per shape and M, the error and the call's seconds, the first call of a configuration being its compile) and exits 0
when every case passes.
    python fi_smoke.py OUT"""
import json, os, subprocess, sys, time

SHAPES = {"self_attn.fused_qkv_a_proj": (2624, 6144), "self_attn.q_b_proj": (2048, 2048), "self_attn.kv_b_proj": (3584, 512),
          "self_attn.o_proj": (6144, 2048), "self_attn.indexer.wq_b": (4096, 2048), "mlp.gate_up_proj": (3072, 6144),
          "mlp.down_proj": (6144, 1536), "mlp.shared_experts.gate_up_proj": (512, 6144),
          "mlp.shared_experts.down_proj": (6144, 256)}
TOL, B = 0.08, 128


def block_quant(w):
    """w (N, K) f32 -> the e4m3 weight, its (ceil(N/128), K/128) f32 scales, and the f32 weight they stand for."""
    import torch
    N, K = w.shape
    Np = -(-N // B) * B
    wp = torch.zeros(Np, K, device=w.device)
    wp[:N] = w
    blocks = wp.view(Np // B, B, K // B, B)
    s = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
    q = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn)
    deq = (q.float() * s[:, None, :, None]).view(Np, K)[:N]
    return q.view(Np, K)[:N].contiguous(), s.contiguous(), deq


def shard(names, out):
    import torch
    from flashinfer.gemm import fp8_blockscale_gemm_sm90
    dev = os.environ.get("SMOKE_DEVICE", "cuda")  # cpu checks this script against a stand-in flashinfer
    torch.manual_seed(0)
    cases = {}
    for name in names:
        N, K = SHAPES[name]
        q, s, deq = block_quant(torch.randn(N, K, device=dev) * 0.02)
        for M in range(1, 32):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            t = time.time()
            try:
                y = fp8_blockscale_gemm_sm90(x, q, None, s)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ref = x.float() @ deq.T
                err = ((y.float() - ref).norm() / ref.norm()).item()
                case = {"err": round(err, 5), "s": round(time.time() - t, 2), "pass": err < TOL}
            except Exception as e:
                case = {"error": f"{type(e).__name__}: {e}"[:400], "pass": False}
            cases[f"{name} M={M}"] = case
            if not case["pass"] or case.get("s", 0) > 1:
                print(os.environ.get("CUDA_VISIBLE_DEVICES"), name, M, case, flush=True)
    with open(out, "w") as f:
        json.dump(cases, f)
    return all(c["pass"] for c in cases.values())


if __name__ == "__main__":
    out = sys.argv[1]
    if len(sys.argv) > 2:  # one GPU's share: python fi_smoke.py OUT.i NAME ...
        sys.exit(0 if shard(sys.argv[2:], out) else 1)
    n, t0 = int(os.environ.get("GPUS", "8")), time.time()
    names = list(SHAPES)
    procs = [subprocess.Popen([sys.executable, __file__, f"{out}.{i}", *names[i::n]], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)})
             for i in range(n) if names[i::n]]
    rcs = [p.wait() for p in procs]
    cases = {}
    for i in range(len(procs)):
        try:
            with open(f"{out}.{i}") as f:
                cases.update(json.load(f))
        except OSError:
            pass
    missing = [f"{nm} M={m}" for nm in names for m in range(1, 32) if f"{nm} M={m}" not in cases]
    ok = not any(rcs) and not missing and all(c["pass"] for c in cases.values())
    errs = [c["err"] for c in cases.values() if "err" in c]
    res = {"gpus": n, "rcs": rcs, "tol": TOL, "cases": len(cases), "missing": missing, "max_err": max(errs, default=None),
           "compiles": sum(c.get("s", 0) > 1 for c in cases.values()), "seconds": round(time.time() - t0, 1), "pass": ok, "by_case": cases}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"fi_smoke: {'pass' if ok else 'FAIL'}: {len(cases)} cases, max error {res['max_err']}, {res['compiles']} slow first calls, "
          f"{res['seconds']} s, rcs {rcs}")
    sys.exit(0 if ok else 1)
