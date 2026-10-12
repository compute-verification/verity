"""Overhead of each PoUW scheme's honest GEMM against the plain GEMM of its datatype, on one RTX 4090.

    python benchmarks/pouw/gemm_bench.py --out RESULT.json      (on the pod: torch with CUDA, verity and verity.core.protocols.pouw importable)

Gates, before any timing (the run refuses to time a kernel that fails its gate):
- ``ncp``: the s8 GEMM's every 16-deep checkpoint, dumped at a small lead-domain shape and shifted to route U
  (`ncp.route_u_shift`), gives the tile digests of `verity.core.protocols.pouw.schemes.ncp` (the reference), and its output is exactly
  A·Bᵀ;
- ``pearl``: the e4m3 GEMM on noisy-quantized operands from `verity.core.protocols.pouw.schemes.pearl_kw` equals the reference replay on
  the ``ada`` device, bit for bit.

Timings, per workload shape (m, k, n), median of CUDA-event timed repetitions:
- plain: the int8 GEMM (cuBLASLt ``torch._int_mm`` where it applies, and this kernel without reads), the e4m3 GEMM
  (cuBLASLt ``torch._scaled_mm``) and BF16 (``torch.mm``);
- NCP: the (m, 3k, n) GEMM with every accumulator read once per 16-deep step, forming the three noised operands, and
  SHAKE256 over the transcript (3k/16 words per output) at a measured 650 GB/s (an estimate, not run here);
- Pearl: the windowed e4m3 GEMM, noisy quantization of A (the rank-32 noise product and the elementwise BF16 steps), and
  the peel (A·F_B and the 2r correction columns, BF16).
"""

from __future__ import annotations

import argparse
import os
import json
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kernels import load  # noqa: E402

SHAKE_GBPS = 650.0
R = 32


_FLUSH: list = []


def timed(fn, reps: int = 20, warm: int = 3) -> float:
    """Median CUDA-event time of `fn`, with the 72 MB L2 flushed before every repetition, so operands stream from DRAM
    as a model's weights do in a forward pass."""
    if not _FLUSH:
        _FLUSH.append(torch.empty(256 << 20, dtype=torch.uint8, device="cuda"))
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        _FLUSH[0].zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def s8(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int8).view(torch.uint8).contiguous()


def gate_ncp(ext) -> dict:
    from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
    from verity.core.protocols.pouw.schemes import ncp

    m, k, n = 16, 1088, 256
    s, rng = ncp.NCP(n_min=64), random.Random(7)
    u = Matmul(3, Shape(m, k, n), 5)
    ctx = Context(b"\x11" * 32, u, b"\x22" * 32, b"\x33" * 32)
    a = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(m)]
    b = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(n)]
    full = s.encode(ctx, s.noise(ctx), a, b, Tile(tuple(range(m)), tuple(range(n))))
    X = torch.tensor(full.x, dtype=torch.int16, device="cuda")
    Y = torch.tensor(full.y, dtype=torch.int16, device="cuda")
    C, _, ck = ext.ncp_gemm(s8(X), s8(Y), 2, 1)
    CU, ck_u = ext.ncp_route_u((X + ncp.BETA).to(torch.uint8), s8(Y), 1)
    A = torch.tensor([[x - 256 if x >= 128 else x for x in r] for r in a], dtype=torch.int64)
    B = torch.tensor([[x - 256 if x >= 128 else x for x in r] for r in b], dtype=torch.int64)
    ok_out = torch.equal(C.cpu().long(), A @ B.T)
    tiles = [Tile(tuple(range(16)), tuple(range(j, j + 16))) for j in (0, 112, 240)]
    ck = ck.cpu()
    ck_u = ck_u.cpu()
    ok_tiles = []
    for t in tiles:
        ref = s.digest(ctx, None, s.checked(ctx, s.encode(ctx, s.noise(ctx), a, [b[j] for j in t.cols], t)))
        unbiased = [[[w & 0xFFFFFFFF for w in ck[i, j].tolist()] for j in t.cols] for i in t.rows]
        got_checked = ncp.route_u_shift(unbiased, [full.y[j] for j in t.cols], s.beta)
        native_checked = [[[w & 0xFFFFFFFF for w in ck_u[i, j].tolist()] for j in t.cols] for i in t.rows]
        ok_tiles.append((s.digest(ctx, None, got_checked) == ref, s.digest(ctx, None, native_checked) == ref))
    native_out = torch.equal(CU.cpu(), C.cpu())
    ok = ok_out and native_out and all(all(pair) for pair in ok_tiles)
    return {"shape": [m, k, n], "output_exact": ok_out, "native_output_matches": native_out,
            "tile_digests_match_shift_native": ok_tiles, "ok": ok}


def gate_pearl(ext) -> dict:
    from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
    from verity.core.protocols.pouw.schemes import pearl_kw as P

    m, k, n = 16, 1024, 128
    rng = random.Random(11)
    rows = lambda cnt: [P.fp10_row([rng.randint(-127, 127) for _ in range(k)], [0x3C00 + rng.randrange(0x200) for _ in range(k // 8)])
                        for _ in range(cnt)]
    a, b = rows(m), rows(n)
    s = P.PearlKW("ada")
    ctx = Context(b"\x44" * 32, Matmul(1, Shape(m, k, n), 2), b"\x55" * 32, b"\x66" * 32)
    ops = s.encode(ctx, s.noise(ctx), a, b, Tile(tuple(range(m)), tuple(range(n))))
    Aq = torch.tensor(ops.a.codes, dtype=torch.uint8, device="cuda")
    Bq = torch.tensor(ops.b.codes, dtype=torch.uint8, device="cuda")
    C = ext.fp8_gemm(Aq, Bq, 1).cpu()
    cells = [(i, j) for i in range(0, m, 5) for j in range(0, n, 9)]
    ref = [P.replay(P.ADA, ops.a.codes[i], ops.b.codes[j])[0] for i, j in cells]
    got = [C[i, j].item() for i, j in cells]
    same = [P.f32_bits(x) == P.f32_bits(y) for x, y in zip(got, ref)]
    return {"shape": [m, k, n], "cells": len(cells), "bit_exact": sum(same), "ok": all(same)}


def bench_shape(ext, m: int, k: int, n: int, gates: dict) -> dict:
    dev = "cuda"
    cfg = 1 if m <= 32 else 0
    pad = (m + 15) // 16 * 16 if cfg == 1 else (m + 127) // 128 * 128
    out: dict = {"shape": [m, k, n], "m_padded": pad}
    A8 = torch.randint(-64, 64, (pad, k), dtype=torch.int8, device=dev)
    B8 = torch.randint(-64, 64, (n, k), dtype=torch.int8, device=dev)
    if pad > 16:
        out["int8_cublaslt_ms"] = timed(lambda: torch._int_mm(A8, B8.T))
    out["int8_kernel_bare_ms"] = timed(lambda: ext.ncp_gemm(A8.view(torch.uint8), B8.view(torch.uint8), 0, cfg))
    Abf = torch.randn(pad, k, dtype=torch.bfloat16, device=dev)
    Bbf = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
    out["bf16_ms"] = timed(lambda: Abf @ Bbf.T)
    Af8, Bf8 = Abf.to(torch.float8_e4m3fn), Bbf.to(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    try:
        out["fp8_cublaslt_ms"] = timed(lambda: torch._scaled_mm(Af8, Bf8.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    except RuntimeError as e:
        out["fp8_cublaslt_error"] = str(e)[:200]
    if gates["ncp"]["ok"]:
        X = torch.randint(-127, 127, (pad, 3 * k), dtype=torch.int8, device=dev).view(torch.uint8)
        Y = torch.randint(-127, 127, (n, 3 * k), dtype=torch.int8, device=dev).view(torch.uint8)
        E1 = torch.randint(-31, 33, (pad, k), dtype=torch.int8, device=dev)
        perm = torch.randperm(k, device=dev)

        def form():
            e2 = E1[:, perm]
            return torch.cat([A8 + E1, A8 + e2, -(A8 + E1 + e2)], dim=1)

        out["ncp_gemm_reads_ms"] = timed(lambda: ext.ncp_gemm(X, Y, 1, cfg))
        out["ncp_gemm_bare3k_ms"] = timed(lambda: ext.ncp_gemm(X, Y, 0, cfg))
        out["ncp_form_ms"] = timed(form)
        words = m * n * (3 * k // 16)
        out["ncp_transcript_bytes"] = 4 * words
        out["ncp_shake_est_ms"] = 4 * words / (SHAKE_GBPS * 1e9) * 1e3
        base = out.get("int8_cublaslt_ms", out["int8_kernel_bare_ms"])
        out["ncp_kernel_over_int8"] = out["ncp_gemm_reads_ms"] / base
        out["ncp_total_over_int8"] = (out["ncp_gemm_reads_ms"] + out["ncp_form_ms"] + out["ncp_shake_est_ms"]) / base
    if gates["pearl"]["ok"]:
        Aq = torch.randint(0, 0x7E, (pad, k), dtype=torch.uint8, device=dev)
        Bq = torch.randint(0, 0x7E, (n, k), dtype=torch.uint8, device=dev)
        E = torch.randn(pad, R, dtype=torch.bfloat16, device=dev)
        F = torch.randn(k, R, dtype=torch.bfloat16, device=dev)
        FB = torch.randn(k, R, dtype=torch.bfloat16, device=dev)
        EB = torch.randn(n, 2 * R, dtype=torch.bfloat16, device=dev)
        alpha, beta = torch.rand(pad, 1, dtype=torch.bfloat16, device=dev), torch.rand(pad, 1, dtype=torch.bfloat16, device=dev)

        def quantize():
            noise = E @ F.T
            return torch.clamp(alpha * Abf + beta * noise, -448, 448).to(torch.float8_e4m3fn)

        def peel():
            af = Abf @ FB
            return torch.cat([af, E], dim=1) @ EB.T

        out["pearl_gemm_ms"] = timed(lambda: ext.fp8_gemm(Aq, Bq, 0 if pad >= 128 and n % 128 == 0 else 1))
        out["pearl_quantize_ms"] = timed(quantize)
        out["pearl_peel_ms"] = timed(peel)
        base = out.get("fp8_cublaslt_ms", out["bf16_ms"])
        out["pearl_kernel_over_fp8"] = out["pearl_gemm_ms"] / base
        out["pearl_total_over_fp8"] = (out["pearl_gemm_ms"] + out["pearl_quantize_ms"] + out["pearl_peel_ms"]) / base
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "gemm.json")
    ap.add_argument("--workload", type=Path, default=Path(__file__).with_name("workloads.json"))
    args = ap.parse_args()
    ext = load()
    gates = {"ncp": gate_ncp(ext), "pearl": gate_pearl(ext)}
    print(json.dumps(gates), flush=True)
    work = json.loads(args.workload.read_text())
    shapes = [(t, k, n) for t in work["decode_batches"] + work["prefill_tokens"] for k, n in work["model"]["linears"].values()]
    shapes += [tuple(s) for s in work["reference_shapes"]]
    rows = []
    for m, k, n in shapes:
        r = bench_shape(ext, m, k, n, gates)
        rows.append(r)
        print(json.dumps(r), flush=True)
    env = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    args.out.write_text(json.dumps({"schema": "verity-pouw/gemm-bench/v1", "env": env, "gates": gates, "rows": rows}, indent=1))
    import run_result
    ms = [(f"{key}@{'x'.join(map(str, r['shape']))}", r[key], "ratio") for r in rows if tuple(r["shape"]) in ((1, 2048, 2048), (2048, 2048, 2048), (4096, 4096, 4096))
          for key in ("ncp_kernel_over_int8", "ncp_total_over_int8", "pearl_kernel_over_fp8", "pearl_total_over_fp8") if key in r]
    run_result.write(args.out, gates["ncp"]["ok"] and gates["pearl"]["ok"], "gemm_bench gates",
                     f"ncp {gates['ncp']}; pearl {gates['pearl']}", ms)


if __name__ == "__main__":
    main()
