#!/usr/bin/env python3
"""Forced work beside the served step (CMP-8 round 2, "windows restricted to GEMMs" against filling the gaps).

Llama-3.1-8B's decoder layer as vLLM's FP8 path runs it, from vLLM's own ops: fused add-RMSNorm, per-token FP8 quant,
`cutlass_scaled_mm`, rotary, FA2 varlen attention, SiLU-and-mul. That layer is repeated over 32 layers with distinct
weights, captured as one CUDA graph and replayed on the highest-priority stream. Next to it, on the lowest-priority stream,
runs one of two kinds of forced work, in short Triton programs that stop when the sequence ends:

  pump  φ chains (phi.cuh's 1× alu step: t = y ^ w; y = rotl13(t) + t + k), 8 independent chains a lane, on registers only;
        CUDA work that a verifier could force with live coins (the "other forced work" of the brief)
  tag   reads a 1 GiB store with 16-byte loads and folds each word into a φ chain: a stand-in for the bandwidth audit's
        tag or sketch pass over its store (DRAM-bound plus light CUDA)

For each phase (prefill 2,048 tokens; decode 32 sequences at context 1,024) and each sequence (the whole layer stack, its
GEMMs only, its other kernels only), it reports the sequence's time alone and beside each filler, and the filler's work
done while the sequence ran, as a share of what the filler does alone on the whole GPU in the same time (its "soak").

  python gap_filler.py [--phases prefill2048,decode32] [--reps 7] > gap-filler.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
import triton
import triton.language as tl

H, NH, NKV, HD, FF, LAYERS = 4096, 32, 8, 128, 14336, 32
PUMP_ITERS, PUMP_BLOCK, PUMP_CHAINS = 48, 128, 8
TAG_BLOCK, TAG_WORDS = 256, 64  # a program reads 256 lanes x 64 x 16 B = 256 KiB
STORE_BYTES = 1 << 30


@triton.jit
def _rotl13(t):
    return (t << 13) | (t >> 19)


@triton.jit
def pump_kernel(out_ptr, go_ptr, stop_ptr, done_ptr, seed, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if tl.load(stop_ptr, volatile=True) == 0:
        _pump_body(out_ptr, go_ptr, stop_ptr, done_ptr, pid, seed, ITERS, BLOCK)


@triton.jit
def _pump_body(out_ptr, go_ptr, stop_ptr, done_ptr, pid, seed, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    lane = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.uint32)
    k = lane * 0x9E3779B9 + seed
    y0 = lane ^ 0x01
    y1 = lane ^ 0x02
    y2 = lane ^ 0x03
    y3 = lane ^ 0x04
    y4 = lane ^ 0x05
    y5 = lane ^ 0x06
    y6 = lane ^ 0x07
    y7 = lane ^ 0x08
    for i in range(ITERS):
        w = i.to(tl.uint32) * 0x7F4A7C15 + seed
        t = y0 ^ w
        y0 = _rotl13(t) + t + k
        t = y1 ^ w
        y1 = _rotl13(t) + t + k
        t = y2 ^ w
        y2 = _rotl13(t) + t + k
        t = y3 ^ w
        y3 = _rotl13(t) + t + k
        t = y4 ^ w
        y4 = _rotl13(t) + t + k
        t = y5 ^ w
        y5 = _rotl13(t) + t + k
        t = y6 ^ w
        y6 = _rotl13(t) + t + k
        t = y7 ^ w
        y7 = _rotl13(t) + t + k
    f = y0 ^ _rotl13(y1) ^ y2 ^ _rotl13(y3) ^ y4 ^ _rotl13(y5) ^ y6 ^ _rotl13(y7)
    tl.store(out_ptr + (pid % 65536) * BLOCK + tl.arange(0, BLOCK), f)
    if (tl.load(go_ptr, volatile=True) != 0) & (tl.load(stop_ptr, volatile=True) == 0):
        tl.atomic_add(done_ptr, 1)


@triton.jit
def tag_kernel(store_ptr, out_ptr, go_ptr, stop_ptr, done_ptr, n_chunks, seed, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if tl.load(stop_ptr, volatile=True) == 0:
        _tag_body(store_ptr, out_ptr, go_ptr, stop_ptr, done_ptr, n_chunks, pid, seed, WORDS, BLOCK)


@triton.jit
def _tag_body(store_ptr, out_ptr, go_ptr, stop_ptr, done_ptr, n_chunks, pid, seed, WORDS: tl.constexpr,
              BLOCK: tl.constexpr):
    base = (pid % n_chunks).to(tl.int64) * (WORDS * BLOCK * 4)
    lane = tl.arange(0, BLOCK)
    k = (pid * BLOCK + lane).to(tl.uint32) * 0x9E3779B9 + seed
    y = k
    for i in range(WORDS):
        offs = base + (i * 4 * BLOCK + lane).to(tl.int64)
        a = tl.load(store_ptr + offs)
        b = tl.load(store_ptr + offs + BLOCK)
        c = tl.load(store_ptr + offs + 2 * BLOCK)
        d = tl.load(store_ptr + offs + 3 * BLOCK)
        t = y ^ a
        y = _rotl13(t) + t + k
        t = y ^ b
        y = _rotl13(t) + t + k
        t = y ^ c
        y = _rotl13(t) + t + k
        t = y ^ d
        y = _rotl13(t) + t + k
    tl.store(out_ptr + (pid % 65536) * BLOCK + lane, y)
    if (tl.load(go_ptr, volatile=True) != 0) & (tl.load(stop_ptr, volatile=True) == 0):
        tl.atomic_add(done_ptr, 1)


def e4m3(shape, gen):
    x = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8, generator=gen)
    x = (x & 0x87) | 0x20  # exponent field 4..7: no NaN, moderate magnitudes
    return x.view(torch.float8_e4m3fn)


class Layers:
    """32 layers' weights and one step's activations for `tokens` query tokens over `ctx` keys per sequence."""

    def __init__(self, phase: str, gen):
        from vllm import _custom_ops as ops

        self.ops = ops
        self.phase = phase
        if phase.startswith("prefill"):
            self.T, self.seqs, self.ctx = int(phase[len("prefill"):]), 1, int(phase[len("prefill"):])
        else:
            self.seqs, self.ctx = int(phase[len("decode"):]), 1024
            self.T = self.seqs
        T = self.T
        self.w = []
        for _ in range(LAYERS):
            self.w.append(
                {
                    "qkv": e4m3((6144, H), gen).t(),
                    "o": e4m3((H, H), gen).t(),
                    "gu": e4m3((2 * FF, H), gen).t(),
                    "down": e4m3((H, FF), gen).t(),
                    "s_qkv": torch.full((6144, 1), 1e-2, device="cuda"),
                    "s_o": torch.full((H, 1), 1e-2, device="cuda"),
                    "s_gu": torch.full((2 * FF, 1), 1e-2, device="cuda"),
                    "s_down": torch.full((H, 1), 1e-2, device="cuda"),
                    "n1": torch.ones(H, device="cuda", dtype=torch.bfloat16),
                    "n2": torch.ones(H, device="cuda", dtype=torch.bfloat16),
                }
            )
        self.h = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        self.res = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        self.act = torch.empty(T, FF, device="cuda", dtype=torch.bfloat16)
        self.pos = torch.arange(T, device="cuda") if self.seqs == 1 else torch.full((T,), self.ctx - 1, device="cuda")
        inv = 1.0 / (500000.0 ** (torch.arange(0, HD, 2, device="cuda", dtype=torch.float32) / HD))
        f = torch.outer(torch.arange(max(self.ctx, T) + 1, device="cuda", dtype=torch.float32), inv)
        self.cos_sin = torch.cat([f.cos(), f.sin()], dim=-1).to(torch.bfloat16)
        if self.seqs == 1:
            self.cu_q = torch.tensor([0, T], device="cuda", dtype=torch.int32)
            self.cu_k = self.cu_q
            self.kc = self.vc = None
        else:
            self.cu_q = torch.arange(0, self.seqs + 1, device="cuda", dtype=torch.int32)
            self.cu_k = torch.arange(0, self.seqs + 1, device="cuda", dtype=torch.int32) * self.ctx
            self.kc = torch.randn(self.seqs * self.ctx, NKV, HD, device="cuda", dtype=torch.bfloat16, generator=gen)
            self.vc = torch.randn(self.seqs * self.ctx, NKV, HD, device="cuda", dtype=torch.bfloat16, generator=gen)

    def layer(self, l: int, gemm: bool, other: bool):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        ops, w, T = self.ops, self.w[l], self.T
        if other:
            ops.fused_add_rms_norm(self.h, self.res, w["n1"], 1e-5)
        q8, s8 = ops.scaled_fp8_quant(self.h, use_per_token_if_dynamic=True) if other else (self.q8_h, self.s8_h)
        qkv = ops.cutlass_scaled_mm(q8, w["qkv"], s8, w["s_qkv"], torch.bfloat16) if gemm else self.qkv
        q, k, v = qkv.split([NH * HD, NKV * HD, NKV * HD], dim=-1)
        if other:
            ops.rotary_embedding(self.pos, q, k, HD, self.cos_sin, True)
            if self.seqs == 1:
                attn = flash_attn_varlen_func(
                    q.reshape(T, NH, HD), k.reshape(T, NKV, HD), v.reshape(T, NKV, HD), T, self.cu_q, T, self.cu_k,
                    causal=True, fa_version=2)
            else:
                attn = flash_attn_varlen_func(
                    q.reshape(T, NH, HD), self.kc, self.vc, 1, self.cu_q, self.ctx, self.cu_k, causal=True, fa_version=2)
            a8, sa8 = ops.scaled_fp8_quant(attn.reshape(T, H), use_per_token_if_dynamic=True)
        else:
            a8, sa8 = self.q8_h, self.s8_h
        o = ops.cutlass_scaled_mm(a8, w["o"], sa8, w["s_o"], torch.bfloat16) if gemm else self.h
        if other:
            ops.fused_add_rms_norm(o, self.res, w["n2"], 1e-5)
            q8, s8 = ops.scaled_fp8_quant(o, use_per_token_if_dynamic=True)
        gu = ops.cutlass_scaled_mm(q8, w["gu"], s8, w["s_gu"], torch.bfloat16) if gemm else self.gu
        if other:
            torch.ops._C.silu_and_mul(self.act, gu)
            q8, s8 = ops.scaled_fp8_quant(self.act, use_per_token_if_dynamic=True)
        else:
            q8, s8 = self.q8_ff, self.s8_ff
        if gemm:
            self.h_out = ops.cutlass_scaled_mm(q8, w["down"], s8, w["s_down"], torch.bfloat16)

    def prepare(self):
        ops, T = self.ops, self.T
        self.q8_h, self.s8_h = ops.scaled_fp8_quant(self.h, use_per_token_if_dynamic=True)
        self.q8_ff, self.s8_ff = ops.scaled_fp8_quant(torch.randn(T, FF, device="cuda", dtype=torch.bfloat16),
                                                      use_per_token_if_dynamic=True)
        self.qkv = torch.randn(T, 6144, device="cuda", dtype=torch.bfloat16)
        self.gu = torch.randn(T, 2 * FF, device="cuda", dtype=torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="prefill2048,decode32")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(7)
    least, greatest = torch.cuda.Stream.priority_range()
    hi = torch.cuda.Stream(priority=greatest)
    lo = torch.cuda.Stream(priority=least)
    flags = torch.zeros(4, device="cuda", dtype=torch.int32)  # go, stop, done
    go, stop, done = flags[0:1], flags[1:2], flags[2:3]
    sink = torch.empty(65536 * 256, device="cuda", dtype=torch.uint32)
    store = torch.randint(0, 2**31, (STORE_BYTES // 4,), device="cuda", dtype=torch.int32, generator=gen).view(torch.uint32)
    n_chunks = STORE_BYTES // (TAG_WORDS * TAG_BLOCK * 16)
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def launch_filler(kind: str, programs: int):
        with torch.cuda.stream(lo):
            if kind == "pump":
                pump_kernel[(programs,)](sink, go, stop, done, 12345, ITERS=PUMP_ITERS, BLOCK=PUMP_BLOCK)
            else:
                tag_kernel[(programs,)](store, sink, go, stop, done, n_chunks, 12345, WORDS=TAG_WORDS, BLOCK=TAG_BLOCK)

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    rate_cache: dict[str, float] = {}

    def grid_for(kind: str, ms: float) -> int:
        # Enough programs to outlast a sequence slowed 30x; the ones dispatched after stop exit at once.
        return min(2**31 - 1, int(rate_cache[kind] * ms * 30) + 4 * sms * 16)

    def filler_alone(kind: str, ms_target: float) -> float:
        # Rate on the whole GPU with nothing beside it: a grid's programs over the grid's own time, go already set,
        # the grid grown until it lasts at least ms_target.
        def timed(n: int) -> float:
            flags.zero_()
            go.fill_(1)
            torch.cuda.synchronize()
            e0.record(lo)
            launch_filler(kind, n)
            e1.record(lo)
            torch.cuda.synchronize()
            assert done.item() == n, (kind, n, done.item())
            return e0.elapsed_time(e1)

        if kind not in rate_cache:
            launch_filler(kind, 64)  # compile
            torch.cuda.synchronize()
        n = 4 * sms * 16
        target = max(ms_target, 5.0)
        while (ms := timed(n)) < target and n < 2**30:
            n = min(2**30, int(n * max(2.0, 1.2 * target / max(ms, 1e-3))))
        rates = [n / timed(n) for _ in range(3)]
        rate_cache[kind] = statistics.median(rates)
        return rate_cache[kind]

    out = {"schema": "verity-pouw/saturation-gap-filler/v1", "device": torch.cuda.get_device_name(0), "sms": sms,
           "priority_range": [least, greatest], "pump": {"iters": PUMP_ITERS, "block": PUMP_BLOCK, "chains": PUMP_CHAINS},
           "tag": {"block": TAG_BLOCK, "words": TAG_WORDS, "bytes_per_program": TAG_WORDS * TAG_BLOCK * 16}, "phases": []}
    for phase in [p for p in a.phases.split(",") if p]:
        L = Layers(phase, gen)
        L.prepare()
        torch.cuda.synchronize()
        res = {"phase": phase, "tokens": L.T, "ctx": L.ctx, "seqs": []}
        for seq, (gemm, other) in {"all": (True, True), "gemm": (True, False), "other": (False, True)}.items():
            def body():
                go.fill_(1)
                for l in range(LAYERS):
                    L.layer(l, gemm, other)
                stop.fill_(1)

            with torch.cuda.stream(hi):
                for _ in range(2):
                    body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=hi):
                body()
            torch.cuda.synchronize()

            def replay(kind: str | None) -> tuple[float, int]:
                flags.zero_()
                torch.cuda.synchronize()
                if kind:
                    launch_filler(kind, grid_for(kind, row["ms_alone"]))
                with torch.cuda.stream(hi):
                    e0.record(hi)
                    g.replay()
                    e1.record(hi)
                torch.cuda.synchronize()
                return e0.elapsed_time(e1), done.item()

            replay(None)
            row = {"seq": seq}
            alone = [replay(None)[0] for _ in range(a.reps)]
            row["ms_alone"] = statistics.median(alone)
            for kind in ("pump", "tag"):
                rate = filler_alone(kind, row["ms_alone"])
                runs = [replay(kind) for _ in range(a.reps)]
                ms = statistics.median(r[0] for r in runs)
                progs = statistics.median(r[1] for r in runs)
                row[kind] = {
                    "ms_with": ms,
                    "slowdown": ms / row["ms_alone"],
                    "grid": grid_for(kind, row["ms_alone"]),
                    "programs": progs,
                    "alone_programs_per_ms": rate,
                    "soak": progs / (rate * ms) if rate > 0 else None,
                }
                if kind == "tag":
                    row[kind]["store_gb_per_s"] = progs * TAG_WORDS * TAG_BLOCK * 16 / (ms * 1e6)
                    row[kind]["alone_gb_per_s"] = rate * TAG_WORDS * TAG_BLOCK * 16 / 1e6
            print(f"{phase} {seq}: {json.dumps(row)}", file=sys.stderr)
            res["seqs"].append(row)
            del g
        out["phases"].append(res)
        del L
        torch.cuda.empty_cache()
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
