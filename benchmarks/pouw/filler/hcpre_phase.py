"""HC-PREFILL's phase probe (cmp-13, approach pouw/filler-prefill): the prefill step's parts that the filler scorer doesn't
measure, run with vLLM's own kernels, alone and beside the DRAM-free filler.

The scorer's prefill point times only the 32 layers of int8 linears; FA2 attention (73.11 ms for causal 8,192) and the
non-GEMM kernels (38.37 ms: norm, activation, quantization, rotary, KV write) are constants it stretches by the linears'
slowdown and credits at the linears' filler rate. This probe builds those parts for Llama-3.1-8B at one 8,192-token
prompt from vLLM's kernels on random BF16 activations (no weights, no linears): per layer fused add + RMSNorm, per-token
FP8 quantization of the qkv input, rotary on q and k, the KV write into a paged cache, FA2 over the paged cache (the
`flash_fwd_splitkv_kernel` vLLM ran in note:20261015T0417Z-draft-saturation-budget), FP8 quantization of the o input, the
second fused add + RMSNorm and its quantization, SiLU-and-mul on the 28,672-wide gate_up output, and the down input's
quantization; captured as one CUDA graph per part (`att`: KV write + FA2; `ng`: the rest; `both`: the layer order).

Each part is then timed as `dscore` times a point: ROUNDS rounds in ABBA order of K reps of the part alone and K reps with
dcap.cu's filler (libhcpre_fill.so, the scorer's `co`: chunks queued on the lowest-priority stream, the part on the
highest), each arm block followed by a standalone block of the filler alone as long; every rep's SM clock and board power
read; each round's tiles numbered from a public range fixed before its coins (os.urandom), receipts checked on the host.
Rows are dscore's JSON shape (kind "phase-<part>"), so the locked `scoring.point_stats` reads them.

  python hcpre_phase.py --lib libhcpre_fill.so --out DIR [--parts both,att,ng] [--rounds 4] [--reps 3]
        [--fill W,NC,S,LOG2,MAP ...] [--probe]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

import torch

T, H, NQ, NKV, HD, INTER, LAYERS = 8192, 4096, 32, 8, 128, 14336, 32
BLOCK = 16
POINT_SLOTS = 256
RANGE_LOG2 = 32


def build(dev):
    from vllm import _custom_ops as ops

    bf = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(7)

    def rnd(*shape, scale=0.5):
        return (torch.randn(*shape, device=dev, generator=g, dtype=torch.float32) * scale).to(bf)

    x = rnd(T, H)
    res = rnd(T, H)
    w_ln = [rnd(H, scale=0.1) + 1 for _ in range(2)]
    q = rnd(T, NQ * HD)
    k = rnd(T, NKV * HD)
    v = rnd(T, NKV * HD)
    att_out = rnd(T, NQ * HD)
    gu = rnd(T, 2 * INTER)
    act = torch.empty(T, INTER, device=dev, dtype=bf)
    pos = torch.arange(T, device=dev, dtype=torch.int64)
    inv = 1.0 / (500000.0 ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
    fr = torch.outer(torch.arange(131072, device=dev, dtype=torch.float32), inv)
    cos_sin = torch.cat([fr.cos(), fr.sin()], dim=-1).to(bf)
    nblk = T // BLOCK
    kc = [torch.zeros(nblk, BLOCK, NKV, HD, device=dev, dtype=bf) for _ in range(LAYERS)]
    vc = [torch.zeros(nblk, BLOCK, NKV, HD, device=dev, dtype=bf) for _ in range(LAYERS)]
    slots = torch.arange(T, device=dev, dtype=torch.int64)
    block_table = torch.arange(nblk, device=dev, dtype=torch.int32).view(1, nblk)
    cu_q = torch.tensor([0, T], device=dev, dtype=torch.int32)
    seqused = torch.tensor([T], device=dev, dtype=torch.int32)
    one = torch.ones(1, device=dev, dtype=torch.float32)
    qv = q.view(T, NQ, HD)
    av = att_out.view(T, NQ, HD)

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def fa(layer):
        flash_attn_varlen_func(q=qv, k=kc[layer], v=vc[layer], max_seqlen_q=T, cu_seqlens_q=cu_q, max_seqlen_k=T,
                               seqused_k=seqused, softmax_scale=HD ** -0.5, causal=True, block_table=block_table,
                               out=av, fa_version=2)

    def quant(t):
        return ops.scaled_fp8_quant(t, None, use_per_token_if_dynamic=True)

    def norm(i):
        ops.fused_add_rms_norm(x, res, w_ln[i], 1e-5)

    def rope():
        ops.rotary_embedding(pos, q, k, HD, cos_sin, True)

    def kvw(layer):
        ops.reshape_and_cache_flash(k.view(T, NKV, HD), v.view(T, NKV, HD), kc[layer], vc[layer], slots, "auto", one, one)

    def silu():
        torch.ops._C.silu_and_mul(act, gu)

    def layer_att(layer):
        kvw(layer)
        fa(layer)

    def layer_ng(layer):
        norm(0)
        quant(x)
        rope()
        quant(att_out)
        norm(1)
        quant(x)
        silu()
        quant(act)

    def layer_both(layer):
        norm(0)
        quant(x)
        rope()
        kvw(layer)
        fa(layer)
        quant(att_out)
        norm(1)
        quant(x)
        silu()
        quant(act)

    parts = {"att": layer_att, "ng": layer_ng, "both": layer_both}
    cats = {"norm": lambda l: (norm(0), norm(1)), "quant": lambda l: (quant(x), quant(att_out), quant(x), quant(act)),
            "rope": lambda l: rope(), "kv_write": kvw, "attention": fa, "act": lambda l: silu()}
    return parts, cats


def graph_of(fn, stream):
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        for _ in range(2):
            for layer in range(LAYERS):
                fn(layer)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=stream):
        for layer in range(LAYERS):
            fn(layer)
    torch.cuda.synchronize()
    return gr


class Lib:
    def __init__(self, path):
        L = ctypes.CDLL(path)
        L.hf_init.argtypes = [ctypes.c_int] * 6 + [ctypes.POINTER(ctypes.c_int)]
        L.hf_chunk_macs.restype = ctypes.c_double
        L.hf_per.restype = ctypes.c_longlong
        L.hf_now_ms.restype = ctypes.c_double
        L.hf_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_ulonglong]
        L.hf_done.restype = ctypes.c_longlong
        L.hf_stop.restype = ctypes.c_longlong
        L.hf_count.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        L.hf_count.restype = ctypes.c_longlong
        L.hf_check.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong)]
        L.hf_seed.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_ulonglong]
        L.hf_seed.restype = ctypes.c_uint64
        L.hf_mhz.restype = ctypes.c_double
        L.hf_watt.restype = ctypes.c_double
        self.L = L


def timed(lib, gr, stream, reps):
    """K reps of the graph on `stream`: each rep's (ms, SM MHz, W), NVML polled while it runs (dcap's time_work)."""
    out = []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        e0.record(stream)
        with torch.cuda.stream(stream):
            gr.replay()
        e1.record(stream)
        cs = ps = 0.0
        nc = np_ = 0
        while not e1.query():
            c, p = lib.L.hf_mhz(), lib.L.hf_watt()
            if c > 0:
                cs, nc = cs + c, nc + 1
            if p > 0:
                ps, np_ = ps + p, np_ + 1
        e1.synchronize()
        if not nc:
            cs, nc = lib.L.hf_mhz(), 1
        if not np_:
            ps, np_ = lib.L.hf_watt(), 1
        out.append([e0.elapsed_time(e1), cs / nc, ps / np_])
    return out


def round_open(point, slot):
    rid = point * POINT_SLOTS + slot
    c0, c1 = (int.from_bytes(os.urandom(8), "little") for _ in range(2))
    return {"id": rid, "slot": slot, "range": [rid << RANGE_LOG2, (rid + 1) << RANGE_LOG2], "c0": c0, "c1": c1}


def round_close(lib, rd, n=4):
    launched = lib.L.hf_stop()
    tiles = (ctypes.c_ulonglong * n)()
    bad = lib.L.hf_check(n, tiles)
    lo = rd["range"][0]
    end = lo + launched * lib.L.hf_per()
    rec = {"id": rd["id"], "slot": rd["slot"], "range": rd["range"], "tiles": [lo, end],
           "coins": [f"{rd['c0']:016x}", f"{rd['c1']:016x}"], "seed0": f"{lib.L.hf_seed(rd['c0'], rd['c1'], lo):016x}",
           "checked": list(tiles)}
    if end > rd["range"][1]:
        raise SystemExit(f"round {rd['id']} ran past its range")
    return rec, bad


def pump_until(lib, n):
    while lib.L.hf_done() < n:
        time.sleep(0.0002)


def measure(lib, part, gr, stream, point, rounds, reps, fill):
    W, NC, S, LOG2, MAP = fill
    bad = checked = 0
    rd0 = round_open(point, 0)
    lib.L.hf_start(rd0["c0"], rd0["c1"], rd0["range"][0])
    pump_until(lib, 2)
    ta = lib.L.hf_now_ms()
    pump_until(lib, 2 + 60)
    tb = lib.L.hf_now_ms()
    sp = ctypes.c_double()
    n = lib.L.hf_count(ta, tb, ctypes.byref(sp))
    cal = (n - 1) * lib.L.hf_chunk_macs() / (sp.value * 1e-3)
    r0, b = round_close(lib, rd0)
    bad, checked = bad + b, checked + 4
    blocks = []
    for r in range(rounds):
        blk = {"order": "alone,arm,solo" if r % 2 == 0 else "arm,solo,alone"}
        if r % 2 == 0:
            blk["alone"] = {"window_ms": 0, "reps": timed(lib, gr, stream, reps)}
        rd = round_open(point, 1 + r)
        lib.L.hf_start(rd["c0"], rd["c1"], rd["range"][0])
        pump_until(lib, 2)
        t0 = lib.L.hf_now_ms()
        arm = timed(lib, gr, stream, reps)
        t1 = lib.L.hf_now_ms()
        n = lib.L.hf_count(t0, t1, ctypes.byref(sp))
        rec, b = round_close(lib, rd)
        bad, checked = bad + b, checked + 4
        blk["arm"] = {"window_ms": t1 - t0, "chunks": n, "span_ms": sp.value, "round": rec, "reps": arm}
        rd = round_open(point, 1 + rounds + r)
        lib.L.hf_start(rd["c0"], rd["c1"], rd["range"][0])
        pump_until(lib, 2)
        s0 = lib.L.hf_now_ms()
        while lib.L.hf_now_ms() < s0 + (t1 - t0) or lib.L.hf_done() < 2 + 16:
            time.sleep(0.0002)
        s1 = lib.L.hf_now_ms()
        n = lib.L.hf_count(s0, s1, ctypes.byref(sp))
        rec, b = round_close(lib, rd)
        bad, checked = bad + b, checked + 4
        blk["solo"] = {"window_ms": s1 - s0, "chunks": n, "span_ms": sp.value, "round": rec, "reps": []}
        if r % 2 == 1:
            blk["alone"] = {"window_ms": 0, "reps": timed(lib, gr, stream, reps)}
        blocks.append(blk)
    return {"mode": "dscore", "kind": f"phase-{part}", "B": T, "L": 0, "place": "co", "sms": 188, "fill_sms": 188,
            "W": W, "NC": NC, "S": S, "chunk_log2": LOG2, "map": MAP, "chunk_macs": lib.L.hf_chunk_macs(),
            "point": point, "slots": POINT_SLOTS, "rounds": rounds, "K": reps, "kv_wrapped": False, "setup_ms": 0.0,
            "spread": {}, "alone_fill": {"mac_per_s": cal, "round": r0}, "rec_checked": checked, "rec_bad": bad,
            "blocks": blocks}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="both,att,ng")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fill", action="append", default=[], help="W,NC,S,LOG2,MAP (repeatable; one filler a process)")
    ap.add_argument("--point0", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="time each kernel category alone and stop")
    a = ap.parse_args(argv)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    lib = Lib(a.lib)
    if lib.L.hf_nvml_init():
        raise SystemExit("NVML init failed")
    fills = [tuple(int(x) for x in f.split(",")) for f in a.fill] or [(4, 4, 512, 17, 1)]
    if len(fills) > 1:
        raise SystemExit("one filler shape a process (hf_init allocates once)")
    W, NC, S, LOG2, MAP = fills[0]
    prio = (ctypes.c_int * 2)()
    lib.L.hf_init(188, W, NC, S, LOG2, MAP, prio)
    stream = torch.cuda.Stream(device=dev, priority=prio[1])
    parts, cats = build(dev)
    out = open(os.path.join(a.out, "phase.jsonl"), "a")
    info = {"mode": "info", "prio": [prio[0], prio[1]], "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception as ex:  # noqa: BLE001
        info["vllm"] = repr(ex)
    print(json.dumps(info), file=out, flush=True)
    t_cat = {}
    for name, fn in cats.items():
        gr = graph_of(fn, stream)
        t_cat[name] = statistics.median(r[0] for r in timed(lib, gr, stream, 5))
        del gr
    print(json.dumps({"mode": "categories", "ms": t_cat, "sum": sum(t_cat.values())}), file=out, flush=True)
    if a.probe:
        return 0
    point = a.point0
    for part in a.parts.split(","):
        gr = graph_of(parts[part], stream)
        row = measure(lib, part, gr, stream, point, a.rounds, a.reps, fills[0])
        print(json.dumps(row), file=out, flush=True)
        point += 1
        del gr
    return 0


if __name__ == "__main__":
    sys.exit(main())
