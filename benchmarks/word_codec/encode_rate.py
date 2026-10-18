"""kv.cu's encode throughput on one GPU at the stack's codec (p1024, masked, FOLD 6), against its squaring and the
stated ceiling, and what binds it.

`--lib NAME=PATH` is kv.cu built at that codec (`encode_rate.sh`). Compute-only rates first (`kv_chain`, each the best
of 2, 4 and 8 CTAs an SM, `kv_bench.chain`):

    square      squarings/s, ILP 1 and 2 words a thread
    products    FOLD 6's product with its fold replaced by the halves' XOR (`what` 3): the squaring less its reduction
    fold        the reduction mod p alone (`what` 4)
    encode      at each k of --ks, words/s, as squarings/s (k root chains of `root_squarings` a word: 98,016 a block at
                k = 3) and as blocks/s (TLW words a 4 KiB block)
    decode      likewise, k squarings a word

each over the reference (`kv.cu` before FOLD 6, 1.499e10) and the stated ceiling's GPU part, `timing.GPU_SQUARINGS_PER_S`
(1.597 × the reference) less the host's share h = 12.5% (s = 1.42, ra-squaring-ceiling's wide report). Then the encode
through memory (`kv_encode_bound` of --blocks blocks, each under an offset of its own, in place) alone and on a
low-priority stream beside a bf16 GEMM loop (--gemm M,K,N) on a high-priority one, at grids of two CTAs an SM, one, and
a word a thread (CTAs that end after one word each, so the GEMM's take the SMs as they free): the encode's rate under a
GEMM loop that outlasts it, and the GEMM's time under an encode that outlasts it. Writes OUT/encode-rate.json.
`--ncu` instead launches each timed chain once (square, encode at the largest k, products, fold), for Nsight Compute.

    python encode_rate.py --lib p1024=kv.so --out OUT [--ks 2,3 --blocks 65536 --gemm 2048,4096,14336] [--ncu]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import kv as K  # noqa: E402
import kv_bench as KB  # noqa: E402
from verity.core.protocols.remote_bandwidth import timing  # noqa: E402

REFERENCE = 1.499e10
HOST_SHARE = 0.125
SPIN_CYCLES = 20_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="NAME=PATH, kv.cu built at field NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--blocks", type=int, default=65536)
    ap.add_argument("--gemm", default="2048,4096,14336")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ncu", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "encode-rate.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    lib = KB.Lib(*a.lib.split("=", 1))
    f = lib.field
    stated = timing.GPU_SQUARINGS_PER_S[f.name]
    gpu_part = stated / (1 + HOST_SHARE)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    W = K.token_layer_words(f)
    res = {"lib": lib.name, "device": props.name, "sms": sms, "reference": REFERENCE, "stated": stated,
           "stated_gpu_part": gpu_part, "host_share": HOST_SHARE, "root_squarings": f.root_squarings,
           "words_a_block": W, "ks": [int(v) for v in a.ks.split(",")]}

    def over(x: float) -> dict:
        return {"over_reference": round(x / REFERENCE, 4), "over_stated_gpu_part": round(x / gpu_part, 4),
                "over_stated": round(x / stated, 4)}

    if a.ncu:
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        k = max(res["ks"])
        KB.set_key(lib, KB.lib_key(lib, k))
        for what, steps in ((0, 20_000), (2, 8), (3, 20_000), (4, 20_000)):
            lib.call("kv_chain", what, k, 1, steps, 2 * sms, xr.data_ptr(), KB.stream(torch))
        torch.cuda.synchronize()
        log(f"ncu: one launch each of chain 0, 2 (k = {k}), 3 and 4 at {2 * sms} CTAs")
        return 0
    KB.set_key(lib, KB.lib_key(lib, lib.kmax))
    sq = {ilp: KB.chain(lib, torch, 0, 1, ilp, sms, a.reps) for ilp in (1, 2)}
    best = max(r["per_s"] for r in sq.values())
    prod = KB.chain(lib, torch, 3, 1, 1, sms, a.reps)
    fold = KB.chain(lib, torch, 4, 1, 1, sms, a.reps)
    res["square"] = {"ilp": sq, "per_s": best, **over(best)}
    res["products"] = {**prod, "share_of_square": round(best / prod["per_s"], 4)}
    res["fold"] = {**fold, "share_of_square": round(best / fold["per_s"], 4)}
    log(f"{lib.name}: squarings {best:.4g}/s ({res['square']['over_reference']}x the reference, "
        f"{res['square']['over_stated_gpu_part']}x the stated GPU part {gpu_part:.4g}); FOLD 6's products alone "
        f"{prod['per_s']:.4g}/s ({res['products']['share_of_square']:.1%} of a squaring's time), the fold alone "
        f"{fold['per_s']:.4g}/s ({res['fold']['share_of_square']:.1%})")
    if best > gpu_part:
        log(f"the squaring beats the stated ceiling's GPU part ({best:.4g} > {gpu_part:.4g})")
    res["encode"], res["decode"] = {}, {}
    for k in res["ks"]:
        KB.set_key(lib, KB.lib_key(lib, k))
        e = KB.chain(lib, torch, 2, k, 1, sms, a.reps)
        d = KB.chain(lib, torch, 1, k, 1, sms, a.reps)
        es = e["per_s"] * k * f.root_squarings
        res["encode"][k] = {**e, "squarings_s": es, "blocks_s": e["per_s"] / W, "over_square": round(es / best, 4),
                            **over(es)}
        res["decode"][k] = {**d, "squarings_s": d["per_s"] * k, "blocks_s": d["per_s"] / W}
        log(f"k={k}: encode {e['per_s']:.4g} words/s = {es:.4g} squarings/s ({es / best:.1%} of the squaring, "
            f"{res['encode'][k]['over_reference']}x the reference, {res['encode'][k]['over_stated_gpu_part']}x the "
            f"stated GPU part) = {e['per_s'] / W:.4g} blocks/s; decode {d['per_s'] / W:.4g} blocks/s")
    res["memory"] = memory(lib, torch, a, sms, W, log)
    (out_dir / "encode-rate.json").write_text(json.dumps(res, indent=1, default=str))
    log("done")
    return 0


def memory(lib, torch, a, sms: int, W: int, log) -> dict:
    """kv_encode_bound in place over a.blocks blocks, alone and beside a GEMM."""
    f = lib.field
    n = a.blocks * W
    buf = KB.random_words(torch, n, f, 5)
    offs = torch.randint(-2 ** 31, 2 ** 31 - 1, (a.blocks, f.limbs), dtype=torch.int32, device="cuda")
    offs[:, -1] = 0
    lo, hi = torch.cuda.Stream.priority_range()
    s_lo, s_hi = torch.cuda.Stream(priority=lo), torch.cuda.Stream(priority=hi)
    M, Kd, N = (int(v) for v in a.gemm.split(","))
    ga = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
    gb = torch.randn(Kd, N, device="cuda").to(torch.bfloat16)
    gc = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    grids = {"two_an_sm": 2 * sms, "one_an_sm": sms, "a_word_a_thread": 0}
    out = {"blocks": a.blocks, "grids": grids, "gemm": {"m": M, "k": Kd, "n": N}, "priorities": [lo, hi], "ks": {}}

    def on(s, fn):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(s)
        fn()
        e1.record(s)
        return e0, e1

    def gemms(cnt):
        def run():
            with torch.cuda.stream(s_hi):
                for _ in range(cnt):
                    torch.matmul(ga, gb, out=gc)
        return run

    def alone(s, fn):
        ts = []
        for _ in range(a.reps + 1):
            torch.cuda.synchronize()
            e0, e1 = on(s, fn)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) / 1e3)
        return statistics.median(ts[1:])

    def together(enc, cnt):
        torch.cuda.synchronize()
        lib.call("kv_spin", SPIN_CYCLES, KB.stream(torch))
        ev = torch.cuda.Event()
        ev.record()
        s_lo.wait_event(ev)
        s_hi.wait_event(ev)
        r = on(s_lo, enc)
        g = on(s_hi, gemms(cnt))
        torch.cuda.synchronize()
        return r[0].elapsed_time(r[1]) / 1e3, g[0].elapsed_time(g[1]) / 1e3

    g1 = alone(s_hi, gemms(10)) / 10
    out["gemm"]["alone_ms"] = round(g1 * 1e3, 4)
    out["gemm"]["tflops"] = round(2 * M * Kd * N / g1 / 1e12, 1)
    log(f"memory: {a.blocks} blocks in place; GEMM {M}x{Kd}x{N} bf16 alone {g1 * 1e3:.3f} ms "
        f"({out['gemm']['tflops']} TFLOP/s); stream priorities encode {lo}, GEMM {hi}")
    for k in [int(v) for v in a.ks.split(",")]:
        KB.set_key(lib, KB.lib_key(lib, k))
        sq = W * k * f.root_squarings
        cell = {}
        for name, grid in grids.items():
            def enc(grid=grid):
                lib.call("kv_encode_bound", buf.data_ptr(), buf.data_ptr(), n, 0, k, offs.data_ptr(), grid,
                         s_lo.cuda_stream)

            t = alone(s_lo, enc)
            outlast, under = max(4, math.ceil(1.5 * t / g1)), max(1, math.floor(0.5 * t / g1))
            tb = statistics.median(together(enc, outlast)[0] for _ in range(a.reps))
            gm = statistics.median(together(enc, under)[1] for _ in range(a.reps)) / under
            c = {"grid": grid or -(-n // 128), "alone_s": round(t, 5), "blocks_s": round(a.blocks / t),
                 "squarings_s": a.blocks / t * sq, "beside_blocks_s": round(a.blocks / tb), "beside_squarings_s": a.blocks / tb * sq,
                 "beside_over_alone": round(tb / t, 3), "gemm_beside_ms": round(gm * 1e3, 4),
                 "gemm_slowdown": round(gm / g1, 3), "gemms": {"outlasting": outlast, "under": under}}
            cell[name] = c
            log(f"memory k={k} {name} ({c['grid']} CTAs): alone {c['blocks_s']:.4g} blocks/s "
                f"({c['squarings_s']:.4g} sq/s); beside the GEMM {c['beside_blocks_s']:.4g} blocks/s ({c['beside_over_alone']}x the time), the GEMM "
                f"{c['gemm_beside_ms']} ms ({c['gemm_slowdown']}x)")
        out["ks"][k] = cell
    return out


if __name__ == "__main__":
    sys.exit(main())
