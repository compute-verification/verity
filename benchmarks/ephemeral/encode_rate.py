"""What registering an activation with the codec costs: one launch of `kv.cu`'s encode (Sloth, k rounds of a square root
mod p a word) over a tensor of an activation's size, against reading the same bytes from HBM, on one GPU.

`kv.cu`, `kv.py` and `kv_bench.py` are ra-word-codec's (``--src``, shipped with ``research run --send``).  Before anything is
timed, `kv_bench.gate` checks the device's encode against `kv.encode_word` and its decode back, at every k.  Then for each
size: the encode's latency (one launch queued behind a spin, so the launch is not in it), its decode's, and the time to read
the bytes at the measured read rate R.  The ratio encode / read is what registration multiplies an activation's read time by
(it is read about once a write); at sizes below the GPU's words in flight the encode is latency-bound.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SIZES = [16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the directory holding kv.py and kv_bench.py")
    ap.add_argument("--lib", required=True, help="kv.cu built at KV_FIELD = 512")
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    import kv_bench as B
    import torch

    ks = [int(k) for k in args.ks.split(",")]
    lib = B.Lib("p512", args.lib)
    f = lib.field
    log = lambda m: print(m, flush=True)  # noqa: E731
    res: dict = {"field": f.name, "word_bytes": f.word_bytes, "gpu": torch.cuda.get_device_name(0),
                 "sms": torch.cuda.get_device_properties(0).multi_processor_count}
    res["gate"] = B.gate(lib, torch, ks, 48, 1 << 16, log)
    if not all(g["ok"] for g in res["gate"].values()):
        Path(args.out).write_text(json.dumps(res, indent=1))
        return 2
    store = B.random_words(torch, (2 << 30) // f.word_bytes, f, 3)
    res["read"] = B.read_rate(lib, torch, store, res["sms"], args.reps)
    R = res["read"]["gbps"] * 1e9
    del store
    st = B.stream(torch)
    words = B.random_words(torch, max(SIZES) // f.word_bytes, f, 9)
    y = torch.empty_like(words)
    back = torch.empty_like(words)
    res["sizes"] = []
    for k in ks:
        B.set_key(lib, B.K.key(f.name, B.SEED, k))
        for size in SIZES:
            n = size // f.word_bytes
            lib.call("kv_encode", B.ptr(words), B.ptr(y), n, 0, k, 0, st)
            enc = [B.queued(lib, torch, lambda: lib.call("kv_encode", B.ptr(words), B.ptr(y), n, 0, k, 0, st))
                   for _ in range(args.reps)]
            dec = [B.queued(lib, torch, lambda: lib.call("kv_decode", B.ptr(y), B.ptr(back), n, 0, k, 0, 0, st))
                   for _ in range(args.reps)]
            torch.cuda.synchronize()
            ok = bool(torch.equal(back[: n * f.word_bytes], words[: n * f.word_bytes]))
            e, d = statistics.median(enc), statistics.median(dec)
            r = size / R
            res["sizes"].append({"k": k, "bytes": size, "words": n, "encode_s": e, "decode_s": d, "read_s": r,
                                 "encode_over_read": e / r, "decode_over_read": d / r, "round_trip": ok})
            log(f"k={k} {size >> 10:8d} KiB  encode {e * 1e6:9.1f} us  decode {d * 1e6:8.1f} us  read {r * 1e6:8.2f} us  "
                f"encode/read {e / r:8.1f}  {'ok' if ok else 'ROUND TRIP FAILED'}")
    del words, y, back
    torch.cuda.empty_cache()
    res["lm_head"] = lm_head(torch, R, args.reps, log)
    Path(args.out).write_text(json.dumps(res, indent=1))
    return 0 if all(s["round_trip"] for s in res["sizes"]) else 3


def lm_head(torch, R: float, reps: int, log) -> list[dict]:
    """The lm_head GEMM as vLLM leaves it (BF16, F.linear) and in FP8 (torch._scaled_mm, per-tensor scales), at Llama-3.1's
    vocabulary and 8B's and 70B's hidden sizes, against reading its weight at R.  The weight is far larger than L2, so
    every call reads it from HBM."""
    import torch.nn.functional as F
    out = []
    for hidden in (4096, 8192):
        w = torch.randn(128256, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
        w8 = w.to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda", dtype=torch.float32)
        for b in (1, 4, 16, 64):
            x = torch.randn(b, hidden, device="cuda", dtype=torch.bfloat16)
            x8 = torch.randn(max(b, 16), hidden, device="cuda").to(torch.float8_e4m3fn)  # cuBLASLt's FP8 wants M >= 16
            row = {"hidden": hidden, "batch": b, "bf16_bytes": w.numel() * 2, "fp8_bytes": w8.numel()}
            for name, fn in (("bf16", lambda: F.linear(x, w)),
                             ("fp8", lambda: torch._scaled_mm(x8, w8.t(), one, one, out_dtype=torch.bfloat16))):
                fn()
                ts = []
                for _ in range(reps):
                    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    fn()
                    e.record()
                    e.synchronize()
                    ts.append(a.elapsed_time(e) / 1e3)
                row[f"{name}_s"] = statistics.median(ts)
            row["bf16_read_s"], row["fp8_read_s"] = row["bf16_bytes"] / R, row["fp8_bytes"] / R
            out.append(row)
            log(f"lm_head H={hidden} B={b}: bf16 {row['bf16_s'] * 1e6:.0f} us (read {row['bf16_read_s'] * 1e6:.0f}), "
                f"fp8 {row['fp8_s'] * 1e6:.0f} us (read {row['fp8_read_s'] * 1e6:.0f})")
        del w, w8
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    sys.exit(main())
