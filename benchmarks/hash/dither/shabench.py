"""shabench.cu's SHA-512 compression variants: each first checked bit-exact (every thread's chain of d <- SHA-512(d)
against hashlib), then timed for latency (one warp, a chain of N compressions: clock64 cycles a compression) and for
throughput (a block of THREADS on every SM, compressions a second).

    python shabench.py --lib libshabench.so --out OUT [--n 256]
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

VARIANTS = {0: "sha512.cuh (unroll 1, K by index)", 1: "fully unrolled", 2: "fully unrolled, 32-bit halves"}


def run(lib, v: int, blocks: int, threads: int, n: int, d0: torch.Tensor) -> tuple:
    out = torch.empty_like(d0)
    cyc = torch.empty(blocks * threads, dtype=torch.int64, device=d0.device)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st = torch.cuda.current_stream().cuda_stream
    a.record()
    rc = lib.sha_chain(v, blocks, threads, n, d0.data_ptr(), out.data_ptr(), cyc.data_ptr(), st)
    b.record()
    b.synchronize()
    if rc:
        raise RuntimeError(f"sha_chain V{v} returned {rc}")
    return out, cyc, a.elapsed_time(b)


def expect(d0: torch.Tensor, n: int) -> list:
    words = d0.cpu().view(-1, 8).tolist()
    res = []
    for row in words:
        m = b"".join((x & (2**64 - 1)).to_bytes(8, "big") for x in row)
        for _ in range(n):
            m = hashlib.sha512(m).digest()
        res.append([int.from_bytes(m[8 * j:8 * j + 8], "big") for j in range(8)])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    lib = C.CDLL(a.lib)
    lib.sha_chain.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    g = torch.Generator().manual_seed(5)

    def seeds(k):
        return torch.randint(-2**63, 2**63 - 1, (k * 8,), dtype=torch.int64, generator=g).to(dev)

    doc: dict = {"gpu": torch.cuda.get_device_name(dev), "sms": sms, "n": a.n, "variants": {}}
    ok_all = True
    for v, name in VARIANTS.items():
        d0 = seeds(128)
        got, _, _ = run(lib, v, 2, 64, 7, d0)
        want = expect(d0, 7)
        got_l = [[x & (2**64 - 1) for x in r] for r in got.cpu().view(-1, 8).tolist()]
        bad = sum(1 for x, y in zip(got_l, want) if x != y)
        rec: dict = {"name": name, "bad_threads": bad, "checked_threads": len(want)}
        ok_all &= bad == 0
        if bad == 0:
            lat = []
            for _ in range(5):
                d0 = seeds(32)
                _, cyc, ms = run(lib, v, 1, 32, a.n, d0)
                lat.append((statistics.median(cyc.cpu().tolist()) / a.n, 1e3 * ms / a.n))
            rec["latency_cycles"] = statistics.median(x[0] for x in lat)
            rec["latency_us"] = statistics.median(x[1] for x in lat)
            rec["clock_ghz"] = rec["latency_cycles"] / rec["latency_us"] / 1e3
            thr = {}
            for threads in (32, 128, 256, 512):
                n = max(16, a.n // 4)
                d0 = seeds(sms * threads)
                try:
                    run(lib, v, sms, threads, n, d0)
                    ms = statistics.median(run(lib, v, sms, threads, n, d0)[2] for _ in range(3))
                except RuntimeError as e:
                    thr[threads] = {"error": str(e)}
                    continue
                thr[threads] = {"g_per_s": sms * threads * n / ms / 1e6, "us_a_chain_step": 1e3 * ms / n}
            rec["throughput"] = thr
        doc["variants"][v] = rec
        print(json.dumps({"v": v, **rec}), flush=True)
    doc["ok"] = ok_all
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "shabench.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
