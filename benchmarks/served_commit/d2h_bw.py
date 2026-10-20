"""Device-to-host bandwidth through the copy engines (cudaMemcpyAsync into pinned host memory, no SMs) on one GPU: one large copy
on 1, 2 and 4 streams, a stream of copies of each chunk size the Commit's buffers come in, H2D and both directions at once, and
a bf16 GEMM loop timed alone and beside a running D2H, so a slowdown of the GEMM would show the copies taking SMs or HBM.
Feasibility of hashing the Commit's values on the host; measurement only, JSON to $OUT/d2h_bw.json.

    bash benchmarks/served_commit/d2h_bw.sh     (TOTAL_MB: bytes per timed copy set, default 2048; REPS, default 7)
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
from collections.abc import Callable

import torch

MB = 1 << 20


def timed(fn: Callable[[], None], reps: int) -> float:
    """Median ms of fn on the current stream's clock, each rep behind a 2 ms device sleep so the launches queue ahead."""
    fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(2_000_000)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    return statistics.median(ms)


def copies(dst: torch.Tensor, src: torch.Tensor, chunk: int, streams: list[torch.cuda.Stream]) -> Callable[[], None]:
    """Copy src to dst in chunk-byte pieces, dealt round-robin to streams that start on and rejoin the current stream."""
    n = src.numel()
    pieces = [(o, min(chunk, n - o)) for o in range(0, n, chunk)]

    def run() -> None:
        cur = torch.cuda.current_stream()
        for s in streams:
            s.wait_stream(cur)
        for i, (o, k) in enumerate(pieces):
            with torch.cuda.stream(streams[i % len(streams)]):
                dst[o:o + k].copy_(src[o:o + k], non_blocking=True)
        for s in streams:
            cur.wait_stream(s)

    return run


def main() -> None:
    out = os.environ.get("OUT", os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out"))
    total, reps = int(os.environ.get("TOTAL_MB", "2048")) * MB, int(os.environ.get("REPS", "7"))
    query = "name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    smi = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True,
                         check=False).stdout.strip()
    props = torch.cuda.get_device_properties(0)
    res: dict = {"device": props.name, "nvidia_smi_pcie": smi, "async_engines": getattr(props, "async_engine_count", None),
                 "total_mb": total // MB, "reps": reps}
    print(json.dumps({k: res[k] for k in ("device", "nvidia_smi_pcie", "async_engines")}))
    dev = torch.empty(total, dtype=torch.uint8, device="cuda").random_(0, 255)
    dev2 = torch.empty_like(dev)
    host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    host2 = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    pool = [torch.cuda.Stream() for _ in range(4)]

    def gbs(ms: float, nbytes: int = total) -> float:
        return round(nbytes / ms / 1e6, 2)

    rows = []
    for direction, dst, src in (("d2h", host, dev), ("h2d", dev2, host2)):
        for ns in (1, 2, 4):
            ms = timed(copies(dst, src, -(-total // ns), pool[:ns]), reps)
            rows.append({"what": f"{direction} one copy per stream", "streams": ns, "chunk_mb": total // ns / MB,
                         "ms": round(ms, 3), "gb_per_s": gbs(ms)})
    for chunk in (128 << 10, 1 * MB, 4 * MB, 16 * MB, 64 * MB):
        for ns in (1, 2):
            ms = timed(copies(host, dev, chunk, pool[:ns]), reps)
            rows.append({"what": "d2h chunked", "streams": ns, "chunk_mb": chunk / MB, "ms": round(ms, 3), "gb_per_s": gbs(ms)})
    d2h, h2d = copies(host, dev, -(-total // 2), pool[:2]), copies(dev2, host2, -(-total // 2), pool[2:])
    ms = timed(lambda: (d2h(), h2d()), reps)
    rows.append({"what": "d2h + h2d at once", "streams": 4, "chunk_mb": total / 2 / MB, "ms": round(ms, 3),
                 "gb_per_s": gbs(ms, 2 * total)})
    res["copies"] = rows
    for r in rows:
        print(json.dumps(r))

    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    n_mm = 40

    def gemms() -> None:
        for _ in range(n_mm):
            torch.mm(a, b, out=c)

    alone = timed(gemms, reps)
    # The GEMMs' own span with eight D2H copies of TOTAL_MB queued beside them on another stream, which outlast the GEMMs.
    side, g = torch.cuda.Stream(), []
    for _ in range(reps):
        cur = torch.cuda.current_stream()
        torch.cuda._sleep(2_000_000)
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            for _ in range(8):
                host.copy_(dev, non_blocking=True)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gemms()
        e1.record()
        cur.wait_stream(side)
        torch.cuda.synchronize()
        g.append(e0.elapsed_time(e1))
    beside = statistics.median(g)
    res["gemm"] = {"alone_ms": round(alone, 3), "alone_tflops": round(2 * 8192**3 * n_mm / alone / 1e9, 1),
                   "beside_d2h_ms": round(beside, 3), "slowdown_pct": round(100 * (beside / alone - 1), 2)}
    print(json.dumps(res["gemm"]))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "d2h_bw.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
