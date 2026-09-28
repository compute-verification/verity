"""Native route-U kernel gate and throughput against the existing signed-s8 dump kernel, on one RTX 4090.

    python benchmarks/pouw/route_u_bench.py --out RESULT.json

The gate uses ncp-v1's production P at k=1088, compares every 16x16 tile digest against ``checked()``, and checks the
exact int32 output.  Timing uses Qwen2.5-3B's k=2048 stacked depth (3k), m=16, n=256; both kernels dump every d=16
checkpoint, so the ratio isolates mixed u8xs8 plus c0 from signed s8.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kernels import load, ncp_route_u_tiles  # noqa: E402
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.core.protocols.pouw.schemes import ncp  # noqa: E402


_FLUSH: list[torch.Tensor] = []


def timed(fn, reps: int = 30, warm: int = 5) -> float:
    if not _FLUSH:
        _FLUSH.append(torch.empty(256 << 20, dtype=torch.uint8, device="cuda"))
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        _FLUSH[0].zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def s8(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int8).view(np.uint8)).cuda()


def gate(ext) -> dict:
    m, k, n = 16, 1088, 256
    scheme, rng = ncp.NCP(n_min=64), random.Random(19)
    ctx = Context(b"\x71" * 32, Matmul(17, Shape(m, k, n), 23), b"\x72" * 32, b"\x73" * 32)
    a = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(m)]
    b = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(n)]
    ops = scheme.encode(ctx, scheme.noise(ctx), a, b, Tile(tuple(range(m)), tuple(range(n))))
    x = np.asarray(ops.x, dtype=np.int16)
    y = np.asarray(ops.y, dtype=np.int8)
    xu = torch.from_numpy(np.ascontiguousarray(x + ncp.BETA, dtype=np.uint8)).cuda()
    yu = s8(y)
    tiles = ncp_route_u_tiles(ext, xu, yu, cfg=1)
    refs = [Tile(tuple(range(16)), tuple(range(j, j + 16))) for j in range(0, n, 16)]
    digest_ok = []
    useful_ok = []
    for (digest, useful), tile in zip(tiles, refs, strict=True):
        sub = scheme.encode(ctx, scheme.noise(ctx), a, [b[j] for j in tile.cols], tile)
        checked = scheme.checked(ctx, sub)
        digest_ok.append(digest == scheme.digest(ctx, None, checked))
        useful_ok.append(useful == scheme.useful(ctx, None, sub, checked))
    aa = np.stack([np.frombuffer(row, dtype=np.int8) for row in a]).astype(np.int64)
    bb = np.stack([np.frombuffer(row, dtype=np.int8) for row in b]).astype(np.int64)
    output = np.concatenate([np.asarray(t[1], dtype=np.int64) for t in tiles], axis=1)
    exact = np.array_equal(output, aa @ bb.T)
    return {
        "shape": [m, k, n],
        "tiles": len(tiles),
        "tile_digests_match": sum(digest_ok),
        "tile_outputs_match": sum(useful_ok),
        "output_exact": exact,
        "ok": exact and all(digest_ok) and all(useful_ok),
    }


def benchmark(ext, reps: int) -> dict:
    m, k, n, stacked = 16, 2048, 256, 3 * 2048
    x = torch.randint(-127, 127, (m, stacked), dtype=torch.int8, device="cuda")
    y = torch.randint(-127, 127, (n, stacked), dtype=torch.int8, device="cuda")
    xu, xs, ys = (x.to(torch.int16) + 128).to(torch.uint8), x.view(torch.uint8), y.view(torch.uint8)
    native = lambda: ext.ncp_route_u(xu, ys, 1)
    signed = lambda: ext.ncp_gemm(xs, ys, 2, 1)
    c_u, ck_u = native()
    c_s, _, ck_s = signed()
    same = torch.equal(c_u, c_s) and ck_u.shape == ck_s.shape
    u_ms, s_ms = timed(native, reps), timed(signed, reps)
    stacked_macs = m * n * stacked
    return {
        "shape_m_k_n": [m, k, n],
        "stacked_depth": stacked,
        "checkpoints": stacked // 16,
        "checkpoint_bytes": m * n * (stacked // 16) * 4,
        "final_outputs_equal_s8": bool(same),
        "route_u_ms": u_ms,
        "s8_ms": s_ms,
        "route_u_over_s8": u_ms / s_ms,
        "route_u_stacked_tmac_s": stacked_macs / (u_ms * 1e-3) / 1e12,
        "s8_stacked_tmac_s": stacked_macs / (s_ms * 1e-3) / 1e12,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "route_u.json")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090":
        raise SystemExit("RTX 4090 required")
    ext = load()
    result = {
        "schema": "verity-pouw/route-u-bench/v1",
        "env": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "gate": gate(ext),
        "timing": benchmark(ext, args.reps),
    }
    print(json.dumps(result, indent=1))
    args.out.write_text(json.dumps(result, indent=1))
    if not result["gate"]["ok"] or not result["timing"]["final_outputs_equal_s8"]:
        raise SystemExit(2)
    import run_result
    run_result.write(
        args.out,
        True,
        "native route-U tile digests and outputs match checked(); timing final outputs match signed s8",
        json.dumps({"gate": result["gate"], "ratio": result["timing"]["route_u_over_s8"]}),
        [
            ("route_u_over_s8", result["timing"]["route_u_over_s8"], "ratio"),
            ("route_u_stacked_tmac_s", result["timing"]["route_u_stacked_tmac_s"], "TMAC/s"),
            ("s8_stacked_tmac_s", result["timing"]["s8_stacked_tmac_s"], "TMAC/s"),
        ],
    )


if __name__ == "__main__":
    main()
