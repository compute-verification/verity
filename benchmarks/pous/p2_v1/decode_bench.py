"""P2 v1's decode throughput on one GPU, kernel by kernel, each gated byte-exact against `reference.py`.

For every library given (`--lib DIR`, each a `libp2v1.so` built with other switches), on `--blocks` uniformly random canonical
blocks of `C` with the v1 keys of a fixed setup:

- `keys_s`: the SHAKE256 expansion of every block's `K` and `t` (`gpu.Kernels.keys`), alone;
- `kernel_s`: `p2v1_decode` alone, on keys expanded beforehand;
- `decode_s`: the harness's decode pass (`gpu.Kernels.decode`: keys, then the kernel), or the library's fused decode
  (`p2v1_decode_fused`, the keys expanded inside it) when the library has one;
- the gate: sampled blocks (and edge values of `c` in the first blocks) decoded on the device equal the reference's decode
  under the reference's keys, every block of `m` equals the first library's, and no kernel flag is raised.

GB/s is of `W`, the payload bytes (2,054 of each 2,056-byte block), over the median time.  Writes `decode-bench.json`."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from p2_v1 import reference as R  # noqa: E402
from p2_v1.gpu import Kernels  # noqa: E402

SALT = hashlib.sha256(b"verity-pous/bench/p2-v1/decode-bench/salt").digest()
MODEL = hashlib.sha512(b"verity-pous/bench/p2-v1/decode-bench/model").digest()


def timed(torch, fn, reps: int) -> list[float]:
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    return out


def edge_values(p: int) -> list[int]:
    return [0, 1, 2, 3, p - 1, p - 2, (p - 1) // 2, (1 << 16447) - 1, 1 << 16447, p - (1 << 32)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, help="a directory holding a libp2v1.so; the first is the baseline")
    ap.add_argument("--build-dir", required=True, help="pous_native's JIT build directory")
    ap.add_argument("--blocks", type=int, default=8 << 16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--gate", type=int, default=48, help="pseudo-random blocks checked against the reference")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    P = R.DEPLOYMENT.with_segments(-(-a.blocks // R.DEPLOYMENT.blocks_per_segment))
    n, nb = a.blocks, P.width_bytes
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(20261006)
    c = torch.randint(0, 256, (n, nb), dtype=torch.uint8, generator=g)
    c[:, 0] &= 0x7F
    edges = edge_values(P.p)
    for i, v in enumerate(edges):
        c[i] = torch.tensor(list(v.to_bytes(nb, "big")), dtype=torch.uint8)
    c = c.to(dev)
    c_host = c.cpu().numpy().tobytes()
    picks = sorted(set(range(len(edges))) | {n - 1} | {
        int.from_bytes(hashlib.sha256(f"decode-bench/{j}".encode()).digest()[:8], "big") % n for j in range(a.gate)})
    want = {}
    for i in picks:
        k, t, _ = R.keys(SALT, P, MODEL, i // P.blocks_per_segment, i % P.blocks_per_segment)
        want[i] = R.decode(P, int.from_bytes(c_host[i * nb:(i + 1) * nb], "big"), k, t).to_bytes(nb, "big")

    results, base_m = [], None
    for lib_dir in a.lib:
        kern = Kernels(lib_dir, a.build_dir, P, R.expand_prefix(SALT, P, MODEL))
        lib = kern.lib
        fused = hasattr(lib, "p2v1_decode_fused")
        k, t = kern.keys(0, n)
        m = torch.empty_like(c)
        stream = lambda: torch.cuda.current_stream().cuda_stream  # noqa: E731

        def kernel_only() -> None:
            rc = lib.p2v1_decode(c.data_ptr(), m.data_ptr(), k.data_ptr(), t.data_ptr(), n, P.c0, kern.flags.data_ptr(), stream())
            assert rc == 0, rc

        keys_reps = timed(torch, lambda: kern.keys(0, n), a.reps)
        kernel_reps = timed(torch, kernel_only, a.reps)
        out = {}
        if fused:
            def full() -> None:
                out["m"] = kern.decode_fused(c, 0)
        else:
            def full() -> None:
                out["m"] = kern.decode(c, 0)
        decode_reps = timed(torch, full, a.reps)
        m_full = out["m"]
        del k, t
        flags = kern.raised()
        torch.cuda.synchronize()
        gate = {i: bytes(m_full[i].cpu().numpy()) == want[i] for i in picks}
        kernel_equal = bool(torch.equal(m, m_full))
        if base_m is None:
            base_m = m_full.clone()
        same_as_base = bool(torch.equal(m_full, base_m))
        pay = n * P.payload_bytes / 1e9
        r = {"lib": lib_dir, "fused": fused, "blocks": n, "keys_s": statistics.median(keys_reps),
             "kernel_s": statistics.median(kernel_reps), "decode_s": statistics.median(decode_reps),
             "keys_reps_s": keys_reps, "kernel_reps_s": kernel_reps, "decode_reps_s": decode_reps,
             "kernel_GBps": pay / statistics.median(kernel_reps), "decode_GBps": pay / statistics.median(decode_reps),
             "keys_GBps": pay / statistics.median(keys_reps),
             "gate": {"passed": all(gate.values()) and kernel_equal and same_as_base and flags == 0,
                      "reference_blocks": len(picks), "reference_equal": sum(gate.values()), "kernel_equals_full": kernel_equal,
                      "all_blocks_equal_baseline": same_as_base, "flags": flags,
                      "failed_blocks": [i for i, ok in gate.items() if not ok][:16]}}
        try:
            r["build"] = json.loads((Path(lib_dir) / "build.json").read_text())
        except OSError:
            r["build"] = None
        results.append(r)
        print(json.dumps({x: r[x] for x in ("lib", "fused", "keys_GBps", "kernel_GBps", "decode_GBps")} | {"gate": r["gate"]["passed"]}),
              flush=True)
        del m, m_full, kern, lib
        torch.cuda.empty_cache()

    doc = {"device": torch.cuda.get_device_name(dev), "blocks": n, "payload_bytes": n * P.payload_bytes, "reps": a.reps,
           "results": results, "passed": all(r["gate"]["passed"] for r in results)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "decode-bench.json").write_text(json.dumps(doc, indent=1))
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
