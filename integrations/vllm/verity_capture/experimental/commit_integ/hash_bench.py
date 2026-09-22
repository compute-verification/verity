"""Host-side hashing microbenchmark (CPU only; runs anywhere): the hashing component of the commit paths, measured in isolation.

  native_host   pos_leaf over 4096-B chunks, hashlib, 1 and N threads (hashlib releases the GIL) + fasttree fold
  cmt_ref_host  CMT-1 `leaf_digests_host` (one hashlib call per 64-B leaf, Python loop) + fold
  cmt_ref_torch CMT-1 torch-vectorised SHA-256 on the CPU tensor backend (correctness vs hashlib asserted) + fold

Output: JSON with MB/s per path and the calculated per-request hashing time for the manifest's naturally-materialised bytes.
Labels: every rate here is MEASURED on this host; the per-request seconds are CALCULATED (bytes / rate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

from verity_capture.experimental.fa2_commit import engine as eng


def bench(fn, reps: int) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--natural-bytes", type=int, default=106_995_516, help="manifest naturally-materialised bytes per request (CMT-1 manifest)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.mb << 20
    data = os.urandom(n)
    mv = memoryview(data)
    res = {"host": platform.node(), "cpu": platform.processor() or platform.machine(), "python": platform.python_version(), "bytes": n,
           "threads": a.threads, "paths": {}}

    chunk = 4096

    def native_1():
        leaves = [eng.pos_leaf(mv[o:o + chunk]) for o in range(0, n, chunk)]
        return eng.ft.fold_levels(leaves)[-1][0]

    pool = ThreadPoolExecutor(a.threads)

    def native_n():
        per = max(chunk, ((n // a.threads) // chunk) * chunk)
        rng = [(o, min(o + per, n)) for o in range(0, n, per)]

        def part(r):
            lo, hi = r
            return [eng.pos_leaf(mv[o:min(o + chunk, hi)]) for o in range(lo, hi, chunk)]

        leaves = []
        for lst in pool.map(part, rng):
            leaves.extend(lst)
        return eng.ft.fold_levels(leaves)[-1][0]

    def raw_sha():
        return hashlib.sha256(data).digest()

    for name, fn in (("sha256_flat_1thread", raw_sha), ("native_host_4096B_1thread", native_1), (f"native_host_4096B_{a.threads}threads", native_n)):
        t = bench(fn, a.reps)
        res["paths"][name] = {"s": t, "MB_s": n / t / 1e6, "per_request_s_calculated": a.natural_bytes / (n / t)}
        print(f"{name:36s} {n / t / 1e6:9.1f} MB/s   per-request(natural {a.natural_bytes / 1e6:.1f} MB) = {a.natural_bytes / (n / t):.3f} s")

    try:
        from verity_capture.experimental import commit as C

        m = min(n, 8 << 20)             # the per-leaf Python loop is slow: 8 MB sample
        sub = data[:m]

        def ref_host():
            leaves = C.leaf_digests_host(sub)
            return eng.ft.fold_levels(list(leaves))[-1][0]

        t = bench(ref_host, 1)
        res["paths"]["cmt_ref_host_64B_leaves"] = {"s": t, "bytes": m, "MB_s": m / t / 1e6, "per_request_s_calculated": a.natural_bytes / (m / t)}
        print(f"{'cmt_ref_host_64B_leaves':36s} {m / t / 1e6:9.1f} MB/s   per-request = {a.natural_bytes / (m / t):.3f} s  (sample {m >> 20} MB)")
        try:
            import torch

            from verity_capture.experimental.commit import engine as E
            from verity_capture.experimental.commit import torch_sha256 as TS
            torch.set_num_threads(a.threads)
            m2 = min(n, 16 << 20)
            st = torch.frombuffer(bytearray(data[:m2]), dtype=torch.uint8)
            nl = m2 // 64
            # correctness: first 1024 leaves vs hashlib
            d = E.leaf_digests_torch(st[: 1024 * 64], 1024)
            ref = C.leaf_digests_host(data[: 1024 * 64])
            ok = all(bytes(d[i].numpy().tobytes()) == ref[i] for i in range(1024))
            res["torch_sha256_matches_hashlib_1024_leaves"] = ok

            def ref_torch():
                leaves = E.leaf_digests_torch(st, nl)
                return TS.fold_levels(leaves)[-1][0]

            t = bench(ref_torch, 1)
            res["paths"]["cmt_ref_torch_cpu_64B_leaves"] = {"s": t, "bytes": m2, "MB_s": m2 / t / 1e6, "per_request_s_calculated": a.natural_bytes / (m2 / t),
                                                            "torch": torch.__version__, "threads": torch.get_num_threads()}
            print(f"{'cmt_ref_torch_cpu_64B_leaves':36s} {m2 / t / 1e6:9.1f} MB/s   per-request = {a.natural_bytes / (m2 / t):.3f} s  (sample {m2 >> 20} MB; "
                  f"digests==hashlib: {ok})")
        except Exception as e:  # noqa: BLE001
            res["torch_error"] = repr(e)
            print("torch backend skipped:", repr(e))
    except Exception as e:  # noqa: BLE001
        res["cmt_ref_error"] = repr(e)
        print("cmt_ref engine skipped:", repr(e))
    pool.shutdown()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
