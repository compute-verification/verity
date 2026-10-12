"""Measure the fused encode -> column-hash streaming commitment (``commit_stream``) at the campaign's Ligero shapes.

    python3 -m encode.commit_bench --reps 7 --out commit_bench.json [--shapes B,A] [--hashes sha256,blake3] [--quick]

Every configuration is checked bit-exactly against the unfused path (v2 encoder + ``hash_gpu.merkle``) on 1,300 rows
before it is timed; timings are CUDA events around ``StreamingCommitter.commit`` (slice loop as a CUDA graph + the
16,384-leaf tree), warm, median of ``--reps``.  The input matrix is resident on the device (data-independent cost).
Configurations: slice rows sweep (L2-resident slices vs HBM-sized slices), overlap on/off, ``keep_codeword``
(cosets streamed to a 15.3 GB store), and the encode-only / hash-only halves of the fused pipeline (diagnostics).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"))

import cupy as cp
from hash_gpu import merkle

from encode import babybear as bb
from encode.commit_stream import StreamingCommitter
from encode.encode_ntt_simt import RSEncoderSIMT

ROWS = {"B": 310_602, "A": 48_873}
PEAK_BW = 1.008e12
UNFUSED_MS_4090 = {"B": {"encode": 32.2, "sha256": 45.7, "blake3": 56.6}, "A": {"encode": 5.3, "sha256": 7.5, "blake3": 9.2}}


def device_info() -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    _free, total = cp.cuda.runtime.memGetInfo()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=10, check=False).stdout.strip()
    return {"gpu": props["name"].decode(), "sms": props["multiProcessorCount"], "l2_bytes": props["l2CacheSize"],
            "device_memory_bytes": total, "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(), "cupy": cp.__version__, "numpy": np.__version__,
            "nvidia_smi_driver": smi}


def timed(fn, reps: int):
    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(ts), ts


def random_rows(R: int, seed: int):
    rs = cp.random.RandomState(seed)
    return rs.randint(0, bb.P, size=(R, bb.K_CODE), dtype=cp.int32).view(cp.uint32)


def check(sc: StreamingCommitter, rows: int = 1300) -> bool:
    x = bb.fill_rows(rows, bb.K_CODE, seed=rows)
    xd = cp.asarray(x)
    code = RSEncoderSIMT()(xd)
    t = merkle.commit(code, sc.hash)
    leaves, root = sc.commit(xd)
    cp.cuda.Device().synchronize()
    ok = bool(cp.array_equal(leaves, t.levels[0])) and bool(cp.array_equal(root, t.root))
    if sc.keep:
        ok = ok and bool(cp.array_equal(sc.store[:rows], code[:, bb.K_CODE:]))
    sc._graphs.clear()
    return ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="commit_bench.json")
    ap.add_argument("--shapes", default="B,A")
    ap.add_argument("--hashes", default="sha256,blake3")
    ap.add_argument("--slices", default="256,512,1024,4096,65536")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        ROWS.update({"B": 20_000, "A": 8_192})
        a.reps = min(a.reps, 3)
    res = {"device": device_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a), "rows": dict(ROWS),
           "unfused_ms_4090_reference": UNFUSED_MS_4090, "results": [], "checks": {}, "bench_kind": "commit",
           "scope": "component: fused RS-encode -> column-hash -> Merkle root of the Ligero witness matrix; nothing else of the prover",
           "bit_exact_note": "every timed fused configuration equals the unfused path (v2 encoder + hash_gpu.merkle) on 1,300 rows: leaves, root and (keep) store"}
    print(json.dumps(res["device"]), flush=True)
    configs = []
    for hash in a.hashes.split(","):
        for sl in (int(s) for s in a.slices.split(",")):
            for ov in (False, True):
                configs.append({"hash": hash, "slice_rows": sl, "overlap": ov, "keep": False, "mode": "fused"})
        configs.append({"hash": hash, "slice_rows": 512, "overlap": False, "keep": True, "mode": "fused"})
        configs.append({"hash": hash, "slice_rows": 512, "overlap": False, "keep": False, "mode": "encode_only"})
        configs.append({"hash": hash, "slice_rows": 512, "overlap": False, "keep": False, "mode": "hash_only"})
    for shape in a.shapes.split(","):
        R = ROWS[shape]
        rows = random_rows(R, seed=1)
        for cfg in configs:
            sc = StreamingCommitter(hash=cfg["hash"], slice_rows=cfg["slice_rows"], overlap=cfg["overlap"], keep_codeword=cfg["keep"])
            key = f"{cfg['hash']}_s{cfg['slice_rows']}_ov{int(cfg['overlap'])}_keep{int(cfg['keep'])}"
            if cfg["mode"] == "fused" and key not in res["checks"]:
                res["checks"][key] = check(sc)
                print(f"check {key}: {res['checks'][key]}", flush=True)
            sc.skip_encode = cfg["mode"] == "hash_only"
            sc.skip_hash = cfg["mode"] == "encode_only"
            try:
                med, ts = timed(lambda sc=sc, rows=rows: sc.commit(rows), a.reps)
            except cp.cuda.memory.OutOfMemoryError:
                print(f"{shape} {key} {cfg['mode']}: OOM", flush=True)
                del sc
                cp.get_default_memory_pool().free_all_blocks()
                continue
            free, total = cp.cuda.runtime.memGetInfo()
            r = {**cfg, "name": f"{key}_{cfg['mode']}", "shape": shape, "rows": R, "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R,
                 "bytes_in": R * bb.K_CODE * 4, "codeword_bytes": R * bb.N_CODE * 4,
                 "read_floor_ms": R * bb.K_CODE * 4 / PEAK_BW * 1e3, "codeword_gb_per_s": R * bb.N_CODE * 4 / (med / 1e3) / 1e9,
                 "device_memory_in_use_bytes": total - free}
            if cfg["mode"] == "fused":
                unf = UNFUSED_MS_4090[shape]["encode"] + UNFUSED_MS_4090[shape][cfg["hash"]]
                r["unfused_ms_reference"] = unf
                r["speedup_vs_unfused"] = unf / med if not a.quick else None
            res["results"].append(r)
            print(f"{shape} {key} {cfg['mode']}: {med:.2f} ms ({r['us_per_row']:.3f} us/row), codeword {r['codeword_gb_per_s']:.0f} GB/s, "
                  f"mem {r['device_memory_in_use_bytes'] / 2**30:.1f} GiB" + (f", {r['speedup_vs_unfused']:.2f}x vs unfused" if r.get("speedup_vs_unfused") else ""), flush=True)
            del sc
            cp.get_default_memory_pool().free_all_blocks()
        del rows
        cp.get_default_memory_pool().free_all_blocks()
    res["all_bit_exact"] = all(res["checks"].values())
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["run_id"] = os.environ.get("RESEARCH_RUN_ID")
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "commit_bench.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
