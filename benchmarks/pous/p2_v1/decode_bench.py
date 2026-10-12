"""P2 v1's decode throughput on one GPU, kernel by kernel, each gated byte-exact against `reference.py`.

For every library given (`--lib DIR`, each a `libp2v1.so` built with other switches), on `--blocks` uniformly random canonical
blocks of `C` with the v4 keys of a fixed setup:

- `keys_s`: the counter-mode SHA-512 expansion of every block's `K` and `t` (`gpu.Kernels.keys`), alone;
- `kernel_s`: `p2v1_decode` alone, on keys expanded beforehand;
- `decode_s`: the harness's decode pass (`gpu.Kernels.decode`: keys, then the kernel), or the library's fused decode
  (`p2v1_decode_fused`, the keys expanded inside it) when the library has one;
- the gate: sampled blocks (and edge values of `c` in the first blocks) decoded on the device equal the reference's decode
  under the reference's keys, their `K` and `t` rows equal the reference's (`reference.keys`), the library reports v4's
  keys (`Kernels.keys_version`), the deployment entry of the pinned `p2-v4-keys.json` (`harness.KEYS`) expands on the device
  to its pinned `K` and `t`, every block of `m` equals the first library's, and no kernel flag is raised; with `--keys-all`,
  every block's `K` and `t` rows equal `reference.stream`'s (hashlib on every CPU, compared by SHA-512 per range of
  blocks).

GB/s is of `W`, the payload bytes (2,054 of each 2,056-byte block), over the median time.  `setup_s` is the wall time of the
first expansion of every block's keys (the setup's key derivation for the whole store, allocation included), before the timed
repetitions.  Writes `decode-bench.json`, and with `--result-dir` the run's `result.json` (`result_doc`): the device as
nvidia-smi names it, `keys_per_s`, `setup_s` and the keys' agreement with the reference, of the first library."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from p2_v1 import reference as R  # noqa: E402
from p2_v1.gpu import Kernels  # noqa: E402
from p2_v1.harness import KEYS  # noqa: E402

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


KEYS_RANGE = 2048


def _range_digests(job: tuple[int, int, int]) -> tuple[int, str, str]:
    """SHA-512 of the `K` rows and of the `t` rows (first candidates) of blocks `lo .. hi − 1`."""
    lo, hi, segments = job
    P = R.DEPLOYMENT.with_segments(segments)
    prefix = R.expand_prefix(SALT, P, MODEL)
    hk, ht = hashlib.sha512(), hashlib.sha512()
    for i in range(lo, hi):
        base = prefix + (i // P.blocks_per_segment).to_bytes(8, "big") + (i % P.blocks_per_segment).to_bytes(8, "big")
        hk.update(R.stream(base, b"K", P.width_bytes))
        ht.update(R.stream(base, b"t", P.width_bytes))
    return lo, hk.hexdigest(), ht.hexdigest()


def keys_all(k, t, n: int, segments: int) -> dict[str, object]:
    """Every block's `K` and `t` rows on the device against hashlib's, a range of `KEYS_RANGE` blocks a job on every CPU."""
    import multiprocessing as mp
    import os

    kb, tb = k.cpu().numpy(), t.cpu().numpy()
    jobs = [(lo, min(lo + KEYS_RANGE, n), segments) for lo in range(0, n, KEYS_RANGE)]
    with mp.get_context("fork").Pool(len(os.sched_getaffinity(0))) as pool:
        want = pool.map(_range_digests, jobs)
    bad = [lo for lo, wk, wt in want
           if hashlib.sha512(kb[lo:lo + KEYS_RANGE].tobytes()).hexdigest() != wk
           or hashlib.sha512(tb[lo:lo + KEYS_RANGE].tobytes()).hexdigest() != wt]
    return {"blocks": n, "ranges": len(jobs), "passed": not bad, "failed_ranges": bad[:16]}


def pinned_keys(lib_dir: str, build_dir: str) -> dict[str, object]:
    """The pinned deployment keys (`harness.KEYS`, w = 16,448) expanded by this library at their setup and block."""
    v = next(x for x in json.loads(KEYS.read_text())["vectors"] if int(x["w"]) == R.DEPLOYMENT.w)
    seg, blk = int(v["segment"]), int(v["block"])
    P = R.DEPLOYMENT.with_segments(seg + 1)
    kern = Kernels(lib_dir, build_dir, P, R.expand_prefix(bytes.fromhex(v["setup_salt"]), P,
                                                          bytes.fromhex(v["model_commitment"])))
    k, t = kern.keys(seg * P.blocks_per_segment + blk, 1)
    got = {"K": k[0].cpu().numpy().tobytes().hex() == v["K"], "t": t[0].cpu().numpy().tobytes().hex() == v["t"]}
    return {"vector": v["id"], "keys_version": kern.keys_version, "passed": all(got.values()) and kern.keys_version == 4, **got}


def smi_device(uuid: str | None) -> str:
    """nvidia-smi's `index, uuid, name` line of the leased GPU (`GPU_LEASE_UUID`'s first), else of the first visible one."""
    sel = ["-i", uuid.split(",")[0]] if uuid else []
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader", *sel],
                         capture_output=True, text=True, check=False).stdout
    return out.strip().splitlines()[0] if out.strip() else ""


def result_doc(doc: dict, smi: str, run_id: str, run_dir: Path, out: Path, argv: list[str]) -> dict:
    """The run's `result.json` (research/result/v0.1) from `decode-bench.json`'s document, for its first library: `device`
    from nvidia-smi's `index, uuid, name` line, `keys_per_s` (blocks whose `K` and `t` are derived per second, at the median
    of the timed expansions), `setup_s`, and `agreement`, the keys' checks against the reference."""
    r = doc["results"][0]
    g = r["gate"]
    index, uuid, name = ([x.strip() for x in smi.split(",", 2)] + ["", "", ""])[:3]
    every, pinned = g["keys_all"], g["pinned_keys"]
    agreement = {"passed": (g["reference_keys_equal"] == g["reference_blocks"] and pinned["passed"]
                            and every is not None and every["passed"]),
                 "reference_keys_equal": g["reference_keys_equal"], "reference_blocks": g["reference_blocks"],
                 "keys_all": every, "pinned_keys": pinned}
    keys_per_s = r["blocks"] / r["keys_s"]
    passed = doc["passed"] and agreement["passed"]
    files = [str(p.relative_to(run_dir)) for p in [out / "decode-bench.json"] if p.is_relative_to(run_dir)]
    return {"schema": "research/result/v0.1", "run_id": run_id,
            "workload_fingerprint": {"harness": "benchmarks/pous/p2_v1/decode_bench.py", "argv": argv,
                                     "source_sha": os.environ.get("RESEARCH_SOURCE_SHA"), "build": r["build"]},
            "validation": {"status": "passed" if passed else "failed", "checker": "decode_bench gate + keys agreement",
                           "detail": f"gate {'passed' if doc['passed'] else 'failed'}; keys agreement "
                                     f"{'passed' if agreement['passed'] else 'failed'}", "evidence": files},
            "device": {"name": name or doc["device"], "index": int(index) if index.isdigit() else None, "uuid": uuid or None},
            "profile": "p2-2054B", "blocks": r["blocks"], "keys_version": r["keys_version"],
            "keys_per_s": keys_per_s, "setup_s": r["setup_s"], "keys_s": r["keys_s"], "agreement": agreement,
            "measurements": [{"name": "keys_per_s", "value": keys_per_s, "unit": "blocks/s"},
                             {"name": "setup_s", "value": r["setup_s"], "unit": "s"},
                             {"name": "keys_s", "value": r["keys_s"], "unit": "s"},
                             {"name": "keys_GBps", "value": r["keys_GBps"], "unit": "GB/s"},
                             {"name": "kernel_GBps", "value": r["kernel_GBps"], "unit": "GB/s"},
                             {"name": "decode_GBps", "value": r["decode_GBps"], "unit": "GB/s"}],
            "measurement_files": files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, help="a directory holding a libp2v1.so; the first is the baseline")
    ap.add_argument("--build-dir", required=True, help="pous_native's JIT build directory")
    ap.add_argument("--blocks", type=int, default=8 << 16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--gate", type=int, default=48, help="pseudo-random blocks checked against the reference")
    ap.add_argument("--keys-all", action="store_true", help="check every block's v4 K and t against hashlib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--result-dir", help="the run directory to write result.json in (RESEARCH_RUN_ID names the run)")
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
    ref: dict[int, tuple[bytes, bytes, bytes]] = {}
    for i in picks:
        k, t, _ = R.keys(SALT, P, MODEL, i // P.blocks_per_segment, i % P.blocks_per_segment)
        m = R.decode(P, int.from_bytes(c_host[i * nb:(i + 1) * nb], "big"), k, t)
        ref[i] = (m.to_bytes(nb, "big"), k.to_bytes(nb, "big"), t.to_bytes(nb, "big"))

    results, base_m = [], {}
    for lib_dir in a.lib:
        pinned = pinned_keys(lib_dir, a.build_dir)
        kern = Kernels(lib_dir, a.build_dir, P, R.expand_prefix(SALT, P, MODEL))
        lib = kern.lib
        fused = hasattr(lib, "p2v1_decode_fused")
        version = kern.keys_version
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        k, t = kern.keys(0, n)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - t0
        keys_gate = {i: bytes(k[i].cpu().numpy()) == ref[i][1] and bytes(t[i].cpu().numpy()) == ref[i][2] for i in picks}
        every = keys_all(k, t, n, P.segments) if a.keys_all else None
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
        gate = {i: bytes(m_full[i].cpu().numpy()) == ref[i][0] for i in picks}
        kernel_equal = bool(torch.equal(m, m_full))
        if version not in base_m:
            base_m[version] = m_full.clone()
        same_as_base = bool(torch.equal(m_full, base_m[version]))
        pay = n * P.payload_bytes / 1e9
        keys_ok = (all(keys_gate.values()) and version == 4 and pinned["passed"]
                   and (every is None or every["passed"]))
        r = {"lib": lib_dir, "fused": fused, "keys_version": version, "blocks": n, "setup_s": setup_s,
             "keys_s": statistics.median(keys_reps),
             "kernel_s": statistics.median(kernel_reps), "decode_s": statistics.median(decode_reps),
             "keys_reps_s": keys_reps, "kernel_reps_s": kernel_reps, "decode_reps_s": decode_reps,
             "kernel_GBps": pay / statistics.median(kernel_reps), "decode_GBps": pay / statistics.median(decode_reps),
             "keys_GBps": pay / statistics.median(keys_reps),
             "gate": {"passed": all(gate.values()) and keys_ok and kernel_equal and same_as_base and flags == 0,
                      "reference_blocks": len(picks), "reference_equal": sum(gate.values()),
                      "reference_keys_equal": sum(keys_gate.values()), "pinned_keys": pinned, "keys_all": every,
                      "kernel_equals_full": kernel_equal,
                      "all_blocks_equal_baseline": same_as_base, "flags": flags,
                      "failed_blocks": [i for i, ok in gate.items() if not ok][:16]}}
        try:
            r["build"] = json.loads((Path(lib_dir) / "build.json").read_text())
        except OSError:
            r["build"] = None
        results.append(r)
        print(json.dumps({x: r[x] for x in ("lib", "fused", "keys_version", "keys_GBps", "kernel_GBps", "decode_GBps")}
                         | {"gate": r["gate"]["passed"]}),
              flush=True)
        del m, m_full, kern, lib
        torch.cuda.empty_cache()

    doc = {"device": torch.cuda.get_device_name(dev), "blocks": n, "payload_bytes": n * P.payload_bytes, "reps": a.reps,
           "results": results, "passed": all(r["gate"]["passed"] for r in results)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "decode-bench.json").write_text(json.dumps(doc, indent=1))
    if a.result_dir:
        run_dir = Path(a.result_dir).resolve()
        res = result_doc(doc, smi_device(os.environ.get("GPU_LEASE_UUID")), os.environ["RESEARCH_RUN_ID"], run_dir,
                         Path(a.out).resolve(), sys.argv)
        tmp = run_dir / ".result.json.tmp"
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, run_dir / "result.json")
        doc["passed"] = res["validation"]["status"] == "passed"
        print(json.dumps({x: res[x] for x in ("device", "keys_per_s", "setup_s")} | {"agreement": res["agreement"]["passed"]}),
              flush=True)
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
