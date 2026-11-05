"""Driver for the PoUS bulk-bandwidth audit measurements on an RTX PRO 6000 (Blackwell, 96 GB GDDR7, PCIe Gen5 x16).

Three measurements, each written to one JSON under `--out`:

- `validate`: the Triton digest kernel against the numpy reference on a small `C`, bit-exact. A failure aborts the job so
  no wrong number is published.
- `honest`: the honest digest time over `R` random 64 KiB labels of a `C` resident in HBM, its distribution over many
  trials and the HBM read rate it achieves, for a sweep of `C` sizes and `R`; plus the raw HBM read rate and the PCIe H2D
  rate (contiguous and gathered) as references.
- `cheater`: for a fixed `C`, the best time a server reaches when a fraction `f` of the challenged labels live in pinned
  host DRAM, by async H2D copies overlapped with the digest of the HBM-resident rest, over a sweep of `R` and `f`; and the
  PCIe H2D rate it measures.

Run on a leased GPU. `honest` and `cheater`
write results per point to `--out` and skip points already there, so a preempted sweep resumes from a persistent `--out`
instead of starting over. The honest/cheater separation and the recommended `R`, `T` are derived from these JSONs by
`analyze.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digest as D  # noqa: E402

GIB = 1 << 30


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    return torch


def gpu_info(torch) -> dict:
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    out = {
        "name": p.name,
        "total_mem_bytes": p.total_memory,
        "multi_processor_count": p.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    try:
        import pynvml  # noqa: F401
    except Exception:
        pass
    return out


def make_C(torch, n_blocks: int, seed: int = 0):
    """A `[n_blocks, WORDS] int32` store of random bytes in HBM."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-(2**31), 2**31, (n_blocks, D.WORDS), dtype=torch.int32, device="cuda", generator=g)


def random_idx(torch, n_blocks: int, r: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, n_blocks, (r,), dtype=torch.int64, device="cuda", generator=g)


def time_events(torch, fn, iters: int, warmup: int = 3) -> list[float]:
    """Milliseconds per call of `fn` over `iters` runs (plus `warmup` discarded), each timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def dist(ms: list[float], nbytes: int) -> dict:
    a = np.array(ms, dtype=np.float64)
    gbps = nbytes / (a / 1e3) / 1e9
    return {
        "trials": len(a),
        "bytes": nbytes,
        "ms_min": float(a.min()),
        "ms_p50": float(np.percentile(a, 50)),
        "ms_p99": float(np.percentile(a, 99)),
        "ms_p999": float(np.percentile(a, 99.9)),
        "ms_max": float(a.max()),
        "ms_mean": float(a.mean()),
        "gbps_at_min": float(gbps.max()),
        "gbps_at_p50": float(nbytes / (float(np.percentile(a, 50)) / 1e3) / 1e9),
    }


# --------------------------------------------------------------------------- validate


def cmd_validate(args, torch, meta):
    prev = load_existing(args, "validate.json", {"kind": "validate"})
    if prev.get("all_ok"):  # a resume need not re-validate; the kernel is bit-exact and the hash tables are fixed
        print("validate: OK (cached)")
        return
    n = 64
    C = make_C(torch, n, seed=1)
    host = C.to("cpu").numpy().view(np.uint32)
    results = []
    for key, r in [(0xABCDEF, 8), (0x1234, 17), (0x55AA55AA, 32)]:
        idx = random_idx(torch, n, r, seed=key & 0xFFFF)
        idx_np = idx.to("cpu").numpy()
        ref = D.reference_digest(host, idx_np, key, lane=0)
        dev = D.device_digest(C, idx, key, lane=0)
        g, _ = D.lane_points(key, 0)
        bh_ref = int(D.reference_block_hash(host[int(idx_np[0])], g))
        bh_int = int(D.reference_block_hash_interleaved(host[int(idx_np[0])], g))
        ok = ref == dev and bh_ref == bh_int
        results.append({"key": key, "R": r, "ref": ref, "dev": dev, "interleave_ok": bh_ref == bh_int, "ok": ok})
    g0, _ = D.lane_points(777, 0)
    g1, _ = D.lane_points(777, 1)
    idx = random_idx(torch, n, 12, seed=3)
    idx_np = idx.to("cpu").numpy()
    o0, o1 = D.block_hashes2(C, idx, g0, g1)
    two_ok = all(int(o0[r]) == int(D.reference_block_hash(host[int(idx_np[r])], g0)) for r in range(12)) and all(
        int(o1[r]) == int(D.reference_block_hash(host[int(idx_np[r])], g1)) for r in range(12)
    )
    ok = all(x["ok"] for x in results) and two_ok
    out = {"meta": meta, "kind": "validate", "cases": results, "two_lane_ok": two_ok, "all_ok": ok}
    write(args, "validate.json", out)
    if not ok:
        raise SystemExit("VALIDATION FAILED")
    print("validate: OK")


# --------------------------------------------------------------------------- honest


def raw_bandwidth(args, torch):
    """Reference rates: HBM read (big contiguous reduction), and PCIe H2D contiguous and gathered."""
    out = {}
    # HBM read: sum a big device buffer (read-dominated) to anchor the peak the digest can approach.
    nb = 4 * GIB
    buf = torch.empty(nb, dtype=torch.int8, device="cuda")
    ms = time_events(torch, lambda: buf.view(torch.float32).sum(), iters=30)
    out["hbm_read_reduce"] = dist(ms, nb)
    del buf
    torch.cuda.empty_cache()
    # PCIe H2D contiguous: pinned host -> device.
    for sz in (256 * (1 << 20), 1 * GIB):
        hpin = torch.empty(sz, dtype=torch.int8, pin_memory=True)
        dev = torch.empty(sz, dtype=torch.int8, device="cuda")
        s = torch.cuda.Stream()
        def cp(hpin=hpin, dev=dev, s=s):
            with torch.cuda.stream(s):
                dev.copy_(hpin, non_blocking=True)
            s.synchronize()
        ms = time_events(torch, cp, iters=20)
        out[f"pcie_h2d_contig_{sz >> 20}MiB"] = dist(ms, sz)
        del hpin, dev
        torch.cuda.empty_cache()
    # PCIe H2D gathered: many 64 KiB labels (what a real cheater's scattered host labels cost per-copy).
    nblk = 8192
    hpin = torch.empty((nblk, D.WORDS), dtype=torch.int32, pin_memory=True)
    dev = torch.empty((nblk, D.WORDS), dtype=torch.int32, device="cuda")
    s = torch.cuda.Stream()
    def gcp(hpin=hpin, dev=dev, s=s):
        with torch.cuda.stream(s):
            for i in range(nblk):
                dev[i].copy_(hpin[i], non_blocking=True)
        s.synchronize()
    ms = time_events(torch, gcp, iters=5, warmup=1)
    out[f"pcie_h2d_gathered_{nblk}x64KiB"] = dist(ms, nblk * D.BLOCK_BYTES)
    del hpin, dev
    torch.cuda.empty_cache()
    # Zero-copy read: the kernel digests labels straight out of pinned host memory over PCIe (UVA), no staging copy.
    try:
        hz = torch.empty((nblk, D.WORDS), dtype=torch.int32, pin_memory=True)
        hz.random_(-(2**31), 2**31)
        zidx = torch.arange(nblk, dtype=torch.int64, device="cuda")
        g0, _ = D.lane_points(0xB01C, 0)
        ms = time_events(torch, lambda: D.block_hashes(hz, zidx, g0), iters=10, warmup=2)
        out[f"zerocopy_digest_{nblk}x64KiB"] = dist(ms, nblk * D.BLOCK_BYTES)
        del hz
    except Exception as e:  # UVA host access may be unsupported; record it rather than abort the run
        out["zerocopy_digest_error"] = repr(e)
    torch.cuda.empty_cache()
    return out


def cmd_honest(args, torch, meta):
    sizes = [int(x) for x in args.c_gib.split(",")]
    rs = [1 << e for e in range(args.r_lo, args.r_hi + 1)]
    out = load_existing(args, "honest.json", {"meta": meta, "kind": "honest", "l2_mib": 128, "raw": {}, "points": []})
    done = {(p["c_gib"], p["R"]) for p in out["points"]}
    for cg in sizes:
        n = (cg * GIB) // D.BLOCK_BYTES
        C = None  # allocate C lazily, so a resume that skips a whole size never touches HBM for it
        for r in rs:
            if r > n or (cg, r) in done:
                continue
            if C is None:
                C = make_C(torch, n, seed=cg)
            idx = random_idx(torch, n, r, seed=0x1000 + r)
            g, z = D.lane_points(0xB01C, 0)
            g1, _ = D.lane_points(0xB01C, 1)
            nbytes = r * D.BLOCK_BYTES
            # precompute key-dependent tables once (off the timed path), as a native responder would; time only device work
            combine0 = D.combine_table(g, "cuda")
            combine1 = D.combine_table(g1, "cuda")
            gp0, gp1 = pow(g, D.INTERLEAVE, D.MOD), pow(g1, D.INTERLEAVE, D.MOD)
            zp = D.z_powers_dev(z, r, "cuda")
            hbuf = torch.empty(r, dtype=torch.int64, device="cuda")
            hbuf1 = torch.empty(r, dtype=torch.int64, device="cuda")
            # kernel-only (the HBM read + fold of one lane), many trials for the distribution
            k_ms = time_events(torch, lambda: D.block_hashes_pre(C, idx, combine0, gp0, hbuf), iters=args.trials)
            # full one-lane digest (audit answer latency = read + fold), and the two-lane digest (one read)
            f_ms = time_events(torch, lambda: D.fold_pre(D.block_hashes_pre(C, idx, combine0, gp0, hbuf), zp),
                               iters=max(30, args.trials // 4))
            two_ms = time_events(torch, lambda: D.block_hashes2_pre(C, idx, combine0, combine1, gp0, gp1, hbuf, hbuf1),
                                 iters=max(30, args.trials // 4))
            out["points"].append({
                "c_gib": cg, "n_blocks": n, "R": r,
                "kernel1": dist(k_ms, nbytes),
                "digest1": dist(f_ms, nbytes),
                "kernel2": dist(two_ms, nbytes),
            })
            write(args, "honest.json", out)  # per-point: a preemption keeps every point already measured
            print(f"honest C={cg}GiB R={r}: kernel1 p50 {dist(k_ms, nbytes)['ms_p50']:.3f} ms "
                  f"{dist(k_ms, nbytes)['gbps_at_p50']:.0f} GB/s")
        if C is not None:
            del C
            torch.cuda.empty_cache()
    if not out.get("raw") or out["raw"].get("error"):  # the reference rates once, after the sweep (or on a resume that lacks them)
        try:
            out["raw"] = raw_bandwidth(args, torch)
        except Exception as e:  # a reference-rate hiccup must not cost the sweep
            out["raw"] = {"error": repr(e)}
        write(args, "honest.json", out)
    want = [(cg, 1 << e) for cg in sizes for e in range(args.r_lo, args.r_hi + 1)
            if (1 << e) <= (cg * GIB) // D.BLOCK_BYTES]
    have = {(p["c_gib"], p["R"]) for p in out["points"]}
    out["complete"] = all(k in have for k in want) and bool(out.get("raw")) and not out["raw"].get("error")
    write(args, "honest.json", out)
    if out["complete"]:
        print("HONEST_COMPLETE")


# --------------------------------------------------------------------------- cheater


def host_resident_mask(torch, n_blocks: int, f: float, seed: int):
    """A deterministic random subset of labels (fraction ~f) that the cheater keeps only in host DRAM."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    thr = int(f * (1 << 30))
    return (torch.randint(0, 1 << 30, (n_blocks,), dtype=torch.int64, device="cuda", generator=g) < thr)


def measure_cheater_async(torch, C_dev, host_pin, idx, host_hit, g):
    """Best overlapped cheater time: copy the host-resident challenged labels H2D on one stream while the HBM-resident
    ones are digested on another, then digest the staged labels. `host_hit` is a bool mask over `idx`; `host_pin` is a
    pinned host buffer of at least `R_host` rows (its contents are irrelevant — only the transfer size and the overlap
    matter for timing).

    The host labels are copied as one contiguous `R_host * 64 KiB` transfer (the cheater's best case: pre-arranged, peak
    PCIe), which is optimistic for the cheater and so conservative for the audit.
    """
    idx_dev = idx[~host_hit]
    rh = int(host_hit.sum().item())
    s_copy = torch.cuda.Stream()
    s_cmp = torch.cuda.Stream()
    out_dev = torch.empty(max(idx_dev.numel(), 1), dtype=torch.int64, device="cuda")
    stage = torch.empty((max(rh, 1), D.WORDS), dtype=torch.int32, device="cuda")
    stage_idx = torch.arange(max(rh, 1), dtype=torch.int64, device="cuda")
    out_stage = torch.empty(max(rh, 1), dtype=torch.int64, device="cuda")
    combine = torch.tensor([pow(int(g), D.INTERLEAVE - 1 - l, D.MOD) for l in range(D.INTERLEAVE)],
                           dtype=torch.int64, device="cuda")
    gp = pow(int(g), D.INTERLEAVE, D.MOD)
    kern = D._kernel()

    def once():
        cur = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(cur)
        s_copy.wait_stream(cur)
        s_cmp.wait_stream(cur)
        if rh:
            with torch.cuda.stream(s_copy):
                stage.copy_(host_pin[:rh], non_blocking=True)
                kern[(rh,)](stage, stage_idx, combine, out_stage, gp, D.MOD, D.INTERLEAVE, D.N_STEPS)
        if idx_dev.numel():
            with torch.cuda.stream(s_cmp):
                kern[(idx_dev.numel(),)](C_dev, idx_dev, combine, out_dev, gp, D.MOD, D.INTERLEAVE, D.N_STEPS)
        cur.wait_stream(s_copy)
        cur.wait_stream(s_cmp)
        end.record(cur)
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(2):
        once()
    torch.cuda.synchronize()
    ms = [once() for _ in range(20)]
    return ms, rh


def measure_cheater_zerocopy(torch, C_dev, host_pin, idx, host_hit, g):
    """The other honest way to touch host-resident labels: the digest kernel reads them straight out of pinned host
    memory over PCIe (UVA zero-copy), with no staging copy, while the HBM-resident labels are digested from device
    memory on another stream. `host_pin` is a pinned host buffer of `R` rows; the host-resident challenged labels are
    read at their scattered positions within it (a realistic, non-coalesced access pattern), overlapped with the device
    digest of the HBM-resident rest."""
    idx_dev = idx[~host_hit]
    idx_host_local = torch.nonzero(host_hit, as_tuple=True)[0]  # scattered rows in [0, R) that live in host DRAM
    rh = int(idx_host_local.numel())
    s_zc = torch.cuda.Stream()
    s_cmp = torch.cuda.Stream()
    out_dev = torch.empty(max(idx_dev.numel(), 1), dtype=torch.int64, device="cuda")
    out_host = torch.empty(max(rh, 1), dtype=torch.int64, device="cuda")
    combine = torch.tensor([pow(int(g), D.INTERLEAVE - 1 - l, D.MOD) for l in range(D.INTERLEAVE)],
                           dtype=torch.int64, device="cuda")
    gp = pow(int(g), D.INTERLEAVE, D.MOD)
    kern = D._kernel()

    def once():
        cur = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(cur)
        s_zc.wait_stream(cur)
        s_cmp.wait_stream(cur)
        if rh:
            with torch.cuda.stream(s_zc):
                kern[(rh,)](host_pin, idx_host_local, combine, out_host, gp, D.MOD, D.INTERLEAVE, D.N_STEPS)
        if idx_dev.numel():
            with torch.cuda.stream(s_cmp):
                kern[(idx_dev.numel(),)](C_dev, idx_dev, combine, out_dev, gp, D.MOD, D.INTERLEAVE, D.N_STEPS)
        cur.wait_stream(s_zc)
        cur.wait_stream(s_cmp)
        end.record(cur)
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(2):
        once()
    torch.cuda.synchronize()
    ms = [once() for _ in range(20)]
    return ms, rh


def cmd_cheater(args, torch, meta):
    cg = int(args.cheater_c_gib)
    n = (cg * GIB) // D.BLOCK_BYTES
    g, z = D.lane_points(0xB01C, 0)
    rs = [1 << e for e in range(args.r_lo, args.r_hi + 1)]
    fs = [float(x) for x in args.fracs.split(",")]
    out = load_existing(args, "cheater.json", {"meta": meta, "kind": "cheater", "c_gib": cg, "n_blocks": n, "points": []})
    done = {(p["R"], p["f"]) for p in out["points"]}
    zc_ok = not out.get("zerocopy_error")  # once zero-copy fails (e.g. UVA unsupported), stop retrying it
    if not all((r, f) in done for r in rs for f in fs):
        C = make_C(torch, n, seed=cg)  # C in HBM (8 GiB); the device-resident challenged labels are digested from here
        for r in rs:
            idx = None
            host_pin = None  # a pinned host buffer sized to this R, allocated lazily so a short lease window fits
            for f in fs:
                if (r, f) in done:
                    continue
                if idx is None:
                    idx = random_idx(torch, n, r, seed=0x2000 + r)
                    host_pin = torch.empty((r, D.WORDS), dtype=torch.int32, pin_memory=True)
                mask = host_resident_mask(torch, n, f, seed=0x3000 + int(f * 1000))
                host_hit = mask[idx]
                ms, rh = measure_cheater_async(torch, C, host_pin, idx, host_hit, g)
                nbytes = r * D.BLOCK_BYTES
                host_bytes = rh * D.BLOCK_BYTES
                d = dist(ms, nbytes)
                d["f_nominal"] = f
                d["R_host"] = rh
                d["host_bytes"] = host_bytes
                d["pcie_gbps_at_p50"] = (host_bytes / (d["ms_p50"] / 1e3) / 1e9) if rh else 0.0
                dz = None
                if zc_ok:
                    try:
                        zc_ms, _ = measure_cheater_zerocopy(torch, C, host_pin, idx, host_hit, g)
                        dz = dist(zc_ms, nbytes)
                        dz["pcie_gbps_at_p50"] = (host_bytes / (dz["ms_p50"] / 1e3) / 1e9) if rh else 0.0
                    except Exception as e:
                        zc_ok = False
                        out["zerocopy_error"] = repr(e)
                out["points"].append({"R": r, "f": f, "async": d, "zerocopy": dz,
                                      "R_host": rh, "host_bytes": host_bytes})
                write(args, "cheater.json", out)  # incremental: a later crash keeps the points already measured
                zp = f"{dz['ms_p50']:.3f}" if dz else "n/a"
                print(f"cheater C={cg}GiB R={r} f={f}: Rh={rh} async p50 {d['ms_p50']:.3f} ms zerocopy p50 {zp} ms")
            del host_pin, idx
            torch.cuda.empty_cache()
        del C
        torch.cuda.empty_cache()
    have = {(p["R"], p["f"]) for p in out["points"]}
    out["complete"] = all((r, f) in have for r in rs for f in fs)
    write(args, "cheater.json", out)
    if out["complete"]:
        print("CHEATER_COMPLETE")


# --------------------------------------------------------------------------- io


def write(args, name: str, obj: dict):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / name
    p.write_text(json.dumps(obj, indent=2))
    print(f"wrote {p}")


def load_existing(args, name: str, default: dict) -> dict:
    """Resume support: the prior JSON at `--out/name` if it is readable, else `default`. `--out` is a path that persists
    across runs, so a preempted sweep continues from the points it already wrote instead of starting over."""
    p = Path(args.out) / name
    if p.exists():
        try:
            prev = json.loads(p.read_text())
            if isinstance(prev, dict) and prev.get("kind") == default["kind"]:
                prev.setdefault("points", default.get("points", []))
                return prev
        except (ValueError, OSError):
            pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "honest", "cheater", "all"])
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--c-gib", default="1,4,8,16")
    ap.add_argument("--cheater-c-gib", default="8")
    ap.add_argument("--r-lo", type=int, default=10)
    ap.add_argument("--r-hi", type=int, default=16)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--fracs", default="0.01,0.05,0.10,0.50,1.0")
    args = ap.parse_args()

    torch = _torch()
    meta = {**gpu_info(torch), "argv": sys.argv, "unix_time": time.time()}

    if args.cmd in ("validate", "all"):
        cmd_validate(args, torch, meta)
    if args.cmd in ("honest", "all"):
        cmd_honest(args, torch, meta)
    if args.cmd in ("cheater", "all"):
        cmd_cheater(args, torch, meta)


if __name__ == "__main__":
    main()
