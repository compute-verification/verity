"""Driver for the volatile-memory secure-erasure calibration: how long an honest prover takes to fill all of its HBM (and
host DRAM) with a fresh-key stream and answer a keyed digest over it, on node 2's RTX PRO 6000s, under co-tenant load.

Commands, each writing one JSON under `--out` (and only there; nothing else touches the disk):

- `validate`: every fill stream, layout and digest order on the device against `fill.py`'s CPU references, bit-exact, on
  small buffers. A mismatch aborts the job, so no number from a wrong kernel is published. `--device cpu` runs the same
  check through Triton's interpreter (`TRITON_INTERPRET=1`), without a GPU.
- `sweep`: fill-then-digest runs over allocation sizes (up to all of HBM that is safe to take: free memory less
  `--reserve-gib`), fill streams, layouts and digest orders, each run under a fresh key; per run the fill and digest
  device times (CUDA events) and the end-to-end wall time from the key's arrival to the digest on the host.
- `host`: host-DRAM fill of a `--host-gib` buffer by `--procs` forked workers (memset, the write ceiling; AES-128-CTR and
  ChaCha20 keystreams through OpenSSL, `cryptography`), cold (first touch) and warm; and a GPU-generated fill landed in a
  pinned host buffer over PCIe (`d2h`).
- `soak`: back-to-back fill-digest runs at the largest size for `--minutes` (or until `--stop-epoch`, or SIGTERM), every
  `--idle-every`th run after `--idle-s` of idle (an audit that finds the GPU idle), with a host-stall probe process beside
  it (`probe`), the node's pressure (PSI) and load every 10 s, and the GPUs' co-tenants from the lease owner lines.
- `result`: `result.json` for the run, from what `--out` holds.

Per-run data are written as they are measured (a preempted lease keeps every point it finished), and a rerun with the same
`--out` skips points already there.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill as F  # noqa: E402

GIB = 1 << 30
MIB = 1 << 20
STOP = threading.Event()


def _on_term(signum, frame):
    STOP.set()


# --------------------------------------------------------------------------- io


def write(out: str, name: str, obj: dict):
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    tmp = p / f".{name}.tmp"
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    os.replace(tmp, p / name)


def load_existing(out: str, name: str, default: dict) -> dict:
    p = Path(out) / name
    if p.exists():
        try:
            prev = json.loads(p.read_text())
            if isinstance(prev, dict) and prev.get("kind") == default["kind"]:
                return prev
        except (ValueError, OSError):
            pass
    return default


def pct(a, q) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if len(a) else float("nan")


def dist(ms) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "p50": pct(a, 50), "p99": pct(a, 99), "p999": pct(a, 99.9),
            "max": float(a.max()), "mean": float(a.mean())}


def node_meta() -> dict:
    m = {"host": platform.node(), "python": platform.python_version(), "argv": sys.argv, "unix_time": time.time(),
         "gpu_lease_uuid": os.environ.get("GPU_LEASE_UUID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "run_id": os.environ.get("RESEARCH_RUN_ID"), "source_sha": os.environ.get("RESEARCH_SOURCE_SHA")}
    try:
        m["affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return m


def gpu_meta(torch) -> dict:
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    out = {"name": p.name, "total_mem_bytes": p.total_memory, "sm_count": p.multi_processor_count,
           "capability": list(torch.cuda.get_device_capability(i)), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton

        out["triton"] = triton.__version__
    except ImportError:
        pass
    uuid = os.environ.get("GPU_LEASE_UUID", "").split(",")[0]
    if uuid:
        try:
            q = "index,uuid,pci.bus_id,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,power.limit,temperature.gpu"
            out["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--id={uuid}", f"--query-gpu={q}", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def co_tenants(lease_dir: str = "/run/gpu-lease") -> list[str]:
    """Who holds each GPU of the node right now (the gpu-lease owner lines, shortened), to say what the run shared with."""
    out = []
    try:
        for p in sorted(Path(lease_dir).glob("*.owner")):
            line = p.read_text(errors="replace").strip().replace("\n", " ")
            head, _, cmd = line.partition("cmd=")
            fields = dict(w.split("=", 1) for w in head.split() if "=" in w)
            keep = [f"{k}={fields[k]}" for k in ("run", "who", "since", "until", "preempt", "timed") if k in fields]
            # an owner file outlives its lease; a holder whose pid is gone is stale
            live = fields.get("pid", "").isdigit() and Path(f"/proc/{fields['pid']}").exists()
            out.append(" ".join([p.stem, "held" if live else "stale", *keep, f"cmd={cmd[:80]}"]) if line
                       else f"{p.stem} free")
    except OSError:
        pass
    return out


def pressure() -> dict:
    """The node's CPU/memory/IO pressure (PSI `some avg10`) and load average: what the co-tenants put on the host."""
    d = {}
    for k in ("cpu", "memory", "io"):
        try:
            first = Path(f"/proc/pressure/{k}").read_text().split("\n")[0]
            d[k] = float(first.split("avg10=")[1].split()[0])
        except (OSError, IndexError, ValueError):
            pass
    try:
        d["load1"] = float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, IndexError, ValueError):
        pass
    return d


# --------------------------------------------------------------------------- device context


def _torch(device: str):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    return torch


class Ctx:
    """The buffers one process reuses for every run: the fill buffer (`n_max` blocks), the digest's scratch, the key."""

    def __init__(self, torch, n_max: int, device: str = "cuda"):
        self.torch, self.device, self.n_max = torch, device, n_max
        pin = device == "cuda"
        self.key_host = torch.zeros(16, dtype=torch.int32, pin_memory=pin)
        self.key_np = self.key_host.numpy()
        self.key_dev = torch.zeros(16, dtype=torch.int32, device=device)
        self.hbuf = torch.empty(n_max, dtype=torch.int64, device=device)
        self.zbuf = torch.empty(n_max, dtype=torch.int64, device=device)
        self.cbuf = torch.empty(F.D.INTERLEAVE, dtype=torch.int64, device=device)
        self.ar = torch.arange(n_max, dtype=torch.int64, device=device)
        self.buf = torch.empty(n_max * F.WORDS, dtype=torch.int32, device=device)
        self.perm_gen = torch.Generator(device=device)
        if device == "cuda":
            self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    def set_key(self, key):
        self.key_np[: len(key)] = F.signed_words(key)
        self.key_dev.copy_(self.key_host, non_blocking=self.device == "cuda")

    def order_idx(self, order: str, n: int, perm_seed: int):
        if order == "seq":
            return self.ar[:n]
        self.perm_gen.manual_seed(perm_seed)
        return self.torch.randperm(n, generator=self.perm_gen, device=self.device)

    def digest(self, n: int, idx, g: int, z: int):
        return F.device_digest(self.torch, self.buf[: n * F.WORDS].view(n, F.WORDS), idx, g, z, self.hbuf, self.zbuf,
                               self.cbuf)

    def run(self, gen: str, layout: str, order: str, n: int, seed: int, run: int) -> dict:
        """One timed fill-then-digest under run `run`'s fresh keys. `wall_ms` runs from before the key is derived to the
        digest's value on the host; `fill_ms`/`digest_ms` are the device's own (CUDA events) times of the two phases."""
        e0, e1, e2 = self.ev
        t0 = time.perf_counter()
        self.set_key(F.fill_key(seed, run))
        e0.record()
        F.device_fill(self.buf, self.key_dev, n, gen, layout)
        e1.record()
        g, z, ps = F.digest_key(seed, run)
        d = self.digest(n, self.order_idx(order, n, ps), g, z)
        e2.record()
        val = int(d.item())
        t1 = time.perf_counter()
        return {"fill_ms": e0.elapsed_time(e1), "digest_ms": e1.elapsed_time(e2), "gpu_ms": e0.elapsed_time(e2),
                "wall_ms": (t1 - t0) * 1e3, "digest": val, "t0": t0}


def max_blocks(torch, reserve_gib: float) -> tuple[int, int, int]:
    """The largest fill that leaves `reserve_gib` free once the context's scratch (48 B a block) is allocated."""
    free, total = torch.cuda.mem_get_info()
    return int((free - reserve_gib * GIB) // (F.BLOCK_BYTES + 48)), free, total


# --------------------------------------------------------------------------- validate


def _interpreter_closures():
    """`hbm_audit/digest.py` nests its kernel's helpers in a closure, which the compiler captures but the interpreter
    (`TRITON_INTERPRET=1`, re-executing the kernel against its module globals) does not; lift them into the globals."""
    for k in F.D._build():
        fn = k.fn
        fn.__globals__.update(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ())))
        for name in fn.__code__.co_freevars:
            inner = getattr(fn.__globals__[name], "fn", None)
            if inner is not None and inner.__closure__:
                fn.__globals__.update(zip(inner.__code__.co_freevars, (c.cell_contents for c in inner.__closure__)))


def cmd_validate(args, meta):
    torch = _torch(args.device)
    if os.environ.get("TRITON_INTERPRET") == "1":
        _interpreter_closures()
    seed = 0xE5A5E
    ctx = Ctx(torch, 64, args.device)
    cases, run = [], 0
    for n in (5, 64):
        for gen in F.GENERATORS:
            for layout in F.LAYOUTS:
                run += 1
                key = F.fill_key(seed, run)
                ref = F.reference_fill(gen, key, n, layout)
                ctx.buf.fill_(0)
                ctx.set_key(key)
                F.device_fill(ctx.buf, ctx.key_dev, n, gen, layout)
                dev = ctx.buf[: n * F.WORDS].cpu().numpy().view(np.uint32).reshape(n, F.WORDS)
                fill_ok = bool(np.array_equal(dev, ref))
                case = {"n": n, "gen": gen, "layout": layout, "fill_ok": fill_ok,
                        "first_mismatch": None if fill_ok else int(np.flatnonzero(dev.reshape(-1) != ref.reshape(-1))[0])}
                g, z, ps = F.digest_key(seed, run)
                for order in F.ORDERS:
                    idx = ctx.order_idx(order, n, ps)
                    want = F.reference_digest(ref, idx.cpu().numpy(), g, z)
                    got = int(ctx.digest(n, idx, g, z).item())
                    case[f"digest_{order}_ok"] = got == want
                cases.append(case)
    # a chunked fill (`block_base`) continues the stream: blocks 3..8 of a seq fill, written as their own 5-block fill
    key = F.fill_key(seed, 999)
    chunk_ok = {}
    for gen in F.GENERATORS:
        ctx.set_key(key)
        F.device_fill(ctx.buf, ctx.key_dev, 5, gen, "seq", block_base=3)
        dev = ctx.buf[: 5 * F.WORDS].cpu().numpy().view(np.uint32)
        chunk_ok[gen] = bool(np.array_equal(dev, F.logical_words(gen, key, 5, block_base=3)))
    ok = all(c["fill_ok"] and c["digest_seq_ok"] and c["digest_perm_ok"] for c in cases) and all(chunk_ok.values())
    write(args.out, "validate.json", {"kind": "validate", "meta": meta, "device": args.device, "cases": cases,
                                      "chunked_fill_ok": chunk_ok, "all_ok": ok})
    print(f"validate: {'OK' if ok else 'FAILED'} ({len(cases)} cases, device {args.device})")
    if not ok:
        raise SystemExit("VALIDATION FAILED")


# --------------------------------------------------------------------------- sweep


def parse_sizes(spec: str, n_max: int) -> list[int]:
    out = []
    for s in spec.split(","):
        n = n_max if s == "max" else min(n_max, int(float(s) * GIB) // F.BLOCK_BYTES)
        if n > 0 and n not in out:
            out.append(n)
    return out


def summarize(rows: list[dict], nbytes: int) -> dict:
    s = {k: dist([r[k] for r in rows]) for k in ("fill_ms", "digest_ms", "gpu_ms", "wall_ms")}
    s["host_ms"] = dist([r["wall_ms"] - r["gpu_ms"] for r in rows])
    for k in ("fill_ms", "digest_ms", "wall_ms"):
        if s[k]["n"]:
            s[k.replace("_ms", "_gbps_at_p50")] = nbytes / (s[k]["p50"] / 1e3) / 1e9
    return s


def cmd_sweep(args, meta):
    torch = _torch("cuda")
    meta = {**meta, "gpu": gpu_meta(torch)}
    n_max, free, total = max_blocks(torch, args.reserve_gib)
    ctx = Ctx(torch, n_max)
    out = load_existing(args.out, "sweep.json", {"kind": "sweep", "points": []})
    out.update(meta=meta, hbm={"free_bytes_before": free, "total_bytes": total, "n_max_blocks": n_max,
                               "max_fill_bytes": n_max * F.BLOCK_BYTES, "max_fill_fraction_of_total": n_max * F.BLOCK_BYTES / total,
                               "free_bytes_after_alloc": torch.cuda.mem_get_info()[0]},
               co_tenants_start=co_tenants())
    done = {(p["n_blocks"], p["gen"], p["layout"], p["order"]) for p in out["points"]}
    run = args.seed_run
    for n in parse_sizes(args.sizes, n_max):
        for gen in args.gens.split(","):
            for layout in args.layouts.split(","):
                for order in args.orders.split(","):
                    run += 1000
                    if (n, gen, layout, order) in done or STOP.is_set():
                        continue
                    trials = args.trials_slow if layout == "wordstride" else args.trials
                    for w in range(args.warmup):
                        ctx.run(gen, layout, order, n, args.seed, run + 900 + w)
                    rows = [ctx.run(gen, layout, order, n, args.seed, run + t) for t in range(trials)]
                    # the same keys again without refilling: the digest is a function of the buffer and the key alone
                    g, z, ps = F.digest_key(args.seed, run + trials - 1)
                    again = int(ctx.digest(n, ctx.order_idx(order, n, ps), g, z).item())
                    nbytes = n * F.BLOCK_BYTES
                    out["points"].append({
                        "n_blocks": n, "bytes": nbytes, "gib": nbytes / GIB, "gen": gen, "layout": layout, "order": order,
                        "trials": trials, "repeat_digest_equal": again == rows[-1]["digest"],
                        "distinct_digests": len({r["digest"] for r in rows}), "summary": summarize(rows, nbytes),
                        "runs": {k: [round(r[k], 4) for r in rows] for k in ("fill_ms", "digest_ms", "gpu_ms", "wall_ms")},
                        "pressure": pressure()})
                    write(args.out, "sweep.json", out)
                    s = out["points"][-1]["summary"]
                    print(f"sweep {nbytes / GIB:.2f}GiB {gen}/{layout}/{order}: fill p50 {s['fill_ms']['p50']:.2f} ms "
                          f"({s['fill_gbps_at_p50']:.0f} GB/s) digest {s['digest_ms']['p50']:.2f} ms wall p50 "
                          f"{s['wall_ms']['p50']:.2f} p999 {s['wall_ms']['p999']:.2f} ms", flush=True)
    out["co_tenants_end"] = co_tenants()
    want = {(n, g, l, o) for n in parse_sizes(args.sizes, n_max) for g in args.gens.split(",")
            for l in args.layouts.split(",") for o in args.orders.split(",")}
    out["complete"] = want <= {(p["n_blocks"], p["gen"], p["layout"], p["order"]) for p in out["points"]}
    write(args.out, "sweep.json", out)


# --------------------------------------------------------------------------- host DRAM

_MM = None  # the shared anonymous mapping the forked workers fill


def _addr(off: int) -> int:
    import ctypes

    return ctypes.addressof(ctypes.c_char.from_buffer(_MM, off))


def _host_region(gen: str, key: bytes, off: int, ln: int) -> int:
    """Fill `[off, off + ln)` of the shared buffer (in a forked worker). The keystreams are one stream over the whole
    buffer whatever the split: each region starts its counter at its own offset."""
    import ctypes

    if gen == "memset":
        ctypes.memset(_addr(off), 0xA5, ln)
        return ln
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if gen == "aesctr":
        iv = ((int.from_bytes(key[16:32], "big") + off // 16) % (1 << 128)).to_bytes(16, "big")
        enc = Cipher(algorithms.AES(key[:16]), modes.CTR(iv)).encryptor()
    elif gen == "chacha20":
        nonce = (off // 64).to_bytes(4, "little") + key[32:44]
        enc = Cipher(algorithms.ChaCha20(key[:32], nonce), mode=None).encryptor()
    else:
        raise ValueError(gen)
    sub = MIB
    zeros = memoryview(bytes(sub))
    mv = memoryview(_MM)
    o, end = off, off + ln
    while o < end:
        m = min(sub, end - o)
        if o + m + 64 <= end:  # update_into wants room for a block past the data; stay inside this region
            enc.update_into(zeros[:m], mv[o : o + m + 64])
        else:
            mv[o : o + m] = enc.update(zeros[:m])
        o += m
    return ln


MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)  # Linux 5.14+


def _worker_init(barrier=None):
    """Pool initializer. Restores the default SIGTERM (the driver's handler only sets `STOP`, so a worker that kept it
    would outlive `Pool.terminate`). With a barrier it also maps the whole shared buffer into this worker's page tables
    (a forked child does not inherit them for a shared mapping), so a warm pass times the fill and not ~4 KiB minor
    faults."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if barrier is None:
        return
    try:
        _MM.madvise(MADV_POPULATE_WRITE)
    except OSError:
        pass
    barrier.wait()


def _parts(size: int, procs: int):
    regions = procs * 4
    step = size // regions // 4096 * 4096
    return [(i * step, step if i < regions - 1 else size - i * step) for i in range(regions)]


def cmd_host(args, meta):
    """Host-DRAM fill rate. `cold` is the first touch of a fresh mapping (page allocation and zeroing included), by the
    first `--procs` count; `warm` overwrites resident memory from workers whose page tables already map it, the state
    an honest prover answers from."""
    global _MM
    import multiprocessing as mp

    size = int(args.host_gib * GIB) // MIB * MIB
    out = {"kind": "host", "points": [], "populate": []}
    out.update(meta={**meta, "pressure_start": pressure()}, host_bytes=size)
    key = F._derive(args.seed, 0, "host") + F._derive(args.seed, 1, "host")
    gens = args.host_gens.split(",")
    cpu_gens = [g for g in gens if g != "d2h"]
    procs_list = [int(x) for x in args.procs.split(",")]
    ctx = mp.get_context("fork")

    def point(pool, gen, procs, phase, rep):
        t0 = time.perf_counter()
        # bounded: a worker that dies mid-task leaves `starmap` waiting forever
        pool.starmap_async(_host_region, [(gen, key, o, ln) for o, ln in _parts(size, procs)], chunksize=1).get(
            timeout=max(120.0, size / 0.2e9))
        s = time.perf_counter() - t0
        out["points"].append({"gen": gen, "procs": procs, "phase": phase, "rep": rep, "seconds": s,
                              "gbps": size / s / 1e9, "pressure": pressure()})
        write(args.out, "host.json", out)
        print(f"host {gen} procs={procs} {phase}: {size / s / 1e9:.1f} GB/s ({s:.2f} s for {size / GIB:.1f} GiB)",
              flush=True)

    if cpu_gens:
        _MM = mmap.mmap(-1, size + 4096, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
        with ctx.Pool(procs_list[0], initializer=_worker_init) as pool:
            point(pool, cpu_gens[0], procs_list[0], "cold", 0)
        for procs in procs_list:
            if STOP.is_set():
                break
            barrier = ctx.Barrier(procs + 1)
            t0 = time.perf_counter()
            try:
                with ctx.Pool(procs, initializer=_worker_init, initargs=(barrier,)) as pool:
                    barrier.wait(timeout=600)
                    out["populate"].append({"procs": procs, "seconds": time.perf_counter() - t0})
                    for gen in cpu_gens:
                        for rep in range(args.host_reps):
                            if STOP.is_set():
                                break
                            point(pool, gen, procs, "warm", rep)
            except Exception as e:  # recorded; the other worker counts still run
                out.setdefault("errors", []).append({"procs": procs, "error": repr(e)})
                write(args.out, "host.json", out)
                print(f"host procs={procs} failed: {e!r}", flush=True)
        _MM.close()
        _MM = None
    if "d2h" in gens and not STOP.is_set():
        try:
            out["d2h"] = host_d2h(args)
        except Exception as e:
            out.setdefault("errors", []).append({"d2h": True, "error": repr(e)})
        write(args.out, "host.json", out)
    out["complete"] = not STOP.is_set()
    write(args.out, "host.json", out)


def host_d2h(args) -> dict:
    """The GPU fills `--d2h-gib` of pinned host memory: ChaCha8 into two 1 GiB device stages on two streams, each copied
    out over PCIe while the other is filled. The stream is the same one a sequential device fill writes (`block_base`)."""
    torch = _torch("cuda")
    size = int(args.d2h_gib * GIB) // GIB * GIB
    stage_blocks = GIB // F.BLOCK_BYTES
    t0 = time.perf_counter()
    pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    pin_s = time.perf_counter() - t0
    stages = [torch.empty(stage_blocks * F.WORDS, dtype=torch.int32, device="cuda") for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    key = torch.tensor(F.signed_words(F.fill_key(args.seed, 7)), dtype=torch.int32, device="cuda")
    res = {"bytes": size, "pin_alloc_s": pin_s, "reps": []}

    def once():
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(size // GIB):
            s, st = streams[i % 2], stages[i % 2]
            with torch.cuda.stream(s):
                F.device_fill(st, key, stage_blocks, "chacha8", "seq", block_base=i * stage_blocks)
                pinned[i * GIB : (i + 1) * GIB].copy_(st.view(torch.uint8), non_blocking=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    once()
    for _ in range(args.host_reps):
        s = once()
        res["reps"].append({"seconds": s, "gbps": size / s / 1e9})
        print(f"host d2h: {size / s / 1e9:.1f} GB/s ({s:.2f} s for {size / GIB:.0f} GiB)", flush=True)
    check = pinned[:64].clone()
    want = F.logical_words("chacha8", F.fill_key(args.seed, 7), 1)[:16].view(np.uint8)
    res["first_bytes_ok"] = bool(np.array_equal(check.numpy(), want))
    del stages
    try:
        res["uva_digest"] = host_uva_digest(torch, args, pinned)
    except Exception as e:  # UVA host access may be unsupported; record it rather than lose the fill numbers
        res["uva_digest"] = {"error": repr(e)}
    del pinned
    torch.cuda.empty_cache()
    return res


def host_uva_digest(torch, args, pinned) -> dict:
    """The digest of the pinned host buffer read by the GPU over PCIe (UVA zero-copy, no staging), in sequential order:
    the host-DRAM half of an erasure answer when the GPU computes it. Checked against the same blocks copied to the device."""
    n = pinned.numel() // F.BLOCK_BYTES
    host2d = pinned.view(torch.int32).view(n, F.WORDS)
    ar = torch.arange(n, dtype=torch.int64, device="cuda")
    hbuf = torch.empty(n, dtype=torch.int64, device="cuda")
    zbuf = torch.empty(n, dtype=torch.int64, device="cuda")
    cbuf = torch.empty(F.D.INTERLEAVE, dtype=torch.int64, device="cuda")
    g, z, _ = F.digest_key(args.seed, 8)
    k = min(64, n)
    uva = int(F.device_digest(torch, host2d[:k], ar[:k], g, z, hbuf, zbuf, cbuf).item())
    dev = int(F.device_digest(torch, host2d[:k].cuda(), ar[:k], g, z, hbuf, zbuf, cbuf).item())
    res = {"bytes": n * F.BLOCK_BYTES, "check_ok": uva == dev, "reps": []}
    for _ in range(args.host_reps + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        int(F.device_digest(torch, host2d, ar, g, z, hbuf, zbuf, cbuf).item())
        s = time.perf_counter() - t
        res["reps"].append({"seconds": s, "gbps": n * F.BLOCK_BYTES / s / 1e9})
    res["reps"] = res["reps"][1:]
    print(f"host uva digest: {res['reps'][-1]['gbps']:.1f} GB/s, check {'OK' if res['check_ok'] else 'FAILED'}", flush=True)
    return res


# --------------------------------------------------------------------------- soak

STALL_EDGES_MS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000)


def cmd_probe(args, meta):
    """A host-stall probe (its own process, so the soak's GIL never shows as a stall): sleep 1 ms, record how late it
    woke. Writes `--name` on SIGTERM or at `--stop-epoch`."""
    signal.signal(signal.SIGTERM, _on_term)
    hist = [0] * (len(STALL_EDGES_MS) + 1)
    events, n, worst = [], 0, 0.0
    t_start = time.time()
    last_flush = time.perf_counter()
    prev = time.perf_counter()

    def flush():
        write(args.out, args.name, {"kind": "probe", "meta": meta, "t_start": t_start, "t_end": time.time(),
                                    "requested_ms": 1.0, "samples": n, "edges_ms": STALL_EDGES_MS, "hist": hist,
                                    "max_ms": worst, "events_over_2ms": events})

    while not STOP.is_set() and time.time() < args.stop_epoch:
        time.sleep(0.001)
        now = time.perf_counter()
        late = (now - prev) * 1e3 - 1.0
        prev = now
        n += 1
        k = int(np.searchsorted(STALL_EDGES_MS, late, side="right"))
        hist[k] += 1
        worst = max(worst, late)
        if late > 2.0 and len(events) < 20000:
            events.append([round(time.time() - t_start, 3), round(late, 3)])
        if now - last_flush > 30:
            flush()
            last_flush = now
    flush()


def cmd_soak(args, meta):
    torch = _torch("cuda")
    meta = {**meta, "gpu": gpu_meta(torch)}
    t_start = time.time()
    stop_at = min(t_start + args.minutes * 60, args.stop_epoch)
    name = f"soak-{args.tag}.json"
    probe = subprocess.Popen([sys.executable, __file__, "probe", "--out", args.out, "--name", f"probe-{args.tag}.json",
                              "--stop-epoch", str(stop_at + 5)])
    n_max, free, total = max_blocks(torch, args.reserve_gib)
    ctx = Ctx(torch, n_max)
    nbytes = n_max * F.BLOCK_BYTES
    orders = args.orders.split(",")
    for o in orders:
        for w in range(args.warmup):
            ctx.run(args.gen, args.layout, o, n_max, args.seed, args.seed_run + 900_000 + w)
    cols = {k: [] for k in ("t", "order", "idle", "fill_ms", "digest_ms", "gpu_ms", "wall_ms")}
    doc = {"kind": "soak", "meta": meta, "tag": args.tag, "gen": args.gen, "layout": args.layout, "orders": orders,
           "n_blocks": n_max, "bytes": nbytes, "hbm": {"free_bytes_before": free, "total_bytes": total},
           "t_start": t_start, "idle_every": args.idle_every, "idle_s": args.idle_s, "co_tenants_start": co_tenants(),
           "pressure": [], "cols": cols}
    t0p = time.perf_counter()
    doc["t0_epoch"] = time.time()  # cols["t"] is seconds after this, so runs line up with the probe's events
    last_flush = last_psi = time.perf_counter()
    i = 0
    while not STOP.is_set() and time.time() < stop_at:
        idle = args.idle_every > 0 and i > 0 and i % args.idle_every == 0
        if idle:
            time.sleep(args.idle_s)
        order = orders[i % len(orders)]
        r = ctx.run(args.gen, args.layout, order, n_max, args.seed, args.seed_run + i)
        cols["t"].append(round(r["t0"] - t0p, 4))
        cols["order"].append(orders.index(order))
        cols["idle"].append(int(idle))
        for k in ("fill_ms", "digest_ms", "gpu_ms", "wall_ms"):
            cols[k].append(round(r[k], 4))
        i += 1
        now = time.perf_counter()
        if now - last_psi > 10:
            doc["pressure"].append([round(now - t0p, 1), pressure()])
            last_psi = now
        if now - last_flush > 60:
            doc.update(runs=i, t_end=time.time())
            write(args.out, name, doc)
            last_flush = now
    doc.update(runs=i, t_end=time.time(), co_tenants_end=co_tenants(), stopped_by_signal=STOP.is_set())
    busy = [w for w, idl in zip(cols["wall_ms"], cols["idle"]) if not idl]
    doc["summary"] = {"busy": dist(busy), "after_idle": dist([w for w, idl in zip(cols["wall_ms"], cols["idle"]) if idl]),
                      "fill_ms": dist(cols["fill_ms"]), "digest_ms": dist(cols["digest_ms"])}
    write(args.out, name, doc)
    probe.send_signal(signal.SIGTERM)
    try:
        probe.wait(timeout=30)
    except subprocess.TimeoutExpired:
        probe.kill()
    print(f"soak {args.tag}: {i} runs, busy wall p50 {doc['summary']['busy'].get('p50', float('nan')):.2f} "
          f"p999 {doc['summary']['busy'].get('p999', float('nan')):.2f} max "
          f"{doc['summary']['busy'].get('max', float('nan')):.2f} ms", flush=True)


# --------------------------------------------------------------------------- result.json


def cmd_result(args, meta):
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir:
        print("result: no RESEARCH_RUN_DIR; nothing written")
        return
    out = Path(args.out)
    files = sorted(p for p in out.glob("*.json"))
    meas, status, detail = [], "not_run", []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if d.get("kind") == "validate":
            status = "passed" if d.get("all_ok") else "failed"
        if d.get("kind") == "sweep":
            meas.append({"name": "sweep_points", "value": len(d["points"]), "unit": "count"})
            meas.append({"name": "sweep_runs", "value": sum(x["trials"] for x in d["points"]), "unit": "count"})
        if d.get("kind") == "soak":
            meas.append({"name": f"soak_runs_{d['tag']}", "value": d.get("runs", 0), "unit": "count"})
            b = d.get("summary", {}).get("busy", {})
            if b.get("n"):
                meas.append({"name": f"soak_wall_p999_{d['tag']}", "value": b["p999"], "unit": "ms"})
        if d.get("kind") == "host":
            detail.append(f"host points {len(d.get('points', []))}")
    rel = [str(p.relative_to(run_dir)) if str(p).startswith(run_dir) else str(p) for p in files]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID"),
           "workload_fingerprint": {"harness": "benchmarks/pous/erase_calib", "argv": sys.argv,
                                    "source_sha": os.environ.get("RESEARCH_SOURCE_SHA")},
           "validation": {"status": status, "checker": "erase_calib.driver validate", "detail": "; ".join(detail)},
           "measurements": meas, "measurement_files": rel}
    tmp = Path(run_dir) / ".result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")
    print(f"result: {status}, {len(rel)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "sweep", "host", "soak", "probe", "result"])
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--seed-run", type=int, default=0, help="offset of the run counter (distinct keys across jobs)")
    ap.add_argument("--reserve-gib", type=float, default=1.0)
    ap.add_argument("--sizes", default="1,4,16,48,max")
    ap.add_argument("--gens", default="mix,chacha8,chacha12,chacha20")
    ap.add_argument("--layouts", default="seq,blockstride,wordstride")
    ap.add_argument("--orders", default="seq,perm")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trials-slow", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--host-gib", type=float, default=32)
    ap.add_argument("--d2h-gib", type=float, default=16)
    ap.add_argument("--procs", default="1,4,8,16")
    ap.add_argument("--host-gens", default="memset,aesctr,chacha20,d2h")
    ap.add_argument("--host-reps", type=int, default=3)
    ap.add_argument("--gen", default="chacha8")
    ap.add_argument("--layout", default="seq")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--stop-epoch", type=float, default=float("inf"))
    ap.add_argument("--idle-every", type=int, default=50)
    ap.add_argument("--idle-s", type=float, default=1.5)
    ap.add_argument("--tag", default="0")
    ap.add_argument("--name", default="probe.json")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    meta = node_meta()
    {"validate": cmd_validate, "sweep": cmd_sweep, "host": cmd_host, "soak": cmd_soak, "probe": cmd_probe,
     "result": cmd_result}[args.cmd](args, meta)


if __name__ == "__main__":
    main()
