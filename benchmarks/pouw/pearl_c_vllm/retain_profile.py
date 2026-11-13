"""Where a retained Pearl-C serve's time goes (`served_passes.py serve`, every pass retained), and what the node's copy and
write paths give.

  retain_profile.py probe --scratch DIR --out probe.json [--gb 8]
  retain_profile.py serve --profile PROFILE.json [--seed S] [--wait-ends] SERVE_ARGS...

``probe`` measures the node: 1 GiB device-to-host copies into pinned and pageable memory and a device-to-device copy, fresh
pinned allocations of 1 and 8 GiB, page-cache writes of ``--gb`` GiB from pinned memory in 256 MiB files by 1, 4, 8 and 12
threads (each then flushed by ``sync``: the disk's rate), one file of ``--gb`` GiB written and fdatasync'ed, and the
kernel's dirty-page limits.

``serve`` runs `served_passes.main` with ``serve SERVE_ARGS`` and timers around the retention path's parts (`PARTS`, each
wrapped where its module holds it, skipped where it is absent): per pass, each part's calls, host seconds and bytes, the y
trees' seconds on the GPU (CUDA events around `served_y.y_tree`, read at the run's end), the writer threads' seconds, and
/proc/meminfo's Dirty and Writeback at each pass end, with the ending thread's and the process's CPU seconds and the wall
clock there (``main_cpu_s``, ``process_cpu_s``, ``wall_s``: a step's main-thread CPU against its wall time).  It writes the parts by pass, with served.json's ``step_s`` beside
them, to ``--profile``.  ``--seed S`` draws the epoch's coins and the hm96 salt key from ``random.Random(S)`` (two serves of
one load then retain the same bytes, so two trees' retention paths compare file by file); ``--wait-ends`` completes each
step's retained pass in its step (`PassEnds.retain_behind` off).

  retain_profile.py digests DIR --out DIGESTS.json

``digests`` writes each file under DIR (relative path) and its bytes' SHA-256.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GiB = 1 << 30
#: (module, attribute) of each part timed by `serve`, by its name in the profile; a dotted attribute is a class's method
PARTS = {
    "gemm": ("verity_vllm.protocols.pouw.pearl_c", "PearlCDeviceRun._gemm"),
    "after": ("verity_vllm.protocols.pouw.pearl_c", "PearlCDeviceRun._after"),
    "begin": ("verity_vllm.protocols.pouw.pearl_c", "PearlCDeviceRun.begin"),
    "poison": ("verity_vllm.protocols.pouw.pearl_c", "PearlCDeviceRun.poison"),
    "join_sides": ("verity_vllm.protocols.pouw.pearl_c", "PearlCDeviceRun.join_sides"),
    "retained_call": ("verity_vllm.protocols.pouw.pearl_c.ends", "PassEnds._retained"),
    "excluded": ("verity_vllm.protocols.pouw.pearl_c.ends", "excluded"),
    "excluded_host": ("verity_vllm.protocols.pouw.pearl_c.ends", "excluded_host"),
    "y_tree": ("verity_vllm.protocols.pouw.pearl_c.served_y", "y_tree"),
    "write_behind": ("verity_vllm.protocols.pouw.pearl_c.ends", "write_behind"),
    "copies": ("verity_vllm.protocols.pouw.pearl_c.host", "_Copies.start"),
    "arena_take": ("verity_vllm.protocols.pouw.pearl_c.host", "_Arena.take"),
    "stream_sync": ("torch.cuda.streams", "Stream.synchronize"),
    "event_sync": ("torch.cuda.streams", "Event.synchronize"),
    "landed": ("verity_vllm.protocols.pouw.pearl_c.host", "_landed"),
    "writer": ("verity_vllm.protocols.pouw.pearl_c.host", "_write_host"),
    "join": ("verity_vllm.protocols.pouw.pearl_c.ends", "joined"),
    "write_retained": ("verity_vllm.protocols.pouw.pearl_c.ends", "PassEnds._write_retained"),
    "weights_files": ("verity_vllm.protocols.pouw.pearl_c.ends", "_linked"),
    "pass_files": ("verity_vllm.protocols.pouw.pearl_c.ends", "_write"),
    "manifest": ("verity_vllm.protocols.pouw.pearl_c.ends", "write_manifest"),
    "readback": ("verity_vllm.protocols.pouw.pearl_c.ends", "_bytes"),
    "end": ("verity_vllm.protocols.pouw.pearl_c.ends", "PassEnds.end"),
    "retained_ends": ("verity_vllm.protocols.pouw.pearl_c.ends", "PassEnds.retained_ends"),
    "finish": ("verity_vllm.protocols.pouw.pearl_c.ends", "PassEnds.finish"),
}


def _meminfo() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("Dirty", "Writeback", "MemAvailable", "Cached"):
            out[k] = int(v.split()[0]) << 10
    return out


def _vm() -> dict[str, str]:
    return {k: Path(f"/proc/sys/vm/{k}").read_text().strip()
            for k in ("dirty_ratio", "dirty_background_ratio", "dirty_bytes", "dirty_background_bytes", "dirty_expire_centisecs")}


def probe(args: argparse.Namespace) -> int:
    import torch
    scratch, out = Path(args.scratch), {"vm": _vm(), "meminfo": _meminfo(), "cpus": os.cpu_count()}
    dev = torch.device("cuda")
    src = torch.randint(0, 255, (GiB,), dtype=torch.uint8, device=dev)

    def timed(f, reps=5):
        best = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            f()
            torch.cuda.synchronize()
            best.append(time.perf_counter() - t)
        return min(best)

    t = time.perf_counter()
    pinned = torch.empty(GiB, dtype=torch.uint8, pin_memory=True)
    out["pin_alloc_1g_s"] = time.perf_counter() - t
    t = time.perf_counter()
    big = torch.empty(8 * GiB, dtype=torch.uint8, pin_memory=True)
    out["pin_alloc_8g_s"] = time.perf_counter() - t
    pageable = torch.empty(GiB, dtype=torch.uint8)
    pageable.fill_(1)
    out["d2h_pinned_gbs"] = GiB / timed(lambda: pinned.copy_(src, non_blocking=True)) / 1e9
    out["d2h_pageable_gbs"] = GiB / timed(lambda: pageable.copy_(src)) / 1e9
    dst = torch.empty_like(src)
    out["d2d_gbs"] = GiB / timed(lambda: dst.copy_(src)) / 1e9
    side = torch.cuda.Stream()
    view = big[:GiB]

    def overlapped():
        with torch.cuda.stream(side):
            view.copy_(src, non_blocking=True)
        pinned.copy_(dst, non_blocking=True)
    out["d2h_two_streams_gbs"] = 2 * GiB / timed(overlapped) / 1e9
    host = memoryview(big.numpy())
    files, chunk = (args.gb * GiB) // (256 << 20), 256 << 20
    writes = {}
    for threads in (1, 4, 8, 12):
        os.sync()
        d = scratch / f"w{threads}"
        d.mkdir(parents=True, exist_ok=True)

        def one(i, d=d):
            with open(d / f"{i}.bin", "wb") as f:
                f.write(host[(i % 32) * chunk:(i % 32 + 1) * chunk])
        before = _meminfo()
        t = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(one, range(files)))
        w = time.perf_counter() - t
        mid = _meminfo()
        t = time.perf_counter()
        os.sync()
        s = time.perf_counter() - t
        writes[threads] = {"page_cache_gbs": files * chunk / w / 1e9, "write_s": w, "sync_s": s,
                           "to_disk_gbs": files * chunk / (w + s) / 1e9, "dirty_before": before["Dirty"], "dirty_after": mid["Dirty"]}
        for p in d.iterdir():
            p.unlink()
        print(f"probe: {threads} threads: {writes[threads]}", flush=True)
    out["writes"] = writes
    os.sync()
    path = scratch / "fdatasync.bin"
    t = time.perf_counter()
    with open(path, "wb") as f:
        for i in range(files):
            f.write(host[(i % 32) * chunk:(i % 32 + 1) * chunk])
        os.fdatasync(f.fileno())
    out["fdatasync_gbs"] = files * chunk / (time.perf_counter() - t) / 1e9
    path.unlink()
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    return 0


class _Profile:
    """Each part's calls, host seconds and bytes, by pass (the pass ends seen so far when the part started)."""

    def __init__(self):
        self.lock, self.ends = threading.Lock(), 0
        self.parts: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0]))
        self.mem: dict[int, dict] = {}
        self.events: list[tuple[int, object, object]] = []

    def add(self, at: int, name: str, s: float, n: int = 0) -> None:
        with self.lock:
            p = self.parts[at][name]
            p[0], p[1], p[2] = p[0] + 1, p[1] + s, p[2] + n

    def wrap(self, name: str, f):
        @functools.wraps(f)
        def timed(*a, **kw):
            at, t = self.ends, time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                self.add(at, name, time.perf_counter() - t, _nbytes(name, a))
                if name == "end":
                    self.mem[at] = {**_meminfo(), "main_cpu_s": time.thread_time(), "process_cpu_s": time.process_time(),
                                    "wall_s": time.perf_counter()}
                    self.ends += 1
        return timed

    def wrap_y_tree(self, f):
        @functools.wraps(f)
        def timed(run, y, index, *rest):
            import torch
            at = self.ends
            if not y.is_cuda:
                return f(run, y, index, *rest)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            t = time.perf_counter()
            out = f(run, y, index, *rest)
            b.record()
            self.add(at, "y_tree", time.perf_counter() - t, y.numel() * y.element_size())
            self.events.append((at, a, b))
            return out
        return timed

    def doc(self) -> dict:
        import torch
        if self.events:
            torch.cuda.synchronize()
        for at, a, b in self.events:
            self.add(at, "y_tree_gpu", a.elapsed_time(b) / 1e3)
        return {str(at): {"parts": {k: {"calls": v[0], "s": v[1], "bytes": v[2]} for k, v in parts.items()},
                          "meminfo_at_end": self.mem.get(at)} for at, parts in sorted(self.parts.items())}


def _nbytes(name: str, a: tuple) -> int:
    if name == "writer" and len(a) > 1:
        return getattr(a[1], "nbytes", 0) or 0
    return 0


def _install(prof: _Profile) -> list[str]:
    got = []
    for name, (mod, attr) in PARTS.items():
        m = importlib.import_module(mod)
        owner, *rest = attr.split(".")
        if rest:
            cls = getattr(m, owner, None)
            if cls is None or not hasattr(cls, rest[0]):
                continue
            setattr(cls, rest[0], prof.wrap(name, getattr(cls, rest[0])))
        elif hasattr(m, owner):
            f = getattr(m, owner)
            setattr(m, owner, prof.wrap_y_tree(f) if name == "y_tree" else prof.wrap(name, f))
        else:
            continue
        got.append(name)
    return got


def serve(args: argparse.Namespace, rest: list[str]) -> int:
    sys.path.insert(0, str(HERE))
    import served_passes as SP

    if args.seed is not None:
        import random

        from verity.protocols.pouw.verifier import verifier
        from verity_vllm.protocols.pouw import pearl_c as PPD
        draw = random.Random(args.seed)
        verifier.coins = draw.randbytes
        PPD.new_salt_key = lambda scheme: draw.randbytes(32)
    if args.wait_ends:
        from verity_vllm.protocols.pouw import pearl_c as PPD
        PPD.PearlCDeviceRun.retain_behind = False
    prof = _Profile()
    parts = _install(prof)
    t = time.perf_counter()
    rc = SP.main(["serve", *rest])
    total = time.perf_counter() - t
    out = Path(rest[rest.index("--out") + 1])
    served = json.loads(out.read_text()) if out.exists() else {}
    doc = {"parts_timed": parts, "rc": rc, "process_s": total, "vm": _vm(), "seed": args.seed, "wait_ends": args.wait_ends,
           **{k: served.get(k) for k in ("retained", "generate_s", "step_s", "engine_steps", "requests", "prompt_tokens",
                                         "output_tokens", "close_s")},
           "passes_timed": [p["pass"] for p in served.get("passes", []) if p.get("timed")],
           "by_pass": prof.doc()}
    Path(args.profile).write_text(json.dumps(doc, indent=1) + "\n")
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--scratch", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--gb", type=int, default=8)
    sv = sub.add_parser("serve")
    sv.add_argument("--profile", required=True)
    sv.add_argument("--seed", type=int)
    sv.add_argument("--wait-ends", action="store_true")
    dg = sub.add_parser("digests")
    dg.add_argument("dir")
    dg.add_argument("--out", required=True)
    args, rest = p.parse_known_args(argv)
    if args.cmd == "digests":
        return digests(args)
    return probe(args) if args.cmd == "probe" else serve(args, rest)


def digests(args: argparse.Namespace) -> int:
    import hashlib
    root = Path(args.dir)

    def one(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(64 << 20):
                h.update(chunk)
        return h.hexdigest()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        got = dict(zip((str(p.relative_to(root)) for p in files), pool.map(one, files)))
    Path(args.out).write_text(json.dumps(got, indent=0, sort_keys=True) + "\n")
    print(f"digests: {len(got)} files under {root}, {hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
