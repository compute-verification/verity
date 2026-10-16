"""Why the matmul chain ran 9–14% faster beside the PoUS harness in a process of its own (the ``separate`` round, during the
audit and after it) than inside it or alone in the sweep (runs r20261012-003617-f349 and r20261012-005734-9db8, at one SM
clock and one affinity mask): the shape's native chains at its smallest batch, every condition of a phase in turn rep by
rep, each item timed on the host (`chain.Native.item`), on the device by events, and from its start to its last launch on
the host, with the CPU its thread ended on and that CPU's frequency.

* ``alone``: on the legacy default stream, a stream of the normal priority, one of the highest, the normal one with a
  device synchronization after each item (the harness's `sync()` after each load), and the default and normal streams
  held (``+held``: the stream spins first, so every replay is queued before the device reaches it, and the device time is
  the device's alone);
* ``sweep``: the default stream, each chain item followed by its GEMM-only graph's, as the sweep orders them;
* ``near``, ``far``: the default and normal streams with this thread pinned to the GPU's NUMA node's CPUs, then to another
  node's (the reserved CPUs left out), its mask restored after;
* ``beside-idle``: as ``alone``, with a second process holding a CUDA context on the GPU and doing nothing;
* ``beside-copies``: with that process copying a 2,056-byte block to the host every 5 ms on a stream of the highest
  priority, as the PoUS responder answers;
* ``beside-spin``: with `SPINNERS` processes spinning on the CPU (no CUDA), as the harness's memory challenger and
  responder spin;
* ``after``: as ``alone`` once they have exited.

Run r20261012-023723-e009 found every ``beside`` phase fast and ``after`` fast for its first 3–4 reps only, at one SM and
memory clock; the ``course`` then times the native chains back to back on the default stream around one co-tenant at a
time (`COURSE`: a process that sleeps, `SPINNERS` that spin, one holding an idle CUDA context), `COURSE_S` seconds before it
starts, after it is ready and after it exits, so the onset and the offset of the fast state are seen and each kind of
co-tenant is seen alone.

nvidia-smi reads the clocks before and after each phase (pynvml is not in the harness's venv).

``decay`` (`python probe.py decay`, run.sh ``DECAY=1``) asks why joint v1's tag passes ran at 1.17–1.32 ms (the tag kernel)
after the chain's GEMMs and at 0.84 ms after a long idle (runs r20261012-051020-03ac, r20261012-035705-85d4): in one process,
on a store of its own in the tagger's memory, it times (d)'s pass from a scrubbed L2, then a raw read of the store and an
op on an L2-resident tensor, `DECAY_PRE` times before each burst of `DECAY` and at `DECAY_AT_MS` after its last kernel,
with nvidia-smi sampling the clocks beside it.  ``steps`` (run.sh ``DECAY=steps``) times `STEP_PASSES` passes after each
step of `STEPS`: before any torch work, after torch's operands, the L2 op and GEMMs, after an idle, and after setting the
context's L1/shared-memory split preference, each followed by GEMMs (whose time a launch it records), since a kernel with
no preference keeps the split the last kernel left (CU_FUNC_CACHE_PREFER_NONE).  Run r20261012-055226-416b found the pass
at 0.84 ms before any torch work, 1.1–1.4 after torch's kernels of any kind, idle or not, and 0.83 once the context
prefers L1, the GEMMs' time a launch unchanged (`joint.prefer_l1`).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPS = 10
SPINNERS = 4
#: seconds of chains before a co-tenant starts, after all of it is ready, and after it has exited
COURSE_S = (5.0, 6.0, 10.0)
#: the most seconds a co-tenant may take to be ready, or to exit once its stdin closes
COURSE_WAIT_S = 60.0
#: an effect smaller than this share of the chain's time before the co-tenant gets no onset or offset
COURSE_EFFECT = 0.02
#: (stream, a device synchronization after each item, held)
STREAMS = (("default", False, False), ("normal", False, False), ("highest", False, False), ("normal", True, False),
           ("default", False, True), ("normal", False, True))
SMI = "clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,clocks_throttle_reasons.active,pci.bus_id"
HOLD_HZ = 2.2e9
SECOND = r"""
import select, sys, torch
x = torch.zeros(2056, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
print("ready", flush=True)
if sys.argv[1] == "idle":
    sys.stdin.read()
else:
    s = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    h = torch.empty(2056, dtype=torch.uint8, pin_memory=True)
    while not select.select([sys.stdin], [], [], 0.005)[0]:
        with torch.cuda.stream(s):
            h.copy_(x, non_blocking=True)
        s.synchronize()
"""
SPIN = r"""
import select, sys
print("ready", flush=True)
n = 0
while True:
    n += 1
    if n % 100_000 == 0 and select.select([sys.stdin], [], [], 0)[0]:
        break
"""
SLEEP = r"""
import sys
print("ready", flush=True)
sys.stdin.read()
"""
#: the course's co-tenants in turn: (name, script, its arguments, how many)
COURSE = (("sleeper", SLEEP, [], 1), ("spinners", SPIN, [], SPINNERS), ("context", SECOND, ["idle"], 1))
#: the decay probe's bursts in turn, (kind, seconds): GEMMs at o (B = 128) and at 8192³, 1 GiB copies through DRAM, an op on
#: a 32 MB tensor (L2-resident), 8 GiB allocated and freed with no kernel, and nothing
DECAY = (("none", 2.0), ("gemm-o-bf16", 0.01), ("gemm-o-fp8", 0.01), ("gemm-o-bf16", 2.0), ("gemm-o-fp8", 2.0),
         ("gemm-big", 2.0), ("copy", 2.0), ("l2", 2.0), ("alloc", 0.0), ("none", 2.0))
#: ms after a burst's last kernel at which a timed point starts
DECAY_AT_MS = (0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 20000)
DECAY_PRE = 3
DECAY_TAG = "ro-blake3/v1"
DECAY_SMI = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,clocks_event_reasons.active"
#: the steps probe's plan: nothing yet, torch's operands, the L2 op, GEMMs, an idle, then the context's L1/shared split
#: preferences with GEMMs between
STEPS = ("fresh", "torch", "l2", "gemm-o-bf16", "idle-30", "ctx-l1", "gemm-o-bf16", "gemm-o-fp8", "gemm-big", "ctx-shared",
         "ctx-none", "gemm-o-fp8", "gemm-o-bf16")
STEP_PASSES = 5


def course_summary(series: list[list], marks: dict, after_s: float = COURSE_S[2]) -> dict:
    """A co-tenant's course (`course`), each item ``[t_s, arm, device_s]`` with ``t_s`` from its start, and `marks` the
    seconds from its start at which all of it was ``ready``, its stdin was closed (``close``) and all of it had
    ``exited``: per arm, the median device ms before it started, while it started, while it was ready, while it exited, and
    in each whole second after it exited; with an effect (ready against before) of at least `COURSE_EFFECT`, the onset (the
    first item from its start nearer the ready median than the before one) and the offset (the first item after it exited
    from which every later item is nearer the before median)."""
    ready, close, exited = (marks.get(k) for k in ("ready", "close", "exited"))
    out = {}
    for arm in dict.fromkeys(a for _, a, _ in series):
        xs = [(t, d) for t, a, d in series if a == arm]
        s = {"before_ms": _span_ms(xs, None, 0.0), "starting_ms": _span_ms(xs, 0.0, ready),
             "ready_ms": _span_ms(xs, ready, close), "closing_ms": _span_ms(xs, close, exited),
             "after_ms": [_span_ms(xs, exited + b, exited + b + 1) for b in range(int(after_s))] if exited is not None else []}
        slow, fast = s["before_ms"], s["ready_ms"]
        s["effect"] = (slow - fast) / slow if slow and fast else None
        s["onset_s"] = s["offset_s"] = None
        if s["effect"] is not None and abs(s["effect"]) >= COURSE_EFFECT:
            nearer = [abs(d * 1e3 - fast) < abs(d * 1e3 - slow) for _, d in xs]
            s["onset_s"] = next((t for (t, _), f in zip(xs, nearer) if t >= 0 and f), None)
            post = [(t, f) for (t, _), f in zip(xs, nearer) if exited is not None and t >= exited]
            last = max((i for i, (_, f) in enumerate(post) if f), default=-1)
            s["offset_s"] = post[last + 1][0] - exited if last + 1 < len(post) else None
        out[arm] = s
    return out


def _span_ms(xs: list[tuple[float, float]], lo: float | None, hi: float | None) -> float | None:
    """The median device ms of the items ``(t_s, device_s)`` with ``lo <= t_s < hi`` (None: unbounded)."""
    ds = [d for t, d in xs if (lo is None or t >= lo) and (hi is None or t < hi)]
    return statistics.median(ds) * 1e3 if ds else None


def course(natives: dict, chains: dict[str, int], script: str, args: list[str], count: int,
           spans: tuple[float, float, float] = COURSE_S) -> dict:
    """The native chains back to back on the default stream, arm by arm in turn, `spans[0]` seconds before `count` copies
    of `script` start, until all are ready, `spans[1]` seconds more, until all have exited once their stdin closes, and
    `spans[2]` seconds after: each item's time from their start and its device time, the marks and `course_summary`."""
    import torch
    before, during, after = spans
    arms, raw, marks, i = list(natives), [], {}, 0
    base = time.monotonic()
    now = lambda: time.monotonic() - base

    def one() -> None:
        nonlocal i
        arm = arms[i % len(arms)]
        i += 1
        t = now()
        with torch.cuda.stream(torch.cuda.default_stream()):
            raw.append([t, arm, natives[arm].item(chains[arm])["device_s"]])

    def until(done, limit: float) -> bool:
        while not done() and now() < limit:
            one()
        return done()

    until(lambda: False, before)
    marks["start"] = now()
    procs = [subprocess.Popen([sys.executable, "-c", script, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
             for _ in range(count)]
    ups: list[float] = []
    for p in procs:
        threading.Thread(target=lambda p=p: p.stdout.readline().strip() == "ready" and ups.append(now()), daemon=True).start()
    error = None
    try:
        if until(lambda: len(ups) == count, marks["start"] + COURSE_WAIT_S):
            marks["ready"] = max(ups)
            until(lambda: False, marks["ready"] + during)
        else:
            error = f"{count - len(ups)} of {count} never got ready in {COURSE_WAIT_S:.0f} s"
    finally:
        marks["close"] = now()
        for p in procs:
            p.stdin.close()
        downs: list[float] = []
        threading.Thread(target=lambda: [p.wait() for p in procs] and downs.append(now()), daemon=True).start()
        if until(lambda: bool(downs), marks["close"] + COURSE_WAIT_S):
            marks["exited"] = downs[0]
        else:
            for p in procs:
                p.kill()
            error = error or f"still running {COURSE_WAIT_S:.0f} s after its stdin closed: killed"
    if "exited" in marks:
        until(lambda: False, marks["exited"] + after)
    t0 = marks["start"]
    series = [[round(t - t0, 4), arm, d] for t, arm, d in raw]
    rel = {k: round(v - t0, 4) for k, v in marks.items() if k != "start"}
    return {"count": count, "marks": rel, "rc": [p.poll() for p in procs], "error": error, "series": series,
            "summary": course_summary(series, rel, after)}


class Sidecar:
    """An idle CUDA context in a process of its own (`SECOND` idle) while the ``with`` lasts.  The course of run
    r20261012-031720-786d found every native chain faster from the moment one is ready (3.5% BF16, 11.5% FP8 at o, B = 1)
    and still faster 10 s after it exits, at one SM and memory clock: an arm measured beside it runs at that speed."""

    def __enter__(self) -> dict:
        t0 = time.monotonic()
        self.proc = subprocess.Popen([sys.executable, "-c", SECOND, "idle"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True)
        if self.proc.stdout.readline().strip() != "ready":
            self.proc.kill()
            raise RuntimeError(f"the sidecar never got ready (rc {self.proc.wait()})")
        self.record = {"pid": self.proc.pid, "ready_s": round(time.monotonic() - t0, 3)}
        return self.record

    def __exit__(self, *exc) -> None:
        self.proc.stdin.close()
        try:
            self.record["rc"] = self.proc.wait(timeout=COURSE_WAIT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.record["rc"] = self.proc.wait()


def smi(uuid: str | None) -> dict:
    if not uuid:
        return {"error": "no GPU_LEASE_UUID"}
    try:
        out = subprocess.run(["nvidia-smi", f"--id={uuid}", f"--query-gpu={SMI}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}
    return dict(zip(SMI.split(","), (x.strip() for x in out.splitlines()[0].split(","))))


def cpu_now() -> int:
    """The CPU this thread last ran on (`/proc/thread-self/stat`'s 39th field)."""
    stat = Path("/proc/thread-self/stat").read_text()
    return int(stat[stat.rindex(")") + 2:].split()[36])


def khz(cpu: int) -> int | None:
    f = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    try:
        return int(f.read_text())
    except (OSError, ValueError):
        return None


def cpulist(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def numa(bus_id: str | None) -> dict:
    """The GPU's NUMA node (sysfs, by its PCI bus id) and every node's CPUs."""
    nodes = {int(d.name[4:]): cpulist((d / "cpulist").read_text())
             for d in Path("/sys/devices/system/node").glob("node[0-9]*")}
    gpu = None
    if bus_id:
        dev = Path("/sys/bus/pci/devices") / bus_id.lower().replace("00000000:", "0000:")
        if (dev / "numa_node").exists():
            gpu = int((dev / "numa_node").read_text())
    return {"gpu_node": gpu, "nodes": {k: sorted(v) for k, v in sorted(nodes.items())}}


def node_of(cpu: int, nodes: dict[int, list[int]]) -> int | None:
    return next((n for n, cpus in nodes.items() if cpu in cpus), None)


def summarize(items: list[list]) -> dict:
    """An item is ``[host_s, device_s, launch_s, held_s | None, cpu, kHz | None, chains]`` (the times a chain's): their
    medians in ms and MHz, and for held items whether every hold outlasted its item's launches."""
    med = lambda i: statistics.median(x[i] for x in items) * 1e3
    mhz = [x[5] / 1e3 for x in items if x[5]]
    out = {"n": len(items), "host_ms": med(0), "device_ms": med(1), "launch_ms": med(2),
           "cpu_mhz": statistics.median(mhz) if mhz else None}
    if items[0][3] is not None:
        out["queued"] = all(x[3] > x[2] * x[6] for x in items)
    return out


def run(natives: dict, chains: dict[str, int], uuid: str | None, reserved: set[int] = frozenset(), reps: int = REPS) -> dict:
    """Every phase over `natives` (arm -> `chain.Native` at the shape's smallest batch), `chains[arm]` chains an item."""
    import torch
    hi = torch.cuda.Stream.priority_range()[1]
    streams = {"default": torch.cuda.default_stream(), "normal": torch.cuda.Stream(), "highest": torch.cuda.Stream(priority=hi)}
    first = smi(uuid)
    topo = numa(first.get("pci.bus_id"))
    nodes = {int(k): v for k, v in topo["nodes"].items()}
    hold: dict[str, int] = {}
    for arm, nat in natives.items():
        with torch.cuda.stream(streams["normal"]):
            launch = nat.item(chains[arm])["launch_s"] * chains[arm]
        hold[arm] = int((3 * launch + 2e-3) * HOLD_HZ)
    out: dict = {"reps": reps, "chains": chains, "hold_cycles": hold, "numa": topo, "phases": {}}

    def phase(name: str, conds, which=("chain",)) -> None:
        rec = {"clocks_before": smi(uuid), "items": {}}
        turn = [(arm, c, w) for arm in natives for c in conds for w in which]
        for rep in range(reps):
            for arm, (s, sync, held), w in turn[rep % len(turn):] + turn[:rep % len(turn)]:
                with torch.cuda.stream(streams[s]):
                    r = natives[arm].item(chains[arm], w, hold[arm] if held else 0)
                if sync:
                    torch.cuda.synchronize()
                label = f"{arm}/{s}" + ("+sync" if sync else "") + ("+held" if held else "") + ("/gemm" if w == "gemm" else "")
                cpu = cpu_now()
                rec["items"].setdefault(label, []).append([r["seconds"], r["device_s"], r["launch_s"], r.get("held_s"), cpu,
                                                           khz(cpu), chains[arm]])
        rec["clocks_after"] = smi(uuid)
        rec["summary"] = {label: {**summarize(xs), "nodes": sorted({node_of(x[4], nodes) for x in xs}, key=str)}
                          for label, xs in rec["items"].items()}
        out["phases"][name] = rec

    def beside(name: str, script: str, args: list[str], count: int = 1) -> None:
        procs = [subprocess.Popen([sys.executable, "-c", script, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                 for _ in range(count)]
        try:
            if all(p.stdout.readline().strip() == "ready" for p in procs):
                phase(name, STREAMS[:4])
            else:
                out["phases"][name] = {"error": "a second process never got ready"}
        finally:
            for p in procs:
                p.stdin.close()
            out.setdefault("second_rc", {})[name] = [p.wait() for p in procs]

    phase("alone", STREAMS)
    phase("sweep", STREAMS[:1], ("chain", "gemm"))
    mask = os.sched_getaffinity(0)
    allowed, gpu_node = mask - set(reserved), topo["gpu_node"]
    try:
        for name, pick in (("near", lambda n: n == gpu_node), ("far", lambda n: n != gpu_node)):
            cpus = next((set(c) & allowed for n, c in nodes.items() if gpu_node is not None and pick(n) and set(c) & allowed),
                        None)
            if not cpus:
                out["phases"][name] = {"skipped": f"no CPU of this mask on a node {'at' if name == 'near' else 'off'} the GPU's"}
                continue
            os.sched_setaffinity(0, cpus)
            phase(name, STREAMS[:2])
            out["phases"][name]["cpus"] = [min(cpus), max(cpus), len(cpus)]
    finally:
        os.sched_setaffinity(0, mask)
    beside("beside-idle", SECOND, ["idle"])
    beside("beside-copies", SECOND, ["copies"])
    beside("beside-spin", SPIN, [], SPINNERS)
    phase("after", STREAMS)
    out["course"] = {}
    for name, script, args, count in COURSE:
        clocks = smi(uuid)
        try:
            out["course"][name] = {"clocks_before": clocks, **course(natives, chains, script, args, count),
                                   "clocks_after": smi(uuid)}
        except Exception as ex:  # noqa: BLE001 - one co-tenant's failure is recorded and the next runs
            out["course"][name] = {"clocks_before": clocks, "error": f"{type(ex).__name__}: {ex}"}
    return out


def decay_summary(conditions: list[dict]) -> list[dict]:
    """Per burst (`decay`'s conditions), the median of its points before it, and each point after it, in ms: the tag kernel,
    the raw read and the L2 op."""
    keys = ("tags_ms", "raw_ms", "l2_ms")
    return [{"kind": c["kind"], "seconds": c["seconds"],
             "before": {k: statistics.median(p[k] for p in c["pre"]) for k in keys},
             "after": [{"at_ms": p["at_ms"], **{k: p[k] for k in keys}} for p in c["post"]]} for c in conditions]


class _Words:
    """A device buffer as torch's int64 tensor (`torch.as_tensor`), through the CUDA array interface."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes // 8,), "typestr": "<i8", "data": (ptr, False), "version": 3}


class _Bench:
    """The probes' device side on the one visible GPU: a store of `blocks` rows in the tagger's memory (the bandwidth
    harness's at `lib`), its scrub buffer and `DECAY_TAG`'s workspace; torch's operands, made by `operands` (the process's
    first torch work on the device); (d)'s pass from a scrubbed L2 (`tag`), with the raw read and the L2 op (`point`);
    and the context's L1/shared-memory split preference (`cuCtxSetCacheConfig`)."""

    def __init__(self, lib: str | Path, blocks: int) -> None:
        import torch

        import joint as J
        self.torch, self.J, self.G = torch, J, J.load_gpu()
        self.t = t = self.G.Tagger(lib)
        self.blocks, self.nbytes = blocks, blocks * self.G.BLOCK_BYTES
        self.store, self.scrub = t.alloc(self.nbytes), t.alloc(self.G.SCRUB_BYTES)
        t.fill_test_store(self.store, blocks)
        t.memset(self.scrub, 0, self.G.SCRUB_BYTES)
        self.ws = t.alloc(t.workspace_bytes(DECAY_TAG, blocks))
        self.ops: dict = {}

    def operands(self) -> None:
        torch = self.torch
        self.words = torch.as_tensor(_Words(self.store, self.nbytes), device="cuda")
        a = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        a8, w8, one = a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn), torch.ones((), device="cuda")
        big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        dst, self.l2 = torch.empty_like(src), torch.ones(8 << 20, device="cuda")
        self.ops = {"gemm-o-bf16": (64, lambda: torch.matmul(a, w.t())),
                    "gemm-o-fp8": (64, lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)),
                    "gemm-big": (2, lambda: torch.matmul(big, big)), "copy": (4, lambda: dst.copy_(src)),
                    "l2": (64, lambda: self.l2.mul_(1.0))}
        torch.cuda.synchronize()

    def timed(self, f) -> float:
        torch = self.torch
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        f()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1)

    def tag(self) -> dict:
        self.t.read_scrub(self.scrub, self.G.SCRUB_BYTES)
        self.t.tag_pass(DECAY_TAG, self.store, self.blocks, self.J.THROWAWAY, self.ws)
        tags_ms, pass_ms = self.t.last_pass_ms()
        return {"tags_ms": tags_ms, "pass_ms": pass_ms}

    def point(self, t0: float) -> dict:
        s = time.monotonic()
        out = {"t_s": round(s - t0, 6), "unix_ns": time.time_ns(), **self.tag()}
        self.t.read_scrub(self.scrub, self.G.SCRUB_BYTES)
        out["raw_ms"] = self.timed(lambda: self.torch.sum(self.words))
        out["l2_ms"] = self.timed(lambda: [self.l2.mul_(1.0) for _ in range(10)])
        return out

    def burst(self, kind: str, seconds: float) -> int:
        """`kind` for `seconds` (at least one batch): its launches."""
        torch, n = self.torch, 0
        if kind == "alloc":
            x = torch.empty(8 << 30, dtype=torch.uint8, device="cuda")
            del x
            torch.cuda.empty_cache()
        elif kind == "none":
            time.sleep(seconds)
        else:
            each, op = self.ops[kind]
            end = time.monotonic() + seconds
            while n == 0 or time.monotonic() < end:
                for _ in range(each):
                    op()
                n += each
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        return n

    def batch_ms(self, kind: str) -> float:
        """One batch of `kind` on the device, ms a launch."""
        each, op = self.ops[kind]
        return self.timed(lambda: [op() for _ in range(each)]) / each

    def prefer(self, which: str) -> int:
        """`ctx-none|ctx-shared|ctx-l1`: the context's preference (CU_FUNC_CACHE_PREFER_*), CUDA's return code."""
        import ctypes
        return ctypes.CDLL("libcuda.so.1").cuCtxSetCacheConfig({"ctx-none": 0, "ctx-shared": 1, "ctx-l1": 2}[which])

    def close(self) -> None:
        for p in (self.ws, self.scrub, self.store):
            self.t.free(p)


def _smi_proc(smi_csv: Path | None, uuid: str | None):
    if smi_csv is None or not uuid:
        return None
    return subprocess.Popen(["nvidia-smi", "-i", uuid, "--format=csv", "-lms", "20", f"--query-gpu={DECAY_SMI}"],
                            stdout=smi_csv.open("w"), stderr=subprocess.STDOUT)


def decay(lib: str | Path, blocks: int, smi_csv: Path | None, uuid: str | None, bursts=DECAY, at_ms=DECAY_AT_MS) -> dict:
    """The decay probe (module docstring) through the bandwidth harness's tagger at `lib`."""
    b = _Bench(lib, blocks)
    b.operands()
    for _, op in b.ops.values():
        op()
    b.tag()
    smi_proc, conditions = _smi_proc(smi_csv, uuid), []
    try:
        time.sleep(at_ms[-1] / 1e3)
        for kind, seconds in bursts:
            pre = []
            for _ in range(DECAY_PRE):
                pre.append(b.point(time.monotonic()))
                time.sleep(0.1)
            b0 = time.monotonic()
            n = b.burst(kind, seconds)
            end = time.monotonic()
            post = []
            for ms in at_ms:
                wait = end + ms / 1e3 - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                post.append({"at_ms": ms, **b.point(end)})
            conditions.append({"kind": kind, "seconds": seconds, "launches": n, "burst_s": round(end - b0, 6), "pre": pre,
                               "post": post})
            after = " ".join(f"{p['tags_ms']:.3f}" for p in post)
            print(f"probe decay: {kind} {seconds} s ({n} launches) before {statistics.median(p['tags_ms'] for p in pre):.3f}"
                  f" after {after}", flush=True)
    finally:
        if smi_proc is not None:
            smi_proc.terminate()
            smi_proc.wait()
        b.close()
    return {"tag": DECAY_TAG, "blocks": blocks, "bytes": b.nbytes, "scrub_bytes": b.G.SCRUB_BYTES, "at_ms": list(at_ms),
            "device": b.G.uuid_text(b.t.device()), "smi": smi(uuid), "conditions": conditions,
            "summary": decay_summary(conditions)}


def steps(lib: str | Path, blocks: int, smi_csv: Path | None, uuid: str | None, plan=STEPS, passes: int = STEP_PASSES) -> dict:
    """The steps probe (module docstring) through the bandwidth harness's tagger at `lib`: each step of `plan`, then
    `passes` of (d)'s passes."""
    b = _Bench(lib, blocks)
    smi_proc, out = _smi_proc(smi_csv, uuid), []
    try:
        for kind in plan:
            rec: dict = {"step": kind}
            try:
                if kind == "torch":
                    b.operands()
                elif kind.startswith("idle-"):
                    time.sleep(float(kind[5:]))
                elif kind.startswith("ctx-"):
                    rec["rc"] = b.prefer(kind)
                elif kind != "fresh":
                    b.burst(kind, 0.0)
                    rec["launch_ms"] = b.batch_ms(kind)
            except Exception as ex:  # noqa: BLE001 - recorded; the passes still run
                rec["error"] = f"{type(ex).__name__}: {ex}"
            rec["passes"] = [b.tag() for _ in range(passes)]
            rec["tags_ms"] = statistics.median(p["tags_ms"] for p in rec["passes"])
            out.append(rec)
            tags = " ".join(f"{p['tags_ms']:.3f}" for p in rec["passes"])
            print(f"probe steps: {kind} rc {rec.get('rc', '-')} launch {rec.get('launch_ms', 0):.4f} ms, tags {tags}",
                  flush=True)
    finally:
        if smi_proc is not None:
            smi_proc.terminate()
            smi_proc.wait()
        b.close()
    return {"tag": DECAY_TAG, "blocks": blocks, "bytes": b.nbytes, "device": b.G.uuid_text(b.t.device()), "smi": smi(uuid),
            "steps": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("decay", "the tag pass before and after each burst (module docstring)"),
                       ("steps", "the tag pass after each step, the context's L1/shared split preferences among them")):
        d = sub.add_parser(name, help=text)
        d.add_argument("--lib", type=Path, required=True, help="the bandwidth tagger's libbwtag.so")
        d.add_argument("--blocks", type=int, default=1 << 19, help="the store's rows of 2,056 bytes (default E's, 2^19)")
        d.add_argument("--smi-csv", type=Path, help="nvidia-smi's samples every 20 ms through the probe (GPU_LEASE_UUID's)")
        d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    result = (decay if a.cmd == "decay" else steps)(a.lib, a.blocks, a.smi_csv, os.environ.get("GPU_LEASE_UUID"))
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
