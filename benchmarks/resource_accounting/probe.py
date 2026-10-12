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
"""

from __future__ import annotations

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


def course_summary(series: list[list], marks: dict, after_s: float = COURSE_S[2]) -> dict:
    """A co-tenant's course (`course`), each item ``[t_s, arm, device_s]`` with ``t_s`` from its start, and `marks` the
    seconds from its start at which all of it was ``ready``, its stdin was closed (``close``) and all of it had
    ``exited``: per arm, the median device ms before it started, while it started, while it was ready, while it exited, and
    in each whole second after it exited; with an effect (ready against before) of at least `COURSE_EFFECT`, the onset (the
    first item from its start nearer the ready median than the before one) and the offset (the first item after it exited
    from which every later item is nearer the before median)."""
    med = lambda xs: statistics.median(xs) * 1e3 if xs else None
    ready, close, exited = (marks.get(k) for k in ("ready", "close", "exited"))
    out = {}
    for arm in dict.fromkeys(a for _, a, _ in series):
        xs = [(t, d) for t, a, d in series if a == arm]
        span = lambda lo, hi: med([d for t, d in xs if (lo is None or t >= lo) and (hi is None or t < hi)])
        s = {"before_ms": span(None, 0.0), "starting_ms": span(0.0, ready), "ready_ms": span(ready, close),
             "closing_ms": span(close, exited),
             "after_ms": [span(exited + b, exited + b + 1) for b in range(int(after_s))] if exited is not None else []}
        slow, fast = s["before_ms"], s["ready_ms"]
        s["effect"] = (slow - fast) / slow if slow and fast else None
        s["onset_s"] = s["offset_s"] = None
        if s["effect"] is not None and abs(s["effect"]) >= COURSE_EFFECT:
            nearer_fast = lambda d: abs(d * 1e3 - fast) < abs(d * 1e3 - slow)
            s["onset_s"] = next((t for t, d in xs if t >= 0 and nearer_fast(d)), None)
            post = [(t, d) for t, d in xs if exited is not None and t >= exited]
            last = max((i for i, (_, d) in enumerate(post) if nearer_fast(d)), default=-1)
            s["offset_s"] = post[last + 1][0] - exited if last + 1 < len(post) else None
        out[arm] = s
    return out


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
