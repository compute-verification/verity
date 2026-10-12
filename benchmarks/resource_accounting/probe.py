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

nvidia-smi reads the clocks before and after each phase (pynvml is not in the harness's venv).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
from pathlib import Path

REPS = 10
SPINNERS = 4
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
    return out
