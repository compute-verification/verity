"""`verity-vllm sweep`: a night's config runs on one host, packed onto its GPUs by RAM and CPU (docs/vllm-config-sweep-plan.md §3).

    verity-vllm sweep plan SWEEP_JSON [--gpus N] [--ram-gb G] [--cpus C]          the packing, simulated on the estimates (nothing runs)
    verity-vllm sweep run  SWEEP_JSON --out DIR [--program-cache D] [--bootstrap] [--gpus N] [--ram-gb G] [--cpus C]

SWEEP_JSON is ``{"configs": [{"row": <row id>, "gpus"?, "ram_gb"?, "cpus"?, "minutes"?}, ...]}``.  A config's needs default by its class
(`estimate`, the plan's §2 table): a large batch (B >= 32) or a long context (input >= 4096) takes 500 GB, 32 CPUs and ~150 min; any
other 150 GB, 16 CPUs and ~60 min; the GPUs are the row's TP world.  An MoE config states its needs in the sweep file (the row id does
not say MoE).  The host's GPUs, RAM and CPUs are read (nvidia-smi, MemTotal, this process's CPU affinity) unless given.

Packing: configs in descending RAM, so the large Builds start first and the small ones fill around them; each starts as soon as its GPUs,
RAM share and CPUs are free.  A config larger than the host is refused by name and never started.  A run is ``verity-vllm row run ROW
ROLE REPO REV --stages build,commit --config-run 1`` with its own CUDA_VISIBLE_DEVICES and CPU affinity, its RAM share as the Build's
parallel-derive budget (BUILD_RAM_BUDGET_GB), the host's shared Program cache, unit-rule cache (VERITY_UNIT_RULE_CACHE) and JIT build dir
(safe for concurrent jobs: `native_jit.jit_load`), and its row dir under DIR/rows/.  ``--bootstrap`` runs
pod_bootstrap.sh once for every role the sweep names before any config starts: the venv, taps and checkpoints are per host.

Output: DIR/sweep.json, rewritten as each run ends: per config its resources, rc, wall and its config record's outcome and reason.  A
config that fails is a finding of the sweep, not a stop.
"""
from __future__ import annotations

import heapq
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT, RowId, UsageError, environ_snapshot, option

SCHEMA = "verity-vllm/sweep/v1"
LARGE = {"ram_gb": 500, "cpus": 32, "minutes": 150}
SMALL = {"ram_gb": 150, "cpus": 16, "minutes": 60}


@dataclass(frozen=True)
class Need:
    gpus: int
    ram_gb: int
    cpus: int
    minutes: int


@dataclass
class Host:
    gpus: list[int]
    ram_gb: int
    cpus: list[int]


@dataclass
class Config:
    row: str
    need: Need
    extra: dict[str, Any] = field(default_factory=dict)


def estimate(row: str) -> Need:
    """A config's needs from its row id (`config.RowId`): large when B >= 32 or the input >= 4096."""
    r = RowId.parse(row)
    c = LARGE if r.batch >= 32 or r.max_input >= 4096 else SMALL
    return Need(gpus=r.world, ram_gb=c["ram_gb"], cpus=c["cpus"], minutes=c["minutes"])


def load(path: str | Path) -> list[Config]:
    doc = json.loads(Path(path).read_text())
    out = []
    for c in doc.get("configs") or []:
        e = estimate(c["row"])
        need = Need(gpus=int(c.get("gpus", e.gpus)), ram_gb=int(c.get("ram_gb", e.ram_gb)), cpus=int(c.get("cpus", e.cpus)),
                    minutes=int(c.get("minutes", e.minutes)))
        out.append(Config(row=c["row"], need=need, extra={k: v for k, v in c.items() if k not in ("row", "gpus", "ram_gb", "cpus", "minutes")}))
    return out


def order(configs: list[Config]) -> list[Config]:
    """Descending RAM, then descending minutes, then the row id: the large Builds first."""
    return sorted(configs, key=lambda c: (-c.need.ram_gb, -c.need.minutes, c.row))


class Packer:
    """The host's free GPUs, RAM and CPUs; `take` claims a config's share (lowest ids first), `give` returns it."""

    def __init__(self, host: Host):
        self.host = host
        self.gpus, self.ram, self.cpus = sorted(host.gpus), int(host.ram_gb), sorted(host.cpus)

    def too_big(self, need: Need) -> str | None:
        h = self.host
        for what, want, have in (("GPUs", need.gpus, len(h.gpus)), ("GB of RAM", need.ram_gb, h.ram_gb), ("CPUs", need.cpus, len(h.cpus))):
            if want > have:
                return f"needs {want} {what}, the host has {have}"
        return None

    def fits(self, need: Need) -> bool:
        return need.gpus <= len(self.gpus) and need.ram_gb <= self.ram and need.cpus <= len(self.cpus)

    def take(self, need: Need) -> tuple[list[int], list[int]]:
        g, c = self.gpus[:need.gpus], self.cpus[:need.cpus]
        self.gpus, self.cpus = self.gpus[need.gpus:], self.cpus[need.cpus:]
        self.ram -= need.ram_gb
        return g, c

    def give(self, need: Need, gpus: list[int], cpus: list[int]) -> None:
        self.gpus, self.cpus = sorted(self.gpus + gpus), sorted(self.cpus + cpus)
        self.ram += need.ram_gb


def plan(configs: list[Config], host: Host) -> dict[str, Any]:
    """The packing simulated on the estimated minutes: each config's start and end (minutes), GPUs and CPUs, and the makespan."""
    pk, pending, running, t, out, refused = Packer(host), order(configs), [], 0, [], []
    for c in list(pending):
        why = pk.too_big(c.need)
        if why:
            refused.append({"row": c.row, "why": why})
            pending.remove(c)
    while pending or running:
        for c in list(pending):
            if pk.fits(c.need):
                g, cp = pk.take(c.need)
                heapq.heappush(running, (t + c.need.minutes, c.row, c, g, cp))
                out.append({"row": c.row, "start_min": t, "end_min": t + c.need.minutes, "gpus": g, "cpus": [cp[0], cp[-1]] if cp else [],
                            "need": asdict(c.need)})
                pending.remove(c)
        if running:
            t, _, c, g, cp = heapq.heappop(running)
            pk.give(c.need, g, cp)
    return {"schema": SCHEMA + "/plan", "host": asdict(host), "configs": out, "refused": refused, "makespan_min": t}


def host_of(gpus: int | None, ram_gb: int | None, cpus: int | None) -> Host:
    if gpus is None:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True)
        gpus = len([ln for ln in r.stdout.splitlines() if ln.strip()]) if r.returncode == 0 else 0
    if ram_gb is None:
        kb = next((int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines() if ln.startswith("MemTotal:")), 0)
        ram_gb = kb >> 20
    cpu_ids = sorted(os.sched_getaffinity(0))
    return Host(gpus=list(range(gpus)), ram_gb=int(ram_gb), cpus=cpu_ids[:cpus] if cpus else cpu_ids)


def _record(row_dir: Path) -> dict[str, Any]:
    try:
        rec = json.loads((row_dir / "config_record.json").read_text())
    except (OSError, ValueError):
        return {"outcome": None, "why": "no config record"}
    return {"outcome": rec.get("outcome"), "why": rec.get("why")}


def run(configs: list[Config], host: Host, out: Path, *, start: Callable[[Config, list[int], list[int]], Any],
        wait: Callable[[], tuple[Any, int]], now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Start each config as soon as it fits (`order`), wait for any to end (`wait` -> (handle, rc)), free its share, and rewrite
    DIR/sweep.json as each ends.  `start(config, gpus, cpus)` returns a handle; `now` is the clock the record reads."""
    pk, pending, live, done, refused = Packer(host), order(configs), {}, [], []
    for c in list(pending):
        why = pk.too_big(c.need)
        if why:
            refused.append({"row": c.row, "why": why})
            pending.remove(c)

    def write() -> dict[str, Any]:
        doc = {"schema": SCHEMA, "host": asdict(host), "done": done, "running": [v["entry"] for v in live.values()],
               "pending": [c.row for c in pending], "refused": refused}
        (out / "sweep.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        return doc

    out.mkdir(parents=True, exist_ok=True)
    while pending or live:
        for c in list(pending):
            if pk.fits(c.need):
                g, cp = pk.take(c.need)
                h = start(c, g, cp)
                live[h] = {"config": c, "gpus": g, "cpus": cp, "t0": now(),
                           "entry": {"row": c.row, "need": asdict(c.need), "gpus": g, "cpus": [cp[0], cp[-1]] if cp else []}}
                pending.remove(c)
        write()
        if not live:
            break
        h, rc = wait()
        v = live.pop(h)
        pk.give(v["config"].need, v["gpus"], v["cpus"])
        done.append({**v["entry"], "rc": rc, "wall_s": round(now() - v["t0"], 1), **_record(out / "rows" / v["config"].row)})
    return write()


def child_env(env: dict[str, str], c: Config, gpus: list[int], cpus: list[int], out: Path, unit_rule_cache: str | None) -> dict[str, str]:
    """A config run's environment: its GPUs and CPUs, its RAM share as the Build's derive budget, the host's caches and its row dirs."""
    e = {**env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "SWEEP_DIR": str(out / "rows"), "OMP_NUM_THREADS": str(len(cpus)),
         "BUILD_RAM_BUDGET_GB": str(c.need.ram_gb)}
    if unit_rule_cache:
        e["VERITY_UNIT_RULE_CACHE"] = unit_rule_cache
    return e


def launcher(out: Path, program_cache: str | None, env: dict[str, str], unit_rule_cache: str | None = None) -> tuple[Callable, Callable]:
    """(start, wait) over real `verity-vllm row run --config-run 1` processes."""
    from verity_vllm.pipeline.epoch import resolve

    procs: dict[int, subprocess.Popen] = {}
    (out / "logs").mkdir(parents=True, exist_ok=True)

    def start(c: Config, gpus: list[int], cpus: list[int]) -> int:
        role, repo, rev, _world = resolve(c.row)
        cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "row", "run", c.row, role, repo, rev, "--stages", "build,commit",
               "--config-run", "1", *(["--program-cache", program_cache] if program_cache else []), *map(str, c.extra.get("args") or [])]
        e = child_env(env, c, gpus, cpus, out, unit_rule_cache)
        log = open(out / "logs" / f"{c.row}.log", "w")
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=e, stdout=log, stderr=subprocess.STDOUT, preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        procs[p.pid] = p
        return p.pid

    def wait() -> tuple[int, int]:
        pid, status = os.wait()
        procs.pop(pid, None)
        return pid, os.waitstatus_to_exitcode(status)

    return start, wait


def bootstrap(configs: list[Config], out: Path, env: dict[str, str]) -> int:
    """pod_bootstrap.sh once for every role the sweep names (B0 included), before any config starts."""
    from verity_vllm.pipeline.epoch import resolve

    roles = sorted({"B0", *(resolve(c.row)[0] for c in configs)})
    d = out / "bootstrap"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "bootstrap.log", "w") as log:
        return subprocess.run(["bash", "verity_vllm/ops/pod_bootstrap.sh", "--cases", ",".join(roles), "--out", str(d), "--gpu"],
                              cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT).returncode


# ------------------------------------------------------------------------------------------------------------------------------ the CLI
@dataclass(frozen=True, kw_only=True)
class _HostOptions:
    sweep: str = option("sweep", help="the sweep file: {\"configs\": [{\"row\": ..., optional needs}, ...]}")
    gpus: int | None = option("--gpus", type=int, default=None, help="the host's GPUs (default: nvidia-smi)")
    ram_gb: int | None = option("--ram-gb", type=int, default=None, help="the host's RAM in GB (default: MemTotal)")
    cpus: int | None = option("--cpus", type=int, default=None, help="the host's CPUs (default: this process's affinity)")


@dataclass(frozen=True, kw_only=True)
class PlanOptions(_HostOptions):
    """The packing, simulated on the estimates; nothing runs."""

    cmd: str = field(default="plan", init=False)


@dataclass(frozen=True, kw_only=True)
class RunOptions(_HostOptions):
    """Run the sweep's config runs on this host."""

    out: str = option("--out", help="the sweep dir: rows/, logs/, sweep.json")
    program_cache: str | None = option("--program-cache", env="PROGRAM_CACHE", default=None, help="the host's Program cache (default DIR/program-cache)")
    unit_rule_cache: str | None = option("--unit-rule-cache", env="VERITY_UNIT_RULE_CACHE", default=None,
                                         help="the host's word-check unit-rule cache, exported to each run (default DIR/unit-rule-cache)")
    do_bootstrap: bool = option("--bootstrap", action="store_true", default=False, help="run pod_bootstrap.sh once for the sweep's roles first")
    environ: dict[str, str] = environ_snapshot()
    cmd: str = field(default="run", init=False)


COMMANDS = {"plan": PlanOptions, "run": RunOptions}


def main(a: Any) -> int:
    configs = load(a.sweep)
    host = host_of(a.gpus, a.ram_gb, a.cpus)
    if a.cmd == "plan":
        print(json.dumps(plan(configs, host), indent=1))
        return 0
    if not host.gpus:
        raise UsageError("sweep run: this host has no GPU (nvidia-smi) and --gpus was not given")
    out = Path(a.out)
    env = dict(a.environ)
    if a.do_bootstrap:
        rc = bootstrap(configs, out, env)
        if rc != 0:
            print(f"[sweep] bootstrap FAIL rc={rc} ({out / 'bootstrap' / 'bootstrap.log'}): no config started", flush=True)
            return rc
    start, wait = launcher(out, a.program_cache or str(out / "program-cache"), env, a.unit_rule_cache or str(out / "unit-rule-cache"))
    doc = run(configs, host, out, start=start, wait=wait)
    n_pass = sum(1 for d in doc["done"] if d.get("outcome") == "PASS")
    print(f"[sweep] {n_pass}/{len(doc['done'])} configs PASS, {len(doc['refused'])} refused; {out / 'sweep.json'}", flush=True)
    return 0
