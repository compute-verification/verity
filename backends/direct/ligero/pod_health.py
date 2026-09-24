"""Is this pod fit to measure on?  ``research pods health <pod>`` runs it over ssh; ``pod_bootstrap.sh`` runs it at the end.

    python -m backends.direct.ligero.pod_health [--json] [--no-bench] [--seconds S]

Facts: the cgroup CPU quota vs ``nproc`` (and whether ``OMP_NUM_THREADS`` is sized to it), the GPU (name, SM / memory clocks
and their maxima, power limit vs default, PCIe gen / width vs max, persistence mode, temperature; read while the matmul
benchmark keeps the GPU busy, so idle down-clocking is not a finding), other GPU compute processes, the load average, free
disk on /workspace.

Then a microbenchmark, about ``2 x --seconds`` (default 10) of timed calls after a warm-up, each call synchronised so host
noise shows: the SIMT RS encoder (the prover's encode + NTT) at l = 4096, n = 4 l, 2048 rows, and a bf16 matmul 8192^3.  The
medians are compared with this SKU's entry in ``pod_health_ref.json`` (the best pod seen; ``research pods health POD
--record`` updates it): ``DEGRADED`` (exit 3, replace the pod) when either is more than :data:`DEGRADED` slower.  Warnings (a
quota the thread pools ignore, a power cap, a narrow PCIe link, other GPU processes, little disk) are printed, not fatal.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from backends.direct.ligero import timing_guard

REF = Path(__file__).with_name("pod_health_ref.json")
DEGRADED = 0.20
ENCODE_L, ENCODE_ROWS, MATMUL_N = 4096, 2048, 8192
GPU_FIELDS = ("name", "clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem", "power.limit", "power.default_limit",
              "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max", "persistence_mode",
              "temperature.gpu")
MIN_FREE_GIB = 20


def _value(s: str) -> Any:
    try:
        return float(s)
    except ValueError:
        return None if s.startswith("[") else s


def gpu_state() -> dict[str, Any] | None:
    """The first GPU's :data:`GPU_FIELDS` from nvidia-smi (numbers as floats, ``[N/A]`` as None); None without nvidia-smi."""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else []
    return dict(zip(GPU_FIELDS, (_value(c.strip()) for c in lines[0].split(",")))) if lines else None


def gpu_apps() -> list[str] | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [" ".join(c.strip() for c in ln.split(",")) for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else None


def host() -> dict[str, Any]:
    disk = Path("/workspace") if Path("/workspace").is_dir() else Path("/")
    return {"cpu_quota": timing_guard.cpu_quota(), "nproc": os.cpu_count(), "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "load1": os.getloadavg()[0], "disk": str(disk), "disk_free_gib": round(shutil.disk_usage(disk).free / 2**30, 1)}


def warnings(h: dict[str, Any], gpu: dict[str, Any] | None, apps: list[str] | None) -> list[str]:
    w = []
    q, omp = h["cpu_quota"], h["omp_threads"]
    if q and q < (h["nproc"] or 0) and (not (omp or "").isdigit() or int(omp) > q):
        w.append(f"OMP_NUM_THREADS={omp or 'unset'} but the cgroup quota is {q:.1f} cores (nproc {h['nproc']}): source /workspace/env.sh")
    g = gpu or {}
    if isinstance(g.get("power.limit"), float) and isinstance(g.get("power.default_limit"), float) and g["power.limit"] < 0.95 * g["power.default_limit"]:
        w.append(f"power limit {g['power.limit']:.0f} W < default {g['power.default_limit']:.0f} W")
    if isinstance(g.get("pcie.link.width.current"), float) and isinstance(g.get("pcie.link.width.max"), float) and g["pcie.link.width.current"] < g["pcie.link.width.max"]:
        w.append(f"PCIe link x{g['pcie.link.width.current']:.0f} of x{g['pcie.link.width.max']:.0f}")
    if isinstance(g.get("pcie.link.gen.current"), float) and isinstance(g.get("pcie.link.gen.max"), float) and g["pcie.link.gen.current"] < g["pcie.link.gen.max"]:
        w.append(f"PCIe gen {g['pcie.link.gen.current']:.0f} of {g['pcie.link.gen.max']:.0f} under load")
    if apps:
        w.append(f"{len(apps)} other GPU compute process(es): {'; '.join(apps)}")
    if h["disk_free_gib"] < MIN_FREE_GIB:
        w.append(f"{h['disk_free_gib']} GiB free on {h['disk']}")
    if h["nproc"] and h["load1"] > h["nproc"]:
        w.append(f"load average {h['load1']:.1f} > {h['nproc']} cores")
    return w


def _median_time(fn, seconds: float) -> tuple[float, int]:
    import torch

    fn()
    torch.cuda.synchronize()
    ts: list[float] = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or len(ts) < 5:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), len(ts)


def bench(seconds: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``({"encode_ms", "matmul_tflops", "encode_calls", "matmul_calls"}, the GPU state read under load)``."""
    import torch

    from backends.direct.ligero.encode_simt import encoder_for
    from backends.direct.ligero.field import P

    g = torch.Generator(device="cuda").manual_seed(0)
    enc = encoder_for(ENCODE_L, 4 * ENCODE_L, 0, "cuda")
    rows = torch.randint(0, P, (ENCODE_ROWS, ENCODE_L), dtype=torch.int64, device="cuda", generator=g)
    enc_s, enc_n = _median_time(lambda: enc.encode(rows), seconds)
    a, b = (torch.randn(MATMUL_N, MATMUL_N, device="cuda", generator=g).to(torch.bfloat16) for _ in range(2))
    mm_s, mm_n = _median_time(lambda: a @ b, seconds)
    for _ in range(max(int(1.0 / mm_s), 1)):                     # about a second of queued work: read the clocks under load
        a @ b
    loaded = gpu_state()
    torch.cuda.synchronize()
    return {"encode_ms": round(enc_s * 1e3, 4), "matmul_tflops": round(2 * MATMUL_N**3 / mm_s / 1e12, 2),
            "encode_calls": enc_n, "matmul_calls": mm_n}, loaded


def sku_of(name: str | None) -> str | None:
    try:
        from verity_numerical.bench.tables import normalise_sku
    except ImportError:
        return name
    return normalise_sku(name)


def judge(measured: dict[str, Any] | None, ref: dict[str, Any] | None) -> tuple[str, dict[str, float]]:
    """``(verdict, slowdown per metric)``: the slowdown is measured time over the reference's (1.25 = 25 % slower)."""
    if measured is None:
        return "NO-BENCH", {}
    if not ref:
        return "NO-REFERENCE", {}
    slow = {}
    if (r := (ref.get("encode_ms") or {}).get("value")) and measured.get("encode_ms"):
        slow["encode"] = measured["encode_ms"] / r
    if (r := (ref.get("matmul_tflops") or {}).get("value")) and measured.get("matmul_tflops"):
        slow["matmul"] = r / measured["matmul_tflops"]
    return ("DEGRADED" if any(s > 1 + DEGRADED for s in slow.values()) else "OK"), slow


def report(seconds: float | None, ref_path: Path = REF) -> dict[str, Any]:
    h, idle, apps = host(), gpu_state(), gpu_apps()
    measured, loaded = bench(seconds) if seconds else (None, None)
    gpu = loaded or idle
    sku = sku_of((gpu or {}).get("name"))
    refs = json.loads(ref_path.read_text()).get("skus", {}) if ref_path.is_file() else {}
    verdict, slow = judge(measured, refs.get(sku))
    return {"schema": "pod-health/v1", "sku": sku, "host": h, "gpu": gpu, "gpu_apps": apps, "bench": measured, "ref": refs.get(sku),
            "slowdown": {k: round(v, 3) for k, v in slow.items()}, "verdict": verdict, "warnings": warnings(h, gpu, apps)}


def render(rep: dict[str, Any]) -> str:
    h, g, b = rep["host"], rep["gpu"] or {}, rep["bench"]
    f = lambda k, fmt="{:.0f}": "?" if not isinstance(g.get(k), float) else fmt.format(g[k])
    q = f"{h['cpu_quota']:.1f}" if h["cpu_quota"] else "unlimited"
    lines = [(f"host  cpu quota {q} of {h['nproc']} cores (OMP_NUM_THREADS={h['omp_threads'] or 'unset'})  load {h['load1']:.1f}"
              f"  {h['disk_free_gib']} GiB free on {h['disk']}"),
             (f"gpu   {g.get('name')} ({rep['sku']})  SM {f('clocks.sm')}/{f('clocks.max.sm')} MHz  mem {f('clocks.mem')}/{f('clocks.max.mem')} MHz"
              f"  power {f('power.limit')}/{f('power.default_limit')} W  PCIe gen{f('pcie.link.gen.current')} x{f('pcie.link.width.current')}"
              f" (max gen{f('pcie.link.gen.max')} x{f('pcie.link.width.max')})  persistence {g.get('persistence_mode')}  {f('temperature.gpu')} C"),
             f"gpu processes: {'; '.join(rep['gpu_apps']) if rep['gpu_apps'] else 'none' if rep['gpu_apps'] is not None else 'unknown'}"]
    if b:
        ref, slow = rep["ref"] or {}, rep["slowdown"]
        vs = lambda m, k: f" (ref {ref[m]['value']}: {slow[k]:.2f}x)" if k in slow else ""
        lines.append(f"bench encode l={ENCODE_L} n={4 * ENCODE_L} x{ENCODE_ROWS} rows: {b['encode_ms']:.3f} ms{vs('encode_ms', 'encode')}"
                     f"  |  matmul bf16 {MATMUL_N}^3: {b['matmul_tflops']:.1f} TFLOP/s{vs('matmul_tflops', 'matmul')}")
    lines += [f"warning: {w}" for w in rep["warnings"]]
    v = rep["verdict"]
    lines.append({"DEGRADED": f"DEGRADED: more than {DEGRADED:.0%} slower than the {rep['sku']} reference -- replace this pod",
                  "NO-REFERENCE": f"NO-REFERENCE: no {rep['sku']} entry in {REF.name} (research pods health POD --record adds this pod)",
                  "NO-BENCH": "NO-BENCH: facts only"}.get(v, "OK"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pod_health", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-bench", action="store_true", help="facts only (no GPU work)")
    ap.add_argument("--seconds", type=float, default=10.0, help="timed seconds per benchmark (default 10)")
    ns = ap.parse_args(argv)
    rep = report(None if ns.no_bench else ns.seconds)
    print(json.dumps(rep, indent=1) if ns.json else render(rep))
    return 3 if rep["verdict"] == "DEGRADED" else 0


if __name__ == "__main__":
    sys.exit(main())
