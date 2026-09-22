"""Measure the fused degree-3 round kernels (``round_simt``, ``round_tc``) on one GPU: per-kernel time vs the roofline,
the 6-layer / 130-round model as a real sequence (eager and CUDA-graphed), the launch-latency tail, and the previous
torch composition from its recorded run.  Writes ``bench.json`` + the ``research/result/v0.1`` envelope.

Pod::

    /workspace/tc/venv/bin/python backends/gkr/tensor/fused/bench.py --out "$RESEARCH_RUN_DIR"

Labels: ``ARITHMETIC_DIAGNOSTIC``, ``authentication excluded``, ``security.target = -128`` (BabyBear^6 challenge).  Kernel
measurements on random tables, not a proof.  Timing: CUDA events, warm kernels, median of ``--repeats``; per-kernel
numbers are taken with a cold L2 (a 256 MB write between repetitions) so that GB/s is HBM traffic; the layer sequence is
run as the prover would (round i+1 reads what round i wrote, L2 as it falls).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import torch

from tensor import reference as ref
from tensor.fused import roofline
from tensor.fused import round_simt as rs

try:
    from tensor.fused import round_tc as rtc
except Exception as e:  # noqa: BLE001  - the TC kernel is optional until it compiles on the box
    rtc = None
    RTC_ERROR = f"{type(e).__name__}: {e}"
else:
    RTC_ERROR = None

GPU_PEAKS = {
    "NVIDIA GeForce RTX 4090": {"int8_dense_tops": 660.6, "hbm_bytes_per_s": 1008e9, "fp32_tflops": 82.6, "bf16_dense_tflops": 165.2,
                                "int32_ops_per_s": roofline.SIMT_INT32_OPS,
                                "source": "NVIDIA Ada GPU architecture whitepaper / RTX 4090 datasheet (dense, no sparsity)"},
}
PROFILE = "first-campaign-target/2026-09-21"
K_TARGET, B_TARGET, UNITS = 1536, 4096, 4096
LAYER_WIDTHS = (137, 679, 673, 669, 647, 533)
PREVIOUS_RUN = HERE.parent.parent.parent / "numerical" / "reports" / "tc_sumcheck_4090.json"


def pow2_pad(w: int) -> int:
    return 1 << max(1, math.ceil(math.log2(w)))


def median(xs):
    return statistics.median(xs)


def probe_hardware() -> dict:
    gpu = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "sm": ".".join(map(str, torch.cuda.get_device_capability(0)))}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        gpu["driver"], gpu["memory"] = [x.strip() for x in q.stdout.strip().split("\n")[0].split(",")]
    except Exception as e:  # noqa: BLE001
        gpu["probe_error"] = f"{type(e).__name__}: {e}"
    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"gpu": gpu, "cpu": {"model": cpu, "count": os.cpu_count()}}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("RESEARCH_SOURCE_SHA")


# ---------------------------------------------------------------------------------------------------------------------
# timing primitives
# ---------------------------------------------------------------------------------------------------------------------


class Flusher:
    def __init__(self, device, mb: int = 256):
        self.buf = torch.empty(mb << 20, dtype=torch.int8, device=device)

    def __call__(self):
        self.buf.fill_(1)


def timed(fn, repeats: int, warm: int, flush=None) -> list[float]:
    """Seconds per call (CUDA events), ``flush()`` before every timed call when given."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        if flush is not None:
            flush()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    return out


def make_kernel(impl: str, device, block: int | None, warps: int | None):
    if impl == "simt":
        return rs.FusedRound(device, block or 256, warps or 4)
    if impl == "tc":
        if rtc is None:
            raise RuntimeError(f"round_tc unavailable: {RTC_ERROR}")
        return rtc.FusedRoundTC(device, block or 64, warps or 4)
    raise ValueError(impl)


# ---------------------------------------------------------------------------------------------------------------------
# per-kernel measurement against the roofline
# ---------------------------------------------------------------------------------------------------------------------


def measure_kernel(kern, n: int, deg_in: int, mode: str, repeats: int, warm: int, peaks: dict, flush, device) -> dict:
    """One kernel shape: ``mode`` in fold+eval | fold | eval (first round); ``n`` input elements per table."""
    f, g, h = (rs.FusedTable.random(n, deg_in, device, 11 + i) for i in range(3))
    r = rs.challenge_tensor(ref.random_ext(random.Random(n)), device)
    fold = mode != "eval"
    ev = mode != "fold"
    outs = [kern.alloc_out(n // 2) for _ in range(3)] if fold else None
    nprog = kern.grid_size(n, fold)
    s_buf = torch.zeros((nprog, 24), dtype=torch.int64, device=device) if ev else None

    def run():
        kern.launch(f, g, h, r if fold else None, ev, outs, s_buf)

    sec = timed(run, repeats, warm, flush)
    t = median(sec)
    in_bytes = 4 if deg_in == 1 else 24
    rf = roofline.kernel(n, in_bytes=in_bytes, first_round=(mode == "eval"), fold_only=(mode == "fold"))
    nbytes = rf["bytes"]
    out = {"impl": kern.name, "n": n, "deg_in": deg_in, "mode": mode, "seconds": t, "seconds_all": sec, "block": kern.block, "num_warps": kern.num_warps,
           "bytes": nbytes, "gb_per_s": nbytes / t / 1e9, "hbm_fraction": nbytes / t / peaks["hbm_bytes_per_s"],
           "t_hbm_floor": rf["t_hbm_us"] * 1e-6, "floor_multiple": t / (rf["t_hbm_us"] * 1e-6),
           "ext_muls": rf["ext_muls"], "simt_ops_model": rf["simt_ops"], "simt_ops_per_s": rf["simt_ops"] / t,
           "simt_fraction_of_peak": rf["simt_ops"] / t / peaks["int32_ops_per_s"],
           "tc_macs_model": rf["tc_macs"], "tc_macs_per_s": rf["tc_macs"] / t, "tc_fraction_of_peak": rf["tc_macs"] / t / (peaks["int8_dense_tops"] * 1e12 / 2),
           "nprog": nprog}
    if hasattr(kern, "macs_performed"):
        out["tc_macs_performed"] = kern.macs_performed(n, deg_in, mode)
        out["tc_performed_fraction_of_peak"] = out["tc_macs_performed"] / t / (peaks["int8_dense_tops"] * 1e12 / 2)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the layer model as a real sequence
# ---------------------------------------------------------------------------------------------------------------------


def layer_plan(width: int, units: int = UNITS) -> list[tuple[int, int, str]]:
    """Kernel sequence of one layer: ``(n_in, deg_in, mode)``.  Round 1 is S over the base tables (no fold); round 2
    folds the base tables with r_1 (extension output) and evaluates; then extension rounds; a final fold to 1 element."""
    n0 = units * pow2_pad(width)
    plan = [(n0, 1, "eval"), (n0, 1, "fold+eval")]
    n = n0 // 2
    while n >= 4:
        plan.append((n, 6, "fold+eval"))
        n //= 2
    plan.append((2, 6, "fold"))
    return plan


class LayerRunner:
    """Pre-allocated buffers for one layer's kernel sequence so the whole sequence can be captured in a CUDA graph."""

    def __init__(self, kern, width: int, device, seed: int = 5):
        self.kern = kern
        self.plan = layer_plan(width)
        n0 = self.plan[0][0]
        self.tables = [rs.FusedTable.random(n0, 1, device, seed + i) for i in range(3)]
        rng = random.Random(seed)
        self.challenges = [rs.challenge_tensor(ref.random_ext(rng), device) for _ in self.plan]
        self.outs = []
        self.s_bufs = []
        for (n, deg, mode) in self.plan:
            fold = mode != "eval"
            self.outs.append([kern.alloc_out(n // 2) for _ in range(3)] if fold else None)
            self.s_bufs.append(torch.zeros((kern.grid_size(n, fold), 24), dtype=torch.int64, device=device) if mode != "fold" else None)
        self.s_total = torch.zeros((len(self.plan), 24), dtype=torch.int64, device=device)

    def run(self, lo: int = 0, hi: int | None = None):
        """Kernels ``lo:hi`` of the plan (all by default); each round's partial sums are reduced on the device."""
        cur = self.tables
        hi = len(self.plan) if hi is None else hi
        for i, (n, deg, mode) in enumerate(self.plan):
            if i >= hi:
                break
            fold = mode != "eval"
            if i >= lo:
                s, _ = self.kern.launch(cur[0], cur[1], cur[2], self.challenges[i] if fold else None, mode != "fold", self.outs[i], self.s_bufs[i])
                if s is not None:
                    torch.sum(s, 0, out=self.s_total[i])
            if fold:
                cur = [rs.FusedTable(o) for o in self.outs[i]]

    def bytes(self, lo: int = 0, hi: int | None = None) -> int:
        hi = len(self.plan) if hi is None else hi
        return sum(roofline.kernel(n, in_bytes=4 if deg == 1 else 24, first_round=(mode == "eval"), fold_only=(mode == "fold"))["bytes"]
                   for (n, deg, mode) in self.plan[lo:hi])


def graph_of(fn) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def measure_layers(kern, repeats: int, warm: int, peaks: dict, device, tail_log2: int = 15) -> dict:
    layers = []
    tot = {"eager_seconds": 0.0, "graphed_seconds": 0.0, "bytes": 0, "kernels": 0, "rounds": 0,
           "tail_eager_seconds": 0.0, "tail_graphed_seconds": 0.0, "tail_kernels": 0, "head_graphed_seconds": 0.0}
    for w in LAYER_WIDTHS:
        L = LayerRunner(kern, w, device)
        eager = median(timed(L.run, repeats, warm))
        g = graph_of(L.run)
        graphed = median(timed(g.replay, repeats, warm))
        split = next(i for i, (n, _, _) in enumerate(L.plan) if n <= (1 << tail_log2))
        head_g = graph_of(lambda L=L, split=split: L.run(0, split))
        tail_g = graph_of(lambda L=L, split=split: L.run(split))
        tail_eager = median(timed(lambda L=L, split=split: L.run(split), repeats, warm))
        tail_graphed = median(timed(tail_g.replay, repeats, warm))
        head_graphed = median(timed(head_g.replay, repeats, warm))
        nb = L.bytes()
        layers.append({"width": w, "n0": L.plan[0][0], "kernels": len(L.plan), "rounds": len(L.plan) - 1, "eager_seconds": eager, "graphed_seconds": graphed,
                       "bytes": nb, "gb_per_s_graphed": nb / graphed / 1e9, "hbm_floor_seconds": nb / peaks["hbm_bytes_per_s"],
                       "tail": {"first_kernel": split, "n_max": L.plan[split][0], "kernels": len(L.plan) - split, "eager_seconds": tail_eager,
                                "graphed_seconds": tail_graphed, "bytes": L.bytes(split)},
                       "head_graphed_seconds": head_graphed})
        tot["eager_seconds"] += eager
        tot["graphed_seconds"] += graphed
        tot["bytes"] += nb
        tot["kernels"] += len(L.plan)
        tot["rounds"] += len(L.plan) - 1
        tot["tail_eager_seconds"] += tail_eager
        tot["tail_graphed_seconds"] += tail_graphed
        tot["tail_kernels"] += len(L.plan) - split
        tot["head_graphed_seconds"] += head_graphed
        print(f"  layer w={w}: {len(L.plan)} kernels  eager {eager*1e3:.2f} ms  graphed {graphed*1e3:.2f} ms  ({nb/graphed/1e9:.0f} GB/s; floor {nb/peaks['hbm_bytes_per_s']*1e3:.2f} ms)"
              f"  tail(n<=2^{tail_log2}: {len(L.plan)-split} kernels) eager {tail_eager*1e6:.0f} us graphed {tail_graphed*1e6:.0f} us", flush=True)
        del L, g, head_g, tail_g
        torch.cuda.empty_cache()
    tot["hbm_floor_seconds"] = tot["bytes"] / peaks["hbm_bytes_per_s"]
    tot["floor_multiple_graphed"] = tot["graphed_seconds"] / tot["hbm_floor_seconds"]
    tot["layers"] = layers
    return tot


def measure_launch(kern, repeats: int, device) -> dict:
    """Launch latency: a tiny fold+eval kernel eager (CPU launch + GPU) and as 32 graphed replays."""
    n = 64
    f, g, h = (rs.FusedTable.random(n, 6, device, i) for i in range(3))
    r = rs.challenge_tensor(ref.random_ext(random.Random(3)), device)
    outs = [kern.alloc_out(n // 2) for _ in range(3)]
    s_buf = torch.zeros((kern.grid_size(n, True), 24), dtype=torch.int64, device=device)
    run = lambda: kern.launch(f, g, h, r, True, outs, s_buf)
    eager = median(timed(run, repeats, 3))

    def many():
        for _ in range(32):
            run()

    eager32 = median(timed(many, repeats, 2)) / 32
    gr = graph_of(many)
    graphed32 = median(timed(gr.replay, repeats, 2)) / 32
    return {"n": n, "eager_single_seconds": eager, "eager_per_kernel_in_32": eager32, "graphed_per_kernel_in_32": graphed32}


# ---------------------------------------------------------------------------------------------------------------------
# previous composition (recorded)
# ---------------------------------------------------------------------------------------------------------------------


def previous_composition() -> dict | None:
    """The torch limb-composition numbers of lane tc-sumcheck on the same GPU (its recorded run), for the comparison."""
    if not PREVIOUS_RUN.exists():
        return None
    doc = json.loads(PREVIOUS_RUN.read_text())
    m = doc["measured"]
    rounds = {(r["t"], r["deg"]): r for r in m["rounds"]}
    per_round = {f"t{t}.deg{d}": {"graphed_seconds": r["seconds"].get("total_graphed"), "kernel_gpu_busy_seconds": (r.get("gpu_kernel_seconds") or {}).get("gpu_busy")}
                 for (t, d), r in rounds.items()}
    return {"run_id": doc["source"]["run_id"], "layer_model_seconds": m["layer_model_seconds"], "rounds": per_round, "utilisation": m["utilisation"]}


# ---------------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------------


def run_agreement_test() -> str:
    env = dict(os.environ, TC_DEVICE="cuda")
    r = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q", "-x"], capture_output=True, text=True, env=env, cwd=HERE.parent.parent, check=False)
    return (r.stdout + r.stderr)[-4000:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR") or "backends/gkr/tensor/fused/out")
    ap.add_argument("--impl", nargs="*", default=["simt", "tc"])
    ap.add_argument("--sizes", type=int, nargs="*", default=[23, 22, 21, 20, 18, 16, 14, 12], help="log2 of input elements per table")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--warps", type=int, default=None)
    ap.add_argument("--no-test", dest="test", action="store_false")
    ap.add_argument("--no-layers", dest="layers", action="store_false")
    ap.add_argument("--no-flush", dest="flush", action="store_false", help="warm-L2 per-kernel numbers")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    hw = probe_hardware()
    peaks = GPU_PEAKS.get(hw["gpu"]["name"])
    if peaks is None:
        print(f"WARNING: no peak table for {hw['gpu']['name']}; using the 4090's", file=sys.stderr)
        peaks = GPU_PEAKS["NVIDIA GeForce RTX 4090"]
    test_log = run_agreement_test() if a.test else "skipped"
    (out / "agreement_test.log").write_text(test_log)
    tail = test_log.strip().splitlines()[-1] if test_log.strip() else "(no test output)"
    print(tail, flush=True)
    t0 = time.time()
    flush = Flusher(device) if a.flush else None
    torch.cuda.reset_peak_memory_stats()
    doc = {"schema": "fused-sumcheck-bench/v1", "hardware": hw, "peaks": peaks,
           "software": {"torch": torch.__version__, "triton": __import__("triton").__version__, "python": platform.python_version(), "commit": git_head()},
           "config": vars(a) | {"field": f"BabyBear p=2^31-2^27+1, F_p^6 = F_p[X]/(X^6 - {ref.W})", "units": UNITS, "layer_widths": LAYER_WIDTHS,
                                "l2_flush": bool(flush), "tc_error": RTC_ERROR},
           "roofline": {"per_kernel_ext": [roofline.kernel(1 << e) for e in a.sizes]},
           "kernels": [], "layers": {}, "launch": {}, "previous_composition": previous_composition(), "agreement_test_tail": tail}
    for impl in a.impl:
        try:
            kern = make_kernel(impl, device, a.block, a.warps)
        except Exception as e:  # noqa: BLE001
            print(f"{impl}: unavailable ({e})", flush=True)
            doc["config"][f"{impl}_error"] = f"{type(e).__name__}: {e}"
            continue
        print(f"== {impl} (block {kern.block}, warps {kern.num_warps})", flush=True)
        for e in a.sizes:
            n = 1 << e
            for (deg, mode) in ((6, "fold+eval"), (6, "fold"), (6, "eval"), (1, "fold+eval"), (1, "eval")):
                if not kern.supports(deg, mode):
                    continue
                if deg == 1 and e not in (22, 20):
                    continue
                m = measure_kernel(kern, n, deg, mode, a.repeats, a.warm, peaks, flush, device)
                doc["kernels"].append(m)
                print(f"  n=2^{e} deg_in={deg} {mode:9}: {m['seconds']*1e6:8.1f} us  {m['gb_per_s']:6.0f} GB/s  {m['floor_multiple']:.2f}x HBM floor  "
                      f"SIMT-model {m['simt_fraction_of_peak']:.2f} of INT32 peak  TC-model {m['tc_fraction_of_peak']:.3f} of INT8 peak", flush=True)
        if a.layers:
            print("  layer model:", flush=True)
            doc["layers"][impl] = measure_layers(kern, a.repeats, a.warm, peaks, device)
            L = doc["layers"][impl]
            print(f"  {impl} 6 layers / {L['rounds']} rounds: eager {L['eager_seconds']*1e3:.2f} ms  graphed {L['graphed_seconds']*1e3:.2f} ms  "
                  f"floor {L['hbm_floor_seconds']*1e3:.2f} ms ({L['floor_multiple_graphed']:.2f}x)  tail eager {L['tail_eager_seconds']*1e3:.2f} graphed {L['tail_graphed_seconds']*1e3:.2f} ms", flush=True)
        doc["launch"][impl] = measure_launch(kern, a.repeats, device)
        print(f"  launch: single eager {doc['launch'][impl]['eager_single_seconds']*1e6:.1f} us; per kernel in a run of 32: eager {doc['launch'][impl]['eager_per_kernel_in_32']*1e6:.1f} us, "
              f"graphed {doc['launch'][impl]['graphed_per_kernel_in_32']*1e6:.1f} us", flush=True)
    doc["wall_seconds"] = time.time() - t0
    doc["mem.peak_device_bytes"] = int(torch.cuda.max_memory_allocated())
    (out / "bench.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    env = envelope(doc, os.environ.get("RESEARCH_RUN_ID", "local"), ["bench.json", "agreement_test.log"])
    if a.test and ("failed" in tail or "passed" not in tail):
        env["validation"]["status"] = "failed"
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(env, indent=1, default=str))
    os.replace(tmp, out / "result.json")
    return 0


def envelope(doc: dict, run_id: str, artifacts: list[str]) -> dict:
    ms = []

    def m(name, value, unit):
        if value is not None:
            ms.append({"name": name, "value": value, "unit": unit})

    for k in doc["kernels"]:
        key = f"{k['impl']}.n{k['n']}.deg{k['deg_in']}.{k['mode'].replace('+', '_')}"
        m(f"fused.seconds.{key}", k["seconds"], "s")
        m(f"fused.gb_per_s.{key}", k["gb_per_s"], "GB/s")
        m(f"fused.floor_multiple.{key}", k["floor_multiple"], "ratio")
    for impl, L in doc["layers"].items():
        m(f"fused.layers.{impl}.eager", L["eager_seconds"], "s")
        m(f"fused.layers.{impl}.graphed", L["graphed_seconds"], "s")
        m(f"fused.layers.{impl}.hbm_floor", L["hbm_floor_seconds"], "s")
        m(f"fused.layers.{impl}.tail_graphed", L["tail_graphed_seconds"], "s")
        if impl == "simt":
            m("t.total", L["graphed_seconds"], "s")
            m("t.arithmetic", L["graphed_seconds"], "s")
    for impl, la in doc["launch"].items():
        m(f"fused.launch.{impl}.eager_single", la["eager_single_seconds"], "s")
        m(f"fused.launch.{impl}.graphed_per_kernel", la["graphed_per_kernel_in_32"], "s")
    m("mem.peak_device_bytes", doc["mem.peak_device_bytes"], "B")
    hw = doc["hardware"]
    fp = {"profile": PROFILE, "domain": "finite", "K": K_TARGET, "B": B_TARGET, "N_subbatches": 1,
          "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "zk_mode": "none",
          "security": {"target": -128, "challenge_field": "BabyBear^6", "note": "operating point of the priced design; nothing is proved here"},
          "hardware": {"gpu": hw["gpu"], "cpu": hw["cpu"]},
          "software": {"backend": {"name": "backends/gkr/tensor/fused", "commit": doc["software"]["commit"], "version": None},
                       "toolchain": {"torch": doc["software"]["torch"], "triton": doc["software"]["triton"], "python": doc["software"]["python"], "cuda": hw["gpu"].get("cuda")}},
          "preprocessing": "none", "retention": "retained", "instances": None,
          "stage": "tc.sumcheck.fused", "kind": "kernel measurement (not a proof): fused degree-3 GKR round (SIMT and tensor-core Triton kernels), random tables",
          "config": doc["config"], "peaks": doc["peaks"]}
    status = "passed" if doc["layers"] else "not_run"
    return {"schema": "research/result/v0.1", "run_id": run_id, "workload_fingerprint": fp,
            "validation": {"status": status, "checker": "backends/gkr/tensor/fused/tests/test_fused.py (CUDA path, run before the sweep)",
                           "detail": "kernels == BabyBear^6 reference bit-for-bit on t <= 4096 in every mode; timings are medians of CUDA-event repeats", "evidence": artifacts},
            "measurements": ms, "measurement_files": artifacts, "artifacts": artifacts}


if __name__ == "__main__":
    sys.exit(main())
