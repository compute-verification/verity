"""Measure the INT8 tensor-core GKR round (``sumcheck_int8``) on one GPU and write ``bench.json`` + the
``research/result/v0.1`` envelope; ``--reprice`` (laptop side, needs ``verity_numerical``) turns a fetched ``bench.json``
into the re-priced Candidate A overheads.

Pod::

    /workspace/tc/venv/bin/python backends/gkr/tensor/bench.py --out "$RESEARCH_RUN_DIR"

Laptop::

    uv run --package verity-numerical python backends/gkr/tensor/bench.py --reprice <run>/bench.json \
        --report backends/numerical/reports/tc_sumcheck_4090

Every number is ``ARITHMETIC_DIAGNOSTIC``, ``authentication excluded``, ``security.target = -128`` (degree-6 challenge):
a kernel measurement, not a proof.  Timing: CUDA events, warm, median of ``--repeats``; MACs are the unpadded ``M K N``
of the matmuls the round issues (``Stats.macs_performed``; the cost model's unit), padding reported beside.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tensor import reference as ref
from tensor import sumcheck_int8 as k

#: RTX 4090 (Ada, AD102) datasheet: 660.6 TOPS INT8 dense tensor (1321.2 with sparsity), 1008 GB/s GDDR6X, 82.6 TFLOPS FP32.
GPU_PEAKS = {
    "NVIDIA GeForce RTX 4090": {"int8_dense_tops": 660.6, "hbm_bytes_per_s": 1008e9, "fp32_tflops": 82.6, "bf16_dense_tflops": 165.2,
                                "source": "NVIDIA Ada GPU architecture whitepaper / RTX 4090 datasheet (dense, no sparsity)"},
}
#: The frozen target's name (``verity_numerical.bench.contract.PROFILE``), copied so the pod venv needs no workspace install;
#: ``--reprice`` re-validates the envelope with the real contract.
PROFILE = "first-campaign-target/2026-09-21"
DOMAIN = "finite"
K_TARGET, B_TARGET, UNITS = 1536, 4096, 4096
#: Candidate A's checker layers (``accounting.CHECKER_LAYER_WIDTHS_A``, output layer first): carried width per unit.
LAYER_WIDTHS = (137, 679, 673, 669, 647, 533)
DL = 24  # limbs of a BabyBear^6 challenge element


def pow2_pad(w: int) -> int:
    return 1 << max(1, math.ceil(math.log2(w)))


def layer_rounds(width: int, units: int = UNITS) -> list[tuple[int, int]]:
    """The (t, deg) sequence of the U-parallel rounds of one layer: table ``units x 2^ceil(log2 width)`` elements,
    round 1 on base tables, then extension tables halving down to 2."""
    t = units * pow2_pad(width)
    out = [(t, 1)]
    t //= 2
    while t >= 2:
        out.append((t, 6))
        t //= 2
    return out


def model_macs_layer(width: int, units: int = UNITS, padded: bool = True) -> int:
    """The cost model's limb MACs for one layer (``explore.tensor.sumcheck_macs(n, 3, 24)`` with the first-round F_t
    credit ``price_design`` applies): ``n = units x width``."""
    n = units * (pow2_pad(width) if padded else width)
    return n * DL * DL * (8 + 8 * 3) - n * (DL * DL - 4 * DL) * 8 // 2


# ---------------------------------------------------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------------------------------------------------


def probe_hardware() -> dict:
    gpu = {"name": None, "driver": None, "cuda": None}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10, check=False).stdout.strip().split("\n")[0]
        name, driver, mem = [x.strip() for x in q.split(",")]
        gpu = {"name": name, "driver": driver, "cuda": None, "memory": mem}
    except Exception as e:  # noqa: BLE001
        gpu["probe_error"] = f"{type(e).__name__}: {e}"
    if k.torch is not None and k.torch.cuda.is_available():
        gpu["cuda"] = k.torch.version.cuda
        gpu["name"] = gpu["name"] or k.torch.cuda.get_device_name(0)
        gpu["sm"] = ".".join(map(str, k.torch.cuda.get_device_capability(0)))
    cpu = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"gpu": gpu if gpu["name"] else None, "cpu": {"model": cpu or "unknown", "count": os.cpu_count()}}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("RESEARCH_SOURCE_SHA")


def median(xs):
    return statistics.median(xs)


# ---------------------------------------------------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------------------------------------------------


def time_round(kern: k.Kernel, t: int, deg: int, repeats: int, warm: int, seed: int = 1) -> dict:
    ops = kern.ops
    f, g, h = (kern.random_table(t, deg, seed + i) for i in range(3))
    r = ref.random_ext(__import__("random").Random(seed))
    C = k.fold_matrix(ops, r, deg, kern.reduce_in_matrix)
    for _ in range(warm):
        kern.round_step(f, g, h, r, fold_C=C)
    ops.synchronize()
    runs = []
    stats = None
    for _ in range(repeats):
        stats = k.Stats()
        timer = k.Timer(ops, stats)
        kern.round_step(f, g, h, r, stats, timer, fold_C=C)
        seg = timer.finish()
        seg["total"] = sum(v for kk, v in seg.items() if kk != "total")
        runs.append(seg)
    segs = {name: median([r_[name] for r_ in runs]) for name in runs[0]}
    segs["tensor_bucket"] = segs["total"] - segs["hadamard"]              # everything the model prices as matmul-side work
    segs["matmul"] = segs["matmul.inner"] + segs["matmul.fold"]
    out = {"t": t, "deg": deg, "seconds": segs, "macs_performed": stats.macs_performed, "macs_padded": stats.macs_padded,
           "hadamard_products": stats.hadamard_products, "repeats": repeats, "warm": warm}
    # graphed whole round (launch overhead removed): CUDA only
    if ops.cuda:
        try:
            out["seconds"]["total_graphed"] = time_round_graphed(kern, f, g, h, r, C, repeats)
        except Exception as e:  # noqa: BLE001
            out["graph_error"] = f"{type(e).__name__}: {e}"
    return out


def time_round_graphed(kern: k.Kernel, f, g, h, r, C, repeats: int) -> float:
    torch = k.torch
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            kern.round_step(f, g, h, r, fold_C=C)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kern.round_step(f, g, h, r, fold_C=C)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return median(times)


def time_int_mm(ops: k.Ops, M: int, K: int, N: int, repeats: int, warm: int = 2) -> dict:
    """A plain ``_int_mm`` of the given shape: the library ceiling."""
    torch = k.torch
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=ops.device)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=ops.device)
    for _ in range(warm):
        torch._int_mm(a, b)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch._int_mm(a, b)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    sec = median(times)
    return {"M": M, "K": K, "N": N, "seconds": sec, "macs": M * K * N, "tmacs_per_s": M * K * N / sec / 1e12}


def time_inner_chunk_loop(ops: k.Ops, half: int, chunk: int, repeats: int) -> dict:
    """The chunked ``F^T x GH`` loop alone (48 x chunk x 96 per launch) for a table of ``half`` rows: separates launch
    cost from the round's other passes."""
    torch = k.torch
    F = torch.randint(-128, 128, (48, half), dtype=torch.int8, device=ops.device)
    GH = torch.randint(-128, 128, (half, 96), dtype=torch.int8, device=ops.device)
    n = -(-half // chunk)
    buf = torch.zeros((n, 48, 96), dtype=torch.int32, device=ops.device)

    def run():
        for c in range(n):
            lo, hi = c * chunk, min(half, (c + 1) * chunk)
            torch._int_mm(F[:, lo:hi], GH[lo:hi], out=buf[c])

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    sec = median(times)
    return {"half": half, "chunk": chunk, "launches": n, "seconds": sec, "macs": 48 * half * 96, "tmacs_per_s": 48 * half * 96 / sec / 1e12,
            "us_per_launch": sec / n * 1e6}


# ---------------------------------------------------------------------------------------------------------------------
# the campaign: sizes -> layers -> effective utilisation
# ---------------------------------------------------------------------------------------------------------------------


def aggregate(rounds: dict[tuple[int, int], dict], peak_macs: float, widths=LAYER_WIDTHS) -> dict:
    """Sum the measured rounds over the six layers' round sequences and derive the effective utilisations."""
    layers = []
    tot = {"seconds": {}, "macs_performed": 0, "macs_padded": 0, "macs_model_padded": 0, "macs_model_census": 0, "rounds": 0}
    for w in widths:
        seq = layer_rounds(w)
        L = {"width": w, "width_padded": pow2_pad(w), "rounds": len(seq), "seconds": {}, "macs_performed": 0, "macs_padded": 0,
             "macs_model_padded": model_macs_layer(w, padded=True), "macs_model_census": model_macs_layer(w, padded=False),
             "missing": []}
        for key in seq:
            m = rounds.get(key)
            if m is None:
                L["missing"].append(key)
                continue
            for name, v in m["seconds"].items():
                L["seconds"][name] = L["seconds"].get(name, 0.0) + v
            L["macs_performed"] += m["macs_performed"]
            L["macs_padded"] += m["macs_padded"]
        layers.append(L)
        for name, v in L["seconds"].items():
            tot["seconds"][name] = tot["seconds"].get(name, 0.0) + v
        for kk in ("macs_performed", "macs_padded", "macs_model_padded", "macs_model_census", "rounds"):
            tot[kk] += L[kk]
    sec = tot["seconds"]

    def util(macs, seconds):
        return macs / (peak_macs * seconds) if seconds > 0 else None

    tot["utilisation"] = {
        # achieved fraction of the INT8 dense peak, performed MACs over ...
        "matmul_only": util(tot["macs_performed"], sec.get("matmul", 0)),
        "tensor_bucket": util(tot["macs_performed"], sec.get("tensor_bucket", 0)),      # + limb passes, chunk sums, reductions
        "round_total": util(tot["macs_performed"], sec.get("total", 0)),                # + torch-int64 Hadamards
        "round_graphed": util(tot["macs_performed"], sec.get("total_graphed", 0)),
        # the number to plug into explore.tensor's ``utilisation``: the MODEL's MACs over the measured time
        "model_equivalent.tensor_bucket": util(tot["macs_model_padded"], sec.get("tensor_bucket", 0)),
        "model_equivalent.matmul_only": util(tot["macs_model_padded"], sec.get("matmul", 0)),
        "model_equivalent.census_width.tensor_bucket": util(tot["macs_model_census"], sec.get("tensor_bucket", 0)),
    }
    tot["layers"] = layers
    return tot


def run_campaign(a) -> dict:
    ops = k.default_ops(prefer_cuda=True)
    if not ops.cuda:
        print("WARNING: no CUDA device; timings are the CPU fallback and mean nothing for the tensor-core question", file=sys.stderr)
    kern = k.Kernel(ops, chunk=a.chunk, reduce_in_matrix=(a.layout == "reduced"), inner_impl=a.inner)
    if a.compile:
        # fuse the SIMT passes (Hadamards, limb decomposition, recomposition) into one kernel each instead of ~40 unfused torch ops
        k.ext_mul = k.torch.compile(k.ext_mul, dynamic=True)
        k.decompose = k.torch.compile(k.decompose, dynamic=True)
        k.recompose = k.torch.compile(k.recompose, dynamic=True)
    hw = probe_hardware()
    peaks = GPU_PEAKS.get((hw["gpu"] or {}).get("name"), None)
    peak_macs = peaks["int8_dense_tops"] * 1e12 / 2 if peaks else None
    if ops.cuda:
        k.torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sizes = [(1 << e, 6) for e in range(a.ext_max_log2, 0, -1)] + [(1 << e, 1) for e in a.base_log2]
    rounds = {}
    for (t, deg) in sizes:
        m = time_round(kern, t, deg, a.repeats, a.warm)
        if peak_macs:
            s = m["seconds"]
            m["tmacs_per_s"] = {kk: m["macs_performed"] / s[kk] / 1e12 for kk in ("matmul", "tensor_bucket", "total", "total_graphed") if s.get(kk)}
            m["peak_fraction"] = {kk: v * 1e12 / peak_macs for kk, v in m["tmacs_per_s"].items()}
        rounds[(t, deg)] = m
        pf = m.get("peak_fraction", {})
        print(f"t=2^{math.log2(t):.0f}={t:>8} deg={deg}  round {m['seconds']['total']*1e3:8.3f} ms  matmul {m['seconds']['matmul']*1e3:8.3f} ms  "
              f"hadamard {m['seconds']['hadamard']*1e3:8.3f} ms  graphed {m['seconds'].get('total_graphed', float('nan'))*1e3:8.3f} ms  "
              f"MACs {m['macs_performed']:.3e} (pad x{m['macs_padded']/max(1, m['macs_performed']):.2f})  "
              f"frac(matmul) {pf.get('matmul', float('nan')):.4f}  frac(tensor) {pf.get('tensor_bucket', float('nan')):.4f}", flush=True)
    ceilings = []
    chunk_loops = []
    if ops.cuda:
        for n in a.ceiling:
            c = time_int_mm(ops, n, n, n, a.repeats)
            c["peak_fraction"] = c["tmacs_per_s"] * 1e12 / peak_macs if peak_macs else None
            ceilings.append(c)
            print(f"_int_mm {n}^3: {c['seconds']*1e3:.3f} ms  {c['tmacs_per_s']:.1f} T MAC/s  frac {c['peak_fraction']}", flush=True)
        for shape in ((48, 32768, 96), (48, 131072, 96), (2**21, 48, 48), (2**21, 48, 96), (24, 32768, 24)):
            c = time_int_mm(ops, *shape, a.repeats)
            c["peak_fraction"] = c["tmacs_per_s"] * 1e12 / peak_macs if peak_macs else None
            ceilings.append(c)
            print(f"_int_mm {shape}: {c['seconds']*1e6:.1f} us  {c['tmacs_per_s']:.2f} T MAC/s  frac {c['peak_fraction']}", flush=True)
        for half in (2**20, 2**16):
            for chunk in (32768, 131072):
                c = time_inner_chunk_loop(ops, half, chunk, a.repeats)
                c["peak_fraction"] = c["tmacs_per_s"] * 1e12 / peak_macs if peak_macs else None
                chunk_loops.append(c)
                print(f"inner chunk loop half={half} chunk={chunk}: {c['seconds']*1e6:.1f} us, {c['launches']} launches, "
                      f"{c['us_per_launch']:.1f} us/launch, frac {c['peak_fraction']}", flush=True)
    agg = aggregate(rounds, peak_macs) if peak_macs else None
    mem = int(k.torch.cuda.max_memory_allocated()) if ops.cuda else None
    doc = {
        "schema": "tc-sumcheck-bench/v1", "wall_seconds": time.time() - t0,
        "hardware": hw, "peaks": peaks, "peak_macs_per_s": peak_macs,
        "software": {"torch": k.torch.__version__ if k.torch else None, "python": platform.python_version(), "backend": ops.backend, "device": ops.device,
                     "commit": git_head()},
        "config": {"chunk": a.chunk, "layout": a.layout, "repeats": a.repeats, "warm": a.warm, "limbs": "signed balanced 8-bit",
                   "inner_impl": kern.inner_impl, "splitk_blocks": k.SPLITK_BLOCKS, "compiled_simt_passes": a.compile,
                   "hadamard": "torch int64 coefficient arithmetic (36 products + 36 reductions per ext x ext)" + (" fused by torch.compile" if a.compile else ", unfused kernels"),
                   "field": f"BabyBear p=2^31-2^27+1, F_p^6 = F_p[X]/(X^6 - {ref.W})", "units": UNITS, "layer_widths": LAYER_WIDTHS,
                   "round_shape": "3 tables (f, g, h), degree 3, 8 triple inner products + 3 folds per round; layer tables units x 2^ceil(log2 width)"},
        "rounds": [{"key": f"n{t}.deg{deg}", **m} for (t, deg), m in rounds.items()],
        "int_mm_ceilings": ceilings, "inner_chunk_loops": chunk_loops,
        "layers": agg, "mem.peak_device_bytes": mem,
    }
    return doc


# ---------------------------------------------------------------------------------------------------------------------
# result envelope
# ---------------------------------------------------------------------------------------------------------------------


def envelope(doc: dict, run_id: str, artifacts: list[str]) -> dict:
    hw = doc["hardware"]
    agg = doc["layers"]
    ms = []

    def m(name, value, unit):
        if value is not None:
            ms.append({"name": name, "value": value, "unit": unit})

    if agg:
        m("t.total", agg["seconds"]["total"], "s")
        m("t.arithmetic", agg["seconds"]["total"], "s")
        for name in ("matmul", "matmul.inner", "matmul.fold", "hadamard", "limb.decompose", "chunk_sum", "reduce.inner", "reduce.fold",
                     "tensor_bucket", "total_graphed"):
            m(f"tc.seconds.{name}", agg["seconds"].get(name), "s")
        for name, v in agg["utilisation"].items():
            m(f"tc.peak_fraction.{name}", v, "ratio")
        m("tc.macs_performed", agg["macs_performed"], "count")
        m("tc.macs_model_padded", agg["macs_model_padded"], "count")
    for r in doc["rounds"]:
        m(f"tc.round.{r['key']}", r["seconds"]["total"], "s")
        if "peak_fraction" in r:
            m(f"tc.peak_fraction.round.{r['key']}.matmul", r["peak_fraction"]["matmul"], "ratio")
    for c in doc["int_mm_ceilings"]:
        if c["M"] == c["K"] == c["N"]:
            m(f"tc.peak_fraction.int_mm_{c['M']}", c["peak_fraction"], "ratio")
    m("mem.peak_device_bytes", doc["mem.peak_device_bytes"], "B")
    fp = {
        "profile": PROFILE, "domain": DOMAIN, "K": K_TARGET, "B": B_TARGET, "N_subbatches": 1,
        "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "zk_mode": "none",
        "security": {"target": -128, "challenge_field": "BabyBear^6 (24 limbs)", "note": "operating point of the priced design; nothing is proved here"},
        "hardware": {"gpu": hw["gpu"], "cpu": hw["cpu"]},
        "software": {"backend": {"name": "backends/gkr/tensor/sumcheck_int8", "commit": doc["software"]["commit"], "version": None},
                     "toolchain": {"torch": doc["software"]["torch"], "python": doc["software"]["python"], "cuda": (hw["gpu"] or {}).get("cuda")}},
        "preprocessing": "none", "retention": "retained", "instances": None,
        "stage": "tc.sumcheck.round", "kind": "kernel measurement (not a proof): Candidate A GKR c-round on INT8 tensor cores, random tables",
        "config": doc["config"], "peaks": doc["peaks"],
    }
    status = "passed" if agg and not any(L["missing"] for L in agg["layers"]) else "not_run"
    return {"schema": "research/result/v0.1", "run_id": run_id, "workload_fingerprint": fp,
            "validation": {"status": status, "checker": "backends/gkr/tensor/tests/test_reference_agreement.py (CUDA path, run before the sweep)",
                           "detail": "kernel == BabyBear^6 reference bit-for-bit on t <= 64; timings are medians of CUDA-event repeats", "evidence": artifacts},
            "measurements": ms, "measurement_files": artifacts, "artifacts": artifacts}


def run_agreement_test() -> str:
    """The CUDA-path agreement test, as evidence in the envelope."""
    env = dict(os.environ, TC_DEVICE="cuda")
    r = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q"], capture_output=True, text=True, env=env, cwd=HERE.parent, check=False)
    return (r.stdout + r.stderr)[-4000:]


def main_measure(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test_log = run_agreement_test() if a.test else "skipped"
    (out / "agreement_test.log").write_text(test_log)
    print(test_log.strip().splitlines()[-1] if test_log.strip() else "(no test output)", flush=True)
    doc = run_campaign(a)
    doc["agreement_test_tail"] = test_log.strip().splitlines()[-1] if test_log.strip() else None
    (out / "bench.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    env = envelope(doc, os.environ.get("RESEARCH_RUN_ID", "local"), ["bench.json", "agreement_test.log"])
    if a.test and "passed" not in (doc["agreement_test_tail"] or "") or (a.test and "failed" in (doc["agreement_test_tail"] or "")):
        env["validation"]["status"] = "failed"
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(env, indent=1, default=str))
    os.replace(tmp, out / "result.json")
    if doc["layers"]:
        print(json.dumps(doc["layers"]["utilisation"], indent=1))
        print(f"layer-model total {doc['layers']['seconds']['total']*1e3:.2f} ms over {doc['layers']['rounds']} rounds; peak device memory {doc['mem.peak_device_bytes']/1e9:.2f} GB")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# re-pricing (laptop side)
# ---------------------------------------------------------------------------------------------------------------------


def reprice(bench_path: Path, report_stem: Path) -> int:
    from verity_numerical.bench import contract
    from verity_numerical.explore import tensor as tx

    doc = json.loads(bench_path.read_text())
    res_path = bench_path.with_name("result.json")
    problems = contract.validate(json.loads(res_path.read_text())) if res_path.exists() else ["result.json not beside bench.json"]
    agg = doc["layers"]
    u = agg["utilisation"]
    peaks = doc["peaks"]
    gpu4090 = tx.Gpu(doc["hardware"]["gpu"]["name"], peaks["int8_dense_tops"], peaks["bf16_dense_tflops"], peaks["fp32_tflops"],
                     peaks["hbm_bytes_per_s"], 1.0e11 * peaks["fp32_tflops"] / 19.5, peaks["source"])
    gpus = {"RTX4090": gpu4090, **tx.GPUS}
    fs = tx.FIELDS["babybear"]
    hr = tx.HASH_RATES["blake3_simt"]["bytes_per_s"]
    rows = {}
    for dname in ("A", "A-hintfree-depth630"):
        p = tx.price_design(tx.DESIGNS[dname], fs, -128)
        rows[dname] = {}
        for gk, gpu in gpus.items():
            rows[dname][gk] = {}
            for label, util in (("0.172 (TensorZKP assumption)", 0.172),
                                ("measured tensor_bucket", u["model_equivalent.tensor_bucket"]),
                                ("measured matmul_only", u["model_equivalent.matmul_only"])):
                tm = p.times(gpu, gk if gk in tx.GPUS else "A100", util, hr, "rs_matrix_ntt", fs)
                rows[dname][gk][label] = {"utilisation": util, "overhead.vs_native_peak": tm["overhead.vs_native_peak"], "t.total": tm["t.total"],
                                          "t.matmul": tm["t.matmul"], "t.simt_field": tm["t.simt_field"], "t.hash": tm["t.hash"],
                                          "matmul_share": tm["matmul_share_of_time"]}
        # the gkr_checker line alone at the measured wall time: replace the model's bucket by the measured layer-model total
        gkr_model_macs = p.matmul["gkr_checker"]
        rows[dname]["gkr_checker_line"] = {
            "model_limb_macs_per_batch": gkr_model_macs, "measured_layer_model_seconds_tensor_bucket": agg["seconds"]["tensor_bucket"],
            "measured_layer_model_seconds_total_with_torch_hadamards": agg["seconds"]["total"],
            "measured_layer_model_seconds_graphed": agg["seconds"].get("total_graphed"),
            "model_seconds_at_0.172_on_4090": gkr_model_macs / (gpu4090.peak_macs_per_s * 0.172),
        }
    out = {"source": {"bench": str(bench_path), "run_id": json.loads(res_path.read_text())["run_id"] if res_path.exists() else None,
                      "commit": doc["software"]["commit"]},
           "labels": {"proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "security.target": -128,
                      "hardware": doc["hardware"], "software": doc["software"]},
           "contract_validation_problems": problems,
           "measured": {"utilisation": u, "layer_model_seconds": agg["seconds"], "macs": {kk: agg[kk] for kk in ("macs_performed", "macs_padded", "macs_model_padded", "macs_model_census")},
                        "rounds": doc["rounds"], "int_mm_ceilings": doc["int_mm_ceilings"], "inner_chunk_loops": doc["inner_chunk_loops"],
                        "mem.peak_device_bytes": doc["mem.peak_device_bytes"]},
           "repriced": rows,
           "scaling_note": "A100/H100 rows apply the 4090-measured utilisation to their INT8 dense peaks (peak-ratio scaling only; "
                           "nothing was measured on those cards; their HBM/launch behaviour differs)."}
    report_stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"utilisation": u, "problems": problems}, indent=1))
    for dname, g in rows.items():
        for gk, labels in g.items():
            if gk == "gkr_checker_line":
                continue
            for label, v in labels.items():
                print(f"{dname:22} {gk:8} {label:32} u={v['utilisation']:.4f}  overhead {v['overhead.vs_native_peak']:.3g}  t.total {v['t.total']:.3f} s")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR") or "backends/gkr/tensor/out")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=k.CHUNK)
    ap.add_argument("--layout", choices=("reduced", "paper"), default="reduced", help="fold output: 48 columns (X^6=W in the matrix) or the paper's 96")
    ap.add_argument("--ext-max-log2", type=int, default=21, help="extension rounds at t = 2^e for e = ext_max_log2 .. 1")
    ap.add_argument("--base-log2", type=int, nargs="*", default=[22, 20], help="first (base-field) rounds at t = 2^e")
    ap.add_argument("--ceiling", type=int, nargs="*", default=[8192, 4096])
    ap.add_argument("--no-test", dest="test", action="store_false")
    ap.add_argument("--inner", choices=("auto", "int_mm", "triton"), default="auto", help="inner-product matmul: chunked torch._int_mm or the Triton split-K kernel")
    ap.add_argument("--compile", action="store_true", help="torch.compile the SIMT passes (Hadamards, limb passes)")
    ap.add_argument("--reprice", type=Path, default=None, help="laptop: a fetched bench.json")
    ap.add_argument("--report", type=Path, default=Path("backends/numerical/reports/tc_sumcheck_4090"))
    a = ap.parse_args(argv)
    if a.reprice:
        return reprice(a.reprice, a.report)
    return main_measure(a)


if __name__ == "__main__":
    sys.exit(main())
