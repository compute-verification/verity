"""Measure the packed sumcheck (``sumcheck_packed``) on one GPU: per-round times for k in {0..4}, the whole sumcheck at
several table sizes, a linear fit -> A-GKR's per-VU checker-sumcheck seconds, achieved fraction of the INT8 peak,
memory; writes ``bench.json`` and the ``research/result/v0.1`` envelope.  ``--reprice`` (laptop, needs
``verity_numerical``) turns a fetched ``bench.json`` into the re-priced A-GKR overheads.

Pod::

    /workspace/venv312/bin/python archive/gkr/packed/bench.py --out "$RESEARCH_RUN_DIR" --log2 22 24 --ks 0 1 2 3 4

Laptop::

    uv run python archive/gkr/packed/bench.py --reprice <run>/bench.json --report backends/numerical/reports/packed_sumcheck_h100

Every number is ``ARITHMETIC_DIAGNOSTIC``, ``authentication excluded``, ``security.target = -128`` (BabyBear^6): a
kernel measurement on random tables, not a proof.  Timing: CUDA events; a round is captured in a CUDA graph and
replayed (median of ``--repeats`` after ``--warm``), so launch gaps are excluded; eager segment times (Hadamard+limbs,
matmul, reduce, fold) are recorded beside it.  MACs are the ``M K N`` of the matmuls as issued (``Stats.macs_performed``,
the cost model's unit); padding to cuBLAS's minimum shapes is reported separately.
"""
from __future__ import annotations

# ruff: noqa: B023 F821  -- the timing lambdas are called inside the loop iteration that builds them; ruff misreads the
# later ``del g, h`` (memory) as making the names undefined.
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from packed import reference as ref
from packed import sumcheck_packed as sp

#: Datasheet peaks (dense, no sparsity).  H100 SXM: INT8 1979 TOPS = 989.5 T MAC/s, HBM3 3.35 TB/s.
GPU_PEAKS = {
    "NVIDIA H100 80GB HBM3": {"int8_dense_tops": 1979.0, "hbm_bytes_per_s": 3.35e12, "fp32_tflops": 67.0, "bf16_dense_tflops": 989.0,
                              "source": "NVIDIA H100 Tensor Core GPU datasheet (SXM5): INT8 1,979 TOPS dense, HBM3 3.35 TB/s, FP32 67 TFLOPS"},
    "NVIDIA GeForce RTX 4090": {"int8_dense_tops": 660.6, "hbm_bytes_per_s": 1008e9, "fp32_tflops": 82.6, "bf16_dense_tflops": 165.2,
                                "source": "NVIDIA Ada whitepaper / RTX 4090 datasheet (dense)"},
    # anchor-a100 lane: the campaign anchor (contract.NATIVE_REFERENCE is this card's BF16 peak)
    "NVIDIA A100-SXM4-80GB": {"int8_dense_tops": 624.0, "hbm_bytes_per_s": 2.039e12, "fp32_tflops": 19.5, "bf16_dense_tflops": 312.0,
                              "source": "NVIDIA A100 Tensor Core GPU datasheet (SXM4 80GB): INT8 624 TOPS dense, HBM2e 2,039 GB/s, FP32 19.5 TFLOPS"},
}
PROFILE = "first-campaign-target/2026-09-21"   # verity.ml.circuits.targets.target.FIRST.name (copied: the pod has no workspace install)
DOMAIN = "finite"
K_TARGET, B_TARGET = 1536, 4096
UNITS = B_TARGET * K_TARGET // 16              # 393,216 transition units in the batch
#: A-GKR's checker layers (``accounting.CHECKER_LAYER_WIDTHS_A``, output layer first): carried width per unit;
#: the c-round tables of layer i are ``units x (input width)`` (PROTOCOL.md 1: reads 2^10 for the computed layers, 2^9 for depth 1).
LAYER_WIDTHS = (137, 679, 673, 669, 647, 533)
LAYER_READ_LOG2 = (10, 10, 10, 10, 10, 9)
DL = sp.DL


# ---------------------------------------------------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------------------------------------------------


def probe_hardware() -> dict:
    gpu = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10, check=False).stdout.strip().splitlines()
        if q:
            name, drv, mem = [x.strip() for x in q[0].split(",")]
            gpu = {"name": name, "driver": drv, "memory": mem, "cuda": None}
    except (OSError, subprocess.TimeoutExpired):
        pass
    cpu = {"model": platform.processor() or platform.machine(), "count": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu["model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"gpu": gpu, "cpu": cpu}


def git_commit() -> str | None:
    for env in ("PACKED_COMMIT", "RESEARCH_SOURCE_SHA"):
        if os.environ.get(env):
            return os.environ[env]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, timeout=5, check=False).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------------------------------------------


def _median_events(fn, repeats: int, warm: int) -> float:
    import torch

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


def time_graphed(fn, repeats: int, warm: int) -> tuple[float, str]:
    """Median seconds of ``fn`` replayed as a CUDA graph (falls back to eager timing when capture fails)."""
    import torch

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        t = _median_events(graph.replay, repeats, warm)
        del graph
        return t, "cuda_graph"
    except Exception as e:  # noqa: BLE001 -- report the fallback, do not hide it
        return _median_events(fn, repeats, warm), f"eager ({type(e).__name__})"


def segments(kernel: sp.Kernel, fn_with_marks) -> dict[str, float]:
    """Eager segment seconds from the round's CUDA-event marks."""
    import torch

    marks: dict = {}
    start = kernel._mark()
    fn_with_marks(marks)
    torch.cuda.synchronize()
    out, prev = {}, start
    for name, ev in marks.items():
        out[name] = prev.elapsed_time(ev) / 1e3
        prev = ev
    return out


# ---------------------------------------------------------------------------------------------------------------------
# one sumcheck
# ---------------------------------------------------------------------------------------------------------------------


def run_sumcheck(kernel: sp.Kernel, n: int, k: int, repeats: int, warm: int, seed: int, peak_macs: float) -> dict:
    """All rounds of one (t = 2^n, k) sumcheck: per-round graphed time, eager segments, MACs, utilisation."""
    import torch

    dev = kernel.device
    rng_t = torch.Generator(device="cpu").manual_seed(seed)
    t = 1 << n
    g = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=rng_t).to(torch.int32).to(dev)
    h = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=rng_t).to(torch.int32).to(dev)
    rng = __import__("random").Random(seed + 1)
    gammas = [ref.random_ext(rng) for _ in range(n)]
    rs_all = [ref.random_ext(rng) for _ in range(n)]      # the coins are drawn ahead: timing does not wait on the transcript
    rounds = []
    rs: list[ref.Ext] = []
    torch.cuda.reset_peak_memory_stats(dev)
    total = 0.0
    total_mm = 0.0
    macs = 0
    macs_padded = 0
    # packed rounds
    for i in range(1, k + 1):
        inp = kernel.inputs(gammas, rs, i, k)
        st = sp.Stats()
        kernel.packed_round(g, h, i, inp, st)
        tm, how = time_graphed(lambda: kernel.packed_round(g, h, i, inp), repeats, warm)
        seg = segments(kernel, lambda marks: kernel.packed_round(g, h, i, inp, None, marks))
        if any(lbl == "inner.packed.fused" for (lbl, *_rest) in st.matmuls):
            # the fused kernel IS the matmul (Hadamard limbs formed in registers): time it alone
            K1, K2 = kernel.split(t >> i)
            split = kernel.split_b1(K1, K2)
            t_mm = _median_events(lambda: kernel._tk.packed_inner_product(g, h, inp.e_lo32, i, K1, K2, split), repeats, warm)
        else:
            A = kernel._packed_hadamard(g, h, i, *kernel.split(t >> i))
            t_mm = _median_events(lambda: sp.mm_int8(A, inp.e_lo_limbs), repeats, warm)
            del A
        rounds.append(_round_row(i, "packed", t >> (i - 1), tm, how, seg, st, t_mm, peak_macs))
        total += tm
        total_mm += t_mm
        macs += st.macs_performed
        macs_padded += st.macs_padded
        rs.append(rs_all[i - 1])
    # the fold by k challenges
    if k:
        kernel.set_fold(inp, rs[:-1], k, k, rs[-1])
        C = inp.fold_C
        st = sp.Stats()
        gE = kernel.fold_k(g, k, inp, st)
        hE = kernel.fold_k(h, k, inp, st)
        tm, how = time_graphed(lambda: (kernel.fold_k(g, k, inp), kernel.fold_k(h, k, inp)), repeats, warm)
        L = sp.decompose(g.view(t >> k, 1 << k))
        t_mm = 2 * _median_events(lambda: sp.mm_int8(L, C), repeats, warm)
        del L
        rounds.append(_round_row(k, "fold_k", t, tm, how, {}, st, t_mm, peak_macs))
        total += tm
        total_mm += t_mm
        macs += st.macs_performed
        macs_padded += st.macs_padded
    else:
        gE = torch.zeros((t, sp.DEGREE), dtype=torch.int32, device=dev)
        gE[:, 0] = g
        hE = torch.zeros_like(gE)
        hE[:, 0] = h
    del g, h
    # extension rounds (the fold by r_i is part of the round)
    for i in range(k + 1, n + 1):
        r_i = rs_all[i - 1]
        inp = kernel.inputs(gammas, rs, i, k, r_i)
        st = sp.Stats()
        _, g2, h2 = kernel.ext_round(gE, hE, i, inp, st)
        tm, how = time_graphed(lambda: kernel.ext_round(gE, hE, i, inp), repeats, warm)
        seg = segments(kernel, lambda marks: kernel.ext_round(gE, hE, i, inp, None, marks))
        m = gE.shape[0] // 2
        L = sp.decompose(torch.stack([gE, hE], 0).view(2 * m, 2 * sp.DEGREE))
        if any(lbl == "inner.ext.fused" for (lbl, *_rest) in st.matmuls):
            K1, K2 = kernel.split(m)
            split = kernel.split_b1(K1, K2)
            t_mm = _median_events(lambda: kernel._tk.ext_inner_product(gE, hE, inp.e_lo32, K1, K2, split), repeats, warm)
        else:
            A = kernel._ext_hadamard3(gE, hE, *kernel.split(m))
            t_mm = _median_events(lambda: sp.mm_int8(A, inp.e_lo_limbs), repeats, warm)
            del A
        t_mm += _median_events(lambda: sp.mm_int8(L, inp.fold_C), repeats, warm)
        del L
        rounds.append(_round_row(i, "ext", gE.shape[0], tm, how, seg, st, t_mm, peak_macs))
        total += tm
        total_mm += t_mm
        macs += st.macs_performed
        macs_padded += st.macs_padded
        rs.append(r_i)
        gE, hE = g2, h2
    mem = torch.cuda.max_memory_allocated(dev)
    del gE, hE
    torch.cuda.empty_cache()
    return {"log2_t": n, "k": k, "seconds": total, "seconds_matmul_only": total_mm, "macs_performed": macs, "macs_padded": macs_padded,
            "utilisation.round_graphed": macs / total / peak_macs, "utilisation.matmul_only": macs / total_mm / peak_macs,
            "mem.peak_device_bytes": mem, "rounds": rounds}


def _round_row(i, kind, t_i, tm, how, seg, st: sp.Stats, t_mm, peak_macs) -> dict:
    return {"i": i, "kind": kind, "table_elements": t_i, "seconds": tm, "timing": how, "segments_eager": seg,
            "seconds_matmul_only": t_mm, "macs_performed": st.macs_performed, "macs_padded": st.macs_padded,
            "matmuls": [{"label": l, "M": M, "K": K, "N": N} for (l, M, K, N) in st.matmuls],
            "hadamard_base": st.hadamard_base, "hadamard_ext": st.hadamard_ext,
            "utilisation.round": st.macs_performed / tm / peak_macs if tm else None,
            "utilisation.matmul_only": st.macs_performed / t_mm / peak_macs if t_mm else None}


# ---------------------------------------------------------------------------------------------------------------------
# layer model
# ---------------------------------------------------------------------------------------------------------------------


def linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares ``seconds = a t + b`` over (t, seconds)."""
    if len(points) == 1:
        return points[0][1] / points[0][0], 0.0
    xs, ys = zip(*points)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    a = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return a, my - a * mx


def layer_model(results: list[dict], units: int = UNITS) -> dict:
    """Per k: the batch's six c-round sumchecks (tables ``units x 2^read`` padded, and ``units x width`` un-padded) from
    the linear fit of the measured whole-sumcheck seconds, and the per-VU checker-sumcheck seconds."""
    out = {}
    for k in sorted({r["k"] for r in results}):
        pts = sorted((1 << r["log2_t"], r["seconds"]) for r in results if r["k"] == k)
        a, b = linear_fit(pts)
        padded = [units * (1 << rd) for rd in LAYER_READ_LOG2]
        unpadded = [units * w for w in LAYER_WIDTHS]
        t_pad = sum(a * t + b for t in padded)
        t_unp = sum(a * t + b for t in unpadded)
        pts_mm = sorted((1 << r["log2_t"], r["seconds_matmul_only"]) for r in results if r["k"] == k)
        am, bm = linear_fit(pts_mm)
        macs_pts = sorted((1 << r["log2_t"], r["macs_performed"]) for r in results if r["k"] == k)
        ma, mb = linear_fit(macs_pts)
        out[str(k)] = {"fit.seconds_per_element": a, "fit.intercept_seconds": b, "fit.points": pts,
                       "layer_tables_padded": padded, "layer_tables_unpadded": unpadded,
                       "batch_seconds.padded": t_pad, "batch_seconds.unpadded": t_unp,
                       "per_vu_seconds.padded": t_pad / B_TARGET, "per_vu_seconds.unpadded": t_unp / B_TARGET,
                       "batch_seconds_matmul_only.padded": sum(am * t + bm for t in padded),
                       "batch_macs.padded": sum(ma * t + mb for t in padded), "batch_macs.unpadded": sum(ma * t + mb for t in unpadded)}
    return out


def mac_model(n: int, k: int) -> dict:
    """The limb-MAC count of the algorithm at ``t = 2^n`` (DESIGN.md 4): packed rounds ``3 . 4^(i-1) . (t / 2^i) . 96``,
    fold_k ``2 . 192 t``, extension rounds ``(t / 2^i) (3 . 576 + 2 . 2304)``; the ordinary (eq as a folded table)
    algorithm as ``explore.tensor.sumcheck_macs`` prices it: ``t . 576 . 32`` (eq-table fold and 8 inner products)."""
    t = 1 << n
    packed = sum(3 * 4 ** (i - 1) * (t >> i) * 96 for i in range(1, k + 1))
    fold = 2 * 192 * t if k else 0
    ext = sum((t >> i) * (3 * 576 + 2 * 2304) for i in range(k + 1, n + 1))
    if k == 0:
        ext = sum((t >> i) * (3 * 576 + 2 * 2304) for i in range(1, n + 1))
    return {"packed_rounds": packed, "fold_k": fold, "ext_rounds": ext, "total": packed + fold + ext,
            "ordinary_eq_table_model": t * 576 * 32, "ordinary_eq_table_model_first_round_credited": t * 576 * 32 - t * (576 - 96) * 4}


# ---------------------------------------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------------------------------------


def run_campaign(a) -> dict:
    import torch

    hw = probe_hardware()
    gpu_name = (hw["gpu"] or {}).get("name")
    peaks = GPU_PEAKS.get(gpu_name)
    if peaks is None:
        raise SystemExit(f"no datasheet peak for {gpu_name!r}: add it to GPU_PEAKS")
    peak_macs = peaks["int8_dense_tops"] * 1e12 / 2
    if hw["gpu"]:
        hw["gpu"]["cuda"] = torch.version.cuda
    kernel = sp.Kernel("cuda", k1_log2=a.k1_log2, mode=a.mode, fused=a.fused)
    results = []
    t0 = time.time()
    for n in a.log2:
        for k in a.ks:
            r = run_sumcheck(kernel, n, k, a.repeats, a.warm, seed=11 + n, peak_macs=peak_macs)
            r["mac_model"] = mac_model(n, k)
            results.append(r)
            print(f"t=2^{n} k={k}: {r['seconds']*1e3:8.3f} ms  matmul-only {r['seconds_matmul_only']*1e3:8.3f} ms  "
                  f"util round {r['utilisation.round_graphed']:.4f} matmul {r['utilisation.matmul_only']:.4f}  "
                  f"MACs {r['macs_performed']:.3e} (model {r['mac_model']['total']:.3e})  mem {r['mem.peak_device_bytes']/1e9:.2f} GB  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    doc = {
        "hardware": hw, "peaks": peaks,
        "software": {"torch": torch.__version__, "python": platform.python_version(), "commit": git_commit(), "cuda": torch.version.cuda},
        "config": {"log2": a.log2, "ks": a.ks, "repeats": a.repeats, "warm": a.warm, "k1_log2": a.k1_log2, "mode": a.mode, "fused": a.fused,
                   "units": UNITS, "layer_widths": LAYER_WIDTHS, "layer_read_log2": LAYER_READ_LOG2, "B": B_TARGET, "K": K_TARGET},
        "results": results,
        "layers": layer_model(results),
    }
    return doc


def envelope(doc: dict, run_id: str, artifacts: list[str], test_tail: str | None) -> dict:
    hw = doc["hardware"]
    ms = []

    def m(name, value, unit):
        if value is not None:
            ms.append({"name": name, "value": value, "unit": unit})

    layers = doc["layers"]
    best_k = min(layers, key=lambda kk: layers[kk]["batch_seconds.padded"])
    best = layers[best_k]
    m("t.total", best["batch_seconds.padded"], "s")
    m("t.arithmetic", best["batch_seconds.padded"], "s")
    m("mem.peak_device_bytes", max(r["mem.peak_device_bytes"] for r in doc["results"]), "B")
    m("packed.best_k", int(best_k), "count")
    for kk, L in layers.items():
        m(f"packed.k{kk}.batch_seconds.padded", L["batch_seconds.padded"], "s")
        m(f"packed.k{kk}.batch_seconds.unpadded", L["batch_seconds.unpadded"], "s")
        m(f"packed.k{kk}.per_vu_seconds.padded", L["per_vu_seconds.padded"], "s")
        m(f"packed.k{kk}.fit.seconds_per_element", L["fit.seconds_per_element"], "s")
    for r in doc["results"]:
        m(f"packed.t2^{r['log2_t']}.k{r['k']}.seconds", r["seconds"], "s")
        m(f"packed.t2^{r['log2_t']}.k{r['k']}.seconds_matmul_only", r["seconds_matmul_only"], "s")
        m(f"packed.t2^{r['log2_t']}.k{r['k']}.utilisation.round", r["utilisation.round_graphed"], "ratio")
        m(f"packed.t2^{r['log2_t']}.k{r['k']}.utilisation.matmul_only", r["utilisation.matmul_only"], "ratio")
        m(f"packed.t2^{r['log2_t']}.k{r['k']}.macs_performed", r["macs_performed"], "count")
    fp = {
        "profile": PROFILE, "domain": DOMAIN, "K": K_TARGET, "B": B_TARGET, "N_subbatches": 1,
        "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "zk_mode": "none",
        "security": {"target": -128, "challenge_field": "BabyBear^6 (24 limbs)", "note": "operating point of the priced design; nothing is proved here"},
        "hardware": {"gpu": hw["gpu"], "cpu": hw["cpu"]},
        "software": {"backend": {"name": "archive/gkr/packed/sumcheck_packed", "commit": doc["software"]["commit"], "version": None},
                     "toolchain": {"torch": doc["software"]["torch"], "python": doc["software"]["python"], "cuda": doc["software"]["cuda"]}},
        "preprocessing": "none", "retention": "retained", "instances": None,
        "stage": "packed.sumcheck", "kind": "kernel measurement (not a proof): Candidate A GKR c-round sumcheck (eq-trick, k packed rounds) "
                                            "on INT8 tensor cores, random tables; t.total is the six-layer batch model from the linear fit",
        "config": doc["config"], "peaks": doc["peaks"],
    }
    status = "passed" if (test_tail and "passed" in test_tail and "failed" not in test_tail) else ("not_run" if test_tail is None else "failed")
    return {"schema": "research/result/v0.1", "run_id": run_id, "workload_fingerprint": fp,
            "validation": {"status": status, "checker": "archive/gkr/packed/tests (CUDA path, run before the sweep)",
                           "detail": f"kernel == BabyBear^6 reference bit-for-bit for k = 0..4, n <= 9; {test_tail}", "evidence": artifacts},
            "measurements": ms, "measurement_files": artifacts, "artifacts": artifacts}


def run_agreement_test() -> str:
    env = dict(os.environ, PACKED_DEVICE="cuda")
    r = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q"], capture_output=True, text=True, env=env, cwd=HERE.parent, check=False)
    return (r.stdout + r.stderr)[-4000:]


def main_measure(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tail = None
    if a.test:
        log = run_agreement_test()
        (out / "agreement_test.log").write_text(log)
        tail = log.strip().splitlines()[-1] if log.strip() else "(no output)"
        print(tail, flush=True)
    doc = run_campaign(a)
    doc["agreement_test_tail"] = tail
    (out / "bench.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    arts = ["bench.json"] + (["agreement_test.log"] if a.test else [])
    env = envelope(doc, os.environ.get("RESEARCH_RUN_ID", "local"), arts, tail)
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(env, indent=1, default=str))
    os.replace(tmp, out / "result.json")
    for kk, L in doc["layers"].items():
        print(f"k={kk}: batch {L['batch_seconds.padded']*1e3:.1f} ms (padded) / {L['batch_seconds.unpadded']*1e3:.1f} ms (unpadded); "
              f"per VU {L['per_vu_seconds.padded']*1e6:.2f} us; fit {L['fit.seconds_per_element']*1e9:.3f} ns/elem + {L['fit.intercept_seconds']*1e3:.3f} ms")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# re-pricing (laptop side)
# ---------------------------------------------------------------------------------------------------------------------


def reprice(bench_path: Path, report_stem: Path) -> int:
    from verity_numerical.bench import contract
    from verity_numerical.explore import tensor as tx

    doc = json.loads(bench_path.read_text())
    res_path = bench_path.with_name("result.json")
    run_id = json.loads(res_path.read_text())["run_id"] if res_path.exists() else None
    problems = contract.validate(json.loads(res_path.read_text())) if res_path.exists() else ["result.json not beside bench.json"]
    fs = tx.FIELDS["babybear"]
    hr = tx.HASH_RATES["blake3_simt"]["bytes_per_s"]
    gpu = tx.GPUS["H100"]
    layers = doc["layers"]
    rows = {}
    for dname in ("A", "A-hintfree-depth630"):
        p = tx.price_design(tx.DESIGNS[dname], fs, -128)
        base = p.times(gpu, "H100", 0.172, hr, "rs_matrix_ntt", fs)
        model_gkr_seconds = p.matmul["gkr_checker"] / (gpu.peak_macs_per_s * 0.172)
        rows[dname] = {"model@0.172": {"overhead": base["overhead.vs_native_peak"], "t.total": base["t.total"], "t.gkr_checker_matmul": model_gkr_seconds,
                                       "t.gkr_hadamards_simt": p.simt_field_mults["gkr_hadamards"] * fs.simt_mult_vs_goldilocks / gpu.simt_field_mults_per_s}}
        for kk, L in layers.items():
            # substitute the measured six-layer sumcheck (matmul + Hadamards + limb passes, all measured) for the model's
            # gkr_checker matmul bucket AND its SIMT Hadamard bucket; every other bucket unchanged
            simt_gkr = p.simt_field_mults["gkr_hadamards"] * fs.simt_mult_vs_goldilocks / gpu.simt_field_mults_per_s
            for tag, meas in (("padded", L["batch_seconds.padded"]), ("unpadded", L["batch_seconds.unpadded"])):
                total = base["t.total"] - model_gkr_seconds - simt_gkr + meas
                rows[dname][f"measured k={kk} ({tag})"] = {"overhead": contract.overhead(total / tx.B_TARGET, tx.K_TARGET), "t.total": total,
                                                           "t.gkr_checker_measured": meas, "gkr_share": meas / total}
    out = {"source": {"bench": str(bench_path), "run_id": run_id, "commit": doc["software"]["commit"]},
           "labels": {"proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "security.target": -128,
                      "hardware": doc["hardware"], "software": doc["software"]},
           "contract_validation_problems": problems, "layers": layers,
           "results_summary": [{k: r[k] for k in ("log2_t", "k", "seconds", "seconds_matmul_only", "macs_performed", "utilisation.round_graphed",
                                                  "utilisation.matmul_only", "mem.peak_device_bytes", "mac_model")} for r in doc["results"]],
           "repriced": rows,
           "note": "the model's gkr_checker matmul bucket (5.4e9 MACs/VU at 0.172) and its SIMT Hadamard bucket are replaced by the measured "
                   "six-layer sumcheck time (linear fit over the measured table sizes, tables units x 2^read per layer); logUp GKR, encoding, "
                   "row combination, hashing and hint generation stay modelled."}
    report_stem.parent.mkdir(parents=True, exist_ok=True)
    report_stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"problems": problems}, indent=1))
    for dname, g in rows.items():
        for label, v in g.items():
            extra = f"  gkr {v['t.gkr_checker_matmul']*1e3:.0f} ms + hadamards {v['t.gkr_hadamards_simt']*1e3:.0f} ms" if "t.gkr_checker_matmul" in v else f"  gkr measured {v['t.gkr_checker_measured']*1e3:.0f} ms ({v['gkr_share']:.0%})"
            print(f"{dname:22} {label:28} overhead {v['overhead']:.3g}  t.total {v['t.total']:.3f} s{extra}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR") or "archive/gkr/packed/out")
    ap.add_argument("--log2", type=int, nargs="*", default=[22, 24])
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--k1-log2", type=int, default=sp.K1_LOG2_DEFAULT)
    ap.add_argument("--mode", choices=["eager", "compile", "triton"], default="eager", help="SIMT passes: torch eager, torch.compile, or fused Triton kernels")
    ap.add_argument("--no-fused", dest="fused", action="store_false", help="triton mode: two-pass packed rounds (limbs to HBM, cuBLAS)")
    ap.add_argument("--no-test", dest="test", action="store_false")
    ap.add_argument("--reprice", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("backends/numerical/reports/packed_sumcheck_h100"))
    a = ap.parse_args(argv)
    if a.reprice:
        return reprice(a.reprice, a.report)
    return main_measure(a)


if __name__ == "__main__":
    raise SystemExit(main())
