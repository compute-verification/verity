"""native_peak: the card's *measured* dense-GEMM throughput for one dtype, as a `native-peak/v1` result.

    python tools/native_peak/measure.py --dtype bf16  [--sizes 8192,16384] [--warmup 3] [--reps 5] [--sustain-s 2.0]
    python tools/native_peak/measure.py --dtype e4m3   # FP8 e4m3 inputs, FP32 accumulate (torch._scaled_mm); skips cleanly where unsupported

One run = one dtype = one artifact.  For each square size n the tool runs `--warmup` untimed GEMMs, a `--sustain-s` phase of
back-to-back GEMMs (so clocks and power settle where they will sit under load, sampled by `nvidia-smi` every 100 ms), then
`--reps` timed GEMMs bracketed by CUDA events; the reported number is `2 n^3 / median(reps)`.  BF16 goes through `torch.matmul`
(cuBLAS, FP32 accumulate is cuBLAS's default for bf16 and reduced-precision reduction is switched off explicitly); e4m3 through
`torch._scaled_mm` with unit scales and `out_dtype=torch.float32`.  A small sub-block of every timed product is checked against an
FP32 reference (`validation`).  The cuBLAS kernel name comes from `torch.profiler`; the SASS opcode is taken from `ncu` only if
that binary is already on the machine, otherwise it is recorded as "not captured".

result.json (`research/result/v0.1`, kind `native-peak/v1`) carries the headline fields at the top level -- device, dtype,
accumulate, method, flop_per_second, spec_flop_per_second, fraction_of_spec, sass, kernel, driver, cuda, clocks_mhz, power_w,
sizes -- so `research data select --where meta.dtype=bf16` works, and the same identity in `workload_fingerprint`.  Spec peaks are
datasheet dense (non-sparse) tensor-core numbers with FP32 accumulate; a card the table does not know gets `spec_flop_per_second:
null`.  Nothing here is a proof; it is the native anchor for `overhead.vs_native_peak`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULT_SCHEMA = "research/result/v0.1"
KIND = "native-peak/v1"
DTYPES = ("bf16", "e4m3")

#: Datasheet dense tensor-core peaks (flop/s) with FP32 accumulate, keyed by a regex on `torch.cuda.get_device_name`.  First match
#: wins.  RTX 4090: NVIDIA Ada GPU architecture whitepaper (BF16 with FP32 acc 165.2 TFLOPS, FP8 with FP32 acc 330.3 TFLOPS,
#: boost clock, dense).  A100 SXM4 / PCIe 80GB: NVIDIA A100 datasheet (BF16 312 TFLOPS dense; no FP8 tensor cores).
SPEC_PEAKS: list[tuple[str, dict[str, float | None], str]] = [
    (r"RTX\s*4090", {"bf16": 165.2e12, "e4m3": 330.3e12}, "NVIDIA Ada GPU Architecture whitepaper, RTX 4090 dense, FP32 accumulate"),
    (r"A100-SXM4", {"bf16": 312.0e12, "e4m3": None}, "NVIDIA A100 Tensor Core GPU datasheet (SXM4 80GB), BF16 dense; Ampere has no FP8"),
    (r"A100.*PCIe", {"bf16": 312.0e12, "e4m3": None}, "NVIDIA A100 Tensor Core GPU datasheet (PCIe 80GB), BF16 dense; Ampere has no FP8"),
]

SMI_FIELDS = ("timestamp", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "clocks_throttle_reasons.active")


def spec_peak(device_name: str, dtype: str) -> tuple[float | None, str | None]:
    """(flop/s, source) for the datasheet peak of `dtype` on `device_name`; (None, None) when the card is not in the table."""
    for pat, peaks, source in SPEC_PEAKS:
        if re.search(pat, device_name or "", re.I):
            return peaks.get(dtype), source
    return None, None


def parse_sizes(text: str) -> list[int]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            n = int(tok)
            if n <= 0 or n % 16:
                raise ValueError(f"size {n} must be a positive multiple of 16")
            out.append(n)
    if not out:
        raise ValueError("--sizes is empty")
    return out


# ---- nvidia-smi sampling ----------------------------------------------------------------------------------------------------------------------

class SmiSampler:
    """`nvidia-smi --query-gpu=... -lms 100` in the background; `window(t0, t1)` summarises the samples between two wall-clock times."""

    def __init__(self, path: Path, interval_ms: int = 100):
        self.path = path
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.handle = None
        self.rows: list[dict] = []

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        cmd = ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)]
        self.handle = open(self.path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.handle, stderr=subprocess.DEVNULL, text=True)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.path.is_file():
            self.rows = parse_smi_csv(self.path.read_text(errors="replace"))

    def window(self, t0: float, t1: float) -> dict:
        return summarize_samples([r for r in self.rows if r["t"] is not None and t0 <= r["t"] <= t1])


def _smi_time(text: str) -> float | None:
    """nvidia-smi prints local time `YYYY/MM/DD HH:MM:SS.mmm`; `time.mktime` maps that back to the epoch `time.time()` uses."""
    base, _, frac = text.strip().partition(".")
    try:
        t = time.mktime(time.strptime(base, "%Y/%m/%d %H:%M:%S"))
    except ValueError:
        return None
    return t + (float("0." + frac) if frac.isdigit() else 0.0)


def parse_smi_csv(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        def num(s):
            try:
                return float(s)
            except ValueError:
                return None
        rows.append({"t": _smi_time(parts[0]), "clocks_sm_mhz": num(parts[1]), "clocks_mem_mhz": num(parts[2]), "power_w": num(parts[3]),
                     "temperature_c": num(parts[4]) if len(parts) > 4 else None, "throttle_reasons": parts[5] if len(parts) > 5 else None})
    return rows


def summarize_samples(rows: list[dict]) -> dict:
    def stats(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    reasons = sorted({r["throttle_reasons"] for r in rows if r.get("throttle_reasons")})
    return {"n_samples": len(rows), "clocks_sm_mhz": stats("clocks_sm_mhz"), "clocks_mem_mhz": stats("clocks_mem_mhz"), "power_w": stats("power_w"),
            "temperature_c": stats("temperature_c"), "throttle_reasons": reasons}


# ---- the GEMMs ---------------------------------------------------------------------------------------------------------------------------------

CHECK_BLOCK = 256


def _make_operands(n: int, dtype: str, device: str):
    """(a, b, ref_a, ref_b): the operands in `dtype` and fp32 copies of the rows of a / columns of b the check needs (same rounding)."""
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(20260922)
    a32 = torch.randn(n, n, device=device, generator=g)
    b32 = torch.randn(n, n, device=device, generator=g)
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float8_e4m3fn   # randn magnitudes (< ~6) sit comfortably inside e4m3's range
    a, b = a32.to(tdt), b32.to(tdt)
    del a32, b32
    if dtype != "bf16":
        b = b.t()          # cuBLASLt's FP8 path wants mat2 column-major: the operand the GEMM multiplies by IS this transposed view
    ref_a, ref_b = a[:CHECK_BLOCK].float(), b[:, :CHECK_BLOCK].float()
    return a, b, ref_a, ref_b


def _gemm_fn(dtype: str, a, b, device: str):
    import torch

    if dtype == "bf16":
        return lambda: torch.matmul(a, b)
    one = torch.ones((), device=device, dtype=torch.float32)

    def fn():
        out = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
        return out[0] if isinstance(out, tuple) else out
    return fn


def _check(out, ref_a, ref_b, dtype: str, block: int = CHECK_BLOCK) -> dict:
    """max relative error of the `block x block` corner of `out` against an FP32 product of the same rounded operands."""
    ref = ref_a @ ref_b                          # fp32 on the device (TF32 is off)
    got = out[:block, :block].float()
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item() or 1.0
    # bf16: the OUTPUT is rounded to bf16 (8 bits) -> ~2.5e-3 observed.  e4m3 -> fp32 output: the tensor core's "fp32" accumulate is not an
    # IEEE sequential sum (alignment bits are truncated inside the MMA); 1.4e-4 observed on sm_89 at K=8192, so 1e-3 is a sanity bound
    # (right operand, right layout), not an exactness claim.  The measured max_rel_err is recorded either way.
    tol = 2e-2 if dtype == "bf16" else 1e-3
    rel = err / scale
    return {"max_abs_err": err, "ref_abs_max": scale, "max_rel_err": rel, "tolerance": tol, "ok": rel <= tol, "block": block,
            "what": "out[:block,:block] vs fp32 (a[:block] @ b[:, :block]) computed on the device"}


def kernel_names(fn) -> list[str]:
    """The CUDA kernels one call launches, by `torch.profiler` (cuBLAS's kernel name encodes the arch/dtype/tile, e.g. sm80_xmma_gemm_bf16bf16_bf16f32_f32...)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = []
        for ev in prof.events():
            if getattr(ev, "device_type", None) is not None and "CUDA" in str(ev.device_type) and ev.name not in names:
                names.append(ev.name)
        return names
    except Exception as ex:  # noqa: BLE001  profiling is a convenience, never a reason to fail the measurement
        return [f"not captured ({type(ex).__name__}: {ex})"]


def measure_size(n: int, dtype: str, device: str, warmup: int, reps: int, sustain_s: float) -> dict:
    import torch

    a, b, ref_a, ref_b = _make_operands(n, dtype, device)
    fn = _gemm_fn(dtype, a, b, device)
    flops = 2.0 * n ** 3
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # sustain: back-to-back GEMMs for >= sustain_s so clocks/power settle; also yields a sustained-throughput number
    t_sus0 = time.time()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    k = 0
    while True:
        fn()
        k += 1
        if k % 4 == 0:
            torch.cuda.synchronize()
            if time.time() - t_sus0 >= sustain_s:
                break
    e_ev.record()
    e_ev.synchronize()
    sustain_wall = s_ev.elapsed_time(e_ev) / 1e3
    # timed reps
    t_rep0 = time.time()
    times = []
    out = None
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    t_rep1 = time.time()
    med = statistics.median(times)
    check = _check(out, ref_a, ref_b, dtype)
    names = kernel_names(fn)
    peak_mem = torch.cuda.max_memory_allocated()
    del a, b, ref_a, ref_b, out
    torch.cuda.empty_cache()
    return {"n": n, "m": n, "k": n, "flop": flops, "warmup": warmup, "reps": reps, "seconds_all": times, "seconds_median": med,
            "seconds_min": min(times), "seconds_max": max(times), "flop_per_second": flops / med,
            "sustain": {"seconds": sustain_wall, "iterations": k, "flop_per_second": flops * k / sustain_wall},
            "smi_window": {"t0": t_sus0, "t1": t_rep1, "sustain_s": t_rep0 - t_sus0, "timed_s": t_rep1 - t_rep0},
            "check": check, "kernels": names, "peak_device_bytes": peak_mem}


# ---- SASS (only with tools already present) ------------------------------------------------------------------------------------------------

def capture_sass(dtype: str, n: int, device: str, out_dir: Path, timeout_s: float = 240.0) -> dict:
    """`ncu --metrics sass__inst_executed_per_opcode` on one GEMM in a child process, if `ncu` is on PATH; else "not captured".
    The raw output is saved as ncu_sass.txt; `sass` lists the tensor-core opcodes (HMMA/QMMA/IMMA/HGMMA/QGMMA/...) seen."""
    tools = {t: shutil.which(t) for t in ("ncu", "nsys", "cuobjdump")}
    rec = {"tools_present": {k: bool(v) for k, v in tools.items()}, "sass": "not captured", "opcodes": [], "raw_file": None, "error": None}
    if not tools["ncu"]:
        rec["error"] = "ncu not on PATH (nsys/cuobjdump alone cannot name the executed SASS opcode of a cuBLAS kernel)"
        return rec
    child = (
        "import torch,sys\n"
        f"n={n}; d={device!r}\n"
        "a=torch.randn(n,n,device=d); b=torch.randn(n,n,device=d)\n"
        f"if {dtype!r}=='bf16':\n"
        "    a=a.to(torch.bfloat16); b=b.to(torch.bfloat16); torch.cuda.synchronize(); torch.matmul(a,b); torch.cuda.synchronize()\n"
        "else:\n"
        "    a=a.to(torch.float8_e4m3fn); b=b.to(torch.float8_e4m3fn).t(); one=torch.ones((),device=d); torch.cuda.synchronize()\n"
        "    torch._scaled_mm(a,b,scale_a=one,scale_b=one,out_dtype=torch.float32); torch.cuda.synchronize()\n"
    )
    cmd = [tools["ncu"], "--csv", "--metrics", "sass__inst_executed_per_opcode", "--kernel-name", "regex:.*", "--launch-skip", "0", "--launch-count", "8",
           sys.executable, "-c", child]
    raw = out_dir / "ncu_sass.txt"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        raw.write_text((r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""))
        rec["raw_file"] = raw.name
        ops = sorted(set(re.findall(r"\b((?:H|Q|I|D|B)?G?MMA(?:\.[A-Z0-9_]+)*)\b", r.stdout or "")))
        rec["opcodes"] = ops
        if ops:
            rec["sass"] = ", ".join(ops)
        else:
            rec["error"] = f"ncu rc={r.returncode}; no MMA opcode in its output (see {raw.name})"
    except subprocess.TimeoutExpired:
        rec["error"] = f"ncu timed out after {timeout_s}s"
    except OSError as ex:
        rec["error"] = f"ncu failed to start: {ex}"
    return rec


# ---- environment -----------------------------------------------------------------------------------------------------------------------------

def environment(device: int) -> dict:
    import torch

    p = torch.cuda.get_device_properties(device)
    smi = None
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.memory,power.limit,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=False)
        smi = r.stdout.strip() or None
    driver = None
    if smi:
        parts = [s.strip() for s in smi.split(",")]
        driver = parts[1] if len(parts) > 1 else None
    return {"device": torch.cuda.get_device_name(device), "sm": f"{p.major}.{p.minor}", "sms": p.multi_processor_count, "total_memory_bytes": p.total_memory,
            "driver": driver, "nvidia_smi": smi, "cuda": torch.version.cuda, "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(), "host": platform.node(),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def source_commit() -> str | None:
    env = os.environ.get("RESEARCH_SOURCE_COMMIT")
    if env:
        return env
    name = Path.cwd().name
    return name if len(name) == 40 and all(c in "0123456789abcdef" for c in name) else None


# ---- result.json -------------------------------------------------------------------------------------------------------------------------------

def build_result(run_id: str, dtype: str, envd: dict, sizes: list[dict], sass: dict, smi_all: dict, skipped: str | None, elapsed: float,
                 measurement_files: list[str]) -> dict:
    """The `native-peak/v1` document: headline = the best size's median-of-reps throughput."""
    spec, spec_src = spec_peak(envd.get("device") or "", dtype)
    best = max(sizes, key=lambda s: s["flop_per_second"]) if sizes else None
    fps = best["flop_per_second"] if best else None
    frac = (fps / spec) if (fps and spec) else None
    per_size = {f"{s['n']}^3": {"seconds_median": s["seconds_median"], "seconds_all": s["seconds_all"], "flop_per_second": s["flop_per_second"],
                               "fraction_of_spec": (s["flop_per_second"] / spec) if spec else None, "sustained_flop_per_second": s["sustain"]["flop_per_second"],
                               "sustain_iterations": s["sustain"]["iterations"], "check_ok": s["check"]["ok"], "max_rel_err": s["check"]["max_rel_err"],
                               "kernels": s["kernels"], "clocks_mhz": s.get("smi", {}).get("clocks_sm_mhz"), "clocks_mem_mhz": s.get("smi", {}).get("clocks_mem_mhz"),
                               "power_w": s.get("smi", {}).get("power_w"), "throttle_reasons": s.get("smi", {}).get("throttle_reasons"),
                               "peak_device_bytes": s["peak_device_bytes"]} for s in sizes}
    head_smi = best.get("smi", {}) if best else {}
    clocks = (head_smi.get("clocks_sm_mhz") or {}).get("median") if head_smi else None
    power = (head_smi.get("power_w") or {}).get("median") if head_smi else None
    method = (f"cublas dense gemm {best['n']}^3, median of {best['reps']} (after {best['warmup']} warmups + {best['sustain']['seconds']:.1f} s sustained), "
              f"{'torch.matmul bf16' if dtype == 'bf16' else 'torch._scaled_mm e4m3 x e4m3 -> fp32, unit scales'}" if best else
              f"skipped: {skipped}")
    checks_ok = bool(sizes) and all(s["check"]["ok"] for s in sizes)
    if skipped:
        validation = {"status": "not_run", "checker": "fp32 reference sub-block", "detail": f"skipped: {skipped}", "evidence": []}
    else:
        validation = {"status": "passed" if checks_ok else "failed", "checker": "fp32 reference on a 256x256 sub-block of every timed product",
                      "detail": "; ".join(f"{s['n']}^3 max_rel_err {s['check']['max_rel_err']:.2e} (tol {s['check']['tolerance']:.0e}) {'ok' if s['check']['ok'] else 'FAIL'}" for s in sizes),
                      "evidence": measurement_files[:1]}
    meas = []
    if best:
        meas += [{"name": "flop_per_second", "value": fps, "unit": "flop/s"}]
        if frac is not None:
            meas += [{"name": "fraction_of_spec", "value": frac, "unit": "ratio"}]
        for s in sizes:
            meas += [{"name": f"gemm_{s['n']}_seconds", "value": s["seconds_median"], "unit": "s"},
                     {"name": f"gemm_{s['n']}_flop_per_second", "value": s["flop_per_second"], "unit": "flop/s"},
                     {"name": f"gemm_{s['n']}_sustained_flop_per_second", "value": s["sustain"]["flop_per_second"], "unit": "flop/s"}]
        if clocks is not None:
            meas += [{"name": "clocks_sm_mhz_median", "value": clocks, "unit": "MHz"}]
        if power is not None:
            meas += [{"name": "power_w_median", "value": power, "unit": "W"}]
    meas += [{"name": "elapsed", "value": elapsed, "unit": "s"}]
    top = {"device": envd.get("device"), "dtype": dtype, "accumulate": "fp32", "method": method, "flop_per_second": fps, "spec_flop_per_second": spec,
           "spec_source": spec_src, "fraction_of_spec": frac, "sass": sass.get("sass", "not captured"), "sass_detail": sass,
           "kernel": (best["kernels"] if best else None), "driver": envd.get("driver"), "cuda": envd.get("cuda"), "torch": envd.get("torch"),
           "clocks_mhz": clocks, "power_w": power, "smi_headline_window": head_smi or None, "smi_whole_run": smi_all, "sizes": per_size,
           "skipped": skipped, "environment": envd}
    fp = {"tool": "native_peak", "version": "1", "device": envd.get("device"), "sm": envd.get("sm"), "dtype": dtype, "accumulate": "fp32", "method": method,
          "sizes": [s["n"] for s in sizes], "warmup": sizes[0]["warmup"] if sizes else None, "reps": sizes[0]["reps"] if sizes else None,
          "torch": envd.get("torch"), "cuda": envd.get("cuda"), "driver": envd.get("driver"), "commit": source_commit(), "spec_flop_per_second": spec}
    return {"schema": RESULT_SCHEMA, "run_id": run_id, "kind": KIND, **top, "workload_fingerprint": fp, "validation": validation, "measurements": meas,
            "measurement_files": measurement_files, "artifacts": []}


def write_result(run_dir: Path, doc: dict) -> Path:
    tmp = run_dir / f"result.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, run_dir / "result.json")
    return run_dir / "result.json"


# ---- main --------------------------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", required=True, choices=DTYPES)
    ap.add_argument("--sizes", default="8192,16384", help="comma-separated square GEMM sizes (multiples of 16); a size that does not fit is skipped")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sustain-s", type=float, default=2.0, help="seconds of back-to-back GEMMs before the timed reps (clock/power settle)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="output directory (default $RESEARCH_RUN_DIR, else tools/native_peak/out)")
    ap.add_argument("--no-sass", action="store_true", help="do not attempt the ncu SASS capture even if ncu is present")
    a = ap.parse_args(argv)
    sizes = parse_sizes(a.sizes)
    run_dir = Path(a.out or os.environ.get("RESEARCH_RUN_DIR") or "tools/native_peak/out")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RESEARCH_RUN_ID") or f"local-{int(time.time())}"
    t0 = time.time()

    import torch

    if not torch.cuda.is_available():
        print("native_peak: no CUDA device", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()   # the memory/capability queries want an index
    torch.cuda.set_device(idx)
    envd = environment(idx)
    print(json.dumps(envd), flush=True)

    skipped = None
    if a.dtype == "e4m3":
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
            skipped = f"torch {torch.__version__} has no float8_e4m3fn/_scaled_mm"
        else:
            major, minor = torch.cuda.get_device_capability(idx)
            if (major, minor) < (8, 9):
                skipped = f"FP8 tensor cores need sm_89+; this card is sm_{major}{minor} ({envd['device']})"
            else:
                try:   # a tiny probe: the runtime is the authority on whether cuBLASLt has an FP8 path here
                    x = torch.ones(32, 32, device=a.device).to(torch.float8_e4m3fn)
                    one = torch.ones((), device=a.device)
                    torch._scaled_mm(x, x.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
                    torch.cuda.synchronize()
                except Exception as ex:  # noqa: BLE001
                    skipped = f"torch._scaled_mm e4m3 unsupported here: {type(ex).__name__}: {str(ex)[:200]}"

    files: list[str] = []
    results: list[dict] = []
    smi_path = run_dir / "nvidia_smi_samples.csv"
    sampler = SmiSampler(smi_path)
    if skipped:
        print(f"native_peak: SKIP {a.dtype}: {skipped}", flush=True)
    else:
        sampler.start()
        time.sleep(0.3)
        for n in sizes:
            need = 2 * n * n * 4                              # the fp32 randn sources, freed once converted
            need += (3 * n * n * 2) if a.dtype == "bf16" else (2 * n * n + 4 * n * n)
            need += 3 * n * CHECK_BLOCK * 4                   # the fp32 check strips
            free, total = torch.cuda.mem_get_info(idx)
            if need > free * 0.9:
                print(f"native_peak: skip {n}^3: needs ~{need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free", flush=True)
                continue
            torch.cuda.reset_peak_memory_stats(idx)
            r = measure_size(n, a.dtype, a.device, a.warmup, a.reps, a.sustain_s)
            results.append(r)
            print(f"native_peak {a.dtype} {n}^3: median {r['seconds_median'] * 1e3:.3f} ms = {r['flop_per_second'] / 1e12:.1f} TFLOP/s "
                  f"(sustained {r['sustain']['flop_per_second'] / 1e12:.1f} over {r['sustain']['iterations']} it), check {'ok' if r['check']['ok'] else 'FAIL'} "
                  f"rel {r['check']['max_rel_err']:.2e}, kernels {r['kernels'][:2]}", flush=True)
        time.sleep(0.3)
        sampler.stop()
        for r in results:
            r["smi"] = sampler.window(r["smi_window"]["t0"], r["smi_window"]["t1"])
            c, p = r["smi"].get("clocks_sm_mhz"), r["smi"].get("power_w")
            print(f"  smi {r['n']}^3: {r['smi']['n_samples']} samples, sm clock median {c and c['median']} MHz, power median {p and p['median']} W, "
                  f"throttle {r['smi']['throttle_reasons']}", flush=True)
        if smi_path.is_file():
            files.append(smi_path.name)
    smi_all = summarize_samples(sampler.rows) if sampler.rows else {"n_samples": 0}
    sass = {"sass": "not captured", "error": "skipped (--no-sass or nothing measured)", "tools_present": {t: bool(shutil.which(t)) for t in ("ncu", "nsys", "cuobjdump")}}
    if results and not a.no_sass:
        sass = capture_sass(a.dtype, results[0]["n"], a.device, run_dir)
        if sass.get("raw_file"):
            files.append(sass["raw_file"])
    print(f"native_peak: sass = {sass['sass']}" + (f" ({sass.get('error')})" if sass.get("error") else ""), flush=True)
    detail = {"run_id": run_id, "dtype": a.dtype, "environment": envd, "sizes": results, "sass": sass, "smi_whole_run": smi_all, "skipped": skipped,
              "args": vars(a)}
    (run_dir / "native_peak.json").write_text(json.dumps(detail, indent=1, default=str) + "\n")
    files.insert(0, "native_peak.json")
    doc = build_result(run_id, a.dtype, envd, results, sass, smi_all, skipped, time.time() - t0, files)
    write_result(run_dir, doc)
    print(json.dumps({k: doc[k] for k in ("device", "dtype", "flop_per_second", "spec_flop_per_second", "fraction_of_spec", "sass", "clocks_mhz", "power_w")},
                     indent=1), flush=True)
    return 0 if (skipped or doc["validation"]["status"] == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
