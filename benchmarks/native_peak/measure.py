"""native_peak: the card's *measured* dense-GEMM throughput for one dtype, as a `native-peak/v1` result.

    python benchmarks/native_peak/measure.py --dtype bf16  [--sizes 8192,16384] [--warmup 3] [--reps 5] [--sustain-s 2.0]
    python benchmarks/native_peak/measure.py --dtype e4m3   # FP8 e4m3 inputs, FP32 accumulate (torch._scaled_mm); skips cleanly where unsupported
    python benchmarks/native_peak/measure.py --dtype nvfp4  # sm_120 only: the block-scaled FP4 mma.sync instruction itself (mma_peak_fp4.cu),
                                                       # register-resident operands, every warp of every SM -- an instruction peak, not a GEMM;
                                                       # a cuBLASLt NVFP4 GEMM through torch._scaled_mm is attempted beside it and recorded

One run = one dtype = one artifact.  For each square size n the tool runs `--warmup` untimed GEMMs, a `--sustain-s` phase of
back-to-back GEMMs (so clocks and power settle where they will sit under load, sampled by `nvidia-smi` every 100 ms), then
`--reps` timed GEMMs bracketed by CUDA events; the reported number is `2 n^3 / median(reps)`.  BF16 goes through `torch.matmul`
(cuBLAS, FP32 accumulate is cuBLAS's default for bf16 and reduced-precision reduction is switched off explicitly); e4m3 through
`torch._scaled_mm` with unit scales and `out_dtype=torch.float32`.  A small sub-block of every timed product is checked against an
FP32 reference (`validation`).  The cuBLAS kernel name comes from `torch.profiler`; the SASS opcode is taken from `ncu` only if
that binary is already on the machine, otherwise it is recorded as "not captured".

result.json (`research/result/v0.1`, kind `native-peak/v1`) carries the headline fields at the top level -- device, dtype,
accumulate, method, flop_per_second, spec_flop_per_second, fraction_of_spec, sass, kernel, driver, cuda, clocks_mhz, power_w,
sizes -- so a store query can select on them (`meta.dtype`), and the same identity in `workload_fingerprint`.  Spec peaks are
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "config"))
from verity_config.settings import SHARED

RESULT_SCHEMA = "research/result/v0.1"
KIND = "native-peak/v1"
DTYPES = ("bf16", "e4m3", "nvfp4")

#: Datasheet dense tensor-core peaks (flop/s) with FP32 accumulate, keyed by a regex on `torch.cuda.get_device_name`.  First match
#: wins.  RTX 4090: NVIDIA Ada GPU architecture whitepaper (BF16 with FP32 acc 165.2 TFLOPS, FP8 with FP32 acc 330.3 TFLOPS,
#: boost clock, dense).  A100 SXM4 / PCIe 80GB: NVIDIA A100 datasheet (BF16 312 TFLOPS dense; no FP8 tensor cores).  RTX 5090:
#: NVIDIA RTX Blackwell GPU Architecture whitepaper, Appendix A table "GeForce RTX 5090 vs RTX 4090 vs RTX 3090" -- BF16 with FP32
#: accumulate 209.5, FP8 with FP32 accumulate 419, "Peak FP4 Tensor TFLOPS with FP32 Accumulate (FP4 AI TOPS)" 1676 dense / 3352
#: with 2:4 sparsity (the card's headline "3352 AI TOPS" is the sparse figure), all at the 2407 MHz boost clock.
SPEC_PEAKS: list[tuple[str, dict[str, float | None], str]] = [
    (r"RTX\s*4090", {"bf16": 165.2e12, "e4m3": 330.3e12, "nvfp4": None}, "NVIDIA Ada GPU Architecture whitepaper, RTX 4090 dense, FP32 accumulate; Ada has no FP4"),
    (r"A100-SXM4", {"bf16": 312.0e12, "e4m3": None, "nvfp4": None}, "NVIDIA A100 Tensor Core GPU datasheet (SXM4 80GB), BF16 dense; Ampere has no FP8"),
    (r"A100.*PCIe", {"bf16": 312.0e12, "e4m3": None, "nvfp4": None}, "NVIDIA A100 Tensor Core GPU datasheet (PCIe 80GB), BF16 dense; Ampere has no FP8"),
    # H100 SXM5: benchmarks/census/hardware.json h100-sxm5-80gb/bf16 and h100-sxm5-80gb/e4m3 (tests/test_native_peak.py checks every census line).
    (r"H100.*(SXM|HBM3)", {"bf16": 989.4e12, "e4m3": 1978.9e12, "nvfp4": None},
     "NVIDIA Hopper architecture whitepaper, H100 SXM5 dense (BF16 1978.9 / FP8 3957.8 TFLOPS with sparsity, halved), FP32 accumulate; Hopper has no FP4"),
    (r"H100.*PCIe", {"bf16": 756.0e12, "e4m3": 1513.0e12, "nvfp4": None}, "NVIDIA H100 PCIe datasheet, dense (sparse figures halved), FP32 accumulate; Hopper has no FP4"),
    (r"L40S", {"bf16": 362.05e12, "e4m3": 733.0e12, "nvfp4": None},
     "NVIDIA L40S GPU datasheet, dense (BF16 733 / FP8 1,466 TFLOPS with sparsity, halved); Ada has no FP4"),
    (r"RTX\s*PRO\s*6000.*Server", {"bf16": 500.0e12, "e4m3": 1000.0e12, "nvfp4": 2000.0e12},
     "NVIDIA RTX PRO 6000 Blackwell Server Edition product page, dense (FP16 | BF16 1 / FP8 2 / FP4 4 PFLOPS with sparsity, halved), "
     "FP32 accumulate at full rate (RTX PRO Blackwell whitepaper Table 4); census rtx-pro-6000-server/{bf16,e4m3,e2m1}"),
    (r"RTX\s*5090", {"bf16": 209.5e12, "e4m3": 419.0e12, "nvfp4": 1676.0e12},
     "NVIDIA RTX Blackwell GPU Architecture whitepaper, Appendix A, RTX 5090 dense with FP32 accumulate (FP4: 1676 dense, 3352 sparse)"),
    # benchmarks/census/hardware.json rtx-pro-6000-server/{bf16,e4m3,e2m1}.  The Workstation Edition's peaks differ (a 2617 MHz boost clock), hence "Server".
    (r"RTX\s*PRO\s*6000.*Server", {"bf16": 500.0e12, "e4m3": 1000.0e12, "nvfp4": 2000.0e12},
     "NVIDIA RTX PRO 6000 Blackwell Server Edition product page, dense (BF16 1 / FP8 2 / FP4 4 PFLOPS with sparsity, halved); FP32 "
     "accumulate at the FP16-accumulate rate (RTX PRO Blackwell whitepaper, Table 4)"),
]

#: --dtype nvfp4: the block-scaled FP4 instruction itself (`mma_peak_fp4.cu`) -- see `measure_nvfp4_mma`.
NVFP4_MMA_FLOP = 2 * 16 * 8 * 64          # one mma.sync m16n8k64
NVFP4_SOURCE = "mma_peak_fp4.cu"
NVFP4_ARCH = "sm_120a"

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


# ---- nvfp4: the instruction peak ------------------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


def nvcc_bin() -> str | None:
    return shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").is_file() else None)


def build_nvfp4(out: Path) -> tuple[Path, str]:
    """nvcc the microbenchmark for sm_120a (compute_120a PTX named explicitly, as the probe does: ptxas refuses `.block_scale`
    on plain sm_120); returns (path of the .so, nvcc version line)."""
    nvcc = nvcc_bin()
    if nvcc is None:
        raise FileNotFoundError("nvcc not found (PATH or /usr/local/cuda/bin)")
    so = out / f"lib{Path(NVFP4_SOURCE).stem}_{NVFP4_ARCH}.so"
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    subprocess.run([nvcc, "-O2", f"-gencode=arch=compute_{NVFP4_ARCH[3:]},code={NVFP4_ARCH}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so),
                    str(HERE / NVFP4_SOURCE)], check=True)
    return so, " / ".join(line.strip() for line in ver)


def nvfp4_sass(so: Path) -> dict:
    """`cuobjdump -sass` on the microbenchmark: the MMA opcodes and their static count in the loop body (NSETS x NACC = 32 in
    the unrolled body + nothing else that is an MMA).  The same opcode name the probe records (OMMA.16864.F32.E2M1.E2M1.UE4M3)."""
    nvcc = nvcc_bin()
    cuobjdump = str(Path(nvcc).parent / "cuobjdump") if nvcc else shutil.which("cuobjdump")
    if not cuobjdump or not Path(cuobjdump).is_file():
        return {"sass": "not captured", "opcodes": [], "mma_count": 0, "error": "cuobjdump not found"}
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    ops: dict[str, int] = {}
    for line in text.splitlines():
        m = re.search(r"\b((?:H|Q|I|O|D|B)?G?MMA(?:\.[A-Z0-9_]+)*)\b", line)
        if m and "*/" in line:
            ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    return {"sass": ", ".join(sorted(ops)) if ops else "not captured", "opcodes": sorted(ops), "mma_count": sum(ops.values()),
            "per_opcode": ops, "error": None if ops else "no MMA opcode in cuobjdump -sass"}


def nvfp4_fragments(nwarps: int, words_per_lane: int, seed: int = 20260922):
    """Random register fragments per lane: E2M1 nibbles for a[4] / b[2] (every code, both signs), UE4M3 scale bytes for sfa / sfb
    sampled from the finite codes around 1 (0x30..0x47: 2^-3 .. 2^4 x mantissa) -- toggling operand bits, as a GEMM would, and
    accumulators that stay finite over any run length this tool uses."""
    import numpy as np

    rng = np.random.default_rng(seed)
    nsets = words_per_lane // 8
    frags = np.zeros((nwarps * 32, nsets, 8), dtype=np.uint32)
    frags[:, :, :6] = rng.integers(0, 2**32, size=(nwarps * 32, nsets, 6), dtype=np.uint64).astype(np.uint32)
    scales = rng.integers(0x30, 0x48, size=(nwarps * 32, nsets, 2, 4), dtype=np.uint32)
    packed = scales[..., 0] | (scales[..., 1] << 8) | (scales[..., 2] << 16) | (scales[..., 3] << 24)
    frags[:, :, 6:8] = packed
    return np.ascontiguousarray(frags.reshape(-1))


def measure_nvfp4_mma(device: str, warmup: int, reps: int, sustain_s: float, out_dir: Path, warps_per_sm: int = 16,
                      target_rep_s: float = 0.25) -> dict:
    """The sustained issue rate of `mma.sync.kind::mxf4nvf4 ... e2m1.e2m1.f32.ue4m3` from every warp of every SM with
    register-resident operands (`mma_peak_fp4.cu`): flop = launches x warps x iterations x 32 MMAs x 16,384.  Same protocol as
    `measure_size`: `warmup` untimed launches, >= `sustain_s` of back-to-back launches, then `reps` event-timed launches
    (median).  The `check` is finiteness + bit-identical outputs across two launches + the SASS opcode count."""
    import ctypes

    import numpy as np
    import torch

    so, nvcc_ver = build_nvfp4(out_dir)
    lib = ctypes.CDLL(str(so))
    lib.nvfp4_peak_device.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
    lib.nvfp4_peak_setup.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.nvfp4_peak_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]
    sms, maxthr, major, minor = (ctypes.c_int() for _ in range(4))
    rc = lib.nvfp4_peak_device(sms, maxthr, major, minor)
    if rc:
        raise RuntimeError(f"nvfp4_peak_device rc={rc}")
    warps_per_block = 4
    blocks_per_sm = warps_per_sm // warps_per_block
    nblocks = sms.value * blocks_per_sm
    nwarps = nblocks * warps_per_block
    words = lib.nvfp4_peak_words_per_lane()
    mmas_per_iter = lib.nvfp4_peak_mmas_per_iter()
    frags = nvfp4_fragments(nwarps, words)
    rc = lib.nvfp4_peak_setup(frags.ctypes.data, nwarps)
    if rc:
        raise RuntimeError(f"nvfp4_peak_setup rc={rc} (cuda error)")
    ms = ctypes.c_float()
    sink = np.zeros(nwarps * 32 * 4, dtype=np.float32)

    def run(iters: int, want_sink: bool = False) -> float:
        rc = lib.nvfp4_peak_run(nblocks, warps_per_block, iters, ms, sink.ctypes.data if want_sink else None)
        if rc:
            raise RuntimeError(f"nvfp4_peak_run rc={rc} (cuda error)")
        return ms.value / 1e3

    flop_per_iter = float(nwarps) * mmas_per_iter * NVFP4_MMA_FLOP
    # calibrate the iteration count to ~target_rep_s per launch
    t_cal = run(2000)
    iters = max(2000, int(2000 * target_rep_s / max(t_cal, 1e-6)))
    for _ in range(warmup):
        run(iters)
    t_sus0 = time.time()
    k, sus_secs = 0, 0.0
    while True:
        sus_secs += run(iters)
        k += 1
        if time.time() - t_sus0 >= sustain_s:
            break
    t_rep0 = time.time()
    times = [run(iters) for _ in range(reps)]
    t_rep1 = time.time()
    med = statistics.median(times)
    flop = flop_per_iter * iters
    # check: finite, deterministic across two launches, expected SASS
    run(iters, want_sink=True)
    first = sink.copy()
    run(iters, want_sink=True)
    finite = bool(np.isfinite(first).all())
    same = bool(np.array_equal(first.view(np.uint32), sink.view(np.uint32)))
    sass = nvfp4_sass(so)
    # the iteration loop is `#pragma unroll 1` in the kernel, so the SASS carries the body once: exactly NSETS x NACC MMAs.
    # (Should a toolchain unroll it anyway, whole copies of the body -- a multiple -- still mean one body per iteration.)
    n_sass = sass["mma_count"]
    copies = n_sass // mmas_per_iter if mmas_per_iter and n_sass % mmas_per_iter == 0 else 0
    ok = finite and same and copies >= 1 and all("E2M1" in op for op in sass["opcodes"])
    check = {"ok": ok, "finite": finite, "deterministic": same, "sass_mma_count": n_sass, "expected_mma_count": mmas_per_iter,
             "sass_body_copies": copies, "opcodes": sass["opcodes"], "tolerance": 0.0, "max_rel_err": 0.0,
             "what": "outputs finite and bit-identical across two launches; cuobjdump counts NSETS x NACC E2M1 MMAs in the kernel "
                     "(the body once: #pragma unroll 1; whole unrolled copies would be a multiple)"}
    lib.nvfp4_peak_teardown()
    label = f"mma_m16n8k64_x{iters}"
    return {"n": None, "m": 16, "k": 64, "label": label, "instruction": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
            "grid": {"sms": sms.value, "blocks": nblocks, "warps_per_block": warps_per_block, "warps": nwarps, "mmas_per_iteration": mmas_per_iter,
                     "iterations": iters, "accumulator_chains": 8, "operand_sets": words // 8}, "cc": f"{major.value}.{minor.value}", "nvcc": nvcc_ver,
            "so": so.name, "flop": flop, "warmup": warmup, "reps": reps, "seconds_all": times, "seconds_median": med, "seconds_min": min(times),
            "seconds_max": max(times), "flop_per_second": flop / med,
            "sustain": {"seconds": sus_secs, "iterations": k, "flop_per_second": flop * k / sus_secs},
            "smi_window": {"t0": t_sus0, "t1": t_rep1, "sustain_s": t_rep0 - t_sus0, "timed_s": t_rep1 - t_rep0},
            "check": check, "kernels": ["nvfp4_peak_kernel"], "sass": sass, "peak_device_bytes": int(frags.nbytes + sink.nbytes)}


def try_nvfp4_cublaslt_gemm(n: int, device: str, warmup: int, reps: int) -> dict:
    """Best-effort: an NVFP4 GEMM through `torch._scaled_mm` (cuBLASLt's block-scaled FP4 path; torch >= 2.8 exposes
    float4_e2m1fn_x2 operands with 1x16 float8_e4m3fn block scales).  Records what the toolchain does; never raises."""
    import torch

    rec: dict = {"attempted": True, "n": n, "supported": False, "reason": None, "flop_per_second": None, "seconds_median": None}
    try:
        if not hasattr(torch, "float4_e2m1fn_x2"):
            rec["reason"] = f"torch {torch.__version__} has no float4_e2m1fn_x2 dtype"
            return rec
        blk = 16
        a = torch.randint(0, 256, (n, n // 2), device=device, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (n, n // 2), device=device, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t()
        sa = torch.full((n, n // blk), 0x38, device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)     # 1.0 scales, 1x16 blocks
        sb = torch.full((n, n // blk), 0x38, device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)
        fn = None
        err = []
        variants = []
        if hasattr(torch, "_scaled_mm_v2") and hasattr(torch, "ScalingType"):
            variants.append(("_scaled_mm_v2 BlockWise1x16", lambda: torch._scaled_mm_v2(a, b, [sa], [sb], [torch.ScalingType.BlockWise1x16],
                                                                                          [torch.ScalingType.BlockWise1x16], None, torch.float32)))
        variants.append(("_scaled_mm blockwise scales", lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.float32)))
        for name, cand in variants:
            try:
                cand()
                torch.cuda.synchronize()
                fn, rec["variant"] = cand, name
                break
            except Exception as ex:  # noqa: BLE001
                err.append(f"{name}: {type(ex).__name__}: {str(ex)[:300]}")
        if fn is None:
            rec["reason"] = "; ".join(err)
            return rec
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) / 1e3)
        med = statistics.median(times)
        rec.update({"supported": True, "seconds_all": times, "seconds_median": med, "flop_per_second": 2.0 * n ** 3 / med,
                    "kernels": kernel_names(fn), "note": "unit scales, random nibbles; throughput only (no numerical check)"})
    except Exception as ex:  # noqa: BLE001
        rec["reason"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    finally:
        torch.cuda.empty_cache()
    return rec


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
    env = SHARED.RESEARCH_SOURCE_COMMIT
    if env:
        return env
    name = Path.cwd().name
    return name if len(name) == 40 and all(c in "0123456789abcdef" for c in name) else None


# ---- result.json -------------------------------------------------------------------------------------------------------------------------------

def _label(s: dict) -> str:
    return s["label"] if s.get("label") else f"{s['n']}^3"


def build_result(run_id: str, dtype: str, envd: dict, sizes: list[dict], sass: dict, smi_all: dict, skipped: str | None, elapsed: float,
                 measurement_files: list[str], gemm_attempt: dict | None = None) -> dict:
    """The `native-peak/v1` document: headline = the best size's median-of-reps throughput (nvfp4: the instruction loop; the
    cuBLASLt GEMM attempt, when the toolchain supports one, is recorded beside it as `gemm_attempt`)."""
    spec, spec_src = spec_peak(envd.get("device") or "", dtype)
    best = max(sizes, key=lambda s: s["flop_per_second"]) if sizes else None
    fps = best["flop_per_second"] if best else None
    frac = (fps / spec) if (fps and spec) else None
    per_size = {_label(s): {"seconds_median": s["seconds_median"], "seconds_all": s["seconds_all"], "flop_per_second": s["flop_per_second"],
                               "fraction_of_spec": (s["flop_per_second"] / spec) if spec else None, "sustained_flop_per_second": s["sustain"]["flop_per_second"],
                               "sustain_iterations": s["sustain"]["iterations"], "check_ok": s["check"]["ok"], "max_rel_err": s["check"]["max_rel_err"],
                               "kernels": s["kernels"], "clocks_mhz": s.get("smi", {}).get("clocks_sm_mhz"), "clocks_mem_mhz": s.get("smi", {}).get("clocks_mem_mhz"),
                               "power_w": s.get("smi", {}).get("power_w"), "throttle_reasons": s.get("smi", {}).get("throttle_reasons"),
                               "peak_device_bytes": s["peak_device_bytes"]} for s in sizes}
    head_smi = best.get("smi", {}) if best else {}
    clocks = (head_smi.get("clocks_sm_mhz") or {}).get("median") if head_smi else None
    power = (head_smi.get("power_w") or {}).get("median") if head_smi else None
    if dtype == "nvfp4" and best:
        g = best["grid"]
        method = (f"instruction microbenchmark: {best['instruction']} issued back to back from {g['warps']} warps ({g['warps_per_block']}/block x "
                  f"{g['blocks']} blocks on {g['sms']} SMs), {g['accumulator_chains']} independent accumulator chains x {g['operand_sets']} register-resident "
                  f"operand sets, {g['iterations']} x {g['mmas_per_iteration']} MMAs per launch, median of {best['reps']} event-timed launches (after "
                  f"{best['warmup']} warmups + {best['sustain']['seconds']:.1f} s sustained); no memory traffic -- an upper bound on any GEMM built from it")
    else:
        method = (f"cublas dense gemm {best['n']}^3, median of {best['reps']} (after {best['warmup']} warmups + {best['sustain']['seconds']:.1f} s sustained), "
                  f"{'torch.matmul bf16' if dtype == 'bf16' else 'torch._scaled_mm e4m3 x e4m3 -> fp32, unit scales'}" if best else
                  f"skipped: {skipped}")
    checks_ok = bool(sizes) and all(s["check"]["ok"] for s in sizes)
    if skipped:
        validation = {"status": "not_run", "checker": "fp32 reference sub-block", "detail": f"skipped: {skipped}", "evidence": []}
    elif dtype == "nvfp4":
        c = best["check"]
        validation = {"status": "passed" if checks_ok else "failed",
                      "checker": "outputs finite and bit-identical across two launches; cuobjdump -sass counts the loop body's E2M1 MMAs "
                                 "(NSETS x NACC, or whole unrolled copies of it)",
                      "detail": f"finite={c['finite']} deterministic={c['deterministic']} sass {c['sass_mma_count']}/{c['expected_mma_count']} "
                                f"(x{c.get('sass_body_copies', 0)}) {c['opcodes']} {'ok' if c['ok'] else 'FAIL'}", "evidence": measurement_files[:1]}
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
            stem = s["label"] if s.get("label") else f"gemm_{s['n']}"
            meas += [{"name": f"{stem}_seconds", "value": s["seconds_median"], "unit": "s"},
                     {"name": f"{stem}_flop_per_second", "value": s["flop_per_second"], "unit": "flop/s"},
                     {"name": f"{stem}_sustained_flop_per_second", "value": s["sustain"]["flop_per_second"], "unit": "flop/s"}]
        if gemm_attempt and gemm_attempt.get("supported"):
            meas += [{"name": f"cublaslt_gemm_{gemm_attempt['n']}_flop_per_second", "value": gemm_attempt["flop_per_second"], "unit": "flop/s"}]
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
    if dtype == "nvfp4":
        top["instruction"] = best["instruction"] if best else None
        top["grid"] = best["grid"] if best else None
        top["gemm_attempt"] = gemm_attempt
        if best:
            top["sass"] = best["sass"]["sass"]
            top["sass_detail"] = best["sass"]
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
    ap.add_argument("--out", default=None, help="output directory (default $RESEARCH_RUN_DIR, else benchmarks/native_peak/out)")
    ap.add_argument("--no-sass", action="store_true", help="do not attempt the ncu SASS capture even if ncu is present")
    a = ap.parse_args(argv)
    sizes = parse_sizes(a.sizes)
    run_dir = Path(a.out or SHARED.RESEARCH_RUN_DIR or "benchmarks/native_peak/out")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = SHARED.RESEARCH_RUN_ID or f"local-{int(time.time())}"
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
    if a.dtype == "nvfp4":
        major, minor = torch.cuda.get_device_capability(idx)
        if (major, minor) != (12, 0):
            skipped = f"the sm_120 mma.sync kind::mxf4nvf4 instruction needs cc 12.0; this card is sm_{major}{minor} ({envd['device']})"
        elif nvcc_bin() is None:
            skipped = "nvcc not found (PATH or /usr/local/cuda/bin): cannot build mma_peak_fp4.cu"

    files: list[str] = []
    results: list[dict] = []
    gemm_attempt = None
    smi_path = run_dir / "nvidia_smi_samples.csv"
    sampler = SmiSampler(smi_path)
    if skipped:
        print(f"native_peak: SKIP {a.dtype}: {skipped}", flush=True)
    elif a.dtype == "nvfp4":
        sampler.start()
        time.sleep(0.3)
        r = measure_nvfp4_mma(a.device, a.warmup, a.reps, a.sustain_s, run_dir)
        results.append(r)
        print(f"native_peak nvfp4 {r['label']}: median {r['seconds_median'] * 1e3:.3f} ms = {r['flop_per_second'] / 1e12:.1f} TFLOP/s "
              f"(sustained {r['sustain']['flop_per_second'] / 1e12:.1f} over {r['sustain']['iterations']} launches), check {'ok' if r['check']['ok'] else 'FAIL'} "
              f"{r['check']['opcodes']} x{r['check']['sass_mma_count']}", flush=True)
        time.sleep(0.3)
        sampler.stop()
        r["smi"] = sampler.window(r["smi_window"]["t0"], r["smi_window"]["t1"])
        c, p = r["smi"].get("clocks_sm_mhz"), r["smi"].get("power_w")
        print(f"  smi: {r['smi']['n_samples']} samples, sm clock median {c and c['median']} MHz, power median {p and p['median']} W, "
              f"throttle {r['smi']['throttle_reasons']}", flush=True)
        if smi_path.is_file():
            files.append(smi_path.name)
        gemm_attempt = try_nvfp4_cublaslt_gemm(sizes[0], a.device, a.warmup, a.reps)
        print(f"native_peak nvfp4 cuBLASLt GEMM {sizes[0]}^3 via torch._scaled_mm: "
              + (f"{gemm_attempt['flop_per_second'] / 1e12:.1f} TFLOP/s ({gemm_attempt.get('variant')})" if gemm_attempt["supported"] else
                 f"unsupported -- {gemm_attempt['reason']}"), flush=True)
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
    if results and a.dtype == "nvfp4":
        sass = results[0]["sass"]                              # cuobjdump on our own kernel: no ncu needed
    elif results and not a.no_sass:
        sass = capture_sass(a.dtype, results[0]["n"], a.device, run_dir)
        if sass.get("raw_file"):
            files.append(sass["raw_file"])
    print(f"native_peak: sass = {sass['sass']}" + (f" ({sass.get('error')})" if sass.get("error") else ""), flush=True)
    detail = {"run_id": run_id, "dtype": a.dtype, "environment": envd, "sizes": results, "sass": sass, "smi_whole_run": smi_all, "skipped": skipped,
              "gemm_attempt": gemm_attempt, "args": vars(a)}
    (run_dir / "native_peak.json").write_text(json.dumps(detail, indent=1, default=str) + "\n")
    files.insert(0, "native_peak.json")
    doc = build_result(run_id, a.dtype, envd, results, sass, smi_all, skipped, time.time() - t0, files, gemm_attempt)
    write_result(run_dir, doc)
    print(json.dumps({k: doc[k] for k in ("device", "dtype", "flop_per_second", "spec_flop_per_second", "fraction_of_spec", "sass", "clocks_mhz", "power_w")},
                     indent=1), flush=True)
    return 0 if (skipped or doc["validation"]["status"] == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
