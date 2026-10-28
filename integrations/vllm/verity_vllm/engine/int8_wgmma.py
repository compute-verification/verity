"""`int8_wgmma.cu` from Python: `int8_fused.scaled_mm`'s product on sm_90a (the H100 and H200) as a persistent,
warp-specialized wgmma GEMM (ping-pong or cooperative), bit-identical to `exactlin_scaled_mm_tma` (the same s32 sums, and
the same epilogue in the same order: cvt.rn.f32.s32, mul.rn by s_a, mul.rn by s_w, cvt.rn.bf16).

`scaled_mm(xq, s_a, w, s_w, y, cfg)` launches it on torch's current stream: xq [M, K] int8, s_a [M] float32, w [N, K] int8
(row-major), s_w [N] float32 -> y [M, N] bf16, in place.  It allocates nothing and a CUDA graph captures it (the tensor maps
are kernel parameters, encoded on the host each call).  A config (`WgCfg`, text "ppm2_128x128:g8") is the kernel (`VARIANTS`:
ping-pong with a consumer tile of 128 x 128, or cooperative on 128 x 256; `m2` for clusters of two CTAs sharing W's boxes
by multicast), the raster group (tile rows, or pairs of them, a group) and at most how many CTAs (0: one an SM).

`scaled_mm_zy(xq, s_a, w, s_w, y, zy, rows, cfg)` is `int8_fused.scaled_mm_zy`'s product the same way: the same y, and from
the same accumulators the call's z/y records over `rows` slots (`int8_exact.zy_records`), byte for byte `_zy_epilogue`'s.

The library is built by the nvcc on the path, else torch's CUDA_HOME's (`-fmad=false`, no FTZ or fast-math flag), for
`int8_gemm_configs.WGMMA_ARCH`, and kept under ``~/.cache/verity/int8_wgmma``, keyed by its sources, flags and nvcc's
version, as `commit.row_tables_device` builds its kernels.  `available()` says whether the current device can run it;
`int8_fused.pick_wgmma` takes it only then.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import torch

from verity_vllm.engine.int8_exact import ZY_BYTES, zy_units
from verity_vllm.engine.profiles.int8_gemm_configs import WGMMA, WGMMA_ARCH

HERE = Path(__file__).resolve().parent
SOURCES = ("int8_wgmma_acc.h", "int8_wgmma.cu")
NVCC_FLAGS = ("-O3", "-std=c++17", "-gencode", f"arch=compute_{WGMMA_ARCH.removeprefix('sm_')},code={WGMMA_ARCH}",
              "-fmad=false", "-Xcompiler", "-fPIC", "-shared", "-Xptxas", "-v")
BUILD_DIR = Path.home() / ".cache" / "verity" / "int8_wgmma"
#: the library's variants, in `int8_wgmma_layout`'s order: (tile rows, columns, cluster); `pp` ping-pong (each consumer
#: its own tile), `co` cooperative (both consumers one tile, 64 rows each)
VARIANTS = {"pp_128x128": (128, 128, 1), "ppm2_128x128": (128, 128, 2), "co_128x256": (128, 256, 1),
            "com2_128x256": (128, 256, 2)}
_vp, _i = ctypes.c_void_p, ctypes.c_int


class WgCfg(tuple):
    """(kernel, raster group, CTAs); `WgCfg.parse("ppm2_128x128:g8")`, `"pp_128x128:g16:c66"`."""

    def __new__(cls, kernel: str, group: int = 8, ctas: int = 0):
        if kernel not in VARIANTS or group < 1 or ctas < 0:
            raise ValueError(f"no such int8_wgmma config: {kernel}:g{group}:c{ctas}")
        return super().__new__(cls, (kernel, group, ctas))

    @classmethod
    def parse(cls, text: str) -> WgCfg:
        kernel, *rest = text.split(":")
        kw = {"g": 8, "c": 0}
        for r in rest:
            kw[r[0]] = int(r[1:])
        return cls(kernel, kw["g"], kw["c"])

    def __str__(self) -> str:
        kernel, group, ctas = self
        return f"{kernel}:g{group}" + (f":c{ctas}" if ctas else "")


def row(m: int, n: int, k: int, sm: str) -> WgCfg | None:
    """`int8_gemm_configs.WGMMA`'s config for one call on `sm`: the last (m_min, cfg) row for (n, k) with m >= m_min."""
    text = None
    for m_min, cfg in WGMMA.get(sm, {}).get((n, k), ()):
        if m >= m_min:
            text = cfg
    return None if text is None else WgCfg.parse(text)


def nvcc() -> str | None:
    """nvcc on the path, else the one under torch's CUDA_HOME."""
    found = shutil.which("nvcc")
    if found:
        return found
    from torch.utils.cpp_extension import CUDA_HOME

    home = Path(CUDA_HOME) / "bin" / "nvcc" if CUDA_HOME else None
    return str(home) if home is not None and home.exists() else None


def build(tool: str) -> Path:
    """The library for sm_90a, built once per sources, flags and nvcc version; ptxas's report lands beside it."""
    h = hashlib.sha256()
    for s in SOURCES:
        h.update(s.encode() + b"\0" + (HERE / s).read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(NVCC_FLAGS).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / "libint8_wgmma.so"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        stubs = Path(tool).resolve().parents[1] / "lib64" / "stubs"
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([tool, *NVCC_FLAGS, "-o", str(tmp), str(HERE / SOURCES[-1]), f"-L{stubs}", "-lcuda"],
                             capture_output=True, text=True)
        lib.with_name("ptxas.txt").write_text(res.stdout + res.stderr)
        if res.returncode:
            raise RuntimeError(f"nvcc failed on {SOURCES[-1]}:\n{res.stderr[-4000:]}")
        tmp.replace(lib)
    return lib


@cache
def lib() -> ctypes.CDLL:
    tool = nvcc()
    if tool is None:
        raise RuntimeError("no nvcc on the path")
    dll = ctypes.CDLL(str(build(tool)))
    dll.int8_wgmma_layout.argtypes = [ctypes.POINTER(_i)]
    dll.int8_wgmma_launch.argtypes = [_i, _vp, _vp, _vp, _vp, _vp, _i, _i, _i, _i, _i, _i, _vp, ctypes.POINTER(_i)]
    dll.int8_wgmma_launch_zy.argtypes = [_i, _vp, _vp, _vp, _vp, _vp, _vp, _i, _i, _i, _i, _i, _i, _i, _vp,
                                         ctypes.POINTER(_i)]
    dll.int8_wgmma_layout.restype = dll.int8_wgmma_launch.restype = dll.int8_wgmma_launch_zy.restype = _i
    out = (_i * 40)()
    n = dll.int8_wgmma_layout(out)
    got = {name: tuple(out[5 * v: 5 * v + 3]) for v, name in enumerate(VARIANTS)} if n == len(VARIANTS) else None
    if got != VARIANTS:
        raise RuntimeError(f"int8_wgmma's layout {list(out)[:5 * n]} is not int8_wgmma.py's {VARIANTS}")
    return dll


def layout() -> dict[str, dict[str, int]]:
    """Each variant's tile, cluster, stages and shared memory, as the library reports them."""
    out = (_i * 40)()
    lib().int8_wgmma_layout(out)
    return {name: dict(zip(("wm", "bn", "cluster", "stages", "smem"), out[5 * v: 5 * v + 5])) for v, name in enumerate(VARIANTS)}


@cache
def available(index: int) -> bool:
    """Whether device `index` is sm_90 and the library builds and loads."""
    if torch.cuda.get_device_capability(index) != (9, 0):
        return False
    try:
        lib()
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return False
    return True


def scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
              cfg: WgCfg, flags: int = 0) -> int:
    """xq [M, K] int8, s_a [M] f32, w [N, K] int8 (row-major), s_w [N] f32 -> y [M, N] bf16, in place; returns the grid.
    y goes out by TMA stores where N is a multiple of 8 (else by each thread); `flags` 16 keeps the threads' stores, and
    1, 2 and 4 are timing probes (streaming stores; no epilogue and no stores, which leave y unwritten)."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.is_contiguous() and w.is_contiguous()
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.is_contiguous()
    assert m >= 1 and k % 16 == 0 and n % 2 == 0, (m, n, k)
    assert xq.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and s_w.data_ptr() % 8 == 0 and y.data_ptr() % 4 == 0
    kernel, group, ctas = cfg
    grid = _i()
    rc = lib().int8_wgmma_launch(list(VARIANTS).index(kernel), xq.data_ptr(), w.data_ptr(), s_a.data_ptr(),
                                 s_w.data_ptr(), y.data_ptr(), m, n, k, group, ctas, flags,
                                 torch.cuda.current_stream().cuda_stream, ctypes.byref(grid))
    if rc:
        raise RuntimeError(f"int8_wgmma_launch({cfg}, M={m}, N={n}, K={k}) failed at step {rc}")
    return grid.value


def scaled_mm_zy(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
                 zy: torch.Tensor, rows: int, cfg: WgCfg, flags: int = 0) -> int:
    """`scaled_mm` into y, and zy [units, ZY_BYTES] uint8 (16-byte aligned), the call's records over `rows` slots
    (`int8_exact.zy_records`), in place; returns the grid.  `flags` as `scaled_mm`'s, and 32 and 64 time the records
    (streaming stores; none, which leaves them unwritten)."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.is_contiguous() and w.is_contiguous()
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.is_contiguous()
    assert zy.dtype == torch.uint8 and zy.shape == (zy_units(m, n, rows), ZY_BYTES) and zy.is_contiguous(), zy.shape
    assert m >= 1 and k % 16 == 0, (m, n, k)
    assert xq.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and s_w.data_ptr() % 8 == 0 and y.data_ptr() % 4 == 0
    assert zy.data_ptr() % 16 == 0
    kernel, group, ctas = cfg
    grid = _i()
    rc = lib().int8_wgmma_launch_zy(list(VARIANTS).index(kernel), xq.data_ptr(), w.data_ptr(), s_a.data_ptr(),
                                    s_w.data_ptr(), y.data_ptr(), zy.data_ptr(), m, n, k, rows, group, ctas, flags,
                                    torch.cuda.current_stream().cuda_stream, ctypes.byref(grid))
    if rc:
        raise RuntimeError(f"int8_wgmma_launch_zy({cfg}, M={m}, N={n}, K={k}, T={rows}) failed at step {rc}")
    return grid.value


def fits(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
         zy: torch.Tensor | None = None) -> bool:
    """Whether `scaled_mm` (with zy, `scaled_mm_zy`) takes these operands (else `int8_fused` runs its Triton plan, with the
    same values)."""
    m, k = xq.shape
    n = w.shape[0]
    return (xq.is_contiguous() and w.is_contiguous() and s_a.is_contiguous() and s_w.is_contiguous() and y.is_contiguous()
            and m >= 1 and k % 16 == 0 and n % 2 == 0 and xq.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0
            and s_w.data_ptr() % 8 == 0 and y.data_ptr() % 4 == 0
            and (zy is None or (zy.is_contiguous() and zy.data_ptr() % 16 == 0)))
