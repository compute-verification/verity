"""units.sm_90a.cubin (build.sh's) in torch's context: its kernels, launched on torch's current stream through the driver
API.  The cubin is $UNITS_CUBIN, else units.sm_90a.cubin beside this file, else build/units.sm_90a.cubin here.  torch is
imported only here, so the twins run without it.
"""
from __future__ import annotations

import ctypes as C
import functools
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = ("units_rmsnorm", "units_quant", "units_quant4", "units_softmax", "units_argmax", "units_relu", "units_add",
           "units_sweep")
#: units.cu's RT and SR: the threads of an RMSNorm, quantizer or softmax CTA
ROW_THREADS = 128

u, i32, u64, vp, P = C.c_uint, C.c_int, C.c_uint64, C.c_void_p, C.POINTER
CUDA = {
    "cuDeviceGet": [P(i32), i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuGetErrorString": [i32, P(C.c_char_p)],
}


def cubin_path() -> Path:
    env = os.environ.get("UNITS_CUBIN")
    if env:
        return Path(env)
    beside = HERE / "units.sm_90a.cubin"
    return beside if beside.exists() else HERE / "build" / "units.sm_90a.cubin"


class Lib:
    """The cubin's kernels in torch's context on the current device."""

    def __init__(self, path: Path):
        import torch
        self.torch = torch
        self.path = Path(path)
        self.cubin = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.cubin).hexdigest()
        self.cu = C.CDLL(os.environ.get("UNITS_LIBCUDA", "libcuda.so.1"))
        for f, a in CUDA.items():
            getattr(self.cu, f).argtypes, getattr(self.cu, f).restype = a, i32
        torch.cuda.init()
        self.device = torch.cuda.current_device()
        dev, ctx = i32(), vp()
        self.ck("cuDeviceGet", C.byref(dev), self.device)
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), self.cubin)
        self.fns = {name: self.fn(name) for name in KERNELS}

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f} returned CUresult {r} ({s.value.decode() if s.value else 'unknown'})")

    def fn(self, name: str) -> vp:
        f = vp()
        self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
        return f

    def launch(self, name: str, grid: int | tuple[int, int], block: int, args: list) -> None:
        gx, gy = (grid, 1) if isinstance(grid, int) else grid
        params = (vp * len(args))(*(C.cast(C.pointer(a), vp) for a in args))
        stream = vp(self.torch.cuda.current_stream().cuda_stream)
        self.ck("cuLaunchKernel", self.fns[name], gx, gy, 1, block, 1, 1, 0, stream, params, None)


@functools.cache
def lib() -> Lib:
    return Lib(cubin_path())


def ptr(t) -> vp:
    """A tensor's device pointer, or NULL for None."""
    return vp(0 if t is None else t.data_ptr())


def threads(n: int) -> int:
    """A block of at least n threads (at least one warp), a whole number of warps."""
    return max(32, -(-n // 32) * 32)


def words(t, dtype, shape: tuple, name: str):
    """``t`` as a contiguous CUDA tensor of ``dtype`` and ``shape``, refused otherwise."""
    if t.dtype != dtype or tuple(t.shape) != tuple(shape) or not t.is_cuda:
        raise ValueError(f"{name}: want {dtype} {tuple(shape)} on CUDA, got {t.dtype} {tuple(t.shape)} on {t.device}")
    return t.contiguous()


def aligned(name: str, *ts) -> None:
    """Refuses a tensor whose data isn't 16-byte aligned: the kernels move rows as uint4s, and a misaligned one is a sticky
    CUDA error that ends the context."""
    for t in ts:
        if t is not None and t.data_ptr() % 16:
            raise ValueError(f"{name}: a tensor at {t.data_ptr():#x} is not 16-byte aligned")


def out_or_new(out, dtype, shape: tuple, like, name: str):
    """``out`` if it is a contiguous ``dtype`` tensor of ``shape`` on ``like``'s device (refused otherwise), else a new one."""
    import torch
    if out is None:
        return torch.empty(shape, dtype=dtype, device=like.device)
    if out.dtype != dtype or tuple(out.shape) != tuple(shape) or not out.is_contiguous() or out.device != like.device:
        raise ValueError(f"{name}: out must be contiguous {dtype} {tuple(shape)} on {like.device}, got {out.dtype} "
                         f"{tuple(out.shape)} on {out.device}")
    return out


def unit_outputs(parts: dict | None, want: dict, like, name: str) -> dict:
    """The unit outputs ``want`` ({key: (dtype, shape)}): the caller's tensors in ``parts`` (each checked), new ones for the
    rest."""
    parts = parts or {}
    extra = set(parts) - set(want)
    if extra:
        raise ValueError(f"{name}: no unit outputs {sorted(extra)} (it has {sorted(want)})")
    return {k: out_or_new(parts.get(k), dt, shape, like, f"{name} {k}") for k, (dt, shape) in want.items()}
