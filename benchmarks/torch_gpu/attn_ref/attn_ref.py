"""attn_ref.sm_90a.cubin (build.sh's) on torch tensors: attention's two BF16 GEMMs for decode, each output word exactly
`GemmCoordinate_v2{K, GEMM_HOPPER}` (attn_ref.cu says how).

    qk(q, k, lens=None) -> s   q [B, H, D], k [B, L, D] (MQA) or [B, Hkv, L, D] -> s [B, H, L]
        s[b, h, j] = GemmCoordinate{K=D}(q[b, h], k[b, kv(h), j])               j < lens[b]
        s[b, h, j] = fill (default 0xFF80, -inf)                                lens[b] <= j < L
    pv(p, v, lens=None) -> o   p [B, H, L], v [B, L, D] or [B, Hkv, L, D] -> o [B, H, D]
        o[b, h, d] = GemmCoordinatePad{T=n, K=kp}(p[b, h, :n], v[b, kv(h), :n, d])     n = lens[b], kp = max(kpv, pad16(n))

Every tensor is bf16 but lens (int32 [B], on the device; default: every sequence holds L keys), D is a multiple of 16,
kv(h) = h // (H // Hkv), and p's and v's words past lens[b] are never read as values (they enter as zero codes, as the
Definition pads both operands).  GemmCoordinatePad_v1{T, K} is GemmCoordinate_v2{K} on both operands padded with K - T
words of +0, and `kpv` (keyword, a multiple of 16) defaults to PV_K = 32, the H200 circuit's PV:
GemmCoordinatePad_v1{T=n, K=32} at every length 1..32.  `out=` takes a contiguous bf16 tensor of the result's shape and
writes every word of it.
`check_lens` (default True) refuses lens outside [0, L], at the price of a device sync; pass False under a CUDA graph.
`warps` (1, 2, 4 or 8 a block) decides only which warp runs a tile, never a word.  Kernels run on torch's current stream.

The cubin is $ATTN_REF_CUBIN, else attn_ref.sm_90a.cubin beside this file, else build/attn_ref.sm_90a.cubin here.
"""
from __future__ import annotations

import ctypes as C
import functools
import hashlib
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NEG_INF = 0xFF80
PV_K = 32
WARPS = (1, 2, 4, 8)

u, i32, vp, P = C.c_uint, C.c_int, C.c_void_p, C.POINTER
CUDA = {
    "cuDeviceGet": [P(i32), i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuGetErrorString": [i32, P(C.c_char_p)],
}


def cubin_path() -> Path:
    env = os.environ.get("ATTN_REF_CUBIN")
    if env:
        return Path(env)
    beside = HERE / "attn_ref.sm_90a.cubin"
    return beside if beside.exists() else HERE / "build" / "attn_ref.sm_90a.cubin"


class Lib:
    """The cubin's two kernels in torch's context on the current device."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cubin = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.cubin).hexdigest()
        self.cu = C.CDLL(os.environ.get("ATTN_REF_LIBCUDA", "libcuda.so.1"))
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
        self.fns = {name: self.fn(name) for name in ("attn_ref_qk", "attn_ref_pv")}

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

    def launch(self, name: str, grid: tuple[int, int, int], warps: int, args: list) -> None:
        params = (vp * len(args))(*(C.cast(C.pointer(a), vp) for a in args))
        stream = vp(torch.cuda.current_stream().cuda_stream)
        self.ck("cuLaunchKernel", self.fns[name], *grid, 32 * warps, 1, 1, 0, stream, params, None)


@functools.cache
def lib() -> Lib:
    return Lib(cubin_path())


def _flat(x: torch.Tensor, kv: torch.Tensor, lens: torch.Tensor | None, name: str):
    """(x [B*Hkv, H/Hkv, X], kv [B*Hkv, L, D], lens [B*Hkv] int32, B, H, Hkv, L, D) from x [B, H, X] and kv [B, (Hkv,) L, D]."""
    if x.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError(f"{name}: operands must be bf16, got {x.dtype} and {kv.dtype}")
    if not (x.is_cuda and kv.is_cuda and x.device == kv.device):
        raise ValueError(f"{name}: operands must be on one CUDA device")
    if kv.dim() == 3:
        kv = kv.unsqueeze(1)
    if x.dim() != 3 or kv.dim() != 4 or kv.shape[0] != x.shape[0]:
        raise ValueError(f"{name}: shapes {tuple(x.shape)} and {tuple(kv.shape)} are not [B, H, *] and [B, (Hkv,) L, D]")
    B, H = x.shape[:2]
    Hkv, L, D = kv.shape[1:]
    if Hkv < 1 or H % Hkv:
        raise ValueError(f"{name}: {H} query heads don't split evenly over {Hkv} KV heads")
    if D % 16 or D < 16:
        raise ValueError(f"{name}: head dim {D} is not a positive multiple of 16")
    if lens is None:
        lens = torch.full((B,), L, dtype=torch.int32, device=x.device)
    elif lens.dtype != torch.int32 or lens.shape != (B,) or lens.device != x.device:
        raise ValueError(f"{name}: lens must be int32 [{B}] on {x.device}, got {lens.dtype} {tuple(lens.shape)} on {lens.device}")
    if Hkv > 1:
        lens = lens.repeat_interleave(Hkv)
    G = H // Hkv
    return x.contiguous().view(B * Hkv, G, x.shape[2]), kv.contiguous().view(B * Hkv, L, D), lens.contiguous(), B, H, Hkv, L, D


def _check(lens: torch.Tensor, L: int, name: str) -> None:
    lo, hi = int(lens.min()), int(lens.max())
    if lo < 0 or hi > L:
        raise ValueError(f"{name}: lens span [{lo}, {hi}], outside [0, {L}], the keys the cache holds")


def _out(out: torch.Tensor | None, shape: tuple, like: torch.Tensor, name: str) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=like.device)
    if out.dtype != torch.bfloat16 or tuple(out.shape) != shape or not out.is_contiguous() or out.device != like.device:
        raise ValueError(f"{name}: out must be contiguous bf16 {shape} on {like.device}, got {out.dtype} {tuple(out.shape)}")
    return out


def _grid(nb: int, rows: int, cols: int, warps: int) -> tuple[int, int, int]:
    if warps not in WARPS:
        raise ValueError(f"warps must be one of {WARPS}, got {warps}")
    tiles = -(-rows // 16)
    return nb, -(-tiles // warps), -(-cols // 8)


def qk(q: torch.Tensor, k: torch.Tensor, lens: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
       fill: int = NEG_INF, warps: int = 4, check_lens: bool = True) -> torch.Tensor:
    """S = Q K^T per sequence: q [B, H, D], k [B, (Hkv,) L, D] bf16 -> [B, H, L] bf16 (the module docstring)."""
    x, kv, ln, B, H, Hkv, L, D = _flat(q, k, lens, "qk")
    if q.shape[2] != D:
        raise ValueError(f"qk: q's head dim {q.shape[2]} is not k's {D}")
    if not 0 <= fill <= 0xFFFF:
        raise ValueError(f"qk: fill {fill:#x} is not a 16-bit word")
    if check_lens:
        _check(ln, L, "qk")
    s = _out(out, (B, H, L), q, "qk")
    G = H // Hkv
    if B and H and L:
        lib().launch("attn_ref_qk", _grid(B * Hkv, G, L, warps), warps,
                     [vp(x.data_ptr()), vp(kv.data_ptr()), vp(s.data_ptr()), vp(ln.data_ptr()), i32(G), i32(D), i32(L), u(fill)])
    return s


def pv(p: torch.Tensor, v: torch.Tensor, lens: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
       kpv: int = PV_K, warps: int = 4, check_lens: bool = True) -> torch.Tensor:
    """O = P V per sequence: p [B, H, L], v [B, (Hkv,) L, D] bf16 -> [B, H, D] bf16, K = max(kpv, pad16(lens[b])) (the
    module docstring)."""
    x, kv, ln, B, H, Hkv, L, D = _flat(p, v, lens, "pv")
    if p.shape[2] != L:
        raise ValueError(f"pv: p's {p.shape[2]} keys are not v's {L}")
    if kpv < 0 or kpv % 16:
        raise ValueError(f"pv: kpv {kpv} is not a multiple of 16")
    if check_lens:
        _check(ln, L, "pv")
    o = _out(out, (B, H, D), p, "pv")
    G = H // Hkv
    if B and H:
        lib().launch("attn_ref_pv", _grid(B * Hkv, G, D, warps), warps,
                     [vp(x.data_ptr()), vp(kv.data_ptr()), vp(o.data_ptr()), vp(ln.data_ptr()), i32(G), i32(D), i32(L),
                      i32(kpv)])
    return o
