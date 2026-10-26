"""attn_bf16's kernels as torch functions, with KATTN-REF's `qk(q, k)` / `pv(p, v)` signature (the shapes of the skeleton's
`e2e_h200.attention.scores` / `pv`), so the skeleton swaps one for the other:

  qk(q, k)   q (B, H, D), k (B, T, D) or (B, Hkv, T, D)  ->  S (B, H, T)
             S[b,h,j] = GemmCoordinate_v2{K=D, GEMM_HOPPER}(q[b,h,:], k[b,kv(h),j,:])
  pv(p, v)   p (B, H, T), v (B, T, D) or (B, Hkv, T, D)  ->  O (B, H, D)
             O[b,h,d] = GemmCoordinate_v2{K=kp, GEMM_HOPPER}(p[b,h,:] ++ 0, v[b,kv(h),:,d] ++ 0)

kv(h) = h // (H / Hkv) (H / Hkv a power of two; MQA's k and v may drop the head axis), D 64 or 128, and kp = max(kpv,
pad16(T)) with kpv = 32 by default: the H200 model's PV, every KV length 1..32 padded with +0 to K = 32.  Tensors are bf16
(or int16 words) on one CUDA device; a copy is made only of one whose last dimension isn't contiguous (or, for q and k,
whose rows aren't 4-byte aligned).  ``lens`` (B,) int32 in 1..T, optional: sequence b's keys are its first lens[b], S is +0
past them, P and V past them are read as +0, and its kp is max(kpv, pad16(lens[b])).  Each call is one launch on torch's
current stream, from the cubin build.sh ships: ATTN_BF16_CUBIN, else attn_bf16.<arch>.cubin beside this file (sm_90a on
an H100/H200, sm_120a on an RTX PRO 6000).
"""

from __future__ import annotations

import ctypes as C
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAD_K = 32
ARCH = {(9, 0): "sm_90a", (12, 0): "sm_120a"}
#: the head-group tilings' group sizes; a tiling serves where its size divides the query heads a KV head serves
GROUP = {"h16": 16, "h8": 8}
#: each op's launch: (tiling, warps a block, head groups a warp), falling back to h8 and then q8 where the tiling can't
#: serve; run.py's timing sweeps the alternatives
LAUNCH = {"qk": ("h16", 8, 1), "pv": ("h16", 8, 1)}

u64 = C.c_uint64


class Args(C.Structure):
    """attn_bf16.cu's AttnArgs."""
    _fields_ = [("x", u64), ("kv", u64), ("out", u64), ("lens", u64), ("nq", u64), ("xs", C.c_int64 * 3),
                ("kvs", C.c_int64 * 3), ("os", C.c_int64 * 3), ("B", C.c_int32), ("H", C.c_int32), ("Mq", C.c_int32),
                ("Lcap", C.c_int32), ("gshift", C.c_int32), ("kpv", C.c_int32)]


def grid(tiling: str, B: int, H: int, Mq: int, Y: int = 0) -> tuple[int, int, int]:
    """q8: a block per (8 query rows, head, sequence); h8, h16: per (query row, sequence) and Y blocks of head groups
    (default one group a block), each block's warps running every Y-th group."""
    return (-(-Mq // 8), H, B) if tiling == "q8" else (Mq, Y or H // GROUP[tiling], B)


def kernel(op: str, D: int, tiling: str) -> str:
    return f"attn_{op}_d{D}" + ("" if tiling == "q8" else f"_{tiling}")


KERNELS = [kernel(op, d, t) for op in ("qk", "pv") for d in (64, 128) for t in ("q8", *GROUP)]


_DRIVER = {"cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)], "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
           "cuDevicePrimaryCtxRetain": [C.POINTER(C.c_void_p), C.c_int], "cuCtxSetCurrent": [C.c_void_p],
           "cuModuleLoadData": [C.POINTER(C.c_void_p), C.c_char_p],
           "cuModuleGetFunction": [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
           "cuLaunchKernel": [C.c_void_p, *[C.c_uint] * 7, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]}
_MODULES: dict[int, tuple] = {}


def _lib():
    lib = C.CDLL(os.environ.get("ATTN_LIBCUDA", "libcuda.so.1"))
    for f, a in _DRIVER.items():
        getattr(lib, f).argtypes, getattr(lib, f).restype = a, C.c_int
    return lib


def _ok(lib, f: str, *args) -> None:
    r = getattr(lib, f)(*args)
    if r:
        s = C.c_char_p()
        lib.cuGetErrorString(r, C.byref(s))
        raise RuntimeError(f"{f}: {r} {s.value.decode() if s.value else ''}")


def _module(index: int) -> tuple:
    """(driver, the device's primary context, {kernel name: function}) for CUDA device ``index``, loaded once."""
    import torch

    if index not in _MODULES:
        cc = torch.cuda.get_device_capability(index)
        if cc not in ARCH:
            raise RuntimeError(f"attn_bf16: no cubin for compute capability {cc} (have {sorted(ARCH)})")
        path = os.environ.get("ATTN_BF16_CUBIN") or os.path.join(HERE, f"attn_bf16.{ARCH[cc]}.cubin")
        with open(path, "rb") as f:
            cubin = f.read()
        torch.cuda.init()
        lib, dev, ctx, mod = _lib(), C.c_int(), C.c_void_p(), C.c_void_p()
        _ok(lib, "cuDeviceGet", C.byref(dev), index)
        _ok(lib, "cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        _ok(lib, "cuCtxSetCurrent", ctx)
        _ok(lib, "cuModuleLoadData", C.byref(mod), cubin)
        funcs = {}
        for name in KERNELS:
            fn = C.c_void_p()
            _ok(lib, "cuModuleGetFunction", C.byref(fn), mod, name.encode())
            funcs[name] = fn
        _MODULES[index] = (lib, ctx, funcs)
    return _MODULES[index]


def _words(t, name: str, pairs: bool):
    """``t`` as bf16 with its last dimension contiguous (and, where the kernel loads word pairs, even strides and a 4-byte
    aligned base), copied only when it isn't."""
    import torch

    if t.dtype == torch.int16:
        t = t.view(torch.bfloat16)
    if t.dtype != torch.bfloat16 or not t.is_cuda:
        raise TypeError(f"attn_bf16: {name} must be a CUDA bf16 or int16 tensor, not {t.dtype} on {t.device}")
    if t.stride(-1) != 1 or (pairs and (t.data_ptr() % 4 or any(s % 2 for s in t.stride()[:-1]))):
        t = t.contiguous()
    return t


def _kv4(t):
    return t.unsqueeze(1) if t.dim() == 3 else t


def _launch(op: str, x, kv, out, lens, kpv: int) -> None:
    import torch

    B, H, D = x.shape[0], x.shape[1], kv.shape[-1]
    Hkv, T = kv.shape[1], kv.shape[2]
    G = H // Hkv
    if H % Hkv or G & (G - 1):
        raise ValueError(f"attn_bf16: {H} query heads on {Hkv} KV heads (a power of two a KV head)")
    if D not in (64, 128):
        raise ValueError(f"attn_bf16: head dim {D}, not 64 or 128")
    if kpv % 16 or kpv < 0:
        raise ValueError(f"attn_bf16: kpv {kpv} is not a multiple of 16")
    dev = x.device
    if lens is None:
        lens = torch.full((B,), T, dtype=torch.int32, device=dev)
    else:
        lens = lens.to(device=dev, dtype=torch.int32).contiguous()
        if lens.shape != (B,):
            raise ValueError(f"attn_bf16: lens {tuple(lens.shape)}, want ({B},)")
    nq = torch.ones((B,), dtype=torch.int32, device=dev)
    tiling, warps, per = LAUNCH[op]
    if G % GROUP[tiling]:
        tiling = "h8" if G % 8 == 0 else "q8"
    lib, ctx, funcs = _module(dev.index if dev.index is not None else torch.cuda.current_device())
    args = Args(x.data_ptr(), kv.data_ptr(), out.data_ptr(), lens.data_ptr(), nq.data_ptr(),
                (C.c_int64 * 3)(x.stride(0), x.stride(1), 0), (C.c_int64 * 3)(kv.stride(0), kv.stride(1), kv.stride(2)),
                (C.c_int64 * 3)(out.stride(0), out.stride(1), 0), B, H, 1, T, G.bit_length() - 1, kpv)
    if tiling in GROUP:
        Y = max(1, -(-(H // GROUP[tiling]) // (warps * per)))
    else:
        warps, Y = (min(8, -(-T // 16)) if op == "qk" else D // 16), 0
    params = (C.c_void_p * 1)(C.cast(C.pointer(args), C.c_void_p))
    stream = C.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    with torch.cuda.device(dev):
        _ok(lib, "cuCtxSetCurrent", ctx)
        _ok(lib, "cuLaunchKernel", funcs[kernel(op, D, tiling)], *grid(tiling, B, H, 1, Y), 32 * warps, 1, 1, 0, stream, params,
            None)


def qk(q, k, lens=None):
    """S (B, H, T) = q (B, H, D) against k (B, T, D) or (B, Hkv, T, D); see the module."""
    import torch

    q, k = _words(q, "q", True), _kv4(_words(k, "k", True))
    if q.dim() != 3 or k.dim() != 4 or k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError(f"attn_bf16.qk: q {tuple(q.shape)}, k {tuple(k.shape)}")
    B, H, T = q.shape[0], q.shape[1], k.shape[2]
    out = (torch.zeros if lens is not None else torch.empty)((B, H, T), dtype=torch.bfloat16, device=q.device)
    _launch("qk", q, k, out, lens, 0)
    return out


def pv(p, v, lens=None, kpv: int = PAD_K):
    """O (B, H, D) = p (B, H, T) times v (B, T, D) or (B, Hkv, T, D), PV's K = max(kpv, pad16(T)); see the module."""
    import torch

    p, v = _words(p, "p", False), _kv4(_words(v, "v", False))
    if p.dim() != 3 or v.dim() != 4 or v.shape[0] != p.shape[0] or v.shape[2] != p.shape[2]:
        raise ValueError(f"attn_bf16.pv: p {tuple(p.shape)}, v {tuple(v.shape)}")
    B, H, D = p.shape[0], p.shape[1], v.shape[-1]
    out = torch.empty((B, H, D), dtype=torch.bfloat16, device=p.device)
    _launch("pv", p, v, out, lens, kpv)
    return out
