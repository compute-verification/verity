"""attn_bf16's kernels as torch functions, `qk(q, k, lens=None, *, out, fill, warps, check_lens, path)` and `pv(p, v,
lens=None, *, out, kpv, warps, check_lens, path)`, the shapes of the skeleton's `e2e_h200.attention.scores` / `pv`:

  qk(q, k)   q (B, H, D), k (B, T, D) or (B, Hkv, T, D)  ->  S (B, H, T)
             S[b,h,j] = GemmCoordinate_v2{K=D, GEMM_HOPPER}(q[b,h,:], k[b,kv(h),j,:])     j < lens[b]
             S[b,h,j] = fill (default 0xFF80, -inf)                                       lens[b] <= j < T
  pv(p, v)   p (B, H, T), v (B, T, D) or (B, Hkv, T, D)  ->  O (B, H, D)
             O[b,h,d] = GemmCoordinate_v2{K=kp, GEMM_HOPPER}(p[b,h,:n] ++ 0, v[b,kv(h),:n,d] ++ 0)    n = lens[b]

kv(h) = h // (H / Hkv) (H / Hkv a power of two; MQA's k and v may drop the head axis), D 64 or 128, and kp = max(kpv,
pad16(n)) with keyword ``kpv`` = 32 by default: the H200 circuit's PV, GemmCoordinatePad_v1{T=n, K=32}, every KV length
1..32 padded with +0 to K = 32, and pad16(n) past 32 keys.  Tensors are bf16 (or int16
words) on one CUDA device; a copy is made only of one whose last dimension isn't contiguous (or, for q and k, whose rows
aren't 4-byte aligned).  ``lens`` (B,) int32 in 0..T, default T each: words of p and v past lens[b] are read as +0, and a
sequence of no keys gets O = +0.  ``check_lens`` (default True) refuses lens outside [0, T], at the price of a device sync;
pass False under a CUDA graph, where the caller vouches for them.  ``out`` takes a contiguous bf16 tensor of the result's shape and writes
every word of it.  ``warps`` (1, 2, 4 or 8 a CTA; default attn.LAUNCH's) decides only which warp runs a tile, never a word.
``path`` "fast" (the default) launches attn.LAUNCH's tiling, and "ref" the reference kernels (the _ref kernels, slow and
tuned for nothing), for the same words.
Without lens each call is one launch; with lens, S is filled before the launch (one more pass over S) and, without
check_lens, O is zeroed first (one over O).  Launches are on torch's current stream, from the cubin build.sh ships:
ATTN_BF16_CUBIN, else attn_bf16.<arch>.cubin beside this file (sm_90a on an H100/H200, sm_120a on an RTX PRO 6000).
"""

from __future__ import annotations

import ctypes as C
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAD_K = 32
NEG_INF = 0xFF80
WARPS = (1, 2, 4, 8)
ARCH = {(9, 0): "sm_90a", (12, 0): "sm_120a"}
#: the head-group tilings' group sizes
GROUP = {"s16": 16, "h16": 16, "h8": 8}
#: the s16 kernels' most keys a query sees, and PV's largest K
S16_KEYS = 32
#: each op's launch: (tiling, warps a CTA, head groups a warp), falling back to h16, h8 and then q8 where the tiling can't
#: serve; run.py's timing sweeps the alternatives (on the H200 at the frozen shape, QK^T and PV are each fastest with every
#: group of a sequence in one warp, at M = 4,096, 16,384 and 20,480 within 3%)
LAUNCH = {"qk": ("s16", 1, 16), "pv": ("s16", 1, 16)}
#: qk's and pv's ``path``: attn.LAUNCH's tiling, or the reference kernels
PATHS = ("fast", "ref")


def serves(tiling: str, G: int, T: int, kpv: int) -> bool:
    """Whether ``tiling`` computes every word for G query heads a KV head, T keys at most and PV's least K kpv: a head
    tiling where its group size divides G, s16 only for T and kpv at most S16_KEYS; q8 and ref everywhere."""
    if tiling in ("q8", "ref"):
        return True
    return G % GROUP[tiling] == 0 and (tiling != "s16" or (T <= S16_KEYS and kpv <= S16_KEYS))


u64 = C.c_uint64


class Args(C.Structure):
    """attn_bf16.cu's AttnArgs."""
    _fields_ = [("x", u64), ("kv", u64), ("out", u64), ("lens", u64), ("nq", u64), ("xs", C.c_int64 * 3),
                ("kvs", C.c_int64 * 3), ("os", C.c_int64 * 3), ("B", C.c_int32), ("H", C.c_int32), ("Mq", C.c_int32),
                ("Lcap", C.c_int32), ("gshift", C.c_int32), ("kpv", C.c_int32)]


def grid(tiling: str, B: int, H: int, Mq: int, Y: int = 0, cols: int = 0, Hkv: int = 1) -> tuple[int, int, int]:
    """q8: a CTA per (8 query rows, head, sequence); h8, h16, s16: per (query row, sequence) and Y CTAs of head groups
    (default one group a CTA), each CTA's warps running every Y-th group; ref: per (8 of the op's ``cols`` columns, keys
    or d, KV head, sequence)."""
    if tiling == "ref":
        return -(-cols // 8), Hkv, B
    return (-(-Mq // 8), H, B) if tiling == "q8" else (Mq, Y or H // GROUP[tiling], B)


def kernel(op: str, D: int, tiling: str) -> str:
    return f"attn_{op}_d{D}" + ("" if tiling == "q8" else f"_{tiling}")


KERNELS = [kernel(op, d, t) for op in ("qk", "pv") for d in (64, 128) for t in ("q8", *GROUP, "ref")]


_DRIVER = {"cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)], "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
           "cuDevicePrimaryCtxRetain": [C.POINTER(C.c_void_p), C.c_int], "cuCtxSetCurrent": [C.c_void_p],
           "cuModuleLoadData": [C.POINTER(C.c_void_p), C.c_char_p],
           "cuModuleGetFunction": [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
           "cuLaunchKernel": [C.c_void_p, *[C.c_uint] * 7, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]}
_MODULES: dict[int, tuple] = {}
_CONSTS: dict[tuple, tuple] = {}


def _consts(dev, B: int, T: int) -> tuple:
    """(lens of T each, nq of 1 each) for B sequences on ``dev``, made once outside a CUDA graph's capture (inside one, a
    tensor first filled there would hold its words only once the graph replays)."""
    import torch

    key = (dev.index, B, T)
    got = _CONSTS.get(key)
    if got is None:
        got = (torch.full((B,), T, dtype=torch.int32, device=dev), torch.ones((B,), dtype=torch.int32, device=dev))
        if not torch.cuda.is_current_stream_capturing():
            if len(_CONSTS) >= 256:
                _CONSTS.clear()
            _CONSTS[key] = got
    return got


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


def cubin_path(arch: str = "sm_90a") -> str:
    """The cubin launches on an ``arch`` card load: ATTN_BF16_CUBIN, else build.sh's one beside this file."""
    return os.environ.get("ATTN_BF16_CUBIN") or os.path.join(HERE, f"attn_bf16.{arch}.cubin")


def _module(index: int) -> tuple:
    """(driver, the device's primary context, {kernel name: function}) for CUDA device ``index``, loaded once."""
    import torch

    if index not in _MODULES:
        cc = torch.cuda.get_device_capability(index)
        if cc not in ARCH:
            raise RuntimeError(f"attn_bf16: no cubin for compute capability {cc} (have {sorted(ARCH)})")
        with open(cubin_path(ARCH[cc]), "rb") as f:
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


def _lens(lens, B: int, T: int, dev, check: bool, name: str):
    """(lens as int32 on ``dev``, or None, and whether some sequence has no keys: None when unchecked)."""
    import torch

    if lens is None:
        return None, False
    lens = lens.to(device=dev, dtype=torch.int32).contiguous()
    if tuple(lens.shape) != (B,):
        raise ValueError(f"attn_bf16.{name}: lens {tuple(lens.shape)}, want ({B},)")
    if not check or not B:
        return lens, None
    lo, hi = int(lens.min()), int(lens.max())
    if lo < 0 or hi > T:
        raise ValueError(f"attn_bf16.{name}: lens span [{lo}, {hi}], outside [0, {T}], the keys the cache holds")
    return lens, lo == 0


def _out(out, shape: tuple, like, name: str):
    import torch

    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=like.device)
    if out.dtype == torch.int16:
        out = out.view(torch.bfloat16)
    if out.dtype != torch.bfloat16 or tuple(out.shape) != shape or not out.is_contiguous() or out.device != like.device:
        raise ValueError(f"attn_bf16.{name}: out must be contiguous bf16 {shape} on {like.device}, got {out.dtype} "
                         f"{tuple(out.shape)} on {out.device}")
    return out


def _launch(op: str, x, kv, out, lens, kpv: int, warps: int | None, path: str) -> None:
    import torch

    if path not in PATHS:
        raise ValueError(f"attn_bf16: path must be one of {PATHS}, got {path!r}")
    B, H, D = x.shape[0], x.shape[1], kv.shape[-1]
    Hkv, T = kv.shape[1], kv.shape[2]
    G = H // Hkv
    if H % Hkv or G & (G - 1):
        raise ValueError(f"attn_bf16: {H} query heads on {Hkv} KV heads (a power of two a KV head)")
    if D not in (64, 128):
        raise ValueError(f"attn_bf16: head dim {D}, not 64 or 128")
    if kpv % 16 or kpv < 0:
        raise ValueError(f"attn_bf16: kpv {kpv} is not a multiple of 16")
    if warps is not None and warps not in WARPS:
        raise ValueError(f"attn_bf16: warps must be one of {WARPS}, got {warps}")
    if not (B and H and T):
        return
    dev = x.device
    full, nq = _consts(dev, B, T)
    lens = full if lens is None else lens
    tiling, w, per = LAUNCH[op] if path == "fast" else ("ref", 4, 0)
    if not serves(tiling, G, T, kpv):
        tiling = next(t for t in ("h16", "h8", "q8") if serves(t, G, T, kpv))
    if tiling in GROUP:
        w = warps or w
        Y = max(1, -(-(H // GROUP[tiling]) // (w * per)))
    elif tiling == "q8":
        w, Y = warps or (min(8, -(-T // 16)) if op == "qk" else D // 16), 0
    else:
        w, Y = warps or w, 0
    lib, ctx, funcs = _module(dev.index if dev.index is not None else torch.cuda.current_device())
    args = Args(x.data_ptr(), kv.data_ptr(), out.data_ptr(), lens.data_ptr(), nq.data_ptr(),
                (C.c_int64 * 3)(x.stride(0), x.stride(1), 0), (C.c_int64 * 3)(kv.stride(0), kv.stride(1), kv.stride(2)),
                (C.c_int64 * 3)(out.stride(0), out.stride(1), 0), B, H, 1, T, G.bit_length() - 1, kpv)
    params = (C.c_void_p * 1)(C.cast(C.pointer(args), C.c_void_p))
    stream = C.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    with torch.cuda.device(dev):
        _ok(lib, "cuCtxSetCurrent", ctx)
        _ok(lib, "cuLaunchKernel", funcs[kernel(op, D, tiling)], *grid(tiling, B, H, 1, Y, T if op == "qk" else D, Hkv),
            32 * w, 1, 1, 0, stream, params, None)


def qk(q, k, lens=None, *, out=None, fill: int = NEG_INF, warps: int | None = None, check_lens: bool = True,
       path: str = "fast"):
    """S (B, H, T) = q (B, H, D) against k (B, T, D) or (B, Hkv, T, D), the word ``fill`` past lens; see the module."""
    import torch

    q, k = _words(q, "q", True), _kv4(_words(k, "k", True))
    if q.dim() != 3 or k.dim() != 4 or k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError(f"attn_bf16.qk: q {tuple(q.shape)}, k {tuple(k.shape)}")
    if not 0 <= fill <= 0xFFFF:
        raise ValueError(f"attn_bf16.qk: fill {fill:#x} is not a 16-bit word")
    B, H, T = q.shape[0], q.shape[1], k.shape[2]
    lens, _ = _lens(lens, B, T, q.device, check_lens, "qk")
    s = _out(out, (B, H, T), q, "qk")
    if lens is not None:
        s.view(torch.int16).fill_(fill - 0x10000 if fill & 0x8000 else fill)
    _launch("qk", q, k, s, lens, 0, warps, path)
    return s


def pv(p, v, lens=None, *, out=None, kpv: int = PAD_K, warps: int | None = None, check_lens: bool = True,
       path: str = "fast"):
    """O (B, H, D) = p (B, H, T) times v (B, T, D) or (B, Hkv, T, D), PV's K = max(kpv, pad16(lens)); see the module."""

    p, v = _words(p, "p", False), _kv4(_words(v, "v", False))
    if p.dim() != 3 or v.dim() != 4 or v.shape[0] != p.shape[0] or v.shape[2] != p.shape[2]:
        raise ValueError(f"attn_bf16.pv: p {tuple(p.shape)}, v {tuple(v.shape)}")
    B, H, T, D = p.shape[0], p.shape[1], p.shape[2], v.shape[-1]
    lens, empty = _lens(lens, B, T, p.device, check_lens, "pv")
    o = _out(out, (B, H, D), p, "pv")
    if empty is None or not T:
        o.zero_()
    _launch("pv", p, v, o, lens, kpv, warps, path)
    if empty:
        o.masked_fill_((lens == 0).view(B, 1, 1), 0)
    return o
