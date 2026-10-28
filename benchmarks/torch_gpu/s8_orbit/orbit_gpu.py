"""s8_orbit.cubin (build.sh's) from Python: one fused Strassen level of the int8 GEMM on sm_90a, on torch tensors.

`Lib(cubin)` loads it into torch's context and checks its `s8orbit_layout` against this file. `b_forms(w)` makes B's seven
wrapped forms of a weight w [N, K] int8 once (orbit.B_FORMS, as one [7, N/2, K/2] int8 tensor). `Lib.mm(x, wb, ...)`
launches `s8orbit_<BN>` on torch's current stream: x [M, K] int8 against those forms, into y32 [M, N] int32, or into
y [M, N] bf16 with s_a [M] and s_w [N] (the exact linear's words). The kernel makes no carry corrections: its output is
X·Wᵀ exactly when no form carries (`Lib.flags` counts A's carry bits since `Lib.reset`, `b_forms` returns B's), and
X·Wᵀ mod 2^8 always. `corrections(x, w)` is the rest, 256·(c_A·w_Bᵀ + w_A·c_Bᵀ) + 65536·c_A·c_Bᵀ assembled, in torch.
"""
from __future__ import annotations

import ctypes as C
import hashlib
from pathlib import Path

import torch

TILES = (96, 80, 64)
BK, TM, XCH, MAX_STAGES, SMEM_LIMIT, NT = 32, 64, 8, 8, 232448, 288
#: CUtensorMapDataType UINT8, CUtensorMapSwizzle 32B, CUtensorMapL2promotion 128B; CUfunction_attribute
#: MAX_DYNAMIC_SHARED_SIZE_BYTES; CUdevice_attribute MULTIPROCESSOR_COUNT
TMAP_UINT8, TMAP_SWIZZLE_32B, TMAP_L2_128B, MAX_DYN_SMEM, SM_COUNT = 0, 1, 2, 8, 16
A_FORMS = ((("11", 1), ("22", 1)), (("21", 1), ("22", 1)), (("11", 1),), (("22", 1),), (("11", 1), ("12", 1)),
           (("21", 1), ("11", -1)), (("12", 1), ("22", -1)))
B_FORMS = ((("11", 1), ("22", 1)), (("11", 1),), (("12", 1), ("22", -1)), (("21", 1), ("11", -1)), (("22", 1),),
           (("11", 1), ("12", 1)), (("21", 1), ("22", 1)))
C_SUMS = {"11": ((0, 1), (3, 1), (4, -1), (6, 1)), "12": ((2, 1), (4, 1)), "21": ((1, 1), (3, 1)),
          "22": ((0, 1), (1, -1), (2, 1), (5, 1))}

u, i32, vp, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_uint64, C.POINTER
CUDA = {
    "cuDeviceGet": [P(i32), i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuDeviceGetAttribute": [P(i32), i32, i32],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuGetErrorString": [i32, P(C.c_char_p)],
    "cuTensorMapEncodeTiled": [vp, i32, u, vp, P(u64), P(u64), P(u), P(u), i32, i32, i32, i32],
}


class Params(C.Structure):
    _fields_ = [("a", C.c_ubyte * 128), ("b", C.c_ubyte * 128), ("sa", u64), ("sw", u64), ("y", u64), ("y32", u64),
                ("flags", u64), ("clk", u64), ("M", C.c_int32), ("N", C.c_int32), ("K", C.c_int32), ("group", C.c_int32),
                ("pad", C.c_ubyte * 64)]


assert C.sizeof(Params) == 384 and Params.sa.offset == 256 and Params.M.offset == 304


def stage_bytes(bn: int) -> int:
    return 4 * TM * BK + 7 * bn * BK + 5 * TM * BK


def stages(bn: int) -> int:
    return min((SMEM_LIMIT - 1024 - 3 * XCH * 128 * 4) // (stage_bytes(bn) + 16), MAX_STAGES)


def smem(bn: int) -> int:
    return 1024 + stages(bn) * stage_bytes(bn) + 3 * XCH * 128 * 4 + 16 * stages(bn)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def check_shape(m: int, n: int, k: int) -> None:
    if m < 2 or m % 2 or n % 4 or k % 64:
        raise ValueError(f"M {m} N {n} K {k}: the orbit needs M even, N a multiple of 4 and K of 64")


def blocks(t: torch.Tensor, rows_first: bool) -> dict[str, torch.Tensor]:
    """A's blocks of X (rows_first) or B's blocks of W [N, K] (B12 = W[n1, k0], B21 = W[n0, k1])."""
    h, hk = t.shape[0] // 2, t.shape[1] // 2
    if rows_first:
        return {"11": t[:h, :hk], "12": t[:h, hk:], "21": t[h:, :hk], "22": t[h:, hk:]}
    return {"11": t[:h, :hk], "12": t[h:, :hk], "21": t[:h, hk:], "22": t[h:, hk:]}


def forms(t: torch.Tensor, which: tuple, rows_first: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Seven wrapped int8 forms [7, R/2, K/2] and their carries (int8, -1/0/1)."""
    bl = blocks(t, rows_first)
    ws, cs = [], []
    for terms in which:
        f = sum(s * bl[name].to(torch.int16) for name, s in terms)
        w = ((f + 128) & 0xFF) - 128
        ws.append(w.to(torch.int8))
        cs.append(((f - w) // 256).to(torch.int8))
    return torch.stack(ws).contiguous(), torch.stack(cs).contiguous()


def b_forms(w: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """B's forms of w [N, K] int8 as the kernel reads them, and each form's carry count."""
    check_shape(2, w.shape[0], w.shape[1])
    wb, cb = forms(w, B_FORMS, False)
    return wb, [int(torch.count_nonzero(cb[j])) for j in range(7)]


def _int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a·bᵀ int32 of int8 a [R, K], b [S, K] (cuBLASLt), exact."""
    return torch._int_mm(a.contiguous(), b.contiguous().t())


def assemble(p: list[torch.Tensor]) -> torch.Tensor:
    q = {name: sum(s * p[j] for j, s in terms) for name, terms in C_SUMS.items()}
    return torch.cat([torch.cat([q["11"], q["12"]], 1), torch.cat([q["21"], q["22"]], 1)], 0)


def corrections(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """What the kernel leaves out, [M, N] int64: O - (the wrapped forms' orbit), from both sides' carries."""
    wa, ca = forms(x, A_FORMS, True)
    wb, cb = forms(w, B_FORMS, False)
    p = []
    for j in range(7):
        acc = torch.zeros((x.shape[0] // 2, w.shape[0] // 2), dtype=torch.int64, device=x.device)
        if ca[j].any():
            acc += 256 * _int_mm(ca[j], wb[j]).long()
        if cb[j].any():
            acc += 256 * _int_mm(wa[j], cb[j]).long()
        if ca[j].any() and cb[j].any():
            acc += 65536 * _int_mm(ca[j], cb[j]).long()
        p.append(acc)
    return assemble(p)


class Lib:
    """The cubin in torch's context on the current device: its kernels, their shared memory, the tensor maps."""

    def __init__(self, path: Path, libcuda: str = "libcuda.so.1"):
        self.path = Path(path)
        self.cubin = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.cubin).hexdigest()
        self.cu = C.CDLL(libcuda)
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
        sms = i32()
        self.ck("cuDeviceGetAttribute", C.byref(sms), SM_COUNT, dev)
        self.sms = sms.value
        self.fns = {}
        for bn in TILES:
            f = self.fn(f"s8orbit_{bn}")
            self.ck("cuFuncSetAttribute", f, MAX_DYN_SMEM, smem(bn))
            self.fns[bn] = f
        self.maps: dict[tuple, bytes] = {}
        self.flags_t = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.layout = self._layout()

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

    def _launch(self, f: vp, grid: int, threads: int, shared: int, param) -> None:
        params = (vp * 1)(C.addressof(param))
        self.ck("cuLaunchKernel", f, grid, 1, 1, threads, 1, 1, shared, vp(torch.cuda.current_stream().cuda_stream), params, None)

    def _layout(self) -> dict:
        out = torch.zeros(16, dtype=torch.int32, device="cuda")
        self._launch(self.fn("s8orbit_layout"), 1, 1, 0, u64(out.data_ptr()))
        got = out.tolist()
        want = [C.sizeof(Params), Params.sa.offset, Params.M.offset] + [v for bn in TILES for v in (smem(bn), stages(bn))]
        if got[:len(want)] != want:
            raise RuntimeError(f"{self.path}'s s8orbit_layout {got[:len(want)]} is not orbit_gpu.py's {want}: rebuild or update")
        return {f"n{bn}": {"smem": smem(bn), "stages": stages(bn)} for bn in TILES}

    def _encode(self, key: tuple, ptr: int, dims: tuple, strides: tuple, box: tuple) -> bytes:
        if key not in self.maps:
            raw = C.create_string_buffer(128 + 64)
            at = (C.addressof(raw) + 63) & ~63
            r = len(dims)
            self.ck("cuTensorMapEncodeTiled", vp(at), TMAP_UINT8, r, vp(ptr), (u64 * r)(*dims), (u64 * (r - 1))(*strides),
                    (u * r)(*box), (u * r)(*([1] * r)), 0, TMAP_SWIZZLE_32B, TMAP_L2_128B, 0)
            self.maps[key] = C.string_at(at, 128)
        return self.maps[key]

    def tmap_a(self, x: torch.Tensor) -> bytes:
        """X [M, K] as [2 halves of M][M/2 rows][2 halves of K][K/2 bytes], boxes of 32 bytes x 64 rows x 2 x 2."""
        m, k = x.shape
        return self._encode(("a", x.data_ptr(), m, k), x.data_ptr(), (k // 2, m // 2, 2, 2), (k, k // 2, (m // 2) * k),
                            (BK, TM, 2, 2))

    def tmap_b(self, wb: torch.Tensor, bn: int) -> bytes:
        """B's forms [7, N/2, K/2], boxes of 32 bytes x BN rows x 7."""
        _, hn, hk = wb.shape
        return self._encode(("b", wb.data_ptr(), hn, hk, bn), wb.data_ptr(), (hk, hn, 7), (hk, hn * hk), (BK, bn, 7))

    def grid(self, tiles: int, ctas: int = 0) -> int:
        cap = min(ctas or self.sms, self.sms)
        return cdiv(tiles, cdiv(tiles, cap))

    def reset(self) -> None:
        self.flags_t.zero_()

    @property
    def flags(self) -> int:
        return int(self.flags_t.item())

    def mm(self, x: torch.Tensor, wb: torch.Tensor, y32: torch.Tensor | None = None, y: torch.Tensor | None = None,
           sa: torch.Tensor | None = None, sw: torch.Tensor | None = None, bn: int = 96, ctas: int = 0, group: int = 8,
           clk: torch.Tensor | None = None) -> dict:
        m, k = x.shape
        _, hn, hk = wb.shape
        n = 2 * hn
        check_shape(m, n, k)
        assert x.dtype == torch.int8 and wb.dtype == torch.int8 and x.is_contiguous() and wb.is_contiguous() and hk == k // 2
        assert (y32 is None) != (y is None), "one of y32 and y"
        if y32 is not None:
            assert y32.dtype == torch.int32 and y32.shape == (m, n) and y32.is_contiguous()
        else:
            assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.is_contiguous()
            assert sa is not None and sw is not None and sa.numel() == m and sw.numel() == n
        p = Params()
        C.memmove(C.addressof(p) + Params.a.offset, self.tmap_a(x), 128)
        C.memmove(C.addressof(p) + Params.b.offset, self.tmap_b(wb, bn), 128)
        p.sa = sa.data_ptr() if sa is not None else 0
        p.sw = sw.data_ptr() if sw is not None else 0
        p.y = y.data_ptr() if y is not None else 0
        p.y32 = y32.data_ptr() if y32 is not None else 0
        p.flags = self.flags_t.data_ptr()
        p.M, p.N, p.K, p.group = m, n, k, group
        tiles = cdiv(m // 2, TM) * cdiv(hn, bn)
        grid = self.grid(tiles, ctas)
        if clk is not None:
            assert clk.dtype == torch.int64 and clk.numel() >= 4 * grid
            p.clk = clk.data_ptr()
        self._launch(self.fns[bn], grid, NT, smem(bn), p)
        return {"grid": grid, "tiles": tiles, "bn": bn, "stages": stages(bn)}
