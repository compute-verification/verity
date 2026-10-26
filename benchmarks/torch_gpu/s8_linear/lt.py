"""cuBLASLt's plain int8 GEMM through the cuBLASLt torch already loaded, by ctypes: TN (W [N, K] and X [M, K] both K-major),
s8 x s8 -> s32 out, alpha 1, beta 0, the harness's int8 family (benchmarks/pouw/harness/native/lt.cu) without its separate
library. `Plan(m, n, k).candidates()` is cuBLASLt's top `want` heuristic algorithms; bench.py gates each (z equal to
torch._int_mm's) and times them, and keeps the fastest. Graph-capturable: one 32 MiB workspace, the stream torch's current."""
from __future__ import annotations

import ctypes as C
from functools import cache

import torch

CUDA_R_8I, CUDA_R_32I = 3, 10
CUBLAS_COMPUTE_32I = 72
DESC_TRANSA, DESC_TRANSB = 3, 4
OP_N, OP_T = 0, 1
PREF_MAX_WORKSPACE_BYTES = 1
CONFIG = {"algo_id": (0, C.c_int32), "tile": (1, C.c_uint32), "splitk": (2, C.c_int32), "reduction": (3, C.c_uint32),
          "swizzle": (4, C.c_uint32), "custom": (5, C.c_uint32), "stages": (6, C.c_uint32)}
WORKSPACE = 32 << 20


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [("algo", Algo), ("workspace", C.c_size_t), ("state", C.c_int), ("waves", C.c_float), ("reserved", C.c_int * 4)]


@cache
def lib() -> tuple[C.CDLL, str]:
    """The libcublasLt torch's torch._int_mm loaded (forced by one call), and its path."""
    torch._int_mm(torch.zeros(32, 32, dtype=torch.int8, device="cuda"), torch.zeros(32, 32, dtype=torch.int8, device="cuda"))
    torch.cuda.synchronize()
    with open("/proc/self/maps") as f:
        paths = sorted({line.split()[-1] for line in f if "libcublasLt.so" in line})
    if not paths:
        raise RuntimeError("torch loaded no libcublasLt")
    L = C.CDLL(paths[0])
    vp, pvp = C.c_void_p, C.POINTER(C.c_void_p)
    L.cublasLtGetVersion.restype = C.c_size_t
    L.cublasLtCreate.argtypes = [pvp]
    L.cublasLtMatmulDescCreate.argtypes = [pvp, C.c_int, C.c_int]
    L.cublasLtMatmulDescSetAttribute.argtypes = [vp, C.c_int, vp, C.c_size_t]
    L.cublasLtMatrixLayoutCreate.argtypes = [pvp, C.c_int, C.c_uint64, C.c_uint64, C.c_int64]
    L.cublasLtMatmulPreferenceCreate.argtypes = [pvp]
    L.cublasLtMatmulPreferenceSetAttribute.argtypes = [vp, C.c_int, vp, C.c_size_t]
    L.cublasLtMatmulAlgoGetHeuristic.argtypes = [vp, vp, vp, vp, vp, vp, vp, C.c_int, C.POINTER(Heuristic), C.POINTER(C.c_int)]
    L.cublasLtMatmulAlgoConfigGetAttribute.argtypes = [C.POINTER(Algo), C.c_int, vp, C.c_size_t, C.POINTER(C.c_size_t)]
    L.cublasLtMatmul.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, C.POINTER(Algo), vp, C.c_size_t, vp]
    for name in ("cublasLtMatmulDescDestroy", "cublasLtMatrixLayoutDestroy", "cublasLtMatmulPreferenceDestroy"):
        getattr(L, name).argtypes = [vp]
    return L, paths[0]


def _ck(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what}: cublasStatus {status}")


@cache
def handle() -> int:
    L, _ = lib()
    h = C.c_void_p()
    _ck(L.cublasLtCreate(C.byref(h)), "cublasLtCreate")
    return h.value


@cache
def workspace() -> torch.Tensor:
    return torch.empty(WORKSPACE, dtype=torch.uint8, device="cuda")


def version() -> dict:
    L, path = lib()
    return {"cublaslt": int(L.cublasLtGetVersion()), "path": path}


class Plan:
    """One (m, n, k): D [M, N] s32 row-major = (W [N, K]) (X [M, K])^T in cuBLASLt's column-major terms: A = W as K x N, op T;
    B = X as K x M, op N; C = D as N x M."""

    def __init__(self, m: int, n: int, k: int):
        L, _ = lib()
        self.m, self.n, self.k = m, n, k
        self.desc, self.a, self.b, self.c, self.pref = (C.c_void_p() for _ in range(5))
        _ck(L.cublasLtMatmulDescCreate(C.byref(self.desc), CUBLAS_COMPUTE_32I, CUDA_R_32I), "desc")
        t, nn = C.c_int32(OP_T), C.c_int32(OP_N)
        _ck(L.cublasLtMatmulDescSetAttribute(self.desc, DESC_TRANSA, C.byref(t), 4), "transa")
        _ck(L.cublasLtMatmulDescSetAttribute(self.desc, DESC_TRANSB, C.byref(nn), 4), "transb")
        _ck(L.cublasLtMatrixLayoutCreate(C.byref(self.a), CUDA_R_8I, k, n, k), "layout a")
        _ck(L.cublasLtMatrixLayoutCreate(C.byref(self.b), CUDA_R_8I, k, m, k), "layout b")
        _ck(L.cublasLtMatrixLayoutCreate(C.byref(self.c), CUDA_R_32I, n, m, n), "layout c")
        _ck(L.cublasLtMatmulPreferenceCreate(C.byref(self.pref)), "pref")
        ws = C.c_uint64(WORKSPACE)
        _ck(L.cublasLtMatmulPreferenceSetAttribute(self.pref, PREF_MAX_WORKSPACE_BYTES, C.byref(ws), 8), "pref ws")
        self.alpha, self.beta = C.c_int32(1), C.c_int32(0)

    def candidates(self, want: int = 32) -> list[Heuristic]:
        L, _ = lib()
        arr, got = (Heuristic * want)(), C.c_int()
        _ck(L.cublasLtMatmulAlgoGetHeuristic(handle(), self.desc, self.a, self.b, self.c, self.c, self.pref, want, arr,
                                             C.byref(got)), "heuristic")
        out = []
        for i in range(got.value):
            if arr[i].state == 0:
                h = Heuristic()
                C.memmove(C.byref(h), C.byref(arr[i]), C.sizeof(h))
                out.append(h)
        return out

    def run(self, h: Heuristic, x: torch.Tensor, w: torch.Tensor, z: torch.Tensor) -> None:
        L, _ = lib()
        ws = workspace()
        _ck(L.cublasLtMatmul(handle(), self.desc, C.byref(self.alpha), w.data_ptr(), self.a, x.data_ptr(), self.b,
                             C.byref(self.beta), z.data_ptr(), self.c, z.data_ptr(), self.c, C.byref(h.algo), ws.data_ptr(),
                             WORKSPACE, torch.cuda.current_stream().cuda_stream), "cublasLtMatmul")

    def __del__(self):
        try:
            L, _ = lib()
            L.cublasLtMatmulPreferenceDestroy(self.pref)
            for lay in (self.a, self.b, self.c):
                L.cublasLtMatrixLayoutDestroy(lay)
            L.cublasLtMatmulDescDestroy(self.desc)
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


def config(h: Heuristic) -> dict:
    L, _ = lib()
    out = {}
    for name, (attr, ty) in CONFIG.items():
        v, written = ty(), C.c_size_t()
        L.cublasLtMatmulAlgoConfigGetAttribute(C.byref(h.algo), attr, C.byref(v), C.sizeof(v), C.byref(written))
        out[name] = int(v.value)
    out["workspace"] = int(h.workspace)
    return out


def name(h: Heuristic, rank: int) -> str:
    c = config(h)
    return f"lt{rank}_algo{c['algo_id']}_tile{c['tile']}_st{c['stages']}_sk{c['splitk']}"
