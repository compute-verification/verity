"""The witness program as ONE fused CUDA kernel (lane hostphase): ``System.program`` compiled to straight-line CUDA C,
one thread per column, every derived row computed and stored in program order -- the same integer/mod-p arithmetic
``witness._run_program`` performs with ~4000 torch kernels per proof (which were launched eagerly: the CUDA-graph
capture of the torch program fails behind a failed capture of the hint generator, see ``hints_torch``).

Exactness: every value is a residue in [0, p) computed exactly (per-term reduction of ``coef * W_j`` in 64 bits, the
same Python-int constants reduced mod p at code-generation time, ``bits`` = ``(x >> b) & 1`` on the residue, ``sels``
on the signed representative ``x - p`` for ``x > p // 2``, ``inv`` by Fermat with ``0 -> 0``), so ``W`` is bit-identical
to the torch program's (asserted by ``witness_device_test.py`` and by the dump-tree comparison against main).  The
output is int32 (values < p < 2^31): the SIMT encoder, the fused tests and the mutation hooks all accept it.
"""
from __future__ import annotations

import hashlib

import torch

from .compile import Expr, P, System

__all__ = ["FusedWitness", "available", "fused_for"]

THREADS = 256

_HEAD = r"""
#define P 2013265921ULL
#define HALF_P 1006632960LL
typedef unsigned long long u64;
typedef long long i64;

__device__ __forceinline__ u64 modp_signed(i64 v) { i64 r = v % (i64)P; return (u64)(r < 0 ? r + (i64)P : r); }
__device__ __forceinline__ u64 modinv_p(u64 x) {   /* Fermat: x^(p-2) mod p; 0 -> 0 like batch_modinv */
    u64 result = 1, base = x % P; u64 e = P - 2;
    while (e) { if (e & 1) result = (result * base) % P; base = (base * base) % P; e >>= 1; }
    return result;
}

extern "C" __global__ void __launch_bounds__(@THREADS@)
witness_program(const i64* __restrict__ pub, const i64* __restrict__ hints, unsigned* __restrict__ Wm, int l)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= l) return;
#define W(j) Wm[(size_t)(j) * l + col]
#define PUB(i) pub[(size_t)(i) * l + col]
#define HINT(i) hints[(size_t)(i) * l + col]
    u64 x, y; i64 sx;
"""

_TAIL = r"""
}
"""

# Dense programs (lane hash-relation: the Poseidon2 sponges' S-box operands are 24-term affine forms, ~74k terms per
# unit against ~2k for the v1 units) are emitted TABLE-DRIVEN: an operand with >= TAB_MIN_TERMS terms becomes one
# ``affine(J, C, off, n, c)`` loop over a (row, coefficient) table passed as two extra kernel arguments, instead of
# unrolled straight-line code (2.7 MB of source that NVRTC needs > 10 min to optimise).  Same residue per term, same
# summation order, so W is identical; the mode only switches on when the unrolled form would exceed TAB_THRESHOLD terms,
# so every v1 system compiles to exactly the source it did before.
TAB_THRESHOLD = 8000
TAB_MIN_TERMS = 3
# In table mode the program is also cut into __noinline__ device functions of CHUNK_OPS operations each and ``affine``
# is not inlined: NVRTC's optimiser is superlinear in the size of a straight-line function (one 6k-statement kernel
# with 1.8k inlined loops took ~9 min on the 4090 pod; ``affine`` __noinline__ 32 s; 8 chunks 36 s, 29 chunks 22 s,
# 116 chunks 20 s -- the floor is the front end).  W is unchanged: the ops are independent except through the W rows
# they read and write, which are in program order in both forms; ``bits`` is emitted as a loop (same residue bits).
CHUNK_OPS = 64

_AFFINE = r"""
__device__ __noinline__ u64 affine(const int* __restrict__ J, const unsigned* __restrict__ C, int off, int n, u64 c,
                                   const unsigned* __restrict__ Wm, int l, int col) {
    u64 acc = 0;
    for (int t = off; t < off + n; ++t) acc += ((u64)Wm[(size_t)J[t] * l + col] * (u64)C[t]) % P;
    return (acc + c) % P;
}
#define W(j) Wm[(size_t)(j) * l + col]
#define PUB(i) pub[(size_t)(i) * l + col]
#define HINT(i) hints[(size_t)(i) * l + col]
"""

_CHUNK_HEAD = r"""
__device__ __noinline__ void chunk_@K@(const i64* __restrict__ pub, const i64* __restrict__ hints,
                                       unsigned* __restrict__ Wm, int l, int col,
                                       const int* __restrict__ J, const unsigned* __restrict__ C)
{
    u64 x, y; i64 sx;
"""

_KERNEL_TAB = r"""
extern "C" __global__ void __launch_bounds__(@THREADS@)
witness_program(const i64* __restrict__ pub, const i64* __restrict__ hints, unsigned* __restrict__ Wm, int l,
                const int* __restrict__ J, const unsigned* __restrict__ C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= l) return;
"""


def _eval(e: Expr, tab: list | None = None) -> str:
    """The C expression of an affine ``Expr`` on the column: exact residue in [0, p)."""
    c = e.c % P
    if not e.lin:
        return f"{c}ULL"
    if tab is not None and len(e.lin) >= TAB_MIN_TERMS:
        off = len(tab)
        tab.extend((j, v % P) for j, v in e.lin.items())
        return f"affine(J, C, {off}, {len(e.lin)}, {c}ULL, Wm, l, col)"
    terms = " + ".join(f"((u64)W({j}) * {v % P}ULL) % P" for j, v in e.lin.items())
    return f"(({terms}) + {c}ULL) % P"


def _n_terms(program: list) -> int:
    n = 0
    for op in program:
        for x in op[2:]:
            if isinstance(x, Expr):
                n += len(x.lin)
    return n


def _program(sys: System) -> list:
    """``sys.program`` without the product rows a witness hook writes (``sys.witness_hooks``: objects with ``rows``,
    the row indices they fill, and ``launch(cp, W_u32, l)`` run after the program on the same stream; hashchain.py's
    Poseidon2 kernel).  Nothing in the program reads a hook's rows, so the order is immaterial."""
    hooks = getattr(sys, "witness_hooks", None) or ()
    skip = set().union(*(h.rows for h in hooks)) if hooks else set()
    return [op for op in sys.program if not (op[0] == "prod" and op[1] in skip)]


def generate(sys: System, tab: list | None = None, program: list | None = None) -> str:
    """The kernel source; with ``tab`` (a list the caller keeps) dense operands are appended to it as (row, coef)."""
    hpos = {n: i for i, (_, n) in enumerate(sys.hints)}
    ppos = {n: i for i, (_, n) in enumerate(sys.pins)}
    stmts: list[list[str]] = []          # one list of C statements per program op
    for op in (sys.program if program is None else program):
        kind = op[0]
        out: list[str] = []
        if kind == "pin":
            _, idx, name = op
            out.append(f"    W({idx}) = (unsigned)PUB({ppos[name]});")
        elif kind == "hint":
            _, idx, name = op
            out.append(f"    W({idx}) = (unsigned)modp_signed(HINT({hpos[name]}));")
        elif kind == "prod":
            _, idx, a, b = op
            out.append(f"    x = {_eval(a, tab)}; y = {_eval(b, tab)}; W({idx}) = (unsigned)((x * y) % P);")
        elif kind == "bits":
            _, row0, nbits, e = op
            out.append(f"    x = {_eval(e, tab)};")
            if tab is not None:
                out.append(f"    for (int b = 0; b < {nbits}; ++b) W({row0} + b) = (unsigned)((x >> b) & 1ULL);")
            else:
                for b in range(nbits):
                    out.append(f"    W({row0 + b}) = (unsigned)((x >> {b}) & 1ULL);")
        elif kind == "sels":
            _, row0, k0s, k1s, key = op
            out.append(f"    x = {_eval(key, tab)}; sx = (x > (u64)HALF_P) ? (i64)x - (i64)P : (i64)x;")
            for i, (k0, k1) in enumerate(zip(k0s, k1s)):
                if not (-(2**63) <= k0 < 2**63 and -(2**63) <= k1 < 2**63):
                    raise ValueError("selector bound outside int64")
                out.append(f"    W({row0 + i}) = (unsigned)((sx >= {int(k0)}LL) && (sx <= {int(k1)}LL));")
        elif kind == "inv":
            _, idx, x = op
            out.append(f"    x = {_eval(x, tab)}; W({idx}) = (unsigned)modinv_p(x);")
        else:
            raise ValueError(kind)
        stmts.append(out)
    if tab is None:                      # the v1 form: one straight-line kernel, source unchanged
        head = _HEAD.replace("@THREADS@", str(THREADS))
        return "\n".join([head, *(s for out in stmts for s in out), _TAIL])
    pre, _, _ = _HEAD.partition("extern \"C\" __global__")
    src = [pre, _AFFINE]
    n_chunks = -(-len(stmts) // CHUNK_OPS) if stmts else 0
    for k in range(n_chunks):
        src.append(_CHUNK_HEAD.replace("@K@", str(k)))
        src.extend(s for out in stmts[k * CHUNK_OPS:(k + 1) * CHUNK_OPS] for s in out)
        src.append("}")
    src.append(_KERNEL_TAB.replace("@THREADS@", str(THREADS)))
    src.extend(f"    chunk_{k}(pub, hints, Wm, l, col, J, C);" for k in range(n_chunks))
    src.append(_TAIL)
    return "\n".join(src)


def available() -> bool:
    try:
        import cupy  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class FusedWitness:
    """``run(pub_rows (n_pins, l) int64, hints (n_hints, l) int64) -> W (m, l) int32`` on the device."""

    def __init__(self, sys: System, device):
        import cupy as cp
        self.cp, self.sys, self.device = cp, sys, torch.device(device)
        self.hooks = list(getattr(sys, "witness_hooks", None) or ())
        program = _program(sys)
        tab: list | None = [] if _n_terms(program) > TAB_THRESHOLD else None
        self.src = generate(sys, tab, program)
        self.digest = hashlib.sha256(self.src.encode()).hexdigest()
        self.mod = cp.RawModule(code=self.src, options=("-std=c++17",))
        self.kernel = self.mod.get_function("witness_program")
        self.tables = None
        if tab is not None:
            import numpy as np
            with cp.cuda.Device(self.device.index or 0):
                self.tables = (cp.asarray(np.array([j for j, _ in tab], dtype=np.int32)),
                               cp.asarray(np.array([v for _, v in tab], dtype=np.uint32)))

    def run(self, pub_rows: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        import numpy as np
        cp = self.cp
        l = hints.shape[1]
        assert pub_rows.shape == (len(self.sys.pins), l) and hints.shape[0] == len(self.sys.hints)
        pub_rows = pub_rows.to(torch.int64).contiguous()
        hints = hints.to(torch.int64).contiguous()
        W = torch.empty((self.sys.m, l), dtype=torch.int32, device=self.device)
        import warnings
        ts = torch.cuda.current_stream(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stream = cp.cuda.ExternalStream(ts.cuda_stream)
        W_u32 = cp.asarray(W).view(cp.uint32)
        args = (cp.asarray(pub_rows), cp.asarray(hints), W_u32, np.int32(l))
        if self.tables is not None:
            args = args + self.tables
        with stream:
            self.kernel((-(-l // THREADS),), (THREADS,), args)
            for h in self.hooks:
                h.launch(cp, W_u32, l)
        return W


_FW: dict = {}


def fused_for(sys: System, device) -> FusedWitness | None:
    """The fused program for this system on this CUDA device, or None (CPU / no cupy / NVRTC failure)."""
    if not str(device).startswith("cuda") or not available():
        return None
    key = (id(sys), str(device))
    hit = _FW.get(key)
    if hit is not None and hit[0] is sys:
        return hit[1]
    try:
        f = FusedWitness(sys, device)
    except Exception as e:  # noqa: BLE001 -- no NVRTC: the torch program
        import warnings
        warnings.warn(f"fused witness kernel unavailable ({e!r}); the torch program runs instead")
        f = None
    _FW[key] = (sys, f)
    return f
