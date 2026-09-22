"""Fused Triton kernels for the SIMT passes of ``sumcheck_packed`` (CUDA only; the torch-eager path stays the reference).

Eager torch runs the Hadamards, limb decompositions and recompositions as dozens of memory passes (an ``F_{p^6}``
product alone is ~150 kernels); each kernel here reads its operands once and writes its result once:

* :func:`ext_hadamard3_limbs` -- ``g_i(x, b) o h_i(x, b)`` for ``x in {0, 1, 2}`` in ``F_{p^6}`` and the 24 limbs of each,
  written straight into the matmul operand layout ``A[(b2, x, s, j), b1]``.
* :func:`packed_hadamard_limbs` -- the ``3 . 4^(i-1)`` base x base products ``g(a, x, b) h(a', x, b) mod p`` and their 4
  limbs, layout ``A[(b2, x, a, a', j), b1]``.
* :func:`fold_k` -- ``g'(b) = sum_a eq(r_1..r_k, a) g(a, b)`` (base x extension, SIMT: ``2^k . 6`` products per output;
  the equivalent matmul has ``K = 4 . 2^k <= 64`` and cuBLAS runs it at 0.1% of peak).
* :func:`fold_ext` -- the extension fold as ONE tensor-core kernel: limb decomposition of ``[x_e | x_o]`` in registers,
  ``tl.dot`` with the ``[C_(1-r); C_r]`` convolution matrix (``K = 64``: 48 limb rows + zero padding, ``N = 64``: 48
  limb-position columns + padding), recomposition ``sum_m 2^(8m) . acc[8d + m] mod p`` in registers, 6 coefficients out.
  Reads 48 B and writes 24 B per pair instead of cuBLAS's 48 B in / 192 B INT32 out plus two more passes.

Triton's front end has no containers, so the Hadamard kernels (36 coefficient products, ``HALF^2`` column pairs) are
generated as unrolled source (:func:`_generated`) into a cache file and imported.

All arithmetic is exact: products of two ``< 2^31`` values in INT64 are reduced before accumulation; limb products
``< 2^14`` accumulate over ``<= 64`` terms in INT32.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from . import reference as ref

P = ref.P
W = ref.W
POW2 = [pow(2, 8 * m, P) for m in range(8)]


@triton.jit
def _limbs(v):
    """4 balanced INT8 limbs of an int32 tensor in [0, p)."""
    l0 = ((v + 128) & 255) - 128
    v = (v - l0) >> 8
    l1 = ((v + 128) & 255) - 128
    v = (v - l1) >> 8
    l2 = ((v + 128) & 255) - 128
    v = (v - l2) >> 8
    l3 = ((v + 128) & 255) - 128
    return l0.to(tl.int8), l1.to(tl.int8), l2.to(tl.int8), l3.to(tl.int8)


# ---------------------------------------------------------------------------------------------------------------------
# generated Hadamard kernels
# ---------------------------------------------------------------------------------------------------------------------

_HEADER = f"""
import triton
import triton.language as tl
P = {P}
W = {W}

@triton.jit
def _limbs(v):
    l0 = ((v + 128) & 255) - 128
    v = (v - l0) >> 8
    l1 = ((v + 128) & 255) - 128
    v = (v - l1) >> 8
    l2 = ((v + 128) & 255) - 128
    v = (v - l2) >> 8
    l3 = ((v + 128) & 255) - 128
    return l0.to(tl.int8), l1.to(tl.int8), l2.to(tl.int8), l3.to(tl.int8)
"""


def _store4(lines: list[str], base_expr: str, val: str, ind: str) -> None:
    lines.append(f"{ind}l0, l1, l2, l3 = _limbs(({val}).to(tl.int32))")
    lines.append(f"{ind}_b = {base_expr}")
    for j in range(4):
        lines.append(f"{ind}tl.store(_b + {j} * K1, l{j}, mask=mask)")


def _ext_hadamard_source() -> str:
    L: list[str] = []
    L.append("@triton.jit")
    L.append("def ext_hadamard3(g_ptr, h_ptr, a_ptr, K1, BLOCK: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    b1 = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)")
    L.append("    mask = b1 < K1")
    L.append("    b = b2.to(tl.int64) * K1 + b1")
    L.append("    row = b * 12")
    L.append("    a_base = a_ptr + (b2.to(tl.int64) * 72) * K1 + b1")
    for s in range(6):
        L.append(f"    ge{s} = tl.load(g_ptr + row + {s}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    go{s} = tl.load(g_ptr + row + {6 + s}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    he{s} = tl.load(h_ptr + row + {s}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    ho{s} = tl.load(h_ptr + row + {6 + s}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    g2{s} = (go{s} * 2 - ge{s}) % P")
        L.append(f"    h2{s} = (ho{s} * 2 - he{s}) % P")
    for x, (gp, hp) in enumerate((("ge", "he"), ("go", "ho"), ("g2", "h2"))):
        for d in range(11):
            terms = [f"({gp}{i} * {hp}{d - i}) % P" for i in range(6) if 0 <= d - i < 6]
            L.append(f"    acc{d} = " + " + ".join(terms))
        for d in range(6):
            c = f"(acc{d} + (acc{d + 6} % P) * W) % P" if d < 5 else f"acc{d} % P"
            _store4(L, f"a_base + {x * 24 + 4 * d} * K1", c, "    ")
    return "\n".join(L) + "\n"


def _packed_hadamard_source(half: int) -> str:
    L: list[str] = []
    L.append("@triton.jit")
    L.append(f"def packed_hadamard_{half}(g_ptr, h_ptr, a_ptr, K1, BLOCK: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    b1 = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)")
    L.append("    mask = b1 < K1")
    L.append("    b = b2.to(tl.int64) * K1 + b1")
    L.append(f"    col = b * {2 * half}")
    rows = 3 * half * half
    L.append(f"    a_base = a_ptr + (b2.to(tl.int64) * {rows * 4}) * K1 + b1")
    for a in range(half):
        L.append(f"    g0_{a} = tl.load(g_ptr + col + {a}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    g1_{a} = tl.load(g_ptr + col + {half + a}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    g2_{a} = (g1_{a} * 2 - g0_{a}) % P")
        L.append(f"    h0_{a} = tl.load(h_ptr + col + {a}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    h1_{a} = tl.load(h_ptr + col + {half + a}, mask=mask, other=0).to(tl.int64)")
        L.append(f"    h2_{a} = (h1_{a} * 2 - h0_{a}) % P")
    for x in range(3):
        for a in range(half):
            for a2 in range(half):
                r = (x * half + a) * half + a2
                _store4(L, f"a_base + {r * 4} * K1", f"(g{x}_{a} * h{x}_{a2}) % P", "    ")
    return "\n".join(L) + "\n"


_MODULE = None


def _generated():
    """The module of generated kernels (written once to a cache file so ``triton.jit`` can read its source)."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    src = _HEADER + "\n" + _ext_hadamard_source() + "\n" + "\n".join(_packed_hadamard_source(1 << e) for e in range(4)) + "\n"
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("PACKED_KERNEL_CACHE") or Path(tempfile.gettempdir()) / "packed_kernels")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"gen_{digest}.py"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(src)
        os.replace(tmp, path)
    name = f"packed_kernels_gen_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE = mod
    return mod


def _block(K1: int, want: int) -> int:
    return min(want, max(16, triton.next_power_of_2(K1)))


def ext_hadamard3_limbs(gE: torch.Tensor, hE: torch.Tensor, K1: int, K2: int, BLOCK: int = 256) -> torch.Tensor:
    """``(t_i, 6) x 2 -> (K2 . 72, K1) int8`` (the ``A`` operand of the weighted inner product)."""
    A = torch.empty((K2 * 72, K1), dtype=torch.int8, device=gE.device)
    blk = _block(K1, BLOCK)
    _generated().ext_hadamard3[(K2, triton.cdiv(K1, blk))](gE, hE, A, K1, BLOCK=blk)
    return A


def packed_hadamard_limbs(g: torch.Tensor, h: torch.Tensor, i: int, K1: int, K2: int, BLOCK: int = 128) -> torch.Tensor:
    half = 1 << (i - 1)
    if half > 8:
        raise ValueError("generated kernels cover k <= 4 (HALF <= 8)")
    A = torch.empty((K2 * 3 * half * half * 4, K1), dtype=torch.int8, device=g.device)
    blk = _block(K1, BLOCK)
    getattr(_generated(), f"packed_hadamard_{half}")[(K2, triton.cdiv(K1, blk))](g, h, A, K1, BLOCK=blk)
    return A


# ---------------------------------------------------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _fold_k_kernel(g_ptr, c_ptr, out_ptr, m, P: tl.constexpr, TWOK: tl.constexpr, BLOCK: tl.constexpr):
    """out[b, s] = sum_a c[a, s] g[b TWOK + a] mod p; c (TWOK, 6) int32."""
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < m
    acc0 = tl.zeros([BLOCK], dtype=tl.int64)
    acc1 = tl.zeros([BLOCK], dtype=tl.int64)
    acc2 = tl.zeros([BLOCK], dtype=tl.int64)
    acc3 = tl.zeros([BLOCK], dtype=tl.int64)
    acc4 = tl.zeros([BLOCK], dtype=tl.int64)
    acc5 = tl.zeros([BLOCK], dtype=tl.int64)
    for a in range(TWOK):
        gv = tl.load(g_ptr + b.to(tl.int64) * TWOK + a, mask=mask, other=0).to(tl.int64)
        acc0 += (gv * tl.load(c_ptr + a * 6 + 0).to(tl.int64)) % P
        acc1 += (gv * tl.load(c_ptr + a * 6 + 1).to(tl.int64)) % P
        acc2 += (gv * tl.load(c_ptr + a * 6 + 2).to(tl.int64)) % P
        acc3 += (gv * tl.load(c_ptr + a * 6 + 3).to(tl.int64)) % P
        acc4 += (gv * tl.load(c_ptr + a * 6 + 4).to(tl.int64)) % P
        acc5 += (gv * tl.load(c_ptr + a * 6 + 5).to(tl.int64)) % P
    o = out_ptr + b.to(tl.int64) * 6
    tl.store(o + 0, (acc0 % P).to(tl.int32), mask=mask)
    tl.store(o + 1, (acc1 % P).to(tl.int32), mask=mask)
    tl.store(o + 2, (acc2 % P).to(tl.int32), mask=mask)
    tl.store(o + 3, (acc3 % P).to(tl.int32), mask=mask)
    tl.store(o + 4, (acc4 % P).to(tl.int32), mask=mask)
    tl.store(o + 5, (acc5 % P).to(tl.int32), mask=mask)


def fold_k(g: torch.Tensor, k: int, coeffs: torch.Tensor, BLOCK: int = 256) -> torch.Tensor:
    """``(t,) base -> (t / 2^k, 6)``; ``coeffs`` = ``eq(r_1..r_k, a)`` as ``(2^k, 6) int32``."""
    m = g.numel() >> k
    out = torch.empty((m, 6), dtype=torch.int32, device=g.device)
    _fold_k_kernel[(triton.cdiv(m, BLOCK),)](g, coeffs, out, m, P=P, TWOK=1 << k, BLOCK=BLOCK)
    return out


@triton.jit
def _fold_ext_kernel(gh_ptr, C_ptr, pow2_ptr, out_ptr, rows, P: tl.constexpr, BLOCK_M: tl.constexpr):
    """Row r of gh (rows, 12) int32 = [x_e (6) | x_o (6)]; C (64, 64) int8 (rows 4 s + j for s < 12, cols 8 d + m for d < 6);
    out (rows, 6) = coefficients of (1 - r) x_e + r x_o."""
    offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = offs < rows
    cols = tl.arange(0, 16)
    v = tl.load(gh_ptr + offs[:, None].to(tl.int64) * 12 + cols[None, :], mask=rmask[:, None] & (cols[None, :] < 12), other=0)
    l0, l1, l2, l3 = _limbs(v)
    # join appends a trailing axis: [.., c, p, q] with q = outer pair, p = inner -> column 4 c + 2 p + q must be limb j
    A = tl.reshape(tl.join(tl.join(l0, l2), tl.join(l1, l3)), (BLOCK_M, 64), can_reorder=False)   # column 4 s + j
    kk = tl.arange(0, 64)
    C = tl.load(C_ptr + kk[:, None] * 64 + kk[None, :])
    acc = tl.dot(A, C, out_dtype=tl.int32)                                            # (BM, 64): column 8 d + m
    z = tl.reshape(acc, (BLOCK_M, 8, 8)).to(tl.int64)
    pw = tl.load(pow2_ptr + tl.arange(0, 8)).to(tl.int64)
    z = (z * pw[None, None, :]) % P
    c = tl.sum(z, axis=2) % P
    d = tl.arange(0, 8)
    tl.store(out_ptr + offs[:, None].to(tl.int64) * 6 + d[None, :], c.to(tl.int32), mask=rmask[:, None] & (d[None, :] < 6))


def fold_ext_matrix(fold_C48: np.ndarray, device) -> torch.Tensor:
    """The (48 x 48) ``[C_(1-r); C_r]`` embedded in the kernel's (64 x 64) operand."""
    C = np.zeros((64, 64), dtype=np.int8)
    C[:48, :48] = fold_C48
    return torch.tensor(C, device=device)


def fold_ext(gh: torch.Tensor, C64: torch.Tensor, pow2: torch.Tensor, BLOCK_M: int = 128) -> torch.Tensor:
    """``gh (rows, 12) int32`` (``[x_e | x_o]`` per pair, both tables stacked) -> ``(rows, 6)`` folded coefficients."""
    rows = gh.shape[0]
    out = torch.empty((rows, 6), dtype=torch.int32, device=gh.device)
    _fold_ext_kernel[(triton.cdiv(rows, BLOCK_M),)](gh, C64, pow2, out, rows, P=P, BLOCK_M=BLOCK_M, num_warps=4)
    return out
