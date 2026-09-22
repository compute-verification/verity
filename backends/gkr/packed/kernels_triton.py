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
P: tl.constexpr = {P}
W: tl.constexpr = {W}

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
    lines.append(f"{ind}_v = {val}")
    lines.append(f"{ind}_v = tl.where(_v < 0, _v + P, _v)")   # Triton % truncates toward zero
    lines.append(f"{ind}l0, l1, l2, l3 = _limbs(_v.to(tl.int32))")
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


def _join_tree(L: list[str], names: list[str], ind: str) -> str:
    """Emit joins so that ``tl.reshape(result, (BLOCK, len(names)))`` has column ``c`` = ``names[c]``: the innermost
    join must split on the most significant bit of the column index (the last join axis is the fastest one)."""
    level = list(names)
    depth = 0
    while len(level) > 1:
        n = len(level) // 2
        nxt = []
        for c in range(n):
            nm = f"_j{depth}_{c}"
            L.append(f"{ind}{nm} = tl.join({level[c]}, {level[c + n]})")
            nxt.append(nm)
        level = nxt
        depth += 1
    return level[0]


def _packed_ip_source(half: int) -> str:
    """Packed round ``i`` (``half = 2^(i-1)``) fused: Hadamard limbs formed in registers, ``tl.dot`` against the eq-weight
    limbs, INT32 accumulation over a ``b1`` split; nothing is written to HBM but the ``(ROWS 4) x 32`` partial sums."""
    rows = 3 * half * half
    rows4 = rows * 4
    rows_pad = max(16, triton.next_power_of_2(rows4))
    L: list[str] = []
    L.append("@triton.jit")
    L.append(f"def packed_ip_{half}(g_ptr, h_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    offs_n = tl.arange(0, 32)")
    L.append(f"    acc = tl.zeros(({rows_pad}, 32), dtype=tl.int32)")
    L.append("    zero8 = tl.zeros((BLOCK_K,), dtype=tl.int8)")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK_K):")
    L.append("        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)")
    L.append(f"        col = (b2.to(tl.int64) * K1 + b1) * {2 * half}")
    L.append("        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])")
    for a in range(half):
        L.append(f"        g0_{a} = tl.load(g_ptr + col + {a}).to(tl.int64)")
        L.append(f"        g1_{a} = tl.load(g_ptr + col + {half + a}).to(tl.int64)")
        L.append(f"        g2_{a} = (g1_{a} * 2 - g0_{a}) % P")
        L.append(f"        g2_{a} = tl.where(g2_{a} < 0, g2_{a} + P, g2_{a})")
        L.append(f"        h0_{a} = tl.load(h_ptr + col + {a}).to(tl.int64)")
        L.append(f"        h1_{a} = tl.load(h_ptr + col + {half + a}).to(tl.int64)")
        L.append(f"        h2_{a} = (h1_{a} * 2 - h0_{a}) % P")
        L.append(f"        h2_{a} = tl.where(h2_{a} < 0, h2_{a} + P, h2_{a})")
    names = []
    for x in range(3):
        for a in range(half):
            for a2 in range(half):
                r = (x * half + a) * half + a2
                L.append(f"        _v = (g{x}_{a} * h{x}_{a2}) % P")
                L.append(f"        l{r}_0, l{r}_1, l{r}_2, l{r}_3 = _limbs(_v.to(tl.int32))")
                names += [f"l{r}_{j}" for j in range(4)]
    names += ["zero8"] * (rows_pad - rows4)
    top = _join_tree(L, names, "        ")
    L.append(f"        A = tl.reshape({top}, (BLOCK_K, {rows_pad}), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append(f"    offs_m = tl.arange(0, {rows_pad})")
    L.append(f"    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * {rows4}) * 24")
    L.append(f"    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=(offs_m[:, None] < {rows4}) & (offs_n[None, :] < 24))")
    return "\n".join(L) + "\n"


def _ext_ip_source() -> str:
    """Extension round fused: the 3 ``F_{p^6}`` Hadamards per pair and their 24 limbs each (72 rows, padded to 128) formed
    in registers and fed to ``tl.dot`` against the eq-weight limbs; INT32 partials over a ``b1`` split."""
    L: list[str] = []
    L.append("@triton.jit")
    L.append("def ext_ip(g_ptr, h_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    offs_n = tl.arange(0, 32)")
    L.append("    acc = tl.zeros((128, 32), dtype=tl.int32)")
    L.append("    zero8 = tl.zeros((BLOCK_K,), dtype=tl.int8)")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK_K):")
    L.append("        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)")
    L.append("        row = (b2.to(tl.int64) * K1 + b1) * 12")
    L.append("        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])")
    for s in range(6):
        L.append(f"        ge{s} = tl.load(g_ptr + row + {s}).to(tl.int64)")
        L.append(f"        go{s} = tl.load(g_ptr + row + {6 + s}).to(tl.int64)")
        L.append(f"        he{s} = tl.load(h_ptr + row + {s}).to(tl.int64)")
        L.append(f"        ho{s} = tl.load(h_ptr + row + {6 + s}).to(tl.int64)")
        L.append(f"        g2{s} = (go{s} * 2 - ge{s}) % P")
        L.append(f"        g2{s} = tl.where(g2{s} < 0, g2{s} + P, g2{s})")
        L.append(f"        h2{s} = (ho{s} * 2 - he{s}) % P")
        L.append(f"        h2{s} = tl.where(h2{s} < 0, h2{s} + P, h2{s})")
    names = []
    for x, (gp, hp) in enumerate((("ge", "he"), ("go", "ho"), ("g2", "h2"))):
        L.append("        " + ", ".join(f"c{x}_{d}" for d in range(6)) + " = _ext_mul6(" + ", ".join(f"{gp}{s}" for s in range(6)) + ", " + ", ".join(f"{hp}{s}" for s in range(6)) + ")")
        for d in range(6):
            L.append(f"        l{x}_{d}_0, l{x}_{d}_1, l{x}_{d}_2, l{x}_{d}_3 = _limbs(c{x}_{d}.to(tl.int32))")
            names += [f"l{x}_{d}_{j}" for j in range(4)]
    names += ["zero8"] * (128 - 72)
    top = _join_tree(L, names, "        ")
    L.append("        A = tl.reshape(" + top + ", (BLOCK_K, 128), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append("    offs_m = tl.arange(0, 128)")
    L.append("    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * 72) * 24")
    L.append("    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=(offs_m[:, None] < 72) & (offs_n[None, :] < 24))")
    return "\n".join(L) + "\n"


def ext_inner_product(gE: torch.Tensor, hE: torch.Tensor, e_lo32: torch.Tensor, K1: int, K2: int, split: int,
                      BLOCK_K: int = 128, num_warps: int = 4) -> torch.Tensor:
    """Extension round fused: ``(split, K2 . 72, 24) int32`` partials of what ``mm_int8(ext_hadamard3_limbs(...), e_lo)`` returns."""
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    Z = torch.empty((split, K2 * 72, 24), dtype=torch.int32, device=gE.device)
    _generated().ext_ip[(K2, split)](gE, hE, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z


def _reduce_contract_source(in_s: int) -> str:
    """``T[r, d] = sum_{b2} e_hi[b2] . X[b2, r]`` with ``X[b2, r]`` recomposed from the INT32 limb block
    ``Z[(b2, r, s, j), (s', j')]`` (``in_s = 1``: base x ext block ``4 x 24``; ``in_s = 6``: ext x ext ``24 x 24`` with
    ``X^6 = W``).  One program per (row, b2-split); partial sums (< 2^17 . p) out, summed and reduced by torch."""
    L: list[str] = []
    L.append("@triton.jit")
    L.append(f"def reduce_contract_{in_s}(z_ptr, e_ptr, pw_ptr, out_ptr, rows, K2, SPLIT_LEN, BLOCK: tl.constexpr):")
    L.append("    r = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    o16 = tl.arange(0, 16)")
    L.append("    joff = (o16 // 4) * 24 + (o16 % 4)")
    L.append("    pw = tl.load(pw_ptr + o16).to(tl.int64)")
    for d in range(6):
        L.append(f"    acc{d} = tl.zeros([BLOCK], dtype=tl.int64)")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK):")
    L.append("        b2 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK)")
    L.append("        m = b2 < K2")
    L.append(f"        base = ((b2.to(tl.int64) * rows + r) * {in_s * 4}) * 24")
    for s in range(in_s):
        for sp_ in range(6):
            L.append(f"        v = tl.load(z_ptr + base[:, None] + {s * 4 * 24 + sp_ * 4} + joff[None, :], mask=m[:, None], other=0).to(tl.int64)")
            # |v| < 2^27 (K1 <= 2^13 products < 2^14), pw < 2^31: 16 terms < 2^62, one reduction
            L.append(f"        y{s}_{sp_} = tl.sum(v * pw[None, :], axis=1) % P")
            L.append(f"        y{s}_{sp_} = tl.where(y{s}_{sp_} < 0, y{s}_{sp_} + P, y{s}_{sp_})")
    for d in range(6):
        lo = [f"y{s}_{d - s}" for s in range(in_s) if 0 <= d - s < 6]
        hi = [f"y{s}_{d + 6 - s}" for s in range(in_s) if 0 <= d + 6 - s < 6]
        expr = " + ".join(lo) if lo else "0"
        if hi:
            expr = f"({expr} + (({' + '.join(hi)}) % P) * W) % P"
        L.append(f"        x{d} = ({expr}) % P")
    for d in range(6):
        L.append(f"        e{d} = tl.load(e_ptr + b2 * 6 + {d}, mask=m, other=0).to(tl.int64)")
    for d in range(11):
        terms = [f"(x{i} * e{d - i}) % P" for i in range(6) if 0 <= d - i < 6]
        L.append(f"        p{d} = " + " + ".join(terms))
    for d in range(6):
        if d < 5:
            L.append(f"        acc{d} += (p{d} + (p{d + 6} % P) * W) % P")
        else:
            L.append(f"        acc{d} += p{d} % P")
    L.append("    o = out_ptr + (sp.to(tl.int64) * rows + r) * 6")
    for d in range(6):
        L.append(f"    tl.store(o + {d}, tl.sum(acc{d}, axis=0))")
    return "\n".join(L) + "\n"


def _ext_mul_helper_source() -> str:
    """``_ext_mul6(a0..a5, b0..b5) -> (c0..c5)`` in ``F_{p^6}``, inputs in ``[0, p)`` (INT64 tensors or scalars)."""
    L = ["@triton.jit", "def _ext_mul6(" + ", ".join(f"a{i}" for i in range(6)) + ", " + ", ".join(f"b{i}" for i in range(6)) + "):"]
    for d in range(11):
        terms = [f"(a{i} * b{d - i}) % P" for i in range(6) if 0 <= d - i < 6]
        L.append(f"    p{d} = " + " + ".join(terms))
    outs = []
    for d in range(6):
        L.append(f"    c{d} = (p{d} + (p{d + 6} % P) * W) % P" if d < 5 else f"    c{d} = p{d} % P")
        outs.append(f"c{d}")
    L.append("    return " + ", ".join(outs))
    return "\n".join(L) + "\n"


def _finish_round_source() -> str:
    """``S(x) = prefix . eq_x[x] . Q(x)``, ``x = 0..3``, from ``T (3, R, 6)`` and ``coeff (R, 6)`` (``Q(x) = sum_r coeff[r] T[x, r]``;
    ``R = 1`` with ``coeff = 1`` for the extension rounds), ``Q(3) = 3 Q(2) - 3 Q(1) + Q(0)``.  One program."""
    L = ["@triton.jit", "def finish_round(t_ptr, c_ptr, pre_ptr, eqx_ptr, s_ptr, R, BLOCK_R: tl.constexpr):",
         "    r = tl.arange(0, BLOCK_R)", "    m = r < R"]
    for d in range(6):
        L.append(f"    c{d} = tl.load(c_ptr + r * 6 + {d}, mask=m, other=0).to(tl.int64)")
        L.append(f"    pre{d} = tl.load(pre_ptr + {d}).to(tl.int64)")
    for x in range(3):
        for d in range(6):
            L.append(f"    t{d} = tl.load(t_ptr + ({x} * R + r) * 6 + {d}, mask=m, other=0).to(tl.int64)")
        L.append("    " + ", ".join(f"u{d}" for d in range(6)) + " = _ext_mul6(" + ", ".join(f"t{d}" for d in range(6)) + ", " + ", ".join(f"c{d}" for d in range(6)) + ")")
        for d in range(6):
            L.append(f"    q{x}_{d} = tl.sum(u{d}, axis=0) % P")
    for d in range(6):
        L.append(f"    q3_{d} = (3 * q2_{d} - 3 * q1_{d} + q0_{d}) % P")
        L.append(f"    q3_{d} = tl.where(q3_{d} < 0, q3_{d} + P, q3_{d})")
    for x in range(4):
        for d in range(6):
            L.append(f"    e{d} = tl.load(eqx_ptr + {x * 6 + d}).to(tl.int64)")
        L.append("    " + ", ".join(f"v{d}" for d in range(6)) + " = _ext_mul6(" + ", ".join(f"pre{d}" for d in range(6)) + ", " + ", ".join(f"e{d}" for d in range(6)) + ")")
        L.append("    " + ", ".join(f"w{d}" for d in range(6)) + " = _ext_mul6(" + ", ".join(f"v{d}" for d in range(6)) + ", " + ", ".join(f"q{x}_{d}" for d in range(6)) + ")")
        for d in range(6):
            L.append(f"    tl.store(s_ptr + {x * 6 + d}, w{d}.to(tl.int32))")
    return "\n".join(L) + "\n"


def finish_round(T: torch.Tensor, coeff: torch.Tensor, prefix: torch.Tensor, eq_x: torch.Tensor) -> torch.Tensor:
    """``T (3, R, 6) int32``, ``coeff (R, 6) int32``, ``prefix (6,)``, ``eq_x (4, 6)`` -> ``S (4, 6) int32``."""
    R = T.shape[1]
    S = torch.empty((4, 6), dtype=torch.int32, device=T.device)
    _generated().finish_round[(1,)](T.contiguous(), coeff.contiguous(), prefix, eq_x, S, R, BLOCK_R=max(16, triton.next_power_of_2(R)))
    return S


def reduce_contract(Z: torch.Tensor, e_hi: torch.Tensor, pw16: torch.Tensor, K2: int, rows: int, in_s: int, BLOCK: int = 32) -> torch.Tensor:
    """``(K2 . rows . in_s . 4, 24) int32`` limb blocks + ``e_hi (K2, 6) int32`` -> ``T (rows, 6) int32`` mod p."""
    Z = Z.contiguous()          # mm_int8 returns a view into its padded output at small shapes
    nsplit = max(1, min(K2 // BLOCK, 32))
    split_len = -(-K2 // nsplit)
    split_len = -(-split_len // BLOCK) * BLOCK
    nsplit = -(-K2 // split_len)
    out = torch.empty((nsplit, rows, 6), dtype=torch.int64, device=Z.device)
    getattr(_generated(), f"reduce_contract_{in_s}")[(rows, nsplit)](Z, e_hi, pw16, out, rows, K2, split_len, BLOCK=BLOCK)
    return (out.sum(0) % P).to(torch.int32)


def pow2_pair16(device) -> torch.Tensor:
    """``2^(8 (j + j')) mod p`` at index ``4 j + j'``."""
    return torch.tensor([POW2[j + jp] for j in range(4) for jp in range(4)], dtype=torch.int64, device=device)


_MODULE = None


def _generated():
    """The module of generated kernels (written once to a cache file so ``triton.jit`` can read its source)."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    src = "\n".join([_HEADER, _ext_mul_helper_source(), _ext_hadamard_source(), *[_packed_hadamard_source(1 << e) for e in range(4)],
                     *[_packed_ip_source(1 << e) for e in range(FUSED_MAX_LOG2_HALF + 1)], _ext_ip_source(),
                     _reduce_contract_source(1), _reduce_contract_source(6), _finish_round_source()]) + "\n"
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


#: the fused Hadamard + tl.dot inner product exists for half = 2^(i-1) <= 4 (rows 12 / 48 / 192, tile 16 / 64 / 256);
#: half = 8 would need a 1024-row accumulator tile.
FUSED_MAX_LOG2_HALF = 2


def has_fused_packed(i: int) -> bool:
    return 1 <= i <= FUSED_MAX_LOG2_HALF + 1


def packed_inner_product(g: torch.Tensor, h: torch.Tensor, e_lo32: torch.Tensor, i: int, K1: int, K2: int, split: int,
                         BLOCK_K: int = 128, num_warps: int | None = None) -> torch.Tensor:
    """Round ``i <= k`` fused: ``(split, K2 . rows . 4, 24) int32`` partial inner products (sum over ``split`` in INT64
    to get what ``mm_int8(packed_hadamard_limbs(...), e_lo)`` returns).  ``e_lo32`` is ``(K1, 32) int8`` (24 limbs + 0)."""
    half = 1 << (i - 1)
    if not has_fused_packed(i):
        raise ValueError("fused packed inner product covers i <= 3")
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    rows4 = 3 * half * half * 4
    Z = torch.empty((split, K2 * rows4, 24), dtype=torch.int32, device=g.device)
    warps = num_warps or 4          # measured on the H100: BLOCK_K = 128, 4 warps (8 warps or BLOCK_K <= 64 are 3-10x slower)
    getattr(_generated(), f"packed_ip_{half}")[(K2, split)](g, h, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=warps)
    return Z


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
    # |acc| < 2^18 (<= 12 limb products), pw < 2^31: the 8-term sum is < 2^52, ONE reduction; Triton's % truncates
    # toward zero (C semantics) so the signed residue is lifted into [0, p)
    c = tl.sum(z * pw[None, None, :], axis=2) % P
    c = tl.where(c < 0, c + P, c)
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
