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

All arithmetic is exact.  Field products on SIMT are **32-bit subtractive Montgomery** (``R = 2^32``, the algorithm and
constants of lane ``a-fused``'s ``backends/gkr/tensor/fused/field.py``: ``mul.lo`` + ``umulhi`` + one conditional add,
7 INT32 ops; no INT64, no ``%``).  Tables and every value written to HBM stay **plain** (``[0, p)``); a product
``mmul(a, b) = a b R^-1`` carries one ``R^-1``, which is absorbed on the host: the eq-weight limbs ``E`` are the limbs of
``eq(.) . R mod p`` (so ``sum E . mmul(g, h) = sum eq g h``), challenges and fold coefficients are pre-scaled by ``R``
(``mmul(r R, x) = r x``).  ``mont_form`` / ``pre_scale`` in :mod:`sumcheck_packed` do the scaling.  Limb products
``< 2^14`` accumulate over ``<= 64`` terms in INT32; the low-volume limb recomposition and ``b2`` contraction
(``reduce_contract``, ``finish_round``) stay in INT64 with ``%``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import reference as ref

P = ref.P
W = ref.W
POW2 = [pow(2, 8 * m, P) for m in range(8)]

# -- Montgomery constants (R = 2^32), as lane a-fused's ``field.py`` defines them ---------------------------------------
R_MONT = 1 << 32
MU = pow(P, -1, R_MONT)                 # 0x88000001 = p^-1 mod 2^32 (subtractive form: hi(T) - hi(m p))
MU_I32 = MU - R_MONT                    # the same bit pattern as a signed int32 constant
R_MOD_P = R_MONT % P
R2_MOD_P = R_MONT * R_MONT % P
WR = W * R_MOD_P % P                    # W in Montgomery form: mmul(x, WR) = W x


def mont_mul_py(a: int, b: int) -> int:
    """``a b R^-1 mod p`` exactly as the kernel computes it (the Python oracle)."""
    t = a * b
    lo, hi = t & (R_MONT - 1), t >> 32
    m = (lo * MU) & (R_MONT - 1)
    u = (m * P) >> 32
    r = hi - u
    return r + P if r < 0 else r


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


@triton.jit
def mmul(a, b):
    """Montgomery product ``a b R^-1 mod p`` of int32 values in ``[0, p)``: mul.lo, umulhi, mul.lo, umulhi, sub, select."""
    lo = a * b
    hi = tl.umulhi(a, b)
    m = lo * MU_I32_C
    u = tl.umulhi(m, P_C)
    r = hi - u
    return tl.where(r < 0, r + P_C, r)


@triton.jit
def madd(a, b):
    s = (a - P_C) + b
    return tl.where(s < 0, s + P_C, s)


@triton.jit
def msub(a, b):
    s = a - b
    return tl.where(s < 0, s + P_C, s)


P_C = tl.constexpr(P)
MU_I32_C = tl.constexpr(MU_I32)
WR_C = tl.constexpr(WR)
# Triton's on-disk cache key is the kernel source + argument specialisation, NOT the values of the ``tl.constexpr``
# globals above; a cache populated under another field (see ``gpu/kernels.py`` and reports/a_kernel_a100.md: W 31 -> 22)
# would be served silently.  Every module-level kernel here (and in kernels_fused / kernels_graph) takes ``FK=FIELD_KEY``
# as an unused constexpr argument so the field constants are part of the key.  The generated kernels (``_HEADER``)
# inline the constants into their source and need no key.
FIELD_KEY = int(hashlib.sha256(f"babybear6:{P}:{W}:{R_MONT}".encode()).hexdigest()[:7], 16)


# ---------------------------------------------------------------------------------------------------------------------
# generated Hadamard kernels
# ---------------------------------------------------------------------------------------------------------------------

_HEADER = f"""
import triton
import triton.language as tl
P: tl.constexpr = {P}
W: tl.constexpr = {W}
MU_I32: tl.constexpr = {MU_I32}
WR: tl.constexpr = {WR}

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

@triton.jit
def mmul(a, b):
    lo = a * b
    hi = tl.umulhi(a, b)
    m = lo * MU_I32
    u = tl.umulhi(m, P)
    r = hi - u
    return tl.where(r < 0, r + P, r)

@triton.jit
def madd(a, b):
    s = (a - P) + b
    return tl.where(s < 0, s + P, s)

@triton.jit
def msub(a, b):
    s = a - b
    return tl.where(s < 0, s + P, s)
"""


def _ext_mmul_helper_source() -> str:
    """``_ext_mmul6(a0..a5, b0..b5) -> (c0..c5)``: ``a b R^-1`` in ``F_{p^6}`` (int32, schoolbook, ``X^6 = W`` through
    ``W b_j`` in Montgomery form); the same operation order as a-fused's ``ext_mmul`` so the two kernels agree bit for bit."""
    L = ["@triton.jit", "def _ext_mmul6(" + ", ".join(f"a{i}" for i in range(6)) + ", " + ", ".join(f"b{i}" for i in range(6)) + "):"]
    for j in range(1, 6):
        L.append(f"    w{j} = mmul(b{j}, WR)")
    for d in range(6):
        terms = [f"mmul(a{i}, b{d - i})" for i in range(6) if 0 <= d - i < 6]
        terms += [f"mmul(a{i}, w{d + 6 - i})" for i in range(6) if 1 <= d + 6 - i < 6]
        expr = terms[0]
        for t in terms[1:]:
            expr = f"madd({expr}, {t})"
        L.append(f"    c{d} = {expr}")
    L.append("    return " + ", ".join(f"c{d}" for d in range(6)))
    return "\n".join(L) + "\n"


def _store4(lines: list[str], base_expr: str, val: str, ind: str) -> None:
    lines.append(f"{ind}l0, l1, l2, l3 = _limbs({val})")
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
        L.append(f"    ge{s} = tl.load(g_ptr + row + {s}, mask=mask, other=0)")
        L.append(f"    go{s} = tl.load(g_ptr + row + {6 + s}, mask=mask, other=0)")
        L.append(f"    he{s} = tl.load(h_ptr + row + {s}, mask=mask, other=0)")
        L.append(f"    ho{s} = tl.load(h_ptr + row + {6 + s}, mask=mask, other=0)")
        L.append(f"    g2{s} = msub(madd(go{s}, go{s}), ge{s})")
        L.append(f"    h2{s} = msub(madd(ho{s}, ho{s}), he{s})")
    for x, (gp, hp) in enumerate((("ge", "he"), ("go", "ho"), ("g2", "h2"))):
        L.append("    " + ", ".join(f"c{d}" for d in range(6)) + " = _ext_mmul6(" + ", ".join(f"{gp}{s}" for s in range(6)) + ", " + ", ".join(f"{hp}{s}" for s in range(6)) + ")")
        for d in range(6):
            _store4(L, f"a_base + {x * 24 + 4 * d} * K1", f"c{d}", "    ")
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
        L.append(f"    g0_{a} = tl.load(g_ptr + col + {a}, mask=mask, other=0)")
        L.append(f"    g1_{a} = tl.load(g_ptr + col + {half + a}, mask=mask, other=0)")
        L.append(f"    g2_{a} = msub(madd(g1_{a}, g1_{a}), g0_{a})")
        L.append(f"    h0_{a} = tl.load(h_ptr + col + {a}, mask=mask, other=0)")
        L.append(f"    h1_{a} = tl.load(h_ptr + col + {half + a}, mask=mask, other=0)")
        L.append(f"    h2_{a} = msub(madd(h1_{a}, h1_{a}), h0_{a})")
    for x in range(3):
        for a in range(half):
            for a2 in range(half):
                r = (x * half + a) * half + a2
                _store4(L, f"a_base + {r * 4} * K1", f"mmul(g{x}_{a}, h{x}_{a2})", "    ")
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


def _packed_ip_source(half: int, linear: bool = False) -> str:
    """Packed round ``i`` (``half = 2^(i-1)``) fused: Hadamard limbs formed in registers, ``tl.dot`` against the eq-weight
    limbs, INT32 accumulation over a ``b1`` split; nothing is written to HBM but the ``(ROWS 4) x 32`` partial sums.

    ``linear`` (the logUp leaf level, kernel ``logup_ip_{half}``): per ``x`` the ``half^2`` product rows are followed by
    ``half`` *linear* rows ``limbs(mmul(g(a, x) + h(a, x), 1))`` = ``(w0 + w1) R^-1``, so that ``sum E . row`` gives
    ``sum_b eq_b (w0 + w1)(a, x, b)`` with the same ``R``-scaled weights."""
    rows = 3 * (half * half + (half if linear else 0))
    rows4 = rows * 4
    rows_pad = max(16, triton.next_power_of_2(rows4))
    L: list[str] = []
    L.append("@triton.jit")
    L.append(f"def {'logup' if linear else 'packed'}_ip_{half}(g_ptr, h_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
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
        L.append(f"        g0_{a} = tl.load(g_ptr + col + {a})")
        L.append(f"        g1_{a} = tl.load(g_ptr + col + {half + a})")
        L.append(f"        g2_{a} = msub(madd(g1_{a}, g1_{a}), g0_{a})")
        L.append(f"        h0_{a} = tl.load(h_ptr + col + {a})")
        L.append(f"        h1_{a} = tl.load(h_ptr + col + {half + a})")
        L.append(f"        h2_{a} = msub(madd(h1_{a}, h1_{a}), h0_{a})")
    names = []
    for x in range(3):
        for a in range(half):
            for a2 in range(half):
                r = (x * half + a) * half + a2
                L.append(f"        l{r}_0, l{r}_1, l{r}_2, l{r}_3 = _limbs(mmul(g{x}_{a}, h{x}_{a2}))")
                names += [f"l{r}_{j}" for j in range(4)]
        if linear:
            for a in range(half):
                L.append(f"        s{x}_{a}_0, s{x}_{a}_1, s{x}_{a}_2, s{x}_{a}_3 = _limbs(mmul(madd(g{x}_{a}, h{x}_{a}), 1))")
                names += [f"s{x}_{a}_{j}" for j in range(4)]
    names += ["zero8"] * (rows_pad - rows4)
    top = _join_tree(L, names, "        ")
    L.append(f"        A = tl.reshape({top}, (BLOCK_K, {rows_pad}), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append(f"    offs_m = tl.arange(0, {rows_pad})")
    L.append(f"    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * {rows4}) * 24")
    L.append(f"    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=(offs_m[:, None] < {rows4}) & (offs_n[None, :] < 24))")
    return "\n".join(L) + "\n"


def _packed_ip_split_source(half: int) -> str:
    """Packed round with ``half^2 . 4 > 256`` limb rows (``half = 8``, round 4): the ``x`` block is a third grid axis, each
    program forms the ``half^2`` Hadamards of ONE ``x`` (``256`` limb rows for ``half = 8``) so the register tile stays at
    256 rows; ``g, h`` are read three times (once per ``x`` program).  Output layout = :func:`_packed_ip_source`'s."""
    rows_x = half * half            # Hadamards per x
    rows4 = rows_x * 4
    rows_pad = max(16, triton.next_power_of_2(rows4))
    L: list[str] = []
    L.append("@triton.jit")
    L.append(f"def packed_ip_{half}(g_ptr, h_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    xx = tl.program_id(2)")
    L.append("    offs_n = tl.arange(0, 32)")
    L.append(f"    acc = tl.zeros(({rows_pad}, 32), dtype=tl.int32)")
    L.append("    zero8 = tl.zeros((BLOCK_K,), dtype=tl.int8)")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK_K):")
    L.append("        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)")
    L.append(f"        col = (b2.to(tl.int64) * K1 + b1) * {2 * half}")
    L.append("        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])")
    for a in range(half):
        # the x-point of table g at a: x = 0 -> g0, 1 -> g1, 2 -> 2 g1 - g0 (runtime select on the program's x)
        L.append(f"        g0_{a} = tl.load(g_ptr + col + {a})")
        L.append(f"        g1_{a} = tl.load(g_ptr + col + {half + a})")
        L.append(f"        gx_{a} = tl.where(xx == 0, g0_{a}, tl.where(xx == 1, g1_{a}, msub(madd(g1_{a}, g1_{a}), g0_{a})))")
        L.append(f"        h0_{a} = tl.load(h_ptr + col + {a})")
        L.append(f"        h1_{a} = tl.load(h_ptr + col + {half + a})")
        L.append(f"        hx_{a} = tl.where(xx == 0, h0_{a}, tl.where(xx == 1, h1_{a}, msub(madd(h1_{a}, h1_{a}), h0_{a})))")
    names = []
    for a in range(half):
        for a2 in range(half):
            r = a * half + a2
            L.append(f"        l{r}_0, l{r}_1, l{r}_2, l{r}_3 = _limbs(mmul(gx_{a}, hx_{a2}))")
            names += [f"l{r}_{j}" for j in range(4)]
    names += ["zero8"] * (rows_pad - rows4)
    top = _join_tree(L, names, "        ")
    L.append(f"        A = tl.reshape({top}, (BLOCK_K, {rows_pad}), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append(f"    offs_m = tl.arange(0, {rows_pad})")
    L.append(f"    z = z_ptr + (((sp * tl.num_programs(0) + b2) * 3 + xx).to(tl.int64) * {rows4}) * 24")
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
        L.append(f"        ge{s} = tl.load(g_ptr + row + {s})")
        L.append(f"        go{s} = tl.load(g_ptr + row + {6 + s})")
        L.append(f"        he{s} = tl.load(h_ptr + row + {s})")
        L.append(f"        ho{s} = tl.load(h_ptr + row + {6 + s})")
        L.append(f"        g2{s} = msub(madd(go{s}, go{s}), ge{s})")
        L.append(f"        h2{s} = msub(madd(ho{s}, ho{s}), he{s})")
    names = []
    for x, (gp, hp) in enumerate((("ge", "he"), ("go", "ho"), ("g2", "h2"))):
        L.append("        " + ", ".join(f"c{x}_{d}" for d in range(6)) + " = _ext_mmul6(" + ", ".join(f"{gp}{s}" for s in range(6)) + ", " + ", ".join(f"{hp}{s}" for s in range(6)) + ")")
        for d in range(6):
            L.append(f"        l{x}_{d}_0, l{x}_{d}_1, l{x}_{d}_2, l{x}_{d}_3 = _limbs(c{x}_{d})")
            names += [f"l{x}_{d}_{j}" for j in range(4)]
    names += ["zero8"] * (128 - 72)
    top = _join_tree(L, names, "        ")
    L.append("        A = tl.reshape(" + top + ", (BLOCK_K, 128), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append("    offs_m = tl.arange(0, 128)")
    L.append("    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * 72) * 24")
    L.append("    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=(offs_m[:, None] < 72) & (offs_n[None, :] < 24))")
    return "\n".join(L) + "\n"


def _packed_u_source(k: int) -> str:
    """a-fusion's U pass in the generated (explicit-name) style of :func:`_packed_ip_source`: per ``b'`` all ``4^k``
    products ``g(c, b') h(c', b')``, 4 limbs each, one ``tl.dot`` against the eq limbs of ``b'``; rows ``r = c 2^k + c'``."""
    twok = 1 << k
    rows4 = twok * twok * 4
    rows_pad = max(16, triton.next_power_of_2(rows4))
    L: list[str] = []
    L.append("@triton.jit")
    L.append(f"def packed_u_{k}(g_ptr, h_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    offs_n = tl.arange(0, 32)")
    L.append(f"    acc = tl.zeros(({rows_pad}, 32), dtype=tl.int32)")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK_K):")
    L.append("        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)")
    L.append(f"        col = (b2.to(tl.int64) * K1 + b1) * {twok}")
    L.append("        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])")
    for c in range(twok):
        L.append(f"        g_{c} = tl.load(g_ptr + col + {c})")
        L.append(f"        h_{c} = tl.load(h_ptr + col + {c})")
    names = []
    for c in range(twok):
        for cp in range(twok):
            r = c * twok + cp
            L.append(f"        l{r}_0, l{r}_1, l{r}_2, l{r}_3 = _limbs(mmul(g_{c}, h_{cp}))")
            names += [f"l{r}_{j}" for j in range(4)]
    top = _join_tree(L, names, "        ")
    L.append(f"        A = tl.reshape({top}, (BLOCK_K, {rows_pad}), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append(f"    offs_m = tl.arange(0, {rows_pad})")
    L.append(f"    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * {rows4}) * 24")
    L.append("    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=offs_n[None, :] < 24)")
    return "\n".join(L) + "\n"


def packed_u_inner_product_gen(g: torch.Tensor, h: torch.Tensor, e_lo32: torch.Tensor, k: int, K1: int, K2: int, split: int,
                               BLOCK_K: int = 128, num_warps: int = 4) -> torch.Tensor:
    """Generated-style twin of ``kernels_fused.packed_u_inner_product`` (same output layout)."""
    if not 1 <= k <= 3:
        raise ValueError("packed U pass covers k = 1..3")
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    rows4 = 4 << (2 * k)
    Z = torch.empty((split, K2 * rows4, 24), dtype=torch.int32, device=g.device)
    getattr(_generated(), f"packed_u_{k}")[(K2, split)](g, h, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z


def _ext_fold_ip_source() -> str:
    """Extension round ``i`` with round ``i - 1``'s fold fused in (lane a-fusion): the tables are read *pre-fold* (4 entries
    per pair ``b``: rows ``4 b .. 4 b + 3`` of the ``(t_{i-1}, 6)`` table), folded by ``r_{i-1}`` in registers into the pair
    ``(e, o)`` = ``(fold(4b, 4b+1), fold(4b+2, 4b+3))``, the folded pair written to ``gf, hf`` (round ``i + 1``'s input) and
    round ``i``'s 3 Hadamards + limbs formed from the register copy (:func:`_ext_ip_source` layout).  Per round the folded
    tables are written once and not re-read: element traffic ``4 M`` (fold read 2, write 1; Hadamard read 1) -> ``3 M``."""
    L: list[str] = []
    L.append("@triton.jit")
    L.append("def ext_fold_ip(g_ptr, h_ptr, r_ptr, gf_ptr, hf_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    offs_n = tl.arange(0, 32)")
    L.append("    acc = tl.zeros((128, 32), dtype=tl.int32)")
    L.append("    zero8 = tl.zeros((BLOCK_K,), dtype=tl.int8)")
    for s in range(6):
        L.append(f"    r{s} = tl.load(r_ptr + {s})")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK_K):")
    L.append("        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)")
    L.append("        b = b2.to(tl.int64) * K1 + b1")
    L.append("        row = b * 24")
    L.append("        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])")
    for tbl, out, pfx in (("g", "gf", "g"), ("h", "hf", "h")):
        for half, off in (("e", 0), ("o", 12)):
            for s in range(6):
                L.append(f"        a{s} = tl.load({tbl}_ptr + row + {off + s})")
                L.append(f"        d{s} = msub(tl.load({tbl}_ptr + row + {off + 6 + s}), a{s})")
            L.append("        " + ", ".join(f"t{s}" for s in range(6)) + " = _ext_mmul6(" + ", ".join(f"r{s}" for s in range(6)) + ", " + ", ".join(f"d{s}" for s in range(6)) + ")")
            for s in range(6):
                L.append(f"        {pfx}{half}{s} = madd(a{s}, t{s})")
                L.append(f"        tl.store({out}_ptr + b * 12 + {(0 if half == 'e' else 6) + s}, {pfx}{half}{s})")
    for s in range(6):
        L.append(f"        g2{s} = msub(madd(go{s}, go{s}), ge{s})")
        L.append(f"        h2{s} = msub(madd(ho{s}, ho{s}), he{s})")
    names = []
    for x, (gp, hp) in enumerate((("ge", "he"), ("go", "ho"), ("g2", "h2"))):
        L.append("        " + ", ".join(f"c{x}_{d}" for d in range(6)) + " = _ext_mmul6(" + ", ".join(f"{gp}{s}" for s in range(6)) + ", " + ", ".join(f"{hp}{s}" for s in range(6)) + ")")
        for d in range(6):
            L.append(f"        l{x}_{d}_0, l{x}_{d}_1, l{x}_{d}_2, l{x}_{d}_3 = _limbs(c{x}_{d})")
            names += [f"l{x}_{d}_{j}" for j in range(4)]
    names += ["zero8"] * (128 - 72)
    top = _join_tree(L, names, "        ")
    L.append("        A = tl.reshape(" + top + ", (BLOCK_K, 128), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append("    offs_m = tl.arange(0, 128)")
    L.append("    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * 72) * 24")
    L.append("    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=(offs_m[:, None] < 72) & (offs_n[None, :] < 24))")
    return "\n".join(L) + "\n"


def _logup_ext_ip_source() -> str:
    """logUp level round on four extension tables ``p0, p1, q0, q1`` (``(t_i, 6)`` each, pairs ``(2b, 2b+1)``): per pair
    and ``x in {0, 1, 2}`` the two ``F_{p^6}`` Hadamards ``A_x = p0 q1 + p1 q0`` and ``B_x = q0 q1`` (Montgomery, one
    ``R^-1`` each), 24 limbs each -> rows ``(x, term)`` = ``2 x + term`` (144 limb rows, padded to 256) against the
    ``R``-scaled eq-weight limbs.  ``lambda`` is applied downstream (``finish_round`` with ``coeff = [1, lambda]``)."""
    L: list[str] = []
    L.append("@triton.jit")
    L.append("def logup_ext_ip(p0_ptr, p1_ptr, q0_ptr, q1_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, BLOCK_K: tl.constexpr):")
    L.append("    b2 = tl.program_id(0)")
    L.append("    sp = tl.program_id(1)")
    L.append("    offs_n = tl.arange(0, 32)")
    L.append("    acc = tl.zeros((256, 32), dtype=tl.int32)")
    L.append("    zero8 = tl.zeros((BLOCK_K,), dtype=tl.int8)")
    L.append("    for c in range(0, SPLIT_LEN, BLOCK_K):")
    L.append("        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)")
    L.append("        row = (b2.to(tl.int64) * K1 + b1) * 12")
    L.append("        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])")
    for nm, ptr in (("a", "p0_ptr"), ("b", "p1_ptr"), ("c", "q0_ptr"), ("d", "q1_ptr")):
        for s in range(6):
            L.append(f"        {nm}e{s} = tl.load({ptr} + row + {s})")
            L.append(f"        {nm}o{s} = tl.load({ptr} + row + {6 + s})")
            L.append(f"        {nm}2{s} = msub(madd({nm}o{s}, {nm}o{s}), {nm}e{s})")
    names = []
    for x, sfx in enumerate(("e", "o", "2")):
        a = ", ".join(f"a{sfx}{s}" for s in range(6))
        b = ", ".join(f"b{sfx}{s}" for s in range(6))
        c = ", ".join(f"c{sfx}{s}" for s in range(6))
        d = ", ".join(f"d{sfx}{s}" for s in range(6))
        L.append("        " + ", ".join(f"u{d_}" for d_ in range(6)) + f" = _ext_mmul6({a}, {d})")
        L.append("        " + ", ".join(f"v{d_}" for d_ in range(6)) + f" = _ext_mmul6({b}, {c})")
        L.append("        " + ", ".join(f"B{x}_{d_}" for d_ in range(6)) + f" = _ext_mmul6({c}, {d})")
        for d_ in range(6):
            L.append(f"        A{x}_{d_} = madd(u{d_}, v{d_})")
        for term in ("A", "B"):
            for d_ in range(6):
                L.append(f"        l{term}{x}_{d_}_0, l{term}{x}_{d_}_1, l{term}{x}_{d_}_2, l{term}{x}_{d_}_3 = _limbs({term}{x}_{d_})")
                names += [f"l{term}{x}_{d_}_{j}" for j in range(4)]
    names += ["zero8"] * (256 - 144)
    top = _join_tree(L, names, "        ")
    L.append("        A = tl.reshape(" + top + ", (BLOCK_K, 256), can_reorder=False)")
    L.append("        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)")
    L.append("    offs_m = tl.arange(0, 256)")
    L.append("    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * 144) * 24")
    L.append("    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=(offs_m[:, None] < 144) & (offs_n[None, :] < 24))")
    return "\n".join(L) + "\n"


def logup_ext_inner_product(T4: torch.Tensor, e_lo32: torch.Tensor, K1: int, K2: int, split: int, BLOCK_K: int = 128,
                            num_warps: int = 4) -> torch.Tensor:
    """``T4 (4, t_i, 6) int32`` = ``p0, p1, q0, q1`` -> ``(split, K2 . 144, 24) int32`` partials, rows ``(b2, x, term, s, j)``."""
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    Z = torch.empty((split, K2 * 144, 24), dtype=torch.int32, device=T4.device)
    _generated().logup_ext_ip[(K2, split)](T4[0], T4[1], T4[2], T4[3], e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z


def logup_packed_inner_product(g: torch.Tensor, h: torch.Tensor, e_lo32: torch.Tensor, i: int, K1: int, K2: int, split: int,
                               BLOCK_K: int = 128, num_warps: int = 4) -> torch.Tensor:
    """logUp leaf level, packed round ``i``: ``(split, K2 . rows . 4, 24)`` with ``rows = 3 (half^2 + half)`` (products then
    linear rows per ``x``)."""
    half = 1 << (i - 1)
    if half > (1 << FUSED_MAX_LOG2_HALF) or K1 % (split * BLOCK_K):
        raise ValueError("fused logUp packed round covers i <= 3 with K1 a multiple of split * BLOCK_K")
    rows4 = 3 * (half * half + half) * 4
    Z = torch.empty((split, K2 * rows4, 24), dtype=torch.int32, device=g.device)
    getattr(_generated(), f"logup_ip_{half}")[(K2, split)](g, h, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z


def ext_inner_product(gE: torch.Tensor, hE: torch.Tensor, e_lo32: torch.Tensor, K1: int, K2: int, split: int,
                      BLOCK_K: int = 128, num_warps: int = 4) -> torch.Tensor:
    """Extension round fused: ``(split, K2 . 72, 24) int32`` partials of what ``mm_int8(ext_hadamard3_limbs(...), e_lo)`` returns."""
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    Z = torch.empty((split, K2 * 72, 24), dtype=torch.int32, device=gE.device)
    _generated().ext_ip[(K2, split)](gE, hE, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z


def ext_fold_inner_product(gE: torch.Tensor, hE: torch.Tensor, r_R: torch.Tensor, e_lo32: torch.Tensor, K1: int, K2: int, split: int,
                           BLOCK_K: int = 128, num_warps: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold + next Hadamard fused (lane a-fusion): ``gE, hE (4 K1 K2, 6)`` plain *pre-fold* tables, ``r_R (6,)`` = ``r . R``;
    returns ``(Z, gF, hF)`` with ``gF, hF (2 K1 K2, 6)`` = ``fold_ext(gE, r_R)`` and ``Z`` = ``ext_inner_product(gF, hF, ...)``."""
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    gF = torch.empty((gE.shape[0] // 2, 6), dtype=torch.int32, device=gE.device)
    hF = torch.empty_like(gF)
    Z = torch.empty((split, K2 * 72, 24), dtype=torch.int32, device=gE.device)
    _generated().ext_fold_ip[(K2, split)](gE, hE, r_R, gF, hF, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z, gF, hF


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


def finish_round(T: torch.Tensor, coeff: torch.Tensor, prefix: torch.Tensor, eq_x: torch.Tensor,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """``T (3, R, 6) int32``, ``coeff (R, 6) int32``, ``prefix (6,)``, ``eq_x (4, 6)`` -> ``S (4, 6) int32`` (into ``out`` if given)."""
    R = T.shape[1]
    S = out if out is not None else torch.empty((4, 6), dtype=torch.int32, device=T.device)
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
    src = "\n".join([_HEADER, _ext_mul_helper_source(), _ext_mmul_helper_source(), _ext_hadamard_source(), *[_packed_hadamard_source(1 << e) for e in range(4)],
                     *[_packed_ip_source(1 << e) for e in range(FUSED_MAX_LOG2_HALF + 1)], _packed_ip_split_source(8), _ext_ip_source(), _ext_fold_ip_source(), *[_packed_u_source(kk) for kk in (1, 2, 3)],
                     *[_packed_ip_source(1 << e, linear=True) for e in range(FUSED_MAX_LOG2_HALF + 1)], _logup_ext_ip_source(),
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


#: the fused Hadamard + tl.dot inner product exists for half = 2^(i-1) <= 4 (rows 12 / 48 / 192, tile 16 / 64 / 256) in one
#: program per (b2, split); half = 8 (round 4: 768 limb rows) is split over a third grid axis (one x per program, 256 rows).
FUSED_MAX_LOG2_HALF = 2
FUSED_SPLIT_HALVES = (8,)


def has_fused_packed(i: int) -> bool:
    return 1 <= i <= FUSED_MAX_LOG2_HALF + 1 or (1 << (i - 1)) in FUSED_SPLIT_HALVES


def packed_inner_product(g: torch.Tensor, h: torch.Tensor, e_lo32: torch.Tensor, i: int, K1: int, K2: int, split: int,
                         BLOCK_K: int = 128, num_warps: int | None = None) -> torch.Tensor:
    """Round ``i <= k`` fused: ``(split, K2 . rows . 4, 24) int32`` partial inner products (sum over ``split`` in INT64
    to get what ``mm_int8(packed_hadamard_limbs(...), e_lo)`` returns).  ``e_lo32`` is ``(K1, 32) int8`` (24 limbs + 0)."""
    half = 1 << (i - 1)
    if not has_fused_packed(i):
        raise ValueError("fused packed inner product covers i <= 4")
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    rows4 = 3 * half * half * 4
    Z = torch.empty((split, K2 * rows4, 24), dtype=torch.int32, device=g.device)
    warps = num_warps or 4          # measured on the H100: BLOCK_K = 128, 4 warps (8 warps or BLOCK_K <= 64 are 3-10x slower)
    grid = (K2, split, 3) if half in FUSED_SPLIT_HALVES else (K2, split)
    getattr(_generated(), f"packed_ip_{half}")[grid](g, h, e_lo32, Z, K1, K1 // split, BLOCK_K=BLOCK_K, num_warps=warps)
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
def _fold_k_kernel(g_ptr, c_ptr, out_ptr, m, TWOK: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    """out[b, s] = sum_a c[a, s] g[b TWOK + a] (plain); c (TWOK, 6) int32 = eq(r_1..r_k, a) . R (Montgomery pre-scaled)."""
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < m
    acc0 = tl.zeros([BLOCK], dtype=tl.int32)
    acc1 = tl.zeros([BLOCK], dtype=tl.int32)
    acc2 = tl.zeros([BLOCK], dtype=tl.int32)
    acc3 = tl.zeros([BLOCK], dtype=tl.int32)
    acc4 = tl.zeros([BLOCK], dtype=tl.int32)
    acc5 = tl.zeros([BLOCK], dtype=tl.int32)
    for a in tl.static_range(TWOK):
        gv = tl.load(g_ptr + b.to(tl.int64) * TWOK + a, mask=mask, other=0)
        acc0 = madd(acc0, mmul(gv, tl.load(c_ptr + a * 6 + 0)))
        acc1 = madd(acc1, mmul(gv, tl.load(c_ptr + a * 6 + 1)))
        acc2 = madd(acc2, mmul(gv, tl.load(c_ptr + a * 6 + 2)))
        acc3 = madd(acc3, mmul(gv, tl.load(c_ptr + a * 6 + 3)))
        acc4 = madd(acc4, mmul(gv, tl.load(c_ptr + a * 6 + 4)))
        acc5 = madd(acc5, mmul(gv, tl.load(c_ptr + a * 6 + 5)))
    o = out_ptr + b.to(tl.int64) * 6
    tl.store(o + 0, acc0, mask=mask)
    tl.store(o + 1, acc1, mask=mask)
    tl.store(o + 2, acc2, mask=mask)
    tl.store(o + 3, acc3, mask=mask)
    tl.store(o + 4, acc4, mask=mask)
    tl.store(o + 5, acc5, mask=mask)


def fold_k(g: torch.Tensor, k: int, coeffs_R: torch.Tensor, BLOCK: int = 256) -> torch.Tensor:
    """``(t,) base -> (t / 2^k, 6)`` plain; ``coeffs_R`` = ``eq(r_1..r_k, a) . R mod p`` as ``(2^k, 6) int32``."""
    m = g.numel() >> k
    out = torch.empty((m, 6), dtype=torch.int32, device=g.device)
    _fold_k_kernel[(triton.cdiv(m, BLOCK),)](g, coeffs_R, out, m, TWOK=1 << k, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _ext_mmul6_j(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
    """Module-level twin of the generated ``_ext_mmul6`` (same operation order)."""
    w1 = mmul(b1, WR_C)
    w2 = mmul(b2, WR_C)
    w3 = mmul(b3, WR_C)
    w4 = mmul(b4, WR_C)
    w5 = mmul(b5, WR_C)
    c0 = madd(madd(madd(madd(madd(mmul(a0, b0), mmul(a1, w5)), mmul(a2, w4)), mmul(a3, w3)), mmul(a4, w2)), mmul(a5, w1))
    c1 = madd(madd(madd(madd(madd(mmul(a0, b1), mmul(a1, b0)), mmul(a2, w5)), mmul(a3, w4)), mmul(a4, w3)), mmul(a5, w2))
    c2 = madd(madd(madd(madd(madd(mmul(a0, b2), mmul(a1, b1)), mmul(a2, b0)), mmul(a3, w5)), mmul(a4, w4)), mmul(a5, w3))
    c3 = madd(madd(madd(madd(madd(mmul(a0, b3), mmul(a1, b2)), mmul(a2, b1)), mmul(a3, b0)), mmul(a4, w5)), mmul(a5, w4))
    c4 = madd(madd(madd(madd(madd(mmul(a0, b4), mmul(a1, b3)), mmul(a2, b2)), mmul(a3, b1)), mmul(a4, b0)), mmul(a5, w5))
    c5 = madd(madd(madd(madd(madd(mmul(a0, b5), mmul(a1, b4)), mmul(a2, b3)), mmul(a3, b2)), mmul(a4, b1)), mmul(a5, b0))
    return c0, c1, c2, c3, c4, c5


@triton.jit
def _fold_ext_kernel(gh_ptr, r_ptr, out_ptr, rows, BLOCK_M: tl.constexpr, FK: tl.constexpr):
    """Row of gh (rows, 12) int32 = [x_e (6) | x_o (6)] plain; r_ptr = r . R (6 int32); out (rows, 6) = x_e + r (x_o - x_e),
    plain.  SIMT Montgomery: one F_{p^6} product per pair (36 mmul), HBM-bound."""
    offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < rows
    base = gh_ptr + offs.to(tl.int64) * 12
    e0 = tl.load(base + 0, mask=mask, other=0)
    e1 = tl.load(base + 1, mask=mask, other=0)
    e2 = tl.load(base + 2, mask=mask, other=0)
    e3 = tl.load(base + 3, mask=mask, other=0)
    e4 = tl.load(base + 4, mask=mask, other=0)
    e5 = tl.load(base + 5, mask=mask, other=0)
    d0 = msub(tl.load(base + 6, mask=mask, other=0), e0)
    d1 = msub(tl.load(base + 7, mask=mask, other=0), e1)
    d2 = msub(tl.load(base + 8, mask=mask, other=0), e2)
    d3 = msub(tl.load(base + 9, mask=mask, other=0), e3)
    d4 = msub(tl.load(base + 10, mask=mask, other=0), e4)
    d5 = msub(tl.load(base + 11, mask=mask, other=0), e5)
    r0 = tl.load(r_ptr + 0)
    r1 = tl.load(r_ptr + 1)
    r2 = tl.load(r_ptr + 2)
    r3 = tl.load(r_ptr + 3)
    r4 = tl.load(r_ptr + 4)
    r5 = tl.load(r_ptr + 5)
    t0, t1, t2, t3, t4, t5 = _ext_mmul6_j(r0, r1, r2, r3, r4, r5, d0, d1, d2, d3, d4, d5)
    o = out_ptr + offs.to(tl.int64) * 6
    tl.store(o + 0, madd(e0, t0), mask=mask)
    tl.store(o + 1, madd(e1, t1), mask=mask)
    tl.store(o + 2, madd(e2, t2), mask=mask)
    tl.store(o + 3, madd(e3, t3), mask=mask)
    tl.store(o + 4, madd(e4, t4), mask=mask)
    tl.store(o + 5, madd(e5, t5), mask=mask)


@triton.jit
def _fold_ext4_kernel(t_ptr, out_ptr, r_ptr, m, stride_in, stride_out, BLOCK_M: tl.constexpr, FK: tl.constexpr):
    """Four ``(., 6)`` tables at ``t_ptr + j stride_in`` (pairs ``(2y, 2y+1)``, ``y < m``) folded by ``r`` (pre-scaled) into
    ``out_ptr + j stride_out`` rows ``y < m``; rows beyond ``m`` of the output are left as they are (zero-padded buffers)."""
    j = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < m
    base = t_ptr + j.to(tl.int64) * stride_in + offs.to(tl.int64) * 12
    e0 = tl.load(base + 0, mask=mask, other=0)
    e1 = tl.load(base + 1, mask=mask, other=0)
    e2 = tl.load(base + 2, mask=mask, other=0)
    e3 = tl.load(base + 3, mask=mask, other=0)
    e4 = tl.load(base + 4, mask=mask, other=0)
    e5 = tl.load(base + 5, mask=mask, other=0)
    d0 = msub(tl.load(base + 6, mask=mask, other=0), e0)
    d1 = msub(tl.load(base + 7, mask=mask, other=0), e1)
    d2 = msub(tl.load(base + 8, mask=mask, other=0), e2)
    d3 = msub(tl.load(base + 9, mask=mask, other=0), e3)
    d4 = msub(tl.load(base + 10, mask=mask, other=0), e4)
    d5 = msub(tl.load(base + 11, mask=mask, other=0), e5)
    r0 = tl.load(r_ptr + 0)
    r1 = tl.load(r_ptr + 1)
    r2 = tl.load(r_ptr + 2)
    r3 = tl.load(r_ptr + 3)
    r4 = tl.load(r_ptr + 4)
    r5 = tl.load(r_ptr + 5)
    t0, t1, t2, t3, t4, t5 = _ext_mmul6_j(r0, r1, r2, r3, r4, r5, d0, d1, d2, d3, d4, d5)
    o = out_ptr + j.to(tl.int64) * stride_out + offs.to(tl.int64) * 6
    tl.store(o + 0, madd(e0, t0), mask=mask)
    tl.store(o + 1, madd(e1, t1), mask=mask)
    tl.store(o + 2, madd(e2, t2), mask=mask)
    tl.store(o + 3, madd(e3, t3), mask=mask)
    tl.store(o + 4, madd(e4, t4), mask=mask)
    tl.store(o + 5, madd(e5, t5), mask=mask)


def fold_ext4(T4: torch.Tensor, m_pairs: int, r_R: torch.Tensor, out: torch.Tensor, BLOCK_M: int = 256) -> torch.Tensor:
    """``T4 (4, Mp, 6)`` (logical ``2 m_pairs`` rows per table) -> ``out (4, Mp', 6)`` rows ``< m_pairs`` written."""
    grid = (triton.cdiv(m_pairs, BLOCK_M), 4)
    _fold_ext4_kernel[grid](T4, out, r_R, m_pairs, T4.stride(0), out.stride(0), BLOCK_M=BLOCK_M, num_warps=4, FK=FIELD_KEY)
    return out


@triton.jit
def _tree_level_kernel(p_ptr, q_ptr, po_ptr, qo_ptr, m, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    """Fraction-tree level: children ``(p, q)`` stored de-interleaved (``[evens | odds]``, ``2 m`` rows each) -> parents
    ``p' = p0 q1 + p1 q0``, ``q' = q0 q1`` for ``x < m``, written de-interleaved for the next level (``x`` even -> row
    ``x / 2``, odd -> ``m / 2 + x / 2``).  Plain in, plain out (``mmul(., R^2)`` undoes the Montgomery factor)."""
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = x < m
    e = x.to(tl.int64) * 6
    o = (m + x).to(tl.int64) * 6
    a0 = tl.load(p_ptr + e + 0, mask=mask, other=0)
    a1 = tl.load(p_ptr + e + 1, mask=mask, other=0)
    a2 = tl.load(p_ptr + e + 2, mask=mask, other=0)
    a3 = tl.load(p_ptr + e + 3, mask=mask, other=0)
    a4 = tl.load(p_ptr + e + 4, mask=mask, other=0)
    a5 = tl.load(p_ptr + e + 5, mask=mask, other=0)
    b0 = tl.load(p_ptr + o + 0, mask=mask, other=0)
    b1 = tl.load(p_ptr + o + 1, mask=mask, other=0)
    b2 = tl.load(p_ptr + o + 2, mask=mask, other=0)
    b3 = tl.load(p_ptr + o + 3, mask=mask, other=0)
    b4 = tl.load(p_ptr + o + 4, mask=mask, other=0)
    b5 = tl.load(p_ptr + o + 5, mask=mask, other=0)
    c0 = tl.load(q_ptr + e + 0, mask=mask, other=0)
    c1 = tl.load(q_ptr + e + 1, mask=mask, other=0)
    c2 = tl.load(q_ptr + e + 2, mask=mask, other=0)
    c3 = tl.load(q_ptr + e + 3, mask=mask, other=0)
    c4 = tl.load(q_ptr + e + 4, mask=mask, other=0)
    c5 = tl.load(q_ptr + e + 5, mask=mask, other=0)
    d0 = tl.load(q_ptr + o + 0, mask=mask, other=0)
    d1 = tl.load(q_ptr + o + 1, mask=mask, other=0)
    d2 = tl.load(q_ptr + o + 2, mask=mask, other=0)
    d3 = tl.load(q_ptr + o + 3, mask=mask, other=0)
    d4 = tl.load(q_ptr + o + 4, mask=mask, other=0)
    d5 = tl.load(q_ptr + o + 5, mask=mask, other=0)
    u0, u1, u2, u3, u4, u5 = _ext_mmul6_j(a0, a1, a2, a3, a4, a5, d0, d1, d2, d3, d4, d5)
    v0, v1, v2, v3, v4, v5 = _ext_mmul6_j(b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5)
    w0, w1, w2, w3, w4, w5 = _ext_mmul6_j(c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5)
    half = m // 2
    dst = tl.where((x & 1) == 0, x // 2, half + x // 2).to(tl.int64) * 6
    tl.store(po_ptr + dst + 0, mmul(madd(u0, v0), R2), mask=mask)
    tl.store(po_ptr + dst + 1, mmul(madd(u1, v1), R2), mask=mask)
    tl.store(po_ptr + dst + 2, mmul(madd(u2, v2), R2), mask=mask)
    tl.store(po_ptr + dst + 3, mmul(madd(u3, v3), R2), mask=mask)
    tl.store(po_ptr + dst + 4, mmul(madd(u4, v4), R2), mask=mask)
    tl.store(po_ptr + dst + 5, mmul(madd(u5, v5), R2), mask=mask)
    tl.store(qo_ptr + dst + 0, mmul(w0, R2), mask=mask)
    tl.store(qo_ptr + dst + 1, mmul(w1, R2), mask=mask)
    tl.store(qo_ptr + dst + 2, mmul(w2, R2), mask=mask)
    tl.store(qo_ptr + dst + 3, mmul(w3, R2), mask=mask)
    tl.store(qo_ptr + dst + 4, mmul(w4, R2), mask=mask)
    tl.store(qo_ptr + dst + 5, mmul(w5, R2), mask=mask)


def tree_level(pq: torch.Tensor, BLOCK: int = 256) -> torch.Tensor:
    """``pq (2, 2m, 6)`` = de-interleaved children ``p, q`` -> ``(2, m, 6)`` de-interleaved parents (plain values)."""
    m = pq.shape[1] // 2
    out = torch.empty((2, m, 6), dtype=torch.int32, device=pq.device)
    _tree_level_kernel[(triton.cdiv(m, BLOCK),)](pq[0], pq[1], out[0], out[1], m, R2=R2_MOD_P, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _leaf_level_kernel(w_ptr, z_ptr, zR_ptr, z2_ptr, po_ptr, qo_ptr, m, half_total, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    """Witness leaves ``(1, z - w_j)``, pairs ``(2x, 2x+1)`` of ``w`` -> level ``n-1`` values ``p = 2 z - (w0 + w1)``,
    ``q = z^2 - z (w0 + w1) + w0 w1`` for ``x < m``, de-interleaved as :func:`tree_level` writes them (odd ``x`` at
    ``half_total + x / 2``, ``half_total`` = half the whole level).  ``z`` plain, ``zR`` = ``z . R``, ``z2`` = ``z^2`` plain."""
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = x < m
    zeros = x * 0
    w0 = tl.load(w_ptr + x.to(tl.int64) * 2, mask=mask, other=0)
    w1 = tl.load(w_ptr + x.to(tl.int64) * 2 + 1, mask=mask, other=0)
    s = madd(w0, w1)
    prod = mmul(mmul(w0, w1), R2)
    dst = tl.where((x & 1) == 0, x // 2, half_total + x // 2).to(tl.int64) * 6
    z0 = tl.load(z_ptr + 0)
    z1 = tl.load(z_ptr + 1)
    z2 = tl.load(z_ptr + 2)
    z3 = tl.load(z_ptr + 3)
    z4 = tl.load(z_ptr + 4)
    z5 = tl.load(z_ptr + 5)
    zr0 = tl.load(zR_ptr + 0)
    zr1 = tl.load(zR_ptr + 1)
    zr2 = tl.load(zR_ptr + 2)
    zr3 = tl.load(zR_ptr + 3)
    zr4 = tl.load(zR_ptr + 4)
    zr5 = tl.load(zR_ptr + 5)
    # p = 2 z - s: coefficient 0 gets -s, the others are the constants 2 z_c
    tl.store(po_ptr + dst + 0, msub(madd(zeros + z0, zeros + z0), s), mask=mask)
    tl.store(po_ptr + dst + 1, madd(zeros + z1, zeros + z1), mask=mask)
    tl.store(po_ptr + dst + 2, madd(zeros + z2, zeros + z2), mask=mask)
    tl.store(po_ptr + dst + 3, madd(zeros + z3, zeros + z3), mask=mask)
    tl.store(po_ptr + dst + 4, madd(zeros + z4, zeros + z4), mask=mask)
    tl.store(po_ptr + dst + 5, madd(zeros + z5, zeros + z5), mask=mask)
    # q = z^2 - z s + w0 w1
    tl.store(qo_ptr + dst + 0, madd(msub(zeros + tl.load(z2_ptr + 0), mmul(zeros + zr0, s)), prod), mask=mask)
    tl.store(qo_ptr + dst + 1, msub(zeros + tl.load(z2_ptr + 1), mmul(zeros + zr1, s)), mask=mask)
    tl.store(qo_ptr + dst + 2, msub(zeros + tl.load(z2_ptr + 2), mmul(zeros + zr2, s)), mask=mask)
    tl.store(qo_ptr + dst + 3, msub(zeros + tl.load(z2_ptr + 3), mmul(zeros + zr3, s)), mask=mask)
    tl.store(qo_ptr + dst + 4, msub(zeros + tl.load(z2_ptr + 4), mmul(zeros + zr4, s)), mask=mask)
    tl.store(qo_ptr + dst + 5, msub(zeros + tl.load(z2_ptr + 5), mmul(zeros + zr5, s)), mask=mask)


def leaf_level(w: torch.Tensor, z: torch.Tensor, zR: torch.Tensor, z2: torch.Tensor, po: torch.Tensor, qo: torch.Tensor,
               half_total: int, BLOCK: int = 256) -> None:
    """``w (2m,) int32`` witness queries -> level rows ``x < m`` of the de-interleaved arrays ``po, qo`` (``2 half_total`` rows)."""
    m = w.numel() // 2
    _leaf_level_kernel[(triton.cdiv(m, BLOCK),)](w, z, zR, z2, po, qo, m, half_total, R2=R2_MOD_P, BLOCK=BLOCK, FK=FIELD_KEY)


@triton.jit
def _ext_mul_rows_kernel(a_ptr, b_ptr, out_ptr, n, stride_b, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    """``out[i] = a[i] b[i]`` (``stride_b = 6``) or ``a[i] b[0]`` (``stride_b = 0``), plain in / plain out."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    ia = i.to(tl.int64) * 6
    ib = i.to(tl.int64) * stride_b
    a0 = tl.load(a_ptr + ia + 0, mask=mask, other=0)
    a1 = tl.load(a_ptr + ia + 1, mask=mask, other=0)
    a2 = tl.load(a_ptr + ia + 2, mask=mask, other=0)
    a3 = tl.load(a_ptr + ia + 3, mask=mask, other=0)
    a4 = tl.load(a_ptr + ia + 4, mask=mask, other=0)
    a5 = tl.load(a_ptr + ia + 5, mask=mask, other=0)
    b0 = tl.load(b_ptr + ib + 0, mask=mask, other=0)
    b1 = tl.load(b_ptr + ib + 1, mask=mask, other=0)
    b2 = tl.load(b_ptr + ib + 2, mask=mask, other=0)
    b3 = tl.load(b_ptr + ib + 3, mask=mask, other=0)
    b4 = tl.load(b_ptr + ib + 4, mask=mask, other=0)
    b5 = tl.load(b_ptr + ib + 5, mask=mask, other=0)
    c0, c1, c2, c3, c4, c5 = _ext_mmul6_j(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
    tl.store(out_ptr + ia + 0, mmul(c0, R2), mask=mask)
    tl.store(out_ptr + ia + 1, mmul(c1, R2), mask=mask)
    tl.store(out_ptr + ia + 2, mmul(c2, R2), mask=mask)
    tl.store(out_ptr + ia + 3, mmul(c3, R2), mask=mask)
    tl.store(out_ptr + ia + 4, mmul(c4, R2), mask=mask)
    tl.store(out_ptr + ia + 5, mmul(c5, R2), mask=mask)


def ext_mul_rows(a: torch.Tensor, b: torch.Tensor, BLOCK: int = 256) -> torch.Tensor:
    """``(n, 6) x (n, 6) | (1, 6) -> (n, 6)`` plain ``F_{p^6}`` Hadamard, one launch (the torch ``ext_mul`` is ~150)."""
    a = a.contiguous()
    b = b.contiguous()
    n = a.shape[0]
    out = torch.empty_like(a)
    _ext_mul_rows_kernel[(triton.cdiv(n, BLOCK),)](a, b, out, n, 0 if b.shape[0] == 1 else 6, R2=R2_MOD_P, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _eq_prepend_kernel(t_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr, FK: tl.constexpr):
    """``out[2 b + bit] = T[b] . (bit ? g : 1 - g)``: one more eq variable on bit 0.  ``g_ptr`` holds ``(1 - g) R`` then
    ``g R`` (12 int32); ``T`` plain in, ``out`` plain out."""
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < n
    ib = b.to(tl.int64) * 6
    t0 = tl.load(t_ptr + ib + 0, mask=mask, other=0)
    t1 = tl.load(t_ptr + ib + 1, mask=mask, other=0)
    t2 = tl.load(t_ptr + ib + 2, mask=mask, other=0)
    t3 = tl.load(t_ptr + ib + 3, mask=mask, other=0)
    t4 = tl.load(t_ptr + ib + 4, mask=mask, other=0)
    t5 = tl.load(t_ptr + ib + 5, mask=mask, other=0)
    zeros = b * 0
    for bit in tl.static_range(2):
        g0 = zeros + tl.load(g_ptr + bit * 6 + 0)
        g1 = zeros + tl.load(g_ptr + bit * 6 + 1)
        g2 = zeros + tl.load(g_ptr + bit * 6 + 2)
        g3 = zeros + tl.load(g_ptr + bit * 6 + 3)
        g4 = zeros + tl.load(g_ptr + bit * 6 + 4)
        g5 = zeros + tl.load(g_ptr + bit * 6 + 5)
        c0, c1, c2, c3, c4, c5 = _ext_mmul6_j(t0, t1, t2, t3, t4, t5, g0, g1, g2, g3, g4, g5)
        o = (2 * b + bit).to(tl.int64) * 6
        tl.store(out_ptr + o + 0, c0, mask=mask)
        tl.store(out_ptr + o + 1, c1, mask=mask)
        tl.store(out_ptr + o + 2, c2, mask=mask)
        tl.store(out_ptr + o + 3, c3, mask=mask)
        tl.store(out_ptr + o + 4, c4, mask=mask)
        tl.store(out_ptr + o + 5, c5, mask=mask)


def eq_prepend(T: torch.Tensor, g_pair_R: torch.Tensor, BLOCK: int = 256) -> torch.Tensor:
    """``T (n, 6)`` = ``eq(gammas, .)`` -> ``(2n, 6)`` = ``eq([g] + gammas, .)`` (``g`` on bit 0).  ``g_pair_R`` = the 12 int32
    ``[(1 - g) R, g R]``."""
    n = T.shape[0]
    out = torch.empty((2 * n, 6), dtype=torch.int32, device=T.device)
    _eq_prepend_kernel[(triton.cdiv(n, BLOCK),)](T, g_pair_R, out, n, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _eq_limbs32_kernel(e_ptr, out_ptr, n, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    """``e (n, 6)`` plain -> ``(n, 32) int8``: the 24 balanced limbs of ``e . R`` (columns ``4 s + j``) and 8 zero columns."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    o = i.to(tl.int64) * 32
    for s in tl.static_range(6):
        v = tl.load(e_ptr + i.to(tl.int64) * 6 + s, mask=mask, other=0)
        vR = mmul(v, R2)                     # v R^2 R^-1 = v R
        l0, l1, l2, l3 = _limbs(vR)
        tl.store(out_ptr + o + 4 * s + 0, l0, mask=mask)
        tl.store(out_ptr + o + 4 * s + 1, l1, mask=mask)
        tl.store(out_ptr + o + 4 * s + 2, l2, mask=mask)
        tl.store(out_ptr + o + 4 * s + 3, l3, mask=mask)
    z = tl.zeros((BLOCK,), dtype=tl.int8)
    for c in tl.static_range(24, 32):
        tl.store(out_ptr + o + c, z, mask=mask)


def eq_limbs32(e: torch.Tensor, BLOCK: int = 256) -> torch.Tensor:
    """``eq`` weights ``(K1, 6)`` plain -> the fused kernels' ``e_lo32`` operand ``(K1, 32) int8`` (``R``-scaled limbs), one launch."""
    n = e.shape[0]
    out = torch.empty((n, 32), dtype=torch.int8, device=e.device)
    _eq_limbs32_kernel[(triton.cdiv(n, BLOCK),)](e.contiguous(), out, n, R2=R2_MOD_P, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _load6(ptr, idx, mask):
    return (tl.load(ptr + idx + 0, mask=mask, other=0), tl.load(ptr + idx + 1, mask=mask, other=0),
            tl.load(ptr + idx + 2, mask=mask, other=0), tl.load(ptr + idx + 3, mask=mask, other=0),
            tl.load(ptr + idx + 4, mask=mask, other=0), tl.load(ptr + idx + 5, mask=mask, other=0))


@triton.jit
def _logup_small_round_kernel(t_ptr, e_ptr, lam_ptr, q_ptr, m, stride_t, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    """One program: ``Q(x) = sum_{y<m} e(y) [A_x(y) + lambda B_x(y)]`` for ``x = 0, 1, 2`` from the four tables at
    ``t_ptr + j stride_t`` (pairs ``(2y, 2y+1)``), ``e (m, 6)`` plain, ``lam_ptr`` = ``lambda R``.  Plain ``(3, 6)`` out."""
    y = tl.arange(0, BLOCK)
    mask = y < m
    ie = (2 * y).to(tl.int64) * 6
    io = ie + 6
    zeros = y * 0
    lr0 = zeros + tl.load(lam_ptr + 0)
    lr1 = zeros + tl.load(lam_ptr + 1)
    lr2 = zeros + tl.load(lam_ptr + 2)
    lr3 = zeros + tl.load(lam_ptr + 3)
    lr4 = zeros + tl.load(lam_ptr + 4)
    lr5 = zeros + tl.load(lam_ptr + 5)
    e0, e1, e2, e3, e4, e5 = _load6(e_ptr, y.to(tl.int64) * 6, mask)
    ae0, ae1, ae2, ae3, ae4, ae5 = _load6(t_ptr, ie, mask)
    ao0, ao1, ao2, ao3, ao4, ao5 = _load6(t_ptr, io, mask)
    be0, be1, be2, be3, be4, be5 = _load6(t_ptr + stride_t, ie, mask)
    bo0, bo1, bo2, bo3, bo4, bo5 = _load6(t_ptr + stride_t, io, mask)
    ce0, ce1, ce2, ce3, ce4, ce5 = _load6(t_ptr + 2 * stride_t, ie, mask)
    co0, co1, co2, co3, co4, co5 = _load6(t_ptr + 2 * stride_t, io, mask)
    de0, de1, de2, de3, de4, de5 = _load6(t_ptr + 3 * stride_t, ie, mask)
    do0, do1, do2, do3, do4, do5 = _load6(t_ptr + 3 * stride_t, io, mask)
    for x in tl.static_range(3):
        if x == 0:
            a0, a1, a2, a3, a4, a5 = ae0, ae1, ae2, ae3, ae4, ae5
            b0, b1, b2, b3, b4, b5 = be0, be1, be2, be3, be4, be5
            c0, c1, c2, c3, c4, c5 = ce0, ce1, ce2, ce3, ce4, ce5
            d0, d1, d2, d3, d4, d5 = de0, de1, de2, de3, de4, de5
        elif x == 1:
            a0, a1, a2, a3, a4, a5 = ao0, ao1, ao2, ao3, ao4, ao5
            b0, b1, b2, b3, b4, b5 = bo0, bo1, bo2, bo3, bo4, bo5
            c0, c1, c2, c3, c4, c5 = co0, co1, co2, co3, co4, co5
            d0, d1, d2, d3, d4, d5 = do0, do1, do2, do3, do4, do5
        else:
            a0, a1, a2, a3, a4, a5 = msub(madd(ao0, ao0), ae0), msub(madd(ao1, ao1), ae1), msub(madd(ao2, ao2), ae2), msub(madd(ao3, ao3), ae3), msub(madd(ao4, ao4), ae4), msub(madd(ao5, ao5), ae5)
            b0, b1, b2, b3, b4, b5 = msub(madd(bo0, bo0), be0), msub(madd(bo1, bo1), be1), msub(madd(bo2, bo2), be2), msub(madd(bo3, bo3), be3), msub(madd(bo4, bo4), be4), msub(madd(bo5, bo5), be5)
            c0, c1, c2, c3, c4, c5 = msub(madd(co0, co0), ce0), msub(madd(co1, co1), ce1), msub(madd(co2, co2), ce2), msub(madd(co3, co3), ce3), msub(madd(co4, co4), ce4), msub(madd(co5, co5), ce5)
            d0, d1, d2, d3, d4, d5 = msub(madd(do0, do0), de0), msub(madd(do1, do1), de1), msub(madd(do2, do2), de2), msub(madd(do3, do3), de3), msub(madd(do4, do4), de4), msub(madd(do5, do5), de5)
        u0, u1, u2, u3, u4, u5 = _ext_mmul6_j(a0, a1, a2, a3, a4, a5, d0, d1, d2, d3, d4, d5)
        v0, v1, v2, v3, v4, v5 = _ext_mmul6_j(b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5)
        w0, w1, w2, w3, w4, w5 = _ext_mmul6_j(c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5)   # B R^-1
        l0, l1, l2, l3, l4, l5 = _ext_mmul6_j(w0, w1, w2, w3, w4, w5, lr0, lr1, lr2, lr3, lr4, lr5)  # lambda B R^-1
        s0 = madd(madd(u0, v0), l0)
        s1 = madd(madd(u1, v1), l1)
        s2 = madd(madd(u2, v2), l2)
        s3 = madd(madd(u3, v3), l3)
        s4 = madd(madd(u4, v4), l4)
        s5 = madd(madd(u5, v5), l5)          # U R^-1
        g0, g1, g2, g3, g4, g5 = _ext_mmul6_j(s0, s1, s2, s3, s4, s5, e0, e1, e2, e3, e4, e5)   # U e R^-2
        for d_ in tl.static_range(6):
            if d_ == 0:
                gg = g0
            elif d_ == 1:
                gg = g1
            elif d_ == 2:
                gg = g2
            elif d_ == 3:
                gg = g3
            elif d_ == 4:
                gg = g4
            else:
                gg = g5
            tot = (tl.sum(gg.to(tl.int64), axis=0) % P_C).to(tl.int32)
            tl.store(q_ptr + x * 6 + d_, mmul(mmul(tot, R2), R2))


def logup_small_round(T4: torch.Tensor, e: torch.Tensor, lam_R: torch.Tensor) -> torch.Tensor:
    """``T4 (4, 2m, 6)``, ``e (m, 6)`` plain, ``lam_R (6,)`` = ``lambda R`` -> ``Q (3, 6)`` plain; ``m <= 64``, one launch."""
    m = T4.shape[1] // 2
    Q = torch.empty((3, 6), dtype=torch.int32, device=T4.device)
    _logup_small_round_kernel[(1,)](T4, e.contiguous(), lam_R, Q, m, T4.stride(0), R2=R2_MOD_P, BLOCK=max(2, triton.next_power_of_2(m)), FK=FIELD_KEY)
    return Q


def fold_ext(gh: torch.Tensor, r_R: torch.Tensor, BLOCK_M: int = 256) -> torch.Tensor:
    """``gh (rows, 12) int32`` (``[x_e | x_o]`` per pair) -> ``(rows, 6)`` = ``(1 - r) x_e + r x_o``; ``r_R`` = ``r . R`` (6 int32)."""
    rows = gh.shape[0]
    out = torch.empty((rows, 6), dtype=torch.int32, device=gh.device)
    _fold_ext_kernel[(triton.cdiv(rows, BLOCK_M),)](gh, r_R, out, rows, BLOCK_M=BLOCK_M, num_warps=4, FK=FIELD_KEY)
    return out
