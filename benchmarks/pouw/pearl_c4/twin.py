"""Pearl-C4's numpy twin (``pearl-c-nvfp4-v0``, ``pearl-c-mxfp4-v0``): one 64 x 64 tile as the sm_120 kernels compute it,
from its 64 A rows and 64 B rows (BF16 words) and the job's seeds.

The per-element and per-word work is batched over `verity.ml.kernels.tc`' bit-exact steps: the noise atoms, the noisy block
scales and the E2M1 cast, the chain, the fused A′·F_B and B̃·F_A, and the peel.  The per-row statistics, the lines and
P_B's FP32 FMA chain come from the scalar reference (`verity.core.protocols.pouw.schemes.pearl_c4`).  The FP4 bench arm's gate compares
the device's sampled tiles with `tile`; benchmarks/pouw/tests/test_pearl_c4_twin.py holds it to the scalar reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from verity.ml.kernels.tc import block_scaled_batch, group_sum_total_batch
from verity.ml.circuits.silicon import fp32
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4, HOPPER_BF16_M16N8K16
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_kw as P

T = F.TILE
R = F.R
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 2, np.float32) * np.repeat(np.float32([1, -1]), 8)


def f32(words: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(words, np.uint32).view(np.float32)


def u32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, np.float32).view(np.uint32)


def e2m1(words: np.ndarray) -> np.ndarray:
    """`pearl_c4.e2m1` on FP32 words: nearest even, ±6 past the top, the sign kept."""
    w = np.asarray(words, np.uint32)
    v = np.abs(f32(w)).astype(np.float64)
    code = np.zeros(w.shape, np.uint8)
    for i, mid in enumerate(F._E2M1_MIDPOINTS):
        up = (v > mid) | ((v == mid) & ((i + 1) % 2 == 0))
        code = np.where(up & (code == i), i + 1, code)
    return (code | ((w >> 31).astype(np.uint8) << 3)).astype(np.uint8)


def ue4m3(words: np.ndarray) -> np.ndarray:
    """`pearl_c4.ue4m3` on non-negative FP32 words: nearest even, saturating to 448."""
    w = np.asarray(words, np.uint32)
    e = ((w >> 23) & 0xFF).astype(np.int64) - 127
    small = np.rint(np.where(e < -6, f32(w).astype(np.float64), 0.0) * 512.0).astype(np.int64)
    fm = (w & 0x7FFFFF).astype(np.int64)
    mant, rem = fm >> 20, fm & 0xFFFFF
    mant = mant + ((rem > 0x80000) | ((rem == 0x80000) & ((mant & 1) == 1)))
    exp = e + 7 + (mant == 8)
    mant = np.where(mant == 8, 0, mant)
    big = np.where((exp > 15) | ((exp == 15) & (mant >= 7)), 0x7E, (exp << 3) | mant)
    return np.where(e < -6, small, big).astype(np.uint8)


def ue4m3_value(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, np.int64)
    e, m = (c >> 3) & 0xF, c & 7
    return np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7)).astype(np.float32)


def ue8m0_up(words: np.ndarray) -> np.ndarray:
    """`pearl_c4.ue8m0_up`: the UE8M0 byte of 2^⌈log2 v⌉."""
    w = np.asarray(words, np.uint32).astype(np.int64)
    field, frac = (w >> 23) & 0xFF, w & 0x7FFFFF
    sub_e = np.frexp(np.maximum(frac, 1).astype(np.float64))[1] - 1 - 149 + ((frac & (frac - 1)) != 0)
    e = np.where(field > 0, field - 127 + (frac != 0), sub_e)
    return np.where(w == 0, 0, np.clip(e + 127, 0, 254)).astype(np.uint8)


def block_scales(fmt: F.Fmt, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(scale bytes, FP32 reciprocal words) per block of `fmt.block` along k of the (rows, k) FP32 words d."""
    rows, k = d.shape
    amax = (d.reshape(rows, k // fmt.block, fmt.block) & np.uint32(0x7FFFFFFF)).max(axis=2)
    v = u32(f32(amax) * f32(np.uint32(F.SIXTH)))
    if fmt is F.NVFP4:
        s = np.maximum(ue4m3(v), F.UE4M3_MIN)
        return s, u32(np.float32(1) / ue4m3_value(s))
    s = ue8m0_up(v)
    return s, u32(np.ldexp(np.float32(1), 127 - s.astype(np.int32)).astype(np.float32))


def quantize(fmt: F.Fmt, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s, r = block_scales(fmt, d)
    return e2m1(u32(f32(d) * np.repeat(f32(r), fmt.block, axis=1))), s


@dataclass
class Side:
    """Rows formed as the kernels form them."""

    x: np.ndarray                 # rows x k FP32 input words (BF16 << 16)
    alpha: np.ndarray             # FP32 words
    beta: np.ndarray
    ae: np.ndarray                # a_E, UE4M3 bytes
    e: np.ndarray                 # rows x 32 E codes
    codes: np.ndarray             # rows x k E2M1
    scales: np.ndarray            # rows x k/block


def form(fmt: F.Fmt, x: np.ndarray, e: np.ndarray, basis: np.ndarray, block: int = 1 << 16) -> Side:
    """`pearl_c4.form` with the noise atoms batched."""
    rows, k = x.shape
    st = [F.stats(fmt, [int(v) for v in row]) for row in x]
    alpha, beta = (np.array([s[i] for s in st], np.uint32) for i in (2, 3))
    ae = np.array([s[4] for s in st], np.uint8)
    acc = u32(f32(x) * f32(alpha)[:, None]).ravel()
    d = np.empty(rows * k, np.uint32)
    for lo in range(0, rows * k, block):
        ii, tt = np.divmod(np.arange(lo, min(lo + block, rows * k)), k)
        pad = np.zeros((len(ii), F.ATOM - R), np.uint8)
        d[lo:lo + len(ii)], bad = block_scaled_batch(
            BLACKWELL_SM120_NVF4, acc[lo:lo + len(ii)], np.concatenate([e[ii], pad], axis=1),
            np.concatenate([basis[tt], pad], axis=1), np.repeat(ae[ii, None], 4, axis=1),
            np.full((len(ii), 4), F.NOISE_F_SCALE, np.uint8))
        if bad.any():
            raise P.Reject("a noise atom left the model")
    codes, scales = quantize(fmt, d.reshape(rows, k))
    return Side(x, alpha, beta, ae, e, codes, scales)


def chain(fmt: F.Fmt, a: np.ndarray, sa: np.ndarray, b: np.ndarray, sb: np.ndarray, ii: np.ndarray, jj: np.ndarray
          ) -> np.ndarray:
    """`pearl_c4.chain` for the cells (ii, jj) of code matrices a, b with their scale bytes sa, sb."""
    k, ns = a.shape[1], F.ATOM // fmt.block
    acc = np.zeros(len(ii), np.uint32)
    for s in range(k // F.ATOM):
        ks, ss = slice(s * F.ATOM, (s + 1) * F.ATOM), slice(s * ns, (s + 1) * ns)
        acc, bad = block_scaled_batch(fmt.atom, acc, a[ii, ks], b[jj, ks], sa[ii, ss], sb[jj, ss])
        if bad.any():
            raise P.Reject("a chain step left the model")
    return acc


def cells(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return ii.ravel(), jj.ravel()


def rne_bf16(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.uint64)
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16).astype(np.uint16)


def e_tilde(side: Side) -> np.ndarray:
    """Ẽ = bf16(E · a_E · 2^-4), exact."""
    v = _E2M1[side.e] * ue4m3_value(side.ae)[:, None] * np.float32(1 / 16)
    return (u32(v) >> 16).astype(np.uint16)


@dataclass
class Job:
    """What every tile of a shape shares: F_A and F_B (k x 32 codes, rows along k) and G."""

    fmt: F.Fmt
    seed_a: bytes
    seed_b: bytes
    fa: np.ndarray
    fb: np.ndarray
    g: np.ndarray
    ones: np.ndarray


def job(fmt: F.Fmt, seed_a: bytes, seed_b: bytes, k: int) -> Job:
    fa, fb = (np.array(F.basis(seed_b, role, k), np.uint8) for role in (F.ROLE_FA, F.ROLE_FB))
    ones = np.full((R, k // fmt.block), fmt.one, np.uint8)
    gi, gj = cells(R, R)
    g = chain(fmt, fb.T.copy(), ones, fa.T.copy(), ones, gi, gj).reshape(R, R)
    return Job(fmt, seed_a, seed_b, fa, fb, g, ones)


@dataclass
class Tile:
    a: Side
    b: Side
    pa: np.ndarray                # 64 x 64 BF16
    pb: np.ndarray
    c: np.ndarray                 # 64 x 64 FP32 words
    u: np.ndarray
    leaf: bytes
    tickets: list[bytes]


def tile(jb: Job, a_rows: np.ndarray, b_rows: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> Tile:
    """The tile of A rows `rows` and B rows `cols` (their BF16 words a_rows, b_rows)."""
    fmt = jb.fmt
    a = form(fmt, a_rows.astype(np.uint32) << 16, np.array([F.line(jb.seed_a, F.ROLE_EA, int(i)) for i in rows], np.uint8),
             jb.fa)
    b = form(fmt, b_rows.astype(np.uint32) << 16, np.array([F.line(jb.seed_b, F.ROLE_EB, int(j)) for j in cols], np.uint8),
             jb.fb)
    na, nb = len(rows), len(cols)
    ri, rq = cells(na, R)
    af = chain(fmt, a.codes, a.scales, jb.fb.T.copy(), jb.ones, ri, rq).reshape(na, R)
    ci, cq = cells(nb, R)
    bf = chain(fmt, b.codes, b.scales, jb.fa.T.copy(), jb.ones, ci, cq).reshape(nb, R)
    ea, eb = e_tilde(a), e_tilde(b)
    pa = np.concatenate([ea, rne_bf16(af)], axis=1)
    g = jb.g.tolist()
    mid = []
    for et, bfr in zip(eb.tolist(), bf.tolist()):
        row = []
        for q in range(R):
            acc = 0
            for p in range(R):
                acc = fp32.fma(et[p] << 16, g[p][q], acc)
            row.append(C._bf16_of_word(fp32.add(acc, fp32.neg(bfr[q]))))
        mid.append(row)
    pb = np.concatenate([np.array(mid, np.uint16), eb ^ np.uint16(0x8000)], axis=1)
    ti, tj = cells(na, nb)
    c = chain(fmt, a.codes, a.scales, b.codes, b.scales, ti, tj)
    u = c.copy()
    for s in range(0, F.PEEL_K, 16):
        u = group_sum_total_batch(HOPPER_BF16_M16N8K16, u, pa[ti, s:s + 16], pb[tj, s:s + 16])[0]
    c, u = c.reshape(na, nb), u.reshape(na, nb)
    checked = C.Checked(c.tolist(), u.tolist())
    tickets = C.tickets(jb.seed_a, checked, "h3")
    return Tile(a, b, pa, pb, c, u, C.leaf(checked), tickets)
