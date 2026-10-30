"""The 2:4 sparse E4M3 row on sm_120 against two dense k32 atoms on the same operands (``tc_sparse_e4m3``,
``tc-model/sm120-e4m3-sp-k64``), split like recheck.py so that no CPU verification runs inside a GPU lease.

    python benchmarks/tc_probe_fp4/sp_e4m3.py capture --seeds 20261201,20261202 --n-random 327680 --prebuilt SO \
        --expect-uuid "$GPU_LEASE_UUID" --out DIR --budget-s 360          # a fill chunk; exit 99 while seeds remain
    python benchmarks/tc_probe_fp4/sp_e4m3.py verify --seed 20261201 --n-random 327680 --capture DIR --prebuilt SO --out OUT   # CPU

The sparse row is ``mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32`` (sp_e4m3.cu's
``sp_e4m3_kernel``, one per tile).  A is 16 x 64 logical, 2:4 structured along k: each group of four logical columns keeps two,
at indices i0 < i1 named by the group's metadata nibble ``(i1 << 2) | i0`` (six valid values), stored in that order; the stored
row is 32 bytes, group g's two at stored columns 2g, 2g + 1.  B is all of the 64 x 8 logical operand.  The dense atom is
``mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32`` (``dense_chain_kernel``, L dependent atoms in registers).

The host packs every register (``pack_*``), so the layout check settles the fragment layout on the device against exact integer
tiles: the packings in ``META_READINGS`` x ``B_READINGS`` are each tried, the one that reproduces C + A.B on every word is used
for the capture and recorded (none: the chunk fails with the tiles dumped); the dense chain has one packing and must reproduce it.

``capture``: the UUID gate, the SASS gate (exactly one MMA per kernel, the sparse kernel's a ``.SP`` one), the layout check,
then per seed the families of ``families`` through the sparse kernel and, on the same operands, the dense twins (hardware against
hardware): ``dense_two_atoms`` (two chained atoms, logical columns 0..31 then 32..63, from the family's accumulator, +0 in the
``*zero_acc`` families), ``dense_two_atoms_swapped`` (32..63 first) and, where the tile's rows share one metadata row,
``dense_on_compressed`` (one atom on the stored A and the 32 B rows the metadata selects).  Each family's words are checkpointed
with the sha256 of its operands; no family starts once ``--budget-s`` has passed (exit 99).

``verify``: regenerates the operands, checks both sha256s, and counts, per family, the words where the sparse instruction differs
from each dense twin (the finding: whether 2:4-sparse weights run the dense chain's numerics) and from each candidate model on the
stored operands (``candidates``: ``gather_g32``, one GroupSum (32,)/26/-133 over the 32 kept products, the dense atom on the
compressed operands; ``two_atoms`` and ``two_atoms_swapped``, the same adder as two groups of 16, one per logical half, in either
order).  The gate: the capture's integrity, and every dense twin reproduced by its model on every word (the dense E4M3 atom's
GroupSum, the vLLM port's sm_120 pin, is the control); which candidates reproduce every sparse word is recorded, never gated.
Each family's counts are checkpointed in OUT/families/, bound to its capture record; no family's models start once
``--budget-s`` has passed (exit 99), so a CPU fill job resumes where it stopped.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as P
import k32

from verity_config.settings import SHARED

SCHEMA = "pouw-rtxpro/tc-sparse-e4m3/v1"
INSTRUCTION_ID = "sm120.mma.sp.m16n8k64.e4m3"
PTX = "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32"
DENSE_PTX = "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32"
KERNELS = ("sp_e4m3_kernel", "dense_chain_kernel")
K_LOGICAL, K_STORED, GROUPS = 64, 32, 16
MAX_TILES_PER_CALL = 1 << 16

#: the metadata nibbles mma.sp::ordered_metadata accepts: kept indices i0 < i1 of a group of four as (i1 << 2) | i0
META_VALID = np.array([0b0100, 0b1000, 0b1100, 0b1001, 0b1101, 0b1110], np.uint8)
_PAIR = np.zeros((16, 2), np.int64)
for _v in META_VALID.tolist():
    _PAIR[_v] = [_v & 3, _v >> 2]

_LANE = np.arange(32)
_G, _Q = _LANE >> 2, _LANE & 3


def logical_columns(meta: np.ndarray) -> np.ndarray:
    """(..., 16) group nibbles -> (..., 32): the logical column of each stored column."""
    return (4 * np.arange(GROUPS)[:, None] + _PAIR[np.asarray(meta, np.int64)]).reshape(*np.shape(meta)[:-1], K_STORED)


def expand(stored: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Stored A (..., 32) and its metadata (..., 16) -> the logical A (..., 64), code 0 (+0) where nothing is kept."""
    out = np.zeros(stored.shape[:-1] + (K_LOGICAL,), stored.dtype)
    np.put_along_axis(out, logical_columns(meta), stored, axis=-1)
    return out


def gather_b(B: np.ndarray, meta_row: np.ndarray) -> np.ndarray:
    """B (n, 8, 64) and one metadata row per tile (n, 16) -> the 32 B rows it selects, (n, 8, 32)."""
    cols = np.broadcast_to(logical_columns(meta_row)[:, None, :], (B.shape[0], 8, K_STORED))
    return np.take_along_axis(B, cols, axis=-1)


# --- register packing (host side) ------------------------------------------------------------------------------------------------


def _words(b: np.ndarray) -> np.ndarray:
    """(..., 4) bytes -> (...) little-endian u32."""
    return np.ascontiguousarray(b.astype(np.uint8)).view("<u4")[..., 0]


def pack_a(A: np.ndarray, k0: int = 0) -> np.ndarray:
    """A (n, 16, >= k0 + 32) bytes -> (n, 32, 4) u32: reg r, byte j = row g + 8(r&1), column k0 + 4q + j + 16(r>>1)."""
    r = np.arange(4)
    rows = (_G[:, None] + 8 * (r[None, :] & 1))[:, :, None]
    cols = (k0 + 4 * _Q[:, None, None] + np.arange(4)[None, None, :] + 16 * (r[None, :, None] >> 1))
    return _words(A[:, rows, cols])


def _pack_b(B: np.ndarray, kmap) -> np.ndarray:
    nreg = kmap.shape[1]
    return _words(B[:, np.broadcast_to(_G[:, None, None], (32, nreg, 4)), kmap])


#: how the sparse instruction may read B's 64 x 8 fragment (4 registers): lane 4g + q, reg r, byte j -> column g, row k
B_READINGS = {
    "k_4q_16r": 4 * _Q[:, None, None] + np.arange(4)[None, None, :] + 16 * np.arange(4)[None, :, None],
    "k_16q_4r": 16 * _Q[:, None, None] + np.arange(4)[None, None, :] + 4 * np.arange(4)[None, :, None],
}


def pack_b_dense(B: np.ndarray, k0: int = 0) -> np.ndarray:
    """B (n, 8, >= k0 + 32) -> (n, 32, 2) u32: reg r, byte j = column g, row k0 + 4q + j + 16r (the dense k32 fragment)."""
    kmap = k0 + 4 * _Q[:, None, None] + np.arange(4)[None, None, :] + 16 * np.arange(2)[None, :, None]
    return _pack_b(B, kmap)


def _meta_word(meta: np.ndarray, rows: np.ndarray, groups: np.ndarray) -> np.ndarray:
    nib = meta[:, rows[:, None], groups].astype(np.uint32)  # (n, 32, 8)
    return (nib << (4 * np.arange(8, dtype=np.uint32))).sum(axis=-1, dtype=np.uint32)


#: how the hardware may read the metadata word of lane 4g + q: (row, the eight groups in nibble order)
META_READINGS = {
    "row_q1_groups_q2": (_G + 8 * (_Q & 1), 8 * (_Q[:, None] >> 1) + np.arange(8)[None, :]),
    "row_q2_groups_q1": (_G + 8 * (_Q >> 1), 8 * (_Q[:, None] & 1) + np.arange(8)[None, :]),
    "row_q1_groups_q2_reversed": (_G + 8 * (_Q & 1), 8 * (_Q[:, None] >> 1) + 7 - np.arange(8)[None, :]),
}


def pack_meta(meta: np.ndarray, reading: str) -> np.ndarray:
    rows, groups = META_READINGS[reading]
    return _meta_word(np.asarray(meta, np.uint8), rows, groups)


def pack_c(C: np.ndarray) -> np.ndarray:
    """C (n, 16, 8) u32 -> (n, 32, 4): c0 = (g, 2q), c1 = (g, 2q+1), c2 = (g+8, 2q), c3 = (g+8, 2q+1)."""
    rows = np.stack([_G, _G, _G + 8, _G + 8], axis=1)
    cols = np.stack([2 * _Q, 2 * _Q + 1, 2 * _Q, 2 * _Q + 1], axis=1)
    return np.ascontiguousarray(np.asarray(C, np.uint32)[:, rows, cols])


def unpack_d(R: np.ndarray) -> np.ndarray:
    D = np.zeros((R.shape[0], 16, 8), np.uint32)
    rows = np.stack([_G, _G, _G + 8, _G + 8], axis=1)
    cols = np.stack([2 * _Q, 2 * _Q + 1, 2 * _Q, 2 * _Q + 1], axis=1)
    D[:, rows, cols] = R
    return D


# --- candidates ------------------------------------------------------------------------------------------------------------------


def candidates() -> dict:
    """On the stored operands (32 kept products per element, logical half 0 in stored columns 0..15): see the module docstring."""
    gs = k32.GroupSumCandidate
    return {"gather_g32": gs("gather_g32", "e4m3", "e4m3", (32,), 26, -133),
            "two_atoms": gs("two_atoms", "e4m3", "e4m3", (16, 16), 26, -133),
            "two_atoms_swapped": gs("two_atoms_swapped", "e4m3", "e4m3", (16, 16), 26, -133)}


#: the model each dense twin must reproduce on every word (the control), on the same stored operands
TWIN_MODEL = {"dense_two_atoms": "two_atoms", "dense_two_atoms_swapped": "two_atoms_swapped", "dense_on_compressed": "gather_g32"}


def model_words(models: dict, As, META, B, C) -> tuple[dict, dict]:
    n = As.shape[0]
    cols = logical_columns(META).tolist()
    Al, Bl, Cl = As.tolist(), B.tolist(), np.asarray(C, np.uint32).tolist()
    out = {m: np.zeros((n, 16, 8), np.uint32) for m in models}
    err = {m: np.zeros((n, 16, 8), np.bool_) for m in models}
    for t in range(n):
        for i in range(16):
            a, ci = Al[t][i], cols[t][i]
            a_sw = a[16:] + a[:16]
            for j in range(8):
                bj = Bl[t][j]
                b = [bj[c] for c in ci]
                for m, model in models.items():
                    try:
                        out[m][t, i, j] = (model.step_scaled(Cl[t][i][j], a_sw, b[16:] + b[:16]) if m == "two_atoms_swapped"
                                           else model.step_scaled(Cl[t][i][j], a, b))
                    except Exception:  # noqa: BLE001 - outside the model's domain (non-finite accumulator, sum leaving binary32)
                        err[m][t, i, j] = True
    return out, err


_CTX: dict = {}


def _init() -> None:
    _CTX["models"] = candidates()


def _eval_chunk(args):
    return model_words(_CTX["models"], *args)


def evaluate(pool, As, META, B, C, procs: int):
    if pool is None:
        return model_words(candidates(), As, META, B, C)
    idx = np.array_split(np.arange(As.shape[0]), max(1, min(As.shape[0] // 16 + 1, procs * 8)))
    parts = pool.map(_eval_chunk, [(As[i], META[i], B[i], C[i]) for i in idx])
    return ({m: np.concatenate([p[0][m] for p in parts]) for m in parts[0][0]},
            {m: np.concatenate([p[1][m] for p in parts]) for m in parts[0][1]})


# --- devices ---------------------------------------------------------------------------------------------------------------------


class Device:
    tcgen05 = False

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        V = ctypes.c_void_p
        self.lib.sp_e4m3_tiles.restype = ctypes.c_int
        self.lib.sp_e4m3_tiles.argtypes = [V] * 5 + [ctypes.c_int]
        self.lib.dense_e4m3_chain.restype = ctypes.c_int
        self.lib.dense_e4m3_chain.argtypes = [V] * 4 + [ctypes.c_int, ctypes.c_int]
        self.lib.device_uuid.restype = ctypes.c_int
        self.lib.device_uuid.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.has_mxf4.restype = ctypes.c_int
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]

    def identity(self) -> dict:
        ident = P.Probe.identity(self)
        ident.pop("has_mxf4", None)
        return ident

    clocks = P.Probe.clocks

    def set_launch(self, on: bool) -> int:
        self.lib.set_kernel_launch(1 if on else 0)
        return int(self.lib.d_poison_byte())

    def sparse(self, As, B, META, C, meta_reading: str, b_reading: str) -> np.ndarray:
        n, D = As.shape[0], np.zeros((As.shape[0], 16, 8), np.uint32)
        for s in range(0, n, MAX_TILES_PER_CALL):
            e = min(n, s + MAX_TILES_PER_CALL)
            ra = np.ascontiguousarray(pack_a(As[s:e]))
            rb = np.ascontiguousarray(_pack_b(B[s:e], B_READINGS[b_reading]))
            re_ = np.ascontiguousarray(pack_meta(META[s:e], meta_reading))
            rc = pack_c(C[s:e])
            rd = np.zeros((e - s, 32, 4), np.uint32)
            err = self.lib.sp_e4m3_tiles(ra.ctypes.data, rb.ctypes.data, re_.ctypes.data, rc.ctypes.data, rd.ctypes.data, e - s)
            if err != 0:
                raise RuntimeError(f"CUDA error {err} in the sparse E4M3 kernel")
            D[s:e] = unpack_d(rd)
        return D

    def dense(self, A, B, C, order) -> np.ndarray:
        """``order``: the 32-column slices of A (n, 16, 32 len(order)) / B (n, 8, ...) the chain's atoms take, in order."""
        n, L, D = A.shape[0], len(order), np.zeros((A.shape[0], 16, 8), np.uint32)
        for s in range(0, n, MAX_TILES_PER_CALL):
            e = min(n, s + MAX_TILES_PER_CALL)
            ra = np.ascontiguousarray(np.stack([pack_a(A[s:e], 32 * h) for h in order], axis=1))
            rb = np.ascontiguousarray(np.stack([pack_b_dense(B[s:e], 32 * h) for h in order], axis=1))
            rc = pack_c(C[s:e])
            rd = np.zeros((e - s, 32, 4), np.uint32)
            err = self.lib.dense_e4m3_chain(ra.ctypes.data, rb.ctypes.data, rc.ctypes.data, rd.ctypes.data, e - s, L)
            if err != 0:
                raise RuntimeError(f"CUDA error {err} in the dense E4M3 chain kernel")
            D[s:e] = unpack_d(rd)
        return D


class FakeDevice:
    """CPU stand-in (``--fake-silicon MODEL``): the sparse words are MODEL's, the dense twins the dense atom's model; a dry run,
    refused with ``RESEARCH_RUN_ID`` set."""

    def __init__(self, model_name: str):
        models = candidates()
        if model_name not in models:
            raise SystemExit(f"--fake-silicon: {model_name!r} is not a candidate ({sorted(models)})")
        self.model_name = model_name

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.model_name})", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def set_launch(self, on: bool) -> int:
        return 0xA5

    def sparse(self, As, B, META, C, meta_reading: str, b_reading: str) -> np.ndarray:
        words, errs = model_words({self.model_name: candidates()[self.model_name]}, As, META, B, C)
        return np.where(errs[self.model_name], np.uint32(0x7FFFFFFF), words[self.model_name])

    def dense(self, A, B, C, order) -> np.ndarray:
        dense = k32.GroupSumCandidate("dense", "e4m3", "e4m3", (32,), 26, -133)
        n, D = A.shape[0], np.asarray(C, np.uint32).copy()
        Al, Bl = A.tolist(), B.tolist()
        for h in order:
            Dl = D.tolist()
            for t in range(n):
                for i in range(16):
                    for j in range(8):
                        D[t, i, j] = dense.step_scaled(Dl[t][i][j], Al[t][i][32 * h:32 * h + 32], Bl[t][j][32 * h:32 * h + 32])
        return D


# --- the layout check ------------------------------------------------------------------------------------------------------------


def layout_check(dev, debug_dir: Path | None) -> dict:
    """Exact integer tiles (every sum exact in FP32): the sparse packing that reproduces C + A.B on every word, and the dense chain."""
    rng = np.random.default_rng(1)
    n = 64
    va, As = k32.small_operands(rng, "e4m3", (n, 16, K_STORED))
    vb, B = k32.small_operands(rng, "e4m3", (n, 8, K_LOGICAL))
    vb, B = np.where(vb == 0, 1.0, vb), np.where(B == 0, np.uint8(0x38), B)  # nonzero everywhere: unselected B rows must not count
    meta = META_VALID[rng.integers(0, len(META_VALID), (n, 16, GROUPS))]
    c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float64)
    C = P.f32_bits(c)
    al = expand(va, meta)
    want = P.f32_bits(c + np.einsum("nik,njk->nij", al, vb))
    fits = []
    for mr in META_READINGS:
        for br in B_READINGS:
            if np.array_equal(dev.sparse(As, B, meta, C, mr, br), want):
                fits.append([mr, br])
    got_dense = dev.dense(expand(As, meta), B, C, (0, 1))
    dense_ok = bool(np.array_equal(got_dense, want))
    out = {"sparse_packings_reproducing": fits, "sparse_packing": fits[0] if fits else None,
           "dense_chain": "passed" if dense_ok else f"{int(np.count_nonzero(got_dense != want))} of {want.size} words differ"}
    if (not fits or not dense_ok) and debug_dir is not None:
        np.savez(Path(debug_dir) / "layout_debug_sp_e4m3.npz", As=As, B=B, meta=meta, C=C, want=want, dense=got_dense,
                 **{f"sparse_{mr}_{br}": dev.sparse(As, B, meta, C, mr, br) for mr in META_READINGS for br in B_READINGS})
    return out


# --- families --------------------------------------------------------------------------------------------------------------------


def families(seed: int, n_random: int, fixed_mult: int = 1):
    """Yield (name, As (n,16,32) stored E4M3 codes, META (n,16,16), B (n,8,64), C (n,16,8), shared).  ``shared``: the tile's rows
    share one metadata row (so one dense atom on the compressed operands is its twin).  ``fixed_mult`` multiplies ``signed_zero``."""
    rng = np.random.default_rng(seed)

    def codes(shape, nonzero=False):
        return k32._codes(rng, "e4m3", shape, nonzero)

    def meta(n, shared=False):
        m = META_VALID[rng.integers(0, len(META_VALID), (n, 1 if shared else 16, GROUPS))]
        return np.repeat(m, 16, axis=1) if shared else m

    def acc_random(n, lo=-30, hi=40):
        return k32._f32((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(lo, hi, (n, 16, 8)))).astype(np.float32))

    def zeros(n):
        return np.zeros((n, 16, 8), np.uint32)

    n = n_random // 2
    yield "random_acc", codes((n, 16, K_STORED)), meta(n), codes((n, 8, K_LOGICAL)), acc_random(n), False
    n = n_random // 4
    yield "zero_acc", codes((n, 16, K_STORED)), meta(n), codes((n, 8, K_LOGICAL)), zeros(n), False
    n = n_random // 4
    yield "shared_meta_random_acc", codes((n, 16, K_STORED)), meta(n, True), codes((n, 8, K_LOGICAL)), acc_random(n), True
    n = n_random // 8
    yield "shared_meta_zero_acc", codes((n, 16, K_STORED)), meta(n, True), codes((n, 8, K_LOGICAL)), zeros(n), True
    # large accumulators over the products: the accumulator sets the alignment and each atom truncates onto its grid
    n = n_random // 8
    yield "acc_window", codes((n, 16, K_STORED)), meta(n), codes((n, 8, K_LOGICAL)), acc_random(n, 10, 50), False
    n = n_random // 8
    yield "tiny_acc", codes((n, 16, K_STORED)), meta(n), codes((n, 8, K_LOGICAL)), k32._uniform_acc(rng, n, 0, 30), False

    # the second logical half's kept products cancel the first half's exactly (10% residual codes): a chain rounds the first
    # atom at the products' magnitude before the cancellation, one group over the 32 kept products does not
    n = n_random // 8
    As, M = codes((n, 16, K_STORED)), meta(n)
    M[:, :, 8:] = M[:, :, :8]
    As[:, :, 16:] = As[:, :, :16]
    res = rng.random((n, 16, 16)) < 0.1
    As[:, :, 16:] = np.where(res, codes((n, 16, 16)), As[:, :, 16:])
    B = codes((n, 8, K_LOGICAL))
    B[:, :, 32:] = np.where(B[:, :, :32] == 0, 0, B[:, :, :32] ^ 0x80)
    C = zeros(n)
    C[n // 2:] = acc_random(n - n // 2, -40, -10)
    yield "half_cancel", As, M, B, C, False

    # one logical half live, the other all +0 codes (either half): one atom of the chain carries only the accumulator
    n = n_random // 8
    As = codes((n, 16, K_STORED))
    first = rng.random(n) < 0.5
    As[first, :, :16] = 0
    As[~first, :, 16:] = 0
    yield "one_half_live", As, meta(n), codes((n, 8, K_LOGICAL)), acc_random(n, -20, 20), False

    # one kept product in each logical half, the accumulator walked around the larger (each atom's alignment edge)
    n = n_random // 8
    As = np.zeros((n, 16, K_STORED), np.uint8)
    t = np.arange(n)
    s0, s1 = rng.integers(0, 16, n), rng.integers(16, 32, n)
    As[t, :, s0] = codes((n, 16), True)
    As[t, :, s1] = codes((n, 16), True)
    M, B = meta(n), codes((n, 8, K_LOGICAL), True)
    cols = logical_columns(M)
    pmax = np.zeros((n, 16, 8))
    for s in (s0, s1):
        va = np.abs(k32.E4M3_VALUE[As[t, :, s].astype(np.int64)])  # (n,16)
        vb = np.abs(k32.E4M3_VALUE[B[t[:, None, None], np.arange(8)[None, None, :], cols[t, :, s][:, :, None]].astype(np.int64)])
        pmax = np.maximum(pmax, va[:, :, None] * vb)
    yield "two_product_walk", As, M, B, k32._acc_near(rng, pmax, -40, 10), False

    # E4M3 subnormal operands (exponent field 0) mixed with random codes, accumulators +0 or tiny
    n = n_random // 8
    sub = (rng.integers(1, 8, (n, 16, K_STORED)) | (rng.integers(0, 2, (n, 16, K_STORED)) << 7)).astype(np.uint8)
    As = np.where(rng.random((n, 16, K_STORED)) < 0.5, sub, codes((n, 16, K_STORED)))
    subb = (rng.integers(1, 8, (n, 8, K_LOGICAL)) | (rng.integers(0, 2, (n, 8, K_LOGICAL)) << 7)).astype(np.uint8)
    B = np.where(rng.random((n, 8, K_LOGICAL)) < 0.5, subb, codes((n, 8, K_LOGICAL)))
    C = zeros(n)
    C[n // 2:] = k32._uniform_acc(rng, n - n // 2, 0, 30)
    yield "subnormal_operands", As, meta(n), B, C, False

    # signed zeros: +-0 operands against +-0 accumulators; half the tiles cancel exactly (+-1 x 3 in the first half)
    n = 256 * fixed_mult
    As = np.where(rng.random((n, 16, K_STORED)) < 0.5, 0, 0x80).astype(np.uint8)
    As[n // 2:, :, :16] = np.where(np.arange(16) % 2 == 0, 0x38, 0xB8).astype(np.uint8)
    B = codes((n, 8, K_LOGICAL))
    B[n // 2:, :, :32] = 0x44
    C = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", As, meta(n), B, C, False


def twins(As, M, B, C, shared: bool) -> dict:
    """label -> (A, B, C, order) for the dense chain on the same operands."""
    A = expand(As, M)
    out = {"dense_two_atoms": (A, B, C, (0, 1)), "dense_two_atoms_swapped": (A, B, C, (1, 0))}
    if shared:
        out["dense_on_compressed"] = (As, gather_b(B, M[:, 0, :]), C, (0,))
    return out


def sass_gate(sass: dict) -> tuple[dict, list[str]]:
    """This build's two kernels (their names are mangled) and what fails the gate: a kernel without exactly one MMA, a missing
    kernel, a sparse kernel whose MMA is not a sparse one."""
    mine = {k: next((r for f, r in sass.items() if k in f), None) for k in KERNELS}
    bad = [f for f, r in sass.items() if r["mma_count"] != 1] + [k for k, r in mine.items() if r is None]
    if mine["sp_e4m3_kernel"] is not None and not all(".SP" in op for op in mine["sp_e4m3_kernel"]["opcodes"]):
        bad.append(f"sp_e4m3_kernel lowers to {mine['sp_e4m3_kernel']['opcodes']}, not a sparse MMA")
    return {k: (r or {}).get("opcodes") for k, r in mine.items()}, bad


def operands_sha256(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def _save(path: Path, D: np.ndarray) -> str:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, D)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- capture (a fill chunk) ------------------------------------------------------------------------------------------------------


def capture(args) -> int:
    t_all = time.time()
    root = Path(os.path.expandvars(args.out))
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    todo = [s for s in seeds if not (root / str(s) / "done.json").exists()]
    if not todo:
        print("SPARSE CAPTURE: every seed recorded", flush=True)
        return 0
    if args.fake_silicon:
        so, dev = None, FakeDevice(args.fake_silicon)
    else:
        so = Path(os.path.expandvars(args.prebuilt))
        dev = Device(so)
    identity = dev.identity()
    uuid = identity.get("gpu_uuid")
    expect = os.path.expandvars(args.expect_uuid) if args.expect_uuid else None
    if not args.fake_silicon and not (expect and P.uuid_allowed(uuid, expect)):
        print(f"FAILURES device UUID {uuid} != expected {expect}: stopped before any launch", flush=True)
        return 2
    head = {"schema": SCHEMA, "instruction": PTX, "instruction_id": INSTRUCTION_ID, "dense": DENSE_PTX, "n_random": args.n_random,
            "fixed_mult": args.fixed_mult, "identity": identity, "expect_uuid": expect, "fake_silicon": args.fake_silicon,
            "library": None if so is None else {"path": str(so), "sha256": hashlib.sha256(so.read_bytes()).hexdigest()}}
    if so is not None:
        head["sass"], bad = sass_gate(P.sass_check(so))
        if bad:
            print("FAILURES sass", bad, flush=True)
            return 1
        poison = np.uint32(int.from_bytes(bytes([dev.set_launch(False)] * 4), "little"))
        z = np.zeros((4, 16, 8), np.uint32)
        nowrite = [dev.sparse(np.zeros((4, 16, 32), np.uint8), np.zeros((4, 8, 64), np.uint8), np.full((4, 16, 16), 0b0100, np.uint8),
                              z, "row_q1_groups_q2", "k_4q_16r"),
                   dev.dense(np.zeros((4, 16, 64), np.uint8), np.zeros((4, 8, 64), np.uint8), z, (0, 1))]
        dev.set_launch(True)
        if not all(np.all(w == poison) for w in nowrite):
            print("FAILURES the no-write control did not read back the poison", flush=True)
            return 1
        head["no_write_control"] = "passed"
    head["layout_check"] = layout_check(dev, root)
    if head["layout_check"]["sparse_packing"] is None or head["layout_check"]["dense_chain"] != "passed":
        print("FAILURES layout check", json.dumps(head["layout_check"]), flush=True)
        return 1
    mr, br = head["layout_check"]["sparse_packing"]
    print("layout_check", json.dumps(head["layout_check"]), flush=True)
    captured = 0
    for seed in todo:
        d = root / str(seed)
        d.mkdir(exist_ok=True)
        fams = {}
        for name, As, M, B, C, shared in families(seed, args.n_random, args.fixed_mult):
            rec_path = d / f"{name}.json"
            if rec_path.exists():
                fams[name] = json.loads(rec_path.read_text())
                continue
            if captured and time.time() - t_all > args.budget_s:
                print(f"SPARSE CAPTURE: budget reached after {captured} families; exit 99", flush=True)
                return 99
            t0 = time.time()
            words = {"sparse": dev.sparse(As, B, M, C, mr, br)}
            for label, (a, b, c, order) in twins(As, M, B, C, shared).items():
                words[label] = dev.dense(a, b, c, order)
            dev_s = time.time() - t0
            rec = {"family": name, "tiles": int(As.shape[0]), "elements": int(As.shape[0]) * 128, "shared_metadata": shared,
                   "device_seconds": round(dev_s, 3), "operands_sha256": operands_sha256(As, M, B, C),
                   "words_sha256": {k: _save(d / f"{name}.{k}.npy", w) for k, w in words.items()}}
            write_json(rec_path, rec)
            fams[name] = rec
            captured += 1
            print(seed, name, rec["tiles"], "tiles", f"{dev_s:.2f}s", flush=True)
        write_json(d / "done.json", {**head, "seed": seed, "families": fams, "clocks": dev.clocks(uuid)})
        print(f"SPARSE CAPTURE: seed {seed} recorded ({len(fams)} families)", flush=True)
    return 0


# --- verify (CPU) ----------------------------------------------------------------------------------------------------------------


def _first(mask, As, M, B, C, D, other, n=5) -> list:
    out = []
    for t, i, j in list(zip(*np.nonzero(mask)))[:n]:
        out.append({"tile": int(t), "i": int(i), "j": int(j), "acc": f"{int(C[t, i, j]):08x}", "sparse": f"{int(D[t, i, j]):08x}",
                    "other": f"{int(other[t, i, j]):08x}", "a_stored": As[t, i].tobytes().hex(), "meta": M[t, i].tobytes().hex(),
                    "b": B[t, j].tobytes().hex()})
    return out


def verify(args) -> int:
    t_all = time.time()
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    d = Path(os.path.expandvars(args.capture)) / str(args.seed)
    failures: list[str] = []
    done = json.loads((d / "done.json").read_text())
    ident = done.get("identity") or {}
    if done.get("fake_silicon") or ident.get("backend") == "fake":
        failures.append("the capture ran on fake silicon (a dry run's invented words)")
    if not done.get("expect_uuid") or not P.uuid_allowed(ident.get("gpu_uuid"), done["expect_uuid"]):
        failures.append(f"the capture's device {ident.get('gpu_uuid')} was not gated on its lease ({done.get('expect_uuid')})")
    for k, v in (("n_random", args.n_random), ("seed", args.seed), ("schema", SCHEMA), ("fixed_mult", args.fixed_mult)):
        if done.get(k) != v:
            failures.append(f"the capture's {k} {done.get(k)!r} != {v!r}")
    if args.prebuilt and hashlib.sha256(Path(os.path.expandvars(args.prebuilt)).read_bytes()).hexdigest() != \
            (done.get("library") or {}).get("sha256"):
        failures.append("the capture loaded another library than --prebuilt")
    fams: dict = {}
    models = sorted(candidates())
    ck = out / "families"
    ck.mkdir(exist_ok=True)
    evaluated = 0
    if not failures:
        pool = mp.Pool(args.procs, initializer=_init) if args.procs > 1 else None
        try:
            for name, As, M, B, C, shared in families(args.seed, args.n_random, args.fixed_mult):
                rec = done["families"].get(name)
                if rec is None:
                    failures.append(f"{name}: not in the capture")
                    continue
                if operands_sha256(As, M, B, C) != rec["operands_sha256"]:
                    failures.append(f"{name}: the regenerated operands differ from the captured ones")
                    continue
                words = {}
                for k, h in rec["words_sha256"].items():
                    p = d / f"{name}.{k}.npy"
                    if hashlib.sha256(p.read_bytes()).hexdigest() != h:
                        failures.append(f"{name}: {p.name} changed since the capture")
                    words[k] = np.load(p)
                if any(f.startswith(f"{name}:") for f in failures):
                    continue
                cp, key = ck / f"{name}.json", hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
                if cp.exists() and (c := json.loads(cp.read_text()))["record_sha256"] == key:
                    fams[name] = c["family"]
                    failures += c["failures"]
                    continue
                if evaluated and time.time() - t_all > args.budget_s:
                    print(f"SPARSE VERIFY: budget reached after {evaluated} families; exit 99", flush=True)
                    return 99
                n0 = len(failures)
                t0 = time.time()
                mw, merr = evaluate(pool, As, M, B, C, args.procs)
                D = words["sparse"]
                fam = {"tiles": rec["tiles"], "elements": rec["elements"], "shared_metadata": shared,
                       "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)),
                       "sparse_vs_dense": {}, "sparse_vs_model": {}, "dense_vs_model": {}, "domain_errors": {m: int(merr[m].sum()) for m in models},
                       "first_differences": {}, "model_seconds": 0.0}
                for label in (k for k in words if k != "sparse"):
                    diff = words[label] != D
                    fam["sparse_vs_dense"][label] = int(diff.sum())
                    fam["first_differences"][label] = _first(diff, As, M, B, C, D, words[label])
                    m = TWIN_MODEL[label]
                    bad = (mw[m] != words[label]) | merr[m]
                    fam["dense_vs_model"][label] = {"model": m, "mismatches": int(bad.sum())}
                    if bad.any():
                        t, i, j = next(zip(*np.nonzero(bad)))
                        failures.append(f"{name}: the dense twin {label} differs from {m} on {int(bad.sum())} words "
                                        f"(first tile {int(t)} ({int(i)},{int(j)}): device {int(words[label][t, i, j]):08x})")
                for m in models:
                    fam["sparse_vs_model"][m] = int(((mw[m] != D) | merr[m]).sum())
                fam["model_seconds"] = round(time.time() - t0, 1)
                fams[name] = fam
                write_json(cp, {"record_sha256": key, "family": fam, "failures": failures[n0:]})
                evaluated += 1
                print(name, json.dumps({k: v for k, v in fam.items() if k != "first_differences"}), flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    labels = sorted({k for f in fams.values() for k in f["sparse_vs_dense"]})
    finding = {
        "sparse_vs_dense": {k: {"elements": sum(f["elements"] for f in fams.values() if k in f["sparse_vs_dense"]),
                                "differing_words": sum(f["sparse_vs_dense"][k] for f in fams.values() if k in f["sparse_vs_dense"])}
                            for k in labels},
        "candidates_reproducing_every_sparse_word": [m for m in models if fams and all(f["sparse_vs_model"][m] == 0 for f in fams.values())],
        "sparse_vs_model": {m: sum(f["sparse_vs_model"][m] for f in fams.values()) for m in models},
    }
    doc = {"schema": SCHEMA + "/verify", "instruction": PTX, "dense": DENSE_PTX, "seed": args.seed, "n_random": args.n_random,
           "fixed_mult": args.fixed_mult, "capture": {"dir": str(d), "done_sha256": hashlib.sha256((d / "done.json").read_bytes()).hexdigest()},
           "identity": ident, "layout_check": done.get("layout_check"), "sass": done.get("sass"), "families": fams,
           "elements": sum(f["elements"] for f in fams.values()),
           "min_family_elements": min((f["elements"] for f in fams.values()), default=0),
           "finding": finding, "failures": failures, "seconds": round(time.time() - t_all, 1)}
    (out / "sparse.json").write_text(json.dumps(doc, indent=1))
    ok = not failures and bool(fams)
    if SHARED.RESEARCH_RUN_ID:
        res = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": SHARED.need("RESEARCH_RUN_ID"),
               "instruction": INSTRUCTION_ID, "ptx": PTX, "sku": ident.get("device"), "gpu_uuid": ident.get("gpu_uuid"),
               "driver": ident.get("driver"), "compute_capability": ident.get("compute_capability"), "model_agrees": None,
               "candidate_finding": {"agreeing": finding["candidates_reproducing_every_sparse_word"],
                                     "mismatches": finding["sparse_vs_model"]},
               "dense_twin_finding": finding["sparse_vs_dense"],
               "counts": {"families": len(fams), "elements": doc["elements"], "min_family_elements": doc["min_family_elements"]},
               "workload_fingerprint": {"tool": "tc_sparse_e4m3", "seed": args.seed, "n_random": args.n_random,
                                        "fixed_mult": args.fixed_mult, "library_sha256": (done.get("library") or {}).get("sha256")},
               "validation": {"status": "passed" if ok else "failed",
                              "checker": "a fill chunk's capture (UUID, SASS, no-write and exact-integer layout gates), its operands "
                                         "regenerated from the seed and their sha256 checked, every dense twin reproduced by its "
                                         "model on every word; the sparse words against the twins and the candidates recorded",
                              "detail": "; ".join(failures) or (f"{len(fams)} families, {doc['elements']} words; sparse vs dense: "
                                                                + json.dumps(finding["sparse_vs_dense"])),
                              "evidence": ["sparse.json"]},
               "measurements": [{"name": "elements", "value": doc["elements"], "unit": "words"}],
               "measurement_files": ["sparse.json"], "artifacts": []}
        write_json(out / "result.json", res)
    print("SPARSE VERIFY", "ok" if ok else "FAILED", failures[:5], json.dumps(finding), f"{doc['seconds']}s", flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--seeds", required=True)
    c.add_argument("--n-random", type=int, required=True)
    c.add_argument("--fixed-mult", type=int, default=1)
    c.add_argument("--prebuilt", default=None)
    c.add_argument("--expect-uuid", default=None)
    c.add_argument("--budget-s", type=float, default=360.0)
    c.add_argument("--fake-silicon", default=None)
    c.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--n-random", type=int, required=True)
    v.add_argument("--fixed-mult", type=int, default=1)
    v.add_argument("--capture", required=True)
    v.add_argument("--prebuilt", default=None)
    v.add_argument("--procs", type=int, default=max(1, min(32, (os.cpu_count() or 4) - 2)))
    v.add_argument("--budget-s", type=float, default=float("inf"))
    v.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--arch", default="sm_120a")
    args = ap.parse_args(argv)
    if getattr(args, "fake_silicon", None) and SHARED.RESEARCH_RUN_ID:
        ap.error("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    if args.cmd == "build":
        out = Path(os.path.expandvars(args.out))
        out.mkdir(parents=True, exist_ok=True)
        so, nvcc = P.build(args.arch, out, "sp_e4m3.cu")
        sass = P.sass_check(so)
        mine, bad = sass_gate(sass)
        doc = {"schema": SCHEMA + "/build", "nvcc": nvcc, "arch": args.arch, "library": so.name,
               "sha256": hashlib.sha256(so.read_bytes()).hexdigest(), "sass": sass, "kernels": mine, "failures": bad}
        write_json(out / "build.json", doc)
        print("sass", json.dumps(mine), flush=True)
        print("FAILURES sass" if bad else "SASS GATE PASSED", bad, flush=True)
        return 1 if bad else 0
    if args.cmd == "capture" and not args.fake_silicon and not args.prebuilt:
        ap.error("capture needs --prebuilt (compile off the lease)")
    return capture(args) if args.cmd == "capture" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
