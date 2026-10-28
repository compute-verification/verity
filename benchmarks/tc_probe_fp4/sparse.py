"""The 2:4 sparse FP4 rows on sm_120 (``probe.py --instruction sp_nvf4 | sp_mxf4``): one
``mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale`` per tile (``mma_fp4.cu``'s
``sparse_tiles_kernel``, SASS ``OMMA.SF.SP.168128``), raw bits in and out, against candidate models.  A capture: which candidates
reproduce every word is the finding (``candidate_finding``); ``--expect-model NAME`` makes one the gate (a fresh-seed confirmation).

Operands (PTX ISA "Sparse matrix storage", .e2m1 with kind::mxf4nvf4): A is 16 x 128 logical, pair-wise 4:8 structured sparse.
Each 8-wide chunk of a row keeps two of its four element pairs p0 < p1, named by the chunk's metadata nibble ``(p1 << 2) | p0``
(six valid values); the stored row is 64 codes, chunk c's four at stored columns 4c..4c+3, pair p0's two first.  B is all of the
128 x 8 logical operand.  ``scale_vec::4X`` (UE4M3) gives each row / column four scales, each over 32 logical columns (16 stored);
``2X`` (UE8M0) two, each over 64 logical (32 stored).

Candidates (every one a ``verity.ml.circuits.silicon.models.BlockScaledAlignAdd`` with the pinned sm_120 adder constants, or a chain of the
pinned dense model; the hardware scale bytes are duplicated onto the model's fixed blocks of 16 / 32 logical columns):

* ``gather_dense``: k = 128 logical, groups of 32 logical columns.  Each such group holds exactly the 16 stored products of one
  dense group under one scale pair, so this is the dense pinned model (``BLACKWELL_SM120_NVF4`` / ``_MXF4``) on the stored A and
  the B rows its metadata selects: "the sparse instruction is the dense one on the compressed operands";
* ``logical_g16`` / ``logical_g64`` (mxf4 only: a group may not span two scales): the same adder over eight groups of 16 / two
  groups of 64 logical columns;
* ``two_halves``: the dense pinned model stepped twice, on logical columns 0..63 then 64..127 (a rounding in between);
* ``gather_dense_underflow_signed`` (mxf4): ``gather_dense`` with log entry D13's refuted signed underflow, the named alternative.

Gates, in order, before any family is captured: the device UUID (``--expect-uuid``), the SASS gate (the row's kernel present
with exactly one ``OMMA.SF.SP.168128`` of the row's type, every kernel of the build one MMA), and the layout check on exact small
integers (random valid metadata per row and chunk, nonzero filler in the B rows no metadata selects, distinct power-of-two scales
per block).  A failed layout check names which alternative metadata layouts would have reproduced it and dumps its operands.

Beside the models, every family with a dense twin also runs the DENSE ``mma.sync`` row on twin operands in the same process
(``dense_twin_finding``, hardware against hardware; see ``families``): whether the sparse word equals the dense instruction's word
on the compressed operands, on the noise atom (a k64 NVF4 atom with 32 zero lanes on both operands), and on two atoms chained.
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import probe as P
from verity.ml.circuits.silicon import models as M
from verity_config.settings import SHARED

KIND = {"sp_nvf4": 0, "sp_mxf4": 1}
PTX = {
    "sp_nvf4": "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
    "sp_mxf4": "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
}
INSTRUCTION_ID = {"sp_nvf4": "sm120.mma.sp.m16n8k128.e2m1.nvf4", "sp_mxf4": "sm120.mma.sp.m16n8k128.e2m1.mxf4"}
SCALE_KIND = {"sp_nvf4": "ue4m3", "sp_mxf4": "ue8m0"}
NBLK = {"sp_nvf4": 4, "sp_mxf4": 2}
KERNEL = {"sp_nvf4": "sparse_tiles_kernelILi0EE", "sp_mxf4": "sparse_tiles_kernelILi1EE"}
SASS_OPCODE = {"sp_nvf4": "OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X", "sp_mxf4": "OMMA.SF.SP.168128.F32.E2M1.E2M1.E8"}
PINNED = {"sp_nvf4": M.BLACKWELL_SM120_NVF4, "sp_mxf4": M.BLACKWELL_SM120_MXF4}
#: the dense mma.sync row (probe.py) whose hardware words the sparse ones are compared with
DENSE = {"sp_nvf4": "nvf4", "sp_mxf4": "mxf4"}
K_LOGICAL, K_STORED, CHUNKS = 128, 64, 16

#: the metadata nibbles mma.sp::ordered_metadata accepts: pairs p0 < p1 as (p1 << 2) | p0
META_VALID = np.array([0b0100, 0b1000, 0b1001, 0b1100, 0b1101, 0b1110], np.uint8)


def _chunk_offsets() -> np.ndarray:
    off = np.zeros((16, 4), np.int64)
    for v in META_VALID.tolist():
        p0, p1 = v & 3, v >> 2
        off[v] = [2 * p0, 2 * p0 + 1, 2 * p1, 2 * p1 + 1]
    return off


OFFSETS = _chunk_offsets()


def logical_columns(meta: np.ndarray) -> np.ndarray:
    """(..., 16) chunk nibbles -> (..., 64): the logical column of each stored column."""
    return (8 * np.arange(CHUNKS)[:, None] + OFFSETS[np.asarray(meta, np.int64)]).reshape(*np.shape(meta)[:-1], K_STORED)


def expand(stored: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Stored A (..., 64) and its metadata (..., 16) -> the logical A (..., 128), zero (code 0 / value 0) where no pair is kept."""
    out = np.zeros(stored.shape[:-1] + (K_LOGICAL,), stored.dtype)
    np.put_along_axis(out, logical_columns(meta), stored, axis=-1)
    return out


def model_scales(s: np.ndarray) -> np.ndarray:
    """Hardware scale bytes (..., NB) -> the model's (..., 2 NB): each byte covers 128 / NB logical columns, two model blocks."""
    return np.repeat(np.asarray(s, np.uint8), 2, axis=-1)


# --- candidates ------------------------------------------------------------------------------------------------------------------


def _adder(instr: str, group: int, underflow_zero: str | None = None) -> M.BlockScaledAlignAdd:
    base = PINNED[instr]
    return M.BlockScaledAlignAdd(name=f"probe_{instr}_k128_g{group}", arch="sm_120", dtype="e2m1", operand_bits=4, scale=base.scale,
                                 k_per_instruction=K_LOGICAL, group=group, frac_bits=base.frac_bits, acc_window=base.acc_window,
                                 grid_floor=base.grid_floor, underflow_zero=underflow_zero or base.underflow_zero,
                                 validated="probe candidate for the 2:4 sparse row (hypothesis)")


@dataclass(frozen=True)
class TwoHalves:
    """The dense pinned model on logical columns 0..63, its word fed back as the accumulator of columns 64..127."""

    name: str
    dense: M.BlockScaledAlignAdd

    def step_scaled(self, c_word: int, a_codes, b_codes, a_scales, b_scales) -> int:
        h, s = K_LOGICAL // 2, len(a_scales) // 2
        c_word = self.dense.step_scaled(c_word, a_codes[:h], b_codes[:h], a_scales[:s], b_scales[:s])
        return self.dense.step_scaled(c_word, a_codes[h:], b_codes[h:], a_scales[s:], b_scales[s:])


def candidates(instr: str) -> dict:
    out = {"gather_dense": _adder(instr, 32), "logical_g16": _adder(instr, 16),
           "two_halves": TwoHalves(f"probe_{instr}_two_halves", PINNED[instr])}
    if SCALE_KIND[instr] == "ue8m0":
        # a group takes the scale of its first block, so a group may not span two hardware scales: 64 only under 2X
        out["logical_g64"] = _adder(instr, 64)
        out["gather_dense_underflow_signed"] = _adder(instr, 32, "signed")
    return out


def model_words(model, A, B, SA, SB, C) -> tuple[np.ndarray, np.ndarray]:
    """One model's word per element on logical operands: A (n,16,128), B (n,8,128) codes, SA (n,16,s) / SB (n,8,s) model scale
    bytes, C (n,16,8) -> (D, domain-error mask)."""
    n = A.shape[0]
    out = np.zeros((n, 16, 8), np.uint32)
    err = np.zeros((n, 16, 8), np.bool_)
    Al, Bl, SAl, SBl, Cl = A.tolist(), B.tolist(), SA.tolist(), SB.tolist(), C.tolist()
    for t in range(n):
        for i in range(16):
            for j in range(8):
                try:
                    out[t, i, j] = model.step_scaled(Cl[t][i][j], Al[t][i], Bl[t][j], SAl[t][i], SBl[t][j])
                except Exception:  # noqa: BLE001 - outside the model's domain (non-finite accumulator, sum leaving binary32)
                    err[t, i, j] = True
    return out, err


_CTX: dict = {}


def _init(instr: str) -> None:
    _CTX["models"] = candidates(instr)


def _eval_chunk(args):
    A, B, SA, SB, C = args
    outs, errs = {}, {}
    for name, model in _CTX["models"].items():
        outs[name], errs[name] = model_words(model, A, B, SA, SB, C)
    return outs, errs


def evaluate(pool, instr, As, B, META, SFA, SFB, C, procs: int):
    """Every candidate's words for a family (the pool's workers hold the candidates; ``pool`` None evaluates in this process)."""
    A, SA, SB = expand(As, META), model_scales(SFA), model_scales(SFB)
    if pool is None:
        _init(instr)
        return _eval_chunk((A, B, SA, SB, C))
    idx = np.array_split(np.arange(A.shape[0]), max(1, min(A.shape[0] // 16 + 1, procs * 8)))
    parts = pool.map(_eval_chunk, [(A[i], B[i], SA[i], SB[i], C[i]) for i in idx])
    return ({m: np.concatenate([p[0][m] for p in parts]) for m in parts[0][0]},
            {m: np.concatenate([p[1][m] for p in parts]) for m in parts[0][1]})


# --- devices ---------------------------------------------------------------------------------------------------------------------


class SparseDevice:
    tcgen05 = False

    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        V = ctypes.c_void_p
        fn = self.lib.mma_sparse_tiles
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int] + [V] * 7 + [ctypes.c_int]
        self.lib.device_uuid.restype = ctypes.c_int
        self.lib.device_uuid.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.has_mxf4.restype = ctypes.c_int
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def tiles(self, instr: str, As, B, META, SFA, SFB, C) -> np.ndarray:
        """As (n,16,64) stored codes, B (n,8,128) codes (n-major, k contiguous), META (n,16,16) nibbles, SFA (n,16,NB), SFB (n,8,NB),
        C (n,16,8) FP32 words -> D (n,16,8)."""
        n, nb = As.shape[0], NBLK[instr]
        arrs = [np.ascontiguousarray(x, np.uint8).reshape(n, *s) for x, s in
                ((As, (16, K_STORED)), (B, (8, K_LOGICAL)), (META, (16, CHUNKS)), (SFA, (16, nb)), (SFB, (8, nb)))]
        C = np.ascontiguousarray(C, np.uint32).reshape(n, 16, 8)
        D = np.zeros((n, 16, 8), np.uint32)
        for s in range(0, n, P.MAX_TILES_PER_CALL):
            e = min(n, s + P.MAX_TILES_PER_CALL)
            a, b, m, sa, sb, c = (np.ascontiguousarray(x[s:e]) for x in (*arrs, C))
            d = np.zeros((e - s, 16, 8), np.uint32)
            rc = self.lib.mma_sparse_tiles(KIND[instr], a.ctypes.data, b.ctypes.data, m.ctypes.data, sa.ctypes.data, sb.ctypes.data,
                                           c.ctypes.data, d.ctypes.data, e - s)
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in the {instr} sparse kernel")
            D[s:e] = d
        return D


class FakeSparse:
    """CPU stand-in (``--fake-silicon MODEL``): MODEL's words for the device; refused with ``RESEARCH_RUN_ID`` set."""

    tcgen05 = False

    def __init__(self, instr: str, model_name: str):
        models = candidates(instr)
        if model_name not in models:
            raise SystemExit(f"--fake-silicon: {model_name!r} is not a candidate of {instr} ({sorted(models)})")
        self.model_name, self.model = model_name, models[model_name]

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.model_name})", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def tiles(self, instr: str, As, B, META, SFA, SFB, C) -> np.ndarray:
        D, err = model_words(self.model, expand(np.asarray(As), META), np.asarray(B), model_scales(SFA), model_scales(SFB),
                             np.asarray(C, np.uint32))
        D[err] = 0x7FFFFFFF
        return D


# --- the layout check ------------------------------------------------------------------------------------------------------------

#: how the hardware might read the metadata the kernel packs (lane 4g+q, nibble c <- row g + 8(q&1), chunk 8(q>>1) + c): the
#: metadata each alternative would make the hardware see, as a function of the array the kernel was given
META_READINGS = {
    "as_packed": lambda m: m,
    "quad_bits_swapped": lambda m: m.reshape(-1, 2, 8, 2, 8).transpose(0, 3, 2, 1, 4).reshape(m.shape),
    "nibbles_reversed": lambda m: m.reshape(-1, 16, 2, 8)[..., ::-1].reshape(m.shape),
}


def exact_reference(a, b, meta, pa, pb, c) -> np.ndarray:
    """C + A.B on exact values: a (n,16,64) stored values, b (n,8,128), meta (n,16,16), pa (n,16,NB) / pb (n,8,NB) scale values."""
    n, nb = a.shape[0], pa.shape[2]
    al = expand(a.astype(np.float64), meta).reshape(n, 16, nb, K_LOGICAL // nb)
    prod = np.einsum("nigk,njgk->nijg", al, b.astype(np.float64).reshape(n, 8, nb, K_LOGICAL // nb))
    return P.f32_bits(c + np.einsum("nijg,nig,njg->nij", prod, pa, pb))


def layout_check(dev, instr: str, debug_dir: Path | None) -> str:
    rng = np.random.default_rng(1)
    n, nb = 64, NBLK[instr]
    vals = np.array([0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6])
    a = vals[rng.integers(0, len(vals), (n, 16, K_STORED))]
    b = vals[rng.integers(1, len(vals), (n, 8, K_LOGICAL))]  # nonzero everywhere: B rows no metadata selects must not count
    meta = META_VALID[rng.integers(0, len(META_VALID), (n, 16, CHUNKS))]
    c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float64)
    # distinct power-of-two scales in 2^-2..2^3 per (row, block) and (column, block); every sum stays exact in FP32
    if SCALE_KIND[instr] == "ue4m3":
        ea, eb = rng.integers(5, 11, (n, 16, nb)), rng.integers(5, 11, (n, 8, nb))
        sfa, sfb, pa, pb = (ea << 3).astype(np.uint8), (eb << 3).astype(np.uint8), 2.0 ** (ea - 7), 2.0 ** (eb - 7)
    else:
        ea, eb = rng.integers(125, 131, (n, 16, nb)), rng.integers(125, 131, (n, 8, nb))
        sfa, sfb, pa, pb = ea.astype(np.uint8), eb.astype(np.uint8), 2.0 ** (ea - 127), 2.0 ** (eb - 127)
    got = dev.tiles(instr, P.e2m1_encode_ints(a), P.e2m1_encode_ints(b), meta, sfa, sfb, P.f32_bits(c))
    want = exact_reference(a, b, meta, pa, pb, c)
    if np.array_equal(want, got):
        return "passed"
    fits = [name for name, read in META_READINGS.items() if np.array_equal(exact_reference(a, b, read(meta), pa, pb, c), got)]
    if debug_dir is not None:
        np.savez(Path(debug_dir) / f"layout_debug_{instr}.npz", want=want, got=got, a=a, b=b, meta=meta, sfa=sfa, sfb=sfb, c=c)
    bad = int(np.count_nonzero(want != got))
    raise RuntimeError(f"layout check failed for {instr}: {bad}/{want.size} elements differ from C + A.B on exact integer tiles; "
                       f"metadata readings that reproduce every word: {fits or 'none'}")


# --- families --------------------------------------------------------------------------------------------------------------------


def families(rng, n_random: int, instr: str):
    """Yield (name, As (n,16,64), B (n,8,128), META (n,16,16), SFA, SFB, C, twins).  ``twins`` names operand sets
    (A, Bt, SFA, SFB, C, steps) for the DENSE instruction (``DENSE[instr]``) whose hardware words the sparse words are compared
    with, word for word.

    The dense families of ``probe.families_nvf4`` carry over whole: each tile's A is the stored A under one metadata row shared by
    its 16 rows, the dense B fills the selected logical rows of B and random codes the rest; their twin ``dense_on_compressed`` is
    the dense family itself, so ``gather_dense`` on these tiles is the pinned dense model on the dense family.  The noise-atom
    families are k64 NVF4 atoms with 32 zero lanes on both operands (A's and B's zeros on the unselected pairs; one scale per 32
    logical columns, so each atom's four 16-column scales come in equal pairs): ``noise_atom_single`` holds one atom in logical
    columns 0..63 and zeros after it (twin ``dense_atom``: the dense instruction on the atom), ``noise_atom_pair`` two atoms
    (twin ``dense_two_atoms``: two dense MMAs, the atoms in order).  Then families with metadata drawn per row and chunk (the shared
    metadata never exercises the per-row gather)."""
    sk = SCALE_KIND[instr]
    nb = NBLK[instr]

    def shared_meta(n):
        return np.repeat(META_VALID[rng.integers(0, len(META_VALID), (n, 1, CHUNKS))], 16, axis=1)

    for name, A, Bt, SFA, SFB, C in P.families_nvf4(rng, n_random, sk):
        n = A.shape[0]
        meta = shared_meta(n)
        B = rng.integers(0, 16, (n, 8, K_LOGICAL)).astype(np.uint8)
        np.put_along_axis(B, np.broadcast_to(logical_columns(meta[:, :1, :]), (n, 8, K_STORED)), Bt, axis=-1)
        yield name, A, B, meta, SFA, SFB, C, {"dense_on_compressed": (A, Bt, SFA, SFB, C, 1)}

    def scales(shape):
        if sk == "ue4m3":
            return P.UE4M3_VALID[rng.integers(0, len(P.UE4M3_VALID), shape)]
        return rng.integers(107, 148, shape).astype(np.uint8)

    def acc(n, lo, hi):
        return P.f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(lo, hi, (n, 16, 8)))).astype(np.float32))

    for name, atoms in (("noise_atom_single", 1), ("noise_atom_pair", 2)):
        n = n_random // 4
        meta = shared_meta(n)
        As = rng.integers(0, 16, (n, 16, K_STORED)).astype(np.uint8)
        Bs = rng.integers(0, 16, (n, 8, K_STORED)).astype(np.uint8)
        if atoms == 1:
            As[:, :, K_STORED // 2:] = 0
            Bs[:, :, K_STORED // 2:] = 0
        B = expand(Bs, meta[:, :8, :])
        SFA, SFB = scales((n, 16, nb)), scales((n, 8, nb))
        C = acc(n, -20, 20)
        C[: n // 2] = 0
        A = expand(As, meta)
        sa, sb = model_scales(SFA), model_scales(SFB)
        h, s = K_LOGICAL // 2, sa.shape[2] // 2
        twins = {"dense_on_compressed": (As, Bs, SFA, SFB, C, 1)}
        if atoms == 1:
            twins["dense_atom"] = (A[:, :, :h], B[:, :, :h], sa[:, :, :s], sb[:, :, :s], C, 1)
        else:
            twins["dense_two_atoms"] = (A, B, sa, sb, C, 2)
        yield name, As, B, meta, SFA, SFB, C, twins

    def per_row(n):
        return (rng.integers(0, 16, (n, 16, K_STORED)).astype(np.uint8), rng.integers(0, 16, (n, 8, K_LOGICAL)).astype(np.uint8),
                META_VALID[rng.integers(0, len(META_VALID), (n, 16, CHUNKS))], scales((n, 16, nb)), scales((n, 8, nb)))

    n = n_random // 2
    yield ("per_row_metadata", *per_row(n), acc(n, -30, 40), {})
    n = n_random // 4
    yield ("per_row_metadata_zero_acc", *per_row(n), np.zeros((n, 16, 8), np.uint32), {})
    # large accumulators over the products: the accumulator window sets the grid and the group dots are truncated onto it, the
    # regime where the grouping (16 / 32 / 64 logical columns) and a rounding between the halves show
    n = n_random // 4
    yield ("per_row_acc_window", *per_row(n), acc(n, 10, 40), {})
    # scales near 1 (products in quarters, group dots up to 2^8) under accumulators +-2^30..2^42, whose window puts the grid
    # 2^(E-35) across the group dots: each dot is truncated onto it on its own
    n = n_random // 4
    As, B, meta, _, _ = per_row(n)
    lo, hi = (0x30, 0x41) if sk == "ue4m3" else (126, 129)
    yield ("grid_straddle", As, B, meta, rng.integers(lo, hi, (n, 16, nb)).astype(np.uint8),
           rng.integers(lo, hi, (n, 8, nb)).astype(np.uint8), P._signed_pow2_acc(rng, (n, 16, 8), 30, 42), {})


# --- driver ----------------------------------------------------------------------------------------------------------------------


def _hexrec(acc, a, meta, b, d, sfa, sfb) -> dict:
    return {"acc": f"{int(acc):08x}", "a_stored": "".join(f"{int(x):x}" for x in a), "meta": "".join(f"{int(x):x}" for x in meta),
            "b": "".join(f"{int(x):x}" for x in b), "sfa": "".join(f"{int(x):02x}" for x in sfa),
            "sfb": "".join(f"{int(x):02x}" for x in sfb), "d": f"{int(d):08x}"}


def main(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    instr, arch = args.instruction, args.arch or "sm_120a"
    if args.chain != 1:
        raise SystemExit("--chain is not defined for the sparse rows")
    if args.fake_silicon and SHARED.RESEARCH_RUN_ID:
        raise SystemExit("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    models = candidates(instr)
    if args.expect_model and args.expect_model not in models:
        raise SystemExit(f"--expect-model {args.expect_model!r} is not a candidate of {instr} ({sorted(models)})")
    sass_check = args.sass_check or args.sass_only
    if args.fake_silicon and not P.shutil.which(P.nvcc_bin()):
        so, nvcc = None, "not built (fake silicon)"
    else:
        so, nvcc = P.build(arch, out, "mma_fp4.cu")
    dev = FakeSparse(instr, args.fake_silicon) if args.fake_silicon else (None if args.sass_only else SparseDevice(so))
    # the dense twin: the same build's dense kernels (a fake run: the pinned dense model's words)
    dense = P.FakeProbe(DENSE[instr], "verity") if args.fake_silicon else (None if args.sass_only else P.Probe(so))
    identity = dev.identity() if dev is not None else {"device": None, "gpu_uuid": None, "driver": None, "compute_capability": None,
                                                       "backend": "none (--sass-only)"}
    identity.update(nvcc=nvcc, arch=arch)
    if args.expect_uuid and dev is not None and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        msg = f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement"
        (out / "probe_results.json").write_text(json.dumps({"identity": identity, "failures": [msg]}, indent=1))
        print("FAILURES", [msg], flush=True)
        return 2
    primary = args.expect_model
    results: dict = {"schema": "pouw-rtxpro/tc-probe-fp4-sparse/v1", "instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr],
                     "identity": identity, "model": "sparse.candidates: " + ", ".join(sorted(models)),
                     "model_source": str(Path(__file__).resolve()) + " + verity.ml.circuits.silicon.models", "primary_model": primary,
                     "fake_silicon": args.fake_silicon, "corrupt_word": bool(args.corrupt_word)}
    print("device", json.dumps(identity), flush=True)
    failures: list[str] = []

    def finish() -> int:
        results["seconds"] = round(time.time() - t_all, 1)
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        return 1 if failures else 0

    if so is not None and sass_check:
        sass = P.sass_check(so)
        results["sass"] = sass
        for kern, rec in sass.items():
            if rec["mma_count"] != 1:
                failures.append(f"sass: {kern} has {rec['mma_count']} MMA instructions")
        mine = [rec for kern, rec in sass.items() if KERNEL[instr] in kern]
        if len(mine) != 1 or mine[0]["opcodes"] != [SASS_OPCODE[instr]]:
            failures.append(f"sass: the {instr} kernel {KERNEL[instr]} must hold exactly one {SASS_OPCODE[instr]}: {mine}")
        print("sass", json.dumps({k: v for k, v in sass.items() if "sparse" in k}), flush=True)
    if args.sass_only or failures:
        print("FAILURES" if failures else "SASS GATE PASSED", failures, flush=True)
        return finish()

    results["clocks_start"] = dev.clocks(identity.get("gpu_uuid"))
    try:
        results["layout_check"] = layout_check(dev, instr, out)
    except RuntimeError as e:
        failures.append(str(e))
        print("FAILURES", failures, flush=True)
        return finish()
    print("layout_check", instr, results["layout_check"], flush=True)

    fams: dict = {}
    if args.sweep:
        only = set(args.families.split(",")) if args.families else None
        rng = np.random.default_rng(args.seed)
        capture = {"instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr], "identity": identity, "records": [],
                   "note": "acc/d FP32 words (hex); a_stored: the 64 stored e2m1 codes of the row; meta: its 16 chunk nibbles; b: the "
                           "column's 128 logical e2m1 codes; sfa/sfb: the scale bytes (hex), as measured on the device in identity"}
        corrupted = False
        pool = mp.Pool(args.procs, initializer=_init, initargs=(instr,)) if args.procs > 1 else None
        try:
            for name, As, B, META, SFA, SFB, C, twins in families(rng, args.n_random, instr):
                if only is not None and name not in only:
                    continue
                t0 = time.time()
                D = dev.tiles(instr, As, B, META, SFA, SFB, C)
                twin_words = {label: dense.tiles(DENSE[instr], a, bt, c, sa, sb, steps=steps)
                              for label, (a, bt, sa, sb, c, steps) in twins.items()}
                if args.corrupt_word and not corrupted and name not in P.RECORD_ONLY_FAMILIES:
                    D = D.copy()
                    D[0, 0, 0] ^= 1
                    corrupted = True
                t_gpu = time.time() - t0
                t0 = time.time()
                words, errs = evaluate(pool, instr, As, B, META, SFA, SFB, C, args.procs)
                fam = {"tiles": int(As.shape[0]), "elements": int(As.shape[0]) * 128, "record_only": name in P.RECORD_ONLY_FAMILIES,
                       "metadata": "per row and chunk" if name.startswith("per_row") else "per tile (shared by its rows)",
                       "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)),
                       "model_domain_errors": {m: int(e.sum()) for m, e in errs.items()}, "gpu_seconds": round(t_gpu, 2),
                       "model_seconds": round(time.time() - t0, 1), "mismatches_vs_model": {}, "first_mismatches": {}}
                for m, w in words.items():
                    mism = (w != D) | errs[m]
                    fam["mismatches_vs_model"][m] = int(mism.sum())
                    fam["first_mismatches"][m] = [
                        {**_hexrec(C[t, i, j], As[t, i], META[t, i], B[t, j], D[t, i, j], SFA[t, i], SFB[t, j]),
                         "model": "error" if errs[m][t, i, j] else f"{int(w[t, i, j]):08x}"}
                        for t, i, j in list(zip(*np.nonzero(mism)))[:20]]
                fam["dense_twins"] = {}
                for label, w in twin_words.items():
                    diff = w != D
                    fam["dense_twins"][label] = {
                        "differing_words": int(diff.sum()),
                        "first": [{**_hexrec(C[t, i, j], As[t, i], META[t, i], B[t, j], D[t, i, j], SFA[t, i], SFB[t, j]),
                                   "dense_word": f"{int(w[t, i, j]):08x}"} for t, i, j in list(zip(*np.nonzero(diff)))[:10]]}
                np.savez_compressed(out / f"tiles_{name}.npz", As=As, B=B, META=META, SFA=SFA, SFB=SFB, C=C, D=D,
                                    **{f"model_{m}": w for m, w in words.items()}, **{f"dense_{k}": w for k, w in twin_words.items()})
                for f in sorted(rng.choice(As.shape[0] * 128, size=min(args.golden_per_family, As.shape[0] * 128), replace=False).tolist()):
                    t, i, j = f // 128, (f % 128) // 8, f % 8
                    capture["records"].append({**_hexrec(C[t, i, j], As[t, i], META[t, i], B[t, j], D[t, i, j], SFA[t, i], SFB[t, j]),
                                               "family": name})
                fams[name] = fam
                print(name, json.dumps({k: v for k, v in fam.items() if k != "first_mismatches"}), flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        gated = {k: f for k, f in fams.items() if not f["record_only"]}
        totals = {m: sum(f["mismatches_vs_model"][m] for f in gated.values()) for m in models}
        results["sweep"] = {"seed": args.seed, "n_random": args.n_random, "families": fams,
                            "total_elements": sum(f["elements"] for f in fams.values()),
                            "gated_elements": sum(f["elements"] for f in gated.values()),
                            "record_only_families": sorted(k for k, f in fams.items() if f["record_only"]),
                            "total_mismatches_vs_model": totals,
                            "record_only_mismatches_vs_model": {m: sum(f["mismatches_vs_model"][m] for f in fams.values() if f["record_only"])
                                                                for m in models}}
        results["candidate_finding"] = {"agreeing": sorted(m for m, c in totals.items() if c == 0), "mismatches": totals}
        labels = sorted({k for f in gated.values() for k in f["dense_twins"]})
        results["dense_twin_finding"] = {
            k: {"elements": sum(f["elements"] for f in gated.values() if k in f["dense_twins"]),
                "differing_words": sum(f["dense_twins"][k]["differing_words"] for f in gated.values() if k in f["dense_twins"]),
                "families": sorted(n for n, f in gated.items() if k in f["dense_twins"])} for k in labels}
        results["dense_twin_finding"]["note"] = (
            "hardware against hardware: the sparse word against the dense mma.sync " + DENSE[instr] + " word on the twin operands "
            "(dense_on_compressed: the stored A and the selected B; dense_atom: the k64 atom in logical columns 0..63; "
            "dense_two_atoms: the two atoms as two dense MMAs)")
        (out / "capture_sample.json").write_text(json.dumps(capture, separators=(",", ":")))
        if primary is not None and totals[primary]:
            failures.append(f"sweep mismatches vs {primary}={totals[primary]} (record-only families excluded)")
        print("sweep TOTAL", results["sweep"]["total_elements"], "elements,", totals, flush=True)
        print("candidates reproducing every word:", results["candidate_finding"]["agreeing"], flush=True)
        print("sparse vs dense hardware words:", json.dumps({k: v for k, v in results["dense_twin_finding"].items() if k != "note"}),
              flush=True)

    results["clocks_end"] = dev.clocks(identity.get("gpu_uuid"))
    rc = finish()
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        sweep = results.get("sweep", {})
        agree = {m: c == 0 for m, c in sweep.get("total_mismatches_vs_model", {}).items()}
        first = [{"model": m, **r} for f in fams.values() for m, lst in f["first_mismatches"].items() for r in lst][:20]
        finding = results.get("candidate_finding")
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": INSTRUCTION_ID[instr],
            "ptx": PTX[instr], "arch": arch, "sku": identity["device"], "compute_capability": identity.get("compute_capability"),
            "driver": identity.get("driver"), "gpu_uuid": identity.get("gpu_uuid"), "nvcc": nvcc,
            "cuda_visible_devices": identity.get("cuda_visible_devices"),
            "sass_mma_counts": {k: v["mma_count"] for k, v in results.get("sass", {}).items()},
            "sass_opcodes": {k: sorted(set(v["opcodes"])) for k, v in results.get("sass", {}).items()},
            "families": {k: {"elements": f["elements"], "mismatches": f["mismatches_vs_model"], "record_only": f["record_only"],
                             "metadata": f["metadata"], "domain_errors": f["model_domain_errors"]} for k, f in fams.items()},
            "counts": {"elements": sweep.get("total_elements", 0), "gated_elements": sweep.get("gated_elements", 0),
                       "mismatches": sweep.get("total_mismatches_vs_model", {}),
                       "record_only_mismatches": sweep.get("record_only_mismatches_vs_model", {})},
            "model": results["model"], "primary_model": primary, "expected_model": primary,
            "model_agrees": agree.get(primary) if primary else None, "agreement": agree, "candidate_finding": finding,
            "dense_twin_finding": results.get("dense_twin_finding"),
            "first_mismatches": first, "seed": args.seed, "n": args.n_random, "layout_check": results.get("layout_check"),
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"),
                       "note": "recorded, not locked: a capture's words do not depend on the clock"},
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/sparse.py", "instruction": INSTRUCTION_ID[instr], "arch": arch,
                                     "device": identity["device"], "seed": args.seed, "n_random": args.n_random},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 sparse.py: UUID, SASS and exact-integer layout gates, then the candidates",
                           "detail": "; ".join(failures) or (f"every hardware word reproduced by {primary}" if primary else
                                                             f"capture recorded; candidates reproducing every word: "
                                                             f"{(finding or {}).get('agreeing')}"),
                           "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": (["capture_sample.json"] if fams else []) + sorted(p.name for p in out.glob("tiles_*.npz")),
        }
        (out / "result.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(out / "result.json.tmp", out / "result.json")
    print("FAILURES" if failures else ("ALL HARDWARE WORDS REPRODUCED" if primary else "CAPTURE RECORDED"), failures,
          f"{results['seconds']}s", flush=True)
    return rc
