"""dither-a's -zy records pinned to h1zy's: `benchmarks/hash/dither/dither_a.py`'s `_dithera_epilogue` at ZL = 2,
one record a `dither_a.UNIT` unit, through `dithera_mm` and `dithera_mm_tma` as `Side.gemm` launches them. A -zy row commits
these records, and a unit opens only a record that parses as `h1zy.RECORD` into its exact z and `h1zy.dequant`'s y, as
integrations/vllm's `test_int8_zy_h1zy` pins `verity_int8::scaled_mm_zy`'s.

On CUDA (skipped without it) each call runs `Side.gemm` and then `Side.ycommit`, as `Side.call` does, on a stand-in window
of T slots, the first m live (its step's m), into 0xA5-poisoned y and records, at the config dither-a picks or a forced
one:

* every record parses as `h1zy.RECORD`, and as dither-a's own `zy_split`, into z: ``int8_exact.int_mm``'s block on a live
  slot, 0 on an empty one;
* every cell of the served y (all M rows) is `h1zy.dequant` of its z at its row's s_a and its column's s_w, and every cell
  of every record's y is `h1zy.dequant` of the record's z at the slot's s_a: an empty slot is zero codes under s_a = +0,
  so its y is dq(0, +0, s_w), +0 under a finite positive s_w (the served case), 0x8000 under a negative one and 0x7FFF
  under an infinite or NaN one (CA's : Served's DequantOf on every cell);
* the records are `scaled_mm_zy`'s byte for byte on the same inputs (its rows the m live ones);
* `ycommit`'s y records (the served y's) are the records' y halves byte for byte, so the two y roots agree;
* the cells where the y changed from dither-a's earlier "0 past m" (0x0000 on every empty slot) are exactly the empty
  slots of columns whose dq(0, +0, s_w) isn't +0: under served scales none, so the fix leaves every served byte as it was.

The edge call is `test_int8_zy_h1zy`'s, rebuilt from the same design: K = 14,336, every row scale of `SA` over 36 design
rows, every column scale of `SW` over 6 column kinds, and 32 empty slots.
"""
from __future__ import annotations

import functools
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops as OPS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DITHER = Path(__file__).resolve().parents[2] / "hash" / "dither"
UR, UC = E.UNIT
M32 = (1 << 32) - 1
K_EDGE = 14336


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


NEG = 0x80000000
ONE, TWO, HALF, THREE, P2M10 = f32(1.0), f32(2.0), f32(0.5), f32(3.0), f32(2.0 ** -10)
INF, NINF, QNAN, SNAN, NQNAN, NAN1S = 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00000, 0x7FFFFFFF
MAX, HUGE, TINY = 0x7F7FFFFF, f32(2.0 ** 100), f32(2.0 ** -100)
SUB_MIN, SUB_MID, SUB_MAX, MIN_NORMAL = 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000
SA = [ONE, NEG | ONE, 0, NEG, P2M10, HALF, f32(0.0123 / 127), NEG | f32(0.0123 / 127), SUB_MIN, SUB_MID, SUB_MAX,
      NEG | SUB_MIN, MIN_NORMAL, MAX, NEG | MAX, HUGE, TINY, NEG | TINY, INF, NINF, QNAN, SNAN, NQNAN, NAN1S]
SW = [ONE, NEG | ONE, 0, NEG, P2M10, HALF, TWO, THREE, f32(1 / 3), f32(1 / 127), f32(0.0042 / 127),
      NEG | f32(0.0042 / 127), SUB_MIN, SUB_MID, SUB_MAX, NEG | SUB_MID, MIN_NORMAL, MAX, NEG | MAX, HUGE, NEG | HUGE, TINY,
      INF, NINF, QNAN, SNAN, NQNAN, NEG | NAN1S, f32(0.0003 / 127), f32(0.07 / 127), f32(1e-20), f32(1e20)]
ZFULLC = 127 * 127 * (K_EDGE - 1) + 63
TARGETS = [0, 1, -1, 3, -7, 127, -16129, 257, 259, -257, -259,
           (1 << 23) + 0x8000, (1 << 23) + 0x18000, -((1 << 23) + 0x18000),
           (1 << 24) + 1, (1 << 24) + 3, -((1 << 24) + 1), -((1 << 24) + 3),
           (1 << 24) + (1 << 16) + 1, (1 << 24) + (1 << 17) + (1 << 16),
           (1 << 25) + 2, (1 << 25) + 6, (1 << 27) + 8, (1 << 27) + 24, (1 << 27) + 5, -((1 << 27) + 13),
           200_000_001, -199_999_999, ZFULLC, -ZFULLC]
LLAMA = {"qkv": (4096, 6144), "o": (4096, 4096), "gate_up": (4096, 28672), "down": (14336, 4096)}


@functools.cache
def dither_a():
    """`benchmarks/hash/dither/dither_a.py`, imported as its scripts import it (its directory on sys.path); it needs
    Triton, so only the CUDA tests call this."""
    if str(DITHER) not in sys.path:
        sys.path.insert(0, str(DITHER))
    import dither_a as DA
    return DA


# -- the edge call (test_int8_zy_h1zy's) ---------------------------------------------------------------------------------

def row_for(t: int, k: int) -> np.ndarray:
    """K int8 codes whose dot with a "fullc" column (k - 1 codes 127, then 1) is t."""
    r = (t + 63) % 127 - 63
    s = (t - r) // 127
    n, rem = divmod(abs(s), 127)
    assert n + (rem > 0) <= k - 1, t
    q = np.zeros(k, np.int8)
    q[:n] = 127 if s >= 0 else -127
    if rem:
        q[n] = rem if s >= 0 else -rem
    q[k - 1] = r
    return q


def _flipped(k: int, d: int) -> np.ndarray:
    q = np.full(k, 127, np.int8)
    q[[5 + 1000 * i for i in range(d)]] = -127
    return q


def _designs(k: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3018)
    return ([row_for(t, k) for t in TARGETS] + [_flipped(k, 0), -_flipped(k, 0), _flipped(k, 1), _flipped(k, 3),
            rng.choice(np.array([-127, 127], np.int8), k), rng.integers(-127, 128, k).astype(np.int8)])


def _columns(k: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3053)
    fullc = np.full(k, 127, np.int8)
    fullc[-1] = 1
    e0 = np.zeros(k, np.int8)
    e0[0] = 1
    return [fullc, np.full(k, 127, np.int8), np.full(k, -127, np.int8), np.zeros(k, np.int8), e0,
            rng.integers(-127, 128, k).astype(np.int8)]


@functools.cache
def edge_call() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(A [M, K] int8, s_a [M], B [N, K] int8, s_w [N], T): each design row under each of `SA`, each column kind under each
    of `SW`, over M slots and 32 empty ones; scales as f32 encodings (uint32)."""
    designs, cols = _designs(K_EDGE), _columns(K_EDGE)
    a = np.stack([d for _s in SA for d in designs])
    sa = np.array([s for s in SA for _d in designs], np.uint32)
    b = np.stack([c for c in cols for _s in SW])
    sw = np.array([s for _c in cols for s in SW], np.uint32)
    return a, sa, b, sw, -(-a.shape[0] // UR) * UR + UR


# -- the Definition and the records --------------------------------------------------------------------------------------

def rn_bf16(f: np.ndarray) -> np.ndarray:
    b = f.view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(f), np.uint16(0x7FFF), r)


def oracle(z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """int8_exact's RN_bf16(RN32(RN32(RN32(z) s_a) s_w)) step by step in numpy float32, on encodings."""
    with np.errstate(all="ignore"):
        zf = np.asarray(z, np.uint32).view(np.int32).astype(np.float32)
        return rn_bf16((zf * np.asarray(sa, np.uint32).view(np.float32)) * np.asarray(sw, np.uint32).view(np.float32))


def expected(z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """y [rows, N] bf16 encodings: `h1zy.dequant` of each cell's (z, its row's s_a, its column's s_w), once a distinct
    triple."""
    tri = np.stack([z, np.broadcast_to(sa[:, None], z.shape), np.broadcast_to(sw[None, :], z.shape)], axis=-1)
    uniq, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    y = np.array([Z.dequant(int(p), int(q), int(r)) for p, q, r in uniq], np.uint16)
    return y[inv.reshape(-1)].reshape(z.shape)


def record_fields(zy: np.ndarray) -> dict[str, np.ndarray]:
    """Records [units, 6 UR UC] uint8 read as `h1zy.RECORD(UR, UC)`: the bytes' bits, little-endian and LSB first, are its
    leaves, field after field, each an array of words whose leaves are their bits, LSB first. {field: [units, words]}."""
    t = Z.RECORD(UR, UC)
    assert zy.shape[1] * 8 == t.leaves == 48 * UR * UC
    out: dict[str, list[np.ndarray]] = {name: [] for name, _ft in t.fields}
    for lo in range(0, zy.shape[0], 64):
        leaves, at = np.unpackbits(zy[lo:lo + 64], axis=1, bitorder="little"), 0
        for name, ft in t.fields:
            n, w = ft.n, ft.elem.n
            assert ft.elem.elem.leaves == 1
            words = leaves[:, at:at + n * w].reshape(-1, n, w).astype(np.uint64)
            out[name].append((words << np.arange(w, dtype=np.uint64)).sum(axis=2))
            at += n * w
        assert at == t.leaves
    return {k: np.concatenate(v) for k, v in out.items()}


def grid(cells: np.ndarray, rows: int, n: int) -> np.ndarray:
    """Units' cells [units, UR UC] as [rows, n]: unit u is row block u // (n / UC), column block u % (n / UC)."""
    return cells.reshape(rows // UR, n // UC, UR, UC).transpose(0, 2, 1, 3).reshape(rows, n)


def records_of(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[units, 6 UR UC] uint8: z [rows, n] u32 then y [rows, n] u16 a unit, cells row-major, little-endian."""
    rows, n = z.shape

    def cells(v: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(v.reshape(rows // UR, UR, n // UC, UC).transpose(0, 2, 1, 3).reshape(-1, UR * UC))
    return np.concatenate([cells(z.astype("<u4")).view(np.uint8), cells(y.astype("<u2")).view(np.uint8)], axis=1)


def _diff(what: str, got: np.ndarray, want: np.ndarray, z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> str:
    bad = np.argwhere(got != want)
    lines = [f"{what}: {len(bad)} of {got.size} cells differ from h1zy.dequant"]
    for i, j in bad[:12]:
        zz, a, b = z[i, j], sa[i], sw[j]
        o = int(oracle(np.array([zz]), np.array([a]), np.array([b]))[0])
        lines.append(f"  cell ({i}, {j}): z {int(np.uint32(zz).view(np.int32))} (0x{int(zz):08x}), s_a 0x{int(a):08x}, "
                     f"s_w 0x{int(b):08x}: dither-a y 0x{int(got[i, j]):04x}, h1zy.dequant 0x{int(want[i, j]):04x}, "
                     f"numpy oracle 0x{o:04x}")
    return "\n".join(lines)


def f32_tensor(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.astype(np.uint32).view(np.int32).copy()).view(torch.float32)


def f32_bits(t: torch.Tensor) -> np.ndarray:
    return t.reshape(-1).cpu().numpy().view(np.uint32)


# -- the calls -----------------------------------------------------------------------------------------------------------

def side(DA, rows: int, m: int, n: int, work=None) -> SimpleNamespace:
    """What `Side.gemm` and `Side.ycommit` read of a -zy Side over a window of `rows` slots at step m, linear 0 of N = n:
    its zy and y records (0xA5), the window's step words S = (parity, m, pad, flags = 0) and slots, ZL = 2, `UNIT`, a
    split-K workspace."""
    units = (rows // UR) * (n // UC)
    zy = torch.full((units * DA.zy_bytes(DA.UNIT),), 0xA5, dtype=torch.uint8, device="cuda")
    ysv = torch.full((units * DA.y_bytes(DA.UNIT),), 0xA5, dtype=torch.uint8, device="cuda")
    win = SimpleNamespace(t=rows, S=torch.tensor([0, m, 0, 0], dtype=torch.int32, device="cuda"), ydump=None)
    return SimpleNamespace(win=win, zy=[zy], ysv=[ysv], unit_of=[DA.UNIT], zl=DA.ZMODES["zy"], arch=DA.EL.arch(), cfgs={},
                           work=work or DA.EL.Workspace("cuda"))


def dither_call(DA, sd: SimpleNamespace, xq: torch.Tensor, sa: torch.Tensor, w: torch.Tensor,
                sw: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One call through `Side.gemm` and `Side.ycommit` into a 0xA5-poisoned y and sd's records: (y [M, N] u16, records
    [units, 6 UR UC] u8, ycommit's y records [units, 2 UR UC] u8)."""
    m, n = xq.shape[0], w.shape[0]
    y = torch.full((m, 2 * n), 0xA5, dtype=torch.uint8, device="cuda").view(torch.bfloat16)
    sd.zy[0].fill_(0xA5)
    sd.ysv[0].fill_(0xA5)
    DA.Side.gemm(sd, 0, xq, sa, w, sw, y)
    DA.Side.ycommit(sd, 0, y, sw)
    torch.cuda.synchronize()
    zy = sd.zy[0].cpu().numpy().reshape(-1, DA.zy_bytes(DA.UNIT))
    ys = sd.ysv[0].cpu().numpy().reshape(-1, DA.y_bytes(DA.UNIT))
    return y.view(torch.int16).cpu().numpy().view(np.uint16), zy, ys


def check_call(DA, y: np.ndarray, zy: np.ndarray, ys: np.ndarray, zfull: np.ndarray, sa: np.ndarray, sw: np.ndarray,
               rows: int, m: int, ref: np.ndarray) -> dict:
    """Hold one dither-a call to h1zy: served y [M, N], records zy and ycommit's y records ys of the call whose exact
    product is ``zfull`` [M, N] (u32), over ``rows`` slots with m live; ``ref`` `scaled_mm_zy`'s records over the same live
    rows. Returns the empty-slot cells (and record bytes) the fix moved from dither-a's earlier 0x0000, by h1zy's word."""
    n = zfull.shape[1]
    assert zy.shape == (E.zy_units(m, n, rows), E.ZY_BYTES) == ref.shape and DA.zy_bytes(DA.UNIT) == E.ZY_BYTES
    assert ys.shape == (zy.shape[0], 2 * UR * UC) and DA.y_bytes(DA.UNIT) == 2 * UR * UC
    f = record_fields(zy)
    zr, yr = grid(f["z"].astype(np.uint32), rows, n), grid(f["y"].astype(np.uint16), rows, n)
    dz, dy = DA.zy_split(torch.from_numpy(zy.reshape(-1)), rows, n, DA.UNIT)
    assert np.array_equal(dz.numpy().view(np.uint32), zr) and np.array_equal(dy.view(torch.int16).numpy().view(np.uint16), yr), \
        "dither_a.zy_split reads the records other than h1zy.RECORD"
    z = np.zeros((rows, n), np.uint32)
    z[:m] = zfull[:m]
    assert np.array_equal(zr, z), f"{int((zr != z).sum())} record z words differ from int_mm's block (0 on empty slots)"
    sas = np.zeros(rows, np.uint32)
    sas[:m] = sa[:m]
    want, served = expected(z, sas, sw), expected(zfull, sa, sw)
    assert np.array_equal(y, served), _diff("served y", y, served, zfull, sa, sw)
    live = np.arange(rows) < m
    assert np.array_equal(yr[live], want[live]), _diff("live slots' record y", yr[live], want[live], z[live], sas[live], sw)
    assert np.array_equal(yr[~live], want[~live]), _diff("empty slots' record y", yr[~live], want[~live], z[~live],
                                                         sas[~live], sw)
    assert np.array_equal(zy, records_of(z, want)), "record bytes other than the words parsed"
    assert np.array_equal(ref, zy), f"{int((ref != zy).sum())} record bytes differ from scaled_mm_zy's"
    assert np.array_equal(ys, zy[:, 4 * UR * UC:]), \
        f"{int((ys != zy[:, 4 * UR * UC:]).sum())} bytes of ycommit's y records differ from the records' y halves (RT4-INT-11)"
    moved = (~live)[:, None] & (want != 0)
    cols = np.array([Z.dequant(0, 0, int(s)) != 0 for s in sw])
    assert np.array_equal(moved, (~live)[:, None] & cols[None, :])
    before = records_of(z, np.where((~live)[:, None], 0, want))
    bytes_moved = int((zy != before).sum())
    assert bytes_moved == int(sum((((want[moved] >> s) & 0xFF) != 0).sum() for s in (0, 8)))
    words: dict[str, int] = {}
    for v in want[moved]:
        words[f"0x{int(v):04x}"] = words.get(f"0x{int(v):04x}", 0) + 1
    return {"cells": int(zy.size // 6 + y.size), "empty slots": int((~live).sum()),
            "empty-slot y not +0 (moved from 0x0000)": int(moved.sum()), "by word": words,
            "record bytes moved from before": bytes_moved}


def reference(xq: torch.Tensor, sa: torch.Tensor, w: torch.Tensor, sw: torch.Tensor, rows: int, m: int) -> np.ndarray:
    """`verity_int8::scaled_mm_zy`'s records over `rows` slots for the call's m live rows."""
    _y, zy = OPS.scaled_mm_zy(xq[:m].contiguous(), sa[:m].contiguous(), w, sw, rows)
    torch.cuda.synchronize()
    return zy.cpu().numpy()


@functools.cache
def edge_tensors():
    a, sa, b, sw, rows = edge_call()
    xq, w = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    z = E.int_mm(xq, w).cpu().numpy()
    exact = (a.astype(np.float64) @ b.astype(np.float64).T).astype(np.int64)
    assert np.array_equal(z.astype(np.int64), exact)
    return xq, f32_tensor(sa).cuda(), w, f32_tensor(sw).cuda(), rows, z.view(np.uint32)


def _edge(DA, sd: SimpleNamespace) -> tuple[dict, np.ndarray]:
    xq, sa, w, sw, rows, z = edge_tensors()
    y, zy, ys = dither_call(DA, sd, xq, sa, w, sw)
    ref = reference(xq, sa, w, sw, rows, xq.shape[0])
    return check_call(DA, y, zy, ys, z, f32_bits(sa), f32_bits(sw), rows, xq.shape[0], ref), zy


# -- CPU -----------------------------------------------------------------------------------------------------------------

def test_the_byte_layout_compared_is_int8_exacts_and_parses_as_h1zy_record():
    """`records_of`, the layout the CUDA tests compare dither-a's bytes in, is `int8_exact.zy_records` ('s records),
    and `record_fields` reads it back as `h1zy.RECORD` into the same z and y: random words over 3 x 5 units."""
    rng = np.random.default_rng(2930)
    rows, n = 3 * UR, 5 * UC
    z = rng.integers(0, 1 << 32, (rows, n), dtype=np.uint64).astype(np.uint32)
    y = rng.integers(0, 1 << 16, (rows, n), dtype=np.uint64).astype(np.uint16)
    rec = records_of(z, y)
    ref = E.zy_records(torch.from_numpy(z.view(np.int32)), torch.from_numpy(y.view(np.int16)).view(torch.bfloat16))
    assert np.array_equal(rec, ref.numpy())
    f = record_fields(rec)
    assert np.array_equal(grid(f["z"].astype(np.uint32), rows, n), z)
    assert np.array_equal(grid(f["y"].astype(np.uint16), rows, n), y)


# -- CUDA ----------------------------------------------------------------------------------------------------------------

@cuda
@pytest.mark.slow
def test_cuda_dither_records_are_h1zys_on_the_edge_call(record_property):
    """The edge call at the config dither-a picks: records parse as `h1zy.RECORD` into int_mm's z, every record y and the
    served y are `h1zy.dequant`'s, the records are `scaled_mm_zy`'s and ycommit's y records their y halves, byte for byte;
    the fix moved exactly the empty slots of the columns whose dq(0, +0, s_w) isn't +0."""
    DA = dither_a()
    assert DA.UNIT == E.UNIT and DA.ZMODES["zy"] == 2
    xq, _sa, w, _sw, rows, _z = edge_tensors()
    sd = side(DA, rows, xq.shape[0], w.shape[0])
    got, _zy = _edge(DA, sd)
    record_property("config", sd.cfgs)
    record_property("result", got)
    kinds = w.shape[0] // len(SW)
    assert got["empty-slot y not +0 (moved from 0x0000)"] == (rows - xq.shape[0]) * kinds * sum(Z.dequant(0, 0, s) != 0
                                                                                              for s in SW)


@cuda
@pytest.mark.slow
def test_cuda_dither_empty_slots_are_h1zys():
    """h1zy's rule on every cell, empty slots included: on the edge call an empty slot's record y, and ycommit's, is
    `h1zy.dequant(0, +0, s_w)` of its column, among them +0, 0x8000 (a negative s_w) and 0x7FFF (an infinite or NaN one)."""
    DA = dither_a()
    xq, sa, w, sw, rows, _z = edge_tensors()
    sd = side(DA, rows, xq.shape[0], w.shape[0])
    _y, zy, ys = dither_call(DA, sd, xq, sa, w, sw)
    n, m = w.shape[0], xq.shape[0]
    yr = grid(record_fields(zy)["y"].astype(np.uint16), rows, n)
    yc = grid(ys.view("<u2"), rows, n)
    pad = np.array([Z.dequant(0, 0, int(s)) for s in f32_bits(sw)], np.uint16)
    assert {0x0000, 0x8000, 0x7FFF} <= set(pad.tolist())
    for what, got in (("record y", yr), ("ycommit's y", yc)):
        assert np.array_equal(got[m:], np.broadcast_to(pad, (rows - m, n))), \
            f"{what}: {int((got[m:] != pad).sum())} empty-slot cells differ from h1zy.dequant(0, +0, s_w)"


CONFIGS = ["16x64x128:w4:s4:k1", "32x32x256:w4:s3:k2", "16x32x256:w4:s3:k4", "64x64x128:w4:s3:k3", "128x128x64:w4:s3:k1",
           "128x64x128:w4:s3:k1:t1"]


@cuda
@pytest.mark.slow
def test_cuda_every_config_writes_h1zys_records_on_the_edge_call(monkeypatch, record_property):
    """`Side.gemm` at each forced config (pointer tiles narrower and wider than a unit, split-K 2, 3 and 4, a row tile past
    T, TMA; warp-specialized TMA on sm_90 and sm_100), twice through one workspace, zero again after: the edge call's
    records and served y as at dither-a's own config."""
    DA = dither_a()
    xq, _sa, w, _sw, rows, _z = edge_tensors()
    cfgs = CONFIGS + (["64x64x128:w4:s3:k1:t2"] if DA.EL.arch() in ("sm_90", "sm_100") else [])
    sd = side(DA, rows, xq.shape[0], w.shape[0])
    done = {}
    for text in cfgs:
        cfg = DA.EL.Cfg.parse(text)
        monkeypatch.setattr(DA.EL, "pick", lambda *_args, cfg=cfg: cfg)
        for _ in range(2):
            got, _zy = _edge(DA, sd)
        assert not sd.work.ws.any() and not sd.work.cnt.any(), text
        done[text] = got["empty-slot y not +0 (moved from 0x0000)"]
    record_property("configs", done)


#: (linear, M, m, T, scales): calls at Llama-3.1-8B's linears; m < M is a padded batch (rows m to M served, not live);
#: "wide" scales are random encodings over most of f32's exponents and both signs
RANDOM = [("qkv", 200, 200, 224, "served"), ("o", 37, 37, 64, "served"), ("gate_up", 1, 1, 64, "served"),
          ("gate_up", 17, 17, 32, "served"), ("down", 64, 64, 64, "served"), ("down", 5, 5, 64, "served"),
          ("qkv", 48, 41, 64, "served"), ("down", 32, 29, 32, "served"), ("o", 256, 250, 256, "served"),
          ("down", 33, 33, 64, "wide"), ("qkv", 32, 32, 64, "wide"), ("gate_up", 16, 9, 32, "wide")]


@cuda
@pytest.mark.slow
@pytest.mark.parametrize("linear,M,m,rows,scales", RANDOM)
def test_cuda_random_calls_at_llamas_linears_are_h1zys(linear, M, m, rows, scales, record_property):
    """Random calls at dither-a's config: records parse as `h1zy.RECORD` into int_mm's z, every record y and the served y
    are `h1zy.dequant`'s, the records are `scaled_mm_zy`'s and ycommit's y records their y halves, byte for byte; under
    served scales no record byte moved from before the fix."""
    DA = dither_a()
    k, n = LLAMA[linear]
    g = torch.Generator("cuda").manual_seed(M * 131 + m + k)
    xq, sa = OPS.row_quant((torch.randn(M, k, generator=g, device="cuda") * 3).bfloat16())
    w = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8, device="cuda")
    sw = torch.rand(n, generator=g, device="cuda") * 0.01 + 1e-4
    sa = sa.reshape(-1).contiguous()
    if scales == "wide":
        r = np.random.default_rng(M * 7 + n)
        sa, sw = (f32_tensor((r.integers(0, 2, c) << 31) | (r.integers(0x30, 0xD0, c) << 23) | r.integers(0, 1 << 23, c)).cuda()
                  for c in (M, n))
    sd = side(DA, rows, m, n)
    y, zy, ys = dither_call(DA, sd, xq, sa, w, sw)
    z = E.int_mm(xq, w).cpu().numpy().view(np.uint32)
    got = check_call(DA, y, zy, ys, z, f32_bits(sa), f32_bits(sw), rows, m, reference(xq, sa, w, sw, rows, m))
    record_property("config", sd.cfgs)
    record_property("result", got)
    if scales == "served":
        assert got["empty-slot y not +0 (moved from 0x0000)"] == 0 and got["record bytes moved from before"] == 0
