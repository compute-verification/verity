"""Is an FP8 or FP4 MMA chain exact integer arithmetic on sm_120? The sweep, the host's recheck and the verdicts.

  python fp_exact.py configs [--quick]                 # the sweep, one line a config (fp_exact.cu's input)
  python fp_exact.py run BINARY --out DIR [--quick]    # one GPU: the sweep, the no-write control, the recheck
  python fp_exact.py summarize DIR                     # the verdicts from a run's outputs
  python fp_exact.py limits DIR/verdicts.json          # the K limits per format, grid and C0, as a table

`fp_exact.cu` chains K / k_step MMAs of one format into one FP32 accumulator from the word C0, with operands from one
of GRIDS and per-block scales from one of SCALES (NVFP4's UE4M3 odd integers, MXFP4's UE8M0 powers 2^k, and two draws
that straddle the 27-bit anchor), and counts every output whose word isn't the exact sum's. Here the host regenerates the first tiles of every
config from the same counter hash, recomputes their sums in int64 and judges their words itself; a config whose device
reference disagrees with the host's is refused, and so is every config of the no-write control. For K <= REPLAY_K the
host also replays the first tile's identity-order chain through `verity.ml.circuits.silicon`'s model of the firing,
so an inexact word is shown to be the modelled truncation and not a fault.

A partial sum is *representable* when FP32 holds it exactly (its odd part under 2^24). The question per format is
whether every output whose partial sums at the MMA boundaries were all representable comes out as the exact sum's
word (`bad_rep` = 0), whatever the accumulation order (`order_diff` = 0), with zero as +0 (`negzero` = 0).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the GPU side (configs, run) needs only the standard library
    np = None

FMTS = ("e4m3", "e4m3_kind", "mxf4", "mxf4nv", "nvf4", "sp_mxf4", "sp_nvf4")
KSTEP = {"e4m3": 32, "e4m3_kind": 32, "mxf4": 64, "mxf4nv": 64, "nvf4": 64, "sp_mxf4": 128, "sp_nvf4": 128}
MACS = {"e4m3": 4096, "e4m3_kind": 4096, "mxf4": 8192, "mxf4nv": 8192, "nvf4": 8192, "sp_mxf4": 16384, "sp_nvf4": 16384}
DISTS = ("U", "MAX", "MAXSIGN", "RAMP", "CANCEL", "EDGE")
# Per-block scales, mirroring fp_exact.cu's Scale enum. UE4M3 (nvf4) carries odd integers; UE8M0 (mxf4) carries 2^k.
SCALES = ("unit", "lo", "hi", "mix", "bind", "bind26")
NORD = 3
SPARSE = {"sp_mxf4", "sp_nvf4"}
PAT0 = (0, 0, 0, 1, 1, 2)
PAT1 = (1, 2, 3, 2, 3, 3)
REC = np.dtype([("d", "<u4", (NORD,)), ("flags", "<u4"), ("ref", "<i8")]) if np else None
TAG = 0x5A000000


@dataclass(frozen=True)
class GridDef:
    name: str
    e2m1: bool
    s: int                    # resolution 2^-s
    vals: tuple[int, ...]     # in units of 2^-s


# Must match fp_exact.cu's grids().
GRIDS = (
    GridDef("e4m3_int16", False, 0, tuple(range(-16, 17))),
    GridDef("e4m3_half8", False, 1, tuple(range(-16, 17))),
    GridDef("e4m3_quarter4", False, 2, tuple(range(-16, 17))),
    GridDef("e2m1_x2", True, 1, (0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12)),
    GridDef("e2m1_int", True, 1, (0, 2, -2, 4, -4, 6, -6, 8, -8, 12, -12)),
)
GRID = {g.name: i for i, g in enumerate(GRIDS)}


def e4m3_value(c: int) -> float:
    s, e, m = c >> 7, (c >> 3) & 15, c & 7
    if e == 15 and m == 7:
        return float("nan")
    v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return -v if s else v


def e4m3_code(v: float) -> int:
    if v == 0:
        return 0
    for c in range(256):
        if e4m3_value(c) == v:
            return c
    raise ValueError(f"{v} is not an E4M3 value")


def e2m1_code(halves: int) -> int:
    mag = (0, 1, 2, 3, 4, 6, 8, 12)
    a = abs(halves)
    if a not in mag:
        raise ValueError(f"{halves} halves is not an E2M1 value")
    return mag.index(a) | (8 if halves < 0 else 0)


def codes(g: GridDef, vals) -> list[int]:
    return [e2m1_code(v) if g.e2m1 else e4m3_code(v * 2.0 ** -g.s) for v in vals]


def ue8m0_code(k: int) -> int:
    return 127 + k  # 2^k, k in [-127, 127]


@dataclass(frozen=True)
class ScaleSet:
    """A per-block scale set (fp_exact.cu's ScaleSet). The scale value is num * 2^exp: UE4M3 odd integers have num odd
    and exp 0; UE8M0 powers 2^k have num 1 and exp k. Both operands draw from one set."""

    mode: int                  # 0 fixed (index 0); 1 drawn per block and operand; 2 alternating (block & 1)
    code: tuple[int, ...]      # the UE4M3 / UE8M0 byte
    num: tuple[int, ...]
    exp: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.code)

    @property
    def sh(self) -> int:
        """The reference's extra fraction bits: -2 x the most negative scale exponent (so every shift is >= 0), else 0."""
        return max(0, -2 * min(self.exp))


def make_scale(fmt: str, scale: str) -> ScaleSet:
    """Must match fp_exact.cu's make_scale. UE4M3 (nvf4): odd integers; UE8M0 (mxf4): powers 2^k."""
    ue4m3 = fmt in ("nvf4", "sp_nvf4")
    unit = 0x7F if fmt in ("mxf4", "mxf4nv", "sp_mxf4") else 0x38  # UE8M0 2^0 / UE4M3 (or E4M3) 1.0
    if scale == "unit":
        return ScaleSet(0, (unit,), (1,), (0,))
    if scale == "lo":
        return ScaleSet(0, (e4m3_code(3),), (3,), (0,)) if ue4m3 else ScaleSet(0, (ue8m0_code(-3),), (1,), (-3,))
    if scale == "hi":
        return ScaleSet(0, (e4m3_code(15),), (15,), (0,)) if ue4m3 else ScaleSet(0, (ue8m0_code(3),), (1,), (3,))
    if scale == "mix":
        if ue4m3:
            vs = tuple(range(1, 16, 2))
            return ScaleSet(1, tuple(e4m3_code(v) for v in vs), vs, (0,) * len(vs))
        kk = tuple(range(-3, 4))
        return ScaleSet(1, tuple(ue8m0_code(k) for k in kk), (1,) * len(kk), kk)
    if scale == "bind":
        return ScaleSet(2, (ue8m0_code(7), ue8m0_code(-7)), (1, 1), (7, -7))
    if scale == "bind26":
        return ScaleSet(2, (ue8m0_code(7), ue8m0_code(-6)), (1, 1), (7, -6))
    raise ValueError(f"unknown scale {scale}")


def scale_block(fmt: str) -> int:
    """Elements a scale block: 16 (UE4M3, nvf4), 32 (UE8M0, mxf4); 0 when the format carries no block scale."""
    return 16 if fmt in ("nvf4", "sp_nvf4") else 32 if fmt in ("mxf4", "mxf4nv", "sp_mxf4") else 0


def max_scale(fmt: str, scale: str) -> float:
    s = make_scale(fmt, scale)
    return max(n * 2.0 ** e for n, e in zip(s.num, s.exp))


def word(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def word_units(w: int, s_tot: int) -> int:
    """The FP32 word `w`'s value in units of 2^-s_tot (the device's `c0_units`)."""
    return int(struct.unpack("<f", struct.pack("<I", w))[0] * 2 ** s_tot)


def c0_words(g: GridDef, sh: int = 0) -> dict[str, int]:
    """The initial accumulators: zero; the binade whose ulp is the scaled product resolution 2^-(2s + sh) (NOBITS' form,
    generalized to carry the blocks' extra fraction bits); NOBITS' own 0x4B400000 (ulp 1); and the binade above the first
    (ulp twice the resolution: FP32 must round)."""
    u = 2 * g.s + sh
    out = {"zero": 0, "binade": 0x4B400000 - (u << 23), "over": 0x4BC00000 - (u << 23)}
    if u:
        out["nobits"] = 0x4B400000
    return out


def ks(fmt: str, quick: bool, scale: str = "unit") -> list[int]:
    # A non-unit scale stretches the product magnitude, so its representability limit is reached well below the unit
    # sweep's top; a short span keeps the sweep inside one lease while still crossing every scaled limit.
    cap = 4096 if scale in ("bind", "bind26") else 65536 if scale != "unit" else 262144
    k, out = KSTEP[fmt], []
    while k <= cap:
        out.append(k)
        k *= 2
    return out[::3] if quick else out


# The scales each format sweeps: NVFP4 the UE4M3 odd integers, MXFP4 (both mnemonics) the UE8M0 powers and the two
# anchor-binding draws; E4M3 and the 2:4-sparse route (whose MMA takes unit scales) stay on unit scales.
SCALED = {"nvf4": ("unit", "lo", "hi", "mix"),
          "mxf4": ("unit", "lo", "hi", "mix", "bind", "bind26"),
          "mxf4nv": ("unit", "lo", "hi", "mix", "bind", "bind26")}


def scales_of(fmt: str) -> tuple[str, ...]:
    return SCALED.get(fmt, ("unit",))


@dataclass(frozen=True)
class Config:
    fmt: str
    grid: str
    dist: str
    scale: str
    c0: str
    c0_word: int
    K: int
    tiles: int
    seed: int
    nrec: int

    @property
    def name(self) -> str:
        return f"{self.fmt}/{self.grid}/{self.scale}/{self.dist}/{self.c0}/K{self.K}"

    def line(self) -> str:
        return (f"{self.name} {FMTS.index(self.fmt)} {GRID[self.grid]} {DISTS.index(self.dist)} "
                f"{SCALES.index(self.scale)} {self.c0_word:08x} {self.K} {self.tiles} {self.seed} {self.nrec}")


def configs(quick: bool = False) -> list[Config]:
    out = []
    for fmt in FMTS:
        for gname in (n for n in GRID if n.startswith("e2m1") == (fmt not in ("e4m3", "e4m3_kind"))):
            g = GRIDS[GRID[gname]]
            for scale in scales_of(fmt):
                sh = make_scale(fmt, scale).sh
                for dist in DISTS:
                    c0s = c0_words(g, sh)
                    if scale != "unit":  # a non-unit scale carries only the K-limit's two cases
                        c0s = {k: v for k, v in c0s.items() if k in ("zero", "binade")}
                    if dist == "CANCEL":
                        c0s["negzero"] = 0x80000000
                    for c0, w in c0s.items():
                        for K in ks(fmt, quick, scale):
                            tiles = max(128, min(4096, (1 << 23) // K))
                            seed = (len(out) + 1) * 0x9E3779B1 & 0xFFFFFFFFFFFF
                            nrec = 2 if K <= 1024 else 1 if K <= 16384 or len(out) % 4 == 0 else 0
                            out.append(Config(fmt, gname, dist, scale, c0, w, K, tiles, seed, nrec))
    return out


# ---- the host's regeneration (fp_exact.cu's elA / elB / logical, in numpy) ----

U64 = np.uint64 if np else None


def mix64(z):
    with np.errstate(over="ignore"):
        z = z + U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
    return z ^ (z >> U64(31))


def hkey(seed: int, tile: int, mat: int, rc, k):
    key = (U64(tile) << U64(32)) | (U64(mat) << U64(30)) | (np.asarray(rc, dtype=U64) << U64(24)) | np.asarray(k, dtype=U64)
    return mix64(U64(seed) ^ key)


def pick(n: int, h):
    return ((h >> U64(32)) * U64(n)) >> U64(32)


def skey(seed: int, tile: int, op: int, rc, block):
    """fp_exact.cu's skey: the salt and layout keep a block's scale draw apart from the element hash (hkey)."""
    key = (U64(seed) ^ U64(0x5CA1E5CA1E000000) ^ (U64(tile) << U64(40)) ^ (U64(op) << U64(38))
           ^ (np.asarray(rc, dtype=U64) << U64(30)) ^ np.asarray(block, dtype=U64))
    return mix64(key)


def scale_arrays(c: "Config", tile: int):
    """The per-k scale factors of A (16 x K) and B (8 x K) in logical (order 0) k order, as positive int64 multipliers
    that fold num x 2^(exp + sh/2) so A's factor times B's is num_a num_b 2^(exp_a + exp_b + sh); and sh itself. The
    device scales each logical block of each operand row (A by its row, B by its column)."""
    s = make_scale(c.fmt, c.scale)
    scblk = scale_block(c.fmt) or c.K
    sh = s.sh
    nblk = c.K // scblk
    num = np.array(s.num, dtype=np.int64)
    shift = np.array(s.exp, dtype=np.int64) + sh // 2  # >= 0: the set's own exponents minus the most negative
    kblk = np.arange(c.K) // scblk

    def factor(op: int, n_rc: int):
        blk = np.arange(nblk)
        if s.mode == 0:
            idx = np.zeros((n_rc, nblk), dtype=np.int64)
        elif s.mode == 2:
            idx = np.broadcast_to((blk & 1).astype(np.int64), (n_rc, nblk))
        else:
            idx = pick(s.n, skey(c.seed, tile, op, np.arange(n_rc)[:, None], blk[None, :])).astype(np.int64)
        idxk = idx[:, kblk]
        return num[idxk] * np.left_shift(np.int64(1), shift[idxk])

    return factor(0, 16), factor(1, 8), sh


def scale_codes(c: "Config", tile: int):
    """The scale byte of each (row / column, logical block), for the silicon-model replay. None when the format is
    unscaled on the host side (E4M3, whose replay passes no scale bytes)."""
    g = GRIDS[GRID[c.grid]]
    if not g.e2m1:
        return None, None
    s = make_scale(c.fmt, c.scale)
    scblk = scale_block(c.fmt) or c.K
    nblk = c.K // scblk
    code = np.array(s.code, dtype=np.int64)

    def bytes_of(op: int, n_rc: int):
        blk = np.arange(nblk)
        if s.mode == 0:
            idx = np.zeros((n_rc, nblk), dtype=np.int64)
        elif s.mode == 2:
            idx = np.broadcast_to((blk & 1).astype(np.int64), (n_rc, nblk))
        else:
            idx = pick(s.n, skey(c.seed, tile, op, np.arange(n_rc)[:, None], blk[None, :])).astype(np.int64)
        return code[idx]

    return bytes_of(0, 16), bytes_of(1, 8)


@dataclass(frozen=True)
class Dist:
    a1: tuple[int, ...]
    a2: tuple[int, ...]
    b1: tuple[int, ...]
    b2: tuple[int, ...]
    ksplit: int
    cancel: bool


def dist_of(c: Config) -> Dist:
    g = GRIDS[GRID[c.grid]]
    mx = max(abs(v) for v in g.vals)
    mn = min(abs(v) for v in g.vals if v)
    if c.dist == "MAX":
        return Dist((mx,), g.vals, (mx,), g.vals, c.K, False)
    if c.dist == "MAXSIGN":
        return Dist((mx,), g.vals, (mx, -mx), g.vals, c.K, False)
    if c.dist == "RAMP":
        return Dist((mx,), (mn, -mn), (mx,), (mn,), c.K // 2, False)
    if c.dist == "EDGE":
        return Dist((mx,), (mn, 0), (mx,), (mn,), c.K // 2, False)
    return Dist(g.vals, g.vals, g.vals, g.vals, c.K, c.dist == "CANCEL")


def values(c: Config, tile: int):
    """A (16 x K) and B (8 x K, n-major) in the grid's units, in logical k order."""
    d = dist_of(c)
    k = np.arange(c.K, dtype=np.int64)
    kk = (k & ~1) if d.cancel else k
    first = k < d.ksplit

    def draw(mat, n_rc, g1, g2, neg):
        h = hkey(c.seed, tile, mat, np.arange(n_rc)[:, None], kk[None, :])
        i1, i2 = pick(len(g1), h).astype(np.int64), pick(len(g2), h).astype(np.int64)
        if neg:
            odd = (k & 1).astype(bool)[None, :]
            n1 = np.array([g1.index(-v) for v in g1]); n2 = np.array([g2.index(-v) for v in g2])
            i1, i2 = np.where(odd, n1[i1], i1), np.where(odd, n2[i2], i2)
        return np.where(first[None, :], np.array(g1)[i1], np.array(g2)[i2]).astype(np.int64)

    A = draw(0, 16, d.a1, d.a2, False)
    B = draw(1, 8, d.b1, d.b2, d.cancel)
    if c.fmt in SPARSE:
        pat = pick(6, hkey(c.seed, tile, 2, np.arange(16)[:, None], (k >> 3)[None, :])).astype(np.int64)
        pair = ((k & 7) >> 1)[None, :]
        A = np.where((pair == np.array(PAT0)[pat]) | (pair == np.array(PAT1)[pat]), A, 0)
    return A, B


def representable(v) -> np.ndarray:
    u = np.abs(np.asarray(v, dtype=np.int64)).astype(np.uint64)
    low = u & (~u + U64(1))
    odd = np.where(u == 0, U64(0), u // np.where(low == 0, U64(1), low))
    return odd < U64(1 << 24)


def exact_words(ref: np.ndarray, s_tot: int) -> np.ndarray:
    return np.array([word(float(int(r)) * 2.0 ** -s_tot) for r in ref.ravel()], dtype=np.uint64).reshape(ref.shape)


def recheck(c: Config, c0_units: int, recs: np.ndarray) -> dict:
    """The host's own sums and verdicts for the config's recorded tiles."""
    units = 2 * GRIDS[GRID[c.grid]].s + make_scale(c.fmt, c.scale).sh
    nsteps, KS = c.K // KSTEP[c.fmt], KSTEP[c.fmt]
    out = {"tiles": int(c.nrec), "outputs": 0, "unwritten": 0, "ref_disagree": 0, "flag_disagree": 0, "exact": 0,
           "bad_rep": 0, "order_diff": 0, "in_binade": 0, "binade_linear": 0}
    for t in range(c.nrec):
        r = recs[t * 128:(t + 1) * 128].reshape(16, 8)
        A, B = values(c, t)
        fA, fB, _ = scale_arrays(c, t)
        sA = (A * fA).reshape(16, nsteps, KS)
        sB = (B * fB).reshape(8, nsteps, KS)
        part = np.einsum("isk,jsk->ijs", sA, sB).cumsum(axis=2) + c0_units
        ref = part[:, :, -1]
        rep_all = representable(part).all(axis=2)
        frep = representable(ref)
        want = exact_words(ref, units)
        d0 = r["d"][:, :, 0].astype(np.uint64)
        ex = frep & (d0 == want)
        out["outputs"] += 128
        out["unwritten"] += int(((r["flags"] & 0xFF000000) != TAG).sum())
        out["ref_disagree"] += int((r["ref"] != ref).sum())
        out["flag_disagree"] += int(((r["flags"] & 3) != (rep_all.astype(np.uint32) | (frep.astype(np.uint32) << 1))).sum())
        out["exact"] += int(ex.sum())
        out["bad_rep"] += int((rep_all & ~ex).sum())
        out["order_diff"] += int((r["d"] != r["d"][:, :, :1]).any(axis=2).sum())
        if c.c0 == "binade":
            # NOBITS' form: while every partial stays in C0's binade (ulp one unit), the word is C0's word plus T.
            inside = ((part >= 1 << 23) & (part < 1 << 24)).all(axis=2)
            lin = d0.astype(np.int64) - c.c0_word == ref - c0_units
            out["in_binade"] += int(inside.sum())
            out["binade_linear"] += int((inside & lin).sum())
    return out


REPLAY_K = 256


def model(fmt: str):
    """One firing of `fmt` as `verity.ml.circuits.silicon` models it. The 2:4-sparse firing has no model of its own:
    it is replayed as the dense adder over its 128 logical elements (one align-add and one RZ a firing), which is the
    hypothesis its replay tests."""
    try:
        from verity.ml.circuits.silicon import models as m
    except ImportError:  # the node's python3 runs this from the source tree, outside the workspace's environment
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from verity.ml.circuits.silicon import models as m
    import dataclasses
    if fmt.startswith("e4m3"):
        return m.BLACKWELL_SM120_E4M3_M16N8K32
    base = m.BLACKWELL_SM120_NVF4 if fmt.endswith("nvf4") else m.BLACKWELL_SM120_MXF4
    if fmt in SPARSE:
        return dataclasses.replace(base, name=base.name.replace("m16n8k64", "sp_m16n8k128"), k_per_instruction=128)
    return base


def replay(c: Config, recs: np.ndarray) -> dict:
    """The silicon model's words for the config's first recorded tile, chained in the identity order from C0, against
    the device's: every output, and the inexact ones among them. Each firing carries its blocks' scale bytes (block b
    of firing f is logical scale block f * scales_per_step + b, the order-0 layout of fp_exact.cu's pack_sf)."""
    g = GRIDS[GRID[c.grid]]
    m, step = model(c.fmt), KSTEP[c.fmt]
    nbf = m.scales_per_step if g.e2m1 else 0
    code = dict(zip(g.vals, codes(g, g.vals)))
    A, B = values(c, 0)
    a = [[code[int(v)] for v in row] for row in A]
    b = [[code[int(v)] for v in row] for row in B]
    sca, scb = scale_codes(c, 0)
    r = recs[:128].reshape(16, 8)
    ref = r["ref"].astype(np.int64)
    units = 2 * g.s + make_scale(c.fmt, c.scale).sh
    exact = representable(ref) & (r["d"][:, :, 0].astype(np.uint64) == exact_words(ref, units))
    out = {"model": m.name, "outputs": 0, "agree": 0, "inexact": 0, "inexact_agree": 0}
    for i in range(16):
        for j in range(8):
            w = c.c0_word
            for f, s in enumerate(range(0, c.K, step)):
                fa = [int(x) for x in sca[i, f * nbf:(f + 1) * nbf]] if g.e2m1 else []
                fb = [int(x) for x in scb[j, f * nbf:(f + 1) * nbf]] if g.e2m1 else []
                w = m.step(w, a[i][s:s + step] + fa, b[j][s:s + step] + fb)
            agree = w == int(r["d"][i, j, 0])
            out["outputs"] += 1
            out["agree"] += agree
            out["inexact"] += not exact[i, j]
            out["inexact_agree"] += agree and not exact[i, j]
    return out


def read_run(d: Path, cfgs: list[Config], control: bool = False) -> list[dict]:
    lines = [json.loads(x) for x in (d / ("control.jsonl" if control else "counts.jsonl")).read_text().splitlines()]
    head, rows = lines[0], lines[1:]
    if len(rows) != len(cfgs):
        raise SystemExit(f"fp_exact: {len(rows)} results for {len(cfgs)} configs")
    recs = np.fromfile(d / ("control.bin" if control else "records.bin"), dtype=REC)
    out, at = [], 0
    for c, row in zip(cfgs, rows):
        if row["name"] != c.name:
            raise SystemExit(f"fp_exact: result {row['name']} where {c.name} was expected")
        rc = recs[at:at + c.nrec * 128]
        at += c.nrec * 128
        row["refused"] = refusal(c, row, rc)
        row["host"] = recheck(c, row["c0_units"], rc) if not row["refused"] else None
        if row["host"] and (row["host"]["ref_disagree"] or row["host"]["flag_disagree"]):
            row["refused"] = (f"the host's sums disagree with the device's on {row['host']['ref_disagree']} outputs "
                              f"(flags on {row['host']['flag_disagree']})")
        row["replay"] = replay(c, rc) if not row["refused"] and c.nrec and c.K <= REPLAY_K else None
        out.append({**row, "head": head})
    return out


def refusal(c: Config, row: dict, recs: np.ndarray) -> str:
    if row["tag"] != (TAG | FMTS.index(c.fmt)):
        return f"the kernel didn't write its counts (tag 0x{row['tag']:08x})"
    if row["out"] != c.tiles * 128:
        return f"{row['out']} outputs counted, {c.tiles * 128} expected"
    units = 2 * GRIDS[GRID[c.grid]].s + make_scale(c.fmt, c.scale).sh
    if row["c0_units"] != (u := word_units(c.c0_word, units)):
        return f"the device started from {row['c0_units']} units, not C0's {u}"
    if int(((recs["flags"] & 0xFF000000) != TAG).sum()):
        return "a recorded tile left poisoned words"
    return ""


def verdicts(rows: list[dict], cfgs: list[Config]) -> dict:
    """Per format, grid and scale: where every output was exact, where rounding first appeared, and whether any output
    lost bits while every partial was representable."""
    by: dict[tuple, list] = {}
    for c, r in zip(cfgs, rows):
        by.setdefault((c.fmt, c.grid, c.scale), []).append((c, r))
    out = {}
    for (fmt, grid, scale), xs in by.items():
        refused = [c.name for c, r in xs if r["refused"]]
        ok = [(c, r) for c, r in xs if not r["refused"]]
        series = {}
        for c, r in ok:
            series.setdefault(f"{c.dist}/{c.c0}", []).append((c.K, r))
        per = {}
        for key, ser in series.items():
            ser.sort(key=lambda x: x[0])
            exact_to = 0
            for K, r in ser:
                if r["exact"] != r["out"]:
                    break
                exact_to = K
            first = next(((K, r) for K, r in ser if r["exact"] != r["out"]), None)
            per[key] = {
                "exact_through_K": exact_to,
                "first_inexact_K": first[0] if first else None,
                "first_inexact": ({k: first[1][k] for k in ("out", "exact", "rep_all", "bad_rep", "unrep", "bad_unrep",
                                                              "rz_final", "order_diff", "max_partial")}
                                  if first else None),
                "max_partial": max(r["max_partial"] for _, r in ser),
            }
        out[f"{fmt}/{grid}/{scale}"] = {
            "configs": len(xs),
            "max_scale": max_scale(fmt, scale),
            "refused": refused,
            "outputs": sum(r["out"] for _, r in ok),
            "exact": sum(r["exact"] for _, r in ok),
            "rep_all": sum(r["rep_all"] for _, r in ok),
            "bad_rep": sum(r["bad_rep"] for _, r in ok),
            "unrep": sum(r["unrep"] for _, r in ok),
            "bad_unrep": sum(r["bad_unrep"] for _, r in ok),
            "unrep_rz_final": sum(r["rz_final"] for _, r in ok),
            "unrep_rn_final": sum(r["rn_final"] for _, r in ok),
            "order_diff": sum(r["order_diff"] for _, r in ok),
            "order_diff_when_exact": sum(r["order_diff"] for _, r in ok if r["exact"] == r["out"]),
            "zero": sum(r["zero"] for _, r in ok),
            "negzero": sum(r["negzero"] for _, r in ok),
            "host_outputs": sum(r["host"]["outputs"] for _, r in ok),
            "host_bad_rep": sum(r["host"]["bad_rep"] for _, r in ok),
            "host_in_binade": sum(r["host"]["in_binade"] for _, r in ok),
            "host_binade_linear": sum(r["host"]["binade_linear"] for _, r in ok),
            "replay": {k: sum(r["replay"][k] for _, r in ok if r.get("replay"))
                       for k in ("outputs", "agree", "inexact", "inexact_agree")}
                      | {"model": next((r["replay"]["model"] for _, r in ok if r.get("replay")), None)},
            "series": per,
        }
    return out


def limits(v: dict) -> list[str]:
    """Per format, grid, scale and C0: the largest K through which every sample was exact, and the first K any draw rounded.
    The max scale product is the largest scale a block times itself (both operands draw from the one set)."""
    out = ["| format/grid | scale | max scale product | C0 | exact for every draw through K | first inexact K (draws) |",
           "|---|---|---|---|---|---|"]
    for key, x in v.items():
        fmt, grid, scale = key.split("/")
        msp = max_scale(fmt, scale) ** 2
        by: dict[str, list] = {}
        for s, y in x["series"].items():
            by.setdefault(s.split("/")[1], []).append((s.split("/")[0], y))
        for c0, ys in sorted(by.items()):
            first = [y["first_inexact_K"] for _, y in ys if y["first_inexact_K"]]
            k1 = min(first) if first else None
            cell = f"{k1} ({','.join(d for d, y in ys if y['first_inexact_K'] == k1)})" if k1 else "-"
            out.append(f"| {fmt}/{grid} | {scale} | {msp:g} | {c0} | "
                       f"{min(y['exact_through_K'] for _, y in ys)} | {cell} |")
    return out


PROBE_MACS = {"imma_s8": 4096, "qmma_e4m3": 4096, "omma_mxf4": 8192, "omma_nvf4": 8192, "omma_sp_mxf4": 16384,
              "omma_sp_nvf4": 16384,
              # the same opcodes on the sweep's drawn operands (a data-dependent-clock check against the fixed probes)
              "qmma_e4m3_data": 4096, "omma_mxf4_data": 8192, "omma_nvf4_data": 8192, "omma_sp_mxf4_data": 16384,
              "omma_sp_nvf4_data": 16384}


def rates(rows: list[dict], sms: int = 188) -> dict[str, dict]:
    """opcode_probe's rows for probes/mma.cu as MACs an SM a clock, against the same run's IMMA. A `_data` probe runs the
    same opcode on the sweep's drawn operands; `vs_fixed` (its rate over the fixed probe's) rules out a data clock."""
    by = {r["probe"]: r for r in rows if r["probe"] in PROBE_MACS}
    if "imma_s8" not in by:
        raise SystemExit("fp_exact: no imma_s8 row to compare with")
    base = by["imma_s8"]["rate"] * PROBE_MACS["imma_s8"]
    out = {}
    for name, r in by.items():
        m = r["rate"] * PROBE_MACS[name]
        n = r.get("distinct_sms") or sms
        fixed = by.get(name[:-5]) if name.endswith("_data") else None
        out[name] = {"rate": r["rate"], "macs_sm_clk": m, "vs_imma": m / base, "sm_mhz": r.get("sm_mhz"), "sms": n,
                     "tmacs": m * n * (r.get("sm_mhz") or 0) * 1e6 / 1e12, "tags": r.get("tags", []),
                     "vs_fixed": r["rate"] / fixed["rate"] if fixed else None,
                     "run": r.get("run", ""), "sass_sha256": r.get("sass_sha256", "")}
    return out


EXPECT_MMA = {"e4m3": "QMMA.16832.F32.E4M3.E4M3", "e4m3_kind": "QMMA.16832.F32.E4M3.E4M3",
              "mxf4": "OMMA.SF.16864.F32.E2M1.E2M1.E8", "mxf4nv": "OMMA.SF.16864.F32.E2M1.E2M1.E8",
              "nvf4": "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", "sp_mxf4": "OMMA.SF.SP.168128.F32.E2M1.E2M1.E8",
              "sp_nvf4": "OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X"}


def sass_refusal(mmas: dict[str, list[str]]) -> str:
    """Each format's kernel must issue exactly its one expected MMA (the chain loop is `#pragma unroll 1`)."""
    for i, fmt in enumerate(FMTS):
        got = mmas.get(f"_Z12exact_kernelILi{i}EEv6ParamsP6CountsP3Rec")
        if got != [EXPECT_MMA[fmt]]:
            return f"fp_exact: {fmt}'s kernel issues {got}, not one {EXPECT_MMA[fmt]}"
    return ""


def sass_mmas(binary: Path) -> dict[str, list[str]]:
    text = subprocess.run(["cuobjdump", "-sass", str(binary)], check=True, capture_output=True, text=True).stdout
    out: dict[str, list[str]] = {}
    fn = None
    for line in text.splitlines():
        if "Function :" in line:
            fn = line.split("Function :")[1].strip()
            out[fn] = []
        elif fn and "MMA" in line:
            ins = line.split("*/")[1].strip().split()[0] if "*/" in line else line.split()[0]
            out[fn].append(ins)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fp_exact.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs")
    c.add_argument("--quick", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("binary", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--quick", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("dir", type=Path)
    t = sub.add_parser("rates")
    t.add_argument("rows", type=Path, help="opcode_probe's rows.jsonl for probes/mma.cu")
    lm = sub.add_parser("limits")
    lm.add_argument("verdicts", type=Path, help="a run's exact/verdicts.json")
    a = ap.parse_args(argv)
    if a.cmd == "limits":
        print("\n".join(limits(json.loads(a.verdicts.read_text())["verdicts"])))
        return 0
    if a.cmd == "configs":
        print("\n".join(x.line() for x in configs(a.quick)))
        return 0
    if a.cmd == "run":
        a.out.mkdir(parents=True, exist_ok=True)
        cfgs = configs(a.quick)
        (a.out / "configs.txt").write_text("\n".join(x.line() for x in cfgs) + "\n")
        (a.out / "sweep.json").write_text(json.dumps({"quick": a.quick}))
        mmas = sass_mmas(a.binary)
        (a.out / "sass_mmas.json").write_text(json.dumps(mmas, indent=1))
        if why := sass_refusal(mmas):
            raise SystemExit(why)
        for args in ((), ("--control",)):
            stem = ("control", "control") if args else ("counts", "records")
            subprocess.run([str(a.binary), str(a.out / "configs.txt"), str(a.out / f"{stem[0]}.jsonl"),
                            str(a.out / f"{stem[1]}.bin"), *args], check=True)
        return 0
    if a.cmd == "rates":
        r = rates([json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()])
        (a.rows.parent / "rates.json").write_text(json.dumps(r, indent=1))
        for name, x in r.items():
            vf = f", {x['vs_fixed']:.4f}x the fixed probe" if x["vs_fixed"] is not None else ""
            print(f"{name}: {x['rate']:.5f} instructions/SM/clk, {x['macs_sm_clk']:.0f} MACs/SM/clk, "
                  f"{x['vs_imma']:.3f}x IMMA, {x['tmacs']:.1f} TMAC/s at {x['sm_mhz']} MHz{vf}, tags {x['tags']}")
        return 0
    quick = json.loads((a.dir / "sweep.json").read_text())["quick"]
    cfgs = configs(quick)
    rows = read_run(a.dir, cfgs)
    ctrl = read_run(a.dir, cfgs, control=True)
    accepted = [r["name"] for r in ctrl if not r["refused"]]
    v = {"run": os.environ.get("RESEARCH_RUN_ID", ""), "device": rows[0]["head"], "control_accepted": accepted,
         "verdicts": verdicts(rows, cfgs)}
    (a.dir / "verdicts.json").write_text(json.dumps(v, indent=1))
    for key, x in v["verdicts"].items():
        print(f"{key}: {x['outputs']} outputs, {x['exact']} exact, bad_rep {x['bad_rep']}, unrep {x['unrep']} "
              f"(inexact {x['bad_unrep']}), order_diff {x['order_diff']} (when exact {x['order_diff_when_exact']}), "
              f"negzero {x['negzero']}/{x['zero']}, host bad_rep {x['host_bad_rep']}/{x['host_outputs']}, "
              f"binade word = C0 + T {x['host_binade_linear']}/{x['host_in_binade']}, "
              f"{x['replay']['model']} replay {x['replay']['agree']}/{x['replay']['outputs']} "
              f"(inexact {x['replay']['inexact_agree']}/{x['replay']['inexact']}), refused {len(x['refused'])}")
    if accepted:
        print(f"fp_exact: the no-write control was accepted for {len(accepted)} configs", file=sys.stderr)
        return 9
    if any(x["refused"] for x in v["verdicts"].values()):
        return 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
