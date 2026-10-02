"""The dependent chain's derive, f: the next call's A from this call's y, and its numpy twin (`native/harness.cu`'s `k_derive`
must give the same bytes: the preflight checks every kind on the device once per run, and each chain's gate checks the A it
derived).

Row r of A (m x k) reads row r of y (m x n, row-major), column c reads column c mod n. Each row is scaled by 2^(T - E), where
the row's amax is in [2^(E-1), 2^E) (a zero row is not scaled), so |x| < 2^T and the chain's values stay bounded over any
number of calls; the scale is not passed on (the chain is a timing construction, not a model's numerics). Then x is
converted to the consumer's format, rounding to nearest even and saturating; the block-scaled formats get real scales in the
128 x 4 tiled layout (cuBLASLt's and CUTLASS's): MXFP8 and MXFP4 the OCP MX rule (2^(floor(log2 amax) - emax) per 32),
NVFP4 UE4M3(amax / 6) per 16 and E2M1(x / scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import fill as F
import numpy as np

SRC = {"bf16": 0, "int32": 1, "f32": 2}


@dataclass(frozen=True)
class Kind:
    name: str
    id: int
    t: int                 # |x| < 2^t after the row's scaling
    block: int = 1         # elements per scale (1: none)
    emax: int = 0          # the element format's largest exponent (the MX rule)
    elem_bytes: float = 1
    what: str = ""


KINDS = {k.name: k for k in (
    Kind("f32", 0, 2, elem_bytes=4, what="FP32 x"),
    Kind("bf16", 1, 2, elem_bytes=2, what="BF16 RNE(x)"),
    Kind("e4m3", 2, 8, what="E4M3 RNE-satfinite(x), the per-tensor scale left at 1"),
    Kind("mxfp8", 3, 0, 32, 8, what="E4M3 codes with UE8M0 scales per 32 (OCP MX)"),
    Kind("nvfp4", 4, 11, 16, elem_bytes=0.5, what="E2M1 codes with UE4M3 scales per 16 (amax / 6), the global scale left at 1"),
    Kind("mxfp4", 5, 0, 32, 2, elem_bytes=0.5, what="E2M1 codes with UE8M0 scales per 32 (OCP MX)"),
    Kind("int8", 6, 7, what="INT8 RNE(x) clamped to +-127"),
)}

#: the derive each baseline family's chain uses, and what it leaves out
FAMILY_KIND = {"bf16": "bf16", "fp8-e4m3": "e4m3", "fp8-e4m3-rowwise": "e4m3", "mxfp8": "mxfp8", "fp8-blk128": "e4m3",
               "nvfp4": "nvfp4", "mxfp4": "mxfp4", "int8": "int8"}
FAMILY_SHORTCUT = {"fp8-blk128": "A's 1 x 128 FP32 scales are left at 1: the derive writes E4M3 codes only"}
#: an arm's `inputs` -> its derive
INPUT_KIND = {"f32": "f32", "bf16": "bf16", "e4m3": "e4m3", "e2m1-nv": "nvfp4", "e2m1-mx": "mxfp4", "int8": "int8"}


def a_bytes(kind: Kind, m: int, k: int) -> int:
    return int(m * k * kind.elem_bytes)


def sf_offsets(rows: int, cols: int) -> np.ndarray:
    """(rows, cols) byte offsets of each block's scale in the 128 x 4 tiled layout."""
    r, c = np.arange(rows)[:, None], np.arange(cols)[None, :]
    return ((r // 128) * (-(-cols // 4)) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)


def load(src: str, raw: bytes, m: int, n: int) -> np.ndarray:
    """y's first m rows as float32 (m, n); int32 converts rounding to nearest even."""
    if src == "bf16":
        return (np.frombuffer(raw, "<u2", m * n).astype(np.uint32) << 16).view(np.float32).reshape(m, n)
    if src == "int32":
        return np.frombuffer(raw, "<i4", m * n).astype(np.float32).reshape(m, n)
    return np.frombuffer(raw, "<f4", m * n).reshape(m, n).copy()


_E4M3_POS = F.E4M3_VALUES[:127]


def e4m3_rne(x: np.ndarray) -> np.ndarray:
    """Nearest E4M3 code, ties to the even code, saturating at 448; the sign kept (zero included)."""
    a = np.abs(np.asarray(x, np.float32)).astype(np.float64)
    i = np.clip(np.searchsorted(_E4M3_POS, a), 1, 126)
    lo, hi = i - 1, i
    dlo, dhi = a - _E4M3_POS[lo], _E4M3_POS[hi] - a
    code = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    code = np.where(a >= 448, 126, code)
    return (code | np.where(np.signbit(x), 0x80, 0)).astype(np.uint8)


def e2m1_rne(x: np.ndarray) -> np.ndarray:
    """Nearest E2M1 code of {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even code, saturating at 6; the sign kept."""
    a = np.abs(np.asarray(x, np.float32))
    c = np.select([a <= 0.25, a < 0.75, a <= 1.25, a < 1.75, a <= 2.5, a < 3.5, a <= 5.0], [0, 1, 2, 3, 4, 5, 6], 7)
    return (c | np.where(np.signbit(x), 8, 0)).astype(np.uint8)


def _exp(v: np.ndarray) -> np.ndarray:
    """E with v in [2^(E-1), 2^E) (v > 0)."""
    return np.frexp(np.asarray(v, np.float32))[1].astype(np.int32)


def scaled(kind: Kind, src: str, y: bytes, m: int, n: int, k: int) -> np.ndarray:
    """Each row of A before conversion: y's row (columns mod n) times 2^(T - E)."""
    v = load(src, y, m, n)
    amax = np.max(np.abs(np.where(np.isnan(v), np.float32(0), v)), axis=1)
    ok = (amax > 0) & np.isfinite(amax)
    shift = np.where(ok, kind.t - _exp(np.where(ok, amax, np.float32(1))), 0).astype(np.int32)
    return np.ldexp(v[:, np.arange(k) % n], shift[:, None]).astype(np.float32)


def derive(kind_name: str, src: str, y: bytes, m: int, n: int, k: int) -> tuple[bytes, np.ndarray | None, np.ndarray | None]:
    """A's bytes and, for the block-scaled kinds, the scale codes (m, k / block) with their offsets in the scale buffer."""
    kind = KINDS[kind_name]
    x = scaled(kind, src, y, m, n, k)
    if kind.name == "f32":
        return x.astype("<f4").tobytes(), None, None
    if kind.name == "bf16":
        return F.bf16_rne(x).astype("<u2").tobytes(), None, None
    if kind.name == "e4m3":
        return e4m3_rne(x).tobytes(), None, None
    if kind.name == "int8":
        return np.clip(np.rint(x), -127, 127).astype(np.int8).tobytes(), None, None
    bs, cols = kind.block, k // kind.block
    blocks = x.reshape(m, cols, bs)
    bmax = np.max(np.abs(blocks), axis=2)
    if kind.name == "nvfp4":
        scale = e4m3_rne(bmax / np.float32(6))
        s = F.E4M3_VALUES[scale].astype(np.float32)[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            codes = np.where(s > 0, e2m1_rne(blocks / np.where(s > 0, s, np.float32(1))), 0).astype(np.uint8)
    else:
        xs = np.where(bmax > 0, _exp(np.where(bmax > 0, bmax, np.float32(1))) - 1 - kind.emax, -127)
        xs = np.clip(xs, -127, 127).astype(np.int32)
        scale = (xs + 127).astype(np.uint8)
        q = np.ldexp(blocks, (-xs)[..., None]).astype(np.float32)
        codes = e4m3_rne(q) if kind.name == "mxfp8" else e2m1_rne(q)
    codes = codes.reshape(m, k)
    if kind.elem_bytes == 0.5:
        codes = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return codes.tobytes(), scale, sf_offsets(m, cols)


def check(kind_name: str, src: str, y: bytes, a: bytes, sf: bytes | None, m: int, n: int, k: int) -> dict:
    """The device's A (and scale buffer) against the twin's f(y): mismatching bytes, and the first few."""
    want, scale, offs = derive(kind_name, src, y, m, n, k)
    got = np.frombuffer(a, np.uint8, len(want))
    bad = np.flatnonzero(got != np.frombuffer(want, np.uint8))
    out = {"kind": kind_name, "src": src, "a_bytes": len(want), "a_mismatches": int(len(bad)),
           "a_first": [{"byte": int(i), "got": int(got[i]), "want": want[i]} for i in bad[:4]]}
    if scale is not None:
        g = np.frombuffer(sf, np.uint8)[offs]
        sbad = np.argwhere(g != scale)
        out.update(scales=int(scale.size), scale_mismatches=int(len(sbad)),
                   scale_first=[{"row": int(r), "block": int(c), "got": int(g[r, c]), "want": int(scale[r, c])}
                                for r, c in sbad[:4]])
    out["ok"] = out["a_mismatches"] == 0 and out.get("scale_mismatches", 0) == 0
    return out


# ---- the preflight's y: every kind's rounding ties, zeros of both signs, saturation, rows of one scale -----------------------


def _ulps(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Non-negative float32 x moved d ulps, stopping at +0 (whose bits less one ulp would be a NaN)."""
    return np.maximum(np.asarray(x, np.float32).view(np.int32) + np.asarray(d, np.int32), np.int32(0)).view(np.float32)


def _nvfp4_edges(rng: np.random.Generator, n: int) -> np.ndarray:
    """A row where NVFP4's roundings tie or turn (at a row scale of 1: its amax, 2016, is in [2^10, 2^11)): each block of 16 has
    its amax at, or an ulp either side of, 6 times a midpoint between two E4M3 scales, and its other values at, or an ulp
    either side of, an E2M1 threshold times the block's scale."""
    pos = F.E4M3_VALUES[:127].astype(np.float32)
    six_mid = np.float32(3) * (pos[:-1] + pos[1:])
    six_mid = six_mid[six_mid < 2**11]
    th = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
    v = np.empty(n, np.float32)
    for b0 in range(0, n, 16):
        w = min(16, n - b0)
        bmax = _ulps(six_mid[rng.integers(len(six_mid))], rng.integers(-1, 2))
        s = pos[e4m3_rne(bmax / np.float32(6))]
        blk = _ulps(th[rng.integers(0, 7, w)] * s, rng.integers(-1, 2, w)) * rng.choice(np.float32([-1, 1]), w)
        blk[rng.integers(w)] = bmax
        v[b0:b0 + w] = blk
    v[0] = 2016
    return v


def preflight_y(src: str, m: int, n: int, seed: int) -> bytes:
    """A y whose derive exercises the kinds' edge cases: random full-entropy rows, rows of small integers (exact ties in every
    format after scaling), a zero row, a row of negative zeros, rows with one large outlier (underflow to zero and
    subnormals), a row of the same value, a row of FP32 subnormals (a row scale past 2^127), a row near FP32's top (past
    2^-126) and a row on NVFP4's rounding edges (`_nvfp4_edges`, exact from f32)."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(m):
        t = r % 10
        if t == 0:
            v = rng.standard_normal(n) * 2.0 ** rng.integers(-8, 12)
        elif t == 1:
            v = rng.integers(-300, 301, n).astype(np.float64)
        elif t == 2:
            v = np.zeros(n)
        elif t == 3:
            v = -np.zeros(n)
        elif t == 4:
            v = rng.integers(-40, 41, n).astype(np.float64) * 2.0 ** -rng.integers(0, 20)
            v[rng.integers(n)] = 2.0 ** 30
        elif t == 5:
            v = np.full(n, 3.0 * 2.0 ** rng.integers(-5, 5))
        elif t == 6:
            v = rng.standard_normal(n) * np.exp(rng.standard_normal(n) * 3)
        elif t == 7:
            v = rng.standard_normal(n) * 2.0 ** -140
        elif t == 8:
            v = np.clip(rng.standard_normal(n), -7, 7) * 2.0 ** 125
        else:
            v = _nvfp4_edges(rng, n)
        rows.append(v)
    y = np.array(rows)
    if src == "bf16":
        return F.bf16_rne(y.astype(np.float32)).astype("<u2").tobytes()
    if src == "int32":
        return np.clip(np.rint(y * 1000), -(2**31), 2**31 - 1).astype("<i4").tobytes()
    return y.astype("<f4").tobytes()
