"""IR-registered MUFU primitives: the special-function unit's `ex2`, `rcp`, `sqrt`, `rsqrt` and `tanh` approximations as sm_89
computes them, FA2's inverse sum, and the range scaling `div.full.f32` puts around `rcp`.

Encodings: f32 = `Value<32>`.  Every evaluator is total on encodings except `MufuTanh`, which raises `MufuTanhUnmeasured` on a word
its measurement does not cover.  A MUFU operation is a measured table read inside an exact bit rule: the rule handles sign,
exponent and special encodings, the table gives the mantissa word of one binade (two for `sqrt` / `rsq`, by exponent parity).

Tables (package data under `tables/`).  `mufu_{rsq,sqrt,rcp,ex2}_delta_int8.xz` hold the int8 first differences of a table's
little-endian u32 words, xz-compressed, from the first word 1.0 (`0x3F800000`).  `table(key)` decodes one by its
`verity.ml.circuits.library` name or SHA-512 and refuses words whose SHA-512 is not the library's; the Lean verifier's archive holds the
same words under that SHA-512.  `mufu_tanh_sm89.{json,xzblocks}` is MUFU.TANH's measured shard with its manifest (see the
MufuTanh section).  All were measured on an RTX 4090 (sm_89): conformance evidence for that architecture.  The delta tables are
the captures of records `W11-40f0cebeb670-20260907T1800Z` (ex2, rcp) and `W11C-40f0cebeb670-20260907T1900Z` (rsq, sqrt); the
Triton record `W11R-40f0cebeb670-20260907T1802Z` measured the same rcp and sqrt words.

Batch kernels.  Every primitive but `MufuTanh` has a numpy kernel registered with `verity.ml.circuits.evaluation` (domain: every encoding).

Provenance.  These are the definitions the vLLM integration registered in `verity_vllm.program.registry.prims` (same ids,
signatures and conformance statuses), which re-exports them from here.  The bit rules are the integration's C++ models
(`fa2_model.cpp`: `mufu_ex2_bits`, `mufu_rcp_bits`; `rms_triton_model.cpp`: `mufu_sqrt`, `mufu_rsq`, `div_full_parts`,
`rsqrt_nonftz`) on integers; the kernels are its numpy ports (`integrations/vllm/verity_vllm/program/kernels/derived_rows.py`).
"""
from __future__ import annotations

import hashlib
import lzma
import os
from functools import lru_cache
from importlib import resources

import numpy as np

from verity.ml.circuits.evaluation import register_kernel
from verity.ml.circuits.defs import primitive
from verity.ml.circuits.types import Value

from verity.ml.circuits import library
from .fp32 import bits_of, f32_is_nan, f32_of

F32 = Value(32)

__all__ = ["DELTA_TABLES", "F32", "MUFU_TANH_RULES", "MUFU_TANH_SHARD_FORMAT", "MUFU_TANH_TABLE_DIR", "MUFU_TANH_TABLE_FORMAT",
           "MUFU_TANH_TABLE_MANIFEST", "MUFU_TANH_TABLE_MANIFEST_SHA512", "TABLE_DIR", "DivFullRcp", "DivFullScaleA", "Fa2InvSum",
           "MufuEx2Ftz", "MufuRcpFtz", "MufuSqrtFtz", "MufuTanh", "MufuTanhUnmeasured", "RsqrtApprox", "mufu_ex2_bits",
           "mufu_rcp_bits", "mufu_rsq_bits", "mufu_sqrt_bits", "mufu_tanh_coverage", "mufu_tanh_shards", "mufu_tanh_table_manifest",
           "table", "table_path"]

TABLE_DIR = str(resources.files(__package__) / "tables")
#: the library tables stored as delta files, each read by index
DELTA_TABLES = ("rsq", "sqrt", "rcp", "ex2")

_ONE = 0x3F800000
_INF = 0x7F800000
_NAN = 0x7FFFFFFF  # the one NaN every MUFU operation returns
_MIN_NORMAL = 0x00800000  # 2^-126
_TWO_POW_126 = 0x7E800000


# ---------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------


def table_path(name: str) -> str:
    return os.path.join(TABLE_DIR, f"mufu_{name}_delta_int8.xz")


@lru_cache(maxsize=None)
def _decoded(name: str) -> np.ndarray:
    entry = library.table(name)
    with open(table_path(name), "rb") as f:
        delta = np.frombuffer(lzma.decompress(f.read()), dtype=np.int8)
    if delta.shape[0] != entry.words - 1:
        raise ValueError(f"MUFU table {name}: {delta.shape[0]} differences, expected {entry.words - 1}")
    words = np.empty(entry.words, dtype=np.int64)
    words[0] = _ONE
    np.cumsum(delta, dtype=np.int64, out=words[1:])
    words[1:] += _ONE
    words = words.astype(np.uint32)
    if hashlib.sha512(words.astype("<u4", copy=False)).hexdigest() != entry.sha512:
        raise ValueError(f"MUFU table {name} at {table_path(name)}: its words are not library table {entry.sha512}")
    words.flags.writeable = False
    return words


def table(key: str) -> np.ndarray:
    """The words of the MUFU table ``key`` (a `verity.ml.circuits.library` name or SHA-512), read-only, checked against the library."""
    name = library.table(key).name
    if name not in DELTA_TABLES:
        raise KeyError(f"library table {name!r} is not stored here (the delta tables are {DELTA_TABLES})")
    return _decoded(name)


@lru_cache(maxsize=None)
def _row(name: str) -> memoryview:
    return memoryview(table(name))


# ---------------------------------------------------------------------------------------------------------
# bit rules (t: the table's words, indexable)
# ---------------------------------------------------------------------------------------------------------


def mufu_ex2_bits(x: int, t) -> int:
    """`ex2.approx.ftz.f32`: t[j] = hw(2^(j / 2^23)); x is read as the signed fixed-point number floor(x 2^23) (the negative
    ones rounded down), whose integer part n scales t[fraction] by 2^n."""
    x &= 0xFFFFFFFF
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    if e == 0xFF:
        return _NAN if ax & 0x7FFFFF else (0 if x >> 31 else _INF)
    if e == 0:
        return _ONE
    unb = e - 127
    if unb >= 30:
        return 0 if x >> 31 else _INF
    mant = (ax & 0x7FFFFF) | 0x800000
    k = mant << unb if unb >= 0 else mant >> -unb
    fixed = k if not x >> 31 else (-k if k & 0x7FFFFF == 0 else -k - 1)
    w = int(t[fixed & 0x7FFFFF])
    re = ((w >> 23) & 0xFF) + (fixed >> 23)
    if re >= 0xFF:
        return _INF
    if re <= 0:
        return 0
    return (re << 23) | (w & 0x7FFFFF)


def mufu_rcp_bits(x: int, t) -> int:
    """`rcp.approx.ftz.f32`: t[m] = hw(1 / 1.m); the exponent is negated exactly."""
    x &= 0xFFFFFFFF
    s, ax = x & 0x80000000, x & 0x7FFFFFFF
    e = ax >> 23
    if e == 0xFF:
        return _NAN if ax & 0x7FFFFF else s
    if e == 0:
        return s | _INF
    w = int(t[ax & 0x7FFFFF])
    re = ((w >> 23) & 0xFF) - (e - 127)
    if re <= 0:
        return s
    if re >= 0xFF:
        return s | _INF
    return s | (re << 23) | (w & 0x7FFFFF)


def _sqrt_like(x: int, t, rsq: bool) -> int:
    x &= 0xFFFFFFFF
    s, ax = x & 0x80000000, x & 0x7FFFFFFF
    e = ax >> 23
    if e == 0xFF:
        if ax & 0x7FFFFF or s:
            return _NAN
        return 0 if rsq else _INF
    if e == 0:
        return s | _INF if rsq else s
    if s:
        return _NAN
    k = e - 127
    par = k & 1
    w = int(t[(par << 23) | (ax & 0x7FFFFF)])
    half = (k - par) // 2
    return ((((w >> 23) & 0xFF) + (-half if rsq else half)) << 23) | (w & 0x7FFFFF)


def mufu_sqrt_bits(x: int, t) -> int:
    """`sqrt.approx.ftz.f32`: t[p 2^23 + m] = hw(sqrt(2^p 1.m)) for exponent parity p; ±0 and subnormals -> ±0, negatives -> NaN."""
    return _sqrt_like(x, t, False)


def mufu_rsq_bits(x: int, t) -> int:
    """`rsqrt.approx.ftz.f32`: t[p 2^23 + m] = hw(1 / sqrt(2^p 1.m)); ±0 and subnormals -> ±inf, negatives -> NaN."""
    return _sqrt_like(x, t, True)


def _times_2_24(u: int) -> int:
    """u * 2^24 for |u| < 2^-126 (exact: the product is a normal number or zero)."""
    m = u & 0x7FFFFF
    if m == 0:
        return u
    k = m.bit_length()
    return (u & 0x80000000) | ((k + 1) << 23) | ((m << (24 - k)) & 0x7FFFFF)


def _div_full_divisor(b: int) -> int:
    """`div.full.f32`'s scaled divisor: b * 0.25 when |b| > 2^126, b * 2^24 when |b| < 2^-126, else b (exact in both ranges)."""
    b &= 0xFFFFFFFF
    ab = b & 0x7FFFFFFF
    if ab < _MIN_NORMAL:
        return _times_2_24(b)
    if _TWO_POW_126 < ab < _INF:
        return b - (2 << 23)
    return b


# ---------------------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------------------


@primitive("MufuEx2Ftz", 1, [("x", F32)], F32, conformance="exact-model-tested",
           tables=(("ex2", library.LIBRARY),))
def MufuEx2Ftz(x: int) -> int:
    """`ex2.approx.ftz.f32` (MUFU.EX2), measured table + reduction rule (attention-model.md §3.1)."""
    return mufu_ex2_bits(x, _row("ex2"))


@primitive("MufuRcpFtz", 1, [("x", F32)], F32, conformance="exact-model-tested",
           tables=(("rcp", library.LIBRARY),))
def MufuRcpFtz(x: int) -> int:
    """`rcp.approx.ftz.f32` (MUFU.RCP), measured 2^23 mantissa table (attention-model.md §3.1)."""
    return mufu_rcp_bits(x, _row("rcp"))


@primitive("Fa2InvSum", 1, [("sum", F32)], F32, conformance="exact-model-tested",
           tables=(("rcp", library.LIBRARY),))
def Fa2InvSum(s: int) -> int:
    """FA2 `normalize_softmax_lse`: (sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)."""
    if (s & 0x7FFFFFFF) == 0 or f32_is_nan(s):
        return 0x3F800000
    return MufuRcpFtz.evaluate(s)


@primitive("MufuSqrtFtz", 1, [("x", F32)], F32, conformance="exact-model-tested",
           tables=(("sqrt", library.LIBRARY),))
def MufuSqrtFtz(x: int) -> int:
    """`sqrt.approx.ftz.f32` (MUFU.SQRT) as the Triton RMSNorm kernel uses it (rmsnorm-model.md)."""
    return mufu_sqrt_bits(x, _row("sqrt"))


@primitive("DivFullRcp", 1, [("b", F32)], F32, conformance="exact-model-tested",
           tables=(("rcp", library.LIBRARY),))
def DivFullRcp(b: int) -> int:
    """`div.full.f32` reciprocal part as ptxas emits it: range-scale b (|b| > 2^126: *0.25; |b| < 2^-126:
    *2^24) then MUFU.RCP; `DivFullScaleA` produces the matching scaled numerator (rms_triton_model.cpp)."""
    return mufu_rcp_bits(_div_full_divisor(b), _row("rcp"))


@primitive("DivFullScaleA", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def DivFullScaleA(a: int, b: int) -> int:
    """`div.full.f32` numerator range scaling: a*0.25 when |b| > 2^126, a*2^24 when |b| < 2^-126, else a."""
    fb = f32_of(b)
    ab = abs(fb)
    fa = f32_of(a)
    if ab > np.float32(8.507059173023462e37):
        return bits_of(np.float32(fa * np.float32(0.25)))
    if ab < np.float32(1.1754943508222875e-38):
        return bits_of(np.float32(fa * np.float32(16777216.0)))
    return a & 0xFFFFFFFF


@primitive("RsqrtApprox", 1, [("x", F32)], F32, conformance="exact-model-tested",
           tables=(("rsq", library.LIBRARY),))
def RsqrtApprox(x: int) -> int:
    """`rsqrt.approx.f32` (rsqrtf without fast-math): MUFU.RSQ with the subnormal range scaling
    (rms_triton_model.rsqrt_f32; CUDA fused_add_rms_norm / rms_norm kernels)."""
    x &= 0xFFFFFFFF
    if (x & 0x7FFFFFFF) >= _MIN_NORMAL:
        return mufu_rsq_bits(x, _row("rsq"))
    r = mufu_rsq_bits(_times_2_24(x), _row("rsq"))  # then r * 2^12, exact: r is ±inf, NaN or a normal number below 2^63
    return r if (r & _INF) == _INF else r + (12 << 23)


# ---------------------------------------------------------------------------------------------------------
# MUFU.TANH (`tanh.approx.f32`, FA2 `apply_softcap` = `cutlass::fast_tanh`) — measured-shard pilot 2026-09-17 (lane gemma)
#
# The hardware tanh is NOT RN32(tanh) (98 % of words differ on [0.5, 0.75), up to 126 ulp there, 133 ulp in the ov-dense samples),
# is not monotone (1081 adjacent inversions per 2^22 words, drops of up to 38 ulp at mantissa-segment boundaries), and no closed form
# is known — so, like `MufuEx2Ftz` / `MufuRcpFtz`, it is a measured word table.  Unlike ex2/rcp it has no exponent reduction: the
# table spans every f32 word between the small-|x| identity region and the saturation point, hence SHARDS of contiguous word ranges
# (measured by `measure_tanh.py`, whose sha256 the manifest records; its last copy in this repository,
# `integrations/vllm/verity_capture/bench/measure_tanh.py`, not that revision, was deleted in ca5d65e8e; format
# `verity/mufu-tanh-shard/xz-blocks/v1`: xz blocks of int8 first differences of an injective ordered image of the output words,
# ~0.005 B/word).  The evaluator answers
#   (a) NaN inputs and the closed regions in `MUFU_TANH_RULES` (each established by an EXHAUSTIVE scan of the whole region),
#   (b) every other word from the table: the in-tree `tables/mufu_tanh_sm89.{json,xzblocks}` (manifest with
#       sha512, hardware/software, rules, evidence job ids, a 512-word raw sample; table [0x31000000, 0x41000000) = [2^-29, 8), which
#       covers the table-only region [2^-8, 8) with margin; the manifest is pinned by `MUFU_TANH_TABLE_MANIFEST_SHA512`); negative
#       words via the table's odd symmetry, which is used only where the measurement shows 0 violations over the table's whole range.
# `MufuTanhUnmeasured` is raised only when the table does not cover the word or does not load: manifest missing or not the pinned
# one, table file sha512 != manifest.  Never an approximation.
# Hardware/software of the table: RTX 4090 (sm_89), driver 580.178.04, torch 2.13.0+cu129 / triton 3.7.1
# (`inline_asm_elementwise("tanh.approx.f32 $0, $1;")`); the sub-ranges re-measured on driver 580.126.20 (cp-cov-a) hashed identical.
# The table is conformance evidence for that stack only.
# ---------------------------------------------------------------------------------------------------------

MUFU_TANH_SHARD_FORMAT = "verity/mufu-tanh-shard/xz-blocks/v1"
MUFU_TANH_TABLE_FORMAT = "verity/mufu-tanh-table/v1"
MUFU_TANH_TABLE_DIR = TABLE_DIR
MUFU_TANH_TABLE_MANIFEST = os.path.join(MUFU_TANH_TABLE_DIR, "mufu_tanh_sm89.json")
MUFU_TANH_TABLE_MANIFEST_SHA512 = ("e206d35a2b05819d283694486510e115df650eb775433d1c0f4c66f6d683f701"
                                   "d104fde89ee22232d11b9c143ee66e2c95dbcc695e95c1d121d4ef1b94b40312")  # another table is another MufuTanh version
_F32_SIGN = 0x80000000

# Closed regions of the domain, each established ONLY by an exhaustive measurement of the whole region; None = not established, the
# evaluator raises there.  Evidence (2026-09-17, RTX 4090 sm_89, driver 580.178.04, `measure_tanh.py`; the job summaries were
# lane w-gemma's local output, not kept, and the manifest records each job's id, range and word sha256; every job also ran the
# sign-flipped words and found 0 odd-symmetry violations):
#   identity_below 0x3B800001: job 0917T215010-79fd04 scanned [0x00000000, 0x31000000): 822,083,584 / 822,083,584 words hw(x) == x
#                              (word sha 125078cf…); job 0917T214646-3b1c9a's shard [0x31000000, 0x41000000) has its first hw != x word
#                              at 0x3B800001 (= 2^-8 + ulp) -> hw(x) == x for every |x| <= 2^-8, both signs (±0 included).
#   saturate_from  0x41000000: job 0917T215017-e0d178 scanned [0x41000000, 0x80000000): 1,048,576,001 / 1,048,576,001 non-NaN words
#                              (8.0 .. max finite, +inf) -> 0x3F800000 exactly (word sha aa255aa2…); negatives -> 0xBF800000 by the
#                              0-violation symmetry check.  Below 8.0 no word saturates (last non-one word 0x40FFFFFF, shard 3b1c9a).
#   nan_out        0x7FFFFFFF: job e0d178: all 8,388,607 positive NaN payloads -> 0x7FFFFFFF; job 0917T215355-08991c scanned
#                              [0xFF800001, 0x100000000): all 8,388,607 negative payloads -> 0x7FFFFFFF (word sha 2e0c30da…).
# Between the two rules, [0x3B800001, 0x41000000) is TABLE ONLY (shard 3b1c9a = the in-tree `mufu_tanh_sm89.xzblocks`, 449,932 B,
# sha256 5e261b88…, pulled from cp-cov-d by job 0917T220405-876a84 and re-hashed on the laptop).
MUFU_TANH_RULES: dict = {
    "identity_below": 0x3B800001,   # word w: hw(x) == x (bitwise) for every word 0 <= |x| < w
    "saturate_from": 0x41000000,    # word w: hw(x) == ±1.0 for every word |x| >= w, inf included
    "nan_out": 0x7FFFFFFF,          # the one word every NaN input (any payload, either sign) maps to
}


class MufuTanhUnmeasured(LookupError):
    """`MufuTanh.evaluate` refuses inputs outside the measured shards / established regions (never approximates silently)."""


class _TanhShard:
    """One measured word range: `header` = block index (`verity/mufu-tanh-shard/xz-blocks/v1`), `sym` = the odd-symmetry record of
    the measurement over exactly this range (`{"checked", "violations"}`); the file sha512 is verified against the header at load
    (mismatch -> `MufuTanhUnmeasured`); blocks are decoded on demand."""

    def __init__(self, path: str, header: dict, sym: dict | None, sha_words: str | None = None, job: str | None = None) -> None:
        import hashlib as _hl
        self.path, self.header = path, header
        if header.get("format") != MUFU_TANH_SHARD_FORMAT:
            raise MufuTanhUnmeasured(f"MufuTanh table {path}: unknown format {header.get('format')!r}")
        self.lo, self.hi = int(header["lo"], 16), int(header["hi"], 16)
        self.block_words = int(header["block_words"])
        if not os.path.exists(path):
            raise MufuTanhUnmeasured(f"MufuTanh table file missing: {path}")
        h = _hl.sha512()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        if h.hexdigest() != header["sha512_file"]:
            raise MufuTanhUnmeasured(f"MufuTanh table {path}: file sha512 {h.hexdigest()} != recorded {header['sha512_file']} (corrupt)")
        sym = sym or {}
        self.sym_ok = sym.get("violations") == 0 and sym.get("checked") == self.hi - self.lo
        self.sha_words, self.job = sha_words, job
        self._cache: dict[int, np.ndarray] = {}

    def block(self, k: int) -> np.ndarray:
        import lzma as _lzma
        b = self._cache.get(k)
        if b is None:
            hb = self.header["blocks"][k]
            with open(self.path, "rb") as f:
                f.seek(hb["offset"])
                raw = f.read(hb["bytes"])
            d = np.frombuffer(_lzma.decompress(raw), dtype=np.dtype(hb["dtype"])).astype(np.int64)
            if d.shape[0] != hb["n"] - 1:
                raise ValueError(f"MufuTanh shard {self.path} block {k}: {d.shape[0]} diffs for {hb['n']} words")
            o = np.empty(hb["n"], dtype=np.int64)
            o[0] = hb["first_ordered"]
            np.cumsum(d, out=o[1:])
            o[1:] += hb["first_ordered"]
            b = np.where(o < 0, (-(o + 1)) | _F32_SIGN, o).astype(np.uint32)  # inverse of measure_tanh.ordered_inj (deleted; see above)
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = b
        return b

    def lookup(self, w: int) -> int | None:
        if not self.lo <= w < self.hi:
            return None
        i = w - self.lo
        return int(self.block(i // self.block_words)[i % self.block_words])


def mufu_tanh_table_manifest(path: str = MUFU_TANH_TABLE_MANIFEST) -> dict:
    """The in-tree table's manifest (format `verity/mufu-tanh-table/v1`): digests, hardware/software, rules, evidence job ids, the
    512-word raw sample (`sample.pairs` = [[x_hex, hw_hex], ...]) and the xz-block index (`table`)."""
    import json as _json
    with open(path) as f:
        m = _json.load(f)
    if m.get("format") != MUFU_TANH_TABLE_FORMAT:
        raise MufuTanhUnmeasured(f"MufuTanh manifest {path}: unknown format {m.get('format')!r}")
    return m


@lru_cache(maxsize=None)
def mufu_tanh_shards() -> tuple:
    """The loaded table: the in-tree manifest, which must be the pinned one (`MUFU_TANH_TABLE_MANIFEST_SHA512`).  Raises
    `MufuTanhUnmeasured` when it is missing / not the pinned one / corrupt or does not cover the table-only region
    [identity_below, saturate_from) (a configuration error, not a measurement gap)."""
    import hashlib as _hl
    if not os.path.exists(MUFU_TANH_TABLE_MANIFEST):
        raise MufuTanhUnmeasured(f"MufuTanh table manifest missing: {MUFU_TANH_TABLE_MANIFEST}")
    with open(MUFU_TANH_TABLE_MANIFEST, "rb") as f:
        digest = _hl.sha512(f.read()).hexdigest()
    if digest != MUFU_TANH_TABLE_MANIFEST_SHA512:
        raise MufuTanhUnmeasured(f"MufuTanh table manifest {MUFU_TANH_TABLE_MANIFEST}: sha512 {digest} != pinned {MUFU_TANH_TABLE_MANIFEST_SHA512}")
    man = mufu_tanh_table_manifest()
    t = man["table"]
    s = _TanhShard(os.path.join(MUFU_TANH_TABLE_DIR, t["file"]), t, t.get("odd_symmetry"), t.get("sha256_words_le_u32"),
                   (man.get("evidence") or {}).get("table_job", {}).get("id"))
    lo_rule, hi_rule = MUFU_TANH_RULES["identity_below"], MUFU_TANH_RULES["saturate_from"]
    if lo_rule is not None and hi_rule is not None and not (s.lo <= lo_rule and s.hi >= hi_rule and s.sym_ok):
        raise MufuTanhUnmeasured(f"MufuTanh table [0x{s.lo:08X}, 0x{s.hi:08X}) sym_ok={s.sym_ok} does not close the table-only "
                                 f"region [0x{lo_rule:08X}, 0x{hi_rule:08X}) for both signs")
    return (s,)


def mufu_tanh_coverage() -> dict:
    """What the evaluator can answer right now: table word ranges (with their odd-symmetry verdict) and the established rules."""
    try:
        shards = mufu_tanh_shards()
    except MufuTanhUnmeasured as e:
        return {"shards": [], "rules": {k: (None if v is None else f"0x{v:08X}") for k, v in MUFU_TANH_RULES.items()}, "error": str(e)}
    return {"shards": [{"lo": f"0x{s.lo:08X}", "hi": f"0x{s.hi:08X}", "n_words": s.hi - s.lo, "odd_symmetry_verified": s.sym_ok,
                        "sha256_words": s.sha_words, "path": s.path} for s in shards],
            "rules": {k: (None if v is None else f"0x{v:08X}") for k, v in MUFU_TANH_RULES.items()}}


@primitive("MufuTanh", 1, [("x", F32)], F32,
           conformance="quarantine: measured exact table 2026-09-17 (RTX 4090 sm_89, driver 580.178.04, triton 3.7.1) covering all 2^32 "
                       "words — in-tree quarantine/dense/tables/mufu_tanh_sm89.xzblocks (sha256 5e261b88…) + three exhaustively-scanned "
                       "rules; on-worker conformance job 0917T215619-1af45f 400,077 words 0 mismatches; exercised end-to-end by the "
                       "dense/AttentionSoftcap@v1 difftest vs FA2 (job 0917T223335-9ad747, 25/25 bitwise, driver 595.91.07)",
           tables=(("tanh_mufu", library.LIBRARY),))
def MufuTanh(x: int) -> int:
    """`tanh.approx.f32` (MUFU.TANH) as FA2 `apply_softcap` executes it on sm_89: NaN and the exhaustively scanned regions from
    `MUFU_TANH_RULES`, every other word from the measured table (see the section comment).  Raises `MufuTanhUnmeasured` only when
    the table does not cover the word or does not load (missing, not the pinned one, corrupt)."""
    u = x & 0xFFFFFFFF
    sign, a = u & _F32_SIGN, u & 0x7FFFFFFF
    if a > 0x7F800000:  # NaN input
        r = MUFU_TANH_RULES["nan_out"]
        if r is None:
            raise MufuTanhUnmeasured(f"MufuTanh(0x{u:08X}): NaN inputs not exhaustively measured (only sampled payloads)")
        return r
    w = MUFU_TANH_RULES["saturate_from"]
    if w is not None and a >= w:
        return 0x3F800000 | sign
    w = MUFU_TANH_RULES["identity_below"]
    if w is not None and a < w:
        return u
    shards = mufu_tanh_shards()  # raises MufuTanhUnmeasured itself when the table does not load
    for s in shards:
        y = s.lookup(u)
        if y is not None:
            return y
    if sign:
        for s in shards:
            if s.sym_ok:
                y = s.lookup(a)
                if y is not None:
                    return y ^ _F32_SIGN
    cov = mufu_tanh_coverage()
    raise MufuTanhUnmeasured(f"MufuTanh(0x{u:08X}) = tanh.approx.f32({f32_of(u)!r}): word outside the loaded tables "
                             f"{[(s['lo'], s['hi']) for s in cov['shards']]} and the established regions {cov['rules']}")


# ---------------------------------------------------------------------------------------------------------
# numpy kernels (u32 word arrays in and out)
# ---------------------------------------------------------------------------------------------------------


def _as_f32(w: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(w, dtype=np.uint32).view(np.float32)


def _as_u32(f: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)


def mufu_ex2_words(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint32)
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    neg = (x >> 31).astype(bool)
    mant = ((ax & 0x7FFFFF) | 0x800000).astype(np.int64)
    unb = e - 127
    # K = floor(|x| 2^23); lanes with unb >= 30 are overwritten below, so the left shift is clipped where it would overflow
    k = np.where(unb >= 0, mant << np.clip(unb, 0, 29), mant >> np.clip(-unb, 0, 24))
    fixed = np.where(~neg, k, np.where((k & 0x7FFFFF) == 0, -k, -k - 1))
    t = T[fixed & 0x7FFFFF].astype(np.int64)
    re = ((t >> 23) & 0xFF) + (fixed >> 23)
    out = np.where(re >= 0xFF, _INF, np.where(re <= 0, 0, (np.clip(re, 0, 0xFF) << 23) | (t & 0x7FFFFF)))
    out = np.where(e == 0, _ONE, out)
    out = np.where(unb >= 30, np.where(neg, 0, _INF), out)  # includes ±inf
    out = np.where((e == 0xFF) & ((ax & 0x7FFFFF) != 0), _NAN, out)
    return out.astype(np.uint32)


def mufu_rcp_words(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint32)
    s = (x & 0x80000000).astype(np.int64)
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    r = T[ax & 0x7FFFFF].astype(np.int64)
    re = ((r >> 23) & 0xFF) - (e - 127)
    out = np.where(re <= 0, s, np.where(re >= 0xFF, s | _INF, s | (np.clip(re, 0, 0xFF) << 23) | (r & 0x7FFFFF)))
    out = np.where(e == 0, s | _INF, out)
    out = np.where(e == 0xFF, np.where((ax & 0x7FFFFF) != 0, _NAN, s), out)
    return out.astype(np.uint32)


def _sqrt_like_words(x: np.ndarray, T: np.ndarray, rsq: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint32)
    s = (x & 0x80000000).astype(np.int64)
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    k = e - 127
    par = k & 1
    t = T[(par << 23) | (ax & 0x7FFFFF)].astype(np.int64)
    half = (k - par) // 2
    out = ((((t >> 23) & 0xFF) + (-half if rsq else half)) << 23) | (t & 0x7FFFFF)
    out = np.where(s != 0, _NAN, out)
    out = np.where(e == 0, (s | _INF) if rsq else s, out)
    inf = np.where(s != 0, _NAN, 0 if rsq else _INF)
    out = np.where(e == 0xFF, np.where((ax & 0x7FFFFF) != 0, _NAN, inf), out)
    return (out & 0xFFFFFFFF).astype(np.uint32)


def mufu_sqrt_words(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    return _sqrt_like_words(x, T, False)


def mufu_rsq_words(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    return _sqrt_like_words(x, T, True)


def _small(w: np.ndarray) -> np.ndarray:
    return (w & 0x7FFFFFFF) < _MIN_NORMAL  # |w| < 2^-126


def _times_2_24_words(w: np.ndarray) -> np.ndarray:
    """`_times_2_24` on integers, so that the host's denormals-are-zero mode cannot change it."""
    w = np.asarray(w, dtype=np.uint32)
    m = (w & 0x7FFFFF).astype(np.int64)
    k = np.clip((m.astype(np.float64).view(np.int64) >> 52) - 1022, 1, 23)  # bit_length(m) for m >= 1
    scaled = (w & 0x80000000).astype(np.int64) | ((k + 1) << 23) | ((m << (24 - k)) & 0x7FFFFF)
    return np.where(m == 0, w, scaled).astype(np.uint32)


def div_full_divisor_words(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint32)
    ab = b & 0x7FFFFFFF
    big = (ab > _TWO_POW_126) & (ab < _INF)
    return np.where(_small(b), _times_2_24_words(b), np.where(big, b - np.uint32(2 << 23), b))


def rsqrt_approx_words(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint32)
    small = _small(x)
    r = mufu_rsq_words(np.where(small, _times_2_24_words(x), x), T)
    return np.where(small & ((r & _INF) != _INF), r + np.uint32(12 << 23), r)


def div_full_scale_a_words(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint32)
    ab = np.abs(_as_f32(b))
    big = ab > np.float32(8.507059173023462e37)
    small = ab < np.float32(1.1754943508222875e-38)
    af = _as_f32(a)
    return np.where(big, _as_u32(af * np.float32(0.25)), np.where(small, _as_u32(af * np.float32(16777216.0)), a))


def fa2_inv_sum_words(s: np.ndarray, T: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.uint32)
    ax = s & 0x7FFFFFFF
    return np.where((ax == 0) | (ax > _INF), np.uint32(_ONE), mufu_rcp_words(s, T))


def _register_kernels() -> None:
    every = "every encoding"
    register_kernel(MufuEx2Ftz, lambda x: (mufu_ex2_words(x, table("ex2")), None), domain=every)
    register_kernel(MufuRcpFtz, lambda x: (mufu_rcp_words(x, table("rcp")), None), domain=every)
    register_kernel(Fa2InvSum, lambda s: (fa2_inv_sum_words(s, table("rcp")), None), domain=every)
    register_kernel(MufuSqrtFtz, lambda x: (mufu_sqrt_words(x, table("sqrt")), None), domain=every)
    register_kernel(DivFullRcp, lambda b: (mufu_rcp_words(div_full_divisor_words(b), table("rcp")), None), domain=every)
    register_kernel(DivFullScaleA, lambda a, b: (div_full_scale_a_words(a, b), None), domain=every)
    register_kernel(RsqrtApprox, lambda x: (rsqrt_approx_words(x, table("rsq")), None), domain=every)


_register_kernels()
