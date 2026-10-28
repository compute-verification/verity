"""The numpy twins of `registry.dsv4_indexer`'s state score and inverse-RoPE quantisation (vLLM d9105ea80's `save_partial_states` and
`fused_inv_rope_fp8_quant`): `score_twin` (`Dsv4CompressorScore_v1{W}`) and `heads_twin` (`Dsv4InvRopeQuantHead_v1{CHUNKS,QG,ROPE}`),
each the Definition's own body (`compressor_score`, `inv_rope`, `inv_rope_quant`) run over numpy arrays of words by `NumpyOps`, whose
every step is its `dsv4_indexer.WORD` primitive word for word.  The Definitions are the reference: tests/properties/
test_dsv4_indexer_capture.py holds each step to its primitive and each twin to its Definition, and the GPU capture
tests/properties/dsv4_indexer_capture_gpu.py holds the kernels to the twins on every captured row."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from verity.ml.circuits.ops import mufu as MU
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics.fp8_twins import np_f32_to_e4m3_sat
from verity.ml.definitions.registry import dsv4_indexer as I

NAN = np.uint32(0x7FFFFFFF)


def _f(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint32).view(np.float32)


def _w(f) -> np.ndarray:
    return np.asarray(f, dtype=np.float32).view(np.uint32)


def _is_nan(u) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint32)
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)


def _gpu(u) -> np.ndarray:
    """Every NaN the GPU's 0x7FFFFFFF."""
    u = np.asarray(u, dtype=np.uint32)
    return np.where(_is_nan(u), NAN, u)


class NumpyOps:
    """`dsv4_indexer.WORD`'s steps on numpy arrays of words, so the Definitions' bodies (`compress_column`, `norm_rope`, `k_quant`,
    `inv_rope`, `inv_rope_quant`) run vectorised over many instances: the f32 arithmetic of `F32AddV2_v1` .. `F32FmaFtzV2_v1` (numpy's
    IEEE single rounding, `derived_rows.fma_rn`'s exact FMA, each NaN 0x7FFFFFFF), MUFU and div.full through core's registered kernels
    (`verity.ml.circuits.ops.mufu`, exact on every encoding), the rest on the encodings."""

    @staticmethod
    def word(bits: int):
        return np.uint32(bits)

    @staticmethod
    def neg(x):
        return np.asarray(x, dtype=np.uint32) ^ np.uint32(0x80000000)

    @staticmethod
    def add(a, b):
        return _gpu(_w(_f(a) + _f(b)))

    @staticmethod
    def sub(a, b):
        return _gpu(_w(_f(a) - _f(b)))

    @staticmethod
    def mul(a, b):
        return _gpu(_w(_f(a) * _f(b)))

    @staticmethod
    def fma(a, b, c):
        return _gpu(DR.fma_rn(a, b, c))

    @staticmethod
    def fma_ftz(a, b, c):
        return _gpu(DR.fma_ftz(a, b, c))

    @staticmethod
    def rcp(b):
        return MU.mufu_rcp_words(MU.div_full_divisor_words(np.atleast_1d(np.asarray(b, dtype=np.uint32))), MU.table("rcp"))

    @staticmethod
    def scale_a(a, b):
        return MU.div_full_scale_a_words(*np.broadcast_arrays(np.atleast_1d(np.asarray(a, np.uint32)), np.atleast_1d(np.asarray(b, np.uint32))))

    @staticmethod
    def rsqrt(x):
        return MU.rsqrt_approx_words(np.atleast_1d(np.asarray(x, dtype=np.uint32)), MU.table("rsq"))

    @staticmethod
    def ex2(x):
        return MU.mufu_ex2_words(np.atleast_1d(np.asarray(x, dtype=np.uint32)), MU.table("ex2"))

    @staticmethod
    def fabs(x):
        return np.asarray(x, dtype=np.uint32) & np.uint32(0x7FFFFFFF)

    @staticmethod
    def _maxmin(a, b, larger: bool):
        """`max.f32` / `min.f32`: a NaN yields the other operand (two give 0x7FFFFFFF), max(+0, -0) = +0 and min(+0, -0) = -0."""
        a, b = np.broadcast_arrays(np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32))
        fa, fb = _f(a), _f(b)
        out = np.where(fa > fb if larger else fa < fb, a, b)
        out = np.where(fa == fb, a & b if larger else a | b, out)
        na, nb = _is_nan(a), _is_nan(b)
        return np.where(na & nb, NAN, np.where(na, b, np.where(nb, a, out)))

    def fmaxf(self, a, b):
        return self._maxmin(a, b, True)

    def fminf(self, a, b):
        return self._maxmin(a, b, False)

    @staticmethod
    def e4m3(x):
        return np_f32_to_e4m3_sat(_f(x)).astype(np.uint32)

    @staticmethod
    def iadd(a, b):
        return ((np.asarray(a, dtype=np.uint64) + np.asarray(b, dtype=np.uint64)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    @staticmethod
    def iand(a, b):
        return np.asarray(a, dtype=np.uint32) & np.asarray(b, dtype=np.uint32)

    @staticmethod
    def isub(a, b):
        return ((np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)) & 0xFFFFFFFF).astype(np.uint32)

    @staticmethod
    def i2f(a):
        return _w(np.asarray(a, dtype=np.uint32).view(np.int32).astype(np.float32))

    @staticmethod
    def ceil(x):
        """`cvt.rpi.ftz.f32.f32`: a subnormal flushed to its signed zero, then rounded up (-1 < x < 0 gives -0), NaN 0x7FFFFFFF."""
        return _gpu(_w(np.ceil(_f(DR.ftz(x)))))

    @staticmethod
    def cvt(w):
        return np.asarray(w, dtype=np.uint32) << np.uint32(16)

    @staticmethod
    def bf16(x):
        return DR.f2fp_bf16(np.atleast_1d(np.asarray(x, dtype=np.uint32)))

    @staticmethod
    def fsel(c, a, b):
        return np.where(np.asarray(c, dtype=np.uint32) & 1, np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32))

    @staticmethod
    def gt(a, b):
        return (_f(a) > _f(b)).astype(np.uint32)

    @staticmethod
    def ile(a, b):
        return (np.asarray(a, dtype=np.uint32).view(np.int32) <= np.asarray(b, dtype=np.uint32).view(np.int32)).astype(np.uint32)


OPS = NumpyOps()


def score_twin(score: np.ndarray, ape: np.ndarray) -> np.ndarray:
    """`Dsv4CompressorScore_v1{W}` over rows: score and ape (n, W) f32 words -> (n, W)."""
    with np.errstate(all="ignore"):
        return np.asarray(I.compressor_score(OPS, None, [score], [ape])[0], dtype=np.uint32)


def heads_twin(x: np.ndarray, cs: np.ndarray, QG: int, ROPE: int) -> tuple[np.ndarray, np.ndarray]:
    """`Dsv4InvRopeQuantHead_v1{CHUNKS,QG,ROPE}` over n heads: x (n, CHUNKS * QG) bf16 words, cs (n, ROPE) f32 words -> (n, CHUNKS
    * QG) e4m3 bytes and (n, CHUNKS) f32 scales."""
    I.check_head_statics(QG, ROPE)
    x, cs = np.asarray(x, dtype=np.uint16), np.asarray(cs, dtype=np.uint32)
    n, hd = x.shape
    if hd % QG or cs.shape != (n, ROPE):
        raise ValueError(f"heads_twin: x {x.shape} and cs {cs.shape} are not heads of whole groups of {QG} and {ROPE} rotary words")
    with np.errstate(all="ignore"):
        words = [OPS.cvt(x[:, i]) for i in range(hd - ROPE)]
        words += I.inv_rope(OPS, SimpleNamespace(ROPE=ROPE), [x[:, i] for i in range(hd - ROPE, hd)], [cs[:, j] for j in range(ROPE)])
        q, s = [], []
        for g in range(hd // QG):
            out = I.inv_rope_quant(OPS, SimpleNamespace(N=QG), words[g * QG:(g + 1) * QG])
            q += out[:QG]
            s.append(out[QG])
    return (np.stack([np.broadcast_to(v, (n,)) for v in q], axis=1).astype(np.uint8),
            np.stack([np.broadcast_to(v, (n,)) for v in s], axis=1).astype(np.uint32))
