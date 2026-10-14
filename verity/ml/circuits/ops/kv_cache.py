"""The scalar conversion used when vLLM appends BF16 keys and values to an E4M3 KV cache.

``KvCacheAppendE4m3_v1`` is one cache byte.  vLLM d9105ea80's
``reshape_and_cache_flash_kernel`` applies it independently to every key and
value element: widen BF16 exactly, divide by the per-tensor FP32 key or value
scale with ``div.rn.f32``, then convert with
``cvt.rn.satfinite.e4m3x2.f32``.  Cache addressing is wiring, not arithmetic.
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite

from . import fp32 as F
from . import prims as P

__all__ = ["KvCacheAppendE4m3", "kv_cache_append_e4m3_word"]


def kv_cache_append_e4m3_word(x: int, scale: int) -> int:
    """CPU reference for one vLLM FP8 KV-cache output byte.

    ``x`` is a BF16 encoding and ``scale`` an FP32 encoding.  The function is
    total on encodings: FP32 division supplies the GPU's canonical NaN,
    infinities saturate to signed 448, and every NaN converts to ``0x7f``.
    """
    widened = P.Bf16ToF32.evaluate(x)
    scaled = F.F32DivV2.evaluate(widened, scale)
    return P.F32ToE4m3Sat.evaluate(scaled)


@composite(
    "KvCacheAppendE4m3",
    1,
    [],
    lambda _S: ((("x", P.BF16), ("scale", P.F32)), P.E4M3),
    conformance="exact-model-tested; vLLM d9105ea80 FlashInfer reshape_and_cache_flash on the RTX PRO 6000 "
                "(run r20261014-060040-6de8): 43,008/43,008 cache bytes bit-exact over one prefill and three decode appends",
)
def KvCacheAppendE4m3(B, _S, x, scale):
    """One BF16 key/value element appended to vLLM's per-tensor E4M3 KV cache.

    The scale is the dequantization scale: the stored byte is
    ``satfinite_e4m3_rne(div_rne(f32(x), scale))``.  In particular, vLLM
    divides by the scale; it does not multiply by a precomputed reciprocal.
    ``BaseKVCacheMethod`` loads separate key and value scales from the
    checkpoint (or duplicates its legacy combined scale); absent scales
    default independently to FP32 ``1.0``.
    """
    widened = B.call(P.Bf16ToF32, x)
    scaled = B.call(F.F32DivV2, widened, scale)
    return B.call(P.F32ToE4m3Sat, scaled)
