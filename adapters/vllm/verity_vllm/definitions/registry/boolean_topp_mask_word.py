"""The top-p keep word on bits: `sampling.topp_mask_word`'s primitive at every vocabulary size, made on demand as it is.

    TopPKeepWord_v3{V,S}    σ of `dead_free.TopPKeepWord_v2{V,S}`, word view: the keep bits at the constant split count S
    TopPMaskWordx{V}_v2     (logits: Array<V, Array<32, Value<1>>>, top_p: Array<32, Value<1>>, splits: Array<32, Value<1>>)
                            -> Array<V, Value<1>>; word view: TopPMaskWordx{V}_v1

`TopPMaskWordx{V}_v2` runs `TopPKeepWord_v3{V,S}` at each launched split count S and keeps lane v where some S has
``splits == S`` and its keep bit; at most one S equals ``splits``, and none for a value outside {1, 2, 4, 8, 16, 32}, whose word is
0 as v1's is.  Its body is that word composite's (`_splits_word`) lifted by `lift_body`.
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.boolean.versions import lift_body, sigma_lift
from verity.circuits.ops.prims import const
from verity.circuits.ops.scalar import BitAnd, BitOr, I32Eq
from verity.circuits.population import word_of
from verity.circuits.types import Array, Value

from verity_vllm.definitions.registry import dead_free as DF
from verity_vllm.definitions.registry import sampling as SM
from verity_vllm.definitions.registry.prims import F32, I32

TopPKeepWord = sigma_lift(DF.TopPKeepWordV2, 3)

_MASK_WORDS: dict[int, CompositeDefinition] = {}


def _splits_word_body(B, st, logits, top_p, splits):
    keep = None
    for S in SM.SPLIT_COUNTS:
        k = B.call(bind(DF.TopPKeepWordV2, V=st.V, S=S), logits, top_p)
        k = B.batch(BitAnd, k, B.call(I32Eq, splits, B.call(const(32, S))), axes=(0, None))
        keep = k if keep is None else B.batch(BitOr, keep, k, axes=(0, 0))
    return B.call(word_of(int(st.V)), keep)


_splits_word = CompositeDefinition(
    "TopPMaskWordSplits", 1, ("V",),
    lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32), ("splits", I32)), Value(st.V)), _splits_word_body,
    doc="TopPMaskWordx{V}_v1 from TopPKeepWord_v2{V,S} at each launched split count S", conformance="exact-model-tested",
    register=False)


def topp_mask_word(vocab: int) -> CompositeDefinition:
    """`TopPMaskWordx{V}_v2`: `TopPMaskWordx{V}_v1` on bits."""
    V = int(vocab)
    if V < 1:
        raise ValueError(f"a vocabulary has at least one token, not {vocab}")
    d = _MASK_WORDS.get(V)
    if d is None:
        word = SM.topp_mask_word(V)
        params, ret = word.signature()
        sig = (tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret))
        d = _MASK_WORDS[V] = CompositeDefinition(
            f"TopPMaskWordx{V}", 2, (), lambda st: sig,
            lambda B, st, *args: lift_body(bind(_splits_word, V=V))(B, st, *args),
            doc=f"TopPMaskWordx{V}_v1 on bits: TopPKeepWord_v3{{V={V},S}} at each launched split count S, lane v kept where "
                "splits == S and its keep bit", conformance="exact-model-tested")
        d.word = word  # type: ignore[attr-defined]
    return d


__all__ = ["TopPKeepWord", "topp_mask_word"]
