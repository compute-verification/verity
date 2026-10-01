"""`registry.moe`'s element-wise Definitions as Boolean Definitions: the next version of each word Definition, the same function on
bits.

    MoeSumCoordinate_v2{TOPK}   (col: Array<TOPK, Array<16, Value<1>>>) -> Array<16, Value<1>>        word view: MoeSumCoordinate_v1{TOPK}
    MoeSum_v2{TOPK,H}           (y: Array<TOPK, Array<H, Array<16, Value<1>>>>) -> Array<H, ...>    word view: MoeSum_v1{TOPK,H}

`MoeSumCoordinate_v2{TOPK}` is traced for each TOPK (`elementwise.specialized`) from ``bf16_rn(((0 + f32(col[0])) + f32(col[1])) +
...)`` through `verity.ml.boolean.fp`, the first add written as what it does (``+0 + x`` is x but for -0, which becomes +0). Its
adds make the canonical NaN where `F32Add_v1` keeps an operand's payload; both are NaN on the same inputs, and the cast ends every
NaN in F2FP's 0x7FFF, so the word is `MoeSumCoordinate_v1`'s. `MoeSum_v2` is `MoeSum_v1`'s body on bits: one coordinate per column of the
TOPK expert outputs. No sub-Call is a word Definition.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.types import Array
from verity.ml.boolean import elementwise as EW
from verity.ml.boolean import fp
from verity.ml.boolean.forms import or_reduce
from verity_vllm.program.registry import moe as M
from verity_vllm.program.registry import prims as P

__all__ = ["MoeSum", "MoeSumCoordinate", "moe_sum"]

BF16 = Array(16, BL.BIT)


def _on_bits(name: str, statics: tuple[str, ...], signature, body, word) -> CompositeDefinition:
    """The Boolean composite ``name_v2`` with body ``body``, whose word view is ``word`` at the same statics."""
    d = CompositeDefinition(name, 2, statics, signature, body, doc=body.__doc__ or "", conformance="exact-model-tested")
    d.word = lambda b: bind(word, **{k: b[k] for k in statics})  # type: ignore[attr-defined]
    return d


def moe_sum(C, *col):
    """`MoeSumCoordinate_v1`: ``bf16_rn(sum_k f32(col[k]))``, ascending k from +0, every NaN 0x7FFF."""
    c0 = col[0]
    acc = fp.bf16_to_f32(list(c0[:15]) + [C.AND(c0[15], or_reduce(C, c0[:15]))])
    for c in col[1:]:
        acc = fp.f32_add(C, acc, fp.bf16_to_f32(c))
    return fp.f32_to_bf16(C, acc, P.BF16_NAN_F2FP)


MoeSumCoordinate = EW.specialized("MoeSumCoordinate", 2, ["TOPK"], lambda S: moe_sum,
                                  lambda S: ((("col", Array(S.TOPK, BF16)),), BF16),
                                  doc="moe_sum_vec_kernel's coordinate on bits: MoeSumCoordinate_v1, bf16_rn(sum_k f32(col[k])), "
                                      "NaN 0x7FFF", word=lambda b: bind(M.MoeSumCoordinate, TOPK=b["TOPK"]))


def _moe_sum(B, S, y):
    """`out[h] = MoeSumCoordinate_v2{TOPK}(y[:, h])` on bits."""
    return B.batch(bind(MoeSumCoordinate, TOPK=S.TOPK), y, axes=(1,))


MoeSum = _on_bits("MoeSum", ("TOPK", "H"), lambda S: ((("y", Array(S.TOPK, Array(S.H, BF16))),), Array(S.H, BF16)), _moe_sum,
                  M.MoeSum)
