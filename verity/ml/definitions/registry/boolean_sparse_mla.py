"""`sparse_mla`'s FA3 sparse MLA attention on bits: version 2 of each Definition, its word body line for line over Boolean
Definitions (the bodies are `sparse_mla`'s, written once over `sparse_mla.Steps`), so each one's word view is its `_v1` at the
same statics with the Boolean ``DOT`` and ``INV`` read as their word views.

    SparseMlaBlock_v2{DR,DV,NVIS,FIRST,BN,DOT,SCALE}   DOT: `gemm.HopperBF16WgmmaDot16_v2` (FA3's wgmma k16 step)
    SparseMlaHead_v2{T,DR,DV,BN,DOT,INV,SCALE}         INV: `boolean_attention.Fa3InvSum_v2`
    SparseMla_v2{T,NH,DR,DV,BN,DOT,INV,SCALE}

The callees are `boolean_attention`'s: `DotBf16_v3{K = DR + DV, DOT}` and `DOT`, `F2fpBf16_v2`, the FTZ f32 operations,
`F32Max_v2`, `GuardNegInfZero_v2`, `MufuEx2Ftz_v2` and the inverse sum; a constant word is constant bits.  No table or lookup gate.
Each is one traced circuit (`boolean_dsa.flat`), so the query every score reads and the rescale every output reads are decoded
once.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.definitions.registry import sparse_mla as M
from verity.ml.definitions.registry.boolean_attention import BF16, F32, Fa3InvSum, _Bits
from verity.ml.definitions.registry.boolean_dsa import flat

BITS = dict(max=BA.F32Max, guard=BA.GuardNegInfZero, ex2=BM.MufuEx2Ftz, mul=BA.F32MulFtz, sub=BA.F32SubFtz, fmasub=BA.F32FmaSubFtz,
            fma=BA.F32FmaFtz, add=BA.F32AddFtz, cvt=BG.F2fpBf16)

#: GLM-5.3 at TP8 on bits: `sparse_mla.SPARSE_MLA_GLM53` with the Boolean DOT and INV
SPARSE_MLA_GLM53 = {**M.SPARSE_MLA_GLM53, "DOT": BG.HopperBF16WgmmaDot16, "INV": Fa3InvSum}
_LIFTED = ("DOT", "INV")


def bit_steps(B) -> M.Steps:
    c = _Bits(B)
    return M.Steps(B, BITS, lambda bits: c.word(bits, 32), lambda: c.word(0, 16), 16, BF16, F32)


def _block(B, S, q, kv, m_prev=None, l_prev=None, o_prev=None):
    """`SparseMlaBlock_v1`'s body on bits: the S chain `DotBf16_v3{K = DR + DV, DOT}`, the P V step `DOT`."""
    M.check_block_statics(S.DR, S.DV, S.NVIS, S.BN)
    return M.block(bit_steps(B), S, bind(BG.DotBf16, K=S.DR + S.DV, DOT=S.DOT), S.DOT, q, kv, m_prev, l_prev, o_prev)


SparseMlaBlockV2 = flat("SparseMlaBlock", 2, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"],
                        lambda S: M._block_type(S, BF16, F32), _block, M.SparseMlaBlock, lifted=_LIFTED)


def _head(B, S, q_nope, q_rope, kv):
    """`SparseMlaHead_v1`'s body on bits over `SparseMlaBlock_v2`, epilogue `INV`."""
    def block_of(nvis, first):
        return bind(SparseMlaBlockV2, DR=S.DR, DV=S.DV, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return M.head(bit_steps(B), S, block_of, S.INV, q_nope, q_rope, kv)


SparseMlaHeadV2 = flat("SparseMlaHead", 2, ["T", "DR", "DV", "BN", "DOT", "INV", "SCALE"], lambda S: M._head_type(S, BF16), _head,
                       M.SparseMlaHead, lifted=_LIFTED)


def _mla(B, S, q_nope, q_rope, kv):
    """`SparseMla_v1`'s body on bits over `SparseMlaHead_v2`."""
    hd = bind(SparseMlaHeadV2, T=S.T, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    return B.batch(hd, q_nope, q_rope, kv, axes=(0, 0, None))


SparseMlaV2 = flat("SparseMla", 2, ["T", "NH", "DR", "DV", "BN", "DOT", "INV", "SCALE"], lambda S: M._mla_type(S, BF16), _mla,
                   M.SparseMla, lifted=_LIFTED)

BOOLEAN_SPARSE_MLA = [SparseMlaBlockV2, SparseMlaHeadV2, SparseMlaV2]
