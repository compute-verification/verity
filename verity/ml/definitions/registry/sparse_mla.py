"""[GLM-5.3] Sparse MLA attention over the indexer's selected keys, as the H200 runs it: vLLM d9105ea80's `FLASH_ATTN_MLA_SPARSE`
(vllm-flash-attn 506341a1, FA3) on sm_90, bf16 KV cache.

The backend (vllm/platforms/cuda.py:136-148): on capability 9 the sparse candidates are `FLASH_ATTN_MLA_SPARSE`, `FLASHMLA_SPARSE`,
then `FLASHINFER_MLA_SPARSE_SM90` (first only when the MLA head size is 512; GLM-5.3's is 576).  `FLASH_ATTN_MLA_SPARSE` takes a
bf16 or fp16 KV cache, no decode context parallelism and a model with `index_topk` (flashattn_mla_sparse.py:75-105), so the
default serve picks it; FlashInfer's sparse path is taken only with an fp8 KV cache or DCP refused by both (or named with
`--attention-backend`), and has no Definition here.

`forward_mqa` (flashattn_mla_sparse.py:201-248) calls `flash_attn_varlen_func(q=q_rope, k=k_pe, v=kv_c, q_v=q_nope,
max_seqlen_q=1, seqused_k=valid_counts, block_table=topk_indices, softmax_scale, causal=True, fa_version=3)`: each query token is
one row over its `valid_counts` selected cache rows (page size 1, in the top-k's order; `-1` entries last), each row
[kv_c 512 | k_pe 64] (kv_rows, flash_attn's `k = kv_rows[:, 512:]`, `v = kv_rows[:, :512]`).  FA3 (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
`mma()`, hdim 64 with hdimv 512, kBlockN 64) computes S = Q K^T then += Qv V^T into one accumulator: per key a bf16 k16 `wgmma` chain
over [q_rope | q_nope] . [k_pe | kv_c] from 0 (4 then 32 steps); the online softmax and P V as `fa3_check_inf.AttnBlock_v4`
with O of width 512, the blocks visited last to first; with one query row at the end of the T keys the causal tile's
`n_block_min_causal_local_mask` is (T - 1) / 64, so only the first visited block runs `Check_inf`.  The epilogue is
`AttentionHead_v4`'s: `inv = Fa3InvSum((l0 + l2) + (l1 + l3))`, `out = bf16(o * inv)`.  `scale_softmax_log2 = float(softmax_scale)
* M_LOG2E` in double, narrowed to f32 (flash_api.cpp `set_params_fprop`); GLM-5.3's softmax_scale is (192 + 64)^-0.5 = 1/16.

Registered functions (all `Name_v1`):
  SparseMlaBlock{DR,DV,NVIS,FIRST,BN,DOT,SCALE}   (q: DR+DV bf16, kv: BN x (DV+DR) bf16, m, l[4], o[DV]) -> (m, l[4], o[DV]);
                                                  the first visited block (FIRST) takes q and kv only
  SparseMlaHead{T,DR,DV,BN,DOT,INV,SCALE}         (q_nope: DV bf16, q_rope: DR bf16, kv: T x (DV+DR) bf16) -> DV bf16
  SparseMlaHeadBlock{DR,DV,NVIS,FIRST,BN,DOT,SCALE}  (q_nope, q_rope, kv: NVIS x (DV+DR), m, l[4], o[DV]) -> (m, l[4], o[DV]);
                                                  one block of a head's chain, a unit of its own (FIRST takes no state)
  SparseMlaHeadEnd{DV,INV}                        (l[4], o[DV]) -> DV bf16; the head's epilogue, the chain's last unit
  SparseMla{T,NH,DR,DV,BN,DOT,INV,SCALE}          (q_nope: NH x DV, q_rope: NH x DR, kv: T x (DV+DR)) -> NH x DV bf16

T is the row's valid count (at most `index_topk`), kv its selected rows in order; `SPARSE_MLA_GLM53` holds GLM-5.3's statics.
A head is cut exactly into its key blocks (`visits`), each a `SparseMlaHeadBlock` carrying the online softmax's state to the next,
and `SparseMlaHeadEnd`, its epilogue: at GLM-5.3's statics a full block fits a Flock block and a head of two blocks does not, so
serving stages a `SparseMla` row as that chain (`head_chain`, ``CUTS``).
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import const
from verity.ml.definitions.registry import hopper
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import _lane
from verity.ml.definitions.registry.prims import BF16, F32

_CONF = ("model-untested: read from vllm-flash-attn 506341a1's FA3 mainloop and softmax.h (the module docstring's files) at hdim "
         "64 / hdimv 512; tests/properties/glm53_dsa_capture_gpu.py is its hardware check")

#: GLM-5.3 at TP8: 8 of the 64 heads per rank, rope 64, latent 512, FA3's kBlockN at hdim 64 / hdimv 512, scale 1/16
SPARSE_MLA_GLM53 = {"NH": 8, "DR": 64, "DV": 512, "BN": 64, "DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum,
                    "SCALE": 0.0625}


def scale_log2_bits(scale: float) -> int:
    """`float(softmax_scale) * M_LOG2E` in double, narrowed to f32."""
    return P.bits_of(np.float32(float(np.float32(scale)) * 1.4426950408889634))


class Steps:
    """The callees of the bodies: ``defs[name]`` by attribute (``steps.mul(a, b)``), ``word(bits)`` a constant f32 word (each once),
    ``zero16()`` the bf16 zero, ``leaf`` the leaves of one bf16 (1 on words, 16 on bits)."""

    def __init__(self, B, defs: dict, word, zero16, leaf: int, bf16=BF16, f32=F32):
        self.B, self._defs, self._word, self._zero16, self.leaf, self.bf16, self.f32 = B, defs, word, zero16, leaf, bf16, f32
        self._words: dict = {}
        self._z16 = None

    def word(self, bits: int):
        if bits not in self._words:
            self._words[bits] = self._word(bits)
        return self._words[bits]

    def zero16(self):
        if self._z16 is None:
            self._z16 = self._zero16()
        return self._z16

    def __getattr__(self, name: str):
        d = self._defs[name]
        return lambda *a: self.B.call(d, *a)


WORD = dict(max=P.F32Max, guard=P.GuardNegInfZero, ex2=P.MufuEx2Ftz, mul=P.F32MulFtz, sub=P.F32SubFtz, fmasub=P.F32FmaSubFtz,
            fma=P.F32FmaFtz, add=P.F32AddFtz, cvt=P.F2fpBf16)


def word_steps(B) -> Steps:
    return Steps(B, WORD, lambda bits: B.call(const(32, bits)), lambda: B.call(const(16, 0)), 1)


def check_block_statics(DR: int, DV: int, NVIS: int, BN: int) -> None:
    if DR % 16 or DV % 16 or BN % 16 or not 1 <= NVIS <= BN:
        raise ValueError(f"SparseMlaBlock: DR={DR}, DV={DV}, BN={BN} must be multiples of 16 and 1 <= NVIS={NVIS} <= BN")


def block(st: Steps, S, dot, DOT, q, kv, m_prev=None, l_prev=None, o_prev=None):
    """`SparseMlaBlock`'s body over ``st``'s steps: ``dot`` the S chain (`DotBf16{K = DR + DV}`), ``DOT`` the P V step."""
    B, DR, DV, NVIS, FIRST, BN = st.B, S.DR, S.DV, S.NVIS, S.FIRST, S.BN
    check_block_statics(DR, DV, NVIS, BN)
    KW, L = DR + DV, st.leaf
    s = [B.call(dot, q, Coll(Array(KW, st.bf16), concat([kv[c][DV:KW].refs, kv[c][0:DV].refs]))) for c in range(NVIS)]
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = st.max(bmax, s[c])
    m_new = bmax if FIRST else st.max(m_prev, bmax)
    m_use = st.guard(m_new) if FIRST else m_new
    scale_log2 = st.word(scale_log2_bits(S.SCALE))
    if not FIRST:
        scores_scale = st.ex2(st.mul(st.sub(m_prev, m_use), scale_log2))
    max_scaled = st.mul(m_use, scale_log2)
    p = [st.ex2(st.fmasub(s[c], scale_log2, max_scaled)) for c in range(NVIS)]
    lanes = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        first = p[cols[0]] if cols else st.word(0)
        acc = first if FIRST else st.fma(scores_scale, l_prev[ln], first)
        for c in cols[1:]:
            acc = st.add(acc, p[c])
        lanes.append(acc)
    pb = [st.cvt(x) for x in p]
    o = [st.word(0) for _ in range(DV)] if FIRST else [st.mul(o_prev[d], scores_scale) for d in range(DV)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = Coll(Array(16, st.bf16), concat([(pb[k0 + j] if k0 + j < NVIS else st.zero16()).refs for j in range(16)]))
        vt = Coll(Array(DV, Array(16, st.bf16)), concat([strided_view(kv, ((k0 * KW) + d) * L, L, KW * L, 1, 16 * L)
                                                          for d in range(DV)]))
        o = list(B.batch(DOT, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


def _block_type(S, bf16=BF16, f32=F32):
    KW = S.DR + S.DV
    state = () if S.FIRST else (("m_prev", f32), ("l_prev", Array(4, f32)), ("o_prev", Array(S.DV, f32)))
    return ((("q", Array(KW, bf16)), ("kv", Array(S.BN, Array(KW, bf16)))) + state,
            Tuple(f32, Array(4, f32), Array(S.DV, f32)))


@composite("SparseMlaBlock", 1, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"], _block_type, conformance=_CONF)
def SparseMlaBlock(B, S, q, kv, m_prev=None, l_prev=None, o_prev=None):
    """One BN-key block of FA3's MLA online softmax for one query row (`AttnBlock_v4`'s steps): S[c] = the bf16 k16 chain over
    [q_rope | q_nope] . [k_pe | kv_c] of row c (`DotBf16_v2{K = DR + DV}`), the row max, the running max guarded
    (`GuardNegInfZero`) only in the first visited block (FIRST: the one `Check_inf` iteration), the rescale, P = `ex2(s * scale -
    max * scale)`, the per-lane sums, and O = O * rescale + bf16(P) kv_c over BN/16 ascending k-steps (`DOT`)."""
    check_block_statics(S.DR, S.DV, S.NVIS, S.BN)
    return block(word_steps(B), S, bind(DotBf16V2, K=S.DR + S.DV, DOT=S.DOT), S.DOT, q, kv, m_prev, l_prev, o_prev)


def visits(T: int, BN: int) -> list[tuple[int, int]]:
    """A head's key blocks over its T keys in FA3's order, last to first: each one's keys [lo, hi); only the first visited one
    can be partial."""
    if T < 1:
        raise ValueError(f"SparseMlaHead: T={T} selected keys")
    return [(b * BN, min(T, (b + 1) * BN)) for b in range((T + BN - 1) // BN - 1, -1, -1)]


def _padded(st: Steps, kv, nvis: int, BN: int, KW: int):
    """A block's ``nvis`` key rows as the BN rows FA3 loads: the out-of-bounds rows cleared to zero (their columns are masked and
    never read; their zero P times a zero V leaves O unchanged)."""
    if nvis == BN:
        return kv
    row = Coll(Array(KW, st.bf16), concat([st.zero16().refs] * KW))
    return Coll(Array(BN, Array(KW, st.bf16)), concat([kv.refs] + [row.refs] * (BN - nvis)))


def head(st: Steps, S, tile_of, INV, q_nope, q_rope, kv):
    """`SparseMlaHead`'s body: ``tile_of(nvis, first)`` the bound block, ``INV`` the epilogue's inverse."""
    B, T, DR, DV, BN = st.B, S.T, S.DR, S.DV, S.BN
    KW = DR + DV
    q = Coll(Array(KW, st.bf16), concat([q_rope.refs, q_nope.refs]))
    m = l = o = None
    for j, (lo, hi) in enumerate(visits(T, BN)):
        kvb = _padded(st, kv[lo:hi], hi - lo, BN, KW)
        if j == 0:
            r = B.call(tile_of(hi - lo, True), q, kvb)
        else:
            r = B.call(tile_of(hi - lo, False), q, kvb, m, l, o)
        m, l, o = r[0], r[1], r[2]
    return head_end(st, INV, l, o)


def head_end(st: Steps, INV, l, o):
    """A head's epilogue on its last block's lanes and O: bf16(o * INV((l0 + l2) + (l1 + l3)))."""
    B = st.B
    inv = B.call(INV, st.add(st.add(l[0], l[2]), st.add(l[1], l[3])))
    return B.batch(st._defs["cvt"], B.batch(st._defs["mul"], o, inv, axes=(0, None)), axes=(0,))


def _head_type(S, bf16=BF16):
    return ((("q_nope", Array(S.DV, bf16)), ("q_rope", Array(S.DR, bf16)), ("kv", Array(S.T, Array(S.DR + S.DV, bf16)))),
            Array(S.DV, bf16))


@composite("SparseMlaHead", 1, ["T", "DR", "DV", "BN", "DOT", "INV", "SCALE"], _head_type, conformance=_CONF)
def SparseMlaHead(B, S, q_nope, q_rope, kv):
    """One query head over its row's T selected KV rows: ceil(T / BN) `SparseMlaBlock_v1`s visited last to first (the first, partial
    when BN does not divide T, zero-padded), then `bf16(o * INV((l0 + l2) + (l1 + l3)))` (`AttentionHead_v4`'s epilogue)."""
    def tile_of(nvis, first):
        return bind(SparseMlaBlock, DR=S.DR, DV=S.DV, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return head(word_steps(B), S, tile_of, S.INV, q_nope, q_rope, kv)


def head_block(st: Steps, S, dot, DOT, q_nope, q_rope, kv, m_prev=None, l_prev=None, o_prev=None):
    """`SparseMlaHeadBlock`'s body: `head`'s block call on one block, its query and padding its own."""
    if S.NVIS != S.BN and not S.FIRST:
        raise ValueError(f"SparseMlaHeadBlock: NVIS={S.NVIS} keys of BN={S.BN}, and only the first visited block is partial")
    KW = S.DR + S.DV
    q = Coll(Array(KW, st.bf16), concat([q_rope.refs, q_nope.refs]))
    return block(st, S, dot, DOT, q, _padded(st, kv, S.NVIS, S.BN, KW), m_prev, l_prev, o_prev)


def _head_block_type(S, bf16=BF16, f32=F32):
    state = () if S.FIRST else (("m_prev", f32), ("l_prev", Array(4, f32)), ("o_prev", Array(S.DV, f32)))
    return ((("q_nope", Array(S.DV, bf16)), ("q_rope", Array(S.DR, bf16)), ("kv", Array(S.NVIS, Array(S.DR + S.DV, bf16))))
            + state, Tuple(f32, Array(4, f32), Array(S.DV, f32)))


@composite("SparseMlaHeadBlock", 1, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"], _head_block_type, conformance=_CONF)
def SparseMlaHeadBlock(B, S, q_nope, q_rope, kv, m_prev=None, l_prev=None, o_prev=None):
    """One key block of a `SparseMlaHead_v1`'s chain on the head's own operands: its query ``[q_rope | q_nope]``, its NVIS keys
    (zero-padded to BN, which only the first visited block (FIRST) needs) and the state the block before it returns, then
    `SparseMlaBlock_v1`'s steps.  A head of T keys is its blocks in `visits` order, each a Call of this, then
    `SparseMlaHeadEnd_v1` on the last one's lanes and O: its function, step for step."""
    check_block_statics(S.DR, S.DV, S.NVIS, S.BN)
    return head_block(word_steps(B), S, bind(DotBf16V2, K=S.DR + S.DV, DOT=S.DOT), S.DOT, q_nope, q_rope, kv, m_prev, l_prev,
                      o_prev)


def _head_end_type(S, bf16=BF16, f32=F32):
    return ((("l", Array(4, f32)), ("o", Array(S.DV, f32))), Array(S.DV, bf16))


@composite("SparseMlaHeadEnd", 1, ["DV", "INV"], _head_end_type, conformance=_CONF)
def SparseMlaHeadEnd(B, S, l, o):
    """A head's end after its last `SparseMlaHeadBlock_v1`: `SparseMlaHead_v1`'s epilogue on the block's lanes and O,
    bf16(o * INV((l0 + l2) + (l1 + l3))) (`head_end`; `dead_free.AttnHeadEnd_v1{D = DV, INV}`'s steps)."""
    return head_end(word_steps(B), S.INV, l, o)


def head_chain(row) -> list[tuple[object, int]]:
    """The units a `SparseMla_v1` binding ``row`` is cut into, each Definition bound and how many of it: per head, its first
    visited block (`SparseMlaHeadBlock_v1`, FIRST, NVIS its keys), each later block (NVIS = BN) and its end
    (`SparseMlaHeadEnd_v1`).  What `serving_templates` stages a row as, and what circuit-check holds to the block in its
    place."""
    if getattr(row, "defn", None) is not SparseMla:
        raise ValueError(f"head_chain cuts a SparseMla_v1 binding, not {row!r}")
    S = row.bindings
    blk = {k: S[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    (lo, hi), *rest = visits(int(S["T"]), int(S["BN"]))
    NH = int(S["NH"])
    out = [(bind(SparseMlaHeadBlock, NVIS=hi - lo, FIRST=True, **blk), NH)]
    if rest:
        out.append((bind(SparseMlaHeadBlock, NVIS=int(S["BN"]), FIRST=False, **blk), NH * len(rest)))
    return out + [(bind(SparseMlaHeadEnd, DV=S["DV"], INV=S["INV"]), NH)]


def _mla_type(S, bf16=BF16):
    return ((("q_nope", Array(S.NH, Array(S.DV, bf16))), ("q_rope", Array(S.NH, Array(S.DR, bf16))),
             ("kv", Array(S.T, Array(S.DR + S.DV, bf16)))), Array(S.NH, Array(S.DV, bf16)))


@composite("SparseMla", 1, ["T", "NH", "DR", "DV", "BN", "DOT", "INV", "SCALE"], _mla_type, conformance=_CONF)
def SparseMla(B, S, q_nope, q_rope, kv):
    """One query token's NH heads over the same T selected rows (MLA's one latent KV head): head h `SparseMlaHead_v1` over
    q_nope[h], q_rope[h]."""
    hd = bind(SparseMlaHead, T=S.T, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    return B.batch(hd, q_nope, q_rope, kv, axes=(0, 0, None))


SPARSE_MLA_COMPOSITES = [SparseMlaBlock, SparseMlaHead, SparseMlaHeadBlock, SparseMlaHeadEnd, SparseMla]
#: each family serving stages as its cut, never whole, with the cut at a binding's statics (circuit-check's ``block`` holds
#: each unit to the Flock block in the Call's place)
CUTS = {"SparseMla_v1": head_chain}
