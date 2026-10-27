"""FA2's split-KV decode as vLLM serves it at its defaults: `Fa2SplitEnd_v1`, `Fa2Combine_v1`, `Fa2AttentionHeadSplit_v1`,
`Fa2AttentionSplit_v1`.

How many splits.  vllm-flash-attn 506341a1's `mha_varlen_fwd` (flash_api.cpp:646-761) swaps a decode launch (max_seqlen_q 1) of a
GQA model with no window into rows of each KV group's query heads and takes `set_params_splitkv`'s count:
`num_splits_heuristic(b * h_k * ceil(g / 64), 2 * SMs, ceil(max_seqlen_k / kBlockN), 128)` (`fa2_num_splits`), since vLLM passes
`num_splits = 0` (its FA2 metadata never sets `max_num_splits`; only FA3's AOT schedule does).  `run_mha_fwd_splitkv_dispatch`
(flash_fwd_launch_template.h:172-191) runs a count of exactly 1 in the "align" kernel, kBlockN 128 at head dim <= 64 and 64 above
(`AttentionHead_v5`), and any other count, 0 included, in the split-KV kernel's tile (`fa2_split_kblock_n`): kBlockN 256 at head
dim <= 64, 128 at <= 128, 64 above.  So a launch that is not swapped (prefill, mixed steps) runs that tile unsplit
(`AttentionHead_v5`), and a swapped one splits NS ways when its count is above 1.

A split.  Sequence b's split s runs the blocks `split_blocks(T, NS, BN)` gives it (`compute_attn_1rowblock_splitkv`:
`n_blocks_per_split = ceil(ceil(T / kBlockN) / NS)` from the sequence's own key count T) last to first through
`AttnBlock_v5`, its first block FIRST and CHECK (a decode launch is non-causal: one masking step, `Check_inf` there), and
`normalize_softmax_lse<Split>` ends it; a split with no block writes o = 0 and lse = -inf.

The PTX (the library carries compute_80 PTX and sm_80 SASS only, so every other GPU runs the driver's JIT of the PTX; every
float operation below is an explicit .ftz instruction or fma.rn there, which the JIT does not contract further):

  * the split end: `sum = (l0 + l2) + (l1 + l3)` (add.ftz, the quad butterfly as lane 0 sees it), `o * Fa2InvSum(sum)`
    (setp.equ.ftz, rcp.approx.ftz), `lse = sum != 0 ? fma.rn.ftz(m, scale_softmax, lg2(sum) * LN2) : -inf`, `scale_softmax`
    the launch's softmax scale as f32 (FA3's split end multiplies by scale_log2 * LN2: `SparseMlaSplitEnd_v1`);
  * `combine_attn_seqk_parallel<kBlockM, Log_max_splits>` (flash_fwd_kernel.h): kBlockM 4 at head dim % 128 == 0, 8 at % 64 == 0,
    else 16; kMaxSplits the least power of two >= NS (at least 2), the template the launch picks; R = min(128 / kBlockM,
    kMaxSplits) lanes per row, lane t holding splits l R + t for l < ceil(kMaxSplits kBlockM / 128), -inf past NS
    (`combine_geometry`).  The max is each lane's own over l, then the max.ftz butterfly (own, partner) over the R lanes, -inf
    guarded to 0; the sum is each lane's sum of `ex2((lse - max) * LOG2E)` over l (`expf`: sub.ftz, mul.ftz, ex2.approx.ftz),
    then the add.ftz butterfly; `logsum = sum != 0 ? fma.rn.ftz(lg2(sum), LN2, max) : +inf` (setp.equ.ftz); the weights
    `ex2((lse_s - logsum) * LOG2E)`; then `acc = fma.rn.ftz(w_s, o_s, acc)` over every split s < NS in order from +0 (an empty
    split's zero output and weight included), and `out = bf16(acc)` (cvt.rn.bf16x2.f32).

Registered functions (all `Name_v1`):
  Fa2SplitEnd{D,INV,SCALE}                                 (m, l[4], o[D]) -> (o_partial[D], lse)
  Fa2Combine{NS,D}                                         (lse: NS f32, o: NS x D f32) -> D bf16
  Fa2AttentionHeadSplit{T,NS,D,BN,DOT,INV,MASKED_FROM}     (q: D, k: T x D, v: T x D bf16) -> D bf16
  Fa2AttentionSplit{T,NS,NH,KVH,D,BN,DOT,INV,MASKED_FROM}  (q: NH x D, kc: T x KVH x D, vc: T x KVH x D bf16) -> NH x D bf16

NS = 1 is `AttentionHead_v5` / `Attention_v5` (the caller binds the align kernel's BN there).
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import sparse_mla as M
from verity.ml.definitions.registry import sparse_mla_split as SMS
from verity.ml.definitions.registry.fa2_check_inf import AttentionHeadV5, AttnBlockV5
from verity.ml.definitions.registry.prims import BF16, F32

NEG_INF, INF, LN2, LOG2E = SMS.NEG_INF, SMS.INF, SMS.LN2, SMS.LOG2E
#: `set_params_splitkv`'s and the combine's bound on the split count
MAX_SPLITS = 128
#: the combine's threads per block
THREADS = 128

_CONF = ("model-untested: read from vllm-flash-attn 506341a1's split-KV kernel, combine kernel and their compute_80 PTX (the module "
         "docstring's files); the served Match on cc 12.0 is its hardware check")


def fa2_split_kblock_n(D: int) -> int:
    """kBlockN of FA2's split-KV kernel (`run_mha_fwd_splitkv_dispatch`, any split count but 1)."""
    return 256 if D <= 64 else 128 if D <= 128 else 64


def fa2_align_kblock_n(D: int) -> int:
    """kBlockN of FA2's `num_splits == 1` align kernel (`run_mha_fwd_splitkv_align`)."""
    return 128 if D <= 64 else 64


def fa2_num_splits(batch: int, heads_kv: int, groups: int, max_seqlen_k: int, D: int, sms: int) -> int:
    """`set_params_splitkv`'s count for a swapped decode launch of `batch` sequences: `num_splits_heuristic(batch * heads_kv *
    ceil(groups / 64), 2 * sms, ceil(max_seqlen_k / kBlockN), 128)` (flash_api.cpp:307-341), its efficiencies in f32."""
    f = np.float32
    bnm, n_sms = batch * heads_kv * -(-groups // 64), 2 * sms
    nnb = -(-max_seqlen_k // fa2_split_kblock_n(D))
    if f(bnm) >= f(0.8) * f(n_sms):
        return 1
    top = min(MAX_SPLITS, n_sms, nnb)
    eligible = [s == 1 or -(-nnb // s) != -(-nnb // (s - 1)) for s in range(1, top + 1)]
    eff = []
    for s in range(1, top + 1):
        waves = f(f(bnm * s) / f(n_sms))
        eff.append(f(waves / f(np.ceil(waves))) if eligible[s - 1] else f(0))
    best = max(eff, default=f(0))
    return next((s for s in range(1, top + 1) if eligible[s - 1] and float(eff[s - 1]) >= 0.85 * float(best)), 1)


def split_ranges(T: int, NS: int, BN: int) -> list[tuple[int, int]]:
    """Every split's [b0, b1) block range in split order, an empty split's b0 >= b1 (`split_blocks` keeps the non-empty)."""
    nb = -(-T // BN)
    per = -(-nb // NS)
    return [(s * per, min((s + 1) * per, nb)) for s in range(NS)]


def combine_geometry(NS: int, D: int) -> tuple[int, int]:
    """(R, L): the combine's lanes per row and the splits each lane holds (module docstring)."""
    kbm = 4 if D % 128 == 0 else 8 if D % 64 == 0 else 16
    kmax = max(2, 1 << (NS - 1).bit_length())
    return min(THREADS // kbm, kmax), -(-(kmax * kbm) // THREADS)


def check_split_statics(T: int, NS: int, D: int) -> None:
    if T < 1 or not 1 <= NS <= MAX_SPLITS or D % 16:
        raise ValueError(f"Fa2AttentionHeadSplit: T={T} keys in NS={NS} splits (1 <= NS <= {MAX_SPLITS}), D={D} a multiple of 16")


def scale_bits(scale: float) -> int:
    """`params.scale_softmax`: the launch's softmax scale as f32."""
    return P.bits_of(np.float32(scale))


# --- the bodies (one per Definition, over `sparse_mla.Steps`) ------------------------------------------------------------

def split_end(st: M.Steps, S, m, l, o):
    """`Fa2SplitEnd`'s body: `SparseMlaSplitEnd`'s with the running max multiplied by `scale_softmax`."""
    return SMS.split_end(st, S, S.INV, m, l, o, scale_bits(S.SCALE))


def combine(st: M.Steps, S, lse, o):
    """`Fa2Combine`'s body: each lane's max and sum over its splits, the butterflies, logsum, the weights, the ordered FMAs."""
    NS, D, B = S.NS, S.D, st.B
    if not 2 <= NS <= MAX_SPLITS:
        raise ValueError(f"Fa2Combine: NS={NS} splits, the combine runs 2 to {MAX_SPLITS}")
    R, L = combine_geometry(NS, D)
    pad = st.word(NEG_INF)
    keys = [tuple(i if i < NS else "pad" for i in range(t, L * R, R)) for t in range(R)]
    word = {i: (pad if i == "pad" else lse[i]) for k in keys for i in k}
    log2e = st.word(LOG2E)

    def lanes(op, of):
        """Each lane's ``op`` over its splits' ``of`` in order (equal lanes once)."""
        memo: dict = {}
        for k in keys:
            if k not in memo:
                acc = of[k[0]]
                for i in k[1:]:
                    acc = op(acc, of[i])
                memo[k] = acc
        return [memo[k] for k in keys]
    cur = st.guard(SMS._lane_tree(st.max, lanes(st.max, word), keys))
    scaled = {i: st.ex2(st.mul(st.sub(w, cur), log2e)) for i, w in word.items()}
    tot = SMS._lane_tree(st.add, lanes(st.add, scaled), keys)
    logsum = st.sel(st.nez(tot), st.fma(st.lg2(tot), st.word(LN2), cur), st.word(INF))
    acc = array_of([st.word(0)] * D)
    for s in range(NS):
        w = st.ex2(st.mul(st.sub(lse[s], logsum), log2e))
        acc = B.batch(st._defs["fma"], w, o[s], acc, axes=(None, 0, 0))
    return B.batch(st._defs["cvt"], acc, axes=(0,))


def head_split(st: M.Steps, S, block_of, unsplit, end, combine_def, q, k, v):
    """`Fa2AttentionHeadSplit`'s body: ``unsplit`` at NS = 1; else every split's blocks (``block_of(nvis, first, check)``) and
    ``end``, an empty split (-inf, 0), and ``combine_def``."""
    B, T, NS, D, BN, MF = st.B, S.T, S.NS, S.D, S.BN, S.MASKED_FROM
    check_split_statics(T, NS, D)
    if NS == 1:
        return B.call(unsplit, q, k, v)
    nb = -(-T // BN)
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"Fa2AttentionHeadSplit: MASKED_FROM {MF} outside blocks 0..{nb - 1}")
    row = None
    lses, parts = [], []
    for b0, b1 in split_ranges(T, NS, BN):
        if b0 >= b1:
            lses.append(st.word(NEG_INF))
            parts.append(array_of([st.word(0)] * D))
            continue
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = k[lo:hi], v[lo:hi]
            if hi - lo < BN:  # Clear_OOB_MN: the rows past the sequence are zero, their columns masked
                if row is None:
                    row = Coll(Array(D, st.bf16), concat([st.zero16().refs] * D))
                kb = Coll(Array(BN, Array(D, st.bf16)), concat([kb.refs] + [row.refs] * (BN - hi + lo)))
                vb = Coll(Array(BN, Array(D, st.bf16)), concat([vb.refs] + [row.refs] * (BN - hi + lo)))
            first = b == b1 - 1
            blk = block_of(hi - lo, first, first or b >= MF)
            if first:
                z = st.word(0)
                r = B.call(blk, q, kb, vb, z, array_of([z] * 4), array_of([z] * D))
            else:
                r = B.call(blk, q, kb, vb, m, l, o)
            m, l, o = r[0], r[1], r[2]
        e = B.call(end, m, l, o)
        parts.append(e[0])
        lses.append(e[1])
    return B.call(combine_def, array_of(lses), array_of(parts))


# --- the Definitions --------------------------------------------------------------------------------------------------------

def _end_type(S, f32=F32):
    return ((("m", f32), ("l", Array(4, f32)), ("o", Array(S.D, f32))), Tuple(Array(S.D, f32), f32))


@composite("Fa2SplitEnd", 1, ["D", "INV", "SCALE"], _end_type, conformance=_CONF)
def Fa2SplitEnd(B, S, m, l, o):
    """A split's end (`normalize_softmax_lse<Split>`): sum = (l0 + l2) + (l1 + l3), the partial output o * INV(sum), and lse =
    sum != 0 ? fma(m, f32(SCALE), MufuLg2Ftz(sum) * LN2) : -inf (`F32NeZeroFtz_v1`, `SelectI32_v1`)."""
    return split_end(SMS.word_steps(B), S, m, l, o)


def _combine_type(S, bf16=BF16, f32=F32):
    return ((("lse", Array(S.NS, f32)), ("o", Array(S.NS, Array(S.D, f32)))), Array(S.D, bf16))


@composite("Fa2Combine", 1, ["NS", "D"], _combine_type, conformance=_CONF)
def Fa2Combine(B, S, lse, o):
    """`combine_attn_seqk_parallel` for one query row of NS splits (every split of the launch, an empty one's lse -inf and o 0):
    the max and the sum of `MufuEx2Ftz((lse - max) * LOG2E)` over `combine_geometry`'s lanes, logsum = sum != 0 ?
    fma(MufuLg2Ftz(sum), LN2, max) : +inf, then `acc = fma(MufuEx2Ftz((lse_s - logsum) * LOG2E), o_s, acc)` over the splits in
    order from +0, bf16(acc)."""
    return combine(SMS.word_steps(B), S, lse, o)


def _head_type(S, bf16=BF16):
    return ((("q", Array(S.D, bf16)), ("k", Array(S.T, Array(S.D, bf16))), ("v", Array(S.T, Array(S.D, bf16)))), Array(S.D, bf16))


@composite("Fa2AttentionHeadSplit", 1, ["T", "NS", "D", "BN", "DOT", "INV", "MASKED_FROM"], _head_type, conformance=_CONF)
def Fa2AttentionHeadSplit(B, S, q, k, v):
    """One query head over T keys in NS splits: `AttentionHead_v5` at NS = 1; else each split's `AttnBlock_v5`s last to first
    (FIRST and CHECK at its first visited block, CHECK at blocks >= MASKED_FROM, the sequence's last block zero-padded) and its
    `Fa2SplitEnd_v1{D, INV, SCALE=D^-1/2}`, an empty split's (-inf, 0), and `Fa2Combine_v1{NS, D}`."""
    def block_of(nvis, first, check):
        return bind(AttnBlockV5, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, CHECK=check)
    unsplit = bind(AttentionHeadV5, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    end = bind(Fa2SplitEnd, D=S.D, INV=S.INV, SCALE=S.D ** -0.5)
    return head_split(SMS.word_steps(B), S, block_of, unsplit, end, bind(Fa2Combine, NS=S.NS, D=S.D), q, k, v)


def _split_type(S, bf16=BF16):
    return ((("q", Array(S.NH, Array(S.D, bf16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, bf16)))),
             ("vc", Array(S.T, Array(S.KVH, Array(S.D, bf16))))), Array(S.NH, Array(S.D, bf16)))


@composite("Fa2AttentionSplit", 1, ["T", "NS", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"], _split_type, conformance=_CONF)
def Fa2AttentionSplit(B, S, q, kc, vc):
    """One query token's NH heads over T keys in NS splits (GQA: head h reads KV head h // (NH / KVH)), each
    `Fa2AttentionHeadSplit_v1`."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    head = bind(Fa2AttentionHeadSplit, T=T, NS=S.NS, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


FA2_SPLIT_COMPOSITES = [Fa2SplitEnd, Fa2Combine, Fa2AttentionHeadSplit, Fa2AttentionSplit]
