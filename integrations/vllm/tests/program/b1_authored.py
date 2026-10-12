"""The B1 serving program authored in torch through the frontend (lane FE2).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

Every composite of TA1's hand-built program from `RMSNormTriton` up to `Serve` (`verity_vllm/program/registry/b1.py`)
written as `@function`s over tensors: `ops.<Prim>` for the registered primitives, `ops.Gemm(K=, N=)` etc. for
the registered leaf composites (`DotBf16`, `Gemm`, `BiasAdd`, `RoPE`, `SiluMul`, `Embedding`, `TokenSelect` — the
same corrected TA1 definitions on both sides), `batch`/`bind` for the structural forms, and torch views
(`slice`, `select`, `stack`, `cat`, `expand`, `t`, `view`) for the reference rearrangements. Nothing here is a
new definition: each function carries TA1's `Name_vN` and statics, and `check_identity` (run by `export`)
rejects it if its body differed from the registered one. Where the digests nevertheless differ it is the
reference *encoding* of one broadcast (`[z, z, ..., z]`: TA1 writes `Concat`/`Explicit` runs, the frontend a
zero-stride `Strided`); `verity_vllm.program.frontend.correspond` shows the reference sequences are the same leaf by
leaf. Needs torch (the pod); `tests/program/test_b1_authored.py` skips without it.
"""

import torch

from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity_vllm.program.registry import b1
from verity_vllm.program.registry.prims import BF16, F32, I32
from verity.ml.circuits.types import Array, Tuple as TupleT
from verity_vllm.program.frontend.torch_frontend import Arr, Bound, Dep, Prim, Registered, Tup, batch, bind, function, ops

BN = b1.BN  # kBlockN = 64


def _c32(bits: int):
    return ops.const(32, bits)


def _f32c(x: float):
    """A `Const32[bits(f32(x))]` gate (TA1's `f32c`)."""
    return Prim(b1.f32c(x))()


# -- RMSNorm ------------------------------------------------------------------------------------------------


def _butterfly(vals: list, masks: tuple[int, ...], add) -> torch.Tensor:
    """Lane 0's value after an xor-butterfly reduction (`v = v + shfl_xor(v, m)`)."""
    n = len(vals)
    cur = list(vals)
    for m in masks:
        cur = [add(cur[i], cur[i ^ m]) if (i ^ m) < n else cur[i] for i in range(n)]
    return cur[0]


@function(name="RMSNormTriton", version=1, static=("N", "EPS"))
def RMSNormTriton(x: Arr["N", BF16], w: Arr["N", BF16], *, N, EPS) -> Arr["N", BF16]:
    """vLLM batch-invariant Triton `_rms_norm_kernel` (b1.RMSNormTriton, gate for gate)."""
    xf = batch(ops.Bf16ToF32, x, axes=(0,))
    chunk_sums = []
    for c0 in range(0, N, 1024):
        warps = []
        for wi in range(4):
            lanes = []
            for ln in range(32):
                t = wi * 32 + ln
                base = c0 + 8 * t
                if base >= N:
                    lanes.append(_c32(0))
                    continue
                xs = [xf[base + j] for j in range(8)]
                acc = ops.F32Mul(xs[1], xs[1])
                acc = ops.F32Fma(xs[0], xs[0], acc)
                for j in range(2, 8):
                    acc = ops.F32Fma(xs[j], xs[j], acc)
                lanes.append(acc)
            warps.append(_butterfly(lanes, (16, 8, 4, 2, 1), ops.F32Add))
        chunk_sums.append(ops.F32Add(ops.F32Add(warps[0], warps[2]), ops.F32Add(warps[1], warps[3])))
    total = _c32(0)
    for cs in chunk_sums:
        total = ops.F32Add(total, cs)
    rcp_n = ops.DivFullRcp(_f32c(float(N)))
    mean_eps = ops.F32Fma(rcp_n, total, _f32c(EPS))
    rms = ops.MufuSqrtFtz(mean_eps)
    inv = ops.F32Mul(ops.DivFullScaleA(_f32c(1.0), rms), ops.DivFullRcp(rms))
    wf = batch(ops.Bf16ToF32, w, axes=(0,))
    sx = batch(ops.F32Mul, inv, xf, axes=(None, 0))
    y = batch(ops.F32Mul, sx, wf, axes=(0, 0))
    return batch(ops.F2fpBf16, y, axes=(0,))


def _cub_block_reduce_sum(partials: list) -> torch.Tensor:
    """`cub::BlockReduce<float, 1024>` (BLOCK_REDUCE_WARP_REDUCTIONS): per-warp shfl_down tree 1,2,4,8,16
    guarded by the valid-peer predicate, then thread 0 adds the warp aggregates from warp 0."""
    warps = []
    for wi in range((len(partials) + 31) // 32):
        lanes = dict(enumerate(partials[32 * wi:32 * wi + 32]))
        for off in (1, 2, 4, 8, 16):
            lanes = {i: (ops.F32Add(v, lanes[i + off]) if i + off in lanes else v)
                     for i, v in lanes.items() if i % (2 * off) == 0}
        warps.append(lanes[0])
    var = warps[0]
    for wv in warps[1:]:
        var = ops.F32Add(var, wv)
    return var


def _fused_rms_epilogue(N: int, EPS: float, r, rf, var, w):
    """`s = rsqrtf(var / N + eps)`; `out_j = bf16((res'_j * s) * w_j)`; returns (normed, res')."""
    mean = ops.F32Div(var, _f32c(float(N)))
    arg = ops.F32Add(mean, _f32c(EPS))
    sc = ops.RsqrtApprox(arg)
    wf = batch(ops.Bf16ToF32, w, axes=(0,))
    y = batch(ops.F32Mul, batch(ops.F32Mul, rf, sc, axes=(0, None)), wf, axes=(0, 0))
    return (batch(ops.F2fpBf16, y, axes=(0,)), r)


@function(name="RMSNormFusedCuda", version=2, static=("N", "EPS"))
def RMSNormFusedCuda(x: Arr["N", BF16], res: Arr["N", BF16], w: Arr["N", BF16], *, N, EPS) -> Tup[Arr["N", BF16], Arr["N", BF16]]:
    """vLLM generic `fused_add_rms_norm_kernel<bf16, 0, true>` under `VLLM_BATCH_INVARIANT=1` (b1.RMSNormFusedCuda v2)."""
    r = batch(ops.Bf16AddF2fp, x, res, axes=(0, 0))
    rf = batch(ops.Bf16ToF32, r, axes=(0,))
    nthreads = min(N, b1.FUSED_RMS_BLOCK)
    partials = []
    for t in range(nthreads):
        acc = ops.F32Mul(rf[t], rf[t])
        for j in range(t + nthreads, N, nthreads):
            acc = ops.F32Fma(rf[j], rf[j], acc)
        partials.append(acc)
    var = _cub_block_reduce_sum(partials)
    return _fused_rms_epilogue(N, EPS, r, rf, var, w)


# -- attention (FA2 forward, one (query, head) row) ------------------------------------------------------


@function(name="AttnBlock", version=1, static=("D", "NVIS", "FIRST"))
def AttnBlock(q: Arr["D", BF16], k: Arr[BN, "D", BF16], v: Arr[BN, "D", BF16], m_prev: F32, l_prev: Arr[4, F32],
              o_prev: Arr["D", F32], *, D, NVIS, FIRST) -> Tup[F32, Arr[4, F32], Arr["D", F32]]:
    """One 64-key block of the FA2 online softmax for one query row (b1.AttnBlock)."""
    dot = ops.DotBf16(K=D)
    s = [dot(q, k[c]) for c in range(NVIS)]  # A: S = Q K^T
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = ops.F32Max(bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else ops.F32Max(m_prev, bmax)
    m_use = ops.GuardNegInfZero(m_new)
    scale_log2 = _c32(b1._scale_log2_bits(D))
    if not FIRST:  # B2: rescale factor
        scores_scale = ops.MufuEx2Ftz(ops.F32MulFtz(ops.F32SubFtz(m_prev, m_use), scale_log2))
    max_scaled = ops.F32MulFtz(m_use, scale_log2)
    p = [ops.MufuEx2Ftz(ops.F32FmaSubFtz(s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    lanes = []  # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    for ln in range(4):
        cols = [c for c in range(NVIS) if b1._lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else _c32(0)
        else:
            first = p[cols[0]] if cols else _c32(0)
            acc = ops.F32FmaFtz(scores_scale, l_prev[ln], first)
        for c in cols[1:]:
            acc = ops.F32AddFtz(acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in ascending 16-key k-steps
    pb = [ops.F2fpBf16(x) for x in p]
    zero16 = ops.const(16, 0) if NVIS < BN else None
    o = torch.stack([_c32(0) for _ in range(D)]) if FIRST else torch.stack([ops.F32MulFtz(o_prev[d], scores_scale) for d in range(D)])
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = torch.stack([(pb[k0 + j] if k0 + j < NVIS else zero16) for j in range(16)])
        vt = v[k0:k0 + 16].t()  # V column d over the 16 keys: a [D][16] strided view
        o = batch(ops.AmpereBF16TcDot16, o, prow, vt, axes=(0, None, 0))
    return (m_new, torch.stack(lanes), o)


@function(name="AttentionHead", version=1, static=("T", "D"))
def AttentionHead(q: Arr["D", BF16], k: Arr["T", "D", BF16], v: Arr["T", "D", BF16], *, T, D) -> Arr["D", BF16]:
    """Causal attention of one query at position T-1 over keys 0..T-1, blocks visited last -> first (b1.AttentionHead)."""
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = ops.const(16, 0).expand(D)  # one zero gate broadcast: a zero-stride reference run
            pad = zero_row.expand(BN - nvis, D)
            kb = torch.cat([kb, pad])
            vb = torch.cat([vb, pad])
        first = b == nb - 1
        blk = bind(AttnBlock, D=D, NVIS=nvis, FIRST=first)
        if first:
            z = _c32(0)
            st = blk(q, kb, vb, z, z.expand(4), z.expand(D))
        else:
            st = blk(q, kb, vb, m, l, o)
        m, l, o = st
    total = ops.F32AddFtz(ops.F32AddFtz(l[0], l[2]), ops.F32AddFtz(l[1], l[3]))
    inv = ops.Fa2InvSum(total)
    y = batch(ops.F32MulFtz, o, inv, axes=(0, None))
    return batch(ops.F2fpBf16, y, axes=(0,))


@function(name="Attention", version=1, static=("T", "NH", "KVH", "D"))
def Attention(q: Arr["NH", "D", BF16], kc: Arr["T", "KVH", "D", BF16], vc: Arr["T", "KVH", "D", BF16], *, T, NH, KVH, D) -> Arr["NH", "D", BF16]:
    """GQA over `[T][KVH][D]` caches: head h reads KV head h // (NH/KVH) (b1.Attention)."""
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = kc.select(1, g)  # one KV head's rows: a strided view of the cache
        vg = vc.select(1, g)
        outs.append(batch(bind(AttentionHead, T=T, D=D), q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return torch.cat(outs, 0)


# -- layer, final, serve -------------------------------------------------------------------------------------


def _hidden(S):
    return Array(S["C"]["H"], BF16)


def _heads(S):
    return Array(S["C"]["NH"], Array(S["C"]["D"], BF16))


def _kv_heads(S):
    return Array(S["C"]["KVH"], Array(S["C"]["D"], BF16))


def _layer_w(S):
    return b1.layer_weights_type(S["C"])


@function(name="LayerPre", version=1, static=("C", "L0"))
def LayerPre(h: Dep(_hidden), res: Dep(_hidden), w: Dep(_layer_w), cs: Dep(lambda S: Array(S["C"]["D"], BF16)), *, C, L0) \
        -> Dep(lambda S: TupleT(_hidden(S), _heads(S), _kv_heads(S), _kv_heads(S))):
    """Layer front half at one position: input norm, QKV GEMM (+ bias), RoPE on q, k (b1.LayerPre)."""
    H, NH, KVH, D = C["H"], C["NH"], C["KVH"], C["D"]
    if L0:
        x = h
        n1 = RMSNormTriton(h, w["ln1"], N=H, EPS=C["EPS"])
    else:
        n1, x = RMSNormFusedCuda(h, res, w["ln1"], N=H, EPS=C["EPS"])
    qkv = ops.Gemm(K=H, N=C["QKV"])(n1, w["qkv_w"])
    if C.get("BIAS", True):
        qkv = ops.BiasAdd(N=C["QKV"])(qkv, w["qkv_b"])
    q = qkv[0:NH * D].view(NH, D)
    k = qkv[NH * D:(NH + KVH) * D].view(KVH, D)
    v = qkv[(NH + KVH) * D:(NH + 2 * KVH) * D].view(KVH, D)
    qr = ops.RoPE(NHEADS=NH, D=D)(q, cs)
    kr = ops.RoPE(NHEADS=KVH, D=D)(k, cs)
    return (x, qr, kr, v)


@function(name="LayerPost", version=1, static=("C",))
def LayerPost(attn: Dep(_heads), res: Dep(_hidden), w: Dep(_layer_w), *, C) -> Dep(lambda S: TupleT(_hidden(S), _hidden(S))):
    """Layer back half: o_proj GEMM, fused add+RMSNorm, gate_up GEMM, SiLU-mul, down GEMM (b1.LayerPost)."""
    H = C["H"]
    a = attn.reshape(C["NH"] * C["D"])
    o = ops.Gemm(K=C["NH"] * C["D"], N=H)(a, w["o_w"])
    n2, r2 = RMSNormFusedCuda(o, res, w["ln2"], N=H, EPS=C["EPS"])
    gu = ops.Gemm(K=H, N=C["GU"])(n2, w["gate_up_w"])
    act = ops.SiluMul(I=C["I"])(gu)
    d = ops.Gemm(K=C["I"], N=H)(act, w["down_w"])
    return (d, r2)


@function(name="Final", version=1, static=("C",))
def Final(h: Dep(_hidden), res: Dep(_hidden), norm_w: Dep(_hidden), lm_head: Dep(lambda S: Array(S["C"]["V"], _hidden(S))), *, C) -> I32:
    """Final fused add+norm, LM head GEMM (tied embedding), greedy token selection (b1.Final)."""
    n = RMSNormFusedCuda(h, res, norm_w, N=C["H"], EPS=C["EPS"])[0]
    logits = ops.Gemm(K=C["H"], N=C["V"])(n, lm_head)
    return ops.TokenSelect(V=C["V"])(logits)


def _forward_positions(C: dict, toks, weights, T0: int, n_pos: int, caches):
    """All layers for positions T0..T0+n_pos-1 (b1._forward_positions): prefill when `caches` is None."""
    NL, H = C["NL"], C["H"]
    embed = ops.Embedding(V=C["V"], H=H)
    hs = [embed(toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [LayerPre(hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i], C=C, L0=(L == 0),
                        name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = torch.stack([p[2] for p in pre])
        v_rows = torch.stack([p[3] for p in pre])
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = torch.cat([caches[L][0], k_rows])
            vc = torch.cat([caches[L][1], v_rows])
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = Attention(pre[i][1], kc[0:T], vc[0:T], T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], name=f"t={T - 1}/layers.{L}/attn")
            post = LayerPost(attn, pre[i][0], w, C=C, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post
    return new_kv, list(zip(hs, res))


@function(name="Serve", version=1, static=("C", "LP", "STEPS"))
def Serve(prompt: Arr["LP", I32], weights: Dep(lambda S: b1.weights_type(S["C"])), *, C, LP, STEPS) -> Arr["STEPS + 1", I32]:
    """Greedy serving of one request: prefill of LP prompt tokens, then STEPS decode steps (b1.Serve)."""
    caches, finals = _forward_positions(C, prompt, weights, 0, LP, None)
    tok = Final(finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], C=C, name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _forward_positions(C, tok.view(1), weights, T0, 1, caches)
        caches = [(torch.cat([caches[L][0], new_kv[L][0]]), torch.cat([caches[L][1], new_kv[L][1]])) for L in range(C["NL"])]
        tok = Final(fin[0][0], fin[0][1], weights["norm"], weights["embed"], C=C, name=f"t={T0}/final")
        toks.append(tok)
    return torch.stack(toks)


# -- [E7 D54/D57 2026-09-13] the attention chain with the kernel's key-block size as a DECLARED static BN (b1.AttnBlockV3 / -----
# -- AttentionHeadV3 / AttentionV3) and Serve_v4 (b1.ServeV4): the v1 bodies with `BN` bound from the program identity instead ---
# -- of the constant 64.  Registered ids AttnBlock_v3{D,NVIS,FIRST,BN} / AttentionHead_v3{T,D,BN} / Attention_v3{...,BN} / ----------
# -- Serve_v4{C,LP,STEPS,BN}; `check_identity` against the registry as for v1. ---------------------------------------------------


@function(name="AttnBlock", version=3, static=("D", "NVIS", "FIRST", "BN"))
def AttnBlockV3(q: Arr["D", BF16], k: Arr["BN", "D", BF16], v: Arr["BN", "D", BF16], m_prev: F32, l_prev: Arr[4, F32],
                o_prev: Arr["D", F32], *, D, NVIS, FIRST, BN) -> Tup[F32, Arr[4, F32], Arr["D", F32]]:
    """One BN-key block of the FA2 online softmax for one query row (b1.AttnBlockV3; BN the declared kBlockN)."""
    dot = ops.DotBf16(K=D)
    s = [dot(q, k[c]) for c in range(NVIS)]  # A: S = Q K^T
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = ops.F32Max(bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else ops.F32Max(m_prev, bmax)
    m_use = ops.GuardNegInfZero(m_new)
    scale_log2 = _c32(b1._scale_log2_bits(D))
    if not FIRST:  # B2: rescale factor
        scores_scale = ops.MufuEx2Ftz(ops.F32MulFtz(ops.F32SubFtz(m_prev, m_use), scale_log2))
    max_scaled = ops.F32MulFtz(m_use, scale_log2)
    p = [ops.MufuEx2Ftz(ops.F32FmaSubFtz(s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    lanes = []  # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    for ln in range(4):
        cols = [c for c in range(NVIS) if b1._lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else _c32(0)
        else:
            first = p[cols[0]] if cols else _c32(0)
            acc = ops.F32FmaFtz(scores_scale, l_prev[ln], first)
        for c in cols[1:]:
            acc = ops.F32AddFtz(acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in ascending 16-key k-steps
    pb = [ops.F2fpBf16(x) for x in p]
    zero16 = ops.const(16, 0) if NVIS < BN else None
    o = torch.stack([_c32(0) for _ in range(D)]) if FIRST else torch.stack([ops.F32MulFtz(o_prev[d], scores_scale) for d in range(D)])
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = torch.stack([(pb[k0 + j] if k0 + j < NVIS else zero16) for j in range(16)])
        vt = v[k0:k0 + 16].t()  # V column d over the 16 keys: a [D][16] strided view
        o = batch(ops.AmpereBF16TcDot16, o, prow, vt, axes=(0, None, 0))
    return (m_new, torch.stack(lanes), o)


@function(name="AttentionHead", version=3, static=("T", "D", "BN"))
def AttentionHeadV3(q: Arr["D", BF16], k: Arr["T", "D", BF16], v: Arr["T", "D", BF16], *, T, D, BN) -> Arr["D", BF16]:
    """Causal attention of one query at position T-1 over keys 0..T-1, ceil(T / BN) blocks visited last -> first (b1.AttentionHeadV3)."""
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = ops.const(16, 0).expand(D)  # one zero gate broadcast: a zero-stride reference run
            pad = zero_row.expand(BN - nvis, D)
            kb = torch.cat([kb, pad])
            vb = torch.cat([vb, pad])
        first = b == nb - 1
        blk = bind(AttnBlockV3, D=D, NVIS=nvis, FIRST=first, BN=BN)
        if first:
            z = _c32(0)
            st = blk(q, kb, vb, z, z.expand(4), z.expand(D))
        else:
            st = blk(q, kb, vb, m, l, o)
        m, l, o = st
    total = ops.F32AddFtz(ops.F32AddFtz(l[0], l[2]), ops.F32AddFtz(l[1], l[3]))
    inv = ops.Fa2InvSum(total)
    y = batch(ops.F32MulFtz, o, inv, axes=(0, None))
    return batch(ops.F2fpBf16, y, axes=(0,))


@function(name="Attention", version=3, static=("T", "NH", "KVH", "D", "BN"))
def AttentionV3(q: Arr["NH", "D", BF16], kc: Arr["T", "KVH", "D", BF16], vc: Arr["T", "KVH", "D", BF16], *, T, NH, KVH, D, BN) -> Arr["NH", "D", BF16]:
    """GQA over `[T][KVH][D]` caches through AttentionHead_v3{T, D, BN} (b1.AttentionV3)."""
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = kc.select(1, g)  # one KV head's rows: a strided view of the cache
        vg = vc.select(1, g)
        outs.append(batch(bind(AttentionHeadV3, T=T, D=D, BN=BN), q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return torch.cat(outs, 0)


def _forward_positions_v4(C: dict, BN: int, toks, weights, T0: int, n_pos: int, caches):
    """`_forward_positions` with Attention_v3{..., BN} at the attention call (b1._forward_positions_v4)."""
    NL, H = C["NL"], C["H"]
    embed = ops.Embedding(V=C["V"], H=H)
    hs = [embed(toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [LayerPre(hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i], C=C, L0=(L == 0),
                        name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = torch.stack([p[2] for p in pre])
        v_rows = torch.stack([p[3] for p in pre])
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = torch.cat([caches[L][0], k_rows])
            vc = torch.cat([caches[L][1], v_rows])
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = AttentionV3(pre[i][1], kc[0:T], vc[0:T], T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=BN, name=f"t={T - 1}/layers.{L}/attn")
            post = LayerPost(attn, pre[i][0], w, C=C, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post
    return new_kv, list(zip(hs, res))


@function(name="Serve", version=4, static=("C", "LP", "STEPS", "BN"))
def ServeV4(prompt: Arr["LP", I32], weights: Dep(lambda S: b1.weights_type(S["C"])), *, C, LP, STEPS, BN) -> Arr["STEPS + 1", I32]:
    """Greedy serving of one request with the attention key-block size BN declared by the program identity (b1.ServeV4)."""
    caches, finals = _forward_positions_v4(C, BN, prompt, weights, 0, LP, None)
    tok = Final(finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], C=C, name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _forward_positions_v4(C, BN, tok.view(1), weights, T0, 1, caches)
        caches = [(torch.cat([caches[L][0], new_kv[L][0]]), torch.cat([caches[L][1], new_kv[L][1]])) for L in range(C["NL"])]
        tok = Final(fin[0][0], fin[0][1], weights["norm"], weights["embed"], C=C, name=f"t={T0}/final")
        toks.append(tok)
    return torch.stack(toks)


# -- CF1: the factored definitions (composition.md §2) — the same bodies, the accidental constants and hard-coded ------
# -- callees as statics (`verity_vllm/program/registry/b1.py`: DotBf16_v2 … Serve_v2) ------------------------------------------
#
# `DOT` (the k16 tensor-core step), `BN` (kBlockN), `INV` (the softmax epilogue inverse sum) and the serving-level
# records `GEMM = {"DOT"}`, `ATTENTION = {"BN", "DOT", "INV"}`, `NORM = {"L0", "FUSED"}` are declared statics, so
# they are part of each specialisation's identity (§8.3) and the Ampere and Hopper programs are two binding records
# of one `Serve_v2`. Function-valued statics may be given as frontend callables (`ops.AmpereBF16TcDot16`, a `bind`
# of an authored norm) or as the registry's own definitions (`b1.GEMM_AMPERE`, `hopper.ATTENTION_FA3`,
# `b1.norm_bundle(C)`): `_callable` accepts both.


def _callable(f):
    """A static-valued function as a frontend callable: a `Prim`/`Bound`/`Function` as is; a registered primitive or a
    complete registered specialisation wrapped (`_ir_static` maps both back to the same IR definition)."""
    if hasattr(f, "_resolve"):
        return f
    if isinstance(f, PrimitiveDefinition):
        return Prim(f)
    if isinstance(f, SpecializedDefinition):
        return Bound(Registered(f.defn), f.bindings)
    raise TypeError(f"{f!r} is not a function-valued static")


GEMM_AMPERE = {"DOT": ops.AmpereBF16TcDot16}
ATTENTION_FA2 = {"BN": BN, "DOT": ops.AmpereBF16TcDot16, "INV": ops.Fa2InvSum}  # the HEAD DIM 128 record (W11): Serve_v1's geometry


def attention_fa2(D: int) -> dict:
    """[E7 D54] The FA2 attention record at head dim `D` (`b1.attention_fa2`): `BN = b1.fa2_kblock_n(D)` -- Serve_v4's geometry."""
    return {"BN": b1.fa2_kblock_n(D), "DOT": ops.AmpereBF16TcDot16, "INV": ops.Fa2InvSum}


def norm_bundle(C: dict) -> dict:
    """The `NORM` record over the authored norms at width `C["H"]` (the registry's `b1.norm_bundle` binds TA1's)."""
    return {"L0": bind(RMSNormTriton, N=C["H"], EPS=C["EPS"]), "FUSED": bind(RMSNormFusedCuda, N=C["H"], EPS=C["EPS"])}


@function(name="AttnBlock", version=2, static=("D", "NVIS", "FIRST", "BN", "DOT"))
def AttnBlockV2(q: Arr["D", BF16], k: Arr["BN", "D", BF16], v: Arr["BN", "D", BF16], m_prev: F32, l_prev: Arr[4, F32],
                o_prev: Arr["D", F32], *, D, NVIS, FIRST, BN, DOT) -> Tup[F32, Arr[4, F32], Arr["D", F32]]:
    """One BN-key block of the flash online softmax for one query row, the k16 step `DOT` (b1.AttnBlockV2)."""
    step = _callable(DOT)
    dot = ops.DotBf16_v2(K=D, DOT=DOT)
    s = [dot(q, k[c]) for c in range(NVIS)]  # A: S = Q K^T
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = ops.F32Max(bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else ops.F32Max(m_prev, bmax)
    m_use = ops.GuardNegInfZero(m_new)
    scale_log2 = _c32(b1._scale_log2_bits(D))
    if not FIRST:  # B2: rescale factor
        scores_scale = ops.MufuEx2Ftz(ops.F32MulFtz(ops.F32SubFtz(m_prev, m_use), scale_log2))
    max_scaled = ops.F32MulFtz(m_use, scale_log2)
    p = [ops.MufuEx2Ftz(ops.F32FmaSubFtz(s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    lanes = []  # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    for ln in range(4):
        cols = [c for c in range(NVIS) if b1._lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else _c32(0)
        else:
            first = p[cols[0]] if cols else _c32(0)
            acc = ops.F32FmaFtz(scores_scale, l_prev[ln], first)
        for c in cols[1:]:
            acc = ops.F32AddFtz(acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in ascending 16-key k-steps
    pb = [ops.F2fpBf16(x) for x in p]
    zero16 = ops.const(16, 0) if NVIS < BN else None
    o = torch.stack([_c32(0) for _ in range(D)]) if FIRST else torch.stack([ops.F32MulFtz(o_prev[d], scores_scale) for d in range(D)])
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = torch.stack([(pb[k0 + j] if k0 + j < NVIS else zero16) for j in range(16)])
        vt = v[k0:k0 + 16].t()  # V column d over the 16 keys: a [D][16] strided view
        o = batch(step, o, prow, vt, axes=(0, None, 0))
    return (m_new, torch.stack(lanes), o)


@function(name="AttentionHead", version=2, static=("T", "D", "BN", "DOT", "INV"))
def AttentionHeadV2(q: Arr["D", BF16], k: Arr["T", "D", BF16], v: Arr["T", "D", BF16], *, T, D, BN, DOT, INV) -> Arr["D", BF16]:
    """Causal attention of one query at position T-1, ceil(T/BN) blocks visited last -> first, epilogue `INV` (b1.AttentionHeadV2)."""
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = ops.const(16, 0).expand(D)  # one zero gate broadcast: a zero-stride reference run
            pad = zero_row.expand(BN - nvis, D)
            kb = torch.cat([kb, pad])
            vb = torch.cat([vb, pad])
        first = b == nb - 1
        blk = bind(AttnBlockV2, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=DOT)
        if first:
            z = _c32(0)
            st = blk(q, kb, vb, z, z.expand(4), z.expand(D))
        else:
            st = blk(q, kb, vb, m, l, o)
        m, l, o = st
    total = ops.F32AddFtz(ops.F32AddFtz(l[0], l[2]), ops.F32AddFtz(l[1], l[3]))
    inv = _callable(INV)(total)
    y = batch(ops.F32MulFtz, o, inv, axes=(0, None))
    return batch(ops.F2fpBf16, y, axes=(0,))


@function(name="Attention", version=2, static=("T", "NH", "KVH", "D", "BN", "DOT", "INV"))
def AttentionV2(q: Arr["NH", "D", BF16], kc: Arr["T", "KVH", "D", BF16], vc: Arr["T", "KVH", "D", BF16], *, T, NH, KVH, D, BN, DOT, INV) \
        -> Arr["NH", "D", BF16]:
    """GQA over `[T][KVH][D]` caches through `AttentionHead_v2` (b1.AttentionV2)."""
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = kc.select(1, g)
        vg = vc.select(1, g)
        outs.append(batch(bind(AttentionHeadV2, T=T, D=D, BN=BN, DOT=DOT, INV=INV), q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return torch.cat(outs, 0)


@function(name="LayerPre", version=2, static=("C", "L0", "GEMM", "NORM"))
def LayerPreV2(h: Dep(_hidden), res: Dep(_hidden), w: Dep(_layer_w), cs: Dep(lambda S: Array(S["C"]["D"], BF16)), *, C, L0, GEMM, NORM) \
        -> Dep(lambda S: TupleT(_hidden(S), _heads(S), _kv_heads(S), _kv_heads(S))):
    """Layer front half with the linear and norm implementations as statics (b1.LayerPreV2)."""
    H, NH, KVH, D = C["H"], C["NH"], C["KVH"], C["D"]
    if L0:
        x = h
        n1 = _callable(NORM["L0"])(h, w["ln1"])
    else:
        n1, x = _callable(NORM["FUSED"])(h, res, w["ln1"])
    qkv = ops.Gemm_v2(K=H, N=C["QKV"], DOT=GEMM["DOT"])(n1, w["qkv_w"])
    if C.get("BIAS", True):
        qkv = ops.BiasAdd(N=C["QKV"])(qkv, w["qkv_b"])
    q = qkv[0:NH * D].view(NH, D)
    k = qkv[NH * D:(NH + KVH) * D].view(KVH, D)
    v = qkv[(NH + KVH) * D:(NH + 2 * KVH) * D].view(KVH, D)
    qr = ops.RoPE(NHEADS=NH, D=D)(q, cs)
    kr = ops.RoPE(NHEADS=KVH, D=D)(k, cs)
    return (x, qr, kr, v)


@function(name="LayerPost", version=2, static=("C", "GEMM", "NORM"))
def LayerPostV2(attn: Dep(_heads), res: Dep(_hidden), w: Dep(_layer_w), *, C, GEMM, NORM) -> Dep(lambda S: TupleT(_hidden(S), _hidden(S))):
    """Layer back half with the linear and norm implementations as statics (b1.LayerPostV2)."""
    H = C["H"]
    DOT = GEMM["DOT"]
    a = attn.reshape(C["NH"] * C["D"])
    o = ops.Gemm_v2(K=C["NH"] * C["D"], N=H, DOT=DOT)(a, w["o_w"])
    n2, r2 = _callable(NORM["FUSED"])(o, res, w["ln2"])
    gu = ops.Gemm_v2(K=H, N=C["GU"], DOT=DOT)(n2, w["gate_up_w"])
    act = ops.SiluMul(I=C["I"])(gu)
    d = ops.Gemm_v2(K=C["I"], N=H, DOT=DOT)(act, w["down_w"])
    return (d, r2)


@function(name="Final", version=2, static=("C", "GEMM", "NORM"))
def FinalV2(h: Dep(_hidden), res: Dep(_hidden), norm_w: Dep(_hidden), lm_head: Dep(lambda S: Array(S["C"]["V"], _hidden(S))), *, C, GEMM, NORM) -> I32:
    """Final fused add+norm, LM head GEMM, greedy token selection, implementations as statics (b1.FinalV2)."""
    n = _callable(NORM["FUSED"])(h, res, norm_w)[0]
    logits = ops.Gemm_v2(K=C["H"], N=C["V"], DOT=GEMM["DOT"])(n, lm_head)
    return ops.TokenSelect(V=C["V"])(logits)


def _forward_positions_v2(C: dict, GEMM, ATTENTION, NORM, toks, weights, T0: int, n_pos: int, caches):
    """`_forward_positions` over the v2 layer functions (b1._forward_positions_v2)."""
    NL, H = C["NL"], C["H"]
    embed = ops.Embedding(V=C["V"], H=H)
    hs = [embed(toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [LayerPreV2(hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i], C=C, L0=(L == 0), GEMM=GEMM, NORM=NORM,
                          name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = torch.stack([p[2] for p in pre])
        v_rows = torch.stack([p[3] for p in pre])
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = torch.cat([caches[L][0], k_rows])
            vc = torch.cat([caches[L][1], v_rows])
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = AttentionV2(pre[i][1], kc[0:T], vc[0:T], T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], **ATTENTION,
                               name=f"t={T - 1}/layers.{L}/attn")
            post = LayerPostV2(attn, pre[i][0], w, C=C, GEMM=GEMM, NORM=NORM, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post
    return new_kv, list(zip(hs, res))


@function(name="Serve", version=2, static=("C", "LP", "STEPS", "GEMM", "ATTENTION", "NORM"))
def ServeV2(prompt: Arr["LP", I32], weights: Dep(lambda S: b1.weights_type(S["C"])), *, C, LP, STEPS, GEMM, ATTENTION, NORM) \
        -> Arr["STEPS + 1", I32]:
    """Greedy serving of one request with the profile's implementation records as statics (b1.ServeV2): the Ampere
    program is `ServeV2` under `GEMM_AMPERE` / `ATTENTION_FA2` / `norm_bundle(C)`, the Hopper one differs in `GEMM` and
    `ATTENTION` only (`verity_vllm.program.registry.hopper.GEMM_HOPPER`, `.ATTENTION_FA3`)."""
    caches, finals = _forward_positions_v2(C, GEMM, ATTENTION, NORM, prompt, weights, 0, LP, None)
    tok = FinalV2(finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], C=C, GEMM=GEMM, NORM=NORM, name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _forward_positions_v2(C, GEMM, ATTENTION, NORM, tok.view(1), weights, T0, 1, caches)
        caches = [(torch.cat([caches[L][0], new_kv[L][0]]), torch.cat([caches[L][1], new_kv[L][1]])) for L in range(C["NL"])]
        tok = FinalV2(fin[0][0], fin[0][1], weights["norm"], weights["embed"], C=C, GEMM=GEMM, NORM=NORM, name=f"t={T0}/final")
        toks.append(tok)
    return torch.stack(toks)


AUTHORED = {f.id: f for f in (RMSNormTriton, RMSNormFusedCuda, AttnBlock, AttentionHead, Attention, LayerPre, LayerPost, Final, Serve,
                              AttnBlockV2, AttentionHeadV2, AttentionV2, LayerPreV2, LayerPostV2, FinalV2, ServeV2,
                              AttnBlockV3, AttentionHeadV3, AttentionV3, ServeV4)}

__all__ = ["RMSNormTriton", "RMSNormFusedCuda", "AttnBlock", "AttentionHead", "Attention", "LayerPre", "LayerPost", "Final", "Serve",
           "AttnBlockV2", "AttentionHeadV2", "AttentionV2", "LayerPreV2", "LayerPostV2", "FinalV2", "ServeV2",
           "AttnBlockV3", "AttentionHeadV3", "AttentionV3", "ServeV4",
           "GEMM_AMPERE", "ATTENTION_FA2", "attention_fa2", "norm_bundle", "AUTHORED"]
