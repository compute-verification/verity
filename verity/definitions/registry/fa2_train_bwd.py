"""torch's FlashAttention-2 backward as a training step runs it on cc 12.0: `aten._scaled_dot_product_flash_attention_backward`
(causal, no dropout, the default scale) of bf16 values, torch's `pytorch_flash` backward kernels under deterministic algorithms,
one Definition per head.

    Fa2TrainBwdPre_v1{D}                    a query row before the loop: its dP sum (`flash_bwd_dot_do_o_kernel`) and its LSE in
                                            log2 units
    Fa2TrainBwdRow_v1{D,BN,NVIS}            a query row against one key block, NVIS of its keys visible: P and dS (bf16) and the
                                            row's dQ part
    Fa2TrainBwdKey_v1{D,BM,FIRST}           a key against one query block: its dV and dK accumulators through the block
    Fa2TrainBwdTile_v1{D,BM,BN,OFF,FIRST}   the loop body of `compute_dq_dk_dv_1colblock`: a query block against a key block
    Fa2TrainBwdDkDv_v1{D}                   a key after the loop: dK (bf16) and dV (bf16)
    Fa2TrainBwdDq_v1{D,NP,NS}               a query row's dQ (bf16) from its NP key blocks' parts in NS buffers
                                            (`flash_bwd_convert_dq_kernel`)
    Fa2TrainBwdHead_v1{T,D,BM,BN,NS}        a head of T rows

The kernel computes the gradients from its own inputs (dO, q, k, v, the forward's O and LSE) in five matrix products, each of
tensor-core k16 steps (`HopperBF16WgmmaDot16`) on bf16 operands into f32 accumulators, the scores among them, computed again.  A
thread block takes one key block of BN keys and walks the query blocks of BM rows from the last down to the key block's first
(BN a multiple of BM: only the blocks at or below the diagonal).  In a tile, for query row i and key j:

    S    = `DotBf16_v2{D}`(q_i, k_j), -inf at a masked key (j > i)
    P    = ex2(S * L - ms_i)                  FMUL, FADD, `Ex2Approx`; L = D^-0.5 * log2 e, ms_i = guard(LSE_i) * log2 e (FMUL)
    dP   = `DotBf16_v2{D}`(dO_i, v_j)
    dS   = P * (dP - dPsum_i)                 FADD, FMUL; a masked key's dP is computed too (its P times an inf is a NaN)
    dV_j += sum_i bf16(P) dO_i,  dK_j += sum_i bf16(dS) q_i     k16 steps over the block's rows in turn, one accumulator a key
                                              through its query blocks
    dQ part of row i = sum_j bf16(dS) k_j     k16 steps over the block's keys in turn, from +0

dPsum_i is dO_i . O_i in f32 as `dot_do_o` adds it: a thread holds 8 head dims of each page of W (64 at a D that is a multiple of
64, else 32), multiplies and adds them in turn (FMUL, FADD), and the W / 8 threads of the row add by a butterfly, xor W / 16 down
to 1.  After the loop dV_j = bf16(acc) and dK_j = bf16(acc * D^-0.5).  A row's dQ parts are added in f32: thread block x adds the
parts of key blocks x, x + NS, .. in turn into its own buffer from +0, `flash_bwd_convert_dq_kernel` adds the buffers in turn,
and dQ_i = bf16(sum * D^-0.5).  NS is the kernel's split count under deterministic algorithms, ceil(SMs / (B H)).  Without
deterministic algorithms every thread block adds into one buffer with atomics, in the order the card schedules them: no function
of the inputs, and no Definition here.

torch builds these kernels without fast math, so every f32 operation is the GPU's without flush-to-zero, as in `fa2_train`.
What the Definitions leave out is the identity on every word: the dP sum's and dV's FMUL by a dropout factor of 1, the convert
kernel's first add (+0 plus a buffer that is never -0), its adds of the buffers no thread block wrote (+0 to a sum that is
never -0), and a masked score's FMUL (-inf times L).
"""
from __future__ import annotations

import numpy as np

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll as _Coll, array_of, concat, strided_view, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.evidence import EVIDENCE
from verity.definitions.ops import scalar as SC
from verity.definitions.ops.gemm import DotBf16 as DotBf16V2
from verity.definitions.ops.prims import HopperBF16WgmmaDot16, ZERO32, const
from verity.definitions.registry import prims as P
from verity.definitions.registry.b1 import NEG_INF32, scale_log2_bits
from verity.definitions.registry.dsv4_moe import Ex2Approx
from verity.definitions.registry.prims import A16, BF16, F32

#: float(M_LOG2E)
LOG2E = 0x3FB8AA3B
#: the SMs of the capture's GPU, which torch divides by B * H for the dQ buffers under deterministic algorithms
SMS = 188

#: torch's flash attention backward at each (B, H, T, D) the capture covers: (BM, BN) of the head-dim-64 kernel
TRAIN_SDPA_BWD = {(2, 8, 1024, 64): (64, 128), (2, 12, 1024, 64): (64, 128), (2, 16, 1024, 64): (64, 128),
                  (2, 24, 1024, 64): (64, 128)}

_FA2 = EVIDENCE["cc12_fa2_backward_capture"]
CONFORMANCE = ("exact-model-tested: every word of dQ, dK and dV of torch's flash attention backward "
               "(aten._scaled_dot_product_flash_attention_backward, causal, bf16, head dim 64) under deterministic algorithms "
               "on cc 12.0 with 188 SMs (torch 2.14.0+cu130) at B = 2, T = 1,024 and H = 8, 12, 16 and 24 "
               "(`flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, BM = 64, BN = 128, 12, 8, 6 and 4 dQ buffers), on each batch "
               "member's first and last heads with q and k N(0, 1) and N(0, 9), v N(0, 1) and a random bf16 dO "
               f"(capture {_FA2['capture']}, run record {_FA2['run_record']})")


def _refuse(b: dict) -> str | None:
    D, BM, BN, NVIS, OFF, NP, NS = (b.get(n) for n in ("D", "BM", "BN", "NVIS", "OFF", "NP", "NS"))
    if D is not None and (D < 32 or D % 32):
        return f"D = {D}: a row's threads hold 8 head dims of each page of 32 or 64 (D a positive multiple of 32)"
    for name, n in (("BM", BM), ("BN", BN)):
        if n is not None and (n < 16 or n % 16):
            return f"{name} = {n}: the products are whole k16 steps over a block ({name} a positive multiple of 16)"
    if BM is not None and BN is not None and BN % BM:
        return f"BN = {BN}: a key block starts at a query block (BN a multiple of BM = {BM})"
    if NVIS is not None and BN is not None and not 1 <= NVIS <= BN:
        return f"NVIS = {NVIS}: a row of a visited tile sees 1 to BN = {BN} of its keys"
    if OFF is not None and BM is not None and BN is not None and (OFF % BM or not 0 <= OFF <= BN):
        return f"OFF = {OFF}: a tile's first row is 0, BM, .. below its key block's first key, BN = {BN} where all are visible"
    for name, n in (("NP", NP), ("NS", NS)):
        if n is not None and n < 1:
            return f"{name} = {n}: a row has at least one part, in at least one buffer"
    return None


def _refuse_head(b: dict) -> str | None:
    T, BN = b.get("T"), b.get("BN")
    if T is not None and BN is not None and (T < BN or T % BN):
        return f"T = {T}: the head is whole key blocks of BN = {BN}"
    return _refuse(b)


def _c(B, bits: int):
    return B.call(const(32, bits))


def _sub(B, a, b):
    """a - b: FADD of a and b negated."""
    return B.call(P.F32AddV2, a, B.call(SC.F32Neg, b))


def _scale(D: int) -> int:
    """`scale_softmax_rp_dropout`: float(D^-0.5) times a dropout factor of 1."""
    return P.bits_of(np.float32(D ** -0.5))


def _zeros(B, n: int):
    return _Coll(Array(n, F32), concat([B.call(ZERO32).refs] * n))


def _steps(B, acc, a, b, D: int):
    """acc[d] += sum_i a[i] b[i][d] by k16 steps over i in turn: a is n bf16 words, b n rows of D."""
    for s in range(len(a) // 16):
        cols = _Coll(Array(D, A16), strided_view(b[16 * s:16 * s + 16], 0, 16, 1, D, 16 * D))
        acc = B.batch(HopperBF16WgmmaDot16, acc, _Coll(A16, a[16 * s:16 * s + 16].refs), cols, axes=(0, None, 0))
    return acc


def _vec(S):
    return Array(S.D, BF16)


def _rows(S, n: int, t=BF16):
    return Array(n, Array(S.D, t))


@composite("Fa2TrainBwdPre", 1, ["D"], lambda S: ((("do", _vec(S)), ("o", _vec(S)), ("lse", F32)), Tuple(F32, F32)),
           conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBwdPre(B, S, do, o, lse):
    """A query row's dP sum and scaled LSE: dPsum = dO . O, each of the row's W / 8 threads (W = 64 at a D that is a multiple of
    64, else 32) adding the products of its 8 head dims of each page in turn and the threads adding by a butterfly; ms =
    guard(lse) * log2 e."""
    D = S.D
    W = 64 if D % 64 == 0 else 32
    x = []
    for t in range(W // 8):
        acc = None
        for d in (page * W + 8 * t + e for page in range(D // W) for e in range(8)):
            prod = B.call(P.F32MulV2, B.call(P.Bf16ToF32, do[d]), B.call(P.Bf16ToF32, o[d]))
            acc = prod if acc is None else B.call(P.F32AddV2, acc, prod)
        x.append(acc)
    while len(x) > 1:
        half = len(x) // 2
        x = [B.call(P.F32AddV2, x[t], x[t + half]) for t in range(half)]
    return tuple_of(x[0], B.call(P.F32MulV2, B.call(P.GuardNegInfZero, lse), _c(B, LOG2E)))


@composite("Fa2TrainBwdRow", 1, ["D", "BN", "NVIS"],
           lambda S: ((("q", _vec(S)), ("k", _rows(S, S.BN)), ("v", _rows(S, S.BN)), ("do", _vec(S)), ("dpsum", F32),
                       ("ms", F32)), Tuple(Array(S.BN, BF16), Array(S.BN, BF16), Array(S.D, F32))),
           conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBwdRow(B, S, q, k, v, do, dpsum, ms):
    """A query row against a key block whose keys 0..NVIS-1 it sees.  Returns P and dS of the BN keys, in bf16, and the row's dQ
    part: BN / 16 k16 steps of bf16(dS) and the keys, from +0."""
    D, BN, NVIS = S.D, S.BN, S.NVIS
    L = _c(B, scale_log2_bits(D))
    dot = bind(DotBf16V2, K=D, DOT=HopperBF16WgmmaDot16)
    if NVIS < BN:
        masked = B.call(Ex2Approx, _sub(B, B.call(NEG_INF32), ms))
        masked16 = B.call(P.F2fpBf16, masked)
    pb, dsb = [], []
    for c in range(BN):
        p = B.call(Ex2Approx, _sub(B, B.call(P.F32MulV2, B.call(dot, q, k[c]), L), ms)) if c < NVIS else masked
        pb.append(B.call(P.F2fpBf16, p) if c < NVIS else masked16)
        dsb.append(B.call(P.F2fpBf16, B.call(P.F32MulV2, p, _sub(B, B.call(dot, do, v[c]), dpsum))))
    ds16 = array_of(dsb)
    return tuple_of(array_of(pb), ds16, _steps(B, _zeros(B, D), ds16, k, D))


def _key_type(S):
    col = Array(S.BM, BF16)
    params = (("p", col), ("ds", col), ("do", _rows(S, S.BM)), ("q", _rows(S, S.BM)))
    if not S.FIRST:
        params += (("dv_prev", Array(S.D, F32)), ("dk_prev", Array(S.D, F32)))
    return params, Tuple(Array(S.D, F32), Array(S.D, F32))


@composite("Fa2TrainBwdKey", 1, ["D", "BM", "FIRST"], _key_type, conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBwdKey(B, S, p, ds, do, q, dv_prev=None, dk_prev=None):
    """A key through one query block: dV += sum_i p_i dO_i and dK += sum_i ds_i q_i, BM / 16 k16 steps each over the block's
    rows in turn.  FIRST: the key block's first visited query block, both accumulators from +0."""
    D = S.D
    dv = _steps(B, _zeros(B, D) if S.FIRST else dv_prev, p, do, D)
    dk = _steps(B, _zeros(B, D) if S.FIRST else dk_prev, ds, q, D)
    return tuple_of(dv, dk)


def _tile_type(S):
    params = (("q", _rows(S, S.BM)), ("k", _rows(S, S.BN)), ("v", _rows(S, S.BN)), ("do", _rows(S, S.BM)),
              ("pre", Array(S.BM, Tuple(F32, F32))))
    if not S.FIRST:
        params += (("dv_prev", _rows(S, S.BN, F32)), ("dk_prev", _rows(S, S.BN, F32)))
    return params, Tuple(_rows(S, S.BN, F32), _rows(S, S.BN, F32), _rows(S, S.BM, F32))


@composite("Fa2TrainBwdTile", 1, ["D", "BM", "BN", "OFF", "FIRST"], _tile_type, conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBwdTile(B, S, q, k, v, do, pre, dv_prev=None, dk_prev=None):
    """A query block against a key block whose first key is OFF rows above the query block's first row: row r is
    `Fa2TrainBwdRow_v1{NVIS = min(BN, OFF + r + 1)}` with its `Fa2TrainBwdPre_v1` pair, then key c is `Fa2TrainBwdKey_v1` on
    column c of the rows' P and dS.  Returns the keys' dV and dK accumulators and the rows' dQ parts."""
    D, BM, BN = S.D, S.BM, S.BN
    rows = [B.call(bind(Fa2TrainBwdRow, D=D, BN=BN, NVIS=min(BN, S.OFF + r + 1)), q[r], k, v, do[r], pre[r][0], pre[r][1])
            for r in range(BM)]
    key = bind(Fa2TrainBwdKey, D=D, BM=BM, FIRST=S.FIRST)
    dv, dk = [], []
    for c in range(BN):
        col = [_Coll(Array(BM, BF16), concat([rows[r][f][c].refs for r in range(BM)])) for f in (0, 1)]
        y = B.call(key, *col, do, q, *(() if S.FIRST else (dv_prev[c], dk_prev[c])))
        dv.append(y[0])
        dk.append(y[1])
    return tuple_of(array_of(dv), array_of(dk), array_of([y[2] for y in rows]))


@composite("Fa2TrainBwdDkDv", 1, ["D"],
           lambda S: ((("dv", Array(S.D, F32)), ("dk", Array(S.D, F32))), Tuple(_vec(S), _vec(S))),
           conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBwdDkDv(B, S, dv, dk):
    """A key's accumulators after its last query block -> (dK, dV): dK = bf16(dk * D^-0.5), dV = bf16(dv)."""
    scaled = B.batch(P.F32MulV2, dk, _c(B, _scale(S.D)), axes=(0, None))
    return tuple_of(B.batch(P.F2fpBf16, scaled, axes=(0,)), B.batch(P.F2fpBf16, dv, axes=(0,)))


@composite("Fa2TrainBwdDq", 1, ["D", "NP", "NS"], lambda S: ((("parts", _rows(S, S.NP, F32)),), _vec(S)),
           conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBwdDq(B, S, parts):
    """A query row's dQ from its parts of key blocks 0..NP-1: buffer x < NS is +0 plus the parts of blocks x, x + NS, .. in turn,
    the buffers are added in turn, and dQ = bf16(sum * D^-0.5)."""
    zeros = _zeros(B, S.D)
    acc = None
    for x in range(min(S.NS, S.NP)):
        buf = zeros
        for b in range(x, S.NP, S.NS):
            buf = B.batch(P.F32AddV2, buf, parts[b], axes=(0, 0))
        acc = buf if acc is None else B.batch(P.F32AddV2, acc, buf, axes=(0, 0))
    scaled = B.batch(P.F32MulV2, acc, _c(B, _scale(S.D)), axes=(0, None))
    return B.batch(P.F2fpBf16, scaled, axes=(0,))


def _head_type(S):
    rows = _rows(S, S.T)
    return ((("do", rows), ("q", rows), ("k", rows), ("v", rows), ("o", rows), ("lse", Array(S.T, F32))),
            Tuple(rows, rows, rows))


@composite("Fa2TrainBwdHead", 1, ["T", "D", "BM", "BN", "NS"], _head_type, conformance=CONFORMANCE, refuse=_refuse_head)
def Fa2TrainBwdHead(B, S, do, q, k, v, o, lse):
    """The causal backward of a head's T rows, the operator's arguments in its order (the upstream gradient, q, k, v, the
    forward's O and LSE): `Fa2TrainBwdPre_v1` on every row; key block n's tiles over query blocks T / BM - 1 down to n BN / BM,
    the first one FIRST, each with OFF = min(BN, the query block's first row less the key block's first key); `Fa2TrainBwdDkDv_v1`
    on every key; `Fa2TrainBwdDq_v1{NP}` on every row of a query block that NP key blocks visit.  Returns (dQ, dK, dV)."""
    T, D, BM, BN = S.T, S.D, S.BM, S.BN
    pre = B.batch(bind(Fa2TrainBwdPre, D=D), do, o, lse, axes=(0, 0, 0))
    last = T // BM - 1
    parts: list[list] = [[] for _ in range(T // BM)]
    dk, dv = [], []
    for n in range(T // BN):
        keys = slice(n * BN, (n + 1) * BN)
        st = None
        for m in range(last, n * BN // BM - 1, -1):
            rows = slice(m * BM, (m + 1) * BM)
            tile = bind(Fa2TrainBwdTile, D=D, BM=BM, BN=BN, OFF=min(BN, m * BM - n * BN), FIRST=m == last)
            st = B.call(tile, q[rows], k[keys], v[keys], do[rows], pre[rows], *(() if m == last else (st[0], st[1])))
            parts[m].append(st[2])
        out = B.batch(bind(Fa2TrainBwdDkDv, D=D), st[0], st[1], axes=(0, 0))
        dk.append(strided_view(out, 0, D, 2 * D, 1, BN * D))
        dv.append(strided_view(out, D, D, 2 * D, 1, BN * D))
    dq = [B.batch(bind(Fa2TrainBwdDq, D=D, NP=len(ps), NS=S.NS), array_of(ps), axes=(1,)).refs for ps in parts]
    rows_t = _rows(S, T)
    return tuple_of(_Coll(rows_t, concat(dq)), _Coll(rows_t, concat(dk)), _Coll(rows_t, concat(dv)))


def train_sdpa_backward(B: int, H: int, T: int, D: int):
    """The head Definition of torch's flash attention backward at (B, H, T, D) on cc 12.0 under deterministic algorithms, bound
    (`TRAIN_SDPA_BWD`, the dQ buffers ceil(`SMS` / (B H))); ValueError at a shape the capture doesn't cover."""
    blocks = TRAIN_SDPA_BWD.get((B, H, T, D))
    if blocks is None:
        raise ValueError(f"(B, H, T, D) = {(B, H, T, D)} is not captured: torch's flash attention backward is fitted at "
                         f"{sorted(TRAIN_SDPA_BWD)}")
    return bind(Fa2TrainBwdHead, T=T, D=D, BM=blocks[0], BN=blocks[1], NS=-(-SMS // (B * H)))


__all__ = ["CONFORMANCE", "Fa2TrainBwdDkDv", "Fa2TrainBwdDq", "Fa2TrainBwdHead", "Fa2TrainBwdKey", "Fa2TrainBwdPre",
           "Fa2TrainBwdRow", "Fa2TrainBwdTile", "LOG2E", "SMS", "TRAIN_SDPA_BWD", "train_sdpa_backward"]
