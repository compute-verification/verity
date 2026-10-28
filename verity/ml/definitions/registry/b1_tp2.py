"""B1 served with `tensor_parallel_size=2`: TA1's typed serving program with the tensor-parallel functions
written as compositions of ordinary functions (coverage wave axis 2, lane TP).

Nothing in `b1.py` is modified. The TP=2 configuration is a different static configuration
(`B1_TP2_CONFIG`) whose layer functions replace the unsharded `Gemm` calls by

  * `batch(Gemm{K, N/2}, x, w_by_rank, axes=(None, 0))`   column-parallel: rank r computes output columns
    r*N/2 .. (r+1)*N/2 of the same input; no collective (qkv_proj, gate_up_proj, lm_head);
  * `batch(Gemm{K/2, N}, x_by_rank, w_by_rank, axes=(0, 0))` followed by `AllReduce2{N}` row-parallel:
    rank r contracts its half of K, the two bf16 partials are summed (o_proj, down_proj);
  * `EmbeddingShard{VS, H, START}` twice (rank as a static) + `AllReduce2{H}` for the vocab-parallel embedding;
  * `AllGather2{N}` for the logits: a gate-free reference rearrangement (a copy).

The rank axis is an ordinary `batch` axis: "rank r" is member r of a batched call, and the operands that
differ per rank are static views of the unsharded weight collections (`rank_rows`, `rank_cols`) — no gate
is added for sharding. Weights keep TA1's unsharded types, so the program has exactly one `Input` gate per
checkpoint scalar (one-producer rule): a rank's weight shard is a projection of those gates, and a weight
both ranks hold (norms, cos/sin) is one value with two authentication locations.

Every body node that production materialises carries `name=<capture op suffix>`; with the rank as the batch
member this *is* the identity <-> FunctionCall mapping (`profiles/b1-tp2/capture_mapping.py` reads it).
"""

from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, I32
from verity.ml.definitions.registry.b1 import (
    B1_CONFIG,
    ZERO16,
    Attention,
    BiasAdd,
    Gemm,
    RMSNormFusedCuda,
    RMSNormTriton,
    RoPE,
    SiluMul,
    TokenSelect,
    const,
    kv_type,
    layer_weights_type,
    weights_type,
)

B1_TP2_CONFIG: dict[str, Any] = dict(B1_CONFIG, TP=2)

# ---------------------------------------------------------------------------------------------------------
# static shard views (reference rearrangements: no gates)
# ---------------------------------------------------------------------------------------------------------


def rank_rows(w: Coll, N: int, K: int, TP: int) -> Coll:
    """`Array(N, Array(K))` -> `Array(TP, Array(N/TP, Array(K)))`: rank r owns output rows r*N/TP.. (column-parallel)."""
    return w.reshape(Array(TP, Array(N // TP, Array(K, BF16))))


def rank_cols(w: Coll, N: int, K: int, TP: int) -> Coll:
    """`Array(N, Array(K))` -> `Array(TP, Array(N, Array(K/TP)))`: rank r owns input columns r*K/TP.. (row-parallel)."""
    Kr = K // TP
    return Coll(Array(TP, Array(N, Array(Kr, BF16))),
                concat([strided_view(w, r * Kr, Kr, K, 1, N * Kr) for r in range(TP)]))


def rank_row_groups(parts: list[tuple[Coll, int, int]], TP: int) -> Coll:
    """Rank r's rows of a fused weight `[A; B; ...]` whose sub-tensors are each row-sharded: for every
    (sub-tensor, rows, K), rank r takes rows r*rows/TP.. of that sub-tensor (vLLM's `MergedColumnParallelLinear`
    / `QKVParallelLinear` shard each fused block separately)."""
    K = parts[0][2]
    per_rank = sum(rows // TP for _, rows, _ in parts)
    refs = []
    for r in range(TP):
        for coll, rows, _ in parts:
            n = rows // TP
            refs.append(coll[r * n:(r + 1) * n].refs)
    return Coll(Array(TP, Array(per_rank, Array(K, BF16))), concat(refs))


def rank_vec_groups(parts: list[tuple[Coll, int]], TP: int) -> Coll:
    per_rank = sum(n // TP for _, n in parts)
    refs = []
    for r in range(TP):
        for coll, n in parts:
            m = n // TP
            refs.append(coll[r * m:(r + 1) * m].refs)
    return Coll(Array(TP, Array(per_rank, BF16)), concat(refs))


# ---------------------------------------------------------------------------------------------------------
# communication computations: ordinary functions
# ---------------------------------------------------------------------------------------------------------


@composite("AllReduce2", 1, ["N"], lambda S: ((("parts", Array(2, Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="exact-model-tested")
def AllReduce2(B, S, parts):
    """NCCL ring/tree all-reduce of two bf16 vectors (`ncclSum`, `ncclBfloat16`): one `Bf16Add` per element,
    `bf16_rne(f32(a) + f32(b))`. With two ranks there is exactly one addition per element and addition is
    commutative, so the summation order is fixed by the arity: no rank-order dependence to specify.
    Conformance: B1 TP=2 commit, request 917b9faa, 11 positions x 4 layers (o_proj, down_proj, embedding):
    99/99 leaves exact from the two captured partials; 99/99 rank-1 copies bit-equal to rank 0's."""
    return B.batch(P.Bf16Add, parts[0], parts[1], axes=(0, 0))


@composite("AllGather2", 1, ["N"], lambda S: ((("parts", Array(2, Array(S.N, BF16))),), Array(2 * S.N, BF16)),
           conformance="exact-model-tested")
def AllGather2(B, S, parts):
    """`torch.ops.vllm.all_gather` along the column axis: rank r's `[N]` becomes columns r*N..(r+1)*N of
    every rank's `[2N]`. A copy: the output *is* the concatenation of the inputs (no gate).
    Conformance: 4/4 gathered logits rows are the two captured `lm_head.gemm` shards side by side; 4/4 rank-1
    copies bit-equal."""
    return parts.reshape(Array(2 * S.N, BF16))


# ---------------------------------------------------------------------------------------------------------
# World-parametric collectives.  IDENTITY RULE: a Definition's identity is `Name_vN{statics}` (defs.SpecializedDefinition.id),
# so the family NAME is part of every recorded Program's digest.  `AllReduce2_v1{N}` / `AllGather2_v1{N}` therefore stay the identities of
# every world-2 collective (the frozen rank Programs name them); `AllReduce_v2{WORLD,N}` / `AllGather_v1{WORLD,N}` are the
# identities for WORLD > 2 only.  `all_reduce(world)` / `all_gather(world)` are the ONE place a caller picks the family: world 2 -> the
# `*2` composites (bit-identical body: one Bf16Add per element), world > 2 -> the parametric ones.  Nothing in the two-rank path changes.
# ---------------------------------------------------------------------------------------------------------


def allreduce_order(world: int) -> tuple[int, ...]:
    """The hop order of the WORLD-rank all-reduce, `(WORLD-1, .., 1, 0)`: vLLM's batch-invariant mode pins `NCCL_ALGO=allreduce:tree`
    `NCCL_PROTO=Simple`, and NCCL's tree inside one node is a chain rooted at rank 0 that the reduce climbs from the leaf -- `acc =
    parts[WORLD-1]; acc = Bf16Add(acc, parts[r])` for r = WORLD-2..0.  The ONE statement of the order: the Definition body, the traced
    op, the Match value leg and the Commit's cross-rank pass all read it here."""
    return tuple(range(int(world) - 1, -1, -1))


@composite("AllReduce", 2, ["WORLD", "N"], lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="exact-model-tested WORLD=4 (4x L40S, SYS topology, NCCL_P2P_DISABLE=1, batch-invariant NCCL tree/Simple): 1056 of "
                       "1056 received all-reduces of a llama32-1b TP4 B8 capture equal this chain bit-exactly (32 of 1056 the ascending chain "
                       "of AllReduce_v1).  The chain is NCCL's intra-node tree in its topology order; the Match value leg (tp.match "
                       "provenance) checks every collective, so another topology's order is a named mismatch.  WORLD=2 is AllReduce2_v1.")
def AllReduce(B, S, parts):
    """`ncclAllReduce(ncclSum, ncclBfloat16)` over WORLD ranks' bf16 vectors in `allreduce_order(WORLD)`: one RNE rounding per addition, as
    NCCL's `FuncSum<__nv_bfloat16>`.  For WORLD = 2 this is one commutative addition per element -- the body of `AllReduce2` (kept as
    the world-2 identity)."""
    W = int(S.WORLD)
    if W < 2:
        raise ValueError(f"AllReduce{{WORLD={W}}}: an all-reduce needs at least two ranks")
    order = allreduce_order(W)
    acc = parts[order[0]]
    for r in order[1:]:
        acc = B.batch(P.Bf16Add, acc, parts[r], axes=(0, 0))
    return acc


@composite("AllGather", 1, ["WORLD", "N"], lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(int(S.WORLD) * S.N, BF16)),
           conformance="exact-model-tested (WORLD=2 as AllGather2_v1); WORLD>2: a reference rearrangement, no gates")
def AllGather(B, S, parts):
    """`torch.ops.vllm.all_gather` along the trailing axis over WORLD ranks: rank r's `[N]` becomes columns r*N..(r+1)*N of every rank's
    `[WORLD*N]`.  A copy: the output IS the concatenation of the parts in rank order (no gate)."""
    return parts.reshape(Array(int(S.WORLD) * S.N, BF16))


def all_reduce(world: int, N: int):
    """The all-reduce Definition bound for a `world`-rank communicator: `AllReduce2_v1{N}` at world 2 (the recorded identity), else
    `AllReduce_v2{WORLD,N}`.  Both take ONE `parts` operand `Array<world, Array<N, bf16>>` in rank order."""
    world = int(world)
    if world < 2:
        raise ValueError(f"all_reduce(world={world}): an all-reduce needs at least two ranks")
    return bind(AllReduce2, N=N) if world == 2 else bind(AllReduce, WORLD=world, N=N)


def all_gather(world: int, N: int):
    """The all-gather Definition for a `world`-rank communicator: `AllGather2_v1{N}` at world 2, else `AllGather_v1{WORLD,N}`."""
    world = int(world)
    if world < 2:
        raise ValueError(f"all_gather(world={world}): an all-gather needs at least two ranks")
    return bind(AllGather2, N=N) if world == 2 else bind(AllGather, WORLD=world, N=N)


@composite("EmbeddingShard", 1, ["VS", "H", "START"],
           lambda S: ((("tok", I32), ("table", Array(S.VS, Array(S.H, BF16)))), Array(S.H, BF16)),
           conformance="exact-model-tested")
def EmbeddingShard(B, S, tok, table):
    """One rank of `VocabParallelEmbedding` (`_C.vocab_parallel_embedding`): the row `table[tok - START]` if
    START <= tok < START + VS, else zeros (the other rank owns the token).
    Conformance: 22/22 per-rank partial rows exact (both ranks, incl. the zero rows of the non-owning rank)."""
    local = B.call(P.I32Add, tok, B.call(const(32, (-S.START) & 0xFFFFFFFF)))
    ge = B.call(P.I32Le, B.call(const(32, S.START)), tok)
    le = B.call(P.I32Le, tok, B.call(const(32, S.START + S.VS - 1)))
    cols = Coll(Array(S.H, Array(S.VS, BF16)), strided_view(table, 0, S.VS, 1, S.H, S.VS * S.H))
    row = B.batch(P.gather_bf16(S.VS), local, cols, axes=(None, 0))
    zero = B.call(ZERO16)
    lo_ok = B.batch(P.SelectBf16, ge, row, zero, axes=(None, 0, None))
    return B.batch(P.SelectBf16, le, lo_ok, zero, axes=(None, 0, None))


# ---------------------------------------------------------------------------------------------------------
# TP=2 layer functions
# ---------------------------------------------------------------------------------------------------------


def _sig_embedding_tp2(S):
    C = S.C
    return ((("tok", I32), ("table", Array(C["V"], Array(C["H"], BF16)))), Array(C["H"], BF16))


@composite("EmbeddingTP2", 1, ["C"], _sig_embedding_tp2, conformance="exact-model-tested")
def EmbeddingTP2(B, S, tok, table):
    """Vocab-parallel embedding: two `EmbeddingShard` calls (rank r owns rows r*V/2..(r+1)*V/2) summed by
    `AllReduce2`. Conformance: every leaf exact (22 EmbeddingShard + 11 AllReduce2)."""
    C = S.C
    V, H, TP = C["V"], C["H"], C["TP"]
    VS = V // TP
    parts = [B.call(bind(EmbeddingShard, VS=VS, H=H, START=r * VS), tok, table[r * VS:(r + 1) * VS],
                    name=f"embed_tokens.embedding/s={r}") for r in range(TP)]
    return B.call(bind(AllReduce2, N=H), array_of(parts), name="embed_tokens.all_reduce")


def _sig_layer_pre_tp2(S):
    C = S.C
    H, NH, KVH, D, TP = C["H"], C["NH"], C["KVH"], C["D"], C["TP"]
    return ((("h", Array(H, BF16)), ("res", Array(H, BF16)), ("w", layer_weights_type(C)), ("cs", Array(D, BF16))),
            Tuple((Array(H, BF16), Array(TP, Array(NH // TP, Array(D, BF16))), Array(KVH, Array(D, BF16)),
                   Array(KVH, Array(D, BF16)))))


@composite("LayerPreTP2", 1, ["C", "L0"], _sig_layer_pre_tp2, conformance="exact-model-tested")
def LayerPreTP2(B, S, h, res, w, cs):
    """Norm (replicated on both ranks), column-parallel qkv_proj (rank r: heads r*NH/TP.., kv head r),
    per-rank bias add and RoPE. Returns q as [rank][head][D]; k, v as [kv head = rank][D].
    Conformance: every leaf exact except TA1's recorded `RMSNormFusedCuda` tie-rounding class (2 words of
    248,832 compared across the TP=2 run, both on rank 0 with rank 1's copy bit-equal)."""
    C = S.C
    H, NH, KVH, D, TP = C["H"], C["NH"], C["KVH"], C["D"], C["TP"]
    assert KVH == TP, "one kv head per rank (Qwen2.5-1.5B: KVH=2)"
    NHr = NH // TP
    Q, KV = NH * D, KVH * D
    QKVr = (Q + 2 * KV) // TP
    if S.L0:
        n1 = B.call(bind(RMSNormTriton, N=H, EPS=C["EPS"]), h, w["ln1"], name="input_layernorm.rmsnorm")
        x = h
    else:
        n1x = B.call(bind(RMSNormFusedCuda, N=H, EPS=C["EPS"]), h, res, w["ln1"], name="input_layernorm.rmsnorm")
        n1, x = n1x[0], n1x[1]
    wq, wk, wv = w["qkv_w"][0:Q], w["qkv_w"][Q:Q + KV], w["qkv_w"][Q + KV:Q + 2 * KV]
    w_by_rank = rank_row_groups([(wq, Q, H), (wk, KV, H), (wv, KV, H)], TP)
    b = w["qkv_b"]
    b_by_rank = rank_vec_groups([(b[0:Q], Q), (b[Q:Q + KV], KV), (b[Q + KV:Q + 2 * KV], KV)], TP)
    qkv = B.batch(bind(Gemm, K=H, N=QKVr), n1, w_by_rank, axes=(None, 0), name="self_attn.qkv_proj.gemm")
    qkvb = B.batch(bind(BiasAdd, N=QKVr), qkv, b_by_rank, axes=(0, 0), name="self_attn.qkv_proj.add")
    Qr, KVr = NHr * D, D
    q_by_rank = Coll(Array(TP, Array(NHr, Array(D, BF16))), concat([qkvb[r].refs.slice(0, Qr) for r in range(TP)]))
    k_by_rank = Coll(Array(TP, Array(1, Array(D, BF16))), concat([qkvb[r].refs.slice(Qr, Qr + KVr) for r in range(TP)]))
    v_by_rank = Coll(Array(TP, Array(1, Array(D, BF16))), concat([qkvb[r].refs.slice(Qr + KVr, Qr + 2 * KVr) for r in range(TP)]))
    qr = B.batch(bind(RoPE, NHEADS=NHr, D=D), q_by_rank, cs, axes=(0, None), name="self_attn.rotary_emb.rope")
    kr = B.batch(bind(RoPE, NHEADS=1, D=D), k_by_rank, cs, axes=(0, None), name="self_attn.rotary_emb.rope.k")
    return tuple_of(x, qr, kr.reshape(Array(KVH, Array(D, BF16))), v_by_rank.reshape(Array(KVH, Array(D, BF16))))


def _sig_attention_tp2(S):
    C = S.C
    NH, KVH, D, TP, T = C["NH"], C["KVH"], C["D"], C["TP"], S.T
    return ((("q", Array(TP, Array(NH // TP, Array(D, BF16)))), ("kc", kv_type(C, T)), ("vc", kv_type(C, T))),
            Array(TP, Array(NH // TP, Array(D, BF16))))


@composite("AttentionTP2", 1, ["T", "C"], _sig_attention_tp2, conformance="exact-model-tested")
def AttentionTP2(B, S, q, kc, vc):
    """Heads split across ranks: rank r runs TA1's `Attention{T, NH/TP, KVH=1}` on its heads against its kv
    head's cache rows — a `batch` over ranks; no communication.
    Conformance: 88/88 per-rank 6-head attention outputs exact, T = 1 ... 527."""
    C = S.C
    NH, KVH, D, TP, T = C["NH"], C["KVH"], C["D"], C["TP"], S.T
    NHr = NH // TP
    kc_by_rank = Coll(Array(TP, Array(T, Array(1, Array(D, BF16)))),
                      concat([strided_view(kc, r * D, D, KVH * D, 1, T * D) for r in range(TP)]))
    vc_by_rank = Coll(Array(TP, Array(T, Array(1, Array(D, BF16)))),
                      concat([strided_view(vc, r * D, D, KVH * D, 1, T * D) for r in range(TP)]))
    return B.batch(bind(Attention, T=T, NH=NHr, KVH=1, D=D), q, kc_by_rank, vc_by_rank, axes=(0, 0, 0),
                   name="self_attn.attn.attn")


def _sig_layer_post_tp2(S):
    C = S.C
    H, NH, D, TP = C["H"], C["NH"], C["D"], C["TP"]
    return ((("attn", Array(TP, Array(NH // TP, Array(D, BF16)))), ("res", Array(H, BF16)), ("w", layer_weights_type(C))),
            Tuple((Array(H, BF16), Array(H, BF16))))


@composite("LayerPostTP2", 1, ["C"], _sig_layer_post_tp2, conformance="exact-model-tested")
def LayerPostTP2(B, S, attn, res, w):
    """Row-parallel o_proj + AllReduce2, fused norm (replicated), column-parallel gate_up_proj, per-rank
    SiluMul, row-parallel down_proj + AllReduce2. Returns (mlp output, residual) as TA1's LayerPost.
    Conformance: partials exact per rank, AllReduce2 exact, gate_up/SiluMul exact per rank; the fused norm
    carries TA1's recorded tie-rounding class (see LayerPreTP2)."""
    C = S.C
    H, NH, D, I, TP = C["H"], C["NH"], C["D"], C["I"], C["TP"]
    Q = NH * D
    a_by_rank = attn.reshape(Array(TP, Array(Q // TP, BF16)))
    o_part = B.batch(bind(Gemm, K=Q // TP, N=H), a_by_rank, rank_cols(w["o_w"], H, Q, TP), axes=(0, 0),
                     name="self_attn.o_proj.gemm")
    o = B.call(bind(AllReduce2, N=H), o_part, name="self_attn.o_proj.all_reduce")
    n2r = B.call(bind(RMSNormFusedCuda, N=H, EPS=C["EPS"]), o, res, w["ln2"], name="post_attention_layernorm.rmsnorm")
    n2, r2 = n2r[0], n2r[1]
    gu = w["gate_up_w"]
    gu_by_rank = rank_row_groups([(gu[0:I], I, H), (gu[I:2 * I], I, H)], TP)
    gate_up = B.batch(bind(Gemm, K=H, N=2 * I // TP), n2, gu_by_rank, axes=(None, 0), name="mlp.gate_up_proj.gemm")
    act = B.batch(bind(SiluMul, I=I // TP), gate_up, axes=(0,), name="mlp.act_fn.act_mul")
    d_part = B.batch(bind(Gemm, K=I // TP, N=H), act, rank_cols(w["down_w"], H, I, TP), axes=(0, 0),
                     name="mlp.down_proj.gemm")
    d = B.call(bind(AllReduce2, N=H), d_part, name="mlp.down_proj.all_reduce")
    return tuple_of(d, r2)


def _sig_final_tp2(S):
    C = S.C
    H, V = C["H"], C["V"]
    return ((("h", Array(H, BF16)), ("res", Array(H, BF16)), ("norm_w", Array(H, BF16)),
             ("lm_head", Array(V, Array(H, BF16)))), I32)


@composite("FinalTP2", 1, ["C"], _sig_final_tp2, conformance="model-untested")
def FinalTP2(B, S, h, res, norm_w, lm_head):
    """Final norm (replicated), column-parallel lm_head (rank r: vocab rows r*V/TP..), AllGather2 of the
    logits (every rank then holds the full row), greedy select — run identically on both ranks."""
    C = S.C
    H, V, TP = C["H"], C["V"], C["TP"]
    n = B.call(bind(RMSNormFusedCuda, N=H, EPS=C["EPS"]), h, res, norm_w, name="norm.rmsnorm")[0]
    logits_by_rank = B.batch(bind(Gemm, K=H, N=V // TP), n, rank_rows(lm_head, V, H, TP), axes=(None, 0), name="lm_head.gemm")
    logits = B.call(bind(AllGather2, N=V // TP), logits_by_rank, name="lm_head.all_gather")
    return B.call(bind(TokenSelect, V=V), logits, name="sampler.select")


# ---------------------------------------------------------------------------------------------------------
# the server: TA1's Serve with the TP=2 layer functions (same signature, same root call names)
# ---------------------------------------------------------------------------------------------------------


def _forward_positions_tp2(B, C: dict, toks: Coll, weights: Coll, T0: int, n_pos: int, caches: list[tuple[Coll, Coll]] | None):
    """As `b1._forward_positions` with the TP=2 functions: positions T0..T0+n_pos-1 through all layers."""
    NL, H = C["NL"], C["H"]
    hs = [B.call(bind(EmbeddingTP2, C=C), toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv: list[tuple[Coll, Coll]] = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [B.call(bind(LayerPreTP2, C=C, L0=(L == 0)), hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i],
                      name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = Coll(kv_type(C, n_pos), concat([p[2].refs for p in pre]))
        v_rows = Coll(kv_type(C, n_pos), concat([p[3].refs for p in pre]))
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = Coll(kv_type(C, T0 + n_pos), concat([caches[L][0].refs, k_rows.refs]))
            vc = Coll(kv_type(C, T0 + n_pos), concat([caches[L][1].refs, v_rows.refs]))
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = B.call(bind(AttentionTP2, T=T, C=C), pre[i][1], kc[0:T], vc[0:T], name=f"t={T - 1}/layers.{L}/attn")
            post = B.call(bind(LayerPostTP2, C=C), attn, pre[i][0], w, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post[0], post[1]
    return new_kv, list(zip(hs, res))


@composite("ServeTP2", 1, ["C", "LP", "STEPS"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))), Array(S.STEPS + 1, I32)),
           conformance="model-untested")
def ServeTP2(B, S, prompt, weights):
    """Greedy serving of one request on the TP=2 server: TA1's `Serve` shape (prefill of LP tokens, STEPS
    decode steps, STEPS+1 selections) over the TP=2 layer functions. Root call names are TA1's
    (`t=<t>/embed`, `t=<t>/layers.<L>/pre|attn|post`, `t=<t>/final`)."""
    C, LP, STEPS = S.C, S.LP, S.STEPS
    caches, finals = _forward_positions_tp2(B, C, prompt, weights, 0, LP, None)
    tok = B.call(bind(FinalTP2, C=C), finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _forward_positions_tp2(B, C, Coll(Array(1, I32), tok.refs), weights, T0, 1, caches)
        caches = [(Coll(kv_type(C, T0 + 1), concat([caches[L][0].refs, new_kv[L][0].refs])),
                   Coll(kv_type(C, T0 + 1), concat([caches[L][1].refs, new_kv[L][1].refs]))) for L in range(C["NL"])]
        tok = B.call(bind(FinalTP2, C=C), fin[0][0], fin[0][1], weights["norm"], weights["embed"], name=f"t={T0}/final")
        toks.append(tok)
    return Coll(Array(STEPS + 1, I32), concat([t.refs for t in toks]))


TP2_COMPOSITES = [AllReduce2, AllGather2, EmbeddingShard, EmbeddingTP2, LayerPreTP2, AttentionTP2, LayerPostTP2, FinalTP2, ServeTP2]


def serve_tp2(LP: int, STEPS: int, C: dict[str, Any] | None = None):
    return bind(ServeTP2, C=dict(C or B1_TP2_CONFIG), LP=LP, STEPS=STEPS)


TP_N_COMPOSITES = [AllReduce, AllGather]
