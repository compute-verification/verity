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

from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ir.types import Array, Tuple
from verity_vllm.registry import prims as P
from verity_vllm.registry.prims import BF16, I32
from verity_vllm.registry.b1 import (
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
# [lane vllm-tp-n] world-parametric collectives.  IDENTITY RULE: a Definition's identity is `Name_vN{statics}` (defs.SpecializedDefinition.id),
# so the family NAME is part of every recorded Program's digest.  `AllReduce2_v1{N}` / `AllGather2_v1{N}` therefore stay the identities of
# every world-2 collective (the frozen #70 / #75 rank Programs name them); `AllReduce_v1{WORLD,N}` / `AllGather_v1{WORLD,N}` are the
# identities for WORLD > 2 only.  `all_reduce(world)` / `all_gather(world)` are the ONE place a caller picks the family: world 2 -> the
# `*2` composites (bit-identical body: one Bf16Add per element), world > 2 -> the parametric ones.  Nothing in the two-rank path changes.
# ---------------------------------------------------------------------------------------------------------


@composite("AllReduce", 1, ["WORLD", "N"], lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="model-untested (WORLD>2): NCCL reduces in the datatype, one bf16 rounding per hop; the hop ORDER is NCCL's "
                       "algorithm/topology (ring: per-chunk start rank; tree: leaves->root), not recorded in the capture -- this body STATES "
                       "the sequential rank order 0->WORLD-1 and the Match value leg (tp.match provenance) checks it bit-exactly against the "
                       "received bytes; a mismatch is a named finding, never relabelled.  WORLD=2 is AllReduce2_v1 (exact-model-tested).")
def AllReduce(B, S, parts):
    """`ncclAllReduce(ncclSum, ncclBfloat16)` over WORLD ranks' bf16 vectors: `acc = parts[0]; acc = Bf16Add(acc, parts[r])` for r = 1..WORLD-1
    (one RNE rounding per addition, as NCCL's `FuncSum<__nv_bfloat16>`), the accumulation order being the sequential rank order.  For
    WORLD = 2 this is one commutative addition per element -- the body of `AllReduce2` (kept as the world-2 identity)."""
    W = int(S.WORLD)
    if W < 2:
        raise ValueError(f"AllReduce{{WORLD={W}}}: an all-reduce needs at least two ranks")
    acc = parts[0]
    for r in range(1, W):
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
    `AllReduce_v1{WORLD,N}`.  Both take ONE `parts` operand `Array<world, Array<N, bf16>>` in rank order."""
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


TP_N_COMPOSITES = [AllReduce, AllGather]


def serve_tp2(LP: int, STEPS: int, C: dict[str, Any] | None = None):
    return bind(ServeTP2, C=dict(C or B1_TP2_CONFIG), LP=LP, STEPS=STEPS)
