"""GLM-5.3's TP8 all-reduces on 8x H200 (vLLM d9105ea80, FlashInfer 0.6.18), as word Definitions.

    AllReduceMnnvlCoordinate_v1{WORLD,SHOT}   (col: Array<WORLD, bf16>) -> bf16
    AllReduceMnnvl_v1{WORLD,N,SHOT}           (parts: Array<WORLD, Array<N, bf16>>) -> Array<N, bf16>
    AllReduceRmsNormMnnvl_v1{WORLD,N,SHOT,CLUSTER,EPS,FMA}
                                              (parts, res: Array<N, bf16>, w: Array<N, bf16>) -> (normed, residual')
    AllReduceCustomCoordinate_v1{WORLD,START} (col: Array<WORLD, bf16>) -> bf16
    AllReduceCustom_v1{WORLD,SIZE,STAGE}      (parts: Array<WORLD, Array<SIZE, bf16>>) -> Array<SIZE, bf16>
    AllReduceMultimemCoordinate_v1{WORLD,TREE,FTZ}
                                              (col: Array<WORLD, bf16>) -> bf16
    AllReduceMultimem_v1{WORLD,SIZE,TREES,FTZ}
                                              (parts: Array<WORLD, Array<SIZE, bf16>>) -> Array<SIZE, bf16>

Which kernel reduces a call (vLLM d9105ea80):

* `fused_allreduce_rms_norm` (`vllm/models/common/ops/fused_allreduce_rms_norm.py`, every o_proj / MoE output of DeepSeek-V3.2's
  decoder layer) runs FlashInfer's `trtllm_allreduce_fusion` when `_can_use_flashinfer` admits the call: M <= 0.5 MiB / (H * 2 B),
  42 tokens at H 6144 and TP 8, and the workspace's buffer holds it. The backend is `_resolve_fi_ar_backend`'s: "mnnvl" on one
  node with NVSwitch multicast (`VLLM_FLASHINFER_ALLREDUCE_BACKEND` pins it), whose kernels are compiled with ``-use_fast_math``.
  `mnnvl_shot`: one-shot (`oneshotAllreduceFusionKernel`) when M * H * WORLD * 2 B <= 1 MiB (M <= 10 at H 6144, TP 8), else
  two-shot (`twoshotAllreduceKernel` then `rmsNormLamport`). `AllReduceRmsNormMnnvl_v1` is either, one token.
* Otherwise the call is `tensor_model_parallel_all_reduce` then `RMSNorm.forward_cuda` (`fused_add_rms_norm`,
  `b1.RMSNormFusedCuda_v2`). On sm_90 at TP 8 that all-reduce is torch's symmetric-memory `multimem_all_reduce_` (NVSwitch
  in-network reduction, whose order no Definition states) unless ``VLLM_ALLREDUCE_USE_SYMM_MEM=0``; then vLLM's custom all-reduce
  up to 8 MiB (`AllReduceCustom_v1`; `custom_stage` picks its kernel, ``VLLM_CUSTOM_ALLREDUCE_ALGO`` pins it) and NCCL past it.

The MNNVL all-reduce (`reduceOneshotDeterministic`, `reduceLamportRanksChunked`): every rank's bf16 payload has -0 (0x8000, the
Lamport sentinel) written as +0 (`sanitizeLamportPayload`); the f32 sum starts at +0 and adds rank 0, 1, ..., WORLD-1 (FADD.FTZ)
on every rank; two-shot adds that sum to a +0 once more (``accum += chunkAccum``) before F2FP. A one-shot sum is -0 when a
negative subnormal sum flushes; the two-shot one never is.

The fused norm, one token over CLUSTER CTAs of BLOCK = N / (8 * CLUSTER) threads (`mnnvl_grid`, FlashInfer's `adjustGridConfig`),
thread t of CTA b holding elements b * N / CLUSTER + 8t .. + 8:

* r = a + res (bf16 ``__hadd``, `Bf16AddF2fp_v1`), the new residual;
* each thread's sum from +0: one-shot adds ``f32(bf16(r * r))`` (``__hmul``, `Bf16MulBf16_v1`), two-shot ``r * r`` contracted
  into FFMA;
* `blockReduceSum`: each warp's xor butterfly 16, 8, 4, 2, 1 (lane 0's cone: every lane ends with its value), the warp sums
  into the lanes of one more butterfly, lanes past the warps holding +0;
* CLUSTER > 1: the CTAs' sums added to +0 in CTA order;
* ``rsqrtf(sum / N + EPS)``: ``-use_fast_math`` makes the division MUFU.RCP of N then FMUL.FTZ, and FMA names whether ptxas
  contracts that FMUL with the + EPS (the H200 capture decides); MUFU.RSQ on an argument >= EPS (a positive normal) is
  `RsqrtApprox_v1`;
* the output ``(f32(r) * rs) * (0 + f32(w))`` one-shot, ``((0 + f32(w)) * f32(r)) * rs`` two-shot (``weightBias`` = 0.0 is an
  FADD), F2FP.

vLLM's custom all-reduce (`csrc/custom_all_reduce.cuh`, built without fast math) sums f32 from the first rank's value (FADD)
and casts with ``__float2bfloat16``. The 1-stage kernel reads ranks 0, 1, ..., WORLD-1 on every rank; the 2-stage one cuts the
8-element packs into WORLD parts, the last rank taking the rest, and the owner of a pack reads ranks from its own: owner,
owner + 1, ... mod WORLD (`custom_owners`).

Past FlashInfer's 0.5 MiB at TP 8 on sm_90, `CudaCommunicator.all_reduce` hands a bf16 tensor to torch's symmetric memory (to
64 MiB) when vLLM's custom all-reduce is off, as the served engine has it (``disable_custom_all_reduce``). At 4, 6 or 8 ranks on
cc 9.0 that is `torch.ops.symm_mem.multimem_all_reduce_` (torch 2.13 `CUDASymmetricMemoryOps.cu`): rank r issues
``multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2`` over its slice of ``round_up(SIZE, 16 * WORLD) / WORLD``
elements (`multimem_owners`) and ``multimem.st``s the result to every rank. The NVSwitch reduces the WORLD payloads in f32 in an
order no source states; `AllReduceMultimem_v1` takes it per owner as TREES, binary trees over the ranks whose nodes are f32 adds
(IEEE, or FTZ when FTZ), the 8-GPU capture's probes reading each tree off the switch (`tree_from_clusters`), then RN to bf16.

f32 NaN words are the GPU's (`F32AddFtz_v2`, `F32MulFtz_v2`, `F32FmaFtz_v2`, `F32Add_v2`); every output is F2FP's bf16, whose NaN
is 0x7FFF.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.circuits.refs import array_of, coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16

__all__ = ["AllReduceCustom", "AllReduceCustomCoordinate", "AllReduceMnnvl", "AllReduceMnnvlCoordinate", "AllReduceMultimem",
           "AllReduceMultimemCoordinate", "AllReduceRmsNormMnnvl", "CUSTOM_MAX_BYTES", "FI_MAX_BYTES", "H200_SMS",
           "MNNVL_ONE_SHOT_BYTES", "SYMM_MEM_MAX_BYTES", "WARP", "chain", "custom_order", "custom_owners", "custom_stage",
           "eps_bits", "fused_max_tokens", "mnnvl_grid", "mnnvl_shot", "multimem_owners", "multimem_runs", "multimem_tree",
           "multimem_trees", "ranks", "strategy", "tree_clusters", "tree_from_clusters", "tree_leaves"]

WARP = 32
#: FlashInfer `MNNVL_ONE_SHOT_THRESHOLD`: the one-shot kernel up to this many bytes of M * H * WORLD
MNNVL_ONE_SHOT_BYTES = 64 * 1024 * 8 * 2
#: vLLM `_max_token_num`'s budget: `fused_allreduce_rms_norm` admits M * H * 2 B up to it
FI_MAX_BYTES = 512 * 1024
#: vLLM's custom all-reduce buffer (`CustomAllreduce.max_size`)
CUSTOM_MAX_BYTES = 8 * 1024 * 1024
#: vLLM `SYMM_MEM_ALL_REDUCE_MAX_SIZES["9.0"][8]`: torch's symmetric memory takes a bf16 all-reduce up to this many bytes
SYMM_MEM_MAX_BYTES = 64 * 1024 * 1024
#: torch's `multimem_all_reduce_` slices are whole 16-byte vectors per rank
MULTIMEM_ALIGN_BYTES = 16
H200_SMS = 132
_NEG_ZERO32 = const(32, 0x80000000)


def ranks(W) -> int:
    W = int(W)
    if W < 2:
        raise ValueError(f"WORLD={W}: an all-reduce needs at least two ranks")
    return W


def strategy(shot) -> int:
    shot = int(shot)
    if shot not in (1, 2):
        raise ValueError(f"SHOT={shot}: FlashInfer's MNNVL all-reduce is one-shot (1) or two-shot (2)")
    return shot


def eps_bits(eps: float) -> int:
    """EPS as an f32 word, which must be a positive normal: then ``sum / N + EPS`` is one too, or +inf or NaN, where
    ``rsqrt.approx.ftz`` and `RsqrtApprox_v1` agree."""
    u = P.bits_of(np.float32(eps))
    if u >> 31 or not (0 < (u >> 23) & 0xFF < 0xFF):
        raise ValueError(f"EPS={eps!r}: the fused norm's epsilon must be a positive normal f32")
    return u


def mnnvl_shot(M: int, N: int, world: int) -> int:
    """FlashInfer's MNNVL strategy for M tokens of N bf16 over `world` ranks: 1 (one-shot) or 2 (two-shot)."""
    return 1 if M * N * world * 2 <= MNNVL_ONE_SHOT_BYTES else 2


def fused_max_tokens(N: int, tp: int) -> int:
    """vLLM's `_max_token_num(tp, N, bf16)` for TP <= 8 on sm_90: the most tokens `fused_allreduce_rms_norm` hands FlashInfer
    (the workspace's buffer check may admit fewer, which the capture records per call)."""
    if tp > 8:
        raise ValueError(f"tp={tp}: vLLM's fused all-reduce budget is stated here for TP <= 8")
    return FI_MAX_BYTES // (N * 2)


def mnnvl_grid(M: int, N: int, sm_count: int = H200_SMS) -> tuple[int, int]:
    """(BLOCK, CLUSTER) of FlashInfer's `adjustGridConfig(M, N, 8, useCluster=true)` with one load per thread, the geometry of
    both fused kernels; at N = 6144 on an H200: M <= 16 (96, 8), 17..33 (192, 4), 34..66 (384, 2), past it (768, 1)."""
    ceil = lambda a, b: -(-a // b)  # noqa: E731
    threads = ceil(N, 8)
    cluster = 8
    while threads % cluster and cluster > 1:
        cluster //= 2
    most = cluster
    block = ceil(threads, cluster)
    while block < 128 and cluster >= 2:
        block, cluster = block * 2, cluster // 2
    while M * cluster > sm_count and cluster > 1 and block <= 512:
        block, cluster = block * 2, cluster // 2
    while cluster < most:
        cand = cluster * 2
        cand_block = ceil(threads, cand)
        if cand_block < 64 or M * cand > sm_count:
            break
        cluster, block = cand, cand_block
    if block > 1024:
        raise ValueError(f"N={N}: past one load per thread, which the fused Definition does not state")
    return block, cluster


def _fused_geometry(N: int, cluster: int) -> tuple[int, int]:
    """(BLOCK, warps per CTA) for N elements over CLUSTER CTAs of 8-element threads, every warp full."""
    if cluster not in (1, 2, 4, 8):
        raise ValueError(f"CLUSTER={cluster}: a cluster of 1, 2, 4 or 8 CTAs")
    if N % (8 * cluster):
        raise ValueError(f"N={N}: the CTAs of a {cluster}-CTA cluster each hold whole 8-element threads")
    block = N // (8 * cluster)
    if block % WARP or block > 1024:
        raise ValueError(f"N={N}, CLUSTER={cluster}: {block} threads per CTA; the Definition states whole warps, at most 1024")
    return block, block // WARP


def custom_stage(world: int, nbytes: int, fully_connected: bool = True) -> int:
    """vLLM's custom all-reduce kernel for `nbytes` over `world` ranks (``VLLM_CUSTOM_ALLREDUCE_ALGO`` unset): 1 or 2."""
    if world == 2:
        return 1
    if not fully_connected:
        raise ValueError("custom all-reduce past two ranks needs a fully connected topology")
    return 1 if (world <= 4 and nbytes < 512 * 1024) or (world <= 8 and nbytes < 256 * 1024) else 2


def custom_order(world: int, start: int) -> tuple[int, ...]:
    """The ranks `packed_reduce` reads, from `start`: start, start + 1, ... mod WORLD."""
    return tuple((start + i) % world for i in range(world))


def custom_owners(world: int, size: int, stage: int) -> list[tuple[int, int, int]]:
    """[(lo, hi, owner)]: elements lo..hi of a SIZE-element all-reduce are summed from rank `owner` (`custom_order`). The 1-stage
    kernel is one run from rank 0; the 2-stage kernel's rank r owns packs r * part .. (r + 1) * part of the SIZE / 8 packs,
    part = packs // WORLD, the last rank to the end."""
    if size % 8:
        raise ValueError(f"SIZE={size}: vLLM's custom all-reduce takes whole 16-byte packs of bf16")
    if stage == 1:
        return [(0, size, 0)]
    if stage != 2:
        raise ValueError(f"STAGE={stage}: 1 or 2")
    packs = size // 8
    part = packs // world
    runs = [(8 * r * part, 8 * (r + 1) * part, r) for r in range(world - 1) if part]
    runs.append((8 * (world - 1) * part, size, world - 1))
    return runs


def tree_leaves(t) -> tuple[int, ...]:
    return (t,) if isinstance(t, int) else tuple(r for c in t for r in tree_leaves(c))


def multimem_tree(t, world: int):
    """`t` as a reduction tree over ranks 0..WORLD-1 in canonical form: a rank, or a pair of trees (one f32 add), every rank once,
    each pair's children ordered by their least rank (an IEEE add commutes)."""
    def canon(x):
        if isinstance(x, bool) or not isinstance(x, (int, tuple, list)):
            raise ValueError(f"TREE {t!r}: a node is a rank or a pair of nodes, not {x!r}")
        if isinstance(x, int):
            return x
        if len(x) != 2:
            raise ValueError(f"TREE {t!r}: the node {x!r} adds {len(x)} subtrees; each add takes two")
        a, b = canon(x[0]), canon(x[1])
        return (a, b) if min(tree_leaves(a)) < min(tree_leaves(b)) else (b, a)

    c = canon(t)
    if sorted(tree_leaves(c)) != list(range(world)):
        raise ValueError(f"TREE {t!r}: its leaves must be the ranks 0..{world - 1}, each once")
    return c


def multimem_trees(trees, world: int) -> tuple:
    """One canonical tree per owner rank."""
    trees = tuple(trees)
    if len(trees) != world:
        raise ValueError(f"TREES: one tree per owner rank, {world}, not {len(trees)}")
    return tuple(multimem_tree(t, world) for t in trees)


def chain(world: int, order=None):
    """The tree adding the ranks of `order` (default 0, 1, ..., WORLD-1) one at a time onto the first."""
    order = tuple(range(world)) if order is None else tuple(order)
    t = order[0]
    for r in order[1:]:
        t = (t, r)
    return multimem_tree(t, world)


def tree_clusters(t) -> set[frozenset[int]]:
    """The ranks under each add of `t`."""
    out: set[frozenset[int]] = set()

    def walk(x) -> frozenset[int]:
        if isinstance(x, int):
            return frozenset((x,))
        s = walk(x[0]) | walk(x[1])
        out.add(s)
        return s

    walk(t)
    return out


def tree_from_clusters(clusters, world: int):
    """The binary tree whose adds hold exactly `clusters` (sets of ranks), or None when none does: WORLD - 1 sets that nest (any
    two disjoint or one inside the other), the largest every rank, each split by its largest proper subsets and its loose ranks
    into exactly two parts."""
    cs = {frozenset(c) for c in clusters}
    full = frozenset(range(world))
    if len(cs) != world - 1 or full not in cs or any(len(c) < 2 or not c <= full for c in cs):
        return None
    if any(a & b and not (a <= b or b <= a) for a in cs for b in cs):
        return None

    def build(s: frozenset[int]):
        if len(s) == 1:
            return next(iter(s))
        inner = [c for c in cs if c < s]
        top = [c for c in inner if not any(c < d for d in inner)]
        parts = top + [frozenset((r,)) for r in sorted(s - frozenset().union(*top))]
        if len(parts) != 2:
            return None
        a, b = build(parts[0]), build(parts[1])
        return None if a is None or b is None else (a, b)

    t = build(full)
    return None if t is None else multimem_tree(t, world)


def multimem_owners(world: int, size: int) -> list[tuple[int, int, int]]:
    """[(lo, hi, owner)]: torch's `multimem_all_reduce_` reduces elements lo..hi of a SIZE-element bf16 all-reduce from rank
    `owner`'s ``multimem.ld_reduce``, each rank ``round_up(SIZE, 16 * WORLD) / WORLD`` elements (the kernel's 16-byte alignment
    taken as elements), empty slices dropped."""
    if size % 8:
        raise ValueError(f"SIZE={size}: torch's multimem all-reduce at 16-byte alignment takes whole 8-element vectors of bf16")
    per = -(-size // (MULTIMEM_ALIGN_BYTES * world)) * MULTIMEM_ALIGN_BYTES
    return [(r * per, min((r + 1) * per, size), r) for r in range(world) if r * per < size]


def multimem_runs(world: int, size: int, trees) -> list[tuple[int, int, object]]:
    """[(lo, hi, tree)]: `multimem_owners`' slices with each owner's tree, adjacent slices of one tree merged."""
    trees = multimem_trees(trees, world)
    runs: list[tuple[int, int, object]] = []
    for lo, hi, owner in multimem_owners(world, size):
        if runs and runs[-1][2] == trees[owner]:
            runs[-1] = (runs[-1][0], hi, trees[owner])
        else:
            runs.append((lo, hi, trees[owner]))
    return runs


# ---------------------------------------------------------------------------------------------------------------------------------
# FlashInfer MNNVL
# ---------------------------------------------------------------------------------------------------------------------------------


def _mnnvl_sum(B, S, col, zero, neg0):
    acc = zero
    for r in range(ranks(S.WORLD)):
        xf = B.call(P.Bf16ToF32, col[r])
        acc = B.call(P.F32AddFtzV2, acc, B.call(P.SelectF32, B.call(P.I32Eq, xf, neg0), zero, xf))
    if strategy(S.SHOT) == 2:
        acc = B.call(P.F32AddFtzV2, zero, acc)
    return B.call(P.F2fpBf16, acc)


@composite("AllReduceMnnvlCoordinate", 1, ["WORLD", "SHOT"], lambda S: ((("col", Array(int(S.WORLD), BF16)),), BF16),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200 (tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMnnvlCoordinate(B, S, col):
    """One element of FlashInfer's MNNVL all-reduce: ``bf16(+0 + s(col[0]) + ... + s(col[WORLD-1]))`` in FTZ f32, s writing -0 as
    +0; SHOT 2 adds the sum to +0 once more."""
    return _mnnvl_sum(B, S, col, B.call(ZERO32), B.call(_NEG_ZERO32))


@composite("AllReduceMnnvl", 1, ["WORLD", "N", "SHOT"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200 (tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMnnvl(B, S, parts):
    """FlashInfer's MNNVL all-reduce of WORLD ranks' N-vectors (`trtllm_allreduce_fusion` without the norm): one
    `AllReduceMnnvlCoordinate_v1` per element."""
    return B.batch(bind(AllReduceMnnvlCoordinate, WORLD=S.WORLD, SHOT=S.SHOT), parts, axes=(1,))


def _butterfly0(B, add, vals: list):
    """Lane 0 of the xor butterfly ``v += shfl_xor(v, m)``, m = 16, 8, 4, 2, 1, over 32 lanes' values."""
    v = list(vals)
    m = WARP // 2
    while m:
        v = [B.call(add, v[j], v[j + m]) for j in range(m)]
        m //= 2
    return v[0]


@composite("AllReduceRmsNormMnnvl", 1, ["WORLD", "N", "SHOT", "CLUSTER", "EPS", "FMA"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
                      Tuple(Array(S.N, BF16), Array(S.N, BF16))),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200 (tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceRmsNormMnnvl(B, S, parts, res, w):
    """FlashInfer's MNNVL all-reduce + residual add + RMSNorm (`kARResidualRMSNorm`, fp32 accumulation, weight bias 0) of one
    token: `AllReduceMnnvl_v1`, r = a + res, the sum of squares in `blockReduceSum`'s geometry over CLUSTER CTAs,
    ``rs = rsqrt(sum * rcp(N) + EPS)`` (FMA: one FFMA), and the normed row. Returns (normed, r)."""
    N, shot, cluster = int(S.N), strategy(S.SHOT), int(S.CLUSTER)
    block, warps = _fused_geometry(N, cluster)
    zero = B.call(ZERO32)
    a = B.call(bind(AllReduceMnnvl, WORLD=S.WORLD, N=N, SHOT=shot), parts)
    r = B.batch(P.Bf16AddF2fp, a, res, axes=(0, 0))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    if shot == 1:
        sq = B.batch(P.Bf16ToF32, B.batch(P.Bf16MulBf16, r, r, axes=(0, 0)), axes=(0,))
        step = lambda acc, j: B.call(P.F32AddFtzV2, acc, sq[j])  # noqa: E731
    else:
        step = lambda acc, j: B.call(P.F32FmaFtzV2, rf[j], rf[j], acc)  # noqa: E731
    sums = []
    for b in range(cluster):
        warp_sums = []
        for k in range(warps):
            lanes = []
            for lane in range(WARP):
                j0 = b * (N // cluster) + 8 * (k * WARP + lane)
                acc = zero
                for i in range(8):
                    acc = step(acc, j0 + i)
                lanes.append(acc)
            warp_sums.append(_butterfly0(B, P.F32AddFtzV2, lanes))
        sums.append(_butterfly0(B, P.F32AddFtzV2, warp_sums + [zero] * (WARP - warps)))
    total = sums[0]
    if cluster > 1:
        total = zero
        for s in sums:
            total = B.call(P.F32AddFtzV2, total, s)
    rcp = B.call(P.MufuRcpFtz, B.call(const(32, P.bits_of(np.float32(N)))))
    eps = B.call(const(32, eps_bits(S.EPS)))
    if S.FMA:
        arg = B.call(P.F32FmaFtzV2, total, rcp, eps)
    else:
        arg = B.call(P.F32AddFtzV2, B.call(P.F32MulFtzV2, total, rcp), eps)
    rs = B.call(P.RsqrtApprox, arg)
    g = B.batch(P.F32AddFtzV2, zero, B.batch(P.Bf16ToF32, w, axes=(0,)), axes=(None, 0))
    if shot == 1:
        y = B.batch(P.F32MulFtzV2, B.batch(P.F32MulFtzV2, rf, rs, axes=(0, None)), g, axes=(0, 0))
    else:
        y = B.batch(P.F32MulFtzV2, B.batch(P.F32MulFtzV2, g, rf, axes=(0, 0)), rs, axes=(0, None))
    return tuple_of(B.batch(P.F2fpBf16, y, axes=(0,)), r)


# ---------------------------------------------------------------------------------------------------------------------------------
# vLLM custom all-reduce
# ---------------------------------------------------------------------------------------------------------------------------------


@composite("AllReduceCustomCoordinate", 1, ["WORLD", "START"], lambda S: ((("col", Array(int(S.WORLD), BF16)),), BF16),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200 under VLLM_ALLREDUCE_USE_SYMM_MEM=0 "
                       "(tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceCustomCoordinate(B, S, col):
    """One element of vLLM's custom all-reduce (`packed_reduce`): ``bf16(f32(col[o0]) + f32(col[o1]) + ...)``, IEEE f32 in
    `custom_order(WORLD, START)`."""
    order = custom_order(ranks(S.WORLD), int(S.START))
    acc = B.call(P.Bf16ToF32, col[order[0]])
    for r in order[1:]:
        acc = B.call(P.F32AddV2, acc, B.call(P.Bf16ToF32, col[r]))
    return B.call(P.F2fpBf16, acc)


@composite("AllReduceCustom", 1, ["WORLD", "SIZE", "STAGE"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200 under VLLM_ALLREDUCE_USE_SYMM_MEM=0 "
                       "(tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceCustom(B, S, parts):
    """vLLM's custom all-reduce of WORLD ranks' SIZE bf16 (the whole flattened tensor, which the 2-stage kernel's ownership
    reads): each run of `custom_owners` one `AllReduceCustomCoordinate_v1{START=owner}` per element."""
    W = ranks(S.WORLD)
    outs = []
    for lo, hi, owner in custom_owners(W, int(S.SIZE), int(S.STAGE)):
        cols = array_of([parts[r][lo:hi] for r in range(W)])
        outs.append(B.batch(bind(AllReduceCustomCoordinate, WORLD=W, START=owner), cols, axes=(1,)))
    return outs[0] if len(outs) == 1 else coll_concat(Array(int(S.SIZE), BF16), outs)


# ---------------------------------------------------------------------------------------------------------------------------------
# torch symmetric memory: NVSwitch multimem
# ---------------------------------------------------------------------------------------------------------------------------------


@composite("AllReduceMultimemCoordinate", 1, ["WORLD", "TREE", "FTZ"], lambda S: ((("col", Array(int(S.WORLD), BF16)),), BF16),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200, the tree read off the switch "
                       "(tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMultimemCoordinate(B, S, col):
    """One element of torch's multimem all-reduce: the f32 adds of TREE over ``f32(col[r])`` (FTZ: FADD.FTZ, else IEEE), then
    RN to bf16."""
    add = P.F32AddFtzV2 if S.FTZ else P.F32AddV2

    def value(t):
        if isinstance(t, int):
            return B.call(P.Bf16ToF32, col[t])
        return B.call(add, value(t[0]), value(t[1]))

    return B.call(P.F2fpBf16, value(multimem_tree(S.TREE, ranks(S.WORLD))))


@composite("AllReduceMultimem", 1, ["WORLD", "SIZE", "TREES", "FTZ"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)),
           conformance="not-yet-established: GLM-5.3 TP8 on 8x H200, the trees read off the switch "
                       "(tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMultimem(B, S, parts):
    """torch's `multimem_all_reduce_` of WORLD ranks' SIZE bf16 (the whole flattened tensor, whose slices name each element's
    owner): each run of `multimem_runs` one `AllReduceMultimemCoordinate_v1{TREE=TREES[owner]}` per element."""
    W = ranks(S.WORLD)
    outs = []
    for lo, hi, tree in multimem_runs(W, int(S.SIZE), S.TREES):
        cols = array_of([parts[r][lo:hi] for r in range(W)])
        outs.append(B.batch(bind(AllReduceMultimemCoordinate, WORLD=W, TREE=tree, FTZ=bool(S.FTZ)), cols, axes=(1,)))
    return outs[0] if len(outs) == 1 else coll_concat(Array(int(S.SIZE), BF16), outs)
