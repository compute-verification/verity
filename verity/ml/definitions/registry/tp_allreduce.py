"""vLLM's TP all-reduces as word Definitions: GLM-5.3's TP8 ones on 8x H200 (vLLM d9105ea80, FlashInfer 0.6.18), and NCCL's tree
all-reduce vLLM's batch-invariant mode pins pynccl to, on 4x RTX PRO 6000 with P2P off.

    AllReduceMnnvlCoordinate_v1{WORLD,SHOT}   (col: Array<WORLD, bf16>) -> bf16
    AllReduceMnnvl_v1{WORLD,N,SHOT}           (parts: Array<WORLD, Array<N, bf16>>) -> Array<N, bf16>
    AllReduceRmsNormMnnvl_v1{WORLD,N,SHOT,CLUSTER,EPS,FMA}
                                              (parts, res: Array<N, bf16>, w: Array<N, bf16>) -> (normed, residual')
    AllReduceResidualMnnvl_v1{WORLD,N,SHOT}   (parts, res) -> residual'   (the fused one's all-reduce and residual add)
    RmsNormMnnvl_v1{N,SHOT,CLUSTER,EPS,FMA}   (r: Array<N, bf16>, w) -> normed   (the fused one's norm of residual' r)
    AllReduceCustomCoordinate_v1{WORLD,START} (col: Array<WORLD, bf16>) -> bf16
    AllReduceCustom_v1{WORLD,SIZE,STAGE}      (parts: Array<WORLD, Array<SIZE, bf16>>) -> Array<SIZE, bf16>
    AllReduceMultimemCoordinate_v1{WORLD,TREE,FTZ}
                                              (col: Array<WORLD, bf16>) -> bf16
    AllReduceMultimemVector_v1{WORLD,TREE,FTZ}
                                              (parts: Array<WORLD, Array<8, bf16>>) -> Array<8, bf16>
    AllReduceMultimem_v1{WORLD,SIZE,TREES,FTZ}
                                              (parts: Array<WORLD, Array<SIZE, bf16>>) -> Array<SIZE, bf16>
    AllReduceNcclTreeCoordinate_v1{WORLD,CHAIN}
                                              (col: Array<WORLD, bf16>) -> bf16
    AllReduceNcclTree_v1{WORLD,SIZE,CHAIN}    (parts: Array<WORLD, Array<SIZE, bf16>>) -> Array<SIZE, bf16>

Which kernel reduces a call (vLLM d9105ea80):

* `fused_allreduce_rms_norm` (`vllm/models/common/ops/fused_allreduce_rms_norm.py`, every o_proj / MoE output of DeepSeek-V3.2's
  decoder layer) runs FlashInfer's `trtllm_allreduce_fusion` when `_can_use_flashinfer` admits the call: M <= 0.5 MiB / (H * 2 B),
  42 tokens at H 6144 and TP 8, and the workspace's buffer holds it. The backend is `_resolve_fi_ar_backend`'s: "mnnvl" on one
  node with NVSwitch multicast (`VLLM_FLASHINFER_ALLREDUCE_BACKEND` pins it), whose kernels are compiled with ``-use_fast_math``.
  `mnnvl_shot`: one-shot (`oneshotAllreduceFusionKernel`) when M * H * WORLD * 2 B <= 1 MiB (M <= 10 at H 6144, TP 8), else
  two-shot (`twoshotAllreduceKernel` then `rmsNormLamport`). `AllReduceRmsNormMnnvl_v1` is either, one token.
* Otherwise the call is `tensor_model_parallel_all_reduce` then `RMSNorm.forward_cuda` (`fused_add_rms_norm`,
  `b1.RMSNormFusedCuda_v2`). `_can_use_flashinfer` reads none of the settings below, so this split is the same under each.
* `tensor_model_parallel_all_reduce` (`CudaCommunicator.all_reduce`, `cuda_communicator.py:305-377`) tries, of the routes on at
  TP 8 on cc 9.0, in order, FlashInfer's MNNVL all-reduce (``fi_ar_comm``) within the same 0.5 MiB; vLLM's custom all-reduce
  (``ca_comm``, `AllReduceCustom_v1`) when it is on, below its ``max_size``; torch's symmetric memory (``symm_mem_comm``,
  `AllReduceMultimem_v1`) to 64 MiB when it is on; then NCCL. The custom all-reduce's ``max_size`` is 8 MiB (`CUSTOM_MAX_BYTES`),
  cut to `CUSTOM_SYMM_MAX_BYTES` (256 KiB at 8 ranks on cc 9.0, `custom_all_reduce.py:206-220`) when symmetric memory is on, which
  FlashInfer's 0.5 MiB holds: at vLLM's defaults (`SETTINGS`) it takes none of these all-reduces, and runs only with
  ``VLLM_ALLREDUCE_USE_SYMM_MEM=0`` (the "switch"), up to 8 MiB (`custom_stage` picks its kernel, ``VLLM_CUSTOM_ALLREDUCE_ALGO``
  pins it).

What the 8-GPU capture (`tests/properties/glm53_tp8_allreduce_capture_gpu.py`) has shown, per setting, hidden size and route, is
`CAPTURED`; `covered` answers for one call, and a route it does not name is not covered, whatever the dispatch says. The third
GO run served vLLM's defaults and also captured DeepSeek-V4-Pro's 7168-wide all-reduces.

The NVSwitch's reduction trees are the node's, not the H200's: two 8x H200 nodes have shown different ones.  So symmetric
memory's routes, and NCCL's past them where NCCL runs NVLS (one in-switch f32 sum on one tree, `nvls_tree`), come from node
records (`NODE_FILE`, `check_node_record`: the device, versions, each owner's tree and FTZ, NCCL's tuning and runs, the routes
and the runs they rest on), each written by the capture's ``--register`` on one node.  `covered` takes a node's routes only at
its device and the TREES and FTZ a registration states, `UNSTATED_NODE` when it states none; `node_refusal` names a refusal.

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
64 MiB) with vLLM's custom all-reduce on (vLLM's default) or off, its cap with symmetric memory on being below FlashInfer's. At
4, 6 or 8 ranks on cc 9.0 that is `torch.ops.symm_mem.multimem_all_reduce_` (torch 2.13 `CUDASymmetricMemoryOps.cu`): rank r
issues ``multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2`` over its slice of ``round_up(SIZE, 16 * WORLD) / WORLD``
elements (`multimem_owners`) and ``multimem.st``s the result to every rank. The NVSwitch reduces the WORLD payloads in f32 in an
order no source states; `AllReduceMultimem_v1` takes it per owner as TREES, binary trees over the ranks whose nodes are f32 adds
(IEEE, or FTZ when FTZ), the 8-GPU capture's probes reading each tree off the switch (`tree_from_clusters`), then RN to bf16. Each
``ld_reduce`` reduces one 16-byte vector, `MULTIMEM_VECTOR` elements, and an owner's slice is whole vectors, so the Definition is a
batch of `AllReduceMultimemVector_v1` per run of one tree, no vector reading another's words. The capture's kept words replay on
this form bit for bit (`MULTIMEM_REPLAY_RUN`, tests/properties/glm53_multimem_replay.py: 20 all-reduces, 17,063,936 elements).

Under vLLM's batch-invariant mode (``VLLM_BATCH_INVARIANT=1``) none of those routes is on: `ParallelConfig` disables the custom
all-reduce, `override_envs_for_invariance` sets ``VLLM_ALLREDUCE_USE_SYMM_MEM=0`` and `CudaCommunicator` leaves FlashInfer's
all-reduce off, so every call is pynccl's ``ncclAllReduce`` (`BATCH_INVARIANT`, route ``pynccl_comm``), and the same function pins
NCCL to one channel (``NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS = 1``) of ``NCCL_ALGO=allreduce:tree``, ``NCCL_PROTO=Simple``. Inside
one node NCCL's tree is a chain (NCCL 2.29.7, torch 2.13's, `graph/connect.cc` `ncclTopoPreset`: a rank's parent is the one before
it in the channel's intra-node order, the first is the root), and with ``NCCL_P2P_DISABLE=1`` the ranks talk over NCCL's SHM
transport, whose intra-node search tries the GPUs in PCI order first (`graph/search.cc`, ``FORCED_ORDER_PCI``): CHAIN, root first.
The reduce runs up the chain from the leaf (`all_reduce.h` `runTreeSplit`: a rank adds its own input to what its child sent) and the
root broadcasts its sum down, so every rank receives ``x[c0] + (x[c1] + (... + x[c_last]))``, each add `FuncSum<__nv_bfloat16>`'s
``__hadd2`` (`reduce_kernel.h`, sm_80 and later): one bf16 rounding of the exact sum, which is `Bf16AddF2fp_v1` (the f32 sum of two
bf16 words rounded again to bf16 rounds once, 24 >= 2 * 8 + 2), NaN 0x7FFF. Neither the chunking nor the call's size changes an
element's adds.

f32 NaN words are the GPU's (`F32AddFtz_v2`, `F32MulFtz_v2`, `F32FmaFtz_v2`, `F32Add_v2`); every output is F2FP's bf16, whose NaN
is 0x7FFF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.circuits.refs import array_of, coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16

__all__ = ["AllReduceCustom", "AllReduceCustomCoordinate", "AllReduceMnnvl", "AllReduceMnnvlCoordinate", "AllReduceMultimem",
           "AllReduceMultimemCoordinate", "AllReduceMultimemVector", "AllReduceNcclTree", "AllReduceNcclTreeCoordinate",
           "AllReduceResidualMnnvl", "AllReduceRmsNormMnnvl", "BATCH_INVARIANT", "CAPTURED", "CAPTURE_RUNS", "CUSTOM_MAX_BYTES",
           "CUSTOM_SYMM_MAX_BYTES", "Captured", "FI_MAX_BYTES", "MNNVL_ONE_SHOT_BYTES", "MULTIMEM_REPLAY_RUN", "MULTIMEM_VECTOR",
           "NCCL_TREE_RUNS", "NODE_FILE", "NODE_RECORDS", "NODE_SCHEMA", "RTX_PRO_6000", "RmsNormMnnvl", "SETTINGS",
           "SYMM_MEM_MAX_BYTES", "TREE_UNRECORDED", "TREE_UNSTATED", "UNSTATED_NODE", "WARP", "chain", "check_node_record", "covered",
           "custom_order", "custom_owners", "custom_stage", "device_of", "eps_bits", "fused_max_tokens", "load_node_records",
           "mnnvl_grid", "mnnvl_shot", "multimem_owners", "multimem_runs", "multimem_tree", "multimem_trees", "nccl_chain",
           "node_key", "node_refusal", "nvls_tree", "offset_free", "ranks", "route_binding", "served_route", "settings_of", "strategy",
           "tree_clusters", "tree_from_clusters", "tree_leaves"]

WARP = 32
#: FlashInfer `MNNVL_ONE_SHOT_THRESHOLD`: the one-shot kernel up to this many bytes of M * H * WORLD
MNNVL_ONE_SHOT_BYTES = 64 * 1024 * 8 * 2
#: vLLM `_max_token_num`'s budget: `fused_allreduce_rms_norm` admits M * H * 2 B up to it
FI_MAX_BYTES = 512 * 1024
#: vLLM's custom all-reduce buffer (`CustomAllreduce.max_size`): it takes fewer bytes than this
CUSTOM_MAX_BYTES = 8 * 1024 * 1024
#: vLLM `CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"][8]`: the custom all-reduce's ``max_size`` when torch's symmetric memory is on
CUSTOM_SYMM_MAX_BYTES = 256 * 1024
#: vLLM `SYMM_MEM_ALL_REDUCE_MAX_SIZES["9.0"][8]`: torch's symmetric memory takes a bf16 all-reduce up to this many bytes
SYMM_MEM_MAX_BYTES = 64 * 1024 * 1024
#: torch's `multimem_all_reduce_` slices are whole 16-byte vectors per rank
MULTIMEM_ALIGN_BYTES = 16
#: the bf16 elements of one 16-byte vector, what one ``multimem.ld_reduce.v4.bf16x2`` reduces
MULTIMEM_VECTOR = MULTIMEM_ALIGN_BYTES // 2
#: the record of the run that replayed `CAPTURE_RUNS`' kept multimem words on `AllReduceMultimemVector_v1`'s batch
#: (tests/properties/glm53_multimem_replay.py)
MULTIMEM_REPLAY_RUN = EVIDENCE["glm53_multimem_replay"]["run_record"]
H200_SMS = 132
_NEG_ZERO32 = const(32, 0x80000000)

#: vLLM's all-reduce settings: name -> (custom all-reduce on, torch symmetric memory on), FlashInfer's all-reduce on in each.
#: "defaults" are vLLM's (``disable_custom_all_reduce`` false, ``VLLM_ALLREDUCE_USE_SYMM_MEM`` and ``VLLM_ALLREDUCE_USE_FLASHINFER``
#: 1); "custom-off" is ``disable_custom_all_reduce`` true, what the two GO runs served; "switch" is ``VLLM_ALLREDUCE_USE_SYMM_MEM=0``.
SETTINGS = {"defaults": (True, True), "custom-off": (False, True), "switch": (True, False)}
#: the records of the 8-GPU capture's GO runs on an 8-GPU H200 node (the first with NCCL_P2P_DISABLE=1, which only NCCL's routes
#: read)
CAPTURE_RUNS = tuple(EVIDENCE["glm53_tp8_allreduce_capture"]["run_records"])
_CUSTOM_OFF_RUNS = CAPTURE_RUNS[:2]
_DEFAULT_RUNS = CAPTURE_RUNS[2:]
#: the tree every owner's multimem reduction took on the node of `CAPTURE_RUNS` (its record in `NODE_RECORDS`, including the
#: 7168-wide pass): another node's may differ, so it is a fixed tree for bindings and tests, not the H200's
GLM53_H200_TREE = ((((((0, (1, 2)), 3), 5), 6), 4), 7)
#: vLLM's batch-invariant mode (``VLLM_BATCH_INVARIANT=1``), the setting under which every TP all-reduce is pynccl's
BATCH_INVARIANT = "batch-invariant"
#: (compute capability, SMs) of the RTX PRO 6000 Blackwell Server Edition
RTX_PRO_6000 = ("12.0", 188)
#: the records of three runs on an RTX PRO 6000 node: Llama-3.1-70B int8's TP4 capture (GPUs 2, 3, 6, 7, NCCL 2.29.7), the replay
#: of its every all-reduce through `AllReduceNcclTree_v1` (`tests/properties/tp4_nccl_tree_replay.py replay`: 5152 calls,
#: 1,514,110,976 rank words, GO) and NCCL's tree probed on GPUs 0, 1, 2, 5 (``probe``: the same chain, NaNs 0x7FFF)
NCCL_TREE_RUNS = tuple(EVIDENCE["tp4_nccl_tree_replay"]["run_records"])


class Captured(NamedTuple):
    """A route the capture showed: at `setting`, hidden size N, the calls of lo..hi tokens on `route` (``fused``:
    `fused_allreduce_rms_norm`'s FlashInfer call; else the communicator `CudaCommunicator.all_reduce` hands it to) equal
    `definition` at `statics` on every word, in the runs whose records are `runs`, over `world` ranks of `device` ((compute
    capability, SMs)) with NCCL's `transport` between them (``"shm"``: ``NCCL_P2P_DISABLE=1``), on the node whose multimem
    trees and FTZ are `node` (`node_key`'s last two). A device, transport or node of None is one the entry is not keyed on (the
    H200 entries', routes NCCL does not run, keyed on their setting at world 8; every route but multimem's and NVLS's)."""
    setting: str
    N: int
    route: str
    lo: int
    hi: int
    definition: str
    statics: dict
    runs: tuple[str, ...]
    world: int = 8
    device: tuple[str, int] | None = None
    transport: str | None = None
    node: tuple | None = None


#: The covered routes no node's tree keys, lo and hi the fewest and most tokens the capture launched on it (its `FUSED_M`, `AR_M`,
#: `DSV4_AR_M` and `SWITCH_M`, with counts between; the TP4 capture launched 256 and 1, and NCCL's tree reads no size).
#: `CAPTURED` adds each node record's routes.
_BASE_CAPTURED = (
    Captured("custom-off", 6144, "fused", 1, 42, "AllReduceRmsNormMnnvl_v1", {"FMA": True}, _CUSTOM_OFF_RUNS),
    Captured("custom-off", 6144, "fi_ar_comm", 1, 42, "AllReduceMnnvl_v1", {}, _CUSTOM_OFF_RUNS),
    Captured("defaults", 6144, "fused", 1, 42, "AllReduceRmsNormMnnvl_v1", {"FMA": True}, _DEFAULT_RUNS),
    Captured("defaults", 6144, "fi_ar_comm", 1, 42, "AllReduceMnnvl_v1", {}, _DEFAULT_RUNS),
    Captured("defaults", 7168, "fi_ar_comm", 1, 36, "AllReduceMnnvl_v1", {}, _DEFAULT_RUNS),
    Captured("switch", 6144, "ca_comm", 43, 682, "AllReduceCustom_v1", {"STAGE": 2}, CAPTURE_RUNS),
    Captured(BATCH_INVARIANT, 8192, "pynccl_comm", 1, 256, "AllReduceNcclTree_v1", {"CHAIN": (0, 1, 2, 3)}, NCCL_TREE_RUNS,
             world=4, device=RTX_PRO_6000, transport="shm"),
)


def device_of(compute_capability, sms) -> tuple[str, int]:
    """(``"major.minor"``, SMs) of a device: its capability as ``"12.0"`` or ``(12, 0)``."""
    cc = compute_capability if isinstance(compute_capability, str) else ".".join(str(int(x)) for x in compute_capability)
    return str(cc), int(sms)


def covered(setting: str, N: int, route: str, M: int, *, world: int = 8, device=None, transport: str | None = None,
            trees=None, ftz: bool | None = None) -> Captured | None:
    """The `CAPTURED` entry covering a call of M tokens of N bf16 on `route` at `setting` over `world` ranks of `device`
    (`device_of`) with NCCL's `transport` (`settings_of`'s ``nccl_transport``), on the node whose multimem `trees` (one per
    owner) and `ftz` the registration states, or None: the call is not covered (`node_refusal` names why a node's route is not).
    A device or transport the caller does not know (None) matches only an entry not keyed on it; a node record's entries match
    only at its device and the TREES and FTZ stated, or `UNSTATED_NODE` when neither is (its device when the caller names none)."""
    if setting not in SETTINGS and setting != BATCH_INVARIANT:
        raise ValueError(f"setting {setting!r}: one of {sorted(SETTINGS) + [BATCH_INVARIANT]}")
    dev = None if device is None else device_of(*device)
    node, node_dev = _stated_node(trees, ftz, int(world)), dev
    if trees is None and ftz is None and UNSTATED_NODE is not None:
        node, node_dev = UNSTATED_NODE[1:], dev or UNSTATED_NODE[0]
    return next((c for c in CAPTURED if (c.setting, c.N, c.route, c.world) == (setting, N, route, int(world)) and c.lo <= M <= c.hi
                 and c.transport in (None, transport)
                 and (c.device in (None, dev) if c.node is None else (c.device, c.node) == (node_dev, node))), None)


def _stated_node(trees, ftz, world: int) -> tuple | None:
    """A registration's stated (TREES, FTZ) in canonical form, or None when it states neither or a tree that is not one per owner
    of `world` ranks."""
    if trees is None or ftz is None:
        return None
    try:
        return multimem_trees(trees, world), bool(ftz)
    except ValueError:
        return None


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
    if size % MULTIMEM_VECTOR:
        raise ValueError(f"SIZE={size}: torch's multimem all-reduce at 16-byte alignment takes whole {MULTIMEM_VECTOR}-element "
                         "vectors of bf16")
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
# Node records
# ---------------------------------------------------------------------------------------------------------------------------------

NODE_SCHEMA = "tp-allreduce-node/1"
#: every node record (`check_node_record`), one per node the 8-GPU capture's ``--register`` (or a whole capture) ran on
NODE_FILE = Path(__file__).with_name("tp_allreduce_nodes.json")
#: the models an NCCL probe fits: bf16 at every hop, or one in-switch f32 sum rounded once, IEEE or FTZ
NODE_NCCL_FITS = ("bf16", "f32", "f32_ftz")
#: the routes a node record keys: torch's symmetric memory, and NCCL's past it when its probe fits one in-switch f32 sum (NVLS)
NODE_ROUTES = ("symm_mem_comm", "pynccl_comm")
#: `node_refusal`'s names: the registration states no multimem tree and FTZ; no node record holds the ones it states
TREE_UNSTATED, TREE_UNRECORDED = "multimem_tree_unstated", "multimem_tree_unrecorded"
_NODE_KEYS = ("schema", "device", "world", "versions", "multimem", "nccl", "routes", "source")


def _fields(where: str, d, keys) -> dict:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise ValueError(f"node record: {where} has {got}, not the keys {sorted(keys)}")
    return d


def _int(where: str, v, lo: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"node record: {where} is {v!r}, not an integer >= {lo}")
    return v


def _text(where: str, v, *, none: bool = False):
    if not (isinstance(v, str) and v) and not (none and v is None):
        raise ValueError(f"node record: {where} is {v!r}, not a string{' or null' if none else ''}")
    return v


def _list(where: str, v, *, empty: bool = False) -> list:
    if not isinstance(v, list) or (not v and not empty):
        raise ValueError(f"node record: {where} is {v!r}, not a {'' if empty else 'nonempty '}list")
    return v


def _trees(where: str, trees, world: int) -> tuple:
    try:
        return tuple(multimem_tree(t, world) for t in _list(where, trees))
    except ValueError as e:
        raise ValueError(f"node record: {where}: {e}") from None


def nvls_tree(nccl: dict | None, world: int):
    """The one tree a node record's NCCL all-reduces (its ``nccl``) take at every probed size over `world` ranks, when its fit is
    one in-switch f32 sum (NVLS, `AllReduceMultimem_v1` at that tree on every owner), else ValueError."""
    if nccl is None or nccl["fit"] not in ("f32", "f32_ftz"):
        raise ValueError("node record: a pynccl_comm route needs nccl.fit f32 or f32_ftz (one in-switch f32 sum)")
    trees = {multimem_tree(s["trees"][s["runs"][0][2]], world) for s in nccl["sizes"] if len(s["runs"]) == 1}
    if len(trees) != 1 or any(len(s["runs"]) != 1 for s in nccl["sizes"]):
        raise ValueError("node record: a pynccl_comm route needs NCCL's every probed size one run of one tree, the same tree")
    return trees.pop()


def check_node_record(rec, *, cited: bool = True) -> dict:
    """`rec` when it is a node record (`NODE_SCHEMA`), else ValueError naming the field. ``device`` {name, compute_capability
    "major.minor", sms}, ``world``, ``versions`` {driver, nccl, torch, vllm, flashinfer: each a string, or null where the run did
    not log it}; ``multimem`` {trees: one per owner rank as nested [a, b] lists, ftz, probes: the [N, M] all-reduces the probes
    read them off}; ``nccl``: null, or NCCL's all-reduces past symmetric memory {fit (`NODE_NCCL_FITS`), sizes: per probed [N, M]
    its bytes, NCCL's tuning (algo, proto, channels) and the runs [lo, hi, t] of the decoded ``trees`` over every element};
    ``routes``: per (setting, N, route) the lo..hi tokens its GO launches showed and the run records they rest on, a pynccl_comm
    one needing NCCL's fit one in-switch f32 sum in one run of one tree; ``source``: each run record with the commit it ran. A
    `cited` record (the tree's) has every route resting on run records ``source`` lists; the probe writes one uncited, before its
    run has a record."""
    _fields("the record", rec, _NODE_KEYS)
    if rec["schema"] != NODE_SCHEMA:
        raise ValueError(f"node record: schema {rec['schema']!r}, not {NODE_SCHEMA!r}")
    dev = _fields("device", rec["device"], ("name", "compute_capability", "sms"))
    _text("device.name", dev["name"])
    cc = _text("device.compute_capability", dev["compute_capability"])
    if len(cc.split(".")) != 2 or not all(p.isdigit() for p in cc.split(".")):
        raise ValueError(f"node record: device.compute_capability {cc!r}, not major.minor")
    _int("device.sms", dev["sms"])
    world = ranks(_int("world", rec["world"], 2))
    for k, v in _fields("versions", rec["versions"], ("driver", "nccl", "torch", "vllm", "flashinfer")).items():
        _text(f"versions.{k}", v, none=True)
    mm = _fields("multimem", rec["multimem"], ("trees", "ftz", "probes"))
    if len(_trees("multimem.trees", mm["trees"], world)) != world:
        raise ValueError(f"node record: multimem.trees holds {len(mm['trees'])} trees, one per owner rank of {world}")
    if not isinstance(mm["ftz"], bool):
        raise ValueError(f"node record: multimem.ftz is {mm['ftz']!r}, not a boolean")
    for i, p in enumerate(_list("multimem.probes", mm["probes"])):
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError(f"node record: multimem.probes[{i}] is {p!r}, not [N, M]")
        _int(f"multimem.probes[{i}].N", p[0])
        _int(f"multimem.probes[{i}].M", p[1])
    if rec["nccl"] is not None:
        n = _fields("nccl", rec["nccl"], ("fit", "sizes"))
        if n["fit"] not in NODE_NCCL_FITS:
            raise ValueError(f"node record: nccl.fit {n['fit']!r}, not one of {NODE_NCCL_FITS}")
        for i, s in enumerate(_list("nccl.sizes", n["sizes"])):
            w = f"nccl.sizes[{i}]"
            _fields(w, s, ("N", "M", "bytes", "algo", "proto", "channels", "runs", "trees"))
            size = _int(f"{w}.N", s["N"]) * _int(f"{w}.M", s["M"])
            if _int(f"{w}.bytes", s["bytes"]) != 2 * size:
                raise ValueError(f"node record: {w}.bytes {s['bytes']}, not the {2 * size} of {s['M']} x {s['N']} bf16")
            for k in ("algo", "proto", "channels"):
                _text(f"{w}.{k}", s[k])
            trees = _trees(f"{w}.trees", s["trees"], world)
            at = 0
            for j, r in enumerate(_list(f"{w}.runs", s["runs"])):
                if (not isinstance(r, list) or len(r) != 3 or r[0] != at or _int(f"{w}.runs[{j}].hi", r[1]) <= at
                        or not 0 <= _int(f"{w}.runs[{j}].t", r[2], 0) < len(trees)):
                    raise ValueError(f"node record: {w}.runs[{j}] is {r!r}, not [lo, hi, t] from {at} with t a tree of {w}.trees")
                at = r[1]
            if at != size:
                raise ValueError(f"node record: {w}.runs end at {at}, not the call's {size} elements")
    source = _list("source", rec["source"])
    for i, s in enumerate(source):
        _fields(f"source[{i}]", s, ("run_record", "commit"))
        _text(f"source[{i}].run_record", s["run_record"], none=not cited)
        _text(f"source[{i}].commit", s["commit"], none=True)
    records = {s["run_record"] for s in source}
    seen = set()
    for i, r in enumerate(_list("routes", rec["routes"])):
        w = f"routes[{i}]"
        _fields(w, r, ("setting", "N", "route", "lo", "hi", "run_records"))
        if r["setting"] not in SETTINGS or r["route"] not in NODE_ROUTES:
            raise ValueError(f"node record: {w} is setting {r['setting']!r} route {r['route']!r}: settings {sorted(SETTINGS)}, "
                             f"routes {NODE_ROUTES}")
        if _int(f"{w}.hi", r["hi"]) < _int(f"{w}.lo", r["lo"]):
            raise ValueError(f"node record: {w} lo {r['lo']} > hi {r['hi']}")
        key = (r["setting"], _int(f"{w}.N", r["N"]), r["route"])
        if key in seen:
            raise ValueError(f"node record: {w} repeats {key}")
        seen.add(key)
        arts = _list(f"{w}.run_records", r["run_records"], empty=not cited)
        if cited and not set(arts) <= records:
            raise ValueError(f"node record: {w}.run_records {sorted(set(arts) - records)} are not in source")
        if r["route"] == "pynccl_comm":
            nvls_tree(rec["nccl"], world)
    return rec


def node_key(rec: dict) -> tuple:
    """(device (`device_of`), world, TREES, FTZ) of a node record: what a registration's stated node tree must equal."""
    d = rec["device"]
    return device_of(d["compute_capability"], d["sms"]), rec["world"], multimem_trees(rec["multimem"]["trees"], rec["world"]), \
        rec["multimem"]["ftz"]


def load_node_records(path: Path = NODE_FILE) -> tuple[dict, ...]:
    """Every node record of `path` (a JSON list), each `check_node_record` cited, no two of one node tree (`node_key`)."""
    recs = tuple(check_node_record(r) for r in json.loads(Path(path).read_text()))
    keys = [node_key(r) for r in recs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: two node records hold one node tree")
    return recs


def _node_entries(rec: dict) -> list[Captured]:
    """A node record's routes as `CAPTURED` entries keyed on its node tree: multimem's at its TREES and FTZ, NVLS's at NCCL's
    one tree for every owner."""
    dev, world, trees, ftz = node_key(rec)
    out = []
    for r in rec["routes"]:
        statics = ({"TREES": trees, "FTZ": ftz} if r["route"] == "symm_mem_comm" else
                   {"TREES": (nvls_tree(rec["nccl"], world),) * world, "FTZ": rec["nccl"]["fit"] == "f32_ftz"})
        out.append(Captured(r["setting"], r["N"], r["route"], r["lo"], r["hi"], "AllReduceMultimem_v1", statics,
                            tuple(r["run_records"]), world=world, device=dev, node=(trees, ftz)))
    return out


NODE_RECORDS = load_node_records()
#: Every covered route: `_BASE_CAPTURED`'s, and each node record's keyed on its node tree (`_node_entries`)
CAPTURED = (*_BASE_CAPTURED, *(e for rec in NODE_RECORDS for e in _node_entries(rec)))
#: the node (device, TREES, FTZ) `covered` takes for a registration that states no TREES and FTZ, on a device the caller names
#: or this one: `CAPTURE_RUNS`' node, `GLM53_H200_TREE` on every owner, FTZ off; None refuses it (`node_refusal`'s `TREE_UNSTATED`)
UNSTATED_NODE = (("9.0", H200_SMS), multimem_trees((GLM53_H200_TREE,) * 8, 8), False)


def node_refusal(trees, ftz, *, device, world: int = 8) -> str | None:
    """Why a registration's stated node tree (multimem `trees`, one per owner, and `ftz`) keys no node record at `device` and
    `world`, by name (`TREE_UNSTATED`, `TREE_UNRECORDED`), or None when a record holds it.  Stating neither is `UNSTATED_NODE`."""
    dev = device_of(*device)
    if trees is None and ftz is None and UNSTATED_NODE is not None:
        trees, ftz = UNSTATED_NODE[1:]
    if trees is None or ftz is None:
        return (f"{TREE_UNSTATED}: the registration states no multimem TREES and FTZ for its {world}-rank {dev} node, which "
                "symmetric memory's and NVLS's routes are keyed on")
    node = _stated_node(trees, ftz, int(world))
    if node is None or not any(node_key(r) == (dev, int(world), *node) for r in NODE_RECORDS):
        return (f"{TREE_UNRECORDED}: no node record ({NODE_FILE.name}) holds TREES {trees!r} FTZ {ftz!r} at {world} ranks of "
                f"{dev}; run the 8-GPU capture's --register on the node and add its record")
    return None


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
           conformance="not-yet-established: exercised through the exact-model-tested AllReduceMnnvl_v1 captures at vLLM's "
                       "defaults and with the custom all-reduce off, not tested separately "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMnnvlCoordinate(B, S, col):
    """One element of FlashInfer's MNNVL all-reduce: ``bf16(+0 + s(col[0]) + ... + s(col[WORLD-1]))`` in FTZ f32, s writing -0 as
    +0; SHOT 2 adds the sum to +0 once more."""
    return _mnnvl_sum(B, S, col, B.call(ZERO32), B.call(_NEG_ZERO32))


@composite("AllReduceMnnvl", 1, ["WORLD", "N", "SHOT"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="exact-model-tested: GLM-5.3 TP8 on 8 cc 9.0 GPUs (132 SMs) at vLLM's defaults and with the custom all-reduce off, "
                       "1-42 tokens at N=6144, and at the defaults 1-36 tokens at N=7168 "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
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
           conformance="exact-model-tested: GLM-5.3 TP8 on 8 cc 9.0 GPUs (132 SMs) at vLLM's defaults and with the custom all-reduce off, "
                       "1-42 tokens at N=6144 with both shots and every launch geometry "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceRmsNormMnnvl(B, S, parts, res, w):
    """FlashInfer's MNNVL all-reduce + residual add + RMSNorm (`kARResidualRMSNorm`, fp32 accumulation, weight bias 0) of one
    token: `AllReduceMnnvl_v1`, r = a + res, the sum of squares in `blockReduceSum`'s geometry over CLUSTER CTAs,
    ``rs = rsqrt(sum * rcp(N) + EPS)`` (FMA: one FFMA), and the normed row. Returns (normed, r)."""
    N, shot = int(S.N), strategy(S.SHOT)
    _fused_geometry(N, int(S.CLUSTER))
    zero = B.call(ZERO32)
    a = B.call(bind(AllReduceMnnvl, WORLD=S.WORLD, N=N, SHOT=shot), parts)
    r = B.batch(P.Bf16AddF2fp, a, res, axes=(0, 0))
    return tuple_of(_mnnvl_norm(B, S, zero, r, w), r)


@composite("AllReduceResidualMnnvl", 1, ["WORLD", "N", "SHOT"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))), ("res", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="not-yet-established: AllReduceRmsNormMnnvl_v1's r, whose captures are exact-model-tested; it is that "
                       "Definition's AllReduceMnnvl_v1 and residual add node for node, not captured on its own "
                       "(tests/program/test_tp_allreduce_parts.py)")
def AllReduceResidualMnnvl(B, S, parts, res):
    """`AllReduceRmsNormMnnvl_v1`'s new residual on its own: ``r = a + res`` (`Bf16AddF2fp_v1`) of the MNNVL all-reduce a,
    element by element, so any block of its columns is the Definition at that width on those columns."""
    a = B.call(bind(AllReduceMnnvl, WORLD=S.WORLD, N=S.N, SHOT=strategy(S.SHOT)), parts)
    return B.batch(P.Bf16AddF2fp, a, res, axes=(0, 0))


@composite("RmsNormMnnvl", 1, ["N", "SHOT", "CLUSTER", "EPS", "FMA"],
           lambda S: ((("r", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="not-yet-established: AllReduceRmsNormMnnvl_v1's normed row from its r, whose captures are "
                       "exact-model-tested; it is that Definition's norm node for node, not captured on its own "
                       "(tests/program/test_tp_allreduce_parts.py)")
def RmsNormMnnvl(B, S, r, w):
    """`AllReduceRmsNormMnnvl_v1`'s norm on its own: the normed row of the new residual r with weight w, in the fused kernel's
    geometry (`_fused_geometry`), so ``(RmsNormMnnvl(AllReduceResidualMnnvl(parts, res), w), AllReduceResidualMnnvl(parts,
    res))`` is that Definition."""
    _fused_geometry(int(S.N), int(S.CLUSTER))
    return _mnnvl_norm(B, S, B.call(ZERO32), r, w)


def _mnnvl_norm(B, S, zero, r, w):
    """The fused kernel's normed row of r (one token, `AllReduceRmsNormMnnvl_v1`'s statics), the f32 zero ``zero`` already
    called."""
    N, shot, cluster = int(S.N), strategy(S.SHOT), int(S.CLUSTER)
    block, warps = _fused_geometry(N, cluster)
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
    return B.batch(P.F2fpBf16, y, axes=(0,))


# ---------------------------------------------------------------------------------------------------------------------------------
# vLLM custom all-reduce
# ---------------------------------------------------------------------------------------------------------------------------------


@composite("AllReduceCustomCoordinate", 1, ["WORLD", "START"], lambda S: ((("col", Array(int(S.WORLD), BF16)),), BF16),
           conformance="not-yet-established: exercised through the exact-model-tested AllReduceCustom_v1 captures under "
                       "VLLM_ALLREDUCE_USE_SYMM_MEM=0, not tested separately "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
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
           conformance="exact-model-tested: GLM-5.3 TP8 on 8 cc 9.0 GPUs (132 SMs) under VLLM_ALLREDUCE_USE_SYMM_MEM=0, where alone "
                       "it takes GLM-5.3's all-reduces, 43-682 tokens at N=6144 "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
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
           conformance="not-yet-established: exercised through the exact-model-tested AllReduceMultimem_v1 captures at vLLM's "
                       "defaults and with the custom all-reduce off, not tested separately "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMultimemCoordinate(B, S, col):
    """One element of torch's multimem all-reduce: the f32 adds of TREE over ``f32(col[r])`` (FTZ: FADD.FTZ, else IEEE), then
    RN to bf16."""
    add = P.F32AddFtzV2 if S.FTZ else P.F32AddV2

    def value(t):
        if isinstance(t, int):
            return B.call(P.Bf16ToF32, col[t])
        return B.call(add, value(t[0]), value(t[1]))

    return B.call(P.F2fpBf16, value(multimem_tree(S.TREE, ranks(S.WORLD))))


@composite("AllReduceMultimemVector", 1, ["WORLD", "TREE", "FTZ"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(MULTIMEM_VECTOR, BF16))),), Array(MULTIMEM_VECTOR, BF16)),
           conformance="not-yet-established: exercised through the exact-model-tested AllReduceMultimem_v1 captures at vLLM's "
                       "defaults and with the custom all-reduce off, not tested separately "
                       "(`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py)")
def AllReduceMultimemVector(B, S, parts):
    """One 16-byte vector of torch's multimem all-reduce, the `MULTIMEM_VECTOR` bf16 one ``multimem.ld_reduce`` returns: one
    `AllReduceMultimemCoordinate_v1{TREE}` per element."""
    return B.batch(bind(AllReduceMultimemCoordinate, WORLD=S.WORLD, TREE=S.TREE, FTZ=bool(S.FTZ)), parts, axes=(1,))


@composite("AllReduceMultimem", 1, ["WORLD", "SIZE", "TREES", "FTZ"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)),
           conformance="exact-model-tested: TP8 on 8 cc 9.0 GPUs (132 SMs) at vLLM's defaults and with the custom all-reduce off, the tree "
                       "read off the switch with IEEE adds: GLM-5.3 at N=6144, 43-2048 tokens, and N=7168, "
                       "37-2048 tokens (`CAPTURED`, tests/properties/glm53_tp8_allreduce_capture_gpu.py), its kept words replayed "
                       "on the vector batch (`MULTIMEM_REPLAY_RUN`, tests/properties/glm53_multimem_replay.py)")
def AllReduceMultimem(B, S, parts):
    """torch's `multimem_all_reduce_` of WORLD ranks' SIZE bf16 (the whole flattened tensor, whose slices name each element's
    owner): each run of `multimem_runs` one `AllReduceMultimemVector_v1{TREE=TREES[owner]}` per 16-byte vector."""
    W, V = ranks(S.WORLD), MULTIMEM_VECTOR
    outs = []
    for lo, hi, tree in multimem_runs(W, int(S.SIZE), S.TREES):
        vectors = Array((hi - lo) // V, Array(V, BF16))
        cols = array_of([parts[r][lo:hi].reshape(vectors) for r in range(W)])
        outs.append(B.batch(bind(AllReduceMultimemVector, WORLD=W, TREE=tree, FTZ=bool(S.FTZ)), cols, axes=(1,)))
    return coll_concat(Array(int(S.SIZE), BF16), outs)


# ---------------------------------------------------------------------------------------------------------------------------------
# NCCL's tree all-reduce inside one node
# ---------------------------------------------------------------------------------------------------------------------------------


def nccl_chain(chain, world: int) -> tuple[int, ...]:
    """CHAIN as NCCL's intra-node tree: the ranks 0..WORLD-1 each once, the root first and the leaf last."""
    c = tuple(chain)
    if any(isinstance(r, bool) or not isinstance(r, int) for r in c) or sorted(c) != list(range(world)):
        raise ValueError(f"CHAIN {chain!r}: NCCL's intra-node chain names the ranks 0..{world - 1}, each once, root first")
    return c


@composite("AllReduceNcclTreeCoordinate", 1, ["WORLD", "CHAIN"], lambda S: ((("col", Array(int(S.WORLD), BF16)),), BF16),
           conformance="not-yet-established: exercised through the exact-model-tested AllReduceNcclTree_v1 capture at TP4 on "
                       "RTX PRO 6000 with P2P off, not tested separately (`CAPTURED`, tests/properties/tp4_nccl_tree_replay.py)")
def AllReduceNcclTreeCoordinate(B, S, col):
    """One element of NCCL's tree all-reduce down CHAIN (root first): the leaf's word, then each rank up the chain adds its own,
    ``col[c0] + (col[c1] + (... + col[c_last]))``, each add one bf16 rounding (`Bf16AddF2fp_v1`, ``__hadd2``)."""
    chain = nccl_chain(S.CHAIN, ranks(S.WORLD))
    acc = col[chain[-1]]
    for r in reversed(chain[:-1]):
        acc = B.call(P.Bf16AddF2fp, col[r], acc)
    return acc


@composite("AllReduceNcclTree", 1, ["WORLD", "SIZE", "CHAIN"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)),
           conformance="exact-model-tested: the int8 70B TP4 capture `NCCL_TREE_RUNS` names, on 4 cc 12.0 GPUs (188 SMs) under "
                       "vLLM's batch invariance with NCCL_P2P_DISABLE=1, every all-reduce of its 32 steps (256 and 1 tokens at "
                       "N=8192) on every rank (`CAPTURED`, tests/properties/tp4_nccl_tree_replay.py)")
def AllReduceNcclTree(B, S, parts):
    """pynccl's ``ncclAllReduce(ncclSum, ncclBfloat16)`` of WORLD ranks' SIZE bf16 on NCCL's intra-node tree (the flattened
    tensor; no element's adds read its position): one `AllReduceNcclTreeCoordinate_v1` per element."""
    W = ranks(S.WORLD)
    return B.batch(bind(AllReduceNcclTreeCoordinate, WORLD=W, CHAIN=nccl_chain(S.CHAIN, W)), parts, axes=(1,))


def _int_on(env, name: str) -> bool:
    """An integer switch as NCCL (``strtol``) and vLLM (``int``) read it: unset, empty or not a number is 0."""
    try:
        return int(str(env.get(name, "")).strip() or "0") != 0
    except ValueError:
        return False


def settings_of(disable_custom_all_reduce: bool, env) -> dict:
    """vLLM's all-reduce switches as `served_route` reads them: the engine's ``disable_custom_all_reduce`` and the process's
    ``VLLM_ALLREDUCE_USE_SYMM_MEM`` / ``VLLM_ALLREDUCE_USE_FLASHINFER`` (unset is vLLM's default, on) and ``VLLM_BATCH_INVARIANT``;
    and ``nccl_transport``, the transport NCCL's routes take between one node's GPUs: ``"shm"`` under ``NCCL_P2P_DISABLE`` (with
    ``NCCL_SHM_DISABLE`` off), else None, NCCL's own pick."""
    on = lambda name: str(env.get(name, "1")).strip().lower() not in ("0", "false", "no", "off")  # noqa: E731
    shm = _int_on(env, "NCCL_P2P_DISABLE") and not _int_on(env, "NCCL_SHM_DISABLE")
    return {"custom_all_reduce": not bool(disable_custom_all_reduce), "symmetric_memory": on("VLLM_ALLREDUCE_USE_SYMM_MEM"),
            "flashinfer": on("VLLM_ALLREDUCE_USE_FLASHINFER"), "batch_invariant": _int_on(env, "VLLM_BATCH_INVARIANT"),
            "nccl_transport": "shm" if shm else None}


def served_route(settings: dict, M: int, N: int) -> tuple[str | None, str]:
    """The setting name and communicator route vLLM d9105ea80 takes for one bf16 [M,N] all-reduce: under batch invariance
    pynccl's at any world (`BATCH_INVARIANT`), else the H200 dispatch at world 8."""
    if settings.get("batch_invariant"):
        return BATCH_INVARIANT, "pynccl_comm"
    custom = bool(settings.get("custom_all_reduce"))
    symm = bool(settings.get("symmetric_memory"))
    flashinfer = bool(settings.get("flashinfer"))
    setting = ("custom-off" if not custom and symm and flashinfer else
               "defaults" if custom and symm and flashinfer else
               "switch" if custom and not symm and flashinfer else None)
    nbytes = M * N * 2
    if flashinfer and nbytes <= FI_MAX_BYTES:
        route = "fi_ar_comm"
    elif custom and nbytes < (CUSTOM_SYMM_MAX_BYTES if symm else CUSTOM_MAX_BYTES):
        route = "ca_comm"
    elif symm and nbytes <= SYMM_MEM_MAX_BYTES:
        route = "symm_mem_comm"
    else:
        route = "pynccl_comm"
    return setting, route


def offset_free(c: Captured, world: int) -> bool:
    """Whether every element of the covered route is reduced the same way wherever it sits in the call: MNNVL's per-element sum,
    NCCL's tree (one chain for every element), multimem with one tree for every owner, the custom all-reduce's 1-stage kernel
    (`custom_owners`)."""
    free = {
        "AllReduceMnnvl_v1": lambda: True,
        "AllReduceNcclTree_v1": lambda: True,
        "AllReduceMultimem_v1": lambda: len(set(multimem_trees(c.statics["TREES"], world))) == 1,
        "AllReduceCustom_v1": lambda: int(c.statics["STAGE"]) == 1,
    }.get(c.definition)
    return free is not None and free()


def route_binding(c: Captured, world: int, M: int, N: int, rows: int | None = None) -> tuple[object, dict]:
    """The Definition and statics of one covered communicator all-reduce of [M,N] over `world` ranks: one call over the `rows` * N
    words of this request's rows among the launch's M tokens (default all M).  The launch's M picks the kernel; a call of fewer
    rows than the launch is only stated for an `offset_free` route, since the others reduce an element by its offset in the launch."""
    rows = M if rows is None else int(rows)
    if rows != M and not offset_free(c, world):
        raise ValueError(f"route {c.route!r} ({c.definition} {c.statics}) reduces an element by its offset among the launch's {M} "
                         f"tokens, which a call of this request's {rows} rows does not know")
    size = rows * N
    binding = {
        "AllReduceMnnvl_v1": lambda: (AllReduceMnnvl, {"WORLD": world, "N": size, "SHOT": mnnvl_shot(M, N, world)}),
        "AllReduceMultimem_v1": lambda: (AllReduceMultimem, {"WORLD": world, "SIZE": size, **c.statics}),
        "AllReduceCustom_v1": lambda: (AllReduceCustom, {"WORLD": world, "SIZE": size, **c.statics}),
        "AllReduceNcclTree_v1": lambda: (AllReduceNcclTree,
                                         {"WORLD": world, "SIZE": size, "CHAIN": nccl_chain(c.statics["CHAIN"], world)}),
    }.get(c.definition)
    if binding is None:
        raise ValueError(f"covered route {c.route!r} names unsupported definition {c.definition!r}")
    return binding()
