"""`AllReduceSumBf16{N, R}` — the body of one NCCL sum all-reduce over R ranks' bf16 vectors of length N (lane tp2, quarantine).

Production op (vLLM d9105ea80, `tensor_parallel_size=2`, batch-invariant profile, `disable_custom_all_reduce=True`):
`torch.ops.vllm.all_reduce(tensor, group_name="tp:0")` → `GroupCoordinator.all_reduce` → `PyNcclCommunicator.all_reduce(in, out=None,
op=ReduceOp.SUM, stream=None)` → `ncclAllReduce(send, recv, count, ncclBfloat16, ncclSum, comm, current_stream)`; the device kernel the
census sees is `ncclDevKernel_AllReduce_Sum_bf16_TREE_LL(ncclDevKernelArgsStorage<4096ul>)` (grid 1, block 640), NCCL 2.29.7
(`out/gen/lanes/ov-tp2/evidence/census_reconcile.rank0.md`, `stream_facts.json`).

What is KNOWN about the arithmetic (evidence + NCCL source reading):
  * reduction op `ncclSum`, datatype `ncclBfloat16`: NCCL reduces IN THE DATATYPE — the partial sum that travels between ranks is a
    bf16 word (`src/device/reduce_kernel.h`, `FuncSum<__nv_bfloat16>`: `__hadd2` on bf16x2 for sm_80+, i.e. `add.rn.bf16x2`, one RNE
    rounding of the exact sum of two bf16 values; the pre-sm_80 fallback widens to f32, adds and narrows with `__floats2bfloat162_rn`,
    which is the same function of two bf16 words — see `_exact_bf16_sum` in the tests: RN32 of the sum of two bf16 values is exact or
    far from every bf16 midpoint, so RN_bf16(RN32(a+b)) = RN_bf16(a+b)).  Each binary addition therefore costs one bf16 rounding, and
    a reduction over R ranks has R-1 roundings, one per hop.  (Source reading from memory of NCCL 2.1x–2.2x; not re-read against the
    captured 2.29.7 in this window — the R = 2 statement does not depend on it, the per-hop statement for R > 2 does.)
  * for R = 2 there is one addition per element; addition is commutative, so the per-element body is the order-free
    `bf16_rne(f32(x_0) + f32(x_1))`.  Measured (lane ov-tp2, `tp2_body.summary.json`): 114/114 snapshotted all-reduces of the TP=2
    capture satisfy it, both ranks' outputs bit-equal (`out_sha_equal_across_ranks`); lane tp2 (TA1 commit 917b9faa): 99/99 leaves.
  * every rank receives the same output word (tree: the root reduces and broadcasts; ring: the reduce-scatter result is all-gathered).

What is NOT known (and why this stays in quarantine):
  * for R > 2 the ORDER of the R-1 additions is a property of NCCL's algorithm choice and topology, not of the call: TREE
    (leaves → root along the tree NCCL built for the communicator; with two interleaved trees per channel), or RING (for chunk c the
    reduce-scatter accumulation starts at rank (c + const) mod R and walks the ring — different elements of ONE buffer are summed in
    different rank orders).  The kernel name records the algorithm (`TREE`) and protocol (`LL`); the tree/ring structure itself is only
    printed under `NCCL_DEBUG=INFO` and is not in the record.  This module therefore refuses R > 2: the one statement of the order is
    `b1_tp2.allreduce_order`, and the R > 2 all-reduce is `b1_tp2.all_reduce(R, N)`.
  * the NaN word `__hadd2` produces (0x7FFF as `F2fpBf16`, or payload-preserving) — no NaN occurred in the evidence; `F2fpBf16` is used.
  * the custom all-reduce path (`_C.custom_ar`, `cross_device_reduce_1stage/2stage`) accumulates in F32 in rank order 0 → R-1 and
    rounds ONCE at the end (`custom_all_reduce.cuh::packed_reduce`): a different function for R > 2 (equal for R = 2).  It was disabled
    in every capture so far and is not modelled here.

Conformance: quarantine — `difftest` pending (the production body needs a 2-GPU producer: two processes and an NCCL communicator; the
protected `difftest.produce` is single-process, so the adapter `allreduce_difftest.py` spawns the second rank itself; untested on GPU in
this window).  Nothing is claimed exact beyond the R = 2 statement above.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16
from verity.ml.circuits.types import Array

SEMANTIC_ID = "tp2/AllReduceSumBf16@v1"


def _rank_params(S):
    return tuple((f"x{r}", BF16) for r in range(int(S.R)))


def _two_ranks(R: int) -> None:
    if R < 2:
        raise ValueError("an all-reduce needs at least two ranks")
    if R > 2:
        raise ValueError(f"R = {R}: the hop order over more than two ranks is b1_tp2.allreduce_order; use b1_tp2.all_reduce({R}, N)")


@composite("AllReduceSumBf16Elem", 1, ["R"], lambda S: (_rank_params(S), BF16),
           conformance="quarantine (lane tp2): NCCL ncclSum/ncclBfloat16 reduces in the datatype, one RNE rounding per hop; R=2 only "
                       "(R>2 refused: its hop order is b1_tp2.allreduce_order, its all-reduce b1_tp2.AllReduce_v2); R=2 is one "
                       "commutative add = b1_tp2.AllReduce2 on non-NaN words (ov-tp2: 114/114 snapshotted all-reduces, both ranks "
                       "bit-equal); difftest pending")
def AllReduceSumBf16Elem(B, S, *xs):
    """One element of the all-reduce output: `acc = x_0; acc = F2fpBf16(F32Add(Bf16ToF32(acc), Bf16ToF32(x_r)))` for r = 1..R-1.
    `R = 2`: `bf16_rne(f32(x_0) + f32(x_1))`, order-free."""
    R = int(S.R)
    _two_ranks(R)
    if len(xs) != R:
        raise TypeError(f"AllReduceSumBf16Elem{{R={R}}}: {len(xs)} operands")
    acc = xs[0]
    for r in range(1, R):
        s = B.call(P.F32Add, B.call(P.Bf16ToF32, acc), B.call(P.Bf16ToF32, xs[r]))
        acc = B.call(P.F2fpBf16, s)
    return acc


@composite("AllReduceSumBf16", 1, ["N", "R"], lambda S: (tuple((f"x{r}", Array(int(S.N), BF16)) for r in range(int(S.R))), Array(int(S.N), BF16)),
           conformance="quarantine (lane tp2): elementwise AllReduceSumBf16Elem{R}; see that definition")
def AllReduceSumBf16(B, S, *xs):
    """`out[i] = AllReduceSumBf16Elem{R}(x_0[i], ..., x_{R-1}[i])` for i < N — every rank's copy of the all-reduce result.  The R
    operands are the R ranks' operand vectors (rank r's row of the row-parallel Gemm / masked embedding lookup); the output is the
    value every rank's result view receives.  Program-side the same collection is bound once and read by every rank's consumers."""
    R = int(S.R)
    if len(xs) != R:
        raise TypeError(f"AllReduceSumBf16{{N={S.N},R={R}}}: {len(xs)} operands")
    return B.batch(bind(AllReduceSumBf16Elem, R=R), *xs, axes=(0,) * R)


def all_reduce_sum_bf16(N: int, R: int = 2):
    """`bind(AllReduceSumBf16, N=N, R=R)` — the specialization one launch record needs (`N = count`, `R = world_size`)."""
    return bind(AllReduceSumBf16, N=int(N), R=int(R))


def all_reduce_sum_bf16_words(xs: list[list[int]]) -> list[int]:
    """The same function evaluated directly on bf16 encodings (R lists of N words), for tests and the linkage tool's self-check:
    hop r: `F2fpBf16(F32Add(Bf16ToF32(acc), Bf16ToF32(x_r)))`."""
    R = len(xs)
    _two_ranks(R)
    N = len(xs[0])
    out: list[int] = []
    for i in range(N):
        acc = xs[0][i]
        for r in range(1, R):
            acc = P.F2fpBf16.evaluate(P.F32Add.evaluate(P.Bf16ToF32.evaluate(acc), P.Bf16ToF32.evaluate(xs[r][i])))
        out.append(acc)
    return out


ALL = [AllReduceSumBf16Elem, AllReduceSumBf16]
