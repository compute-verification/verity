"""The collective launch record — what the trusted recorder must produce for ONE collective on ONE rank, and the cross-rank link record
built from R of them.  Written from the ov-tp2 evidence (Qwen2.5-1.5B, TP=2, eager, PyNccl; `out/gen/lanes/ov-tp2/{MECHANISM,REPORT}.md`,
`evidence/collectives.json`, `fold_gap.shard.rank{0,1}.json`, `tp2_body.summary.json`, `census_reconcile.rank0.md`).

1. Per-rank collective launch record (`SCHEMA_SHARD`, one row per collective launch of one rank's log)
------------------------------------------------------------------------------------------------------

    rank, world_size            the process's rank and the communicator size (header `notes.tp2`; ranks.json)
    event, step, phase          the launch record (DispatchEnter id of `vllm.all_reduce` / `vllm.all_gather`), its engine step
    kind                        "all_reduce" | "all_gather" | "reduce_scatter" | "vocab_shard_lookup" (the last is rank-LOCAL: no communicator)
    op, module, stream          dispatcher op, module-path hint (diagnostic), CUDA stream handle (one per rank in this configuration)
    communicator                the group the collective runs on — today only the group NAME (`"tp:0"`, a scalar argument of the wrapper op);
                                the NCCL communicator identity (ncclUniqueId / comm hash) is NOT in the record
    k                           position of this collective in the communicator's call sequence WITHIN the step (0-based; all_reduce and
                                all_gather share one sequence — NCCL's contract: the k-th collective on a communicator matches the k-th on
                                every other rank).  `seq` = the same position cumulative over the log; the absolute NCCL sequence number
                                (which counts warm-up / profiling collectives before the log starts) is NOT in the record.
    ordinal_in_step             position among all rows of this table in the step (all_reduce + all_gather + vocab_shard_lookup) — the
                                lane's `ordinal_in_step`, for cross-reference with `collectives.json`
    reduce_op, dtype, count     "sum" (vLLM's wrapper is sum-only), the operand dtype ("bfloat16"), element count N
    operand                     this rank's send buffer view (alloc, offset, shape, strides, dtype, numel, byte range) — the per-rank
                                Gemm output rows (o_proj / down_proj: `a391`, `a395`, …) or the masked embedding lookup (`a383`)
    operand_alloc               {alloc, nbytes, origin} of the operand's storage
    operand_touches             every launch record that referenced the operand allocation before this launch (bounded list, in log
                                order): {event, kind: "dispatch-out" | "dispatch-arg" | "triton-arg", op, module, step, arg, view}
    operand_last_touch          the last of those = the candidate PRODUCER launch (e.g. event 125 `matmul_kernel_persistent` c_ptr for the
                                o_proj all-reduce 126; event 87 `_C.vocab_parallel_embedding` for the embedding all-reduce 88).  Whether
                                that launch WROTE the bytes is the fold's memory model's determination (pattern-declared writes), not this
                                tool's; the fold's instance for that event is joined in the link record when instances are given.
    result                      this rank's receive buffer view (`a384`, `a392`, …; a fresh allocation from DispatchExit) — the bytes
                                that depend on the peer ranks' operands.  Today NO write is journaled for it (fold: `Unresolved: unwritten`).
    result_consumers            every launch record that referenced the result allocation after this launch until its free (bounded)
    result_first_consumer       the first of those (event 92 `_rms_norm_kernel` for `a384`: the fold's first insufficiency point)
    result_freed                whether a StorageFree of the result allocation was seen
    rank_statics                per-rank scalar arguments that differ across ranks (vocab_shard_lookup: org_vocab_start/end — the shard)
    scalars                     all non-tensor arguments of the wrapper op, verbatim

2. Cross-rank link record (`SCHEMA_LINK`)
-----------------------------------------

    key                         (communicator, step, k) for communicator collectives; ("local", step, ordinal_in_step) for the vocab lookup
    kind, op, module, reduce_op, dtype, count, world_size
    ranks[r]                    {event, operand, result, producer (= operand_last_touch), producer_instance, consumers, consumer_instances}
    definition                  the Verity function the launch instantiates: all_reduce → `AllReduceSumBf16_v1{N=count,R=2}` at world 2
                                (quarantine, `verity_vllm.program.quarantine.collective.allreduce`), `AllReduce_v2{WORLD,N}` above
                                (`verity.ml.definitions.registry.b1_tp2`); all_gather → `AllGather2_v1{N}` (a
                                reference rearrangement, no gates); vocab_shard_lookup → `EmbeddingShard_v1{VS,H,START}` per rank
    required_writes[r]          the result view on each rank that the instance's output must be journaled to
    agreement                   per-field agreement across ranks (op, module, operand geometry, result geometry)
    status                      "linked" (all ranks present, all fields agree) | "rank-mismatch" | "missing-rank"

3. What the current record LACKS for admission (facts, from the evidence) — `MISSING_IN_RECORD`
-----------------------------------------------------------------------------------------------

  * the NCCL communicator identity and absolute sequence number (only the group name and the in-log position are recorded);
  * the reduction algorithm / tree-or-ring structure for R > 2 (the kernel name `…_TREE_LL` is in the census, not in the log; the
    order fixes the bf16 rounding sequence for R > 2 — see the Definition's docstring; irrelevant for R = 2);
  * a journaled write of the result view (fold-side: a `Collective` pattern + instance; not an observer fact).
  What is NOT needed: the peer's bytes — the body is checkable from the R ranks' operand records (116/116 snapshotted collectives).
"""
from __future__ import annotations

SCHEMA_SHARD = "verity-capture/tp2-collective-shard/v1"
SCHEMA_LINK = "verity-capture/tp2-collective-link/v1"

#: dispatcher op -> kind.  `vocab_shard_lookup` is rank-local (no communicator) but listed: it produces the embedding all-reduce's operand.
COLLECTIVE_KINDS: dict[str, str] = {
    "vllm.all_reduce.default": "all_reduce",
    "vllm.all_gather.default": "all_gather",
    "vllm.reduce_scatter.default": "reduce_scatter",
    "_C.vocab_parallel_embedding.default": "vocab_shard_lookup",
}
COMMUNICATOR_KINDS = ("all_reduce", "all_gather", "reduce_scatter")

REQUIRED_LAUNCH_FIELDS = ("rank", "world_size", "event", "step", "phase", "kind", "op", "module", "stream", "communicator", "k", "seq",
                          "ordinal_in_step", "reduce_op", "dtype", "count", "operand", "operand_alloc", "operand_touches", "operand_last_touch",
                          "result", "result_consumers", "result_first_consumer", "result_freed", "rank_statics", "scalars")

MISSING_IN_RECORD = (
    "nccl communicator identity (only the group name 'tp:N' is recorded as a scalar argument)",
    "absolute nccl sequence number (only the in-log position k / seq; warm-up collectives precede the log)",
    "reduction algorithm and tree/ring structure (census kernel name only: *_TREE_LL / *_RING_LL; order matters for R > 2 only)",
    "a journaled write of the result view on each rank (fold-side Collective pattern; the result is 'unwritten' today)",
)


def validate_launch(d: dict) -> list[str]:
    """Missing / ill-typed required fields of one per-rank launch record (empty list = valid)."""
    problems = [f"missing field {f!r}" for f in REQUIRED_LAUNCH_FIELDS if f not in d]
    if not problems:
        if d["kind"] not in set(COLLECTIVE_KINDS.values()):
            problems.append(f"unknown kind {d['kind']!r}")
        if d["kind"] in COMMUNICATOR_KINDS and not d["communicator"]:
            problems.append(f"{d['kind']} without a communicator")
        if d["kind"] in COMMUNICATOR_KINDS and d["result"] is None:
            problems.append(f"{d['kind']} without a result view")
        if d["operand"] is None:
            problems.append("no operand view")
    return problems
