"""GEN llama: pattern specializations found necessary by the LLAMA32_1B capture (none by QWEN05).

`GemmLaunchTuned` — vLLM d9105ea80's batch-invariant GEMM does not always launch with the tiling the base profile pins.
`vllm/model_executor/determinism/batch_invariant.py::matmul_persistent` passes the launch through
`batch_invariant_configs._get_matmul_config(M, N, K, dtype, default)`: for bf16 on an arch family with a tuned table
("ada" = sm_89 = the pinned RTX 4090; also "hopper", "blackwell") a GEMM whose **(N, K)** is a key of that table takes the
table's tiling for the M-bucket (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `num_stages`; `BLOCK_SIZE_K` shape-wide,
`GROUP_SIZE_M` 8) instead of the default `128/128/64, 8 warps, 3 stages` the base `GemmLaunch` pins.  SmolLM2-135M and
Qwen2.5-1.5B/0.5B never hit a key; Llama-3.2-1B's o_proj is `[M, 2048] x [2048, 2048]` and `(2048, 2048)` is a key, so all
16 layers x 16 steps of it resolved `Unsupported: launch constexpr/geometry differs from the pinned profile`
(`BLOCK_SIZE_M 16` at prefill M=32; `32` at decode M=1; `num_stages 5`).

Numerics: same kernel source (`src_sha` identical to the default launches), same `accumulator = tl.dot(a, b, accumulator)`
chain over 64-wide k tiles in ascending k — `BLOCK_SIZE_M/N`, `num_warps`, `num_stages` only change which output
coordinates a CTA owns and how the pipeline is staged, not the per-coordinate k-order — so the launch is the registered
`Gemm_v1{K,N}` body; Tier A (G4, k >= 3 per specialization against the captured values) is the bit-exactness evidence,
not this docstring.  Rows with another `BLOCK_SIZE_K` are deliberately not in the table below (a different k tiling is a
different numerical claim).

This is a sibling specialization, not a widened matcher: every constexpr and launch-geometry field is still pinned exactly —
to the table's values for that (N, K, M-bucket) instead of the default — and the pattern owns ONLY launches whose (N, K)
is in its explicit table copy.  It returns None for every other GEMM so the base `GemmLaunch` (which must follow it in
the pattern list; the resolver is first-match and `GemmLaunch` answers `Unsupported`, not None, for a mismatched
tiling) keeps deciding those.  A tabled (N, K) launched with any tiling other than its bucket's stays Unsupported."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET

from verity_vllm.observe.fold.patterns import GemmLaunch, base_op, bound_args, gemm_target_of, triton_name
from verity_vllm.observe.fold.resolver import Resolution, StepContext
from verity_vllm.observe.fold.tree import TreeNode

#: [R13 ampere] the pinned sm_89 table now lives in the GEMM target registry (`verity_vllm.program.frontend.gemm_targets.ADA_TUNED_GEMM`, the
#: `ada` record's `tuned_table`) and is re-exported here unchanged for the tests / profile notes that cite it by this name.
from verity_vllm.program.frontend.gemm_targets import ADA_TUNED_GEMM  # noqa: E402,F401


def tuned_launch_config(table: dict, M: int, N: int, K: int) -> tuple[dict[str, Any], int, int] | None:
    """`_get_matmul_config`'s answer for a bf16 `[M,K] x [K,N]` on the tabled device: `(constexprs, num_warps, num_stages)`,
    or None when `(N, K)` is not a key (the default tiling applies)."""
    row = table.get((N, K))
    if row is None:
        return None
    bm, bn, warps, stages = row["m_buckets"][-1][1]
    for max_m, cfg in row["m_buckets"]:
        if max_m >= M:
            bm, bn, warps, stages = cfg
            break
    consts = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": row["BLOCK_SIZE_K"], "GROUP_SIZE_M": 8,
              "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False}
    return consts, warps, stages


@dataclass
class GemmLaunchTuned:
    """`GemmLaunch` for the (N, K) keys of the record TARGET's tuned batch-invariant GEMM table (module docstring).  Same Compute
    (`Gemm_v1{K,N}`, rows=M, (PerRow(a), Shared(w)) -> (PerRow(c),)), same operand / dtype / bias / weight-view checks
    (delegated to a `GemmLaunch` pinned to the bucket's tiling), different pins.

    [R13 ampere] `table=None` (the default) means the registered GEMM target's table (`patterns.gemm_target_of(ctx)` ->
    `GemmTarget.tuned_table`: the sm_89 rows on an `ada` target, nothing on an `ampere` target, whose every launch is the base
    `GemmLaunch`'s default tiling); the pattern is named after the target (`…tuned_ada`).  An explicit `table` keeps the old
    behaviour for synthetic tests.  A target without a registered specialization: None here, refused by the base pattern."""

    table: dict[tuple[int, int], dict[str, Any]] | None = None
    num_sms: int | None = DEFAULT_TARGET.num_sms   # [X-01] profile default only; the delegate GemmLaunch pins the record target's SM count
    name: str = "gemm.matmul_kernel_persistent.tuned"
    kernel: str = "matmul_kernel_persistent"

    def _launch(self, node: TreeNode) -> TreeNode | None:
        if node.is_triton:
            return node if triton_name(node) == self.kernel else None
        if base_op(node.op) in GemmLaunch.wrapper_ops:
            launches = [n for n in node.walk() if n.is_triton and triton_name(n) == self.kernel]
            return launches[0] if len(launches) == 1 else None        # anything else is the base pattern's to report
        return None

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        launch = self._launch(node)
        if launch is None:
            return None
        a = bound_args(launch)
        M, N, K = a.get("M"), a.get("N"), a.get("K")
        if not all(isinstance(v, int) for v in (M, N, K)):
            return None
        table, name = self.table, self.name
        if table is None:
            gt, _refusal = gemm_target_of(ctx)
            if gt is None:
                return None                                            # unregistered target: the base pattern refuses it
            table, name = gt.tuned_table, f"gemm.matmul_kernel_persistent.tuned_{gt.key}"
        cfg = tuned_launch_config(table, M, N, K)
        if cfg is None:
            return None                                                # not a tabled shape: the default tiling, base pattern
        consts, warps, stages = cfg
        pinned = GemmLaunch(consts, num_warps=warps, num_stages=stages, num_sms=self.num_sms, name=name, kernel=self.kernel)
        return pinned.match(node, ctx)


def with_tuned_gemm(patterns: list, **kw: Any) -> list:
    """The base pattern list with `GemmLaunchTuned` inserted right before the base `GemmLaunch` (first-match resolver)."""
    out = list(patterns)
    idx = next(i for i, p in enumerate(out) if type(p) is GemmLaunch)
    out.insert(idx, GemmLaunchTuned(**kw))
    return out
