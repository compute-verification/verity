"""GEMM patterns: the batch-invariant persistent matmul launch and the linear's separate bias add."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_weight_view, is_row_major
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args, first, triton_name
from verity_vllm.observe.fold.patterns.resolutions import _check_pins, _fmt, _group_compute, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import _targets, gemm_target_of, pinned_num_sms, target_of
from verity_vllm.observe.fold.patterns.transparent import scope_transparents
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, Shared, StepContext, Transparent
from verity_vllm.observe.fold.tree import TreeNode


@dataclass
class GemmLaunch:
    """`TritonLaunch …matmul_kernel_persistent` (vLLM batch-invariant GEMM, launched from Python by
    `matmul_persistent(a, b)`; no `aten.mm` in the eager batch-invariant path — the launch is a root-level event
    preceded by the `aten.t` alias of the weight) → `Compute(Gemm_v1{K,N}, rows=M, (PerRow(a), Shared(w)),
    (PerRow(c),))` with `w` the `[N,K]` parameter view behind the transposed `b_ptr`.  Also matches an
    `aten.mm/matmul/linear/addmm` dispatcher node that wraps exactly one such launch (the launch fired inside the
    op's CUDA implementation), consuming the wrapper as scope.

    Constexpr pins (`constexprs`, `num_warps`, `num_stages`) must match exactly.  [R13 ampere] The TILING pinned is the
    record target's launch geometry for (M, N, K) — `gemm_target_of(ctx)`: the registered GEMM specialization of the
    header's capability (`verity_vllm.program.frontend.gemm_targets`; `ada` = vLLM's sm_89 tuned table rows else the default, `ampere`
    = the default for every shape); `constexprs` / `num_warps` / `num_stages` given at construction are the profile's default
    expectation for the fields the target does not decide.  A target without a registered specialization is Unsupported
    (`numerics_unregistered`) whatever the launch shows.  The `NUM_SMS` pin is the record target's SM count ([X-01]
    `pinned_num_sms`: one `TargetProfile` fact, read from the run header's device document; `num_sms` is the profile's own
    default when the header has none).  A bound bias (`HAS_BIAS` / non-null `bias_ptr`) is the
    dispatcher-invisible `addmm` path: Unsupported ("bias path") — `Gemm` then `BiasAdd` from one launch is not
    expressible as one Compute (report proposal)."""

    constexprs: dict[str, Any]
    num_warps: int | None = None
    num_stages: int | None = None
    num_sms: int | None = None
    name: str = "gemm.matmul_kernel_persistent"
    kernel: str = "matmul_kernel_persistent"
    wrapper_ops: frozenset = frozenset({"aten.mm", "aten.matmul", "aten.linear", "aten.addmm", "aten.bmm"})

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        before: list[Transparent] = []
        if node.is_triton:
            if triton_name(node) != self.kernel:
                return None
            launch = node
        elif base_op(node.op) in self.wrapper_ops:
            launches = [n for n in node.walk() if n.is_triton and triton_name(n) == self.kernel]
            if len(launches) != 1:
                return unsupported(node, f"{base_op(node.op)} without exactly one batch-invariant {self.kernel} launch",
                                   launches=[n.enter.fn for n in launches])
            launch = launches[0]
            before, bad = scope_transparents(node, ctx, {launch.id}, self.name)
            if bad is not None:
                return bad
        else:
            return None
        a = bound_args(launch)
        consts = dict(launch.enter.constexprs)
        # [R13 ampere] the tiling (BLOCK_SIZE_*, num_warps, num_stages) a launch must show is the record TARGET's launch geometry for
        # (M, N, K) — the registered GEMM specialization's `launch_config` (verity_vllm.program.frontend.gemm_targets: vLLM's tuned-table row for the
        # target's arch family, else the default), the same source Bet A pins.  An unregistered target is refused whatever the launch shows.
        gt, refusal = gemm_target_of(ctx)
        if gt is None:
            return unsupported(launch, refusal, compute_capability=list(target_of(ctx)[0].compute_capability))
        expected = dict(self.constexprs)
        exp_warps, exp_stages = self.num_warps, self.num_stages
        Mi, Ni, Ki = a.get("M"), a.get("N"), a.get("K")
        if all(isinstance(v, int) for v in (Mi, Ni, Ki)):
            tconsts, twarps, tstages, _src = gt.launch_config(Mi, Ni, Ki)
            expected.update({k: v for k, v in tconsts.items() if k in expected or k.startswith("BLOCK_SIZE_")})
            if exp_warps is not None:
                exp_warps = twarps
            if exp_stages is not None:
                exp_stages = tstages
        got = {k: first(consts.get(k), a.get(k)) for k in expected}
        bad = _check_pins(launch, got, expected)
        if bad is not None:
            return bad
        geom = {}
        pins = {}
        if exp_warps is not None:
            geom["num_warps"], pins["num_warps"] = launch.enter.num_warps, exp_warps
        if exp_stages is not None:
            geom["num_stages"], pins["num_stages"] = launch.enter.num_stages, exp_stages
        if self.num_sms is not None:   # [X-01] the pin is the record target's SM count (one TargetProfile fact), else the profile's
            geom["NUM_SMS"], pins["NUM_SMS"] = first(consts.get("NUM_SMS"), a.get("NUM_SMS")), pinned_num_sms(ctx, self.num_sms)
        bad = _check_pins(launch, geom, pins)
        if bad is not None:
            return bad
        a_ptr, b_ptr, c_ptr, bias = a.get("a_ptr"), a.get("b_ptr"), a.get("c_ptr"), a.get("bias_ptr")
        M, N, K = a.get("M"), a.get("N"), a.get("K")
        if not all(isinstance(v, ViewDesc) for v in (a_ptr, b_ptr, c_ptr)) or not all(isinstance(v, int) for v in (M, N, K)):
            return unsupported(launch, "matmul launch arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if bias is not None or first(consts.get("HAS_BIAS"), a.get("HAS_BIAS")):
            return unsupported(launch, "bias path (HAS_BIAS / bias_ptr bound): Gemm then BiasAdd from one launch is not one Compute",
                               bias=repr(bias) if isinstance(bias, ViewDesc) else bias)
        if a_ptr.shape != (M, K) or not is_row_major(a_ptr):
            return unsupported(launch, "a_ptr is not a row-major [M,K] view", a=repr(a_ptr), M=M, K=K)
        if c_ptr.shape != (M, N) or not is_row_major(c_ptr):
            return unsupported(launch, "c_ptr is not a row-major [M,N] view", c=repr(c_ptr), M=M, N=N)
        try:
            w = as_weight_view(b_ptr, K, N)
        except GeometryError as e:
            return unsupported(launch, str(e))
        if not (a_ptr.dtype == b_ptr.dtype == c_ptr.dtype == "bfloat16"):
            return unsupported(launch, "Gemm_v1 is the bf16 kernel", dtypes=[a_ptr.dtype, b_ptr.dtype, c_ptr.dtype])
        if K % 16:
            return unsupported(launch, "K is not a multiple of the 16-wide k-step", K=K)
        if not node.is_triton:
            outs = node.output_views()
            if not outs or outs[0].alloc != c_ptr.alloc:
                return unsupported(node, "wrapper output is not the launch's c_ptr allocation")
        roots = ctx.roots_of(w)
        wname = roots[0].name if roots else ""
        rows = M
        # [R13 hopper] the registered GEMM Definition of the record's target (`frontend.targets.gemm_spec`: Gemm_v1 on the accepted
        # Ampere-family target, Gemm_v2{DOT=<target step>} otherwise) — the same selection the Bet A builder makes
        try:
            spec = _targets().gemm_spec(target_of(ctx)[0], K, N)
        except _targets().UnregisteredTarget as e:
            return unsupported(launch, f"target has no registered GEMM binding: {e}")
        comp = Compute(spec, rows, (PerRow(a_ptr), Shared(w)), (PerRow(c_ptr),),
                       tuple(sorted(node.ids())), hint=hint_of(launch, wname or "gemm"), kernels=(self.kernel,),
                       attribution=attribution_for(ctx, rows, node.phase), pattern=self.name)
        if before:
            return _group_compute(comp, before=tuple(before))
        return comp


@dataclass
class BiasAdd:
    """`aten.add.Tensor(y[M,N] bf16, b[N] bf16 weight Root)` — the batch-invariant linear's separate bias add
    (`linear_batch_invariant`: `output + bias`) → `Compute(BiasAdd_v1{N}, rows=M, (PerRow(y), Shared(b)), (PerRow(out),))`.
    Absent for SmolLM2 (`BIAS=False`); present for Qwen2 qkv_proj."""

    name: str = "bias_add.aten_add"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.add":
            return None
        args = node.args
        if len(args) < 2 or not isinstance(args[0], ViewDesc) or not isinstance(args[1], ViewDesc):
            return None
        y, b = args[0], args[1]
        outs = node.output_views()
        if y.rank != 2 or b.rank != 1 or y.shape[1] != b.shape[0] or not outs:
            return None
        if not any(r.role == "weight" for r in ctx.roots_of(b)):
            return None
        alpha = bound_args(node).get("alpha", 1)
        if alpha not in (1, 1.0):
            return unsupported(node, "aten.add with alpha != 1", alpha=alpha)
        if not (y.dtype == b.dtype == outs[0].dtype == "bfloat16") or not is_row_major(y) or not is_row_major(outs[0]):
            return unsupported(node, "BiasAdd_v1 is bf16 over row-major [M,N] views", dtypes=[y.dtype, b.dtype, outs[0].dtype])
        M, N = y.shape
        return Compute(bind(b1.BiasAdd, N=N), M, (PerRow(y), Shared(b)), (PerRow(outs[0]),), (node.id,),
                       hint=hint_of(node, ctx.roots_of(b)[0].name), kernels=("vectorized_elementwise_kernel", "elementwise_kernel"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


def fuse_bias_adds(resolutions: list, ctx: StepContext, roots: list[TreeNode]) -> list:
    """The linear's bias add as part of its GEMM: the record target's GEMM Compute (the launch) whose output the step names ONLY in the
    next Compute, a `BiasAdd_v1` over exactly that output (only aliasing / no-op resolutions between them), becomes a no-op at the launch
    and ONE Compute of the target's one-Call Definition at the add (`_gemm_bias_of`) -- the frontend's binding of the same pair
    (`TritonGemmRule`), so the pre-bias word is interior to each coordinate on both sides.  `roots[k]` is the event subtree
    `resolutions[k]` resolves; any other event of the step naming the launch's output allocation (a view of it, another reader) keeps the
    pair.  That allocation is then never written: a read of it anywhere is an uninitialised read, never a silent value.  One resolution
    per root is kept."""
    out = list(resolutions)
    for i, g in enumerate(out):
        if type(g) is not Compute or not isinstance(g.outputs[0], PerRow):
            continue
        spec = _gemm_bias_of(g, ctx)
        if spec is None:
            continue
        c = g.outputs[0].view
        j = i + 1
        while j < len(out) and type(out[j]) is Transparent and out[j].effect in ("alias", "noop") and not _names(roots[j], c.alloc):
            j += 1
        if j >= len(out):
            continue
        a = out[j]
        if (type(a) is not Compute or a.spec.id.split("{")[0] != "BiasAdd_v1" or a.rows != g.rows or a.operands[0] != PerRow(c)
                or not isinstance(a.operands[1], Shared)
                or any(_names(r, c.alloc) for k, r in enumerate(roots) if k not in (i, j) and not _allocates(out[k]))):
            continue
        w, b = g.operands[1].view, a.operands[1].view
        names = tuple(rs[0].name if rs else None for rs in (ctx.roots_of(w), ctx.roots_of(b)))
        out[i] = Transparent("noop", g.consumed, rows=g.rows, hint=g.hint, pattern="gemm_bias.launch")
        out[j] = Compute(spec, a.rows, (g.operands[0], g.operands[1], a.operands[1]), a.outputs, a.consumed,
                         hint=a.hint, kernels=g.kernels + a.kernels, attribution=a.attribution, pattern="gemm_bias.fused",
                         roots=(None, *names))
    return out


def _gemm_bias_of(g: Compute, ctx: StepContext):
    """The record target's one-Call Definition of a biased linear whose launch is `g` (`frontend.targets.gemm_bias_spec`: `GemmBias_v1`
    or `GemmBias_v2{DOT}`), or None: `g` is not the target's GEMM Definition of its (K, N), or the target keeps the pair as two Calls."""
    T = _targets()
    statics = getattr(g.spec, "bindings", None) or {}
    K, N = statics.get("K"), statics.get("N")
    if K is None or N is None:
        return None
    tp = target_of(ctx)[0]
    try:
        return T.gemm_bias_spec(tp, K, N) if g.spec.id == T.gemm_spec(tp, K, N).id else None
    except T.UnregisteredTarget:
        return None


def _allocates(r) -> bool:
    """An allocation (`aten.empty` / `zeros` / ... : `matmul_persistent`'s `c = torch.empty(...)`): it reads nothing."""
    return type(r) is Transparent and r.effect == "alloc"


def _names(root: TreeNode, alloc: int) -> bool:
    """Whether any event of `root`'s subtree reads or writes a view of allocation `alloc`."""
    return any(v.alloc == alloc for n in root.walk() for v in (*n.input_views(), *n.output_views()))
