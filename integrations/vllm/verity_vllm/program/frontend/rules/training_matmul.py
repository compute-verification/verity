"""The training rules' matrix products (`rules.training`): `mm`, `bmm` and `mv` as `LinearRef_v1` rows, and at a compute capability
whose cuBLAS routes are captured, a bf16 `mm` or `mv` as F.linear's GEMM on that GPU and, under allow_tf32, an f32 `mm` as its
TF32 GEMM (`Matmul`)."""

from __future__ import annotations

import torch

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.training_emit import _DT, BF16, F32, _batch, _shape, _TrainRule, aten
from verity_vllm.program.frontend.rules.training_views import _info, _view_permute
from verity_vllm.program.frontend.strided_views import _tensor_type

#: F.linear's GEMM at x (M, K), W (N, K) by its mm's operand layouts (`_layout`) as autograd lays them out: x @ W^T, dY @ W, dY^T @ X
_ROLES = {("row", "col"): "fwd", ("row", "row"): "dX", ("col", "row"): "dW"}
#: F.linear's (M, K, N) from its role's mm(a[Mm, Km], b[Km, Nm])
_LINEAR_DIMS = {"fwd": lambda Mm, Km, Nm: (Mm, Km, Nm), "dX": lambda Mm, Km, Nm: (Mm, Nm, Km), "dW": lambda Mm, Km, Nm: (Km, Nm, Mm)}
#: a layout's transpose: `mv(A, x)` is the mm x[1, K] @ A^T
_FLIP = {"row": "col", "col": "row"}


def _layout(t) -> str | None:
    """A dense matrix's layout in memory: "row" (row-major), "col" (column-major: a row-major one transposed), else None."""
    if t is None or t.dim() != 2 or t.storage_offset() != 0:
        return None
    rows, cols = t.shape
    return {(cols, 1): "row", (1, rows): "col"}.get(tuple(t.stride()))


class Matmul(_TrainRule):
    """`mm(a[M,K], b[K,N])` / `bmm(a[B,M,K], b[B,K,N])` of f32 or bf16 values -> one `batch` of `LinearRef_v1{K,N,DT}` over the rows
    of `a` (per batch index for `bmm`), with `b` transposed by reference, and a bf16 `mv(A[N,K], x[K])` -> `LinearRef_v1{K,N,DT}`
    of x against A's rows, the one row of x[1, K] @ A^T that torch CPU's mv computes.  At a compute capability ``cc`` whose cuBLAS
    routes are captured (`cublas_bf16.TRAIN_CC`), a bf16 `mm` is one batch of F.linear's row GEMM as its cuBLAS computes it
    instead: its role read off the operands' layouts (`_ROLES`), its route the capture's (`cublas_bf16.train_route`) and its
    Definition the route's on words, an unsplit one padded with +0 words to whole k16 steps (`cublas_bf16.route_gemm`); a bf16
    `mv` is that GEMM at one row, x[1, K] @ A^T, as torch's CUDA gemv calls cuBLAS's GEMM with one column.  With ``tf32``
    (cuda.matmul.allow_tf32) an f32 `mm` of dense row- or column-major operands at a shape the TF32 capture covers
    (`cublas_bf16.TRAIN_TF32`) is one batch of cuBLAS's unsplit TF32 GEMM's row (`cublas_bf16.train_tf32`, `GemmTf32_v1`) over a's
    rows against b's columns.  The rule still claims a product it can't lower that way and refuses it with the capture it needs,
    so that no other rule lowers it with torch CPU's arithmetic: at ``cc`` a bf16 `bmm`, a layout no role has, a shape the capture
    doesn't cover, a route with no Definition (`cublas_bf16.route_gemm` refuses it), with ``tf32`` an f32 `bmm`, or an f32 `mm`
    at another shape or layout (cuBLAS's TF32 GEMM); an f32 `mv` (cuBLAS's sgemv, or torch CPU's BLAS gemv, whose order is not
    its mm's row); and a `dot` (cuBLAS's dot kernel, or torch CPU's f32 BLAS dot)."""
    info = _info("Matmul", "aten.mm / aten.bmm / aten.mv of f32 or bf16 values", "both operands are values of one dtype, f32 or bf16",
                 "one batch of LinearRef rows per matrix")

    def __init__(self, cc: tuple[int, int] | None = None, tf32: bool = False):
        if cc is not None and tuple(cc) != CB.TRAIN_CC:
            raise ValueError(f"Matmul: cc {tuple(cc)} has no captured cuBLAS routes (cublas_bf16.TRAIN_CC is {CB.TRAIN_CC})")
        if tf32 and cc is None:
            raise ValueError("Matmul: tf32 is a GPU's (cuda.matmul.allow_tf32); without a cc the products are torch CPU's")
        self.cc, self.tf32 = None if cc is None else f"cc{cc[0]}.{cc[1]}", tf32
        if cc is not None:
            self.info = _info(f"Matmul_{self.cc}" + ("_tf32" if tf32 else ""), "aten.mm / aten.bmm / aten.mv of f32 or bf16 values",
                              "both operands are values of one dtype, f32 or bf16",
                              f"a bf16 mm or mv: one of F.linear's GEMMs, at a shape and route {self.cc}'s capture covers",
                              "an f32 mm or bmm without tf32; with it an f32 mm at a shape the TF32 capture covers",
                              "one batch of rows per matrix: the cuBLAS route's GEMM for a bf16 mm or mv, its TF32 GEMM for "
                              "an f32 mm with tf32, else LinearRef")

    def targets(self):
        return {aten.mm.default, aten.bmm.default, aten.mv.default, aten.dot.default}

    def applies(self, node, ctx):
        for a in node.args[:2]:
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return "an operand is not a value"
        dts = {ctx.dtype_of(a) for a in node.args[:2]}
        if len(dts) != 1 or not dts <= set(_DT):
            return f"numerics_unregistered: dtypes {sorted(map(str, dts))}"
        return True

    def emit(self, node, ctx):
        dt, mv = ctx.dtype_of(node.args[0]), node.target is aten.mv.default
        if self.tf32 and dt == torch.float32 and node.target is aten.mm.default:
            return self._emit_tf32(node, ctx)
        if node.target is aten.dot.default or dt == torch.float32 and (mv or self.tf32):
            self._no_order(node, ctx, dt)
        if self.cc is not None and dt == torch.bfloat16:
            return self._emit_gpu(node, ctx)
        E = BF16 if dt == torch.bfloat16 else F32
        if mv:
            (N, K) = _shape(ctx, node.args[0])
            fn = ctx.kind("linear", node, K=K, N=N, DT=_DT[dt], BIAS=False)
            x = ctx.coll(node.args[1]).reshape(Array(1, Array(K, E)))
            out = _batch(ctx, node, fn, x, ctx.coll(node.args[0]).reshape(Array(N, Array(K, E))), axes=(0, None))
            ctx.define(node, Coll(_tensor_type([N], E.w), out.refs))
            ctx.record(self, node, {"batch": 1, "M": 1, "K": K, "N": N}, emitted=f"{fn.id} x1")
            return
        a, b = ctx.coll(node.args[0]), ctx.coll(node.args[1])
        sa, sb = _shape(ctx, node.args[0]), _shape(ctx, node.args[1])
        nb = sa[0] if len(sa) == 3 else 1
        M, K, N = sa[-2], sa[-1], sb[-1]
        fn = ctx.kind("linear", node, K=K, N=N, DT=_DT[dt], BIAS=False)
        a3 = a.reshape(Array(nb, Array(M, Array(K, E))))
        b3 = b.reshape(Array(nb, Array(K, Array(N, E))))
        outs = []
        for i in range(nb):
            bT = _view_permute(b3[i], (1, 0), node.name)
            outs.append(_batch(ctx, node, fn, a3[i], bT, axes=(0, None)))
        ctx.define(node, Coll(_tensor_type(list(sa[:-1]) + [N], E.w), concat([o.refs for o in outs])))
        ctx.record(self, node, {"batch": nb, "M": M, "K": K, "N": N}, emitted=f"{fn.id} x{nb * M}")

    def _no_order(self, node, ctx, dt):
        """Refuses a product whose arithmetic is an order no Definition has, naming the capture that would give it one."""
        shapes = " @ ".join(str(list(_shape(ctx, a))) for a in node.args[:2])
        op, where = str(node.target).split(".")[1], f"cc {CB.TRAIN_CC[0]}.{CB.TRAIN_CC[1]}"
        if self.cc is None:
            why = (f"torch CPU's {op}: no rule lowers it, since its f32 {op} is its BLAS's ({'sdot' if op == 'dot' else 'sgemv'}), "
                   f"whose order is not its mm's row (LinearRef_v1)")
        elif node.target is aten.dot.default:
            why = (f"a {str(dt).split('.')[1]} dot on {self.cc} is cuBLAS's dot kernel, its per-block partial sums and their "
                   f"reduction in an order no capture covers: it needs torch.dot's words on {where} at {shapes}, on words whose "
                   f"sums each candidate order rounds apart")
        elif node.target is aten.mv.default:
            why = (f"an f32 mv on {self.cc} is cuBLAS's sgemv, its lanes' order uncaptured: it needs torch.mv's kernel and words on "
                   f"{where} at {shapes}")
        else:
            why = (f"an f32 {op} with cuda.matmul.allow_tf32 on {self.cc} is cuBLAS's TF32 GEMM (each operand rounded to TF32 and "
                   f"multiplied in the tensor cores' TF32 steps), which no capture covers here (the TF32 capture's are dense mm's "
                   f"at (M, K, N) in {CB.TRAIN_TF32}): it needs torch.{op}'s kernels and words on {where} with allow_tf32 at "
                   f"{shapes}, fitted to a TF32 step's route")
        refuse("numerics_unregistered", node, ctx.module_path(node), why)

    def _emit_tf32(self, node, ctx):
        """An f32 `mm(a[M, K], b[K, N])` under allow_tf32: `GemmTf32_v1{K, N}` of each of a's rows against b^T's rows, where the
        TF32 capture covers its (M, K, N) and both operands are dense row- or column-major; else `_no_order`'s refusal."""
        (M, K), N = _shape(ctx, node.args[0]), _shape(ctx, node.args[1])[1]
        if None in (_layout(ctx.tensor_meta(a)) for a in node.args[:2]) or (M, K, N) not in CB.TRAIN_TF32:
            self._no_order(node, ctx, torch.float32)
        fn = CB.train_tf32(M, K, N)
        a = ctx.coll(node.args[0]).reshape(Array(M, Array(K, F32)))
        w = _view_permute(ctx.coll(node.args[1]).reshape(Array(K, Array(N, F32))), (1, 0), node.name)
        out = _batch(ctx, node, fn, a, w, axes=(0, None))
        ctx.define(node, Coll(_tensor_type([M, N], 32), out.refs))
        ctx.record(self, node, {"M": M, "K": K, "N": N, "tf32": True}, emitted=f"{fn.id} x{M}")

    def _emit_gpu(self, node, ctx):
        def no(why: str):
            refuse("numerics_unregistered", node, ctx.module_path(node), f"a bf16 {node.target} on {self.cc}: {why}")

        if node.target not in (aten.mm.default, aten.mv.default):
            no("only F.linear's GEMMs have captured routes (cublas_bf16.TRAIN_CAPTURE); without a cc it is torch CPU's")
        mv = node.target is aten.mv.default
        if mv:
            (Nm, Km), Mm = _shape(ctx, node.args[0]), 1
            layouts = ("row", _FLIP.get(_layout(ctx.tensor_meta(node.args[0]))))
        else:
            layouts = tuple(_layout(ctx.tensor_meta(a)) for a in node.args[:2])
            (Mm, Km), Nm = _shape(ctx, node.args[0]), _shape(ctx, node.args[1])[1]
        if layouts not in _ROLES:
            no(f"operand layouts {layouts}, none of F.linear's GEMMs' "
               f"({', '.join(f'{r} {a} x {b}' for (a, b), r in _ROLES.items())})")
        role = _ROLES[layouts]
        M, K, N = _LINEAR_DIMS[role](Mm, Km, Nm)
        try:
            route = CB.train_route(role, M, K, N)
            fn = word_of(CB.route_gemm(route, *CB.TRAIN_LINEAR_ROLES[role][2](M, K, N)))
        except ValueError as e:
            no(f"F.linear's {role} GEMM at x ({M}, {K}), W ({N}, {K}): {e.args[0]}")
        if mv:
            a = ctx.coll(node.args[1]).reshape(Array(1, Array(Km, BF16)))
            w = ctx.coll(node.args[0]).reshape(Array(Nm, Array(Km, BF16)))
        else:
            a = ctx.coll(node.args[0]).reshape(Array(Mm, Array(Km, BF16)))
            w = _view_permute(ctx.coll(node.args[1]).reshape(Array(Km, Array(Nm, BF16))), (1, 0), node.name)
        out = _batch(ctx, node, fn, a, w, axes=(0, None))
        ctx.define(node, Coll(_tensor_type([Nm] if mv else [Mm, Nm], 16), out.refs))
        ctx.record(self, node, {"role": role, "M": M, "K": K, "N": N, "route": route}, emitted=f"{fn.id} x{Mm}")
