"""The token embedding pattern."""

from __future__ import annotations

from dataclasses import dataclass

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_int, is_row_major, sub_rows
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args
from verity_vllm.observe.fold.patterns.resolutions import _fmt, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.transparent import SCRATCH_INT_COPIES
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode


@dataclass
class Embedding:
    """`aten.embedding(weight[V,H], indices[M])` → `Compute(Embedding_v1{V,H}, rows=M, (PerRow(input_ids), Shared(table)),
    (PerRow(out),))`.  The token operand is the step's `input_ids` view (whose rows the fold fills from prompt Roots /
    StepLinks at StepStart) — either the indices view itself or, when vLLM converted it (`input_ids.long()`), the
    integer copy's source recorded by `classify_transparent`."""

    name: str = "embedding.aten_embedding"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.embedding":
            return None
        a = bound_args(node)
        w, idx = a.get("weight", a.get("arg0")), a.get("indices", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(w, ViewDesc) or not isinstance(idx, ViewDesc) or not outs:
            return unsupported(node, "embedding arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if w.rank != 2 or not is_row_major(w) or w.dtype != "bfloat16":
            return unsupported(node, "embedding table is not a row-major bf16 [V,H] view", weight=repr(w))
        if not is_int(idx) or idx.rank != 1:
            return unsupported(node, "embedding indices are not an integer [M] view", indices=repr(idx))
        V, H = w.shape
        M = idx.shape[0]
        out = outs[0]
        if out.shape != (M, H) or not is_row_major(out) or out.dtype != "bfloat16":
            return unsupported(node, "embedding output is not a row-major bf16 [M,H] view", out=repr(out))
        tok = self._token_view(idx, ctx)
        if isinstance(tok, str):
            return unsupported(node, tok, indices=repr(idx))
        roots = ctx.roots_of(w)
        return Compute(bind(b1.Embedding, V=V, H=H), M, (PerRow(tok), Shared(w)), (PerRow(out),), (node.id,),
                       hint=hint_of(node, roots[0].name if roots else "embedding"), kernels=("indexSelectLargeIndex", "indexSelectSmallIndex", "vectorized_gather_kernel"),  # torch 2.13 index_select fast path [L5 census]
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)

    @staticmethod
    def _token_view(idx: ViewDesc, ctx: StepContext) -> ViewDesc | str:
        iv = ctx.start.input_ids_view
        if iv is None:
            return idx
        if idx.alloc == iv.alloc:
            return idx
        copies = ctx.scratch.get(SCRATCH_INT_COPIES, {})
        hit = copies.get(idx.alloc)
        if hit is not None:
            src, dst = hit
            if src.alloc == iv.alloc:
                if src.shape == idx.shape:
                    return src
                if src.rank == 1 and src.shape[0] >= idx.shape[0]:
                    return sub_rows(src, 0, idx.shape[0])
        return "embedding indices are neither the step's input_ids view nor an integer copy of it"
