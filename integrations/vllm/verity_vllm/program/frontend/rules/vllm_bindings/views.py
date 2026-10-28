"""Views of allocations and the token-id widening."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.value_types import shape_of, width_of
from verity_vllm.program.frontend.rules.base import RuleInfo, ViewRule
from verity_vllm.program.frontend.rules.vllm_bindings.operands import aten


# -- views of allocations ---------------------------------------------------------------------------------------------------


class AllocViewRule(ViewRule):
    """`aten.view/reshape(alloc, shape)` of an UNWRITTEN allocation (vLLM's `Attention.forward`: `output = torch.empty(...)` then
    `output.view(-1, NH, D)` before the kernel writes it): the view is the same uninitialised allocation with the viewed type;
    the in-place rule that writes the view rebinds it. Reading either before the write stays refused (Alloc semantics).

    The in-place rule rebinds only the view it wrote, so the base stays an Alloc; a later view of the base over exactly that
    region (vLLM's `DeepseekV4Attention.forward`: the kernel writes `o_padded[:, :NH]`, the layer re-slices `o_padded[:, :NH]`)
    is the written value. Only when that one write went through the base: with two, either may have overwritten the other."""
    info = RuleInfo("AllocView", 3, "view", applies_to="aten.view / reshape / flatten / unit-step slice / permute / t / transpose of an unwritten allocation",
                    preconditions=("operand is an unwritten Alloc", "view/reshape/flatten: leaf count is preserved; slice: step 1 (a sub-allocation "
                                   "of the same storage -- vLLM's MoE workspace carving `buf[:n].view(...)`); permute/t/transpose: the viewed "
                                   "type is the node's (an axis order over the same leaves -- vLLM's column-major FP8 group scales "
                                   "`torch.empty(K/G, M).permute(-1, -2)`)",
                                   "the written value: exactly one in-place write went through the operand's storage, its view's storage offset, "
                                   "shape, stride and dtype are the node's, and it is not stale"), evidence_domain="structural (no gates)")

    def targets(self):
        # v2: + permute / t / transpose (the in-place rule that writes the view rebinds the VIEW node; the base stays an unread Alloc)
        return {aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints, aten.slice.Tensor,
                aten.permute.default, aten.t.default, aten.transpose.int}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Alloc):
            return f"operand is {type(v).__name__}, not an allocation"
        if node.target is aten.slice.Tensor and (len(node.args) > 4 and node.args[4] != 1):
            return f"slice step {node.args[4]} is not 1"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend.value_types import type_of_tensor
        written = _written_region(node, ctx)
        if written is None:
            ctx.define(node, Alloc(type_of_tensor(node.meta["val"]), node), view_of=node.args[0])
            return
        ctx.define(node, ctx.env[written], view_of=written)
        ctx.record(self, node, {"written": written.name, "region": _region(node)}, emitted="(the written value; no gates)")


def _region(node):
    t = node.meta.get("val")
    return None if t is None else [int(t.storage_offset()), [int(s) for s in t.shape], [int(s) for s in t.stride()], str(t.dtype)]


def _written_region(node, ctx):
    """The node an in-place rule wrote through `node`'s storage if it is the only such write and covers exactly `node`'s region."""
    base = ctx.storage_base(node.args[0])
    writes = [w for w, b in ctx.written_allocs.items() if b is base]
    if len(writes) != 1:
        return None
    w = writes[0]
    if w in ctx.stale or not isinstance(ctx.env.get(w), Coll) or _region(node) is None or _region(w) != _region(node):
        return None
    return w


# -- token-id widening ------------------------------------------------------------------------------------------------------


class IntIdWidenRule(ViewRule):
    """`aten.to.dtype(ids: int32 value, int64)` / `_to_copy(..., dtype=int64)`: the I32 token id widened for `aten.embedding`
    (vLLM's `VocabParallelEmbedding.forward`). Value-preserving for ids in [0, 2^31): a reinterpretation, no gate; the value keeps
    its 32-bit leaves (the Embedding kind takes I32)."""
    info = RuleInfo("IntIdWiden", 1, "view", applies_to="aten.to.dtype / _to_copy from a 32-bit integer value to int64",
                    preconditions=("operand is a value collection with 32-bit integer leaves", "target dtype is int64"),
                    evidence_domain="structural: widening of a non-negative index is a reinterpretation")

    def targets(self):
        return {aten.to.dtype, aten._to_copy.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        src, dst = ctx.dtype_of(node.args[0]), ctx.dtype_of(node)
        if src is None or dst is None or src.is_floating_point or dst.is_floating_point:
            return "not an integer -> integer cast"
        if dst != torch.int64 or width_of(src) != 32 or shape_of(v.type)[1] != 32:
            return f"{src} -> {dst} is not a 32-bit integer widened to int64"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.coll(node.args[0]), view_of=node.args[0])
        ctx.record(self, node, {"from": str(ctx.dtype_of(node.args[0])), "to": str(ctx.dtype_of(node)), "carried_as": "I32"}, emitted="(no gates)")
