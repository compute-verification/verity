"""DeepSeek-V4-Pro's H200 observer folds (vLLM d9105ea80).

Each pattern states one of the Build's DeepSeek-V4 rules (`rules/vllm_mhc.py`, `rules/vllm_dsv4.py`, the fused q / kv norm of
`vllm_bindings/triton_launches.py`) over the observer log, and binds
its Definitions through the record target's vocabulary as the Build binds them (`target.vocab_kind`), so each Compute is the Build's
Call.
"""

from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import re
from dataclasses import dataclass

from verity_vllm.observe.events import Root, ViewDesc
from verity_vllm.observe.fold.patterns.collectives import ALL_REDUCE_OP
from verity_vllm.observe.fold.patterns.ops import as_float, base_op, bound_args, op_name, triton_name
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, unsupported
from verity_vllm.observe.fold.patterns.target import target_of, vocab_kind
from verity_vllm.observe.fold.resolver import Compute, Group, PerRow, Resolution, Shared, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.program.frontend.rules.vllm_bindings.pins import QKV_RMS_PINS
from verity_vllm.program.frontend.rules.vllm_bindings.triton_launches import QKV_ARGS, TRITON_DIVISIBILITY, qkv_rms_launch
from verity_vllm.program.frontend.rules.vllm_mhc import MHC_GEOMETRY, post_pre_path
from verity_vllm.program.kernels.mhc_twin import tf32_num_split
from verity_vllm.program.registry.dsa_norm import check_block

LAYER = re.compile(r"model\.layers\.(\d+)")


def _module(node: TreeNode) -> str:
    return getattr(node.enter, "module", None) or ""


def _roots(ctx: StepContext) -> dict[str, Root]:
    """The step's roots by name."""
    if "dsv4_roots" not in ctx.scratch:
        ctx.scratch["dsv4_roots"] = {r.name: r for rs in ctx.roots.values() for r in rs}
    return ctx.scratch["dsv4_roots"]


def _weights(node: TreeNode, ctx: StepContext, names: tuple[str, ...]) -> tuple[Shared, ...] | Unsupported:
    rs = _roots(ctx)
    if (missing := [n for n in names if n not in rs]):
        return unsupported(node, "the capture registered no root for weights the call reads", missing=missing)
    return tuple(Shared(rs[n].view) for n in names)


def _is(v: ViewDesc, shape: tuple[int, ...], dtype: str) -> bool:
    return v.shape == shape and v.dtype == dtype


@dataclass
class Mhc:
    """The mHC (`rules/vllm_mhc.py`).  vLLM calls the TileLang kernels' Python wrappers directly, so no kernel event reaches the log;
    what does is each wrapper's allocations under the layer's module (the model's for the final call), in its order:

        layer 0's pre_broadcast   streams [T,HC,H], post [T,HC], comb [T,HC*HC], layer input [T,H], DeepGEMM's [NS,T,OUT] and [NS,T]
        fused post_pre            the splits' [NS,T,OUT] and [NS,T], then streams, post, comb, layer input
        the final mhc_post        streams, then hc_head's [T,H]

    A call's last allocation resolves to its Computes, which write its outputs there.  Its inputs are the previous call's outputs and
    the sublayer's output, the last all-reduce's received storage: the pattern runs ahead of `TpCollective`, records it and declines.
    Layer L's first call is its attention's pre (`hc_attn_*`, `attn_norm`), its second the FFN's, each reading the sublayer's weights by
    name.  NS is the dispatch's at the step's tokens and must be the leading extent of the split buffer the wrapper allocated."""

    name = "dsv4.mhc"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return None
        st = ctx.scratch.setdefault("dsv4_mhc", {"x": None, "pending": [], "calls": {}, "prev": None})
        if base_op(node.op) == ALL_REDUCE_OP:
            outs = node.output_views()
            st["x"] = outs[0] if len(outs) == 1 else None
            return None
        mod = _module(node)
        if op_name(node.op) not in ("empty", "empty_like") or not (mod == "model" or LAYER.fullmatch(mod)):
            return None
        outs = node.output_views()
        pend = st["pending"]
        if pend and pend[0][0] != mod:
            pend.clear()
        pend.append((mod, outs[0] if len(outs) == 1 else None))
        if len(pend) < (2 if mod == "model" else 6):
            return None
        views = [v for _, v in pend]
        pend.clear()
        if not all(isinstance(v, ViewDesc) for v in views):
            return unsupported(node, "an mHC allocation has no single output view")
        C = ctx.config
        HC, H, T = int(C["HC_MULT"]), int(C["H"]), len(ctx.start.rows)
        g = MHC_GEOMETRY.get(H)
        if g is None:
            return unsupported(node, "no pinned mHC thread geometry at the hidden size", H=H, pinned=sorted(MHC_GEOMETRY))
        x = st["x"]
        if not isinstance(x, ViewDesc) or not _is(x, (T, H), "bfloat16"):
            return unsupported(node, "the sublayer's output is not the last all-reduce's bf16 [T,H]", x=repr(x), T=T, H=H)
        S = {"HC": HC, "H": H, "TN": g["TN"], "REPEAT": int(C["HC_REPEAT"]), "EPS": float(C["EPS"]), "HC_EPS": float(C["HC_EPS"])}
        rows = tuple(ctx.start.rows)
        sms = int(target_of(ctx)[0].num_sms)
        if mod == "model":
            return self._final(node, ctx, st, views, x, S, g, rows)
        L = int(LAYER.fullmatch(mod).group(1))
        k = st["calls"][mod] = st["calls"].get(mod, 0) + 1
        if k > 2:
            return unsupported(node, "a third mHC call in one layer", module=mod)
        side = ("attn", "ffn")[k - 1]
        names = (f"{mod}.hc_{side}_fn" + ("_broadcast" if L == 0 and k == 1 else ""), f"{mod}.hc_{side}_scale", f"{mod}.hc_{side}_base",
                 f"{mod}.{side}_norm.weight")
        ws = _weights(node, ctx, names)
        if isinstance(ws, Unsupported):
            return ws
        OUT = HC * (2 + HC)
        if L == 0 and k == 1:
            streams, post, comb, inp, ys, rs = views
            NS = tf32_num_split(T, H, sms)
            parts = [Compute(vocab_kind(ctx, "mhc_pre_broadcast", NS=NS, **S), T, (PerRow(x), *ws), (PerRow(inp), PerRow(post), PerRow(comb)),
                             (node.id,), hint=f"{mod}/pre", attribution=rows, pattern=self.name, roots=(None, *names)),
                     Compute(vocab_kind(ctx, "mhc_residual_broadcast", HC=HC, H=H), T, (PerRow(x),), (PerRow(streams),), (),
                             hint=f"{mod}/streams", attribution=rows, pattern=self.name)]
        else:
            if st["prev"] is None:
                return unsupported(node, "a fused post_pre with no earlier mHC call in the step", module=mod)
            ys, rs, streams, post, comb, inp = views
            role, NS = post_pre_path(T, HC, H, sms)
            res, post0, comb0 = st["prev"]
            spec = vocab_kind(ctx, role, NS=NS, **S, **({"TH": g["TH"]} if role == "mhc_fused_post_pre" else {}))
            parts = [Compute(spec, T, (PerRow(x), PerRow(res), PerRow(post0), PerRow(comb0), *ws),
                             (PerRow(streams), PerRow(inp), PerRow(post), PerRow(comb)), (node.id,), hint=f"{mod}/{side}.post_pre",
                             attribution=rows, pattern=self.name, roots=(None, None, None, None, *names))]
        want = ((streams, (T, HC, H), "bfloat16"), (post, (T, HC), "float32"), (comb, (T, HC * HC), "float32"), (inp, (T, H), "bfloat16"),
                (ys, (NS, T, OUT), "float32"), (rs, (NS, T), "float32"))
        if (bad := [repr(v) for v, s, d in want if not _is(v, s, d)]):
            return unsupported(node, "the mHC call's allocations differ from its outputs and NS splits", NS=NS, T=T, views=bad)
        st["prev"] = (streams, post, comb)
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=mod, pattern=self.name)

    def _final(self, node: TreeNode, ctx: StepContext, st: dict, views: list, x: ViewDesc, S: dict, g: dict, rows: tuple) -> Resolution:
        streams, head = views
        HC, H = S["HC"], S["H"]
        if st["prev"] is None or not _is(streams, (len(rows), HC, H), "bfloat16") or not _is(head, (len(rows), H), "bfloat16"):
            return unsupported(node, "the final mHC call's allocations are not streams [T,HC,H] and the head's [T,H] after the layers",
                               views=[repr(v) for v in views])
        names = ("model.hc_head_fn", "model.hc_head_scale", "model.hc_head_base")
        ws = _weights(node, ctx, names)
        if isinstance(ws, Unsupported):
            return ws
        res, post, comb = st["prev"]
        head_spec = vocab_kind(ctx, "hc_head", HC=HC, H=H, TH=g["TH_HEAD"], EPS=S["EPS"], HC_EPS=S["HC_EPS"])
        parts = (Compute(vocab_kind(ctx, "mhc_post", HC=HC, N=H), len(rows), (PerRow(x), PerRow(res), PerRow(post), PerRow(comb)),
                         (PerRow(streams),), (node.id,), hint="model/post", attribution=rows, pattern=self.name),
                 Compute(head_spec, len(rows), (PerRow(streams), *ws), (PerRow(head),), (), hint="model/hc_head", attribution=rows,
                         pattern=self.name, roots=(None, *names)))
        st["prev"] = None
        return Group(parts, tuple(sorted(node.ids())), hint="model", pattern=self.name)


@dataclass
class QKvRmsNorm:
    """`_fused_q_kv_rmsnorm_kernel[(T, 2)]` (`fused_q_kv_rmsnorm`, the attention's q and kv norms) -> per token `TritonBlockRMSNorm{N=Q_SIZE}`
    of the q row and `{N=KV_SIZE}` of the kv row over their weights: the Build's `FusedQKvRmsNormRule`, with its pins (the source, the
    wrapper's BLOCK_SIZE and num_warps, packed outputs apart from the inputs, and Triton's 16-divisible specialization the PTX is)."""

    name = "dsv4.fused_q_kv_rmsnorm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or node.enter.fn != QKV_RMS_PINS["kernel"]:
            return None
        if node.enter.src_sha != QKV_RMS_PINS["src_sha256"]:
            return unsupported(node, "the kernel's source differs from the pin", src_sha=node.enter.src_sha, pinned=QKV_RMS_PINS["src_sha256"])
        a = bound_args(node)
        sizes, eps = [a.get(k[-1]) for k in QKV_ARGS], as_float(a.get("eps"))
        if not all(isinstance(n, int) and n > 0 for n in sizes) or eps is None:
            return unsupported(node, "Q_SIZE, KV_SIZE and eps are not static positive sizes and a float", sizes=sizes, eps=repr(a.get("eps")))
        block, warps = qkv_rms_launch(sizes)
        if a.get("BLOCK_SIZE") != block or node.enter.num_warps != warps:
            return unsupported(node, "BLOCK_SIZE and num_warps are not the wrapper's", BLOCK_SIZE=a.get("BLOCK_SIZE"), num_warps=node.enter.num_warps,
                               wrapper=(block, warps))
        try:
            for N in sizes:
                check_block(N, block, warps, eps)
        except ValueError as e:
            return unsupported(node, str(e))
        mod, parts, outs = _module(node), [], set()
        for side, x_arg, out_arg, w_arg, in_stride, out_stride, size_key in QKV_ARGS:
            x, out, w, N = a.get(x_arg), a.get(out_arg), a.get(w_arg), a[size_key]
            if not all(isinstance(v, ViewDesc) and v.dtype == "bfloat16" for v in (x, out, w)):
                return unsupported(node, f"the {side} arguments are not bf16 views", x=repr(x), out=repr(out), w=repr(w))
            M = x.shape[0]
            if x.shape != (M, N) or x.strides != (a.get(in_stride), 1) or out.shape != (M, N) or out.strides != (N, 1) or a.get(out_stride) != N \
                    or w.shape != (N,) or w.strides != (1,) or node.enter.grid != [M, 2]:
                return unsupported(node, f"the {side} row is not [T,{size_key}] at the launch's stride into a packed output over a [{size_key}] weight on a (T, 2) grid",
                                   x=repr(x), out=repr(out), w=repr(w), grid=node.enter.grid)
            if x.strides[0] % TRITON_DIVISIBILITY or N % TRITON_DIVISIBILITY or any(2 * v.offset % TRITON_DIVISIBILITY for v in (x, out, w)):
                return unsupported(node, f"the {side} row strides or a pointer are not {TRITON_DIVISIBILITY}-divisible: another Triton specialization than the PTX",
                                   x=repr(x), out=repr(out), w=repr(w))
            if out.alloc in {x.alloc, w.alloc} or out.alloc in outs:
                return unsupported(node, "q_out and kv_out are not two allocations apart from the inputs", out=repr(out))
            outs.add(out.alloc)
            roots = ctx.roots_of(w)
            if not roots:
                return unsupported(node, f"the {side} norm weight is no root of the capture", w=repr(w))
            parts.append(Compute(vocab_kind(ctx, "triton_block_rmsnorm", N=N, BLOCK=block, WARPS=warps, EPS=eps), M, (PerRow(x), Shared(w)),
                                 (PerRow(out),), () if parts else (node.id,), hint=f"{mod}/{side}_norm", kernels=(triton_name(node),),
                                 attribution=attribution_for(ctx, M, node.phase), pattern=self.name, roots=(None, roots[0].name)))
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=mod, pattern=self.name)


def dsv4_patterns() -> list:
    """The DeepSeek-V4 patterns, ahead of the rank's collectives (`Mhc` records the all-reduces they resolve)."""
    return [Mhc(), QKvRmsNorm()]


__all__ = ["Mhc", "QKvRmsNorm", "dsv4_patterns"]
