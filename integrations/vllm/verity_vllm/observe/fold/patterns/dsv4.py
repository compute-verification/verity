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
from verity_vllm.observe.fold.patterns.geometry import GeometryError, is_int, sub_rows
from verity_vllm.observe.fold.patterns.kv_cache import kv_group_of
from verity_vllm.observe.fold.patterns.ops import as_float, base_op, bound_args, op_name, triton_name
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, unsupported
from verity_vllm.observe.fold.patterns.rotary import step_positions
from verity_vllm.observe.fold.patterns.target import target_of, vocab_kind
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.program.frontend.dsv4_leaves import FP8_DS_MLA_QB
from verity_vllm.program.frontend.rules.vllm_bindings.pins import QKV_RMS_PINS
from verity_vllm.program.frontend.rules.vllm_bindings.triton_launches import QKV_ARGS, TRITON_DIVISIBILITY, qkv_rms_launch
from verity_vllm.program.frontend.rules.vllm_mhc import MHC_GEOMETRY, post_pre_path
from verity_vllm.program.kernels.mhc_twin import tf32_num_split
from verity_vllm.program.registry import dsv4_qnorm_insert as Q
from verity_vllm.program.registry.dsa_norm import check_block

LAYER = re.compile(r"model\.layers\.(\d+)")
QNORM_INSERT_OP = "_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
SAVE_STATES_KERNEL = "vllm.models.deepseek_v4.common.ops.save_partial_states._save_partial_states_kernel"
SAVE_STATES_SRC_SHA = "11230c3682661cfe7eb4f04edc099b0e19fb3efba7ac4e0c5738bcc49d9d19e8"


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


def _slots(node: TreeNode, ctx: StepContext, layer: str, bs: object, sm: ViewDesc, T: int) -> tuple[int, ...] | Unsupported:
    """The T tokens' slots in `layer`'s KV group, once the group's blocks are the launch's `bs` and the slot buffer covers the tokens."""
    try:
        g = kv_group_of(ctx, layer)
    except GeometryError as e:
        return unsupported(node, str(e))
    slots = tuple(int(s) for s in ((g or {}).get("slot_mapping") or ())[:T])
    if g is None or int(g["block_size"]) != bs or len(slots) != T or not is_int(sm) or sm.numel < T or any(s < 0 for s in slots):
        return unsupported(node, "the cache's KV group does not give every token a slot in blocks of the launch's size", layer=layer,
                           block_size=bs, slots=slots[:8])
    return slots


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


def swa_block(cache: ViewDesc, block: int, bs: int, DN: int, DR: int, QB: int) -> tuple[ViewDesc, ViewDesc, ViewDesc]:
    """Block `block` of a layer's window cache (uint8 [blocks, block bytes]) as its bs tokens' NoPE [bs, DN] e4m3, scale bytes
    [bs, DN/QB + 1] and RoPE [bs, DR] bf16: the tokens' DN + 2 DR data bytes each, then their scale bytes each
    (`cache_utils.quantize_and_insert_k_cache`)."""
    row, ns = DN + 2 * DR, DN // QB + 1
    if cache.dtype != "uint8" or cache.rank != 2 or cache.strides[1] != 1 or cache.shape[1] != bs * (row + ns) or not 0 <= block < cache.shape[0]:
        raise GeometryError(f"block {block} of a window cache of {bs} rows of {row} + {ns} bytes: {cache!r}")
    base = cache.offset + block * cache.strides[0]
    if (base + DN) % 2:
        raise GeometryError(f"block {block}'s RoPE bytes are not bf16-aligned: {cache!r}")
    return (ViewDesc(cache.alloc, base, (bs, DN), (row, 1), "float8_e4m3fn", cache.device),
            ViewDesc(cache.alloc, base + bs * row, (bs, ns), (ns, 1), "uint8", cache.device),
            ViewDesc(cache.alloc, (base + DN) // 2, (bs, DR), (row // 2, 1), "bfloat16", cache.device))


def block_runs(slots: tuple[int, ...], bs: int) -> list[tuple[int, int, int]]:
    """(block, first row, end row) of each run of consecutive rows whose slots are in one block."""
    runs: list[list[int]] = []
    for i, s in enumerate(slots):
        if runs and runs[-1][0] == s // bs:
            runs[-1][2] = i + 1
        else:
            runs.append([s // bs, i, i + 1])
    return [(b, lo, hi) for b, lo, hi in runs]


@dataclass
class QNormRopeKvInsert:
    """`_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q, kv, k_cache!, slot_mapping, positions, cos_sin, q_head_padded, eps,
    cache_block_size)` -> per token `Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}` of q[t], kv[t] and cos_sin[positions[t]]: the
    Build's `Dsv4QNormRopeKvInsertRule`, QB fp8_ds_mla's (`dsv4_leaves.FP8_DS_MLA_QB`).  The padded q is the op's output; the kv
    row's NoPE, scale bytes and RoPE go to the token's slot of the layer's window cache, addressed by `<layer>.swa_cache`'s KV group,
    so the tokens of one cache block are one Compute (`swa_block`)."""

    name = "dsv4.qnorm_rope_kv_insert"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != QNORM_INSERT_OP:
            return None
        a, outs = bound_args(node), node.output_views()
        q, kv, cache, sm, cs = (a.get(k) for k in ("q_in", "kv", "k_cache", "slot_mapping", "cos_sin_cache"))
        NHP, bs, eps = a.get("q_head_padded"), a.get("cache_block_size"), as_float(a.get("eps"))
        if not all(isinstance(v, ViewDesc) for v in (q, kv, cache, sm, cs)) or len(outs) != 1 or not isinstance(NHP, int) \
                or not isinstance(bs, int) or eps is None:
            return unsupported(node, "the insert's arguments are not bound as (q, kv, cache, slots, positions, cos_sin, padded heads, eps, block size)")
        qo, rows = outs[0], tuple(ctx.start.rows)
        T = len(rows)
        if q.rank != 3 or q.shape[0] != T or cs.rank != 2:
            return unsupported(node, "q is not [T,NH,D] over the step's tokens or cos_sin is not a table", q=repr(q), cs=repr(cs), T=T)
        _, NH, D = q.shape
        DR, QB = cs.shape[1], FP8_DS_MLA_QB
        S = {"NH": NH, "NHP": NHP, "DN": D - DR, "DR": DR, "QB": QB, "EPS": eps}
        try:
            Q.check_insert_statics(NH, NHP)
            Q.check_q_statics(D, DR)
            Q.check_kv_statics(D - DR, DR, QB)
        except ValueError as e:
            return unsupported(node, str(e))
        if not (_is(q, (T, NH, D), "bfloat16") and _is(kv, (T, D), "bfloat16") and _is(qo, (T, NHP, D), "bfloat16") and cs.dtype == "float32") \
                or not all(v.strides[-1] == 1 for v in (q, kv, qo, cs)):
            return unsupported(node, "q, kv and the padded q are not bf16 [T,NH,D], [T,D], [T,NHP,D] rows, or cos_sin not an f32 table",
                               q=repr(q), kv=repr(kv), out=repr(qo), cs=repr(cs))
        pos = step_positions(node, ctx, a.get("position_ids"), T)
        if isinstance(pos, Unsupported):
            return pos
        slots = _slots(node, ctx, f"{_module(node)}.swa_cache", bs, sm, T)
        if isinstance(slots, Unsupported):
            return slots
        if any(p < 0 or p >= cs.shape[0] for p in pos):
            return unsupported(node, "a position is outside the rotary table", positions=pos[:8], table_rows=cs.shape[0])
        croots = ctx.roots_of(cs)
        if not croots:
            return unsupported(node, "the rotary table is no root of the capture", cs=repr(cs))
        spec, parts = vocab_kind(ctx, "dsv4_qnorm_rope_kv_insert", **S), []
        for b, lo, hi in block_runs(slots, bs):
            try:
                nope, scale, rope = swa_block(cache, b, bs, D - DR, DR, QB)
            except GeometryError as e:
                return unsupported(node, str(e))
            idx = tuple(s % bs for s in slots[lo:hi])
            parts.append(Compute(spec, hi - lo, (PerRow(sub_rows(q, lo, hi)), PerRow(sub_rows(kv, lo, hi)), Gather(cs, pos[lo:hi])),
                                 (PerRow(sub_rows(qo, lo, hi)), Gather(nope, idx), Gather(scale, idx), Gather(rope, idx)),
                                 () if parts else (node.id,), hint=f"{_module(node)}/qnorm_rope_kv_insert", attribution=rows[lo:hi],
                                 pattern=self.name, roots=(None, None, croots[0].name)))
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


def state_block(cache: ViewDesc, block: int, W: int) -> tuple[ViewDesc, ViewDesc]:
    """Block `block` of a compressor's state cache (f32 [blocks, bs, 2W]) as its bs slots' kv words [bs, W] and score words [bs, W]."""
    if cache.dtype != "float32" or cache.rank != 3 or cache.shape[2] != 2 * W or cache.strides[2] != 1 or not 0 <= block < cache.shape[0]:
        raise GeometryError(f"block {block} of a state cache of rows of 2 x {W} f32 words: {cache!r}")
    base, bs = cache.offset + block * cache.strides[0], cache.shape[1]
    return tuple(ViewDesc(cache.alloc, base + k * W, (bs, W), (cache.strides[1], 1), "float32", cache.device) for k in (0, 1))


@dataclass
class SavePartialStates:
    """`_save_partial_states_kernel[(T,)]` (`save_partial_states`, a compressor's state rows) -> per token kv[t] moved to the first W
    words of its slot of the state cache, and `Dsv4CompressorScore_v1{W}` of score[t] and ape[positions[t] % R] in the next W: the
    Build's `Dsv4CompressorStateRule`.  The slot is the token's in `<compressor>.state_cache`'s KV group, so the tokens of one cache
    block are one move and one Compute (`state_block`).  A ratio-4 layer's indexer runs the kernel on its own compressor's state, a
    cache only the indexer's long path reads."""

    name = "dsv4.save_partial_states"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or node.enter.fn != SAVE_STATES_KERNEL:
            return None
        if node.enter.src_sha != SAVE_STATES_SRC_SHA or node.enter.num_warps != 4:
            return unsupported(node, "the kernel's source or warps differ from the pin", src_sha=node.enter.src_sha, num_warps=node.enter.num_warps)
        a = bound_args(node)
        kv, score, ape, cache, sm = (a.get(k) for k in ("kv_ptr", "score_ptr", "ape_ptr", "state_cache_ptr", "slot_mapping_ptr"))
        W, R, bs = a.get("HEAD_SIZE"), a.get("COMPRESS_RATIO"), a.get("block_size")
        rows = tuple(ctx.start.rows)
        T = len(rows)
        if not all(isinstance(v, ViewDesc) for v in (kv, score, ape, cache, sm)) or not all(isinstance(n, int) and n > 0 for n in (W, R, bs)):
            return unsupported(node, "the arguments are not bound as (kv, score, ape, positions, state cache, slots, block size) with static sizes")
        if a.get("STATE_WIDTH") != W or a.get("TRITON_BLOCK_SIZE") != W or node.enter.grid != [T]:
            return unsupported(node, "the launch is not one program per token over state rows of two HEAD_SIZE halves", grid=node.enter.grid,
                               T=T, constexprs=node.enter.constexprs)
        if not (_is(kv, (T, W), "float32") and _is(score, (T, W), "float32") and _is(ape, (R, W), "float32")) \
                or any(v.strides[-1] != 1 for v in (kv, score, ape)) \
                or (a.get("kv_stride"), a.get("score_stride"), a.get("ape_stride")) != (kv.strides[0], score.strides[0], ape.strides[0]):
            return unsupported(node, "kv and score are not f32 [T,W] rows and ape an f32 [R,W] table at the launch's strides",
                               kv=repr(kv), score=repr(score), ape=repr(ape))
        if cache.rank != 3 or cache.shape[1] != bs or (a.get("state_cache_stride0"), a.get("state_cache_stride1")) != cache.strides[:2]:
            return unsupported(node, "the state cache is not [blocks, block size, .] at the launch's strides", cache=repr(cache), block_size=bs)
        pos = step_positions(node, ctx, a.get("positions_ptr"), T)
        if isinstance(pos, Unsupported):
            return pos
        slots = _slots(node, ctx, f"{_module(node)}.state_cache", bs, sm, T)
        if isinstance(slots, Unsupported):
            return slots
        aroots = ctx.roots_of(ape)
        if not aroots:
            return unsupported(node, "ape is no root of the capture", ape=repr(ape))
        spec, hint, parts = vocab_kind(ctx, "dsv4_compressor_score", W=W), f"{_module(node)}/state", []
        for b, lo, hi in block_runs(slots, bs):
            try:
                kv_rows, score_rows = state_block(cache, b, W)
            except GeometryError as e:
                return unsupported(node, str(e))
            idx = tuple(s % bs for s in slots[lo:hi])
            parts += [Transparent("move", () if parts else (node.id,), src=PerRow(sub_rows(kv, lo, hi)), dst=Gather(kv_rows, idx), rows=hi - lo,
                                  hint=f"{hint}.kv", pattern=self.name),
                      Compute(spec, hi - lo, (PerRow(sub_rows(score, lo, hi)), Gather(ape, tuple(p % R for p in pos[lo:hi]))),
                              (Gather(score_rows, idx),), (), hint=f"{hint}.score", kernels=(triton_name(node),), attribution=rows[lo:hi],
                              pattern=self.name, roots=(None, aroots[0].name))]
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


def dsv4_patterns() -> list:
    """The DeepSeek-V4 patterns, ahead of the rank's collectives (`Mhc` records the all-reduces they resolve)."""
    return [Mhc(), QKvRmsNorm(), QNormRopeKvInsert(), SavePartialStates()]


__all__ = ["Mhc", "QKvRmsNorm", "QNormRopeKvInsert", "SavePartialStates", "block_runs", "dsv4_patterns", "state_block", "swa_block"]
