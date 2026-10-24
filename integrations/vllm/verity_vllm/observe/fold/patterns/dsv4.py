"""DeepSeek-V4-Pro's H200 observer folds (vLLM d9105ea80).

Each pattern states one of the Build's DeepSeek-V4 rules (`rules/vllm_mhc.py`, `rules/vllm_dsv4.py`, the fused q / kv norm of
`vllm_bindings/triton_launches.py`) over the observer log, and binds
its Definitions through the record target's vocabulary as the Build binds them (`target.vocab_kind`), so each Compute is the Build's
Call.  FlashMLA's sparse attention is `dsv4_flashmla`'s; the caches both read are `dsv4_cache`'s.
"""

from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import re
from dataclasses import dataclass

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.collectives import ALL_REDUCE_OP
from verity_vllm.observe.fold.patterns.cublas import CublasLinear, CublasMmF32Out
from verity_vllm.observe.fold.patterns.dsv4_cache import _is, _module, _slots, _weights, block_runs, state_block, swa_block
from verity_vllm.observe.fold.patterns.dsv4_flashmla import SparseDecode, SparsePrefill
from verity_vllm.observe.fold.patterns.fp8 import DynamicDeepGemmFp8Linear
from verity_vllm.observe.fold.patterns.geometry import GeometryError, sub_rows
from verity_vllm.observe.fold.patterns.norms import CudaRmsNorm
from verity_vllm.observe.fold.patterns.ops import as_float, base_op, bound_args, op_name, triton_name
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, unsupported
from verity_vllm.observe.fold.patterns.rotary import step_positions
from verity_vllm.observe.fold.patterns.target import target_of, vocab_kind
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.program.frontend.dsv4_leaves import FP8_DS_MLA_QB
from verity_vllm.program.frontend.rules.vllm_bindings.fp8 import FP8_QUANT_PINS
from verity_vllm.program.frontend.rules.vllm_bindings.pins import QKV_RMS_PINS
from verity_vllm.program.frontend.rules.vllm_bindings.triton_launches import QKV_ARGS, TRITON_DIVISIBILITY, qkv_rms_launch
from verity_vllm.program.frontend.rules.vllm_mhc import MHC_GEOMETRY, post_pre_path
from verity.ml.definitions.numerics.mhc_twin import tf32_num_split
from verity.ml.definitions.registry import dsv4_compressor as C
from verity.ml.definitions.registry import dsv4_indexer as I
from verity.ml.definitions.registry import dsv4_qnorm_insert as Q
from verity.ml.definitions.registry.dsa_norm import check_block
from verity.ml.definitions.registry.fp8 import weight_blocks

LAYER = re.compile(r"model\.layers\.(\d+)")
QNORM_INSERT_OP = "_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
#: Triton kernels by name and source text (`TritonLaunch.src_sha`, the sha256 of the kernel's source)
SAVE_STATES_KERNEL = "_save_partial_states_kernel"
SAVE_STATES_SRC_SHA = "11230c3682661cfe7eb4f04edc099b0e19fb3efba7ac4e0c5738bcc49d9d19e8"
INV_ROPE_OP = "vllm.fused_inv_rope_fp8_quant_kernel"
INV_ROPE_KERNEL = "_fused_inv_rope_fp8_quant_per_head"
INV_ROPE_SRC_SHA = "25e6ced0c59d2464bfa83a706c99bfe09d5b00f31e1ab31a600797d0cc8cd5cc"
#: Dsv4InvRopeQuantHead_v1's clamp and amax floor
INV_ROPE_FP8_MAX, INV_ROPE_EPS = 448.0, 1e-10
INDEXER_COMPRESS_KERNEL = "_fused_kv_compress_norm_rope_insert_indexer_attn"
INDEXER_COMPRESS_SRC_SHA = "9fa36754ffe4c4483c15a2f9bd19eee999f1473b5dd95aa467e94b615a0babef"
INDEXER_COMPRESSOR = ".indexer.compressor"



@dataclass
class Mhc:
    """The mHC (`rules/vllm_mhc.py`).  vLLM calls the TileLang kernels' Python wrappers directly, so no kernel event reaches the log;
    what does is each wrapper's allocations under the layer's module (the model's for the final call), in its order:

        layer 0's pre_broadcast   streams [T,HC,H], post [T,HC], comb [T,HC*HC], layer input [T,H], DeepGEMM's [NS,T,OUT] and [NS,T]
        fused post_pre            the splits' [NS,T,OUT] and [NS,T], then streams, post, comb, layer input
        the final mhc_post        streams, then hc_head's [T,H]

    A call's last allocation resolves to its Computes, which write its outputs there, except the final call's: mhc_post resolves at
    streams and hc_head at the head's, since the model drops the sublayer's output between them.  A call's inputs are the previous
    call's outputs and the sublayer's output, the last all-reduce's received storage: the pattern runs ahead of `TpCollective`, records
    it and declines.
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
        if len(pend) < (1 if mod == "model" else 6):
            return None
        views = [v for _, v in pend]
        if len(pend) == (2 if mod == "model" else 6):
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
        HC, H = S["HC"], S["H"]
        if len(views) == 1:
            streams = views[0]
            if st["prev"] is None or not _is(streams, (len(rows), HC, H), "bfloat16"):
                return unsupported(node, "the final mHC call's first allocation is not streams [T,HC,H] after the layers", streams=repr(streams))
            res, post, comb = st["prev"]
            st["prev"] = None
            return Compute(vocab_kind(ctx, "mhc_post", HC=HC, N=H), len(rows), (PerRow(x), PerRow(res), PerRow(post), PerRow(comb)),
                           (PerRow(streams),), tuple(sorted(node.ids())), hint="model/post", attribution=rows, pattern=self.name)
        streams, head = views
        if not _is(head, (len(rows), H), "bfloat16"):
            return unsupported(node, "the final mHC call's second allocation is not the head's [T,H]", head=repr(head))
        names = ("model.hc_head_fn", "model.hc_head_scale", "model.hc_head_base")
        ws = _weights(node, ctx, names)
        if isinstance(ws, Unsupported):
            return ws
        head_spec = vocab_kind(ctx, "hc_head", HC=HC, H=H, TH=g["TH_HEAD"], EPS=S["EPS"], HC_EPS=S["HC_EPS"])
        return Compute(head_spec, len(rows), (PerRow(streams), *ws), (PerRow(head),), tuple(sorted(node.ids())), hint="model/hc_head",
                       attribution=rows, pattern=self.name, roots=(None, *names))


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
        ctx.scratch.setdefault("dsv4_cos_sin", {})[_module(node)] = (cs, croots[0].name)
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



def indexer_short_path(node: TreeNode, pattern: str) -> Transparent:
    """A launch of a ratio-4 layer's indexer compressor (`<layer>.attn.indexer.compressor`): on the indexer's short path, every
    compressed row selected, nothing reads its state rows or key cache, and the Build's indexer leaf states nothing (`dsv4_leaves`), so
    the launch folds to nothing and leaves those caches unwritten.  The long path, which reads them, has no pattern, and the attention
    refuses more compressed rows than the top-k (`dsv4_flashmla.sparse_layer`)."""
    return Transparent("noop", tuple(sorted(node.ids())), hint=f"{_module(node)}: the indexer's short path", kernels=(triton_name(node),),
                       pattern=pattern)


@dataclass
class SavePartialStates:
    """`_save_partial_states_kernel[(T,)]` (`save_partial_states`, a compressor's state rows) -> per token kv[t] moved to the first W
    words of its slot of the state cache, and `Dsv4CompressorScore_v1{W}` of score[t] and ape[positions[t] % R] in the next W: the
    Build's `Dsv4CompressorStateRule`.  The slot is the token's in `<compressor>.state_cache`'s KV group, so the tokens of one cache
    block are one move and one Compute (`state_block`).  A ratio-4 layer's indexer runs the kernel on its own compressor's state
    (`indexer_short_path`).  A head-512 compressor's window ends are recorded for the layer's attention, which emits the compress the
    CuTe DSL runs next (`mla_compress`)."""

    name = "dsv4.save_partial_states"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if triton_name(node) != SAVE_STATES_KERNEL:
            return None
        if node.enter.src_sha != SAVE_STATES_SRC_SHA or node.enter.num_warps != 4:
            return unsupported(node, "the kernel's source or warps differ from the pin", src_sha=node.enter.src_sha, num_warps=node.enter.num_warps)
        if _module(node).endswith(INDEXER_COMPRESSOR):
            return indexer_short_path(node, self.name)
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
        if W // (2 if R == 4 else 1) == C.DSV4_COMPRESSOR["D"]:
            ctx.scratch.setdefault("dsv4_compress", {})[_module(node).removesuffix(".compressor")] = \
                (cache, W, R, tuple((rows[t], p) for t, p in enumerate(pos) if (p + 1) % R == 0))
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


@dataclass
class IndexerCompress:
    """`_fused_kv_compress_norm_rope_insert_indexer_attn[(T,)]`, a ratio-4 layer's indexer compressor's compress into the indexer's
    key cache (`registry.dsv4_indexer`'s `Dsv4IndexerCompress`), under its pin: the indexer's short path (`indexer_short_path`)."""

    name = "dsv4.indexer_compress"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if triton_name(node) != INDEXER_COMPRESS_KERNEL:
            return None
        e = node.enter
        if e.src_sha != INDEXER_COMPRESS_SRC_SHA or e.num_warps != 1 or not _module(node).endswith(INDEXER_COMPRESSOR):
            return unsupported(node, "the kernel's source, warps or module differ from the pin", src_sha=e.src_sha, num_warps=e.num_warps,
                               module=_module(node))
        return indexer_short_path(node, self.name)


@dataclass
class InvRopeQuant:
    """`vllm.fused_inv_rope_fp8_quant_kernel(o, positions, cos_sin, ...)`, the o-projection's input -> per token and head
    `Dsv4InvRopeQuantHead_v1{CHUNKS,QG,ROPE}` of o[t, h] and cos_sin[positions[t]]: the Build's `Dsv4InvRopeQuantRule`, on cc 9.0
    with f32 scales.  Head h = g * HPG + j writes its bytes at [g, t, j * HD : (j + 1) * HD] and its CHUNKS scales at
    [g, t, j * CHUNKS : (j + 1) * CHUNKS], one Compute per head over the step's tokens.  The scale buffer is the wrapper's
    [G, scale_inner, TMA-aligned T] read as [G, T, scale_inner]: the programs past T zero its alignment rows, which no view holds."""

    name = "dsv4.inv_rope_fp8_quant"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != INV_ROPE_OP:
            return None
        if len(node.children) != 1 or triton_name(node.children[0]) != INV_ROPE_KERNEL:
            return unsupported(node, "the op launched other than its one per-head kernel", children=[c.op for c in node.children])
        kid = node.children[0]
        if kid.enter.src_sha != INV_ROPE_SRC_SHA or kid.enter.num_warps != 1:
            return unsupported(node, "the kernel's source or warps differ from the pin", src_sha=kid.enter.src_sha, num_warps=kid.enter.num_warps)
        a, ka, outs = bound_args(node), bound_args(kid), node.output_views()
        sizes = ("heads_per_group", "quant_group_size", "chunks_per_head", "rope_start", "half_rope", "num_tokens", "n_groups", "d",
                 "scale_inner", "tma_aligned_T")
        o, pos_v, cs = (a.get(k) for k in ("o", "positions", "cos_sin_cache"))
        if not all(isinstance(a.get(k), int) for k in sizes) or not isinstance(a.get("tma_aligned_scales"), bool) \
                or as_float(a.get("fp8_max")) is None or not all(isinstance(v, ViewDesc) for v in (o, pos_v, cs, *outs)) or len(outs) != 2:
            return unsupported(node, "the launch's sizes, tma_aligned_scales and fp8_max are not statics over (o, positions, cos_sin) "
                               "into (bytes, scales)")
        if target_of(ctx)[0].compute_capability != (9, 0) or a["tma_aligned_scales"] or as_float(a["fp8_max"]) != INV_ROPE_FP8_MAX:
            return unsupported(node, "the Definition is the cc 9.0 PTX with f32 scales clamped to 448", tma_aligned_scales=a["tma_aligned_scales"],
                               fp8_max=a["fp8_max"])
        rows = tuple(ctx.start.rows)
        T, G, HPG, CH, QG, ROPE, tma = (len(rows), a["n_groups"], a["heads_per_group"], a["chunks_per_head"], a["quant_group_size"],
                                        2 * a["half_rope"], a["tma_aligned_T"])
        HD, fp8, sc = CH * QG, *outs
        if (a["num_tokens"], a["d"], a["scale_inner"], QG - a["rope_start"]) != (T, HPG * HD, HPG * CH, ROPE):
            return unsupported(node, f"the launch is not {T} tokens of {HPG} heads a group, each {CH} groups of {QG} with a rotary suffix of {ROPE}",
                               num_tokens=a["num_tokens"], d=a["d"], scale_inner=a["scale_inner"], rope_start=a["rope_start"])
        pins = {"heads_per_group": HPG, "QUANT_GROUP_SIZE": QG, "CHUNKS_PER_HEAD": CH, "ROPE_START": a["rope_start"],
                "HALF_ROPE": a["half_rope"], "TMA_ALIGNED_SCALES": False, "fp8_max": INV_ROPE_FP8_MAX, "eps": INV_ROPE_EPS}
        if any(kid.enter.constexprs.get(k) != v for k, v in pins.items()) or kid.enter.grid != [tma, G * HPG] or tma < T:
            return unsupported(node, "the kernel's constexprs or grid are not the launch's", constexprs=kid.enter.constexprs, grid=kid.enter.grid,
                               tma_aligned_T=tma)
        try:
            I.check_head_statics(QG, ROPE)
        except ValueError as e:
            return unsupported(node, str(e))
        if not (_is(o, (T, G * HPG, HD), "bfloat16") and _is(fp8, (G, T, HPG * HD), "float8_e4m3fn") and _is(sc, (G, T, HPG * CH), "float32")) \
                or o.strides[2] != 1 or fp8.strides[2] != 1 or sc.strides != (HPG * CH * tma, 1, tma) \
                or cs.dtype != "float32" or cs.rank != 2 or cs.shape[1] != ROPE or cs.strides[1] != 1:
            return unsupported(node, "o, the bytes and the scales are not [T,G*HPG,HD] bf16, [G,T,d] e4m3 and the wrapper's [G,T,scale_inner] f32, "
                               "or cos_sin not an f32 [., ROPE] table", o=repr(o), fp8=repr(fp8), scales=repr(sc), cs=repr(cs))
        if (ka.get("o_ptr"), ka.get("cos_sin_cache_ptr"), ka.get("fp8_ptr"), ka.get("scale_ptr"), ka.get("num_tokens")) != (o, cs, fp8, sc, T) \
                or (ka.get("o_stride_token"), ka.get("o_stride_head"), ka.get("cache_stride_pos"), ka.get("fp8_stride_group"),
                    ka.get("fp8_stride_token"), ka.get("scale_stride_group"), ka.get("scale_stride_k")) \
                != (*o.strides[:2], cs.strides[0], *fp8.strides[:2], sc.strides[0], sc.strides[2]):
            return unsupported(node, "the kernel's pointers and strides are not the op's views'")
        pos = step_positions(node, ctx, ka.get("positions_ptr"), T)
        if isinstance(pos, Unsupported):
            return pos
        if any(p < 0 or p >= cs.shape[0] for p in pos):
            return unsupported(node, "a position is outside the rotary table", positions=pos[:8], table_rows=cs.shape[0])
        croots = ctx.roots_of(cs)
        if not croots:
            return unsupported(node, "the rotary table is no root of the capture", cs=repr(cs))
        spec, hint, parts = vocab_kind(ctx, "dsv4_inv_rope_quant_head", CHUNKS=CH, QG=QG, ROPE=ROPE), f"{_module(node)}/inv_rope_quant", []
        for k in range(G * HPG):
            g, j = divmod(k, HPG)
            head = ViewDesc(o.alloc, o.offset + k * o.strides[1], (T, HD), (o.strides[0], 1), o.dtype, o.device)
            q = ViewDesc(fp8.alloc, fp8.offset + g * fp8.strides[0] + j * HD, (T, HD), (fp8.strides[1], 1), fp8.dtype, fp8.device)
            s = ViewDesc(sc.alloc, sc.offset + g * sc.strides[0] + j * CH * tma, (T, CH), (1, tma), sc.dtype, sc.device)
            parts.append(Compute(spec, T, (PerRow(head), Gather(cs, pos)), (PerRow(q), PerRow(s)), () if parts else (node.id,), hint=f"{hint}[h={k}]",
                                 kernels=(triton_name(kid),), attribution=rows, pattern=self.name, roots=(None, croots[0].name)))
        ctx.scratch["dsv4_o_proj"] = (_module(node), fp8, sc)
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


@dataclass
class Fp8Einsum:
    """DeepGEMM's `fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a.weight, wo_a.weight_scale_inv), z, recipe=(1, 128, 128))`
    (`deep_gemm_fp8_o_proj`, cc 9.0) -> per token `ScaledMmFp8BlockGrouped_v1{H,K,N,G}` of its G groups' bytes and scales over the
    grouped weight: the Build's `DeepGemmFp8EinsumRule`.  The JIT kernel leaves no event; what does is z's allocation, the first in
    the attention's module after the inverse-RoPE quant (`InvRopeQuant`), which this resolves to the Calls writing it."""

    name = "dsv4.o_proj_einsum"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        st = ctx.scratch.get("dsv4_o_proj")
        if node.is_triton or op_name(node.op) != "empty" or st is None or st[0] != _module(node):
            return None
        mod, fp8, sc = ctx.scratch.pop("dsv4_o_proj")
        (G, T, K), QG, outs = fp8.shape, FP8_QUANT_PINS["group_size"], node.output_views()
        z = outs[0] if len(outs) == 1 else None
        if not isinstance(z, ViewDesc) or z.dtype != "bfloat16" or z.rank != 3 or z.shape[:2] != (T, G) or K % QG or sc.shape[2] != K // QG:
            return unsupported(node, "the o-projection's first allocation after the inverse RoPE is not z [T,G,N] bf16 over K/128 scales",
                               z=repr(z), fp8=repr(fp8), scales=repr(sc))
        N, names = z.shape[2], (f"{mod}.wo_a.weight", f"{mod}.wo_a.weight_scale_inv")
        ws = _weights(node, ctx, names)
        if isinstance(ws, Unsupported):
            return ws
        w, s = (x.view for x in ws)
        if not (_is(w, (G, N, K), "float8_e4m3fn") and _is(s, (G, weight_blocks(N, QG), K // QG), "float32")):
            return unsupported(node, "wo_a is not the grouped e4m3 weight [G,N,K] with its f32 block scales [G,cdiv(N,128),K/128]",
                               w=repr(w), s=repr(s))
        xq = ViewDesc(fp8.alloc, fp8.offset, (T, G, K), (fp8.strides[1], fp8.strides[0], 1), fp8.dtype, fp8.device)
        sx = ViewDesc(sc.alloc, sc.offset, (T, G, K // QG), (sc.strides[1], sc.strides[0], sc.strides[2]), sc.dtype, sc.device)
        return Compute(vocab_kind(ctx, "deepgemm_fp8_einsum_grouped", H=G, K=K, N=N, G=QG), T, (PerRow(xq), PerRow(sx), *ws), (PerRow(z),),
                       (node.id,), hint=f"{mod}/wo_a", attribution=tuple(ctx.start.rows), pattern=self.name, roots=(None, None, *names))



def dsv4_patterns() -> list:
    """The DeepSeek-V4 patterns, ahead of the rank's collectives (`Mhc` records the all-reduces they resolve); with them vLLM's default
    Hopper block-FP8 linear, the cuBLAS GEMMs (the f32-output one is the KV compressors') and the final `_C.rms_norm`."""
    return [Mhc(), QKvRmsNorm(), QNormRopeKvInsert(), SavePartialStates(), IndexerCompress(), InvRopeQuant(), Fp8Einsum(), SparseDecode(),
            SparsePrefill(), DynamicDeepGemmFp8Linear(), CublasLinear(), CublasMmF32Out(), CudaRmsNorm()]


__all__ = ["Fp8Einsum", "IndexerCompress", "InvRopeQuant", "Mhc", "QKvRmsNorm", "QNormRopeKvInsert", "SavePartialStates", "dsv4_patterns",
           "indexer_short_path"]
