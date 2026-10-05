"""The `mla-moe` κ patterns (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM` on vLLM d9105ea80, TRITON_MLA decode / FA2 prefill).

The fold-side twins of the Build's MLA, router and shared-expert rules (`rules/vllm_bindings/mla_attention.py`, `f32_linear.py`,
`rules/vllm_moe_router.GroupedTopkRule`), one per custom op or launch the base list does not know:

  `vllm.unified_mla_kv_cache_update` ⊃ `_C_cache_ops.concat_and_cache_mla`   two moves: kv_c -> slot[:L], k_pe -> slot[L:L+DR]
  `vllm.unified_mla_attention_with_output`, decode (Triton MLA)            `MlaAbsorb{NH,DN,L}` (bmm vs W_UK_T), the `[ql | q_pe]` cat
                                                                           (moves), `TritonMlaDecodeWord{NH,T,L,DR,32,1,SCALE}` per
                                                                           distinct T over the latent cache, `MlaBmmWord{NH,L,DV}` (vs W_UV)
  `vllm.unified_mla_attention_with_output`, prefill (FA2 varlen)           the kv_b_proj GEMM (the profile's own GEMM pattern), the
                                                                           `[k_nope | k_pe]` / v copies (moves), FA2's causal attention
                                                                           per query row over this call's own keys
  `_C.rotary_embedding(..., is_neox=False)`                                `RoPEGptJ_v2{NHEADS,D}` on q and k
  `aten.linear(x f32, W f32)` on the widened hidden                       `CublasF32Linear{K,N,TREE}` (the router gate)
  `vllm.moe_forward_shared`                                                the shared expert (GEMM, `SiluMul_v2`, GEMM) and the routed
                                                                           block of `MoEFusedExperts` with the sigmoid, bias-corrected
                                                                           `grouped_topk` router (`MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}`)
  `aten.mul_.Tensor(routed, C)`                                            `Bf16MulScalar{N,C}` (the routed scaling factor)
  `aten.add.Tensor(shared, routed)`                                        `ResidualAdd{N}`
  `_C.silu_and_mul`                                                        `SiluMul_v2{I}` (the Build's `silu_mul` kind)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.ir.defs import bind
from verity_vllm.program.registry import act as _ACT
from verity_vllm.program.registry import boolean_rope as _BROPE
from verity_vllm.program.registry import cublas_f32 as _CUB
from verity_vllm.program.registry import dense as _DN
from verity_vllm.program.registry import mla_word as _MLAW
from verity_vllm.program.registry import moe as _MOE
from verity_vllm.program.registry import targets as T

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (SiluAndMul, _positions_ok, as_float, attribution_for, base_op, bound_args,
                                               classify_transparent, find_descendants, first, hint_of, is_int, is_row_major, op_name,
                                               target_of, triton_name, unsupported, widened_source)
from verity_vllm.observe.fold.patterns.geometry import GeometryError, rows_operand
from verity_vllm.observe.fold.patterns.kv_cache import kv_group_of, kv_layer_of, kv_slots
from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Group, PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.engine.profiles.gen_ln_patterns import ResidualAddPattern
from verity_vllm.engine.profiles.gen_ov_moe_patterns import K_ALIGN_KERNELS, K_FUSED, K_SUM, MoEFusedExperts, _as_rows, _contig, _flat, _root_name

#: `bmm_kernel`'s tiling on the pinned κ (vLLM d9105ea80 `mla_attention` absorb / v-up bmm, sm120, bf16)
BMM_PINS: dict[str, Any] = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}
#: `_fwd_grouped_kernel_stage1`'s constexprs `TritonMlaDecodeWord_v1` states (one split, BN 32, no logit cap, page 16)
DECODE_PINS: dict[str, Any] = {"BLOCK_N": _MLAW.BLOCK_N, "NUM_KV_SPLITS": 1, "PAGE_SIZE": 16, "logit_cap": 0.0, "IS_MLA": True}
K_BMM = "bmm_kernel"
K_STAGE1 = "_fwd_grouped_kernel_stage1"
K_STAGE2 = "_fwd_kernel_stage2"
K_GEMM = "matmul_kernel_persistent"
K_CONCAT_CACHE = "concat_and_cache_mla_kernel"


def _f32_bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def _view(alloc_of: ViewDesc, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]) -> ViewDesc:
    return ViewDesc(alloc_of.alloc, offset, shape, strides, alloc_of.dtype, alloc_of.device)


def _token_major(v: ViewDesc) -> ViewDesc:
    """A bmm operand `[B=NH, M, X]` as the token-major `[M, NH, X]` view of the same words."""
    return _view(v, v.offset, (v.shape[1], v.shape[0], v.shape[2]), (v.strides[1], v.strides[0], v.strides[2]))


def _slot_rows(cache: ViewDesc) -> ViewDesc:
    """The MLA latent cache `[blocks, block_size, L+DR]` (one latent head) as `[slots, L+DR]`."""
    if cache.rank == 4 and cache.shape[2] == 1:
        nb, bs, _, w = cache.shape
        s_b, s_o, _, s_d = cache.strides
    elif cache.rank == 3:
        nb, bs, w = cache.shape
        s_b, s_o, s_d = cache.strides
    else:
        raise GeometryError(f"MLA cache view {cache!r} is not [blocks, block_size, (1,) L+DR]")
    if s_d != 1 or s_o < w or s_b != bs * s_o:
        raise GeometryError(f"MLA cache view {cache!r} is not a uniformly strided slot table")
    return _view(cache, cache.offset, (nb * bs, w), (s_o, 1))


def _weight_root(ctx: StepContext, view: ViewDesc, fallback: str) -> str:
    roots = ctx.roots_of(view)
    w = [r for r in roots if r.role == "weight"]
    return (w or roots)[0].name if roots else fallback


def _glue(node: TreeNode, ctx: StepContext, keep: set[int], pattern: str, extra=None) -> tuple[list[tuple[int, Any]], Unsupported | None]:
    """`scope_transparents` with each move/alloc keyed by its node id, so the scope's parts run in log order (a move inside an MLA
    scope reads what an earlier compute of the scope wrote).  `extra(child)` may explain a child the transparent set does not."""
    out: list[tuple[int, Any]] = []

    def walk(n: TreeNode):
        for c in n.children:
            if c.id in keep:
                continue
            yield c
            yield from walk(c)

    for c in walk(node):
        r = extra(c) if extra is not None else None
        if r is None:
            r = classify_transparent(c, ctx)
        if r is None:
            return [], unsupported(node, f"unexplained child {c.op} inside {op_name(node.op)}", child_id=c.id, child=c.op)
        if isinstance(r, Unsupported):
            return [], unsupported(node, f"child {c.op} inside {op_name(node.op)}: {r.reason}", child_id=c.id, child_detail=r.detail)
        if isinstance(r, Transparent) and r.effect in ("move", "alloc"):
            out.append((c.id, dataclasses.replace(r, consumed=(), pattern=pattern)))
        elif isinstance(r, list):
            out.extend((c.id, x) for x in r)
    return out, None


def _profile_compute(node: TreeNode, child: TreeNode, ctx: StepContext, skip: tuple[type, ...]) -> Compute | Unsupported:
    """An in-scope launch resolved by the profile's own patterns, as the first-match resolver resolves it outside a scope."""
    for pt in ctx.profile.patterns:
        if isinstance(pt, skip):
            continue
        r = pt.match(child, ctx)
        if r is None:
            continue
        if not isinstance(r, Compute):
            return unsupported(node, f"{getattr(pt, 'name', type(pt).__name__)} refused {child.op} inside {op_name(node.op)}",
                               child_id=child.id, child_reason=getattr(r, "reason", None), child_detail=getattr(r, "detail", None))
        return r
    return unsupported(node, f"no profile pattern resolves {child.op} inside {op_name(node.op)}", child_id=child.id)


def _ordered_group(node: TreeNode, parts: list[tuple[int, Any]], hint: str, pattern: str) -> Group:
    return Group(tuple(p for _, p in sorted(parts, key=lambda x: x[0])), tuple(sorted(node.ids())), hint=hint, pattern=pattern)


@dataclass
class MlaKvCacheUpdate:
    """`vllm.unified_mla_kv_cache_update(kv_c [M,L], k_pe [M,1,DR], layer_name, 'auto', k_scale)` ⊃
    `_C_cache_ops.concat_and_cache_mla(kv_c, k_pe [M,DR], kv_cache [blocks,bs,L+DR], slot_mapping, 'auto', scale)` -> two moves:
    `PerRow(kv_c) -> Gather(cache[:, :L], slots)` and `PerRow(k_pe) -> Gather(cache[:, L:], slots)` (the bf16 cache stores the row as is;
    the slots are the step's public slot mapping of the layer's KV-cache group)."""

    name: str = "kv.concat_and_cache_mla"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_mla_kv_cache_update":
            return None
        ccs = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "concat_and_cache_mla")
        if len(ccs) != 1:
            return unsupported(node, "unified_mla_kv_cache_update without exactly one concat_and_cache_mla", found=[n.op for n in ccs])
        cc = ccs[0]
        glue, bad = _glue(node, ctx, {cc.id}, self.name)
        if bad is not None:
            return bad
        a = bound_args(cc)
        kv_c, k_pe, cache, sm, dt = a.get("kv_c"), a.get("k_pe"), a.get("kv_cache"), a.get("slot_mapping"), a.get("kv_cache_dtype")
        if not all(isinstance(v, ViewDesc) for v in (kv_c, k_pe, cache, sm)):
            return unsupported(node, "concat_and_cache_mla arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        if dt != "auto":
            return unsupported(node, "MLA cache dtype is not the bf16 cache ('auto')", kv_cache_dtype=dt)
        if not (kv_c.dtype == k_pe.dtype == cache.dtype == "bfloat16"):
            return unsupported(node, "MLA cache write is bf16 -> bf16 only", dtypes=[kv_c.dtype, k_pe.dtype, cache.dtype])
        if kv_c.rank != 2 or k_pe.rank != 2 or kv_c.strides[1] != 1 or k_pe.strides[1] != 1 or kv_c.shape[0] != k_pe.shape[0]:
            return unsupported(node, "kv_c / k_pe are not [M,L] / [M,DR] rows", kv_c=repr(kv_c), k_pe=repr(k_pe))
        M, L, DR = kv_c.shape[0], kv_c.shape[1], k_pe.shape[1]
        try:
            rows = _slot_rows(cache)
        except GeometryError as e:
            return unsupported(node, str(e))
        if rows.shape[1] != L + DR:
            return unsupported(node, "MLA cache row is not [kv_c | k_pe]", cache=repr(cache), L=L, DR=DR)
        layer = kv_layer_of(node) or kv_layer_of(cc)
        try:
            group = kv_group_of(ctx, layer)
        except GeometryError as e:
            return unsupported(node, str(e), layer=layer)
        public = ctx.start.slot_mapping if group is None else (group.get("slot_mapping") or [])
        if not is_int(sm) or sm.numel < M or len(public) < M:
            return unsupported(node, "slot_mapping does not cover the rows", slot_mapping=repr(sm), M=M, public=len(public))
        slots = tuple(int(s) for s in public[:M])
        if any(s < 0 or s >= rows.shape[0] for s in slots):
            return unsupported(node, "a slot is outside the cache (or a padding slot -1)", slots=slots[:8], cache_slots=rows.shape[0])
        lat = _view(rows, rows.offset, (rows.shape[0], L), rows.strides)
        pe = _view(rows, rows.offset + L, (rows.shape[0], DR), rows.strides)
        moves = [(cc.id, Transparent("move", (), src=PerRow(kv_c), dst=Gather(lat, slots), rows=M, hint=hint_of(node, "kv.latent"),
                                     kernels=(K_CONCAT_CACHE,), pattern=self.name)),
                 (cc.id, Transparent("move", (), src=PerRow(k_pe), dst=Gather(pe, slots), rows=M, hint=hint_of(node, "kv.rope"),
                                     kernels=(), pattern=self.name))]
        return _ordered_group(node, glue + moves, hint_of(node, "kv"), self.name)


@dataclass
class MlaAttentionScope:
    """`vllm.unified_mla_attention_with_output(q [M,NH,DN+DR], kv_c [M,L], k_pe [M,1,DR], output! [M,NH*DV], layer_name, ...)`.

    Decode (the Triton MLA launches `bmm_kernel`, `_fwd_grouped_kernel_stage1`, `_fwd_kernel_stage2`, `bmm_kernel`):
    `MlaAbsorb{NH,DN,L,DOT}` per token of `q_nope` against W_UK_T (kv_b_proj's rows as stored), the cat `[ql | q_pe]` as two moves,
    `TritonMlaDecodeWord{NH,T,L,DR,32,1,SCALE,DOT}` per distinct `T = position + 1` over the latent cache's slots 0..T-1 of the row's
    request, `MlaBmmWord{NH,L,DV,DOT}` against W_UV.  Prefill (`_vllm_fa2_C.varlen_fwd` over this call's own keys, no context): the
    kv_b_proj GEMM by the profile's GEMM pattern, the K / V assembly copies and the identity pad as moves, then one FA2 row per query row
    (`registry.targets.attention_spec`: Attention_v5 under FA2's per-iteration Check_inf) over its request's rows 0..t.  A scope with
    both paths (a mixed decode + prefill step) or a prefill with context is Unsupported."""

    name: str = "attention.mla"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_mla_attention_with_output":
            return None
        a = bound_args(node)
        q, kv_c, k_pe, out = a.get("q"), a.get("kv_c_normed"), a.get("k_pe"), a.get("output")
        if not all(isinstance(v, ViewDesc) for v in (q, kv_c, k_pe, out)):
            return unsupported(node, "unified_mla_attention_with_output arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        if a.get("output_scale") is not None or a.get("output_block_scale") is not None:
            return unsupported(node, "quantised MLA output (output_scale / output_block_scale) is not modelled")
        M = q.shape[0]
        if M != len(ctx.start.rows) or ctx.start.num_tokens != ctx.start.num_tokens_padded:
            return unsupported(node, "MLA query rows are not exactly the step's token rows", M=M, rows=len(ctx.start.rows))
        stage1 = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_STAGE1)
        fas = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "varlen_fwd")
        if stage1 and fas:
            return unsupported(node, "MLA scope runs both the decode and the prefill path (a mixed step): not modelled")
        if stage1:
            return self._decode(node, ctx, q, out, M)
        if fas:
            return self._prefill(node, ctx, q, kv_c, k_pe, out, M, fas[0])
        return unsupported(node, "MLA scope has neither the Triton decode nor the FA2 prefill launch",
                           descendants=[n.op for n in node.walk() if n is not node][:24])

    def _bmm(self, node: TreeNode, launch: TreeNode, NH: int, M: int, K: int, N: int):
        e = launch.enter
        bad = {k: (e.constexprs.get(k), v) for k, v in BMM_PINS.items() if e.constexprs.get(k) != v}
        if bad:
            return unsupported(node, "bmm_kernel tiling differs from the pinned profile", mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()})
        b = bound_args(launch)
        av, bv, cv = b.get("a_ptr"), b.get("b_ptr"), b.get("c_ptr")
        if not all(isinstance(v, ViewDesc) for v in (av, bv, cv)):
            return unsupported(node, "bmm_kernel pointers not bound as expected", args={k: repr(v) for k, v in b.items()})
        if (b.get("B"), b.get("M"), b.get("N"), b.get("K")) != (NH, M, N, K):
            return unsupported(node, "bmm_kernel B/M/N/K differ from the scope's geometry", observed=[b.get(k) for k in "BMNK"], expected=[NH, M, N, K])
        if av.shape != (NH, M, K) or bv.shape != (NH, K, N) or cv.shape != (NH, M, N) or not (av.dtype == bv.dtype == cv.dtype == "bfloat16"):
            return unsupported(node, "bmm_kernel operands are not bf16 [NH,M,K] x [NH,K,N] -> [NH,M,N]", a=repr(av), b=repr(bv), c=repr(cv))
        if av.strides[2] != 1 or cv.strides[2] != 1:
            return unsupported(node, "bmm_kernel a / c rows are not unit-stride", a=repr(av), c=repr(cv))
        return av, bv, cv

    def _decode(self, node: TreeNode, ctx: StepContext, q: ViewDesc, out: ViewDesc, M: int) -> Resolution:
        bmms = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_BMM)
        s1 = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_STAGE1)
        s2 = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_STAGE2)
        cats = find_descendants(node, lambda n: not n.is_triton and base_op(n.op) == "aten.cat")
        if len(bmms) != 2 or len(s1) != 1 or len(s2) != 1 or len(cats) != 1:
            return unsupported(node, "MLA decode scope is not absorb bmm, cat, stage1, stage2, v-up bmm",
                               counts={"bmm": len(bmms), "stage1": len(s1), "stage2": len(s2), "cat": len(cats)})
        absorb, vup, st1, st2, cat = bmms[0], bmms[1], s1[0], s2[0], cats[0]
        if not absorb.id < cat.id < st1.id < st2.id < vup.id:
            return unsupported(node, "MLA decode launches are out of order", ids=[absorb.id, cat.id, st1.id, st2.id, vup.id])
        tp = target_of(ctx)[0]
        try:
            dot = T.tensor_core_dot(tp)
        except T.UnregisteredTarget as e:
            return unsupported(node, f"target has no registered k16 step: {e}")
        if T.uses_accepted_kinds(tp):
            return unsupported(node, "the MLA kinds are bound with a target's k16 DOT step; the accepted (Ampere) kinds have none")
        c1 = st1.enter.constexprs
        bad = {k: (c1.get(k), v) for k, v in DECODE_PINS.items() if c1.get(k) != v}
        if bad:
            return unsupported(node, "MLA decode stage1 constexprs differ from TritonMlaDecodeWord_v1's", mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()})
        NH, Lk, Lv = c1.get("q_head_num"), c1.get("Lk"), c1.get("Lv")
        if c1.get("kv_group_num") != NH or c1.get("BLOCK_DMODEL") != Lv or c1.get("BLOCK_DPE") != Lk - Lv or c1.get("BLOCK_DV") != Lv:
            return unsupported(node, "MLA decode stage1 is not one latent head of [L | DR] for every query head", constexprs=dict(c1))
        L, DR = Lv, Lk - Lv
        c2 = st2.enter.constexprs
        if c2.get("NUM_KV_SPLITS") != 1 or c2.get("Lv") != L or c2.get("BLOCK_DV") != L:
            return unsupported(node, "MLA decode stage2 is not the one-split reduction of [L]", constexprs=dict(c2))
        D = q.shape[2] if q.rank == 3 else None
        if q.rank != 3 or q.shape[1] != NH or q.dtype != "bfloat16":
            return unsupported(node, "q is not bf16 [M,NH,DN+DR]", q=repr(q))
        DN = D - DR
        r = self._bmm(node, absorb, NH, M, DN, L)
        if isinstance(r, Unsupported):
            return r
        xa, wa, ca = r
        if xa.alloc != q.alloc or xa.offset != q.offset or (xa.strides[0], xa.strides[1]) != (q.strides[1], q.strides[0]):
            return unsupported(node, "absorb bmm A operand is not q_nope", a=repr(xa), q=repr(q))
        if not (wa.strides[1] == L and wa.strides[2] == 1):
            return unsupported(node, "absorb bmm B operand is not W_UK_T ([NH][DN][L] rows of kv_b_proj as stored)", b=repr(wa))
        H = wa.strides[0] // L
        tk = attribution_for(ctx, M, node.phase)
        hint = hint_of(node)
        kvb = _weight_root(ctx, wa, "kv_b_proj")
        parts: list[tuple[int, Any]] = [
            (absorb.id, Compute(bind(_MLAW.MlaAbsorb, NHEADS=NH, K=DN, N=L, DOT=dot), M, (PerRow(_token_major(xa)), Shared(wa)),
                                (PerRow(_token_major(ca)),), (absorb.id,), hint=f"{hint}/absorb", kernels=(K_BMM,), attribution=tk,
                                pattern=self.name, roots=(None, kvb)))]
        # the cat [ql | q_pe] -> Q of stage1
        ca_args = bound_args(cat)
        tens, dim = ca_args.get("tensors"), ca_args.get("dim")
        cout = cat.output_views()
        if not isinstance(tens, (list, tuple)) or len(tens) != 2 or not all(isinstance(t, ViewDesc) for t in tens) or dim not in (-1, 2) or len(cout) != 1:
            return unsupported(node, "the decode query cat is not [ql, q_pe] along the last axis", args={k: repr(v) for k, v in ca_args.items()})
        ql, qpe = tens
        qcat = cout[0]
        if ql.shape != (M, NH, L) or qpe.shape != (M, NH, DR) or qcat.shape != (M, NH, L + DR) or qcat.strides[2] != 1:
            return unsupported(node, "the decode query cat geometry is not [M,NH,L] + [M,NH,DR]", ql=repr(ql), q_pe=repr(qpe), out=repr(qcat))
        if (ql.alloc, ql.offset, ql.strides) != (ca.alloc, ca.offset, _token_major(ca).strides):
            return unsupported(node, "the decode query cat does not read the absorb output", ql=repr(ql), absorb_out=repr(ca))
        if (qpe.alloc, qpe.offset) != (q.alloc, q.offset + DN * q.strides[2]):
            return unsupported(node, "the decode query cat does not read q_pe", q_pe=repr(qpe), q=repr(q))
        parts.append((cat.id, Transparent("move", (), src=PerRow(ql), dst=PerRow(_view(qcat, qcat.offset, (M, NH, L), qcat.strides)), rows=M,
                                          hint=f"{hint}/cat.ql", kernels=("CatArrayBatchedCopy",), pattern=self.name)))
        parts.append((cat.id, Transparent("move", (), src=PerRow(qpe), dst=PerRow(_view(qcat, qcat.offset + L, (M, NH, DR), qcat.strides)), rows=M,
                                          hint=f"{hint}/cat.q_pe", kernels=(), pattern=self.name)))
        # stage1 + stage2: the one-split Triton MLA decode
        s1a, s2a = bound_args(st1), bound_args(st2)
        Q, KB, VB, o = s1a.get("Q"), s1a.get("K_Buffer"), s1a.get("V_Buffer"), s2a.get("o")
        if not all(isinstance(v, ViewDesc) for v in (Q, KB, VB, o)):
            return unsupported(node, "MLA decode launch pointers not bound as expected", stage1={k: repr(v) for k, v in s1a.items()})
        if (Q.alloc, Q.offset, Q.shape, Q.strides) != (qcat.alloc, qcat.offset, qcat.shape, qcat.strides):
            return unsupported(node, "MLA decode Q is not the cat output", Q=repr(Q), cat=repr(qcat))
        try:
            kv_rows = _slot_rows(KB)
        except GeometryError as e:
            return unsupported(node, str(e))
        if kv_rows.shape[1] != L + DR or VB.alloc != KB.alloc or VB.offset != KB.offset or VB.shape[-1] != L:
            return unsupported(node, "MLA decode K / V buffers are not the latent cache [L | DR] and its [L] prefix", K=repr(KB), V=repr(VB))
        if o.shape != (M, NH, L) or o.strides[2] != 1 or o.dtype != "bfloat16":
            return unsupported(node, "MLA decode output is not bf16 [M,NH,L]", o=repr(o))
        scale = as_float(s1a.get("sm_scale"))
        if scale is None:
            return unsupported(node, "MLA decode sm_scale is not a static float", sm_scale=repr(s1a.get("sm_scale")))
        SCALE = _f32_bits(scale)
        layer = kv_layer_of(node)
        groups: dict[int, list[int]] = {}
        for i, (req, position) in enumerate(ctx.start.rows):
            Tn = int(position) + 1
            seq = ctx.start.seq_lens.get(req)
            if seq is not None and Tn > seq:
                return unsupported(node, "row position exceeds the request's KV length", row=i, request=req, T=Tn, seq_len=seq)
            groups.setdefault(Tn, []).append(i)
        for Tn, rows in groups.items():
            try:
                slots = tuple(kv_slots(ctx, ctx.start.rows[i][0], Tn, layer=layer) for i in rows)
            except GeometryError as e:
                return unsupported(node, str(e), T=Tn, layer=layer)
            if any(s >= kv_rows.shape[0] for ss in slots for s in ss):
                return unsupported(node, "a KV slot is outside the latent cache", T=Tn, cache_slots=kv_rows.shape[0])
            spec = bind(_MLAW.TritonMlaDecodeWord, H=NH, T=Tn, DL=L, DR=DR, BN=_MLAW.BLOCK_N, SPLITS=1, SCALE=SCALE, DOT=dot)
            # dated at stage2, the launch `o` is bound to: an instance's output snapshot is looked up at its event (`min(consumed)`)
            parts.append((st1.id, Compute(spec, len(rows), (rows_operand(qcat, rows), GatherMany(kv_rows, slots)), (rows_operand(o, rows),),
                                          (st2.id,), hint=f"{hint}[t={Tn - 1}]", kernels=(K_STAGE1, K_STAGE2),
                                          attribution=tuple(ctx.start.rows[i] for i in rows), pattern=self.name)))
        # the v-up bmm against W_UV
        DV = out.shape[1] // NH if out.rank == 2 else None
        r = self._bmm(node, vup, NH, M, L, DV)
        if isinstance(r, Unsupported):
            return r
        xv, wv, cv = r
        if (xv.alloc, xv.offset) != (o.alloc, o.offset) or _token_major(xv).strides != o.strides:
            return unsupported(node, "v-up bmm A operand is not the decode output", a=repr(xv), o=repr(o))
        if (wv.alloc, wv.strides) != (wa.alloc, (H * L, 1, L)) or wv.offset != wa.offset + DN * L:
            return unsupported(node, "v-up bmm B operand is not W_UV (kv_b_proj's value rows)", b=repr(wv), w_uk_t=repr(wa))
        if (cv.alloc, cv.offset) != (out.alloc, out.offset) or _token_major(cv).strides != (out.strides[0], DV, 1):
            return unsupported(node, "v-up bmm C operand is not the layer output", c=repr(cv), out=repr(out))
        w_uv = _view(wv, wv.offset, (NH, DV, L), (H * L, L, 1))
        parts.append((vup.id, Compute(bind(_MLAW.MlaBmmWord, NHEADS=NH, K=L, N=DV, DOT=dot), M, (PerRow(o), Shared(w_uv)),
                                      (PerRow(_token_major(cv)),), (vup.id,), hint=f"{hint}/v_up", kernels=(K_BMM,), attribution=tk,
                                      pattern=self.name, roots=(None, kvb))))
        keep = {absorb.id, vup.id, st1.id, st2.id, cat.id}
        glue, bad = _glue(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        return _ordered_group(node, glue + parts, hint, self.name)

    def _prefill(self, node: TreeNode, ctx: StepContext, q: ViewDesc, kv_c: ViewDesc, k_pe: ViewDesc, out: ViewDesc, M: int,
                 fa: TreeNode) -> Resolution:
        tp = target_of(ctx)[0]
        if tp.effective_flash_attn_version() != 2:
            return unsupported(node, "MLA prefill is FA2's varlen launch; the record target's effective FA version differs",
                               fa=tp.effective_flash_attn_version())
        a = bound_args(fa)
        qf, kf, vf = a.get("q"), a.get("k"), a.get("v")
        outs = fa.output_views()
        if not all(isinstance(v, ViewDesc) for v in (qf, kf, vf)) or not outs:
            return unsupported(node, "varlen_fwd q/k/v not bound by name", names=list(a)[:24])
        of = outs[0]
        if first(a.get("is_causal"), a.get("causal")) is not True:
            return unsupported(node, "MLA prefill is not causal")
        if qf.rank != 3 or (qf.alloc, qf.offset, qf.shape, qf.strides) != (q.alloc, q.offset, q.shape, q.strides):
            return unsupported(node, "varlen_fwd q is not the layer's query", q=repr(qf), query=repr(q))
        _, NH, D = qf.shape
        scale = as_float(a.get("softmax_scale"))
        if scale is None or not math.isclose(scale, float(D) ** -0.5, rel_tol=1e-9, abs_tol=0.0):
            return unsupported(node, "softmax_scale is not D**-0.5", softmax_scale=scale, D=D)
        for key in ("window_size_left", "window_size_right"):
            wv = a.get(key)
            if isinstance(wv, int) and wv >= 0:
                return unsupported(node, "sliding window is not modelled", **{key: wv})
        if as_float(a.get("softcap")) not in (None, 0.0) or a.get("alibi_slopes") is not None or as_float(first(a.get("p_dropout"), a.get("dropout_p"))) not in (None, 0.0):
            return unsupported(node, "softcap / alibi / dropout are not modelled")
        if a.get("block_table") is not None or a.get("seqused_k") is not None:
            return unsupported(node, "MLA prefill over a paged / context KV is not modelled (a prefill with context)")
        if a.get("num_splits") not in (None, 0, 1):
            return unsupported(node, "num_splits differs from the non-split launch", num_splits=a.get("num_splits"))
        for v in (kf, vf, of):
            if v.shape != (M, NH, D) or v.strides[1:] != (D, 1) or v.dtype != "bfloat16":
                return unsupported(node, "varlen_fwd k / v / out are not bf16 [M,NH,D] rows", view=repr(v))
        if not T.fa2_check_inf(tp):
            return unsupported(node, "the MLA prefill binds FA2's per-iteration Check_inf chain (Attention_v5); the target does not register it")
        gemms = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_GEMM)
        if len(gemms) != 1 or gemms[0].id > fa.id:
            return unsupported(node, "MLA prefill scope does not have exactly one kv_b_proj GEMM ahead of the attention", gemms=[g.id for g in gemms])
        g = gemms[0]
        gc = _profile_compute(node, g, ctx, (MlaAttentionScope,))
        if isinstance(gc, Unsupported):
            return gc
        ga = bound_args(g).get("a_ptr")
        if not isinstance(ga, ViewDesc) or (ga.alloc, ga.offset, ga.shape) != (kv_c.alloc, kv_c.offset, kv_c.shape):
            return unsupported(node, "the in-scope GEMM does not read kv_c", a=repr(ga), kv_c=repr(kv_c))
        parts: list[tuple[int, Any]] = [(g.id, gc)]
        rows_of: dict[str, list[int]] = {}
        for i, (req, _pos) in enumerate(ctx.start.rows):
            rows_of.setdefault(req, []).append(i)
        for req, rows in rows_of.items():
            if rows != list(range(rows[0], rows[0] + len(rows))):
                return unsupported(node, "a request's prefill rows are not contiguous in the step", request=req)
            pos = [int(ctx.start.rows[i][1]) for i in rows]
            if pos != list(range(len(rows))):
                return unsupported(node, "MLA prefill does not open the request's history (a prefill with context)", request=req, positions=pos[:8])
        max_sq = max(len(r) for r in rows_of.values())

        def pad(c: TreeNode):
            if base_op(c.op) != "aten.pad":
                return None
            pa = bound_args(c)
            src, dst = pa.get("self"), c.output_views()
            if not isinstance(src, ViewDesc) or len(dst) != 1 or any(int(p) != 0 for p in (pa.get("pad") or ())) or src.shape != dst[0].shape:
                return unsupported(c, "aten.pad is not the identity (all-zero pads)", pad=pa.get("pad"), src=repr(src))
            return [Transparent("move", (), src=PerRow(src), dst=PerRow(dst[0]), rows=src.shape[0], hint=f"{hint_of(c)}: identity pad",
                                kernels=("elementwise_kernel",), pattern=self.name)]

        keep = {g.id, fa.id} | set(g.ids()) | set(fa.ids())
        glue, bad = _glue(node, ctx, keep, self.name, extra=pad)
        if bad is not None:
            return bad
        hint = hint_of(node)
        for req, rows in rows_of.items():
            r0, own = rows[0], len(rows)
            for t, i in enumerate(rows):
                try:
                    spec = T.attention_spec(tp, t + 1, NH, NH, D, max_sq, row=t, own_rows=own)
                except T.UnregisteredTarget as e:
                    return unsupported(node, f"target has no registered attention binding: {e}")
                keys = (tuple(range(r0, r0 + t + 1)),)
                parts.append((fa.id, Compute(spec, 1, (rows_operand(qf, [i]), GatherMany(kf, keys), GatherMany(vf, keys)), (rows_operand(of, [i]),),
                                             (fa.id,), hint=f"{hint}[t={t}]", kernels=("flash_fwd_kernel",), attribution=(ctx.start.rows[i],),
                                             pattern=self.name)))
        return _ordered_group(node, glue + parts, hint, self.name)


@dataclass
class RotaryGptJ:
    """`_C.rotary_embedding(positions, query!, key!, head_size, cos_sin_cache, is_neox=False)` (GPT-J's interleaved pairs, in place) ->
    `RoPEGptJ_v2{NHEADS,D}` per token on q `[M,NH,D]` (any head stride, unit inner stride) and on k, with the cos/sin row gathered by the
    step's public positions."""

    name: str = "rope.rotary_embedding_gptj"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "rotary_embedding":
            return None
        a = bound_args(node)
        if a.get("is_neox", a.get("arg5")) is not False:
            return None
        pos = a.get("positions", a.get("arg0"))
        q, k = a.get("query", a.get("arg1")), a.get("key", a.get("arg2"))
        D, cs = a.get("head_size", a.get("arg3")), a.get("cos_sin_cache", a.get("arg4"))
        if not isinstance(pos, ViewDesc) or not isinstance(q, ViewDesc) or not isinstance(cs, ViewDesc) or not isinstance(D, int):
            return unsupported(node, "rotary_embedding arguments not bound as expected", args={k_: repr(v) for k_, v in a.items()})
        if cs.rank != 2 or cs.shape[1] != D or not is_row_major(cs) or cs.dtype != "bfloat16":
            return unsupported(node, "cos_sin_cache is not a row-major bf16 [MAX_POS, D] table", cs=repr(cs), D=D)
        M = q.shape[0]
        reason = _positions_ok(pos, M, ctx)
        if reason:
            return unsupported(node, reason)
        positions = tuple(int(p) for p in ctx.start.positions[:M])
        if any(p < 0 or p >= cs.shape[0] for p in positions):
            return unsupported(node, "a position is outside the cos/sin table", positions=positions[:8], rows=cs.shape[0])
        parts = []
        for which, v in (("q", q), ("k", k)):
            if v is None:
                continue
            if not isinstance(v, ViewDesc) or v.dtype != "bfloat16":
                return unsupported(node, f"{which} is not a bf16 tensor view", view=repr(v))
            if v.rank == 2 and v.strides[1] == 1 and v.shape[1] % D == 0:
                v = ViewDesc(v.alloc, v.offset, (v.shape[0], v.shape[1] // D, D), (v.strides[0], D, 1), v.dtype, v.device)
            if v.rank != 3 or v.shape[0] != M or v.shape[2] != D or v.strides[2] != 1 or (v.shape[1] > 1 and v.strides[1] < D):
                return unsupported(node, f"{which} view is not [M, NHEADS, D] rows", view=repr(v), D=D, M=M)
            parts.append(Compute(bind(_BROPE.RoPEGptJV2, NHEADS=v.shape[1], D=D), M, (PerRow(v), Gather(cs, positions)), (PerRow(v),),
                                 (node.id,) if not parts else (), hint=hint_of(node, f"rope.{which}"), kernels=("rotary_embedding_kernel",),
                                 attribution=attribution_for(ctx, M, node.phase), pattern=self.name))
        if not parts:
            return unsupported(node, "rotary_embedding without a query tensor")
        if len(parts) == 1:
            return parts[0]
        return Group(tuple(parts), (node.id,), hint=hint_of(node, "rope"), pattern=self.name)


@dataclass
class F32GateLinear:
    """`aten.linear(x f32 [M,K], W f32 [N,K])` with `x` the exact widening of a bf16 activation (the router gate,
    `nn.Linear(params_dtype=float32)` on `hidden.to(float32)`) -> `CublasF32Linear{K,N,TREE}` per row over the bf16 source, TREE the
    cuBLAS kernel family for M rows (`f32_linear.cublas_f32_kernel`)."""

    name: str = "linear.f32_cublas"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.linear":
            return None
        from verity_vllm.program.frontend.rules.vllm_bindings.f32_linear import cublas_f32_kernel  # noqa: PLC0415
        a = bound_args(node)
        x, w, b = a.get("input", a.get("arg0")), a.get("weight", a.get("arg1")), a.get("bias")
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(w, ViewDesc) or not outs:
            return unsupported(node, "aten.linear arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        if x.dtype != "float32" or w.dtype != "float32":
            return None
        if b is not None:
            return unsupported(node, "f32 linear with a bias is not modelled", bias=repr(b))
        src = widened_source(x, ctx)
        if src is None:
            return unsupported(node, "f32 linear input is not the exact widening of a bf16 activation", x=repr(x))
        y = outs[0]
        if x.rank != 2 or w.rank != 2 or not is_row_major(src) or not is_row_major(w) or y.shape != (x.shape[0], w.shape[0]) or not is_row_major(y):
            return unsupported(node, "f32 linear views are not row-major [M,K] x [N,K]^T -> [M,N]", x=repr(x), w=repr(w), y=repr(y))
        if not any(r.role == "weight" for r in ctx.roots_of(w)):
            return unsupported(node, "f32 linear weight is not a weight Root", w=repr(w))
        M, K = x.shape
        N = w.shape[0]
        kt = cublas_f32_kernel(K, N, M)
        if kt is None:
            return unsupported(node, "the cuBLAS f32 kernel for this (K, N, M) is unmeasured", K=K, N=N, M=M)
        return Compute(bind(_CUB.CublasF32LinearWord, K=K, N=N, TREE=kt[1]), M, (PerRow(src), Shared(w)), (PerRow(y),), (node.id,),
                       hint=hint_of(node, _weight_root(ctx, w, "gate")), kernels=(kt[0],), attribution=attribution_for(ctx, M, node.phase),
                       pattern=self.name)


@dataclass
class Bf16MulScalarInPlace:
    """`aten.mul_.Tensor(x bf16 [M,N], C)` (C a Python float: the routed scaling factor) -> `Bf16MulScalar{N,C}` in place."""

    name: str = "dense.bf16_mul_scalar_inplace"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.mul_":
            return None
        a = bound_args(node)
        x, c = a.get("self", a.get("arg0")), a.get("other", a.get("arg1"))
        if not isinstance(x, ViewDesc) or isinstance(c, (bool, ViewDesc)) or as_float(c) is None:
            return None
        if x.rank != 2 or not is_row_major(x) or x.dtype != "bfloat16":
            return unsupported(node, "Bf16MulScalar is over a row-major bf16 [M,N] view", x=repr(x))
        M, N = x.shape
        return Compute(bind(_DN.Bf16MulScalar, N=N, C=float(as_float(c))), M, (PerRow(x),), (PerRow(x),), (node.id,), hint=hint_of(node, "scale"),
                       kernels=("vectorized_elementwise_kernel", "elementwise_kernel"), attribution=attribution_for(ctx, M, node.phase),
                       pattern=self.name)


@dataclass
class SiluAndMulAsBuilt:
    """`_C.silu_and_mul` as `SiluMul_v2{I}` (`registry.act`), the Build's `silu_mul` kind."""

    name: str = "act.silu_and_mul.v2"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        r = SiluAndMul().match(node, ctx)
        if not isinstance(r, Compute):
            return r
        return dataclasses.replace(r, spec=bind(_ACT.SiluMul, I=r.spec.bindings["I"]), pattern=self.name)


@dataclass
class MoEFusedSharedExperts(MoEFusedExperts):
    """`vllm.moe_forward_shared(hidden [M,H], router_logits [M,E] f32, shared_experts_input) -> (shared, routed)`: the shared expert
    (the profile's GEMM, `SiluMul_v2`, GEMM) and the routed block of `MoEFusedExperts` with `_moe_C.grouped_topk(scores, 1, 1, TOPK,
    renormalize=True, scale, bias, sigmoid)` as `MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}` per token (`PerRow(scores)`, `Shared(bias)`)."""

    name = "moe.fused_shared_experts.pairwise"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "moe_forward_shared":
            return None
        from verity_vllm.program.frontend.rules.vllm_moe_router import grouped_topk_kernel  # noqa: PLC0415
        C = self.config
        E, TOPK, H, I = C.get("NUM_EXPERTS"), C.get("TOPK"), C["H"], C["I"]
        sa = bound_args(node)
        hidden, logits, sx = sa.get("hidden_states"), sa.get("router_logits"), sa.get("shared_experts_input")
        if not all(isinstance(v, ViewDesc) for v in (hidden, logits, sx)):
            return unsupported(node, "moe_forward_shared arguments not bound as expected", args={k: repr(v) for k, v in sa.items()})
        M = hidden.shape[0]
        if hidden.shape != (M, H) or M != len(ctx.start.rows) or ctx.start.num_tokens != ctx.start.num_tokens_padded:
            return unsupported(node, "MoE rows are not exactly the step's token rows", hidden=repr(hidden), rows=len(ctx.start.rows))
        if (sx.alloc, sx.offset, sx.shape) != (hidden.alloc, hidden.offset, hidden.shape):
            return unsupported(node, "the shared experts' input is not the routed experts' hidden_states", shared=repr(sx), hidden=repr(hidden))
        routers = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "grouped_topk")
        aligns = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "moe_align_block_size")
        launches = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_FUSED)
        gemms = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_GEMM)
        silus = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "silu_and_mul")
        sums = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "moe_sum")
        if len(routers) != 1 or len(launches) != 2 or len(gemms) != 2 or len(silus) != 2 or len(sums) != 1 or len(aligns) > 1:
            return unsupported(node, "shared-expert MoE scope does not have the shape (2 GEMM, 1 grouped_topk, <=1 align, 2 fused_moe_kernel, 2 silu_and_mul, 1 moe_sum)",
                               counts={"gemm": len(gemms), "grouped_topk": len(routers), "align": len(aligns), "fused_moe_kernel": len(launches),
                                       "silu_and_mul": len(silus), "moe_sum": len(sums)})
        router, up, down, msum = routers[0], launches[0], launches[1], sums[0]
        g_up, g_down = gemms
        s_silu, r_silu = silus
        if not g_up.id < s_silu.id < g_down.id < router.id < up.id < r_silu.id < down.id < msum.id:
            return unsupported(node, "shared-expert MoE launches are out of order",
                               ids=[g_up.id, s_silu.id, g_down.id, router.id, up.id, r_silu.id, down.id, msum.id])
        hint = node.enter.module or "mlp.experts"
        parts: list[tuple[int, Any]] = []
        # the shared expert
        for gl in (g_up, g_down):
            c = _profile_compute(node, gl, ctx, (MoEFusedExperts,))
            if isinstance(c, Unsupported):
                return c
            parts.append((gl.id, c))
        if (bound_args(g_up).get("a_ptr") or hidden).alloc != hidden.alloc:
            return unsupported(node, "the shared gate_up GEMM does not read hidden_states", a=repr(bound_args(g_up).get("a_ptr")))
        sr = SiluAndMulAsBuilt().match(s_silu, ctx)
        if not isinstance(sr, Compute) or sr.rows != M:
            return unsupported(node, "the shared expert's silu_and_mul did not resolve on the M token rows", detail=getattr(sr, "detail", None))
        parts.append((s_silu.id, dataclasses.replace(sr, hint=f"{hint}/shared/silu_and_mul")))
        # the router
        ra = bound_args(router)
        scores, bias = ra.get("scores"), ra.get("bias")
        if not isinstance(scores, ViewDesc) or not isinstance(bias, ViewDesc):
            return unsupported(node, "grouped_topk arguments not bound as expected", args={k: repr(v) for k, v in ra.items()})
        if (ra.get("n_group"), ra.get("topk_group"), ra.get("topk"), ra.get("renormalize"), ra.get("scoring_func")) != (1, 1, TOPK, True, 1):
            return unsupported(node, "grouped_topk is not one group, renormalized, sigmoid, TOPK", args={k: repr(v) for k, v in ra.items()})
        rsf = as_float(ra.get("routed_scaling_factor"))
        if rsf is None:
            return unsupported(node, "grouped_topk routed_scaling_factor is not a static number")
        if (scores.alloc, scores.offset, scores.shape) != (logits.alloc, logits.offset, logits.shape) or scores.shape != (M, E) or scores.dtype != "float32" or not is_row_major(scores):
            return unsupported(node, "grouped_topk scores are not the row-major f32 [M,E] router logits", scores=repr(scores), logits=repr(logits))
        if bias.shape != (E,) or bias.dtype != "float32" or not _contig(bias) or not any(r.role == "weight" for r in ctx.roots_of(bias)):
            return unsupported(node, "grouped_topk bias is not the f32 [E] e_score_correction_bias weight", bias=repr(bias))
        router_outs = router.output_views()
        if len(router_outs) != 2:
            return unsupported(node, "grouped_topk does not return (weights, ids)", outs=[repr(v) for v in router_outs])
        weights, ids = router_outs
        if weights.shape != (M, TOPK) or weights.dtype != "float32" or not _contig(weights) or ids.shape != (M, TOPK) or ids.dtype != "int32" or not _contig(ids):
            return unsupported(node, "grouped_topk outputs are not contiguous f32 / i32 [M,TOPK]", weights=repr(weights), ids=repr(ids))
        kernel = grouped_topk_kernel(E, M)
        if kernel is None:
            return unsupported(node, "the grouped_topk launch at this E is unmeasured", E=E)
        parts.append((router.id, Compute(bind(_MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=float(rsf), KERNEL=kernel), M,
                                         (PerRow(scores), Shared(bias)), (PerRow(weights), PerRow(ids)), (router.id,), hint=f"{hint}/router",
                                         kernels=(f"single_group_topk_{kernel}_kernel",), attribution=tuple(ctx.start.rows), pattern=self.name,
                                         roots=(None, _weight_root(ctx, bias, "e_score_correction_bias")))))
        for al in aligns:
            parts.append((al.id, Transparent("noop", (al.id,), hint=f"{hint}/moe_align_block_size: pair->expert-block schedule; no output word depends on it",
                                             kernels=K_ALIGN_KERNELS, pattern=self.name)))
        # the routed experts over (token, slot) pairs
        P = M * TOPK
        ids_flat = _flat(ids, P)
        try:
            up_spec, down_spec = (T.moe_expert_gemm_spec(target_of(ctx)[0], E, K, N, routed=w) for K, N, w in ((H, 2 * I, False), (I, H, True)))
        except T.UnregisteredTarget as e:
            return unsupported(node, f"target has no registered fused-MoE binding: {e}")
        r = self._fused(node, up, expect_weighted=False, expect_top_k=TOPK, E=E, K=H, N=2 * I, P=P)
        if isinstance(r, Unsupported):
            return r
        x_up, w13, c1 = r
        if x_up.alloc != hidden.alloc or x_up.shape != (M, H) or not is_row_major(x_up):
            return unsupported(node, "up-GEMM A operand is not the [M,H] hidden_states view", a=repr(x_up), hidden=repr(hidden))
        parts.append((up.id, Compute(up_spec, P, (Gather(x_up, tuple(p // TOPK for p in range(P))), PerRow(ids_flat), Shared(w13)),
                                     (PerRow(_as_rows(c1, P, 2 * I)),), (up.id,), hint=f"{hint}/{_root_name(ctx, w13, 'w13')}", kernels=(K_FUSED,),
                                     pattern=self.name)))
        rr = SiluAndMulAsBuilt().match(r_silu, ctx)
        if not isinstance(rr, Compute) or rr.rows != P:
            return unsupported(node, "the routed silu_and_mul did not resolve on the M*TOPK pair rows", detail=getattr(rr, "detail", None))
        parts.append((r_silu.id, dataclasses.replace(rr, hint=f"{hint}/silu_and_mul")))
        r = self._fused(node, down, expect_weighted=True, expect_top_k=1, E=E, K=I, N=H, P=P)
        if isinstance(r, Unsupported):
            return r
        x_dn, w2, c3 = r
        wv = bound_args(down).get("topk_weights_ptr")
        if not isinstance(wv, ViewDesc) or wv.alloc != weights.alloc or wv.offset != weights.offset or wv.shape != (M, TOPK):
            return unsupported(node, "down-GEMM routed weights are not the router's topk_weights view", topk_weights_ptr=repr(wv), weights=repr(weights))
        if x_dn.shape != (P, I) or not is_row_major(x_dn):
            return unsupported(node, "down-GEMM A operand is not the [M*TOPK, I] silu output", a=repr(x_dn))
        parts.append((down.id, Compute(down_spec, P, (PerRow(x_dn), PerRow(ids_flat), PerRow(_flat(weights, P)), Shared(w2)),
                                       (PerRow(_as_rows(c3, P, H)),), (down.id,), hint=f"{hint}/{_root_name(ctx, w2, 'w2')}", kernels=(K_FUSED,),
                                       pattern=self.name)))
        ma = bound_args(msum)
        y, out = ma.get("input"), ma.get("output")
        if not isinstance(y, ViewDesc) or not isinstance(out, ViewDesc):
            return unsupported(node, "moe_sum arguments not bound as expected", args={k: repr(v) for k, v in ma.items()})
        if y.shape != (M, TOPK, H) or not _contig(y) or y.dtype != "bfloat16" or y.alloc != c3.alloc or y.offset != c3.offset:
            return unsupported(node, "moe_sum input is not the contiguous bf16 [M,TOPK,H] down-GEMM output", input=repr(y), cache3=repr(c3))
        if out.shape != (M, H) or not is_row_major(out) or out.dtype != "bfloat16":
            return unsupported(node, "moe_sum output is not a row-major bf16 [M,H] view", output=repr(out))
        parts.append((msum.id, Compute(bind(_MOE.MoeSum, TOPK=TOPK, H=H), M, (PerRow(y),), (PerRow(out),), (msum.id,), hint=f"{hint}/moe_sum",
                                       kernels=(K_SUM,), attribution=tuple(ctx.start.rows), pattern=self.name)))
        keep = {n.id for n in (router, up, down, msum, g_up, g_down, s_silu, r_silu, *aligns)}
        keep |= set().union(*(set(n.ids()) for n in (router, up, down, msum, g_up, g_down, s_silu, r_silu, *aligns)))
        glue, bad = _glue(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        return _ordered_group(node, glue + parts, hint, self.name)


def mla_moe_patterns(C: dict[str, Any]) -> list:
    return [MlaKvCacheUpdate(), MlaAttentionScope(), RotaryGptJ(), F32GateLinear(), MoEFusedSharedExperts(C), Bf16MulScalarInPlace(),
            ResidualAddPattern()]


def with_mla_moe_patterns(patterns: list, C: dict[str, Any]) -> list:
    """The profile's pattern list with the `mla-moe` patterns after the tuned GEMM (ahead of the base `RotaryEmbedding`, which refuses
    GPT-J's rotation) and `_C.silu_and_mul` bound as `SiluMul_v2`."""
    out: list = []
    for p in patterns:
        if type(p) is SiluAndMul:
            p = SiluAndMulAsBuilt()
        out.append(p)
    at = next((i + 1 for i, p in enumerate(out) if type(p).__name__ == "GemmLaunchTuned"), 0)
    out[at:at] = mla_moe_patterns(C)
    return out
