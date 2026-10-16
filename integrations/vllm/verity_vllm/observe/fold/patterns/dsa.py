"""GLM-5.3's cc 9.0 DSA observer folds.

The patterns in this module are deliberately the recorded vLLM d9105ea80,
cc 9.0 specialisations.  They turn the two Triton fusion kernels, DeepGEMM's
unobserved MQA launch, the short-row top-k and FA3 sparse MLA into the same
Definitions used by the export frontend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, is_row_major
from verity_vllm.observe.fold.patterns.kv_cache import kv_slots
from verity_vllm.observe.fold.patterns.ops import bound_args, op_name, triton_name
from verity_vllm.observe.fold.patterns.resolutions import unsupported
from verity_vllm.observe.fold.patterns.rotary import _positions_ok
from verity_vllm.observe.fold.patterns.target import target_of
from verity_vllm.observe.fold.resolver import (
    Compute,
    Const,
    Gather,
    GatherMany,
    Group,
    PerRow,
    Resolution,
    Shared,
    StepContext,
    Transparent,
    Unsupported,
)
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.program.registry import dsa
from verity_vllm.program.registry.boolean_rope import RoPEGptJV2, RoPEHeadGptJV2
from verity_vllm.program.registry.dsa_norm import DsaRMSNorm
from verity_vllm.program.registry.sparse_mla import SPARSE_MLA_GLM53
from verity_vllm.program.registry.sparse_mla_split import SparseMlaSplit, launch_num_splits

NORM_KERNEL = "_fused_norm_rope_kernel"
Q_KERNEL = "_fused_q_kernel"
MQA_STAGE = "cp_gather_indexer_k_quant_cache"
TOPK_PREFILL = "top_k_per_row_prefill"
TOPK_DECODE = "cooperative_topk"
INDEX_SHARE_KERNEL = "vllm.v1.attention.backends.mla.sparse_utils.ConvertReqIndexToGlobalIndexKernel.kernel"
FA3_OP = "fwd"

NORM_SRC_SHA = "2dcf56892eb2b61c0f0249965013fc1f63968c00e473b3df35cb186d42c650de"
Q_SRC_SHA = "7380a1385db787ceded52869fe395160e859d22710ca46f20d1d0fdef2f41043"
INDEX_SHARE_SRC_SHA = "ad40bdb415664d3b65e3e4163863158096f2ea52f2e2a3e01ff8be63aeb98f63"


def _same(a: Any, b: ViewDesc) -> bool:
    return isinstance(a, ViewDesc) and (a.alloc, a.offset, a.shape, a.strides, a.dtype) == (
        b.alloc,
        b.offset,
        b.shape,
        b.strides,
        b.dtype,
    )


def _module(node: TreeNode) -> str:
    return getattr(node.enter, "module", None) or ""


def _root(ctx: StepContext, v: ViewDesc) -> str | None:
    """The checkpoint parameter a weight view reads (`Compute.roots`): GLM-5.3's Triton launches carry no module, so the hint alone
    names one layer's norm weight like every other layer's and the Match's weight map fuses all of them into one overlapping range."""
    rs = ctx.roots_of(v)
    return rs[0].name if rs else None


def _state(ctx: StepContext) -> dict:
    return ctx.scratch.setdefault("glm53_dsa", {"modules": {}, "topk": None})


def _module_state(ctx: StepContext, node: TreeNode) -> dict:
    return _state(ctx)["modules"].setdefault(_module(node), {})


def _one_row(v: ViewDesc, row: int, shape: tuple[int, ...] | None = None) -> ViewDesc:
    """A one-row physical view retaining ``v``'s row stride."""
    if v.rank < 1 or not 0 <= row < v.shape[0]:
        raise GeometryError(f"row {row} outside {v!r}")
    return ViewDesc(v.alloc, v.offset + row * v.strides[0], shape or (1, *v.shape[1:]), (v.strides[0], *v.strides[1:]), v.dtype, v.device)


def _slot_rows(v: ViewDesc, width: int) -> ViewDesc:
    """Flatten ``[blocks, block, physical-width]`` to logical slot rows."""
    if v.rank != 3 or v.shape[2] < width or v.strides[1:] != (v.shape[2], 1):
        raise GeometryError(f"cache is not [blocks, block, physical-width] row-major: {v!r}")
    return ViewDesc(v.alloc, v.offset, (v.shape[0] * v.shape[1], width), (v.strides[1], 1), v.dtype, v.device)


def _index_scale_rows(v: ViewDesc) -> ViewDesc:
    """The f32 word in bytes 128..131 of each 132-byte index-cache row."""
    if v.rank != 3 or v.dtype != "float32" or v.shape[2] != 33 or v.strides[1:] != (33, 1):
        raise GeometryError(f"index scale cache is not f32 [blocks, block, 33]: {v!r}")
    return ViewDesc(v.alloc, v.offset + 32, (v.shape[0] * v.shape[1],), (33,), v.dtype, v.device)


def _positions(node: TreeNode, ctx: StepContext, pos: Any, rows: int) -> tuple[int, ...] | Unsupported:
    if not isinstance(pos, ViewDesc):
        return unsupported(node, "positions argument is not a tensor view", positions=repr(pos))
    why = _positions_ok(pos, rows, ctx)
    if why:
        return unsupported(node, why)
    out = tuple(int(p) for p in ctx.start.positions[:rows])
    if len(out) != rows:
        return unsupported(node, "StepStart positions do not cover the kernel rows", rows=rows, positions=len(out))
    return out


def _slots(node: TreeNode, ctx: StepContext, rows: int, limit: int) -> tuple[int, ...] | Unsupported:
    if len(ctx.start.slot_mapping) < rows:
        return unsupported(node, "StepStart slot_mapping does not cover the kernel rows", rows=rows, slots=len(ctx.start.slot_mapping))
    out = tuple(int(s) for s in ctx.start.slot_mapping[:rows])
    if any(s < 0 or s >= limit for s in out):
        return unsupported(node, "a current-token slot is outside the cache", slots=out[:8], cache_slots=limit)
    return out


def _pin_topk(node: TreeNode, ctx: StepContext, topk: Any) -> Unsupported | None:
    if not isinstance(topk, ViewDesc) or topk.dtype != "int32" or topk.rank != 2 or topk.shape[1] != 2048 or topk.strides != (2048, 1):
        return unsupported(node, "topk_indices_ptr is not a contiguous i32 [capacity,2048] view", topk=repr(topk))
    state = _state(ctx)
    old = state["topk"]
    identity = (topk.alloc, topk.offset, topk.shape[1], topk.strides)
    if old is not None and old != identity:
        return unsupported(node, "IndexShare layer does not use the indexer layers' top-k allocation", observed=identity, established=old)
    state["topk"] = identity
    return None


@dataclass
class DsaNormRope:
    """The recorded ``_fused_norm_rope_kernel`` as four per-token computations."""

    name = "dsa.fused_norm_rope"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or triton_name(node) != NORM_KERNEL:
            return None
        a, c = bound_args(node), node.enter.constexprs
        if node.enter.src_sha != NORM_SRC_SHA or node.enter.num_warps != 4 or node.enter.num_stages != 3:
            return unsupported(node, "fused norm/RoPE source or launch geometry differs from the recorded GLM-5.3 pin",
                               src_sha=node.enter.src_sha, num_warps=node.enter.num_warps, num_stages=node.enter.num_stages)
        pins = {
            "Q_DIM": 2048, "Q_BLOCK_SIZE": 2048, "KV_DIM": 512, "KPE_HALF_ROT_DIM": 32,
            "MLA_CACHE_BLOCK_SIZE": 64, "MLA_CACHE_FP8": False, "MLA_CACHE_DS_MLA": False,
            "MLA_NUM_TILES": 1, "MLA_TILE_DIM": 1, "TOPK": 2048, "TOPK_BLOCK_SIZE": 1024,
            "INDEX_ROPE_INTERLEAVE": True,
        }
        bad = {k: (c.get(k), v) for k, v in pins.items() if c.get(k) != v}
        if bad:
            return unsupported(node, "fused norm/RoPE constexprs differ from the recorded GLM-5.3 pin", mismatches=bad)
        q, qo, qw = a.get("q_c_ptr"), a.get("q_c_out_ptr"), a.get("q_rms_norm_w_ptr")
        kv, kvw = a.get("kv_ptr"), a.get("kv_rms_norm_w_ptr")
        kpe, cs = a.get("kpe_ptr"), a.get("kpe_rope_cos_sin_cache_ptr")
        latent = a.get("mla_cache_ptr")
        if not all(isinstance(v, ViewDesc) for v in (q, qo, qw, kv, kvw, kpe, cs, latent)):
            return unsupported(node, "fused norm/RoPE pointer arguments are not bound", args={k: repr(v) for k, v in a.items() if k.endswith("_ptr")})
        M = q.shape[0]
        pos = _positions(node, ctx, a.get("pos_ptr"), M)
        if isinstance(pos, Unsupported):
            return pos
        if M != len(ctx.start.rows) or q.shape != (M, 2048) or qo.shape != (M, 2048) or kv.shape != (M, 512) or kpe.shape != (M, 64):
            return unsupported(node, "fused norm/RoPE rows or GLM-5.3 dimensions differ", q=repr(q), q_out=repr(qo), kv=repr(kv), kpe=repr(kpe),
                               step_rows=len(ctx.start.rows))
        if not (q.dtype == qo.dtype == qw.dtype == kv.dtype == kvw.dtype == kpe.dtype == cs.dtype == latent.dtype == "bfloat16"):
            return unsupported(node, "fused norm/RoPE is not entirely bf16")
        if qw.shape != (2048,) or kvw.shape != (512,) or cs.rank != 2 or cs.shape[1] != 64 or not is_row_major(cs):
            return unsupported(node, "norm weights or GPT-J rotary table have the wrong shape", qw=repr(qw), kvw=repr(kvw), cs=repr(cs))
        if any(p < 0 or p >= cs.shape[0] for p in pos):
            return unsupported(node, "a position is outside the GPT-J rotary table", positions=pos[:8], table_rows=cs.shape[0])
        try:
            latent_rows = _slot_rows(latent, 576)
        except GeometryError as e:
            return unsupported(node, str(e))
        slots = _slots(node, ctx, M, latent_rows.shape[0])
        if isinstance(slots, Unsupported):
            return slots
        kv_cache = ViewDesc(latent_rows.alloc, latent_rows.offset, (latent_rows.shape[0], 512), (576, 1), "bfloat16", latent_rows.device)
        kpe_cache = ViewDesc(latent_rows.alloc, latent_rows.offset + 512, (latent_rows.shape[0], 64), (576, 1), "bfloat16", latent_rows.device)
        try:
            qeps, kveps = float(a.get("q_rms_eps")), float(a.get("kv_rms_eps"))
        except (TypeError, ValueError):
            return unsupported(node, "norm epsilons are not static floats", q_eps=repr(a.get("q_rms_eps")), kv_eps=repr(a.get("kv_rms_eps")))
        parts: list[Compute] = [
            Compute(bind(DsaRMSNorm, N=2048, EPS=qeps), M, (PerRow(q), Shared(qw)), (PerRow(qo),), (node.id,),
                    hint=f"{_module(node)}/q_a_layernorm", kernels=(NORM_KERNEL,), attribution=tuple(ctx.start.rows), pattern=self.name,
                    roots=(None, _root(ctx, qw))),
            Compute(bind(DsaRMSNorm, N=512, EPS=kveps), M, (PerRow(kv), Shared(kvw)), (Gather(kv_cache, slots),), (),
                    hint=f"{_module(node)}/kv_a_layernorm", pattern=self.name, roots=(None, _root(ctx, kvw))),
            Compute(bind(RoPEHeadGptJV2, D=64), M, (PerRow(kpe), Gather(cs, pos)), (Gather(kpe_cache, slots),), (),
                    hint=f"{_module(node)}/k_pe", pattern=self.name),
        ]
        has = c.get("HAS_INDEXER") is True
        ms = _module_state(ctx, node)
        ms.update({"latent": latent_rows, "kv": kv_cache, "kpe": kpe_cache, "slots": slots})
        if has:
            ik, iw, ib, ics = (a.get(k) for k in ("index_k_ptr", "index_k_layer_norm_w_ptr", "index_k_layer_norm_bias_ptr",
                                                   "index_k_rope_cos_sin_cache_ptr"))
            ikc, isc = a.get("indexer_cache_ptr"), a.get("indexer_cache_scale_ptr")
            if not all(isinstance(v, ViewDesc) for v in (ik, iw, ib, ics, ikc, isc)):
                return unsupported(node, "indexer-layer norm/RoPE pointers are not bound")
            if c.get("INDEX_K_DIM") != 128 or c.get("INDEX_K_BLOCK_SIZE") != 128 or c.get("INDEX_K_HALF_ROT_DIM") != 32:
                return unsupported(node, "index-K dimensions differ from the recorded GLM-5.3 pin")
            if ik.shape != (M, 128) or iw.shape != (128,) or ib.shape != (128,) or ics.rank != 2 or ics.shape[1] != 64:
                return unsupported(node, "index-K inputs have the wrong shape", index_k=repr(ik), weight=repr(iw), bias=repr(ib), cs=repr(ics))
            try:
                ik_rows, is_rows = _slot_rows(ikc, 128), _index_scale_rows(isc)
            except GeometryError as e:
                return unsupported(node, str(e))
            if ik_rows.alloc != is_rows.alloc or 4 * is_rows.offset != ik_rows.offset + 128 or ik_rows.shape[0] != is_rows.shape[0]:
                return unsupported(node, "index-K values and scales are not the two views of one 132-byte slot layout")
            ieps = float(a.get("index_k_layer_norm_eps"))
            parts.append(Compute(bind(dsa.DsaIndexerKRow, D=128, ROPE=64, EPS=ieps), M,
                                 (PerRow(ik), Shared(iw), Shared(ib), Gather(ics, pos)),
                                 (Gather(ik_rows, slots), Gather(is_rows, slots)), (), hint=f"{_module(node)}/indexer_k",
                                 pattern=self.name, roots=(None, _root(ctx, iw), _root(ctx, ib), None)))
            ms.update({"index_k": ik_rows, "index_scale": is_rows})
        else:
            if c.get("INDEX_K_DIM") != 1 or c.get("INDEX_K_BLOCK_SIZE") != 1:
                return unsupported(node, "IndexShare layer has unexpected dummy index-K dimensions",
                                   INDEX_K_DIM=c.get("INDEX_K_DIM"), INDEX_K_BLOCK_SIZE=c.get("INDEX_K_BLOCK_SIZE"))
        bad_topk = _pin_topk(node, ctx, a.get("topk_indices_ptr"))
        if bad_topk is not None:
            return bad_topk
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


@dataclass
class DsaQ:
    """The recorded ``_fused_q_kernel`` as GPT-J RoPE and optional index-Q."""

    name = "dsa.fused_q"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or triton_name(node) != Q_KERNEL:
            return None
        a, c = bound_args(node), node.enter.constexprs
        if node.enter.src_sha != Q_SRC_SHA or node.enter.num_warps != 1 or node.enter.num_stages != 3:
            return unsupported(node, "fused-q source or launch geometry differs from the recorded GLM-5.3 pin",
                               src_sha=node.enter.src_sha, num_warps=node.enter.num_warps, num_stages=node.enter.num_stages)
        pins = {"NUM_Q_HEADS": 8, "Q_PE_HALF_ROT_DIM": 32, "QL_NOPE_DIM": 512, "QL_NOPE_BLOCK": 512,
                "INDEX_ROPE_INTERLEAVE": True, "QUANTIZE_MQA": False}
        bad = {k: (c.get(k), v) for k, v in pins.items() if c.get(k) != v}
        if bad:
            return unsupported(node, "fused-q constexprs differ from the recorded GLM-5.3 pin", mismatches=bad)
        q, qo, cs = a.get("q_pe_ptr"), a.get("q_pe_out_ptr"), a.get("q_pe_cos_sin_ptr")
        if not all(isinstance(v, ViewDesc) for v in (q, qo, cs)):
            return unsupported(node, "fused-q RoPE pointers are not bound")
        M = q.shape[0]
        pos = _positions(node, ctx, a.get("pos_ptr"), M)
        if isinstance(pos, Unsupported):
            return pos
        if q.shape != (M, 8, 64) or qo.shape != (M, 8, 64) or q.dtype != "bfloat16" or qo.dtype != "bfloat16" \
                or cs.dtype != "bfloat16" or cs.rank != 2 or cs.shape[1] != 64:
            return unsupported(node, "fused-q RoPE views differ from bf16 [M,8,64]", q=repr(q), q_out=repr(qo), cs=repr(cs))
        parts: list[Compute] = [
            Compute(bind(RoPEGptJV2, NHEADS=8, D=64), M, (PerRow(q), Gather(cs, pos)), (PerRow(qo),), (node.id,),
                    hint=f"{_module(node)}/q_pe", kernels=(Q_KERNEL,), attribution=tuple(ctx.start.rows), pattern=self.name)
        ]
        ms = _module_state(ctx, node)
        ms.update({"q_rope": qo, "q_v": a.get("ql_nope_ptr")})
        if c.get("HAS_INDEXER") is True:
            iq, ics, iw = a.get("index_q_ptr"), a.get("index_q_cos_sin_ptr"), a.get("index_weights_ptr")
            iqo, iwo = a.get("index_q_fp8_ptr"), a.get("index_weights_out_ptr")
            if not all(isinstance(v, ViewDesc) for v in (iq, ics, iw, iqo, iwo)):
                return unsupported(node, "indexer-layer fused-q pointers are not bound")
            if c.get("NUM_INDEX_Q_HEADS") != 32 or c.get("INDEX_Q_HEAD_DIM") != 128 or c.get("INDEX_Q_HALF_ROT_DIM") != 32:
                return unsupported(node, "index-Q dimensions differ from the recorded GLM-5.3 pin")
            if iq.shape != (M, 32, 128) or iw.shape != (M, 32) or iqo.shape != (M, 32, 128) or iwo.shape != (M, 32):
                return unsupported(node, "index-Q views have the wrong shape", q=repr(iq), weights=repr(iw), q_out=repr(iqo), weights_out=repr(iwo))
            ss, hs = float(a.get("index_weights_softmax_scale")), float(a.get("index_weights_head_scale"))
            if not math.isclose(ss, 128 ** -0.5, rel_tol=1e-15) or not math.isclose(hs, 32 ** -0.5, rel_tol=1e-15):
                return unsupported(node, "index-Q scales differ from D^-0.5 and H^-0.5", softmax_scale=ss, head_scale=hs)
            parts.append(Compute(bind(dsa.DsaIndexerQ, H=32, D=128, ROPE=64), M,
                                 (PerRow(iq), Gather(ics, pos), PerRow(iw)), (PerRow(iqo), PerRow(iwo)), (),
                                 hint=f"{_module(node)}/indexer_q", pattern=self.name))
            ms.update({"index_q": iqo, "index_weights": iwo, "mqa_after": node.id})
        else:
            if c.get("NUM_INDEX_Q_HEADS") != 1 or c.get("INDEX_Q_HEAD_DIM") != 1:
                return unsupported(node, "IndexShare layer has unexpected dummy index-Q dimensions")
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


@dataclass
class DsaMqaAllocation:
    """The f32 allocation DeepGEMM fills without an observer dispatch event."""

    name = "dsa.mqa_logits.invisible_deepgemm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or not op_name(node.op).startswith("empty"):
            return None
        ms = _module_state(ctx, node)
        if not ms.get("mqa_after") or node.id <= ms["mqa_after"]:
            return None
        outs = node.output_views()
        if len(outs) != 1:
            return None
        out = outs[0]
        M = len(ctx.start.rows)
        if out.dtype != "float32" or out.shape != (M, 512) or not is_row_major(out):
            return None
        q, w, kc, ks = (ms.get(k) for k in ("index_q", "index_weights", "index_k", "index_scale"))
        if not all(isinstance(v, ViewDesc) for v in (q, w, kc, ks)):
            return unsupported(node, "invisible MQA allocation lacks the fused-q or index-cache views", module=_module(node))
        parts: list[Compute] = []
        for i, (req, position) in enumerate(ctx.start.rows):
            n = int(position) + 1
            try:
                slots = kv_slots(ctx, req, n, layer=_module(node))
            except GeometryError as e:
                return unsupported(node, str(e), request=req, T=n)
            if n > out.shape[1] or any(s >= kc.shape[0] for s in slots):
                return unsupported(node, "MQA row or cache does not cover its valid keys", T=n, output_width=out.shape[1], cache_slots=kc.shape[0])
            ov = ViewDesc(out.alloc, out.offset + i * out.strides[0], (1, n), (out.strides[0], 1), out.dtype, out.device)
            parts.append(Compute(bind(dsa.DsaMqaLogitsV3, N=n, H=32, D=128), 1,
                                 (PerRow(_one_row(q, i)), PerRow(_one_row(w, i)), GatherMany(kc, (slots,)), GatherMany(ks, (slots,))),
                                 (PerRow(ov),), (node.id,) if not parts else (), hint=f"{_module(node)}/mqa[t={position}]",
                                 kernels=("fp8_mqa_logits", "fp8_paged_mqa_logits"), attribution=((req, int(position)),), pattern=self.name))
        ms.pop("mqa_after", None)
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


@dataclass
class DsaMqaGather:
    """DeepGEMM's K-cache gather is a staging copy; the MQA Definition reads the cache directly."""

    name = "dsa.mqa_cache_gather"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != MQA_STAGE:
            return None
        return Transparent("noop", tuple(sorted(node.ids())), hint=f"{_module(node)}/indexer K-cache staging",
                           kernels=(MQA_STAGE,), pattern=self.name)


@dataclass
class DsaTopK:
    """Both short-row top-k dispatches as ``DsaTopKShortRow_v1``."""

    name = "dsa.topk_short_row"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) not in (TOPK_PREFILL, TOPK_DECODE):
            return None
        a = bound_args(node)
        out = a.get("indices") if op_name(node.op) == TOPK_PREFILL else a.get("output")
        topk = a.get("topK") if op_name(node.op) == TOPK_PREFILL else a.get("k")
        if not isinstance(out, ViewDesc) or topk != 2048 or out.dtype != "int32" or out.shape != (len(ctx.start.rows), 2048) \
                or not is_row_major(out):
            return unsupported(node, "top-k output is not contiguous i32 [step rows,2048]", output=repr(out), topk=topk)
        bad = _pin_topk(node, ctx, out)
        if bad is not None:
            return bad
        parts = []
        kernel = "topKPerRowPrefill" if op_name(node.op) == TOPK_PREFILL else "cooperative_topk"
        for i, (req, position) in enumerate(ctx.start.rows):
            n = int(position) + 1
            if n > 2048:
                return unsupported(node, "DSA top-k row exceeds the short-row shortcut", position=position, keys=n, TOPK=2048)
            parts.append(Compute(bind(dsa.DsaTopKShortRow, TOPK=2048), 1, (Const(32, n, "len"),),
                                 (PerRow(_one_row(out, i)),), (node.id,) if not parts else (),
                                 hint=f"{_module(node)}/topk[t={position}]", kernels=(kernel,),
                                 attribution=((req, int(position)),), pattern=self.name))
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


@dataclass
class DsaIndexShare:
    """The pinned request-index to physical-slot copy between top-k and sparse FA3.

    The conversion is staging: ``SparseMlaSplit`` reads the selected latent-cache rows directly through ``GatherMany``.  This pattern
    nevertheless verifies that each layer's FA3 page table is produced from the one top-k allocation shared by full and IndexShare
    layers, then records that output for ``DsaSparseMla``.
    """

    name = "dsa.index_share_copy"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or node.enter.fn != INDEX_SHARE_KERNEL:
            return None
        a, c = bound_args(node), node.enter.constexprs
        pins = {
            "BLOCK_SIZE": 64, "BLOCK_STRIDE_ROWS": 64, "BLOCK_N": 2048, "HAS_PREFILL": False,
            "COUNT_VALID": True, "SINGLE_TILE": True, "COMPACT_TO_FRONT": True,
            "DCP_SIZE": 1, "DCP_RANK": 0, "DCP_INTERLEAVE": 1,
        }
        bad = {k: (c.get(k), v) for k, v in pins.items() if c.get(k) != v}
        if node.enter.src_sha != INDEX_SHARE_SRC_SHA or node.enter.num_warps != 8 or node.enter.num_stages != 3 or bad:
            return unsupported(node, "IndexShare conversion source or launch geometry differs from the recorded GLM-5.3 pin",
                               src_sha=node.enter.src_sha, num_warps=node.enter.num_warps, num_stages=node.enter.num_stages,
                               mismatches=bad)
        req, blocks, source, out, valid = (a.get(k) for k in
                                           ("req_id_ptr", "block_table_ptr", "token_indices_ptr", "out_ptr", "valid_count_ptr"))
        M = len(ctx.start.rows)
        if not all(isinstance(v, ViewDesc) for v in (req, blocks, source, out, valid)):
            return unsupported(node, "IndexShare conversion pointers are not bound")
        if req.shape != (M,) or req.dtype != "int32" or blocks.dtype != "int32" or blocks.rank != 2 \
                or source.shape != out.shape or source.shape != (M, 2048) or source.dtype != out.dtype or source.dtype != "int32" \
                or valid.shape != (M,) or valid.dtype != "int32" or not is_row_major(source) or not is_row_major(out):
            return unsupported(node, "IndexShare conversion views differ from the pinned i32 layouts",
                               req=repr(req), block_table=repr(blocks), source=repr(source), out=repr(out), valid=repr(valid))
        established = _state(ctx).get("topk")
        source_id = (source.alloc, source.offset, source.shape[1], source.strides)
        if established is None or source_id != established:
            return unsupported(node, "IndexShare conversion does not read the shared top-k allocation",
                               source=source_id, established=established)
        _module_state(ctx, node)["page_table"] = (out.alloc, out.offset, out.shape[1], out.strides)
        return Transparent("noop", tuple(sorted(node.ids())), hint=f"{_module(node)}/IndexShare top-k staging",
                           kernels=("ConvertReqIndexToGlobalIndexKernel",), pattern=self.name)


@dataclass
class DsaSparseMla:
    """FA3 sparse MLA at vLLM's per-row default split count."""

    name = "dsa.fa3_sparse_mla_split"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != FA3_OP or "_vllm_fa3_C" not in node.op:
            return None
        a, outs = bound_args(node), node.output_views()
        q, qv, k, v = a.get("q"), a.get("q_v"), a.get("k"), a.get("v")
        if not all(isinstance(x, ViewDesc) for x in (q, qv, k, v)) or not outs:
            return unsupported(node, "FA3 sparse MLA q/q_v/k/v/out pointers are not bound")
        out = outs[0]
        M, NH, DR, DV = q.shape[0], q.shape[1], q.shape[2], qv.shape[2]
        if (M, NH, DR, DV) != (len(ctx.start.rows), 8, 64, 512) or qv.shape != (M, NH, DV) or out.shape != (M, NH, DV):
            return unsupported(node, "FA3 sparse MLA views differ from [M,8,64]/[M,8,512]", q=repr(q), q_v=repr(qv), out=repr(out))
        if not (q.dtype == qv.dtype == k.dtype == v.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "FA3 sparse MLA views are not bf16")
        pins = {
            "k_new": None, "v_new": None, "out": None, "max_seqlen_q": 1, "max_seqlen_k": 2048,
            "softmax_scale": 0.0625, "is_causal": True, "window_size_left": -1, "window_size_right": -1,
            "softcap": 0.0, "is_rotary_interleaved": True, "scheduler_metadata": None, "pack_gqa": None,
            "sm_margin": 0, "s_aux": None, "cp_world_size": 1, "cp_rank": 0, "cp_tot_seqused_k": None,
        }
        bad = {key: (a.get(key), val) for key, val in pins.items() if a.get(key) != val}
        if bad:
            return unsupported(node, "FA3 sparse MLA arguments differ from the GLM-5.3 default path", mismatches=bad)
        for key in ("cu_seqlens_k", "cu_seqlens_k_new", "seqused_q", "kv_batch_idx", "leftpad_k", "rotary_cos", "rotary_sin",
                    "seqlens_rotary", "q_descale", "k_descale", "v_descale"):
            if a.get(key) is not None:
                return unsupported(node, f"FA3 sparse MLA has unsupported argument {key}", value=repr(a.get(key)))
        if k.alloc != v.alloc or k.offset != v.offset + 512 or k.shape[:3] != v.shape[:3] or k.strides[0] != 576 or v.strides[0] != 576:
            return unsupported(node, "FA3 k/v are not adjacent views of 576-word latent-cache rows", k=repr(k), v=repr(v))
        latent = ViewDesc(v.alloc, v.offset, (v.shape[0], 576), (576, 1), "bfloat16", v.device)
        ms = _module_state(ctx, node)
        if isinstance(ms.get("latent"), ViewDesc) and (ms["latent"].alloc, ms["latent"].offset) != (latent.alloc, latent.offset):
            return unsupported(node, "FA3 does not read the latent cache fused_norm_rope wrote", observed=repr(latent), established=repr(ms["latent"]))
        page = a.get("page_table")
        established = ms.get("page_table")
        page_id = (page.alloc, page.offset, page.shape[1], page.strides) if isinstance(page, ViewDesc) and page.rank == 2 else None
        if page_id is None or established is None or page_id != established:
            return unsupported(node, "FA3 page table is not the shared top-k allocation", page_table=repr(page), established=established)
        ns_arg = a.get("num_splits")
        if not isinstance(ns_arg, int) or ns_arg < 0:
            return unsupported(node, "FA3 num_splits is not a nonnegative integer", num_splits=ns_arg)
        valid = [int(position) + 1 for _, position in ctx.start.rows]
        tp = target_of(ctx)[0]
        if int(tp.num_sms) != 132:
            return unsupported(node, "GLM-5.3 sparse split heuristic is pinned to the record target's 132-SM launch", num_sms=tp.num_sms)
        splits = launch_num_splits(valid, int(tp.num_sms), 2048, ns_arg)
        parts = []
        for i, ((req, position), n, ns) in enumerate(zip(ctx.start.rows, valid, splits)):
            try:
                slots = kv_slots(ctx, req, n, layer=_module(node))
            except GeometryError as e:
                return unsupported(node, str(e), request=req, T=n)
            if any(s >= latent.shape[0] for s in slots):
                return unsupported(node, "a selected latent-cache slot is outside the cache", T=n, cache_slots=latent.shape[0])
            spec = bind(SparseMlaSplit, T=n, NS=ns, NH=NH, DR=DR, DV=DV, BN=SPARSE_MLA_GLM53["BN"],
                        DOT=SPARSE_MLA_GLM53["DOT"], INV=SPARSE_MLA_GLM53["INV"], SCALE=0.0625)
            kernels = ("FlashAttnFwdSm90",) + (("FlashAttnFwdCombine",) if ns > 1 else ())
            parts.append(Compute(spec, 1, (PerRow(_one_row(qv, i)), PerRow(_one_row(q, i)), GatherMany(latent, (slots,))),
                                 (PerRow(_one_row(out, i)),), (node.id,) if not parts else (),
                                 hint=f"{_module(node)}/sparse_mla[t={position},splits={ns}]", kernels=kernels,
                                 attribution=((req, int(position)),), pattern=self.name))
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=_module(node), pattern=self.name)


def dsa_patterns() -> list:
    """Patterns in dependency order, before generic transparent handlers; with them vLLM's default Hopper block-FP8 linear op
    (`fp8.DynamicDeepGemmFp8Linear`), which GLM-5.3's attention and dense projections run."""
    from verity_vllm.observe.fold.patterns.collectives import TpFusedAllReduceRmsNorm
    from verity_vllm.observe.fold.patterns.cublas import CublasLinear, MlaAbsorbBmm
    from verity_vllm.observe.fold.patterns.fp8 import DynamicDeepGemmFp8Linear
    from verity_vllm.observe.fold.patterns.norms import CudaFusedAddRmsNorm, CudaRmsNorm

    return [DsaNormRope(), DsaQ(), DsaMqaGather(), DsaMqaAllocation(), DsaTopK(), DsaIndexShare(), DsaSparseMla(), DynamicDeepGemmFp8Linear(),
            TpFusedAllReduceRmsNorm(world=8), CudaRmsNorm(), CudaFusedAddRmsNorm(), CublasLinear(), MlaAbsorbBmm()]


__all__ = [
    "DsaMqaAllocation",
    "DsaMqaGather",
    "DsaNormRope",
    "DsaQ",
    "DsaIndexShare",
    "DsaSparseMla",
    "DsaTopK",
    "dsa_patterns",
]
