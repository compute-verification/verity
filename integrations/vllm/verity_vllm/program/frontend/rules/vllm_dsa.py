"""[GLM-5.3] DeepSeek sparse attention as the H200 serves it (vLLM d9105ea80 `deepseek_v32/attention.py`, `FLASH_ATTN_MLA_SPARSE`),
over the three leaves `export_compat.dsa_leaves` exports in place of its kernels (`vllm_iface`):

    dsa_norm_rope(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps, k_pe, cos_sin, index_k?, index_k_w?, index_k_b?, index_k_eps,
                  index_cos_sin?, interleave, kv_c_out!, k_pe_out!) -> (q_c, index K e4m3 [T, D], index K scale f32 [T])
        `_fused_norm_rope_kernel` per token: `dsa_rmsnorm{N,EPS}` of q_c and of kv_c (kv_c_out rebound), `dsa_kv_rope{NHEADS=1,D}` of
        k_pe (k_pe_out rebound), and on an indexer layer `DsaIndexerKRow_v1{D,ROPE,EPS}` (the index-K cache slot)
    dsa_q(positions, q_pe, cos_sin, index_q?, index_cos_sin?, index_weights?, softmax_scale, head_scale, interleave)
        -> (q_pe roped [T, NH, R], index q e4m3 [T, H, D], index weights f32 [T, H])
        `_fused_q_kernel` per token: `dsa_q_rope{NHEADS,D}` of q_pe and on an indexer layer `DsaIndexerQ_v1{H,D,ROPE}`
    dsa_indexer_attn(positions, ql_nope, q_pe, kv_c, k_pe, index_q?, index_weights?, index_k?, index_k_scale?, layer_name, topk, scale)
        -> attention rows [T, NH, L]
        the layer's caches as history keyed by `layer_name` (rows [kv_c | k_pe], index-K slots and scales); per query row of n = P + 1
        keys, on an indexer layer `DsaMqaLogits_v3{N=n,H,D}` and `DsaTopKShortRow_v1{TOPK}` (n <= TOPK: the kernels' short-row
        shortcut, slot i = i), on an IndexShare layer the top-k the last indexer layer wrote for the same positions; then
        `SparseMla_v1{T=n,NH,DR,DV,BN,DOT,INV,SCALE}` over the history's first n rows

and two ops around them: the latent absorptions `bmm(q_nope^T, W_UK_T)` and `bmm(attn^T, W_UV)` (`batched_gemm{B,K,N,DOT}` per token)
and `output.copy_(rows)` (an allocation written whole).  A role the vocabulary lacks refuses by name.
"""

from __future__ import annotations

import numpy as np
import torch

from verity.ml.circuits.refs import Coll, PartLog, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.program.frontend.rules.base import InPlaceRule, OpRule, RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, active_target_profile
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.frontend.torch_frontend import _permute, _tensor_type
from verity_vllm.program.registry import dsa
from verity_vllm.program.registry.sparse_mla import SPARSE_MLA_GLM53

aten = torch.ops.aten
BF16, F32, E4M3 = torch.bfloat16, torch.float32, torch.float8_e4m3fn

#: FA3's sparse MLA launch geometry by (rope, latent) width: kBlockN and the epilogue at hdim 64 / hdimv 512 (`registry.sparse_mla`)
FA3_SPARSE_GEOMETRY = {(SPARSE_MLA_GLM53["DR"], SPARSE_MLA_GLM53["DV"]): {k: SPARSE_MLA_GLM53[k] for k in ("BN", "DOT", "INV")}}

#: the roles the vocabulary may lack, and which kernel each states
ROLE_KERNELS = {"dsa_rmsnorm": "q_a_layernorm / kv_a_layernorm in _fused_norm_rope_kernel (DsaRMSNorm_v1{N,EPS})",
                "dsa_kv_rope": "k_pe's interleaved RoPE in _fused_norm_rope_kernel",
                "dsa_q_rope": "q_pe's interleaved RoPE in _fused_q_kernel",
                "dsa_indexer_k_row": "the index-K slot of _fused_norm_rope_kernel (DsaIndexerKRow_v1)",
                "dsa_indexer_q": "the index query of _fused_q_kernel (DsaIndexerQ_v1)",
                "dsa_mqa_logits": "DeepGEMM's fp8 MQA logits (DsaMqaLogits_v3)",
                "dsa_topk_short_row": "the top-k of a row of at most TOPK keys (DsaTopKShortRow_v1)",
                "sparse_mla": "FA3's sparse MLA (SparseMla_v1)",
                "batched_gemm": "the latent absorption bmm, cuBLAS's batched bf16 GEMM (BatchedGemm_v1{B,K,N,DOT})",
                "const": "a constant word"}


def _args(node, names: tuple[str, ...]) -> dict:
    d = dict(zip(names, node.args))
    d.update(node.kwargs)
    return d


def _missing(ctx, roles) -> str | None:
    gone = [r for r in roles if not ctx.vocab.has(r)]
    if not gone:
        return None
    return "numerics_unregistered: the vocabulary has no " + "; no ".join(f"{r} ({ROLE_KERNELS[r]})" for r in gone)


def _value(ctx, a, dt, shp, what: str) -> str | None:
    """None when ``a`` is a value of dtype ``dt`` and shape ``shp`` (None entries: any length)."""
    if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
        return f"{what} is not a value"
    if ctx.dtype_of(a) != dt:
        return f"{what} is {ctx.dtype_of(a)}, not {dt}"
    s = shape(ctx, a)
    if s is None or len(s) != len(shp) or any(w is not None and w != v for w, v in zip(shp, s)):
        return f"{what} has shape {s}, not {list(shp)}"
    return None


def _alloc(ctx, a, shp, dt, what: str) -> str | None:
    if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Alloc):
        return f"{what} is not an unwritten allocation"
    if shape(ctx, a) != tuple(shp) or ctx.dtype_of(a) != dt:
        return f"{what} is {ctx.dtype_of(a)}{list(shape(ctx, a) or ())}, not {dt}{list(shp)}"
    return None


def _positions(ctx, a) -> list[int] | str:
    v = ctx.val(a) if isinstance(a, torch.fx.Node) else None
    if not isinstance(v, StaticIndex) or len(v.shape) != 1:
        return "positions is not a statically known 1-D index"
    return v.flat()


def _table(ctx, a, width: int, positions: list[int], what: str) -> str | None:
    """None when ``a`` is a bf16 rotary cache [MAX_POS, width] holding every position."""
    r = _value(ctx, a, BF16, (None, width), what)
    if r:
        return r
    n = shape(ctx, a)[0]
    if any(not 0 <= p < n for p in positions):
        return f"a position outside {what} ({n} rows)"
    return None


def _static_float(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _f32(x: float) -> float:
    return float(np.float32(x))


def _rows(ctx, a, w: int) -> Coll:
    return ctx.coll(a).reshape(_tensor_type(shape(ctx, a), w))


def _log(ctx, key, rows: list[Coll]) -> PartLog:
    """The rows of one history as a `PartLog` (one row per part), so that the rows a call reads are `log.cut(n)`."""
    logs = ctx.state.setdefault("dsa_logs", {})
    log = logs.get(key)
    if log is None or len(log) > len(rows) or any(p is not r.refs for p, r in zip(log.parts, rows)):
        log = PartLog()
    for r in rows[len(log):]:
        log.append(r.refs)
    logs[key] = log
    return log


class CopyIntoAllocRule(InPlaceRule):
    """`aten.copy_(dst, src)` with dst an unwritten allocation of src's shape and dtype: dst is src's rows (no gates)."""
    modes = frozenset({"profile"})
    info = RuleInfo("CopyIntoAlloc[vllm-dsa]", 1, "in-place-binding", applies_to="aten.copy_(Tensor! dst, src, non_blocking=False)",
                    preconditions=("dst is an unwritten allocation (`empty`, `empty_like`, `new_empty`)",
                                   "src is a value of dst's shape and dtype (no cast, no broadcast)"),
                    evidence_domain="structural: the allocation denotes the copied rows")

    def targets(self):
        return {aten.copy_.default}

    def applies(self, node, ctx):
        dst, src = node.args[0], node.args[1]
        if not isinstance(dst, torch.fx.Node) or not isinstance(ctx.val(dst), Alloc):
            return "dst is not an unwritten allocation"
        return _value(ctx, src, ctx.dtype_of(dst), shape(ctx, dst), "src") or True

    def emit(self, node, ctx):
        dst, src = node.args[0], node.args[1]
        ctx.rebind(dst, ctx.coll(src).reshape(ctx.val(dst).type), self, what="dst")
        ctx.define(node, ctx.env[dst], view_of=dst)
        ctx.record(self, node, {"shape": list(shape(ctx, dst)), "dtype": str(ctx.dtype_of(dst))}, emitted="(rows; no gates)")


_NORM_ROPE = ("positions", "q_c", "q_w", "q_eps", "kv_c", "kv_w", "kv_eps", "k_pe", "cos_sin", "index_k", "index_k_w", "index_k_b",
              "index_k_eps", "index_cos_sin", "index_rope_interleave", "kv_c_out", "k_pe_out")


class DsaNormRopeRule(InPlaceRule):
    """`vllm_iface.dsa_norm_rope` (`_fused_norm_rope_kernel`) per token: the q_c and kv_c RMSNorms, k_pe's RoPE and on an indexer
    layer the index-K slot; kv_c_out and k_pe_out are rebound, the node is (q_c normed, index K [T, D] e4m3, index K scale [T] f32)."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaNormRope[vllm-dsa]", 1, "in-place-binding", applies_to="vllm_iface.dsa_norm_rope(...) (" + ", ".join(_NORM_ROPE) + ")",
                    preconditions=("positions a static 1-D index of T rows; q_c [T, Nq], kv_c [T, L], k_pe [T, R] bf16 values",
                                   "q_w [Nq], kv_w [L] bf16; cos_sin a bf16 rotary cache [MAX_POS, R] holding every position",
                                   "kv_c_out, k_pe_out unwritten allocations of kv_c's and k_pe's shape",
                                   "the index operands all given or all None; given: index_k [T, D] bf16, k_norm weight and bias [D] f32, "
                                   "a bf16 rotary cache [MAX_POS, ROPE], the interleaved RoPE, D and ROPE as DsaIndexerKRow_v1 takes them",
                                   "the eps static floats; not under registered_values"),
                    evidence_domain="structural; numerics: dsa_rmsnorm, dsa_kv_rope and DsaIndexerKRow_v1 (registry.dsa)")

    def targets(self):
        return {iface.dsa_norm_rope.default}

    def applies(self, node, ctx):
        a = _args(node, _NORM_ROPE)
        has = a.get("index_k") is not None
        if (r := _missing(ctx, ["dsa_rmsnorm", "dsa_kv_rope"] + (["dsa_indexer_k_row"] if has else []))):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: the DSA norms take eps as a static; no registered-values twin"
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        qs, ks = shape(ctx, a["q_c"]), shape(ctx, a["kv_c"])
        R = (shape(ctx, a["k_pe"]) or (0, 0))[-1]
        for r in (_value(ctx, a["q_c"], BF16, (T, None), "q_c"), _value(ctx, a["kv_c"], BF16, (T, None), "kv_c"),
                  _value(ctx, a["k_pe"], BF16, (T, None), "k_pe")):
            if r:
                return r
        for r in (_value(ctx, a["q_w"], BF16, (qs[1],), "q_w"), _value(ctx, a["kv_w"], BF16, (ks[1],), "kv_w"),
                  _table(ctx, a["cos_sin"], R, pos, "cos_sin"), _alloc(ctx, a["kv_c_out"], ks, BF16, "kv_c_out"),
                  _alloc(ctx, a["k_pe_out"], (T, R), BF16, "k_pe_out")):
            if r:
                return r
        if not (_static_float(a["q_eps"]) and _static_float(a["kv_eps"])):
            return "q_eps / kv_eps are not static floats"
        ix = [a.get(k) for k in ("index_k", "index_k_w", "index_k_b", "index_cos_sin")]
        if any(x is None for x in ix) != (not has):
            return "the index operands are neither all given nor all None"
        if not has:
            return True
        D = (shape(ctx, a["index_k"]) or (0, 0))[-1]
        ROPE = (shape(ctx, a["index_cos_sin"]) or (0, 0))[-1]
        for r in (_value(ctx, a["index_k"], BF16, (T, D), "index_k"), _value(ctx, a["index_k_w"], F32, (D,), "index_k_w"),
                  _value(ctx, a["index_k_b"], F32, (D,), "index_k_b"), _table(ctx, a["index_cos_sin"], ROPE, pos, "index_cos_sin")):
            if r:
                return r
        if a.get("index_rope_interleave") is not True:
            return "numerics_unregistered: the index-K RoPE is not interleaved (DsaIndexerKRow_v1 states INDEX_ROPE_INTERLEAVE)"
        if not _static_float(a["index_k_eps"]):
            return "index_k_eps is not a static float"
        try:
            dsa.check_row_statics(D, ROPE)
            dsa.check_eps(float(a["index_k_eps"]))
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = _args(node, _NORM_ROPE)
        has = a.get("index_k") is not None
        pos = ctx.static_index(a["positions"]).flat()
        T, h = len(pos), ctx.hint_name(node)
        qc, kvc, kpe = _rows(ctx, a["q_c"], 16), _rows(ctx, a["kv_c"], 16), _rows(ctx, a["k_pe"], 16)
        Nq, L, R = shape(ctx, a["q_c"])[1], shape(ctx, a["kv_c"])[1], shape(ctx, a["k_pe"])[1]
        cs = ctx.coll(a["cos_sin"])
        qn = ctx.kind("dsa_rmsnorm", node, N=Nq, EPS=float(a["q_eps"]))
        kn = ctx.kind("dsa_rmsnorm", node, N=L, EPS=float(a["kv_eps"]))
        kr = ctx.kind("dsa_kv_rope", node, NHEADS=1, D=R)
        qw, kvw = ctx.coll(a["q_w"]), ctx.coll(a["kv_w"])
        q_out = [ctx.call(qn, qc[i], qw, node=node, rule=self, name=f"{h}/q_c[{i}]") for i in range(T)]
        kv_out = [ctx.call(kn, kvc[i], kvw, node=node, rule=self, name=f"{h}/kv_c[{i}]") for i in range(T)]
        pe_out = [ctx.call(kr, kpe[i].reshape(Array(1, Array(R, Value(16)))), cs[pos[i]], node=node, rule=self,
                           name=f"{h}/k_pe[{i}]").reshape(Array(R, Value(16))) for i in range(T)]
        ctx.rebind(a["kv_c_out"], array_of(kv_out), self, what="kv_c_out")
        ctx.rebind(a["k_pe_out"], array_of(pe_out), self, what="k_pe_out")
        ids = [qn.id, kn.id, kr.id]
        ik = iks = None
        pre = {"T": T, "positions": [pos[0], pos[-1]], "Nq": Nq, "L": L, "R": R, "q_eps": float(a["q_eps"]), "kv_eps": float(a["kv_eps"]),
               "indexer": has}
        if has:
            D, ROPE, EPS = shape(ctx, a["index_k"])[1], shape(ctx, a["index_cos_sin"])[1], float(a["index_k_eps"])
            fn = ctx.kind("dsa_indexer_k_row", node, D=D, ROPE=ROPE, EPS=EPS)
            k, w, b, ics = _rows(ctx, a["index_k"], 16), ctx.coll(a["index_k_w"]), ctx.coll(a["index_k_b"]), ctx.coll(a["index_cos_sin"])
            slots = [ctx.call(fn, k[i], w, b, ics[pos[i]], node=node, rule=self, name=f"{h}/index_k[{i}]") for i in range(T)]
            ik, iks = array_of([s[0] for s in slots]), array_of([s[1] for s in slots])
            ctx.bind_weight_role(a["index_k_w"], self, "dsa.indexer.k_norm.weight", D=D)
            ctx.bind_weight_role(a["index_k_b"], self, "dsa.indexer.k_norm.bias", D=D)
            ctx.bind_weight_role(a["index_cos_sin"], self, "dsa.indexer.rope.cos_sin_cache", D=ROPE)
            ids.append(fn.id)
            pre.update(D=D, ROPE=ROPE, index_k_eps=EPS, index_rope_interleave=True)
        ctx.define(node, [array_of(q_out), ik, iks])
        ctx.bind_weight_role(a["q_w"], self, "dsa.q_a_layernorm.weight", N=Nq)
        ctx.bind_weight_role(a["kv_w"], self, "dsa.kv_a_layernorm.weight", N=L)
        ctx.bind_weight_role(a["cos_sin"], self, "rope.cos_sin_cache", D=R)
        ctx.record(self, node, pre, emitted=" + ".join(f"{i} x{T}" for i in ids))


_Q = ("positions", "q_pe", "cos_sin", "index_q", "index_cos_sin", "index_weights", "softmax_scale", "head_scale", "index_rope_interleave")


class DsaQRule(OpRule):
    """`vllm_iface.dsa_q` (`_fused_q_kernel`) per token: q_pe's RoPE and on an indexer layer `DsaIndexerQ_v1{H,D,ROPE}`; the node is
    (q_pe roped [T, NH, R], index q [T, H, D] e4m3, index weights [T, H] f32)."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaQ[vllm-dsa]", 1, "op-binding", applies_to="vllm_iface.dsa_q(...) (" + ", ".join(_Q) + ")",
                    preconditions=("positions a static 1-D index of T rows; q_pe [T, NH, R] bf16; cos_sin a bf16 rotary cache [MAX_POS, R]",
                                   "the index operands all given or all None; given: index_q [T, H, D] bf16, index_weights [T, H] bf16, a bf16 "
                                   "rotary cache [MAX_POS, ROPE], the interleaved RoPE",
                                   "softmax_scale == f32(D^-0.5) and head_scale == f32(H^-0.5), the words DsaIndexerQ_v1 folds in"),
                    evidence_domain="structural; numerics: dsa_q_rope and DsaIndexerQ_v1 (registry.dsa)")

    def targets(self):
        return {iface.dsa_q.default}

    def applies(self, node, ctx):
        a = _args(node, _Q)
        has = a.get("index_q") is not None
        if (r := _missing(ctx, ["dsa_q_rope"] + (["dsa_indexer_q"] if has else []))):
            return r
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        if (r := _value(ctx, a["q_pe"], BF16, (T, None, None), "q_pe")):
            return r
        if (r := _table(ctx, a["cos_sin"], shape(ctx, a["q_pe"])[2], pos, "cos_sin")):
            return r
        ix = [a.get(k) for k in ("index_q", "index_cos_sin", "index_weights")]
        if any(x is None for x in ix) != (not has):
            return "the index operands are neither all given nor all None"
        if not has:
            return True
        H, D = (shape(ctx, a["index_q"]) or (0, 0, 0))[1:3]
        ROPE = (shape(ctx, a["index_cos_sin"]) or (0, 0))[-1]
        for r in (_value(ctx, a["index_q"], BF16, (T, H, D), "index_q"), _value(ctx, a["index_weights"], BF16, (T, H), "index_weights"),
                  _table(ctx, a["index_cos_sin"], ROPE, pos, "index_cos_sin")):
            if r:
                return r
        if a.get("index_rope_interleave") is not True:
            return "numerics_unregistered: the index-q RoPE is not interleaved (DsaIndexerQ_v1 states INDEX_ROPE_INTERLEAVE)"
        ss, hs = a["softmax_scale"], a["head_scale"]
        if not (_static_float(ss) and _static_float(hs) and _f32(ss) == _f32(D ** -0.5) and _f32(hs) == _f32(H ** -0.5)):
            return (f"numerics_unregistered: softmax_scale {ss!r} / head_scale {hs!r} are not f32(D^-0.5) / f32(H^-0.5) at D={D}, H={H}, "
                    f"the words DsaIndexerQ_v1 folds in")
        try:
            dsa.check_row_statics(D, ROPE)
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = _args(node, _Q)
        has = a.get("index_q") is not None
        pos = ctx.static_index(a["positions"]).flat()
        T, h = len(pos), ctx.hint_name(node)
        _, NH, R = shape(ctx, a["q_pe"])
        qpe, cs = _rows(ctx, a["q_pe"], 16), ctx.coll(a["cos_sin"])
        qr = ctx.kind("dsa_q_rope", node, NHEADS=NH, D=R)
        q_out = array_of([ctx.call(qr, qpe[i], cs[pos[i]], node=node, rule=self, name=f"{h}/q_pe[{i}]") for i in range(T)])
        ids, iq, iw = [qr.id], None, None
        pre = {"T": T, "positions": [pos[0], pos[-1]], "NH": NH, "R": R, "indexer": has}
        if has:
            _, H, D = shape(ctx, a["index_q"])
            ROPE = shape(ctx, a["index_cos_sin"])[1]
            fn = ctx.kind("dsa_indexer_q", node, H=H, D=D, ROPE=ROPE)
            q, w, ics = _rows(ctx, a["index_q"], 16), _rows(ctx, a["index_weights"], 16), ctx.coll(a["index_cos_sin"])
            outs = [ctx.call(fn, q[i], ics[pos[i]], w[i], node=node, rule=self, name=f"{h}/index_q[{i}]") for i in range(T)]
            iq, iw = array_of([o[0] for o in outs]), array_of([o[1] for o in outs])
            ctx.bind_weight_role(a["index_cos_sin"], self, "dsa.indexer.rope.cos_sin_cache", D=ROPE)
            ids.append(fn.id)
            pre.update(H=H, D=D, ROPE=ROPE, softmax_scale=float(a["softmax_scale"]), head_scale=float(a["head_scale"]))
        ctx.define(node, [q_out, iq, iw])
        ctx.bind_weight_role(a["cos_sin"], self, "rope.cos_sin_cache", D=R)
        ctx.record(self, node, pre, emitted=" + ".join(f"{i} x{T}" for i in ids))


_ATTN = ("positions", "ql_nope", "q_pe", "kv_c", "k_pe", "index_q", "index_weights", "index_k", "index_k_scale", "layer_name", "topk", "scale")


def _observed_sparse_mla(layer_name: str) -> str | None:
    """None when the layer's OBSERVED implementation is the one `SparseMla_v1` states (`vllm_meta.attention_impls`)."""
    obs = (_OBSERVED["attention_impls"] or {}).get(layer_name)
    if obs is None:
        return f"no observed attention implementation for {layer_name!r} (vllm_meta.attention_impls)"
    if obs.get("kind") != "mla" or not str(obs.get("impl", "")).endswith(".FlashAttnMLASparseImpl"):
        return f"numerics_unregistered: {layer_name!r} runs {obs.get('impl')!r}; SparseMla_v1 states FLASH_ATTN_MLA_SPARSE"
    if obs.get("layer_kv_cache_dtype") not in ("auto", "bfloat16"):
        return f"numerics_unregistered: {layer_name!r} has KV cache dtype {obs.get('layer_kv_cache_dtype')!r}; SparseMla_v1 reads a bf16 cache"
    if obs.get("dcp_world_size") not in (None, 1) or obs.get("use_pcp"):
        return f"numerics_unregistered: {layer_name!r} runs under context parallelism"
    if obs.get("prefill_backend") is not None:
        return (f"numerics_unregistered: {layer_name!r} has the dense-MHA prefill backend {obs['prefill_backend']!r}: a prefill of at most "
                f"TOPK keys would take it, and no Definition states it")
    return None


class DsaIndexerAttnRule(StateRule):
    """`vllm_iface.dsa_indexer_attn` (the sparse indexer and FA3's sparse MLA of one layer): the layer's MLA and index-K caches are
    implicit state keyed by `layer_name`, made explicit as histories; each query row at position P reads the first n = P + 1 rows.
    An indexer layer emits the row's logits and its top-k, an IndexShare layer takes the top-k the last indexer layer selected for
    the same positions, and both `SparseMla_v1` over the selected rows: with n <= TOPK the selection is rows 0..n-1 in order."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaIndexerAttn[vllm-dsa]", 1, "state", applies_to="vllm_iface.dsa_indexer_attn(...) (" + ", ".join(_ATTN) + ")",
                    preconditions=("the layer's OBSERVED implementation is FlashAttnMLASparseImpl on a bf16 cache, without context parallelism "
                                   "or a dense-MHA prefill backend (so every step takes the top-k MQA path)",
                                   "the target is sm_90 and FA3's sparse MLA is pinned to one split (engine.sparse_mla, VERITY_SPARSE_MLA_NUM_SPLITS=1)",
                                   "positions are the layer's next T rows; ql_nope [T, NH, L], q_pe [T, NH, R], kv_c [T, L], k_pe [T, R] bf16",
                                   "the index operands all given (index_q [T, H, D] e4m3, index_weights [T, H] f32, index_k [T, D] e4m3, "
                                   "index_k_scale [T] f32) or all None, then the last indexer layer selected for these positions",
                                   "every row has at most TOPK keys (the top-k kernels' short-row shortcut)",
                                   "layer_name static and written from one module; not under registered_values"),
                    evidence_domain="structural; numerics: DsaMqaLogits_v3, DsaTopKShortRow_v1, SparseMla_v1 (registry.dsa, registry.sparse_mla)")

    def targets(self):
        return {iface.dsa_indexer_attn.default}

    def applies(self, node, ctx):
        a = _args(node, _ATTN)
        has = a.get("index_q") is not None
        layer, topk, scale = a["layer_name"], a["topk"], a["scale"]
        if not isinstance(layer, str) or not isinstance(topk, int) or topk < 1 or not _static_float(scale):
            return "layer_name, topk, scale are not static"
        if (r := _missing(ctx, ["sparse_mla"] + (["dsa_mqa_logits", "dsa_topk_short_row", "const"] if has else []))):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: SparseMla_v1 takes its scale as a static; no registered-values twin"
        if (r := _observed_sparse_mla(layer)):
            return r
        tp = active_target_profile()
        if tp.compute_capability != (9, 0):
            return f"numerics_unregistered: SparseMla_v1 is FA3 on sm_90; the target is {tp.capability_str}"
        from verity_vllm.engine import sparse_mla as SM
        try:
            splits = SM.wanted()
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        if splits != SM.SPLITS:
            return (f"numerics_unregistered: FA3's sparse MLA at vLLM's num_splits heuristic ({SM.ENV} unset or 0): the split-combine has no "
                    f"Definition; SparseMla_v1 is num_splits=1")
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        L = (shape(ctx, a["kv_c"]) or (0, 0))[-1]
        R = (shape(ctx, a["k_pe"]) or (0, 0))[-1]
        NH = (shape(ctx, a["ql_nope"]) or (0, 0, 0))[1]
        for r in (_value(ctx, a["ql_nope"], BF16, (T, NH, L), "ql_nope"), _value(ctx, a["q_pe"], BF16, (T, NH, R), "q_pe"),
                  _value(ctx, a["kv_c"], BF16, (T, L), "kv_c"), _value(ctx, a["k_pe"], BF16, (T, R), "k_pe")):
            if r:
                return r
        if (R, L) not in FA3_SPARSE_GEOMETRY:
            return f"numerics_unregistered: FA3's sparse MLA launch at rope {R}, latent {L} is not registered ({sorted(FA3_SPARSE_GEOMETRY)})"
        ix = [a.get(k) for k in ("index_q", "index_weights", "index_k", "index_k_scale")]
        if any(x is None for x in ix) != (not has):
            return "the index operands are neither all given nor all None"
        if has:
            H, D = (shape(ctx, a["index_q"]) or (0, 0, 0))[1:3]
            for r in (_value(ctx, a["index_q"], E4M3, (T, H, D), "index_q"), _value(ctx, a["index_weights"], F32, (T, H), "index_weights"),
                      _value(ctx, a["index_k"], E4M3, (T, D), "index_k"), _value(ctx, a["index_k_scale"], F32, (T,), "index_k_scale")):
                if r:
                    return r
            try:
                dsa.check_logit_statics(H, D)
            except ValueError as e:
                return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = _args(node, _ATTN)
        has = a.get("index_q") is not None
        layer, TOPK, scale = a["layer_name"], a["topk"], float(a["scale"])
        mp, h = ctx.module_path(node), ctx.hint_name(node)
        pos = ctx.static_index(a["positions"]).flat()
        T = len(pos)
        _, NH, L = shape(ctx, a["ql_nope"])
        R = shape(ctx, a["k_pe"])[1]
        owner = ctx.state.setdefault("dsa_owner", {})
        if owner.setdefault(layer, mp) != mp:
            refuse("side_input", node, mp, f"the caches of {layer!r} are written from two modules ({owner[layer]} and {mp})")
        hist = ctx.state.setdefault("dsa_history", {}).setdefault(layer, {"latent": [], "ik": [], "iks": []})
        P0 = len(hist["latent"])
        if pos != list(range(P0, P0 + T)):
            refuse("shape", node, mp, f"positions {pos[0]}..{pos[-1]} are not {layer!r}'s next rows {P0}..{P0 + T - 1}")
        if has != bool(hist["ik"]) and P0:
            refuse("side_input", node, mp, f"{layer!r} has an index-K cache at some steps only")
        kvc, kpe = _rows(ctx, a["kv_c"], 16), _rows(ctx, a["k_pe"], 16)
        for i in range(T):
            hist["latent"].append(Coll(Array(L + R, Value(16)), concat([kvc[i].refs, kpe[i].refs])))
        latent = _log(ctx, (layer, "latent"), hist["latent"])
        G = FA3_SPARSE_GEOMETRY[(R, L)]
        too_long = [p for p in pos if p + 1 > TOPK]
        if too_long:
            refuse("numerics_unregistered", node, mp, f"{layer!r}: position {too_long[0]} has {too_long[0] + 1} keys > TOPK {TOPK}: the top-k "
                                                     f"kernels' radix selection has no Definition (DsaTopKShortRow_v1 states rows of at most TOPK)")
        if has:
            H, D = shape(ctx, a["index_q"])[1:3]
            iq, iw = _rows(ctx, a["index_q"], 8), _rows(ctx, a["index_weights"], 32)
            ik, iks = _rows(ctx, a["index_k"], 8), _rows(ctx, a["index_k_scale"], 32)
            for i in range(T):
                hist["ik"].append(ik[i])
                hist["iks"].append(iks[i])
            klog, slog = _log(ctx, (layer, "ik"), hist["ik"]), _log(ctx, (layer, "iks"), hist["iks"])
            sel_fn = ctx.kind("dsa_topk_short_row", node, TOPK=TOPK)
            selected = {}
            for i, P in enumerate(pos):
                n = P + 1
                lg = ctx.kind("dsa_mqa_logits", node, N=n, H=H, D=D)
                ctx.call(lg, iq[i], iw[i], Coll(Array(n, Array(D, Value(8))), klog.cut(n)), Coll(Array(n, Value(32)), slog.cut(n)),
                         node=node, rule=self, name=f"{h}/logits[t={P}]")
                ln = ctx.call(ctx.kind("const", node, W=32, BITS=n), node=node, rule=self, name=f"{h}/len[t={P}]")
                selected[P] = ctx.call(sel_fn, ln, node=node, rule=self, name=f"{h}/topk[t={P}]")
            ctx.state["dsa_topk"] = {"layer": layer, "rows": selected}
            source = layer
        else:
            last = ctx.state.get("dsa_topk")
            if last is None or any(P not in last["rows"] for P in pos):
                refuse("side_input", node, mp, f"IndexShare layer {layer!r} reads a top-k no indexer layer selected for positions "
                                               f"{pos[0]}..{pos[-1]}")
            source = last["layer"]
        ql, qr = _rows(ctx, a["ql_nope"], 16), _rows(ctx, a["q_pe"], 16)
        outs = []
        for i, P in enumerate(pos):
            n = P + 1
            attn = ctx.kind("sparse_mla", node, T=n, NH=NH, DR=R, DV=L, BN=G["BN"], DOT=G["DOT"], INV=G["INV"], SM_SCALE=scale)
            outs.append(ctx.call(attn, ql[i], qr[i], Coll(Array(n, Array(L + R, Value(16))), latent.cut(n)), node=node, rule=self,
                                 name=f"{h}[t={P}]"))
        ctx.define(node, array_of(outs))
        from verity_vllm.engine import sparse_mla as SM
        ctx.record(self, node, {"layer_name": layer, "positions": [pos[0], pos[-1]], "history_rows_after": P0 + T, "NH": NH, "DR": R, "DV": L,
                                "BN": G["BN"], "DOT": G["DOT"].name, "INV": G["INV"].name, "scale": scale, "TOPK": TOPK, "indexer": has,
                                "top_k_from": source, "selection": "rows 0..n-1 in order (n <= TOPK: DsaTopKShortRow_v1's slots)",
                                "num_splits": SM.SPLITS, "num_splits_switch": SM.ENV,
                                "attention_impl_observed": (_OBSERVED["attention_impls"] or {}).get(layer)},
                   emitted=f"{attn.id} x{T}" + (f" + DsaMqaLogits_v3, DsaTopKShortRow_v1 x{T}" if has else f" (top-k of {source})"))


class MlaAbsorbBmmRule(OpRule):
    """`aten.bmm(x [B, M, K] bf16, W [B, K, N] bf16 parameter)`, MLA's latent absorptions (`W_UK_T`, `W_UV`): per token m,
    `batched_gemm{B,K,N,DOT}(x[:, m, :], W^T)` over all B heads; the node is [B, M, N]."""
    modes = frozenset({"profile"})
    info = RuleInfo("MlaAbsorbBmm[vllm-dsa]", 1, "op-binding", applies_to="aten.bmm(x [B, M, K], W [B, K, N]) with W a parameter",
                    preconditions=("x a bf16 value [B, M, K]; W a bf16 parameter [B, K, N] (W_UK_T or W_UV)",
                                   "DOT the target's k16 step (frontend.targets.tensor_core_dot)"),
                    evidence_domain="structural; numerics: batched_gemm (cuBLAS's batched bf16 GEMM)")

    def targets(self):
        return {aten.bmm.default}

    def applies(self, node, ctx):
        x, w = node.args[0], node.args[1]
        if ctx.weight_of(w) is None:
            return "the right operand is not a parameter"
        if (r := _missing(ctx, ["batched_gemm"])):
            return r
        if (r := _value(ctx, x, BF16, (None, None, None), "x")):
            return r
        B, _, K = shape(ctx, x)
        if (r := _value(ctx, w, BF16, (B, K, None), "W")):
            return r
        from verity_vllm.program.frontend import targets as TG
        try:
            TG.tensor_core_dot(active_target_profile())
        except TG.UnregisteredTarget as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets as TG
        x, w = node.args[0], node.args[1]
        B, M, K = shape(ctx, x)
        N = shape(ctx, w)[2]
        DOT = TG.tensor_core_dot(active_target_profile())
        fn = ctx.kind("batched_gemm", node, B=B, K=K, N=N, DOT=DOT)
        where = ctx.module_path(node)
        xa = _permute(_rows(ctx, x, 16), (1, 0, 2), where)
        wt = _permute(_rows(ctx, w, 16), (0, 2, 1), where)
        h = ctx.hint_name(node)
        outs = [ctx.call(fn, xa[m], wt, node=node, rule=self, name=f"{h}[{m}]") for m in range(M)]
        ctx.define(node, _permute(array_of(outs), (1, 0, 2), where))
        wl = ctx.weight_of(w)
        ctx.bind_weight_role(w, self, "mla.absorb", B=B, K=K, N=N)
        ctx.record(self, node, {"B": B, "M": M, "K": K, "N": N, "DOT": DOT.name, "weight": wl.name}, emitted=f"{fn.id} x{M}")


VLLM_DSA_RULES = [CopyIntoAllocRule(), DsaNormRopeRule(), DsaQRule(), DsaIndexerAttnRule(), MlaAbsorbBmmRule()]

__all__ = ["CopyIntoAllocRule", "DsaIndexerAttnRule", "DsaNormRopeRule", "DsaQRule", "FA3_SPARSE_GEOMETRY", "MlaAbsorbBmmRule",
           "VLLM_DSA_RULES"]
