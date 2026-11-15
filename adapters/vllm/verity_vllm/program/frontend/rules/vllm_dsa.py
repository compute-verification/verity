"""[GLM-5.3] DeepSeek sparse attention as the H200 serves it (vLLM d9105ea80 `deepseek_v32/attention.py`, `FLASH_ATTN_MLA_SPARSE`),
over the three leaves `export_compat.dsa_leaves` exports in place of its kernels (`vllm_iface`):

    dsa_norm_rope(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps, k_pe, cos_sin, index_k?, index_k_w?, index_k_b?, index_k_eps,
                  index_cos_sin?, interleave, kv_c_out!, k_pe_out!) -> (q_c, index K e4m3 [T, D], index K scale f32 [T])
        `_fused_norm_rope_kernel` per token: `dsa_rmsnorm{N,EPS}` of q_c and of kv_c (kv_c_out rebound), `dsa_kv_rope{NHEADS=1,D}` of
        k_pe (k_pe_out rebound), and on an indexer layer `DsaIndexerKRow_v1{D,ROPE,EPS}` (the index-K cache slot); in class G's
        batched form (`batched_form`) the two are one `DsaLatentRow_v1{KVL,R,EPS}` per token, its row [kv_c | k_pe]
    dsa_q(positions, q_pe, cos_sin, index_q?, index_cos_sin?, index_weights?, softmax_scale, head_scale, interleave)
        -> (q_pe roped [T, NH, R], index q e4m3 [T, H, D], index weights f32 [T, H])
        `_fused_q_kernel` per token: `dsa_q_rope{NHEADS,D}` of q_pe and on an indexer layer `DsaIndexerQ_v1{H,D,ROPE}`
    dsa_indexer_attn(positions, ql_nope, q_pe, kv_c, k_pe, index_q?, index_weights?, index_k?, index_k_scale?, layer_name, topk, scale)
        -> attention rows [T, NH, L]
        the layer's caches as history keyed by `layer_name` (rows [kv_c | k_pe], index-K slots and scales); per query row of n = P + 1
        keys, on an indexer layer `DsaMqaLogits_v3{N=n,H,D}` and `DsaTopKShortRow_v1{TOPK}` (n <= TOPK: the kernels' short-row
        shortcut, slot i = i), or past TOPK, at the row's class N (`long_row_keys`: n rounded up to a multiple of TOPK, the keys
        past n `PAD_SCORE`), the logits' top-TOPK set (`DsaTopKMask_v1{N,TOPK}`) and the concentrator's settings for it
        (`DsaButterflySettings_v1{N,K=TOPK}`); on an IndexShare layer the top-k the last indexer layer wrote for the same
        positions; then explicitly pinned `SparseMla_v1{T,...}` or default `SparseMlaSplit_v1{T,NS,...}` over the history's first n
        rows (T = n), or past TOPK over the rows `DsaButterflyRows_v1{N,K=TOPK,W,E=16}` routes at those settings, in ascending
        index order (T = TOPK; `DsaButterflyRowsPad_v1{T=n,N,...}` where n < N).  In the batched form a call's rows past TOPK go in
        blocks (`long_row_blocks`), each over the history's first N_b rows: `DsaMqaLogits_v3{N=N_b}`, `DsaTopKMaskCounted_v1{N,TOPK}`
        of the logits and the row's own key count, the settings, and the rows routed from N_b (`DsaButterflyRowsPad_v1{T=N_b,N}`),
        each op over the block's rows in turn; under `dsa_table_reads` (`Registered.dsa_table_reads`) the mask's positions
        `DsaTopKIndex_v1{N,TOPK}` in place of the settings, which an IndexShare layer reads as well, and the rows read at them
        from the history's first N_b, `DsaHistoryReads_v1{V=N_b,TOPK,H}` (`program.dsa_table`), in place of the rows network

and `output.copy_(rows)` (an allocation written whole); the latent absorptions around them are `vllm_dsa_absorb.py`'s.  A role the
vocabulary lacks refuses by name.  The leaf the shim exports in dsa_indexer_attn's place on a prefill vLLM serves by dense MHA,
`dsa_dense_mha`, is `vllm_dsa_dense.py`'s; it opens the same histories.
"""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import torch
from verity.circuits.refs import Coll, PartLog, Refs, array_of, concat
from verity.circuits.types import Array, Value
from verity.definitions.registry import dsa
from verity.definitions.registry.sparse_mla import SPARSE_MLA_GLM53
from verity.definitions.registry.sparse_mla_split import launch_num_splits

from verity_vllm.program.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.program.frontend.rules.base import (
    InPlaceRule,
    OpRule,
    RuleInfo,
    StateRule,
)
from verity_vllm.program.frontend.rules.common import known, refuse_cached_prefix, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import (
    _OBSERVED,
    active_target_profile,
    batched_form,
    dsa_table_reads,
    launch_keys_of,
)
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.frontend.torch_frontend import _tensor_type

aten = torch.ops.aten
BF16, F32, E4M3 = torch.bfloat16, torch.float32, torch.float8_e4m3fn

#: FA3's sparse MLA launch geometry by (rope, latent) width: kBlockN and the epilogue at hdim 64 / hdimv 512 (`registry.sparse_mla`)
FA3_SPARSE_GEOMETRY = {(SPARSE_MLA_GLM53["DR"], SPARSE_MLA_GLM53["DV"]): {k: SPARSE_MLA_GLM53[k] for k in ("BN", "DOT", "INV")}}

#: `FlashAttnMLASparseMetadataBuilder`'s decode threshold by query heads per rank, and its default (vLLM d9105ea80
#: flashattn_mla_sparse.py:149; a speculative config only raises it): a call of more query rows is a prefill
DECODE_THRESHOLD, DECODE_THRESHOLD_DEFAULT = {16: 128, 32: 128, 64: 256, 128: 256}, 256

#: the roles the vocabulary may lack, and which kernel each states
ROLE_KERNELS = {"dsa_rmsnorm": "q_a_layernorm / kv_a_layernorm in _fused_norm_rope_kernel (DsaRMSNorm_v1{N,EPS})",
                "dsa_kv_rope": "k_pe's interleaved RoPE in _fused_norm_rope_kernel",
                "dsa_q_rope": "q_pe's interleaved RoPE in _fused_q_kernel",
                "dsa_indexer_k_row": "the index-K slot of _fused_norm_rope_kernel (DsaIndexerKRow_v1)",
                "dsa_indexer_q": "the index query of _fused_q_kernel (DsaIndexerQ_v1)",
                "dsa_mqa_logits": "DeepGEMM's fp8 MQA logits (DsaMqaLogits_v3)",
                "dsa_topk_short_row": "the top-k of a row of at most TOPK keys (DsaTopKShortRow_v1)",
                "dsa_topk_mask": "the top-TOPK set of a row past TOPK keys, a mask of its logits (DsaTopKMask_v1)",
                "dsa_topk_mask_counted": "the same over a block's keys, of the row's own (DsaTopKMaskCounted_v1)",
                "dsa_latent_row": "kv_a_layernorm and k_pe's interleaved RoPE in _fused_norm_rope_kernel as one row (DsaLatentRow_v1)",
                "dsa_butterfly_settings": "the concentrator's settings for a row's top-TOPK mask (DsaButterflySettings_v1)",
                "dsa_butterfly_rows": "the history rows sparse MLA reads past TOPK keys, routed in ascending index order at the "
                                      "concentrator's settings (DsaButterflyRows_v1)",
                "dsa_butterfly_rows_pad": "the same at a row whose keys are no multiple of TOPK: its history and the pad to its "
                                          "size class two arguments (DsaButterflyRowsPad_v1)",
                "dsa_topk_index": "the table-read form's positions of a row's top-TOPK mask, ascending (DsaTopKIndex_v1)",
                "dsa_history_reads": "the table-read form's TOPK history rows sparse MLA reads, one read each at those positions "
                                     "(DsaHistoryReads_v1)",
                "sparse_mla": "FA3's sparse MLA (SparseMla_v1)",
                "sparse_mla_split": "FA3's sparse MLA at vLLM's default split count (SparseMlaSplit_v1)",
                "attention_dot": "FA3's dense causal prefill attention, FlashAttnPrefillBackend's (Attention_v4; AttentionSplit_v1 "
                                 "where FA3 splits it)",
                "batched_gemm": "the latent absorption bmm, cuBLAS's batched bf16 GEMM (BatchedGemm_v1{B,K,N,DOT})",
                "const": "a constant word"}

#: the score of a key past a long row's own (`long_row_keys`): its `dsa_topk.topk_key` is 0, the least, and a real key it ties
#: with has the lower index, so every real key precedes it and the top-TOPK set of a row of at least TOPK keys never holds one
PAD_SCORE = 0xFFFFFFFF


def long_row_keys(n: int, topk: int) -> int:
    """The class a row of n > topk keys is selected over: n rounded up to a multiple of topk, so a request's rows past TOPK share
    one specialization of the mask form per class (each of its Definitions' bodies grows with its N) at under TOPK keys more a
    row.  The keys past n score `PAD_SCORE` and their history rows are zero: no output of the concentrator reads one."""
    return -(-int(n) // int(topk)) * int(topk)


#: the multiple a call's rows past TOPK round their keys up to in the batched form, a block's rows sharing one (`long_row_blocks`)
BLOCK_KEYS = 256


def long_row_blocks(keys: list[int], topk: int) -> list[tuple[int, list[int]]]:
    """A call's rows past `topk` keys (`keys` in row order, increasing) in the batched form, as (the block's keys, its rows'
    indices): a row's keys rounded up to a multiple of `BLOCK_KEYS`, at most the call's last row's, so a block's rows share one
    specialization of each long-row Definition and batch (`prune.batched_form`), each row selecting among its own keys alone
    (`DsaTopKMaskCounted`)."""
    out: list[tuple[int, list[int]]] = []
    for i, n in enumerate(keys):
        if n > topk:
            N = min(-(-n // BLOCK_KEYS) * BLOCK_KEYS, keys[-1])
            if out and out[-1][0] == N:
                out[-1][1].append(i)
            else:
                out.append((N, [i]))
    return out


def pad_words(topk: int, words: int = 1) -> int:
    """The words of a Program's one block of a pad (a batch of one constant, sliced at each long row, so a row's pad is one run
    of references, not one per word): the most keys a row's class adds (`long_row_keys`), topk - 1, of ``words`` words each."""
    return (int(topk) - 1) * int(words)


def _args(node, names: tuple[str, ...]) -> dict:
    d = dict(zip(names, node.args))
    d.update(node.kwargs)
    return d


def _missing(ctx, roles, kernels: dict = ROLE_KERNELS) -> str | None:
    gone = [r for r in roles if not ctx.vocab.has(r)]
    if not gone:
        return None
    return "numerics_unregistered: the vocabulary has no " + "; no ".join(f"{r} ({kernels[r]})" for r in gone)


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


def _table(ctx, a, width: int, positions: list[int], what: str, dt=BF16) -> str | None:
    """None when ``a`` is a rotary cache [MAX_POS, width] of ``dt`` holding every position."""
    r = _value(ctx, a, dt, (None, width), what)
    if r:
        return r
    n = known(shape(ctx, a))[0]
    if any(not 0 <= p < n for p in positions):
        return f"a position outside {what} ({n} rows)"
    return None


def _static_float(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _f32(x: float) -> float:
    return float(np.float32(x))


def _rows(ctx, a, w: int) -> Coll:
    return ctx.coll(a).reshape(_tensor_type(known(shape(ctx, a)), w))


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
        ctx.record(self, node, {"shape": list(known(shape(ctx, dst))), "dtype": str(ctx.dtype_of(dst))}, emitted="(rows; no gates)")


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
                                   ("the index operands all given or all None; given: index_k [T, D] bf16, k_norm weight and bias [D] f32, "
                                   "a bf16 rotary cache [MAX_POS, ROPE], the interleaved RoPE, D and ROPE as DsaIndexerKRow_v1 takes them"),
                                   "the eps static floats; not under registered_values"),
                    evidence_domain="structural; numerics: dsa_rmsnorm, dsa_kv_rope and DsaIndexerKRow_v1 (registry.dsa)")

    def targets(self):
        return {iface.dsa_norm_rope.default}

    def applies(self, node, ctx):
        a = _args(node, _NORM_ROPE)
        has = a.get("index_k") is not None
        if (r := _missing(ctx, ["dsa_rmsnorm", "dsa_latent_row" if batched_form() else "dsa_kv_rope"] + (["dsa_indexer_k_row"] if has else []))):
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
        for r in (_value(ctx, a["q_w"], BF16, (known(qs)[1],), "q_w"), _value(ctx, a["kv_w"], BF16, (known(ks)[1],), "kv_w"),
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
        Nq, L, R = known(shape(ctx, a["q_c"]))[1], known(shape(ctx, a["kv_c"]))[1], known(shape(ctx, a["k_pe"]))[1]
        cs = ctx.coll(a["cos_sin"])
        qn = ctx.kind("dsa_rmsnorm", node, N=Nq, EPS=float(a["q_eps"]))
        qw, kvw = ctx.coll(a["q_w"]), ctx.coll(a["kv_w"])
        q_out = [ctx.call(qn, qc[i], qw, node=node, rule=self, name=f"{h}/q_c[{i}]") for i in range(T)]
        if batched_form():
            lr = ctx.kind("dsa_latent_row", node, KVL=L, R=R, EPS=float(a["kv_eps"]))
            lat = [ctx.call(lr, Coll(Array(L + R, Value(16)), concat([kvc[i].refs, kpe[i].refs])), kvw, cs[pos[i]], node=node, rule=self,
                            name=f"{h}/latent[{i}]") for i in range(T)]
            kv_out = [Coll(Array(L, Value(16)), r.refs.slice(0, L)) for r in lat]
            pe_out = [Coll(Array(R, Value(16)), r.refs.slice(L, L + R)) for r in lat]
            ids = [qn.id, lr.id]
        else:
            kn = ctx.kind("dsa_rmsnorm", node, N=L, EPS=float(a["kv_eps"]))
            kr = ctx.kind("dsa_kv_rope", node, D=R)
            kv_out = [ctx.call(kn, kvc[i], kvw, node=node, rule=self, name=f"{h}/kv_c[{i}]") for i in range(T)]
            pe_out = [ctx.call(kr, kpe[i], cs[pos[i]], node=node, rule=self, name=f"{h}/k_pe[{i}]") for i in range(T)]
            ids = [qn.id, kn.id, kr.id]
        ctx.rebind(a["kv_c_out"], array_of(kv_out), self, what="kv_c_out")
        ctx.rebind(a["k_pe_out"], array_of(pe_out), self, what="k_pe_out")
        ik = iks = None
        pre = {"T": T, "positions": [pos[0], pos[-1]], "Nq": Nq, "L": L, "R": R, "q_eps": float(a["q_eps"]), "kv_eps": float(a["kv_eps"]),
               "indexer": has}
        if has:
            D, ROPE, EPS = known(shape(ctx, a["index_k"]))[1], known(shape(ctx, a["index_cos_sin"]))[1], float(a["index_k_eps"])
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
                                   ("the index operands all given or all None; given: index_q [T, H, D] bf16, index_weights [T, H] bf16, a bf16 "
                                   "rotary cache [MAX_POS, ROPE], the interleaved RoPE"),
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
        if (r := _table(ctx, a["cos_sin"], known(shape(ctx, a["q_pe"]))[2], pos, "cos_sin")):
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
        _, NH, R = known(shape(ctx, a["q_pe"]))
        qpe, cs = _rows(ctx, a["q_pe"], 16), ctx.coll(a["cos_sin"])
        qr = ctx.kind("dsa_q_rope", node, D=R)
        q_out = array_of([array_of([ctx.call(qr, qpe[i][j], cs[pos[i]], node=node, rule=self, name=f"{h}/q_pe[{i},{j}]")
                                     for j in range(NH)]) for i in range(T)])
        ids, iq, iw = [qr.id], None, None
        pre = {"T": T, "positions": [pos[0], pos[-1]], "NH": NH, "R": R, "indexer": has}
        if has:
            _, H, D = known(shape(ctx, a["index_q"]))
            ROPE = known(shape(ctx, a["index_cos_sin"]))[1]
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
_DENSE = ("positions", "q_nope", "q_pe", "k_nope", "v", "kv_c", "k_pe", "index_k", "index_k_scale", "layer_name", "topk", "scale")


def _observed_sparse_mla(layer_name: str) -> str | None:
    """None when the observed implementation is the registered FA3 sparse path, at its default split or the explicit one-split pin."""
    obs = (_OBSERVED["attention_impls"] or {}).get(layer_name)
    if obs is None:
        return f"no observed attention implementation for {layer_name!r} (vllm_meta.attention_impls)"
    if obs.get("kind") != "mla" or not str(obs.get("impl", "")).endswith(".FlashAttnMLASparseImpl"):
        return f"numerics_unregistered: {layer_name!r} runs {obs.get('impl')!r}; SparseMla_v1 states FLASH_ATTN_MLA_SPARSE"
    if obs.get("layer_kv_cache_dtype") not in ("auto", "bfloat16"):
        return f"numerics_unregistered: {layer_name!r} has KV cache dtype {obs.get('layer_kv_cache_dtype')!r}; SparseMla_v1 reads a bf16 cache"
    if obs.get("dcp_world_size") not in (None, 1) or obs.get("use_pcp"):
        return f"numerics_unregistered: {layer_name!r} runs under context parallelism"
    if obs.get("num_splits") not in (None, 1):
        return f"numerics_unregistered: {layer_name!r} passes unsupported num_splits={obs.get('num_splits')!r} to FA3 sparse MLA"
    return None


def _decode_threshold(layer_name: str) -> int | str:
    """The metadata builder's decode threshold at the layer's observed query heads per rank: a call of more query rows is a prefill."""
    heads = _OBSERVED["attention_impls"][layer_name].get("q_heads_per_rank")
    if not isinstance(heads, int):
        return (f"no observed query heads per rank for {layer_name!r}, the decode threshold its prefill backend "
                f"{_OBSERVED['attention_impls'][layer_name].get('prefill_backend')!r} is behind")
    return DECODE_THRESHOLD.get(heads, DECODE_THRESHOLD_DEFAULT)


def _prefill_by_dense_mha(layer_name: str, T: int) -> str | None:
    """None when a call of T query rows takes the top-k MQA path (`MLAAttention.forward_impl`) as a decode: the layer has no dense-MHA
    prefill backend, or `sparse_mla_force_mqa`, or the call is at most the decode threshold.  A longer call is a prefill, which the
    prefill backend serves by dense MHA while its rows have at most TOPK keys (`vllm_iface.dsa_dense_mha`, `DsaDenseMha[vllm-dsa]`);
    one that reaches this op has rows past TOPK, which vLLM serves by sparse MQA over the indexer's prefill top-k."""
    obs = _OBSERVED["attention_impls"][layer_name]
    if obs.get("prefill_backend") is None or obs.get("sparse_mla_force_mqa") is True:
        return None
    limit = _decode_threshold(layer_name)
    if isinstance(limit, str) or limit >= T:
        return limit if isinstance(limit, str) else None
    return (f"numerics_unregistered: {layer_name!r}: a call of {T} query rows is a prefill (decode threshold {limit} at "
            f"{obs.get('q_heads_per_rank')} query heads per rank) that reached the top-k MQA op; vLLM serves a prefill by its prefill "
            f"backend {obs.get('prefill_backend')!r}'s dense MHA while its rows have at most TOPK keys (vllm_iface.dsa_dense_mha) and by "
            f"sparse MQA over the indexer's prefill top-k past them, which no Definition states")


def _open_history(ctx, node, layer: str, pos: list[int], has: bool) -> tuple[dict, int]:
    """The layer's history (rows [kv_c | k_pe], index-K slots and scales) and its length before this call, once the call passes what
    every call of the layer must: one module writes the layer's caches, the positions continue the history, and the layer has an
    index-K cache at every step or at none."""
    mp = ctx.module_path(node)
    owner = ctx.state.setdefault("dsa_owner", {})
    if owner.setdefault(layer, mp) != mp:
        refuse("side_input", node, mp, f"the caches of {layer!r} are written from two modules ({owner[layer]} and {mp})")
    hist = ctx.state.setdefault("dsa_history", {}).setdefault(layer, {"latent": [], "ik": [], "iks": [], "next": None})
    P0 = len(hist["latent"])
    if pos != list(range(pos[0], pos[0] + len(pos))) or (P0 and pos[0] != hist["next"]):
        refuse("shape", node, mp, f"positions {pos[0]}..{pos[-1]} do not continue {layer!r}'s history"
                                  + (f" of {P0} rows (next position {hist['next']})" if P0 else " (not consecutive)"))
    if has != bool(hist["ik"]) and P0:
        refuse("side_input", node, mp, f"{layer!r} has an index-K cache at some steps only")
    hist["next"] = pos[-1] + 1
    return hist, P0


def _append_rows(ctx, hist: dict, a: dict, has: bool) -> None:
    """Append the call's cache rows to the layer's history: [kv_c | k_pe] per row, and on an indexer layer its index-K slot and scale."""
    kvc, kpe = _rows(ctx, a["kv_c"], 16), _rows(ctx, a["k_pe"], 16)
    T, L, R = kvc.type.n, known(shape(ctx, a["kv_c"]))[1], known(shape(ctx, a["k_pe"]))[1]
    for i in range(T):
        hist["latent"].append(Coll(Array(L + R, Value(16)), concat([kvc[i].refs, kpe[i].refs])))
    if has:
        ik, iks = _rows(ctx, a["index_k"], 8), _rows(ctx, a["index_k_scale"], 32)
        for i in range(T):
            hist["ik"].append(ik[i])
            hist["iks"].append(iks[i])


class DsaIndexerAttnRule(StateRule):
    """`vllm_iface.dsa_indexer_attn` (the sparse indexer and FA3's sparse MLA of one layer): the layer's MLA and index-K caches are
    implicit state keyed by `layer_name`, made explicit as histories.  A layer's history opens at its first call, at whatever position
    (the step wrapper's `pos0`, as the dense attention rule's `kv_history` opens), and every later call continues it; row i of a call
    that finds P0 rows reads the first n = P0 + i + 1 rows.  An indexer layer emits the row's logits and its top-k, an IndexShare layer takes the top-k the last indexer layer selected for
    the same positions, and both the pinned one-split or default split sparse MLA over the selected rows: with n <= TOPK the
    selection is rows 0..n-1 in order.  Past TOPK the circuit computes it from the logits at the row's class N (`long_row_keys`,
    the logits past n `PAD_SCORE` and the history rows past n zero): `DsaTopKMask_v1{N,TOPK}` marks the row's top-TOPK set,
    `DsaButterflySettings_v1{N,K=TOPK}` sets an order-preserving concentrator of the N rows for it, and sparse MLA reads the TOPK
    rows `DsaButterflyRows_v1{N,K=TOPK}` routes at those settings, in ascending index order (the row `VERITY_DSA_TOPK=canonical`
    serves); an IndexShare layer reads the indexer layer's settings, so the wiring is static.  Below N it reads them through
    `DsaButterflyRowsPad_v1{T=n,N}`, the history and its pad two arguments, so the history is a cut one row longer than the row
    before's, which the descriptor writes as an alias plus a part.  Under `dsa_table_reads` the batched form's long rows read
    their history by table instead: the indexer layer's `DsaTopKIndex_v1{N,TOPK}` of the mask is what an IndexShare layer reads,
    and each layer's `DsaHistoryReads_v1{V=N_b,TOPK,H}` reads the block's history at it (`program.dsa_table`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaIndexerAttn[vllm-dsa]", 1, "state", applies_to="vllm_iface.dsa_indexer_attn(...) (" + ", ".join(_ATTN) + ")",
                    preconditions=(("the layer's OBSERVED implementation is FlashAttnMLASparseImpl on a bf16 cache, without context parallelism; "
                                   "the workload either declares the one-split pin or leaves FA3's split heuristic at its default"),
                                   ("the call takes the top-k MQA path as a decode: no dense-MHA prefill backend, sparse_mla_force_mqa, or T "
                                   "at most the metadata builder's decode threshold at the observed query heads per rank (DECODE_THRESHOLD); "
                                   "a prefill of at most TOPK keys is DsaDenseMha[vllm-dsa]'s"),
                                   "the target's compute capability is 9.0 (FA3)",
                                   "positions are T consecutive rows continuing the layer's history; ql_nope [T, NH, L], q_pe [T, NH, R], kv_c [T, L], k_pe [T, R] bf16",
                                   ("the index operands all given (index_q [T, H, D] e4m3, index_weights [T, H] f32, index_k [T, D] e4m3, "
                                   "index_k_scale [T] f32) or all None, then the last indexer layer selected for these positions"),
                                   ("a row of at most TOPK keys takes the top-k kernels' short-row shortcut; a longer one reads the rows "
                                   "DsaButterflyRows_v1 (DsaButterflyRowsPad_v1 below its class) routes at the settings "
                                   "DsaButterflySettings_v1 computes for DsaTopKMask_v1's set, or under dsa_table_reads the rows "
                                   "DsaHistoryReads_v1 reads at DsaTopKIndex_v1's positions of that set "
                                   "(an IndexShare layer's rows have the indexer layer's key counts)"),
                                   "layer_name static and written from one module; not under registered_values"),
                    evidence_domain="structural; numerics: DsaMqaLogits_v3, DsaTopKShortRow_v1, DsaTopKMask_v1, DsaButterflySettings_v1, "
                                    "DsaButterflyRows_v1, DsaButterflyRowsPad_v1, DsaTopKIndex_v1, DsaHistoryReads_v1, SparseMla_v1 or "
                                    "SparseMlaSplit_v1")

    def targets(self):
        return {iface.dsa_indexer_attn.default}

    def applies(self, node, ctx):
        a = _args(node, _ATTN)
        has = a.get("index_q") is not None
        layer, topk, scale = a["layer_name"], a["topk"], a["scale"]
        if not isinstance(layer, str) or not isinstance(topk, int) or topk < 1 or not _static_float(scale):
            return "layer_name, topk, scale are not static"
        obs = (_OBSERVED["attention_impls"] or {}).get(layer) or {}
        sparse_role = "sparse_mla" if obs.get("num_splits") == 1 else "sparse_mla_split"
        if (r := _missing(ctx, [sparse_role] + (["dsa_mqa_logits", "dsa_topk_short_row", "const"] if has else []))):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: SparseMla_v1 takes its scale as a static; no registered-values twin"
        if (r := _observed_sparse_mla(layer)):
            return r
        tp = active_target_profile()
        if tp.compute_capability != (9, 0):
            return f"numerics_unregistered: SparseMla_v1 is FA3 at compute capability 9.0; the target is {tp.capability_str}"
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        if (r := _prefill_by_dense_mha(layer, T)):
            return r
        hist = (ctx.state.get("dsa_history") or {}).get(layer)
        table = batched_form() and dsa_table_reads()
        long_roles = (["dsa_history_reads"] + (["dsa_topk_mask_counted", "dsa_topk_index"] if has else [])) if table else (
            ["dsa_butterfly_rows", "dsa_butterfly_rows_pad"]
            + (["dsa_topk_mask_counted" if batched_form() else "dsa_topk_mask", "dsa_butterfly_settings"] if has else []))
        if (len(hist["latent"]) if hist else 0) + T > topk and (r := _missing(ctx, long_roles)):
            return r
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
        refuse_cached_prefix(ctx, node, self.rule_id)
        a = _args(node, _ATTN)
        has = a.get("index_q") is not None
        layer, TOPK, scale = a["layer_name"], a["topk"], float(a["scale"])
        mp, h = ctx.module_path(node), ctx.hint_name(node)
        pos = ctx.static_index(a["positions"]).flat()
        T = len(pos)
        _, NH, L = known(shape(ctx, a["ql_nope"]))
        R = known(shape(ctx, a["k_pe"]))[1]
        hist, P0 = _open_history(ctx, node, layer, pos, has)
        keys = [P0 + i + 1 for i in range(T)]
        _append_rows(ctx, hist, a, has)
        latent = _log(ctx, (layer, "latent"), hist["latent"])
        G = FA3_SPARSE_GEOMETRY[(R, L)]
        obs = (_OBSERVED["attention_impls"] or {}).get(layer) or {}
        pinned_one = obs.get("num_splits") == 1
        if pinned_one:
            splits = [1] * T
        else:
            # one launch over every token of the step, each with its top-k's valid keys: a token's split count is a function of
            # that count given the launch
            batch = [min(n, TOPK) for n in launch_keys_of(node, keys)]
            by_keys = dict(zip(batch, launch_num_splits(batch, int(active_target_profile().num_sms), TOPK, int(obs.get("num_splits") or 0))))
            splits = [by_keys[min(n, TOPK)] for n in keys]
        past = [P for P, n in zip(pos, keys) if n > TOPK]
        pads = ctx.state.setdefault("dsa_pads", {})

        def pad(w: int, bits: int, words: int, count: int) -> Refs:
            if (w, bits) not in pads:
                pads[(w, bits)] = ctx.B.batch(ctx.kind("const", node, W=w, BITS=bits), axes=(), n=words, name=f"{h}/pad{w}").refs
                k = len(ctx.B.nodes) - 1
                ctx.node_hints[k], ctx.node_sources[k], ctx.node_fx[k] = f"{h}/pad{w}", node.name, node
            return pads[(w, bits)].slice(0, count)

        blocks = long_row_blocks(keys, TOPK) if batched_form() else []
        table = bool(blocks) and dsa_table_reads()
        if has:
            H, D = known(shape(ctx, a["index_q"]))[1:3]
            iq, iw = _rows(ctx, a["index_q"], 8), _rows(ctx, a["index_weights"], 32)
            klog, slog = _log(ctx, (layer, "ik"), hist["ik"]), _log(ctx, (layer, "iks"), hist["iks"])
            sel_fn = ctx.kind("dsa_topk_short_row", node, TOPK=TOPK)
            selected = {}
            for i, (P, n) in enumerate(zip(pos, keys)):
                if blocks and n > TOPK:
                    break
                lg = ctx.kind("dsa_mqa_logits", node, N=n, H=H, D=D)
                logits = ctx.call(lg, iq[i], iw[i], Coll(Array(n, Array(D, Value(8))), klog.cut(n)), Coll(Array(n, Value(32)), slog.cut(n)),
                                  node=node, rule=self, name=f"{h}/logits[t={P}]")
                if n <= TOPK:
                    ln = ctx.call(ctx.kind("const", node, W=32, BITS=n), node=node, rule=self, name=f"{h}/len[t={P}]")
                    selected[P] = ctx.call(sel_fn, ln, node=node, rule=self, name=f"{h}/topk[t={P}]")
                    continue
                N = long_row_keys(n, TOPK)
                scores = Coll(Array(N, Value(32)), concat([logits.refs, pad(32, PAD_SCORE, pad_words(TOPK), N - n)]) if n < N else logits.refs)
                mask = ctx.call(ctx.kind("dsa_topk_mask", node, N=N, TOPK=TOPK), scores, node=node, rule=self,
                                name=f"{h}/topk_mask[t={P}]")
                selected[P] = ctx.call(ctx.kind("dsa_butterfly_settings", node, N=N, TOPK=TOPK), mask, node=node, rule=self,
                                       name=f"{h}/topk[t={P}]")
            for Nb, rows in blocks:
                N, lg = long_row_keys(Nb, TOPK), ctx.kind("dsa_mqa_logits", node, N=Nb, H=H, D=D)
                ik, iks = Coll(Array(Nb, Array(D, Value(8))), klog.cut(Nb)), Coll(Array(Nb, Value(32)), slog.cut(Nb))
                logits = [ctx.call(lg, iq[i], iw[i], ik, iks, node=node, rule=self, name=f"{h}/logits[t={pos[i]}]") for i in rows]
                counts = [Coll(Value(32), pad(32, keys[i], 1, 1)) for i in rows]
                tail = [pad(32, PAD_SCORE, pad_words(TOPK), N - Nb)] if Nb < N else []
                fm = ctx.kind("dsa_topk_mask_counted", node, N=N, TOPK=TOPK)
                fs = ctx.kind("dsa_topk_index" if table else "dsa_butterfly_settings", node, N=N, TOPK=TOPK)
                masks = [ctx.call(fm, Coll(Array(N, Value(32)), concat([lgt.refs, *tail])), c, node=node, rule=self,
                                  name=f"{h}/topk_mask[t={pos[i]}]") for i, lgt, c in zip(rows, logits, counts)]
                for i, m in zip(rows, masks):
                    selected[pos[i]] = ctx.call(fs, m, node=node, rule=self, name=f"{h}/topk[t={pos[i]}]")
            ctx.state["dsa_topk"] = {"layer": layer, "rows": selected, "keys": dict(zip(pos, keys))}
            source = layer
        else:
            last = ctx.state.get("dsa_topk")
            if last is None or any(P not in last["rows"] for P in pos):
                refuse("side_input", node, mp, f"IndexShare layer {layer!r} reads a top-k no indexer layer selected for positions "
                                               f"{pos[0]}..{pos[-1]}")
            odd = [(P, n) for P, n in zip(pos, keys) if n > TOPK and last["keys"][P] != n]
            if odd:
                refuse("side_input", node, mp, f"IndexShare layer {layer!r} has {odd[0][1]} keys at position {odd[0][0]}, but "
                                               f"{last['layer']!r} selected its top-{TOPK} among {last['keys'][odd[0][0]]}")
            selected, source = last["rows"], last["layer"]
        ql, qr = _rows(ctx, a["ql_nope"], 16), _rows(ctx, a["q_pe"], 16)
        row = Array(L + R, Value(16))
        rows_of = {}
        for Nb, rows in blocks:
            N, hb = long_row_keys(Nb, TOPK), Coll(Array(Nb, row), latent.cut(Nb))
            rest: tuple[Coll, ...]
            if table:
                fr, rest = ctx.kind("dsa_history_reads", node, V=Nb, TOPK=TOPK, H=L + R), ()
            else:
                rest = (Coll(Array(N - Nb, row), pad(16, 0, pad_words(TOPK, L + R), (N - Nb) * (L + R))),) if Nb < N else ()
                fr = (ctx.kind("dsa_butterfly_rows_pad", node, T=Nb, N=N, TOPK=TOPK, W=L + R) if rest
                      else ctx.kind("dsa_butterfly_rows", node, N=N, TOPK=TOPK, W=L + R))
            for i in rows:
                rows_of[i] = ctx.call(fr, selected[pos[i]], hb, *rest, node=node, rule=self, name=f"{h}/rows[t={pos[i]}]")
        outs = []
        for i, (P, n, ns) in enumerate(zip(pos, keys, splits)):
            role = "sparse_mla" if pinned_one else "sparse_mla_split"
            statics = {"T": min(n, TOPK), "NH": NH, "DR": R, "DV": L, "BN": G["BN"], "DOT": G["DOT"], "INV": G["INV"], "SM_SCALE": scale}
            if not pinned_one:
                statics["NS"] = ns
            attn = ctx.kind(role, node, **statics)
            kv = rows_of[i] if i in rows_of else Coll(Array(n, row), latent.cut(n))
            if n > TOPK and i not in rows_of:
                N = long_row_keys(n, TOPK)
                rest = (Coll(Array(N - n, row), pad(16, 0, pad_words(TOPK, L + R), (N - n) * (L + R))),) if n < N else ()
                rows = (ctx.kind("dsa_butterfly_rows_pad", node, T=n, N=N, TOPK=TOPK, W=L + R) if rest
                        else ctx.kind("dsa_butterfly_rows", node, N=N, TOPK=TOPK, W=L + R))
                kv = ctx.call(rows, selected[P], kv, *rest, node=node, rule=self, name=f"{h}/rows[t={P}]")
            outs.append(ctx.call(attn, ql[i], qr[i], kv, node=node, rule=self, name=f"{h}[t={P}]"))
        ctx.define(node, array_of(outs))
        ctx.record(self, node, {"layer_name": layer, "positions": [pos[0], pos[-1]], "history_rows_after": P0 + T, "NH": NH, "DR": R, "DV": L,
                                "BN": G["BN"], "DOT": G["DOT"].name, "INV": G["INV"].name, "scale": scale, "TOPK": TOPK, "indexer": has,
                                "top_k_from": source, "selection": "rows 0..n-1 in order (n <= TOPK: DsaTopKShortRow_v1's slots); past "
                                "TOPK DsaTopKMask_v1's set of the logits at the row's class (long_row_keys), its rows in ascending index "
                                + ("order read at DsaTopKIndex_v1's positions by DsaHistoryReads_v1 (the history table's reads)" if table else
                                   "order by DsaButterflyRows_v1 (DsaButterflyRowsPad_v1 below the class) at DsaButterflySettings_v1's settings"),
                                "past_topk": past, "attention_impl_observed": (_OBSERVED["attention_impls"] or {}).get(layer)},
                   emitted=f"{attn.id} x{T} (splits {splits})" +
                           (f" + DsaMqaLogits_v3, DsaTopKShortRow_v1 x{T - len(past)}" if has else f" (top-k of {source})") +
                           (f" + {('DsaTopKMaskCounted_v1' if blocks else 'DsaTopKMask_v1') + (', DsaTopKIndex_v1, ' if table else ', DsaButterflySettings_v1, ') if has else ''}"
                            f"{'DsaHistoryReads_v1' if table else 'DsaButterflyRows[Pad]_v1'} x{len(past)}"
                            if past else ""))


VLLM_DSA_RULES = [CopyIntoAllocRule(), DsaNormRopeRule(), DsaQRule(), DsaIndexerAttnRule()]

__all__ = ["CopyIntoAllocRule", "DsaIndexerAttnRule", "DsaNormRopeRule", "DsaQRule", "FA3_SPARSE_GEOMETRY", "VLLM_DSA_RULES",
           "DECODE_THRESHOLD", "DECODE_THRESHOLD_DEFAULT", "PAD_SCORE", "long_row_blocks", "long_row_keys", "pad_words"]
