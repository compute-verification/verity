"""[GLM-5.3] DeepSeek sparse attention over an `fp8_ds_mla` KV cache, as the H200 serves `--kv-cache-dtype fp8` (vLLM d9105ea80:
`FLASHMLA_SPARSE`, `flashmla_sparse.py`), over the leaves `dsa_leaves.dsa_leaves` exports for it (`vllm_iface`):

    dsa_norm_rope_ds_mla(dsa_norm_rope's operands, tile, kv_c_out!, k_pe_out!)
        -> (q_c, index K e4m3 [T, D], index K scale f32 [T], NoPE e4m3 [T, L], scales f32 [T, L / tile])
        `_fused_norm_rope_kernel` with `mla_kv_cache_dtype = "fp8_ds_mla"`, per token one `DsaLatentRowDsMla_v1{KVL,R,TILE,EPS}`: its
        bf16 row is kv_c_out and k_pe_out (rebound), and its NoPE bytes and their scales, with k_pe_out, are the token's cache row
    dsa_flashmla_attn(positions, ql_nope, q_pe, kv_nope, kv_scale, k_pe, index_q?, index_weights?, index_k?, index_k_scale?,
                      layer_name, topk, scale) -> attention rows [T, NH, L]
        the layer's cache rows as history keyed by `layer_name`; the logits and the short-row top-k as `dsa_indexer_attn`'s, then
        per query row of n keys `FlashMlaV32Decode_v1{NH,SPLITS,DN,DR,QT,BN,DOT,SCALE}` (`_forward_fp8_kv_mixed_batch`) over TOPK
        cache rows: the history's first n at index 0.., then row 0's under index -1, which the kernel zeroes.  One call takes
        every sparse row of the launch as one request of that many query rows at `FP8_PADDED_HEADS` heads, so its rows share the
        planner's SPLITS (`decode_splits`)
    dsa_dense_mha_ds_mla(dsa_dense_mha's operands, its cache rows kv_nope, kv_scale and k_pe) -> attention rows [T, NH, DV]
        `dsa_dense_mha`'s prefill over the call's own bf16 rows; the cache rows open the layer's history

`predict.dsa` writes the same calls (`_Model.attention`, `_flashmla_fp8`) and reads this module's constants.  A row past TOPK keys
is refused: its cache rows are the ones the top-k selects, which no Definition routes yet.
"""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

from verity.circuits.refs import Coll, array_of, concat
from verity.circuits.types import Array, Value
from verity.definitions.numerics import flashmla_twin
from verity.definitions.registry import dsa, dsa_ds_mla, flashmla
from verity.definitions.registry.flashmla_dsv32 import FLASHMLA_GLM53

from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import known, refuse_cached_prefix, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import (
    _OBSERVED,
    active_target_profile,
    launch_keys_of,
)
from verity_vllm.program.frontend.rules.vllm_dsa import (
    _NORM_ROPE,
    BF16,
    E4M3,
    F32,
    ROLE_KERNELS,
    DsaNormRopeRule,
    _args,
    _log,
    _missing,
    _open_history,
    _positions,
    _prefill_by_dense_mha,
    _rows,
    _static_float,
    _value,
)
from verity_vllm.program.frontend.rules.vllm_dsa_dense import DsaDenseMhaRule
from verity_vllm.program.frontend.rules.vllm_iface import iface

#: the layout vLLM stores an FP8 KV cache in under FLASHMLA_SPARSE (`phi.STATED_KV_LAYOUT`), a layer's `kv_cache_dtype`
DS_MLA = "fp8_ds_mla"
#: the NoPE values one f32 scale of its row quantizes (kernels.py `_fused_norm_rope_kernel`: four tiles of the 512)
DS_MLA_TILE = 128
#: the implementation that reads the layout
FLASHMLA_SPARSE_IMPL = "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseImpl"
#: flashmla_sparse.py: below this many query heads a rank the cache takes every sparse row of a step in one mixed batch (from it
#: on a prefill upconverts the cache to bf16), its query padded to the decode kernel's 64 heads, the planner's h_q
MIN_HEADS_FOR_BF16_PREFILL, FP8_PADDED_HEADS = 32, 64

#: FlashMLA's sparse fp8 decode by (rope, latent) width: the NoPE values a scale, the rows a block and its k16 step
FLASHMLA_FP8_GEOMETRY = {(FLASHMLA_GLM53["DR"], FLASHMLA_GLM53["DN"]): {k: FLASHMLA_GLM53[k] for k in ("QT", "BN", "DOT")}}

FP8_ROLE_KERNELS = {**ROLE_KERNELS,
                    "dsa_latent_row_ds_mla": "_fused_norm_rope_kernel's row under fp8_ds_mla, with its cache row (DsaLatentRowDsMla_v1)",
                    "flashmla_v32_decode": "FlashMLA's sparse fp8 decode over a token's cache rows (FlashMlaV32Decode_v1)"}


def decode_splits(tokens: int, topk: int, num_sms: int, block: int) -> tuple[int, ...]:
    """SPLITS of each row of a FlashMLA fp8 call of `tokens` sparse rows (`_forward_fp8_kv_mixed_batch`): the planner cuts one
    request of `topk` cache rows over num_sms / tokens / (FP8_PADDED_HEADS / 64) partitions (`get_mla_metadata` with
    `num_q_tokens_per_head_k = tokens * FP8_PADDED_HEADS`), in blocks of `block` rows."""
    metas, ns = flashmla_twin.sched_meta([topk], topk=topk, parts=flashmla_twin.num_sm_parts(FP8_PADDED_HEADS, s_q=tokens,
                                                                                         num_sms=num_sms), block_size_n=block)
    return flashmla_twin.request_splits(metas, ns, 0, topk // block)


def _observed_flashmla(layer_name: str) -> str | None:
    """None when the observed implementation is FlashMLA's sparse one over an `fp8_ds_mla` cache in its mixed batch."""
    obs = (_OBSERVED["attention_impls"] or {}).get(layer_name)
    if obs is None:
        return f"no observed attention implementation for {layer_name!r} (vllm_meta.attention_impls)"
    if obs.get("kind") != "mla" or obs.get("impl") != FLASHMLA_SPARSE_IMPL:
        return f"numerics_unregistered: {layer_name!r} runs {obs.get('impl')!r}; FlashMlaV32Decode_v1 states {FLASHMLA_SPARSE_IMPL}"
    if (obs.get("layer_kv_cache_dtype"), obs.get("kv_cache_dtype")) != (DS_MLA, DS_MLA):
        return (f"numerics_unregistered: {layer_name!r} has KV cache dtype {obs.get('layer_kv_cache_dtype')!r} (its implementation "
                f"{obs.get('kv_cache_dtype')!r}); FlashMlaV32Decode_v1 reads {DS_MLA} rows")
    if obs.get("dcp_world_size") not in (None, 1) or obs.get("use_pcp"):
        return f"numerics_unregistered: {layer_name!r} runs under context parallelism"
    heads = obs.get("q_heads_per_rank")
    if not isinstance(heads, int) or isinstance(heads, bool) or not 0 < heads < MIN_HEADS_FOR_BF16_PREFILL:
        return (f"numerics_unregistered: {layer_name!r} has {heads!r} query heads a rank; FlashMLA's mixed batch serves fewer than "
                f"{MIN_HEADS_FOR_BF16_PREFILL} (from it on, a prefill upconverts the cache to bf16)")
    return None


def _append_ds_mla_rows(ctx, hist: dict, a: dict, has: bool) -> None:
    """Append the call's cache rows to the layer's history: each row's NoPE bytes (`latent`), scales and RoPE words, and on an
    indexer layer its index-K slot and scale."""
    for key, name, w in (("latent", "kv_nope", 8), ("scale", "kv_scale", 32), ("rope", "k_pe", 16)):
        rows = _rows(ctx, a[name], w)
        hist.setdefault(key, []).extend(rows[i] for i in range(rows.type.n))
    if has:
        ik, iks = _rows(ctx, a["index_k"], 8), _rows(ctx, a["index_k_scale"], 32)
        for i in range(ik.type.n):
            hist["ik"].append(ik[i])
            hist["iks"].append(iks[i])


def _cache_rows(ctx, a: dict, T: int) -> list[str | None]:
    """The call's cache-row operands beside k_pe: NoPE e4m3 [T, L] and whole tiles' scales f32 [T, L / tile]."""
    L = (shape(ctx, a["kv_nope"]) or (0, 0))[-1]
    S = (shape(ctx, a["kv_scale"]) or (0, 0))[-1]
    if not (S and L and L % S == 0):
        return [f"kv_scale has {S} scales a row of {L} NoPE values, not one per whole tile"]
    return [_value(ctx, a["kv_nope"], E4M3, (T, L), "kv_nope"), _value(ctx, a["kv_scale"], F32, (T, S), "kv_scale")]


_NORM_ROPE_DS_MLA = (*_NORM_ROPE[:-2], "tile", *_NORM_ROPE[-2:])


class DsaNormRopeDsMlaRule(DsaNormRopeRule):
    """`vllm_iface.dsa_norm_rope_ds_mla` (`_fused_norm_rope_kernel` writing an `fp8_ds_mla` cache row) per token: `DsaNormRope`'s
    q_c norm and index-K slot, and one `DsaLatentRowDsMla_v1{KVL,R,TILE,EPS}`, whose bf16 row kv_c_out and k_pe_out are rebound to;
    the node is (q_c normed, index K, index K scale, NoPE [T, L] e4m3, scales [T, L / tile] f32)."""
    info = RuleInfo("DsaNormRopeDsMla[vllm-dsa]", 1, "in-place-binding",
                    applies_to="vllm_iface.dsa_norm_rope_ds_mla(...) (" + ", ".join(_NORM_ROPE_DS_MLA) + ")",
                    preconditions=(*DsaNormRopeRule.info.preconditions,
                                   "tile a static int, L and R and kv_eps as DsaLatentRowDsMla_v1 takes them (whole tiles of the row)"),
                    evidence_domain="structural; numerics: dsa_rmsnorm, DsaIndexerKRow_v1 (registry.dsa) and DsaLatentRowDsMla_v1 "
                                    "(registry.dsa_ds_mla)")
    names, kernels = _NORM_ROPE_DS_MLA, FP8_ROLE_KERNELS

    def targets(self):
        return {iface.dsa_norm_rope_ds_mla.default}

    def row_roles(self) -> list[str]:
        return ["dsa_latent_row_ds_mla"]

    def row_statics(self, a: dict, L: int, R: int) -> str | None:
        tile = a.get("tile")
        if not isinstance(tile, int) or isinstance(tile, bool):
            return "tile is not a static int"
        try:
            dsa_ds_mla.check_row(L, R, tile, float(a["kv_eps"]))
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return None

    def rows(self, ctx, node, a: dict, kvc: Coll, kpe: Coll, kvw: Coll, cs: Coll, pos: list[int], h: str):
        L, R = known(shape(ctx, a["kv_c"]))[1], known(shape(ctx, a["k_pe"]))[1]
        lr = ctx.kind("dsa_latent_row_ds_mla", node, KVL=L, R=R, TILE=int(a["tile"]), EPS=float(a["kv_eps"]))
        out = [ctx.call(lr, Coll(Array(L + R, Value(16)), concat([kvc[i].refs, kpe[i].refs])), kvw, cs[p], node=node, rule=self,
                        name=f"{h}/latent[{i}]") for i, p in enumerate(pos)]
        return ([Coll(Array(L, Value(16)), o[0].refs.slice(0, L)) for o in out],
                [Coll(Array(R, Value(16)), o[0].refs.slice(L, L + R)) for o in out],
                [array_of([o[1] for o in out]), array_of([o[2] for o in out])], [lr.id])


_FLASH = ("positions", "ql_nope", "q_pe", "kv_nope", "kv_scale", "k_pe", "index_q", "index_weights", "index_k", "index_k_scale",
          "layer_name", "topk", "scale")


class DsaFlashMlaAttnRule(StateRule):
    """`vllm_iface.dsa_flashmla_attn` (the sparse indexer and FlashMLA's sparse fp8 decode of one layer over its `fp8_ds_mla`
    cache): the layer's cache rows are a history keyed by `layer_name`, opened and continued as `DsaIndexerAttn[vllm-dsa]`'s, each
    row its NoPE bytes, scales and RoPE words.  Row i of a call that finds P0 rows has n = P0 + i + 1 keys, at most TOPK: an
    indexer layer emits its logits and the short-row top-k, an IndexShare layer takes the last indexer layer's for the same
    positions, and the row's attention is `FlashMlaV32Decode_v1` over TOPK cache rows, the history's first n at index 0.. and then
    row 0's under index -1, at the SPLITS the planner gives a call of the launch's sparse rows (`decode_splits`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaFlashMlaAttn[vllm-dsa]", 1, "state", applies_to="vllm_iface.dsa_flashmla_attn(...) (" + ", ".join(_FLASH) + ")",
                    preconditions=(("the layer's OBSERVED implementation is FlashMLASparseImpl, the layer's and its KV cache dtype "
                                    "fp8_ds_mla, without context parallelism, at fewer query heads a rank than "
                                    "MIN_HEADS_FOR_BF16_PREFILL (the mixed batch)"),
                                   ("the call takes the top-k MQA path as a decode (DsaIndexerAttn[vllm-dsa]'s precondition); a "
                                    "prefill of at most TOPK keys is DsaDenseMhaDsMla[vllm-dsa]'s"),
                                   "the target's compute capability is 9.0 (FlashMLA's Hopper sparse fp8 decode)",
                                   ("positions are T consecutive rows continuing the layer's history; ql_nope [T, NH, L], q_pe [T, NH, R], "
                                    "k_pe [T, R] bf16, kv_nope [T, L] e4m3, kv_scale [T, L / QT] f32, at a registered (R, L) "
                                    "(FLASHMLA_FP8_GEOMETRY)"),
                                   "TOPK whole blocks of BN rows; every row of the call has at most TOPK keys",
                                   ("the index operands all given (index_q [T, H, D] e4m3, index_weights [T, H] f32, index_k [T, D] e4m3, "
                                    "index_k_scale [T] f32) or all None, then the last indexer layer selected for these positions"),
                                   "layer_name static and written from one module; not under registered_values"),
                    evidence_domain="structural; numerics: DsaMqaLogits_v3, DsaTopKShortRow_v1, FlashMlaV32Decode_v1 "
                                    "(glm53_flashmla_fp8_capture)")

    def targets(self):
        return {iface.dsa_flashmla_attn.default}

    def applies(self, node, ctx):
        a = _args(node, _FLASH)
        has = a.get("index_q") is not None
        layer, topk, scale = a["layer_name"], a["topk"], a["scale"]
        if not isinstance(layer, str) or not isinstance(topk, int) or topk < 1 or not _static_float(scale):
            return "layer_name, topk, scale are not static"
        if (r := _missing(ctx, ["flashmla_v32_decode", "const"] + (["dsa_mqa_logits", "dsa_topk_short_row"] if has else []),
                          FP8_ROLE_KERNELS)):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: FlashMlaV32Decode_v1 takes its scale as a static; no registered-values twin"
        if (r := _observed_flashmla(layer)):
            return r
        tp = active_target_profile()
        if tp.compute_capability != (9, 0):
            return (f"numerics_unregistered: FlashMlaV32Decode_v1 is FlashMLA's Hopper kernel at compute capability 9.0; the target is "
                    f"{tp.capability_str}")
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        if (r := _prefill_by_dense_mha(layer, T)):
            return r
        L = (shape(ctx, a["kv_nope"]) or (0, 0))[-1]
        R = (shape(ctx, a["k_pe"]) or (0, 0))[-1]
        NH = (shape(ctx, a["ql_nope"]) or (0, 0, 0))[1]
        G = FLASHMLA_FP8_GEOMETRY.get((R, L))
        if G is None:
            return (f"numerics_unregistered: FlashMLA's sparse fp8 decode at rope {R}, latent {L} is not registered "
                    f"({sorted(FLASHMLA_FP8_GEOMETRY)})")
        for r in (_value(ctx, a["ql_nope"], BF16, (T, NH, L), "ql_nope"), _value(ctx, a["q_pe"], BF16, (T, NH, R), "q_pe"),
                  _value(ctx, a["kv_nope"], E4M3, (T, L), "kv_nope"), _value(ctx, a["kv_scale"], F32, (T, L // G["QT"]), "kv_scale"),
                  _value(ctx, a["k_pe"], BF16, (T, R), "k_pe")):
            if r:
                return r
        try:
            flashmla.check_row_statics(L, R, G["QT"], "FlashMlaV32KvRow")
            flashmla.check_block_statics(L + R, G["BN"])
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        if topk % G["BN"]:
            return f"numerics_unregistered: TOPK {topk} is not whole blocks of FlashMLA's {G['BN']} rows"
        histories = ctx.state.get("dsa_history")
        hist = None if histories is None else histories.get(layer)
        if (n := (len(hist["latent"]) if hist else 0) + T) > topk:
            return (f"numerics_unregistered: a row of {n} keys, past TOPK ({topk}), over an {DS_MLA} cache: FlashMLA's sparse fp8 "
                    f"decode over the rows its top-k selects, which no Definition routes")
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
        a = _args(node, _FLASH)
        has = a.get("index_q") is not None
        layer, TOPK, scale = a["layer_name"], a["topk"], float(a["scale"])
        mp, h = ctx.module_path(node), ctx.hint_name(node)
        pos = ctx.static_index(a["positions"]).flat()
        T = len(pos)
        _, NH, L = known(shape(ctx, a["ql_nope"]))
        R = known(shape(ctx, a["k_pe"]))[1]
        G = FLASHMLA_FP8_GEOMETRY[(R, L)]
        hist, P0 = _open_history(ctx, node, layer, pos, has)
        keys = [P0 + i + 1 for i in range(T)]
        _append_ds_mla_rows(ctx, hist, a, has)
        logs = [(_log(ctx, (layer, key), hist[key]), Array(w, Value(bits)))
                for key, w, bits in (("latent", L, 8), ("scale", L // G["QT"], 32), ("rope", R, 16))]
        tokens = len(launch_keys_of(node, keys))
        splits = decode_splits(tokens, TOPK, int(active_target_profile().num_sms), G["BN"])
        if has:
            H, D = known(shape(ctx, a["index_q"]))[1:3]
            iq, iw = _rows(ctx, a["index_q"], 8), _rows(ctx, a["index_weights"], 32)
            klog, slog = _log(ctx, (layer, "ik"), hist["ik"]), _log(ctx, (layer, "iks"), hist["iks"])
            sel_fn = ctx.kind("dsa_topk_short_row", node, TOPK=TOPK)
            selected = {}
            for i, (P, n) in enumerate(zip(pos, keys)):
                logits = ctx.kind("dsa_mqa_logits", node, N=n, H=H, D=D)
                ctx.call(logits, iq[i], iw[i], Coll(Array(n, Array(D, Value(8))), klog.cut(n)), Coll(Array(n, Value(32)), slog.cut(n)),
                         node=node, rule=self, name=f"{h}/logits[t={P}]")
                ln = ctx.call(ctx.kind("const", node, W=32, BITS=n), node=node, rule=self, name=f"{h}/len[t={P}]")
                selected[P] = ctx.call(sel_fn, ln, node=node, rule=self, name=f"{h}/topk[t={P}]")
            ctx.state["dsa_topk"] = {"layer": layer, "rows": selected, "keys": dict(zip(pos, keys))}
            source = layer
        else:
            last = ctx.state.get("dsa_topk")
            if last is None or any(P not in last["rows"] for P in pos):
                refuse("side_input", node, mp, f"IndexShare layer {layer!r} reads a top-k no indexer layer selected for positions "
                                               f"{pos[0]}..{pos[-1]}")
            source = last["layer"]
        ql, qr = _rows(ctx, a["ql_nope"], 16), _rows(ctx, a["q_pe"], 16)
        fn = ctx.kind("flashmla_v32_decode", node, NH=NH, SPLITS=splits, DN=L, DR=R, QT=G["QT"], BN=G["BN"], DOT=G["DOT"], SCALE=scale)
        valid, gone = (ctx.call(ctx.kind("const", node, W=32, BITS=v), node=node, rule=self, name=f"{h}/index[{v:#x}]").refs
                       for v in (0, 0xFFFFFFFF))
        outs = []
        for i, (P, n) in enumerate(zip(pos, keys)):
            q = Coll(Array(NH, Array(L + R, Value(16))), concat([p for hd in range(NH) for p in (ql[i][hd].refs, qr[i][hd].refs)]))
            rows = [Coll(Array(TOPK, row), log.cut(n, [log.parts[0]] * (TOPK - n))) for log, row in logs]
            idx = Coll(Array(TOPK, Value(32)), concat([valid] * n + [gone] * (TOPK - n)))
            outs.append(ctx.call(fn, q, *rows, idx, node=node, rule=self, name=f"{h}[t={P}]"))
        ctx.define(node, array_of(outs))
        ctx.record(self, node, {"layer_name": layer, "positions": [pos[0], pos[-1]], "history_rows_after": P0 + T, "NH": NH, "DR": R, "DN": L,
                                "QT": G["QT"], "BN": G["BN"], "DOT": G["DOT"].name, "scale": scale, "TOPK": TOPK, "indexer": has,
                                "top_k_from": source, "launch_sparse_rows": tokens, "padded_heads": FP8_PADDED_HEADS,
                                "splits": list(splits),
                                "selection": "rows 0..n-1 at index 0.. (n <= TOPK: DsaTopKShortRow_v1's slots), then row 0 under "
                                             "index -1 to TOPK rows",
                                "attention_impl_observed": (_OBSERVED["attention_impls"] or {}).get(layer)},
                   emitted=f"{fn.id} x{T}" + (f" + DsaMqaLogits_v3, DsaTopKShortRow_v1 x{T}" if has else f" (top-k of {source})"))


_DENSE_DS_MLA = ("positions", "q_nope", "q_pe", "k_nope", "v", "kv_nope", "kv_scale", "k_pe", "index_k", "index_k_scale", "layer_name",
                 "topk", "scale")


class DsaDenseMhaDsMlaRule(DsaDenseMhaRule):
    """`vllm_iface.dsa_dense_mha_ds_mla`: `DsaDenseMha[vllm-dsa]`'s prefill of a layer whose cache is `fp8_ds_mla`.  Dense MHA reads
    the call's own bf16 rows (`forward_mha` takes kv_c_normed, not the cache), so its calls are that rule's; the call's cache rows
    (NoPE bytes, scales, RoPE words) open the layer's history for the decode steps `DsaFlashMlaAttn[vllm-dsa]` reads them at."""
    info = RuleInfo("DsaDenseMhaDsMla[vllm-dsa]", 1, "state",
                    applies_to="vllm_iface.dsa_dense_mha_ds_mla(...) (" + ", ".join(_DENSE_DS_MLA) + ")",
                    preconditions=(("the layer's OBSERVED implementation is FlashMLASparseImpl, the layer's and its KV cache dtype "
                                    "fp8_ds_mla, without context parallelism, at fewer query heads a rank than "
                                    "MIN_HEADS_FOR_BF16_PREFILL; its prefill backend FlashAttnPrefillBackend on FA3 with v unpadded at "
                                    "the call's scale; sparse_mla_force_mqa off"),
                                   *DsaDenseMhaRule.info.preconditions[1:3],
                                   ("q's and k's parts q_nope, k_nope [T, NH, P], q_pe [T, NH, R], k_pe [T, R], and v [T, NH, D] bf16 "
                                    "at D = P + R; kv_nope [T, L] e4m3 and kv_scale [T, L / tile] f32; scale == f32(D^-0.5)"),
                                   *DsaDenseMhaRule.info.preconditions[4:]),
                    evidence_domain=DsaDenseMhaRule.info.evidence_domain)
    names = _DENSE_DS_MLA

    def targets(self):
        return {iface.dsa_dense_mha_ds_mla.default}

    def observed(self, layer_name: str) -> str | None:
        return _observed_flashmla(layer_name)

    def cache_rows(self, ctx, a: dict, T: int) -> list[str | None]:
        return _cache_rows(ctx, a, T)

    def append(self, ctx, hist: dict, a: dict, has: bool) -> None:
        _append_ds_mla_rows(ctx, hist, a, has)


VLLM_DSA_FP8_RULES = [DsaNormRopeDsMlaRule(), DsaFlashMlaAttnRule(), DsaDenseMhaDsMlaRule()]

__all__ = ["DS_MLA", "DS_MLA_TILE", "DsaDenseMhaDsMlaRule", "DsaFlashMlaAttnRule", "DsaNormRopeDsMlaRule", "FLASHMLA_FP8_GEOMETRY",
           "FLASHMLA_SPARSE_IMPL", "FP8_PADDED_HEADS", "MIN_HEADS_FOR_BF16_PREFILL", "VLLM_DSA_FP8_RULES", "decode_splits"]
