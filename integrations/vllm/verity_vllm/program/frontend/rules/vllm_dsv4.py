"""[DSV4] DeepSeek-V4's sparse MLA as the H200 serves it (vLLM d9105ea80 `models/deepseek_v4/attention.py`, `nvidia/flashmla.py`),
over the leaves `dsv4_leaves.dsv4_attention_leaves` exports in place of its kernels (`vllm_iface`):

    dsv4_qnorm_rope_kv_insert(q [T, NH, D], kv [T, D], positions, cos_sin, padded_heads, eps, quant_block)
        -> (q [T, NHP, D], NoPE [T, DN] e4m3, scale bytes [T, DN/QB + 1], RoPE [T, DR] bf16)
        `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` per token: `Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}` of its live
        heads, its kv row and its position's f32 cos_sin row (`registry.dsv4_qnorm_insert`)
    dsv4_compressor_state(kv_score [T, 2W] f32, positions, ape [R, W] f32) -> (kv [T, W], score [T, W])
        `save_partial_states` per token: kv the first W words, score `Dsv4CompressorScore_v1{W}` of the last W and ape[pos % R]
        (`registry.dsv4_indexer`)
    dsv4_sparse_attn(q, NoPE, scale, RoPE, positions, cos_sin, sink, state kv, state score, norm weight, layer_name, heads, ratio,
                     window, topk, softmax_scale, eps) -> [T, heads, D] bf16
        the layer's window, compressor state and compressed caches as histories, the compressor at each window's end
        (`registry.dsv4_compressor`) and FlashMLA's sparse decode or prefill over the rows each token reads (`registry.flashmla`,
        `registry.flashmla_prefill`)

and vLLM's `fused_inv_rope_fp8_quant_kernel`, the o-projection's input, per token and head (`registry.dsv4_indexer`).  A role
the vocabulary lacks refuses by name.
"""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import call_args, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile, launch_sequences_of
from verity_vllm.program.frontend.rules.vllm_dsa import _missing, _positions, _rows, _static_float, _table, _value
from verity_vllm.program.frontend.rules.vllm_iface import iface, targets_for
from verity.ml.definitions.numerics.flashmla_twin import num_sm_parts, request_splits, sched_meta
from verity.ml.definitions.registry import dsv4_compressor as C
from verity.ml.definitions.registry import dsv4_indexer as I
from verity.ml.definitions.registry import dsv4_qnorm_insert as Q
from verity.ml.definitions.registry.flashmla import FLASHMLA_DSV4

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

BF16, F32, E4M3, U8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.uint8
M32 = 0xFFFFFFFF

#: the roles the vocabulary may lack, and which kernel each states
ROLE_KERNELS = {"dsv4_qnorm_rope_kv_insert": "the query norm, RoPE and window cache insert (Dsv4QNormRopeKvInsert_v1)",
                "dsv4_compressor_score": "save_partial_states' score row (Dsv4CompressorScore_v1)",
                "dsv4_compress_c4": "the ratio-4 compressor (Dsv4MlaCompressC4_v1)",
                "dsv4_compress_c128": "the ratio-128 compressor (Dsv4MlaCompressC128_v1)",
                "flashmla_decode": "FlashMLA's sparse fp8 decode (FlashMlaDecode_v1)",
                "flashmla_gather_row": "the prefill's K-cache gather (FlashMlaGatherRow_v1)",
                "flashmla_prefill": "FlashMLA's sparse prefill (FlashMlaPrefill_v1)",
                "dsv4_inv_rope_quant_head": "the o-projection input's inverse RoPE and FP8 quantisation (Dsv4InvRopeQuantHead_v1)",
                "const": "a constant word"}

#: FlashMLA's sparse fp8 decode and prefill (cc 9.0) and the head-512 compressors by (head, rope) width: the scale group (the
#: compressors' quant block, whose last group is the rope), the block rows and the score chain
FLASHMLA_GEOMETRY = {(FLASHMLA_DSV4["DN"] + FLASHMLA_DSV4["DR"], FLASHMLA_DSV4["DR"]): {k: FLASHMLA_DSV4[k] for k in ("QT", "BN", "DOT")}}
#: the compressor's role by ratio; ratio 1 is a window-only layer
COMPRESSORS = {4: "dsv4_compress_c4", 128: "dsv4_compress_c128"}


class Dsv4QNormRopeKvInsertRule(OpRule):
    """`vllm_iface.dsv4_qnorm_rope_kv_insert` per token: `Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}` of q[t], kv[t] and
    cos_sin[positions[t]]; the node is (q padded, NoPE, scale bytes, RoPE) over the tokens."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4QNormRopeKvInsert[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.dsv4_qnorm_rope_kv_insert",
                    preconditions=("positions a static 1-D index of T rows; q [T, NH, D], kv [T, D] bf16 values",
                                   "cos_sin an f32 rotary cache [MAX_POS, DR] holding every position",
                                   "padded_heads, quant_block static ints and eps a static float as the Definition's statics check them",
                                   "not under registered_values"),
                    evidence_domain="structural; numerics: Dsv4QNormRopeKvInsert_v1 (registry.dsv4_qnorm_insert)")

    def targets(self):
        return {iface.dsv4_qnorm_rope_kv_insert.default}

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["dsv4_qnorm_rope_kv_insert"], ROLE_KERNELS)):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: the insert takes eps as a static; no registered-values twin"
        a = call_args(node)
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        if (r := _value(ctx, a["q"], BF16, (T, None, None), "q")):
            return r
        _, NH, D = shape(ctx, a["q"])
        DR = (shape(ctx, a["cos_sin"]) or (0, 0))[-1]
        if (r := _value(ctx, a["kv"], BF16, (T, D), "kv") or _table(ctx, a["cos_sin"], DR, pos, "cos_sin", F32)):
            return r
        NHP, QB, eps = a["padded_heads"], a["quant_block"], a["eps"]
        if not (isinstance(NHP, int) and isinstance(QB, int) and _static_float(eps)):
            return "padded_heads, quant_block and eps are not statics"
        try:
            Q.check_insert_statics(NH, NHP)
            Q.check_q_statics(D, DR)
            Q.check_kv_statics(D - DR, DR, QB)
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = call_args(node)
        pos = ctx.static_index(a["positions"]).flat()
        T, h = len(pos), ctx.hint_name(node)
        _, NH, D = shape(ctx, a["q"])
        DR = shape(ctx, a["cos_sin"])[1]
        S = {"NH": NH, "NHP": a["padded_heads"], "DN": D - DR, "DR": DR, "QB": a["quant_block"], "EPS": float(a["eps"])}
        fn = ctx.kind("dsv4_qnorm_rope_kv_insert", node, **S)
        q, kv, cs = _rows(ctx, a["q"], 16), _rows(ctx, a["kv"], 16), ctx.coll(a["cos_sin"])
        outs = [ctx.call(fn, q[i], kv[i], cs[pos[i]], node=node, rule=self, name=f"{h}[t={pos[i]}]") for i in range(T)]
        ctx.define(node, [array_of([o[k] for o in outs]) for k in range(4)])
        ctx.bind_weight_role(a["cos_sin"], self, "rope.cos_sin_cache", D=DR)
        ctx.record(self, node, {"T": T, "positions": [pos[0], pos[-1]], **S}, emitted=f"{fn.id} x{T}")


class Dsv4CompressorStateRule(OpRule):
    """`vllm_iface.dsv4_compressor_state` per token: the kv row is kv_score[t, :W] (no gates), the score row
    `Dsv4CompressorScore_v1{W}` of kv_score[t, W:] and ape[positions[t] % R]; the node is (kv, score) over the tokens."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4CompressorState[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.dsv4_compressor_state",
                    preconditions=("positions a static 1-D index of T rows of non-negative positions",
                                   "kv_score an f32 value [T, 2W]; ape an f32 value [R, W]"),
                    evidence_domain="structural; numerics: Dsv4CompressorScore_v1 (registry.dsv4_indexer)")

    def targets(self):
        return {iface.dsv4_compressor_state.default}

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["dsv4_compressor_score"], ROLE_KERNELS)):
            return r
        a = call_args(node)
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        if any(p < 0 for p in pos):
            return "a negative position"
        if (r := _value(ctx, a["ape"], F32, (None, None), "ape")):
            return r
        W = shape(ctx, a["ape"])[1]
        return _value(ctx, a["kv_score"], F32, (len(pos), 2 * W), "kv_score") or True

    def emit(self, node, ctx):
        a = call_args(node)
        pos = ctx.static_index(a["positions"]).flat()
        h = ctx.hint_name(node)
        R, W = shape(ctx, a["ape"])
        fn = ctx.kind("dsv4_compressor_score", node, W=W)
        rows, ape = _rows(ctx, a["kv_score"], 32), ctx.coll(a["ape"])
        score = [ctx.call(fn, rows[i][W:], ape[p % R], node=node, rule=self, name=f"{h}[t={p}]") for i, p in enumerate(pos)]
        ctx.define(node, [array_of([r[:W] for r in rows]), array_of(score)])
        ctx.bind_weight_role(a["ape"], self, "dsv4.compressor.ape", R=R, W=W)
        ctx.record(self, node, {"T": len(pos), "positions": [pos[0], pos[-1]], "R": R, "W": W}, emitted=f"{fn.id} x{len(pos)}")


def _blocks(n: int, bn: int) -> int:
    return -(-n // bn)


def _stack(rows: list[Coll], n: int, w: int) -> Coll:
    return Coll(Array(len(rows), Array(n, Value(w))), concat([r.refs for r in rows]))


def _launch_decodes(node, mp, layer: str, p: int, own: tuple[int, int], window: int, ratio: int) -> tuple[list[tuple[int, int]], int]:
    """The (window, compressed) lengths of every decode of the launch the decode at position p runs in, in the launch's order, and
    its index among them.  A decode at position q of a sequence from position 0 reads min(q + 1, window) window rows and
    (q + 1) / ratio compressed rows; without a declared batch the decode is alone."""
    seqs = launch_sequences_of(node)
    if seqs is None:
        return [own], 0
    decodes = [q for rows, q in seqs if rows == 1]
    if decodes.count(p) != 1:
        refuse("numerics_unregistered", node, mp, f"{layer!r}: the declared launch batch holds {decodes.count(p)} decodes at position "
                                                  f"{p}, not one: FlashMLA splits a decode by its index among the launch's decodes, "
                                                  f"which only its position names")
    lens = [(min(q + 1, window), (q + 1) // ratio if ratio > 1 else 0) for q in decodes]
    i = decodes.index(p)
    if lens[i] != own:
        refuse("shape", node, mp, f"{layer!r}: the decode at position {p} reads {own} (window, compressed) rows, not the {lens[i]} of a "
                                  f"sequence from position 0, as the launch's other decodes are")
    return lens, i


class Dsv4SparseAttnRule(StateRule):
    """`vllm_iface.dsv4_sparse_attn` (`forward_mqa`): the layer's window rows (the insert's NoPE, scale and RoPE by position), its
    compressor's state rows and its compressed rows are implicit cache state keyed by `layer_name`, made explicit as histories.  A
    layer's history opens at its first call, at whatever position P (the step wrapper's `pos0`, as `vllm_dsa.DsaIndexerAttnRule`'s
    opens), and each later call's T tokens are its next positions (one request); no row before P is read.

    * The compressor (`compress_norm_rope_store_cutedsl`) at each position p of the call with (p + 1) % ratio == 0, before the
      attention reads its row: compressed row (p + 1) / ratio - 1 from the state rows of positions p - win + 1 .. p at and past P
      (win = 8 at ratio 4, 128 at ratio 128), row r reading the D state words from (r / ratio) * D (ratio 4 overlaps two windows in
      a state of 2D), and the cos_sin row of position p - ratio + 1: `Dsv4MlaCompressC4_v1` / `Dsv4MlaCompressC128_v1`.
    * Token p reads its window, the rows of positions max(p - window + 1, P) .. p, and the (p + 1) / ratio - P / ratio compressed
      rows the history holds by then, in order (the indexer's short path selects every compressed row in order,
      `_fill_short_context_topk_indices`; ratio 128's rows are `_build_c128a_topk_metadata_kernel`'s, the same).
    * A call of one token is a decode (the metadata builders' decode threshold 1): `FlashMlaDecode_v1` over its window rows padded
      to a multiple of BN, then its compressed rows padded likewise (`get_cur_req_info`'s kernel order), at the splits FlashMLA's
      planner gives it among the launch's decodes (`flashmla_twin.sched_meta` over each decode's lengths in the declared batch's
      order, `launch_sequences_of`, at the padded heads' `num_sm_parts`; alone without a declared batch).  The decode is the
      launch's one at its position, and two at one position are refused.
    * A call of more is a prefill: each row it reads gathered once (`FlashMlaGatherRow_v1`, the one chunk's workspace), and token p
      `FlashMlaPrefill_v1` over its compressed rows, then its window rows (`combine_topk_swa_indices`' order), padded to its NP
      pairs of BN-row blocks.

    The index kernels are not stated: the rows are chosen from the static positions, and a row's index is 0 where the token reads
    it and -1 past its length (its value reaches the Definitions only through its validity), the prefill's workspace one row."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4SparseAttn[vllm-dsv4]", 1, "state", applies_to="vllm_iface.dsv4_sparse_attn",
                    preconditions=("the target's compute capability is 9.0 (FlashMLA's cc 9.0 sparse kernels)",
                                   "positions are T consecutive rows continuing the layer's history (opened at its first call's first position)",
                                   "q [T, NHP, D] bf16, NoPE [T, D - DR] e4m3, scale [T, (D - DR) / QT + 1] u8, RoPE [T, DR] bf16, sink [NHP] f32, "
                                   "(D, DR) a registered FlashMLA geometry; cos_sin an f32 rotary cache [MAX_POS, DR] holding every position",
                                   "ratio 1 without a compressor, or 4 or 128 with state kv and score [T, D] f32 ([T, 2D] at ratio 4), the "
                                   "norm weight [D] bf16 and a static eps",
                                   "at ratio 4, (p + 1) / 4 <= topk at every position (the indexer's short path)",
                                   "layer_name, heads, window, topk and softmax_scale static; the caches written from one module; "
                                   "not under registered_values"),
                    evidence_domain="structural; numerics: Dsv4MlaCompressC4_v1 / C128_v1, FlashMlaDecode_v1, FlashMlaGatherRow_v1, "
                                    "FlashMlaPrefill_v1")

    def targets(self):
        return {iface.dsv4_sparse_attn.default}

    def applies(self, node, ctx):
        a = call_args(node)
        layer, NH, ratio, window, topk = (a[k] for k in ("layer_name", "heads", "ratio", "window", "topk"))
        if not (isinstance(layer, str) and all(isinstance(x, int) for x in (NH, ratio, window, topk)) and window >= 1
                and _static_float(a["softmax_scale"])):
            return "layer_name, heads, ratio, window, topk and softmax_scale are not statics"
        if ratio != 1 and ratio not in COMPRESSORS:
            return f"numerics_unregistered: compress ratio {ratio} (the compressors: {sorted(COMPRESSORS)})"
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        roles = (["flashmla_decode"] if T == 1 else ["flashmla_gather_row", "flashmla_prefill"]) + ["const"]
        if (r := _missing(ctx, roles + ([COMPRESSORS[ratio]] if ratio > 1 else []), ROLE_KERNELS)):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: the attention takes its scale as a static; no registered-values twin"
        tp = active_target_profile()
        if tp.compute_capability != (9, 0):
            return f"numerics_unregistered: the FlashMLA Definitions are its cc 9.0 kernels; the target is cc {tp.capability_str}"
        if (r := _value(ctx, a["q"], BF16, (T, None, None), "q")):
            return r
        _, NHP, D = shape(ctx, a["q"])
        DR = (shape(ctx, a["rope"]) or (0, 0))[-1]
        G = FLASHMLA_GEOMETRY.get((D, DR))
        if G is None:
            return f"numerics_unregistered: no FlashMLA sparse geometry at head {D}, rope {DR} (registered: {sorted(FLASHMLA_GEOMETRY)})"
        DN = D - DR
        for r in (_value(ctx, a["nope"], E4M3, (T, DN), "NoPE"), _value(ctx, a["scale"], U8, (T, DN // G["QT"] + 1), "scale"),
                  _value(ctx, a["rope"], BF16, (T, DR), "RoPE"), _value(ctx, a["sink"], F32, (NHP,), "sink"),
                  _table(ctx, a["cos_sin"], DR, pos, "cos_sin", F32)):
            if r:
                return r
        if NHP not in (64, 128) or not 1 <= NH <= NHP:
            return f"numerics_unregistered: FlashMLA's sparse kernels take 64 or 128 padded heads, at least one live ({NH} of {NHP})"
        given = [a[k] is not None for k in ("state_kv", "state_score", "norm_weight")]
        if ratio == 1:
            return True if not any(given) else "a window-only layer with a compressor's operands"
        if not all(given):
            return f"a ratio-{ratio} layer without its compressor's state and norm weight"
        P = (ctx.state.get("dsv4_history", {}).get(layer) or {"from": pos[0]})["from"]
        if ratio == 4 and (pos[-1] + 1) // 4 - P // 4 > topk:
            return (f"numerics_unregistered: position {pos[-1]} has {(pos[-1] + 1) // 4 - P // 4} compressed rows, more than the "
                    f"indexer's top-{topk}: its long path has no Definition")
        width = D * (2 if ratio == 4 else 1)
        for r in (_value(ctx, a["state_kv"], F32, (T, width), "state kv"), _value(ctx, a["state_score"], F32, (T, width), "state score"),
                  _value(ctx, a["norm_weight"], BF16, (D,), "norm weight")):
            if r:
                return r
        if not _static_float(a["eps"]):
            return "eps is not a static float"
        if G["QT"] != DR:
            return f"numerics_unregistered: the compressor rotates its last quant group, so its block {G['QT']} is the rope width {DR}"
        try:
            C.check_norm_statics(ratio, D, DR, float(a["eps"]))
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = call_args(node)
        layer, NH, ratio, window, scale = a["layer_name"], a["heads"], a["ratio"], a["window"], float(a["softmax_scale"])
        mp, h = ctx.module_path(node), ctx.hint_name(node)
        pos = ctx.static_index(a["positions"]).flat()
        T = len(pos)
        _, NHP, D = shape(ctx, a["q"])
        DR = shape(ctx, a["rope"])[1]
        G = FLASHMLA_GEOMETRY[(D, DR)]
        DN, QT, BN, DOT = D - DR, G["QT"], G["BN"], G["DOT"]
        owner = ctx.state.setdefault("dsv4_owner", {})
        if owner.setdefault(layer, mp) != mp:
            refuse("side_input", node, mp, f"the caches of {layer!r} are written from two modules ({owner[layer]} and {mp})")
        hist = ctx.state.setdefault("dsv4_history", {}).setdefault(layer, {"from": pos[0], "window": [], "state": [], "compressed": []})
        P, P0 = hist["from"], len(hist["window"])
        if pos != list(range(P + P0, P + P0 + T)):
            refuse("shape", node, mp, f"positions {pos[0]}..{pos[-1]} are not the {T} after {layer!r}'s history of {P0} rows from position {P}")
        nope, sc, rope = _rows(ctx, a["nope"], 8), _rows(ctx, a["scale"], 8), _rows(ctx, a["rope"], 16)
        hist["window"] += [(nope[i], sc[i][:DN // QT], rope[i]) for i in range(T)]
        consts: dict[int, Coll] = {}

        def i32(v: int) -> Coll:
            if v not in consts:
                consts[v] = ctx.call(ctx.kind("const", node, W=32, BITS=v & M32), node=node, rule=self, name=f"{h}/i32[{v}]")
            return consts[v]

        made = []
        if ratio > 1:
            kv, score = _rows(ctx, a["state_kv"], 32), _rows(ctx, a["state_score"], 32)
            hist["state"] += [(kv[i], score[i]) for i in range(T)]
            made = [p for p in pos if (p + 1) % ratio == 0]
            for p in made:
                hist["compressed"].append(self._compress(ctx, node, a, hist["state"], P, p, D, QT, h))
        q, sink = _rows(ctx, a["q"], 16), ctx.coll(a["sink"])[:NH]
        pre = {"layer_name": layer, "positions": [pos[0], pos[-1]], "history_from": P, "history_rows_after": P0 + T, "ratio": ratio,
               "window": window, "NH": NH, "NHP": NHP, "D": D, "DR": DR, "QT": QT, "BN": BN, "DOT": DOT.name, "scale": scale,
               "compressed_rows": [(p + 1) // ratio - 1 for p in made],
               "selection": "the window by position, then every compressed row in order (the indexer's short path)"}
        if T == 1:
            p = pos[0]
            s0 = max(p - window + 1, P)
            swa, nc = p - s0 + 1, (p + 1) // ratio - P // ratio if ratio > 1 else 0
            NW, NX = _blocks(swa, BN) * BN, _blocks(nc, BN) * BN
            win, comp = hist["window"][s0 - P:p - P + 1], hist["compressed"][:nc]
            rows = win + win[:1] * (NW - swa) + comp + comp[:1] * (NX - nc)
            idx = [i32(0)] * swa + [i32(-1)] * (NW - swa) + [i32(0)] * nc + [i32(-1)] * (NX - nc)
            # the lengths given, extra_topk only says whether the layer has an extra cache
            lens, req = _launch_decodes(node, mp, layer, p, (swa, nc), window, ratio)
            metas, ns = sched_meta([w for w, _ in lens], [x for _, x in lens] if ratio > 1 else None, topk=window,
                                   extra_topk=int(ratio > 1), parts=num_sm_parts(NHP, 1, int(active_target_profile().num_sms)),
                                   block_size_n=BN)
            SPLITS = request_splits(metas, ns, req, (NW + NX) // BN)
            fn = ctx.kind("flashmla_decode", node, NH=NH, SPLITS=SPLITS, NW=NW, DN=DN, DR=DR, QT=QT, BN=BN, DOT=DOT, SCALE=scale)
            lens = [i32(swa)] + ([i32(nc)] if NX else [])
            outs = [ctx.call(fn, q[0][:NH], _stack([r[0] for r in rows], DN, 8), _stack([r[1] for r in rows], DN // QT, 8),
                             _stack([r[2] for r in rows], DR, 16), array_of(idx), *lens, sink, node=node, rule=self, name=f"{h}[t={p}]")]
            pre.update(path="decode", swa_len=swa, extra_len=nc, NW=NW, SPLITS=list(SPLITS))
            emitted = f"{fn.id}"
        else:
            gather = ctx.kind("flashmla_gather_row", node, DN=DN, DR=DR, QT=QT)
            gathered: dict[tuple[str, int], Coll] = {}

            def row(cache: str, j: int) -> Coll:
                if (cache, j) not in gathered:
                    gathered[(cache, j)] = ctx.call(gather, *hist[cache][j], node=node, rule=self, name=f"{h}/gather[{cache} {j}]")
                return gathered[(cache, j)]

            outs, pairs = [], []
            for i, p in enumerate(pos):
                s0, tl = max(p - window + 1, P), (p + 1) // ratio - P // ratio if ratio > 1 else 0
                n = tl + p - s0 + 1
                NP = _blocks(_blocks(n, BN), 2)
                rows = [row("compressed", j) for j in range(tl)] + [row("window", r - P) for r in range(s0, p + 1)]
                fn = ctx.kind("flashmla_prefill", node, NH=NH, NP=NP, D=D, BN=BN, DOT=DOT, SCALE=scale)
                pad = 2 * NP * BN - n
                outs.append(ctx.call(fn, q[i][:NH], array_of(rows + rows[:1] * pad), array_of([i32(0)] * n + [i32(-1)] * pad), i32(n),
                                     i32(1), sink, node=node, rule=self, name=f"{h}[t={p}]"))
                pairs.append(NP)
            pre.update(path="prefill", gathered_rows=len(gathered), pairs=sorted(set(pairs)))
            emitted = f"{gather.id} x{len(gathered)} + FlashMlaPrefill_v1 x{T} (pairs {sorted(set(pairs))})"
        ctx.define(node, array_of(outs))
        ctx.bind_weight_role(a["sink"], self, "dsv4.attn_sink", NH=NHP)
        ctx.bind_weight_role(a["cos_sin"], self, "rope.cos_sin_cache", D=DR)
        if ratio > 1:
            ctx.bind_weight_role(a["norm_weight"], self, "dsv4.compressor.norm.weight", D=D)
        ctx.record(self, node, pre, emitted=emitted + (f" + {COMPRESSORS[ratio]} x{len(made)}" if made else ""))

    def _compress(self, ctx, node, a, state: list[tuple[Coll, Coll]], P: int, p: int, D: int, QB: int, h: str) -> tuple[Coll, Coll, Coll]:
        """The compressed row of the window ending at p over the state rows from position P, as the attention reads it: (NoPE, scale
        bytes without the pad, RoPE)."""
        ratio, eps = a["ratio"], float(a["eps"])
        win = 2 * ratio if ratio == 4 else ratio
        taken = [(r, s - P) for r, s in enumerate(range(p - win + 1, p + 1)) if s >= P]
        if ratio == 4:
            fn = ctx.kind("dsv4_compress_c4", node, ROWS=len(taken), D=D, QB=QB, EPS=eps)
        else:
            fn = ctx.kind("dsv4_compress_c128", node, W=C.DSV4_COMPRESSOR["W"], R=ratio // C.DSV4_COMPRESSOR["W"], D=D, QB=QB, EPS=eps)
        kv, score = (_stack([state[s][k][r // ratio * D:(r // ratio + 1) * D] for r, s in taken], D, 32) for k in (0, 1))
        out = ctx.call(fn, kv, score, ctx.coll(a["norm_weight"]), ctx.coll(a["cos_sin"])[p - ratio + 1], node=node, rule=self,
                       name=f"{h}/compress[t={p}]")
        return out[0], out[2][:(D - QB) // QB], out[1]


class Dsv4InvRopeQuantRule(OpRule):
    """`vllm.fused_inv_rope_fp8_quant_kernel`, the o-projection's input (`deep_gemm_fp8_o_proj`), per token and head:
    `Dsv4InvRopeQuantHead_v1{CHUNKS,QG,ROPE}` of o[t, h] and cos_sin[positions[t]].  Head h = g * heads_per_group + j writes its
    bytes at [g, t, j * HD : (j + 1) * HD] and its CHUNKS f32 scales at [g, t, j * CHUNKS : (j + 1) * CHUNKS]; the scale buffer's
    rows past T (its TMA alignment) hold no value."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4InvRopeQuant[vllm-dsv4]", 1, "op-binding", applies_to="vllm.fused_inv_rope_fp8_quant_kernel",
                    preconditions=("the target's compute capability is 9.0 and tma_aligned_scales false: f32 scales (cc 10.0 packs UE8M0)",
                                   "positions a static 1-D index of num_tokens = T rows; o [T, n_groups * heads_per_group, HD] bf16",
                                   "cos_sin an f32 rotary cache [MAX_POS, 2 half_rope] holding every position",
                                   "HD = chunks_per_head * quant_group_size, the rotary suffix quant_group_size - rope_start = 2 half_rope, "
                                   "d = heads_per_group * HD, scale_inner = heads_per_group * chunks_per_head; fp8_max 448",
                                   "not under registered_values"),
                    evidence_domain="structural; numerics: Dsv4InvRopeQuantHead_v1 (registry.dsv4_indexer)")

    def targets(self):
        return targets_for("fused_inv_rope_fp8_quant_kernel")

    @staticmethod
    def _geometry(a) -> tuple[int, int, int, int, int]:
        return a["n_groups"], a["heads_per_group"], a["chunks_per_head"], a["quant_group_size"], 2 * a["half_rope"]

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["dsv4_inv_rope_quant_head"], ROLE_KERNELS)):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: the quantisation has no registered-values twin"
        a = call_args(node)
        sizes = ("heads_per_group", "quant_group_size", "chunks_per_head", "rope_start", "half_rope", "num_tokens", "n_groups", "d",
                 "scale_inner")
        if not (all(isinstance(a[k], int) for k in sizes) and isinstance(a["tma_aligned_scales"], bool) and _static_float(a["fp8_max"])):
            return "the launch's sizes, tma_aligned_scales and fp8_max are not statics"
        tp = active_target_profile()
        if tp.compute_capability != (9, 0) or a["tma_aligned_scales"]:
            return (f"numerics_unregistered: the Definition is the cc 9.0 PTX with f32 scales; the target is cc {tp.capability_str}, "
                    f"tma_aligned_scales {a['tma_aligned_scales']}")
        if float(a["fp8_max"]) != 448.0:
            return f"numerics_unregistered: fp8_max {a['fp8_max']} (the Definition clamps to 448)"
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T, (G, HPG, CH, QG, ROPE) = len(pos), self._geometry(a)
        if (r := _value(ctx, a["o"], BF16, (T, G * HPG, CH * QG), "o") or _table(ctx, a["cos_sin_cache"], ROPE, pos, "cos_sin", F32)):
            return r
        if (a["num_tokens"], a["d"], a["scale_inner"], QG - a["rope_start"]) != (T, HPG * CH * QG, HPG * CH, ROPE):
            return (f"the launch is not {T} tokens of {HPG} heads a group, each {CH} groups of {QG} with a rotary suffix of {ROPE}: "
                    f"num_tokens {a['num_tokens']}, d {a['d']}, scale_inner {a['scale_inner']}, rope_start {a['rope_start']}")
        try:
            I.check_head_statics(QG, ROPE)
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = call_args(node)
        pos = ctx.static_index(a["positions"]).flat()
        T, h, (G, HPG, CH, QG, ROPE) = len(pos), ctx.hint_name(node), self._geometry(a)
        fn = ctx.kind("dsv4_inv_rope_quant_head", node, CHUNKS=CH, QG=QG, ROPE=ROPE)
        o, cs = _rows(ctx, a["o"], 16), ctx.coll(a["cos_sin_cache"])
        outs = [[ctx.call(fn, o[i][k], cs[p], node=node, rule=self, name=f"{h}[t={p},h={k}]") for k in range(G * HPG)]
                for i, p in enumerate(pos)]

        def grouped(k: int, n: int, w: int) -> Coll:
            refs = [outs[i][g * HPG + j][k].refs for g in range(G) for i in range(T) for j in range(HPG)]
            return Coll(Array(G, Array(T, Array(HPG * n, Value(w)))), concat(refs))

        ctx.define(node, [grouped(0, CH * QG, 8), grouped(1, CH, 32)])
        ctx.bind_weight_role(a["cos_sin_cache"], self, "rope.cos_sin_cache", D=ROPE)
        ctx.record(self, node, {"T": T, "positions": [pos[0], pos[-1]], "G": G, "HPG": HPG, "CHUNKS": CH, "QG": QG, "ROPE": ROPE},
                   emitted=f"{fn.id} x{T * G * HPG}")


VLLM_DSV4_RULES = [Dsv4QNormRopeKvInsertRule(), Dsv4CompressorStateRule(), Dsv4SparseAttnRule(), Dsv4InvRopeQuantRule()]

__all__ = ["COMPRESSORS", "Dsv4CompressorStateRule", "Dsv4InvRopeQuantRule", "Dsv4QNormRopeKvInsertRule", "Dsv4SparseAttnRule",
           "FLASHMLA_GEOMETRY", "VLLM_DSV4_RULES"]
