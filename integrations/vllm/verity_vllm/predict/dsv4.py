"""The hyper-connected compressed-attention MoE decoder (`FAMILY_FACTS[...]["arch"]` "mhc-moe": DeepSeek-V4's `deepseek_v4`, which
vLLM serves as `DeepseekV4ForCausalLM`) for `build.build_program`: one tensor-parallel rank of its block-FP8 checkpoint (MXFP4
experts) on cc 9.0 without batch invariance under vLLM d9105ea80's served defaults, written with the calls the Build's rules make
(`rules.vllm_mhc`, `vllm_dsv4`, `vllm_bindings.fp8`, `bf16_gemm`, `triton_launches`, `collectives`, `vllm_moe_router`,
`vllm_marlin_moe`) at the selections they read: the captured cuBLAS and all-reduce routes, the Marlin configs the capture saw
(`vllm_marlin_moe.served_config`) and FlashMLA's planner (`kernels.flashmla_twin`).  The configuration is the family's facts over
the rank's config.json (`generic.rank_hf_config`, `family_facts.resolve_facts`); the weights record lists the family's `params` in
their order, each at the shape the model allocates.

Vocabulary: EmbeddingShard and its all-reduce; layer 0's MhcPreBroadcast and MhcResidualBroadcast; per layer the compressor's f32
cuBLAS GEMM (on an indexer layer, ratio 4, the indexer's weights projection and compressor GEMMs too, which its short path never
reads), the block-FP8 linears (FlashInfer below 32 rows), TritonBlockRMSNorm of q and kv, Dsv4QNormRopeKvInsert,
Dsv4CompressorScore, FlashMlaDecode over the layer's window, Dsv4InvRopeQuantHead per head and the grouped o projection
(ScaledMmFp8BlockGrouped); between sublayers the all-reduce and MhcFusedPostPre; the MoE block: the router GEMM, the shared expert
(SiluMulClamp), the table route (GatherI32x, MoeRouterHash) below HASH_LAYERS and MoeRouterSqrtSoftplus from it on, MoeAlignRanks
over the schedule's keys and MoeAlignBlocks, the Marlin expert GEMMs per pair, MoeSum and the shared expert's add; then MhcPost,
HcHead, the final norm, the last row's logits, their all-gather and TokenSelect.  The domain is one engine step's single row, which
opens every layer's history: a fact outside it raises `Unexpressible`, φ's when the decoder is made and a launch's (a prefill, a
second call, a position that completes a compressed row, a Marlin batch no capture determines) when it is written.
"""

from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import Builder
from verity.ml.circuits.refs import Coll, Refs, concat, strided_view
from verity.ml.circuits.types import Type

from . import dsa
from .common import Unexpressible, _arr, _Peers, served_class
from .phi import Phi

_refuse = dsa._refuse
Shape = tuple[tuple[int, ...], int]


class Net:
    """One rank's decoder: the configuration, the weights record (`leaves`), the peers and schedule words a pass reads
    (`peer_shapes`, `schedule_words`) and the target's selections the forward (`model`) reads."""

    DOMAIN = ((8,), "eager", "float8_e4m3fn")

    def __init__(self, phi: Phi, M: int, history: int) -> None:
        from verity_vllm.engine.profiles import family_facts as FF
        from verity_vllm.engine.profiles.generic import rank_hf_config
        from verity_vllm.program import model as analytic
        from verity_vllm.program.frontend import targets as TG
        from verity_vllm.program.frontend.rules.vllm_dsv4 import COMPRESSORS, FLASHMLA_GEOMETRY
        from verity_vllm.program.frontend.rules.vllm_mhc import MHC_GEOMETRY
        from verity_vllm.program.registry import dsv4_indexer
        from verity_vllm.program.registry import dsv4_qnorm_insert as Q
        from verity_vllm.program.registry.dsa_norm import check_block
        from verity_vllm.program.registry.dsv4_moe import TOPK as ROUTER_SLOTS
        from verity_vllm.program.registry.tp_allreduce import settings_of
        hf, ff = phi.hf_config, FF.facts_for(phi.hf_config)
        self.world, self.rank = int(phi.tp["world"]), int(phi.tp.get("rank", 0))
        qc = hf.get("quantization_config") or {}
        try:
            rc = rank_hf_config(hf, self.world)
            C, _ = FF.resolve_facts(rc, {})
            C.update(H=int(hf["hidden_size"]), EPS=float(hf["rms_norm_eps"]), NL=int(hf["num_hidden_layers"]), NH=int(rc["num_attention_heads"]),
                     V=int(rc["vocab_size"]), VOCAB=int(hf["vocab_size"]), I=int(rc["intermediate_size"]),
                     MAX_POS=int(hf["max_position_embeddings"]), FP8_BLOCK=int((qc.get("weight_block_size") or [0])[0]))
            self.ratios = [max(1, int(r)) for r in C["COMPRESS_RATIOS"][:C["NL"]]]
            H, D, DR, QL, NH, NHP = (int(C[k]) for k in ("H", "HEAD_DIM", "QK_ROPE", "Q_LORA", "NH", "PADDED_HEADS"))
            OG = int(C["O_GROUPS"]) // self.world
        except (KeyError, ValueError, TypeError) as e:
            raise Unexpressible(f"config.json: {e}") from None
        self.C, self.tp, self.NL = C, phi.target_profile(), C["NL"]
        self.G = FLASHMLA_GEOMETRY.get((D, DR))
        facts = []
        if self.tp.compute_capability != (9, 0):
            facts.append(f"target cc {self.tp.capability_str} (the served defaults are stated on cc 9.0)")
        if hf.get("tie_word_embeddings"):
            facts.append("tied embeddings")
        if (why := analytic.quant_refusal(qc) if qc else "a checkpoint without its block-FP8 quantization_config"):
            facts.append(why)
        route = (C["EXPERT_DTYPE"], C["SCORING"], C["RENORMALIZE"], C["SCORE_BIAS"], int(C["TOPK"]))
        if route != ("fp4", "sqrtsoftplus", True, "noaux_tc", ROUTER_SLOTS):
            facts.append(f"{C['EXPERT_DTYPE']} experts routed top {C['TOPK']} by {C['SCORING']} scores, renormalize {C['RENORMALIZE']}, "
                         f"topk_method {C['SCORE_BIAS']} (the Marlin MXFP4 experts and the routers' {ROUTER_SLOTS} slots of renormalized "
                         f"sqrtsoftplus scores with a correction bias)")
        if not C["SHARED_I"]:
            facts.append("a MoE block without a shared expert")
        if (odd := sorted(set(self.ratios) - set(COMPRESSORS))):
            facts.append(f"compress ratios {odd} (the compressors: {sorted(COMPRESSORS)}; a window-only layer rotates by a rope cache "
                         f"of its own)")
        if H not in MHC_GEOMETRY:
            facts.append(f"no pinned mHC thread geometry at hidden size {H} (pinned: {sorted(MHC_GEOMETRY)})")
        if OG < 1 or NH % OG:
            facts.append(f"{C['O_GROUPS']} o-projection groups over {self.world} ranks of {NH} heads")
        block = 1 << (max(QL, D) - 1).bit_length()
        self.norm = {"BLOCK": block, "WARPS": 8 if block >= 2048 else 4, "EPS": C["EPS"]}
        if self.G is None:
            facts.append(f"no FlashMLA sparse geometry at head {D}, rope {DR} (registered: {sorted(FLASHMLA_GEOMETRY)})")
        else:
            try:
                if self.G["QT"] != DR:
                    raise ValueError(f"the compressor rotates its last quant group, so its block {self.G['QT']} is the rope width {DR}")
                Q.check_insert_statics(NH, NHP)
                Q.check_q_statics(D, DR)
                Q.check_kv_statics(D - DR, DR, self.G["QT"])
                dsv4_indexer.check_head_statics(int(C["O_QUANT_GROUP"]), DR)
                for N in (QL, D):
                    check_block(N, **self.norm)
                self.dot = TG.tensor_core_dot(self.tp)
            except (ValueError, TG.UnregisteredTarget) as e:
                facts.append(str(e))
        if facts:
            _refuse("; ".join(facts))
        self.settings = settings_of(bool(phi.engine.get("disable_custom_all_reduce")), phi.engine.get("env") or {})
        self.mhc = {"HC": int(C["HC_MULT"]), "H": H, "TN": MHC_GEOMETRY[H]["TN"], "REPEAT": int(C["HC_REPEAT"]), "EPS": C["EPS"],
                    "HC_EPS": float(C["HC_EPS"])}
        self.model_class = served_class(ff["vllm_class"])
        self.nodes_per_row_layer = 20 + NH + 6 * int(C["TOPK"])
        P, whole = ff["params"], self._whole()
        self._leaves: list[tuple[str, Type, object]] = [(n, _arr(*whole[n]), n.removeprefix("model.").removesuffix(".weight")) for n in P["top"]]
        held = [self._layer(L) for L in range(self.NL)]
        for L, h in enumerate(held):
            if (stray := set(h) - set(P["order"]) - set(P["buffers"])):
                raise AssertionError(f"layer {L} holds {sorted(stray)}, which the family's params do not order")
            self._leaves += [(f"model.layers.{L}.{n}", _arr(*h[n]), (L, n)) for n in P["order"] if n in h]
        self._leaves += [(n, _arr(*whole[n]), n.removeprefix("model.").removesuffix(".weight")) for n in P["tail"]]
        self._leaves += [(f"model.layers.{L}.{n}", _arr(*h[n]), (L, n)) for L, h in enumerate(held) for n in P["buffers"] if n in h]

    def _fp8(self, name: str, N: int, K: int) -> dict[str, Shape]:
        G = int(self.C["FP8_BLOCK"])
        return {f"{name}.weight": ((N, K), 8), f"{name}.weight_scale_inv": ((-(-N // G), -(-K // G)), 32)}

    def _whole(self) -> dict[str, Shape]:
        C = self.C
        H, HC, V = int(C["H"]), int(C["HC_MULT"]), int(C["V"])
        return {"model.hc_head_fn": ((HC, HC * H), 32), "model.hc_head_base": ((HC,), 32), "model.hc_head_scale": ((1,), 32),
                "model.embed_tokens.weight": ((V, H), 16), "model.norm.weight": ((H,), 16), "lm_head.weight": ((V, H), 16)}

    def _layer(self, L: int) -> dict[str, Shape]:
        """The parameters and buffers layer L holds, by name below `model.layers.<L>.`: the mHC mixes (layer 0's broadcast pre too),
        the attention (an indexer at ratio 4, a compressor of the D-wide rows at every ratio, overlapping two windows at ratio 4), the
        router (its table below HASH_LAYERS, else its correction bias), the shared and the MXFP4 routed experts; layer 0 holds the one
        rope cache."""
        from verity_vllm.program.registry import marlin_moe as MM
        C, r = self.C, self.ratios[L]
        H, HC, D, DR, QL, NH, NHP, E, I, SI, IH, ID, OL, TOPK = (int(C[k]) for k in (
            "H", "HC_MULT", "HEAD_DIM", "QK_ROPE", "Q_LORA", "NH", "PADDED_HEADS", "NUM_EXPERTS", "I", "SHARED_I", "INDEX_HEADS",
            "INDEX_HEAD_DIM", "O_LORA", "TOPK"))
        OUT, OG = HC * (2 + HC), int(C["O_GROUPS"]) // self.world

        def compressor(prefix: str, width: int) -> dict[str, Shape]:
            cw = width * (2 if r == 4 else 1)
            return {f"{prefix}.ape": ((r, cw), 32), f"{prefix}.fused_wkv_wgate.weight": ((2 * cw, H), 16), f"{prefix}.norm.weight": ((width,), 16)}

        out: dict[str, Shape] = {f"hc_{s}_{k}": v for s in ("attn", "ffn") for k, v in (("fn", ((OUT, HC * H), 32)), ("base", ((OUT,), 32)),
                                                                                         ("scale", ((3,), 32)))}
        if L == 0:
            out |= {"hc_attn_fn_broadcast": ((OUT, H), 32), "attn.rotary_emb.cos_sin_cache": ((int(C["MAX_POS"]), DR), 32)}
        out |= {"attn.attn_sink": ((NHP,), 32), "attn.q_norm.weight": ((QL,), 16), "attn.kv_norm.weight": ((D,), 16)}
        out |= (self._fp8("attn.fused_wqa_wkv", QL + D, H) | self._fp8("attn.wq_b", NH * D, QL) | self._fp8("attn.wo_a", OG * OL, NH * D // OG)
                | self._fp8("attn.wo_b", H, OG * OL))
        if r == 4:
            out |= self._fp8("attn.indexer.wq_b", IH * ID, QL) | compressor("attn.indexer.compressor", ID)
            out["attn.indexer.weights_proj.weight"] = ((IH, H), 16)
        out |= compressor("attn.compressor", D)
        out["ffn.gate.weight"] = ((E, H), 16)
        if L < int(C["HASH_LAYERS"]):
            out["ffn.gate.tid2eid"] = ((int(C["VOCAB"]), TOPK), 32)
        else:
            out["ffn.gate.e_score_correction_bias"] = ((E,), 32)
        out |= self._fp8("ffn.shared_experts.gate_up_proj", 2 * SI, H) | self._fp8("ffn.shared_experts.down_proj", H, SI)
        for k, N, K in (("w13", 2 * I, H), ("w2", H, I)):
            out |= {f"ffn.experts.routed_experts.{k}_weight": ((E, N, K // 2), 8), f"ffn.experts.routed_experts.{k}_weight_scale": ((E, N, K // MM.GROUP), 8)}
        return out | {"attn_norm.weight": ((H,), 16), "ffn_norm.weight": ((H,), 16)}

    def leaves(self) -> list[tuple[str, Type]]:
        return [(name, t) for name, t, _ in self._leaves]

    def peer_shapes(self, r: int) -> list[tuple[int, int]]:
        """A pass of r rows: the embedding's all-reduce, each layer's attention and MoE all-reduces, the sampled row's logits gather."""
        H = int(self.C["H"])
        return [(r, H)] * (1 + 2 * self.NL) + [(1, int(self.C["V"]))]

    def schedule_words(self, r: int) -> int:
        """A pass of r rows: each layer's MoE align reads one key per (row, slot) pair."""
        return self.NL * r * int(self.C["TOPK"])

    def model(self, B: Builder, W: Coll, peers: _Peers) -> "_Model":
        return _Model(self, B, W, peers)


class _Model(dsa._Model):
    """The forward over one call's row, emitted into `B`: the HC residual streams and their (post, comb) mixes carried from sublayer
    to sublayer; the collectives, linears, router GEMM and sampler are `dsa._Model`'s."""

    def __init__(self, net: Net, B: Builder, W: Coll, peers: _Peers) -> None:
        super().__init__(net, B, W, peers)
        self.calls = 0

    def fp8(self, xs: list[Refs], L: int, name: str) -> list[Refs]:
        return self.fp8_linear(xs, self.w[(L, f"{name}.weight")], self.w[(L, f"{name}.weight_scale_inv")])

    def forward(self, toks: list[Coll], pos0: int) -> list[Refs]:
        """Token rows (I32 values) at positions pos0.. -> the final-normed hidden rows."""
        from verity_vllm.program.frontend.rules.vllm_mhc import MHC_GEOMETRY
        C, net, H, M = self.C, self.net, self.H, len(toks)
        if M > 1:
            _refuse(f"a call of {M} rows, a prefill (FlashMLA's sparse prefill and the compressors are not written)")
        if self.calls:
            _refuse("a second call, which reads the window rows of the call before it")
        self.calls += 1
        V = int(C["V"])
        emb = self.k("embedding_shard", VS=V, H=H, START=net.rank * V)
        streams, x, mix = self.pre(self.all_reduce([self.call(emb, t.refs, self.w["embed_tokens"].refs).refs for t in toks]))
        for L in range(net.NL):
            streams, x, mix = self.post_pre(self.attention(L, x, pos0), streams, mix, L, "ffn")
            out = self.moe(L, x, toks)
            if L + 1 < net.NL:
                streams, x, mix = self.post_pre(out, streams, mix, L + 1, "attn")
        post = self.k("mhc_post", HC=net.mhc["HC"], N=H)
        streams = [self.call(post, a, s, p, c).refs for a, s, (p, c) in zip(self.all_reduce(out), streams, mix)]
        head = self.k("hc_head", HC=net.mhc["HC"], H=H, TH=MHC_GEOMETRY[H]["TH_HEAD"], EPS=self.EPS, HC_EPS=net.mhc["HC_EPS"])
        x = [self.call(head, s, self.w["hc_head_fn"].refs, self.w["hc_head_scale"].refs, self.w["hc_head_base"].refs).refs for s in streams]
        norm = self.k("rmsnorm", N=H, EPS=self.EPS, TARGET=net.tp, M=M)
        return [self.call(norm, r, self.w["norm"].refs).refs for r in x]

    # -- the hyper-connections (`vllm_mhc`)

    def pre(self, rows: list[Refs]) -> tuple[list[Refs], list[Refs], list[tuple[Refs, Refs]]]:
        """Layer 0's pre over the embedding rows (`MhcPreBroadcastRule`) -> (streams, layer input, (post, comb)) per row."""
        from verity_vllm.program.kernels.mhc_twin import tf32_num_split
        w = lambda f: self.w[(0, f)].refs  # noqa: E731
        fn = self.k("mhc_pre_broadcast", **self.net.mhc, NS=tf32_num_split(len(rows), self.H, int(self.net.tp.num_sms)))
        outs = [self.call(fn, r, w("hc_attn_fn_broadcast"), w("hc_attn_scale"), w("hc_attn_base"), w("attn_norm.weight")) for r in rows]
        copy = self.k("mhc_residual_broadcast", HC=self.net.mhc["HC"], H=self.H)
        return [self.call(copy, r).refs for r in rows], [o[0].refs for o in outs], [(o[1].refs, o[2].refs) for o in outs]

    def post_pre(self, parts: list[Refs], streams: list[Refs], mix: list, L: int, side: str) -> tuple[list[Refs], list[Refs], list]:
        """The all-reduce of a sublayer's partials, its post into the streams and the pre of layer L's `side` sublayer, on the FMA path
        of at most `SMALL_FMA_TOKENS` rows (`MhcFusedPostPreRule`) -> (streams, layer input, (post, comb)) per row."""
        from verity_vllm.program.frontend.rules.vllm_mhc import MHC_GEOMETRY
        w = lambda f: self.w[(L, f)].refs  # noqa: E731
        x = self.all_reduce(parts)
        fn = self.k("mhc_fused_post_pre", **self.net.mhc, NS=8 if len(x) < 8 and self.H <= 4096 else 4, TH=MHC_GEOMETRY[self.H]["TH"])
        outs = [self.call(fn, a, s, p, c, w(f"hc_{side}_fn"), w(f"hc_{side}_scale"), w(f"hc_{side}_base"), w(f"{side}_norm.weight"))
                for a, s, (p, c) in zip(x, streams, mix)]
        return [o[0].refs for o in outs], [o[1].refs for o in outs], [(o[2].refs, o[3].refs) for o in outs]

    # -- the layer

    def attention(self, L: int, x: list[Refs], pos0: int) -> list[Refs]:
        """Layer L's attention over the row x at position pos0, which opens the layer's history (`vllm_dsv4`, op-major): the
        compressor's state row (its score; no compressed row is complete), the query and the window row, FlashMLA's decode over the
        window, the inverse RoPE and the grouped o projection (`fp8.DeepGemmFp8EinsumRule`) -> the o projection's partials."""
        from verity_vllm.program.frontend.rules.vllm_bindings.fp8 import FP8_QUANT_PINS
        from verity_vllm.program.kernels.flashmla_twin import num_sm_parts, request_splits, sched_meta
        C, net, p, r = self.C, self.net, pos0, self.net.ratios[L]
        if (p + 1) % r == 0:
            _refuse(f"position {p} completes compressed row {(p + 1) // r - 1} of layer {L} (ratio {r}): the compressors are not written")
        D, DR, QL, NH, NHP, QG = (int(C[k]) for k in ("HEAD_DIM", "QK_ROPE", "Q_LORA", "NH", "PADDED_HEADS", "O_QUANT_GROUP"))
        DN, QT, BN, CW, OG = D - DR, net.G["QT"], net.G["BN"], D * (2 if r == 4 else 1), int(C["O_GROUPS"]) // net.world
        w = lambda f: self.w[(L, f)].refs  # noqa: E731
        cs = self.w[(0, "attn.rotary_emb.cos_sin_cache")].refs.slice(p * DR, (p + 1) * DR)
        kvs = self.cublas(x, self.w[(L, "attn.compressor.fused_wkv_wgate.weight")], f32=True)
        indexer = r == 4
        if indexer:
            self.cublas(x, self.w[(L, "attn.indexer.weights_proj.weight")])
            self.cublas(x, self.w[(L, "attn.indexer.compressor.fused_wkv_wgate.weight")], f32=True)
        qkv = self.fp8(x, L, "attn.fused_wqa_wkv")
        qn, kn = (self.k("triton_block_rmsnorm", N=N, **net.norm) for N in (QL, D))
        q = [self.call(qn, a.slice(0, QL), w("attn.q_norm.weight")).refs for a in qkv]
        kv = [self.call(kn, a.slice(QL, QL + D), w("attn.kv_norm.weight")).refs for a in qkv]
        score = self.k("dsv4_compressor_score", W=CW)
        ape = w("attn.compressor.ape").slice(p % r * CW, (p % r + 1) * CW)

        def state() -> None:
            for s in kvs:
                self.call(score, s.slice(CW, 2 * CW), ape)

        if indexer:   # the Build's order: an indexer layer's compressor state precedes its query
            state()
        ins = self.k("dsv4_qnorm_rope_kv_insert", NH=NH, NHP=NHP, DN=DN, DR=DR, QB=QT, EPS=self.EPS)
        rows = [self.call(ins, a, b, cs) for a, b in zip(self.fp8(q, L, "attn.wq_b"), kv)]
        if not indexer:
            state()
        consts: dict[int, Refs] = {}

        def i32(v: int) -> Refs:
            if v not in consts:
                consts[v] = self.call(self.k("const", W=32, BITS=v & 0xFFFFFFFF)).refs
            return consts[v]

        swa, NW = 1, BN
        idx = concat([i32(0)] * swa + [i32(-1)] * (NW - swa))
        metas, ns = sched_meta([swa], [0], topk=int(C["WINDOW"]), extra_topk=1, parts=num_sm_parts(NHP, 1, int(net.tp.num_sms)), block_size_n=BN)
        dec = self.k("flashmla_decode", NH=NH, SPLITS=request_splits(metas, ns, 0, NW // BN), NW=NW, DN=DN, DR=DR, QT=QT, BN=BN, DOT=net.G["DOT"],
                     SCALE=float(C["ATTN_SCALE"]))
        att = [self.call(dec, o[0].refs.slice(0, NH * D), concat([o[1].refs] * NW), concat([o[2].refs.slice(0, DN // QT)] * NW),
                         concat([o[3].refs] * NW), idx, i32(swa), w("attn.attn_sink").slice(0, NH)).refs for o in rows]
        inv = self.k("dsv4_inv_rope_quant_head", CHUNKS=D // QG, QG=QG, ROPE=DR)
        wo_a = self.k("deepgemm_fp8_einsum_grouped", H=OG, K=NH // OG * D, N=int(C["O_LORA"]), G=FP8_QUANT_PINS["group_size"])
        out = []
        for o in att:
            heads = [self.call(inv, o.slice(h * D, (h + 1) * D), cs) for h in range(NH)]
            out.append(self.call(wo_a, concat([h[0].refs for h in heads]), concat([h[1].refs for h in heads]), w("attn.wo_a.weight"),
                                 w("attn.wo_a.weight_scale_inv")).refs)
        return self.fp8(out, L, "attn.wo_b")

    def moe(self, L: int, x: list[Refs], toks: list[Coll]) -> list[Refs]:
        """The MoE block over the rows x (`vllm_moe_router`, `vllm_marlin_moe`, op-major): the router logits, the shared expert, each
        row's six experts and weights, the align over the pairs p = t * TOPK + k and their schedule keys, per pair the w13 GEMM, the
        clamped SwiGLU and the weighted w2 GEMM; each row's sum of its TOPK rows, plus the shared expert's."""
        from verity_vllm.program.frontend.rules.vllm_marlin_moe import served_config
        from verity_vllm.program.kernels.marlin_moe_rows import moe_block
        from verity_vllm.program.registry import marlin_moe as MM
        C, M, H = self.C, len(x), self.H
        E, TOPK, I, SCALE = int(C["NUM_EXPERTS"]), int(C["TOPK"]), int(C["I"]), float(C["ROUTED_SCALING_FACTOR"])
        P = M * TOPK
        w = lambda f: self.w[(L, f)]  # noqa: E731
        logits = self.router(x, w("ffn.gate.weight"))

        def swiglu(rows: list[Refs]) -> list[Refs]:
            return [self.call(self.k("silu_mul_clamp", I=a.count // 2, LIMIT=float(C["SWIGLU_LIMIT"])), a).refs for a in rows]

        shared = self.fp8(swiglu(self.fp8(x, L, "ffn.shared_experts.gate_up_proj")), L, "ffn.shared_experts.down_proj")
        if L < int(C["HASH_LAYERS"]):
            V, table = int(C["VOCAB"]), w("ffn.gate.tid2eid")
            gather = self.k("gather_i32", V=V)
            ids = [concat([self.call(gather, t.refs, strided_view(table, k, V, V * TOPK, TOPK, V)).refs for k in range(TOPK)]) for t in toks]
            wts = [self.call(self.k("moe_router_hash", E=E, SCALE=SCALE), lg, i).refs for lg, i in zip(logits, ids)]
        else:
            top = [self.call(self.k("moe_router_sqrtsoftplus", E=E, SCALE=SCALE), lg, w("ffn.gate.e_score_correction_bias").refs) for lg in logits]
            wts, ids = [t[0].refs for t in top], [t[1].refs for t in top]
        ids, wts, BM = concat(ids), concat(wts), moe_block(M, TOPK, E)
        pairs = concat([self.call(self.k("const", W=32, BITS=p)).refs for p in range(P)])
        rank = self.call(self.k("marlin_moe_rank", E=E, P=P, BM=BM), ids, self.peers.key(P), pairs)
        sched = self.call(self.k("marlin_moe_align", P=P, BM=BM), ids, rank[0].refs, rank[1].refs)
        blk, par = sched[0].refs, sched[1].refs

        def launch(role: str, K: int, N: int, top_k: int):
            cfg = served_config(E, TOPK, K, N, top_k, M, self.net.tp)
            if isinstance(cfg, str):
                _refuse(cfg)
            try:
                MM.check_gemm(SimpleNamespace(K=K, N=N, TK=cfg[0], TN=cfg[1]))
            except ValueError as e:
                _refuse(str(e))
            return self.k(role, E=E, K=K, N=N, TK=cfg[0], TN=cfg[1], GRID=cfg[2], DOT=MM.DOT)

        up, down = launch("marlin_moe_gemm", H, 2 * I, TOPK), launch("marlin_moe_gemm_w", I, H, 1)
        one = lambda a, p: a.slice(p, p + 1)  # noqa: E731
        a = swiglu([self.call(up, x[p // TOPK], one(ids, p), one(blk, p), par, w("ffn.experts.routed_experts.w13_weight").refs,
                              w("ffn.experts.routed_experts.w13_weight_scale").refs).refs for p in range(P)])
        d = [self.call(down, a[p], one(ids, p), one(wts, p), one(blk, p), par, w("ffn.experts.routed_experts.w2_weight").refs,
                       w("ffn.experts.routed_experts.w2_weight_scale").refs).refs for p in range(P)]
        msum = self.k("moe_sum", TOPK=TOPK, H=H)
        routed = [self.call(msum, concat(d[t * TOPK:(t + 1) * TOPK])).refs for t in range(M)]
        add = self.k("residual_add", N=H)
        return [self.call(add, s, r).refs for s, r in zip(shared, routed)]
