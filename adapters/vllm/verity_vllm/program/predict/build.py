"""BuildProgram(φ) -> the Program the Build derives for a row, constructed ahead of time from a hand-written vocabulary.

The Build (`pipeline.build`) traces vLLM's model class on meta with torch.export and lets the binding rules translate each node;
this module writes the same Program down directly from φ (`phi.Phi`: the checkpoint's `config.json`, the declared target, the
wrapper shape and the sampling policy), with the same circuit-building calls the derive makes: one `Builder` over the wrapper's user inputs and
the weights record (vLLM's parameters in module order, then its buffers), one call per row of each launch, the vocabulary's kinds
(`rules.vocab.PROFILE_B1_EAGER_V3.kind`) with the statics the rules select through `frontend.targets`, the flat body wrapped as
`Derived[<wrapper>[<model class>]]_v1{SRC}` and inlined by `Program`.  Nothing here traces, loads vLLM or reads a Build output.
The checkpoint's shape and its weights record are `model_shape`'s (`ModelShape`, `model_shape`, `rank_shape`, `weight_leaves`).

Vocabulary (v1, Llama-family decoders, dense or fused-MoE, under the bi-eager profile): Embedding, RMSNormTriton (layer-0 input
norm, and the q/k norm: per head, or over all heads), Gemm (target-keyed; the q/k/v projection's bias, when the family's rule gives
it one, fused as the target's GemmBias where its record says so, else a BiasAdd per row; on one rank, an exact-int8 checkpoint's
decoder linears under a target whose `int8_exact` is "linears" are Int8RowQuant per row, then ScaledMmS8 per row, its LM head the
bf16 Gemm), RoPE (all q rows, then all k rows),
Attention (target-keyed; one call per query row over the layer's history), RMSNormFusedCuda (post-attention / next-layer / final
norm), SiluMul, the fused-MoE block in its indexed-read construction (router Gemm, MoeRouterTopK per row, MoeExpertGemm / SiluMul /
MoeExpertGemmW per (token, slot) pair, MoeSum per row); for Gemma-2, the embedding's normalizer scale, GemmaRMSNorm's unfused (1 +
w) chain (every norm, the two extra MLP norms included), AttentionSoftcap, GeluTanhMul and the final logits' tanh soft-cap; and the
token select: greedy TokenSelect, or the seeded temperature + top-p GumbelTopPTokenSelect with its per-event constants (the split
count among them, the registered S). Anything φ asks for outside it raises `Unexpressible` naming the fact (a result, never a
silent approximation).

This module is NEVER on a record path: a predicted Program is not written into a Build row, not published as a Build Attempt and
not read by the Build, Match or Commit (`program.frontend.inputs_trace` lists `verity_vllm/program/predict/` as a forbidden input).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from verity.circuits.codec import canonical_json, encode_program, program_digest
from verity.circuits.defs import Builder, CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Coll, concat
from verity.circuits.types import Array, RecordType, Type, Value
from verity.definitions.replay import replayer

from verity_vllm.engine.profiles.family_facts import FAMILY_FACTS
from verity_vllm.program.inputs import check_params
from verity_vllm.program.prune import dead_free as dead_free_form

from .common import (
    Unexpressible,
    _arr,
    _arr_shape,
    _Peers,
    _ptypes,
    _Stochastic,
    drive,
    edge_config,
    in_launch,
    kind,
    launch_batch,
    launch_rows,
    launch_seqlen_q,
    request_passes,
    select_token,
    with_checks,
    with_edge,
)
from . import draft_rows as DR
from .domain import check_phi
from .model_shape import _ACTS, ModelShape, model_shape, rank_shape, weight_leaves
from .phi import Phi

__all__ = ["Unexpressible", "build_program"]

BUILDER_ID = "predict-build/derived/v1"
I32 = Value(32)
BF16 = Value(16)

#: the wrapper kinds of `vllm_meta.make_wrapper` per sampling kind: DERIVE_SOURCE (the SRC class) and the derive name
WRAPPERS = {("step", "greedy"): ("verity-vllm/engine-step/v1", "vLLMStep"),
            ("request", "greedy"): ("verity-vllm/greedy-request/v1", "vLLMRequest"),
            ("step", "stochastic"): ("verity-vllm/stochastic-engine-step/v1", "vLLMStep"),
            ("request", "stochastic"): ("verity-vllm/stochastic-request/v1", "vLLMRequest")}
CACHED_SRC = "verity-vllm/cached-greedy-request/v1"  # `vllm_meta.CachedGreedyRequest.DERIVE_SOURCE`, with a cached prefix
I64 = Value(64)


class _Specs:
    """The vocabulary's kinds for C on the declared target, bound once each (the same objects the rules' `ctx.kind` returns)."""

    def __init__(self, C: ModelShape, target, world: int = 1, rank: int = 0, M: int = 1) -> None:
        from verity_vllm.program.frontend import targets as TG
        VS, C = C.V // world, rank_shape(C, world)
        self.C, self.tp, self.TG = C, target, TG
        parts = target.int8_parts
        if C.int8 != ("linears" in parts):
            raise Unexpressible(f"vocabulary cannot express: {'an exact-int8' if C.int8 else 'a bf16'} checkpoint on a target whose "
                                f"int8_exact is {target.int8_exact!r} (the int8 linears are the checkpoint's and the target's both)")
        if "attention" in parts:
            raise Unexpressible(f"vocabulary cannot express: the exact-int8 attention (int8_exact {target.int8_exact!r}: "
                                "Int8KvQuant_v1 and Int8LutAttentionRow_v1)")
        if C.int8 and world > 1:
            raise Unexpressible(f"vocabulary cannot express: the int8 linears at TP{world} (a row-parallel shard's activation scale "
                                "is its K slice's)")
        self.accepted = TG.uses_accepted_kinds(target)
        self.dot = TG.tensor_core_dot(target)
        if world == 1:
            self.embedding = self.kind("embedding", V=C.V, H=C.H)
            self.all_reduce = self.all_gather = None
        else:
            self.embedding = self.kind("embedding_shard", VS=VS, H=C.H, START=rank * VS)
            self.all_reduce, self.all_gather = self.kind("all_reduce2", N=C.H), self.kind("all_gather2", N=VS)
        self.norm0 = self.kind("rmsnorm", N=C.H, EPS=C.EPS, TARGET=target, M=M)
        self.fused = self.kind("rmsnorm_fused", N=C.H, EPS=C.EPS, TARGET=target, M=M)
        self.qkv, self.o = self.linear(C.H, C.QKV), self.linear(C.NH * C.D, C.H)
        self.qkv_bias = self.qkv_fused = None
        if C.qkv_bias:
            role = self.gemm_bias_role()
            if role is None:
                self.qkv_bias = self.kind("bias_add", N=C.QKV)
            else:
                self.qkv = self.qkv_fused = self.kind(role[0], K=C.H, N=C.QKV, **role[1])
        self.head_norm = self.kind("rmsnorm", N=C.D, EPS=C.EPS, TARGET=target, M=M * C.NH) if C.qk_norm == "head" else None
        if C.qk_norm == "all":
            self.q_norm, self.k_norm = (
                self.kind("rmsnorm", N=n * C.D * world, EPS=C.EPS, TARGET=target, M=M) for n in (C.NH, C.KVH)
            )
            self.q_gather, self.k_gather = [self.kind("all_gather2", N=n * C.D) if world > 1 else None for n in (C.NH, C.KVH)]
        self.gate_up, self.down, self.logits = self.linear(C.H, 2 * C.I), self.linear(C.I, C.H), self.gemm(C.H, VS)
        if C.E:
            self.moe()
        self.rope_q, self.rope_k = self.kind("rope", NHEADS=C.NH, D=C.D), self.kind("rope", NHEADS=C.KVH, D=C.D)
        self.silu = self.kind(_ACTS[C.act], I=C.I)
        if C.norm_chain:
            self.chain = {r: self.kind(r, N=C.H) for r in ("square_bf16", "square_f32", "mean_triton", "add_widened_bf16", "scale_row_bf16",
                                                           "scale_row_f32", "mul_vec_f32", "narrow_f32_bf16")}
            self.chain.update(plus_one=self.kind("add_scalar_bf16", N=C.H, C=1.0), eps=self.kind("add_scalar_f32", N=1, C=C.EPS),
                              rsqrt=self.kind("rsqrt_f32", N=1))
            self.weight_once = target.effective_weight_only_calls() == "once"
        if C.embed_scale:
            self.embed_scale = self.kind("scale_tensor", N=C.H)
        if C.final_softcap is not None:
            self.cap = [self.kind("bf16_div_scalar", N=C.V, C=C.final_softcap), self.kind("bf16_tanh", N=C.V),
                        self.kind("bf16_mul_scalar", N=C.V, C=C.final_softcap)]
        self._attn: dict[tuple, Any] = {}

    def moe(self) -> None:
        """The fused-MoE block's kinds under the target's declared construction (`rules.vllm_moe`): the router GEMM, the top-k
        softmax (in selection order under `indexed-read-ordered`; the renormalizing form when the family renormalizes), the per-pair
        expert GEMMs (the target's k16 step as `DOT` where `frontend.targets.moe_expert_dot` names one) and the routed sum."""
        import torch

        from verity_vllm.program.frontend.rules.vllm_moe import (
            INDEXED_READ,
            INDEXED_READ_ORDERED,
            TOPK_SOFTMAX_VPT,
        )
        C = self.C
        c = self.tp.effective_moe_construction()
        if c not in (INDEXED_READ, INDEXED_READ_ORDERED):
            raise Unexpressible(f"vocabulary cannot express: moe_construction {c!r} (indexed-read only)")
        self.router = self.gemm(C.H, C.E)
        self.topk = self.kind("moe_router_topk" + ("_ordered" if c == INDEXED_READ_ORDERED else "") + ("_norm" if C.renorm else ""),
                              E=C.E, TOPK=C.TOPK, VPT=TOPK_SOFTMAX_VPT[torch.bfloat16])
        try:
            dot = self.TG.moe_expert_dot(self.tp)
        except self.TG.UnregisteredTarget as e:
            raise Unexpressible(f"numerics_unregistered: {e}") from None
        extra = {} if dot is None else {"DOT": dot}
        sfx = "" if dot is None else "_dot"
        self.expert_up = self.kind("moe_expert_gemm" + sfx, E=C.E, K=C.H, N=2 * C.I, **extra)
        self.expert_down = self.kind("moe_expert_gemm_w" + sfx, E=C.E, K=C.I, N=C.H, **extra)
        self.moe_sum = self.kind("moe_sum", TOPK=C.TOPK, H=C.H)

    kind = staticmethod(kind)

    def gemm(self, K: int, N: int):
        return self.kind("gemm", K=K, N=N) if self.accepted else self.kind("gemm_dot", K=K, N=N, DOT=self.dot)

    def linear(self, K: int, N: int):
        """A decoder linear's kind: the GEMM, or under the int8 linears (`rules.vllm_bindings.int8`) its quantizer and product,
        `(Int8RowQuant_v1{K}, ScaledMmS8_v1{K,N,DOT})`."""
        if not self.C.int8:
            return self.gemm(K, N)
        return self.kind("int8_row_quant", K=K), self.kind("int8_scaled_mm", K=K, N=N)

    def gemm_bias_role(self) -> tuple[str, dict] | None:
        """The biased linear's one-Call kind as `TritonGemmRule` picks it (the bias add the GEMM's only reader): `GemmBias_v1` under
        the accepted kinds, `GemmBias_v2{DOT}` where the target's record names a biased-linear step (`frontend.targets.gemm_bias_dot`,
        on the trees that have it), else None: the GEMM, then `BiasAdd_v1` per row."""
        if self.accepted:
            return "gemm_bias", {}
        dot = getattr(self.TG, "gemm_bias_dot", lambda tp: None)(self.tp)
        return None if dot is None else ("gemm_bias_dot", {"DOT": dot})

    def attention(self, P: int, row: int, own_rows: int, launch_sq: int):
        """`UnifiedAttentionWithOutput[vllm]`'s kind for the query row at history length P (row = its index among the sequence's
        query rows in the launch, own_rows = their number, launch_sq = the launch's max_seqlen_q)."""
        C, TG, tp = self.C, self.TG, self.tp
        key = (P, row, own_rows, launch_sq)
        fn = self._attn.get(key)
        if fn is not None:
            return fn
        cap = {} if C.attn_softcap is None else {"CAP": C.attn_softcap}
        if self.accepted:
            fn = self.kind("attention_softcap" if cap else "attention", T=P, NH=C.NH, KVH=C.KVH, D=C.D, **cap)
        else:
            if cap and not TG.fa2_check_inf(tp):
                raise Unexpressible(f"vocabulary cannot express: a soft-capped attention on {tp.tensor_core_family} without FA2's "
                                    "per-iteration Check_inf (the only registered target-keyed softcap chain)")
            g = TG.attention_geometry(tp, C.D, launch_sq, C.NH, C.KVH)
            extra = dict(cap)
            if TG.fa3_check_inf(tp):
                extra["MASKED_FROM"] = TG.fa3_masked_from(tp, P, C.NH, C.KVH, C.D, launch_sq, row)
            if TG.fa2_check_inf(tp):
                extra["FA2_MASKED_FROM"] = TG.fa2_masked_from(tp, P, C.D, launch_sq, row, own_rows)
            fn = self.kind("attention_dot", T=P, NH=C.NH, KVH=C.KVH, D=C.D, BN=g["BN"], DOT=g["DOT"], INV=g["INV"], **extra)
        self._attn[key] = fn
        return fn


@dataclass
class BuildResult:
    wrapper: str
    program: Any
    digest: str
    desc: dict | None
    function: str
    n_root_nodes: int
    timings: dict = field(default_factory=dict)
    named: Any = None     # what a `build_program` `finish` says beside the Program (`naming.Named`)
    checks: dict | None = None   # its checks output, as `derive`'s provenance states it (`common.with_checks`)

    def summary(self) -> dict:
        return {"wrapper": self.wrapper, "function": self.function, "digest": self.digest,
                "n_root_nodes": self.n_root_nodes, "gates": getattr(self.program, "gates", None), "timings": self.timings}


class _Model:
    """The forward of the model class over one call's rows, emitted into `B` with the per-layer K/V history carried across calls
    (the implicit KV cache the attention rule makes explicit).  C is one rank's shape (`rank_shape`); a tensor-parallel rank's
    collectives read `peers`, the other ranks' partials."""

    def __init__(self, B: Builder, specs: _Specs, W: Coll, C: ModelShape, peers: _Peers) -> None:
        self.B, self.S, self.W, self.C = B, specs, W, C
        self.peers, self.rank = peers, peers.rank
        n = 0
        self.embed = W[n]
        n += 1
        self.layers = []
        lin = (lambda k: [k, k + "_s"]) if C.int8 else (lambda k: [k])
        for _ in range(C.NL):
            names = lin("qkv") + (["qkv_b"] if C.qkv_bias else []) + (["q_norm", "k_norm"] if C.qk_norm == "all" else []) + lin("o") + \
                (["q_norm", "k_norm"] if C.qk_norm == "head" else []) + (["gate_w", "w13", "w2"] if C.E else lin("gate_up") + lin("down")) + \
                ["ln1", "ln2"] + (["pre_ff", "post_ff"] if C.extra_norms else [])
            self.layers.append({k: W[n + i] for i, k in enumerate(names)})
            n += len(names)
        self.norm = W[n]
        n += 1
        self.lm_head = self.embed if C.tie else W[n]
        n += 0 if C.tie else 1
        self.normalizer = W[n] if C.embed_scale else None
        n += 1 if C.embed_scale else 0
        self.cos_sin = W[n]
        self.hist: list[tuple[list, list]] = [([], []) for _ in range(C.NL)]
        self.cached = 0
        self.ctx: dict | None = None
        self.lp_of: dict[int, int] = {}
        self.plus_one: dict[int, Coll] = {}

    def open_with(self, prefixes: list[Coll], C: int) -> None:
        """Every layer's history opens with its rows of `prefix` (`rules.vllm_bindings.kv_prefix`: bf16 [NL, 2, C, KVH*D], keys then values)."""
        prefix, = prefixes
        rl = self.C.KVH * self.C.D
        for L, (kh, vh) in enumerate(self.hist):
            kh += [prefix.refs.slice(((2 * L) * C + j) * rl, ((2 * L) * C + j + 1) * rl) for j in range(C)]
            vh += [prefix.refs.slice(((2 * L + 1) * C + j) * rl, ((2 * L + 1) * C + j + 1) * rl) for j in range(C)]
        self.cached = C

    def call(self, fn, *args) -> Coll:
        return self.B.call(fn, *args)

    def linear(self, fn, xs: list[Coll], lw: dict, key: str) -> list[Coll]:
        """The linear `fn` (`_Specs.linear`) over the rows `xs` with the layer's weight `lw[key]`: the GEMM a row, or under the int8
        linears every row's `Int8RowQuant` then every row's `ScaledMmS8` over its codes and scale, the s8 weight and its row scales
        `lw[key + "_s"]` (`Int8RowQuant[vllm]` then `Int8ScaledMm[vllm]`, one op each over the call's rows)."""
        if not self.C.int8:
            p = _ptypes(fn)
            return [self.call(fn, Coll(p[0], x.refs), Coll(p[1], lw[key].refs)) for x in xs]
        quant, mm = fn
        pq, pm = _ptypes(quant), _ptypes(mm)
        qs = [self.call(quant, Coll(pq[0], x.refs)) for x in xs]
        return [self.call(mm, Coll(pm[0], q[0].refs), Coll(pm[1], q[1].refs), Coll(pm[2], lw[key].refs), Coll(pm[3], lw[key + "_s"].refs))
                for q in qs]

    def collective(self, fn, own: list[Coll]) -> list[Coll]:
        """`fn` (`AllReduce2` / `AllGather2`) per row over the ranks' rows in rank order (`rules.vllm_bindings.collectives`), or the
        rows themselves on one rank."""
        if self.peers.world == 1:
            return own
        p = _ptypes(fn)[0]
        return [self.call(fn, Coll(p, r)) for r in self.peers.rows([x.refs for x in own])]

    def norm_all_heads(self, parts: list[tuple]) -> list[list]:
        """The q and k norms over all heads (`OlmoeAttention._apply_qk_norm`), each part (rows, width, gather, norm, weight): each
        row normed whole, which under tensor parallelism is both parts' rows all-gathered across the ranks, both normed, then this
        rank's slice of each normed row."""
        rows = [[c.refs for c in self.collective(g, [Coll(_arr((n,), 16), r) for r in x])] if g else x for x, n, g, _, _ in parts]
        normed = [[self.call(fn, Coll(_ptypes(fn)[0], r), Coll(_ptypes(fn)[1], w.refs)).refs for r in x]
                  for x, (_, _, _, fn, w) in zip(rows, parts)]
        return [[r.slice(self.rank * n, (self.rank + 1) * n) for r in x] if g else x for x, (_, n, g, _, _) in zip(normed, parts)]

    def forward(self, toks: list[Coll], pos0: int) -> list[Coll]:
        """Token rows (I32 values) at positions pos0.. -> the final-normed hidden rows."""
        S, C, M = self.S, self.C, len(toks)
        pe = _ptypes(S.embedding)
        prq, prk, ps = _ptypes(S.rope_q), _ptypes(S.rope_k), _ptypes(S.silu)
        q_w, k_w, v_w = C.NH * C.D, C.KVH * C.D, C.KVH * C.D
        h = self.collective(S.all_reduce, [self.call(S.embedding, Coll(pe[0], t.refs), Coll(pe[1], self.embed.refs)) for t in toks])
        if C.embed_scale:
            pz = _ptypes(S.embed_scale)
            assert self.normalizer is not None, "the weights record holds the normalizer of a family that scales its embedding"
            h = [self.call(S.embed_scale, Coll(pz[0], h[i].refs), Coll(pz[1], self.normalizer.refs)) for i in range(M)]
        x, _ = self.rms(h, self.layers[0]["ln1"])
        res: list | None = [h[i].refs for i in range(M)]
        d: list[Coll] = []
        for L in range(C.NL):
            lw = self.layers[L]
            if L > 0:
                x, res = self.rms(d, lw["ln1"], res)
            if S.qkv_fused is not None:
                pq = _ptypes(S.qkv)
                qkv = [self.call(S.qkv, Coll(pq[0], x[i].refs), Coll(pq[1], lw["qkv"].refs), Coll(pq[2], lw["qkv_b"].refs)) for i in range(M)]
            else:
                qkv = self.linear(S.qkv, x, lw, "qkv")
            if S.qkv_bias is not None:
                pb = _ptypes(S.qkv_bias)
                qkv = [self.call(S.qkv_bias, Coll(pb[0], qkv[i].refs), Coll(pb[1], lw["qkv_b"].refs)) for i in range(M)]
            cs = [Coll(prq[1], self.cos_sin.refs.slice((pos0 + i) * C.rot, (pos0 + i + 1) * C.rot)) for i in range(M)]
            q_in = [qkv[i].refs.slice(0, q_w) for i in range(M)]
            k_in = [qkv[i].refs.slice(q_w, q_w + k_w) for i in range(M)]
            if C.qk_norm == "head":
                q_in, k_in = self.head_norms(q_in, C.NH, lw["q_norm"]), self.head_norms(k_in, C.KVH, lw["k_norm"])
            elif C.qk_norm == "all":
                q_in, k_in = self.norm_all_heads([(q_in, q_w, S.q_gather, S.q_norm, lw["q_norm"]),
                                                  (k_in, k_w, S.k_gather, S.k_norm, lw["k_norm"])])
            q = [self.call(S.rope_q, Coll(prq[0], q_in[i]), cs[i]) for i in range(M)]
            k = [self.call(S.rope_k, Coll(prk[0], k_in[i]), Coll(prk[1], cs[i].refs)) for i in range(M)]
            kh, vh = self.hist[L]
            P0 = len(kh)
            prefill = self.cached == P0
            if prefill:
                self.lp_of[L] = P0 + M
            own_rows = M   # one sequence's rows of the launch: a prefill's, a decode's one, a draft pass's token and drafts
            launch_sq = own_rows if self.ctx is None else launch_seqlen_q(self.ctx, own_rows, 0 if prefill else P0 - self.lp_of[L] + 1, S.tp, S.accepted)
            for i in range(M):
                kh.append(k[i].refs)
                vh.append(qkv[i].refs.slice(q_w + k_w, q_w + k_w + v_w))
            attn = []
            for i in range(M):
                P = P0 + i + 1
                fn = S.attention(P, i, own_rows, launch_sq)
                pa = _ptypes(fn)
                attn.append(self.call(fn, Coll(pa[0], q[i].refs), Coll(pa[1], concat(kh[:P])), Coll(pa[2], concat(vh[:P]))))
            o = self.collective(S.all_reduce, self.linear(S.o, attn, lw, "o"))
            if C.extra_norms:
                o, _ = self.rms(o, lw["ln2"])
                x, res = self.rms(o, lw["pre_ff"], res)
            else:
                x, res = self.rms(o, lw["ln2"], res)
            if C.E:
                d = self.collective(S.all_reduce, self.moe(x, lw))
            else:
                gu = self.linear(S.gate_up, x, lw, "gate_up")
                a = [self.call(S.silu, Coll(ps[0], gu[i].refs)) for i in range(M)]
                d = self.collective(S.all_reduce, self.linear(S.down, a, lw, "down"))
            if C.extra_norms:
                d, _ = self.rms(d, lw["post_ff"])
        return self.rms(d, self.norm, res)[0]

    def rms(self, xs: list[Coll], w: Coll, res: list | None = None) -> tuple[list[Coll], list | None]:
        """The RMSNorm over the rows `xs` with weight `w`, after the residual add when `res` is given -> (the normed rows, the new
        residual rows or None): the fused kernels (`RMSNormTriton` alone, `RMSNormFusedCuda` with the residual), or GemmaRMSNorm's
        `forward_native` as `rules.vllm_bindings.norm_chain` binds it: (1 + w) (once per program under the target's
        `weight_only_calls = once`), [the widened residual add and its narrowing,] x², the mean, + eps, rsqrt, the row scale, the weight
        multiply and the narrowing, each over every row."""
        S, M = self.S, len(xs)
        if not self.C.norm_chain:
            if res is None:
                p = _ptypes(S.norm0)
                return [self.call(S.norm0, Coll(p[0], x.refs), Coll(p[1], w.refs)) for x in xs], None
            p = _ptypes(S.fused)
            f = [self.call(S.fused, Coll(p[0], xs[i].refs), Coll(p[1], res[i]), Coll(p[2], w.refs)) for i in range(M)]
            return [f[i][0] for i in range(M)], [f[i][1].refs for i in range(M)]
        K = S.chain
        call = lambda fn, *a: self.call(fn, *(Coll(t, r) for t, r in zip(_ptypes(fn), a)))  # noqa: E731
        w1 = self.plus_one.get(id(w))
        if w1 is None:
            w1 = call(K["plus_one"], w.refs)
            if S.weight_once:
                self.plus_one[id(w)] = w1
        s, new_res = [x.refs for x in xs], None
        if res is not None:
            s = [call(K["add_widened_bf16"], s[i], res[i]).refs for i in range(M)]
            new_res = [call(K["narrow_f32_bf16"], s[i]).refs for i in range(M)]
        sq, sc = (K["square_bf16"], K["scale_row_bf16"]) if res is None else (K["square_f32"], K["scale_row_f32"])
        v = [call(sq, s[i]) for i in range(M)]
        m = [call(K["mean_triton"], v[i].refs) for i in range(M)]
        e = [call(K["eps"], m[i].refs) for i in range(M)]
        r = [call(K["rsqrt"], e[i].refs) for i in range(M)]
        y = [call(sc, s[i], r[i].refs) for i in range(M)]
        y = [call(K["mul_vec_f32"], y[i].refs, w1.refs) for i in range(M)]
        return [call(K["narrow_f32_bf16"], y[i].refs) for i in range(M)], new_res

    def moe(self, x: list[Coll], lw: dict) -> list[Coll]:
        """The fused-MoE block over the rows `x` as `rules.vllm_moe` binds its launches: each row's router logits and top-k, then
        each (token, slot) pair p = t * TOPK + k in pair order: the w13 expert GEMM (expert id ids[t][k]), the activation, the w2
        expert GEMM with the routed weight w[t][k]; then each row's sum of its TOPK down rows."""
        S, M, K = self.S, len(x), self.C.TOPK
        pr, pt, pu, pa, pw, pm = (_ptypes(fn) for fn in (S.router, S.topk, S.expert_up, S.silu, S.expert_down, S.moe_sum))
        lg = [self.call(S.router, Coll(pr[0], x[t].refs), Coll(pr[1], lw["gate_w"].refs)) for t in range(M)]
        r = [self.call(S.topk, Coll(pt[0], lg[t].refs)) for t in range(M)]
        ids = [r[p // K][1].refs.slice(p % K, p % K + 1) for p in range(M * K)]
        ws = [r[p // K][0].refs.slice(p % K, p % K + 1) for p in range(M * K)]
        up = [self.call(S.expert_up, Coll(pu[0], x[p // K].refs), Coll(pu[1], ids[p]), Coll(pu[2], lw["w13"].refs)) for p in range(M * K)]
        act = [self.call(S.silu, Coll(pa[0], up[p].refs)) for p in range(M * K)]
        down = [self.call(S.expert_down, Coll(pw[0], act[p].refs), Coll(pw[1], ids[p]), Coll(pw[2], ws[p]), Coll(pw[3], lw["w2"].refs))
                for p in range(M * K)]
        return [self.call(S.moe_sum, Coll(pm[0], concat([down[t * K + k].refs for k in range(K)]))) for t in range(M)]

    def head_norms(self, rows: list, heads: int, weight: Coll) -> list:
        """The per-head q/k RMSNorm over every row's heads, rows major (one launch over [rows, heads, D]) -> each row's normed heads."""
        fn, D = self.S.head_norm, self.C.D
        ph = _ptypes(fn)
        out = [self.call(fn, Coll(ph[0], r.slice(h * D, (h + 1) * D)), Coll(ph[1], weight.refs)) for r in rows for h in range(heads)]
        return [concat([out[i * heads + h].refs for h in range(heads)]) for i in range(len(rows))]

    def sample(self, hidden_row: Coll, event: int, pos: int, policy: _Stochastic | None = None) -> Coll:
        """The token of sampling event `event` from the logits of `hidden_row` (at position `pos`), gathered across the ranks and
        soft-capped where the family caps them (`common.select_token`)."""
        S = self.S
        pl = _ptypes(S.logits)
        lg = self.collective(S.all_gather, [self.call(S.logits, Coll(pl[0], hidden_row.refs), Coll(pl[1], self.lm_head.refs))])[0]
        for fn in S.cap if self.C.final_softcap is not None else ():
            lg = self.call(fn, Coll(_ptypes(fn)[0], lg.refs))
        return select_token(lambda fn, *refs: self.call(fn, *(Coll(t, r) for t, r in zip(_ptypes(fn), refs))), kind, lg.refs, self.C.V,
                            pos, policy)


def decoder(phi: Phi, M: int = 1, history: int = 0):
    """φ's decoder at launches of M rows over a history of `history` keys: the family's constructor (`FAMILY_FACTS[...]["arch"]`), a
    DeepSeek sparse-attention MoE (`dsa.Net`), a hyper-connected compressed-attention MoE (`dsv4.Net`) or the dense / fused-MoE
    Llama-family one (`_Decoder`).  Its `leaves()` are the weights record, `peer_shapes(rows)` one forward pass's peer Inputs in
    collective order, `model(B, W, peers)` the forward and the sampler."""
    family = str(phi.hf_config.get("model_type", "")).lower()
    entry = FAMILY_FACTS.get(family)
    arch = entry.get("arch") if entry else None    # a family the table lacks, or one with no constructor, is `model_shape`'s to refuse
    Net: type[Any]
    if arch == "dsa-moe":
        from .dsa import Net
    elif arch == "mhc-moe":
        from .dsv4 import Net
    else:
        Net = _Decoder
    check_phi(phi, Net.DOMAIN)
    if Net is not _Decoder and (phi.engine or {}).get(DR.ENGINE_ARG):
        raise Unexpressible(f"vocabulary cannot express: a launch that drafts {phi.engine[DR.ENGINE_ARG]} tokens a decode row on the "
                            f"{arch} decoder (draft rows are the dense decoder's)")
    return Net(phi, M, history)


def request_shape(wrapper: str, LP: int | None, T: int | None) -> tuple[int, int]:
    """A wrapper's (LP, T): (0, 0) for a `step`, which reads neither; for any other the prompt rows and the decode steps its
    Program is built for, a TypeError without them."""
    if wrapper == "step":
        return 0, 0
    if LP is None or T is None:
        raise TypeError(f"a {wrapper} wrapper's Program is built for a prompt of LP rows and T decode steps (LP={LP!r}, T={T!r})")
    return int(LP), int(T)


class _Decoder:
    """The dense or fused-MoE Llama-family decoder (`ModelShape`) under the bi-eager profile, on one rank or two."""

    DOMAIN = ((1, 2), "bi-eager", ("bfloat16", "int8"))

    def __init__(self, phi: Phi, M: int, history: int) -> None:
        C = model_shape(phi.hf_config)
        if (phi.model or {}).get("dtype", "int8" if C.int8 else "bfloat16") != ("int8" if C.int8 else "bfloat16"):
            raise Unexpressible(f"vocabulary cannot express: checkpoint dtype {phi.model['dtype']!r} for "
                                f"{'an exact-int8' if C.int8 else 'a bf16'} checkpoint (the row's dtype is its checkpoint's)")
        if C.window is not None and history > C.window:
            raise Unexpressible(f"vocabulary cannot express: a history of {history} keys under the sliding window of {C.window} keys")
        self.world, self.rank = int(phi.tp.get("world", 1)), int(phi.tp.get("rank", 0))
        self.C, self.specs = C, _Specs(C, phi.target_profile(), self.world, self.rank, M)
        self.model_class, self.NL, self.nodes_per_row_layer = C.model_class, C.NL, C.nodes_per_row_layer

    def leaves(self) -> list[tuple[str, Type]]:
        return weight_leaves(self.C, self.world)

    def peer_shapes(self, r: int) -> list[tuple[int, int]]:
        """`vllm_meta.WithPeers` over a pass of r rows: the embedding's all-reduce, each layer's q / k all-gathers (a norm over all heads)
        and o / down all-reduces, then the sampled row's logits all-gather."""
        R, C = self.specs.C, self.C
        layer = ([(r, R.NH * R.D), (r, R.KVH * R.D)] if R.qk_norm == "all" else []) + [(r, C.H)] * 2
        return [(r, C.H)] + layer * C.NL + [(1, C.V // self.world)]

    def prefix_params(self, C: int) -> list[tuple[str, Type]]:
        """The Input a request's history of its first C positions opens with (`vllm_meta.prefix_input`): every layer's key and value rows."""
        return [("prefix", _arr((self.C.NL, 2, C, self.C.KVH * self.C.D), 16))]

    def model(self, B: Builder, W: Coll, peers: _Peers) -> _Model:
        return _Model(B, self.specs, W, self.specs.C, peers)


def build_program(phi: Phi, wrapper: str, *, LP: int | None = None, T: int | None = None, M: int = 1, pos0: int = 8, with_desc: bool = True,
                  C: int = 0, dead_free: bool = False, batched: bool = False, table_reads: bool = False, finish: Callable | None = None,
                  batch: list | None = None, chunks: tuple[int, ...] | None = None, edge: tuple[str, ...] = (), stop: dict | None = None, entries: tuple[int, ...] = (),
                  drafts: tuple[tuple[int, int], ...] = (), draft_layer: Any = None) -> BuildResult:
    """The Program of the Build's `wrapper` for φ: `step` (one engine step, M token rows at pos0.., the last row sampled) or
    `request` (prefill of LP rows, then T decode steps each fed the previous greedy sample; returns the T+1 samples). `C` > 0: the
    request's first C positions are a history it reads and doesn't compute, every layer's rows of them Inputs after its own
    (`prefix_params`): a prefix-cache hit (`vllm_meta.CachedGreedyRequest`, the Build's `--cached C`) whose prefill computes rows
    C..LP-1, or on the sparse-attention decoder also a request a Commit window carries in at decode C >= LP, its first decode fed
    the `last_token` Input (`common.request_passes`); on that decoder either's SRC config holds `{"history": C}`, and no Build
    traces it yet. A request on one rank drops its dead units (`program.prune`, as the Build's derive does): the last layer's rows
    before the prefill's last, which no sample reads. A stochastic request takes its seed rule's `seed_rule.PARAMS` after its
    seed, at its policy's ``rid``, and checks the rule (`seed_rule.check`); `stop`: its length rule, the last check (`stop_rule`).
    `dead_free`: a tensor-parallel rank's request too (the Build's `--dead-free`), each dead collective's peer rows then unread.
    `batched`: that form batched (`prune.batched_form`, the Build's `--batched`): class G's request Programs. `table_reads`: in that
    form, DSA's rows past INDEX_TOPK keys by table reads (`Registered.dsa_table_reads`, the Build's `--dsa-table-reads`).
    `finish`: the body's last step in place of that drop, (body, decoder, parameters) -> (body, what it says beside the Program,
    `BuildResult.named`), as `naming.named_program` names a request Program. `batch`: the request's launch batch (`common._Launch`,
    the Build's `--launch-context '{"batch": ...}'`) in place of the one φ's workload gives it (`launch_batch`). `chunks`: a
    request's prefill as the engine ran it, in launches of these rows in turn (the chunked request wrapper), where its kernels read
    the launch: per chunk one forward at its own step of `batch` (its chunks, then its T decodes), the sample after the last chunk,
    a chunk that leaves the prompt incomplete sampling nothing (vLLM discards that sample); the SRC config is `{"chunks": [...]}`
    (its wrapper's class unchanged); one chunk is the request wrapper. `entries`: per chunk, its sequence's rows in that launch, of
    which it computes the first (`steps`' ``recompute_hits``), in the SRC config. `edge`: the history groups (`dsa.Net.edge_groups`)
    a request a Commit window leaves at position LP + T (inside its prompt too: T < 0, sampling nothing) returns after its samples
    and checks, every layer's rows [0, LP + T) of each as `prefix_params(LP + T)` lays them out (`common.with_edge`), in its SRC config.
    `drafts`: a request's draft schedule, each decode pass's (drafts, kept), under a launch that decodes speculatively, with
    `draft_layer` the registration's draft layer (`draft_rows.drive`): its T decode tokens come in those passes, each pass's
    kept count a check, in its SRC config."""
    from verity_vllm.program.frontend import seed_rule as SEED
    t0 = time.time()
    prompt_rows, decode_steps = request_shape(wrapper, LP, T)
    net = decoder(phi, M, pos0 + M if wrapper == "step" else prompt_rows + decode_steps)
    kind = (phi.sampling or {}).get("kind", "greedy")
    if (wrapper, kind) not in WRAPPERS:
        raise Unexpressible(f"wrapper {wrapper!r} (have {sorted({w for w, _ in WRAPPERS})})")
    world, rank, C = int(phi.tp.get("world", 1)), int(phi.tp.get("rank", 0)), int(C)
    if wrapper != "request" and (phi.engine or {}).get(DR.ENGINE_ARG):
        raise Unexpressible(f"vocabulary cannot express: a {wrapper} Program of a launch that drafts {phi.engine[DR.ENGINE_ARG]} tokens "
                            f"a decode row (a draft row is a pass of a request's Program)")
    if (table_reads and not batched) or batched and (finish is not None or wrapper != "request" or not (world == 1 or dead_free)):
        raise ValueError("batched is a request's registered compiler's form (dead_free at a tensor-parallel rank), without finish; "
                         "table_reads is the batched form's (DSA's long rows by table reads)")
    dense = type(net) is _Decoder
    if C and (wrapper != "request" or not hasattr(net, "prefix_params") or (dense and (kind, world) != ("greedy", 1))):
        raise Unexpressible(f"vocabulary cannot express: a cached prefix on a {kind} {wrapper} at TP{world} of the {net.model_class} decoder "
                            f"(a request's, greedy on one rank of the dense or fused-MoE decoder, or on the sparse-attention decoder)")
    resumed, chunks, passes, sampled = request_passes(wrapper, LP, T, M, C, chunks, dense, bool(edge))
    wrec = RecordType(*net.leaves())
    n = M if wrapper == "step" else prompt_rows
    listed: list[tuple[str, Type]] = [("input_ids" if wrapper == "step" and world == 1 else "prompt", Array(n, I32))]
    if kind == "stochastic":
        listed += [("seed", Array(1, I64)), *(SEED.params(SEED.rid_in(phi.sampling)) if wrapper == "request" else ())]
    history = len(listed)
    if C:
        listed += net.prefix_params(C)   # `vllm_meta.prefix_input`
    if resumed:
        listed.append(("last_token", Array(1, I32)))
    n_user = len(listed)
    if world > 1:
        # `vllm_meta.WithPeers`: world - 1 peer Inputs per collective of each forward pass, in call order; a pass that samples
        # nothing gathers no logits, the last of `peer_shapes`
        shapes = [s for r, smp in zip(passes, sampled) for s in net.peer_shapes(r)[:None if smp else -1] for _ in range(world - 1)]
        listed += [(f"peers_{k}", _arr(s, 16)) for k, s in enumerate(shapes)]
    n_peers = len(listed) - n_user
    params = check_params((*listed, ("weights", wrec)))
    B = Builder(params, Value(1))
    inputs = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    ids, W = inputs[0], inputs[-1]
    peers = _Peers(inputs[n_user:n_user + n_peers], world, rank)
    policy = None
    if kind == "stochastic":
        from verity_vllm.program.frontend.rules.vllm_sampling import f32_bits
        s = phi.sampling
        policy = _Stochastic(seed=Coll(I64, Affine("p", 1, 0, 1)), S=int(s["splits"]), temperature_bits=f32_bits(s["temperature"]),
                             top_p_bits=f32_bits(s.get("top_p", 1.0)))
    model = net.model(B, W, peers)
    if batched and hasattr(model, "batched"):
        model.batched, model.table_reads = True, table_reads
    batch = (launch_batch(phi, prompt_rows, decode_steps) if batch is None else batch) if wrapper == "request" else None
    in_launch(model, batch, net.model_class, chunks, decode_steps, entries=tuple(entries))
    if wrapper == "request" and isinstance(model, _Model):
        model.ctx = launch_rows(phi, prompt_rows, decode_steps, C)
    if C:
        model.open_with(inputs[history:n_user - resumed], C)
    rows = [Coll(I32, ids.refs.slice(i, i + 1)) for i in range(n)]
    if drafts:
        plain = (wrapper, kind, world) == ("request", "greedy", 1) and dense and not (C or chunks or edge or batch or batched or finish)
        toks = DR.drive(model, phi, rows, LP=prompt_rows, T=decode_steps, drafts=tuple(drafts), layer=draft_layer, plain=plain)
    else:
        toks = drive(model, wrapper, rows, M=M, pos0=pos0, LP=LP, T=T, policy=policy, batch=batch, C=C, chunks=chunks,
                     last=inputs[n_user - 1] if resumed else None)
    out = Coll(Array(1, I32), toks[0].refs) if wrapper == "step" else Coll(Array(len(toks), I32), concat([t.refs for t in toks]))
    if peers.n != len(peers.inputs):
        raise AssertionError(f"the collectives read {peers.n} of {len(peers.inputs)} peer Inputs")
    out, checks = with_checks(model, out, B, stop, inputs[1:2 + len(SEED.PARAMS)] if (wrapper, kind) == ("request", "stochastic") else None)
    out = with_edge(model, out, edge)
    B.ret_type = out.type
    body, names = B.finish(out), None
    if finish is not None:
        body, names = finish(body, net, params)
    elif wrapper == "request" and (world == 1 or dead_free):
        body = dead_free_form(body, batched=batched, consume=True)[0]
    t_body = time.time()
    src_class, derive_name = (CACHED_SRC, WRAPPERS[(wrapper, kind)][1]) if C and dense else WRAPPERS[(wrapper, kind)]
    if world > 1:
        src_class = f"verity-vllm/with-peers/v1[{src_class}]"
    name = f"{derive_name}[{net.model_class}]"
    src_static = {"class": src_class, "mode": "profile", "profile": phi.profile,
                  "inputs": [list(_arr_shape(t)) for _, t in params[:-1]],
                  "config_sha256": hashlib.sha256(canonical_json({**({"chunks": list(chunks)} if chunks else {}),
                                                                  **({"entries": list(entries)} if entries else {}), **edge_config(edge),
                                                                  **({"history": C} if C and not dense else {}), **({"stop": stop} if stop else {}),
                                                                  **DR.src_config(phi, tuple(drafts), draft_layer)})).hexdigest()[:16]}
    defn = CompositeDefinition(f"Derived[{name}]", 1, ("SRC",), lambda S, _p=params, _r=out.type: (tuple(_p), _r), replayer(body),
                               doc=f"predicted ahead of time by verity_vllm.program.predict.build ({BUILDER_ID})", conformance="derived: predicted, a switched row's, checked by its trace (predict.aot)",
                               register=False)
    sd = bind(defn, SRC=src_static)
    sd.set_body(body)
    program = Program(sd)
    timings = {"body_s": round(t_body - t0, 3), "program_s": round(time.time() - t_body, 3)}
    desc = digest = None
    if with_desc:
        t2 = time.time()
        desc = encode_program(program)
        digest = program_digest(desc)
        timings["encode_digest_s"] = round(time.time() - t2, 3)
    timings["total_s"] = round(time.time() - t0, 3)
    return BuildResult(wrapper=wrapper, program=program, digest=digest or "", desc=desc, function=sd.id, timings=timings, named=names,
                       n_root_nodes=len(program.fn.body.nodes) if hasattr(program.fn, "body") else -1, checks=checks)
