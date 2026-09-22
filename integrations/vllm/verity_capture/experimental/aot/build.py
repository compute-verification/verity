"""BuildProgram(φ, h) -> the Verity Program, constructed ahead of time from a *predicted* call sequence.

The same IR calls as the capture fold (`Builder.call(bind(b1.X, ...), Coll(ptype, refs))`, the `Captured_v1{PROFILE, WORKLOAD}`
binding, `_replayer`, `Program`), driven by `predict.predict_schedule(φ, h)` instead of the dispatcher trace, with the
dataflow tracked logically (which node produced each row's hidden state / K / V / sampled token) instead of through the
memory model.  See `out/gen/r8/aot-builder/how-fold-works.md` §4–5 for the per-step instance sequence this reproduces.

Vocabulary (v1, dense decoder-only families under the eager profile): Embedding, RMSNormTriton (layer-0 input norm),
Gemm, RoPE, Attention_v3 (FA2, grouped by distinct T), RMSNormFusedCuda (post-attention / next-layer / final norm),
SiluMul, TokenSelect (greedy).  Anything φ asks for outside this vocabulary raises `Unexpressible` (a result, never a
silent approximation).  Family switches (LayerNorm, MoE, QK-norm, extra norms, untied lm_head, QKV bias) are detected from
C and reported as unexpressible in v1.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import Any

from verity.ir.codec import encode_program, program_digest
from verity.ir.defs import Builder, CompositeDefinition, SpecializedDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import Affine, Coll, Refs, concat
from verity_vllm.registry import b1
from verity.ir.types import Array, Value
from verity.ir.types import Tuple as TupleT

from ... import analytic
from ...fold import _replayer, config_for_weights
from .hints import Hints
from .phi import Phi
from .predict import PREDICTOR_ID, StepPlan, predict_schedule

I32 = Value(32)
BUILDER_ID = "aot-build/eager/v8"      # v2: + BiasAdd (BIAS), untied lm_head, sampler launch order; v3: + seeded Gumbel sampler rows; v4: + QK-norm, fused-MoE (OLMoE); v5: + GPT-NeoX LayerNorm family (Pythia); v6: content-addressed K/V (prefix caching); v7: pool-model hit producers; v8: n > 1 forks


def max_rss_gb() -> float:
    """Peak RSS of this process in GiB (`ru_maxrss` is bytes on macOS, KiB on Linux)."""
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2), 3)


class Unexpressible(Exception):
    """φ (or a hint) asks for a structure the v1 vocabulary cannot express.  The message names the fact precisely."""


@dataclass
class BuildResult:
    program: Any
    spec: Any
    body: Any
    digest: str
    desc: dict | None
    plans: list[StepPlan]
    params: tuple
    ret_type: Any
    n_instances: int
    timings: dict = field(default_factory=dict)
    max_rss_gb: float = 0.0
    instances: list[dict] = field(default_factory=list)      # optional per-instance meta (step, spec, row) when `with_meta`


# ---------------------------------------------------------------------------------------------------------
# vocabulary checks
# ---------------------------------------------------------------------------------------------------------


def check_vocabulary(phi: Phi) -> None:
    C = phi.config
    facts = []
    if C.get("NUM_EXPERTS"):
        from verity_capture.profiles.gen_ov_moe_patterns import ROUTER_VPT_BY_E
        if not C.get("TOPK"):
            facts.append("NUM_EXPERTS without TOPK")
        if C["NUM_EXPERTS"] not in ROUTER_VPT_BY_E:
            facts.append(f"NUM_EXPERTS={C['NUM_EXPERTS']} has no pinned router geometry (VPT)")
        if C.get("NORM_TOPK_PROB"):
            facts.append("NORM_TOPK_PROB (renormalised router) is not the modelled MoeRouterTopK_v1")
    if C.get("extra_layer_norms"):
        facts.append(f"extra_layer_norms={C['extra_layer_norms']} (Gemma-2 pre/post feed-forward norms)")
    if C.get("embed_normalizer") is not None:
        facts.append("embed_normalizer (Gemma-2 embedding scale)")
    if C.get("ROT") and C["ROT"] != C["D"] and C.get("NORM") != "layernorm":
        facts.append(f"ROT={C['ROT']} != D={C['D']} (partial rotary outside the GPT-NeoX family)")
    s = phi.workload.get("sampling") or {}
    if s.get("top_k") not in (None, 0, -1) or s.get("top_p") not in (None, 1, 1.0):
        facts.append(f"sampling filters top_k/top_p {s.get('top_k')!r}/{s.get('top_p')!r} (no Definition for filtered sampling)")
    if s.get("logprobs"):
        facts.append("logprobs requested")
    T0 = float(s.get("temperature", 0.0))
    if T0 != 0.0:
        for i, r in enumerate(phi.logical_requests()):
            if r.get("seed") is None:
                facts.append(f"request {i}: temperature {T0} sampled without a recorded seed (host-drawn RNG state is not a φ fact)")
    # n > 1 is expressible since v8: the fork is a φ-derived rule (phi.logical_requests); children share the parent's prompt parameter.
    if facts:
        raise Unexpressible("v1 vocabulary cannot express: " + "; ".join(facts))


# ---------------------------------------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------------------------------------


class _Specs:
    """The bound specializations, cached (one object per id, as the fold's `bind` cache gives)."""

    def __init__(self, C: dict) -> None:
        self.C = C
        H, V, NH, KVH, D, I, EPS, QKV, GU, BN = (C["H"], C["V"], C["NH"], C["KVH"], C["D"], C["I"], C["EPS"], C["QKV"], C["GU"], C["BN"])
        self.embedding = bind(b1.Embedding, V=V, H=H)
        self.norm0 = bind(b1.RMSNormTriton, N=H, EPS=EPS)
        self.fused = bind(b1.RMSNormFusedCuda, N=H, EPS=EPS)
        self.qkv = bind(b1.Gemm, K=H, N=QKV)
        self.o = bind(b1.Gemm, K=NH * D, N=H)
        self.gate_up = bind(b1.Gemm, K=H, N=GU)
        self.down = bind(b1.Gemm, K=I, N=H)
        self.logits = bind(b1.Gemm, K=H, N=V)
        self.rope_q = bind(b1.RoPE, NHEADS=NH, D=D)
        self.rope_k = bind(b1.RoPE, NHEADS=KVH, D=D)
        self.silu = bind(b1.SiluMul, I=I)
        self.token_select = bind(b1.TokenSelect, V=V)
        self.bias_add = bind(b1.BiasAdd, N=QKV) if C.get("BIAS") else None      # Qwen2: one aten.add launch after the QKV GEMM
        self.V = V
        # OLMoE: per-head-group q/k RMSNorm (same Triton kernel as the layer-0 norm) and the pair-major fused-MoE block
        self.q_norm = bind(b1.RMSNormTriton, N=NH * D, EPS=EPS) if C.get("QK_NORM") else None
        self.k_norm = bind(b1.RMSNormTriton, N=KVH * D, EPS=EPS) if C.get("QK_NORM") else None
        self.ln = None
        if C.get("NORM") == "layernorm":
            # GPT-NeoX / Pythia (gen_ln_patterns): LayerNorm-with-bias, every dense bias a separate BiasAdd, erf-GELU, parallel
            # residual (two ResidualAdds), partial rotary on the first ROT dims of every head, untied embed_out
            from verity_vllm.registry.quarantine.ln.gelu_erf import GeluErf
            from verity_vllm.registry.quarantine.ln.layer_norm_aten import LayerNormAten
            ROT = C.get("ROT") or D
            self.rope_q = bind(b1.RoPE, NHEADS=NH, D=ROT)
            self.rope_k = bind(b1.RoPE, NHEADS=KVH, D=ROT)
            self.ln = {"norm": bind(LayerNormAten, N=H, EPS=EPS), "qkv_b": bind(b1.BiasAdd, N=QKV), "o_b": bind(b1.BiasAdd, N=H),
                       "up": bind(b1.Gemm, K=H, N=I), "up_b": bind(b1.BiasAdd, N=I), "gelu": bind(GeluErf, N=I),
                       "down_b": bind(b1.BiasAdd, N=H), "add": bind(b1.ResidualAdd, N=H), "ROT": ROT}
        self.moe = None
        if C.get("NUM_EXPERTS"):
            from verity_capture.profiles.gen_ov_moe_patterns import ROUTER_VPT_BY_E
            from verity_vllm.registry.quarantine.ov_moe import moe as Q
            E, TOPK = C["NUM_EXPERTS"], C["TOPK"]
            self.moe = {"gate": bind(b1.Gemm, K=H, N=E), "router": bind(Q.MoeRouterTopK, E=E, TOPK=TOPK, VPT=ROUTER_VPT_BY_E[E]),
                        "up": bind(Q.MoeExpertGemm, E=E, K=H, N=2 * I), "down": bind(Q.MoeExpertGemmW, E=E, K=I, N=H),
                        "sum": bind(Q.MoeSum, TOPK=TOPK, H=H), "TOPK": TOPK}
        self._sampler: dict[tuple, SpecializedDefinition] = {}
        self._attn: dict[int, SpecializedDefinition] = {}
        self.NH, self.KVH, self.D, self.BN, self.H = NH, KVH, D, BN, H

    def sampler(self, T: float, seed: int | None, pos: int) -> SpecializedDefinition:
        """The sampler Definition of one sampled row: greedy -> TokenSelect_v1{V}; seeded temperature-1 sampling -> the quarantine
        GumbelArgmaxBf16_v1{V, SEED, POS} (gen_ov_sampling_patterns); other temperatures -> GumbelArgmaxTempBf16_v1{..., T}."""
        key = (T, seed, pos)
        sp = self._sampler.get(key)
        if sp is None:
            if T == 0.0:
                sp = self.token_select
            else:
                from verity_vllm.registry.quarantine.ov_sampling.gumbel import GumbelArgmaxBf16, GumbelArgmaxTempBf16
                if T == 1.0:
                    sp = bind(GumbelArgmaxBf16, V=self.V, SEED=int(seed), POS=int(pos))
                else:
                    import numpy as np
                    sp = bind(GumbelArgmaxTempBf16, V=self.V, SEED=int(seed), POS=int(pos), T=int(np.float32(T).view(np.uint32)))
            self._sampler[key] = sp
        return sp

    def attention(self, T: int) -> SpecializedDefinition:
        s = self._attn.get(T)
        if s is None:
            s = self._attn[T] = bind(b1.AttentionV3, T=T, NH=self.NH, KVH=self.KVH, D=self.D, BN=self.BN)
        return s


def _ptypes(spec) -> tuple:
    params, _ = spec.signature()
    return tuple(t for _, t in params)


def build_program(phi: Phi, hints: Hints | None = None, *, with_program: bool = True, with_desc: bool = True,
                  with_meta: bool = False, steps_limit: int | None = None) -> BuildResult:
    """The Program of (φ, h).  `steps_limit` builds only the first n engine steps (development: prefill-only comparisons);
    the resulting body is then a prefix of the full one and its return covers the served tokens so far."""
    t0 = time.time()
    hints = hints or Hints(predictor=PREDICTOR_ID)
    check_vocabulary(phi)
    C = config_for_weights(dict(phi.config))
    if "BN" not in C:
        C["BN"] = b1.fa2_kblock_n(C["D"])
    specs = _Specs(C)
    weights_type = analytic.weights_type(C)
    reqs = phi.logical_requests()          # engine requests: n > 1 forks expanded (children in index order, right after each other)
    R = len(reqs)
    hreqs = phi.requests                   # header requests: one prompt parameter each (a child's prompt IS its parent's, as in the fold)
    RP = len(hreqs)
    params: list[tuple[str, Any]] = []
    for i, r in enumerate(hreqs):
        params.append(("prompt" if RP == 1 else f"prompt_{i}", Array(len(r["prompt_token_ids"]), I32)))
    params.append(("weights", weights_type))
    params_t = tuple(params)
    B = Builder(params_t, I32)
    parent_colls = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params_t[:-1])]
    prompt_colls = [parent_colls[r["_parent"]] for r in reqs]
    W = Coll(weights_type, Affine("p", RP, 0, weights_type.leaves))
    embed = W["embed"]
    lm_head = embed if C.get("tie_word_embeddings", True) else W["lm_head"]
    norm_w = W["norm"]
    cos_sin = W["cos_sin"]
    NL = C["NL"]
    layers = [W["layers"][L] for L in range(NL)]
    p_embed = _ptypes(specs.embedding)
    p_norm0 = _ptypes(specs.norm0)
    p_fused = _ptypes(specs.fused)
    p_qkv, p_o, p_gu, p_down, p_logits = (_ptypes(s) for s in (specs.qkv, specs.o, specs.gate_up, specs.down, specs.logits))
    p_rq, p_rk, p_silu, p_ts = (_ptypes(s) for s in (specs.rope_q, specs.rope_k, specs.silu, specs.token_select))
    p_bias = _ptypes(specs.bias_add) if specs.bias_add is not None else None
    p_qn = _ptypes(specs.q_norm) if specs.q_norm is not None else None
    p_kn = _ptypes(specs.k_norm) if specs.k_norm is not None else None
    p_moe = {k: _ptypes(v) for k, v in specs.moe.items() if k != "TOPK"} if specs.moe is not None else None
    p_ln = {k: _ptypes(v) for k, v in specs.ln.items() if k != "ROT"} if specs.ln is not None else None
    smp = phi.workload.get("sampling") or {}
    T0 = float(smp.get("temperature", 0.0))
    temps = {i: float(pr["temperature"]) for pr in (smp.get("per_request") or []) for i, r in enumerate(reqs)
             if pr["request_id"] in (r["request_id"], hreqs[r["_parent"]]["request_id"])}
    seeds = {i: (None if r.get("seed") is None else int(r["seed"])) for i, r in enumerate(reqs)}
    H, D, NH, KVH = C["H"], C["D"], C["NH"], C["KVH"]
    q_w, k_w, v_w = NH * D, KVH * D, KVH * D

    plans = predict_schedule(phi, hints)
    if steps_limit is not None:
        plans = plans[:steps_limit]

    # logical state ---------------------------------------------------------------------------------------------
    kv: list[dict[tuple[int, int], tuple[Refs, Refs]]] = [dict() for _ in range(NL)]   # layer -> (req, pos) -> (k refs, v refs)
    # prefix caching (prefix_cache.py semantics): a request that hits cached blocks reads K/V another request's rows produced.
    # The producer is identified by prompt *content*: kv_content[L][prefix key of prompt[0..pos]] = the FIRST rows that computed
    # that prefix position (vLLM's cache map keeps the first cached block for a hash).  Only prompt positions are content-addressable
    # from φ: a block that contains generated tokens has a hash the builder cannot know (its tokens are circuit values).
    prefix_key: list[list[int]] = []
    for r in reqs:
        hk, keys = 0, []
        for tok in r["prompt_token_ids"]:
            hk = hash((hk, int(tok)))
            keys.append(hk)
        prefix_key.append(keys)
    kv_content: list[dict[int, tuple[Refs, Refs]]] = [dict() for _ in range(NL)]

    hit_src: dict[tuple[int, int], int] = {}     # (req, pos) -> producer request, from the predictor's pool model (v4-pool plan.hits)

    def kv_part(L: int, r: int, p: int) -> tuple[Refs, Refs]:
        part = kv[L].get((r, p))
        if part is None and (r, p) in hit_src:
            part = kv[L].get((hit_src[(r, p)], p))      # v4-pool: the block's writer at hit time (eviction-aware)
            if part is not None:
                kv[L][(r, p)] = part
        if part is None and p < len(prefix_key[r]):
            part = kv_content[L].get(prefix_key[r][p])
            if part is not None:
                kv[L][(r, p)] = part                       # a prefix-cache hit: the producer's K/V, linked once
        if part is None:
            raise Unexpressible(f"layer {L}: request {r} position {p} has no K/V in the run (prefix-cache state from outside the observed run, or a lost chunk)")
        return part

    def kv_put(L: int, r: int, pos: int, part: tuple[Refs, Refs]) -> None:
        kv[L][(r, pos)] = part
        if pos < len(prefix_key[r]):
            kv_content[L].setdefault(prefix_key[r][pos], part)
    sample_at: dict[tuple[int, int], Coll] = {}                                        # (req, token position) -> the TokenSelect output (I32) that produced that token
    served: dict[int, list[Coll]] = {i: [] for i in range(R)}
    meta: list[dict] = []
    n_inst = 0

    def call(spec, colls, step: int, row: int, label: str) -> Coll:
        nonlocal n_inst
        out = B.call(spec, *colls, name=None)
        n_inst += 1
        if with_meta:
            meta.append({"node": len(B.nodes) - 1, "step": step, "spec": spec.id, "row": row, "label": label})
        return out

    for plan in plans:
        s = plan.step
        rows = plan.rows
        M = len(rows)
        for hr, hfirst, hcount, hprod in plan.hits:
            for hp in range(hfirst, hfirst + hcount):
                hit_src[(hr, hp)] = hprod
        # token refs per row: prompt leaf or the previous sample of the request
        toks: list[Coll] = []
        for i, (r, pos) in enumerate(rows):
            LP = len(reqs[r]["prompt_token_ids"])
            if pos < LP:
                toks.append(Coll(I32, prompt_colls[r][pos].refs))
            else:
                # the token at position pos was sampled from the logits row at position pos - 1 (also right for a preemption
                # recompute, which re-feeds every generated token at its own position)
                prev = sample_at.get((r, pos))
                if prev is None:
                    raise Unexpressible(f"step {s} row {i}: request {r} position {pos} is past its prompt ({LP}) and no sample produced that token")
                toks.append(prev)
        # embedding
        h = [call(specs.embedding, (toks[i], Coll(p_embed[1], embed.refs)), s, i, "embed") for i in range(M)]
        def attention_rows(L: int, q: list, kv_of_row) -> list:
            """The attention instances of the step's rows for layer L: rows grouped by distinct T in first-occurrence order
            (patterns.AttentionScope); K/V = the concatenation of the request's per-position parts from kv[L]."""
            groups: dict[int, list[int]] = {}
            for i, (r, pos) in enumerate(rows):
                groups.setdefault(pos + 1, []).append(i)
            attn: list[Coll | None] = [None] * M
            for T, idxs in groups.items():
                spec = specs.attention(T)
                pt = _ptypes(spec)
                for i in idxs:
                    r = rows[i][0]
                    try:
                        parts = [kv_part(L, r, p) for p in range(T)]
                    except Unexpressible as e:
                        raise Unexpressible(f"step {s} {e}") from None
                    kc = Coll(pt[1], concat([kp for kp, _ in parts]))
                    vc = Coll(pt[2], concat([vp for _, vp in parts]))
                    attn[i] = call(spec, (Coll(pt[0], q[i]), kc, vc), s, i, f"L{L}.attn")
            return attn

        if specs.ln is not None:
            # GPT-NeoX layer (vLLM GPTNeoXLayer, use_parallel_residual): x -> ln1 -> qkv -> +b -> rope(q[:ROT], k[:ROT]) -> attn -> dense -> +b;
            # x -> ln2 -> h_to_4h -> +b -> gelu -> 4h_to_h -> +b; x' = (mlp + attn) + x.  Final norm -> embed_out -> sampler.
            ln, pl, ROT, D = specs.ln, p_ln, specs.ln["ROT"], C["D"]
            NH, KVH = C["NH"], C["KVH"]
            xs = [h[i].refs for i in range(M)]                    # the residual stream
            for L in range(NL):
                lw = layers[L]
                n1 = [call(ln["norm"], (Coll(pl["norm"][0], xs[i]), Coll(pl["norm"][1], lw["ln1"].refs), Coll(pl["norm"][2], lw["ln1_b"].refs)), s, i, f"L{L}.ln1") for i in range(M)]
                g = [call(specs.qkv, (Coll(p_qkv[0], n1[i].refs), Coll(p_qkv[1], lw["qkv_w"].refs)), s, i, f"L{L}.qkv") for i in range(M)]
                qkv = [call(ln["qkv_b"], (Coll(pl["qkv_b"][0], g[i].refs), Coll(pl["qkv_b"][1], lw["qkv_b"].refs)), s, i, f"L{L}.qkv_b") for i in range(M)]
                cs = [Coll(p_rq[1], cos_sin[pos].refs) for _, pos in rows]
                # the kernel rotates dims [0, ROT) of every head; dims [ROT, D) keep the BiasAdd's words
                q_rot = [concat([qkv[i].refs.slice(hh * D, hh * D + ROT) for hh in range(NH)]) for i in range(M)]
                k_rot = [concat([qkv[i].refs.slice(q_w + hh * D, q_w + hh * D + ROT) for hh in range(KVH)]) for i in range(M)]
                rq = [call(specs.rope_q, (Coll(p_rq[0], q_rot[i]), cs[i]), s, i, f"L{L}.rope.q") for i in range(M)]
                rk = [call(specs.rope_k, (Coll(p_rk[0], k_rot[i]), cs[i]), s, i, f"L{L}.rope.k") for i in range(M)]
                q_full = [concat([part for hh in range(NH) for part in (rq[i].refs.slice(hh * ROT, (hh + 1) * ROT), qkv[i].refs.slice(hh * D + ROT, (hh + 1) * D))]) for i in range(M)]
                k_full = [concat([part for hh in range(KVH) for part in (rk[i].refs.slice(hh * ROT, (hh + 1) * ROT), qkv[i].refs.slice(q_w + hh * D + ROT, q_w + (hh + 1) * D))]) for i in range(M)]
                for i, (r, pos) in enumerate(rows):
                    kv_put(L, r, pos, (k_full[i], qkv[i].refs.slice(q_w + k_w, q_w + k_w + v_w)))
                attn = attention_rows(L, q_full, None)
                o = [call(specs.o, (Coll(p_o[0], attn[i].refs), Coll(p_o[1], lw["o_w"].refs)), s, i, f"L{L}.o") for i in range(M)]
                ob = [call(ln["o_b"], (Coll(pl["o_b"][0], o[i].refs), Coll(pl["o_b"][1], lw["o_b"].refs)), s, i, f"L{L}.o_b") for i in range(M)]
                n2 = [call(ln["norm"], (Coll(pl["norm"][0], xs[i]), Coll(pl["norm"][1], lw["ln2"].refs), Coll(pl["norm"][2], lw["ln2_b"].refs)), s, i, f"L{L}.ln2") for i in range(M)]
                u = [call(ln["up"], (Coll(pl["up"][0], n2[i].refs), Coll(pl["up"][1], lw["up_w"].refs)), s, i, f"L{L}.up") for i in range(M)]
                ub = [call(ln["up_b"], (Coll(pl["up_b"][0], u[i].refs), Coll(pl["up_b"][1], lw["up_b"].refs)), s, i, f"L{L}.up_b") for i in range(M)]
                ge = [call(ln["gelu"], (Coll(pl["gelu"][0], ub[i].refs),), s, i, f"L{L}.gelu") for i in range(M)]
                dn = [call(specs.down, (Coll(p_down[0], ge[i].refs), Coll(p_down[1], lw["down_w"].refs)), s, i, f"L{L}.down") for i in range(M)]
                db = [call(ln["down_b"], (Coll(pl["down_b"][0], dn[i].refs), Coll(pl["down_b"][1], lw["down_b"].refs)), s, i, f"L{L}.down_b") for i in range(M)]
                a1 = [call(ln["add"], (Coll(pl["add"][0], db[i].refs), Coll(pl["add"][1], ob[i].refs)), s, i, f"L{L}.res1") for i in range(M)]
                a2 = [call(ln["add"], (Coll(pl["add"][0], a1[i].refs), Coll(pl["add"][1], xs[i])), s, i, f"L{L}.res2") for i in range(M)]
                xs = [a2[i].refs for i in range(M)]
            fn = [call(ln["norm"], (Coll(pl["norm"][0], xs[i]), Coll(pl["norm"][1], norm_w.refs), Coll(pl["norm"][2], W["norm_b"].refs)), s, i, "final_ln") for i in range(M)]
            x = [Coll(p_logits[0], fn[i].refs) for i in range(M)]
            NL_dense = 0
        else:
            NL_dense = NL
            # layer-0 input norm (Triton RMSNorm), residual = embedding
            x = [call(specs.norm0, (Coll(p_norm0[0], h[i].refs), Coll(p_norm0[1], layers[0]["ln1"].refs)), s, i, "norm0") for i in range(M)]
            res = [Coll(p_fused[1], h[i].refs) for i in range(M)]
        for L in range(NL_dense):
            lw = layers[L]
            qkv = [call(specs.qkv, (Coll(p_qkv[0], x[i].refs), Coll(p_qkv[1], lw["qkv_w"].refs)), s, i, f"L{L}.qkv") for i in range(M)]
            if specs.bias_add is not None:
                qkv = [call(specs.bias_add, (Coll(p_bias[0], qkv[i].refs), Coll(p_bias[1], lw["qkv_b"].refs)), s, i, f"L{L}.qkv_b") for i in range(M)]
            cs = [Coll(p_rq[1], cos_sin[pos].refs) for _, pos in rows]
            q_in = [qkv[i].refs.slice(0, q_w) for i in range(M)]
            k_in = [qkv[i].refs.slice(q_w, q_w + k_w) for i in range(M)]
            if specs.q_norm is not None:       # OlmoeAttention: q_norm(q), k_norm(k) on contiguous copies, then RoPE
                q_in = [call(specs.q_norm, (Coll(p_qn[0], q_in[i]), Coll(p_qn[1], lw["q_norm"].refs)), s, i, f"L{L}.q_norm").refs for i in range(M)]
                k_in = [call(specs.k_norm, (Coll(p_kn[0], k_in[i]), Coll(p_kn[1], lw["k_norm"].refs)), s, i, f"L{L}.k_norm").refs for i in range(M)]
            q = [call(specs.rope_q, (Coll(p_rq[0], q_in[i]), cs[i]), s, i, f"L{L}.rope.q") for i in range(M)]
            k = [call(specs.rope_k, (Coll(p_rk[0], k_in[i]), cs[i]), s, i, f"L{L}.rope.k") for i in range(M)]
            for i, (r, pos) in enumerate(rows):
                kv_put(L, r, pos, (k[i].refs, qkv[i].refs.slice(q_w + k_w, q_w + k_w + v_w)))
            attn = attention_rows(L, [q[i].refs for i in range(M)], None)
            o = [call(specs.o, (Coll(p_o[0], attn[i].refs), Coll(p_o[1], lw["o_w"].refs)), s, i, f"L{L}.o") for i in range(M)]
            f1 = [call(specs.fused, (Coll(p_fused[0], o[i].refs), res[i], Coll(p_fused[2], lw["ln2"].refs)), s, i, f"L{L}.norm2") for i in range(M)]
            x2 = [Coll(p_gu[0], f1[i][0].refs) for i in range(M)]
            res = [Coll(p_fused[1], f1[i][1].refs) for i in range(M)]
            if specs.moe is None:
                gu = [call(specs.gate_up, (x2[i], Coll(p_gu[1], lw["gate_up_w"].refs)), s, i, f"L{L}.gate_up") for i in range(M)]
                a = [call(specs.silu, (Coll(p_silu[0], gu[i].refs),), s, i, f"L{L}.silu") for i in range(M)]
                d = [call(specs.down, (Coll(p_down[0], a[i].refs), Coll(p_down[1], lw["down_w"].refs)), s, i, f"L{L}.down") for i in range(M)]
            else:
                # gen_ov_moe_patterns.MoEFusedExperts, fold order: router logits GEMM (M) -> topk_softmax (M) -> w13 fused GEMM over
                # the (token, slot) pairs p = t*TOPK + k (M*TOPK; A row = hidden row t, expert id = word p of the router ids,
                # the whole expert table Shared) -> silu_and_mul (M*TOPK) -> weighted w2 GEMM (M*TOPK) -> moe_sum (M).
                # The routing (which expert) is the router's *output value*, an ordinary operand: it never enters advice.
                mo, pm, TOPK = specs.moe, p_moe, specs.moe["TOPK"]
                lg = [call(mo["gate"], (Coll(pm["gate"][0], x2[i].refs), Coll(pm["gate"][1], lw["gate_w"].refs)), s, i, f"L{L}.moe.gate") for i in range(M)]
                rt = [call(mo["router"], (Coll(pm["router"][0], lg[i].refs),), s, i, f"L{L}.moe.router") for i in range(M)]
                w13, w2 = Coll(pm["up"][2], lw["w13"].refs), Coll(pm["down"][3], lw["w2"].refs)
                up = [call(mo["up"], (Coll(pm["up"][0], x2[t].refs), Coll(pm["up"][1], rt[t][1].refs.slice(kk, kk + 1)), w13), s, t, f"L{L}.moe.w13")
                      for t in range(M) for kk in range(TOPK)]
                a = [call(specs.silu, (Coll(p_silu[0], up[pp].refs),), s, pp // TOPK, f"L{L}.moe.silu") for pp in range(M * TOPK)]
                dn = [call(mo["down"], (Coll(pm["down"][0], a[t * TOPK + kk].refs), Coll(pm["down"][1], rt[t][1].refs.slice(kk, kk + 1)),
                                        Coll(pm["down"][2], rt[t][0].refs.slice(kk, kk + 1)), w2), s, t, f"L{L}.moe.w2")
                      for t in range(M) for kk in range(TOPK)]
                d = [call(mo["sum"], (Coll(pm["sum"][0], concat([dn[t * TOPK + kk].refs for kk in range(TOPK)])),), s, t, f"L{L}.moe.sum") for t in range(M)]
            nw = layers[L + 1]["ln1"] if L + 1 < NL else norm_w
            f2 = [call(specs.fused, (Coll(p_fused[0], d[i].refs), res[i], Coll(p_fused[2], nw.refs)), s, i, f"L{L}.norm_next") for i in range(M)]
            x = [Coll(p_norm0[0], f2[i][0].refs) for i in range(M)]
            res = [Coll(p_fused[1], f2[i][1].refs) for i in range(M)]
        # logits rows -> one logits GEMM launch over all sampled rows (M' instances), then one sampler launch (M' TokenSelects):
        # two kernels, so the fold emits all GEMM instances before the first TokenSelect (visible only when M' > 1).
        row_of = {rp: i for i, rp in enumerate(rows)}
        lrows: list[int] = []
        for j, (r, pos) in enumerate(plan.logits_rows):
            i = row_of.get((r, pos))
            if i is None:
                raise Unexpressible(f"step {s}: logits row {j} names (request {r}, position {pos}) which is not a token row of the step")
            lrows.append(i)
        lgs = [call(specs.logits, (Coll(p_logits[0], x[i].refs), Coll(p_logits[1], lm_head.refs)), s, i, "logits") for i in lrows]
        for j, ((r, pos), i, lg) in enumerate(zip(plan.logits_rows, lrows, lgs)):
            sp = specs.sampler(temps.get(r, T0), seeds[r], pos)
            tok = call(sp, (Coll(p_ts[0], lg.refs),), s, i, "token_select")
            sample_at[(r, pos + 1)] = tok
            if plan.served[j]:
                served[r].append(tok)

    # return: served tokens per request, header order (one Array for one request, else a Tuple)
    outs = [Coll(Array(len(served[i]), I32), concat([c.refs for c in served[i]])) for i in range(R)]
    if R == 1:
        out = outs[0]
    else:
        from verity.ir.refs import tuple_of
        out = tuple_of(*outs)
    B.ret_type = out.type
    body = B.finish(out)
    ret_type = out.type
    t_body = time.time()
    defn = CompositeDefinition("Captured", 1, ("PROFILE", "WORKLOAD"), lambda S, _p=params_t, _r=ret_type: (_p, _r), _replayer(body),
                               doc="built ahead of time by verity_capture.experimental.aot.build (P = G(phi, a))",
                               conformance="derived: aot build", register=False)
    spec = bind(defn, PROFILE=phi.profile_id, WORKLOAD=phi.workload_digest())
    spec.set_body(body)
    program = desc = None
    digest = ""
    timings = {"body_s": round(t_body - t0, 3)}
    if with_program:
        t1 = time.time()
        program = Program(spec)
        timings["program_s"] = round(time.time() - t1, 3)
        if with_desc:
            t2 = time.time()
            desc = encode_program(program)
            digest = program_digest(desc)
            timings["encode_digest_s"] = round(time.time() - t2, 3)
    timings["total_s"] = round(time.time() - t0, 3)
    rss_gb = max_rss_gb()
    return BuildResult(program=program, spec=spec, body=body, digest=digest, desc=desc, plans=plans, params=params_t, ret_type=ret_type,
                       n_instances=n_inst, timings=timings, max_rss_gb=round(rss_gb, 3), instances=meta)
