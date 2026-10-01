"""BuildProgram(φ) -> the Program the Build derives for a row, constructed ahead of time from a hand-written vocabulary.

The Build (`pipeline.build`) traces vLLM's model class on meta with torch.export and lets the binding rules translate each node;
this module writes the same Program down directly from φ (`phi.Phi`: the checkpoint's `config.json`, the declared target, the
wrapper shape and the sampling policy), with the same IR calls the derive makes: one `Builder` over the wrapper's user inputs and
the weights record (vLLM's parameters in module order, then its buffers), one call per row of each launch, the vocabulary's kinds
(`rules.vocab.PROFILE_B1_EAGER_V3.kind`) with the statics the rules select through `registry.targets`, the flat body wrapped as
`Derived[<wrapper>[<model class>]]_v1{SRC}` and inlined by `Program`.  Nothing here traces, loads vLLM or reads a Build output.

Vocabulary (v1, dense Llama-family decoders under the bi-eager profile, greedy): Embedding, RMSNormTriton (layer-0 input norm),
Gemm (target-keyed), RoPE (all q rows, then all k rows), Attention (target-keyed; one call per query row over the layer's
history), RMSNormFusedCuda (post-attention / next-layer / final norm), SiluMul, TokenSelect.  Anything φ asks for outside it raises
`Unexpressible` naming the fact (a result, never a silent approximation).

Restored lineage: `verity_capture.experimental.aot.build` (`aot-build/eager/v8`, deleted in 528f9d21f) built the capture fold's
`Captured_v1{PROFILE, WORKLOAD}` Program; the Build of record now derives `Derived[vLLM*]` Programs, so the emitter targets those.

This module is NEVER on a record path: a predicted Program is not written into a Build row, not published as a Build Attempt and
not read by the Build, Match or Commit (`program.frontend.inputs_trace` lists `verity_vllm/predict/` as a forbidden input).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

from verity.ir.codec import canonical_json, encode_program, program_digest
from verity.ir.defs import Builder, CompositeDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import Affine, Coll, concat
from verity.ir.types import Array, Record, Type, Value

from .phi import Phi

BUILDER_ID = "predict-build/derived/v1"
I32 = Value(32)
BF16 = Value(16)

#: the wrapper kinds of `vllm_meta.make_wrapper` (greedy): DERIVE_SOURCE (the SRC class) and the derive name
WRAPPERS = {"step": ("verity-vllm/engine-step/v1", "vLLMStep"), "request": ("verity-vllm/greedy-request/v1", "vLLMRequest")}


class Unexpressible(Exception):
    """φ asks for a structure the vocabulary cannot express.  The message names the fact precisely."""


@dataclass(frozen=True)
class ModelShape:
    arch: str
    model_class: str
    H: int
    V: int
    NL: int
    NH: int
    KVH: int
    D: int
    I: int
    EPS: float
    tie: bool
    max_pos: int
    rot: int

    @property
    def QKV(self) -> int:
        return (self.NH + 2 * self.KVH) * self.D


def model_shape(hf: dict) -> ModelShape:
    """The shape facts of an HF `config.json` the vocabulary reads, through the two declared homes of model facts: the family facts
    table (`engine.profiles.family_facts`: vLLM's class and constructor kind per `model_type`) and `program.model.config_of` (the
    static configuration, with the family's bias rule and the checkpoint's quantization).  The vocabulary is the dense block with
    no per-family facts: a family with any (a norm variant, a soft-cap, per-head q/k norm, MoE, LayerNorm), a linear bias, an
    activation other than SiLU, a partial rotary width or a rotary scaling without an analytic table raises `Unexpressible`."""
    from verity_vllm.engine.profiles.family_facts import FAMILY_FACTS
    from verity_vllm.program import model as analytic
    family = str(hf.get("model_type", "")).lower()
    ff = FAMILY_FACTS.get(family)
    if ff is None:
        raise Unexpressible(f"model_type {family!r}: not in the family facts table")
    rules = dict(ff.get("rules") or {})
    facts = []
    if ff.get("arch") != "dense":
        facts.append(f"family {family!r} constructor {ff.get('arch')!r} (the dense block only)")
    if ff.get("facts") or ff.get("patterns") or rules.get("qk_norm"):
        facts.append(f"family {family!r} per-family facts {sorted(ff.get('facts') or {})}, patterns {ff.get('patterns')}, "
                     f"q/k norm {rules.get('qk_norm')}")
    try:
        C, _ = analytic.config_of(hf, family_rules={family: rules})
    except (KeyError, ValueError) as e:
        raise Unexpressible(f"config.json: {e}") from None
    if C.get("BIAS") or hf.get("mlp_bias"):
        facts.append("linear biases (q/k/v or MLP)")
    act = hf.get("hidden_act", hf.get("hidden_activation", "silu"))
    if act != "silu":
        facts.append(f"hidden_act {act!r} (SiluMul only)")
    if C.get("FP8_BLOCK"):
        facts.append(f"block-scaled FP8 checkpoint (G={C['FP8_BLOCK']})")
    if analytic.rotary_dim(C) != C["D"]:
        facts.append(f"partial rotary width {analytic.rotary_dim(C)} of head_dim {C['D']}")
    try:
        analytic._inv_freq(C)
    except NotImplementedError as e:
        facts.append(str(e))
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))
    cls = re.match(r"\w+", ff["vllm_class"]).group(0)
    return ModelShape(arch=family, model_class=cls, H=int(C["H"]), V=int(C["V"]), NL=int(C["NL"]), NH=int(C["NH"]), KVH=int(C["KVH"]),
                      D=int(C["D"]), I=int(C["I"]), EPS=float(C["EPS"]), tie=bool(C["tie_word_embeddings"]), max_pos=int(C["MAX_POS"]),
                      rot=int(C["D"]))


def _arr(shape: tuple[int, ...], w: int) -> Type:
    t: Type = Value(w)
    for n in reversed(shape):
        t = Array(int(n), t)
    return t


def weight_leaves(C: ModelShape) -> list[tuple[str, Type]]:
    """The weights record of `LlamaForCausalLM` on meta, as `derive._weights_of` lists it: the distinct parameters in module order
    (the tied `lm_head.weight` is the embedding's leaf), then the distinct buffers (the one shared rotary `cos_sin_cache` [max_pos, rot]
    bf16, each attention layer's four f32 scalar scales).  The field names are informational: the descriptor carries types only."""
    bf = lambda *s: _arr(s, 16)  # noqa: E731
    out: list[tuple[str, Type]] = [("model.model.embed_tokens.weight", bf(C.V, C.H))]
    for L in range(C.NL):
        p = f"model.model.layers.{L}."
        out += [(p + "self_attn.qkv_proj.weight", bf(C.QKV, C.H)), (p + "self_attn.o_proj.weight", bf(C.H, C.NH * C.D)),
                (p + "mlp.gate_up_proj.weight", bf(2 * C.I, C.H)), (p + "mlp.down_proj.weight", bf(C.H, C.I)),
                (p + "input_layernorm.weight", bf(C.H)), (p + "post_attention_layernorm.weight", bf(C.H))]
    out.append(("model.model.norm.weight", bf(C.H)))
    if not C.tie:
        out.append(("model.lm_head.weight", bf(C.V, C.H)))
    out.append(("model.model.layers.0.self_attn.rotary_emb.cos_sin_cache", bf(C.max_pos, C.rot)))
    for L in range(C.NL):
        out += [(f"model.model.layers.{L}.self_attn.attn.{s}", Value(32)) for s in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")]
    return out


def _ptypes(spec) -> tuple:
    params, _ = spec.signature()
    return tuple(t for _, t in params)


class _Specs:
    """The vocabulary's kinds for C on the declared target, bound once each (the same objects the rules' `ctx.kind` returns)."""

    def __init__(self, C: ModelShape, target) -> None:
        from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as VOC
        from verity_vllm.program.registry import targets as TG
        self.C, self.tp, self.TG, self.voc = C, target, TG, VOC
        self.accepted = TG.uses_accepted_kinds(target)
        self.dot = TG.tensor_core_dot(target)
        self.embedding = self.kind("embedding", V=C.V, H=C.H)
        self.norm0 = self.kind("rmsnorm", N=C.H, EPS=C.EPS)
        self.fused = self.kind("rmsnorm_fused", N=C.H, EPS=C.EPS)
        self.qkv, self.o = self.gemm(C.H, C.QKV), self.gemm(C.NH * C.D, C.H)
        self.gate_up, self.down, self.logits = self.gemm(C.H, 2 * C.I), self.gemm(C.I, C.H), self.gemm(C.H, C.V)
        self.rope_q, self.rope_k = self.kind("rope", NHEADS=C.NH, D=C.D), self.kind("rope", NHEADS=C.KVH, D=C.D)
        self.silu = self.kind("silu_mul", I=C.I)
        self.token_select = self.kind("token_select", V=C.V)
        self._attn: dict[tuple, Any] = {}

    def kind(self, role: str, **statics):
        fn = self.voc.kind(role, **statics)
        if fn is None:
            raise Unexpressible(f"vocabulary {self.voc.name!r} has no kind for role {role!r} with {statics}")
        return fn

    def gemm(self, K: int, N: int):
        return self.kind("gemm", K=K, N=N) if self.accepted else self.kind("gemm_dot", K=K, N=N, DOT=self.dot)

    def attention(self, P: int, row: int, own_rows: int, launch_sq: int):
        """`UnifiedAttentionWithOutput[vllm]`'s kind for the query row at history length P (row = its index among the sequence's
        query rows in the launch, own_rows = their number, launch_sq = the launch's max_seqlen_q)."""
        C, TG, tp = self.C, self.TG, self.tp
        key = (P, row, own_rows, launch_sq)
        fn = self._attn.get(key)
        if fn is not None:
            return fn
        if self.accepted:
            fn = self.kind("attention", T=P, NH=C.NH, KVH=C.KVH, D=C.D)
        else:
            g = TG.attention_geometry(tp, C.D, launch_sq, C.NH, C.KVH)
            extra = {}
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
    sha512: str | None
    desc: dict | None
    function: str
    n_root_nodes: int
    timings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"wrapper": self.wrapper, "function": self.function, "digest": self.digest, "sha512": self.sha512,
                "n_root_nodes": self.n_root_nodes, "gates": getattr(self.program, "gates", None), "timings": self.timings}


class _Model:
    """The forward of the model class over one call's rows, emitted into `B` with the per-layer K/V history carried across calls
    (the implicit KV cache the attention rule makes explicit)."""

    def __init__(self, B: Builder, specs: _Specs, W: Coll, C: ModelShape) -> None:
        self.B, self.S, self.W, self.C = B, specs, W, C
        n = 0
        self.embed = W[n]
        n += 1
        self.layers = []
        for _ in range(C.NL):
            self.layers.append({"qkv": W[n], "o": W[n + 1], "gate_up": W[n + 2], "down": W[n + 3], "ln1": W[n + 4], "ln2": W[n + 5]})
            n += 6
        self.norm = W[n]
        n += 1
        self.lm_head = self.embed if C.tie else W[n]
        n += 0 if C.tie else 1
        self.cos_sin = W[n]
        self.hist: list[tuple[list, list]] = [([], []) for _ in range(C.NL)]
        self.lp_of: dict[int, int] = {}

    def call(self, fn, *args) -> Coll:
        return self.B.call(fn, *args)

    def forward(self, toks: list[Coll], pos0: int) -> list[Coll]:
        """Token rows (I32 values) at positions pos0.. -> the final-normed hidden rows."""
        S, C, M = self.S, self.C, len(toks)
        pe, pn0, pf = _ptypes(S.embedding), _ptypes(S.norm0), _ptypes(S.fused)
        pq, po, pgu, pd = _ptypes(S.qkv), _ptypes(S.o), _ptypes(S.gate_up), _ptypes(S.down)
        prq, prk, ps = _ptypes(S.rope_q), _ptypes(S.rope_k), _ptypes(S.silu)
        q_w, k_w, v_w = C.NH * C.D, C.KVH * C.D, C.KVH * C.D
        h = [self.call(S.embedding, Coll(pe[0], t.refs), Coll(pe[1], self.embed.refs)) for t in toks]
        x = [self.call(S.norm0, Coll(pn0[0], h[i].refs), Coll(pn0[1], self.layers[0]["ln1"].refs)) for i in range(M)]
        res = [h[i].refs for i in range(M)]
        d = None
        for L in range(C.NL):
            lw = self.layers[L]
            if L > 0:
                f = [self.call(S.fused, Coll(pf[0], d[i].refs), Coll(pf[1], res[i]), Coll(pf[2], lw["ln1"].refs)) for i in range(M)]
                x = [Coll(pq[0], f[i][0].refs) for i in range(M)]
                res = [f[i][1].refs for i in range(M)]
            qkv = [self.call(S.qkv, Coll(pq[0], x[i].refs), Coll(pq[1], lw["qkv"].refs)) for i in range(M)]
            cs = [Coll(prq[1], self.cos_sin.refs.slice((pos0 + i) * C.rot, (pos0 + i + 1) * C.rot)) for i in range(M)]
            q = [self.call(S.rope_q, Coll(prq[0], qkv[i].refs.slice(0, q_w)), cs[i]) for i in range(M)]
            k = [self.call(S.rope_k, Coll(prk[0], qkv[i].refs.slice(q_w, q_w + k_w)), Coll(prk[1], cs[i].refs)) for i in range(M)]
            kh, vh = self.hist[L]
            P0 = len(kh)
            if P0 == 0:
                self.lp_of[L] = M
            own_rows = M if P0 == 0 else 1
            launch_sq = own_rows
            for i in range(M):
                kh.append(k[i].refs)
                vh.append(qkv[i].refs.slice(q_w + k_w, q_w + k_w + v_w))
            attn = []
            for i in range(M):
                P = P0 + i + 1
                fn = S.attention(P, i if P0 == 0 else 0, own_rows, launch_sq)
                pa = _ptypes(fn)
                attn.append(self.call(fn, Coll(pa[0], q[i].refs), Coll(pa[1], concat(kh[:P])), Coll(pa[2], concat(vh[:P]))))
            o = [self.call(S.o, Coll(po[0], attn[i].refs), Coll(po[1], lw["o"].refs)) for i in range(M)]
            f = [self.call(S.fused, Coll(pf[0], o[i].refs), Coll(pf[1], res[i]), Coll(pf[2], lw["ln2"].refs)) for i in range(M)]
            res = [f[i][1].refs for i in range(M)]
            gu = [self.call(S.gate_up, Coll(pgu[0], f[i][0].refs), Coll(pgu[1], lw["gate_up"].refs)) for i in range(M)]
            a = [self.call(S.silu, Coll(ps[0], gu[i].refs)) for i in range(M)]
            d = [self.call(S.down, Coll(pd[0], a[i].refs), Coll(pd[1], lw["down"].refs)) for i in range(M)]
        f = [self.call(S.fused, Coll(pf[0], d[i].refs), Coll(pf[1], res[i]), Coll(pf[2], self.norm.refs)) for i in range(M)]
        return [f[i][0] for i in range(M)]

    def sample(self, hidden_row: Coll) -> Coll:
        S = self.S
        pl, pt = _ptypes(S.logits), _ptypes(S.token_select)
        lg = self.call(S.logits, Coll(pl[0], hidden_row.refs), Coll(pl[1], self.lm_head.refs))
        return self.call(S.token_select, Coll(pt[0], lg.refs))


def _check_phi(phi: Phi) -> None:
    facts = []
    if (phi.sampling or {}).get("kind", "greedy") != "greedy":
        facts.append(f"sampling {phi.sampling.get('kind')!r} (greedy only)")
    if int((phi.tp or {}).get("world", 1)) != 1:
        facts.append(f"tensor parallel world {phi.tp.get('world')} (TP1 only)")
    if phi.profile != "b1-eager-v3":
        facts.append(f"numerics profile {phi.profile!r}")
    if phi.execution != "bi-eager":
        facts.append(f"execution {phi.execution!r}")
    if (phi.model or {}).get("dtype", "bfloat16") != "bfloat16":
        facts.append(f"checkpoint dtype {phi.model.get('dtype')!r} (bf16 only)")
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))


def build_program(phi: Phi, wrapper: str, *, LP: int | None = None, T: int | None = None, M: int = 1, pos0: int = 8,
                  with_desc: bool = True) -> BuildResult:
    """The Program of the Build's `wrapper` for φ: `step` (one engine step, M token rows at pos0.., the last row sampled) or
    `request` (prefill of LP rows, then T decode steps each fed the previous greedy sample; returns the T+1 samples)."""
    t0 = time.time()
    _check_phi(phi)
    if wrapper not in WRAPPERS:
        raise Unexpressible(f"wrapper {wrapper!r} (have {sorted(WRAPPERS)})")
    C = model_shape(phi.hf_config)
    target = phi.target_profile()
    specs = _Specs(C, target)
    leaves = weight_leaves(C)
    wrec = Record(*leaves)
    user = (M,) if wrapper == "step" else (int(LP),)
    params = (("input_ids" if wrapper == "step" else "prompt", Array(user[0], I32)), ("weights", wrec))
    B = Builder(params, Value(1))
    ids = Coll(params[0][1], Affine("p", 0, 0, user[0]))
    W = Coll(wrec, Affine("p", 1, 0, wrec.leaves))
    model = _Model(B, specs, W, C)
    rows = [Coll(I32, ids.refs.slice(i, i + 1)) for i in range(user[0])]
    if wrapper == "step":
        hid = model.forward(rows, pos0)
        tok = model.sample(hid[-1])
        out = Coll(Array(1, I32), tok.refs)
    else:
        hid = model.forward(rows, 0)
        toks = [model.sample(hid[-1])]
        for s in range(int(T)):
            hid = model.forward([Coll(I32, toks[-1].refs)], int(LP) + s)
            toks.append(model.sample(hid[-1]))
        out = Coll(Array(len(toks), I32), concat([t.refs for t in toks]))
    B.ret_type = out.type
    body = B.finish(out)
    t_body = time.time()
    src_class, derive_name = WRAPPERS[wrapper]
    name = f"{derive_name}[{C.model_class}]"
    src_static = {"class": src_class, "mode": "profile", "profile": phi.profile, "inputs": [list(user)],
                  "config_sha256": hashlib.sha256(canonical_json({})).hexdigest()[:16]}
    from verity_vllm.program.frontend.derive import _replayer
    defn = CompositeDefinition(f"Derived[{name}]", 1, ("SRC",), lambda S, _p=params, _r=out.type: (tuple(_p), _r), _replayer(body),
                               doc=f"predicted ahead of time by verity_vllm.predict.build ({BUILDER_ID})", conformance="predicted: not a record",
                               register=False)
    sd = bind(defn, SRC=src_static)
    sd.set_body(body)
    program = Program(sd)
    timings = {"body_s": round(t_body - t0, 3), "program_s": round(time.time() - t_body, 3)}
    desc = digest = sha512 = None
    if with_desc:
        from verity.ir.partition_object import program_sha512
        t2 = time.time()
        desc = encode_program(program)
        digest = program_digest(desc)
        sha512 = program_sha512(desc)
        timings["encode_digest_s"] = round(time.time() - t2, 3)
    timings["total_s"] = round(time.time() - t0, 3)
    return BuildResult(wrapper=wrapper, program=program, digest=digest or "", sha512=sha512, desc=desc, function=sd.id,
                       n_root_nodes=len(program.fn.body.nodes) if hasattr(program.fn, "body") else -1, timings=timings)
