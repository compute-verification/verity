"""Kind vocabularies: the registered definitions a derivation mode binds semantic roles to (GUIDANCE §43.2).

A rule never names a kind directly; it asks `ctx.kind(role, **statics)`. The vocabulary of the (mode, profile)
answers with a bound registered definition or `None` (-> `Unsupported(numerics_unregistered)`). Provenance records
the vocabulary's name and version, so "which kernels does this program claim" is pinned per derivation.

`Vocabulary` is one vocabulary and `REFERENCE_CPU_F32` reference mode's. A profile's vocabulary is an adapter's, which adds it to
`PROFILES` (the vLLM adapter's profiles: `verity_vllm.program.frontend.rules.vocab`); `vocabulary_for` looks it up there.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from verity.circuits.defs import bind as ir_bind
from verity_vllm.definitions.registry import ref_prims as RP
from verity_vllm.definitions.registry import value_inputs as _vi

from verity_torch.frontend.rules import ref_kinds as R

@dataclass
class Vocabulary:
    name: str
    mode: str
    profile: str | None
    kinds: dict[str, Callable[..., object]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def version(self) -> str:
        rows = []
        for role in sorted(self.kinds):
            rows.append(f"{role}:{self.notes.get(role, '')}")
        return hashlib.sha256("\n".join([self.name, self.notes.get("_pin", "")] + rows).encode()).hexdigest()[:16]

    def kind(self, role: str, **statics):
        binder = self.kinds.get(role)
        if binder is None:
            return None
        try:
            return binder(**statics)
        except TypeError:
            return None

    def has(self, role: str) -> bool:
        return role in self.kinds

    def to_json(self) -> dict:
        d = {"name": self.name, "version": self.version, "mode": self.mode, "profile": self.profile,
             "roles": {r: self.notes.get(r, "") for r in sorted(self.kinds)}}
        pin = self.notes.get("_pin")
        if pin is not None:
            d["pin"] = pin
        return d


def _ref_doc(d) -> str:
    return (d.doc or "").split("\n")[0].strip()


REFERENCE_CPU_F32 = Vocabulary(
    "reference-ref-prims", "reference", None,
    kinds={
        "linear": lambda K, N, DT, BIAS: ir_bind(RP.LinearRef, K=K, N=N, DT=DT, BIAS=bool(BIAS)),
        "add": lambda N, DT: ir_bind(RP.AddRef, N=N, DT=DT),
        "mul": lambda N, DT: ir_bind(RP.MulRef, N=N, DT=DT),
        "relu": lambda N, DT: ir_bind(RP.ReluRef, N=N, DT=DT),
        "silu": lambda N, DT: ir_bind(RP.SiluRef, N=N, DT=DT),
        "gelu": lambda N, DT, APPROX: ir_bind(RP.GeluRef, N=N, DT=DT, APPROX=APPROX),
        "layer_norm": lambda N, EPS, DT: ir_bind(RP.LayerNormRef, N=N, EPS=EPS, DT=DT),
        "rmsnorm": lambda N, EPS, DT: ir_bind(RP.RMSNormRef, N=N, EPS=EPS, DT=DT),
        "layer_norm_value": lambda N, DT: ir_bind(_vi.LayerNormRefV2, N=N, DT=DT),
        "rmsnorm_value": lambda N, DT: ir_bind(_vi.RMSNormRefV2, N=N, DT=DT),
        "softmax": lambda N, DT: ir_bind(RP.SoftmaxRef, N=N, DT=DT),
        "embedding": lambda V, H, DT: ir_bind(RP.EmbeddingRef, V=V, H=H, DT=DT),
        "argmax": lambda V, DT: ir_bind(RP.ArgmaxRef, V=V, DT=DT),
        "i64_to_i32": lambda: RP.I64ToI32,
        "eager_attention": lambda T, NH, KVH, D, DT, SCALE_OP: ir_bind(RP.EagerAttentionRef, T=T, NH=NH, KVH=KVH, D=D, DT=DT, SCALE_OP=SCALE_OP),
        "rope_heads": lambda NHEADS, D, DT: ir_bind(RP.RoPEHeadsRef, NHEADS=NHEADS, D=D, DT=DT),
        "bf16_to_f32": lambda N: ir_bind(R.DeriveRefBf16ToF32, N=N),
        "f32_to_bf16": lambda N: ir_bind(R.DeriveRefF32ToBf16Rn, N=N),
    },
    notes={
        "linear": "LinearRef_v1{K,N,DT,BIAS}: " + _ref_doc(RP.LinearRef),
        "add": "AddRef_v1{N,DT}: " + _ref_doc(RP.AddRef), "mul": "MulRef_v1{N,DT}: " + _ref_doc(RP.MulRef),
        "relu": "ReluRef_v1{N,DT}: " + _ref_doc(RP.ReluRef), "silu": "SiluRef_v1{N,DT}: " + _ref_doc(RP.SiluRef),
        "gelu": "GeluRef_v1{N,DT,APPROX}: " + _ref_doc(RP.GeluRef), "layer_norm": "LayerNormRef_v1{N,EPS,DT}: " + _ref_doc(RP.LayerNormRef),
        "rmsnorm": "RMSNormRef_v1{N,EPS,DT}: " + _ref_doc(RP.RMSNormRef),
        "layer_norm_value": "LayerNormRef_v2{N,DT}(x, g, b, eps): LayerNormRef_v1's body with eps an input",
        "rmsnorm_value": "RMSNormRef_v2{N,DT}(x, w, eps): RMSNormRef_v1's body with eps an input", "softmax": "SoftmaxRef_v1{N,DT}: " + _ref_doc(RP.SoftmaxRef),
        "embedding": "EmbeddingRef_v1{V,H,DT}: " + _ref_doc(RP.EmbeddingRef), "argmax": "ArgmaxRef_v1{V,DT}: " + _ref_doc(RP.ArgmaxRef),
        "i64_to_i32": "I64ToI32_v1: " + _ref_doc(RP.I64ToI32),
        "eager_attention": "EagerAttentionRef_v1{T,NH,KVH,D,DT,SCALE_OP}: " + _ref_doc(RP.EagerAttentionRef),
        "rope_heads": "RoPEHeadsRef_v1{NHEADS,D,DT}: " + _ref_doc(RP.RoPEHeadsRef),
        "bf16_to_f32": "batched Bf16ToF32_v1 (exact widening; derive-lane composite over the registered primitive)",
        "f32_to_bf16": "batched F32ToBf16Rn_v1 (RNE narrowing; derive-lane composite over the registered primitive)",
        "_pin": f"{RP.REF_VOCAB_VERSION} digest {RP.ref_vocab_digest()}",
    })


#: profile -> its vocabulary; empty until an adapter adds its profiles
PROFILES: dict[str | None, Vocabulary] = {}


def vocabulary_for(mode: str, profile: str | None) -> Vocabulary:
    if mode == "reference":
        return REFERENCE_CPU_F32
    v = PROFILES.get(profile)
    if v is None:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(k for k in PROFILES if k)}")
    return v


__all__ = ["Vocabulary", "REFERENCE_CPU_F32", "PROFILES", "vocabulary_for"]
