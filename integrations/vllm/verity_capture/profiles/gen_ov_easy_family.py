"""[GEN ov-easy] The analytic serving family for an UNTIED checkpoint (`ServeUntied_v1{C, LP, STEPS, BN}`) — a lane-local
DIAGNOSTIC reference, nothing the gates consume.

Finding (B7 = Qwen/Qwen2.5-7B, `tie_word_embeddings=False`, sealed card G1–G6 pass): `verity_capture.compare`'s `structure`
section matches the derived flat body against the registered family `Serve_v4{C, LP, STEPS, BN}` (`b1.ServeV4`).  That family is
a TIED program by construction — both `Final` calls receive `weights["embed"]` as the lm_head operand (`b1.py:871,878`), whatever
`C["tie_word_embeddings"]` says.  For an untied checkpoint the derived Program's lm_head Gemm reads the `lm_head` weight root (a
field of its own in `analytic.weights_type(C)`; root_policy proves the runtime kept a separate parameter), so the family's
`embed` interval — already bound to the Embedding gathers' flat interval — would have to map onto a second flat interval and the
injective `_IntervalMap` refuses it: the match fails at the very first lm_head leaf (`t=LP-1/final/1 Gemm_v1{K=H,N=V}` … "no
isomorphic derived node (16 derived calls of that kind, 1 unused candidates checked)"), while `tokens`, `boundary` and
`subcircuits` pass.  The gates are unaffected (G1–G6 pass: G5's exhaustive operand provenance + sampled semantics, G3's validated
`lm_head` weight root) — the structure section is `diag_family_match`, informational — but its `fail` is a false alarm for every
untied model and the card's "family match" line goes dark exactly where the extra scrutiny would be welcome.

This module is the missing reference: `Serve_v4`'s body verbatim (`b1._forward_positions_v4`, `b1.Final`, the same names in the
same order) with `Final` reading `weights["lm_head"]` and the weights parameter typed `analytic.weights_type(C)`.  Registered
NOWHERE (`register=False`: the verity.ir registry is frozen for lanes, and a diagnostic reference must not become an admissible
identity by side effect); expanded leaf for leaf it is `Serve_v4` except at the 16 lm_head operands (`tests/test_gen_ov_easy.py`).

Wiring it in needs ONE shared-code change (`compare.py` is PROTECTED, so it is a SHARED-CODE proposal, not a lane edit): `main`
resolves the profile module's `SERVE_FAMILY(C, LP=, STEPS=, BN=)` when present and passes the result to `compare_structure(...,
family=)` — the parameter already exists — else `Serve_v4` as today.  The B7 profile exposes `SERVE_FAMILY = serve_family`; the
fork point ignores the attribute (the structure section keeps failing there, harmlessly).
"""

from __future__ import annotations

from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import Coll as _Coll, concat
from verity_vllm.registry import b1
from verity_vllm.registry.prims import I32
from verity.ir.types import Array

from .. import analytic


def _signature(S):
    return (("prompt", Array(S.LP, I32)), ("weights", analytic.weights_type(S.C))), Array(S.STEPS + 1, I32)


def _serve_untied(B, S, prompt, weights):
    """`Serve_v4`'s body with `Final` reading `weights["lm_head"]`: greedy serving of one request, prefill of LP prompt tokens
    then STEPS decode steps, STEPS+1 sampled tokens; attention `Attention_v3{..., BN}`."""
    C, LP, STEPS, BN = S.C, S.LP, S.STEPS, S.BN
    head = weights["lm_head"]
    caches, finals = b1._forward_positions_v4(B, C, BN, prompt, weights, 0, LP, None)
    tok = B.call(bind(b1.Final, C=C), finals[-1][0], finals[-1][1], weights["norm"], head, name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = b1._forward_positions_v4(B, C, BN, _Coll(Array(1, I32), tok.refs), weights, T0, 1, caches)
        caches = [(_Coll(b1.kv_type(C, T0 + 1), concat([caches[L][0].refs, new_kv[L][0].refs])),
                   _Coll(b1.kv_type(C, T0 + 1), concat([caches[L][1].refs, new_kv[L][1].refs]))) for L in range(C["NL"])]
        tok = B.call(bind(b1.Final, C=C), fin[0][0], fin[0][1], weights["norm"], head, name=f"t={T0}/final")
        toks.append(tok)
    return _Coll(Array(STEPS + 1, I32), concat([t.refs for t in toks]))


#: `Serve_v4` for a checkpoint whose output projection is its own weight (`C["tie_word_embeddings"] is False`).  Not registered.
ServeUntied = CompositeDefinition("ServeUntied", 1, ("C", "LP", "STEPS", "BN"), _signature, _serve_untied,
                                  doc=_serve_untied.__doc__ or "", conformance="model-untested", register=False)


def serve_family(C: dict, *, LP: int = 32, STEPS: int = 15, BN: int | None = None) -> SpecializedDefinition:
    """The analytic reference a profile config `C` should be compared against: `ServeUntied_v1{C, LP, STEPS, BN}` when `C` says
    the output projection is untied, else the registered `Serve_v4{C, LP, STEPS, BN}` (exactly `compare_structure`'s default).
    `BN` defaults as `compare_structure` does (the config's `BN`, else the FA2 rule at `C["D"]`); `C` is passed without `BN`."""
    if BN is None:
        BN = C.get("BN") or b1.fa2_kblock_n(C["D"])
    C = {k: v for k, v in C.items() if k != "BN"}
    fam = b1.ServeV4 if C.get("tie_word_embeddings", True) else ServeUntied
    return bind(fam, C=C, LP=LP, STEPS=STEPS, BN=BN)
