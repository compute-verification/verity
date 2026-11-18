"""The vLLM eager profile's registered program families (`Serve_v1`, `Serve_v3`, the decoder layer) and `DEFAULT_FAMILIES`.

A family claims a registered program identity for a derived flat program only after a verified isomorphism; the matcher
that verifies it is `verity_torch.frontend.rules.family`'s (its docstring states the claim and its checks).
"""

from __future__ import annotations

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition
from verity.circuits.defs import bind as ir_bind
from verity.circuits.refs import Coll as _Coll
from verity.circuits.refs import array_of as _array_of
from verity.circuits.refs import concat, tuple_of
from verity.circuits.types import Array, RecordType, Value
from verity.circuits.types import Tuple as TupleT
from verity_torch.frontend.rules.family import (
    RegisteredFamily,
    expand_leaves,
    match_program,
    node_leaf_id,
)
from verity_torch.frontend.rules.family import _field_intervals as _field_intervals
from verity_torch.frontend.rules.family import _runs as _runs

# ---------------------------------------------------------------------------------------------------------
# the registered families of the vLLM eager profile
# ---------------------------------------------------------------------------------------------------------
from verity_vllm.definitions.registry import b1 as _b1


def _serve_binder(flat_sd, ctx):
    C = ctx.config
    if not all(k in C for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "MAX_POS")):
        return None
    params, ret = flat_sd.signature()
    if len(params) != 2 or not isinstance(ret, Array):
        return None
    LP = params[0][1].n if isinstance(params[0][1], Array) else None
    if LP is None:
        return None
    return {"C": C, "LP": LP, "STEPS": ret.n - 1}


SERVE_FAMILY = RegisteredFamily("Family[Serve_v1]", 1, _b1.Serve, _serve_binder,
                                "structural: verified isomorphism to Serve_v1{C,LP,STEPS} (the certificate of record's program identity)")


def _padded_serve_binder(flat_sd, ctx):
    """`Serve_v3{C3, LP, D_MAX, GEMM, ATTENTION, NORM}` (serve3.py): the padded circuit. `C3` is the derivation's config
    with `EOS_ID` (spec §10: bound at the binding site, never into a v1/v2 dict); the guards the source computed must
    all test that id. The implementation records come from the vocabulary (a v2-skeleton profile)."""
    from verity_vllm.program.frontend.rules.vocab import profile_records
    C = ctx.config
    if not all(k in C for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "MAX_POS", "EOS_ID")):
        return None
    recs = profile_records(ctx.vocab)
    if recs is None:
        return None
    params, ret = flat_sd.signature()
    if len(params) != 2 or not isinstance(ret, TupleT) or len(ret.fields) != 2:
        return None
    LP = params[0][1].n if isinstance(params[0][1], Array) else None
    out_t = ret.fields[0]
    if LP is None or not isinstance(out_t, Array):
        return None
    eos = {a["preconditions_checked"].get("literal") for a in ctx.applications if a["rule"].startswith("EqScalar[pad]")}
    if eos != {C["EOS_ID"]}:
        return None  # the guards test a different id (or none): not this program
    return {"C": C, "LP": LP, "D_MAX": out_t.n - 1, "GEMM": recs["GEMM"], "ATTENTION": recs["ATTENTION"], "NORM": recs["NORM"](C)}


from verity_vllm.definitions.registry import serve3 as _serve3  # noqa: E402

SERVE_V3_FAMILY = RegisteredFamily("Family[Serve_v3]", 1, _serve3.ServeV3, _padded_serve_binder,
                                   "structural: verified isomorphism to Serve_v3{C3,LP,D_MAX,GEMM,ATTENTION,NORM} (the padded circuit, serve3-spec)")


def _layer_weights_with_table(C: dict) -> RecordType:
    lw = _b1.layer_weights_type(C)
    return RecordType(*(list(lw.fields) + [("cos_sin", Array(C["MAX_POS"], Array(C["D"], Value(16))))]))


def _decoder_layer_sig(S):
    C = S.C
    H = C["H"]
    BF = Value(16)
    rows = Array(S.T, Array(H, BF))
    params = [("h", rows)] + ([] if S.L0 else [("res", rows)]) + [("w", _layer_weights_with_table(C))]
    return (tuple(params), TupleT(rows, rows))


def _decoder_layer_body(B, S, h, *rest):
    """TA1's layer at positions T0..T0+T-1 (one layer's prefill): LayerPre x T, Attention x T over this call's
    K/V rows, LayerPost x T. Layer 0 (`L0`) takes no residual (the residual is `h`)."""
    C, T, T0 = S.C, S.T, S.T0
    res, w = (h, rest[0]) if S.L0 else (rest[0], rest[1])
    lw = _Coll(_b1.layer_weights_type(C), w.refs.slice(0, _b1.layer_weights_type(C).leaves))
    cs = w["cos_sin"]
    pre = [B.call(ir_bind(_b1.LayerPre, C=C, L0=S.L0), h[i], res[i], lw, cs[T0 + i], name=f"t={T0 + i}/pre") for i in range(T)]
    kc = _Coll(_b1.kv_type(C, T), concat([p[2].refs for p in pre]))
    vc = _Coll(_b1.kv_type(C, T), concat([p[3].refs for p in pre]))
    outs, ress = [], []
    for i in range(T):
        attn = B.call(ir_bind(_b1.Attention, T=i + 1, NH=C["NH"], KVH=C["KVH"], D=C["D"]), pre[i][1], kc[0:i + 1], vc[0:i + 1], name=f"t={T0 + i}/attn")
        post = B.call(ir_bind(_b1.LayerPost, C=C), attn, pre[i][0], lw, name=f"t={T0 + i}/post")
        outs.append(post[0])
        ress.append(post[1])
    return tuple_of(_array_of(outs), _array_of(ress))


#: The hand-built decoder layer at T positions (TA1's LayerPre/Attention/LayerPost) — the S2 structural oracle.
DecoderLayerT = CompositeDefinition("DeriveOracleDecoderLayer", 1, ("C", "L0", "T", "T0"), _decoder_layer_sig, _decoder_layer_body,
                                    doc="TA1 decoder layer at T positions (prefill of one layer): LayerPre x T, Attention x T, LayerPost x T",
                                    conformance="derived: a structural oracle, composed of exact-model-tested and model-untested TA1 composites", register=False)


def _layer_binder(flat_sd, ctx):
    # imported here: the rule needs torch, which a derivation has and this module's import does not
    from verity_vllm.program.frontend.rules.profile import RotaryEmbeddingRule
    C = ctx.config
    if not all(k in C for k in ("H", "NL", "NH", "KVH", "D", "I", "QKV", "GU", "EPS", "MAX_POS")):
        return None
    params, ret = flat_sd.signature()
    if len(params) not in (2, 3):
        return None
    h_t = params[0][1]
    if not (isinstance(h_t, Array) and isinstance(h_t.elem, Array)):
        return None
    ids = {n.fn.base_id for n in flat_sd.body.nodes if isinstance(n.fn, SpecializedDefinition)}
    L0 = "RMSNormTriton_v1" in ids
    if (len(params) == 2) != L0:
        return None
    T0 = 0
    for a in ctx.applications:
        pos = a.get("preconditions_checked", {}).get("positions")
        if a["rule"] == RotaryEmbeddingRule.info.rule_id and pos:
            T0 = pos[0]
            break
    return {"C": C, "L0": L0, "T": h_t.n, "T0": T0}


LAYER_FAMILY = RegisteredFamily("Family[DecoderLayerT]", 1, DecoderLayerT, _layer_binder,
                                "structural: verified isomorphism to the hand-built TA1 layer at T positions (S2 oracle)")

DEFAULT_FAMILIES = [SERVE_FAMILY, SERVE_V3_FAMILY, LAYER_FAMILY]

__all__ = ["RegisteredFamily", "match_program", "expand_leaves", "node_leaf_id", "SERVE_FAMILY", "SERVE_V3_FAMILY", "LAYER_FAMILY", "DecoderLayerT", "DEFAULT_FAMILIES"]
