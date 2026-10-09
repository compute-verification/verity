"""[R17-42 / R17-45 (a) route A (3), lane moe] The FRONTEND's fixed-slot unit: `MoeFfnFixedL_v1(x, logits, w13, w2)` -- the padded block
from the router LOGITS (vLLM's fused block `vllm.moe_forward(hidden_states, router_logits)` starts at `_moe_C.topk_softmax`; the router
GEMM is the layer's own `mlp.gate` Compute, one resolution per dispatcher root) -- and its hand-authored lift of record `LMoeFfnL_v1`.

  (1) value: `MoeFfnFixedL(x, Gemm(x, wg), w13, w2) == MoeFfnFixed(x, wg, w13, w2) == MOE-01` word for word on every admitted route;
  (2) lift of record: `lift_def(bind(MoeFfnFixedL))` IS `bind(LMoeFfnL)` (registry AND decoded path), never a replay; at enable 1 its
      payload equals the twin's; at enable 0 it is all bot; the unselected slots are DISABLED (bot z_e, a_e = Active(0)) as in `LMoeBlock`;
  (3) the signed Stage 2 object is untouched: `LMoeBlock_v1`'s digest at (E=4, K=2, VPT=2, H=I=16) is what revb signed
      (1dd831feee7ac882..., moe_pad_stage2_routes_E4_K2.json) -- the shared `_lifted_core` emits the same nodes in the same order;
  (4) the frontend vocabulary binds the kind: `PROFILE_B1_EAGER_V3.kind("moe_ffn_fixed", ...)` is `bind(MoeFfnFixedL, ...)`.

The traced Build (the `Padded*` rules on an exported OLMoE forward, `moe_construction = padded`) is exercised on cp-moe with vLLM importable
(`derive_step --target '{"moe_construction": "padded"}'`); this file is the registry half, torch-free.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from verity.ml.circuits import Program, bind
from verity.ml.circuits.codec import decode_program, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, composite
from verity_vllm.program.registry import b1, lifted as L, moe_pad as MP
from verity_vllm.program.kernels import serve3_reference as ref
from verity_vllm.program.registry.prims import BF16
from verity.ml.circuits.types import Array

from tests.program.test_moe_pad_stage2 import (BOT16, BOT32, OFF, ON, E, H, I, VPT, W13_T, W2_T, WG_T, X_T, MoeIndexedReadRef, run, token_for_route,
                                  unlifted_inputs, weights)  # noqa: F401  (weights: the shared fixture)

LOGITS_T = Array(E, BF16)
_HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2_EVIDENCE = os.path.join(_HERE, "..", "..", "out", "gen", "sweep", "evidence", "moe", "moe_pad_stage2", "moe_pad_stage2_routes_E4_K2.json")


def logits_of(x: list[int], w: dict) -> list[int]:
    """The layer's router GEMM as the frontend binds it (`Gemm_v1{K=H,N=E}` over `mlp.gate`): the block's `logits` input."""
    _, _, out = run(bind(b1.Gemm, K=H, N=E), [x, w["wg"]])
    return out


def inputs_l(x, w):
    return [x, logits_of(x, w), w["w13"], w["w2"]]


def lifted_inputs_l(x, w, enable: int = ON):
    return [L.lift_words(X_T, x), L.lift_words(LOGITS_T, logits_of(x, w)), L.lift_words(W13_T, w["w13"]), L.lift_words(W2_T, w["w2"]), [enable]]


@composite("CallsTheTwinL", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: bind(MP.MoeFfnFixedL, **S.__dict__).signature(), conformance="protocol-defined")
def CallsTheTwinL(B, S, x, logits, w13, w2):
    return B.call(bind(MP.MoeFfnFixedL, **S.__dict__), x, logits, w13, w2, name="moe")


@pytest.mark.parametrize("TOPK", [1, 2])
def test_1_the_logits_form_equals_the_router_owning_twin_and_moe01_on_every_admitted_route(TOPK, weights):
    fixed_l = bind(MP.MoeFfnFixedL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    fixed = bind(MP.MoeFfnFixed, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    ref01 = bind(MoeIndexedReadRef, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    for route in MP.admitted_routes(E, TOPK):
        x = token_for_route(route, seed=hash(route) & 0xFFFF)
        _, _, y_l = run(fixed_l, inputs_l(x, weights))
        _, _, y_fixed = run(fixed, unlifted_inputs(x, weights))
        _, _, out01 = run(ref01, unlifted_inputs(x, weights))
        assert y_l == y_fixed == out01[:H]
    # the two twins are different Definitions (one owns the router GEMM, one takes its output): different digests, one shared core
    assert program_digest(encode_program(Program(fixed_l))) != program_digest(encode_program(Program(fixed)))
    assert fixed_l.signature()[0][1][0] == "logits" and fixed_l.signature()[0][1][1] == LOGITS_T


@pytest.mark.parametrize("TOPK", [1, 2])
def test_2_the_lift_of_record_is_the_hand_authored_lmoeffnl_on_both_paths_disabled_slots_bot_disabled_block_all_bot(TOPK, weights):
    fixed_l = bind(MP.MoeFfnFixedL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    padded_l = bind(MP.LMoeFfnL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    # the hook: registry path
    assert MP.HAND_LIFT_L is MP.LMoeFfnL and L.hand_lift_of("MoeFfnFixedL_v1") is MP.LMoeFfnL
    lifted = L.lift_def(fixed_l)
    assert lifted.defn is MP.LMoeFfnL and lifted.bindings == fixed_l.bindings and L.lift_callee(MP.MoeFfnFixedL) is MP.LMoeFfnL
    d_pad = program_digest(encode_program(Program(padded_l)))
    assert program_digest(encode_program(Program(lifted))) == d_pad != program_digest(encode_program(Program(fixed_l)))
    assert L.has_enable(padded_l) and padded_l.signature()[0][-1][0] == L.ENABLE_PARAM       # the enable LAST (B7)
    # the decoded path (GP-01 request_component / codec.decode_program)
    dec = decode_program(encode_program(Program(bind(CallsTheTwinL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I))), REGISTRY)
    node = next(n for n in dec.root.body.nodes if getattr(n.fn, "defn", None) is not None and n.fn.defn.id == "MoeFfnFixedL_v1")
    assert L._is_decoded(node.fn) and L.lift_def(node.fn).defn is MP.LMoeFfnL
    ldec = L.lift_def(dec.root)
    assert any(getattr(n.fn, "defn", None) is MP.LMoeFfnL for n in ldec.body.nodes)
    assert not any(n.fn.id.startswith("Lifted[MoeFfnFixed") for n in ldec.body.nodes)
    # value: enable 1 -> the twin's payload; the assignment record (LMoeBlockL) has the unselected slots DISABLED; enable 0 -> all bot
    block_l = bind(MP.LMoeBlockL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    for route in MP.admitted_routes(E, TOPK):
        x = token_for_route(route, seed=hash(route) & 0xFFFF)
        _, _, y_fixed = run(fixed_l, inputs_l(x, weights))
        _, _, y_pad = run(padded_l, lifted_inputs_l(x, weights))
        assert y_pad == y_fixed                                                           # Active words: payload == word
        _, _, rec = run(block_l, lifted_inputs_l(x, weights))
        y, ids, alpha, s, g, z, u, a_e = (rec[:H], rec[H:H + TOPK], rec[H + TOPK:H + 2 * TOPK], rec[H + 2 * TOPK:H + 2 * TOPK + E],
                                          rec[H + 2 * TOPK + E:H + 2 * TOPK + 2 * E], rec[H + 2 * TOPK + 2 * E:H + 2 * TOPK + 2 * E + E * H],
                                          None, rec[-E:])
        assert y == y_fixed and [L.payload(32, w) for w in ids] == list(route)
        for e in range(E):
            sel = e in route
            assert a_e[e] == (ON if sel else OFF) and s[e] == L.enable_word(1 if sel else 0)
            assert (g[e] != BOT32) is sel
            assert all((w != BOT16) is sel for w in z[e * H:(e + 1) * H])
        _, _, y0 = run(padded_l, lifted_inputs_l(x, weights, enable=OFF))
        assert y0 == [BOT16] * H


def _no_ord(fid: str) -> str:
    """A lifted id without pad G1's `ORD=<mask>` static (absent at pad v0.7): `Lifted[F]_v2{K=16,N=4,ORD=0}` -> `Lifted[F]_v2{K=16,N=4}`."""
    return re.sub(r"\{ORD=[^}]*\}$", "", re.sub(r",ORD=[^,}]*", "", fid))


def test_3_the_signed_stage2_block_is_unchanged_by_the_factoring():
    """`LMoeBlock_v1` (router-owning) emits through the shared `_lifted_core`: its digest at the Stage 2 envelope is the one revb signed
    (S-r17b-S2 (2)); `LMoeBlockL_v1` differs from it (no router GEMM node, `logits` an input) and from nothing else."""
    sp = MP.block(E, 2, VPT, H, I)
    d = program_digest(encode_program(Program(sp)))
    # The PIN is the digest of the Stage 2 block on the lifting stack of record; a pin is a pin, not a claim (rev M-1280).  It moved twice:
    #   1dd831feee7ac882  pad v0.7 / B7 (lifted.py 5460c3e03 .. 90d284951) -- the digest revb signed, S-r17b-S2 (2);
    #   de34ef2c82f65eb4  pad G1 5261740d7 (ORDINARY class: ORD statics in the lifted ids) -- the same block, the same relation; the Stage 2
    #                     evidence JSON (dump_routes.py at v2) still carries the v0.7 digest and is compared only when it names the current stack.
    #   43bb1ede072b2031  the re-baseline re-key (the k16 step AmpereBF16TcDot16_v1 -> core's _v2; nothing else moved).
    #   52de640566245b7f  the same descriptor, digested with SHA-512 (the identity re-baseline).
    STAGE2_DIGESTS = {"1dd831feee7ac882": "pad v0.7 (B7)", "de34ef2c82f65eb4": "pad G1 5261740d7 (ORD statics)", "43bb1ede072b2031": "re-key (TcDot16 v2)",
                      "52de640566245b7f": "SHA-512"}
    assert d[:16] in STAGE2_DIGESTS, d
    if os.path.exists(STAGE2_EVIDENCE):
        ev = json.load(open(STAGE2_EVIDENCE))["program_digest"]
        assert ev[:16] in STAGE2_DIGESTS and (ev == d or STAGE2_DIGESTS[ev[:16]] != STAGE2_DIGESTS[d[:16]])
    dl = program_digest(encode_program(Program(bind(MP.LMoeBlockL, E=E, TOPK=2, VPT=VPT, H=H, I=I))))
    assert dl != d
    # the node census: LMoeBlockL = LMoeBlock minus exactly the router GEMM call
    # (ids read without pad G1's ORD static -- `Lifted[Gemm_v1]_v2{K=16,N=4,ORD=0}` under G1, `{K=16,N=4}` at v0.7: the census is the same)
    names = lambda spec: sorted(_no_ord(n.fn.id) for n in Program(spec).root.body.nodes)  # noqa: E731
    a, b = names(sp), names(bind(MP.LMoeBlockL, E=E, TOPK=2, VPT=VPT, H=H, I=I))
    extra = list(a)
    for n in b:
        extra.remove(n)
    assert extra == ["Lifted[Gemm_v1]_v2{K=16,N=4}"]


def test_4_the_frontend_vocabulary_binds_the_kind():
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3
    fn = PROFILE_B1_EAGER_V3.kind("moe_ffn_fixed", E=64, TOPK=8, VPT=8, H=2048, I=1024)
    assert fn.defn is MP.MoeFfnFixedL and fn.bindings == {"E": 64, "TOPK": 8, "VPT": 8, "H": 2048, "I": 1024}
    assert fn.id == "MoeFfnFixedL_v1{E=64,TOPK=8,VPT=8,H=2048,I=1024}"
    assert "moe_ffn_fixed" in PROFILE_B1_EAGER_V3.notes and "LMoeFfnL_v1" in PROFILE_B1_EAGER_V3.notes["moe_ffn_fixed"]
