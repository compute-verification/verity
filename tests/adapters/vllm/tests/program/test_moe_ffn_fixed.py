"""The fixed-slot MoE Definitions (`registry/moe_pad.py`): `MoeFfnFixedL_v1(x, logits, w13, w2)` is `MoeFfnFixed_v1` minus its router
GEMM, and both equal the indexed-read construction (MOE-01) word for word on every admitted route; the frontend vocabulary binds
the kind.  Torch-free."""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
from verity.circuits import Program, bind
from verity.circuits.codec import encode_program, program_digest
from verity.circuits.defs import composite
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple

from verity_vllm.definitions.numerics import serve3_reference as ref
from verity_vllm.definitions.registry import b1, moe
from verity_vllm.definitions.registry import moe_pad as MP
from verity_vllm.definitions.registry.prims import BF16, F32, I32

E, VPT, H, INTER = 4, 2, 16, 16
LOGITS_T = Array(E, BF16)


def bf16(v) -> list[int]:
    """f32 -> bf16 encodings, RNE."""
    b = np.asarray(v, dtype=np.float32).reshape(-1).view(np.uint32).astype(np.uint64)
    r = ((b >> 16) & 1) + 0x7FFF
    return [int(w) for w in ((b + r) >> 16)]


@pytest.fixture(scope="module")
def weights():
    """A ONE-HOT router (`wg[e][k] = 1.0 iff k == e`, so logit_e = x[e] exactly) makes the route a function of x[0..E-1] the tests
    set explicitly; random bf16 expert weights of moderate scale."""
    rng = np.random.default_rng(1729)
    wg: np.ndarray = np.zeros((E, H), dtype=np.float32)
    for e in range(E):
        wg[e, e] = 1.0
    return {"wg": bf16(wg), "w13": bf16(rng.normal(0, 0.3, (E, 2 * INTER, H))), "w2": bf16(rng.normal(0, 0.3, (E, H, INTER)))}


def admitted_routes(TOPK: int) -> list[tuple[int, ...]]:
    """Every ordered route the router can emit: TOPK distinct expert ids in selection order."""
    return [tuple(p) for p in permutations(range(E), TOPK)]


def token_for_route(route: tuple[int, ...], seed: int = 0) -> list[int]:
    """A token whose router argmax rounds pick exactly `route` (descending logits x[route[0]] > x[route[1]] > the rest)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, H).astype(np.float32)
    x[:E] = -1.0 - np.arange(E) * 0.25
    for j, e in enumerate(route):
        x[e] = 3.0 - j
    return bf16(x)


def run(sp, inputs):
    prog = Program(sp)
    t = ref.materialize(prog, inputs)
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


@composite("MoeIndexedReadRefPart2", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: ((("x", Array(S.H, BF16)), ("wg", Array(S.E, Array(S.H, BF16))), ("w13", Array(S.E, Array(2 * S.I, Array(S.H, BF16)))),
                       ("w2", Array(S.E, Array(S.H, Array(S.I, BF16))))),
                      Tuple(Array(S.H, BF16), Array(S.TOPK, I32), Array(S.TOPK, F32), Array(S.TOPK, Array(S.H, BF16)))),
           conformance="protocol-defined")
def MoeIndexedReadRef(B, S, x, wg, w13, w2):
    """One token through the registered fused-MoE block of record: router GEMM -> MoeRouterTopK_v1 -> per (token, slot) pair
    MoeExpertGemm_v1 (the expert index as an I32 value, the row gathered over the bank) -> SiluMul -> MoeExpertGemmW_v1 -> MoeSum_v1."""
    logits = B.call(bind(b1.Gemm, K=S.H, N=S.E), x, wg, name="router/logits")
    r = B.call(bind(moe.MoeRouterTopK, E=S.E, TOPK=S.TOPK, VPT=S.VPT), logits, name="router/topk")
    w, ids = r[0], r[1]
    ys = []
    for k in range(S.TOPK):
        h = B.call(bind(moe.MoeExpertGemm, E=S.E, K=S.H, N=2 * S.I), x, ids[k], w13, name=f"pair{k}/w13")
        a = B.call(bind(b1.SiluMul, I=S.I), h, name=f"pair{k}/silu_mul")
        ys.append(B.call(bind(moe.MoeExpertGemmW, E=S.E, K=S.I, N=S.H), a, ids[k], w[k], w2, name=f"pair{k}/w2"))
    out = B.call(bind(moe.MoeSum, TOPK=S.TOPK, H=S.H), array_of(ys), name="moe_sum")
    return tuple_of(out, ids, w, array_of(ys))


def unlifted_inputs(x, w):
    return [x, w["wg"], w["w13"], w["w2"]]


def logits_of(x: list[int], w: dict) -> list[int]:
    """The layer's router GEMM as the frontend binds it (`Gemm_v1{K=H,N=E}` over `mlp.gate`): the block's `logits` input."""
    _, _, out = run(bind(b1.Gemm, K=H, N=E), [x, w["wg"]])
    return out


@pytest.mark.parametrize("TOPK", [1, 2])
def test_the_fixed_slot_twin_equals_moe01_on_every_admitted_route(TOPK, weights):
    fixed = bind(MP.MoeFfnFixed, E=E, TOPK=TOPK, VPT=VPT, H=H, I=INTER)
    ref01 = bind(MoeIndexedReadRef, E=E, TOPK=TOPK, VPT=VPT, H=H, I=INTER)
    for route in admitted_routes(TOPK):
        x = token_for_route(route, seed=hash(route) & 0xFFFF)
        _, _, y_fixed = run(fixed, unlifted_inputs(x, weights))
        _, _, out01 = run(ref01, unlifted_inputs(x, weights))
        assert y_fixed == out01[:H]


@pytest.mark.parametrize("TOPK", [1, 2])
def test_the_logits_form_equals_the_router_owning_twin_and_moe01_on_every_admitted_route(TOPK, weights):
    fixed_l = bind(MP.MoeFfnFixedL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=INTER)
    fixed = bind(MP.MoeFfnFixed, E=E, TOPK=TOPK, VPT=VPT, H=H, I=INTER)
    ref01 = bind(MoeIndexedReadRef, E=E, TOPK=TOPK, VPT=VPT, H=H, I=INTER)
    for route in admitted_routes(TOPK):
        x = token_for_route(route, seed=hash(route) & 0xFFFF)
        _, _, y_l = run(fixed_l, [x, logits_of(x, weights), weights["w13"], weights["w2"]])
        _, _, y_fixed = run(fixed, unlifted_inputs(x, weights))
        _, _, out01 = run(ref01, unlifted_inputs(x, weights))
        assert y_l == y_fixed == out01[:H]
    # the two twins are different Definitions (one owns the router GEMM, one takes its output): different digests, one shared core
    assert program_digest(encode_program(Program(fixed_l))) != program_digest(encode_program(Program(fixed)))
    assert fixed_l.signature()[0][1][0] == "logits" and fixed_l.signature()[0][1][1] == LOGITS_T


def test_the_frontend_vocabulary_binds_the_kind():
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3
    fn = PROFILE_B1_EAGER_V3.kind("moe_ffn_fixed", E=64, TOPK=8, VPT=8, H=2048, I=1024)
    assert fn.defn is MP.MoeFfnFixedL and fn.bindings == {"E": 64, "TOPK": 8, "VPT": 8, "H": 2048, "I": 1024}
    assert fn.id == "MoeFfnFixedL_v1{E=64,TOPK=8,VPT=8,H=2048,I=1024}"
    assert "moe_ffn_fixed" in PROFILE_B1_EAGER_V3.notes
