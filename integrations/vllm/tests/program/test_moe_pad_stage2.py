"""[R17-42 / MOE-PAD Stage 2, lane moe] The complete small padded MoE -- PADDING-DECISION s7 Stage 2, the first decisive milestone.

Four experts, top-1 AND top-2, the Program built before any route (`moe_pad.block`), every admitted route walked in full
(`moe_pad.admitted_routes`: 4 at top-1, 12 ordered at top-2), each demonstrated on the reference evaluator (`serve3_reference.materialize`:
every gate of the denoted circuit, nothing shortcut):

  (a) ONE Program per envelope: digest / gate count / node set fixed by (E, TOPK, VPT, H, I) alone; every route is an assignment to the same
      circuit.  The NEGATIVE the ruling names -- a builder that specializes to the observed route -- is written out and its digest moves.
  (b) EXACT active values: the selected slots' z_{r_j}, the ids, the weights and y are bit for bit the MOE-01 indexed-read block's
      (`registry/moe.py`: MoeExpertGemm / MoeExpertGemmW with the gather over the bank, MoeSum) on the same words.
  (c) CORRECT inactive assignments (R17-43 / A12, pad's B7): an unselected slot is a DISABLED computation -- its enable `a_e = LEnable(a, s_e)`
      is Active(0), every word it produces (routed weight, output) is `enc(bot)` -- never Active(0), never INV; its input `x` is wired in
      unchanged; nothing on the selected side is bot.  A DISABLED block (a = 0: the continuation's post-prefix row) is all bot whatever it
      reads, every slot enable Active(0), nothing painted; an ENABLED block reading a bot token is INV by name (the v1 mechanism -- a bot
      input making the row inactive -- is no longer how inactivity is produced, and it cannot pass silently).
  (d) CORRECT selected-output order: u_j = z_{r_j} in slot order; the two orders of one pair are two assignments.
  (e)/(f) full binding + openings and module/fine replay go through the existing Program/query machinery (next; not in this file).

rev's s7 negatives, exercised DIRECTLY on the registered local relations (not by sample): substituted routing ids / expert weights, an
unselected expert influencing the output, an omitted inactive assignment (a disabled slot with Active words), wrong selected order, early
inactivation (a selected slot disabled), cross-request mixing, unsupported index encodings, a route-specialised builder.  (The lag token in the served output is the continuation's relation --
`lifted.ContinuationChain` -- outside this block.)
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import Program, bind
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import array_of, tuple_of
from verity_vllm.program.registry import b1, lifted as L, moe, moe_pad as MP, prims as P
from verity_vllm.program.kernels import serve3_reference as ref
from verity_vllm.program.registry.prims import BF16, F32, I32
from verity.ml.circuits.types import Array, Tuple

E, VPT, H, I = 4, 2, 16, 16
BOT16, BOT32, BOT1 = L.enc_bot(16), L.enc_bot(32), L.enc_bot(1)
INV16, INV32, INV1 = L.inv(16), L.inv(32), L.inv(1)
ON, OFF = L.enable_word(1), L.enable_word(0)      # the enable words: Active(1) / Active(0) (B7)
X_T = Array(H, BF16)
WG_T = Array(E, Array(H, BF16))
W13_T = Array(E, Array(2 * I, Array(H, BF16)))
W2_T = Array(E, Array(H, Array(I, BF16)))


def bf16(v) -> list[int]:
    """f32 -> bf16 encodings, RNE."""
    b = np.asarray(v, dtype=np.float32).reshape(-1).view(np.uint32).astype(np.uint64)
    r = ((b >> 16) & 1) + 0x7FFF
    return [int(w) for w in ((b + r) >> 16)]


@pytest.fixture(scope="module")
def weights():
    """Shared weights: a ONE-HOT router (`wg[e][k] = 1.0 iff k == e`, so logit_e = x[e] exactly) makes the route a function of x[0..E-1]
    the tests set explicitly; random bf16 expert weights of moderate scale."""
    rng = np.random.default_rng(1729)
    wg = np.zeros((E, H), dtype=np.float32)
    for e in range(E):
        wg[e, e] = 1.0
    return {"wg": bf16(wg), "w13": bf16(rng.normal(0, 0.3, (E, 2 * I, H))), "w2": bf16(rng.normal(0, 0.3, (E, H, I))),
            "w13_np": rng.normal(0, 0.3, (E, 2 * I, H)).astype(np.float32)}


def token_for_route(route: tuple[int, ...], seed: int = 0) -> list[int]:
    """An Active token whose router argmax rounds pick exactly `route` (descending logits x[route[0]] > x[route[1]] > the rest)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, H).astype(np.float32)
    x[:E] = -1.0 - np.arange(E) * 0.25            # the unselected experts: distinct, all below the selected
    for j, e in enumerate(route):
        x[e] = 3.0 - j                            # 3.0, 2.0, ... in selection order
    return bf16(x)


def lifted_inputs(x: list[int], w: dict, enable: int = ON) -> list[list[int]]:
    """The block's inputs with its enable LAST (B7): `ON` = the block is an enabled computation (the root's constant 1 in a real Program)."""
    return [L.lift_words(X_T, x), L.lift_words(WG_T, w["wg"]), L.lift_words(W13_T, w["w13"]), L.lift_words(W2_T, w["w2"]), [enable]]


def unlifted_inputs(x: list[int], w: dict) -> list[list[int]]:
    return [x, w["wg"], w["w13"], w["w2"]]


def run(sp, inputs):
    prog = Program(sp)
    t = ref.materialize(prog, inputs)
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


class Assignment:
    """The block's returned record split by field (block_signature order)."""

    def __init__(self, TOPK: int, outs: list[int]) -> None:
        o = iter(outs)
        take = lambda n: [next(o) for _ in range(n)]  # noqa: E731
        self.y = take(H)
        self.ids = take(TOPK)
        self.alpha = take(TOPK)
        self.s = take(E)
        self.g = take(E)
        self.z = [take(H) for _ in range(E)]
        self.u = [take(H) for _ in range(TOPK)]
        self.a = take(E)                      # the slot enables a_e (B7 / A12), last
        assert next(o, None) is None


# ---------------------------------------------------------------------------------------------------------
# the MOE-01 indexed-read block (registry/moe.py, the labelled alternative) as the value reference for (b)
# ---------------------------------------------------------------------------------------------------------


@composite("MoeIndexedReadRefStage2", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: ((("x", Array(S.H, BF16)), ("wg", Array(S.E, Array(S.H, BF16))), ("w13", Array(S.E, Array(2 * S.I, Array(S.H, BF16)))),
                       ("w2", Array(S.E, Array(S.H, Array(S.I, BF16))))),
                      Tuple(Array(S.H, BF16), Array(S.TOPK, I32), Array(S.TOPK, F32), Array(S.TOPK, Array(S.H, BF16)))),
           conformance="protocol-defined")
def MoeIndexedReadRef(B, S, x, wg, w13, w2):
    """One token through the registered fused-MoE block of record (MOE-01): router GEMM -> MoeRouterTopK_v1 -> per (token, slot) pair
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


# ---------------------------------------------------------------------------------------------------------
# (a) one Program per envelope; the route-specialised builder is the negative
# ---------------------------------------------------------------------------------------------------------


@composite("MoeRouteSpecialisedNEGATIVE", 1, ["E", "TOPK", "VPT", "H", "I", "ROUTE"],
           lambda S: (MP.block_signature(S)[0], Array(S.H, MP.V(16))), conformance="protocol-defined")
def MoeRouteSpecialisedNEGATIVE(B, S, x, wg, w13, w2, a):
    """THE PROHIBITED BUILDER (PADDING-DECISION s3 'do not observe the route and construct only the selected expert bodies afterward'):
    the observed route enters as a STATIC and only its slots are built.  Exists here to show its digest moves with the route."""
    body = L.lift_def(bind(MP.MoeExpertBody, H=S.H, I=S.I))
    logits = B.call(L.lift_def(bind(b1.Gemm, K=S.H, N=S.E)), x, wg, a)
    r = B.call(L.lift_def(bind(moe.MoeRouterTopK, E=S.E, TOPK=S.TOPK, VPT=S.VPT)), logits, a)
    us = [B.call(body, x, r[0][j], w13[e], w2[e], a) for j, e in enumerate(S.ROUTE)]
    return B.call(L.lift_def(bind(moe.MoeSum, TOPK=S.TOPK, H=S.H)), array_of(us), a)


@pytest.mark.parametrize("TOPK", [1, 2])
def test_a_one_program_per_envelope_every_admitted_route_is_an_assignment_to_the_same_circuit(TOPK, weights):
    sp = MP.block(E, TOPK, VPT, H, I)
    prog = Program(sp)
    d0, gates0, nodes0 = program_digest(encode_program(prog)), prog.gates, [(n.fn.id, n.form, n.name) for n in sp.body.nodes]
    routes = MP.admitted_routes(E, TOPK)
    assert len(routes) == {1: 4, 2: 12}[TOPK]
    seen: set[tuple[int, ...]] = set()
    patterns: set[tuple[int, ...]] = set()
    for route in routes:
        _, _, outs = run(sp, lifted_inputs(token_for_route(route), weights))
        a = Assignment(TOPK, outs)
        assert tuple(a.ids) == route, (route, a.ids)                       # the value space is walked in full
        seen.add(tuple(a.ids))
        patterns.add(tuple(a.s))
        assert Program(MP.block(E, TOPK, VPT, H, I)).gates == gates0      # rebuilt after seeing the route: the same circuit
    assert seen == set(routes)
    assert len(patterns) == len({frozenset(r) for r in routes})            # each selected SET is one activity pattern of the fixed slots
    # the same envelope built again (a fresh bind) has the same digest, gates and node set: nothing about x, the route or the weights is in it
    p2 = Program(MP.block(E, TOPK, VPT, H, I))
    assert program_digest(encode_program(p2)) == d0 and p2.gates == gates0
    assert [(n.fn.id, n.form, n.name) for n in MP.block(E, TOPK, VPT, H, I).body.nodes] == nodes0
    # the NEGATIVE: the route-specialised builder's digest and gate count move with the route it observed
    digs = {route: program_digest(encode_program(Program(bind(MoeRouteSpecialisedNEGATIVE, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I, ROUTE=route))))
            for route in routes}
    assert len(set(digs.values())) == len(routes), "a builder that specializes to the observed route has one Program per route -- prohibited"
    assert Program(bind(MoeRouteSpecialisedNEGATIVE, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I, ROUTE=routes[0])).gates < gates0


# ---------------------------------------------------------------------------------------------------------
# (b) exact active values against the indexed-read block of record
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("TOPK", [1, 2])
def test_b_selected_slots_ids_weights_and_y_are_bit_exact_to_the_moe01_indexed_read_block(TOPK, weights):
    sp = MP.block(E, TOPK, VPT, H, I)
    rf = bind(MoeIndexedReadRef, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    for route in MP.admitted_routes(E, TOPK):
        x = token_for_route(route, seed=hash(route) % 1000)
        _, _, outs = run(sp, lifted_inputs(x, weights))
        a = Assignment(TOPK, outs)
        _, _, r = run(rf, unlifted_inputs(x, weights))
        y_ref, ids_ref, w_ref, ys_ref = r[:H], r[H:H + TOPK], r[H + TOPK:H + 2 * TOPK], [r[H + 2 * TOPK + k * H:H + 2 * TOPK + (k + 1) * H] for k in range(TOPK)]
        assert a.ids == ids_ref == list(route) and a.alpha == w_ref               # Active words: payload == word (tag 0)
        assert a.y == y_ref, route                                                # the block output, bit for bit
        for j, e in enumerate(route):
            assert a.z[e] == ys_ref[j] == a.u[j], (route, j)                      # the selected slot's z == the pair output of record
            assert a.g[e] == w_ref[j]
        assert all(v >> 16 == 0 for v in a.y) and all(v >> 32 == 0 for v in a.ids + a.alpha)


# ---------------------------------------------------------------------------------------------------------
# (c) inactive assignments: bot exactly where the route did not select, never INV, nothing painted
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("TOPK", [1, 2])
def test_c_unselected_slots_are_disabled_bot_in_weight_and_output_never_inv_and_the_selected_side_is_never_bot(TOPK, weights):
    sp = MP.block(E, TOPK, VPT, H, I)
    for route in MP.admitted_routes(E, TOPK):
        _, t, outs = run(sp, lifted_inputs(token_for_route(route), weights))
        a = Assignment(TOPK, outs)
        for e in range(E):
            if e in route:
                assert a.s[e] == 1 and a.a[e] == ON, (route, e)                   # selected: the slot is ENABLED (a_e = LEnable(1, 1) = 1)
                assert a.g[e] >> 32 == 0 and all(v >> 16 == 0 for v in a.z[e]), (route, e)
            else:
                assert a.s[e] == 0, (route, e)                                    # Active(0): the condition is a VALUE, not bot
                assert a.a[e] == OFF, (route, e)                                  # the slot is DISABLED: a_e = LEnable(1, 0) = Active(0), not bot
                assert a.g[e] == BOT32, (route, e)                                # the slot's routed weight (produced inside it) is bot
                assert a.z[e] == [BOT16] * H, (route, e)                          # every output word of the slot is enc(bot)...
                assert all(v != 0 and v != INV16 for v in a.z[e])                 # ...not Active(0) (a zero filler), not INV
        assert all(v >> 16 == 0 for v in a.y) and all(u >> 16 == 0 for uj in a.u for u in uj)
        # the slot's INNER words (not returned), read from the transcript by node name: the enable gate `slot<e>/enable` is Active(0)/(1),
        # the routed-weight chain `slot<e>/g*` is bot on a disabled slot and Active on an enabled one -- and there is NO `slot<e>/x` gate
        # any more: x is wired in unchanged (A12), the slot's inactivity is its enable, not a replaced input
        prog = Program(sp)
        root = prog.root.body                                                     # the block's nodes are inlined after the Input nodes
        seen_enable, seen_g = 0, 0
        for n_idx, node in enumerate(root.nodes):
            nm = node.name or ""
            if nm.startswith("slot") and nm.endswith("/enable"):
                e = int(nm[len("slot"):-len("/enable")])
                assert t[root.offsets[n_idx]] == (ON if e in route else OFF), (route, e)
                seen_enable += 1
            elif nm.startswith("slot") and "/g" in nm:
                e = int(nm[len("slot"):nm.index("/")])
                w = t[root.offsets[n_idx]]
                assert (w == BOT32) if (e not in route or nm.endswith("/g_none")) else (w >> 32 == 0 or w == BOT32), (route, nm, hex(w))
                seen_g += 1
            assert not nm.endswith("/x"), nm
        assert seen_enable == E and seen_g == E * (TOPK + 1)


@pytest.mark.parametrize("TOPK", [1, 2])
def test_c2_a_disabled_block_is_all_bot_with_every_slot_enable_active0_nothing_is_painted_and_an_enabled_block_on_a_bot_token_is_inv(TOPK, weights):
    sp = MP.block(E, TOPK, VPT, H, I)
    x = token_for_route(MP.admitted_routes(E, TOPK)[-1])
    # a DISABLED block (a = 0: the continuation's post-prefix row) on an ACTIVE token: every value it produces is bot (A9.3), whatever it reads
    _, _, outs = run(sp, lifted_inputs(x, weights, enable=OFF))
    a = Assignment(TOPK, outs)
    assert a.ids == [BOT32] * TOPK and a.alpha == [BOT32] * TOPK                  # no route is assigned by a disabled block
    assert a.s == [BOT1] * E and a.g == [BOT32] * E                               # the conditions are bot (never Active(0) = "unselected")
    assert a.a == [OFF] * E                                                       # the slot enables are Active(0): LEnable(0, bot) = 0, never bot
    assert all(a.z[e] == [BOT16] * H for e in range(E)) and all(uj == [BOT16] * H for uj in a.u) and a.y == [BOT16] * H
    assert INV16 not in a.y and not any(INV16 in z for z in a.z) and INV32 not in a.ids + a.alpha + a.g and INV1 not in a.s
    # ... and on a bot token: the SAME words (the disabled body's values do not depend on its operands)
    inputs = lifted_inputs(x, weights, enable=OFF)
    inputs[0] = L.bot_words(X_T)
    prog_b, t_b, outs_b = run(sp, inputs)
    assert outs_b == outs
    # revb PADDED-ACCEPTANCE Stage 2 v2 item 8 (b): EVERY word the disabled block OWNS -- not only the returned record: the router's, every
    # slot's interior (up/gate rows, the activation, every down coordinate, its f32 accumulators, g chain), the combine, the sum -- is enc(bot)
    # at its own width; the only non-bot computed words are the slot enables (LEnable: Active(0)); the Inputs are read unchanged
    owned, enables = 0, 0
    for gv in prog_b.circuit[0:prog_b.gates]:
        prim = gv.prim
        if getattr(prim, "family", None) == "Input":
            continue
        if prim.id == "LEnable_v1":
            assert t_b[gv.index] == OFF, (gv.index, prim.id, t_b[gv.index])
            enables += 1
        else:
            assert t_b[gv.index] == L.enc_bot(prim.ret.w - 1), (gv.index, prim.id, hex(t_b[gv.index]))
            owned += 1
    assert enables == E and owned == prog_b.gates - prog_b.input_gates - E and owned > 900   # 1,038 (top-2) / 943 (top-1) owned computed words at E=4, H=I=16
    # the NEGATIVE the v2 rules name: an ENABLED block reading a bot token is INV by name -- the router's strict lift yields bot ids, the
    # selection conditions are bot, LEnable(1, bot) = INV_1 (N2: an enable is never bot; a bot local under an enabled parent is ill-formed)
    # and the slot gates it enables are INV_w.  In v1 this passed silently as "the inactive token"; under R17-43 inactivity is the enable's
    inputs = lifted_inputs(x, weights, enable=ON)
    inputs[0] = L.bot_words(X_T)
    _, _, outs_i = run(sp, inputs)
    ai = Assignment(TOPK, outs_i)
    assert ai.ids == [BOT32] * TOPK and ai.s == [BOT1] * E                       # the honest strict facts (A9.4) up to the enable...
    assert ai.a == [INV1] * E                                                     # ...where the ill-formed enable is NAMED
    assert all(v == INV16 for z in ai.z for v in z) and ai.y == [INV16] * H        # and the slots it enables are INV, not bot
    # the NEGATIVE the encoding rules out: an Active all-zero token (a "pad with zeros" filler) DOES route and DOES execute experts
    _, _, outs0 = run(sp, lifted_inputs([0] * H, weights))
    a0 = Assignment(TOPK, outs0)
    assert all(v >> 32 == 0 for v in a0.ids) and sum(a0.s) == TOPK == sum(a0.a)   # an invented route with weight 1/E each: bot is not a filler


# ---------------------------------------------------------------------------------------------------------
# (d) selected-output order
# ---------------------------------------------------------------------------------------------------------


def test_d_selected_outputs_follow_the_slot_order_and_the_two_orders_of_a_pair_are_two_assignments(weights):
    sp = MP.block(E, 2, VPT, H, I)
    for (p, q) in [(0, 3), (3, 0), (1, 2), (2, 1)]:
        _, _, outs = run(sp, lifted_inputs(token_for_route((p, q)), weights))
        a = Assignment(2, outs)
        assert a.ids == [p, q] and a.u[0] == a.z[p] and a.u[1] == a.z[q]          # u_j = z_{r_j}, in slot order
        assert a.u[0] != a.u[1]
    # the same pair in the other order is a different assignment of the same circuit (ids and u reordered); MoeSum over TWO addends from a
    # zero f32 accumulator is commutative, so y agrees only when the two slots' z agree -- here x differs between the two tokens, so y differs too
    _, _, o1 = run(sp, lifted_inputs(token_for_route((0, 3)), weights))
    _, _, o2 = run(sp, lifted_inputs(token_for_route((3, 0)), weights))
    a1, a2 = Assignment(2, o1), Assignment(2, o2)
    assert a1.ids == [0, 3] and a2.ids == [3, 0] and a1.u[0] == a1.z[0] and a2.u[0] == a2.z[3]
    # the direct order check every verifier makes: u_j against the registered gather at ids_j; a swapped u fails by the relation
    lg = MP.fixed_select(E, 16)
    for n in range(H):
        rows = [a1.z[e][n] for e in range(E)]
        assert lg.evaluate(a1.ids[0], *rows) == a1.u[0][n] and lg.evaluate(a1.ids[1], *rows) == a1.u[1][n]
    assert any(lg.evaluate(a1.ids[0], *[a1.z[e][n] for e in range(E)]) != a1.u[1][n] for n in range(H))   # u swapped: caught


# ---------------------------------------------------------------------------------------------------------
# rev's s7 negatives, on the registered relations directly
# ---------------------------------------------------------------------------------------------------------


def test_negatives_substituted_ids_omitted_inactive_early_inactivation_bad_encodings_on_the_local_relations(weights):
    sp = MP.block(E, 2, VPT, H, I)
    route = (2, 0)
    x = token_for_route(route)
    _, _, outs = run(sp, lifted_inputs(x, weights))
    a = Assignment(2, outs)
    lg, lg_en, en, sel32_en = MP.fixed_select(E, 16), MP.fixed_select_enabled(E, 16), L.l_enable(), L.lifted_prim(L.l_select(32))
    # substituted routing id: the selected output must be the gather of the COMMITTED id; another id gives another slot's word (or bot)
    for n in range(H):
        rows = [a.z[e][n] for e in range(E)]
        for wrong in range(E):
            if wrong != a.ids[0]:
                assert lg.evaluate(wrong, *rows) != a.u[0][n] or rows[wrong] == rows[a.ids[0]]
    # omitted inactive assignment (A12): an unselected slot's enable must be Active(0) = LEnable(a = 1, s_e = 0) -- Active(1) violates the
    # enable relation by name -- and every word the disabled slot produces must be bot: its routed weight Active(0) (a zero filler) or the
    # route's own alpha violates Lifted[LSelect32]_v2(eq, alpha, g, enable = 0) = bot
    e_off = next(e for e in range(E) if e not in route)
    assert a.s[e_off] == 0 and a.a[e_off] == OFF == en.evaluate(ON, a.s[e_off]) and en.evaluate(ON, a.s[e_off]) != ON
    for j in range(2):
        assert sel32_en.evaluate(0, a.alpha[j], BOT32, OFF) == BOT32 != 0 and sel32_en.evaluate(0, a.alpha[j], BOT32, OFF) != a.alpha[j]
    # early inactivation: a selected slot's enable must be Active(1) = LEnable(1, 1) -- Active(0) violates it; and under enable 1 the slot's
    # routed weight is the route's alpha (never bot): Lifted[LSelect32]_v2(1, alpha_0, _, 1) = alpha_0
    e_on = route[0]
    assert a.s[e_on] == 1 and a.a[e_on] == ON == en.evaluate(ON, 1) != OFF
    assert sel32_en.evaluate(1, a.alpha[0], BOT32, ON) == a.alpha[0] != BOT32 and a.g[e_on] == a.alpha[0]
    # a disabled BLOCK (a = 0) disables every slot whatever the condition -- LEnable(0, 1) = LEnable(0, bot) = 0 -- and the combine is bot
    assert en.evaluate(OFF, 1) == en.evaluate(OFF, BOT1) == OFF
    assert lg_en.evaluate(a.ids[0], *[a.z[e][0] for e in range(E)], OFF) == BOT16
    # ill-formed enables are NAMED, never a silent Active(0): a bot or INV enable word makes LEnable INV_1 and the gates it enables INV_w
    assert en.evaluate(BOT1, 1) == en.evaluate(ON, BOT1) == en.evaluate(INV1, 1) == INV1
    assert lg_en.evaluate(a.ids[0], *[a.z[e][0] for e in range(E)], INV1) == INV16
    # unsupported index encodings: an INV index is INV (never a slot); an out-of-range Active index is NAMED -- INV_16 under pad's
    # LFixedSelect (MOE-PAD-SPEC s2, G2) -- and the boundary rejects it by name before any gate reads it either way
    rows = [a.z[e][0] for e in range(E)]
    assert lg.evaluate(INV32, *rows) == INV16
    oor = lg.evaluate(E + 3, *rows)
    assert oor == INV16 == lg_en.evaluate(E + 3, *rows, ON)
    assert oor not in [a.z[e][0] for e in range(E)] or oor == INV16                # never a slot's word by default
    with pytest.raises(L.BoundaryFailure, match="out of range"):
        L.check_boundary(32, E + 3, index_range=E)
    with pytest.raises(L.BoundaryFailure, match="invalid representation"):
        L.check_boundary(32, INV32, index_range=E)
    # a bot id at the gather reads bot (the inactive token's u_j), whatever the slots hold
    assert lg.evaluate(BOT32, *rows) == BOT16


def test_negative_an_unselected_expert_cannot_influence_the_output_and_a_selected_one_does(weights):
    sp = MP.block(E, 2, VPT, H, I)
    route = (1, 3)
    x = token_for_route(route)
    _, _, o = run(sp, lifted_inputs(x, weights))
    a = Assignment(2, o)
    e_off, e_on = 0, 1
    w13 = list(weights["w13"])
    per_e = 2 * I * H
    # perturb EVERY word of the unselected expert 0's w13: y, u, the selected z, ids and weights are bit-identical
    w_off = dict(weights, w13=[(v ^ 0x0040) if e_off * per_e <= i < (e_off + 1) * per_e else v for i, v in enumerate(w13)])
    _, _, o2 = run(sp, lifted_inputs(x, w_off))
    b = Assignment(2, o2)
    assert b.y == a.y and b.u == a.u and b.ids == a.ids and b.alpha == a.alpha and all(b.z[e] == a.z[e] for e in route)
    assert b.z[e_off] == a.z[e_off] == [BOT16] * H                              # the unselected slot's output stays bot: no path to y
    # perturb the SELECTED expert 1's w13 the same way: y moves (the selected slot's arithmetic is real)
    w_on = dict(weights, w13=[(v ^ 0x0040) if e_on * per_e <= i < (e_on + 1) * per_e else v for i, v in enumerate(w13)])
    _, _, o3 = run(sp, lifted_inputs(x, w_on))
    c = Assignment(2, o3)
    assert c.ids == a.ids and c.z[e_on] != a.z[e_on] and c.y != a.y


@composite("TwoTokensSharedExpertsStage2", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: ((("x0", Array(S.H, MP.V(16))), ("x1", Array(S.H, MP.V(16)))) + MP.block_signature(S)[0][1:-1]
                      + (("a0", L.ENABLE_T), ("a1", L.ENABLE_T)),
                      Tuple(MP.block_signature(S)[1], MP.block_signature(S)[1])), conformance="protocol-defined")
def TwoTokensSharedExperts(B, S, x0, x1, wg, w13, w2, a0, a1):
    """Two rows on shared weights, each block under ITS OWN enable (the continuation's run bit of its row, in a decode Program)."""
    blk = MP.block(S.E, S.TOPK, S.VPT, S.H, S.I)
    return tuple_of(B.call(blk, x0, wg, w13, w2, a0, name="tok0"), B.call(blk, x1, wg, w13, w2, a1, name="tok1"))


def test_negative_cross_request_mixing_a_disabled_row_beside_an_enabled_one_changes_nothing_of_the_enabled_one(weights):
    TOPK = 2
    two = bind(TwoTokensSharedExperts, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    x1 = token_for_route((3, 1), seed=7)
    _, _, alone = run(MP.block(E, TOPK, VPT, H, I), lifted_inputs(x1, weights))
    n_rec = len(alone)
    shared = [L.lift_words(WG_T, weights["wg"]), L.lift_words(W13_T, weights["w13"]), L.lift_words(W2_T, weights["w2"])]
    _, _, both = run(two, [L.bot_words(X_T), L.lift_words(X_T, x1)] + shared + [[OFF], [ON]])
    rec0, rec1 = both[:n_rec], both[n_rec:]
    assert rec1 == alone                                                          # token 1: every returned word equals its single-token run
    a0 = Assignment(TOPK, rec0)
    assert a0.y == [BOT16] * H and a0.ids == [BOT32] * TOPK and a0.a == [OFF] * E  # token 0: disabled, no route, no output -- and no edge to token 1
    # and two ENABLED tokens with different routes beside each other: each equals its own single-token run (shared weights, no shared state)
    x0 = token_for_route((0, 2), seed=3)
    _, _, alone0 = run(MP.block(E, TOPK, VPT, H, I), lifted_inputs(x0, weights))
    _, _, both2 = run(two, [L.lift_words(X_T, x0), L.lift_words(X_T, x1)] + shared + [[ON], [ON]])
    assert both2[:n_rec] == alone0 and both2[n_rec:] == alone
    # revb item 8 (c): the per-slot enable a_{r,e} = LEnable(a_r, [e selected for r]) on a pattern where the SAME expert is selected by one
    # token and not by the other -- expert 0 and 2 by token 0 only, 3 and 1 by token 1 only: the slot enables are per (row, expert), never shared
    b0, b1_ = Assignment(TOPK, both2[:n_rec]), Assignment(TOPK, both2[n_rec:])
    assert b0.a == [ON, OFF, ON, OFF] and b1_.a == [OFF, ON, OFF, ON]
    assert b0.z[0] != [BOT16] * H and b1_.z[0] == [BOT16] * H                    # slot 0: computed for token 0, disabled (bot) for token 1
    assert b1_.z[3] != [BOT16] * H and b0.z[3] == [BOT16] * H
    # ... and with token 0 DISABLED, its slot 0 is disabled too although its route (had it run) selects it: a_{0,0} = LEnable(0, ·) = 0
    _, _, both3 = run(two, [L.lift_words(X_T, x0), L.lift_words(X_T, x1)] + shared + [[OFF], [ON]])
    c0 = Assignment(TOPK, both3[:n_rec])
    assert c0.a == [OFF] * E and c0.z[0] == [BOT16] * H and both3[n_rec:] == alone


def test_lmoe_ffn_returns_y_only_takes_the_enable_last_and_the_specified_class_is_unchanged():
    sp = bind(MP.LMoeFfn, E=E, TOPK=2, VPT=VPT, H=H, I=I)
    params, ret = sp.signature()
    assert ret == Array(H, MP.V(16))
    assert L.has_enable(sp) and L.has_enable(MP.block(E, 2, VPT, H, I)) and params[-1] == (L.ENABLE_PARAM, L.ENABLE_T)
    # nothing new in pad's specified class: the block uses LSelect32, LBot32, LFixedSelect4x16, LEnable (their enabled _v2 forms) and
    # strict / constant lifts only
    import re
    allowed = re.compile(r"^(LSelect\d+|LIsActive\d+|LBot\d+|LGatherRow\d+x\d+(?:o[0-9a-f]+)?|LStepLive\d+|LSplitsForSMS\d+|LFixedSelect\d+x\d+|LEnable|LIsEOS\S*|LEmitNext\S*)_v1$")
    for fid in L.specified_ids():
        assert allowed.match(fid), fid
    # ids read without pad G1's `ORD=<mask>` static (`Lifted[MoeSum_v1]_v2{TOPK=2,H=16,ORD=0}` under G1 5261740d7; no static at v0.7)
    used = {re.sub(r",ORD=[^,}]*", "", n.fn.id) for n in MP.block(E, 2, VPT, H, I).body.nodes}
    assert {"Lifted[LSelect32_v1]_v2", "Lifted[LBot32_v1]_v2", "LEnable_v1", MP.fixed_select_enabled(E, 16).id, "Lifted[I32Eq_v1]_v2",
            "Lifted[MoeExpertBody_v1]_v2{H=16,I=16}", "Lifted[MoeSum_v1]_v2{TOPK=2,H=16}"} <= used, used
    assert MP.fixed_select_enabled(E, 16).id == f"Lifted[{MP.fixed_select(E, 16).id}]_v2"
    assert all(f.startswith("Lifted[") or f in L.specified_ids() for f in used), used
    # every computed gate of the block takes an enable (B7): the only callee without one is LEnable itself (an enable is owned by no body)
    for n in MP.block(E, 2, VPT, H, I).body.nodes:
        assert L.has_enable(n.fn) or n.fn.id == "LEnable_v1", n.fn.id


# ---------------------------------------------------------------------------------------------------------
# Stage 3 builder, route A (1): the UNLIFTED fixed-slot twin MoeFfnFixed_v1 -- value-equal to MOE-01 and to LMoeFfn_v1's payload
# ---------------------------------------------------------------------------------------------------------


@composite("StrictReplayOfTwinNEGATIVE", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: (MP.block_signature(S)[0], Array(S.H, MP.V(16))), conformance="protocol-defined")
def StrictReplayOfTwinNEGATIVE(B, S, x, wg, w13, w2, a):
    """WHAT THE REPLAY LIFT OF THE TWIN WOULD BE (`Lifted[MoeFfnFixed_v1]_v2` had it not been superseded by the hand lift): the twin's
    body replayed node by node with lifted callees under the one enable -- every slot body ENABLED whatever the route, every slot word
    Active, `g_e = 0.0f` on the unselected slots.  Value-equal to the padded block on y; a different Program with no disabled slot."""
    sp = bind(MP.MoeFfnFixed, E=S.E, TOPK=S.TOPK, VPT=S.VPT, H=S.H, I=S.I)
    return L.replay_lifted(B, sp.body, [x, wg, w13, w2], enable=a)


@composite("CallsTheTwinStage2", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: bind(MP.MoeFfnFixed, **S.__dict__).signature(), conformance="protocol-defined")
def CallsTheTwin(B, S, x, wg, w13, w2):
    """An unlifted Program that CALLS the twin (a Program's root inlines its own body, so the twin must be a callee to be a decoded node)."""
    return B.call(bind(MP.MoeFfnFixed, E=S.E, TOPK=S.TOPK, VPT=S.VPT, H=S.H, I=S.I), x, wg, w13, w2, name="twin")


@pytest.mark.parametrize("TOPK", [1, 2])
def test_moe_ffn_fixed_unlifted_twin_equals_moe01_and_its_lift_of_record_is_the_hand_authored_padded_block_on_both_paths(TOPK, weights):
    fixed = bind(MP.MoeFfnFixed, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    ref01 = bind(MoeIndexedReadRef, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    padded = bind(MP.LMoeFfn, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    d_fixed, d_padded = program_digest(encode_program(Program(fixed))), program_digest(encode_program(Program(padded)))
    for route in MP.admitted_routes(E, TOPK):
        x = token_for_route(route, seed=hash(route) & 0xFFFF)
        _, _, y_fixed = run(fixed, unlifted_inputs(x, weights))
        _, _, out01 = run(ref01, unlifted_inputs(x, weights))
        _, _, y_pad = run(padded, lifted_inputs(x, weights))
        assert y_fixed == out01[:H]                                           # bit-exact to the MOE-01 indexed-read block
        assert y_fixed == y_pad                                               # and to the padded block (Active words: payload == word)
    # route A (2): the lift OF RECORD of the twin is the hand-authored padded block (pad's register_hand_lift, moe_pad.HAND_LIFT) --
    # (i) the registry path: lift_def(bind(twin)) IS bind(LMoeFfn) with the same bindings, same digest as the padded block
    assert MP.HAND_LIFT is MP.LMoeFfn and L.hand_lift_of("MoeFfnFixed_v1") is MP.LMoeFfn
    lifted = L.lift_def(fixed)
    assert lifted.defn is MP.LMoeFfn and lifted.bindings == fixed.bindings and L.lift_callee(MP.MoeFfnFixed) is MP.LMoeFfn
    assert program_digest(encode_program(Program(lifted))) == d_padded != d_fixed
    # (ii) the decoded path (the Build path: GP-01 request_component / codec.decode_program): a Program that CALLS the twin, encoded and
    # decoded, lifts its twin node to bind(LMoeFfn), not to a replay
    from verity.ml.circuits.codec import decode_program
    from verity.ml.circuits.defs import REGISTRY
    dec = decode_program(encode_program(Program(bind(CallsTheTwin, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I))), REGISTRY)
    node = next(n for n in dec.root.body.nodes if getattr(n.fn, "defn", None) is not None and n.fn.defn.id == "MoeFfnFixed_v1")
    assert L._is_decoded(node.fn) and node.fn.bindings == fixed.bindings
    ldec_twin = L.lift_def(node.fn)
    assert ldec_twin.defn is MP.LMoeFfn and ldec_twin.bindings == fixed.bindings
    ldec = L.lift_def(dec.root)
    assert any(getattr(n.fn, "defn", None) is MP.LMoeFfn for n in ldec.body.nodes)
    assert not any(n.fn.id.startswith("Lifted[MoeFfnFixed") for n in ldec.body.nodes)
    # (iii) the NEGATIVE the hook supersedes: the twin's body REPLAYED with lifted callees computes every slot Active (no bot anywhere,
    # g = Active(0.0f) on the unselected slots), a different Program from the padded block that agrees with it on y
    strict = bind(StrictReplayOfTwinNEGATIVE, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
    x = token_for_route(MP.admitted_routes(E, TOPK)[0], seed=1)
    prog_s = Program(strict)
    t_s = ref.materialize(prog_s, lifted_inputs(x, weights))
    y_strict = [t_s[g] for g in prog_s.circuit.root_outputs()]
    _, _, y_pad = run(padded, lifted_inputs(x, weights))
    assert y_strict == y_pad
    assert program_digest(encode_program(prog_s)) != d_padded and program_digest(encode_program(prog_s)) != d_fixed
    words = list(t_s.values()) if hasattr(t_s, "values") else []
    assert BOT16 not in words and BOT32 not in words
    # ... and under enable 0 the replay is all bot too (B7 holds for the replay as for the hand lift): the same y words, no distinction outside
    t_0 = ref.materialize(prog_s, lifted_inputs(x, weights, enable=OFF))
    assert [t_0[g] for g in prog_s.circuit.root_outputs()] == [BOT16] * H
    _, _, y_pad0 = run(padded, lifted_inputs(x, weights, enable=OFF))
    assert y_pad0 == [BOT16] * H
