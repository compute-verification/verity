"""Stage MoE, spec + tiny circuit on CPU — **padded baseline (experimental)**, lane `pad` (R16, BRIEF §7B).

The registered fused-MoE block (`verity_vllm/program/registry/moe.py`, R13 MOE-01: router top-k → expert row gather with the expert index as an
I32 VALUE → expert GEMMs → routed sum) lifted by node replay (`lifted.lift_def`), on a tiny per-token MoE FFN (E=4 experts, TOPK=2,
K=16 hidden, N=16 intermediate, silu_and_mul).  What this establishes, gate for gate:

  * ACTIVE ROUTING COMPUTES NORMALLY: on an Active token every gate of the lifted block equals the unlifted block — the routed ids, the
    routing weights, the expert outputs, the sum (strict lifts are identities on Active operands; `LGatherRow` reads the selected row);
  * AN INACTIVE CONTINUATION NEEDS NO EXPERT ROUTE: on a ⊥ token the router's top-k ids and weights are ⊥ in every slot — the fixed
    Program assigns NO expert to the inactive token (the argmax scan's `SelectI32` with a ⊥ comparison yields ⊥, never the initial
    index); the expert row gather with a ⊥ index yields ⊥ rows (specified `LGatherRow` rule), the GEMMs and the sum are ⊥ by strictness;
    the kernel's block schedule (`moe_align_block_size`) is not a Program value, so nothing about it has to be invented either;
  * THE NEGATIVE — a FILLER row invents a route: the same block on an Active all-zero row (what a "pad with zeros" scheme would feed)
    routes to experts (0, 1) with weight 1/E each and executes them — the invented route the requirement forbids; ⊥ is not a filler;
  * NO CROSS-TOKEN EDGE: in a two-token block sharing the expert weights, a ⊥ token 0 changes no gate of Active token 1 (equal to its
    single-token unlifted run); a ⊥ row of an UNSELECTED expert's weights does not poison an Active token's output (the gather's rule),
    while a ⊥ row of a SELECTED expert does make that expert's contribution ⊥ (strictness through the selected row) — the pair;
  * THE ROUTE IS A VALUE, NOT A STRUCTURE: the lifted block's node count / gate count / digest are fixed by (E, TOPK, K, N) alone and
    every expert's weight row is in the dataflow of every pair (the wide-fan-in gather), so no per-request Program shape depends on
    who is routed where — the fixed population holds for MoE as for dense.

What it does NOT establish: anything about a real OLMoE record (a real OLMoE step under a larger cap is the `moe` lane's pod time; not
tonight).
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import Program, bind
from verity.ml.circuits.codec import _spec_id, encode_program, program_digest
from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import Coll, array_of, coll_concat, tuple_of
from verity_vllm.program.registry import b1, lifted as L, moe, prims as P
from verity_vllm.program.kernels import serve3_reference as ref
from verity_vllm.program.registry.prims import BF16, F32, I32
from verity.ml.circuits.types import Array, Tuple

E, TOPK, VPT, K, N = 4, 2, 2, 16, 16
BOT16, BOT32 = L.enc_bot(16), L.enc_bot(32)


@composite("TinyMoeFfn", 1, ["E", "TOPK", "VPT", "K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("wr", Array(S.E, Array(S.K, BF16))), ("w13", Array(S.E, Array(2 * S.N, Array(S.K, BF16)))),
                       ("w2", Array(S.E, Array(S.K, Array(S.N, BF16))))),
                      Tuple(Array(S.K, BF16), Array(S.TOPK, I32), Array(S.TOPK, F32))),
           conformance="protocol-defined")
def TinyMoeFfn(B, S, x, wr, w13, w2):
    """One token through the registered fused-MoE block: router logits (a GEMM over the E router rows) → `MoeRouterTopK_v1` →
    per slot `MoeExpertGemm_v1` (w13) → `silu_and_mul` → `MoeExpertGemmW_v1` (w2, routed weight in) → `MoeSum_v1`."""
    logits = B.batch(bind(b1.GemmCoordinate, K=S.K), x, wr, axes=(None, 0), name="router/logits")
    r = B.call(bind(moe.MoeRouterTopK, E=S.E, TOPK=S.TOPK, VPT=S.VPT), logits, name="router/topk")
    w, ids = r[0], r[1]
    ys = []
    for k in range(S.TOPK):
        h = B.call(bind(moe.MoeExpertGemm, E=S.E, K=S.K, N=2 * S.N), x, ids[k], w13, name=f"slot{k}/w13")
        a = B.batch(P.SiluMulBf16, h[: S.N], h[S.N:], axes=(0, 0), name=f"slot{k}/silu_mul")
        ys.append(B.call(bind(moe.MoeExpertGemmW, E=S.E, K=S.N, N=S.K), a, ids[k], w[k], w2, name=f"slot{k}/w2"))
    out = B.call(bind(moe.MoeSum, TOPK=S.TOPK, H=S.K), array_of(ys), name="moe_sum")
    return tuple_of(out, ids, w)


@composite("TinyMoeTwoTokens", 1, ["E", "TOPK", "VPT", "K", "N"],
           lambda S: ((("x0", Array(S.K, BF16)), ("x1", Array(S.K, BF16)), ("wr", Array(S.E, Array(S.K, BF16))),
                       ("w13", Array(S.E, Array(2 * S.N, Array(S.K, BF16)))), ("w2", Array(S.E, Array(S.K, Array(S.N, BF16))))),
                      Tuple(Tuple(Array(S.K, BF16), Array(S.TOPK, I32), Array(S.TOPK, F32)),
                            Tuple(Array(S.K, BF16), Array(S.TOPK, I32), Array(S.TOPK, F32)))),
           conformance="protocol-defined")
def TinyMoeTwoTokens(B, S, x0, x1, wr, w13, w2):
    """Two tokens of one engine step through the block on ONE weight set (the batch the fused kernel schedules jointly)."""
    f = bind(TinyMoeFfn, E=S.E, TOPK=S.TOPK, VPT=S.VPT, K=S.K, N=S.N)
    return tuple_of(B.call(f, x0, wr, w13, w2, name="tok0"), B.call(f, x1, wr, w13, w2, name="tok1"))


FFN = bind(TinyMoeFfn, E=E, TOPK=TOPK, VPT=VPT, K=K, N=N)
TWO = bind(TinyMoeTwoTokens, E=E, TOPK=TOPK, VPT=VPT, K=K, N=N)
T_X, T_WR, T_W13, T_W2 = Array(K, BF16), Array(E, Array(K, BF16)), Array(E, Array(2 * N, Array(K, BF16))), Array(E, Array(K, Array(N, BF16)))


def _bf16(rng, n, scale):
    return [int(v) for v in (rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4242)
    x = _bf16(rng, K, 1.0)
    wr = _bf16(rng, E * K, 0.5)
    w13 = _bf16(rng, E * 2 * N * K, 0.2)
    w2 = _bf16(rng, E * K * N, 0.2)
    return x, wr, w13, w2


def _run(sp, inputs):
    prog = Program(sp)
    if L.has_enable(sp) and len(inputs) == len(sp.signature()[0]) - 1:
        inputs = list(inputs) + [[1]]  # v0.6 B7: a lifted composite takes its enable LAST; the tests run it enabled (Active(1))
    t = ref.materialize(prog, inputs)
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _gate_for_gate(pl, tl, pu, tu):
    """v0.6 B7: the lifted Program has ONE more gate -- the enable Input (its LAST prescribed input); every other gate, in order,
    carries the unlifted word (an Active word is its payload; G1 keeps the weights unwidened)."""
    from verity.ml.circuits.layout import resolve
    last = pl.input_coll(len(pl.root.signature()[0]) - 1)
    eg = {resolve(pl.circuit.root_scope, r) for r in last.refs}
    assert len(eg) == 1 and pl.gates == pu.gates + 1, (pl.gates, pu.gates)
    return [tl[g] for g in range(pl.gates) if g not in eg] == [tu[g] for g in range(pu.gates)]


def _split(outs):
    return outs[:K], outs[K:K + TOPK], outs[K + TOPK:K + 2 * TOPK]


def _lifted_inputs(x, wr, w13, w2, *, x_bot=False):
    return [L.bot_words(T_X) if x_bot else L.lift_words(T_X, x), L.lift_words(T_WR, wr), L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)]


def test_active_routing_and_expert_arithmetic_are_bit_exact_gate_for_gate(data):
    x, wr, w13, w2 = data
    pu, tu, ou = _run(FFN, [x, wr, w13, w2])
    pl, tl, ol = _run(L.lift_def(FFN), _lifted_inputs(x, wr, w13, w2))
    assert ol == ou and _gate_for_gate(pl, tl, pu, tu)  # every gate, not only the outputs (modulo the enable Input, B7)
    out, ids, w = _split(ou)
    assert len(set(ids)) == TOPK and all(0 <= e < E for e in ids), ids  # a real route: two distinct experts
    assert all(L.is_active(32, v) for v in ids + w) and all(L.is_active(16, v) for v in out)
    # the lifted block is the node-replay lift: same node count, callees lifted, the specified members where the block selects/gathers
    lsp = L.lift_def(FFN)
    assert _spec_id(lsp) == f"Lifted[TinyMoeFfn_v1]{{E={E},K={K},N={N},TOPK={TOPK},VPT={VPT}}}" or _spec_id(lsp).startswith("Lifted[TinyMoeFfn_v1]")
    ids_in_program = {n.node.fn.id for n in pl.circuit.root_calls()} | {c.node.fn.id for call in pl.circuit.root_calls() for c in call.children()}
    assert any(i.startswith("Lifted[MoeRouterTopK_v1]") for i in ids_in_program) and any(i.startswith("Lifted[MoeExpertGemm") for i in ids_in_program)


def test_an_inactive_token_has_no_route_bot_ids_bot_weights_bot_output_and_no_expert_is_invented(data):
    x, wr, w13, w2 = data
    pl, tl, ol = _run(L.lift_def(FFN), _lifted_inputs(x, wr, w13, w2, x_bot=True))
    out, ids, w = _split(ol)
    assert ids == [BOT32] * TOPK and w == [BOT32] * TOPK and out == [BOT16] * K  # the route is ⊥: no expert is assigned
    assert not any(L.is_active(32, e) for e in ids)  # in particular NOT the argmax scan's initial index 0 — nothing invented
    # the token cone is ⊥ throughout: more than a thousand ⊥ words (16- and 32-bit) inside the block; the Active gates that remain are the
    # ones with no token in their cone (the router's loop counters `i += 1`, the expert-index constants, the −10000 mask, the weights)
    words = [tl[g] for g in range(pl.input_gates, pl.gates)]
    n_bot = sum(1 for v in words if v in (BOT16, BOT32))
    assert n_bot > 1000 and n_bot < len(words)
    # no INV word among the 32-bit/16-bit outputs of the block's named stages: the router's ids/weights, every slot's outputs, the sum
    for call in pl.circuit.root_calls():
        for c in _walk(call):
            nm = c.node.name or ""
            if nm.startswith(("router/", "slot", "moe_sum")):
                for g in c.returns:
                    assert tl[g] not in (L.inv(16), L.inv(32)), (nm, g)
    # the expert weights are read only through the gather: with a ⊥ index every gathered row is ⊥ although every weight is Active
    # (the specified rule); the weights themselves are untouched Inputs
    for pi in (1, 2, 3):
        coll = pl.input_coll(pi)
        assert all(L.is_active(16, tl[r[2]]) for r in coll.refs)


def test_negative_a_filler_row_invents_a_route_where_bot_does_not(data):
    """The thing the requirement forbids, exhibited on the unlifted block: an inactive request padded with an all-zero (Active) row is
    ROUTED — uniform logits, the kernel's tie rule picks experts (0, 1) with weight 1/E — and executes two experts on a token that does not
    exist.  The lifted block on the same zero row (Active) does exactly the same (it must: Active semantics are preserved); only the ⊥
    row is unrouted.  Inactivity is the value ⊥, not a filler."""
    x, wr, w13, w2 = data
    zero = [0] * K
    _, _, ou = _run(FFN, [zero, wr, w13, w2])
    out_u, ids_u, w_u = _split(ou)
    assert ids_u == [0, 1] and w_u == [0x3E800000] * TOPK  # 1/E = 0.25f: the invented route, executed
    _, _, ol = _run(L.lift_def(FFN), _lifted_inputs(zero, wr, w13, w2))
    assert _split(ol) == (out_u, ids_u, w_u)  # Active filler ⇒ the same invented route in the lifted block (A3: a filler is a value)
    _, _, ob = _run(L.lift_def(FFN), _lifted_inputs(x, wr, w13, w2, x_bot=True))
    assert _split(ob)[1] == [BOT32] * TOPK  # only ⊥ carries "no request here"


def test_no_cross_token_edge_and_the_unselected_vs_selected_bot_expert_row_pair(data):
    x, wr, w13, w2 = data
    rng = np.random.default_rng(99)
    x1 = _bf16(rng, K, 1.0)
    # token 0 ⊥, token 1 Active on the shared weights: token 1's every output equals its single-token UNLIFTED run
    _, _, o_single = _run(FFN, [x1, wr, w13, w2])
    _, t2, o2 = _run(L.lift_def(TWO), [L.bot_words(T_X), L.lift_words(T_X, x1), L.lift_words(T_WR, wr), L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)])
    n = K + 2 * TOPK
    assert o2[:n] == [BOT16] * K + [BOT32] * (2 * TOPK) and o2[n:] == o_single
    # the pair: a ⊥ weight row of an UNSELECTED expert leaves the Active token's output bit-exact; of a SELECTED expert makes it ⊥
    _, ids, _ = _split(o_single)
    unselected = next(e for e in range(E) if e not in ids)
    selected = ids[0]

    def poison_w13(expert):
        ws = list(L.lift_words(T_W13, w13))
        base = expert * 2 * N * K
        for j in range(2 * N * K):
            ws[base + j] = BOT16
        return ws
    _, _, o_ok = _run(L.lift_def(FFN), [L.lift_words(T_X, x1), L.lift_words(T_WR, wr), poison_w13(unselected), L.lift_words(T_W2, w2)])
    assert o_ok == o_single  # LGatherRow: an inactive unselected row does not poison the read
    _, _, o_bad = _run(L.lift_def(FFN), [L.lift_words(T_X, x1), L.lift_words(T_WR, wr), poison_w13(selected), L.lift_words(T_W2, w2)])
    out_b, ids_b, w_b = _split(o_bad)
    assert ids_b == ids and w_b == _split(o_single)[2]  # the ROUTE is still Active (the router reads x and wr only) …
    assert out_b == [BOT16] * K  # … and the routed sum is ⊥: the selected expert's rows are ⊥, strictness carries it through MoeSum


def test_the_route_is_a_value_not_a_structure_the_fixed_shape_holds_for_moe(data):
    x, wr, w13, w2 = data
    lsp = L.lift_def(FFN)
    p1 = Program(lsp)
    d1 = program_digest(encode_program(p1))
    # nothing about inputs enters the Program: the same object / digest whatever the token or the weights
    assert program_digest(encode_program(Program(L.lift_def(FFN)))) == d1 and L.lift_def(FFN) is lsp
    # every expert's weight row is in the dataflow of every pair (the gather over the expert axis), for every slot: the shape is fixed by
    # (E, TOPK, K, N) — a Program whose expert set depended on the route would have a different node list per request
    rows = [c for call in p1.circuit.root_calls() for c in _walk(call) if c.node.fn.id.startswith("Lifted[MoeExpertRow_v1]")]
    assert len(rows) == TOPK * (2 * N + K)  # one gathered weight row per (slot, output coordinate) of w13 and of w2
    row = L.lift_def(bind(moe.MoeExpertRow, E=E, K=K))
    assert [(L.member_id(n.fn), n.form, n.n) for n in row.body.nodes] == [(f"LGatherRow16x{E}_v1", "batch", K)]  # reached through the v2 wrapper (B7)  # K gathers over the E experts, per row
    assert all(c.node.fn is row for c in rows)
    # so the gathered weight elements per slot are 2N×K (w13) + K×N (w2), each a fan-in over ALL E experts — the shape does not depend on the route
    assert len(rows) * K == TOPK * (2 * N * K + K * N)
    # the digest differs from the unlifted block's and the unlifted block's shape is the same node list (PAD-03 by construction here)
    assert d1 != program_digest(encode_program(Program(FFN))) and len(lsp.body.nodes) == len(FFN.body.nodes) and lsp.gates == FFN.gates


def _walk(call):
    stack = [call]
    while stack:
        c = stack.pop()
        yield c
        stack.extend(c.children())
