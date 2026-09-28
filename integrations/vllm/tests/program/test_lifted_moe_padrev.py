"""padrev — adversarial tests for pad's MoE stage (`Lifted[TinyMoeFfn]` over the registered fused-MoE block; lane/r16-pad 8c2d7d628), from
LIFTING-REVIEW §3 A19 (per-expert batch composition), A20 (router argmax on encoded words), A7/R5 (Active numerics unchanged, incl. the
special f32/bf16 values the router compares), the INV-laundering rule (a garbage token must not look legitimately inactive), and the
LIFTING-STATUS §4 (i) closure claim (P3's specified list does not grow for MoE).  Padded baseline (experimental).  The circuit and the
fixture data are pad's (`test_lifted_moe`); the properties, vectors and expected values are padrev's."""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import Program
from verity_vllm.program.registry import lifted as L
from verity_vllm.program.kernels import serve3_reference as ref
from tests.program import test_lifted_moe as PMOE
from tests.program.test_lifted_moe import E, FFN, K, N, T_W13, T_W2, T_WR, T_X, TOPK, TWO, _bf16, _split, _walk

BOT16, BOT32, INV16, INV32 = L.enc_bot(16), L.enc_bot(32), L.inv(16), L.inv(32)


@pytest.fixture(scope="module")
def data():
    return PMOE.data.__wrapped__()


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


def _lifted(x, wr, w13, w2):
    return [L.lift_words(T_X, x), L.lift_words(T_WR, wr), L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)]


# ---------------------------------------------------------------------------------------------------------
# A20 — the router's argmax must never let an encoded ⊥/INV logit WIN as an Active expert id
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("e_bad", list(range(E)))
def test_a20_a_bot_logit_at_any_expert_never_yields_an_active_expert_id(data, e_bad):
    """Attack: `enc(⊥) = 2^32` read as an integer is larger than every Active f32 bit pattern, so an argmax over encoded words would elect
    the ⊥ logit's expert.  One router row `wr[e_bad]` is made ⊥ (a smuggled weight — refused at the boundary in a real record; here the
    boundary is bypassed to reach the scan): the route may be ⊥ by strictness, but it must NEVER contain an Active id — least of all
    `e_bad` — and no slot may be Active while another is ⊥ (a half-Active route would be an invented expert)."""
    x, wr, w13, w2 = data
    ws = list(L.lift_words(T_WR, wr))
    for j in range(K):
        ws[e_bad * K + j] = BOT16
    _, _, o = _run(L.lift_def(FFN), [L.lift_words(T_X, x), ws, L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)])
    out, ids, w = _split(o)
    assert not any(L.is_active(32, v) for v in ids), (e_bad, ids)
    assert e_bad not in ids
    assert len({L.is_bot(32, v) for v in ids}) == 1 and len({L.is_bot(32, v) for v in w}) == 1   # all slots alike
    assert ids == [BOT32] * TOPK and w == [BOT32] * TOPK and out == [BOT16] * K   # strictness: ⊥ route, ⊥ output


def test_a20_an_inv_logit_gives_an_inv_route_never_an_active_winner_and_never_a_bot_alibi(data):
    """The other encoded word: `INV_16 = 2^16 + 1` in one router row.  The route must be INV in every slot — not an Active id (the garbage
    word compared as large), and not ⊥ (INV laundered into a legitimate inactivity)."""
    x, wr, w13, w2 = data
    ws = list(L.lift_words(T_WR, wr))
    ws[2 * K + 3] = INV16
    _, _, o = _run(L.lift_def(FFN), [L.lift_words(T_X, x), ws, L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)])
    out, ids, w = _split(o)
    assert ids == [INV32] * TOPK and w == [INV32] * TOPK, (ids, w)
    assert out == [INV16] * K
    assert not any(L.is_bot(32, v) or L.is_active(32, v) for v in ids)


def test_inv_in_the_token_is_not_laundered_to_bot_by_the_moe_block(data):
    """A garbage word inside the hidden state (one INV among Active bf16 words): every named output of the block is INV — the route, the
    weights, the sum — and no word of the token cone is ⊥.  A lift that turned INV into ⊥ would let a corrupt token pass as 'no request'."""
    x, wr, w13, w2 = data
    xl = list(L.lift_words(T_X, x))
    xl[5] = INV16
    pl, tl, o = _run(L.lift_def(FFN), [xl, L.lift_words(T_WR, wr), L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)])
    out, ids, w = _split(o)
    assert ids == [INV32] * TOPK and w == [INV32] * TOPK and out == [INV16] * K
    words = [tl[g] for g in range(pl.input_gates, pl.gates)]
    assert not any(v in (BOT16, BOT32) for v in words)   # nowhere does INV become ⊥
    assert sum(1 for v in words if v in (INV16, INV32)) > 1000


# ---------------------------------------------------------------------------------------------------------
# A7 / R5 — Active numerics unchanged by the lifting, on the special values the router and the experts actually compare
# ---------------------------------------------------------------------------------------------------------

_BF16 = {"zero": 0x0000, "neg_zero": 0x8000, "one": 0x3F80, "inf": 0x7F80, "neg_inf": 0xFF80, "nan": 0x7FC0, "max": 0x7F7F, "min_sub": 0x0001}


@pytest.mark.parametrize("pattern", ["zero", "neg_zero", "inf", "neg_inf", "nan", "max", "min_sub"])
def test_r5_special_active_values_run_gate_for_gate_equal_to_the_unlifted_block(data, pattern):
    """A7: strict lifts are identities on Active operands — including NaN/±inf/−0/denormals, where the router's `F32GtStrict` (NaN false,
    first index wins ties), `F32IsFinite` masking and the bf16 products decide the route.  Every gate of the lifted block equals the
    unlifted block's; the route is whatever the unlifted block computes (Active), never ⊥/INV."""
    x, wr, w13, w2 = data
    xs = list(x)
    xs[0] = _BF16[pattern]
    xs[7] = _BF16[pattern]
    pu, tu, ou = _run(FFN, [xs, wr, w13, w2])
    pl, tl, ol = _run(L.lift_def(FFN), _lifted(xs, wr, w13, w2))
    assert ol == ou
    assert _gate_for_gate(pl, tl, pu, tu), pattern
    out, ids, w = _split(ol)
    assert all(L.is_active(32, v) for v in ids + w) and all(L.is_active(16, v) for v in out) and all(0 <= e < E for e in ids)


def test_r5_a_nan_router_row_still_matches_the_unlifted_route_bit_for_bit(data):
    """NaN in the ROUTER weights (one expert's logit becomes NaN): the unlifted scan's rule decides (NaN never wins a strict compare; the
    finite mask zeroes it) and the lifted block must reproduce it exactly — the encoded-word compare must not reorder NaN against the
    others."""
    x, wr, w13, w2 = data
    for e in range(E):
        wrs = list(wr)
        wrs[e * K + 1] = _BF16["nan"]
        pu, tu, ou = _run(FFN, [x, wrs, w13, w2])
        pl, tl, ol = _run(L.lift_def(FFN), _lifted(x, wrs, w13, w2))
        assert ol == ou and _gate_for_gate(pl, tl, pu, tu), e
        # the unlifted rule decides (the finite mask turns the NaN logit into 0, which CAN win against negative logits — measured: it does
        # for e = 0 on this fixture); the lifted block reproduces that route exactly, Active, never ⊥/INV
        assert all(L.is_active(32, v) for v in _split(ol)[1])


# ---------------------------------------------------------------------------------------------------------
# A19 — per-expert batch composition: the Active token's whole call is independent of its ⊥ batch-mates, in every order
# ---------------------------------------------------------------------------------------------------------


def _call_range(prog, name):
    c = next(c for c in prog.circuit.root_calls() if (c.node.name or "") == name)
    lo = min(g for g in c.returns)
    return c


def test_a19_the_active_tokens_whole_call_is_bit_equal_to_its_single_token_run_whatever_its_batch_mate_is(data):
    """pad's TWO test: token 0 ⊥, token 1 Active.  The reviewer runs every composition — (⊥, A), (A, ⊥), (A, A'), (⊥, ⊥) — and compares
    the Active token's ENTIRE call (every gate under `tok0`/`tok1`, not the outputs alone) with its single-token unlifted run; a per-expert
    count, a routed-sum denominator or a block-schedule value that depended on the batch-mate would show as a differing gate."""
    x, wr, w13, w2 = data
    rng = np.random.default_rng(7)
    x1 = _bf16(rng, K, 1.0)
    single = {}
    for name, xx in (("A", x), ("A1", x1)):
        pu, tu, ou = _run(FFN, [xx, wr, w13, w2])
        single[name] = (pu, tu, ou)
    W = [L.lift_words(T_WR, wr), L.lift_words(T_W13, w13), L.lift_words(T_W2, w2)]
    cases = {"bot,A": (None, "A"), "A,bot": ("A", None), "A,A1": ("A", "A1"), "bot,bot": (None, None)}
    for label, (t0, t1) in cases.items():
        x0w = L.bot_words(T_X) if t0 is None else L.lift_words(T_X, x if t0 == "A" else x1)
        x1w = L.bot_words(T_X) if t1 is None else L.lift_words(T_X, x if t1 == "A" else x1)
        p2, t2, o2 = _run(L.lift_def(TWO), [x0w, x1w] + W)
        n = K + 2 * TOPK
        for slot, (tk, name) in enumerate(((t0, "tok0"), (t1, "tok1"))):
            out_slice = o2[slot * n:(slot + 1) * n]
            if tk is None:
                assert out_slice == [BOT16] * K + [BOT32] * (2 * TOPK), (label, name)
                continue
            pu, tu, ou = single[tk]
            assert out_slice == ou, (label, name)
            # the whole call: the token's call in TWO vs the root body of the single-token unlifted Program, gate for gate
            call = next(c for c in p2.circuit.root_calls() if (c.node.name or "") == name)
            kids, roots = list(call.children()), list(pu.circuit.root_calls())
            assert len(kids) == len(roots) >= len(FFN.body.nodes), (label, name)   # batch nodes expand to n calls on both sides
            n_cmp = 0
            for kid, root in zip(kids, roots):   # the FFN body nodes in order; each subtree walked identically on both sides
                gl = [g for c in _walk(kid) for g in c.returns]
                gu = [g for c in _walk(root) for g in c.returns]
                assert len(gl) == len(gu), (label, name, kid.node.name)
                assert [t2[g] for g in gl] == [tu[g] for g in gu], (label, name, kid.node.name)
                n_cmp += len(gl)
            assert n_cmp > 2000, (label, name, n_cmp)


# ---------------------------------------------------------------------------------------------------------
# The renormalising router (Qwen3-MoE / Mixtral `norm_topk_prob`, R13 MOE-04) — the moe lane's Qwen3-30B-A3B row uses it; pad tested the
# raw router only.  Under ⊥ the epilogue's `denom = acc > 0 ? acc : 1` must not resurrect an Active weight (a `SelectF32(⊥, ·, 1.f)` that
# picked the constant 1.f would give an Active normalised weight on a ⊥ route).
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("norm", [False, True])
def test_the_renormalising_router_keeps_a_bot_row_bot_and_an_active_row_bit_exact(norm):
    from verity.ml.circuits import bind
    from verity_vllm.program.registry import moe
    from verity_vllm.program.registry.prims import BF16
    from verity.ml.circuits.types import Array
    R = bind(moe.MoeRouterTopKNorm if norm else moe.MoeRouterTopK, E=E, TOPK=TOPK, VPT=2)
    T_L = Array(E, BF16)
    rng = np.random.default_rng(11)
    for trial in range(6):
        logits = _bf16(rng, E, 1.0) if trial else [0x3F80, 0x3F80, 0x3F80, 0x3F80]   # ties first (the argmax's first-index rule)
        pu, tu, ou = _run(R, [logits])
        pl, tl, ol = _run(L.lift_def(R), [L.lift_words(T_L, logits)])
        assert ol == ou and _gate_for_gate(pl, tl, pu, tu), (norm, trial)
        assert all(L.is_active(32, v) for v in ol)
    _, _, ob = _run(L.lift_def(R), [L.bot_words(T_L)])
    assert ob == [BOT32] * (2 * TOPK), (norm, ob)      # weights AND ids ⊥: the `: 1.f` branch of the denominator never leaks out
    lg = L.lift_words(T_L, _bf16(rng, E, 1.0))
    lg[1] = BOT16
    _, _, o1 = _run(L.lift_def(R), [lg])
    assert o1 == [BOT32] * (2 * TOPK), (norm, o1)      # one ⊥ logit: strictness through the softmax's reductions, no Active winner


# ---------------------------------------------------------------------------------------------------------
# LIFTING-STATUS §4 (i)/(ii) — the specified list does not grow for MoE; the gather population is the fixed TOPK·(2N+K)·K
# ---------------------------------------------------------------------------------------------------------


def test_p3_closure_the_moe_block_uses_only_lselect_and_lgatherrow_from_the_specified_class_and_the_gather_count_is_fixed():
    lsp = L.lift_def(FFN)
    p = Program(lsp)
    prim_ids = set()
    for call in p.circuit.root_calls():
        for c in _walk(call):
            for nd in c.node.fn.body.nodes if hasattr(c.node.fn, "body") and c.node.fn.body is not None else []:
                prim_ids.add(L.member_id(nd.fn))  # the member the v2 wrapper computes when enabled (B7)
    spec_used = {i for i in prim_ids if i in L.specified_ids()}
    assert spec_used, "the MoE block must reach the specified class (its selects and gathers)"
    families = {i.split("_v1")[0].rstrip("0123456789x") for i in spec_used}
    assert all(i.startswith(("LSelect", "LGatherRow")) for i in spec_used), spec_used   # (i): nothing MoE-specific joined the list
    assert not any(i.startswith("LSplitsFor") or i.startswith("LStepLive") or i.startswith("LBot") or i.startswith("LIsActive") for i in spec_used)
    # (ii) the gather population per pair: TOPK·(2N+K) expert-row calls × K gathers each, over ALL E experts — whatever the route
    rows = [c for call in p.circuit.root_calls() for c in _walk(call) if c.node.fn.id.startswith("Lifted[MoeExpertRow_v1]")]
    assert len(rows) == TOPK * (2 * N + K)
    gathers = sum(1 for c in rows for nd in c.node.fn.body.nodes if L.member_id(nd.fn).startswith("LGatherRow"))
    assert gathers == TOPK * (2 * N + K)   # one batched LGatherRow node (n = K) per row call
    assert all(nd.n == K and L.member_id(nd.fn) == f"LGatherRow16x{E}_v1" for c in rows for nd in c.node.fn.body.nodes if L.member_id(nd.fn).startswith("LGatherRow"))
