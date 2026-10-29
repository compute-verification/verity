"""Adversarial tests for the MIXED BATCH stage (LIFTING-REVIEW attack list A1/A5/A12), Program side, TINY shape.

Two requests share ONE weights Input and one EOS set; request A stops early (EOS at event 3), request B never stops (cap decides).
The attacks these tests aim at:

  A1  ⊥ leaking across requests — A's deactivation must not change one bit of B's values (no cross-request edge in the greedy
      Program; the only cross-request value is the S-migration count of PAD-04, which is NOT a value of B's rows);
  A5  "one inactive request never deactivates its batch-mates" — B's activity vector is [1]*(D_MAX+1) regardless of A's EOS;
  A12 the S relation reads Program activity values, not observed liveness — n_live_t is Σ_r LIsActive(tok_in_{r,t}), a Program
      value, and it drops by exactly one at A's a_r while B's rows keep S Active.

Written against the skeleton (LServe_v1 ×2) before the MIXED BATCH stage was built; the stage's record half (B2 mixedstop) is
reviewed separately. Uses only the public lifted API; nothing here is implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.circuits import Program
from verity.ml.circuits.defs import composite
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.refs import Coll, concat
from verity.ml.definitions.registry import lifted as L
from verity.ml.definitions.numerics import serve3_reference as ref
from verity.ml.definitions.registry.b1 import weights_type
from tests.program.test_serve3 import D_MAX, EOS_FOR, LP, TINY, _inputs
from verity.ml.circuits.types import Array, Value

I32 = Value(32)
NEVER = 0x7FFFFFFF  # an EOS id no TINY event emits: the cap decides


@composite("PadrevServe2", 1, ["C", "LP", "D_MAX", "EOS"],
           lambda S: ((("prompt_a", Array(S.LP, Value(32))), ("prompt_b", Array(S.LP, Value(32))),  # G1 (v0.7 A11): prompts + weights are prescribed -- ordinary
                       ("weights", weights_type(S.C))), Array(2 * (S.D_MAX + 1), Value(33))),
           conformance="protocol-defined")
def PadrevServe2(B, S, prompt_a, prompt_b, weights):
    """Two `LServe_v1` requests on ONE weights Input — the fixed population of a B2 workload, skeleton form."""
    a = B.call(L.lserve(S.C, S.LP, S.D_MAX, S.EOS, lag=0), prompt_a, weights, name="req_a")
    b = B.call(L.lserve(S.C, S.LP, S.D_MAX, S.EOS, lag=0), prompt_b, weights, name="req_b")
    return Coll(Array(2 * (S.D_MAX + 1), Value(33)), concat([a.refs, b.refs]))


def _serve2(eos, prompt_a=None, prompt_b=None):
    prompt, w = _inputs()
    pa = list(prompt) if prompt_a is None else list(prompt_a)
    pb = list(prompt) if prompt_b is None else list(prompt_b)
    prog = Program(L.bind(PadrevServe2, C=TINY, LP=LP, D_MAX=D_MAX, EOS=tuple(int(e) for e in eos)))
    t = ref.materialize(prog, [L.lift_words(Array(LP, I32), pa), L.lift_words(Array(LP, I32), pb),
                               L.lift_words(weights_type(TINY), list(w))])
    outs = [t[g] for g in prog.circuit.root_outputs()]
    return prog, t, outs[: D_MAX + 1], outs[D_MAX + 1:]


def _single(eos, prompt=None):
    p0, w = _inputs()
    prompt = list(p0) if prompt is None else list(prompt)
    prog = Program(L.lserve(TINY, LP, D_MAX, eos, lag=0))
    t = ref.materialize(prog, [L.lift_words(Array(LP, I32), prompt), L.lift_words(weights_type(TINY), list(w))])
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _all_words(prog, t, calls):
    words = []
    stack = list(calls)[::-1]
    while stack:
        c = stack.pop()
        words.extend(t[g] for g in c.returns)
        stack.extend(list(c.children())[::-1])
    return words


def _request_words(prog, t, name):
    """Every materialised word of one request's subtree (its call's children and all descendants), in emission order."""
    call = next(c for c in prog.circuit.root_calls() if c.node.name == name)
    return _all_words(prog, t, call.children())


B_PROMPT_SHIFT = 1  # prompt_b[0] += 1: on TINY the greedy transcript becomes [15, 24, 48, 16, 48, ...] — never the EOS id 17


def _prompt_b():
    prompt, _ = _inputs()
    pb = list(prompt)
    pb[0] = (pb[0] + B_PROMPT_SHIFT) % TINY["V"]
    return pb


def test_activity_is_per_request_a_function_of_its_own_committed_tokens_only():
    """Identical prompts: B's transcript IS A's, so B emits the EOS at ITS OWN event 3 and stops at ITS OWN a_r = 4 — activity is a
    function of each request's committed tokens, never of the set's membership in another row."""
    _, _, out_a, out_b = _serve2((EOS_FOR[3],))
    ia = L.l_is_active(32)
    assert [ia.evaluate(x) for x in out_a] == [ia.evaluate(x) for x in out_b] == [1] * 4 + [0] * (D_MAX - 3)
    assert out_a == out_b  # the same request twice is the same value sequence: one specialization, two addresses


@pytest.mark.slow
def test_a5_the_stopped_request_does_not_change_one_word_of_its_batch_mate():
    """ONE EOS set {17}.  Run 1: A = the TINY prompt (stops: EOS at event 3), B = the shifted prompt (never stops).  Run 2: A = the
    shifted prompt too (nobody stops).  Every word of B's subtree is identical in the two runs, and identical to B served ALONE —
    A's deactivation is A's alone; the shared weights Input leaks no ⊥ (A1/A5).  A's words equal A's single-request run."""
    eos = (EOS_FOR[3],)
    pb = _prompt_b()
    prog1, t1, out_a1, out_b1 = _serve2(eos, prompt_b=pb)
    prog2, t2, out_a2, out_b2 = _serve2(eos, prompt_a=pb, prompt_b=pb)
    ia = L.l_is_active(32)
    assert [ia.evaluate(x) for x in out_a1] == [1] * 4 + [0] * (D_MAX - 3)
    assert EOS_FOR[3] not in [L.payload(32, x) for x in out_b1[:-1]], "fixture: B must not stop on its own"
    assert [ia.evaluate(x) for x in out_b1] == [ia.evaluate(x) for x in out_a2] == [1] * (D_MAX + 1)  # A's EOS deactivated nothing of B
    wb1, wb2 = _request_words(prog1, t1, "req_b"), _request_words(prog2, t2, "req_b")
    assert len(wb1) > 10_000 and wb1 == wb2  # B's WHOLE subtree, word for word, with A stopping vs A live
    # B alone (single-request Program, same EOS set) computes the same words in the same order: the population alters no request
    prog_b, t_b, out_b_alone = _single(eos, pb)
    assert out_b_alone == out_b1 and _all_words(prog_b, t_b, prog_b.circuit.root_calls()) == wb1
    # A's subtree in the two-request Program equals A's single-request run
    prog_a, t_a, out_a_alone = _single(eos)
    assert out_a_alone == out_a1 and _all_words(prog_a, t_a, prog_a.circuit.root_calls()) == _request_words(prog1, t1, "req_a")
    # and A's ⊥ region is EXACTLY the events ≥ a_r: no Active word after, no ⊥ before (Commit's `enc(⊥) iff s ≥ a_r`)
    assert all(x == L.enc_bot(32) for x in out_a1[4:]) and all(ia.evaluate(x) for x in out_a1[:4])


@pytest.mark.slow
def test_a1_the_shared_weights_input_carries_no_bot_and_is_read_identically_by_both_requests():
    """The shared Input is Active in every word (a ⊥ weight would be a deleted parameter, refused at the boundary), and both
    requests' first reads of it are the same words — no per-request copy, one value namespace for the weights."""
    prog, t, out_a, out_b = _serve2((EOS_FOR[3],), prompt_b=_prompt_b())
    coll = prog.input_coll(2)
    gates = [resolve(prog.circuit.root_scope, coll.refs[j]) for j in range(coll.type.leaves)]
    words = [t[g] for g in gates]
    widths = list(_walk(weights_type(TINY)))
    assert len(words) == len(widths) == coll.type.leaves and all(L.is_active(w, x) for w, x in zip(widths, words))
    # a ⊥ weight word is refused at the prescribed-input boundary — a deleted parameter is not a value of the Program
    prompt, w = _inputs()
    lifted_w = L.lift_words(weights_type(TINY), list(w))
    lifted_w[0] = L.enc_bot(widths[0])
    with pytest.raises(Exception):
        for wd, x in zip(widths, lifted_w):
            L.check_boundary(wd, x, what="weights", prescribed=True)
    # both requests' weights reads are the SAME gates: one Input, two readers (no per-request copy of the weights)
    gset = set(gates)
    a_reads = set(g for g in _request_gate_reads(prog, "req_a") if g in gset)
    b_reads = set(g for g in _request_gate_reads(prog, "req_b") if g in gset)
    assert a_reads and a_reads == b_reads


def _request_gate_reads(prog, name):
    """Every gate index a request's subtree READS as an operand (the union of its calls' argument refs)."""
    call = next(c for c in prog.circuit.root_calls() if c.node.name == name)
    reads = set()
    stack = [call]
    while stack:
        c = stack.pop()
        reads.update(c.args)  # gate indices in signature order (layout.FunctionCall.args)
        stack.extend(c.children())
    return reads


def _walk(ty):
    if isinstance(ty, Value):
        yield ty.w
    elif isinstance(ty, Array):
        for _ in range(ty.n):
            yield from _walk(ty.elem)
    else:
        for f in getattr(ty, "fields", ()):
            yield from _walk(f[1] if isinstance(f, tuple) else f)


def test_a12_n_live_is_a_program_value_that_drops_by_one_at_a_r_and_b_keeps_an_active_s():
    """PAD-04 hook on the two-request skeleton: n_live_t = Σ_r LIsActive(tok_in_{r,t}) over the PROGRAM's step inputs (not a
    liveness observed on the record): 2 for t < a_A = 4, 1 from a_A on (B alone); the S operand stays Active while any row is live,
    and the retired hook Σ LIsActive(out_r[t-1]) over-counts by one exactly at t = a_A."""
    if not hasattr(L, "l_splits_for"):
        pytest.skip("PAD-04 candidate not on this tree")
    prog, t, out_a, out_b = _serve2((EOS_FOR[3],), prompt_b=_prompt_b())
    ia, ls = L.l_is_active(32), L.l_splits_for(142)
    named = {}
    for c in prog.circuit.root_calls():
        if c.node.name in ("req_a", "req_b"):
            for k in c.children():
                if k.node.name and k.node.name.endswith("/tok_in"):
                    named[(c.node.name, int(k.node.name.split("=")[1].split("/")[0]))] = k
    b_stops_itself = any(ia.evaluate(x) and L.payload(32, x) == EOS_FOR[3] for x in out_b[:-1])
    if b_stops_itself:
        pytest.skip("TINY B transcript hits the EOS set on its own; the A12 count needs an independent B")
    for s in range(1, D_MAX + 1):
        n_live = ia.evaluate(t[named[("req_a", s)].lo]) + ia.evaluate(t[named[("req_b", s)].lo])
        n_retired = ia.evaluate(out_a[s - 1]) + ia.evaluate(out_b[s - 1])
        assert n_live == (2 if s < 4 else 1), (s, n_live)
        assert ls.evaluate(n_live) == 32  # 142 SMs: 1..4 live rows → S = 32; never ⊥ while B is live
        if s == 4:
            assert n_retired == 2 and n_live == 1  # the phantom live row of the retired hook, at A's a_r
        else:
            assert n_retired == n_live
