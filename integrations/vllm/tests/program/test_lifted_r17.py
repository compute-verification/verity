"""Two rules of the lifted-padded Program as tiny-circuit tests (nothing here is a panel workload).

(a) Total relation behaviour for NONCANONICAL `1‖nonzero` words, stated and tested PER OPERATION CLASS (A3.2): arithmetic (the strict
class), selection (`LSelect`), indexing (`LGatherRow`, `Lifted[Embedding]`), state ops (`Lifted[StepBody]`: the K/V cache), plus the
activity primitives.  Every class maps ANY noncanonical operand, in ANY position (the untaken branch and the unselected row included), to
the one noncanonical word `INV_w` — never to ⊥, never to an Active value, never to the operand's own bits; a RANDOM noncanonical word is
treated like `INV_w` itself.  A3.3 (honest computation emits canonical values) is MEASURED gate for gate on the lifted composites and the
lifted serving Program; A3.4 (the commitment layer never normalizes) on `commit_leaf`; A3.5 (boundaries reject by name) on `check_boundary`.

(b) TRUE widened widths (A4): the width of a lifted set is `Σ (w+1)`; admissibility is the UNCHANGED `partition.validate_width` at the
lifted parameter `W_MAX_LIFTED`; the active-reference rule at 32 fails on a 33-bit leaf BY DESIGN and is reported, never used as the lifted
verdict; a lifted use document carries `kind = vu-partition-lifted` and `rules_version = lifted-v1` (A5) and `gate_G8` refuses it by name.

LAG declared (A2): the attestation (`lag_from_header`) and the declaration (`declared_lag_of_row`) go through GP-01's ONE rule
(`eos_lag_rows_declared`); the lifted tool's R0 needs both, equal; a Build that declared no lag is never ok (tested in test_lifted_record).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.circuits import Program, bind
from verity.ml.circuits.defs import REGISTRY
from verity.ml.definitions.registry import b1, lifted as L, pad_prims as PP, prims as P
from verity_vllm.program.lifting import lifted_records
from verity_vllm.program.lifting import lifted_compose
from verity.ml.definitions.numerics import serve3_reference as ref
from verity.ml.definitions.registry.b1 import weights_type
from tests.program.test_serve3 import D_MAX, LP, TINY, _inputs
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.workload import eos_lag_rows_declared

I32, BF16 = Value(32), Value(16)
BOT32, BOT16, INV32, INV16 = L.enc_bot(32), L.enc_bot(16), L.inv(32), L.inv(16)
RNG = np.random.default_rng(1706)

class _V1AndV2:
    """The lifted relation under test = `Lifted[p]_v1` (the propagation relation) and, on the SAME
    vectors, `Lifted[p]_v2` at `enable = Active(1)` must agree with it (the enabled result is the propagation relation)."""

    def __init__(self, p):
        self.v1, self.v2 = L.lifted_prim_v1(p), L.lifted_prim(p)
        self.id, self.lift_class, self.ret, self.params = self.v2.id, getattr(self.v1, "lift_class", None), self.v1.ret, self.v1.params

    def evaluate(self, *ops):
        r = self.v1.evaluate(*ops)
        assert self.v2.evaluate(*ops, L.enc_active(1, 1)) == r, (self.id, ops)
        assert self.v2.evaluate(*ops, L.enc_active(1, 0)) == L.enc_bot(self.v1.ret.w - 1), (self.id, ops)  # disabled → ⊥ whatever the operands
        return r



def _rand(w: int, n: int) -> list[int]:
    return [int(x) for x in RNG.integers(0, 1 << w, n, dtype=np.uint64)]


def _noncanonical(w: int) -> int:
    """A RANDOM `1‖nonzero` word of `w+1` bits that is neither ⊥ nor `INV_w`: what a dishonest prover might commit.
    At `w = 1` the only noncanonical word IS `INV_1 = 0b11`."""
    if w == 1:
        return L.inv(1)
    while True:
        x = (1 << w) | int(RNG.integers(2, 1 << w, dtype=np.uint64))
        if x != L.inv(w):
            return x


def _run(sp, inputs):
    prog = Program(sp)
    t = ref.materialize(prog, inputs)
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _runl(lsp, inputs, enable: int = 1):
    """Run a LIFTED composite (`lift_def`) as a Program: its last parameter is the enable, a one-leaf Boolean word; the A3
    rows below are its relation at `enable = Active(1)` (facts about the total relation)."""
    return _run(lsp, list(inputs) + [[enable]])


def _all_words_canonical(prog, t) -> bool:
    # G1: an ordinary gate (an input, an unwidened constant) has no tag -- its word is a value, not a lifted word to classify
    return all(L.valid(prog.circuit.gate(g).prim.ret.w - 1, t[g]) for g in range(prog.gates) if L.is_lifted_prim(prog.circuit.gate(g).prim))


# ---------------------------------------------------------------------------------------------------------
# A1 — vocabulary: canonical / noncanonical / INV_w, named
# ---------------------------------------------------------------------------------------------------------


def test_a1_vocabulary_word_class_names_every_word_and_rewrites_none():
    for w in (1, 16, 32):
        assert L.word_class(w, 0) == "active" and L.word_class(w, (1 << w) - 1) == "active"
        assert L.word_class(w, L.enc_bot(w)) == "bot"
        assert L.word_class(w, L.inv(w)) == "noncanonical"  # INV_w is one of the noncanonical words, not a third class
        for _ in range(8):
            x = _noncanonical(w)
            assert L.word_class(w, x) == "noncanonical" and not L.valid(w, x) and (x != L.inv(w) or w == 1)
        with pytest.raises(ValueError):
            L.word_class(w, 1 << (w + 1))
    assert L.inv(w) == (1 << w) | 1  # `1‖0^{w-1}‖1`


# ---------------------------------------------------------------------------------------------------------
# A3.2 — total relation behaviour per operation class on RANDOM noncanonical words
# ---------------------------------------------------------------------------------------------------------

ARITH = [p for p in list(P.ALL_SCALAR) + list(PP.ALL_NEW) if p.id not in ("SelectI32_v1", "SelectF32_v1", "SelectBf16_v1")]


@pytest.mark.parametrize("p", ARITH, ids=lambda p: p.id)
def test_a3_2_arithmetic_any_random_noncanonical_operand_in_any_position_is_inv(p):
    """Arithmetic (the strict class): the output is `INV_w` exactly — not the operand's own bits, not ⊥, not `0‖payload` — for a random
    noncanonical word in EVERY operand position, with the other operands Active, ⊥, or noncanonical themselves."""
    lp = _V1AndV2(p)
    ws = [w for _, t in p.params for w in t.leaf_widths()]
    inv_out = L.inv(p.ret.w)
    ops = [_rand(w, 1)[0] for w in ws]
    for i in range(len(ws)):
        for other in ("active", "bot", "noncanonical"):
            o = list(ops)
            if other == "bot":
                o = [L.enc_bot(w) for w in ws]
            elif other == "noncanonical":
                o = [_noncanonical(w) for w in ws]
            o[i] = _noncanonical(ws[i])
            got = lp.evaluate(*o)
            assert got == inv_out, (p.id, i, other, hex(o[i]), hex(got))
            assert (got != o[i] or ws[i] == 1 == p.ret.w) and not L.is_bot(p.ret.w, got) and L.word_class(p.ret.w, got) == "noncanonical"


def test_a3_2_selection_random_noncanonical_anywhere_including_the_untaken_branch_is_inv():
    for w in (16, 32):
        s = L.l_select(w)
        a, b = _rand(w, 1)[0], _rand(w, 1)[0]
        for _ in range(6):
            x = _noncanonical(w)
            assert s.evaluate(1, x, b) == L.inv(w) and s.evaluate(0, a, x) == L.inv(w)          # taken branch
            assert s.evaluate(1, a, x) == L.inv(w) and s.evaluate(0, x, b) == L.inv(w)          # UNTAKEN branch: poisons too
            assert s.evaluate(L.enc_bot(1), x, b) == L.inv(w) and s.evaluate(L.enc_bot(1), a, x) == L.inv(w)  # ⊥ condition does not hide it
            assert s.evaluate(1, x, L.enc_bot(w)) == L.inv(w)  # INV beats ⊥
        assert s.evaluate(0b11, a, b) == L.inv(w)  # the 2-bit condition's own noncanonical word
        # honest relation untouched
        assert s.evaluate(1, a, b) == a and s.evaluate(0, a, b) == b and s.evaluate(1, a, L.enc_bot(w)) == a


def test_a3_2_indexing_random_noncanonical_index_or_any_row_selected_or_not_is_inv():
    g = L.l_gather_row(16, 8)
    rows = _rand(16, 8)
    assert g.evaluate(3, *rows) == rows[3]
    for _ in range(6):
        xi, xr = _noncanonical(32), _noncanonical(16)
        assert g.evaluate(xi, *rows) == INV16                                   # noncanonical index
        for j in range(8):
            r = list(rows)
            r[j] = xr
            assert g.evaluate(3, *r) == INV16, j                                 # the selected row (j = 3) AND every unselected row
        assert g.evaluate(BOT32, *[rows[0]] * 7 + [xr]) == INV16                 # INV beats ⊥ index
    # through the composite (Lifted[Embedding]): the unselected noncanonical row poisons exactly the columns that read it
    EMB = bind(b1.Embedding, V=64, H=32)
    table = _rand(16, 64 * 32)
    lt = L.lift_words(Array(64, Array(32, BF16)), table)
    _, _, h0 = _runl(L.lift_def(EMB), [[5], lt])
    poisoned = list(lt)
    x = _noncanonical(16)
    poisoned[9 * 32 + 4] = x  # row 9 (unselected), column 4
    _, t, h1 = _runl(L.lift_def(EMB), [[5], poisoned])
    assert h1[4] == INV16 and h1[4] != x and all(h1[j] == h0[j] for j in range(32) if j != 4)


def test_a3_2_state_ops_random_noncanonical_word_in_the_kv_cache_poisons_its_cone_only():
    """State ops (`Lifted[StepBody]`, the K/V cache append + attention read): a random noncanonical word in one cache leaf makes every
    output that READS it `INV_w` (the token, the deeper layers' new rows); the new rows that do not read it (layer 0's K/V, a function of
    the token and the weights alone) stay canonical Active and bit-equal to the honest run.  The word itself appears nowhere in the
    outputs (never passed through as if it were a value)."""
    T = LP + 3
    STEP = bind(L.StepBody, C=TINY, T=T, GEMM=b1.GEMM_AMPERE, ATTENTION=b1.ATTENTION_FA2, NORM=b1.norm_bundle(TINY))
    _, w = _inputs()
    kv_t = Array(TINY["NL"], b1.kv_type(TINY, T - 1))
    kc, vc = _rand(16, kv_t.leaves), _rand(16, kv_t.leaves)
    tok = 5
    lsp = L.lift_def(STEP)
    ins = [[tok], L.lift_words(kv_t, kc), L.lift_words(kv_t, vc), L.lift_words(weights_type(TINY), list(w))]
    prog, t0, y0 = _runl(lsp, ins)
    assert _all_words_canonical(prog, t0)  # A3.3 on the honest run, gate for gate
    n_row = TINY["KVH"] * TINY["D"]  # leaves of one layer's new row
    x = _noncanonical(16)
    bad = [list(v) for v in ins]
    bad[1][0] = x  # layer 0, position 0, head 0, dim 0 of the K cache
    _, t1, y1 = _runl(lsp, bad)
    assert y1[0] == INV32                                                   # the token reads the whole cache through attention
    k_new1 = y1[1:1 + TINY["NL"] * n_row]
    k_new0 = y0[1:1 + TINY["NL"] * n_row]
    assert k_new1[:n_row] == k_new0[:n_row] and all(L.word_class(16, v) == "active" for v in k_new1[:n_row])  # layer 0's new K: untouched
    assert x not in y1 and all(v in (INV16, INV32) or L.valid(16, v) for v in y1)  # never the raw word; INV or canonical
    if TINY["NL"] > 1:
        assert all(v == INV16 for v in k_new1[n_row:2 * n_row])              # layer 1's new K reads layer 0's attention output: INV
    # ⊥ in the same leaf: strict ⊥, not INV — the two inactive-looking words are NOT the same word (A1)
    bot = [list(v) for v in ins]
    bot[1][0] = BOT16
    _, _, yb = _runl(lsp, bot)
    assert yb[0] == BOT32 and yb[0] != y1[0]


def test_a3_2_activity_primitives_on_random_noncanonical_words():
    ia, sl = L.l_is_active(32), L.l_step_live(2)
    for _ in range(6):
        x = _noncanonical(32)
        assert ia.evaluate(x) == 0b11                       # INV_1, never Active(0)/Active(1)
        assert sl.evaluate(x, 7, 9) == 0b11 and sl.evaluate(42, x, 9) == 0b11  # in the token OR in the EOS set
    assert ia.evaluate(BOT32) == 0 and sl.evaluate(BOT32, 7, 9) == 0  # ⊥ is not noncanonical: Active(0)


# ---------------------------------------------------------------------------------------------------------
# A3.3 — honest computation emits canonical values (measured), A3.4 — the commitment layer never normalizes, A3.5 — boundaries
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_a3_3_honest_lifted_serving_program_emits_only_canonical_words_gate_for_gate():
    prompt, w = _inputs()
    eos = (int(prompt[0]) + 1,)  # some id; the transcript of the honest run decides where ⊥ starts
    prog = Program(L.lserve(TINY, LP, D_MAX, eos, lag=0))
    t = ref.materialize(prog, [L.lift_words(Array(LP, I32), list(prompt)), L.lift_words(weights_type(TINY), list(w))])
    assert _all_words_canonical(prog, t)
    outs = [t[g] for g in prog.circuit.root_outputs()]
    assert all(L.word_class(32, v) in ("active", "bot") for v in outs)  # theorem A3.3 on the interface too


def test_a3_4_commit_leaf_commits_the_raw_word_and_names_it_never_normalizes():
    for w in (16, 32):
        for x, cls in ((7, "active"), (L.enc_bot(w), "bot"), (L.inv(w), "noncanonical"), (_noncanonical(w), "noncanonical")):
            c = L.commit_leaf(w, x)
            assert c["class"] == cls and c["word"] == x and c["normalized"] is False and c["lifted_width"] == w + 1
            assert c["words"] == L.pack_leaf(w, x) and L.unpack_leaf(w, c["words"]) == x  # the raw bitstring, round trip
            assert len(c["words"]) == L.leaf_words(w)
        # two DIFFERENT noncanonical words commit to DIFFERENT word lists (distinct raw bitstrings stay distinct)
        a, b = _noncanonical(w), _noncanonical(w)
        while b == a:
            b = _noncanonical(w)
        assert L.commit_leaf(w, a)["words"] != L.commit_leaf(w, b)["words"] != L.commit_leaf(w, L.inv(w))["words"]
        # ... and none of them is what ⊥ or the payload alone would commit to
        assert L.commit_leaf(w, a)["words"] not in (L.pack_leaf(w, L.enc_bot(w)), L.pack_leaf(w, L.payload(w, a)))
    # a 33-bit noncanonical word needs two machine words; a one-word layout is refused by name, never read as Active
    x = _noncanonical(32)
    assert len(L.commit_leaf(32, x)["words"]) == 2
    with pytest.raises(L.LiftedLayoutError):
        L.unpack_leaf(32, L.commit_leaf(32, x)["words"][:1])


def test_a3_5_boundaries_reject_random_noncanonical_words_by_name():
    for w in (16, 32):
        for _ in range(4):
            with pytest.raises(L.BoundaryFailure, match="neither Active nor"):
                L.check_boundary(w, _noncanonical(w), what="committed token")
        assert L.check_boundary(w, L.enc_bot(w)) == L.enc_bot(w)  # ⊥ passes an unprescribed boundary
        with pytest.raises(L.BoundaryFailure, match="prescribed input must be Active"):
            L.check_boundary(w, L.enc_bot(w), prescribed=True)


# ---------------------------------------------------------------------------------------------------------
# A4 / A5 — true widened widths: the unchanged validator at the lifted parameter; the lifted use document; G8 refuses it by name
# ---------------------------------------------------------------------------------------------------------


def test_a4_true_widths_admissibility_is_the_unchanged_validator_at_w_max_lifted():
    from verity.ml.circuits import partition as PT
    TSEL = bind(b1.TokenSelect, V=64)  # 32-bit token / index arithmetic: lifted leaves are 33 bits
    P0, P1 = Program(TSEL), Program(L.lift_def(TSEL))
    v0 = PT.validate_vu_partition(P0, f"default({L.W_MAX})", L.W_MAX)
    assert v0.status == "ok" and v0.width.max_seen <= 32
    # the lifted Program at the ACTIVE-REFERENCE parameter: 33-bit gates -> not ready / width violated, BY DESIGN (A4.2: nothing is split)
    v_ref = PT.validate_vu_partition(P1, f"default({L.W_MAX})", L.W_MAX)
    assert v_ref.status in ("fail", "not-ready") and not v_ref.readiness_32.ok and any(x.w == 33 for x in v_ref.readiness_32.violators)
    # the lifted Program at the LIFTED parameter with the lifted query: the same validator, admissible, real widths reported
    v1 = PT.validate_vu_partition(P1, f"default({L.W_MAX_LIFTED})", L.W_MAX_LIFTED)
    assert v1.status == "ok" and v1.width.status == "ok" and v1.width.w_max == 64 and v1.readiness_32.ok
    assert v1.width.max_seen is not None and v1.width.max_seen > 32  # a true lifted width above 32 was ADMITTED, not hidden
    assert v1.width.max_seen == v0.width.max_seen + (v1.width.max_seen - v0.width.max_seen) and v1.width.max_seen <= 2 * v0.width.max_seen
    # the helper wraps exactly that: the verdict at W_MAX_LIFTED is the lifted verdict; the 32 verdict is information beside it
    from verity.ml.circuits.query_ast import evaluate
    fam = evaluate(P1, f"default({L.W_MAX_LIFTED})")
    adm = L.lifted_width_admissibility(P1, fam)
    assert adm["ok"] is True and adm["kind"] == "vu-partition-lifted" and adm["rules_version"] == "lifted-v1" and adm["w_max_lifted"] == 64
    assert adm["verdict"]["w_max"] == 64 and adm["verdict"]["max_seen"] == v1.width.max_seen
    assert adm["active_reference_rule_verdict"]["w_max"] == 32 and adm["active_reference_rule_verdict"]["status"] != "ok"
    # A4.4: max_seen (the validator's) and the n_out histogram (the boundary module's), complete over the family
    ws = adm["width_summary"]
    assert ws["max_seen"] == v1.width.max_seen and ws["complete"] is True and ws["sets_visited"] == v1.count and ws["sets_unsupported"] == 0
    hist = {int(k): v for k, v in ws["n_out_histogram"].items()}
    assert sum(hist.values()) == v1.count and ws["max_n_out"] == max(hist) >= 1
    # on a lifted Program n_out(S) is exactly the tag bits the set grew by: the widest set's lifted width == its payload width + its n_out
    assert ws["max_seen"] <= 33 * ws["max_n_out"]
    h_small = L.n_out_histogram(fam, max_sets=3)
    assert h_small["complete"] is False and h_small["sets_visited"] == 3
    # the four A4.4 / A7(iii) names on EVERY summary -- w_out_lifted_max (the boundary module's Σ(w+1) per set, == the
    # validator's max_seen on a fully certified family), n_out_max, growth_max (= n_out_max: one tag bit per escaping leaf), the histogram;
    # w_out_payload_max beside them as information; readiness at 32 (not ok: 33-bit gates, by design) and at w_max_lifted (ok)
    assert all(k in ws for k in L.WIDTH_SUMMARY_FIELDS) and ws["rules_version"] == "lifted-v1"
    assert ws["w_out_lifted_max"] == ws["max_seen"] and ws["max_seen_equals_w_out_lifted_max"] is True
    assert ws["n_out_max"] == ws["growth_max"] == ws["max_n_out"] >= 1
    assert ws["w_out_payload_max"] < ws["w_out_lifted_max"] and ws["w_out_payload_max"] <= 32
    assert ws["readiness_at_32"]["ok"] is False and ws["readiness_at_32"]["gates_over"] >= 1 and ws["readiness_at_32"]["max_w"] == 33
    assert ws["readiness_at_w_max_lifted"]["ok"] is True and ws["readiness_at_w_max_lifted"]["gates_over"] == 0
    h_prog = L.n_out_histogram(fam, program=P1)
    assert h_prog["w_out_lifted_max"] == ws["w_out_lifted_max"] and L.n_out_histogram(fam)["w_out_lifted_max"] is None  # no program: no widths, named None
    # the STRUCTURAL bound by member class (real bodies: 10^9 sets, the exact walk is bounded): complete, covers every set, and dominates the
    # exact walk -- a BOUND, not the exact count: it counts a member's distinct returned leaves whether or not a gate reads them, so the tiny
    # body's two unread sets (exact n_out 0) are bounded by 1; the exact histogram has {0: 2}, the bound has none
    sb = L.n_out_structural_bound(P1, fam)
    assert sb["available"] is True and sb["complete"] is True and sb["sets"] == v1.count and sb["classes"] >= 1
    assert sb["n_out_bound_max"] == sb["growth_max_bound"] >= ws["n_out_max"]  # a bound is never below the exact maximum
    hist_b = {int(k): v for k, v in sb["n_out_bound_histogram"].items()}
    assert sum(hist_b.values()) == sum(hist.values()) == v1.count and min(hist_b) >= 1 and 0 in hist
    assert sum(k * v for k, v in hist_b.items()) >= sum(k * v for k, v in hist.items())  # total tag bits bounded from above
    assert hist_b == {1: v1.count} and hist == {0: 2, 1: v1.count - 2}
    # the summary carries it only when the exact walk was bounded (complete False); complete walks carry None (nothing to add)
    assert ws["n_out_structural_bound"] is None
    ws_b = L.width_summary_a44(P1, fam, v1.width, histogram_max_sets=2)
    assert ws_b["complete"] is False and ws_b["sets_visited"] == 2 and ws_b["n_out_structural_bound"]["complete"] is True
    assert ws_b["n_out_structural_bound"]["sets"] == v1.count and ws_b["n_out_structural_bound"]["n_out_bound_max"] >= ws_b["n_out_max"]


def test_a5_lifted_use_document_carries_its_own_kind_and_rules_version_and_gate_g8_refuses_it_by_name(tmp_path):
    from verity.ml.circuits import partition as PT
    from verity_vllm.query import query_artifact as QA
    from verity_vllm.check import gates as G
    TSEL = bind(b1.TokenSelect, V=64)
    P1 = Program(L.lift_def(TSEL))
    q = f"default({L.W_MAX_LIFTED})"
    v1 = PT.validate_vu_partition(P1, q, L.W_MAX_LIFTED)
    qdoc, use = lifted_records.lifted_vu_use_document(q, P1, v1)
    assert use["schema"] == "verity-ir/query-use/v1" and use["use"]["kind"] == L.USE_KIND_LIFTED == "vu-partition-lifted"
    assert use["use"]["w_max"] == 64 and use["use"]["rules_version"] == "lifted-v1" and use["use"]["w_max_active_reference"] == 32
    assert use["verdict"]["status"] == "ok" and "kind" not in qdoc and "w_max" not in qdoc  # the query document is unchanged in shape
    assert use["width_summary"]["max_seen"] == v1.width.max_seen > 32 and use["width_summary"]["complete"] is True
    assert sum(use["width_summary"]["n_out_histogram"].values()) == v1.count
    assert QA.check_query_document(qdoc) == [] and QA.check_use_document(use, qdoc["program_digest"], qdoc) == []  # structurally a use doc
    # G8 (the active-reference gate) refuses the lifted document by NAME: a lifted verdict never reads as a panel-row G8 pass
    (tmp_path / "query_use.json").write_text(json.dumps(use))
    (tmp_path / "query.json").write_text(json.dumps(qdoc))
    r = G.gate_G8(G.RunArtifacts(tmp_path))
    assert r.status == "fail" and any("use.kind" in p and "vu-partition-lifted" in p for p in r.detail.get("problems", []) or [r.reason or ""])
    assert G.QUERY_USE_KIND == "vu-partition" != L.USE_KIND_LIFTED


# ---------------------------------------------------------------------------------------------------------
# A2 — one lag rule (GP-01's), two readers (attestation / declaration)
# ---------------------------------------------------------------------------------------------------------


def test_a2_lag_from_header_is_gp01s_rule_on_the_attested_engine_block():
    for knob, want in ((True, 1), (False, 0)):
        hdr = {"$e": "RunHeader", "profile": {"engine_kwargs": {"async_scheduling": knob}}}
        assert lifted_records.lag_from_header(hdr) == (want, lifted_records.LAG_FIELD) == (eos_lag_rows_declared(hdr["profile"]["engine_kwargs"]), lifted_records.LAG_FIELD)
    for bad in (None, 1, 0, "yes"):  # GP-01 returns None for these: refused by name, never defaulted
        hdr = {"$e": "RunHeader", "profile": {"engine_kwargs": {"async_scheduling": bad}}}
        assert eos_lag_rows_declared(hdr["profile"]["engine_kwargs"]) is None
        with pytest.raises(ValueError, match="not the resolved boolean"):
            lifted_records.lag_from_header(hdr)
    with pytest.raises(ValueError, match="not attested"):
        lifted_records.lag_from_header({"$e": "RunHeader", "profile": {"engine_kwargs": {}}})
    assert L.DP_PAD_02.endswith("RULED R17-1")


def test_a2_declared_lag_of_row_reads_the_builds_declaration_through_the_same_rule():
    # `lag.value` and `stop_rule.lag` first, GP-01's older `stop_rule.eos_lag_rows` as fallback; a bare boolean knob goes through the
    # rule; nothing defaults
    assert lifted_records.declared_lag_of_row({"lag": {"value": 1}, "stop_rule": {"eos_lag_rows": 1}}) == (1, "lag.value")
    assert lifted_records.declared_lag_of_row({"stop_rule": {"lag": 0, "eos_lag_rows": 0}}) == (0, "stop_rule.lag")
    assert lifted_records.declared_lag_of_row({"stop_rule": {"eos_lag_rows": 1}}) == (1, "stop_rule.eos_lag_rows")
    assert lifted_records.declared_lag_of_row({"stop_rule": {"eos_lag_rows": eos_lag_rows_declared({"async_scheduling": True})}}) == (1, "stop_rule.eos_lag_rows")
    assert lifted_records.declared_lag_of_row({"lag": {"value": False}}) == (0, "lag.value")  # the knob, through the one rule
    assert lifted_records.declared_lag_of_row({"stop_rule": {"eos_token_ids": [1]}}) == (None, None)  # a Build from before the lag was declared: NOTHING
    assert lifted_records.declared_lag_of_row({"stop_rule": {"eos_lag_rows": None}}) == (None, None)  # GP-01's None (undeclared engine): nothing
    assert lifted_records.declared_lag_of_row(None) == (None, None) and lifted_records.declared_lag_of_row({}) == (None, None)
    with pytest.raises(ValueError, match="must be declared as one of"):
        lifted_records.declared_lag_of_row({"lag": {"value": 2}})
    with pytest.raises(ValueError, match="disagrees with itself"):  # lag.value == stop_rule.lag == stop_rule.eos_lag_rows, or refused
        lifted_records.declared_lag_of_row({"lag": {"value": 1}, "stop_rule": {"lag": 0}})
    assert lifted_records.declared_lag_of_row({"lag": {"value": 1}, "stop_rule": {"lag": 1, "eos_lag_rows": 1}}) == (1, "lag.value")
    assert lifted_records.LAG_DECLARED_FIELDS == ("lag.value", "stop_rule.lag", "stop_rule.eos_lag_rows")
    # the static lives in the Program's spec id (`workload_fn` = Workload_v1{...,lag=n}), NOT in a bindings dict of the Root
    # wrapper; it is the digest-bound source, read first, and a sidecar edited against it is refused by name
    fn1 = 'Workload_v1{requests=[{"dict":{"arrive_step":0,"cap":32,"fn":{"fn":"WorkloadRequest_v1{program_fn=\\"Derived[x]{lag=7}\\"}"}}}],shared=0,lag=1}'
    assert lifted_records.declared_lag_of_row({"workload_fn": fn1}) == (1, "workload_fn:lag=")                       # the digest's static alone
    assert lifted_records.declared_lag_of_row({"workload_fn": fn1, "lag": {"value": 1}, "stop_rule": {"lag": 1, "eos_lag_rows": 1}}) == (1, "workload_fn:lag=")
    with pytest.raises(ValueError, match="disagrees with itself"):                                      # sidecar 0, digest 1
        lifted_records.declared_lag_of_row({"workload_fn": fn1, "lag": {"value": 0}, "stop_rule": {"lag": 0, "eos_lag_rows": 0}})
    fn0 = fn1[: -len("lag=1}")] + "lag=0}"
    assert lifted_records.declared_lag_of_row({"workload_fn": fn0}) == (0, "workload_fn:lag=")
    assert lifted_records.declared_lag_of_row({"workload_fn": "Workload_v1{requests=[],shared=0}"}) == (None, None)  # an id without the lag static declares NOTHING
    assert lifted_records.declared_lag_of_row({"workload_fn": "Serve_v2{STEPS=31}"}) == (None, None)                 # not a workload id: nothing
    with pytest.raises(ValueError, match="must be declared as one of"):
        lifted_records.declared_lag_of_row({"workload_fn": "Workload_v1{requests=[],shared=0,lag=2}"})
    # the recorded workload artefacts (Builds from before the lag was declared) declare nothing through EVERY source — never a default
    import glob as _glob
    for p in _glob.glob("out/gen/sweep/evidence/pad/*/*/build_workload/workload_program.json"):
        wp = json.load(open(p))
        assert wp["workload_fn"].startswith("Workload_v1{") and lifted_records.declared_lag_of_row(wp) == (None, None), p
    # the compose passes the lag as the Workload_v1 static exactly when the tree's WorkloadProgram takes one; inspected, not guessed
    import inspect
    from verity_vllm.program import workload as W
    takes_lag = "lag" in inspect.signature(W.WorkloadProgram.__init__).parameters
    assert callable(lifted_compose.workload_program) and isinstance(takes_lag, bool)


def test_a2_the_lifted_program_is_a_function_of_the_declared_lag():
    """A different LAG is a different Program (id, gate count) — the value is a static, never read from the record."""
    eos = (15,)
    p0, p1 = L.lserve(TINY, LP, D_MAX, eos, lag=0), L.lserve(TINY, LP, D_MAX, eos, lag=1)
    assert p0.id != p1.id and "LAG" in p0.id
    assert L.guard_gate_count(D_MAX, len(eos), 0) - L.guard_gate_count(D_MAX, len(eos), 1) == 2  # step 1 unguarded under LAG=1
    with pytest.raises(ValueError, match="must be declared"):
        L.lserve(TINY, LP, D_MAX, eos, lag=2)
