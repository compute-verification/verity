"""Adversarial tests for stage DENSE B1, Program side — **padded baseline (experimental)**.  Attack A5 (the 32-bit cap dropping
the lifted bit) on the leaf layout, T-1 refusals, the exact-count arithmetic against the Program's own arity for EVERY `a_r`, and
the deletion rule as a function — on the TINY shape (`test_lifted_dense_b1` measures the real B0 shape).  Skips on a tree without `verity.ml.definitions.registry.lifted`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

L = pytest.importorskip("verity.ml.definitions.registry.lifted")
pytest.importorskip("verity.ml.definitions.registry.lifted", reason="the lifted registry").leaf_words  # DENSE B1 additions present?

from verity.ml.circuits import Program  # noqa: E402
from verity.ml.definitions.registry import b1  # noqa: E402
from tests.program.test_serve3 import LP, TINY  # noqa: E402

RNG = np.random.default_rng(20260920)
CAP = 9  # TINY: D_MAX = 8 decode steps = cap − 1


def _rand(w, n):
    return [int(x) for x in RNG.integers(0, 1 << w, n, dtype=np.uint64)]


# ---------------------------------------------------------------------------------------------------------
# A5 — the leaf layout: word 0 IS the unlifted committed word; the tag lives in its own bit, never dropped
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", [1, 2, 16, 31, 32])
def test_word_zero_of_an_active_leaf_is_the_unlifted_committed_word(w):
    """Payload agreement at the LAYOUT level: an Active leaf's first word equals the value the unlifted verifier commits, so the
    lifted store extends the unlifted one instead of re-encoding it; ⊥ and INV differ from every Active word in the tag word."""
    for v in [0, 1, (1 << w) - 1, 1 << (w - 1)] + _rand(w, 32):
        packed = L.pack_leaf(w, L.enc_active(w, v))
        assert packed[0] == v if w == 32 else packed == [v], (w, v)
        if w == 32:
            assert packed == [v, 0]
    bot, inv = L.pack_leaf(w, L.enc_bot(w)), L.pack_leaf(w, L.inv(w))
    assert bot != inv and L.unpack_leaf(w, bot) == L.enc_bot(w) and L.unpack_leaf(w, inv) == L.inv(w)  # INV is not canonicalised
    if w == 32:
        assert bot == [0, 1] and inv == [1, 1]
    else:
        assert bot == [1 << w] and inv == [(1 << w) + 1]


def test_a_tag_word_with_a_nonzero_payload_is_garbage_and_the_boundary_refuses_it():
    """A5/A4 combined: `[v, 1]` with `v ≠ 0` is a `1‖nonzero` word — unpack is total (returns it), the boundary names it; `[0, 1]` is
    ⊥; `[0, 2]`/`[0, 3]` set bits above the 33rd and are refused by the layout itself."""
    for v in [1, 7, 0xFFFFFFFF]:
        x = L.unpack_leaf(32, [v, 1])
        assert not L.valid(32, x) and not L.is_bot(32, x) and not L.is_active(32, x)
        with pytest.raises(L.BoundaryFailure, match="invalid representation"):
            L.check_boundary(32, x, what="committed leaf")
    assert L.unpack_leaf(32, [0, 1]) == L.enc_bot(32)
    for hi in (2, 3, 0x80000000):
        with pytest.raises(L.LiftedLayoutError, match="above the lifted width"):
            L.unpack_leaf(32, [0, hi])


def test_a_one_word_per_leaf_reader_misreads_a_mixed_width_lifted_store_and_the_size_check_catches_it():
    """The attack in full: a store of leaves of widths [32, 16, 1, 32, 32] holding ⊥, Active, Active(1), Active(max), INV.  The
    lifted reader (offsets from `leaf_words`) round-trips exactly.  A legacy one-word-per-leaf reader reads the ⊥ as Active(0) and
    then every later leaf from the wrong offset; the ONLY thing that stops it is the size check `len(words) == Σ leaf_words(w)` —
    which is therefore a required field of a lifted binding map (a condition for the record phase)."""
    widths = [32, 16, 1, 32, 32]
    leaves = [L.enc_bot(32), L.enc_active(16, 0xBEEF), L.enc_active(1, 1), L.enc_active(32, (1 << 32) - 1), L.inv(32)]
    flat = [wd for w, x in zip(widths, leaves) for wd in L.pack_leaf(w, x)]
    assert len(flat) == sum(L.leaf_words(w) for w in widths) == 8 != len(widths)
    # lifted reader
    got, off = [], 0
    for w in widths:
        n = L.leaf_words(w)
        got.append(L.unpack_leaf(w, flat[off:off + n]))
        off += n
    assert got == leaves and off == len(flat)
    # legacy reader: one word per leaf, no tag
    legacy = flat[:len(widths)]
    assert legacy[0] == 0 and L.is_active(32, legacy[0])  # ⊥ read as Active(0): the deletion of a request disguised as a value
    assert legacy[1] == 1  # the tag word of leaf 0 read as leaf 1's value
    with pytest.raises(L.LiftedLayoutError):
        L.unpack_leaf(32, [legacy[0]])  # the lifted reader refuses the one-word layout by name
    assert len(legacy) != sum(L.leaf_words(w) for w in widths)  # the size check that a lifted binding map must carry


# ---------------------------------------------------------------------------------------------------------
# T-1 — built from the manifest only; every refusal by name; the cap is the population
# ---------------------------------------------------------------------------------------------------------


def test_lifted_request_refusals_and_population_on_tiny():
    sp = L.lifted_request(TINY, LP, CAP, (17,), lag=0)
    steps_key = "STEPS" if "STEPS" in sp.bindings else "D_MAX"  # STEPS (= cap-1, as Serve_v2)
    assert sp.bindings[steps_key] == CAP - 1 and sp.bindings["LP"] == LP and sp.bindings["EOS"] == (17,) and sp.bindings["GUARD"] == "in"
    assert "D_MAX" not in sp.bindings or steps_key == "D_MAX"  # never both names for one static
    for bad_T in (0, CAP - 2, CAP, 100):
        with pytest.raises(ValueError, match="never from a served length"):
            L.lifted_request(TINY, LP, CAP, (17,), lag=0, T=bad_T)
    # the manifest-validity refusals fire EAGERLY at bind — and again at Program() for a direct bind(LServe)
    with pytest.raises(ValueError, match="non-empty declared EOS set"):
        L.lifted_request(TINY, LP, CAP, (), lag=0)  # an empty declared EOS set is not a declaration
    with pytest.raises(ValueError, match="MAX_POS"):
        L.lifted_request(TINY, LP, TINY["MAX_POS"] - LP + 2, (17,), lag=0)  # LP + STEPS > MAX_POS: not a request of this model
    with pytest.raises(ValueError, match="non-empty declared EOS set"):
        Program(L.bind(L.LServe, **{**sp.bindings, "EOS": ()}))  # the direct bind is still refused, at Program(), by the same name
    with pytest.raises((ValueError, L.BoundaryFailure)):
        L.lifted_request(TINY, LP, CAP, (L.enc_bot(32),), lag=0)  # ⊥ as a declared EOS id is not an id
    prog = Program(sp)
    outs = prog.circuit.root_outputs()
    assert len(outs) == CAP  # the event population IS the cap (the spec's R3), whatever any execution serves
    assert L.expected_bot_return_leaves(TINY, CAP, CAP, lag=0) == 0 and L.expected_bot_return_leaves(TINY, CAP, 1, lag=0) == (CAP - 1) * (1 + 2 * TINY["NL"] * TINY["KVH"] * TINY["D"])


def test_exact_count_arithmetic_equals_the_programs_step_arity_for_every_a_r():
    """T-3(ii) closed form vs the Program: for EVERY admissible `a_r ∈ [1, cap]`, the ⊥ leaf count is the sum of the return arities of
    the steps `s ≥ a_r` — and the arithmetic refuses `a_r` outside `[1, cap]` (no negative or over-cap 'activity')."""
    prog = Program(L.lifted_request(TINY, LP, CAP, (17,), lag=0))
    steps = {int(c.node.name.split("=")[1].split("/")[0]): c for c in prog.circuit.root_calls() if c.node.name and c.node.name.endswith("/step")}
    assert sorted(steps) == list(range(1, CAP))
    for a_r in range(1, CAP + 1):
        assert L.expected_bot_return_leaves(TINY, CAP, a_r, lag=0) == sum(len(steps[s].returns) for s in range(a_r, CAP)), a_r
    for bad in (0, -1, CAP + 1):
        with pytest.raises(ValueError):
            L.expected_bot_return_leaves(TINY, CAP, bad, lag=0)


# ---------------------------------------------------------------------------------------------------------
# the deletion rule as a FUNCTION of the lifted root body (what GP-01's comparison would apply)
# ---------------------------------------------------------------------------------------------------------


def _is_controller_row(nm: str) -> bool:
    return nm.startswith("slot=") and nm.endswith(("/emit", "/is_eos", "/out"))


def _delete_guard_rows(prog):
    """The deletion rule: drop the named guard rows (`eos[i]`, `bot32`, `s=<s>/live`, `s=<s>/tok_in`) and the Continuation_v1 controller rows
    (`slot=<t>/emit`, `slot=<t>/is_eos`, `slot=<t>/out`) and renumber the remaining root nodes; return [(name, gate_count)] of the
    body nodes under the survivors, in order."""
    out = []
    for c in prog.circuit.root_calls():
        nm = c.node.name or ""
        if nm.startswith("eos[") or nm in ("bot32", "enable/one") or nm.endswith("/live") or nm.endswith("/tok_in") or _is_controller_row(nm):
            continue
        if nm == "prefill" or nm.endswith("/step"):
            for k in c.children():
                if k.node.name:
                    out.append((k.node.name, k.hi - k.lo))
        else:
            out.append((nm, c.hi - c.lo))
    return out


def test_deletion_rule_recovers_the_unlifted_body_and_nothing_else_is_deletable():
    """After deleting exactly the guard rows, the lifted body's (name, gate-count) sequence equals the unlifted `Serve_v2` skeleton's;
    the deleted rows are exactly `2·D_MAX + |E| + 1` gates of the three guard ids; deleting any OTHER row breaks the equality."""
    D_MAX = CAP - 1
    p_u = Program(b1.serve_v2(TINY, LP, D_MAX))
    p_l = Program(L.lifted_request(TINY, LP, CAP, (17, 23), lag=0))
    u = [(c.node.name, c.hi - c.lo) for c in p_u.circuit.root_calls() if c.node.name]
    assert _delete_guard_rows(p_l) == u
    guard = [c for c in p_l.circuit.root_calls() if (c.node.name or "").startswith("eos[") or c.node.name in ("bot32", "enable/one")
             or (c.node.name or "").endswith(("/live", "/tok_in"))]
    ctrl = [c for c in p_l.circuit.root_calls() if _is_controller_row(c.node.name or "")]
    assert sum(c.hi - c.lo for c in guard) == 2 * D_MAX + 2 + 2 and sum(c.hi - c.lo for c in ctrl) == 3 * CAP  # 3 controller gates per slot
    assert sum(c.hi - c.lo for c in guard + ctrl) == p_l.gates - p_u.gates == L.continuation_gate_count(CAP, 2, 0)
    ids = {c.id.split("{")[0] for c in guard}
    assert ids == {"LStepLive2_v1", "LSelect32_v1", "LBot32_v1", "Const32[0x00000011]_v1", "Const32[0x00000017]_v1", "Const2[0x1]_v1"}  # + the root enable; the EOS set ordinary (G1)
    assert {c.id.split("{")[0] for c in ctrl} == {"Const2[0x1]_v1", "LEmitNext[H=1]_v1", "LIsEOS2_v1", "LSelect32_v1"}
    # deleting a non-guard row (a step) does not recover the unlifted body: the rule is exact, not 'delete until equal'
    fewer = [x for x in _delete_guard_rows(p_l) if not x[0].startswith(f"t={LP + 2}/")]
    assert fewer != u
