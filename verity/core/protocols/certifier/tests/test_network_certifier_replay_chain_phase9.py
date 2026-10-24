"""Red-team (REDTEAM-ADVICE, phase 9 target 2) against ADV-SPEC's replay state chain (`replay_chain`), the lead's
splice/skip/repeat question: can a prover substitute a different committed state between steps, skip a step, or repeat
one, and does a window leak decision bits across its boundary?

Findings, each confirmed executably below:

1. Cross-window state splice is CLOSED. `check_chain`'s C-carry ties window w's first state row (and window row) to
   window w-1's last, byte for byte, under the reused salt, so substituting any other committed row -- even a
   well-formed one under a fresh salt -- is caught.
2. A window OWNS its decisions: `advance` refuses a window whose last step leaves a decision unread, and `load` refuses
   a window that starts while the previous one left decisions or steps. Decisions cannot leak across the boundary
   (reinforces ENC-CODE's per-window charge under A5).
3. `advance`'s `used` and `taken` are a FREE WITNESS within the bound: on a non-last step the chain accepts used 0..A
   (and taken 0..valid-arrivals), so a prover can re-read decisions/arrivals or advance the heads less than the step
   really consumed. The tie of `used`/`taken` to ReplayStep_v1's outputs is the (unbuilt) compile unit's; this piece
   only bounds them.
"""
from __future__ import annotations

from verity.core.protocols.certifier import replay_chain as RC
from verity.core.protocols.sampled_proofs.one_stage import registered as G

SMALL = RC.Shape(R=3, W=2, A=3, L=40, NB=20, IW=4, Q=5, D=6, P=3, K=3)
PROGRAM, REGISTRATION = "replay-chain-rt", "ab" * 64


def _rows(S, first, n):
    """A window's P+2 state rows starting at `first`, the rest arbitrary valid rows."""
    rest = [RC.Row(((7, 1, 1, 0, 0),), (), 5, 1, 1, max(0, n - 1 - k)) for k in range(S.P + 1)]
    return [first] + rest


def test_cross_window_state_splice_is_closed_by_the_carry():
    """Finding 1. A two-window session commits clean; substituting window 1's first state row with a different, equally
    well-formed committed row (fresh salt) is caught as C-carry."""
    S = SMALL
    win0 = RC.Window(((4, 3, 3), (5, 2, 1)), (1, 0, 1))
    win1 = RC.Window(((6, 2, 2),), (0, 1))
    rows0 = _rows(S, RC.empty(S), 2)
    v0 = RC.commit_window(PROGRAM, rows0, win0, S, REGISTRATION)
    rows1 = _rows(S, rows0[-1], 1)                       # window 1 starts in window 0's last state row (the carry)
    v1 = RC.commit_window(PROGRAM, rows1, win1, S, REGISTRATION, carry=v0)
    ops = [RC.openings(v0), RC.openings(v1)]
    assert RC.check_chain(ops, PROGRAM, S) is None

    # splice: a DIFFERENT but well-formed state row committed under a fresh salt in window 1's slot 0.
    other = RC.Row(((9, 2, 2, 1, 3),), (), 7, 0, 0, 1)
    spliced = G.Registered.commit(PROGRAM, RC.VALUE, [RC.row(RC.encode(other, S))] + list(v1.state.rows[1:]),
                                  word_bits=16, schema=RC.SCHEMA, produced=REGISTRATION)
    (s1, w1) = RC.openings(v1)
    bad = [RC.openings(v0), (list(spliced.bcs), w1)]
    assert RC.check_chain(bad, PROGRAM, S).startswith("C-carry")


def test_a_window_owns_its_decisions():
    """Finding 2. The last step must read the window's remaining decisions (advance), and a window cannot start while
    the previous left decisions or steps (load). So decisions do not carry across the boundary."""
    S = SMALL
    win = RC.Window(((4, 3, 3), (5, 2, 1)), (1, 0))      # two decisions, read by the window's single last step
    last = RC.Row(((1, 5, 3, 1, 6),), (), 11, 1, 1, 1)   # steps = 1: this is the last step, dh = 1 of 2 read
    arr, bits = RC.front(S, win, last)
    # the last step leaving a decision (used = 0) is refused; reading the rest (used = 1 -> dh = 2) is accepted.
    assert RC.advance(S, win, last, True, RC.Row(), arr, bits, 0, 0) is None
    done = RC.advance(S, win, last, True, RC.Row(), arr, bits, 0, 1)
    assert done is not None and done.dh == 2

    # and load refuses a new window while the previous left a decision or a step unread.
    last_win = RC.Window(((3, 1, 1), (4, 3, 3)), (1, 1))
    left_a_decision = RC.Row((), (), 11, 1, 1, 0)        # dh = 1 != len(decisions) = 2
    left_a_step = RC.Row((), (), 11, 2, 2, 1)            # steps = 1 != 0
    assert RC.load(S, left_a_decision, last_win, [], (1,), 1) is None
    assert RC.load(S, left_a_step, last_win, [], (1,), 1) is None
    ok = RC.load(S, RC.Row((), (), 11, 2, 2, 0), last_win, [(6, 2, 2)], (1,), 1)
    assert ok is not None


def test_advance_used_and_taken_are_free_witness_within_the_bound():
    """Finding 3. On a non-last step the chain accepts a whole band of `used` and `taken`, so a prover can advance the
    decision/arrival heads less than the step really consumed (re-reading them next step). Pinning them to
    ReplayStep_v1's outputs is the compile unit's."""
    S = SMALL
    win = RC.Window(((4, 3, 3), (5, 2, 1)), (1, 0, 1, 1, 0, 1))   # 2 arrivals, D = 6 decisions
    cur = RC.Row(((1, 5, 3, 1, 6),), (), 11, 0, 0, 3)             # steps = 3: not the last step
    arr, bits = RC.front(S, win, cur)
    stepped = RC.Row(((1, 5, 3, 2, 7),), (), 10)

    used_ok, taken_ok = [], []
    for used in range(S.A + 2):
        ref = RC.advance(S, win, cur, True, stepped, arr, bits, 0, used)
        if ref is not None:
            nxt = RC.encode(ref, S)
            assert RC.evaluate_advance(S, [RC.advance_arguments(S, win, cur, True, stepped, arr, bits, 0, used, nxt)]) == [(1, 1)]
            used_ok.append(used)
    for taken in range(S.A + 2):
        ref = RC.advance(S, win, cur, True, stepped, arr, bits, taken, 0)
        if ref is not None:
            taken_ok.append(taken)
    # used 0..A all accepted (used = 0 re-reads this step's decisions next step); taken 0..valid-arrivals (= 2).
    assert used_ok == [0, 1, 2, 3], used_ok
    assert taken_ok == [0, 1, 2], taken_ok
    # used = 0 leaves the decision head where it was: the next step reads the same decisions.
    stay = RC.advance(S, win, cur, True, stepped, arr, bits, 0, 0)
    assert stay is not None and stay.dh == cur.dh and stay.qh == cur.qh
    print(f"\n[replay-chain] advance free band on a non-last step: used accepted {used_ok}, taken accepted {taken_ok}")
