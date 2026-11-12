"""The falsification test of `tt-out-tile-pearl-c-h100-rev1` (`Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1S8`,
security/Definitions/Assumptions/PearlC.lean; `shape.conjectures` in security/lean-audit.json).

The conjecture is TT_OUT(1/400) per audit tile, rev1, at the H100 record, over W1 on the H100 with INT8: no program
outputs a good tile's words for less than (1 − g₀) of their credit, the credit (1 − ρ)·`creditDevRev1` per share. Each
test takes one of the census's known skip classes (`pearlc_census.py`), builds on census-formed tiles at the H100's atom
the cheapest program the class gives, checks that its words are the honest ones, and asserts it stays within the bound.

It catches what the Lean proofs can't: a change to rev1's credit or W_ref (`pearl_c.PearlC.credit_of`, `wref`, the Python
mirror of `creditDevRev1` and `wrefDevRev1`) that credits an operation a known skip removes, and a census tile on which
a known class beats the bound. A new skip class that the census finds belongs here as another test.
"""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from verity.protocols.pouw.prover.schemes import pearl_c as PC
from verity.protocols.pouw.prover.schemes.pearl_c_device import H100
from verity.protocols.pouw.verifier.protocol import Matmul, Shape

spec = importlib.util.spec_from_file_location("pouw_pearlc_census_ttout", Path(__file__).parents[1] / "pearlc_census.py")
C = importlib.util.module_from_spec(spec)
sys.modules["pouw_pearlc_census_ttout"] = C
spec.loader.exec_module(C)

#: TT_OUT's gap and rev1's cap share, both 1/400 (`TTOutTilePearlCDevRev1`, `pearlCProtocolDevRev1 … (1 / 400)`)
G0 = RHO = Fraction(1, 400)
#: the H100 record's atoms per promotion (`devH100`'s G)
G = H100.g
#: census tiles: families that reach the chain's skip classes, formed under §5 at the H100's atom
FAMILIES = ("gaussian", "spikes-first", "outliers-groupstart", "aligned-spikes-pm1-r444", "xcancel-spikes-r64",
            "xcancel-tail-f8")


def _tile(fam: str, k: int, rows: int = 4, cols: int = 4, run: int = 1, line_norm: float = 16.0):
    """A census tile's codes (rows·cols words, one row of A and one of B each) under §5's forming, and whether its operands
    pass the rules (noise floor and liveness), as `pearlc_census.cell` forms it."""
    rng = C._rng(run, fam, k, 1.0)
    FA = C.lines(k, C._rng(run, "FA", k))
    XA, XB = C.family(fam, rows, cols, k, FA, rng)
    FcA = C.enc(C.unit_lines(k, C._rng(run, "FA-lines", k)) * (line_norm / np.sqrt(32)))
    FcB = C.enc(C.unit_lines(k, C._rng(run, "FB-lines", k)) * (line_norm / np.sqrt(32)))
    a, _, _, floor_a, live_a = C.form_s5(XA, FcA, 1.0, line_norm, C._rng(run, fam, k, 1.0, "unit-A"))
    b, _, _, floor_b, live_b = C.form_s5(XB, FcB, 1.0, line_norm, C._rng(run, fam, k, 1.0, "unit-B"))
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return a[ii.ravel()], b[jj.ravel()], floor_a and live_a and floor_b and live_b


def _fires(credit: Fraction, cost: Fraction) -> bool:
    """TT_OUT's event for one program on one unit or tile: its credited work beats its cost by more than the gap."""
    return (1 - RHO) * credit > cost / (1 - G0)


def _skip_first(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The ticket words with the first promotion not executed: the first group's accumulator is the total."""
    T = A.shape[1] // 32
    total = None
    for g0 in range(0, T, G):
        acc = C.group_run(A, B, list(range(g0, min(g0 + G, T))))[0].astype(np.float32)
        total = acc if total is None else (total + acc).astype(np.float32)
    return total


def test_first_promotion_skip_is_outside_rev1s_credit():
    """The red team's first-promotion skip (rating D of `TTOutPearlC`): T_1 = RNE(+0 + S_1) is S_1, so the chain with the
    first promotion of every word not executed outputs every ticket word. rev1 takes that add out of W_ref and the credit
    (`first_adds`), so the program costs rev1's W_ref and its event never fires on the domain; under the credit before
    rev1 it fires, which is the falsification this test would have caught."""
    C.set_atom("hopper")
    for fam in FAMILIES:
        A, B, ok = _tile(fam, 1024)
        assert ok, fam
        assert (_skip_first(A, B) == C.group_sums(A, B, G)[1]).all(), fam
    sch = PC.PearlC(forming="v1", device=H100)
    ks = [k for k in (256, 1024, 2048, 4096, 8192, 16384, 32768, 65536) if k <= H100.k_max]
    fired_before = False
    for k in ks:
        for m, n in ((64, 64), (8192, 8192), (8192, 2048)):
            s = Shape(m, k, n)
            sch.check(Matmul(0, s, 0))
            cost = Fraction(sch.wref(s))
            assert sch.first_adds(s) == H100.prices.fadd * m * n > 0
            assert not _fires(Fraction(sch.credit_of(s)), cost), s
            fired_before |= _fires(Fraction(sch.credit_of(s) + sch.first_adds(s)), cost)
    assert fired_before


def test_cross_group_joint_skips_stay_within_the_gap():
    """The second hole, cross-group joint skips (`joint_cross_group`), which TT_OUT carries: sets of atoms from different
    promotion groups whose rounding effects cancel in a word's ticket. W1 charges a tensor-core instruction its whole
    fragment (`h100S8Frags`: 64 rows by at least 64 columns), so a program saves an atom only by skipping it for every word
    the instruction covers. With the products in hand (more than any program has before computing them), each word's
    greedy set (`greedy_set`) is intersected over the tile's 4 x 4 words, which over-approximates what one fragment's
    words share, the intersection replayed on every word, and its atoms outside each word's debit (`debit`'s flags) priced
    at a MAC each against the tile's rev1 credit: the event never fires. Priced word by word instead, the same sets do
    fire, so the test reaches the class and the bound rests on the fragment."""
    C.set_atom("hopper")
    sch = PC.PearlC(forming="v1", device=H100)
    k = 1024
    s = Shape(64, k, 64)
    per_word_credit = Fraction(sch.credit_of(s), s.m * s.n)
    per_word_wref = Fraction(sch.wref(s), s.m * s.n)
    fired_per_word = False
    for fam in FAMILIES:
        A, B, ok = _tile(fam, k)
        assert ok, fam
        P = A.shape[0]
        flags, _ = C.debit(A, B, G)
        honest = C.group_sums(A, B, G)[1]
        sets = [C.greedy_set(A[p:p + 1], B[p:p + 1], G) for p in range(P)]
        for p, S in enumerate(sets):
            beyond = 32 * sum(1 for t in S if not flags[p, t])
            fired_per_word |= _fires(per_word_credit, per_word_wref - beyond)
        common = set.intersection(*sets)
        if common:
            assert all(C.word_with_skip(A[p:p + 1], B[p:p + 1], common, G) == honest[p] for p in range(P)), fam
        saving = 32 * sum(1 for p in range(P) for t in common if not flags[p, t])
        assert not _fires(per_word_credit * P, per_word_wref * P - saving), (fam, sorted(common), saving)
    assert fired_per_word
