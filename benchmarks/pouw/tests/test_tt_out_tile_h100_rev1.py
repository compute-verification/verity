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


# ---- dead pairs: a reviewer's class, which the conjecture does not survive ------------------------------------------------
#
# Liveness lets k/64 entries of a row sit so far above its noise that no salt moves their code (a dead entry). Where a row
# of A and a row of B are both dead at a position, that product is the same for every salt, so a program holds it before
# the salt. The debit charges a dead entry's forming and nothing for its products, and P1's flags see only an atom whose
# skip in place leaves the word unchanged, which these atoms are not.

#: code values of the dead entries: each the centre of an E4M3 cell at least 16 wide on either side (448's cell is [432, ∞))
TOP, MID, LOW, HIGH = 448.0, 320.0, 288.0, 352.0
#: max|x| over ρ on a dead-pair row: α·max|x| = 448·400/404 = 443.6, in 448's cell, and ρ·α = 448/404 ≥ 1 (the noise floor)
SPIKE = 400.0
#: W1's prices on `W1Device.s8H100Best` (W1.lean): an INT32 instruction, and one code of a cast (64 an instruction, two codes)
INT32, CAST = 64, 32
#: an upper bound on the W1 cost of one noise line: 32 entries, each a byte extracted and looked up (its value and its
#: square), scaled, rounded to BF16, cast to E4M3, decoded, scaled by β and cast again (about 600 units an entry at the
#: prices above), and the squares summed for the norm's table entry. SHA-512 is free on `s8H100Best`.
LINE = 24_000


def _dead_pair_rows(side: str, rows: int, k: int, groups: int, rng: np.random.Generator) -> np.ndarray:
    """`rows` rows of FP32 values of one operand of the dead-pair family: a Gaussian background, and in each of the first
    `groups` promotion groups every position off the every-8th grid dead, 28 an atom, the same in every row. On A an atom's
    dead codes are 26 of 448, then 320 and 288; on B 13 of 448, 13 of −448, then 320 and −352. So the 28 dead products of an
    atom sum to 320·320 − 288·352 = 1,024 whatever the salt, while two of them are ±448², which sets the atom's adder."""
    assert 112 * groups * 64 <= k, "liveness admits k/64 dead entries a row"
    X = rng.standard_normal((rows, k))
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    alpha = C.Q_MAX / ((SPIKE + 4.0) * rho)
    vals = [TOP] * 26 + [MID, LOW] if side == "A" else [TOP] * 13 + [-TOP] * 13 + [MID, -HIGH]
    for atom in range(4 * groups):
        for l, v in zip([32 * atom + r for r in range(32) if r % 8], vals):
            X[:, l] = np.sign(v) * SPIKE * rho if abs(v) == TOP else v / alpha
    return X.astype(np.float32).astype(np.float64)


def _scheme_rules(X: np.ndarray) -> bool:
    """The scheme's own domain rules on every row (`PearlC.admissible`): the noise floor ρ·α ≥ 1 on the pinned FP32
    statistics, and liveness on the exact one."""
    for row in X.astype("<f4").view("<u4"):
        words = [int(w) for w in row]
        _, rho, alpha, _ = PC.s5_stats(words)
        if not (PC.P.f32_of_bits(rho) * PC.P.f32_of_bits(alpha) >= PC.P.SIGMA_MIN and PC.live_row(words)):
            return False
    return True


def _dead_pair_tile(k: int, groups: int, rows: int = 64, cols: int = 64, run: int = 1, line_norm: float = 16.0):
    """One audit tile of the family, formed under §5 at the H100's atom as `_tile` forms the census's: the P = rows·cols
    words' codes, the tile's rows of A (values and codes), and whether every row of both operands passes the rules."""
    XA = _dead_pair_rows("A", rows, k, groups, C._rng(run, "dead-pairs", k, "A"))
    XB = _dead_pair_rows("B", cols, k, groups, C._rng(run, "dead-pairs", k, "B"))
    FcA = C.enc(C.unit_lines(k, C._rng(run, "FA-lines", k)) * (line_norm / np.sqrt(32)))
    FcB = C.enc(C.unit_lines(k, C._rng(run, "FB-lines", k)) * (line_norm / np.sqrt(32)))
    a, _, _, floor_a, live_a = C.form_s5(XA, FcA, 1.0, line_norm, C._rng(run, "dead-pairs", k, "unit-A"))
    b, _, _, floor_b, live_b = C.form_s5(XB, FcB, 1.0, line_norm, C._rng(run, "dead-pairs", k, "unit-B"))
    ok = floor_a and live_a and floor_b and live_b and _scheme_rules(XA) and _scheme_rules(XB)
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return a[ii.ravel()], b[jj.ravel()], XA, a, FcA, ok


def _dead_pair_program(A: np.ndarray, B: np.ndarray, groups: int) -> tuple[np.ndarray, int]:
    """The ticket words of the W1 program, and the E4M3 atoms it runs per word.

    In an atom of the first `groups` groups the two ±448² products set the adder's unit at 8, so each of the atom's four
    live products is cut to a multiple of 8 on its own and the sum is exact: the atom adds 1,024 and those four multiples
    of 8 to its accumulator, exactly. So the total after the first `groups` groups is 4,096·`groups` plus the `16·groups`
    live products, each cut to a multiple of 8. The program computes that total with one fragment per 30 live products: the
    accumulator starts at the constant 4,096·`groups` (held before the salt), each atom's operand rows carry 448, 448 on A
    and 448, −448 on B (the same adder), then 30 of the live pairs, and the atoms chain. It runs every later group and
    promotion as the reference does, with that total as the running total."""
    P, k = A.shape
    live = np.concatenate([np.arange(128 * g, 128 * g + 128, 8) for g in range(groups)])
    top = C.enc(np.array([TOP]))[0]
    acc, atoms = np.full(P, 4096.0 * groups), 0
    for lo in range(0, len(live), 30):
        cols = live[lo:lo + 30]
        a, b = np.zeros((P, 32), dtype=np.uint8), np.zeros((P, 32), dtype=np.uint8)
        a[:, :2], b[:, 0], b[:, 1] = top, top, top | 0x80
        a[:, 2:2 + len(cols)], b[:, 2:2 + len(cols)] = A[:, cols], B[:, cols]
        acc, atoms = C.step(acc, a, b)[0], atoms + 1
    total = acc.astype(np.float32)
    for g0 in range(G * groups, k // 32, G):
        total = (total + C.group_run(A, B, list(range(g0, g0 + G)))[0].astype(np.float32)).astype(np.float32)
    return total, atoms + k // 32 - G * groups


def _dead_over_count(X: np.ndarray, groups: int, chunk: int = 1 << 16) -> int:
    """At least the salt-dead elements of the rows `X` (`SaltDead.saltDead`: the same code at every pair of flat lines, E's
    entries ±44 and F's ±11/4): the elements whose code is the same at 0, 16 and 32 negative products of the 32, three of
    the pairs the predicate ranges over."""
    rows, k = X.shape
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    alpha = C.Q_MAX / (np.abs(X).max(1) + 4 * rho)
    Cw = (alpha[:, None] * X).astype(np.float32).astype(np.float64).ravel()
    e = np.repeat(C.enc(rho * alpha * (np.sqrt(32) / 4096) * 44.0), k)
    same = np.ones(rows * k, dtype=bool)
    first = None
    for neg in (0, 16, 32):
        f = np.full(32, C.enc(np.array([2.75]))[0], dtype=np.uint8)
        f[:neg] |= 0x80
        codes = np.concatenate([C.enc(C.step(Cw[lo:lo + chunk], np.repeat(e[lo:lo + chunk, None], 32, 1),
                                             np.broadcast_to(f, (len(Cw[lo:lo + chunk]), 32)))[0])
                                for lo in range(0, rows * k, chunk)])
        first = codes if first is None else first
        same &= codes == first
    return int(same.sum())


def _dead_pair_cost(s: Shape, groups: int, atoms: int, units: int) -> Fraction:
    """An upper bound on the W1 cost, on `W1Device.s8H100Best`, of the program on `units` units of shape `s` that share
    one weight. Every instruction is priced; data movement, constants, SHA-512 and fetches are free there, and the program
    takes no branch. The activations are the adversary's own and fixed before the salt, so s, ρ, α, β and α·x are held.

    - chain: `atoms` E4M3 atoms a word, and one FP32 add for each group after the first `groups`;
    - fused A′·F_B: the reference's chain on 32 more columns with its promotions, and the first `groups` groups' atoms on
      fragments of their own, 64 columns wide for 32 used;
    - peel: its BF16 products, and the 64 casts of P_A a row;
    - forming: the noise atom and the cast on the live elements only (a dead element's code is held);
    - a noise line a row, and the live codes of the first `groups` groups packed once more for the short atoms;
    - the weight side, once: B's forming, B̃·F_A and G as chains, P_B's FMA chain and casts, B's lines and both bases'
      lines, and B's packing, all doubled for fragments wider than their use."""
    m, k, n = s.m, s.k, s.n
    fadd, bf = H100.prices.fadd, H100.prices.bf16_mac
    chain = m * n * (32 * atoms + fadd * (k // 128 - groups))
    fused = m * PC.R * (k + fadd * (k // 128 - 1)) + m * G * groups * 32 * 32
    peel = m * n * bf * PC.PEEL_K + m * PC.PEEL_K * CAST
    forming = m * (k - 112 * groups) * (32 + CAST)
    rows = m * (LINE + 16 * groups * INT32)
    job = 2 * (n * k * (32 + CAST) + (n + PC.R) * PC.R * (k + fadd * (k // 128)) + n * PC.R * PC.R * fadd
               + n * PC.PEEL_K * CAST + (n + 2 * k) * LINE + n * 16 * groups * INT32)
    return Fraction(units * (chain + fused + peel + forming + rows) + job)


def _running(group_words: np.ndarray) -> np.ndarray:
    """Each word's promoted total after each group (FP32, nearest even), from its group words."""
    total, out = np.zeros(group_words.shape[0], dtype=np.float32), []
    for g in range(group_words.shape[1]):
        total = (total + group_words[:, g].astype(np.float32)).astype(np.float32)
        out.append(total)
    return np.stack(out, 1)


def _dead_pair_case(k: int, n: int, groups: int, units: int, m: int = 8192, run: int = 1) -> dict:
    """One 64 x 64 audit tile of the dead-pair family at depth `k`, in `units` units of shape (m, k, n) that share one
    weight: whether its rows pass the rules, whether the program's words are the reference's, the tile's replayed debit
    (flagged atom-words, later promotions into +0, and at least every salt-dead element) against its cap, and the
    program's cost against TT_OUT's bound, per word."""
    C.set_atom("hopper")
    sch = PC.PearlC(forming="v1", device=H100)
    s = Shape(m, k, n)
    sch.check(Matmul(0, s, 0))
    A, B, XA, _, _, ok = _dead_pair_tile(k, groups, run=run)
    P = A.shape[0]
    group_words, honest = C.group_sums(A, B, G)
    words, atoms = _dead_pair_program(A, B, groups)
    flags, _ = C.debit(A, B, G)
    flagged = int(flags.sum())
    later_zero = int((_running(group_words)[:, :-1] == 0).sum())
    dead = _dead_over_count(XA, groups)
    unread = 32 * int(flags.reshape(64, 64, -1).all(1).sum())
    debit = ((32 + Fraction(H100.prices.fadd, G)) * flagged + H100.prices.fadd * later_zero
             + H100.prices.forming_credited * Fraction(64, n) * (dead + unread))
    per_word = Fraction(sch.credit_of(s), s.m * s.n)
    cost, credit = _dead_pair_cost(s, groups, atoms, units), units * Fraction(sch.credit_of(s))
    return {"shape": (m, k, n), "groups": groups, "units": units, "run": run, "rules": bool(ok),
            "words": P, "equal": int((words == honest).sum()), "atoms": atoms, "reference_atoms": k // 32,
            "flagged": flagged, "later_zero": later_zero, "dead": dead, "designed_dead": 64 * 112 * groups,
            "unread": unread, "debit": float(debit), "cap": float(RHO * per_word * P),
            "cost": float(cost / (units * s.m * s.n)), "credit": float(per_word),
            "bound": float((1 - G0) * (1 - RHO) * per_word), "fires": _fires(credit, cost)}


def test_dead_pairs_stay_within_the_gap():
    """Dead pairs: positions where a row of A and a row of B are both dead, so the product is fixed before the salt. On
    rows that put liveness's k/64 dead entries in whole promotion groups, at the same positions in A and B, with signs
    that cancel, a W1 program replaces each such group's four atoms by 16/30 of one (`_dead_pair_program`). The test forms
    one 64 x 64 audit tile of such rows under §5, checks the rules on every row, checks the program's 4,096 ticket words
    against the reference's, checks the tile's replayed debit against the cap with every element that could be salt-dead
    charged, prices the program on the unit (`_dead_pair_cost`) and asserts TT_OUT's bound. The bound fails: this is a
    reviewer's falsification of the conjecture as stated, kept failing until the credit or the debit answers it."""
    case = _dead_pair_case(k=8192, n=8192, groups=1, units=256)
    assert case["rules"], case
    assert case["equal"] == case["words"], case
    assert case["debit"] <= case["cap"], case
    assert not case["fires"], (
        f"dead pairs at {case['shape']}: the program costs {case['cost']:.1f} units a word against a credit of "
        f"{case['credit']:.1f}, under (1 - g0)(1 - rho) of it ({case['bound']:.1f}); the tile's debit is "
        f"{case['debit']:.0f} of a cap of {case['cap']:.0f}")


# ---- raised adder: a second class of the same kind, at a fifth of the credit ---------------------------------------------
#
# The H100's atom aligns its 32 products and its accumulator to the largest exponent among them and cuts each to that
# grid. Three products of 448·448 in a group's first atom leave an accumulator of about 602,112, so in the group's other
# three atoms the grid is 64 wide and every product of two ordinary codes (each under 8) is cut to zero. One planted pair
# in each of those atoms, a multiple of 64, keeps it from being an identity, which P1's flags would charge. The group's
# word is then its first atom's, with the three planted products as that atom's accumulator.

#: code values planted in a group's later atoms: each the centre of an E4M3 cell 8 wide on either side
PLANT_A = {0: (160.0, 160.0, 160.0), 1: (160.0, 160.0, 144.0)}
PLANT_B = {0: (160.0, 160.0, 160.0), 1: (-160.0, -160.0, -176.0)}
#: the sum of a group's three planted products, by the group's parity: 3·160² and −(2·160² + 144·176); a pair of groups
#: then adds 256 and a few multiples of 64 to the running total, so no promotion adds into +0
PLANT_SUM = {0: 76_800.0, 1: -76_544.0}


def _raised_rows(side: str, rows: int, k: int, groups: int, rng: np.random.Generator, fill: str = "gauss") -> np.ndarray:
    """`rows` rows of FP32 values of one operand of the raised-adder family: a Gaussian background, and in each of the
    first `groups` promotion groups six planted entries, the same positions in every row of A and of B: positions 1, 2, 3
    of the group's first atom at the top code (−448 on B in an odd group), and position 1 of each later atom at the codes
    of `PLANT_A` or `PLANT_B`. With `fill="zero"` every other entry of those later atoms is zero, so its code is noise
    alone, under 8 for every salt; with `"gauss"` it is the background, and a product of 64 or more has probability about
    10^-12."""
    assert groups % 2 == 0 and 6 * groups * 64 <= k, "groups come in pairs; liveness admits k/64 dead entries a row"
    X = rng.standard_normal((rows, k))
    if fill == "zero":
        for g in range(groups):
            X[:, 128 * g + 32:128 * g + 128] = 0.0
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    alpha = C.Q_MAX / ((SPIKE + 4.0) * rho)
    for g in range(groups):
        sign = -1.0 if side == "B" and g % 2 else 1.0
        for l in (1, 2, 3):
            X[:, 128 * g + l] = sign * SPIKE * rho
        for t, v in zip((1, 2, 3), (PLANT_A if side == "A" else PLANT_B)[g % 2]):
            X[:, 128 * g + 32 * t + 1] = v / alpha
    return X.astype(np.float32).astype(np.float64)


def _raised_tile(k: int, groups: int, fill: str, rows: int = 64, cols: int = 64, run: int = 1, line_norm: float = 16.0):
    """One audit tile of the raised-adder family, formed as `_dead_pair_tile` forms its own."""
    XA = _raised_rows("A", rows, k, groups, C._rng(run, "raised", k, "A"), fill)
    XB = _raised_rows("B", cols, k, groups, C._rng(run, "raised", k, "B"), fill)
    FcA = C.enc(C.unit_lines(k, C._rng(run, "FA-lines", k)) * (line_norm / np.sqrt(32)))
    FcB = C.enc(C.unit_lines(k, C._rng(run, "FB-lines", k)) * (line_norm / np.sqrt(32)))
    a, _, _, floor_a, live_a = C.form_s5(XA, FcA, 1.0, line_norm, C._rng(run, "raised", k, "unit-A"))
    b, _, _, floor_b, live_b = C.form_s5(XB, FcB, 1.0, line_norm, C._rng(run, "raised", k, "unit-B"))
    ok = floor_a and live_a and floor_b and live_b and _scheme_rules(XA) and _scheme_rules(XB)
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return a[ii.ravel()], b[jj.ravel()], XA, ok


def _raised_program(A: np.ndarray, B: np.ndarray, groups: int) -> tuple[np.ndarray, int]:
    """The ticket words of the W1 program, and the E4M3 atoms it runs per word: for each of the first `groups` groups the
    group's first atom alone, on the reference's own operand rows, its accumulator the constant `PLANT_SUM` of the
    group's parity; every promotion, and every later group, as the reference runs it."""
    P, k = A.shape
    total = np.zeros(P, dtype=np.float32)
    for g in range(groups):
        word = C.step(np.full(P, PLANT_SUM[g % 2]), A[:, 128 * g:128 * g + 32], B[:, 128 * g:128 * g + 32])[0]
        total = (total + word.astype(np.float32)).astype(np.float32)
    for g0 in range(G * groups, k // 32, G):
        total = (total + C.group_run(A, B, list(range(g0, g0 + G)))[0].astype(np.float32)).astype(np.float32)
    return total, groups + k // 32 - G * groups


def _raised_skip_program(A: np.ndarray, B: np.ndarray, groups: int) -> tuple[np.ndarray, int]:
    """The ticket words of a program that only skips, and the E4M3 atoms it runs per word: the reference's chain with the
    second and third atoms of each of the first `groups` groups not executed, for every word. In a pair of planted groups
    those four atoms add 160², 160², −160² and −160², so with the four skipped in place the total after the pair, and every
    later total, is the reference's. Skipping any of them within one group alone changes the word, so P1's flags, which
    range over the subsets of one group, are not set: this is a cross-group joint skip that every word of every fragment
    shares."""
    P, k = A.shape
    skip = frozenset(G * g + t for g in range(groups) for t in (1, 2))
    total = np.zeros(P, dtype=np.float32)
    for g0 in range(0, k // 32, G):
        total = (total + C.group_run(A, B, list(range(g0, g0 + G)), skip)[0].astype(np.float32)).astype(np.float32)
    return total, k // 32 - len(skip)


def _raised_cost(s: Shape, atoms: int, units: int) -> Fraction:
    """An upper bound on the W1 cost, on `W1Device.s8H100Best`, of the raised-adder program on `units` units of shape `s`
    that share one weight, counted as `_dead_pair_cost` counts: `atoms` E4M3 atoms a word and the reference's promotions;
    the fused A′·F_B, the peel, the forming of every element and a noise line a row as the reference runs them; and the
    weight side once, doubled. The program packs nothing: its atoms read the reference's operand rows."""
    m, k, n = s.m, s.k, s.n
    fadd, bf = H100.prices.fadd, H100.prices.bf16_mac
    chain = m * n * (32 * atoms + fadd * (k // 128 - 1))
    fused = m * PC.R * (k + fadd * (k // 128 - 1))
    peel = m * n * bf * PC.PEEL_K + m * PC.PEEL_K * CAST
    forming = m * k * (32 + CAST)
    rows = m * LINE
    job = 2 * (n * k * (32 + CAST) + (n + PC.R) * PC.R * (k + fadd * (k // 128)) + n * PC.R * PC.R * fadd
               + n * PC.PEEL_K * CAST + (n + 2 * k) * LINE)
    return Fraction(units * (chain + fused + peel + forming + rows) + job)


def _raised_case(k: int, n: int, groups: int, units: int, m: int = 8192, run: int = 1, fill: str = "gauss",
                 program: str = "one-atom") -> dict:
    """One 64 x 64 audit tile of the raised-adder family at depth `k`, reported as `_dead_pair_case` reports its own, for
    the program that runs one atom a planted group (`"one-atom"`) or the one that only skips (`"skip"`)."""
    C.set_atom("hopper")
    sch = PC.PearlC(forming="v1", device=H100)
    s = Shape(m, k, n)
    sch.check(Matmul(0, s, 0))
    A, B, XA, ok = _raised_tile(k, groups, fill, run=run)
    P = A.shape[0]
    group_words, honest = C.group_sums(A, B, G)
    words, atoms = (_raised_program if program == "one-atom" else _raised_skip_program)(A, B, groups)
    flags, _ = C.debit(A, B, G)
    flagged = int(flags.sum())
    later_zero = int((_running(group_words)[:, :-1] == 0).sum())
    dead = _dead_over_count(XA, groups)
    unread = 32 * int(flags.reshape(64, 64, -1).all(1).sum())
    debit = ((32 + Fraction(H100.prices.fadd, G)) * flagged + H100.prices.fadd * later_zero
             + H100.prices.forming_credited * Fraction(64, n) * (dead + unread))
    per_word = Fraction(sch.credit_of(s), s.m * s.n)
    cost, credit = _raised_cost(s, atoms, units), units * Fraction(sch.credit_of(s))
    return {"class": "raised-adder", "program": program, "fill": fill, "shape": (m, k, n), "groups": groups, "units": units, "run": run,
            "rules": bool(ok), "words": P, "equal": int((words == honest).sum()), "atoms": atoms,
            "reference_atoms": k // 32, "flagged": flagged, "flagged_in_planted_groups": int(flags[:, :G * groups].sum()),
            "later_zero": later_zero, "dead": dead, "designed_dead": 64 * 6 * groups, "unread": unread,
            "debit": float(debit), "cap": float(RHO * per_word * P), "cost": float(cost / (units * s.m * s.n)),
            "credit": float(per_word), "bound": float((1 - G0) * (1 - RHO) * per_word), "fires": _fires(credit, cost)}


def test_raised_adder_groups_stay_within_the_gap():
    """Raised adder: six planted entries in a promotion group, at the same positions in every row of A and of B, make the
    group's last three atoms add a constant whatever else they hold (`_raised_rows`), so a W1 program runs one atom of
    the group's four (`_raised_program`). At 8192 deep liveness admits twenty such groups of the 64. The test forms one
    64 x 64 audit tile under §5 with every unplanted entry of those atoms zero, so the cut to zero holds for every salt,
    checks the rules on every row, the program's 4,096 ticket words against the reference's, the tile's replayed debit
    against the cap, prices the program on one unit, and asserts TT_OUT's bound. The bound fails by about a fifth of the
    credit: a reviewer's falsification of the conjecture as stated, kept failing until the credit or the debit answers
    it."""
    case = _raised_case(k=8192, n=8192, groups=20, units=1, fill="zero")
    assert case["rules"], case
    assert case["equal"] == case["words"], case
    assert case["debit"] <= case["cap"], case
    assert not case["fires"], (
        f"raised adder at {case['shape']}: the program costs {case['cost']:.1f} units a word against a credit of "
        f"{case['credit']:.1f}, under (1 - g0)(1 - rho) of it ({case['bound']:.1f}); the tile's debit is "
        f"{case['debit']:.0f} of a cap of {case['cap']:.0f}")


def test_planted_cross_group_skips_stay_within_the_gap():
    """The cross-group class on operands an adversary chooses, with a program that only skips. `test_cross_group_joint_skips_
    stay_within_the_gap` rests on W1 charging a fragment whole, so that a program saves an atom only where every word the
    fragment covers can skip it, which census tiles do not offer. Planted rows offer it: the second and third atoms of
    every planted group can be skipped in place for every word of the unit at once, in pairs of groups
    (`_raised_skip_program`), and no flag is set. The test forms one 64 x 64 audit tile as the test above does, checks the
    rules, the 4,096 ticket words and the cap, prices the program on one unit, and asserts TT_OUT's bound, which fails by
    about a tenth of the credit: the whole-fragment rule does not bound this class once the adversary chooses the
    operands."""
    case = _raised_case(k=8192, n=8192, groups=20, units=1, fill="zero", program="skip")
    assert case["rules"], case
    assert case["equal"] == case["words"], case
    assert case["flagged_in_planted_groups"] == 0, case
    assert case["debit"] <= case["cap"], case
    assert not case["fires"], (
        f"planted cross-group skips at {case['shape']}: the program costs {case['cost']:.1f} units a word against a "
        f"credit of {case['credit']:.1f}, under (1 - g0)(1 - rho) of it ({case['bound']:.1f}); the tile's debit is "
        f"{case['debit']:.0f} of a cap of {case['cap']:.0f}")
