"""algo_lab (CMP-125): every alternative to the H200 reference kernel computes Dither's words and receipts exactly, and
its DS-90 prices are the ones the note's table quotes."""
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo_lab import lab as LAB  # noqa: E402
from algo_lab import price as PR  # noqa: E402


@pytest.fixture(scope="module")
def words():
    rows = LAB.word_table()
    return rows, rows[0].load


@pytest.fixture(scope="module")
def folds():
    rows, mixes = LAB.fold_table()
    return {r.name: r for r in rows}, mixes


def test_every_route_is_exact(words, folds):
    rows, _ = words
    assert [r.name for r in rows if not r.exact] == []
    frows, mixes = folds
    assert [n for n, r in frows.items() if not r.exact] == []
    assert all(m[2].exact for m in mixes.values())


def test_the_reference_is_alu_bound_by_its_checkpoint(words):
    _, ref = words
    assert PR.time(ref) == (pytest.approx(136 / 128), "ALU")
    assert ref["tensor"] == pytest.approx(1.0)


def test_no_alternative_comes_near_the_reference(words):
    """Strassen at depth 1 is the closest: 7/8 of the tensor work, but its additions put 3.8 times the reference's time on
    the ALU pipe, and 2.86 times even with the ALU and IMAD pipes pooled."""
    rows, ref = words
    cols = {r.name: r.cols(ref) for r in rows[1:]}
    best = min(cols, key=lambda n: cols[n]["time"])
    assert best == "(a) Strassen depth 1, A = B = 0"
    assert cols[best]["tensor"] == pytest.approx(0.875)
    assert cols[best]["time"] == pytest.approx(3.824, abs=1e-3)
    assert min(c["pooled"] for c in cols.values()) > 2.8
    assert min(c["issue"] for c in cols.values()) > 2.8


def test_li_has_nothing_under_the_menus_cp_red2(folds):
    """cp-red2 (3 LOP3 and 2 REDG a segment) balances ALU and issue at 134/128, 1.47% under cp-seg4's 136/128: the
    figure `machine_model/tests/test_eps.py` pins for the design's menu (0.014706). LI's own fold is 1 lane instruction
    a word in every exact form found."""
    by, _ = folds
    red2 = by["LI, cp-red2"]
    assert PR.time(red2.load)[0] == pytest.approx(134 / 128)
    assert 1 - PR.time(red2.load)[0] / PR.time(by["LI, cp-seg4 (reference)"].load)[0] == pytest.approx(0.014706, abs=1e-6)
    assert min(PR.time(r.load)[0] for n, r in by.items() if r.fold == "li") == PR.time(red2.load)[0]
    assert min(r.fold_instr for r in by.values() if r.fold == "li") == pytest.approx(1.0)


def test_duos_imad_form_mix_is_a_sixth_under_leahi_on_ds90(folds):
    """duo's honest form (LOP3, IMAD, LEA.HI) puts 4 on the ALU pipe a warp-word; moving the rotate-add of about a third
    of the words to IMAD.HI + IMAD balances ALU, IMAD and issue at 3.3385 (no checkpoint), 16.5% less, with more
    instructions: on DS_issue alone it costs more."""
    by, mixes = folds
    x, t, row = mixes["duo LEA.HI/IMAD, no checkpoint"]
    assert x == Fraction(127, 384) and t == pytest.approx(3 + 1 / 128 + 127 / 384)
    assert PR.time(by["duo, LEA.HI (reference dc90-l32)"].bare)[0] == pytest.approx(4.0)
    assert PR.time(row.bare)[0] == pytest.approx(3.375)
    assert row.load["issue"] > by["duo, LEA.HI (reference dc90-l32)"].load["issue"]
