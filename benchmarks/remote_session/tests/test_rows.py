"""`rows.py --ctas-at`: a row count runs at the receipt CTA counts its word names, in the word's order, and at
`--sketch-ctas` alone where no word names it; a word for a row count that doesn't run, or one whose counts aren't
numbers, ends the run saying what a word is."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rows


def test_a_row_count_takes_its_words_counts_in_order_and_the_default_elsewhere():
    assert rows.ctas_at(["64=96,128,264"], [16, 64, 16, 64], 128) == {16: [128], 64: [96, 128, 264]}
    assert rows.ctas_at([], [64, 16], 0) == {64: [0], 16: [0]}
    assert rows.ctas_at(["64=132", "16=128"], [64, 16], 64) == {64: [132], 16: [128]}


@pytest.mark.parametrize("word", ["48=128", "64", "64=", "64=128,", "64=a", "x=128", "64=-1"])
def test_a_word_that_names_no_running_row_count_or_no_counts_ends_the_run(word):
    with pytest.raises(SystemExit, match="ROWS=N"):
        rows.ctas_at([word], [64], 128)
