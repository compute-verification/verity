"""The second stage's coins (`second_stage`): its first read against the bytes the second stage's sampler takes, the
units a first read is counted in, a planner's design's cells, and the figures at a design."""
import hashlib
from fractions import Fraction

import pytest
import second_stage as s2
from verity.protocols.spot.device.challenger import COIN_BYTES, MAX_COINS
from verity.protocols.spot.optimize import Cell
from verity.protocols.spot.verifier import service


def test_the_first_read_is_the_bytes_the_second_stage_takes():
    """`service.select_second` on fixed bytes takes each cell's `2 k·width(n) + 64` bytes once, template by template
    and in each by sampled replay block, `first_read` in all: no word of these bytes is rejected, so no cell reads
    again."""
    sample, ns, ks = (3, 8, 20), (5, 300, 70000), (5, 17, 1000)
    asked = []

    def coins(n: int) -> bytes:
        asked.append(n)
        return hashlib.shake_256(len(asked).to_bytes(4, "big")).digest(n)

    got = service.select_second(sample, ns, ks, coins)
    assert asked == [2 * k * s2.width(n) + 64 for n, k in zip(ns, ks) for _ in sample]
    assert sum(asked) == s2.first_read(len(sample), ns, ks)
    assert len(got["units"]) == len(sample) * sum(ks)


def test_a_word_is_the_population_s_bits_and_64_more():
    assert [s2.width(n) for n in (1, 255, 256, 2 ** 16 - 1, 2 ** 16, 2 ** 64)] == [9, 9, 10, 10, 11, 17]


def test_a_first_read_in_rounds_and_in_the_challenger_s_coins():
    assert s2.ROUND_BYTES == 2 ** 20
    assert [s2.stages(b) for b in (1, 2 ** 20, 2 ** 20 + 1)] == [1, 1, 2]
    assert [s2.coins(b) for b in (64, 65)] == [1, 2]
    held = MAX_COINS * COIN_BYTES
    assert [s2.fits(b) for b in (held, held // 2, held // 2 + 1, held + 1)] == [1, 2, 1, 0]
    assert s2.refusal(3, (2, 5)) == Fraction(21, 2 ** 64)


@pytest.mark.parametrize("ns,ks", [((), ()), ((4,), (0,)), ((4,), (5,)), ((4, 4), (1,))])
def test_cells_the_second_stage_refuses_have_no_first_read(ns, ks):
    with pytest.raises(ValueError, match="samples 1 to n_t of each cell's n_t proof blocks"):
        s2.first_read(1, ns, ks)


def test_a_design_s_cells_split_each_stratum_s_count_in_proportion():
    """Two 16-bit types in the merged stratum and a 64-bit type in its own: the merged stratum's 10 go 1 and 9 to
    cells of 3 and 101 proof blocks; a count under the cells' number leaves each cell 1, and one of every proof block
    takes each cell whole."""
    cells = [Cell(3, 16), Cell(100.5, 16), Cell(7, 64)]
    assert s2.design_cells(cells, {"merged": 16, "k": {0: 10, 64: 2}}) == ((3, 101, 7), (1, 9, 2))
    assert s2.design_cells(cells, {"merged": 16, "k": {0: 1, 64: 7}}) == ((3, 101, 7), (1, 1, 7))
    assert s2.design_cells(cells, {"merged": 16, "k": {0: 104, 64: 1}}) == ((3, 101, 7), (3, 101, 1))
    assert s2.design_cells(cells, {"merged": 64, "k": {0: 37}}) == ((3, 101, 7), (1, 34, 2))


def test_the_second_stage_at_a_design():
    """A verification phase of at most 3 sampled replay blocks: their cells' first reads, 82, 226 and 100 bytes a
    replay block, in rounds and in coins, the replay blocks of 408 bytes a commitment's coins hold, and Lean's bound at
    3·12 sampled proof blocks; over the scope's 12.5 sampled replay blocks, 12.5·12 of them."""
    cells = [Cell(3, 16), Cell(100.5, 16), Cell(7, 64)]
    row = {"merged": 16, "k": {0: 10, 64: 2}, "registration_cap": 3, "replay_sample": 12.5}
    assert s2.at_design(cells, row) == {
        "cells": 3, "sampled": 12, "whole": False, "sample": 3, "read": 408, "stages": 1, "coins": 20,
        "fits": MAX_COINS * COIN_BYTES // 408, "refusal": 36 / 2.0 ** 64, "pooled": 150 / 2.0 ** 64}
    assert s2.at_design(cells, row | {"registration_cap": 0})["sample"] == 1
    whole = s2.at_design([Cell(2, 16)], {"merged": 16, "k": {0: 2}, "registration_cap": 1, "replay_sample": 4.0})
    assert whole == {"cells": 1, "sampled": 2, "whole": True, "sample": 1, "read": 0, "stages": 0, "coins": 0,
                     "fits": None, "refusal": 0.0, "pooled": 0.0}
