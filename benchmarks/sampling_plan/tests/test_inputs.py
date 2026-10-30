"""A statement's cap (`inputs.cap`): the C-Flock block a proof block needs is the composer's occupancy of it (the unit's slot
above its ANDs and port bits, an hm96 slot a row, the mask and the sha512x3 slots), not its ANDs with its rows' openings;
past C-Flock's 2^K_MAX a shape is no proof block, and every plan reads the one cap."""

from __future__ import annotations

import h200 as H
import int8_llama as P
import transformer as X
from inputs import K_MAX, block_k_log, cap

COORD_ROWS = (2, 4, 4)


def test_the_block_is_the_composers_at_the_measured_points():
    """ScaledMm's K = 128 (21,963 ANDs) and K = 1024 (139,283) coordinates prove at 22 (cpu_budget's calibration), A6's
    K = 2048 coordinate (271,342) at 23 and the sweep's 14.8M-AND unit at 25."""
    assert block_k_log(21_963, (128, 128) + COORD_ROWS) == 22
    assert block_k_log(139_283, (1024, 1024) + COORD_ROWS) == 22
    assert block_k_log(271_342, (2048, 2048) + COORD_ROWS) == 23
    assert block_k_log(14_800_000, (4096, 4096) + COORD_ROWS) == 25
    assert cap(271_342, (2048, 2048) + COORD_ROWS) == 1 << (35 - 23)


def test_a_one_slot_unit_fits_below_half_the_block():
    """The unit's slot shares the block with its rows' slots, so a unit whose ANDs and port bits reach 2^(K_MAX - 1) needs
    a block of 2^(K_MAX + 1) and is no proof block, though its ANDs are under 2^K_MAX."""
    rows = (1024, 1024) + COORD_ROWS
    under = (1 << (K_MAX - 1)) - 1 - 8 * sum(rows)
    assert block_k_log(under, rows) == K_MAX and cap(under, rows) == 1 << (35 - K_MAX)
    assert block_k_log(under + 1, rows) == K_MAX + 1 and cap(under + 1, rows) == 0 and under + 1 < 1 << K_MAX


def test_the_int8_plan_offers_no_tile_past_the_block():
    """int8 Llama's K = 14336 2 x 1 tile and both 16 x 16 tiles are past C-Flock's block (cap 0, descended into, never proved
    whole), and so is its K = 14336 coordinate, whose 898 noise compressions at `SHA512_ROWS` are 73M ANDs; its K = 4096
    coordinate (258, 21M) fits, 512 a statement."""
    caps = {}

    def walk(s):
        caps[s.id] = s.cap
        for _, parts in s.decompositions():
            for child, _ in parts:
                walk(child)

    for k in (4096, 14336):
        walk(P.linear(k, 16))
    assert {t for t, c in caps.items() if c == 0} == {"S8Tile{K=4096,16x16}", "S8Tile{K=14336,2x1}", "S8Tile{K=14336,16x16}",
                                                      "S8Coord{K=14336}"}
    assert caps["S8Coord{K=4096}"] == 512


def test_a_unit_past_the_block_is_proved_as_its_least_parts():
    """The H200 demo's stub norm row (153.8M ANDs over three 32 KiB rows) needs 2^29 bits whole: it is drawn whole and proved
    as 3 parts of 2^27, 256 parts a statement, so 85 units."""
    t = H.Template("stub:RmsNormRow_v1{N=16384}", 1, 153_800_784, (32768,) * 3, 262_144)
    assert block_k_log(t.ands, t.rows) == K_MAX + 2 and cap(t.ands, t.rows) == 0
    assert (t.parts, t.cap) == (3, 256 // 3)


def test_every_plan_reads_the_one_cap():
    assert P.cap is X.cap is H.cap is cap
