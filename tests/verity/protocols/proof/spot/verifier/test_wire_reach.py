"""`members._reaching`, the per-member index `_wired_roots` reads, gives the wires a scan of every range and wire by
`_wire_reaches` gives, in their order, on random partitions with empty ranges and wires of every reader."""

from __future__ import annotations

import random

from verity.protocols.proof.spot.verifier.one_stage import members as M


def test_reaching_is_the_scan_of_every_range_and_wire():
    rnd = random.Random(1)
    for _ in range(600):
        ranges, at = [], 0
        for k in range(rnd.randint(1, 8)):
            c = rnd.choice([0, 1, 2, 3, 5, 8])
            ranges.append((k, rnd.choice("ABC"), at, c))
            at += c
        lbase = M._local_bases(ranges)
        wires = [{"reader": rnd.choice("ABC"), "cut": (0, rnd.choice([0, 1, 2, 4])), "at": rnd.randint(0, 10),
                  "radix": [rnd.randint(1, 6), rnd.randint(1, 3)], "i": i} for i in range(rnd.randint(0, 6))]
        reach = M._reaching(ranges, lbase, wires)
        for first in range(at + 1):
            for n in range(1, 5):
                assert reach(first, n) == [w for w in wires if M._wire_reaches(ranges, lbase, first, n, w)]
