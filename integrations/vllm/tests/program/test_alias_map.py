"""`AliasMap` (the derive's storage aliases) answers `aliases(base, node)` as the scan it replaces did, in the same order, under any mix
of writes, `setdefault`s and pops."""
from __future__ import annotations

import random

from verity_vllm.program.frontend.alias_map import AliasMap


class _Node:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def test_aliases_are_the_scan_in_dict_order_under_random_edits():
    rng = random.Random(7)
    nodes = [_Node(f"n{i}") for i in range(60)]
    m, ref = AliasMap(), {}
    for _ in range(4000):
        op, a, b = rng.random(), rng.choice(nodes), rng.choice(nodes)
        if op < 0.45:
            m[a] = b
            ref[a] = b
        elif op < 0.7:
            assert m.setdefault(a, b) is ref.setdefault(a, b)
        else:
            assert m.pop(a, None) is ref.pop(a, None)
        assert list(m.items()) == list(ref.items())
        base, node = rng.choice(nodes), rng.choice(nodes)
        assert m.aliases(base, node) == [n for n, x in ref.items() if x is base and n is not node]
