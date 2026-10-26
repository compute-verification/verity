"""The call's reference (slots.py): the checked-in fixture, its leaves and trees rebuilt from verity's own frame and hm96
primitives, C-Flock's salt stream, unit openings and their refusals, the layout's rules and the frozen H200 layout."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import slots as S
from verity.core.protocols.pouw.audit import hm96_salt
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.row_table import root_leaf
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

CTX = bytes(range(64))
KEY = bytes(range(100, 132))


def test_the_fixture_is_the_references() -> None:
    assert json.loads(S.FIXTURE.read_text()) == S.fixture()


def test_salts_are_cflocks_hm96_salt_with_the_slot_as_nonce() -> None:
    for slot, g in ((0, 0), (3, 5), ((1 << 40) + 1, 1000)):
        assert S.row_salts(KEY, slot, g, 2) == hm96_salt(KEY, slot, g) + hm96_salt(KEY, slot, g + 1)


def _slot_root_by_hand(layout: S.Layout, rows: list[bytes], slot: int) -> bytes:
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    salts = hm96_salt
    unit_roots = []
    for u, unit in enumerate(layout.units):
        d = S.unit_domain(CTX, layout, u)
        leaves = []
        for r in range(len(unit)):
            g = layout.row_base[u] + r
            x = hashlib.sha512(sha512_row_v2_prefix(8 * len(rows[g])) + rows[g]).digest()
            bc = inst.commit_string(key, x, salts(KEY, slot, g))
            leaves.append(d.leaf(r, r, "hm96-sha512/row/v2", inst.tree_leaf(key, bc)))
        unit_roots.append(tree_levels(d, leaves)[-1][0])
    sd = S.slot_domain(CTX, layout)
    return tree_levels(sd, [root_leaf(sd.domain_id, u, r) for u, r in enumerate(unit_roots)])[-1][0]


def test_a_step_is_frame_v3_trees_of_hm96_row_v2_leaves() -> None:
    layout, slots = S.edge_layout(), 3
    out = S.random_outputs(layout, slots, np.random.default_rng(1))
    roots = [_slot_root_by_hand(layout, S.slot_rows(layout, out, s), s) for s in range(slots)]
    td = S.step_domain(CTX, layout, 4, slots)
    step_root = tree_levels(td, [root_leaf(td.domain_id, s, r) for s, r in enumerate(roots)])[-1][0]
    assert S.commit(CTX, layout, 4, KEY, out, slots) == S.StepCommit(step_root, tuple(roots))


def test_every_unit_opens_and_a_changed_opening_is_refused() -> None:
    layout, slots, step = S.edge_layout(), 3, 2
    out = S.random_outputs(layout, slots, np.random.default_rng(2))
    c = S.commit(CTX, layout, step, KEY, out, slots)
    for s in range(slots):
        rows = S.slot_rows(layout, out, s)
        for u in range(layout.unit_count):
            o = S.open_unit(CTX, layout, step, KEY, c.slot_roots, s, u, rows)
            b, n = layout.row_base[u], len(layout.units[u])
            salts = S.row_salts(KEY, s, b, n)
            mine = {r: (rows[b + r], salts[192 * r:192 * (r + 1)]) for r in range(n)}
            assert S.verify_unit(c.root, CTX, layout, step, slots, o)
            assert S.verify_unit(c.root, CTX, layout, step, slots, o, mine)
            assert not S.verify_unit(c.root, CTX, layout, step + 1, slots, o)
            assert not S.verify_unit(c.root, CTX, layout, step, slots + 1, o)
            flipped = bytes([o.strings[0][0] ^ 1]) + o.strings[0][1:]
            assert not S.verify_unit(c.root, CTX, layout, step, slots, S.UnitOpening(s, u, (flipped, *o.strings[1:]),
                                                                                     o.unit_path, o.slot_path))
            other = {0: (bytes([mine[0][0][0] ^ 1]) + mine[0][0][1:], mine[0][1]), **{r: mine[r] for r in range(1, n)}}
            assert not S.verify_unit(c.root, CTX, layout, step, slots, o, other)
    o = S.open_unit(CTX, layout, step, KEY, c.slot_roots, 1, 0, S.slot_rows(layout, out, 1))
    assert not S.verify_unit(c.root, CTX, layout, step, slots, S.UnitOpening(2, 0, o.strings, o.unit_path, o.slot_path))


def test_a_replay_of_other_rows_is_refused() -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, 2, np.random.default_rng(3))
    c = S.commit(CTX, layout, 0, KEY, out, 2)
    with pytest.raises(ValueError, match="not the ones committed"):
        S.open_unit(CTX, layout, 0, KEY, c.slot_roots, 0, 0, S.slot_rows(layout, out, 1))


def test_the_layout_refuses_a_unit_over_its_size_class_and_a_part_outside_its_tensor() -> None:
    with pytest.raises(ValueError, match="262144 bits"):
        S.Group((40000,), ((S.Row((S.Part(0, 0, 32768),)), S.Row((S.Part(0, 32768, 1),))),))
    with pytest.raises(ValueError, match="outside"):
        S.Group((10,), ((S.Row((S.Part(0, 5, 6),)),),))
    with pytest.raises(ValueError, match="at least one row"):
        S.Group((10,), ((),))


def test_the_h200_layout() -> None:
    h = S.h200_layout()
    g = h.groups[0]
    assert len(h.groups) == 52 and all(x is g for x in h.groups)
    assert len(g.units) == 25 and h.unit_count == 1300
    assert all(sum(r.bits for r in u) <= S.UNIT_MAX_BITS for u in g.units)
    assert sum(g.sources) == sum(r.nbytes for u in g.units for r in u) == 663_824
    assert h.slot_bytes * 20480 == 706_946_007_040
