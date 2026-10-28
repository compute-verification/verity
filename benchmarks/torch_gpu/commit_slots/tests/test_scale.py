"""Planning a run stays linear in its units at the families map's scale (6,102,102 units a slot): the layout digest
and every unit's domain id are the reference's bytes, computed once (the digest) and once a unit (the ids); a group the
layout repeats is laid out once; and a host Plan of over six million units builds."""
from __future__ import annotations

import hashlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import slots as S
import slots_gpu as G
from verity.ml.commitments.identity import identity_digest_sha512

needs_cxx = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="needs a C++ compiler")

CTX = bytes(range(64))


def families_like(layers: int, coords: int = 4, chunks: int = 1) -> S.Layout:
    """A layout shaped like the families map's: one group a layer, all the same `Group`, of ``coords`` one-row units (a
    coordinate's 16-bit row) and ``chunks`` two-row units (a quantizer chunk's codes, then its f32 scale)."""
    units = [(S.Row((S.Part(0, 2 * i, 2),)),) for i in range(coords)]
    units += [(S.Row((S.Part(1, 64 * i, 64),)), S.Row((S.Part(2, 0, 4),))) for i in range(chunks)]
    return S.Layout(f"families-like/{coords}x{chunks}", (S.Group((2 * coords, 64 * chunks, 4), tuple(units)),) * layers)


LAYOUTS = {
    "edge": S.edge_layout(),
    "h200-3": S.h200_layout(layers=3),
    "h200-3-by-2": G.regroup(S.h200_layout(layers=3), 2),
    "an-empty-group": S.Layout("ünits/v1", (S.Group((4,), ()), *S.edge_layout().groups)),
    "families-like": families_like(5),
}


@pytest.mark.parametrize("name", LAYOUTS)
def test_the_layout_digest_and_every_unit_domain_id_are_the_references(name: str) -> None:
    layout = LAYOUTS[name]
    assert S.layout_digest(layout) == identity_digest_sha512(S.LAYOUT_TAG, layout.manifest())
    ids = S.unit_domain_ids(CTX, layout)
    assert [ids[64 * u:64 * (u + 1)] for u in range(layout.unit_count)] == \
        [S.unit_domain(CTX, layout, u).domain_id for u in range(layout.unit_count)]
    assert layout.row_base == tuple(sum(len(u) for u in layout.units[:u]) for u in range(layout.unit_count))
    assert layout.group_row_base == tuple(sum(layout.group_rows[:i]) for i in range(len(layout.groups)))


TAGGED, SHAPE = S.tagged_sha512, G.Plan._shape


def _plan_counts(layout: S.Layout) -> dict[str, int]:
    """Build a host Plan and a step committer over ``layout``, counting SHA-512 calls in slots.py, layout digests, group
    layouts, and refusing the manifest (O(units) a call)."""
    n = {"sha512": 0, "layout": 0, "shape": 0}

    def sha512(data: bytes = b"") -> object:
        n["sha512"] += 1
        return hashlib.sha512(data)

    def tagged(tag: str, payload: bytes) -> bytes:
        n["layout"] += tag == S.LAYOUT_TAG
        return TAGGED(tag, payload)

    def manifest(self: S.Layout) -> None:
        raise AssertionError("planning builds no manifest")

    def shape(self: G.Plan, g: S.Group) -> object:
        n["shape"] += 1
        return SHAPE(self, g)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(S, "hashlib", SimpleNamespace(sha512=sha512))
        m.setattr(S, "tagged_sha512", tagged)
        m.setattr(S.Layout, "manifest", manifest)
        m.setattr(G.Plan, "_shape", shape)
        plan = G.Plan(CTX, layout, "host")
        G.StepCommitter(plan, 0, 2, bytes(32))
        S.unit_domain(CTX, layout, layout.unit_count - 1)
    return n


@needs_cxx
def test_planning_is_linear_in_the_units() -> None:
    """Doubling the layers doubles the hashing (two SHA-512s a unit: its manifest's, then its domain's) and nothing else: the
    layout digest is computed once and the repeated layer laid out once. Counted, not timed (no test reads a clock)."""
    one, two = families_like(40), families_like(80)
    a, b = _plan_counts(one), _plan_counts(two)
    assert a == {"sha512": 2 * one.unit_count, "layout": 1, "shape": 1}
    assert b == {"sha512": 2 * two.unit_count, "layout": 1, "shape": 1}


@needs_cxx
@pytest.mark.slow
def test_a_host_plan_of_over_six_million_units() -> None:
    """A host Plan at the families map's size: 52 layers of 117,400 units (117,000 one-row, 400 two-row), 6,104,800 units and
    6,125,600 rows a slot. Takes about 20 s and 2.3 GB on a CPU node (the leaf tables are 224 bytes a row); before the
    layout digest was cached, planning was quadratic in the units and this never finished. What it catches that
    test_planning_is_linear_in_the_units can't: a linear but slow step a unit (a Python object or call a row) at full size."""
    layout = families_like(52, coords=117_000, chunks=400)
    plan = G.Plan(CTX, layout, "host")
    assert layout.unit_count == 6_104_800 and layout.row_count == 6_125_600
    assert plan.unit_dids.shape == (layout.unit_count, 8)
    for u in (0, 116_999, 117_000, 117_399, 3_000_000, layout.unit_count - 1):
        assert np.array_equal(plan.unit_dids[u], G.E.words(S.unit_domain(CTX, layout, u).domain_id)[0])
    assert sum(g.leaf_table.G for g in plan.groups) == layout.row_count
