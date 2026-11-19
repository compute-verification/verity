"""The call's reference (slots.py): the checked-in fixture, its leaves and trees rebuilt from verity's own frame and hm96
primitives under the salt bytes it was handed, ``open_unit``'s openings and their refusals, layouts written and loaded back through
``commitments.layout`` to the same digests and roots, and the frozen H200 layout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import salts_of

from verity.commitments.cuda.commit_slots import slots as S
from verity.commitments import hm96
from verity.commitments import layout as L
from verity.commitments.merkle import tree_levels
from verity.commitments.row_table import root_leaf
from verity.commitments.rowleaf import sha512_row_v2_prefix

CTX = bytes(range(64))
#: FAMILIES' shared-row layout at 1 KiB rows, fetched with commitments' layout fixtures (LAYOUTS, a registered fixture no
#: checkout holds until it is fetched), its sha256, slots' digest of it (the artifact's ``layout_digest``),
#: ``commitments.layout``'s ``json_digest`` of it, and a step root over it under ``salts_of``'s test salts
LAYOUTS = "tests/verity/commitments/fixtures/layouts"
FAMILIES = Path(__file__).resolve().parents[5] / LAYOUTS / "families-shared-1024b.layout.json"
FAMILIES_SHA256 = "15fd14be19dea4d4d70b18771d0e185881894c3d5e480db35a9dbe4eb5141cd8"
FAMILIES_DIGEST = ("eb7937f19e2e8e3a0daf9f7b85f33d837b267a079cd0c5c60d3581e8c2b1675b"
                   "095d54fd36f2e8434c67734213533c0aba0e40d9f2601cc4307ca9474e403a03")
FAMILIES_JSON_DIGEST = ("f8d6252c5933f22c8c276c911a7c22e84c7ea12d846d7de8f9ee90e312c68005"
                        "3d41a525a4350c921f9cf12b80fc0dc9d43d3034c8b7a33f83bc069c1e656a9f")
FAMILIES_ROOT = ("9aa80c49a85ad0ce4f574cc22f4c03c9d8fa2dd55b275f34efe9325ef3cffb70"
                 "bfa3890b7650973f42d6e905af73bbecb982fd6d6522dc2cb0f3dc504d556f13")


def test_the_fixture_is_the_references() -> None:
    assert json.loads(S.FIXTURE.read_text()) == S.fixture()


def test_fresh_salts_are_the_operating_systems_one_draw_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fresh_salts`` is ``hm96.SHA512.fresh_salt()`` once a row of a slot, laid out [slot, row], and nothing else: no
    key, slot or row number enters a salt."""
    layout, slots, drawn = S.edge_layout(), 3, []

    def fresh(inst: hm96.Instance) -> bytes:
        drawn.append(bytes([len(drawn) % 251]) * inst.salt_bytes)
        return drawn[-1]

    monkeypatch.setattr(type(S.INST), "fresh_salt", fresh)
    y = S.fresh_salts(layout, slots)
    assert len(drawn) == slots * layout.row_count
    assert y.tobytes() == b"".join(drawn) and y.shape == (slots, layout.row_count, hm96.SHA512.salt_bytes)


def _slot_root_by_hand(layout: S.CommitShape, rows: list[bytes], salts: np.ndarray) -> bytes:
    """A slot's root from hm96's own leaf: each row's string is ``hm96.SHA512.commit_string`` under that row's 192 salt
    bytes, as handed in."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    unit_roots = []
    for u, unit in enumerate(layout.units):
        d = S.unit_domain(CTX, layout, u)
        leaves = []
        for r in range(len(unit)):
            g = layout.row_base[u] + r
            x = hashlib.sha512(sha512_row_v2_prefix(8 * len(rows[g])) + rows[g]).digest()
            bc = inst.commit_string(key, x, salts[g].tobytes())
            leaves.append(d.leaf(r, r, "hm96-sha512/row/v2", inst.tree_leaf(key, bc)))
        unit_roots.append(tree_levels(d, leaves)[-1][0])
    sd = S.slot_domain(CTX, layout)
    return tree_levels(sd, [root_leaf(sd.domain_id, u, r) for u, r in enumerate(unit_roots)])[-1][0]


def test_a_step_is_frame_v3_trees_of_hm96_row_v2_leaves() -> None:
    layout, slots = S.edge_layout(), 3
    out = S.random_outputs(layout, slots, np.random.default_rng(1))
    y = salts_of(layout, slots)
    roots = [_slot_root_by_hand(layout, S.slot_rows(layout, out, s), y[s]) for s in range(slots)]
    td = S.step_domain(CTX, layout, 4, slots)
    step_root = tree_levels(td, [root_leaf(td.domain_id, s, r) for s, r in enumerate(roots)])[-1][0]
    assert S.commit(CTX, layout, 4, y, out, slots) == S.StepCommit(step_root, tuple(roots))


def test_a_replay_of_other_rows_is_refused() -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, 2, np.random.default_rng(3))
    y = salts_of(layout, 2)
    c = S.commit(CTX, layout, 0, y, out, 2)
    with pytest.raises(ValueError, match="not the ones committed"):
        S.open_unit(CTX, layout, 0, y[0], c.slot_roots, 0, 0, S.slot_rows(layout, out, 1))
    with pytest.raises(ValueError, match="not the ones committed"):
        S.open_unit(CTX, layout, 0, y[1], c.slot_roots, 0, 0, S.slot_rows(layout, out, 0))


def _round_trip(layout: S.CommitShape, tmp_path: Path) -> S.CommitShape:
    """``layout`` written in ``_layout_json``'s form (compact JSON, then a newline) and loaded back by ``S.load``
    (``commitments.layout``): an equal layout, its repeated groups one object again, with equal digests, written back to
    the same bytes."""
    assert (S.Part, S.Row, S.Group) == (L.Part, L.Row, L.Group)
    paths = tmp_path / "layout.json", tmp_path / "again.json"
    paths[0].write_text(json.dumps(S._layout_json(layout), separators=(",", ":")) + "\n")
    back = S.load(paths[0])
    assert type(back) is S.CommitShape and back == layout
    assert len({id(g) for g in back.groups}) == len({id(g) for g in layout.groups})
    assert back.digest == layout.digest and back.json_digest == layout.json_digest
    assert S._layout_json(back) == L.to_json(back)
    paths[1].write_text(json.dumps(S._layout_json(back), separators=(",", ":")) + "\n")
    assert paths[1].read_bytes() == paths[0].read_bytes()
    return back


def test_the_edge_layout_round_trips_to_the_fixtures_digest_and_roots(tmp_path: Path) -> None:
    fx = json.loads(S.FIXTURE.read_text())
    layout = S.edge_layout()
    assert S._layout_json(layout) == fx["layout"]
    back = _round_trip(layout, tmp_path)
    assert back.digest.hex() == fx["layout_digest"]
    slots = fx["slots"]
    out = [[np.frombuffer(bytes.fromhex(t), np.uint8).reshape(slots, -1) for t in g] for g in fx["outputs"]]
    y = np.stack([np.frombuffer(bytes.fromhex(h), np.uint8).reshape(-1, 192) for h in fx["salts"]])
    c = S.commit(bytes.fromhex(fx["context"]), back, fx["step"], y, out, slots)
    assert c.root.hex() == fx["root"] and [r.hex() for r in c.slot_roots] == [d["root"] for d in fx["per_slot"]]


def test_families_shared_layout_round_trips_to_its_digest_and_roots(tmp_path: Path) -> None:
    """Skipped where the layout was not fetched: the suites' runner fetches it before this suite, a bare pytest in a
    fresh clone (gpu.sh on a GPU node) only after `research data fetch-fixtures`."""
    if not FAMILIES.is_file():
        pytest.skip(f"{LAYOUTS} is a fetched fixture this checkout does not hold, and the evidence store is not reachable "
                    f"from a test: `research data fetch-fixtures {LAYOUTS}` fetches it (tools/check/suites.py does "
                    "before this suite)")
    raw = FAMILIES.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FAMILIES_SHA256
    back = _round_trip(S.load(FAMILIES), tmp_path)
    assert (tmp_path / "layout.json").read_bytes() == raw
    assert len(back.groups) == 54 and len({id(g) for g in back.groups}) == 3 and back.unit_count == back.row_count == 21_857
    assert back.digest.hex() == FAMILIES_DIGEST and back.json_digest.hex() == FAMILIES_JSON_DIGEST
    out = S.random_outputs(back, 2, np.random.default_rng(20261027))
    assert S.commit(CTX, back, 5, salts_of(back, 2), out, 2).root.hex() == FAMILIES_ROOT


def test_the_h200_layout() -> None:
    h = S.h200_layout()
    g = h.groups[0]
    assert len(h.groups) == 52 and all(x is g for x in h.groups)
    assert len(g.units) == 25 and h.unit_count == 1300
    assert all(sum(r.bits for r in u) <= S.UNIT_MAX_BITS for u in g.units)
    assert sum(g.sources) == sum(r.nbytes for u in g.units for r in u) == 663_824
    assert h.slot_bytes * 20480 == 706_946_007_040
