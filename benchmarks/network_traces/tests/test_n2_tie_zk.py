"""n2_tie_zk's members at the pinned size class: each bucket's output word by IoTieRequest's reference, and whether the
prover's ``pub`` row is the verifier's own, give the member's verdicts (`EXPECT`): honest tied; carried, IoTieRequest
accepting bucket 4 on an added carried field count whose row isn't the verifier's; forged and split, refused. Each
witness row is whole u16 words, and the scan's blobs are the warden's held cells and the rows' salts, never an empty
slot. Nothing here lowers or proves (`n2_tie_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_tie_zk", HERE / "n2_tie_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tz = _load()


def _tie():
    return tz.TS.IngressTie(tz.LINK, tz.PARAMS, tz.SESSION)


def test_each_members_references_give_its_verdicts():
    tie = _tie()
    assert tz.HC.pinned(tie.gates()[0]) and list(tz.EXPECT) == list(tz.MEMBERS)
    for name, (link, adv, edit, _) in tz.members().items():
        rd = tz.link_windows(tie, link, adv, edit)
        assert len(rd.windows) == 2 and all(len(x.words) == tz.T for x in rd.windows), name
        assert tuple(bool(x.expect) for x in rd.windows) == tz.EXPECT[name], name
        assert all(len(tie.row(a, r)) % 2 == 0 for x in rd.windows for a in x.args for r in tz.TS.ROWS)
    carried = tz.link_windows(tie, *tz.members()["carried"][:3])
    assert [x.words for x in carried.windows] == [[1] * tz.T] * 2
    assert [x.opens for x in carried.windows] == [[True] * tz.T, [False] + [True] * (tz.T - 1)]
    forged = tz.link_windows(tie, *tz.members()["forged"][:3])
    assert all(x.own is not None and 0 in x.words for x in forged.windows)


def test_the_served_class_is_one_window_of_60_buckets_of_one_4096_byte_cell_honest_tied_and_forged_refused():
    sv = _load()
    sv.use("served")
    tie = sv.TS.IngressTie(sv.LINK, sv.PARAMS, sv.SESSION)
    assert tie.key == (4096, 1, 1) and sv.T == 60 and sv.HC.pinned(tie.gates()[0])
    assert list(sv.members()) == list(sv.EXPECT) == ["honest", "forged"] and f"pub-{sv.T}.bin" in sv.RECEIVED
    for name, (link, adv, edit, _) in sv.members().items():
        rd = sv.link_windows(tie, link, adv, edit)
        assert len(rd.windows) == 1 and len(rd.windows[0].words) == 60 and all(rd.windows[0].opens), name
        assert tuple(bool(x.expect) for x in rd.windows) == sv.EXPECT[name], name
    assert tz.T == 4 and tz.B == 512


def test_a_warden_runs_release_is_its_cells_and_its_forged_row_is_the_registered_reads_to_refuse():
    """`warden_link`'s reading of a warden's cells (`_released`) gives the release and prompt lengths the warden's own
    ingress served, and `forge`'s row, an id request of the same prompt length, keeps every output word 1 while the
    registered read refuses it, in its window alone."""
    tie = _tie()
    link = tz.served(tz.SPAN, tz.ARRIVALS, tz.RATES)
    cells = [tz.WR.Cells.of(tz.LINK, tz.Direction.INGRESS, rec, tz.PARAMS, tz.SESSION) for rec in link.records]
    releases, lens = tz._released(tz.PARAMS, cells)
    assert releases == list(link.releases) and lens == tz.SPAN
    real = SimpleNamespace(cells=cells, records=link.records, releases=releases)
    f = tz.forge(real)
    rd = tz.link_windows(tie, real, tz.advice(lens, tz.received(real)), f)
    assert f.window == 0 and rd.windows[0].rows[f.bucket] != cells[0].rows[f.bucket]
    assert [x.words for x in rd.windows] == [[1] * tz.T] * 2 and all(all(x.opens) for x in rd.windows)
    assert "does not open" in rd.windows[0].held and rd.windows[1].held is None
    assert [x.expect for x in rd.windows] == [0, 1]
    honest = tz.link_windows(tie, real, tz.advice(lens, tz.received(real)))
    assert [x.expect for x in honest.windows] == [1, 1] and all(x.held is None for x in honest.windows)


def test_the_scan_looks_for_held_cells_and_salts_alone():
    tie = _tie()
    link, adv, _, _ = tz.members()["honest"]
    rd = tz.link_windows(tie, link, adv)
    blobs, held = tz.blobs_of(rd)
    cells = [row[i:i + tz.B] for x in rd.windows for row in x.cells.rows for i in range(0, len(row), tz.B)]
    assert sorted(blobs[k] for k in held) == sorted(c for c in cells if c != bytes(tz.B))
    assert held and sum(1 for k in blobs if k.startswith("salt")) == len(rd.windows) * tz.T
