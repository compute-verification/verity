"""The adapter from the real certifier's records to `datacenter.observe.Record`s, and their audit: a record's counts are
its grid's occupancy and its digest binds the status and the real frames; the audit gives `claims.audit`'s claims and
reasons; and on an injected clock an honest run publishes exactly the records the Program declares, while a block that
holds a session late or writes an extra event is refused in the window it cheated in."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration, grid
from verity.core.protocols.certifier.grid import GridParams, Reason, Status

HERE = Path(__file__).resolve().parents[1]
UP = ("b0.nic0", "b0.cert")


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("network_sim_mockdc", HERE / "mockdc.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        yield mod


def window(frames: list[bytes], per_bucket: list[int], status: Status = grid.COMPLETE) -> grid.Record:
    c = grid.Certifier("responses", GridParams(buckets=len(per_bucket), allocation=2, frame_bytes=8))
    k = 0
    for n in per_bucket:
        for f in frames[k : k + n]:
            c.enqueue(f)
        k += n
        c.tick()
    r = c.close_window()
    return grid.Record(r.link, r.window, r.grid, status)


def test_a_record_is_the_grids_occupancy_and_a_digest_of_its_status_and_real_frames(m):
    rec = m.records
    frames = [bytes([k]) * 8 for k in range(3)]
    r = rec.to_record(window(frames, [2, 1, 0]), UP, "b0.cert")
    assert (r.link, r.window, r.counts, r.certifier, len(r.digest)) == (UP, 0, (2, 1, 0), "b0.cert", 64)
    assert rec.to_record(window(frames, [2, 1, 0]), UP, "b0.cert") == r
    moved = rec.to_record(window(frames, [1, 2, 0]), UP, "b0.cert")
    assert moved.counts == (1, 2, 0) and moved.digest == r.digest  # the counts, not the digest, fix the split
    other = rec.to_record(window([frames[0], frames[1], b"\xff" * 8], [2, 1, 0]), UP, "b0.cert")
    assert other.counts == r.counts and other.digest != r.digest
    failed = rec.to_record(window(frames, [2, 1, 0], Status(Reason.COVERAGE_NEVER_ESTABLISHED)), UP, "b0.cert")
    assert failed.counts == r.counts and failed.digest != r.digest


def test_the_audit_carries_the_declared_windows_and_refuses_with_claims_reasons(m):
    from datacenter.claims import Carried, Refused
    from datacenter.observe import Record

    rec = m.records
    d0, d1 = Record(UP, 0, (2, 1, 0), b"a" * 64, "b0.cert"), Record(UP, 1, (0, 0, 1), b"b" * 64, "b0.cert")
    expected = [d0, d1]
    assert rec.audit(expected, expected, syncs=16) == [Carried(UP, 0, 4.0), Carried(UP, 1, 4.0)]
    where = f"link {UP} window 1"
    assert rec.audit([d0], expected, 16)[0] == Refused(where, rec.NO_RECORD)
    assert Refused(where, rec.OCCUPANCY) in rec.audit([d0, Record(UP, 1, (1, 0, 0), d1.digest, "b0.cert")], expected, 16)
    assert Refused(where, rec.CONTENT) in rec.audit([d0, Record(UP, 1, d1.counts, b"c" * 64, "b0.cert")], expected, 16)
    extra = Record(UP, 2, (0, 0, 0), b"d" * 64, "b0.cert")
    assert Refused(f"link {UP} window 2", rec.OCCUPANCY) in rec.audit([*expected, extra], expected, 16)
    assert rec.profile(expected) == {UP: {0: 2, 1: 1, 5: 1}}


def test_the_records_name_the_blocks_uplink_in_each_direction_of_travel(m):
    dc = m.graph.leaf_spine(2)
    report = calibration.calibrate(m.trace())
    setup = m.ar.Setup(report)
    ends, cert = m.records.directions(dc, "b1", setup)
    assert ends == {setup.egress_link: ("b1.nic0", "b1.cert"), setup.ingress_link: ("b1.cert", "b1.nic0")}
    assert cert == "b1.cert"


def test_on_an_injected_clock_honest_publishes_the_declared_records_and_cheating_is_refused(m, tmp_path):
    from datacenter.claims import Carried, Refused

    dc = m.graph.leaf_spine(1)
    doc = m.trace()
    report = calibration.calibrate(doc)
    setup = m.ar.Setup(report)
    honest = m.ar.plan_configuration(doc["configurations"][0], setup)
    expected = m.records.declared(honest, setup, dc, "b0")
    syncs = len(setup.sync_set.deltas) * len(setup.sync_set.rhos)
    windows = honest["windows"]
    assert len(expected) == 2 * windows
    refused = {}
    for strategy in ("honest", "late", "extra"):
        plan = m.strategy_plan(honest, setup, strategy)
        assert m.ar.run_virtual(plan, report, tmp_path / strategy) == {"server": 0, "proxy": 0, "peer": 0}
        published = m.records.published(tmp_path / strategy, setup, dc, "b0")
        claims = m.records.audit(published, expected, syncs)
        if strategy == "honest":
            assert sorted(published, key=lambda r: (r.link, r.window)) == sorted(expected, key=lambda r: (r.link, r.window))
            assert claims == [Carried(r.link, r.window, math.log2(syncs)) for r in published]
        refused[strategy] = [c for c in claims if isinstance(c, Refused)]
    egress = ("b0.nic0", "b0.cert")
    assert refused["honest"] == []
    assert refused["late"] == [Refused(f"link {egress} window 0", m.records.OCCUPANCY)]
    assert refused["extra"] == [Refused(f"link {egress} window {windows - 1}", m.records.OCCUPANCY)]
