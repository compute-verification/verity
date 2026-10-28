"""The adapter from the real certifier's records to `datacenter.observe.Record`s: a record's port is its block's
certifier in the direction of travel, its counts are its grid's occupancy, and its digest binds the status and the real
frames; the certifiers' parameters make the declaration's cadence; and on an injected clock an honest run publishes exactly
the records the Program declares, which `datacenter.check_records` carries, while a block that holds a session late or
writes an extra event is refused in the window it cheated in."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration, grid
from verity.core.protocols.certifier.grid import GridParams, Reason, Status

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("network_sim_mockdc", HERE / "mockdc.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        yield mod


@pytest.fixture(scope="module")
def setup(m):
    report = calibration.calibrate(m.trace())
    return report, m.ar.Setup(report)


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
    from datacenter import Port

    out = Port("b0.cert", "egress")
    frames = [bytes([k]) * 8 for k in range(3)]
    r = m.records.to_record(window(frames, [2, 1, 0]), out)
    assert (r.port, r.window, r.counts, len(r.digest)) == (out, 0, (2, 1, 0), 64)
    assert m.records.to_record(window(frames, [2, 1, 0]), out) == r
    moved = m.records.to_record(window(frames, [1, 2, 0]), out)
    assert moved.counts == (1, 2, 0) and moved.digest == r.digest  # the counts, not the digest, fix the split
    other = m.records.to_record(window([frames[0], frames[1], b"\xff" * 8], [2, 1, 0]), out)
    assert other.counts == r.counts and other.digest != r.digest
    failed = m.records.to_record(window(frames, [2, 1, 0], Status(Reason.COVERAGE_NEVER_ESTABLISHED)), out)
    assert failed.counts == r.counts and failed.digest != r.digest


def test_the_certifiers_parameters_are_the_declarations_cadence_and_the_ports_its_uplinks(m, setup):
    from datacenter import Cadence, Port

    _, s = setup
    assert m.records.cadence(s) == Cadence(frame_bits=4096 * 8, bucket_ticks=100, buckets=60, jitter_classes=16)
    assert m.records.ports(m.leaf_spine(2), "b1", s) == {
        s.egress_link: Port("b1.cert", "egress"), s.ingress_link: Port("b1.cert", "ingress")}
    with pytest.raises(ValueError, match="frame size, bucket or window"):
        m.records.cadence(type("S", (), {"egress_params": s.egress_params, "sync_set": s.sync_set,
                                      "ingress_params": GridParams(buckets=30, allocation=1, frame_bytes=4096)})())


def test_on_an_injected_clock_honest_publishes_the_declared_records_and_cheating_is_refused(m, setup, tmp_path):
    from datacenter import Carried, Interval, Refused, check_records

    report, s = setup
    doc = m.trace()
    honest = m.ar.plan_configuration(doc["configurations"][0], s)
    dc = m.leaf_spine(1, cadence=m.records.cadence(s))
    declared = m.records.declared(honest, s, dc, "b0")
    windows = honest["windows"]
    assert sorted(declared, key=repr) == sorted(((p, w) for p in dc.ports for w in range(windows)), key=repr)
    claims = {}
    for strategy in ("honest", "late", "extra"):
        plan = m.strategy_plan(honest, s, strategy)
        assert m.ar.run_virtual(plan, report, tmp_path / strategy) == {"server": 0, "proxy": 0, "peer": 0}
        published = m.records.published(tmp_path / strategy, s, dc, "b0")
        if strategy == "honest":
            assert {(r.port, r.window): r for r in published} == declared
        claims[strategy] = check_records(published, declared, dc.cadence)
    assert claims["honest"] == [Carried(p, math.log2(16), Interval(w * 6000, (w + 1) * 6000)) for p, w in declared]
    refused = {k: [(repr(c.on), c.why) for c in v if isinstance(c, Refused)] for k, v in claims.items()}
    both = ["occupancy differs from the declared profile", "a frame is not a committed crossing"]
    assert refused["late"] == [("port b0.cert:out [0, 6000)", why) for why in both]
    assert refused["extra"] == [(f"port b0.cert:out [{(windows - 1) * 6000}, {windows * 6000})", why) for why in both]
    missing = check_records([], declared, dc.cadence)
    assert {c.why for c in missing} == {"no record: the certifier was down or its record was withheld"}
    assert len(missing) == len(declared)


def test_a_block_down_for_maintenance_is_refused_without_the_plan_and_charged_what_it_holds_with_it(m, setup, tmp_path,
                                                                                                     capsys):
    """Under `maintenance` the block's certifier publishes nothing for window 1. Declared from its drained plan, its
    records are the ones the verifier's plan with `Maintenance(b0, window 1)` declares, which leave that window out:
    the check carries the four others and the account is sound, charging everything the block's devices hold. Without
    the plan both port-windows of window 1 are refused "no record", each charged at the uplink's full capacity. The
    certifier's audit has no verdict for window 1. An honest block beside it is charged its timing advice only."""
    from datacenter import Carried, Interval, Maintenance, Plan, check_records

    report, s = setup
    honest = m.ar.plan_configuration(m.trace()["configurations"][0], s)
    dc = m.leaf_spine(2, cadence=m.records.cadence(s))
    plan = m.strategy_plan(honest, s, "maintenance")
    assert plan["strategy"] == "maintenance" and plan["outage"]["window"] == m.OUTAGE_WINDOW == 1
    verifier = m.verifier_plan(dc, {"b0": plan, "b1": honest})
    assert verifier == Plan(Interval(0, 18000), maintenance=[Maintenance("b0", Interval(6000, 12000))])
    assert m.ar.run_virtual(plan, report, tmp_path / "b0") == {"server": 0, "proxy": 0, "peer": 0}
    assert m.ar.run_virtual(honest, report, tmp_path / "b1") == {"server": 0, "proxy": 0, "peer": 0}
    published = {(r.port, r.window): r for r in m.records.published(tmp_path / "b0", s, dc, "b0")}
    excused = m.records.declared(plan, s, dc, "b0", verifier=verifier)
    assert published == excused and sorted({w for _, w in excused}) == [0, 2]
    assert all(isinstance(c, Carried) for c in check_records(list(published.values()), excused, dc.cadence))

    summary = m.audit(tmp_path, dc, honest, {"b0": plan, "b1": honest}, report, {"b0": "maintenance", "b1": "honest"})
    assert summary["plan"] == {"epoch": [0, 18000], "maintenance": [["b0", [6000, 12000]]]}
    other = summary["blocks"]["b1"]["datacenter"]
    assert other["accepted"] and "unplanned" not in other
    assert (other["account"]["covert_bits"], other["account"]["serviced_bits"]) == (24.0, 0.0)
    entry = summary["blocks"]["b0"]
    assert [(link, w, raw) for link, w, raw, _ in entry["verdicts"] if raw] == \
        [("responses", 1, "no-verdict"), ("requests", 1, "no-verdict")]
    held = sum(float(d.memory) for d in dc.block("b0").devices if d.memory is not None)
    line_rate = float(dc.uplink("b0.cert").capacity) * 6.0
    planned, bare = entry["datacenter"], entry["datacenter"]["unplanned"]
    assert planned["accepted"] and planned["refused"] == [] and planned["carried"] == 4
    assert planned["account"] == {"sound": True, "covert_bits": 16.0 + held, "timing_bits": 16.0, "refused_bits": 0.0,
                                  "serviced_bits": held, "carried_bits": held}
    assert not bare["accepted"] and bare["refused"] == [
        f"port b0.cert:{d} [6000, 12000): no record: the certifier was down or its record was withheld"
        for d in ("out", "in")]
    assert bare["account"] == {"sound": False, "covert_bits": 16.0 + 2 * line_rate, "timing_bits": 16.0,
                               "refused_bits": 2 * line_rate, "serviced_bits": 0.0, "carried_bits": held}
    assert "against_honest" not in entry
    line = capsys.readouterr().out
    assert "datacenter records without the plan REFUSED" in line and "with Maintenance(b0, [6000, 12000)) ACCEPTED" in line


def test_a_fleet_is_validated_and_stood_up_merged(m, tmp_path):
    """`--fleet` is `two_sites`' fleet merged, its WAN as long as `--wan-latency-us`; a fleet declaration is merged
    too, and one whose inter-site link leaves a block past its certifier is refused. A shape without the fleet's
    options, as `workload.lab_run.shape_of` builds it, is the default leaf-spine."""
    import argparse

    lab_run = dict(declaration=None, blocks=4, leaves=2, spines=1, fabric_latency_us=2000, fabric_jitter_us=500)
    default = m.lab(argparse.Namespace(**lab_run))
    assert default.name == "leaf-spine-4" and len(m.plan_of(argparse.Namespace(**lab_run), default).namespaces) == 12
    shape = dict(declaration=None, fleet=True, blocks=1, leaves=1, spines=0, fabric_latency_us=2000,
                 wan_latency_us=30_000)
    dc = m.lab(argparse.Namespace(**shape))
    assert dc.name == "east+west" and [b.name for b in dc.blocks] == ["east.b0", "west.b0"]
    assert [(e.a, e.b, e.delay) for e in dc.links if e.kind == "wan"] == [("east.leaf0", "west.leaf0", 0.03)]
    path = tmp_path / "fleets.py"
    path.write_text("from datacenter import Fleet, Link\n"
                    "from datacenter.examples import two_sites\n"
                    "def bad():\n"
                    "    f = two_sites(1)\n"
                    "    return Fleet(f.datacenters, [Link('east.b0.nic0', 'west.leaf0', f.links[0].capacity)])\n")
    assert m.lab(argparse.Namespace(**shape | {"declaration": f"{path}:two_sites"})).name == "east+west"
    with pytest.raises(SystemExit, match="link east.b0.nic0-west.leaf0 passes no certifier"):
        m.lab(argparse.Namespace(**shape | {"declaration": f"{path}:bad"}))


def test_a_fleets_blocks_are_each_audited_and_its_records_checked_on_every_port_of_the_merged_fleet(m, setup, tmp_path,
                                                                                                   capsys):
    """On the merged `two_sites` fleet, a late block on the east site and an honest one on the west each get both
    audits, and one record check on every block's records together covers the fleet's four ports, the WAN between the
    spines having none: it refuses what the late block's own check refuses and carries the rest, and the account
    charges the refused window at the east block's uplink capacity."""
    report, s = setup
    honest = m.ar.plan_configuration(m.trace()["configurations"][0], s)
    dc = m.two_sites(1, cadence=m.records.cadence(s)).merged()
    plans = {"east.b0": m.strategy_plan(honest, s, "late"), "west.b0": honest}
    for b, plan in plans.items():
        assert m.ar.run_virtual(plan, report, tmp_path / b) == {"server": 0, "proxy": 0, "peer": 0}
    summary = m.audit(tmp_path, dc, honest, plans, report, {"east.b0": "late", "west.b0": "honest"})
    assert summary["declaration"] == "east+west"
    assert summary["wan"] == [["east.spine0-west.spine0", "wan/east-west=1e+11 bit/s (declared)",
                               "wan/east-west-latency=0.02 s (declared)"]]
    east, west = summary["blocks"]["east.b0"], summary["blocks"]["west.b0"]
    both = ["occupancy differs from the declared profile", "a frame is not a committed crossing"]
    refused = [f"port east.b0.cert:out [0, 6000): {why}" for why in both]
    assert not east["accepted"] and east["datacenter"]["refused"] == refused
    assert east["against_honest"]["wire_is_honests"] is False
    assert west["accepted"] and west["datacenter"]["accepted"] and west["datacenter"]["carried"] == 6
    fleet = summary["records"]
    assert fleet["ports"] == ["east.b0.cert:in", "east.b0.cert:out", "west.b0.cert:in", "west.b0.cert:out"]
    assert fleet["of"] == len(dc.ports) == 4 and fleet["refused"] == refused and fleet["carried"] == 11
    line_rate = float(dc.uplink("east.b0.cert").capacity) * 6.0
    held = sum(x["datacenter"]["account"]["carried_bits"] for x in (east, west))
    assert fleet["account"] == {"sound": False, "covert_bits": 44.0 + line_rate, "timing_bits": 44.0,
                                "refused_bits": line_rate, "serviced_bits": 0.0, "carried_bits": held}
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == (f"east+west: datacenter records on 4 of its 4 ports REFUSED (port east.b0.cert:out [0, 6000)), "
                       f"account NOT sound, covert budget {44.0 + line_rate:.3g} bits ({line_rate:.3g} on refused "
                       f"windows)")


def test_a_salted_block_publishes_the_records_declared_with_its_revealed_salts(m, setup, tmp_path, capsys):
    """With the certifier's salt on, each ingress window holds one more frame, its first bucket's, and egress is the
    unsalted declaration's. The records declared with the salts the proxy revealed are what it published, and
    `mockdc.audit` accepts the block on both audits and its salt check accepts every window; declared without them, the
    ingress windows are refused."""
    from datacenter import Refused, check_records

    report, s = setup
    honest = m.ar.plan_configuration(m.trace()["configurations"][0], s)
    dc = m.leaf_spine(1, cadence=m.records.cadence(s))
    assert m.ar.run_virtual(honest, report, tmp_path / "b0", salt=True) == {"server": 0, "proxy": 0, "peer": 0}
    salts = {int(w): bytes.fromhex(x) for w, x in m.json.loads((tmp_path / "b0" / "salts.json").read_text()).items()}
    published = {(r.port, r.window): r for r in m.records.published(tmp_path / "b0", s, dc, "b0")}
    salted, plain = m.records.declared(honest, s, dc, "b0", salts), m.records.declared(honest, s, dc, "b0")
    assert published == salted
    for (port, w), r in salted.items():
        if port.direction == "egress":
            assert r == plain[port, w]
        else:
            assert r.counts[0] == plain[port, w].counts[0] + 1 and r.counts[1:] == plain[port, w].counts[1:]
    both = ["occupancy differs from the declared profile", "a frame is not a committed crossing"]
    assert [(repr(c.on), c.why) for c in check_records(list(published.values()), plain, dc.cadence)
            if isinstance(c, Refused)] == \
        [(f"port b0.cert:in [{w * 6000}, {(w + 1) * 6000})", why) for w in range(honest["windows"]) for why in both]

    summary = m.audit(tmp_path, dc, honest, {"b0": honest}, report, {"b0": "honest"}, salt=True)
    entry = summary["blocks"]["b0"]
    assert entry["accepted"] and entry["datacenter"]["accepted"]
    assert entry["salt"] == {"accepted": True, "violations": [], "ingress_salt_frames": [1, 1, 1],
                             "egress_salt_frames": 0, "egress_holds_salt": False, "block_received": True}
    assert "salt frames ingress [1, 1, 1] egress 0, salt check ACCEPTED, block received True" in capsys.readouterr().out


def test_a_salted_block_down_for_maintenance_owes_no_salt_in_its_outage(m, setup, tmp_path):
    """A salted block down for window 1 reveals the salts of windows 0 and 2 only. Its records declared with them are
    what it published, so the record check under the verifier's plan carries them; the certifier's salt check finds
    no record and no salt for window 1, as its audit has no verdict there."""
    report, s = setup
    honest = m.ar.plan_configuration(m.trace()["configurations"][0], s)
    dc = m.leaf_spine(1, cadence=m.records.cadence(s))
    plan = m.strategy_plan(honest, s, "maintenance")
    assert m.ar.run_virtual(plan, report, tmp_path / "b0", salt=True) == {"server": 0, "proxy": 0, "peer": 0}
    assert sorted(m.json.loads((tmp_path / "b0" / "salts.json").read_text())) == ["0", "2"]

    summary = m.audit(tmp_path, dc, honest, {"b0": plan}, report, {"b0": "maintenance"}, salt=True)
    entry = summary["blocks"]["b0"]
    assert entry["datacenter"]["accepted"] and entry["datacenter"]["carried"] == 4
    assert [x.split(": ", 1)[1] for x in entry["datacenter"]["unplanned"]["refused"]] == \
        ["no record: the certifier was down or its record was withheld"] * 2
    assert entry["salt"] == {"accepted": False, "violations": [[1, "salt"]], "ingress_salt_frames": [1, 1],
                             "egress_salt_frames": 0, "egress_holds_salt": False, "block_received": True}
