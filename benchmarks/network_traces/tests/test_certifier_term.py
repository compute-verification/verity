"""leak_bounds' scaling by requests, which `verity bench certifier_term` prints: the network certifier's terms at the
load of a node, the reference datacenter and the world, each check's by its own rule.

- the scan's terms are the rows form's and N2's the clock form's, held here to the Lean's `clog2` on a hand-worked
  execution phase, and the table to a hand-worked world of three machines, in exact arithmetic;
- each of the three changes zeroes what it should and leaves the rest, the failed connections' pooled charge bounds
  the count it states, and the design's term is the jitter and that charge;
- nothing a check counts by capacity scales with the requests;
- the world, the datacenter, the requests and the weights are read from the files, each input marked with its source;
- the command prints the table and its JSON (value, unit, commit, run), through `verity bench`'s runner too."""

from __future__ import annotations

import json
import math
import runpy
import sys
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import leak_bounds as LB  # noqa: E402
from verity.protocols.certifier.prover import ingress_scan as IS  # noqa: E402

#: a link's execution phases of 128 s over a week, a month and a year
PHASES = {"week": 4_725, "month": 20_250, "year": 246_375}
NAMES = ("scan", "N2", "lengths", "timing", "entries", "all three", "design")


def clog2(x: int) -> int:
    """The Lean's `clog2`: if x ≤ 1 then 0 else Nat.log2 (x - 1) + 1."""
    return 0 if x <= 1 else (x - 1).bit_length()


def small(**given) -> dict:
    """The inputs of a world of three machines of 8 GPUs and a datacenter of two, where a link's execution phase, 32
    buckets of 4 s, serves 10 requests of at most 1,000 ids: 3 × 246,375 × 10 requests a year."""
    inp = LB.inputs(**{"response_ids": 1_000, "buckets": 32, "bucket_s": 4,
                       "world_requests": 3 * PHASES["year"] * 10} | given)
    sizes = {"world_machines": 3, "datacenter_machines": 2, "machine_gpus": 8}
    return inp | {k: inp[k]._replace(value=v) for k, v in sizes.items()}


def rows(inp: dict) -> dict:
    return {(r.scope.name, r.period, r.name): r for r in LB.table(inp)}


def test_the_scans_terms_are_the_rows_forms_on_a_hand_worked_execution_phase():
    """4 connections of at most 65 cells and one scan of 32 buckets: ⌈log2 66⌉ = 7 bits a connection, charged for
    its schedule and again for its fail cell, and ⌈log2 D_j^32⌉ of jitter, which is 0, 32 and ⌈32·log2 3⌉ = 51."""
    assert clog2(66) == 7 == (65).bit_length() and [clog2(x) for x in (0, 1, 2, 3, 4, 5)] == [0, 0, 1, 2, 2, 3]
    for dj, jitter in ((1, 0), (2, 32), (3, 51)):
        assert jitter == clog2(dj ** 32)
        got = LB.terms(4, 1, cells=65, steps=32, dj=dj, entries=9_088)
        assert got == LB.Terms(schedules=28, cells=28, jitter=jitter, entries=9_088)
        assert got.certifier == 56 + jitter
    assert LB.terms(0, 0, cells=65, steps=32, dj=2) == LB.Terms(0, 0, 0, 0)
    assert LB.terms(Fraction(7, 2), Fraction(1, 4), cells=1_001, steps=32, dj=2).certifier == 35 + 35 + 8


def test_n2s_terms_are_the_clock_forms_on_a_hand_worked_link():
    """One link of 5 execution phases under 128 syncs: ⌈log2 128^5⌉ = 35 bits of schedules, no fail cell, and the
    jitter a step a bucket. The schedules are a ceiling over a link's phases, not a phase's ceiling summed."""
    assert LB.clock_terms(1, 5, syncs=128, steps=32, dj=2) == LB.Terms(35, 0, 160, None)
    assert LB.clock_terms(3, 5, syncs=128, steps=32).certifier == 105
    assert LB.clock_terms(1, 5, syncs=1, steps=32) == LB.Terms(0, 0, 0, None)
    for syncs, phases in ((3, 5), (3, 4_725), (100, 7), (129, 1)):
        assert LB.clock_terms(1, phases, syncs=syncs, steps=1).schedules == clog2(syncs ** phases)
    assert LB.clock_terms(1, 5, syncs=3, steps=1).schedules == 8 < 5 * clog2(3)
    assert LB.clock_terms(2, Fraction(9, 2), syncs=128, steps=32, dj=2) == LB.Terms(2 * 32, 0, 2 * 144, None)


def test_the_failed_connections_pooled_charge_bounds_the_count_it_states():
    """Of 8 connections of at most 7 cells, at most 2 fail in 1 + 8·8 + 28·64 = 1,857 ways, 10.9 bits; the charge is
    2·(log2(8e/2) + log2 8) = 10 + 2·log2 e, and no failure is charged nothing."""
    assert sum(math.comb(8, j) * 8 ** j for j in range(3)) == 1_857
    assert LB.failed(8, 2, 7) == pytest.approx(10 + 2 * math.log2(math.e)) and LB.failed(8, 2, 7) >= math.log2(1_857)
    assert LB.failed(8, 0, 7) == 0 == LB.failed(0, 0, 7)
    for n in (1, 5, 8, 60):
        for c in (1, 7, 64, 4_097):
            charged = [LB.failed(n, f, c) for f in range(n + 1)]
            assert charged == sorted(charged) and charged[-1] == pytest.approx(n * math.log2(c + 2))
            for f in range(1, n + 1):
                ways = sum(math.comb(n, j) * (c + 1) ** j for j in range(f + 1))
                assert charged[f] >= math.log2(ways) - 1e-9
    held = LB.terms(8, 1, cells=7, steps=32, dj=2, changes=["lengths"], failures=2)
    assert held == LB.Terms(0, LB.failed(8, 2, 7), 32, 0)
    assert LB.terms(8, 1, cells=7, steps=32, failures=2) == LB.Terms(24, 24, 0, 0)
    rate = LB.afforded(1_000.0, 10 ** 9, 4_097)
    assert 0 < rate < 1 and LB.failed(10 ** 9, 10 ** 9 * rate, 4_097) == pytest.approx(1_000.0)
    assert LB.afforded(1e12, 10 ** 9, 4_097) == 1.0 == LB.afforded(1.0, 0, 4_097)


def test_the_assumed_class_has_the_scans_widths():
    """A response of at most KO ids takes at most KO + 1 cells, so a connection's bits are ⌈log2(KO + 2)⌉, and a
    slot entry is its read bit, its count and its KO ids, to a whole 16-bit word."""
    cls = LB.Assumed(KO=1_000, IDB=18, T=32, tau=Fraction(4))
    assert (cls.cells, cls.cell, cls.phase_s) == (1_001, clog2(1_000 + 2), 128) and cls.cell == 10
    assert cls.entry == 18_048 == -(-(1 + 32 + 1_000 * 18) // 16) * 16
    assert all(LB.Assumed(KO=ko, IDB=18, T=32, tau=4).cell == clog2(ko + 2) for ko in range(1, 5_000))
    assert [LB.Assumed(KO=ko, IDB=18, T=32, tau=4).cell for ko in (875, 1_000, 4_096, 4_500)] == [10, 10, 13, 13]
    assert LB.Assumed(KO=4_096, IDB=18, T=1_280, tau=Fraction(1, 10)).entry == 73_776
    scan = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=64, KI=1024)
    same = LB.Assumed(KO=64, IDB=17, T=32, tau=4)
    assert same.entry == IS.entry_bits(scan) == 1_136 and LB.EVERY == scan.EVERY == 16
    assert same.cell == LB.link(scan, 1, T=32).cell == 7


def test_the_table_on_a_hand_worked_world():
    """10 requests a link's execution phase, 10 bits a connection, 18,048 bits a slot entry, 32 buckets a phase, 7
    bits of N2's schedules a phase."""
    got = rows(small())
    assert len(got) == 3 * 3 * 7 and tuple(LB.ROWS) == NAMES and {k[2] for k in got} == set(NAMES)
    for scope, machines in (("node", 1), ("datacenter", 2), ("world", 3)):
        for period, n in PHASES.items():
            r, n2 = got[scope, period, "scan"], got[scope, period, "N2"]
            phases, requests = machines * n, machines * n * 10
            assert (r.scope.machines, r.scope.gpus, r.phases, r.requests) == (machines, 8 * machines, phases, requests)
            assert r.eps == phases * Fraction(1, 2 ** 20)
            for dj, jitter in ((1, 0), (2, 32 * phases)):
                assert r.terms[dj] == LB.Terms(10 * requests, 10 * requests, jitter, 18_048 * requests)
                assert r.terms[dj].certifier == 20 * requests + jitter
                assert n2.terms[dj] == LB.Terms(7 * phases, 0, jitter, None)
    week, year = got["node", "week", "scan"], got["world", "year", "scan"]
    assert week.terms[2] == LB.Terms(472_500, 472_500, 151_200, 852_768_000) and week.terms[2].certifier == 1_096_200
    assert year.requests == 7_391_250 and year.phases == 739_125
    assert (year.terms[1].certifier, year.terms[2].certifier) == (147_825_000, 171_477_000)
    assert year.terms[2].certifier == 3 * got["node", "year", "scan"].terms[2].certifier
    assert got["world", "year", "N2"].terms[2].certifier == 7 * 739_125 + 32 * 739_125 == 28_825_875
    assert rows(small(syncs=1))["world", "year", "N2"].terms[1].certifier == 0


def test_each_change_zeroes_what_it_should_and_leaves_the_rest():
    got = rows(small())
    for scope in LB.SCOPES:
        for period in LB.PERIODS:
            scan, all_three = got[scope, period, "scan"].terms, got[scope, period, "all three"].terms
            lengths, timing, entries, design = (got[scope, period, c].terms
                                                for c in ("lengths", "timing", "entries", "design"))
            for dj in LB.JITTERS:
                assert scan[dj].schedules == scan[dj].cells > 0 and scan[dj].entries > 0
                assert lengths[dj] == scan[dj]._replace(schedules=0, cells=0)
                assert timing[dj] == scan[dj]._replace(jitter=0)
                assert entries[dj] == scan[dj]._replace(entries=None)
                assert entries[dj].certifier == scan[dj].certifier
                assert all_three[dj] == LB.Terms(0, 0, 0, None) and all_three[dj].certifier == 0
                assert design[dj] == lengths[dj]._replace(entries=None) == LB.Terms(0, 0, scan[dj].jitter, None)
            assert scan[1].jitter == 0 < scan[2].jitter == lengths[2].certifier == design[2].certifier
            assert timing[1] == timing[2] == scan[1] and design[1].certifier == 0


def test_the_failed_connections_are_charged_pooled_only_where_the_lengths_are_held():
    got, failing = rows(small()), rows(small(failure_rate=Fraction(1, 1_000)))
    for key, r in got.items():
        f = failing[key]
        assert (f.requests, f.phases) == (r.requests, r.phases)
        for dj in LB.JITTERS:
            if key[2] in ("lengths", "all three", "design"):
                charge = LB.failed(r.requests, r.requests / 1_000, 1_001)
                assert f.terms[dj] == r.terms[dj]._replace(cells=charge) and f.terms[dj].schedules == 0
                assert charge == pytest.approx(float(r.requests) / 1_000 * math.log2(1_000 * math.e * 1_002))
            else:
                assert f.terms[dj] == r.terms[dj]
    design, each = failing["world", "year", "design"].terms, math.log2(1_000 * math.e) + math.log2(1_002)
    assert 21.37 < each < 21.38 and design[1].certifier == pytest.approx(7_391.25 * each)
    assert design[2].certifier == pytest.approx(23_652_000 + 7_391.25 * each)


def test_the_scale_is_read_from_the_files_and_every_input_is_marked():
    inp = LB.inputs()
    assert all(i.mark in LB.MARKS and i.source for i in inp.values())
    fleet = tomllib.loads((ROOT / LB.SLOWDOWN).read_text())["fleet"]["world-h200"]
    card = json.loads((ROOT / LB.WORKLOAD).read_text())
    census = next(e for e in json.loads((ROOT / LB.ADVERSARY).read_text()) if e["id"] == "world-inference-2027")
    peak = next(e for e in json.loads((ROOT / LB.HARDWARE).read_text()) if e["id"] == LB.PEAK)
    assert inp["world_machines"].value == LB.world_links() == fleet["count"]["value"] == 46_870_000
    assert inp["datacenter_machines"].value == LB.datacenter_links() == 12_500 and inp["machine_gpus"].value == 8
    assert (inp["mfu_serve"].value, inp["mfu_serve"].mark) == (fleet["mfu_serve"]["value"], "assumed")
    assert inp["machine_gpus"].mark == inp["peak_flops"].mark == "assumed" and inp["world_machines"].mark == "modeled"
    assert inp["weights"].value == LB.WEIGHTS.bits == card["value"]["params"] * 4
    assert (inp["token_macs"].mark, inp["weights"].mark) == (card["status"], card["status"]) == ("measured", "measured")
    for name, path in (("world_machines", LB.SLOWDOWN), ("machine_gpus", LB.SLOWDOWN_HARDWARE), ("mfu_serve", LB.SLOWDOWN),
                       ("datacenter_machines", LB.DEPLOYMENTS), ("request_tokens", LB.ADVERSARY),
                       ("peak_flops", LB.HARDWARE), ("token_macs", LB.WORKLOAD), ("weights", LB.WORKLOAD)):
        assert inp[name].source.startswith(path) and (ROOT / path).is_file()
    seconds = 46_870_000 * 8 * 365 * 86_400
    flop = census["tokens_per_request"] * 2 * card["value"]["served_macs_per_token"]
    served = seconds * fleet["mfu_serve"]["value"] * peak["flop_per_second"] / flop
    assert inp["world_requests"].value == pytest.approx(served, rel=1e-12) and inp["world_requests"].mark == "modeled"
    for name in ("response_ids", "id_bits", "phase_buckets", "bucket_s", "connections", "links", "failure_rate",
                 "lanes", "syncs"):
        assert inp[name].mark == "assumed"
    assert [inp[k].value for k in ("response_ids", "id_bits", "phase_buckets", "bucket_s", "lanes", "syncs")] == [
        4_096, 18, 1_280, LB.TAU, None, 128]
    assert LB.TAU == Fraction(1, 10) and "calibration.py" in inp["syncs"].source
    assert (ROOT / "verity/protocols/certifier/verifier/calibration.py").is_file()
    given = LB.inputs(world_requests=1e16, response_ids=875, syncs=16, lanes=512)
    assert (given["world_requests"].value, given["world_requests"].mark) == (1e16, "assumed")
    assert [given[k].value for k in ("response_ids", "syncs", "lanes")] == [875, 16, 512]
    assert "--syncs" in given["syncs"].source and "--lanes" in given["lanes"].source


def test_a_scope_serves_the_worlds_requests_in_proportion_to_its_gpu_seconds():
    inp = LB.inputs(world_requests=10 ** 16)
    got = rows(inp)
    world = got["world", "year", "scan"]
    assert world.requests == 10 ** 16 and world.phases == 46_870_000 * PHASES["year"]
    assert got["node", "year", "scan"].requests == Fraction(10 ** 16, 46_870_000)
    assert got["datacenter", "week", "scan"].requests == Fraction(10 ** 16 * 12_500 * 7, 46_870_000 * 365)
    assert world.terms[1].certifier == 2 * 13 * 10 ** 16 and world.terms[2].jitter == 1_280 * world.phases
    assert world.terms[1].entries == 73_776 * 10 ** 16


def test_nothing_counted_by_capacity_scales_with_the_requests():
    """Twice the requests are twice the scan's schedules and fail cells and nothing else: the jitter and N2's
    schedules are by link, and with the lanes given so are the entries, every slot's, 512 lanes × 1,280 / 16 strides
    of 73,776 bits a link's execution phase."""
    once, twice = (rows(LB.inputs(world_requests=n * 10 ** 16, lanes=512)) for n in (1, 2))
    for key, r in once.items():
        t = twice[key]
        assert t.phases == r.phases and t.requests == 2 * r.requests
        for dj in LB.JITTERS:
            a, b = r.terms[dj], t.terms[dj]
            assert (b.jitter, b.entries) == (a.jitter, a.entries)
            assert (b.schedules, b.cells) == ((a.schedules, a.cells) if key[2] == "N2" else
                                              (2 * a.schedules, 2 * a.cells))
    world = once["world", "year", "scan"]
    assert world.terms[1].entries == world.phases * 512 * 80 * 73_776
    assert once["node", "week", "scan"].terms[1].entries == 4_725 * 512 * 80 * 73_776
    assert once["world", "year", "N2"].terms[1] == LB.Terms(7 * world.phases, 0, 0, None)
    least = rows(LB.inputs(world_requests=10 ** 16))["world", "year", "scan"].terms[1].entries
    assert least == 73_776 * 10 ** 16 < world.terms[1].entries


def test_the_document_names_its_value_unit_commit_and_run(capsys):
    doc = LB.document(small(), commit="c0ffee", run="r1")
    assert (doc["unit"], doc["commit"], doc["run"], doc["model"]) == ("GB", "c0ffee", "r1", "GLM-5.3")
    assert doc["value"] == 147_825_000 / LB.GB and "world's year" in doc["name"] and "D_j = 1" in doc["name"]
    assert doc["target"] == {"value": 300.0, "unit": "GB", "bits_a_phase": 300 * LB.GB / 739_125,
                             "bits_a_request": 300 * LB.GB / 7_391_250}
    assert doc["weights"]["value"] == LB.WEIGHTS.bits / LB.GB
    assert doc["phase"] == {"seconds": 128.0, "buckets": 32, "requests": 10.0, "connection_bits": 10, "sync_bits": 7,
                            "entry_bits": 18_048}
    assert doc["failure_rate_at_target"] == 1.0 and "one entry a request" in doc["entries_rule"]
    assert set(doc["inputs"]) == set(small()) and all(set(i) == {"value", "unit", "mark", "source"}
                                                      for i in doc["inputs"].values())
    assert list(doc["legend"]) == list(NAMES) and all(set(k) == {"standing", "counts"} for k in doc["legend"].values())
    assert [k["standing"] for k in doc["legend"].values()] == [
        "on main", "on main", *["a change to the scan"] * 4, "a design with no run behind it"]
    assert "by request" in doc["legend"]["scan"]["counts"] and "by link" in doc["legend"]["N2"]["counts"]
    by = {(r["scope"], r["period"], r["row"]): r for r in doc["rows"]}
    assert len(by) == 63 and all(r["covers"] == 1.0 and r["standing"] == doc["legend"][r["row"]]["standing"]
                                 for r in by.values())
    r = by["world", "year", "scan"]
    assert (r["machines"], r["gpu_seconds"], r["requests"], r["phases"]) == (3, 24 * 31_536_000, 7_391_250, 739_125)
    assert r["schedules"] == r["cells"] == {"bits": 73_912_500, "GB": 73_912_500 / LB.GB}
    assert r["jitter"] == {"1": {"bits": 0, "GB": 0}, "2": {"bits": 23_652_000, "GB": 23_652_000 / LB.GB}}
    assert r["term"]["2"] == {"bits": 171_477_000, "GB": 171_477_000 / LB.GB,
                              "of_target": 171_477_000 / (300 * LB.GB), "of_weights": 171_477_000 / LB.WEIGHTS.bits,
                              "bits_a_phase": 232.0, "bits_a_request": 23.2}
    assert r["entries"]["bits"] == 18_048 * 7_391_250
    assert all(by["world", "year", name]["entries"] is None for name in ("entries", "all three", "design", "N2"))
    assert by["world", "year", "N2"]["term"]["2"]["bits_a_phase"] == 39.0
    assert by["world", "year", "design"]["term"]["2"]["bits"] == 23_652_000
    assert r["eps"] == 739_125 / 2 ** 20
    json.dumps(doc)
    few = LB.document(small(), only=["world"], periods=["year", "week"])
    assert [(x["scope"], x["period"]) for x in few["rows"]] == [("world", "week")] * 7 + [("world", "year")] * 7
    assert few["value"] == doc["value"]
    text = LB.render(doc)
    assert "commit c0ffee, run r1" in text and "## The terms in GB" in text and "## The terms in bits" in text
    assert "the compute side's" in text and "| world      | year   | all three |" in text
    assert "- design (a design with no run behind it): " in text and "- N2 (on main): " in text
    assert "| lanes " in text and "| none " in text
    assert "| world | year   | all three |" in LB.render(few) and "| node" not in LB.render(few)
    assert LB.render(LB.document(small())).splitlines()[2] == "commit unknown, run none"
    slots = LB.document(small(lanes=4))
    assert "4 lanes x 2 strides" in slots["entries_rule"] and "by capacity" in slots["entries_rule"]


def test_the_command_prints_the_table_and_its_json(capsys, monkeypatch):
    assert LB.main(["--json", "--commit", "c0ffee", "--run", "r1", "--world-requests", "1e16"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc == json.loads(json.dumps(LB.document(LB.inputs(world_requests=1e16), commit="c0ffee", run="r1")))
    assert doc["value"] == 2 * 13 * 1e16 / LB.GB and len(doc["rows"]) == 63
    assert doc["target"]["bits_a_phase"] == pytest.approx(300 * LB.GB / (46_870_000 * 246_375))
    rate = doc["failure_rate_at_target"]
    assert 1e-6 < rate < 1e-5 and LB.failed(1e16, 1e16 * rate, 4_097) == pytest.approx(300 * LB.GB)
    assert LB.main(["--scope", "world", "--period", "year", "--commit", "c0ffee", "--failure-rate", "1/1000",
                    "--response-ids", "875", "--target-gb", "150", "--phase-buckets", "32", "--bucket-s", "4",
                    "--syncs", "16", "--lanes", "512"]) == 0
    text = capsys.readouterr().out
    assert "| world | year   | scan " in text and "| node " not in text and "The target is 150 GB" in text
    assert "each charged 10 bits for its schedule" in text and "| failure_rate " in text
    assert "N2's schedules are 4 bits" in text and "512 lanes x 2 strides" in text
    assert "A link's execution phase is 128 s of 32 buckets" in text
    monkeypatch.setattr(sys, "argv", ["run.py", "--json", "--commit", "c0ffee", "--run", "r1", "--world-requests",
                                      "1e16"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(ROOT / "benchmarks" / "certifier_term" / "run.py"), run_name="__main__")
    assert done.value.code == 0 and json.loads(capsys.readouterr().out) == doc


def test_refusals():
    with pytest.raises(ValueError, match="a change is one of lengths, timing, entries"):
        LB.terms(1, 1, cells=65, steps=32, changes=["length"])
    with pytest.raises(ValueError, match="D_j = 0"):
        LB.terms(1, 1, cells=65, steps=32, dj=0)
    with pytest.raises(ValueError, match="2 failed"):
        LB.terms(1, 1, cells=65, steps=32, failures=2)
    with pytest.raises(ValueError, match="egress cells are an integer from 0"):
        LB.terms(1, 1, cells=6.5, steps=32)
    with pytest.raises(ValueError, match="a positive integer of syncs"):
        LB.clock_terms(1, 1, syncs=0, steps=32)
    with pytest.raises(ValueError, match="IDB = 33"):
        LB.Assumed(KO=1, IDB=33, T=32, tau=4)
    with pytest.raises(ValueError, match="failure rate is in \\[0, 1\\]"):
        LB.table(small(failure_rate=2))
    with pytest.raises(ValueError, match="a multiple of the slot stride's 16, not 4 lanes and 40 buckets"):
        LB.table(small(lanes=4, buckets=40))
    with pytest.raises(ValueError, match="a scope one of node, datacenter, world"):
        LB.document(small(), only=["rack"])
    with pytest.raises(ValueError, match="the target is positive"):
        LB.document(small(), target_gb=0)
