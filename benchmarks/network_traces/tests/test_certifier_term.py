"""leak_bounds' scaling by requests, which `verity bench certifier_term` prints: the network certifier's terms at the
load of a node, the reference datacenter and the world, each check's by its own rule.

- the scan's terms are the rows form's and N2's the clock form's, held here to the Lean's `clog2` on a hand-worked
  execution phase, and the table to a hand-worked world of three machines, in exact arithmetic;
- each of the three changes zeroes what it should and leaves the rest, the failed connections' pooled charge bounds
  the count it states, and the design's term is the jitter and that charge;
- the design with no stop block is charged the scan's schedules and is the design's otherwise, and no other row
  reads the stop block;
- nothing a check counts by capacity scales with the requests;
- the world, the datacenter, the requests and the weights are read from the files, each input marked with its source;
- the set-time card is an input marked simulated, the one input so marked, its source the clause a figure under the
  label rests on, and the document's figure at D_j = 1 carries that mark;
- the world's requests are by default the sampling plan's, the break-even planner's own count; the census's where
  asked; a given count in place of either; and every row names which;
- the term of record is the one at D_j = 2: the document's value is the scan's over the world's year at D_j = 2, and
  a figure at D_j = 1 is beside it only under its label, in the document and in every header of the tables;
- the term of record comes first: in each table a column at D_j = 2 stands before its column at D_j = 1, and every
  cell is under its own header;
- a row that rests on the set-time card at either D_j carries the label, in the document and in its printed name,
  and no other row prints a jitter of 0 at D_j = 2;
- the command prints the table and its JSON (value, unit, commit, run), through `verity bench`'s runner too."""

from __future__ import annotations

import itertools
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
#: the design before its stop block has been sampled, the one row with no stop block
NO_STOP = "design, no stop block"
NAMES = ("scan", "N2", "lengths", "timing", "entries", "all three", NO_STOP, "design")
#: the label's short form, as a column's header at D_j = 1 and a row's printed name carry it
CARD = "simulated set-time card"


def clog2(x: int) -> int:
    """The Lean's `clog2`: if x ≤ 1 then 0 else Nat.log2 (x - 1) + 1."""
    return 0 if x <= 1 else (x - 1).bit_length()


def tables(text: str) -> list[list[dict]]:
    """The printed tables of the scopes, the terms in GB and the terms in bits: each one's body rows, a row its
    cells by header, in the header's order."""
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    out, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("| scope "):
            body = itertools.takewhile(lambda x: x.startswith("|"), lines[i + 2:])
            out.append([dict(zip(cells(line), cells(x), strict=True)) for x in body])
    return out


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
    assert len(got) == 3 * 3 * 8 and tuple(LB.ROWS) == NAMES and {k[2] for k in got} == set(NAMES)
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
            if key[2] in ("lengths", "all three", NO_STOP, "design"):
                charge = LB.failed(r.requests, r.requests / 1_000, 1_001)
                assert f.terms[dj] == r.terms[dj]._replace(cells=charge)
                assert (f.terms[dj].schedules == 0) == (key[2] != NO_STOP)
                assert charge == pytest.approx(float(r.requests) / 1_000 * math.log2(1_000 * math.e * 1_002))
            else:
                assert f.terms[dj] == r.terms[dj]
    design, each = failing["world", "year", "design"].terms, math.log2(1_000 * math.e) + math.log2(1_002)
    assert 21.37 < each < 21.38 and design[1].certifier == pytest.approx(7_391.25 * each)
    assert design[2].certifier == pytest.approx(23_652_000 + 7_391.25 * each)


def test_the_design_with_no_stop_block_is_charged_the_scans_schedules_and_is_the_designs_otherwise():
    """Until a stop block holds a response's output count to its ids, the server picks the count the lengths unit
    holds a length to. So the row's schedules are the scan's, and its failed connections' pooled charge, its jitter
    and its entries are the design's: at every scope, monitoring period and jitter bound, with failures and without,
    on the hand-worked world, at the command's defaults and at a given count of a smaller class. It stands where
    the design stands, just before it."""
    names = list(LB.ROWS)
    bare, design = LB.ROWS[NO_STOP], LB.ROWS["design"]
    assert names.index(NO_STOP) + 1 == names.index("design") == len(names) - 1
    assert (bare.standing, bare.changes) == (design.standing, design.changes) and bare.counts != design.counts
    assert "stop block" in bare.counts and "stop block" in design.counts
    assert [name for name, kind in LB.ROWS.items() if not kind.stop] == [NO_STOP]
    rate = Fraction(1, 10 ** 6)
    for inp in (small(), small(failure_rate=Fraction(1, 1_000)), LB.inputs(), LB.inputs(failure_rate=rate),
                LB.inputs(world_requests=1.778e16, response_ids=1_024),
                LB.inputs(world_requests=1.778e16, response_ids=1_024, failure_rate=rate)):
        got = rows(inp)
        for scope in LB.SCOPES:
            for period in LB.PERIODS:
                scan, whole, row = (got[scope, period, name] for name in ("scan", "design", NO_STOP))
                assert (row.requests, row.phases, row.eps) == (whole.requests, whole.phases, whole.eps)
                for dj in LB.JITTERS:
                    t = row.terms[dj]
                    assert t.schedules == scan.terms[dj].schedules > 0 == whole.terms[dj].schedules
                    assert (t.cells, t.jitter, t.entries) == (whole.terms[dj].cells, whole.terms[dj].jitter, None)
                    assert t == whole.terms[dj]._replace(schedules=scan.terms[dj].schedules)
                    more = float(whole.terms[dj].certifier + scan.terms[dj].schedules)
                    assert float(t.certifier) == pytest.approx(more, rel=1e-12)
                    assert (t.cells > 0) == bool(inp["failure_rate"].value)


def test_only_the_lengths_change_reads_the_stop_block():
    """Under every set of the three changes, `terms` with a stop block is `terms` as it is called without the
    argument, and with none it differs only where the lengths are held, and there only in the schedules, which are
    each connection's as under no change. So the rows that have a stop block, every row but one, are not moved by
    the one that has none. A row whose schedules read the stop block says so: the rows that hold the lengths name
    it in their text, and no other row does."""
    given = {"cells": 65, "steps": 32, "dj": 2, "entries": 9_088, "failures": 1}
    scan = LB.terms(4, 1, **given)
    for size in range(len(LB.CHANGES) + 1):
        for changes in itertools.combinations(LB.CHANGES, size):
            held, bare = LB.terms(4, 1, changes=changes, **given), LB.terms(4, 1, changes=changes, stop=False, **given)
            assert held == LB.terms(4, 1, changes=changes, stop=True, **given)
            if "lengths" in changes:
                assert held.schedules == 0 < scan.schedules and bare == held._replace(schedules=scan.schedules)
                assert bare.cells == held.cells == LB.failed(4, 1, 65) != scan.cells
            else:
                assert bare == held
    assert LB.Kind("on main", (), "a row").stop
    reads = {name for name, kind in LB.ROWS.items() if "lengths" in (kind.changes or ())}
    assert reads == {"lengths", "all three", NO_STOP, "design"} == {
        name for name, kind in LB.ROWS.items() if "stop block" in kind.counts}


def test_the_scale_is_read_from_the_files_and_every_input_is_marked():
    inp = LB.inputs(requests="census")
    assert all(i.mark in LB.MARKS and i.source for i in inp.values())
    fleet = tomllib.loads((ROOT / LB.RHO).read_text())["fleet"]["world-h200"]
    card = json.loads((ROOT / LB.WORKLOAD).read_text())
    census = next(e for e in json.loads((ROOT / LB.ADVERSARY).read_text()) if e["id"] == "world-inference-2027")
    peak = next(e for e in json.loads((ROOT / LB.HARDWARE).read_text()) if e["id"] == LB.PEAK)
    assert inp["world_machines"].value == LB.world_links() == fleet["count"]["value"] == 46_870_000
    assert inp["datacenter_machines"].value == LB.datacenter_links() == 12_500 and inp["machine_gpus"].value == 8
    assert (inp["mfu_serve"].value, inp["mfu_serve"].mark) == (fleet["mfu_serve"]["value"], "assumed")
    assert inp["machine_gpus"].mark == inp["peak_flops"].mark == "assumed" and inp["world_machines"].mark == "modeled"
    assert inp["weights"].value == LB.WEIGHTS.bits == card["value"]["params"] * 4
    assert (inp["token_macs"].mark, inp["weights"].mark) == (card["status"], card["status"]) == ("measured", "measured")
    for name, path in (("world_machines", LB.RHO), ("machine_gpus", LB.RHO_HARDWARE), ("mfu_serve", LB.RHO),
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


def test_the_set_time_card_is_an_input_marked_simulated_and_the_documents_figure_carries_its_mark():
    """The card every figure under the label rests on is simulated: a fourth mark, which the card's input alone
    carries, with the clause the figure rests on as its source. The document's figure at D_j = 1 carries that mark
    beside its label, whatever the command asks, and the term of record carries none: it rests on no card."""
    assert LB.MARKS == ("measured", "modeled", "assumed", "simulated") and LB.SIMULATED == "simulated"
    clause = ("`trusted-host`, set-time card clause: the network certifier's card releases each bucket at the tick its "
              "record names, whatever the host and the server send it. It holds only of a calibrated hardware card; no "
              "such card is deployed, and the software pacer that stands in for it does not meet it.")
    assert LB.SET_TIME_CLAUSE == clause and LB.SET_TIME == "with a simulated set-time card"
    for inp in (LB.inputs(), LB.inputs(requests="census"), LB.inputs(world_requests=1e16, lanes=512), small()):
        assert inp["set_time_card"] == LB.Input("simulated", "", "simulated", clause)
        assert [k for k, i in inp.items() if i.mark == "simulated"] == ["set_time_card"]
        doc = LB.document(inp)
        assert doc["inputs"]["set_time_card"] == {"value": "simulated", "unit": "", "mark": "simulated",
                                                  "source": clause}
        assert doc["with_set_time_card"]["mark"] == "simulated" and "mark" not in doc
        text = LB.render(doc)
        assert f"| set_time_card | simulated | | simulated | {clause} |" in " ".join(text.split())
    # the mark is the input's own: the figure of a document whose card is marked otherwise says so
    other = small() | {"set_time_card": LB.Input("simulated", "", "assumed", clause)}
    assert LB.document(other)["with_set_time_card"]["mark"] == "assumed"


def test_the_worlds_requests_are_the_sampling_plans_by_default(monkeypatch):
    """The default count is the planner's own: its case's world year at the 128k serving plan's request, with that
    request's tokens and served MACs a token as the inputs beside it, from which the count follows by the scopes'
    rule. It is the request's count, the same on every case the planner prices at that request, whatever the case's
    levers. Every scope and monitoring period the planner counts, the table counts alike. Nothing else moves from
    the census's inputs."""
    monkeypatch.syspath_prepend(str(ROOT / LB.PLANNER))
    import sweep
    case = sweep.cases()[LB.PLAN_CASE]
    _, cards, *_, at = sweep.build(case)
    inp = LB.inputs()
    assert case.ctx == "c343" and inp == LB.inputs(requests="plan") and inp["request_count"].value == "plan"
    assert inp["world_requests"].value == at["scopes"]["world year"].requests > 0
    alike = [c for c in sweep.cases().values() if c.ctx == case.ctx]
    assert len(alike) > 1
    assert {sweep.build(c)[-1]["scopes"]["world year"].requests for c in alike} == {inp["world_requests"].value}
    assert (inp["request_tokens"].value, inp["token_macs"].value) == (cards["tokens_per_request"].value,
                                                                     at["served_macs"])
    by = f"{LB.PLANNER}/sweep.py"
    for name in ("request_tokens", "token_macs", "world_requests"):
        assert inp[name].mark == "modeled" and inp[name].source.startswith(f"{by}'s case {LB.PLAN_CASE}: ")
    assert (ROOT / by).is_file() and all(i.mark in LB.MARKS and i.source for i in inp.values())
    flop = (inp["world_machines"].value * inp["machine_gpus"].value * LB.PERIODS["year"] * inp["mfu_serve"].value
            * inp["peak_flops"].value)
    assert inp["world_requests"].value == pytest.approx(
        flop / (inp["request_tokens"].value * 2 * inp["token_macs"].value), rel=1e-12)
    got = rows(inp)
    assert sorted(at["scopes"]) == sorted(f"{who} {period}" for who in ("world", "datacenter") for period in LB.PERIODS)
    for scope in at["scopes"].values():
        r = got[scope.who, scope.period, "scan"]
        assert r.scope.gpus * LB.PERIODS[scope.period] == scope.gpu_seconds
        assert float(r.requests) == pytest.approx(scope.requests, rel=1e-12)
    census = LB.inputs(requests="census")
    moved = {k for k in inp if inp[k] != census[k]}
    assert "request_count" in moved <= {"request_tokens", "token_macs", "world_requests", "request_count"}
    assert set(inp) == set(census) and list(inp) == list(census)


def test_every_row_names_its_request_count():
    """Each count's document names it at its top and on every row, its text in every row of its three tables; a
    given count stands in place of either request's."""
    for given, name in (({}, "plan"), ({"requests": "census"}, "census"), ({"world_requests": 1e16}, "given"),
                        ({"requests": "census", "world_requests": 1e16}, "given"),
                        ({"requests": "plan", "world_requests": 1e16}, "given")):
        inp = LB.inputs(**given)
        assert (inp["request_count"].value, inp["request_count"].source) == (name, LB.COUNTS[name])
        assert name != "given" or (inp["world_requests"].value, inp["world_requests"].mark) == (1e16, "assumed")
        doc = LB.document(inp)
        assert doc["request_count"] == doc["inputs"]["request_count"]["value"] == name
        assert len(doc["rows"]) == 72 and {r["request_count"] for r in doc["rows"]} == {name}
        text = LB.render(doc).splitlines()
        assert sum(line.startswith(f"Request count {name}: {LB.COUNTS[name]}. Every row names") for line in text) == 1
        body = [line.split("|") for line in text if line.startswith(("| node ", "| datacenter ", "| world "))]
        assert len(body) == 9 + 72 + 72 and all(name in (c.strip() for c in cells[3:5]) for cells in body)
        heads = [line for line in text if line.startswith("| scope ")]
        assert len(heads) == 3 and all("| request count |" in line for line in heads)
    assert list(LB.COUNTS) == [*LB.REQUESTS, "given"] and LB.REQUESTS[0] == "plan"
    json.dumps(LB.document(LB.inputs()))


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
    assert doc["schema"] == "certifier-term/v2"
    assert doc["value"] == 171_477_000 / LB.GB and "world's year" in doc["name"]
    assert "two release schedules (D_j = 2)" in doc["name"] and "D_j = 1" not in doc["name"]
    assert doc["with_set_time_card"] == {"value": 147_825_000 / LB.GB, "unit": "GB",
                                         "label": "with a simulated set-time card", "mark": "simulated"}
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
        "on main", "on main", *["a change to the scan"] * 4, *["a design with no run behind it"] * 2]
    assert "by request" in doc["legend"]["scan"]["counts"] and "by link" in doc["legend"]["N2"]["counts"]
    by = {(r["scope"], r["period"], r["row"]): r for r in doc["rows"]}
    assert len(by) == 72 and all(r["covers"] == 1.0 and r["standing"] == doc["legend"][r["row"]]["standing"]
                                 for r in by.values())
    r = by["world", "year", "scan"]
    assert (r["machines"], r["gpu_seconds"], r["requests"], r["phases"]) == (3, 24 * 31_536_000, 7_391_250, 739_125)
    assert r["schedules"] == r["cells"] == {"bits": 73_912_500, "GB": 73_912_500 / LB.GB}
    assert r["jitter"] == {"1": {"bits": 0, "GB": 0}, "2": {"bits": 23_652_000, "GB": 23_652_000 / LB.GB}}
    assert r["term"]["2"] == {"bits": 171_477_000, "GB": 171_477_000 / LB.GB,
                              "of_target": 171_477_000 / (300 * LB.GB), "of_weights": 171_477_000 / LB.WEIGHTS.bits,
                              "bits_a_phase": 232.0, "bits_a_request": 23.2}
    assert r["entries"]["bits"] == 18_048 * 7_391_250
    assert all(by["world", "year", name]["entries"] is None
               for name in ("entries", "all three", NO_STOP, "design", "N2"))
    assert by["world", "year", "N2"]["term"]["2"]["bits_a_phase"] == 39.0
    assert by["world", "year", "design"]["term"]["2"]["bits"] == 23_652_000
    bare = by["world", "year", NO_STOP]
    assert bare["schedules"] == r["schedules"] and bare["cells"] == by["world", "year", "design"]["cells"]
    assert bare["term"]["2"]["bits"] == r["schedules"]["bits"] + by["world", "year", "design"]["term"]["2"]["bits"]
    assert r["eps"] == 739_125 / 2 ** 20
    json.dumps(doc)
    few = LB.document(small(), only=["world"], periods=["year", "week"])
    assert [(x["scope"], x["period"]) for x in few["rows"]] == [("world", "week")] * 8 + [("world", "year")] * 8
    assert few["value"] == doc["value"]
    text = LB.render(doc)
    assert "commit c0ffee, run r1" in text and "## The terms in GB" in text and "## The terms in bits" in text
    assert "the compute side's" in text and f"| world      | year   | all three, {CARD} |" in text
    assert f"| world      | year   | {NO_STOP} " in text and f"| world      | year   | timing, {CARD} " in text
    assert "- design (a design with no run behind it): " in text and "- N2 (on main): " in text
    assert f"- {NO_STOP} (a design with no run behind it): " in text
    assert "| lanes " in text and "| none " in text
    assert f"| world | year   | all three, {CARD} |" in LB.render(few) and "| node" not in LB.render(few)
    assert LB.render(LB.document(small())).splitlines()[2] == "commit unknown, run none"
    slots = LB.document(small(lanes=4))
    assert "4 lanes x 2 strides" in slots["entries_rule"] and "by capacity" in slots["entries_rule"]


def test_the_headline_is_the_scans_world_year_term_at_two_release_schedules_and_d_j_1_carries_its_label():
    """The term of record is the one at D_j = 2. On a hand-worked world the scan's year is 2 × 73,912,500 bits of
    schedules and fail cells and 739,125 execution phases × 32 bits of jitter: the document's value is all of it, and
    the term without the jitter is beside it only under the label, in the document and in each table's headers."""
    label = "with a simulated set-time card"
    assert (LB.OF_RECORD, LB.SET_TIME) == (2, label) and LB.OF_RECORD in LB.JITTERS
    for given in ({}, {"failure_rate": Fraction(1, 100), "lanes": 4}):
        for doc in (LB.document(small(**given)), LB.document(small(**given), only=["node"], periods=["week"])):
            assert (doc["value"], doc["unit"]) == ((2 * 73_912_500 + 739_125 * 32) / LB.GB, "GB")
            assert doc["with_set_time_card"] == {"value": 2 * 73_912_500 / LB.GB, "unit": "GB", "label": label,
                                                 "mark": "simulated"}
            assert "two release schedules (D_j = 2)" in doc["name"] and "D_j = 1" not in doc["name"]
    for inp in (small(), LB.inputs(), LB.inputs(requests="census")):
        doc = LB.document(inp)
        by = {(r["scope"], r["period"], r["row"]): r for r in doc["rows"]}
        scan = by["world", "year", "scan"]
        assert doc["value"] == scan["term"]["2"]["GB"] > scan["term"]["1"]["GB"] == doc["with_set_time_card"]["value"]
        jitter = scan["jitter"]["2"]["GB"]
        assert doc["value"] - doc["with_set_time_card"]["value"] == pytest.approx(jitter, rel=1e-9)
        assert jitter == by["world", "year", "design"]["term"]["2"]["GB"] and scan["jitter"]["1"]["GB"] == 0
    text = LB.render(doc).splitlines()
    assert ("The term of record is the one at D_j = 2, two release schedules. The term at D_j = 1 is what a network "
            "certifier whose network card releases at a set time would be charged. That card is simulated, the input "
            "set_time_card, whose source is what the term rests on: none is deployed, so it is printed as "
            f"\"{label}\".") in text
    held = next(line for line in text if line.startswith("With the lengths held by a stop block and the release "))
    assert "taken once by a network certifier whose network card releases at a set time (simulated)" in held
    assert held.endswith(f" Without that card the jitter alone, at D_j = 2, is {LB._n(jitter)} GB over the world's "
                         f"year, {LB._n(jitter / 300)} times the target.")
    heads = [[c.strip() for c in line.strip("|").split("|")] for line in text if line.startswith("| scope")]
    assert [sum("D_j = 1" in c for c in head) for head in heads] == [0, 4, 4]
    assert [c for c in heads[1] if c.startswith("x ")] == [
        f"{c}, D_j = {dj}" for dj in ("2", f"1, {CARD}") for c in ("x target", "x weights")]
    for head in heads:
        assert all(c.endswith(f", D_j = 1, {CARD}") for c in head if "D_j = 1" in c)
        assert sum("D_j = 2" in c for c in head) == sum("D_j = 1" in c for c in head)
        assert not any("set-time card" in c for c in head if "D_j = 1" not in c)
    few = LB.render(LB.document(small(), only=["node"], periods=["week"]))
    assert "Without that card the jitter alone, at D_j = 2, is " in few and f"it is printed as \"{label}\"." in few
    zeroed = {name for name, kind in LB.ROWS.items() if "timing" in (kind.changes or ())}
    assert zeroed == {"timing", "all three"} == {name for name, k in LB.ROWS.items() if "set-time card" in k.counts}
    assert "the card is simulated, and none is deployed" in LB.CHANGES["timing"]


def test_the_term_of_record_comes_first_in_both_tables():
    """In each table a column at D_j = 2 stands before its column at D_j = 1, and every cell is under its own
    header. On the hand-worked world the scan's year is 23,652,000 bits of jitter in a term of 171,477,000 bits at
    D_j = 2, 232 bits a link's phase and 23.2 a request; without the jitter it is 147,825,000 bits, 200 and 20. The
    line on a link's execution phase names the jitter at D_j = 2 first, and the one at D_j = 1 under its label."""
    one, n = f"D_j = 1, {CARD}", LB._n
    text = LB.render(LB.document(small(), only=["world"], periods=["year"]))
    scopes, gb, bits = tables(text)
    assert [len(t) for t in (scopes, gb, bits)] == [1, 8, 8]
    assert list(gb[0]) == [
        "scope", "period", "row", "request count", "schedules", "cells", "jitter, D_j = 2", f"jitter, {one}",
        "term, D_j = 2", "x target, D_j = 2", "x weights, D_j = 2", f"term, {one}", f"x target, {one}",
        f"x weights, {one}", "entries, the compute side's"]
    assert list(bits[0]) == [
        "scope", "period", "row", "request count", "schedules", "cells", "jitter, D_j = 2", f"jitter, {one}",
        "term, D_j = 2", f"term, {one}", "entries, the compute side's", "a link's phase, D_j = 2",
        f"a link's phase, {one}", "a request, D_j = 2", f"a request, {one}"]
    scan = gb[0]
    assert (scan["row"], scan["jitter, D_j = 2"], scan[f"jitter, {one}"]) == ("scan", n(23_652_000 / LB.GB), "0")
    assert (scan["term, D_j = 2"], scan[f"term, {one}"]) == (n(171_477_000 / LB.GB), n(147_825_000 / LB.GB))
    assert scan["x target, D_j = 2"] == n(171_477_000 / (300 * LB.GB)) != scan[f"x target, {one}"]
    assert scan[f"x target, {one}"] == n(147_825_000 / (300 * LB.GB))
    scan = bits[0]
    assert (scan["jitter, D_j = 2"], scan[f"jitter, {one}"]) == ("23,652,000", "0")
    assert (scan["term, D_j = 2"], scan[f"term, {one}"]) == ("171,477,000", "147,825,000")
    assert (scan["a link's phase, D_j = 2"], scan[f"a link's phase, {one}"]) == ("232", "200")
    assert (scan["a request, D_j = 2"], scan[f"a request, {one}"]) == ("23.2", "20")
    assert f"the jitter is 32 bits at D_j = 2, and 0 at D_j = 1 {LB.SET_TIME}; a slot entry is " in text
    for doc in (LB.document(small()), LB.document(LB.inputs())):
        _, gb, bits = tables(LB.render(doc))
        for r, g, b in zip(doc["rows"], gb, bits, strict=True):
            for dj, at in (("2", "D_j = 2"), ("1", one)):
                jitter, term = r["jitter"][dj], r["term"][dj]
                assert (g[f"jitter, {at}"], b[f"jitter, {at}"]) == (n(jitter["GB"]), n(jitter["bits"]))
                assert [g[f"{c}, {at}"] for c in ("term", "x target", "x weights")] == [
                    n(term[k]) for k in ("GB", "of_target", "of_weights")]
                assert [b[f"{c}, {at}"] for c in ("term", "a link's phase", "a request")] == [
                    n(term[k]) for k in ("bits", "bits_a_phase", "bits_a_request")]


def test_a_row_that_rests_on_the_set_time_card_carries_the_label():
    """A row whose release times are taken once has a jitter of 0 at D_j = 2 too, a zero that rests on the set-time
    card. So the row carries the label: in the document as its own ``with_set_time_card``, which is null on every
    other row, and in both tables in its name, beside every cell of it. The rows that carry it are exactly those
    whose changes include the timing's, and no other row prints a jitter of 0 at D_j = 2. A row's name in the
    document is as it was, since readers take rows by name."""
    carries = {name for name, kind in LB.ROWS.items() if "timing" in (kind.changes or ())}
    assert carries == {"timing", "all three"}
    rate = Fraction(1, 10 ** 6)
    for inp in (small(), small(failure_rate=Fraction(1, 100), lanes=4), LB.inputs(), LB.inputs(failure_rate=rate),
                LB.inputs(requests="census"), LB.inputs(world_requests=1.778e16, response_ids=1_024)):
        doc = LB.document(inp)
        assert len(doc["rows"]) == 72 and [r["row"] for r in doc["rows"][:8]] == list(NAMES)
        for r in doc["rows"]:
            assert r["row"] in NAMES and r["with_set_time_card"] == (LB.SET_TIME if r["row"] in carries else None)
            assert (r["jitter"]["2"]["bits"] == 0) == (r["row"] in carries) and r["jitter"]["1"]["bits"] == 0
        _, gb, bits = tables(LB.render(doc))
        for table in (gb, bits):
            for r, printed in zip(doc["rows"], table, strict=True):
                labelled = printed["row"].endswith(f", {CARD}")
                assert printed["row"] == r["row"] + (f", {CARD}" if labelled else "")
                assert labelled == (r["row"] in carries) == (printed["jitter, D_j = 2"] == "0")
    text = LB.render(doc).splitlines()
    assert ("A row whose release times are taken once has a jitter of 0 at either D_j, which rests on the same card, "
            "so its name carries the label.") in text
    assert LB.SET_TIME == f"with a {CARD}"


def test_the_command_prints_the_table_and_its_json(capsys, monkeypatch):
    assert LB.main(["--json", "--commit", "c0ffee", "--run", "r1", "--world-requests", "1e16"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc == json.loads(json.dumps(LB.document(LB.inputs(world_requests=1e16), commit="c0ffee", run="r1")))
    assert doc["with_set_time_card"]["value"] == 2 * 13 * 1e16 / LB.GB and len(doc["rows"]) == 72
    assert doc["value"] == pytest.approx((2 * 13 * 1e16 + 1_280 * 46_870_000 * 246_375) / LB.GB, rel=1e-12)
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
    for argv, given in (([], {}), (["--requests", "plan"], {}), (["--requests", "census"], {"requests": "census"})):
        assert LB.main(["--json", "--commit", "c0ffee", *argv]) == 0
        counted = json.loads(capsys.readouterr().out)
        assert counted == json.loads(json.dumps(LB.document(LB.inputs(**given), commit="c0ffee")))
        assert counted["request_count"] == given.get("requests", "plan")
    assert LB.main(["--json", "--commit", "c0ffee", "--run", "r1", "--requests", "census", "--world-requests",
                    "1e16"]) == 0
    assert json.loads(capsys.readouterr().out) == doc and doc["request_count"] == "given"
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
    with pytest.raises(ValueError, match="the request counted is one of plan, census, not 'given'"):
        LB.inputs(requests="given")
    with pytest.raises(SystemExit):
        LB.main(["--requests", "given"])
