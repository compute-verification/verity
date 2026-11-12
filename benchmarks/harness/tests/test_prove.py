"""`verity bench prove` with no GPU and no node: the rules on small tables (a census and its cut, each source of a
block type's proving time, the totals and their range, SPOT over a sampling plan), then the store side on a real store with
a directory remote: a workload's records as `fixtures/prove/` holds them (each cut to the fields the command reads;
`records.json` names what each was cut from), the hand computation they reproduce, every refusal by name, and the
command.

`fixtures/prove/` is the reader's input, not a result: the store side parses the bytes the producers wrote, so the
census, the cards and the sampling plan are those bytes, never regenerated here. The steady-state rows are not: the
three runs predate the one format the command reads (`benchmarks/one_stage/server_rate.py`'s `rate.json`), so
`steady/RUN/rate/rate.json` holds each of their proof server reports as that script's rows, a staged type and prover
count a row with the fields the command reads: a statement's mean GPU-seconds as its reciprocal `statements_per_s` on
one GPU, a report that was stopped or timed out as a void row whose prover exited nonzero, and one with a session not
accepted as a void row with a refused session. `server-reports/RUN/` holds the runs' reports themselves, as the store does,
for the reader of runs from before `rate.json`. `refusals/rate/rate.json` is made up: one row that is a steady-state
measurement and one of each way a row isn't. The figures pinned below hold this code's arithmetic to a computation done
independently of it on the same records; they publish no measurement, and what the command says of a workload is what
it prints from the store's records of record."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import runpy
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from verity_bench import lean_verify as L
from verity_bench import prove as P
from verity_bench import prove_records as R
from verity_bench import steady as S
from verity_bench.cli import main

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "prove"
W = R.WORKLOADS["glm53"]

# ---- the rules, on small tables ------------------------------------------------------------------------------------------------

#: one request of 4 tokens: two block types cut from a GEMM row, one from a norm row, and one no step ran
CENSUS = {
    "tokens": {"prefill": 3, "decode": 1}, "per_request": {"circuit_macs": 4000, "macs": 3600},
    "definitions": {
        'Gemm_v1{K=64,N=1,DOT={"fn":"Dot_v1"}}': {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 6, "decode": 2},
        "Gemm_v1{K=32,N=1}": {"k_log": 19, "of": {"Linear_v1": "row"}, "prefill": 3, "decode": 1},
        'Norm_v1{N=8,EPS={"f64":"0x1.0p-3"},FMA=true,TREE=[0,[1,2]]}': {"k_log": 22, "of": {"Norm_v1": "row", "Add_v1": "part"},
                                                                    "prefill": 1, "decode": 1},
        "Idle_v1{N=1}": {"k_log": 18, "of": {"Idle_v1": "row"}, "prefill": 0, "decode": 0},
    },
}
GEMM64, GEMM32, NORM = (name for name in CENSUS["definitions"] if not name.startswith("Idle"))
CUT = {"name": "split", "label": "a toy cut", "replaces": ["Add_v1, Norm_v1"],
       "strata": [{"block": "Half_v1{N=4}", "kind": "Half_v1", "per_token": 1.0, "k_log": 21},
                  {"block": "Bare_v1{N=4}", "kind": "Bare_v1", "per_token": 0.25},
                  {"block": "Unused_v1{N=4}", "kind": "Unused_v1", "per_token": 0, "k_log": 21}]}
#: single sessions: the K=64 GEMM's own, and another Definition's, both at 2 s a statement's fixed cost
SESSIONS = (P.Session(GEMM64, 20, 0.5, 2.0, 1 << 15), P.Session("Other_v1{N=2}", 22, 2.0, 2.0, 1 << 13))


def steady(type_: str, definition: str, run: str, blocks: int, gpu: float | None, *, full: int = 8, k_log: int = 20,
           verify: float | None = None, refused: str | None = None, provers: int = 1, art: str | None = None,
           cores: int = P.HOST_CORES, form: str | None = None) -> P.Steady:
    """A steady-state row, timed at the host's cores unless `cores` says otherwise, its statements of `definition`
    unless `form` names another Definition."""

    return P.Steady(type_, definition, k_log, blocks, full, run, R.ART_RATES if art else R.RATES, gpu, verify, refused,
                    provers, art, cores, form)


def test_a_census_counts_a_request_over_its_tokens_and_a_cut_replaces_its_kinds():
    census = P.Census.of(CENSUS)
    assert [(b.name, b.kind, b.per_token, b.k_log) for b in census.blocks] == [
        (GEMM64, "Linear_v1", 2.0, 20), (GEMM32, "Linear_v1", 1.0, 19), (NORM, "Add_v1, Norm_v1", 0.5, 22)]
    assert (census.served_macs_per_token, census.block_macs_per_token, census.macs_in_blocks, census.cut) == (1000, 900, 0.9, None)

    recut = census.recut(CUT)
    assert [b.name for b in recut.blocks] == [GEMM64, GEMM32, "Half_v1{N=4}", "Bare_v1{N=4}"]
    assert recut.cut == "split" and recut.blocks[-1].k_log is None and recut.macs_in_blocks == 0.9
    with pytest.raises(ValueError, match="replaces Router_v1, which the census doesn't hold"):
        census.recut(CUT | {"replaces": ["Router_v1"]})


def test_a_census_name_is_spelled_as_the_definition_id_a_staged_statement_records():
    assert P.definition_id(GEMM64) == "Gemm_v1{K=64,N=1,DOT=Dot_v1}"
    assert P.definition_id(NORM) == "Norm_v1{N=8,EPS=0.125,FMA=True,TREE=(0,(1,2))}"
    assert P.definition_id('Read_v1{READ={"fn":"Row_v1{V=2,H=3}"},SWAP=false}') == "Read_v1{READ=Row_v1{V=2,H=3},SWAP=False}"


def test_a_block_type_takes_the_best_source_that_holds_it():
    """Steady state, then the card's single session (its own block's, else its Definition and K's scaled), then the k_log
    model, then nothing; and a steady-state attempt that didn't stand is named on the row it would have priced."""

    census = P.Census.of(CENSUS).recut(CUT)
    reports = (
        steady("gemm64", P.definition_id(GEMM64), "r20260101-000000-0001", 8, 16.0, verify=4.0),
        steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),       # the newest full one
        steady("gemm64", P.definition_id(GEMM64), "r20260103-000000-0003", 4, 6.0, verify=2.0),       # newer, but half full
        steady("gemm64", P.definition_id(GEMM64), "r20260104-000000-0004", 8, None, refused="timed out"),
        steady("half", "Half_v1{N=4}", "r20260101-000000-0001", 8, 80.0, refused="timed out"),
        steady("half", "Half_v1{N=4}", "r20260103-000000-0003", 4, 80.0, refused="stopped at the host's memory limit"),
    )
    table = P.Flock(reports, SESSIONS, "art:card").table(census)
    by = {r.block: r for r in table.rows}
    assert table.system == "flock" and list(by) == [b.name for b in census.blocks]

    gemm = by[GEMM64].cost
    assert (gemm.source, gemm.seconds, gemm.record, gemm.staged) == (P.STEADY, 1.0, "r20260102-000000-0002", "gemm64")
    assert gemm.verifier_seconds == 0.5 and gemm.at_steady_rate is None and gemm.provers == 1
    assert gemm.note == "`gemm64` (`Gemm_v1{K=64,N=1,DOT=Dot_v1}`), 1 prover, full statements (out/rate/rate.json)"
    assert gemm.definition == P.definition_id(GEMM64) and by[GEMM32].cost.definition is None

    # the card measures the K=64 GEMM alone: K=32 is another type, so it and the cut's block are modeled from k_log, at
    # the card's types' median seconds a witness bit and their fixed seconds' line in k_log (flat here, at 2 s)
    rate = (0.5 / 2**20 + 2.0 / 2**22) / 2
    assert by[GEMM32].cost.source == P.MODELED and by[GEMM32].cost.record == "art:card"
    assert by[GEMM32].cost.seconds == pytest.approx(rate * 2**19 + 2.0 / 2**16)
    half = by["Half_v1{N=4}"].cost
    assert half.source == P.MODELED and half.seconds == pytest.approx(rate * 2**21 + 2.0 / 2**14)
    assert half.staged == "half" and half.provers is None and "memory limit" in half.note
    assert "tried as `half` at 1 prover in r20260103-000000-0003" in half.note
    assert by["Bare_v1{N=4}"].cost == P.Cost(P.MISSING)

    # the range's other end: every type off the steady state at the full-statement steady types' median rate
    assert by[GEMM32].cost.at_steady_rate == 1.0 / 2**20 * 2**19
    assert "9.54e-07 GPU-seconds a witness bit" in table.range_basis
    total = 2.0 * 1.0 + 1.0 * by[GEMM32].cost.seconds + 1.0 * half.seconds
    assert table.gpu_seconds_per_token() == pytest.approx(total)
    assert table.gpu_seconds_range() == pytest.approx((total, 2.0 + 0.5 + 2.0))

    # with no steady-state report the K=64 GEMM takes its own single session, at a full statement's share of the fixed
    # seconds, and no row has another end
    alone = {r.block: r.cost for r in P.Flock((), SESSIONS, "art:card").table(census).rows}
    assert (alone[GEMM64].source, alone[GEMM64].seconds) == (P.SINGLE, 0.5 + 2.0 / (1 << 15))
    assert all(c.at_steady_rate is None for c in alone.values())
    # with no card either, nothing is priced
    assert {r.cost.source for r in P.Flock().table(census).rows} == {P.MISSING}


def test_a_steady_row_prices_its_definition_and_ks_other_statics_and_is_not_measured_for_them():
    """A type with no row of its own takes the steady-state rows of its Definition and K at other statics: at 2^Δk
    their seconds a block, the dearest of them, with no verifier's seconds. Another K is another family. Such a type
    counts in the totals and in neither the steady-state share nor a measured status."""

    defs = {"Head_v1{K=4,FIRST=false}": 20, "Head_v1{K=4,FIRST=true}": 20, "Head_v1{K=4,LAYER=9}": 21,
            "Head_v1{K=8,FIRST=false}": 20}
    census = P.Census.of({"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        name: {"k_log": k, "of": {"Head_v1": "row"}, "prefill": 1, "decode": 0} for name, k in defs.items()}})
    rows = (steady("head", "Head_v1{K=4,FIRST=False}", "r20260101-000000-0001", 8, 8.0, verify=8.0),
            steady("head-first", "Head_v1{K=4,FIRST=True}", "r20260101-000000-0001", 8, 16.0, verify=8.0))
    table = P.Flock(rows, (), None).table(census)
    by = {r.block: r.cost for r in table.rows}
    assert [(by[n].source, by[n].seconds) for n in defs] == [(P.STEADY, 1.0), (P.STEADY, 2.0), (P.FAMILY, 4.0), (P.MISSING, None)]
    kin = by["Head_v1{K=4,LAYER=9}"]
    assert (kin.record, kin.staged, kin.provers, kin.verifier_seconds) == ("r20260101-000000-0001", "head-first", 1, None)
    assert kin.note == ("its Definition and K's, `head-first` (`Head_v1{K=4,FIRST=True}`), 1 prover, full statements, "
                        "at 2^1 its seconds; the dearest of 2 measured")
    assert kin.at_steady_rate == pytest.approx(1.5 / 2**20 * 2**21)
    t = P.report(table, 1.0)["totals"]
    assert t["gpu_seconds_per_token"] == 7.0 and t["steady_time_share"] == pytest.approx(3.0 / 7.0)
    assert t["verifier_seconds_per_token"]["block_types"] == 2
    whole = P.Flock(rows, (), None).table(P.Census.of({"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        name: {"k_log": defs[name], "of": {"Head_v1": "row"}, "prefill": 1, "decode": 0} for name in list(defs)[:3]}}))
    assert P.report(whole, 1.0)["totals"]["status"] == "modeled"


def test_a_types_row_names_the_definition_that_priced_it_and_which_form_of_the_type_it_is():
    """A row staged for its census type in another Definition prices the type as any row of it does, and the type's
    row names the Definition its statements are of: as the type's Boolean version where the tree registers it as that,
    as another Definition where it doesn't, and with no qualifier where it is the type's own. A type priced from its
    family's row names that row's Definition the same way, and a type priced off the steady state names none. The
    form moves no figure."""

    defs = {"Head_v1{K=4,FIRST=false}": 20, "Head_v1{K=4,FIRST=true}": 20, "Head_v1{K=4,LAYER=9}": 21,
            "Tail_v1{K=4,FIRST=false}": 20, "Tail_v1{K=4,LAYER=9}": 21, "Odd_v1{N=2}": 20, "Bare_v1{N=2}": 20}
    census = P.Census.of({"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        name: {"k_log": k, "of": {name.split("{")[0]: "row"}, "prefill": 1, "decode": 0} for name, k in defs.items()}})
    head, first, layer, tail, tail_layer, odd, bare = defs
    run = "r20260101-000000-0001"
    rows = (steady("head-bits", "Head_v1{K=4,FIRST=False}", run, 8, 16.0, form="Head_v2{K=4,FIRST=False}"),
            steady("head-first", "Head_v1{K=4,FIRST=True}", run, 8, 8.0),
            steady("tail", "Tail_v1{K=4,FIRST=False}", run, 8, 8.0),
            steady("odd-v3", "Odd_v1{N=2}", run, 8, 8.0, form="Odd_v3{N=2}"))
    # the tree's Boolean version of each census type, by the type's census name: `Odd_v3` is none of `Odd_v1`
    registered = {head: "Head_v2{K=4,FIRST=False}", odd: "Odd_v2{N=2}"}
    flock = P.Flock(rows, (), None, boolean=lambda block, definition: registered.get(block) == definition)
    table = flock.table(census)
    by = {r.block: r.cost for r in table.rows}
    assert [c.source for c in by.values()] == [P.STEADY, P.STEADY, P.FAMILY, P.STEADY, P.FAMILY, P.STEADY, P.MISSING]

    # a Boolean row prices the word-form census type, and its line says so
    assert by[head].note == ("`head-bits` (its Boolean version, `Head_v2{K=4,FIRST=False}`), 1 prover, full statements "
                             "(out/rate/rate.json)")
    # a row of the type's own Definition names it with no qualifier
    assert by[first].note == "`head-first` (`Head_v1{K=4,FIRST=True}`), 1 prover, full statements (out/rate/rate.json)"
    # a Definition the tree doesn't register as the type's Boolean version is another Definition
    assert by[odd].note == "`odd-v3` (another Definition, `Odd_v3{N=2}`), 1 prover, full statements (out/rate/rate.json)"
    # a type priced from its family's row names the block type that row measured and the Definition it was staged in
    assert by[layer].note == ("its Definition and K's, `head-bits` (`Head_v1{K=4,FIRST=False}` in its Boolean version, "
                              "`Head_v2{K=4,FIRST=False}`), 1 prover, full statements, at 2^1 its seconds; the dearest of 2 "
                              "measured")
    assert by[tail_layer].note == ("its Definition and K's, `tail` (`Tail_v1{K=4,FIRST=False}`), 1 prover, full statements, "
                                   "at 2^1 its seconds")
    named = {head: "Head_v2{K=4,FIRST=False}", first: "Head_v1{K=4,FIRST=True}", layer: "Head_v2{K=4,FIRST=False}",
             tail: "Tail_v1{K=4,FIRST=False}", tail_layer: "Tail_v1{K=4,FIRST=False}", odd: "Odd_v3{N=2}", bare: None}
    assert {name: c.definition for name, c in by.items()} == named
    doc = P.report(table, 1.0)
    assert {r["block"]: r["definition"] for r in doc["rows"]} == named

    # with nothing to say what the tree registers, no row is called a Boolean version
    unsaid = {r.block: r.cost for r in P.Flock(rows, (), None).table(census).rows}
    assert unsaid[head].note == ("`head-bits` (another Definition, `Head_v2{K=4,FIRST=False}`), 1 prover, full statements "
                                 "(out/rate/rate.json)")
    assert "`Head_v1{K=4,FIRST=False}` in another Definition, `Head_v2{K=4,FIRST=False}`" in unsaid[layer].note

    # the form prices nothing: the same rows staged in their types' own Definitions give every figure the same
    plain = P.Flock(tuple(replace(s, form=None) for s in rows), (), None).table(census)

    def figures(t: P.Table) -> list[P.Row]:
        return [replace(r, cost=replace(r.cost, note="", definition=None)) for r in t.rows]

    assert figures(plain) == figures(table)
    same = P.report(plain, 1.0)
    assert (same["totals"], same["sources"]) == (doc["totals"], doc["sources"])
    assert {r["block"]: r["definition"] for r in same["rows"]} == named | {
        head: "Head_v1{K=4,FIRST=False}", layer: "Head_v1{K=4,FIRST=False}", odd: "Odd_v1{N=2}"}
    # a sampling plan takes the prover's rows, each with its Definition and its line
    cells = [{"block": name, "type": "", "taken": 1.0} for name in defs]
    spot = P.Spot(flock, plan(cells), P.Serving(0.5, 4.0, "stated")).table(census)
    assert [(r.cost.definition, r.cost.note) for r in spot.rows] == [(c.definition, c.note) for c in by.values()]


def test_the_steady_rows_are_the_types_measured_at_steady_state_as_the_planner_reads_them():
    """`--steady-rows`' file holds a row for each block type its own steady-state row measures, by its census name, with
    the statement's GPU-seconds and blocks that row timed, and none for a type priced from its family's row, a single
    session or the k_log model; every row has the fields the break-even planner's reader requires."""

    defs = {'Head_v1{K=4,DOT={"fn":"Dot_v1"},FIRST=false}': 20, 'Head_v1{K=4,DOT={"fn":"Dot_v1"},FIRST=true}': 20,
            "Head_v1{K=4,LAYER=9}": 21, "Head_v1{K=8,FIRST=false}": 20}
    census = P.Census.of({"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        name: {"k_log": k, "of": {"Head_v1": "row"}, "prefill": 1, "decode": 0} for name, k in defs.items()}})
    rows = (steady("head", "Head_v1{K=4,DOT=Dot_v1,FIRST=False}", "r20260101-000000-0001", 8, 12.0),
            steady("head-first", "Head_v1{K=4,DOT=Dot_v1,FIRST=True}", "r20260101-000000-0002", 4, 16.0, provers=1, cores=24,
                   art="art:" + "ab" * 32))
    doc = P.report(P.Flock(rows, SESSIONS, "art:card").table(census), 1.0)
    assert [r["source"] for r in sorted(doc["rows"], key=lambda r: list(defs).index(r["block"]))] == [
        P.STEADY, P.STEADY, P.FAMILY, P.MODELED]
    first, later = list(defs)[1], list(defs)[0]
    assert P.steady_rows(doc) == {"rows": [
        {"block": first, "statement_seconds": 16.0, "blocks": 4, "provers": 1, "cores": 24, "record": "art:" + "ab" * 32},
        {"block": later, "statement_seconds": 12.0, "blocks": 8, "provers": 1, "cores": 10, "record": "r20260101-000000-0001"}]}
    reader = ast.parse((REPO / "benchmarks" / "spot_figures" / "breakeven" / "reprice.py").read_text())
    fields = next(ast.literal_eval(n.value) for n in reader.body
                  if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "STEADY_FIELDS")
    assert fields and all(set(fields) <= set(r) for r in P.steady_rows(doc)["rows"])


def test_a_type_is_priced_at_the_stated_prover_count_from_rows_at_that_count_only():
    """Rows at N provers, then the fullest statement, then the newest run: a newer or fuller row at another count
    prices nothing, so a type's figure doesn't depend on which counts were run, or in what order."""

    census = P.Census.of(CENSUS)
    gemm, norm = P.definition_id(GEMM64), P.definition_id(NORM)
    rows = (
        steady("gemm64", gemm, "r20260101-000000-0001", 8, 16.0),                              # one prover, full
        steady("gemm64", gemm, "r20260102-000000-0002", 4, 6.0),                               # one prover, newer, half full
        steady("gemm64", gemm, "r20260103-000000-0003", 8, 4.0, provers=2),                    # newest and full, two provers
        steady("gemm64", gemm, "r20260103-000000-0003", 8, 2.0, provers=4, refused="void: out of device memory"),
        steady("norm", norm, "r20260103-000000-0003", 8, 24.0, provers=2, k_log=22),           # measured at two provers only
    )

    def priced(provers: int) -> dict[str, P.Cost]:
        return {r.block: r.cost for r in P.Flock(rows, SESSIONS, "art:card", provers=provers).table(census).rows}

    one, two, four = priced(1), priced(2), priced(4)
    assert (one[GEMM64].source, one[GEMM64].seconds, one[GEMM64].record, one[GEMM64].provers) == (
        P.STEADY, 2.0, "r20260101-000000-0001", 1)
    assert one[GEMM64].note.startswith(f"`gemm64` (`{gemm}`), 1 prover, full statements")
    # the norm has no row at one prover: it is modeled, and its row says which counts measured it
    assert one[NORM].source == P.MODELED and one[NORM].provers is None
    assert one[NORM].note == "measured at steady state at 2 provers, which price no type at 1"
    assert (two[GEMM64].seconds, two[GEMM64].provers, two[NORM].seconds, two[NORM].source) == (0.5, 2, 3.0, P.STEADY)
    assert two[GEMM64].note.startswith(f"`gemm64` (`{gemm}`), 2 provers, full statements")
    # at four provers the only row is void: the type is priced off the steady state, naming the attempt and the counts
    assert four[GEMM64].source == P.SINGLE and four[GEMM64].provers is None
    assert four[GEMM64].note == ("steady state tried as `gemm64` at 4 provers in r20260103-000000-0003 (out/rate/rate.json): "
                                 "void: out of device memory; measured at steady state at 1, 2 provers, which price no "
                                 "type at 4")
    assert "at 2 provers priced at" in P.Flock(rows, SESSIONS, "art:card", provers=2).table(census).range_basis
    assert [r["provers"] for r in P.report(P.Flock(rows, SESSIONS, "art:card").table(census), 1.0)["rows"]] == [1, None, None]


def test_a_type_is_priced_at_the_hosts_cores_else_from_the_fewest_cores_above():
    """Among the rows at the stated prover count, those timed at the host's cores price a type, however full or new a
    row at more cores is. A type they measured only at more cores takes the fewest cores above, then the fullest
    statement, and its note says so; a row timed at fewer cores prices nothing, and is named like a row at another
    prover count. The host's cores default to `HOST_CORES`, and a row that states none is one at `UNSTATED_CORES`."""

    census = P.Census.of(CENSUS)
    gemm, gemm32, norm = (P.definition_id(name) for name in (GEMM64, GEMM32, NORM))
    rows = (
        steady("gemm64", gemm, "r20260101-000000-0001", 4, 8.0),                               # the host's 10 cores, half full
        steady("gemm64", gemm, "r20260102-000000-0002", 8, 12.0, cores=12),
        steady("gemm64", gemm, "r20260103-000000-0003", 8, 8.0, cores=24),                     # the newest, and full
        steady("norm", norm, "r20260101-000000-0001", 8, 24.0, k_log=22, cores=24),
        steady("norm", norm, "r20260101-000000-0001", 4, 8.0, k_log=22, cores=12),             # the fewest above 10, half full
        steady("norm", norm, "r20260103-000000-0003", 8, 40.0, k_log=22, cores=6),             # under the host's share
        steady("gemm32", gemm32, "r20260103-000000-0003", 8, 4.0, k_log=19, cores=6),          # measured under it only,
        steady("gemm32", gemm32, "r20260103-000000-0003", 8, 4.0, k_log=19, provers=2),        # and at two provers
    )

    def priced(**at: int) -> dict[str, P.Cost]:
        return {r.block: r.cost for r in P.Flock(rows, SESSIONS, "art:card", **at).table(census).rows}

    assert (P.HOST_CORES, P.UNSTATED_CORES, P.Flock().cores) == (10, 24, 10)
    assert P.Steady("gemm64", gemm, 20, 8, 8, "r20260101-000000-0001", R.RATES, 8.0).cores == 24
    ten = priced()
    assert ten == priced(cores=10)
    assert (ten[GEMM64].source, ten[GEMM64].seconds, ten[GEMM64].record, ten[GEMM64].cores) == (
        P.STEADY, 2.0, "r20260101-000000-0001", 10)
    assert ten[GEMM64].note == f"`gemm64` (`{gemm}`), 1 prover, statements 4/8 full (out/rate/rate.json)"
    assert (ten[NORM].source, ten[NORM].seconds, ten[NORM].cores) == (P.STEADY, 2.0, 12)
    assert ten[NORM].note == (f"`norm` (`{norm}`), 1 prover, statements 4/8 full (out/rate/rate.json); timed at 12 cores, "
                              "above the host's 10")
    assert ten[GEMM32].source == P.MODELED and ten[GEMM32].cores is None
    assert ten[GEMM32].note == ("measured at steady state at 2 provers, which price no type at 1; measured at steady state "
                                "at 6 cores, which price no type at 10")
    # on a host of 12 cores the rows at 12 are its own, and the row at 10 prices nothing
    twelve = priced(cores=12)
    assert (twelve[GEMM64].seconds, twelve[GEMM64].record, twelve[GEMM64].cores) == (1.5, "r20260102-000000-0002", 12)
    assert (twelve[NORM].seconds, twelve[NORM].cores) == (2.0, 12)
    assert "timed at" not in twelve[GEMM64].note + twelve[NORM].note
    wide = priced(cores=24)
    assert (wide[GEMM64].seconds, wide[GEMM64].record, wide[NORM].seconds) == (1.0, "r20260103-000000-0003", 3.0)
    assert priced(cores=16)[GEMM64].note == (f"`gemm64` (`{gemm}`), 1 prover, full statements (out/rate/rate.json); timed at "
                                             "24 cores, above the host's 16")
    # on a host of more cores than any row was timed at, no row prices a type: each names the cores that measured it
    over = priced(cores=32)
    assert (over[GEMM64].source, over[NORM].source) == (P.SINGLE, P.MODELED)
    assert over[GEMM64].note == "measured at steady state at 10, 12, 24 cores, which price no type at 32"
    assert over[NORM].note == "measured at steady state at 6, 12, 24 cores, which price no type at 32"
    # the prover count and the cores both hold: at two provers the K=32 GEMM's row at the host's cores prices it
    assert (priced(provers=2)[GEMM32].source, priced(provers=2)[GEMM32].cores) == (P.STEADY, 10)

    # the totals give the share of the proving time that rests on rows timed above the host's cores
    table = P.Flock(rows, SESSIONS, "art:card").table(census)
    total = 2.0 * 2.0 + 1.0 * ten[GEMM32].seconds + 0.5 * 2.0
    out = P.report(table, 1.0)
    assert table.cores == 10 and out["totals"]["gpu_seconds_per_token"] == pytest.approx(total)
    assert out["totals"]["above_cores"] == {
        "cores": 10, "block_types": 1, "time_share": pytest.approx(1.0 / total), "timed_at": [12],
        "sources": [{"source": P.STEADY, "block_types": 1, "time_share": pytest.approx(1.0 / total)}]}
    assert {r["block"]: r["cores"] for r in out["rows"]} == {GEMM64: 10, NORM: 12, GEMM32: None}
    assert P.report(P.Flock(rows, SESSIONS, "art:card", cores=12).table(census), 1.0)["totals"]["above_cores"] == {
        "cores": 12, "block_types": 0, "time_share": 0.0, "timed_at": [], "sources": []}


def test_the_family_rule_takes_the_rows_that_price_each_type_at_the_hosts_cores():
    """A type with no row of its own takes its Definition and K's rows as the host's cores choose them: a row timed
    above them prices its kin and says so, and one timed under them prices neither its type nor its kin. The share of
    the proving time that rests on rows timed above the host's cores counts both, by source, at a sampling plan too."""

    defs = {"Head_v1{K=4,FIRST=false}": 20, "Head_v1{K=4,FIRST=true}": 20, "Head_v1{K=4,LAYER=9}": 21}
    census = P.Census.of({"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        name: {"k_log": k, "of": {"Head_v1": "row"}, "prefill": 1, "decode": 0} for name, k in defs.items()}})
    own, first, layer = defs
    rows = (steady("head", "Head_v1{K=4,FIRST=False}", "r20260101-000000-0001", 8, 8.0),
            steady("head-first", "Head_v1{K=4,FIRST=True}", "r20260101-000000-0001", 8, 16.0, cores=24))
    table = P.Flock(rows, (), None).table(census)
    by = {r.block: r.cost for r in table.rows}
    assert [(c.source, c.seconds, c.cores) for c in by.values()] == [(P.STEADY, 1.0, 10), (P.STEADY, 2.0, 24), (P.FAMILY, 4.0, 24)]
    assert by[layer].note == ("its Definition and K's, `head-first` (`Head_v1{K=4,FIRST=True}`), 1 prover, full statements, "
                              "at 2^1 its seconds; the dearest of 2 measured; timed at 24 cores, above the host's 10")
    out = P.report(table, 1.0)
    t = out["totals"]
    assert t["steady_time_share"] == pytest.approx(3.0 / 7.0)
    assert t["above_cores"] == {
        "cores": 10, "block_types": 2, "time_share": pytest.approx(6.0 / 7.0), "timed_at": [24],
        "sources": [{"source": P.STEADY, "block_types": 1, "time_share": pytest.approx(2.0 / 7.0)},
                    {"source": P.FAMILY, "block_types": 1, "time_share": pytest.approx(4.0 / 7.0)}]}

    def page(out: dict, cores: int = 10) -> str:
        return R.render({"workload": "toy", "model": "Toy", "provers": 1, "cores": cores, "inputs": [], "refused": [],
                         "macs_in_blocks": 1.0, **out,
                         "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
                         "periods": {"gates_basis": "stated", "only": None, "rows": []}})

    text = page(out)
    assert ("- host cores (`--cores`, a stated input): 10 CPU cores a card, node A's share: the cores its provider gives a "
            "card; a type is priced from the rows whose provers held that many, else from those that held the fewest above "
            "it, which its row names; a row that held fewer prices nothing, and one whose record states no cores is read "
            "as 24\n") in text
    assert ("- steady-state rows timed above the host's 10 cores: 85.71% of its proving time rests on them: 2 block types "
            "no row at 10 cores measures, priced from rows timed at 24 cores (1 measured at steady state, 28.57%; 1 steady "
            "state, its family's, 57.14%)\n") in text

    # at a sampling plan the share is of the proving time the plan's blocks take
    cells = [{"block": name, "type": "", "taken": taken} for name, taken in zip(defs, (1.0, 1.0, 2.0))]
    spot = P.Spot(P.Flock(rows, (), None), plan(cells), P.Serving(0.5, 4.0, "stated")).table(census)
    assert spot.cores == 10
    assert P.report(spot, 1.0)["totals"]["above_cores"]["time_share"] == pytest.approx((10 * 2.0 + 20 * 4.0) / 110.0)

    # a row at the host's cores prices its type before the fuller, newer one above them does, and its kin with it
    exact = (*rows, steady("head-first", "Head_v1{K=4,FIRST=True}", "r20260101-000000-0001", 4, 6.0))
    table = P.Flock(exact, (), None).table(census)
    assert [(r.cost.source, r.cost.seconds, r.cost.cores) for r in table.rows] == [
        (P.STEADY, 1.0, 10), (P.STEADY, 1.5, 10), (P.FAMILY, 3.0, 10)]
    out = P.report(table, 1.0)
    assert out["totals"]["above_cores"] == {"cores": 10, "block_types": 0, "time_share": 0.0, "timed_at": [], "sources": []}
    assert "- steady-state rows timed above the host's 10 cores: 0.00% of its proving time rests on them\n" in page(out)

    # on a host of 24 cores the row timed at 10 prices neither its own type nor its kin: both take the row at 24
    wide = P.Flock(rows, (), None, cores=24).table(census)
    by = {r.block: r.cost for r in wide.rows}
    assert [(c.source, c.seconds, c.cores) for c in by.values()] == [(P.FAMILY, 2.0, 24), (P.STEADY, 2.0, 24), (P.FAMILY, 4.0, 24)]
    assert by[own].note == ("its Definition and K's, `head-first` (`Head_v1{K=4,FIRST=True}`), 1 prover, full statements, at "
                            "2^0 its seconds; measured at steady state at 10 cores, which price no type at 24")
    text = page(P.report(wide, 1.0), cores=24)
    assert "- host cores (`--cores`, a stated input): 24 CPU cores a card; a type is priced" in text and "node A" not in text
    assert "- steady-state rows timed above the host's 24 cores: 0.00% of its proving time rests on them\n" in text


def test_a_row_an_art_holds_is_its_runs_and_names_the_art():
    """A row an art holds in place of its run's own is ordered as that run's, and the type's row names the art as its
    record and the run that measured it."""

    census = P.Census.of(CENSUS)
    gemm, norm = P.definition_id(GEMM64), P.definition_id(NORM)
    art = "art:" + "ab" * 32
    rows = (
        steady("gemm64", gemm, "r20260101-000000-0001", 8, 16.0),
        steady("gemm64", gemm, "r20260102-000000-0002", 8, 8.0, art=art),                      # the newer run's, in an art
        steady("norm", norm, "r20260101-000000-0001", 8, 24.0, k_log=22),
        steady("norm", norm, "r20260102-000000-0002", 8, None, k_log=22, art=art, refused="its row lacks `flat`"),
    )
    cost = {r.block: r.cost for r in P.Flock(rows, SESSIONS, "art:card").table(census).rows}
    assert (cost[GEMM64].source, cost[GEMM64].seconds, cost[GEMM64].record) == (P.STEADY, 1.0, art)
    assert cost[GEMM64].note == (f"`gemm64` (`{gemm}`), 1 prover, full statements (rate.json of art:abababababab, measured in "
                                 "r20260102-000000-0002)")
    # the art's row of the norm isn't a steady-state measurement, so the older run's own prices it
    assert (cost[NORM].source, cost[NORM].seconds, cost[NORM].record) == (P.STEADY, 3.0, "r20260101-000000-0001")
    tried = {r.block: r.cost for r in P.Flock(rows[3:], SESSIONS, "art:card").table(census).rows}[NORM]
    assert tried.source == P.MODELED and tried.note == (
        "steady state tried as `norm` at 1 prover in r20260102-000000-0002 (rate.json of art:abababababab): its row lacks `flat`")


def test_a_steady_row_of_no_block_type_of_the_census_is_unread_not_dropped():
    """A steady-state row whose staged Definition the census doesn't hold, or holds no longer under its cut, is on no
    block type's row: the table lists it as unread, at whatever prover count and cores, with its figure where it is a
    steady-state measurement and why it isn't one otherwise. It enters no figure: no row, no total, and not the
    range's other end, whose median is over the census's steady-state types alone."""

    census = P.Census.of(CENSUS)
    norm, art = P.definition_id(NORM), "art:" + "cd" * 32
    priced = steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0)
    strays = (
        steady("norm", norm, "r20260102-000000-0002", 8, 24.0, k_log=22),                     # a block type the cut replaces
        steady("old-linear", "Linear_v0{K=64}", "r20260103-000000-0003", 4, 6.0, provers=2, art=art, cores=24),
        steady("old-linear", "Linear_v0{K=64}", "r20260103-000000-0003", 8, None, refused="void: out of device memory"),
        steady("idle", "Idle_v1{N=1}", "r20260101-000000-0001", 8, 2.0, cores=6),            # a type no step ran
    )
    flock = P.Flock((priced, *strays), SESSIONS, "art:card")
    whole = flock.table(census)
    assert whole.unread == strays[1:] and {r.block: r.cost.source for r in whole.rows}[NORM] == P.STEADY
    recut = flock.table(census.recut(CUT))
    assert recut.unread == strays
    out = P.report(recut, 1.0)
    assert out["unread"] == [
        {"record": "r20260102-000000-0002", "run": "r20260102-000000-0002", "type": "norm", "definition": norm,
         "provers": 1, "cores": 10, "seconds": 3.0, "why": None},
        {"record": art, "run": "r20260103-000000-0003", "type": "old-linear", "definition": "Linear_v0{K=64}",
         "provers": 2, "cores": 24, "seconds": 1.5, "why": None},
        {"record": "r20260103-000000-0003", "run": "r20260103-000000-0003", "type": "old-linear",
         "definition": "Linear_v0{K=64}", "provers": 1, "cores": 10, "seconds": None, "why": "void: out of device memory"},
        {"record": "r20260101-000000-0001", "run": "r20260101-000000-0001", "type": "idle", "definition": "Idle_v1{N=1}",
         "provers": 1, "cores": 6, "seconds": 0.25, "why": None}]
    alone = P.report(P.Flock((priced,), SESSIONS, "art:card").table(census.recut(CUT)), 1.0)
    assert alone["unread"] == [] and {k: v for k, v in out.items() if k != "unread"} == {k: v for k, v in alone.items() if k != "unread"}
    # a sampling plan takes the prover's table, its unread rows with it
    cells = [{"block": b.name, "type": "", "taken": 1.0} for b in census.blocks]
    assert P.Spot(flock, plan(cells), P.Serving(0.5, 4000.0, "stated")).table(census).unread == strays[1:]

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.9, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert text.endswith(
        "\nUnread (labelled steady-state rows whose staged Definition is no block type of this census under its cut; "
        "they are in no figure above):\n\n"
        f"- r20260102-000000-0002: `norm` at 1 prover, `{norm}`: measured at steady state, 3,000 ms per block\n"
        "- art:cdcdcdcdcdcd, run r20260103-000000-0003: `old-linear` at 2 provers, `Linear_v0{K=64}`: measured at steady "
        "state, 1,500 ms per block\n"
        "- r20260103-000000-0003: `old-linear` at 1 prover, `Linear_v0{K=64}`: not a steady-state measurement: void: out "
        "of device memory\n"
        "- r20260101-000000-0001: `idle` at 1 prover, `Idle_v1{N=1}`: measured at steady state, 250 ms per block\n")
    assert "Unread" not in R.render(page | alone)


def test_a_single_session_prices_its_definition_and_k_at_another_size():
    """A block type the card doesn't measure by name takes its Definition and K's measurement at 2^Δk the seconds and
    2^-Δk the statement, Δk between the two k_logs as the census holds them."""

    doc = {"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        "Gemm_v1{K=64,N=1}": {"k_log": 21, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=4}": {"k_log": 23, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=0}": {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0}}}
    # the card staged N=1 at k_log 20; the census holds it at 21, which is what Δk is taken from
    card = (P.Session("Gemm_v1{K=64,N=1}", 20, 0.5, 2.0, 1 << 15),)
    cost = {r.block: r.cost for r in P.Flock((), card, "art:card").table(P.Census.of(doc)).rows}
    assert cost["Gemm_v1{K=64,N=1}"].seconds == 0.5 + 2.0 / (1 << 15)
    assert cost["Gemm_v1{K=64,N=4}"].seconds == 0.5 * 4 + 2.0 / (1 << 13) and "at 2^2 its seconds" in cost["Gemm_v1{K=64,N=4}"].note
    assert cost["Gemm_v1{K=64,N=0}"].seconds == 0.5 / 2 + 2.0 / (1 << 16)
    assert {c.source for c in cost.values()} == {P.SINGLE}


def test_the_report_gives_each_sources_share_and_the_totals_with_their_range():
    census = P.Census.of(CENSUS).recut(CUT)
    table = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),), SESSIONS,
                    "art:card").table(census)
    serving = P.Serving(0.5, 4000.0, "stated")
    assert serving.gpu_seconds_per_token(census) == 2 * 1000 / (0.5 * 4000.0)
    doc = P.report(table, 1.0)
    total = table.gpu_seconds_per_token()
    low, high = table.gpu_seconds_range()
    t = doc["totals"]
    assert t["gpu_seconds_per_token"] == total and t["tokens_per_gpu_second"] == 1 / total
    assert t["gpu_seconds_per_token_range"] == [low, high] and t["tokens_per_gpu_second_range"] == [1 / high, 1 / low]
    assert t["gpus_per_serving_gpu"] == total and t["blocks_per_token"] == 4.25
    # the cut's bare block type has no figure, so the totals are partial and name it; this cut holds no MACs
    assert t["status"] == "partial" and t["steady_time_share"] == pytest.approx(2.0 / total)
    assert t["missing"] == [{"block": "Bare_v1{N=4}", "per_token": 0.25, "block_share": 0.25 / 4.25, "mac_share": None}]
    assert t["verifier_seconds_per_token"] == {"value": 1.0, "block_types": 1, "block_share": 2.0 / 4.25}
    assert t["proof_bytes_per_token"] is None
    by = {s["source"]: s for s in doc["sources"]}
    assert list(by) == list(P.SOURCES)
    assert (by[P.STEADY]["block_types"], by[P.STEADY]["per_token"], by[P.STEADY]["gpu_seconds_per_token"]) == (1, 2.0, 2.0)
    assert (by[P.MODELED]["block_types"], by[P.MISSING]["block_types"], by[P.SINGLE]["block_types"]) == (2, 1, 0)
    assert by[P.MISSING]["gpu_seconds_per_token"] == 0 and by[P.MISSING]["block_share"] == 0.25 / 4.25
    assert sum(s["time_share"] for s in doc["sources"]) == pytest.approx(1.0)
    assert [r["block"] for r in doc["rows"]][-1] == "Bare_v1{N=4}" and doc["rows"][-1]["time_share"] is None
    assert doc["rows"][0]["time_share"] == max(r["time_share"] or 0 for r in doc["rows"])

    # measured only when every block type is measured at steady state; modeled when one is priced another way
    whole = P.Census.of({**CENSUS, "definitions": {GEMM64: CENSUS["definitions"][GEMM64]}})
    unverified = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),)
    measured = P.report(P.Flock(unverified, SESSIONS, "art:card").table(whole), 1.0)["totals"]
    assert measured["status"] == "measured" and measured["verifier_seconds_per_token"] is None and measured["missing"] == []
    modeled = P.report(P.Flock(unverified, SESSIONS, "art:card").table(P.Census.of(CENSUS)), 1.0)["totals"]
    assert modeled["status"] == "modeled" and modeled["missing"] == [] and modeled["steady_time_share"] < 1.0


def test_a_total_that_leaves_a_block_type_out_is_partial_and_names_it(monkeypatch):
    """One steady-state type and one type with no figure at three quarters of the blocks: steady-state measurements
    cover all of the time the total prices, and the total is still partial, never measured. It names the type it leaves
    out with its share of the blocks, and of the MACs where the census holds them."""

    doc = {"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 100, "macs": 80}, "definitions": {
        GEMM64: {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0, "macs": 60},
        "Unpriced_v1{N=1}": {"of": {"Unpriced_v1": "row"}, "prefill": 3, "decode": 0, "macs": 5},
        "Rare_v1{N=1}": {"of": {"Rare_v1": "row"}, "prefill": 1, "decode": 0, "macs": 5}}}
    reports = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),)
    out = P.report(P.Flock(reports, SESSIONS, "art:card").table(P.Census.of(doc)), 1.0)
    t = out["totals"]
    assert {s["source"]: s["block_types"] for s in out["sources"]} == {P.STEADY: 1, P.FAMILY: 0, P.SINGLE: 0, P.MODELED: 0, P.MISSING: 2}
    assert (t["steady_time_share"], t["gpu_seconds_per_token"], t["status"]) == (1.0, 1.0, "partial")
    assert t["missing"] == [
        {"block": "Unpriced_v1{N=1}", "per_token": 3.0, "block_share": 0.6, "mac_share": 0.1875},
        {"block": "Rare_v1{N=1}", "per_token": 1.0, "block_share": 0.2, "mac_share": 0.0625}]

    # a census that holds no MACs for one of its types gives the share of the blocks alone
    del doc["definitions"][GEMM64]["macs"]
    bare = P.report(P.Flock(reports, SESSIONS, "art:card").table(P.Census.of(doc)), 1.0)["totals"]
    assert bare["status"] == "partial" and [(m["block_share"], m["mac_share"]) for m in bare["missing"]] == [(0.6, None), (0.2, None)]
    # at a sampling plan the shares are of the blocks the plan samples
    cells = [{"block": name, "type": "", "taken": 1.0} for name in doc["definitions"]]
    sampled = P.report(P.Spot(P.Flock(reports, SESSIONS, "art:card"), plan(cells), P.Serving(0.5, 400.0, "stated")).table(
        P.Census.of(doc)), 1.0)["totals"]
    assert sampled["status"] == "partial" and [m["block_share"] for m in sampled["missing"]] == pytest.approx([1 / 3, 1 / 3])

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.8, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert "- prover GPUs per serving GPU: 1, partial: steady-state measurements cover 100.0% of the proving time it prices" in text
    assert ("- partial: 2 block type(s) with no figure, whose proving every total here leaves out: `Unpriced_v1{N=1}` "
            "(60.00% of the blocks, 18.75% of the blocks' MACs); `Rare_v1{N=1}` (20.00% of the blocks, 6.25% of the "
            "blocks' MACs)\n") in text
    assert "not measured at that size, and partial as that ratio is:" in text
    monkeypatch.setattr(R, "MISSING_NAMED", 1)
    assert "(60.00% of the blocks, 18.75% of the blocks' MACs); and 1 more, which `--json` names\n" in R.render(page)
    whole = R.render(page | {"totals": t | {"status": "measured", "missing": []}})
    assert "measured: steady-state measurements cover 100.0% of its proving time" in whole and "partial" not in whole


def lean(definition: str, run: str, seconds: tuple[float, ...], size: int, *, k_log: int = 20, blocks: int = 8,
         full: int = 8) -> P.LeanVerify:
    return P.LeanVerify(definition, k_log, blocks, full, seconds, size, run, "node-a (64 cores, a CPU)", L.COVER_LABEL)


def test_the_lean_verifiers_seconds_and_the_proofs_bytes_are_one_types_and_scaled_to_the_rest():
    """A block type the Lean verifier ran on takes its statement's median seconds and its proof's bytes over the
    statement's blocks; every other type with a k_log takes the same seconds and bytes a witness bit; a type with none
    takes nothing. The totals are modeled, and say how much of each the measured type holds."""

    census = P.Census.of(CENSUS).recut(CUT)
    rows = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),)
    timed = (lean(P.definition_id(GEMM64), "r20260105-000000-0005", (40.0, 16.0, 24.0), 4000),
             lean(P.definition_id(GEMM64), "r20260104-000000-0004", (1.0,), 1),             # an older run of the type
             lean("Other_v1{N=2}", "r20260106-000000-0006", (1.0,), 1))                       # no block type of the census
    table = P.Flock(rows, SESSIONS, "art:card", lean=timed).table(census)
    assert [v.run for v in table.lean] == ["r20260105-000000-0005"]
    by = {r.block: r.cost for r in table.rows}
    assert [(by[n].lean_seconds, by[n].proof_bytes, by[n].lean_source) for n in (GEMM64, GEMM32, "Half_v1{N=4}", "Bare_v1{N=4}")] == [
        (3.0, 500.0, P.LEAN_MEASURED), (1.5, 250.0, P.LEAN_SCALED), (6.0, 1000.0, P.LEAN_SCALED), (None, None, None)]
    # the proof server's verifier's seconds stay the steady-state row's own
    assert by[GEMM64].verifier_seconds == 0.5

    doc = P.report(table, 1.0)
    t = doc["totals"]
    assert t["lean_verifier_seconds_per_token"] == {
        "value": 2.0 * 3.0 + 1.5 + 6.0, "status": "modeled", "measured_block_types": 1, "scaled_block_types": 2,
        "unpriced_block_types": 1, "measured_share": 6.0 / 13.5, "basis": P.LEAN_SECONDS_BASIS}
    assert t["proof_bytes_per_token"] == {
        "value": 2250.0, "status": "modeled", "measured_block_types": 1, "scaled_block_types": 2, "unpriced_block_types": 1,
        "measured_share": 1000.0 / 2250.0, "basis": P.PROOF_BYTES_BASIS}
    share = 2.0 * by[GEMM64].seconds / table.gpu_seconds_per_token()
    assert t["lean_verify"] == [{"block": GEMM64, "time_share": share, "k_log": 20, "blocks": 8, "full": 8,
                                 "seconds": [40.0, 16.0, 24.0], "statement_seconds": 24.0, "proof_bytes": 4000,
                                 "run": "r20260105-000000-0005", "host": "node-a (64 cores, a CPU)",
                                     "label": L.COVER_LABEL}]
    row = next(r for r in doc["rows"] if r["block"] == GEMM32)
    assert (row["lean_verifier_seconds"], row["proof_bytes"], row["lean_source"]) == (1.5, 250.0, P.LEAN_SCALED)
    assert t["verifier_seconds_per_token"] == {"value": 1.0, "block_types": 1, "block_share": 2.0 / 4.25}

    # with two types measured, the rest scale at the median of the two statements' seconds, and bytes, a witness bit
    two = P.Flock(rows, SESSIONS, "art:card", lean=(*timed, lean("Half_v1{N=4}", "r20260105-000000-0005", (64.0,), 16000,
                                                                 k_log=21, blocks=4, full=4))).table(census)
    by = {r.block: r.cost for r in two.rows}
    assert (by["Half_v1{N=4}"].lean_seconds, by["Half_v1{N=4}"].proof_bytes, by["Half_v1{N=4}"].lean_source) == (
        16.0, 4000.0, P.LEAN_MEASURED)
    assert by[GEMM32].lean_seconds == pytest.approx((24.0 / 2**23 + 64.0 / 2**23) / 2 * 2**19)
    assert by[GEMM32].proof_bytes == pytest.approx((4000 / 2**23 + 16000 / 2**23) / 2 * 2**19)
    assert [x["block"] for x in P.report(two, 1.0)["totals"]["lean_verify"]] == [GEMM64, "Half_v1{N=4}"]

    # measured only when every block type has a record of its own; none of either with no record
    whole = P.Census.of({**CENSUS, "definitions": {GEMM64: CENSUS["definitions"][GEMM64]}})
    own = P.report(P.Flock(rows, SESSIONS, "art:card", lean=timed).table(whole), 1.0)["totals"]
    assert own["lean_verifier_seconds_per_token"] == {
        "value": 6.0, "status": "measured", "measured_block_types": 1, "scaled_block_types": 0, "unpriced_block_types": 0,
        "measured_share": 1.0, "basis": P.LEAN_SECONDS_BASIS}
    none = P.report(P.Flock(rows, SESSIONS, "art:card").table(census), 1.0)
    assert none["totals"]["lean_verifier_seconds_per_token"] is None and none["totals"]["proof_bytes_per_token"] is None
    assert none["totals"]["lean_verify"] == [] and {r["lean_source"] for r in none["rows"]} == {None}


def test_spot_takes_the_lean_verifiers_seconds_at_the_blocks_its_plan_samples():
    census = P.Census.of(CENSUS)
    flock = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card",
                    lean=(lean(P.definition_id(GEMM64), "r20260105-000000-0005", (24.0,), 4000),))
    cells = [{"block": GEMM64, "type": "gemm64", "taken": 4.0}, {"block": GEMM32, "type": "", "taken": 2.0},
             {"block": NORM, "type": "norm", "taken": 0.8}]
    table = P.Spot(flock, plan(cells), P.Serving(0.5, 4000.0, "stated")).table(census)
    t = P.report(table, 1.0)["totals"]
    # 40, 20 and 8 blocks sampled over 800 tokens, at 3 s a block and at its seconds a witness bit times 2^19 and 2^22
    assert t["lean_verifier_seconds_per_token"]["value"] == pytest.approx((40 * 3.0 + 20 * 1.5 + 8 * 12.0) / 800)
    assert t["proof_bytes_per_token"]["value"] == pytest.approx((40 * 500 + 20 * 250 + 8 * 2000) / 800)
    assert t["lean_verifier_seconds_per_token"]["status"] == "modeled" and t["lean_verify"][0]["block"] == GEMM64
    assert t["lean_verify"][0]["time_share"] == pytest.approx(40 * 1.0 / sum(
        n * r.cost.seconds for n, r in zip((40.0, 20.0, 8.0), table.rows)))


def plan(cells: list[dict], **over) -> P.SamplingPlan:
    return P.SamplingPlan.of({"case": "toy", "scope": "world year", "period_s": 100.0, "q": 0.01, "gpu_seconds": 800.0,
                              "replay_sample": 10.0, "cells": cells} | over, "art:plan")


def test_spot_prices_the_blocks_its_plan_samples_and_refuses_another_census_plan():
    census = P.Census.of(CENSUS)
    flock = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
    serving = P.Serving(0.5, 4000.0, "stated")                    # 1 GPU-second a token: 800 tokens in the plan's period
    cells = [{"block": "Fused[its own name of the K=64 GEMM]", "type": "gemm64", "taken": 4.0},
             {"block": GEMM32, "type": "", "taken": 2.0}, {"block": NORM, "type": "norm", "taken": 0.8}]
    got = plan(cells)
    assert got.sampled == {cells[0]["block"]: 40.0, GEMM32: 20.0, NORM: 8.0}
    assert got.staged == {cells[0]["block"]: "gemm64", NORM: "norm"} and (got.name, got.q) == ("toy", 0.01)

    inner, table = flock.table(census), P.Spot(flock, got, serving).table(census)
    assert table.system == "spot[flock]" and table.range_basis == inner.range_basis
    # a cell is a row's by name, else by the staged type that measured the row; the costs are the prover's own
    assert [(r.block, r.per_token) for r in table.rows] == [(GEMM64, 40 / 800), (GEMM32, 20 / 800), (NORM, 8 / 800)]
    assert [r.cost for r in table.rows] == [r.cost for r in inner.rows]
    # the prover GPUs a serving GPU are the plan's proving over its serving, whatever the serving rate
    proving = sum(n * r.cost.seconds for n, r in zip((40.0, 20.0, 8.0), inner.rows))
    assert P.report(table, 1.0)["totals"]["gpus_per_serving_gpu"] == pytest.approx(proving / 800.0)
    half = P.Serving(0.25, 4000.0, "stated")
    slow = P.report(P.Spot(flock, got, half).table(census), half.gpu_seconds_per_token(census))["totals"]
    assert slow["gpus_per_serving_gpu"] == pytest.approx(proving / 800.0)
    assert slow["tokens_per_gpu_second"] == pytest.approx(400 / proving)

    with pytest.raises(ValueError, match=r"art:plan and the census don't cover each other.*no cell for 1 block type\(s\) "
                                         r"\(Norm_v1.*no block type for 0 cell"):
        P.Spot(flock, plan(cells[:2]), serving).table(census)
    with pytest.raises(ValueError, match=r"no cell for 0 block type\(s\), and the census no block type for 1 cell\(s\) "
                                         r"\(Stray_v1"):
        P.Spot(flock, plan([*cells, {"block": "Stray_v1{N=1}", "type": "", "taken": 1.0}]), serving).table(census)

    # a newer, fuller row staged under another name prices the type; the plan's cell still finds it by the old name
    renamed = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 4, 8.0),
                       steady("gemm-k64", P.definition_id(GEMM64), "r20260103-000000-0003", 8, 4.0)), SESSIONS, "art:card")
    rows = P.Spot(renamed, got, serving).table(census).rows
    assert (rows[0].block, rows[0].per_token, rows[0].cost.staged, rows[0].cost.seconds) == (GEMM64, 40 / 800, "gemm-k64", 0.5)
    assert rows[0].staged_as == ("gemm-k64", "gemm64")
    # the old name's only row superseded: it prices nothing, and still ties the plan's cell to the type
    aliased = P.Flock((steady("gemm-k64", P.definition_id(GEMM64), "r20260103-000000-0003", 8, 4.0),), SESSIONS, "art:card",
                      aliases=((P.definition_id(GEMM64), "gemm64"),))
    rows = P.Spot(aliased, got, serving).table(census).rows
    assert (rows[0].block, rows[0].cost.staged, rows[0].cost.seconds) == (GEMM64, "gemm-k64", 0.5)
    with pytest.raises(ValueError, match="don't cover each other"):
        P.Spot(replace(aliased, aliases=()), got, serving).table(census)


def test_a_plan_that_states_its_own_prices_gets_both_totals_and_where_they_differ():
    """A plan's sample fits its planner's prices, and the table prices the same blocks from its own sources. Where the
    plan states its prices the report gives the total at each, each source's proving time at both and the block type
    that holds most of the difference; the table's own total doesn't move. A plan that prices none of its cells has
    the one total, and one that prices only some is refused."""

    census = P.Census.of(CENSUS)
    flock = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
    serving = P.Serving(0.5, 4000.0, "stated")                    # 800 tokens in the plan's period, as above
    cells = [{"block": GEMM64, "type": "gemm64", "taken": 4.0, "prove_s": 2.0},
             {"block": GEMM32, "type": "", "taken": 2.0, "prove_s": 0.5}, {"block": NORM, "type": "norm", "taken": 0.8, "prove_s": 0.7}]
    got = plan(cells)
    assert got.priced == pytest.approx({GEMM64: 0.5, GEMM32: 0.25, NORM: 0.875})     # a cell's seconds over its blocks
    table = P.Spot(flock, got, serving).table(census)
    assert [r.planned for r in table.rows] == pytest.approx([0.5, 0.25, 0.875])
    out = P.report(table, 1.0)
    t, sampled = out["totals"], (40.0, 20.0, 8.0)
    own = [r.cost.seconds for r in table.rows]
    assert own[0] == 1.0 and t["gpus_per_serving_gpu"] == pytest.approx(sum(n * x for n, x in zip(sampled, own)) / 800)
    its = t["planned"]
    assert its["gpus_per_serving_gpu"] == its["gpu_seconds_per_token"] == pytest.approx((20 + 5 + 7) / 800)
    # no session measured the K=32 GEMM or the norm: one steady-state type, two modeled from k_log
    assert [(x["source"], x["block_types"]) for x in its["sources"]] == [(P.STEADY, 1), (P.MODELED, 2)]
    assert [x["gpu_seconds_per_token"] for x in its["sources"]] == pytest.approx([40 / 800, (20 * own[1] + 8 * own[2]) / 800])
    assert [x["planned_gpu_seconds_per_token"] for x in its["sources"]] == pytest.approx([20 / 800, (5 + 7) / 800])
    gaps = [n * (x - y) for n, x, y in zip(sampled, own, (0.5, 0.25, 0.875))]
    top = max(range(3), key=lambda i: abs(gaps[i]))
    assert its["largest"] == {"block": table.rows[top].block, "seconds": own[top], "planned_seconds": table.rows[top].planned,
                              "difference_share": pytest.approx(gaps[top] / sum(gaps))}
    assert {r["block"]: r["planned_seconds"] for r in out["rows"]} == got.priced

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.9, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert "| ms per block | ms per block in the plan | source |" in text
    assert "| 1,000 | 500 | measured at steady state |" in text
    assert "- the same blocks at the sampling plan's own prices: 0.04 prover GPUs per serving GPU. Its planner sized" in text
    assert ("- where the two differ, this table's proving time over the plan's by this table's source: measured at steady "
            f"state 2x (1 block types), modeled from k_log {(20 * own[1] + 8 * own[2]) / 12:.3g}x (2 block types); `") in text
    assert f"alone is {gaps[top] / sum(gaps):.1%} of the difference" in text
    assert text.count(" --- |") == 9 + 6 + 6                              # the rows' table has the plan's column

    bare = [{k: v for k, v in c.items() if k != "prove_s"} for c in cells]
    one = P.report(P.Spot(flock, plan(bare), serving).table(census), 1.0)
    assert one["totals"]["planned"] is None and one["totals"]["gpus_per_serving_gpu"] == t["gpus_per_serving_gpu"]
    assert "in the plan" not in (plain := R.render(page | one)) and "own prices" not in plain
    assert P.report(flock.table(census), 1.0)["totals"]["planned"] is None
    with pytest.raises(ValueError, match=r"art:plan states its planner's seconds \(`prove_s`\) for 2 cell\(s\) and not for 1 "
                                         r"\(Norm_v1.*prices every cell or none"):
        plan([*cells[:2], bare[2]])
    # a priced cell that takes no block has no seconds a block, and doesn't void the others'
    assert plan([*cells, {"block": "Idle_v1{N=1}", "type": "", "taken": 0.0, "prove_s": 0.0}]).priced["Idle_v1{N=1}"] == 0.0


def test_a_priced_plans_totals_where_the_table_has_no_price_a_higher_one_or_the_same():
    """The edges of a plan's own prices beside the table's. A block type the table has no figure for is in the plan's
    total and in neither side of the comparison. The block type that holds most of the difference is the largest in
    size, whichever price is the higher. Where the two prices agree, or the table prices nothing, no block type is
    named; and a source the plan prices at 0 has no ratio."""

    census = P.Census.of(CENSUS).recut(CUT)
    half, bare = "Half_v1{N=4}", "Bare_v1{N=4}"
    flock = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
    serving = P.Serving(0.5, 4000.0, "stated")                    # 800 tokens in the plan's period
    cells = [{"block": GEMM64, "type": "gemm64", "taken": 4.0, "prove_s": 2.0}, {"block": GEMM32, "type": "", "taken": 2.0, "prove_s": 0.25},
             {"block": half, "type": "", "taken": 1.0, "prove_s": 0.5}, {"block": bare, "type": "", "taken": 0.8, "prove_s": 0.2}]
    got = plan(cells)
    assert got.priced == pytest.approx({GEMM64: 0.5, GEMM32: 0.125, half: 0.5, bare: 0.25})
    own = {r.block: r.cost.seconds for r in flock.table(census).rows}
    assert own[GEMM64] == 1.0 and own[bare] is None and 0.25 < own[GEMM32] < 0.2501 and 1.0 < own[half] < 1.001

    def page(out: dict) -> str:
        return R.render({"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.9,
                         **out, "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
                         "periods": {"gates_basis": "stated", "only": None, "rows": []}})

    def priced(at: dict[str, float]) -> dict:
        return P.report(P.Spot(flock, replace(got, priced=at), serving).table(census), 1.0)

    # a block type with no figure: the plan's total holds it, the table's leaves it out, and the comparison is over
    # the three the table prices
    out = P.report(P.Spot(flock, got, serving).table(census), 1.0)
    t, its = out["totals"], out["totals"]["planned"]
    assert t["status"] == "partial" and [m["block"] for m in t["missing"]] == [bare]
    assert t["gpus_per_serving_gpu"] == pytest.approx((40 + 20 * own[GEMM32] + 10 * own[half]) / 800)
    assert its["gpus_per_serving_gpu"] == pytest.approx((20 + 2.5 + 5 + 2) / 800)
    assert [(x["source"], x["block_types"]) for x in its["sources"]] == [(P.STEADY, 1), (P.MODELED, 2)]
    assert sum(x["planned_gpu_seconds_per_token"] for x in its["sources"]) == pytest.approx((20 + 2.5 + 5) / 800)
    gaps = {GEMM64: 40 * 0.5, GEMM32: 20 * (own[GEMM32] - 0.125), half: 10 * (own[half] - 0.5)}
    assert its["largest"] == {"block": GEMM64, "seconds": 1.0, "planned_seconds": 0.5,
                              "difference_share": pytest.approx(20 / sum(gaps.values()))}
    text = page(out)
    assert "| `Bare_v1{N=4}` | — | 0.01 | — | 250 | missing |" in text          # no ms per block here, the plan's beside it
    assert "- the same blocks at the sampling plan's own prices: 0.0369 prover GPUs per serving GPU." in text

    # the plan prices the K=64 GEMM higher than the table and the rest lower: it is still the block type named, by
    # the size of its difference, and its share is of the net difference, which the others offset
    its = (higher := priced(got.priced | {GEMM64: 3.0}))["totals"]["planned"]
    net = 40 * (1.0 - 3.0) + gaps[GEMM32] + gaps[half]
    assert its["gpus_per_serving_gpu"] > higher["totals"]["gpus_per_serving_gpu"]
    assert its["largest"] == {"block": GEMM64, "seconds": 1.0, "planned_seconds": 3.0,
                              "difference_share": pytest.approx(-80 / net)}
    assert its["largest"]["difference_share"] > 1
    text = page(higher)
    assert "measured at steady state 0.333x (1 block types), modeled from k_log" in text
    assert f"alone is {-80 / net:.1%} of the difference, 1,000 ms per block here and 3,000 ms in the plan" in text

    # the plan's prices are the table's: both totals, and no block type holds a difference
    same = priced({block: 0.25 if seconds is None else seconds for block, seconds in own.items()})
    its = same["totals"]["planned"]
    assert its["largest"] is None
    assert [x["gpu_seconds_per_token"] for x in its["sources"]] == [x["planned_gpu_seconds_per_token"] for x in its["sources"]]
    assert its["gpus_per_serving_gpu"] == pytest.approx(same["totals"]["gpus_per_serving_gpu"] + 8 * 0.25 / 800)
    assert "sampling plan's own prices" in (text := page(same)) and "where the two differ" not in text

    # a table that prices no block type has the plan's total alone
    its = P.report(P.Spot(P.Flock(), got, serving).table(census), 1.0)["totals"]["planned"]
    assert its == {"gpu_seconds_per_token": pytest.approx(29.5 / 800), "gpus_per_serving_gpu": pytest.approx(29.5 / 800),
                   "sources": [], "largest": None}

    # a source whose block types the plan prices at 0 has no ratio; the others keep theirs
    free = P.report(P.Spot(flock, plan([c | {"prove_s": 0.0} if c["block"] in (GEMM32, half) else c for c in cells]),
                           serving).table(census), 1.0)
    its = free["totals"]["planned"]
    assert [(x["source"], x["planned_gpu_seconds_per_token"]) for x in its["sources"]] == [(P.STEADY, 20 / 800), (P.MODELED, 0.0)]
    text = page(free)
    assert "by this table's source: measured at steady state 2x (1 block types); `" in text
    assert "modeled from k_log" not in text[text.index("- where the two differ"):].split("\n")[0]


REFUSALS = json.loads((FIXTURES / "refusals" / "rate" / "rate.json").read_text())
#: each row of the refusals fixture that names a staged Definition, and why it isn't a steady-state measurement: the
#: check of the steady-state rules it didn't pass, the check's answer and its reason, then what it lacks to be priced
ROW_WHYS = {
    "good": None,
    "unaccepted": "`accepted` (fail): 8 of its 9 sessions accepted",
    "refused": "`accepted` (fail): 1 sessions refused",
    "void": "`not_void` (fail): void: out of device memory: prover-gpu0",
    "exited": "`exits` (fail): exited nonzero: prover-gpu0 rc -9",
    "unflat": "`flat` (fail): its statement cache's hit rate wasn't flat",
    "ungpued": "its row lacks `gpus`",
    "bare": "`flat` (cannot tell): its row lacks `flat`; its row lacks `gpus`, `stage.blocks`, `stage.full`",
}
#: a field a made-up row is left without
DROP = object()
#: made-up rows the steady-state rules don't pass and a reader of the fields alone would: what is changed in the good
#: row, and what the rules' check then says
STRICT = [
    ({"unfiled": DROP}, "`accepted` (cannot tell): its row lacks `unfiled`"),
    ({"unfiled": 1}, "`accepted` (fail): 1 sessions never filed by their verifier"),
    ({"warm": True}, "`accepted` (cannot tell): `warm` (True) of its row isn't a count"),
    ({"sessions": "8"}, "`accepted` (cannot tell): `sessions` ('8') of its row isn't a count"),
    ({"refused_n": -1}, "`accepted` (cannot tell): `refused_n` (-1) of its row isn't a count"),
    ({"exits": {}}, "`exits` (cannot tell): its row's `exits` ({}) names no process"),
    ({"exits": []}, "`exits` (cannot tell): its row's `exits` ({}) names no process"),
    ({"exits": {"prover-gpu0": 0.0}}, "`exits` (fail): exited nonzero: prover-gpu0 rc 0.0"),
    ({"exits": [0, False]}, "`exits` (fail): exited nonzero: 1 rc False"),
    ({"exits": 0}, "`exits` (cannot tell): its row's `exits` (0) names no process"),
    ({"error": "RuntimeError: its last verifier died"}, "`not_void` (fail): failed: RuntimeError: its last verifier died"),
    ({"skipped": "past the deadline"}, "`not_void` (fail): skipped: past the deadline"),
    ({"statements_per_s": math.inf}, "`not_void` (fail): no rate: its `statements_per_s` is inf"),
    ({"statements_per_s": math.nan}, "`not_void` (fail): no rate: its `statements_per_s` is nan"),
    ({"statements_per_s": True}, "`not_void` (fail): no rate: its `statements_per_s` is True"),
    ({"statements_per_s": None}, "`not_void` (fail): no rate: its `statements_per_s` is None"),
    ({"flat": None}, "`flat` (fail): its statement cache's hit rate wasn't flat"),
    ({"flat": DROP}, "`flat` (cannot tell): its row lacks `flat`"),
    ({"throttled": {"nr_throttled": 3, "throttled_usec": 250000}},
     "`no_throttling` (fail): its CPUs were throttled 3 times while it ran, for 0.25 s"),
    ({"throttled": DROP}, "`no_throttling` (cannot tell): its row lacks `throttled.nr_throttled`"),
    ({"warm_per_prover": {"prover-gpu0": 0}},
     "`every_prover_warmed` (fail): 0 of its 1 provers finished a warm session before the timed ones"),
    ({"warm_per_prover": DROP}, "`every_prover_warmed` (cannot tell): its row lacks `warm_per_prover`"),
    ({"idle_provers": 1}, "`every_prover_warmed` (fail): 1 of its 1 provers filed no session in the window"),
    ({"idle_provers": DROP}, "`every_prover_warmed` (cannot tell): its row lacks `idle_provers`"),
]
#: the checks of the steady-state rules the reader holds a row to, by the rules' names, and those it doesn't: three read
#: a run's other files, and a labelled row may lack what `one_numa_node` reads
GATES = ("accepted", "exits", "not_void", "flat", "no_throttling", "every_prover_warmed")
NO_GATES = ("one_lease_own_card", "card_clear_start_end", "level", "one_numa_node")


def changed(row: dict, change: dict) -> dict:
    return {k: v for k, v in (row | change).items() if v is not DROP}


def test_a_row_is_judged_by_the_steady_state_rules_own_checks_and_by_no_rule_of_the_readers():
    """The reader's answer on a row is `verity_bench.steady`'s on `GATES` (`accepted`, `exits`, `not_void`, `flat`,
    `no_throttling` and `every_prover_warmed`), check for check and word for word: on the fixture's rows, on rows those
    checks are strict about (`STRICT`: a session never filed, a row that names no exit code, a rate that isn't finite,
    a row that also holds an `error`, a count that isn't one, throttled CPUs, a prover that never warmed or sat idle)
    and on a row several of them fail, which names each. Only a row that passed all six is a measurement, so "cannot
    tell" prices nothing. `one_numa_node` is no condition here: a row that fails it, or can't show it, is still
    priced."""

    assert R.CHECKS == tuple(getattr(S, name) for name in GATES)
    assert S.CHECKS == tuple(sorted(GATES + NO_GATES, key=S.CHECKS.index)) and S.one_numa_node not in R.CHECKS
    good = {row["type"]: row for row in REFUSALS}["good"]
    rows = [row for row in REFUSALS if "stage" in row] + [changed(good, change) for change, _ in STRICT]
    for row in rows:
        got = R.judged(row)
        assert got.checks == tuple(getattr(S, name)(row) for name in GATES)
        assert R.steady_row(row, "r1").measured == (got.passed and R.unpriced(row) is None)
        assert (R.row_why(row) is None) == R.steady_row(row, "r1").measured
    for change, why in STRICT:
        row = changed(good, change)
        read = R.steady_row(row, "r1")
        assert (R.row_why(row), read.refused, read.measured, read.gpu_seconds) == (why, why, False, None), change
        assert R.judged(row).status == (S.CANNOT_TELL if "(cannot tell)" in why else S.FAIL)
    # a row that fails several checks names each, in the rules' order
    killed = good | {"exits": {"verifier-good": 0, "prover-gpu0": -15}, "flat": False, "statements_per_s": None,
                     "void": ["exited nonzero: prover-gpu0 rc -15"]}
    assert R.row_why(killed) == ("`exits` (fail): exited nonzero: prover-gpu0 rc -15; `not_void` (fail): void: exited "
                                 "nonzero: prover-gpu0 rc -15; `flat` (fail): its statement cache's hit rate wasn't flat")
    assert R.row_why(good | {"exits": [0, 1], "refused_n": 2}) == ("`accepted` (fail): 2 sessions refused; `exits` (fail): "
                                                                   "exited nonzero: 1 rc 1")
    # a check the row fails outranks one its fields can't show, as in the rules
    assert R.judged(changed(killed, {"unfiled": DROP})).status == S.FAIL
    # throttled CPUs and a prover that never warmed are gates: a row that fails both names both and prices nothing
    cold = good | {"throttled": {"nr_throttled": 3, "throttled_usec": 250000}, "warm_per_prover": {"prover-gpu0": 0}}
    assert R.row_why(cold) == ("`no_throttling` (fail): its CPUs were throttled 3 times while it ran, for 0.25 s; "
                               "`every_prover_warmed` (fail): 0 of its 1 provers finished a warm session before the "
                               "timed ones")
    assert not R.steady_row(cold, "r1").measured
    # `one_numa_node` is none: a row whose cores span two NUMA nodes, and one that doesn't say, are priced as the good row
    for cores, status in (({"count": 10, "list": "8-17", "numa_nodes": [0, 1]}, S.FAIL), ({"count": 10}, S.CANNOT_TELL)):
        row = good | {"cores": cores}
        assert S.one_numa_node(row).status == status
        assert R.row_why(row) is None and R.steady_row(row, "r1").gpu_seconds == R.steady_row(good, "r1").gpu_seconds


def test_a_row_is_a_steady_state_measurement_only_when_every_session_was_accepted_and_its_cache_flat():
    """`fixtures/prove/refusals`: one row that is a steady-state measurement, and one of each way a row isn't, each
    refused with its reason; a row lacking a field names the field."""

    rows = {row["type"]: row for row in REFUSALS}
    assert {name: R.row_why(rows[name]) for name in ROW_WHYS} == ROW_WHYS
    read = {name: R.steady_row(rows[name], "r20260101-000000-0001") for name in ROW_WHYS}
    assert all(isinstance(s, P.Steady) and s.refused == ROW_WHYS[name] and s.measured == (name == "good")
               for name, s in read.items())
    # a statement's GPU-seconds are the GPUs its provers held over its statements a second, and a block's are that over
    # the statement's blocks
    good = read["good"]
    assert (good.type, good.definition, good.k_log, good.blocks, good.full, good.provers) == ("good", "Good_v1{N=1}", 20, 8, 8, 1)
    assert (good.gpu_seconds, good.verifier_seconds, good.report) == (1 / 0.125, 4.0, "out/rate/rate.json")
    four = {"gpus": 2, "provers": 4, "accepted": 12, "warm_per_prover": {f"prover-gpu{i}": 1 for i in range(4)}}
    assert R.steady_row(rows["good"] | four, "r1").gpu_seconds == 2 / 0.125
    cost = P.Flock((good,)).table(P.Census((P.Block("Good_v1{N=1}", "Good_v1", 1.0, 20),), 1.0, 1.0)).rows[0].cost
    assert (cost.source, 1e3 * cost.seconds, cost.verifier_seconds, cost.provers) == (P.STEADY, 1000.0, 0.5, 1)
    # the verifier's seconds are optional, and a row whose rate isn't one is refused
    assert R.steady_row({k: v for k, v in rows["good"].items() if k != "verify_s"}, "r1").verifier_seconds is None
    assert R.row_why(rows["good"] | {"verify_s": {"mean": 4.0}}) == "its `verify_s` isn't a statement's seconds"
    assert R.row_why(rows["good"] | {"gpus": 0}) == "no rate to price: `statements_per_s` is 0.125 on `gpus` 0"
    assert R.row_why(rows["good"] | {"stage": rows["good"]["stage"] | {"blocks": 16}}).startswith(
        "its `provers` (1), `stage.blocks` (16) and `stage.full` (8) aren't")
    # a row no session of which ran has no prover count to price at
    none = rows["good"] | {"provers": 0, "accepted": 8, "warm_per_prover": {}}
    assert R.row_why(none).startswith("its `provers` (0), `stage.blocks` (8) and")
    # a row that names no staged Definition goes on no block type's row: it is its reason
    assert R.steady_row(rows["failed"], "r1") == "its row `failed` at 2 provers: failed: RuntimeError: the server didn't start"
    assert R.steady_row(rows["late"], "r1") == "its row `late` at 4 provers: skipped: past the deadline"
    assert R.steady_row({"type": "odd", "provers": 1, "stage": {}}, "r1") == "its row `odd` at 1 provers: it lacks `stage.definition`"


def test_a_row_names_the_census_type_it_was_staged_for_not_the_form_staged():
    """A row staged in another form of its census type (the card stages `FlashInferFp8Linear_v1` as `_v2`) prices the
    census's type: `stage.word` names it, and `stage.definition` only where a row has no `stage.word`. The Definition
    staged is the row's `form` where it is another than its census type's, and the type's row names it."""

    good = {row["type"]: row for row in REFUSALS}["good"]
    staged = good | {"stage": good["stage"] | {"definition": "Good_v2{N=1}", "word": "Good_v1{N=1}"}}
    read = R.steady_row(staged, "r1")
    assert (read.definition, read.form, read.proved) == ("Good_v1{N=1}", "Good_v2{N=1}", "Good_v2{N=1}") and read.measured
    cost = P.Flock((read,)).table(P.Census((P.Block("Good_v1{N=1}", "Good_v1", 1.0, 20),), 1.0, 1.0)).rows[0].cost
    assert (cost.source, cost.definition) == (P.STEADY, "Good_v2{N=1}")
    assert cost.note.startswith("`good` (another Definition, `Good_v2{N=1}`), 1 prover, full statements (out/rate/rate.json)")
    # a row that names no census type is its staged Definition's, and one staged in its census type's own Definition
    # has no other form, in the census's spelling of it or the Definition's id's
    bare = R.steady_row(good | {"stage": good["stage"] | {"definition": "Good_v2{N=1}"}}, "r1")
    assert (bare.definition, bare.form, bare.proved) == ("Good_v2{N=1}", None, "Good_v2{N=1}")
    own = R.steady_row(good | {"stage": good["stage"] | {"definition": "Good_v1{N=1,FMA=True}", "word": "Good_v1{N=1,FMA=true}"}}, "r1")
    assert (own.definition, own.form) == ("Good_v1{N=1,FMA=True}", None)
    # a row that isn't a steady-state measurement keeps its form
    assert R.steady_row(staged | {"flat": False}, "r1").form == "Good_v2{N=1}"


def test_a_rows_cores_are_its_count_or_its_mappings_count_and_24_where_it_states_none():
    """A row's `cores` is the CPU cores its provers held for its card: a count, or a mapping whose `count` is one, as a
    row completed after its run states them. A row that states none is read at 24, or at what its run's host states
    where its reader has that; one that states anything else isn't a steady-state measurement, and says so."""

    good = {row["type"]: row for row in REFUSALS}["good"]
    held = {"count": 10, "list": "12-21", "numa_nodes": [0], "of_host": 96}
    assert "cores" not in good and R.steady_row(good, "r1").cores == 24
    assert [R.steady_row(good | {"cores": stated}, "r1").cores for stated in (10, held, 24, held | {"count": 12})] == [10, 10, 24, 12]
    assert [R.row_cores(row) for row in (good, good | {"cores": 10}, good | {"cores": held})] == [24, 10, 10]
    assert (R.steady_row(good, "r1", unstated=10).cores, R.steady_row(good | {"cores": 12}, "r1", unstated=10).cores) == (10, 12)
    assert all(R.steady_row(good | {"cores": stated}, "r1").measured for stated in (10, held))
    for stated in (0, 10.2, "10", True, None, {"list": "12-21"}, held | {"count": 0}):
        read = R.steady_row(good | {"cores": stated}, "r1")
        assert R.row_cores(good | {"cores": stated}) is None and not read.measured
        assert read.refused == f"its `cores` ({stated}) isn't a count of CPU cores, or a mapping whose `count` is one"
    # a row that isn't a steady-state measurement for another reason keeps the cores it states
    assert R.steady_row(good | {"cores": 10, "flat": False}, "r1").cores == 10


# ---- the store side ------------------------------------------------------------------------------------------------------------


class Store:
    """A machine holding a workload's records: a local store with a directory remote, each art and run on it."""

    def __init__(self, tmp: Path):
        from research.store.local import LocalStore

        self.tmp, self.remote = tmp, tmp / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp / "producer", config_path=self.config)
        self._ts = 0

    def fresh(self, name: str = "fresh"):
        from research.store.local import LocalStore

        return LocalStore(self.tmp / name, config_path=self.config)

    def art(self, kind: str, meta: dict, *, file: Path | None = None, tree: Path | None = None) -> str:
        art = self.store.put_artifact(kind, meta, file=file, tree=tree)
        self.store.push(art)
        return art

    def _scratch(self) -> Path:
        self._ts += 1
        path = self.tmp / "docs" / str(self._ts)
        path.mkdir(parents=True)
        return path

    def doc(self, kind: str, name: str, doc: dict, meta: dict | None = None) -> str:
        """A file art holding `doc`."""

        path = self._scratch() / name
        path.write_text(json.dumps(doc))
        return self.art(kind, meta or {}, file=path)

    def tree(self, kind: str, name: str, doc: dict, meta: dict | None = None) -> str:
        """A tree art holding `doc` as its file `name`."""

        root = self._scratch()
        (root / name).write_text(json.dumps(doc))
        return self.art(kind, meta or {}, tree=root)

    def ran(self, run: str, files: Path | None, *, rc: int = 0) -> None:
        """A run's attempt on the remote, its files a run record (`files`: None for a run that kept none)."""

        from research.store import canon

        record = self.art("run-record/v1", {}, tree=files) if files else None
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "bash", "version": None},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T00:00:00Z"}, "cost": None,
            "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": None},
            "outputs": {"result": None, "run_record": record}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))

    def label(self, target: str, key: str, value: str) -> None:
        self._ts += 1
        self.store.add_label(target, key, value, by="benchmarks", ts=f"2026-01-01T00:{self._ts // 60:02d}:{self._ts % 60:02d}Z")
        self.store.push_labels([target])

    def record(self, target: str, slot: str) -> str:
        self.label(target, R.RUN_OF_RECORD, W.record(slot))
        return target


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


RUNS = fixture("records.json")[R.STEADY]


def run_files(tmp: Path, run: str) -> Path:
    """A fixture run's files as its run record holds them: its proof server rate under `out/rate/`."""

    files = tmp / "run-files" / run
    if not files.exists():
        shutil.copytree(FIXTURES / "steady" / run, files / "out")
    return files


def census_art(s: Store, **meta) -> str:
    return s.tree(R.CENSUS_KIND, R.CENSUS_FILE, fixture("census.json"), {"model": W.model, "run": "r20260101-000000-0000"} | meta)


def plan_art(s: Store, **change) -> str:
    return s.tree("evidence/v1", R.PLAN_FILE, fixture("mix.json") | change)


def recorded(tmp: Path, **plan_change) -> tuple[Store, dict[str, str]]:
    """The fixtures' records in a store, each labelled its slot's record: the census, its cut, the single-session card,
    the three proof server runs and the sampling plan (with `plan_change` over its fields)."""

    s = Store(tmp)
    arts = {R.CENSUS: s.record(census_art(s), R.CENSUS),
            R.CUT: s.record(s.doc(R.CARD_SCHEMA, "partition.json", fixture("partition.json")), R.CUT),
            R.SINGLE: s.record(s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json")), R.SINGLE),
            R.SAMPLING: s.record(plan_art(s, **plan_change), R.SAMPLING)}
    for run in RUNS:
        s.ran(run, run_files(tmp, run), rc=0 if run == RUNS[-1] else 1)
        s.record(run, R.STEADY)
    return s, arts


def test_the_hand_computation_is_reproduced_from_the_records_it_read(tmp_path):
    """The figures a hand computation gave on these records, to the last digit it kept: with every block proved, the
    blocks and the prover GPU-seconds a token, the tokens a prover GPU proves a second and each source's share; at the
    sampling plan, the served tokens a prover GPU covers a second and the prover GPUs a serving GPU, with the range's
    other end. A fresh machine gets them from the store's remote alone."""

    s, arts = recorded(tmp_path)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    t = doc["totals"]
    assert doc["cut"] == "per-expert-fused" and len(doc["rows"]) == 150 and doc["refused"] == []
    assert doc["macs_in_blocks"] == pytest.approx(0.98755, abs=5e-6)
    assert t["blocks_per_token"] == pytest.approx(19415356.404181186, rel=1e-12)
    assert t["gpu_seconds_per_token"] == pytest.approx(88303.31936863324, rel=1e-12)
    assert t["tokens_per_gpu_second"] == pytest.approx(1.1324602598746884e-05, rel=1e-12)
    assert t["gpu_seconds_per_token_range"][0] == pytest.approx(27643.734264603525, rel=1e-12)
    assert t["status"] == "modeled" and t["proof_bytes_per_token"] is None
    by = {x["source"]: x for x in doc["sources"]}
    assert [(k, x["block_types"]) for k, x in by.items()] == [(P.STEADY, 8), (P.FAMILY, 59), (P.SINGLE, 27), (P.MODELED, 56), (P.MISSING, 0)]
    assert by[P.STEADY]["per_token"] == pytest.approx(10380036.348432057) and by[P.STEADY]["time_share"] == pytest.approx(
        0.23323941236103127)
    assert by[P.SINGLE]["per_token"] == pytest.approx(6132840.139372822) and by[P.SINGLE]["time_share"] == pytest.approx(
        0.49571996911002464)
    assert by[P.MODELED]["per_token"] == pytest.approx(2902445.3797909454) and by[P.MODELED]["time_share"] == pytest.approx(
        0.2710368026048355)
    assert doc["serving"]["gpu_seconds_per_token"] == pytest.approx(0.00012950804344665488, rel=1e-12)

    # each steady-state type is the newest run's, at the statement it finished; the two killed before any statement are
    # named on the rows the card prices
    rows = {r["block"]: r for r in doc["rows"]}
    steady_rows = [r for r in doc["rows"] if r["source"] == P.STEADY]
    assert {r["record"] for r in steady_rows} == {RUNS[-1]}
    assert sorted(r["note"].split(" (")[0] for r in steady_rows) == [
        "`ar-multimem`", "`dsa-logits-v3`", "`fr-w13`", "`fr-w2`", "`residual-add`", "`rmsnorm-v8`", "`sm-fp8-k2048`",
        "`sm-fp8-k6144`"]
    assert sum("statements 128/256 full" in r["note"] for r in steady_rows) == 2
    top = doc["rows"][0]
    assert top["block"] == "FlashInferFp8Linear_v1{K=2048,N=1}" and top["source"] == P.SINGLE and top["record"] == arts[R.SINGLE]
    assert top["time_share"] == pytest.approx(0.4016, abs=5e-5) and 1e3 * top["seconds"] == pytest.approx(56.611, abs=5e-4)
    assert top["note"] == (f"steady state tried as `fi-linear-k2048` at 1 prover in {RUNS[1]} (out/rate/rate.json): "
                           "`exits` (fail): exited nonzero: prover-gpu0 rc -15; `not_void` (fail): void: exited nonzero: "
                           "prover-gpu0 rc -15; `flat` (fail): its statement cache's hit rate wasn't flat")
    assert {r["provers"] for r in steady_rows} == {1} and top["provers"] is None and doc["provers"] == 1
    # the runs state no cores, in a row or for their host: every row is read as timed at 24, above the default host's
    # 10, so the same rows price the same types and the totals say how much of the proving time rests on them
    assert doc["cores"] == 10 and {r["cores"] for r in steady_rows} == {24} and top["cores"] is None
    assert all(r["note"].endswith("; timed at 24 cores, above the host's 10") for r in steady_rows)
    assert t["above_cores"] == {
        "cores": 10, "block_types": 8 + 59, "timed_at": [24],
        "time_share": pytest.approx(by[P.STEADY]["time_share"] + by[P.FAMILY]["time_share"], rel=1e-12),
        "sources": [{"source": P.STEADY, "block_types": 8, "time_share": by[P.STEADY]["time_share"]},
                    {"source": P.FAMILY, "block_types": 59, "time_share": by[P.FAMILY]["time_share"]}]}
    assert "tried as `gq-k6144`" in rows["Fp8GroupQuantUe8m0_v1{K=6144,G=128}"]["note"]
    assert [i["targets"] for i in doc["inputs"]] == [[arts[R.CENSUS]], [arts[R.CUT]], RUNS[::-1], [arts[R.SINGLE]]]

    spot = R.build("spot", "glm53", sampling=arts[R.SAMPLING], store=s.fresh("second"), root=REPO)
    assert spot == R.build("spot", "glm53", store=s.fresh("third"), root=REPO)          # the plan of record is the default
    t = spot["totals"]
    assert spot["system"] == "spot[flock]" and spot["inputs"][-1]["targets"] == [arts[R.SAMPLING]]
    assert t["tokens_per_gpu_second"] == pytest.approx(14246, abs=0.5)
    assert t["gpus_per_serving_gpu"] == pytest.approx(0.5420038985704952, rel=1e-12)
    assert t["gpus_per_serving_gpu_range"] == pytest.approx([0.1860208609869626, 0.5420038985704952], rel=1e-12)
    assert t["tokens_per_gpu_second_range"][1] == pytest.approx(41509, abs=0.5)
    assert t["gpu_seconds_per_token"] == pytest.approx(7.019e-05, rel=1e-4)
    # the plan covers one compute and period; every block proved is the same ratio at any
    assert [(x["compute"], x["period"]) for x in spot["periods"]["rows"]] == [("world year", "year")]
    assert spot["periods"]["rows"][0]["prover_gpus"] == pytest.approx(0.5420038985704952 * 374960000.0)
    assert [(x["compute"].split()[0], x["period"]) for x in doc["periods"]["rows"]] == [
        (c, p) for c in ("datacenter", "world") for p in ("week", "month", "year")]
    world = doc["periods"]["rows"][-1]
    assert world["serving_gpus"] == pytest.approx(374960000.0, rel=1e-4)
    assert world["prover_gpu_seconds"] == pytest.approx(doc["totals"]["gpus_per_serving_gpu"] * world["serving_gpus"] * 31536000)

    text = R.render(doc, rows=3)
    assert "| `FlashInferFp8Linear_v1{K=2048,N=1}` | 24 | 626,474 | 56.6 | measured in a single session |" in text
    assert "| 147 more block types |" in text and "- proof bytes per token: not measured" in text
    assert "- served MACs inside a proof block: 98.8%; the rest are served rows no cut fits" in text
    assert "- prover GPU-seconds per token: 88,303 (range 27,644 to 88,303)" in text
    assert "the proof server's own verifier" in text and "Refused" not in text
    assert doc["unread"] == [] and "Unread" not in text                                # every labelled row is a block type's
    assert (f"- {R.STEADY}: {', '.join(RUNS[::-1])} (`prove-glm53-flock-steady`): 8 staged types measured at steady state "
            "at 1 prover\n") in text
    assert "- prover count (`--provers`, a stated input): 1;" in text
    assert "- host cores (`--cores`, a stated input): 10 CPU cores a card, node A's share" in text
    assert ("- steady-state rows timed above the host's 10 cores: 23.32% of its proving time rests on them: 67 block types "
            "no row at 10 cores measures, priced from rows timed at 24 cores (8 measured at steady state, 23.32%; 59 "
            "steady state, its family's, 0.00%)\n") in text
    assert "(modeled, a stated input, not a record)" in text and "| measured at steady state | 8 |" in text
    # on a host of 24 cores the rows are its own: the same figures, none resting on a row timed above it; on one of
    # more, no row prices a type
    wide = R.build("flock", "glm53", cores=24, store=s.fresh("wide"), root=REPO)
    assert wide["cores"] == 24 and wide["totals"]["above_cores"]["block_types"] == 0
    assert wide["totals"]["gpu_seconds_per_token"] == doc["totals"]["gpu_seconds_per_token"] and wide["sources"] == doc["sources"]
    assert not any("timed at" in r["note"] for r in wide["rows"])
    over = R.build("flock", "glm53", cores=32, store=s.fresh("over"), root=REPO)
    assert {x["source"]: x["block_types"] for x in over["sources"]}[P.STEADY] == 0 and over["totals"]["range_basis"] is None
    assert sum(r["note"].endswith("measured at steady state at 24 cores, which price no type at 32") for r in over["rows"]) == 8
    with pytest.raises(ValueError, match="--cores 0: a type is priced on a host of one core a card or more"):
        R.build("flock", "glm53", cores=0, store=s.fresh("none"), root=REPO)
    # at two provers only the oldest run's half statements of the six light types measure anything: its two heavy
    # types' row at two provers is void, and every row at one prover prices nothing
    two = R.build("flock", "glm53", provers=2, store=s.fresh("fourth"), root=REPO)
    at_two = [r for r in two["rows"] if r["source"] == P.STEADY]
    assert two["provers"] == 2 and len(at_two) == 6 and {(r["record"], r["provers"]) for r in at_two} == {(RUNS[0], 2)}
    assert all(", 2 provers, statements " in r["note"] and "/" in r["note"] for r in at_two)
    norm = {r["block"]: r for r in two["rows"]}["RMSNormFusedCudaVector8_v1{N=6144,EPS={\"f64\":\"0x1.4f8b588e368f1p-17\"},BLOCK=256}"]
    assert norm["source"] != P.STEADY and "tried as `rmsnorm-v8` at 2 provers" in norm["note"]
    assert norm["note"].endswith("measured at steady state at 1 provers, which price no type at 2")
    assert two["totals"]["range_basis"] is None and "8 staged types" not in R.render(two, rows=0)
    assert "6 staged types measured at steady state at 2 provers" in R.render(two, rows=0)
    with pytest.raises(ValueError, match="--provers 0: a type is priced at one prover or more"):
        R.build("flock", "glm53", provers=0, store=s.fresh("fifth"), root=REPO)

    text = R.render(spot)
    assert "- prover GPUs per serving GPU: 0.542 (range 0.186 to 0.542), modeled" in text
    assert "- served tokens covered per second per prover GPU: 14,246 (range 14,246 to 41,509)" in text
    assert "| world year | year |" in text and "no plan of record covers another period, or a datacenter." in text
    assert text.count("\n| `") == 150


def test_the_plan_of_records_own_prices_are_read_from_its_record(tmp_path):
    """A plan of record whose cells state their planner's seconds: the table's total is the one its sources give, and
    the plan's own is beside it, on every row and in the totals."""

    mix = fixture("mix.json")
    s, _ = recorded(tmp_path, cells=[c | {"prove_s": 0.002 * c["taken"]} for c in mix["cells"]])
    spot = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    t = spot["totals"]
    assert t["gpus_per_serving_gpu"] == pytest.approx(0.5420038985704952, rel=1e-12)
    blocks = mix["replay_sample"] * sum(c["taken"] for c in mix["cells"])
    assert t["planned"]["gpus_per_serving_gpu"] == pytest.approx(0.002 * blocks / mix["gpu_seconds"], rel=1e-9)
    assert sum(x["block_types"] for x in t["planned"]["sources"]) == 150
    assert [r["planned_seconds"] for r in spot["rows"]] == pytest.approx([0.002] * 150)
    text = R.render(spot, rows=2)
    assert "- prover GPUs per serving GPU: 0.542 (range 0.186 to 0.542), modeled" in text
    assert f"- the same blocks at the sampling plan's own prices: {R._n(t['planned']['gpus_per_serving_gpu'])} prover GPUs" in text
    # the plan's column is on every line of the rows' table, the rest's row included
    table = text[text.index("| block type |"):].split("\n\n")[0].splitlines()
    assert table[0].count("|") == 10 and "| ms per block in the plan |" in table[0] and len(table) == 2 + 2 + 1
    assert table[-1].startswith("| 148 more block types | | ") and {line.count("|") for line in table} == {10}


def test_an_unlabelled_or_superseded_input_is_refused_by_name(tmp_path):
    s = Store(tmp_path)
    census = census_art(s)
    cut = s.doc(R.CARD_SCHEMA, "partition.json", fixture("partition.json"))
    single = s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json"))
    mix = plan_art(s)

    def build(system: str = "flock", **kw) -> dict:
        return R.build(system, "glm53", store=s.store, root=REPO, **kw)

    # nothing is labelled: each slot in turn is named with the label that would fix it
    for target, slot, what in ((census, R.CENSUS, "block census"), (cut, R.CUT, "cut"), (single, R.SINGLE, "single-session card")):
        with pytest.raises(ValueError, match=f"no {what} of record: nothing in the store carries "
                                             f"`run_of_record={W.record(slot)}`.*research data label TARGET"):
            build()
        s.record(target, slot)
    doc = build()
    assert doc["inputs"][2] == {"slot": R.STEADY, "record": W.record(R.STEADY), "targets": [],
                                "what": "none of record: no block type is measured at steady state"}
    assert {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 0
    assert doc["totals"]["range_basis"] is None and "(range" not in R.render(doc, rows=1)

    # SPOT's plan: none of record, one given that isn't labelled, and --sampling without spot
    with pytest.raises(ValueError, match=f"no sampling plan of record.*`run_of_record={W.record(R.SAMPLING)}`"):
        build("spot")
    with pytest.raises(ValueError, match=f"{mix} isn't a sampling plan of record: it doesn't carry"):
        build("spot", sampling=mix)
    with pytest.raises(ValueError, match="--sampling is SPOT's"):
        build(sampling=mix)
    s.record(mix, R.SAMPLING)
    # the plan names the two fused block types by their staged types, which only a steady-state row ties to a Definition
    with pytest.raises(ValueError, match=rf"the sampling plan {mix} and the census don't cover each other: the plan has no "
                                         r"cell for 2 block type\(s\) \(MoeExpertFp8ReadGemm_v1.*no block type for 2 "
                                         r"cell\(s\) \(Fused\[.*or no labelled steady-state row names"):
        build("spot")
    s.ran(RUNS[-1], run_files(tmp_path, RUNS[-1]))
    s.record(RUNS[-1], R.STEADY)
    other = s.record(plan_art(s, case="other"), R.SAMPLING)
    with pytest.raises(ValueError, match=f"2 targets carry `run_of_record={W.record(R.SAMPLING)}` and none is superseded: "
                                         f"{mix}, {other}"):
        build("spot")
    assert build("spot", sampling=other)["inputs"][-1]["what"].startswith("`other`")     # naming one of them chooses it
    s.label(mix, R.SUPERSEDED_BY, other)
    assert build("spot")["inputs"][-1]["targets"] == [other]
    with pytest.raises(ValueError, match=f"the sampling plan of record is superseded: {mix} by {other}"):
        build("spot", sampling=mix)

    # a record of the wrong kind, model or slot
    wrong = s.record(census_art(s, model="Llama-3.1-8B"), R.CENSUS)
    s.label(census, R.SUPERSEDED_BY, wrong)
    with pytest.raises(ValueError, match=f"{wrong} isn't a block census of GLM-5.3 .*: it is a block-census/v1 of Llama-3.1-8B"):
        build()
    s.label(wrong, R.SUPERSEDED_BY, census)
    with pytest.raises(ValueError, match=f"the block census of record is superseded: {census} by {wrong}, {wrong} by {census}"):
        build()
    s.record(census_art(s, run="r20260102-000000-0000"), R.CENSUS)
    misfiled = s.record(s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json"), {"filed": "cut"}), R.CUT)
    s.label(cut, R.SUPERSEDED_BY, misfiled)
    with pytest.raises(ValueError, match=f"{misfiled} isn't a `partition` card of GLM-5.3 .*'slot': 'proof_overhead'"):
        build()


def test_a_steady_state_run_that_cant_be_read_is_left_out_and_listed(tmp_path):
    s, _ = recorded(tmp_path)
    newest, older, oldest = RUNS[::-1]
    empty = tmp_path / "empty-run"
    (empty / "out").mkdir(parents=True)
    (empty / "out" / "job.log").write_text("no report")
    s.ran("r20261109-000000-0001", empty)
    s.ran("r20261109-000000-0002", None)
    for run in ("r20261109-000000-0001", "r20261109-000000-0002", "r20261109-000000-0003"):
        s.record(run, R.STEADY)
    s.label(newest, R.SUPERSEDED_BY, "r20261109-000000-0001")
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert {x["target"]: x["why"] for x in doc["refused"]} == {
        "r20261109-000000-0003": "not in the store or on its remote", "r20261109-000000-0002": "kept no run files",
        "r20261109-000000-0001": "holds no proof server rate (`out/rate/rate.json`, which "
                                 "`benchmarks/one_stage/server_rate.py` writes) and no proof server report "
                                 "(`*/server-k*/server.json` with its `types.json`)",
        newest: "superseded by r20261109-000000-0001"}
    assert doc["inputs"][2]["targets"] == [older, oldest]
    # without the newest run each type takes the fullest statement the others measured: the six light types' full
    # statements in the oldest run, not the quarter ones after it, and the two heavy types' half statements
    steady_rows = [r for r in doc["rows"] if r["source"] == P.STEADY]
    assert sorted((r["record"], "full statements" in r["note"]) for r in steady_rows) == [(oldest, True)] * 6 + [(older, False)] * 2
    text = R.render(doc, rows=0)
    assert "Refused (not read):" in text and f"- {newest} (`prove-glm53-flock-steady`): superseded by" in text
    assert "| 150 more block types |" in text

    # a run whose rate.json has rows that name no staged Definition is read, and those rows are listed
    s3 = Store(tmp_path / "third")
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s3.record(s3.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s3.record(census_art(s3), R.CENSUS)
    odd = tmp_path / "odd-run"
    shutil.copytree(FIXTURES / "refusals", odd / "out")
    s3.ran("r20261109-000000-0004", odd)
    s3.record("r20261109-000000-0004", R.STEADY)
    notlist = tmp_path / "notlist-run"
    (notlist / "out" / "rate").mkdir(parents=True)
    (notlist / "out" / "rate" / "rate.json").write_text(json.dumps({"rows": []}))
    s3.ran("r20261109-000000-0005", notlist)
    s3.record("r20261109-000000-0005", R.STEADY)
    doc = R.build("flock", "glm53", store=s3.fresh(), root=REPO)
    assert doc["inputs"][2]["targets"] == ["r20261109-000000-0004"]
    assert [(x["target"], x["why"]) for x in doc["refused"]] == [
        ("r20261109-000000-0005", "its `out/rate/rate.json` isn't a list of rows"),
        ("r20261109-000000-0004", "its row `failed` at 2 provers: failed: RuntimeError: the server didn't start"),
        ("r20261109-000000-0004", "its row `late` at 4 provers: skipped: past the deadline")]
    # its rows are no block type of this census, so they price nothing, and each is listed as unread
    assert {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 0
    assert [(x["record"], x["type"], x["why"]) for x in doc["unread"]] == [
        ("r20261109-000000-0004", name, why) for name, why in ROW_WHYS.items()]
    assert doc["inputs"][2]["what"] == ("0 staged types measured at steady state at 1 prover; 8 row(s) of no block type of "
                                        "the census, listed under Unread")
    assert [x["seconds"] for x in doc["unread"]] == [1.0, *[None] * 7]
    text = R.render(doc, rows=0)
    assert text.count("\n- r20261109-000000-0004: `") == 8 and text.index("Refused (not read):") < text.index("Unread (")
    assert "- r20261109-000000-0004: `good` at 1 prover, `Good_v1{N=1}`: measured at steady state, 1,000 ms per block\n" in text
    assert ("- r20261109-000000-0004: `bare` at 1 prover, `Bare_v1{N=1}`: not a steady-state measurement: `flat` (cannot "
            "tell): its row lacks `flat`; its row lacks `gpus`, `stage.blocks`, `stage.full`\n") in text

    # an art that names its runs stands for them
    s2 = Store(tmp_path / "second")
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s2.record(s2.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s2.record(census_art(s2), R.CENSUS)
    s2.ran(newest, run_files(tmp_path, newest))
    s2.record(s2.doc("evidence/v1", "steady.json", {}, {"runs": [newest]}), R.STEADY)
    doc = R.build("flock", "glm53", store=s2.fresh(), root=REPO)
    assert doc["inputs"][2]["targets"] == [newest] and doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.31936863324)


def test_an_art_that_holds_rate_json_gives_its_runs_rows_in_their_place(tmp_path):
    """A run's rows completed after the run are an art's `rate.json`, with the run in its `runs`: labelled, the art
    prices the types the run's own rows can't, each row naming the art. An art that names no run or several, one whose
    file isn't rows and a superseded one are left out and listed."""

    newest, older = RUNS[-1], RUNS[1]
    whole = json.loads((FIXTURES / "steady" / newest / "rate" / "rate.json").read_text())
    s = Store(tmp_path)
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s.record(s.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s.record(census_art(s), R.CENSUS)
    # the run's own rows lack `flat`, as the script's did before it wrote the field
    bare = tmp_path / "bare-run"
    (bare / "out" / "rate").mkdir(parents=True)
    (bare / "out" / "rate" / "rate.json").write_text(json.dumps([{k: v for k, v in row.items() if k != "flat"} for row in whole]))
    s.ran(newest, bare)
    s.record(newest, R.STEADY)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 0
    assert sum("its row lacks `flat`" in r["note"] for r in doc["rows"]) == 8

    art = s.record(s.tree("evidence/v1", R.ART_RATES, whole, {"runs": [newest]}), R.STEADY)
    doc = R.build("flock", "glm53", store=s.fresh("second"), root=REPO)
    assert doc["inputs"][2]["targets"] == [newest, art] and doc["refused"] == []
    assert doc["inputs"][2]["what"] == "8 staged types measured at steady state at 1 prover"
    assert doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.31936863324, rel=1e-12)
    steady_rows = [r for r in doc["rows"] if r["source"] == P.STEADY]
    assert len(steady_rows) == 8 and {r["record"] for r in steady_rows} == {art}
    # the fixture's rows state no `cores`, so each is read as timed at 24 and says so on a host of 10
    assert all(r["note"].endswith(f"(rate.json of {art[:16]}, measured in {newest}); timed at 24 cores, above the host's 10")
               for r in steady_rows)
    assert f"| {art[:16]} |" in R.render(doc, rows=150)

    # the art alone, its run in no store: the rows are the art's, and the run isn't read. A file art holds them as well
    s2 = Store(tmp_path / "second-store")
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s2.record(s2.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s2.record(census_art(s2), R.CENSUS)
    alone = s2.record(s2.doc("evidence/v1", R.ART_RATES, whole, {"runs": {"only": newest}}), R.STEADY)
    doc = R.build("flock", "glm53", store=s2.fresh(), root=REPO)
    assert doc["inputs"][2]["targets"] == [alone] and doc["refused"] == []
    assert doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.31936863324, rel=1e-12)

    # what can't be read is listed: no run named, two named, a file that isn't rows; a superseded art gives nothing,
    # whether it holds rows or names runs
    unnamed = s2.record(s2.tree("evidence/v1", R.ART_RATES, whole), R.STEADY)
    two = s2.record(s2.tree("evidence/v1", R.ART_RATES, whole, {"runs": [older, newest]}), R.STEADY)
    notlist = s2.record(s2.tree("evidence/v1", R.ART_RATES, {"rows": whole}, {"runs": [newest]}), R.STEADY)
    s2.ran(older, run_files(tmp_path, older))
    stands = s2.record(s2.doc("evidence/v1", "steady.json", {}, {"runs": [older]}), R.STEADY)
    for old in (alone, stands):
        s2.label(old, R.SUPERSEDED_BY, notlist)
    doc = R.build("flock", "glm53", store=s2.fresh("second"), root=REPO)
    assert {x["target"]: x["why"] for x in doc["refused"]} == {
        unnamed: "its meta names no `runs`: the run its `rate.json` is of, or the runs it stands for",
        two: f"holds `rate.json` and names 2 runs ({older}, {newest}): an art's rows are one run's, so each run's go in "
             "an art of their own",
        notlist: "its `rate.json` isn't a list of rows",
        alone: f"superseded by {notlist}", stands: f"superseded by {notlist}"}
    assert doc["inputs"][2]["targets"] == [] and {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 0


HEAD_BLOCK = ('SparseMlaHeadBlock_v1{DR=64,DV=512,NVIS=4,FIRST=true,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
              'SCALE={"f64":"0x1.0000000000000p-4"}}')


def registered(block: str) -> str:
    """The id of the Definition the tree registers as a census type's Boolean version, asked of the registry itself."""

    from verity.definitions import boolean
    from verity.definitions.call_scope import spec_base, spec_statics, specialization

    return boolean.boolean_version(specialization(spec_base(block), spec_statics(block))).id


def test_a_definition_is_a_types_boolean_version_only_where_the_trees_registry_says_so():
    """`boolean_version` asks the tree's registry: a census type's Boolean version is the one Definition
    `verity.definitions.boolean.boolean_version` gives for the type's own, at the statics it binds, a function-valued
    static's Boolean version among them. The type's own Definition, its Boolean Definition at other statics, a
    version the tree doesn't hold, and anything of a type the registry doesn't bind are none."""

    linear = "FlashInferFp8Linear_v1{K=2048,N=1}"
    bits = registered(linear)
    assert bits.startswith("FlashInferFp8Linear_v") and bits.endswith("{K=2048,N=1}") and bits != P.definition_id(linear)
    assert R.boolean_version(linear, bits)
    assert not any(R.boolean_version(linear, other) for other in (
        P.definition_id(linear), registered("FlashInferFp8Linear_v1{K=1536,N=1}"), "FlashInferFp8Linear_v0{K=2048,N=1}"))
    head = registered(HEAD_BLOCK)
    assert head.startswith("SparseMlaHeadBlock_v") and R.boolean_version(HEAD_BLOCK, head)
    assert not R.boolean_version(HEAD_BLOCK, P.definition_id(HEAD_BLOCK))
    # a name the registry doesn't hold, and statics it can't bind
    assert not R.boolean_version("Made_up_v1{N=1}", "Made_up_v2{N=1}")
    assert not R.boolean_version("FlashInferFp8Linear_v1{BOGUS=1}", bits)


def test_the_table_names_each_priced_rows_definition_and_its_form_from_the_trees_registry(tmp_path):
    """On a store: a made-up row staged for a census type in the Definition the tree registers as its Boolean version
    says so on the type's row, one staged in any other Definition is another Definition's, and the fixture's rows,
    each of its type's own Definition, name it with no qualifier. The JSON row holds the Definition, the rendered
    table the line, and the same rows staged in their types' own Definitions give every total the same."""

    newest = RUNS[-1]
    whole = json.loads((FIXTURES / "steady" / newest / "rate" / "rate.json").read_text())
    good = next(row for row in whole if row["type"] == "residual-add")
    linear, wide = "FlashInferFp8Linear_v1{K=2048,N=1}", "FlashInferFp8Linear_v1{K=6144,N=1}"

    def staged(type_: str, word: str, definition: str, k_log: int) -> dict:
        return good | {"type": type_, "stage": good["stage"] | {"word": word, "definition": definition, "k_log": k_log}}

    def built(name: str, rows: list[dict]) -> dict:
        s = Store(tmp_path / name)
        for slot, card in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
            s.record(s.doc(R.CARD_SCHEMA, card, fixture(card)), slot)
        s.record(census_art(s), R.CENSUS)
        s.record(s.tree("evidence/v1", R.ART_RATES, [*whole, *rows], {"runs": [newest]}), R.STEADY)
        return R.build("flock", "glm53", store=s.fresh(), root=REPO)

    bits = registered(linear)
    doc = built("forms", [staged("fi-linear-bits", linear, bits, 24),
                          staged("fi-linear-n2", wide, "FlashInferFp8Linear_v1{K=6144,N=2}", 26)])
    rows = {r["block"]: r for r in doc["rows"]}
    assert bits != linear and (rows[linear]["source"], rows[linear]["definition"]) == (P.STEADY, bits)
    assert rows[linear]["note"].startswith(
        f"`fi-linear-bits` (its Boolean version, `{bits}`), 1 prover, full statements (rate.json of ")
    assert (rows[wide]["source"], rows[wide]["definition"]) == (P.STEADY, "FlashInferFp8Linear_v1{K=6144,N=2}")
    assert rows[wide]["note"].startswith(
        "`fi-linear-n2` (another Definition, `FlashInferFp8Linear_v1{K=6144,N=2}`), 1 prover, full statements (rate.json of ")
    assert rows["ResidualAdd_v1{N=6144}"]["definition"] == "ResidualAdd_v1{N=6144}"
    assert rows["ResidualAdd_v1{N=6144}"]["note"].startswith(
        "`residual-add` (`ResidualAdd_v1{N=6144}`), 1 prover, full statements (rate.json of ")
    # every steady-state and family's row names a Definition, and no other row does
    assert all((r["definition"] is not None) == (r["source"] in (P.STEADY, P.FAMILY)) for r in doc["rows"])
    assert {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 10
    family = next(r for r in doc["rows"] if r["source"] == P.FAMILY)
    assert f"(`{family['definition']}`), 1 prover, full statements, at 2^" in family["note"]
    text = R.render(doc)
    assert f"| `fi-linear-bits` (its Boolean version, `{bits}`), 1 prover, full statements " in text
    assert "| `fi-linear-n2` (another Definition, `FlashInferFp8Linear_v1{K=6144,N=2}`), 1 prover, full statements " in text

    # the same two rows staged in their types' own Definitions: every figure is the same, and no row has a qualifier
    own = built("own", [staged("fi-linear-bits", linear, linear, 24), staged("fi-linear-n2", wide, wide, 26)])
    assert (own["totals"], own["sources"], own["periods"]) == (doc["totals"], doc["sources"], doc["periods"])

    def figures(d: dict) -> list[dict]:
        return [{k: v for k, v in r.items() if k not in ("note", "definition", "record")} for r in d["rows"]]

    assert figures(own) == figures(doc)
    assert {r["block"]: r["definition"] for r in own["rows"] if r["block"] in (linear, wide)} == {linear: linear, wide: wide}
    assert not any("Boolean version" in r["note"] or "another Definition" in r["note"] for r in own["rows"])


def test_a_rows_cores_are_its_own_else_its_runs_host_else_24(tmp_path):
    """Where a row's cores are read from, on a store: a run's row that states none takes the cores its run's
    `out/host.json` states (`cpus_used`), and is read at 24 where the run holds no host file or the file states none;
    an art's row states its own. Each host's cores then choose between a run's rows and an art's."""

    newest = RUNS[-1]
    whole = json.loads((FIXTURES / "steady" / newest / "rate" / "rate.json").read_text())

    def store(name: str, host: dict | None) -> Store:
        s = Store(tmp_path / name)
        for slot, card in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
            s.record(s.doc(R.CARD_SCHEMA, card, fixture(card)), slot)
        s.record(census_art(s), R.CENSUS)
        files = tmp_path / name / "run-files"
        shutil.copytree(FIXTURES / "steady" / newest, files / "out")
        if host is not None:
            (files / "out" / "host.json").write_text(json.dumps(host))
        s.ran(newest, files)
        s.record(newest, R.STEADY)
        return s

    def steady_rows(s: Store, name: str, **at: int) -> tuple[dict, list[dict]]:
        doc = R.build("flock", "glm53", store=s.fresh(name), root=REPO, **at)
        return doc, [r for r in doc["rows"] if r["source"] == P.STEADY]

    # the run's host held 10 cores for its card: its rows are the default host's own, and none is timed above it
    s = store("hosted", {"cpus_host": 96, "cpus_used": 10, "cpus_list": "12-21"})
    doc, rows = steady_rows(s, "ten")
    assert len(rows) == 8 and {r["cores"] for r in rows} == {10} and not any("timed at" in r["note"] for r in doc["rows"])
    assert doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.31936863324, rel=1e-12)
    assert doc["totals"]["above_cores"] == {"cores": 10, "block_types": 0, "time_share": 0.0, "timed_at": [], "sources": []}
    # and on a host of 24 cores they are under its share: they price nothing, and each type's row says what measured it
    doc, rows = steady_rows(s, "wide", cores=24)
    assert rows == [] and sum(r["note"].endswith("measured at steady state at 10 cores, which price no type at 24")
                              for r in doc["rows"]) == 8

    # an art holding the run's rows, each stating its own cores as a mapping: at 24 they are another measurement of
    # the same types, which prices them on a host of 24 cores and leaves the run's own to price them on one of 10
    held = {"count": 24, "list": "24-47", "numa_nodes": [1], "of_host": 96}
    art = s.record(s.tree("evidence/v1", R.ART_RATES, [row | {"cores": held} for row in whole], {"runs": [newest]}), R.STEADY)
    doc, rows = steady_rows(s, "both")
    assert {(r["record"], r["cores"]) for r in rows} == {(newest, 10)} and doc["totals"]["above_cores"]["block_types"] == 0
    doc, rows = steady_rows(s, "both-wide", cores=24)
    assert len(rows) == 8 and {(r["record"], r["cores"]) for r in rows} == {(art, 24)}
    assert doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.31936863324, rel=1e-12)
    assert doc["totals"]["above_cores"]["block_types"] == 0 and not any("which price no type" in r["note"] for r in rows)

    # a run whose host file states no cores, or isn't a mapping, is read as one that holds none: at 24
    for name, host in (("unstated", {"hostname": "h"}), ("uncounted", {"cpus_used": "10"}), ("listed", [10])):
        doc, rows = steady_rows(store(name, host), "reader")
        assert len(rows) == 8 and {r["cores"] for r in rows} == {24}
        assert doc["totals"]["above_cores"]["block_types"] == 8 + 59 and doc["totals"]["above_cores"]["timed_at"] == [24]


def test_a_run_from_before_rate_json_is_read_from_its_proof_server_reports(tmp_path):
    """The three runs as they are in the store, their proof server reports (`server-reports/RUN/`) and no `rate.json`: read from
    the reports, they give the table their `rate.json` rows give, each `server-kK` report's types at K provers. The rows
    name the report they come from."""

    s = Store(tmp_path)
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s.record(s.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s.record(census_art(s), R.CENSUS)
    for run in RUNS:
        files = tmp_path / "report-files" / run
        shutil.copytree(FIXTURES / "server-reports" / run, files / "out")
        s.ran(run, files, rc=0 if run == RUNS[-1] else 1)
        s.record(run, R.STEADY)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert doc["refused"] == [] and doc["inputs"][2]["what"] == "8 staged types measured at steady state at 1 prover"
    assert doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.31936863324, rel=1e-9)
    steady_rows = [r for r in doc["rows"] if r["source"] == P.STEADY]
    assert {r["record"] for r in steady_rows} == {RUNS[-1]} and {r["provers"] for r in steady_rows} == {1}
    assert sum("out/light-va2/server-k1" in r["note"] for r in steady_rows) == 6
    assert {r["cores"] for r in steady_rows} == {24}                                   # a report states no cores

    two = R.build("flock", "glm53", provers=2, store=s.fresh("second"), root=REPO)
    at_two = [r for r in two["rows"] if r["source"] == P.STEADY]
    assert len(at_two) == 6 and {(r["record"], r["provers"]) for r in at_two} == {(RUNS[0], 2)}
    assert all("out/light/server-k2" in r["note"] for r in at_two)


RMSNORM = next(name for name in fixture("census.json")["definitions"] if name.startswith("RMSNormFusedCudaVector8_v1{N=6144,"))
LEAN_HOST = {"node": "cpu-node-0", "cores": 64, "host_cores": 128, "cpu_model": "A CPU 9000", "mem_gib": 512.0, "commit": "c" * 40}


def lean_record(run: str, *, word: str | None = RMSNORM, seconds: tuple[float, ...] = (600.0, 500.0, 520.0),
                size: int = 2_000_000, why: str | None = None, **block) -> dict:
    """A made-up record of `verity bench prove lean-verify`: a full statement of 256 blocks of 2^27 witness bits, each
    run accepted, or its last refused with `why`."""

    runs = [{"wall_s": x, "rc": 0, "accepted": True, "why": None} for x in seconds]
    if why:
        runs[-1] |= {"rc": 1, "accepted": False, "why": why}
    return {"schema": L.SCHEMA, "run": run,
            "block": {"word": word, "definition": "Staged_v2{N=1}", "k_log": 27, "blocks": 256, "full": 256} | block,
            "proof": {"bytes": size}, "verifier": {"repeat": len(seconds), "runs": runs}, "accepted": not why,
            "label": None if why else L.COVER_LABEL, "host": dict(LEAN_HOST)}


def lean_ran(s: Store, tmp: Path, run: str, doc: dict | None, *, name: str = L.RECORD) -> str:
    """`run` in the store with `doc` as its `out/lean-verify.json` (None: a run that wrote none), a Lean verifier run
    of record."""

    files = tmp / "lean-files" / run / "out"
    files.mkdir(parents=True)
    (files / "job.log").write_text("ran")
    if doc is not None:
        (files / name).write_text(json.dumps(doc))
    s.ran(run, files.parent)
    return s.record(run, R.LEAN)


def test_a_lean_verifier_run_of_record_gives_both_figures_as_modeled_and_says_what_was_measured(tmp_path):
    """One block type's statement measured, on made-up figures: its row takes them, every other row with a k_log is
    scaled a witness bit, and both lines say modeled, the measured type and its share of the proving time, the
    statement, the run, the host and the basis. The proof server's verifier line stays."""

    s, arts = recorded(tmp_path)
    run = lean_ran(s, tmp_path, "r20261110-000000-0001", lean_record("r20261110-000000-0001"))
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    rows = {r["block"]: r for r in doc["rows"]}
    own, t = rows[RMSNORM], doc["totals"]
    assert (own["lean_verifier_seconds"], own["proof_bytes"], own["lean_source"]) == (520.0 / 256, 2_000_000 / 256, "measured")
    assert {r["lean_source"] for r in doc["rows"] if r is not own} == {"scaled"}
    assert all(r["lean_verifier_seconds"] == pytest.approx(520.0 * 2 ** r["k_log"] / 2**35) for r in doc["rows"])
    seconds, size = t["lean_verifier_seconds_per_token"], t["proof_bytes_per_token"]
    assert seconds["value"] == pytest.approx(sum(r["per_token"] * 520.0 * 2 ** r["k_log"] / 2**35 for r in doc["rows"]))
    assert size["value"] == pytest.approx(sum(r["per_token"] * 2_000_000 * 2 ** r["k_log"] / 2**35 for r in doc["rows"]))
    assert (seconds["status"], size["status"]) == ("modeled", "modeled")
    assert (seconds["measured_block_types"], seconds["scaled_block_types"], seconds["unpriced_block_types"]) == (1, 149, 0)
    assert seconds["measured_share"] == pytest.approx(own["per_token"] * 520.0 / 256 / seconds["value"])
    assert t["lean_verify"] == [{"block": RMSNORM, "time_share": own["time_share"], "k_log": 27, "blocks": 256, "full": 256,
                                 "seconds": [600.0, 500.0, 520.0], "statement_seconds": 520.0, "proof_bytes": 2_000_000,
                                 "run": run, "host": "cpu-node-0 (64 cores, A CPU 9000)", "label": L.COVER_LABEL}]
    assert doc["inputs"][-1] == {"slot": R.LEAN, "record": "prove-glm53-lean-verify", "targets": [run],
                                 "what": "the Lean verifier (its `verify`) on one statement of each of 1 block "
                                         "type(s); every other type's Lean verifier seconds and proof bytes are scaled "
                                         "from them"}
    assert doc["refused"] == [] and t["verifier_seconds_per_token"]["block_types"] == 8

    text = R.render(doc)
    lines = {line.split(" per token:")[0]: line for line in text.splitlines() if " per token:" in line}
    assert lines["- verifier seconds"].startswith("- verifier seconds per token: ") and "the proof server's own verifier" in lines["- verifier seconds"]
    share = f"{own['time_share']:.1%}"
    assert lines["- Lean verifier seconds"] == (
        f"- Lean verifier seconds per token: {R._n(seconds['value'])}, modeled, not measured. Measured: one block type, "
        f"`{RMSNORM}` ({share} of the proving time), in its statement of 256 blocks of 2^27 witness bits each (a full "
        "statement): accepted on the provisional unit cover, units as stated, the Program not held to the circuit, by the "
        "Lean verifier's `verify` in 520 s (the median of 3 runs: 600, 500, 520 s; a run is one "
        f"process, the statement read and set up in it), in run {run} on cpu-node-0 (64 cores, A CPU 9000). Scaled: the "
        f"other 149 block types, {1 - seconds['measured_share']:.1%} of this figure, each at the measured statement's "
        f"seconds a witness bit times its 2^k_log bits: {P.LEAN_SECONDS_BASIS}.")
    assert lines["- proof bytes"] == (
        f"- proof bytes per token: {R._n(size['value'])}, modeled, not measured. Measured: one block type, `{RMSNORM}` "
        f"({share} of the proving time), in its statement of 256 blocks of 2^27 witness bits each (a full statement), "
        f"whose proof files are 2.000e+06 bytes in all (run {run}). Scaled: the other 149 block types, "
        f"{1 - size['measured_share']:.1%} of this figure, each at the measured statement's bytes a witness bit times "
        f"its 2^k_log bits: {P.PROOF_BYTES_BASIS}.")
    assert "per token: not measured" not in text

    # at the sampling plan: the same record at the blocks the plan samples, the type's share the plan's
    spot = R.build("spot", "glm53", store=s.fresh("second"), root=REPO)
    rows = {r["block"]: r for r in spot["rows"]}
    t = spot["totals"]
    assert t["lean_verifier_seconds_per_token"]["value"] == pytest.approx(
        sum(r["per_token"] * 520.0 * 2 ** r["k_log"] / 2**35 for r in spot["rows"]))
    assert t["lean_verifier_seconds_per_token"]["status"] == "modeled" and t["proof_bytes_per_token"]["status"] == "modeled"
    assert t["lean_verify"][0]["time_share"] == rows[RMSNORM]["time_share"] != own["time_share"]
    assert [i["slot"] for i in spot["inputs"]][-2:] == [R.LEAN, R.SAMPLING]
    assert f"`{RMSNORM}` ({rows[RMSNORM]['time_share']:.1%} of the proving time at the sampling plan), in its statement" in R.render(spot)

    # a newer run of the same type replaces it; a statement that isn't full says so
    newer = lean_ran(s, tmp_path, "r20261110-000000-0002", lean_record("r20261110-000000-0002", seconds=(300.0,), blocks=128))
    doc = R.build("flock", "glm53", store=s.fresh("third"), root=REPO)
    assert [x["run"] for x in doc["totals"]["lean_verify"]] == [newer] and doc["inputs"][-1]["targets"] == [newer, run]
    assert doc["totals"]["lean_verifier_seconds_per_token"]["value"] == pytest.approx(2 * 300.0 / 520.0 * seconds["value"])
    assert (f"in its statement of 128 blocks of 2^27 witness bits each (128/256 of a full statement): {L.COVER_LABEL}, by "
            "the Lean verifier's `verify` in 300 s (one run; a run is one process") in R.render(doc)


def test_a_lean_verifier_run_that_measures_nothing_is_refused_by_name_and_both_figures_stay_not_measured(tmp_path):
    """A refusal by the Lean verifier is its run's result and prices nothing: with it, a run that wrote no record, one
    not in the store, a superseded one, another schema's, one that stopped short and one of no block type of the
    census, both figures are not measured and each run is listed with why."""

    s, _ = recorded(tmp_path)
    runs = [f"r20261110-000000-000{i}" for i in range(1, 9)]
    lean_ran(s, tmp_path, runs[0], lean_record(runs[0], why="lincheck: the folded claim doesn't match"))
    lean_ran(s, tmp_path, runs[1], None)
    s.record(runs[2], R.LEAN)
    lean_ran(s, tmp_path, runs[3], lean_record(runs[3]))
    s.label(runs[3], R.SUPERSEDED_BY, runs[0])
    lean_ran(s, tmp_path, runs[4], lean_record(runs[4]) | {"schema": "verity-bench/lean-verify/v0"})
    lean_ran(s, tmp_path, runs[5], lean_record(runs[5]) | {"accepted": False, "verifier": {"repeat": 3, "runs": [
        {"wall_s": 1.0, "rc": 0, "accepted": True, "why": None}]}})
    lean_ran(s, tmp_path, runs[6], lean_record(runs[6], word=None, definition="Stray_v1{N=1}"))
    record = lean_record(runs[7])
    del record["host"]["cpu_model"], record["proof"]["bytes"]
    lean_ran(s, tmp_path, runs[7], record)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert {x["target"]: x["why"] for x in doc["refused"]} == {
        runs[0]: "the Lean verifier refused its statement: lincheck: the folded claim doesn't match",
        runs[1]: "holds no `out/lean-verify.json`, which `verity bench prove lean-verify` writes",
        runs[2]: "not in the store or on its remote",
        runs[3]: f"superseded by {runs[0]}",
        runs[4]: "its `out/lean-verify.json` isn't a `verity-bench/lean-verify/v1` record",
        runs[5]: "the Lean verifier ran 1 of the 3 runs it set out to",
        runs[6]: "its block type, `Stray_v1{N=1}`, is no block type of the census under its cut",
        runs[7]: "its record lacks `proof.bytes`, `host.cpu_model`"}
    assert {x["record"] for x in doc["refused"]} == {"prove-glm53-lean-verify"}
    t = doc["totals"]
    assert t["lean_verifier_seconds_per_token"] is None and t["proof_bytes_per_token"] is None and t["lean_verify"] == []
    assert doc["inputs"][-1]["targets"] == [] and "not measured" in doc["inputs"][-1]["what"]
    text = R.render(doc)
    assert ("- Lean verifier seconds per token: not measured (no Lean verifier run of record measures a block type: label one "
            "`research data label TARGET run_of_record prove-glm53-lean-verify --by LANE`)\n- proof bytes per token: not "
            "measured\n") in text
    assert f"{runs[0]}" in text.split("Refused (not read):")[1] and "the Lean verifier refused its statement" in text

    bad = lean_record(runs[0], k_log=0)
    assert R.lean_verify(bad, runs[0]).startswith("its `block` ({'k_log': 0, 'blocks': 256, 'full': 256}), `proof.bytes`")
    assert R.lean_verify(lean_record(runs[0], seconds=(0.0,)), runs[0]).endswith("aren't a statement's blocks, bytes and seconds")
    assert R.lean_verify(lean_record(runs[0], word=None, definition=None), runs[0]) == (
        "its record names no block type (`block.word`, `block.definition`)")
    # an accept the record doesn't qualify is printed by no line
    assert R.lean_verify({k: v for k, v in lean_record(runs[0]).items() if k != "label"}, runs[0]) == "its record lacks `label`"
    assert R.lean_verify(lean_record(runs[0]) | {"label": None}, runs[0]) == "its `label` (None) doesn't say what its accept holds"
    assert R.lean_verify([], runs[0]) == "its `out/lean-verify.json` isn't a `verity-bench/lean-verify/v1` record"


# ---- the Lean verifier's runner ------------------------------------------------------------------------------------------------


class Verifier:
    """The Lean verifier as the runner starts it, with no process: `archive-put` names each file by a key, and each
    `verify` takes the next verdict (None: it prints none), a made-up clock advancing by the seconds it took."""

    def __init__(self, monkeypatch, verdicts: list[tuple[dict | None, float]], *, put_rc: int = 0):
        self.verdicts, self.put_rc, self.now, self.calls = list(verdicts), put_rc, 0.0, []
        monkeypatch.setattr(L, "spawn", self.spawn)
        monkeypatch.setattr(L, "clock", lambda: self.now)
        monkeypatch.setattr(L, "host", lambda: LEAN_HOST)

    def spawn(self, cmd, log: Path) -> tuple[int, str, dict]:
        self.calls.append(list(cmd))
        if cmd[1] == "archive-put":
            assert Path(cmd[2]).is_dir()
            self.now += 0.25
            return self.put_rc, "".join(f"key-{Path(p).name} {p}\n" for p in cmd[3:]), {}
        verdict, took = self.verdicts.pop(0)
        self.now += took
        session = cmd[cmd.index("--session") + 1]
        log.write_text(f"inputs: 40 ms\nsetup: 900 ms\n{session}: 2000 ms\n")
        accepted = bool(verdict and verdict["accepted"])
        return (0 if accepted else 1, "" if verdict is None else "VERDICT " + json.dumps(verdict) + "\n",
                {"cpu_s": took * 2, "user_s": took, "maxrss_gib": 1.5})


#: a made-up program descriptor of the census type `RMSNORM`, whose `annotations` are in no digest, and a made-up
#: partition object
PROGRAM = {"schema": "made-up", "program_fn": RMSNORM, "definitions": {"Staged_v1{N=1}": [1, 2]}, "annotations": {"theta": [3]}}
PARTITION = {"format": "made-up", "program": "p", "query": {"name": "Q_made_up", "version": 1, "params": {"X": 16}}}
#: the program's file, spelled as no canonical form is
PROGRAM_FILE = json.dumps(PROGRAM, indent=1).encode()


def _sha512(prefix: bytes, obj: object) -> str:
    return hashlib.sha512(prefix + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


#: the program's digest as the verifier computes it, over the descriptor without `annotations`, and what a unit
#: statement's circuit states as `program_sha512`: the digest of its `program_digests`, which no program file has
PROGRAM_SHA512 = _sha512(b"", {k: v for k, v in PROGRAM.items() if k != "annotations"})
UNIT_SHA512 = _sha512(b"", {"unit/made-up": hashlib.sha256(b"the staged Definition's id and the row layout").hexdigest()})


def circuit_text(rule: str = L.PARTITION_V1, size: int = 4000) -> bytes:
    """A circuit file of `size` bytes whose second line states a unit's `program_sha512` and binds, under `rule`,
    `PARTITION`."""

    meta = {"k_log": 27, "program_sha512": UNIT_SHA512,
            "partition": {"rule": rule, "digest": _sha512(rule.encode() + b"\0", PARTITION)}}
    head = b"flock-circuit unit/made-up\nMETA " + json.dumps(meta).encode() + b"\n"
    return head + b"x" * (size - len(head))


def kept_proof(tmp: Path, *, stage: dict | None = None, rule: str = L.PARTITION_COVER) -> Path:
    """A kept proof's tree: a staged statement's files, its circuit under `rule`, and one session's two proof files."""

    tree = tmp / "proof"
    (tree / "stage").mkdir(parents=True)
    (tree / "session").mkdir()
    for name, data in (("circuit.txt", circuit_text(rule)), ("pub-256.bin", b"x" * 64), ("program-word.json", PROGRAM_FILE)):
        (tree / "stage" / name).write_bytes(data)
    if stage is not None:
        (tree / "stage" / "stage.json").write_text(json.dumps(stage))
    (tree / "session" / "session.json").write_text("{}")
    (tree / "session" / "circuit.rep0.bin").write_bytes(b"p" * 300)
    (tree / "session" / "circuit.rep1.bin").write_bytes(b"p" * 200)
    return tree


STAGE = {"word": RMSNORM, "definition": "Staged_v2{N=1}", "k_log": 27, "n": 256}
#: the path the runner is given for the verifier: the fake starts no process, so no file is there
VERIFIER = "bin/lean-verifier"
ACCEPTED = {"accepted": True}


def test_the_lean_verifiers_runner_records_each_run_of_the_verifier_on_a_kept_proof(tmp_path, monkeypatch, capsys):
    """The runner on a kept proof, the verifier a fake: the statement's files go to the verifier's archive, each run
    is one `verify` on the session with its own seconds, and the record names the block type, the proof's bytes, the
    runs and the host. `prove` reads the record it wrote."""

    tree, out = kept_proof(tmp_path, stage=STAGE), tmp_path / "out"
    verifier = Verifier(monkeypatch, [(ACCEPTED, 12.5), (ACCEPTED, 10.0), (ACCEPTED, 11.0)])
    assert main(["lean-verify", "--lean", VERIFIER, "--proof", str(tree), "--out", str(out)]) == 0
    doc = json.loads((out / L.RECORD).read_text())
    put, *verifies = verifier.calls
    archive = put[2]
    assert put[:2] == [VERIFIER, "archive-put"] and [Path(x).name for x in put[3:]] == ["circuit.txt", "pub-256.bin", "program-word.json"]
    assert not Path(archive).exists()                                # the verifier's archive is scratch, outside the record
    assert verifies == 3 * [[VERIFIER, "verify", "--statement", "verity/flock-circuit", "--archive", archive,
                             "--circuit", "key-circuit.txt", "--public", "key-pub-256.bin", "--program", "key-program-word.json",
                             "--zk", "--session", str(tree / "session")]]
    assert doc["schema"] == L.SCHEMA and doc["accepted"] is True and doc["host"] == LEAN_HOST
    assert doc["block"] == {"word": RMSNORM, "definition": "Staged_v2{N=1}", "k_log": 27, "blocks": 256, "full": 256}
    assert doc["proof"] == {"files": [{"name": "circuit.rep0.bin", "bytes": 300}, {"name": "circuit.rep1.bin", "bytes": 200}],
                            "bytes": 500, "record_bytes": 2}
    assert doc["statement"]["files"] == [{"name": "circuit.txt", "bytes": 4000}, {"name": "pub-256.bin", "bytes": 64},
                                         {"name": "program-word.json", "bytes": len(PROGRAM_FILE)}]
    assert (doc["statement"]["zk"], doc["statement"]["circuit_sha512"]) == (True, "key-circuit.txt")
    # what the circuit states, and the program handed: its digest is not the circuit's, and nothing holds it to the circuit
    assert doc["statement"]["bound"] == {
        "program_sha512": UNIT_SHA512, "held": [], "withheld": [],
        "partition": {"rule": L.PARTITION_COVER, "digest": _sha512(L.PARTITION_COVER.encode() + b"\0", PARTITION)}}
    assert doc["statement"]["program"] == {"sha512": PROGRAM_SHA512, "fn": RMSNORM} and PROGRAM_SHA512 != UNIT_SHA512
    assert doc["label"] == "accepted on the provisional unit cover, units as stated, the Program not held to the circuit"
    v = doc["verifier"]
    assert (v["archive_put_s"], v["repeat"], [r["wall_s"] for r in v["runs"]]) == (0.25, 3, [12.5, 10.0, 11.0])
    assert v["command"][:2] == ["lean-verifier", "verify"]                 # the executable by its file's name, not its path
    assert v["runs"][0] | {"verdict_sha512": None} == {
        "wall_s": 12.5, "cpu_s": 25.0, "user_s": 12.5, "maxrss_gib": 1.5, "rc": 0, "accepted": True, "why": None,
        "verdict_sha512": None, "phases_ms": {"inputs": 40, "setup": 900, "session": 2000}}
    assert capsys.readouterr().out.startswith('LEAN-VERIFY {"accepted": true, ')

    # the record is the one `prove` reads
    read = R.lean_verify(doc, "r20261110-000000-0001")
    assert read == P.LeanVerify(P.definition_id(RMSNORM), 27, 256, 256, (12.5, 10.0, 11.0), 500, "r20261110-000000-0001",
                                "cpu-node-0 (64 cores, A CPU 9000)", L.COVER_LABEL)
    assert read.statement_seconds == 11.0 and read.bits == 2**35


def test_a_refusal_ends_the_lean_verifiers_runner_with_no_further_run(tmp_path, monkeypatch, capsys):
    """The verifier refuses the second of three runs: the third is never started, the record says refused with the
    verifier's reason, the command exits 1, and `prove` takes no figure from it. A run that prints no verdict is a
    refusal too."""

    tree, out = kept_proof(tmp_path, stage=STAGE), tmp_path / "out"
    verifier = Verifier(monkeypatch, [(ACCEPTED, 12.0), ({"accepted": False, "why": "table circuit: lincheck"}, 3.0),
                                      (ACCEPTED, 12.0)])
    assert main(["lean-verify", "--lean", VERIFIER, "--proof", str(tree), "--out", str(out)]) == 1
    doc = json.loads((out / L.RECORD).read_text())
    assert len(verifier.calls) == 3 and len(verifier.verdicts) == 1          # archive-put and two runs: no third
    assert doc["accepted"] is False and [(r["accepted"], r["why"], r["rc"]) for r in doc["verifier"]["runs"]] == [
        (True, None, 0), (False, "table circuit: lincheck", 1)]
    assert "the Lean verifier refused run 2 of 3: table circuit: lincheck" in capsys.readouterr().out
    assert R.lean_verify(doc, "r20261110-000000-0001") == "the Lean verifier refused its statement: table circuit: lincheck"

    Verifier(monkeypatch, [(None, 1.0)])
    assert main(["lean-verify", "--lean", VERIFIER, "--proof", str(tree), "--out", str(out), "--repeat", "1"]) == 1
    assert json.loads((out / L.RECORD).read_text())["verifier"]["runs"][0]["why"] == "no verdict (rc 1)"


def test_the_lean_verifiers_runner_finds_each_file_by_name_and_takes_the_block_from_the_flags(tmp_path, monkeypatch):
    def measured(*argv: str) -> dict:
        parser = argparse.ArgumentParser()
        L.add_arguments(parser)
        return L.measure(parser.parse_args(["--lean", VERIFIER, "--proof", str(tree), "--repeat", "1", *argv]), tmp_path / "out")

    tree = kept_proof(tmp_path, rule=L.PARTITION_V1)
    verifier = Verifier(monkeypatch, [(ACCEPTED, 1.0)] * 3)
    with pytest.raises(ValueError, match="no block type: the proof holds no stage.json"):
        measured()
    with pytest.raises(ValueError, match=r"the statement's `k_log` is None: give --k-log"):
        measured("--definition", "Staged_v2{N=1}")
    assert verifier.calls == []
    # the flags name the block; a full statement's blocks are 2^35 witness bits' worth
    doc = measured("--definition", "Staged_v2{N=1}", "--k-log", "30", "--blocks", "16", "--plain", "--verify-arg=--coins",
                   "--verify-arg=tree")
    assert doc["block"] == {"word": None, "definition": "Staged_v2{N=1}", "k_log": 30, "blocks": 16, "full": 32}
    assert verifier.calls[-1][-6:] == ["--program", "key-program-word.json", "--coins", "tree", "--session", str(tree / "session")]
    assert doc["label"] == f"accepted as a plain session, the circuit's partition under {L.PARTITION_V1}"
    assert R.lean_verify(doc, "r1").definition == "Staged_v2{N=1}"

    # a name two files carry is refused, and its flag says which; partition.json and registered.json go where present
    (tree / "other").mkdir()
    (tree / "other" / "circuit.txt").write_text("another")
    (tree / "stage" / "partition.json").write_text(json.dumps(PARTITION))
    with pytest.raises(ValueError, match=r"2 paths under .* are named `circuit.txt` .*: say which with --circuit"):
        measured("--definition", "D", "--k-log", "30", "--blocks", "16")
    doc = measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))
    assert "--partition" in verifier.calls[-1] and [f["name"] for f in doc["statement"]["files"]] == [
        "circuit.txt", "pub-256.bin", "partition.json", "program-word.json"]
    assert doc["label"] == f"accepted at --zk, the circuit's partition under {L.PARTITION_V1}"
    (tree / "other" / "program.json").write_text("{}")
    with pytest.raises(ValueError, match=r"2 paths under .* are named `program\*.json` .*: say which with --program"):
        measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))
    (tree / "other" / "program.json").unlink()

    (tree / "session" / "circuit.rep0.bin").unlink()
    (tree / "session" / "circuit.rep1.bin").unlink()
    with pytest.raises(ValueError, match="holds no proof file"):
        measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))
    with pytest.raises(ValueError, match="archive-put` exited 1"):
        Verifier(monkeypatch, [], put_rc=1)
        (tree / "session" / "circuit.rep0.bin").write_bytes(b"p")
        measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))


def test_the_lean_verifiers_runner_holds_the_partition_and_the_circuit_and_labels_what_an_accept_holds(tmp_path, monkeypatch):
    """The circuit's second line binds a partition, and states a `program_sha512` that is no program file's digest.
    A partition file that isn't the bound object, or that the circuit binds nothing for, ends the command before any
    verifier process starts; under the provisional cover the verifier takes no partition object, so the bound cover is
    checked and not handed to it. The program is held to nothing the circuit states: the record says which was handed,
    and only one that names another Definition than `stage.json`'s `word` is refused. An accept on the provisional
    cover is labelled as holding no Program to the circuit. The circuit is held to `stage.json`'s `circuit_sha512`."""

    def measured(*argv: str) -> dict:
        parser = argparse.ArgumentParser()
        L.add_arguments(parser)
        return L.measure(parser.parse_args(["--lean", VERIFIER, "--proof", str(tree), "--repeat", "1", "--definition", "D",
                                            "--k-log", "30", "--blocks", "16", *argv]), tmp_path / "out")

    def refused(match: str) -> None:
        calls = len(verifier.calls)
        with pytest.raises(ValueError, match=match):
            measured()
        assert len(verifier.calls) == calls                                  # no verifier process, archive-put included

    tree = kept_proof(tmp_path, rule=L.PARTITION_V1)
    program, partition, circuit, stage = (tree / "stage" / n for n in ("program-word.json", "partition.json", "circuit.txt", "stage.json"))
    verifier = Verifier(monkeypatch, [(ACCEPTED, 1.0)] * 7)

    # the program: any descriptor is handed on, since the circuit's `program_sha512` is no digest of it; the record
    # says which, by the digest the verifier computes (annotations in none) and the Definition it names
    other = {"schema": "another", "definitions": PROGRAM["definitions"]}
    program.write_text(json.dumps(other | {"annotations": {"theta": [9, 9]}}, separators=(",", ":")))
    doc = measured()
    assert doc["statement"]["program"] == {"sha512": _sha512(b"", other), "fn": None} and doc["statement"]["bound"]["held"] == []
    assert verifier.calls[-1][verifier.calls[-1].index("--program") + 1] == "key-program-word.json"
    program.write_text("not a descriptor")
    refused(r"program-word\.json \(--program\) is not JSON .*: no verifier was started")
    # the stage's word is the Definition the program names, or the command ends: the verifier would accept either
    stage.write_text(json.dumps({"word": "Another_v1{N=2}"}))
    program.write_bytes(PROGRAM_FILE)
    refused(r"program-word\.json is the program of another Definition than the statement's: its `program_fn` is RMSNorm.*, and "
            r".*stage\.json names the `word` Another_v1\{N=2\}: no verifier was started")
    stage.write_text(json.dumps({"word": RMSNORM}))
    assert measured()["statement"]["program"] == {"sha512": PROGRAM_SHA512, "fn": RMSNORM}
    stage.unlink()

    # the bound partition object goes to the verifier; another is refused
    partition.write_text(json.dumps(PARTITION, indent=2))
    doc = measured()
    assert doc["statement"]["bound"]["held"] == ["partition"] and doc["statement"]["bound"]["withheld"] == []
    assert doc["label"] == f"accepted at --zk, the circuit's partition under {L.PARTITION_V1}"
    assert verifier.calls[-1][verifier.calls[-1].index("--partition") + 1] == "key-partition.json"
    partition.write_text(json.dumps(PARTITION | {"program": "q"}))
    refused(rf"partition\.json is not the partition the circuit binds: its digest under {L.PARTITION_V1} is .*: no verifier was started")

    # the provisional cover: the bound object is checked and withheld, since the verifier takes none under that rule
    circuit.write_bytes(circuit_text(L.PARTITION_COVER))
    refused(rf"partition\.json is not the partition the circuit binds: its digest under {L.PARTITION_COVER} is ")
    partition.write_text(json.dumps(PARTITION))
    doc = measured()
    put, verify = verifier.calls[-2:]
    assert [Path(x).name for x in put[3:]] == ["circuit.txt", "pub-256.bin", "program-word.json"] and "--partition" not in verify
    assert [x["name"] for x in doc["statement"]["files"]] == ["circuit.txt", "pub-256.bin", "program-word.json"]
    assert doc["statement"]["bound"] == {
        "program_sha512": UNIT_SHA512, "held": ["partition"], "withheld": ["partition"],
        "partition": {"rule": L.PARTITION_COVER, "digest": _sha512(L.PARTITION_COVER.encode() + b"\0", PARTITION)}}
    # the label of an accept there, word for word; a plain session's accept is not that one
    assert doc["label"] == L.COVER_LABEL == "accepted on the provisional unit cover, units as stated, the Program not held to the circuit"
    assert measured("--plain")["label"] == f"accepted as a plain session, the circuit's partition under {L.PARTITION_COVER}"

    # the circuit is the one stage.json names
    (tree / "stage" / "stage.json").write_text(json.dumps({"circuit_sha512": "key-circuit.txt"}))
    assert measured()["accepted"] is True
    (tree / "stage" / "stage.json").write_text(json.dumps({"circuit_sha512": "another"}))
    calls = len(verifier.calls)
    with pytest.raises(ValueError, match=r"circuit\.txt is not the circuit .*stage\.json names: its SHA-512 is key-circuit\.txt"):
        measured()
    assert [c[1] for c in verifier.calls[calls:]] == ["archive-put"]         # the verifier hashed it, and verified nothing

    # a circuit that binds nothing holds no file to it
    (tree / "stage" / "stage.json").unlink()
    circuit.write_bytes(b"x" * 4000)
    refused(r"circuit\.txt binds no partition under a rule this command hashes \(its rule is None; ")
    partition.unlink()
    doc = measured()
    assert doc["statement"]["bound"] == {"program_sha512": None, "partition": None, "held": [], "withheld": []}
    assert doc["label"] == "accepted at --zk, the circuit's partition under no rule"
    circuit.write_bytes(b"flock-circuit unit/made-up\nMETA {not json\n")
    refused(r"circuit\.txt: its second line is not `META` and a JSON object")
    assert verifier.verdicts == []


# ---- the command ---------------------------------------------------------------------------------------------------------------


def test_verity_bench_prove_is_the_commands_runner(tmp_path, monkeypatch, capsys):
    """`verity bench prove` runs benchmarks/prove/run.py, which is `verity-bench prove`: the table from the store alone."""

    s, arts = recorded(tmp_path)
    shutil.rmtree(tmp_path / "producer")
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))

    def run(*argv: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(REPO / "benchmarks" / "prove" / "run.py"), run_name="__main__")
        out = capsys.readouterr()
        return done.value.code, out.out, out.err

    code, out, _ = run("--system", "flock", "--workload", "glm53", "--rows", "2")
    assert code == 0 and out.startswith("# verity bench prove: flock on glm53 (GLM-5.3)\n")
    assert "- prover count (`--provers`, a stated input): 1;" in out and "8 staged types measured at steady state at 1 prover" in out
    assert "## Every block proved" in out and "| 148 more block types |" in out
    assert "- host cores (`--cores`, a stated input): 10 CPU cores a card, node A's share" in out
    code, out, _ = run("--system", "flock", "--workload", "glm53", "--provers", "2", "--json")
    assert code == 0 and (json.loads(out)["provers"], json.loads(out)["cores"]) == (2, 10)
    code, out, _ = run("--system", "flock", "--workload", "glm53", "--json", "--steady-rows", str(tmp_path / "steady.json"))
    wrote = json.loads((tmp_path / "steady.json").read_text())["rows"]
    measured = {r["block"]: r for r in json.loads(out)["rows"] if r["source"] == P.STEADY}
    assert code == 0 and len(wrote) == 8 and {r["block"] for r in wrote} == set(measured)
    assert all(r["statement_seconds"] / r["blocks"] == pytest.approx(measured[r["block"]]["seconds"], rel=1e-12) and r["provers"] == 1
               for r in wrote)
    code, out, _ = run("--system", "flock", "--workload", "glm53", "--cores", "24", "--json")
    doc = json.loads(out)
    assert code == 0 and (doc["provers"], doc["cores"]) == (1, 24) and doc["totals"]["above_cores"]["block_types"] == 0
    code, _, err = run("--system", "flock", "--workload", "glm53", "--cores", "0", "--local-only")
    assert code == 2 and "--cores 0: a type is priced on a host of one core a card or more" in err
    code, out, _ = run("--system", "spot", "--workload", "glm53", "--sampling", arts[R.SAMPLING], "--json")
    doc = json.loads(out)
    assert code == 0 and doc["schema"] == P.SCHEMA and doc["totals"]["gpus_per_serving_gpu"] == pytest.approx(0.5420038985704952)
    code, _, err = run("--system", "spot", "--workload", "glm53", "--sampling", arts[R.CENSUS], "--local-only")
    assert code == 2 and f"{arts[R.CENSUS]} isn't a sampling plan of record" in err
    with pytest.raises(SystemExit):
        main(["prove", "--system", "sp1", "--workload", "glm53"])
    assert "invalid choice: 'sp1'" in capsys.readouterr().err



def test_verity_bench_prove_lean_verify_is_the_lean_verifiers_runner_and_the_table_reads_its_record(tmp_path, monkeypatch, capsys):
    """`verity bench prove lean-verify` reaches the runner through the same file, and the record it writes, kept as a
    run of record's `out/lean-verify.json`, is what the table's two lines are from."""

    s, _ = recorded(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    Verifier(monkeypatch, [(ACCEPTED, 9.0)])

    def run(*argv: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(REPO / "benchmarks" / "prove" / "run.py"), run_name="__main__")
        return done.value.code, capsys.readouterr().out

    out_dir = tmp_path / "lean-out"
    code, out = run("lean-verify", "--lean", VERIFIER, "--proof", str(kept_proof(tmp_path, stage=STAGE)), "--out",
                    str(out_dir), "--repeat", "1")
    assert code == 0 and out.startswith("LEAN-VERIFY ")
    lean_ran(s, tmp_path, "r20261110-000000-0001", json.loads((out_dir / L.RECORD).read_text()))
    code, out = run("--system", "flock", "--workload", "glm53")
    assert code == 0 and ", modeled, not measured. Measured: one block type, `RMSNormFusedCudaVector8_v1{" in out
    assert (": accepted on the provisional unit cover, units as stated, the Program not held to the circuit, by the Lean "
            "verifier's `verify` in 9 s (one run;") in out and "per token: not measured" not in out
    code, out = run("--system", "flock", "--workload", "glm53", "--json")
    t = json.loads(out)["totals"]
    assert (t["lean_verifier_seconds_per_token"]["status"], t["proof_bytes_per_token"]["status"]) == ("modeled", "modeled")
    assert t["lean_verify"][0]["seconds"] == [9.0] and t["lean_verify"][0]["proof_bytes"] == 500
