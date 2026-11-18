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
measurement and one of each way a row isn't. `exfiltration/RUN/rows.jsonl` is the exfiltration run of record's world
year rows of its figure's case, cut to the fields `verity bench exfiltration`'s reader reads. The figures pinned below hold this code's arithmetic to a computation done
independently of it on the same records; they publish no measurement, and what the command says of a workload is what
it prints from the store's records of record."""

from __future__ import annotations

import ast
from dataclasses import replace

import pytest
from prove_helpers import (
    CENSUS,
    CUT,
    GEMM32,
    GEMM64,
    NORM,
    REPO,
    SESSIONS,
    _outside_a_research_run,  # noqa: F401  (autouse)
    plan,
    steady,
)
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import prove_records as R


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
    table = PF.Flock(reports, SESSIONS, "art:card").table(census)
    by = {r.block: r for r in table.rows}
    assert table.system == "flock" and list(by) == [b.name for b in census.blocks]

    gemm = by[GEMM64].cost
    assert (gemm.source, gemm.seconds, gemm.record, gemm.staged) == (P.STEADY, 1.0, "r20260102-000000-0002", "gemm64")
    assert gemm.verifier_seconds == 0.5 and gemm.at_steady_rate is None and gemm.provers == 1
    assert gemm.note == ("`gemm64` (`Gemm_v1{K=64,N=1,DOT=Dot_v1}`), 1 prover, full statements (out/rate/rate.json); "
                         "another size, statements 4/8 full (r20260103-000000-0003)")
    assert gemm.sizes == ((8, 8.0, "r20260102-000000-0002"), (4, 6.0, "r20260103-000000-0003"))
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
    alone = {r.block: r.cost for r in PF.Flock((), SESSIONS, "art:card").table(census).rows}
    assert (alone[GEMM64].source, alone[GEMM64].seconds) == (P.SINGLE, 0.5 + 2.0 / (1 << 15))
    assert all(c.at_steady_rate is None for c in alone.values())
    # with no card either, nothing is priced
    assert {r.cost.source for r in PF.Flock().table(census).rows} == {P.MISSING}


def test_a_types_rows_at_other_statement_sizes_are_its_sizes_and_move_no_figure():
    """A type's measured rows at its prover count and cores that timed statements of another size are carried beside
    the row that prices it, one a size, the most blocks first: the pricing row's run's where it timed that size, else
    the newest. A row at other cores or provers, one refused, and one staged in another Definition are no size of it.
    The sizes move no figure, and a type timed at one size has none."""

    census = P.Census.of(CENSUS)
    gemm, norm = P.definition_id(GEMM64), P.definition_id(NORM)
    run, newer, newest = "r20260102-000000-0002", "r20260103-000000-0003", "r20260104-000000-0004"
    sizes = (
        steady("gemm64", gemm, run, 8, 8.0),                                    # prices the type
        steady("gemm64", gemm, run, 2, 3.0),                                    # a quarter, the pricing row's run
        steady("gemm64", gemm, newest, 2, 2.5),                                 # a newer quarter
        steady("gemm64", gemm, newer, 4, 6.0),                                  # a half, another run
        steady("gemm64", gemm, newest, 1, 1.0, cores=24),                       # other cores
        steady("gemm64", gemm, newest, 1, 1.0, provers=2),                      # another prover count
        steady("gemm64", gemm, newest, 1, None, refused="timed out"),
        steady("gemm64", gemm, newest, 1, 1.0, form="Gemm_v2{K=64,N=1,DOT=Dot_v2}"),
        steady("norm", norm, run, 8, 24.0, k_log=22),
    )
    table = PF.Flock(sizes, SESSIONS, "art:card").table(census)
    by = {r.block: r.cost for r in table.rows}
    assert by[GEMM64].sizes == ((8, 8.0, run), (4, 6.0, newer), (2, 3.0, run))
    assert by[GEMM64].note.endswith(f"full statements (out/rate/rate.json); another size, statements 4/8 full ({newer}); "
                                    f"another size, statements 2/8 full ({run})")
    assert by[NORM].sizes == () and "another size" not in by[NORM].note
    one = PF.Flock((sizes[0], sizes[-1]), SESSIONS, "art:card").table(census)
    assert [replace(r, cost=replace(r.cost, sizes=(), note="")) for r in table.rows] == [
        replace(r, cost=replace(r.cost, note="")) for r in one.rows]
    rows = {r["block"]: r for r in P.report(table, 1.0)["rows"]}
    assert rows[GEMM64]["sizes"] == [{"blocks": 8, "statement_seconds": 8.0, "record": run},
                                     {"blocks": 4, "statement_seconds": 6.0, "record": newer},
                                     {"blocks": 2, "statement_seconds": 3.0, "record": run}]
    assert rows[NORM]["sizes"] is None and rows[GEMM32]["sizes"] is None


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
    table = PF.Flock(rows, (), None).table(census)
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
    whole = PF.Flock(rows, (), None).table(P.Census.of({"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
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
    flock = PF.Flock(rows, (), None, boolean=lambda block, definition: registered.get(block) == definition)
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
    unsaid = {r.block: r.cost for r in PF.Flock(rows, (), None).table(census).rows}
    assert unsaid[head].note == ("`head-bits` (another Definition, `Head_v2{K=4,FIRST=False}`), 1 prover, full statements "
                                 "(out/rate/rate.json)")
    assert "`Head_v1{K=4,FIRST=False}` in another Definition, `Head_v2{K=4,FIRST=False}`" in unsaid[layer].note

    # the form prices nothing: the same rows staged in their types' own Definitions give every figure the same
    plain = PF.Flock(tuple(replace(s, form=None) for s in rows), (), None).table(census)

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
    doc = P.report(PF.Flock(rows, SESSIONS, "art:card").table(census), 1.0)
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
        return {r.block: r.cost for r in PF.Flock(rows, SESSIONS, "art:card", provers=provers).table(census).rows}

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
    assert "at 2 provers priced at" in PF.Flock(rows, SESSIONS, "art:card", provers=2).table(census).range_basis
    assert [r["provers"] for r in P.report(PF.Flock(rows, SESSIONS, "art:card").table(census), 1.0)["rows"]] == [1, None, None]


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
        return {r.block: r.cost for r in PF.Flock(rows, SESSIONS, "art:card", **at).table(census).rows}

    assert (P.HOST_CORES, P.UNSTATED_CORES, PF.Flock().cores) == (10, 24, 10)
    assert PF.Steady("gemm64", gemm, 20, 8, 8, "r20260101-000000-0001", R.RATES, 8.0).cores == 24
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
    table = PF.Flock(rows, SESSIONS, "art:card").table(census)
    total = 2.0 * 2.0 + 1.0 * ten[GEMM32].seconds + 0.5 * 2.0
    out = P.report(table, 1.0)
    assert table.cores == 10 and out["totals"]["gpu_seconds_per_token"] == pytest.approx(total)
    assert out["totals"]["above_cores"] == {
        "cores": 10, "block_types": 1, "time_share": pytest.approx(1.0 / total), "timed_at": [12],
        "sources": [{"source": P.STEADY, "block_types": 1, "time_share": pytest.approx(1.0 / total)}]}
    assert {r["block"]: r["cores"] for r in out["rows"]} == {GEMM64: 10, NORM: 12, GEMM32: None}
    assert P.report(PF.Flock(rows, SESSIONS, "art:card", cores=12).table(census), 1.0)["totals"]["above_cores"] == {
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
    table = PF.Flock(rows, (), None).table(census)
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
    spot = P.Spot(PF.Flock(rows, (), None), plan(cells), P.Serving(0.5, 4.0, "stated")).table(census)
    assert spot.cores == 10
    assert P.report(spot, 1.0)["totals"]["above_cores"]["time_share"] == pytest.approx((10 * 2.0 + 20 * 4.0) / 110.0)

    # a row at the host's cores prices its type before the fuller, newer one above them does, and its kin with it
    exact = (*rows, steady("head-first", "Head_v1{K=4,FIRST=True}", "r20260101-000000-0001", 4, 6.0))
    table = PF.Flock(exact, (), None).table(census)
    assert [(r.cost.source, r.cost.seconds, r.cost.cores) for r in table.rows] == [
        (P.STEADY, 1.0, 10), (P.STEADY, 1.5, 10), (P.FAMILY, 3.0, 10)]
    out = P.report(table, 1.0)
    assert out["totals"]["above_cores"] == {"cores": 10, "block_types": 0, "time_share": 0.0, "timed_at": [], "sources": []}
    assert "- steady-state rows timed above the host's 10 cores: 0.00% of its proving time rests on them\n" in page(out)

    # on a host of 24 cores the row timed at 10 prices neither its own type nor its kin: both take the row at 24
    wide = PF.Flock(rows, (), None, cores=24).table(census)
    by = {r.block: r.cost for r in wide.rows}
    assert [(c.source, c.seconds, c.cores) for c in by.values()] == [(P.FAMILY, 2.0, 24), (P.STEADY, 2.0, 24), (P.FAMILY, 4.0, 24)]
    assert by[own].note == ("its Definition and K's, `head-first` (`Head_v1{K=4,FIRST=True}`), 1 prover, full statements, at "
                            "2^0 its seconds; measured at steady state at 10 cores, which price no type at 24")
    text = page(P.report(wide, 1.0), cores=24)
    assert "- host cores (`--cores`, a stated input): 24 CPU cores a card; a type is priced" in text and "node A" not in text
    assert "- steady-state rows timed above the host's 24 cores: 0.00% of its proving time rests on them\n" in text
