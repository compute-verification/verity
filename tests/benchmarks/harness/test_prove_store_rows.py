"""The steady-state record's runs and arts in the store: a run's own steady-state verdict, an art's rows in place of
its run's, each row's Definition and its form by the tree's registry, its cores, and a run from before
`rate.json`. `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from prove_helpers import (
    FIXTURES,
    REPO,
    RUNS,
    Store,
    _outside_a_research_run,  # noqa: F401  (autouse)
    census_art,
    fixture,
    recorded,
    registered,
    run_files,
)
from verity_bench import prove as P
from verity_bench import prove_records as R
from verity_bench import steady as S
from verity_bench import steady_node as N

#: what a steady-state point's verdict file holds of a row whose run's files didn't show it level, or its card clear
FAILED_VERDICT = {"identity": "i" * 64, "point": "x-k1", "passed": False, "why": "fail: `level` (fail): ...",
                  "verdict": {"type": "x", "provers": 1, "status": S.FAIL, "level": S.LEVEL, "checks": {
                      **{name: {"status": S.PASS, "why": None} for name in S.CHECKS},
                      "level": {"status": S.FAIL, "why": "its window isn't level"},
                      "card_clear_start_end": {"status": S.CANNOT_TELL, "why": "no record of its card"}}}}


def as_steady_point(s: Store, tmp: Path, run: str, validation: str | None, verdict: dict | None = None) -> None:
    """A fixture run as a run of the harness's steady-state point: recorded under its tool, its result's validation
    `validation`, and its files holding `verdict` as the point's verdict file where given."""

    files = tmp / "point-files" / run
    if not files.exists():
        shutil.copytree(run_files(tmp, run), files)
    if verdict is not None:
        (files / N.VERDICT_FILE).write_text(json.dumps(verdict))
    s.ran(run, files, tool=R.STEADY_POINT_TOOL, validation=validation)


def test_the_steady_point_tool_and_its_verdict_file_are_the_harnesss():
    from verity_bench import tool

    assert tool.STEADY_POINT.name == R.STEADY_POINT_TOOL and N.VERDICT_FILE == "out/verdict.json"


def test_a_labelled_steady_point_whose_own_verdict_failed_prices_nothing_and_is_named(tmp_path):
    """A run of the harness's steady-state point that judged itself not passed: each of its rows is refused with the
    run and the checks its verdict file names, whatever the six checks this reader holds a row to say, and every type
    it measured is priced as with no such run (here, as with the run superseded). The run is named among the
    measurements' inputs with why."""

    newest = RUNS[-1]
    s, _ = recorded(tmp_path)
    s.label(newest, R.SUPERSEDED_BY, "r20991231-000000-0000")
    without = R.build("spot", "glm53", store=s.fresh("without"), root=REPO)

    s, _ = recorded(tmp_path / "point")
    as_steady_point(s, tmp_path / "point", newest, "failed", FAILED_VERDICT)
    doc = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    why = (f"its own steady-state verdict failed: run {newest} didn't pass `card_clear_start_end` (cannot tell), "
           "`level` (fail)")
    assert {x["target"]: x["why"] for x in doc["refused"]} == {newest: why}
    assert not any(r.get("record") == newest for r in doc["rows"])
    assert doc["sources"] == without["sources"]
    assert [(r["block"], r["source"], r["seconds"]) for r in doc["rows"]] == [
        (r["block"], r["source"], r["seconds"]) for r in without["rows"]]
    assert doc["totals"]["gpus_per_serving_gpu"] == without["totals"]["gpus_per_serving_gpu"]
    head, seconds = R.measurements(doc, {"a/x": "gpus_per_serving_gpu", "a/y": "gpu_seconds_per_token"})
    assert all(f"refused {newest}: {why}" in m["inputs"] for m in (head, seconds))
    assert (head["mark"], head["value"]) == next((m["mark"], m["value"]) for m in R.measurements(
        without, {"a/x": "gpus_per_serving_gpu"}))
    assert f"- {newest} (`prove-glm53-flock-steady`): {why}" in R.render(doc)


@pytest.mark.parametrize("validation, verdict, says", [
    (None, None, ("can't be told: run {run} is a steady-state point's, and its attempt carries no verdict (its result's "
                  "validation is None)")),
    ("not_run", None, ("can't be told: run {run} is a steady-state point's, and its attempt carries no verdict (its "
                       "result's validation is 'not_run')")),
    ("failed", None, "failed: run {run} keeps no `out/verdict.json` naming the checks it didn't pass"),
    ("failed", {"passed": False, "why": "refused, nothing timed: no lease", "verdict": None},
     "failed: run {run} wasn't a passed point: refused, nothing timed: no lease"),
])
def test_a_steady_point_whose_verdict_cannot_be_read_or_names_no_check_is_refused(tmp_path, validation, verdict, says):
    """A run of the point whose attempt carries no verdict can't be told and prices nothing, as a row whose fields
    can't show a check; a failed one whose verdict file is missing, or names no check, is refused with what it says."""

    newest = RUNS[-1]
    s, _ = recorded(tmp_path)
    as_steady_point(s, tmp_path, newest, validation, verdict)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert {x["target"]: x["why"] for x in doc["refused"]} == {newest: "its own steady-state verdict " + says.format(run=newest)}
    assert not any(r.get("record") == newest for r in doc["rows"])


def test_a_passed_steady_point_and_a_run_with_no_verdict_of_its_own_are_priced_as_before(tmp_path):
    """A run of the point that judged itself passed is read as any run: its rows are priced as the six checks find
    them. A run under any other tool has no verdict of its own, so a failed validation of its result is no verdict on
    its rows: they are priced as before. Either table is the one the same records gave before."""

    newest = RUNS[-1]
    s, _ = recorded(tmp_path)
    before = R.build("spot", "glm53", store=s.fresh("before"), root=REPO)
    as_steady_point(s, tmp_path, newest, "passed", {"passed": True, "why": None, "verdict": None})
    assert R.build("spot", "glm53", store=s.fresh("passed"), root=REPO) == before
    s.ran(newest, run_files(tmp_path, newest), validation="failed")
    assert R.build("spot", "glm53", store=s.fresh("older"), root=REPO) == before
    assert before["refused"] == [] and {x["source"]: x["block_types"] for x in before["sources"]}[P.STEADY] == 8


def test_an_art_completing_a_failed_steady_points_rows_prices_none_of_them(tmp_path):
    """An art holding a run's rows completed after it stands for that run, so the run's own verdict holds of them."""

    newest = RUNS[-1]
    whole = json.loads((FIXTURES / "steady" / newest / "rate" / "rate.json").read_text())
    s = Store(tmp_path)
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s.record(s.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s.record(census_art(s), R.CENSUS)
    as_steady_point(s, tmp_path, newest, "failed", FAILED_VERDICT)
    art = s.record(s.tree("evidence/v1", R.ART_RATES, whole, {"runs": [newest]}), R.STEADY)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert [x["target"] for x in doc["refused"]] == [art]
    assert doc["refused"][0]["why"].startswith(f"its own steady-state verdict failed: run {newest} didn't pass `card_")
    assert {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 0


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


def test_a_definition_is_a_types_boolean_version_only_where_the_trees_registry_says_so():
    """`boolean_version` asks the tree's registry: a census type's Boolean version is the one Definition
    `verity.circuits.ops.boolean.versions.boolean_version` gives for the type's own, at the statics it binds, a function-valued
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

    def store(name: str, host: object) -> Store:
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
