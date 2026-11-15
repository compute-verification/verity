"""The store side on a real store with a directory remote: the hand computation the records reproduce, a plan's own
prices, the prover GPUs a serving GPU at the exfiltration run of record's design, inputs refused by name, and a
run that can't be read. `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

import json
import shutil
from statistics import linear_regression

import pytest
from prove_helpers import (
    DESIGN,
    FIXTURES,
    REPO,
    ROW_WHYS,
    RUNS,
    Store,
    W,
    _outside_a_research_run,  # noqa: F401  (autouse)
    census_art,
    designed,
    fixture,
    plan_art,
    recorded,
    run_files,
)
from verity_bench import exfiltration as E
from verity_bench import kit
from verity_bench import prove as P
from verity_bench import prove_records as R


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
    assert [(k, x["block_types"]) for k, x in by.items()] == [(P.STEADY, 8), (P.WARM, 0), (P.FAMILY, 59), (P.SINGLE, 27), (P.MODELED, 56), (P.MISSING, 0)]
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
    assert text.count("\n| `") == 2 * 150                               # the cost table and the Lean verifier's


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


def test_spot_prints_the_prover_gpus_a_serving_gpu_at_the_exfiltration_run_of_records_design_beside_the_plans(tmp_path):
    """Beside the prover GPUs a serving GPU at the sampling plan of record, SPOT's table gives them at the exfiltration
    run of record's design, each line naming its source: the planner's world year at 10% of total compute, read off by
    `verity bench exfiltration`'s reader from that run's rows (`fixtures/prove/exfiltration/`, the run of record's own
    world year rows), its proving's share of total compute over 1 - its audit budget. With no record, or a superseded
    one, the line says why and the table stands; with every block proved there is no such line."""

    s, arts = recorded(tmp_path)
    bare = R.build("spot", "glm53", store=s.fresh("bare"), root=REPO)
    assert bare["at_record"]["gpus_per_serving_gpu"] is None and bare["at_record"]["run"] is None
    nothing = (f"no planner rows of record: nothing in the store carries `{R.RUN_OF_RECORD}={E.RECORD}` (label each run "
               f"that holds them: `research data label RUN {R.RUN_OF_RECORD} {E.RECORD} --by LANE`)")
    assert bare["at_record"]["why"] == nothing
    text = R.render(bare)
    assert (f"- prover GPUs per serving GPU at the sampling plan of record ({arts[R.SAMPLING]}): 0.542, modeled\n"
            f"- prover GPUs per serving GPU at the exfiltration run of record's design (`{R.RUN_OF_RECORD}={E.RECORD}`): "
            f"not read: {nothing}\n") in text
    (none,) = R.measurements(bare, {"a/z": R.AT_RECORD})
    assert none == {"name": "a/z", "value": None, "unit": "prover GPUs a serving GPU", "mark": "modeled",
                    "why": f"not read: {nothing}", "inputs": [], "target": R.GPUS_PER_SERVING_GPU_TARGET}
    assert kit.writer(tmp_path / "bare-run").measurement(**none)["inputs"] == ["modeled"]

    designed(s, tmp_path)
    spot = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    got = spot["at_record"]
    rows = [json.loads(line) for line in (FIXTURES / "exfiltration" / DESIGN / "rows.jsonl").read_text().splitlines()]
    slope, intercept = linear_regression([r["b10"] for r in rows], [r["total"] for r in rows])
    beta = (0.10 - intercept) / slope
    lo, hi = next((a, b) for a, b in zip(rows, rows[1:], strict=False) if a["b10"] <= beta <= b["b10"])
    prove = lo["prove"] + (hi["prove"] - lo["prove"]) * (beta - lo["b10"]) / (hi["b10"] - lo["b10"])
    assert (got["record"], got["run"], got["scope"], got["share"], got["status"], got["why"]) == (
        E.RECORD, DESIGN, "world year", 0.10, "modeled", None)
    assert got["beta"] == pytest.approx(beta, rel=1e-12) == pytest.approx(0.054555103866200196, rel=1e-12)
    assert got["prove_share"] == pytest.approx(prove, rel=1e-12) == pytest.approx(0.052138852549246315, rel=1e-12)
    assert got["gpus_per_serving_gpu"] == pytest.approx(prove / (1 - beta), rel=1e-12)
    assert got["gpus_per_serving_gpu"] == pytest.approx(0.055147426108551965, rel=1e-12)
    # the table's own figures don't move
    assert spot["totals"] == bare["totals"] and spot["rows"] == bare["rows"]
    text = R.render(spot)
    assert (f"- prover GPUs per serving GPU at the sampling plan of record ({arts[R.SAMPLING]}): 0.542, modeled\n"
            "- prover GPUs per serving GPU at the exfiltration run of record's design: 0.05515, modeled: run "
            f"{DESIGN} (`{R.RUN_OF_RECORD}={E.RECORD}`), the planner's world year at 10% of total compute: its proving's "
            "5.214% of total compute over 1 - its audit budget, 5.456%\n") in text
    (one,) = R.measurements(spot, {"a/z": R.AT_RECORD})
    assert one == {"name": "a/z", "value": got["gpus_per_serving_gpu"], "unit": "prover GPUs a serving GPU",
                   "mark": "modeled", "inputs": [f"modeled: {R._design(got)}", DESIGN], "target": 0.0545,
                   "verdict": "misses"}

    s.label(DESIGN, R.SUPERSEDED_BY, "r20261114-000000-0000")
    gone = R.build("spot", "glm53", store=s.fresh("gone"), root=REPO)["at_record"]
    assert gone["gpus_per_serving_gpu"] is None and gone["why"] == (
        f"no run of record holds the planner's rows: {DESIGN} superseded by r20261114-000000-0000 (label one that does: "
        f"`research data label RUN {R.RUN_OF_RECORD} {E.RECORD} --by LANE`)")

    flock = R.build("flock", "glm53", store=s.fresh("flock"), root=REPO)
    assert "at_record" not in flock and "run of record's design" not in R.render(flock)
    with pytest.raises(ValueError, match="`gpus_per_serving_gpu_at_record` is SPOT's: only a table at a sampling plan"):
        R.measurements(flock, {"a/z": R.AT_RECORD})


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
