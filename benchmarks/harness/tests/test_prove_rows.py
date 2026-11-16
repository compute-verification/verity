"""A steady-state row: judged by the rules' own checks, what it lacks to be priced, the type it names, its cores, and
whether its card was on its CPUs' NUMA node (the off-node scan, and the totals' count of each kind). `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Any

import pytest
from prove_helpers import (
    FIXTURES,
    REPO,
    ROW_WHYS,
    RUNS,
    _outside_a_research_run,  # noqa: F401  (autouse)
    recorded,
)
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import prove_records as R
from verity_bench import prove_rows
from verity_bench import steady as S

REFUSALS = json.loads((FIXTURES / "refusals" / "rate" / "rate.json").read_text())
#: a field a made-up row is left without
DROP = object()
#: made-up rows the steady-state rules don't pass and a reader of the fields alone would: what is changed in the good
#: row, and what the rules' check then says
STRICT: list[tuple[dict[str, Any], str]] = [
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

    assert tuple(getattr(S, name) for name in GATES) == R.CHECKS
    assert tuple(sorted(GATES + NO_GATES, key=S.CHECKS.index)) == S.CHECKS and S.one_numa_node not in R.CHECKS
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
        assert S.one_numa_node(row, lambda path: None).status == status
        assert R.row_why(row) is None and R.steady_row(row, "r1").gpu_seconds == R.steady_row(good, "r1").gpu_seconds


def test_a_row_is_a_steady_state_measurement_only_when_every_session_was_accepted_and_its_cache_flat():
    """`fixtures/prove/refusals`: one row that is a steady-state measurement, and one of each way a row isn't, each
    refused with its reason; a row lacking a field names the field."""

    rows = {row["type"]: row for row in REFUSALS}
    assert {name: R.row_why(rows[name]) for name in ROW_WHYS} == ROW_WHYS
    read = {name: R.steady_row(rows[name], "r20260101-000000-0001") for name in ROW_WHYS}
    assert all(isinstance(s, PF.Steady) and s.refused == ROW_WHYS[name] and s.measured == (name == "good")
               for name, s in read.items())
    # a statement's GPU-seconds are the GPUs its provers held over its statements a second, and a block's are that over
    # the statement's blocks
    good = read["good"]
    assert (good.type, good.definition, good.k_log, good.blocks, good.full, good.provers) == ("good", "Good_v1{N=1}", 20, 8, 8, 1)
    assert (good.gpu_seconds, good.verifier_seconds, good.report) == (1 / 0.125, 4.0, "out/rate/rate.json")
    four = {"gpus": 2, "provers": 4, "accepted": 12, "warm_per_prover": {f"prover-gpu{i}": 1 for i in range(4)}}
    assert R.steady_row(rows["good"] | four, "r1").gpu_seconds == 2 / 0.125
    cost = PF.Flock((good,)).table(P.Census((P.Block("Good_v1{N=1}", "Good_v1", 1.0, 20),), 1.0, 1.0)).rows[0].cost
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
    cost = PF.Flock((read,)).table(P.Census((P.Block("Good_v1{N=1}", "Good_v1", 1.0, 20),), 1.0, 1.0)).rows[0].cost
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


#: a scan that read two runs, and found one of them off its card's node
SCAN = R.OffNodeScan("prove-glm53-flock-steady", frozenset({"r20261110-211750-65a0", "r20261111-000000-0001"}),
                     frozenset({"r20261110-211750-65a0"}))


@pytest.mark.parametrize("row, run, off", [
    ({R.CARD_NUMA: 1, "cores": {"count": 24, "numa_nodes": [0]}}, "r20261112-000000-0001", True),
    ({R.CARD_NUMA: 0, "cores": {"count": 24, "numa_nodes": [0, 1]}}, "r20261112-000000-0001", True),
    ({R.CARD_NUMA: 0, "cores": {"count": 24, "numa_nodes": [0]}}, "r20261112-000000-0001", False),
    # a row that records its card's node is read by it, whatever the scan says of its run
    ({R.CARD_NUMA: 2, "cores": {"count": 24, "numa_nodes": [2]}}, "r20261110-211750-65a0", False),
    ({R.CARD_NUMA: 1, "cores": {"count": 24, "numa_nodes": [0]}}, "r20261111-000000-0001", True),
    ({}, "r20261110-211750-65a0", True),
    ({}, "r20261111-000000-0001", False),
    ({R.CARD_NUMA: None, "cores": {"count": 24, "numa_nodes": [0]}}, "r20261111-000000-0001", False),
    # a run between two the scan read is no run it read
    ({}, "r20261111-000000-0000", None),
    ({}, "r20261112-000000-0001", None),
    ({R.CARD_NUMA: None, "cores": {"count": 24, "numa_nodes": [0]}}, "r20261111-000000-0000", None),
    ({R.CARD_NUMA: 1, "cores": {"count": 24}}, "r20261112-000000-0001", None),
    ({R.CARD_NUMA: True, "cores": {"count": 24, "numa_nodes": [0]}}, "r20261112-000000-0001", None),
], ids=["differs", "spans-two", "equal", "equal-but-listed", "differs-but-read", "listed", "read-unlisted",
        "null-read-unlisted", "unread-between-read", "unread-later", "null-unread", "no-cpu-nodes", "not-a-node"])
def test_a_rows_card_is_off_or_on_its_cpus_numa_node_by_its_own_record_else_by_the_scan_of_its_run(row, run, off):
    assert R.off_node(row, run, SCAN) is off


def test_the_scan_holds_the_six_priced_off_node_runs_among_the_79_it_found_in_the_220_it_read(tmp_path):
    doc = json.loads(R.OFF_NODE.read_text())
    scan = R.off_node_scan()
    assert scan.record == doc["record"] == "prove-glm53-flock-steady"
    assert len(scan.read) == 220 == len(doc["read"]) and len(scan.off) == 79 == len(doc["off"]) and scan.off <= scan.read
    six = {"r20261110-211750-65a0": (2, [0]), "r20261110-190448-3b46": (2, [1]), "r20261110-190404-431a": (3, [2]),
           "r20261110-190457-2ce7": (2, [1]), "r20261110-232129-ed12": (2, [1]), "r20261110-232120-0c7b": (2, [1])}
    assert {r: (doc["off"][r]["card_numa_node"], doc["off"][r]["cpu_numa_nodes"]) for r in six} == six
    assert all(R.off_node({}, r) is True for r in six)
    assert R.off_node({}, "r20261111-013927-ac13") is False
    # inside the span of the runs read, but none of them: a run of a superseded target, and another record's run
    assert min(scan.read) < "r20261111-015333-38d6" < "r20261113-162903-6d87" < max(scan.read)
    assert R.off_node({}, "r20261111-015333-38d6") is None and R.off_node({}, "r20261113-162903-6d87") is None
    assert R.off_node({}, "r20261108-045435-b9cb") is None
    assert "host record and lease line" in doc["source"] and "kernels lane" in doc["source"]

    good = {"source": "s", "record": "prove-glm53-flock-steady", "read": ["r20261110-000000-0001", "r20261110-000000-0002"],
            "off": {"r20261110-000000-0001": {"card_numa_node": 2, "cpu_numa_nodes": [1]}}}

    def loaded(text: str) -> R.OffNodeScan:
        path = tmp_path / "scan.json"
        path.write_text(text)
        return R.off_node_scan.__wrapped__(path)

    scan = loaded(json.dumps(good))
    assert (scan.of("r20261110-000000-0001"), scan.of("r20261110-000000-0002"), scan.of("r20261110-000000-0003")) == (
        True, False, None)
    for bad, match in [
        (good | {"record": "prove-glm53-flock-single"}, "is no workload's steady-state record"),
        (good | {"read": [*good["read"], "r20261110-000000-0002"]}, "`read` lists r20261110-000000-0002 more than once"),
        (good | {"read": [*good["read"], "run-3"]}, "`read` lists what isn't a run id: run-3"),
        (good | {"off": {"r20261110-000000-0009": {"card_numa_node": 2, "cpu_numa_nodes": [1]}}},
         "off-node run r20261110-000000-0009 isn't among the runs the scan read"),
        (good | {"off": {"r20261110-000000-0001": {"card_numa_node": 1, "cpu_numa_nodes": [1]}}},
         "lists no card's NUMA node and CPUs' NUMA nodes other than it alone"),
    ]:
        with pytest.raises(ValueError, match=re.escape(match)):
            loaded(json.dumps(bad))
    twice = json.dumps(good).replace('"off": {', '"off": {"r20261110-000000-0002": {"card_numa_node": 2, "cpu_numa_nodes": [1]}, '
                                     '"r20261110-000000-0002": {"card_numa_node": 2, "cpu_numa_nodes": [1]}, ')
    with pytest.raises(ValueError, match="lists r20261110-000000-0002 more than once"):
        loaded(twice)


def test_the_proving_totals_count_the_priced_steady_rows_off_on_and_not_recorded_against_their_cards_numa_node(
        tmp_path, monkeypatch):
    """Runs the scan lists count off where their rows price a type, and not where they are superseded or their row
    isn't a steady-state measurement. A row that records its card's node counts by it: off where it differs, on where
    it's the CPUs' node. One that doesn't, or records null, counts on where the scan read its run, and not recorded
    where it didn't, as the fixture's own runs. The shares are the table's own weights,
    the rows' shares of the proving time at the sampling plan; the single-session card's rows are counted apart; and
    the count moves no value and no mark."""

    s, _ = recorded(tmp_path)
    rows = {r["type"]: r for r in json.loads((FIXTURES / "steady" / RUNS[-1] / "rate" / "rate.json").read_text())}

    def ran(run: str, row: dict) -> str:
        files = tmp_path / "off-node" / run
        (files / "out" / "rate").mkdir(parents=True)
        (files / "out" / "rate" / "rate.json").write_text(json.dumps([row]))
        s.ran(run, files)
        return s.record(run, R.STEADY)

    def cpus(card: int | None) -> dict:
        return {R.CARD_NUMA: card, "cores": {"count": 24, "list": "0-23", "numa_nodes": [0]}}

    listed = ["r20261110-211750-65a0", "r20261110-211750-65a1", "r20261110-211750-65a2", "r20261110-211750-65a3"]
    monkeypatch.setattr(prove_rows, "off_node_scan", lambda: replace(SCAN, read=SCAN.read | {*listed, "r20261111-000000-0003"},
                                                                     off=frozenset(listed)))
    priced = [ran(listed[0], rows["rmsnorm-v8"]), ran(listed[1], rows["ar-multimem"])]
    unpriced = ran(listed[2], rows["residual-add"] | {"flat": False})
    superseded = ran(listed[3], rows["fr-w2"])
    s.label(superseded, R.SUPERSEDED_BY, listed[0])
    differs = ran("r20261112-000000-0001", rows["fr-w13"] | cpus(1))
    equal = ran("r20261112-000000-0002", rows["sm-fp8-k2048"] | cpus(0))
    read = ran("r20261111-000000-0003", rows["sm-fp8-k6144"] | cpus(None))
    doc = R.build("spot", "glm53", store=s.fresh(), root=REPO)

    steady_rows = [r for r in doc["rows"] if r["source"] in (P.STEADY, P.FAMILY)]
    by = {r["record"] for r in steady_rows}
    assert {*priced, differs, equal, read} <= by and not {unpriced, superseded} & by
    off, t = doc["totals"]["off_node"], doc["totals"]
    assert off["off"]["runs"] == sorted([*priced, differs]) and off["on"]["runs"] == sorted([equal, read])
    assert off["not_recorded"]["runs"] and set(off["not_recorded"]["runs"]) <= set(RUNS)
    assert off["off"]["rows"] == 3 and off["on"]["rows"] == 2
    assert off["rows"] == off["off"]["rows"] + off["on"]["rows"] + off["not_recorded"]["rows"]

    def share(records: list[str]) -> float:
        return sum(r["time_share"] for r in steady_rows if r["record"] in records)

    for kind in ("off", "on", "not_recorded"):
        assert off[kind]["time_share"] == pytest.approx(share(off[kind]["runs"]), rel=1e-12) and off[kind]["time_share"] > 0
    single = [r for r in doc["rows"] if r["source"] in (P.SINGLE, P.MODELED) and r["seconds"] is not None]
    assert single and off["single"] == {"rows": len(single), "time_share": pytest.approx(
        sum(r["time_share"] for r in single), rel=1e-12)}
    line = (f"card off its CPUs' NUMA node: 3 of {off['rows']} priced steady rows, {off['off']['time_share']:.4%} of the "
            f"proving time; on its node by the runs' own records: 2 rows, {off['on']['time_share']:.4%}; not recorded: "
            f"{off['not_recorded']['rows']} rows, {off['not_recorded']['time_share']:.4%}; priced from the single-session "
            f"card, whose table records no CPUs: {len(single)} rows, {off['single']['time_share']:.4%}")
    assert R.off_node_line(off) == line and f"\n- {line}\n" in R.render(doc)
    metrics = {"a/x": "gpus_per_serving_gpu", "a/y": "gpu_seconds_per_token", "a/z": "proof_bytes_per_token"}
    head, seconds, size = R.measurements(doc, metrics)
    assert head["inputs"][3] == seconds["inputs"][1] == line and line not in size.get("inputs", [])

    monkeypatch.setattr(prove_rows, "off_node_scan", lambda: R.OffNodeScan(SCAN.record, frozenset(), frozenset()))
    none = R.build("spot", "glm53", store=s.fresh("unscanned"), root=REPO)
    assert none["totals"]["off_node"]["off"]["runs"] == [differs] and none["totals"]["off_node"]["on"]["runs"] == [equal]
    assert {k: v for k, v in none["totals"].items() if k != "off_node"} == {k: v for k, v in t.items() if k != "off_node"}
    strip = [{k: v for k, v in m.items() if k != "inputs"} for m in R.measurements(none, metrics)]
    assert strip == [{k: v for k, v in m.items() if k != "inputs"} for m in (head, seconds, size)]
