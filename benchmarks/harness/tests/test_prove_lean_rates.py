"""The Lean verifier's seconds and the proof's bytes on small tables: a measured type's figures and the rest priced
by size, each type's mark and the record it rests on, and SPOT at the blocks its plan samples. `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from prove_helpers import (
    CENSUS,
    CUT,
    GEMM32,
    GEMM64,
    NORM,
    SESSIONS,
    _outside_a_research_run,  # noqa: F401  (autouse)
    lean,
    lean_record,
    lean_shares,
    plan,
    steady,
)
from verity_bench import lean_verify as L
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import prove_records as R


def test_the_lean_verifiers_seconds_and_the_proofs_bytes_are_one_types_and_priced_to_the_rest_by_size():
    """A block type the Lean verifier ran on takes its statement's median seconds and its proof's bytes over the
    statement's blocks; another type of its Definition takes them a unit of size times its own, every other type the
    rate fitted over the measured statements times its own: seconds by AND gates, bytes by witness bits. A type with
    no size takes nothing of that figure. The totals are modeled, and say how much of each figure each source holds."""

    census = P.Census.of(CENSUS).recut(CUT)
    rows = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),)
    timed = (lean(P.definition_id(GEMM64), "r20260105-000000-0005", (40.0, 16.0, 24.0), 4000),
             lean(P.definition_id(GEMM64), "r20260104-000000-0004", (1.0,), 1),             # an older run of the type
             lean("Other_v1{N=2}", "r20260106-000000-0006", (1.0,), 1))                       # no block type of the census
    table = PF.Flock(rows, SESSIONS, "art:card", lean=timed).table(census)
    assert [v.run for v in table.lean] == ["r20260105-000000-0005"]
    by = {r.block: r.cost for r in table.rows}
    # 24 s over 8 blocks of 1000 ANDs: 0.003 s an AND; 4000 bytes over 8 blocks of 2^20 witness bits
    assert [(by[n].lean_seconds, by[n].lean_source, by[n].proof_bytes, by[n].proof_bytes_source)
            for n in (GEMM64, GEMM32, "Half_v1{N=4}", "Bare_v1{N=4}")] == [
        (3.0, P.LEAN_MEASURED, 500.0, P.LEAN_MEASURED), (pytest.approx(1.2), P.LEAN_FAMILY, 250.0, P.LEAN_FAMILY),
        (pytest.approx(6.0), P.LEAN_FITTED, 1000.0, P.LEAN_FITTED), (pytest.approx(1.5), P.LEAN_FITTED, None, None)]
    assert table.lean_fitted == (pytest.approx(0.003), 4000 / 2**23)
    # the proof server's verifier's seconds stay the steady-state row's own
    assert by[GEMM64].verifier_seconds == 0.5

    doc = P.report(table, 1.0)
    t = doc["totals"]
    total = 2.0 * 3.0 + 1.2 + 6.0 + 0.25 * 1.5
    assert t["lean_verifier_seconds_per_token"] == {
        "value": pytest.approx(total), "status": "modeled", "unpriced_block_types": 0, "fitted_rate": pytest.approx(0.003),
        "sources": lean_shares((1, 6.0 / total), (1, 1.2 / total), (2, 6.375 / total)), "basis": P.LEAN_SECONDS_BASIS}
    assert t["proof_bytes_per_token"] == {
        "value": 2250.0, "status": "modeled", "unpriced_block_types": 1, "fitted_rate": 4000 / 2**23,
        "sources": lean_shares((1, 1000.0 / 2250), (1, 250.0 / 2250), (1, 1000.0 / 2250)), "basis": P.PROOF_BYTES_BASIS}
    share = 2.0 * by[GEMM64].seconds / table.gpu_seconds_per_token()
    assert t["lean_verify"] == [{"block": GEMM64, "time_share": share, "k_log": 20, "blocks": 8, "full": 8,
                                 "seconds": [40.0, 16.0, 24.0], "statement_seconds": 24.0, "proof_bytes": 4000,
                                 "run": "r20260105-000000-0005", "host": "node-a (64 cores, a CPU)",
                                 "label": L.COVER_LABEL, "form": None}]
    row = next(r for r in doc["rows"] if r["block"] == GEMM32)
    assert (row["lean_verifier_seconds"], row["lean_source"], row["proof_bytes"], row["proof_bytes_source"]) == (
        pytest.approx(1.2), P.LEAN_FAMILY, 250.0, P.LEAN_FAMILY)
    assert t["verifier_seconds_per_token"] == {"value": 1.0, "block_types": 1, "block_share": 2.0 / 4.25}

    # with a second Definition measured, the family keeps its own rate and the rest take the pooled one: 88 s over
    # 8 * 1000 + 2 * 2000 ANDs, and 20000 bytes over 2^23 + 2^22 witness bits
    two = PF.Flock(rows, SESSIONS, "art:card", lean=(*timed, lean("Half_v1{N=4}", "r20260105-000000-0005", (64.0,), 16000,
                                                                 k_log=21, blocks=2, full=4))).table(census)
    by = {r.block: r.cost for r in two.rows}
    assert (by["Half_v1{N=4}"].lean_seconds, by["Half_v1{N=4}"].proof_bytes, by["Half_v1{N=4}"].lean_source) == (
        32.0, 8000.0, P.LEAN_MEASURED)
    assert (by[GEMM32].lean_seconds, by[GEMM32].proof_bytes) == (pytest.approx(1.2), 250.0)
    assert by["Bare_v1{N=4}"].lean_seconds == pytest.approx(88.0 / 12000 * 500)
    assert two.lean_fitted == (pytest.approx(88.0 / 12000), pytest.approx(20000 / (2**23 + 2**22)))
    assert [x["block"] for x in P.report(two, 1.0)["totals"]["lean_verify"]] == [GEMM64, "Half_v1{N=4}"]

    # measured only when every block type has a record of its own; none of either with no record
    whole = P.Census.of({**CENSUS, "definitions": {GEMM64: CENSUS["definitions"][GEMM64]}})
    own = P.report(PF.Flock(rows, SESSIONS, "art:card", lean=timed).table(whole), 1.0)["totals"]
    assert own["lean_verifier_seconds_per_token"] == {
        "value": 6.0, "status": "measured", "unpriced_block_types": 0, "fitted_rate": pytest.approx(0.003),
        "sources": lean_shares((1, 1.0), (0, 0.0), (0, 0.0)), "basis": P.LEAN_SECONDS_BASIS}
    none = P.report(PF.Flock(rows, SESSIONS, "art:card").table(census), 1.0)
    assert none["totals"]["lean_verifier_seconds_per_token"] is None and none["totals"]["proof_bytes_per_token"] is None
    assert none["totals"]["lean_verify"] == [] and {r["lean_source"] for r in none["rows"]} == {None}


#: one served token of a head block measured at two statics and at a third not, a GEMM measured, and a read of no
#: measured Definition
HEADS: dict[str, Any] = {
    "tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 100, "macs": 100},
    "definitions": {
        "Head_v1{NVIS=64,FIRST=false}": {"k_log": 27, "ands": 37_000, "of": {"Attn_v1": "row"}, "prefill": 1, "decode": 0},
        "Head_v1{NVIS=64,FIRST=true}": {"k_log": 27, "ands": 36_000, "of": {"Attn_v1": "row"}, "prefill": 1, "decode": 0},
        "Head_v1{NVIS=8,FIRST=true}": {"k_log": 24, "ands": 4_500, "of": {"Attn_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=1}": {"k_log": 25, "ands": 3_000, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Read_v1{H=4}": {"k_log": 23, "ands": 1_800, "of": {"Read_v1": "row"}, "prefill": 1, "decode": 0},
    },
}


def test_an_unmeasured_type_takes_its_definitions_measured_rate_by_its_ands_else_the_fitted_rate_never_a_median():
    """A head block at other statics than its two measured siblings is priced at the dearer sibling's seconds an AND
    times its own ANDs: its ANDs ratio to that sibling, not a median of unlike statements. A type whose Definition no
    statement measured takes the rate fitted over all of them, their seconds over their ANDs, which is neither the
    old median a witness bit nor the median rate an AND."""

    statements = {"Head_v1{NVIS=64,FIRST=false}": (256, 27, 320.0, 2_400_000),
                  "Head_v1{NVIS=64,FIRST=true}": (256, 27, 350.0, 2_380_000),
                  "Gemm_v1{K=64,N=1}": (1024, 25, 16.0, 2_350_000)}
    timed = tuple(lean(P.definition_id(name), "r20260105-000000-0005", (s,), size, k_log=k, blocks=n, full=n)
                  for name, (n, k, s, size) in statements.items())
    table = PF.Flock((), (), None, lean=timed).table(P.Census.of(HEADS))
    by = {r.block: r.cost for r in table.rows}

    sibling = by["Head_v1{NVIS=8,FIRST=true}"]
    assert sibling.lean_source == sibling.proof_bytes_source == P.LEAN_FAMILY
    # the FIRST=true sibling is the dearer an AND (350 / 36000 over 320 / 37000), and its ANDs ratio prices the block
    assert sibling.lean_seconds == pytest.approx(350.0 / 256 * 4_500 / 36_000)
    # bytes by witness bits: the dearer sibling's a witness bit, at 2^24 of them
    assert sibling.proof_bytes == pytest.approx(2_400_000 / 2**35 * 2**24)

    read = by["Read_v1{H=4}"]
    assert read.lean_source == read.proof_bytes_source == P.LEAN_FITTED
    ands = 256 * 37_000 + 256 * 36_000 + 1024 * 3_000
    assert read.lean_seconds == pytest.approx((320.0 + 350.0 + 16.0) / ands * 1_800)
    assert read.proof_bytes == pytest.approx((2_400_000 + 2_380_000 + 2_350_000) / (3 * 2**35) * 2**23)
    old_median = 320.0 / 2**35 * 2**23
    median_per_and = 320.0 / (256 * 37_000) * 1_800
    assert read.lean_seconds != pytest.approx(old_median) and read.lean_seconds != pytest.approx(median_per_and)

    t = P.report(table, 1.0)["totals"]["lean_verifier_seconds_per_token"]
    assert {k: v["block_types"] for k, v in t["sources"].items()} == {P.LEAN_MEASURED: 3, P.LEAN_FAMILY: 1, P.LEAN_FITTED: 1}
    assert t["sources"][P.LEAN_FAMILY]["share"] == pytest.approx(sibling.lean_seconds / t["value"])
    assert t["fitted_rate"] == pytest.approx((320.0 + 350.0 + 16.0) / ands)


def test_each_block_types_lean_verifier_row_carries_the_mark_its_source_earns_and_the_record_it_rests_on():
    """The Lean verifier's table holds a row for every block type, each marked by where its seconds are from: measured
    only on a statement of its own Definition, which its row cites with the circuit file it was read from (the
    unrolled file, the templates, or neither named); modeled only on its Definition's measured statement, cited; assumed
    at the fitted rate, citing none; none with no AND gates. A mark that says more than its source fails here."""

    heads = {**HEADS, "definitions": {**HEADS["definitions"], "Bare_v1{N=1}": {"of": {"Bare_v1": "row"}, "prefill": 1,
                                                                                 "decode": 0}}}
    statements = {"Head_v1{NVIS=64,FIRST=false}": (256, 27, 320.0, "r20260105-000000-0001", "verity/flock-circuit"),
                  "Head_v1{NVIS=64,FIRST=true}": (256, 27, 350.0, "r20260105-000000-0002", "verity/flock-circuit/tmpl"),
                  "Gemm_v1{K=64,N=1}": (512, 25, 16.0, "r20260105-000000-0003", None)}
    timed = tuple(replace(lean(P.definition_id(name), run, (s,), 1000, k_log=k, blocks=n, full=1024 if k == 25 else n),
                          form=form) for name, (n, k, s, run, form) in statements.items())
    doc = P.report(PF.Flock((), (), None, lean=timed).table(P.Census.of(heads)), 1.0)
    rows = {r["block"]: r for r in doc["rows"]}
    assert {name: (r["lean_mark"], r["lean_record"], r["lean_form"]) for name, r in rows.items()} == {
        "Head_v1{NVIS=64,FIRST=false}": ("measured", "r20260105-000000-0001", "verity/flock-circuit"),
        "Head_v1{NVIS=64,FIRST=true}": ("measured", "r20260105-000000-0002", "verity/flock-circuit/tmpl"),
        "Gemm_v1{K=64,N=1}": ("measured", "r20260105-000000-0003", None),
        "Head_v1{NVIS=8,FIRST=true}": ("modeled", "r20260105-000000-0002", "verity/flock-circuit/tmpl"),
        "Read_v1{H=4}": ("assumed", None, None),
        "Bare_v1{N=1}": (None, None, None)}
    measured = {v.definition for v in timed}
    for r in doc["rows"]:
        assert r["lean_mark"] == P.LEAN_MARKS.get(r["lean_source"])
        assert (r["lean_mark"] == "measured") == (P.definition_id(r["block"]) in measured)
        if r["lean_mark"] == "measured":
            assert r["lean_definition"] == P.definition_id(r["block"])
        elif r["lean_mark"] == "modeled":
            assert r["lean_definition"] in measured and r["lean_definition"] != P.definition_id(r["block"])
        else:
            assert r["lean_record"] is r["lean_definition"] is None
    assert [(x["block"], x["form"]) for x in doc["totals"]["lean_verify"]] == [
        (name, form) for name, (*_, form) in statements.items()]

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 1.0,
            **doc, "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert f"\n{R.LEAN_ROWS}:\n\n| block type | Lean verifier ms per block | mark | record | rests on |\n" in text

    def line(name: str, mark: str, record: str, basis: str) -> str:
        seconds = rows[name]["lean_verifier_seconds"]
        return f"| `{name}` | {'—' if seconds is None else R._n(1e3 * seconds)} | {mark} | {record} | {basis} |"

    assert line("Head_v1{NVIS=64,FIRST=false}", "measured", "r20260105-000000-0001",
                "its own statement of 256 blocks (a full statement), measured from the unrolled circuit file") in text
    assert line("Head_v1{NVIS=64,FIRST=true}", "measured", "r20260105-000000-0002",
                "its own statement of 256 blocks (a full statement), measured from its circuit's templates") in text
    assert line("Gemm_v1{K=64,N=1}", "measured", "r20260105-000000-0003",
                "its own statement of 512 blocks (512/1024 of a full statement), its record names no statement id") in text
    assert line("Head_v1{NVIS=8,FIRST=true}", "modeled", "r20260105-000000-0002",
                f"`{P.definition_id('Head_v1{NVIS=64,FIRST=true}')}`'s statement of 256 blocks (a full statement), "
                "measured from its circuit's templates") in text
    assert line("Read_v1{H=4}", "assumed", "—", "the fitted rate") in text
    assert line("Bare_v1{N=1}", "none", "—", "—") in text
    rest = [r["lean_mark"] or "none" for r in doc["rows"][2:]]
    counts = ", ".join(f"{m} {rest.count(m)}" for m in ("measured", "modeled", "assumed", "none") if m in rest)
    assert f"\n| 4 more block types | | {counts} | | |\n" in R.render(page, rows=2)

    # the reader takes the statement id its record names, and None where it names none
    assert R.lean_verify(lean_record("r1") | {"statement": {"id": "verity/flock-circuit/tmpl"}}, "r1").form == (
        "verity/flock-circuit/tmpl")
    assert R.lean_verify(lean_record("r1"), "r1").form is None


def test_spot_takes_the_lean_verifiers_seconds_at_the_blocks_its_plan_samples():
    census = P.Census.of(CENSUS)
    flock = PF.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card",
                    lean=(lean(P.definition_id(GEMM64), "r20260105-000000-0005", (24.0,), 4000),))
    cells = [{"block": GEMM64, "type": "gemm64", "taken": 4.0}, {"block": GEMM32, "type": "", "taken": 2.0},
             {"block": NORM, "type": "norm", "taken": 0.8}]
    table = P.Spot(flock, plan(cells), P.Serving(0.5, 4000.0, "stated")).table(census)
    t = P.report(table, 1.0)["totals"]
    # 40, 20 and 8 blocks sampled over 800 tokens, at 3 s a block and at its 0.003 s an AND times 400 and 3000 ANDs;
    # bytes at its bytes a witness bit times 2^19 and 2^22
    assert t["lean_verifier_seconds_per_token"]["value"] == pytest.approx((40 * 3.0 + 20 * 1.2 + 8 * 9.0) / 800)
    assert t["proof_bytes_per_token"]["value"] == pytest.approx((40 * 500 + 20 * 250 + 8 * 2000) / 800)
    assert t["lean_verifier_seconds_per_token"]["status"] == "modeled" and t["lean_verify"][0]["block"] == GEMM64
    assert t["lean_verify"][0]["time_share"] == pytest.approx(40 * 1.0 / sum(
        n * r.cost.seconds for n, r in zip((40.0, 20.0, 8.0), table.rows)))
