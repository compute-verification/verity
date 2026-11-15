"""The Lean verifier's runs of record and the warm sessions' runs in the store: what each measures, and a run that
measures nothing refused by name. `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from prove_helpers import (
    REPO,
    RMSNORM,
    Store,
    _outside_a_research_run,  # noqa: F401  (autouse)
    fixture,
    lean_ran,
    lean_record,
    lean_shares,
    recorded,
)
from verity_bench import lean_verify as L
from verity_bench import prove as P
from verity_bench import prove_records as R


def test_a_lean_verifier_run_of_record_gives_both_figures_as_modeled_and_says_what_was_measured(tmp_path):
    """One block type's statement measured, on made-up figures: its row takes them, every other row (no other type
    of its Definition is in the census) takes the fitted rate, the seconds by its census ANDs and the bytes by its
    witness bits, and both lines say modeled, the measured type and its share of the proving time, the statement, the
    run, the host, each source's share and the basis. The proof server's verifier line stays."""

    s, arts = recorded(tmp_path)
    run = lean_ran(s, tmp_path, "r20261110-000000-0001", lean_record("r20261110-000000-0001"))
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    rows = {r["block"]: r for r in doc["rows"]}
    own, t = rows[RMSNORM], doc["totals"]
    ands = {b.name: b.ands for b in P.Census.of(fixture("census.json")).recut(fixture("partition.json")["value"]).blocks}
    per_and = 520.0 / (256 * ands[RMSNORM])
    assert (own["lean_verifier_seconds"], own["proof_bytes"], own["lean_source"]) == (520.0 / 256, 2_000_000 / 256, "measured")
    assert {(r["lean_source"], r["proof_bytes_source"]) for r in doc["rows"] if r is not own} == {("fitted", "fitted")}
    assert all(r["lean_verifier_seconds"] == pytest.approx(per_and * ands[r["block"]]) for r in doc["rows"])
    assert all(r["proof_bytes"] == pytest.approx(2_000_000 * 2 ** r["k_log"] / 2**35) for r in doc["rows"])
    seconds, size = t["lean_verifier_seconds_per_token"], t["proof_bytes_per_token"]
    assert seconds["value"] == pytest.approx(sum(r["per_token"] * per_and * ands[r["block"]] for r in doc["rows"]))
    assert size["value"] == pytest.approx(sum(r["per_token"] * 2_000_000 * 2 ** r["k_log"] / 2**35 for r in doc["rows"]))
    assert (seconds["status"], size["status"]) == ("modeled", "modeled")
    assert seconds["fitted_rate"] == pytest.approx(per_and) and seconds["unpriced_block_types"] == 0
    measured = own["per_token"] * 520.0 / 256 / seconds["value"]
    assert seconds["sources"] == lean_shares((1, measured), (0, 0.0), (149, 1 - measured))
    assert t["lean_verify"] == [{"block": RMSNORM, "time_share": own["time_share"], "k_log": 27, "blocks": 256, "full": 256,
                                 "seconds": [600.0, 500.0, 520.0], "statement_seconds": 520.0, "proof_bytes": 2_000_000,
                                 "run": run, "host": "cpu-node-0 (64 cores, A CPU 9000)", "label": L.COVER_LABEL,
                                 "form": None}]
    assert doc["inputs"][-1] == {"slot": R.LEAN, "record": "prove-glm53-lean-verify", "targets": [run],
                                 "what": "the Lean verifier (its `verify`) on one statement of each of 1 block "
                                         "type(s); every other type's Lean verifier seconds and proof bytes are priced "
                                         "from them, its Definition's at its size, else at the rate fitted over them"}
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
        f"process, the statement read and set up in it), in run {run} on cpu-node-0 (64 cores, A CPU 9000). Of this "
        f"figure: own 1 block type(s), {measured:.1%}; fitted 149 block type(s), {1 - measured:.1%}, each at "
        f"{R._n(per_and)} seconds an AND gate (every measured statement's seconds over their AND gates) times its AND "
        f"gates: {P.LEAN_SECONDS_BASIS}.")
    bytes_measured = size["sources"]["measured"]["share"]
    assert bytes_measured == pytest.approx(own["per_token"] * 2_000_000 / 256 / size["value"])
    assert lines["- proof bytes"] == (
        f"- proof bytes per token: {R._n(size['value'])}, modeled, not measured. Measured: one block type, `{RMSNORM}` "
        f"({share} of the proving time), in its statement of 256 blocks of 2^27 witness bits each (a full statement), "
        f"whose proof files are 2.000e+06 bytes in all (run {run}). Of this figure: own 1 block type(s), "
        f"{bytes_measured:.1%}; fitted 149 block type(s), {size['sources']['fitted']['share']:.1%}, each at "
        f"{R._n(2_000_000 / 2**35)} bytes a witness bit (every measured statement's bytes over their 2^k_log witness "
        f"bits) times its 2^k_log witness bits: {P.PROOF_BYTES_BASIS}.")
    assert "per token: not measured" not in text and "Scaled:" not in text

    # at the sampling plan: the same record at the blocks the plan samples, the type's share the plan's
    spot = R.build("spot", "glm53", store=s.fresh("second"), root=REPO)
    rows = {r["block"]: r for r in spot["rows"]}
    t = spot["totals"]
    assert t["lean_verifier_seconds_per_token"]["value"] == pytest.approx(
        sum(r["per_token"] * per_and * ands[r["block"]] for r in spot["rows"]))
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


def warm_statement(word: str | None, n: int, k_log: int, seconds: float, *, device: str = "gpu", timed: int = 4,
                   accepted: tuple[bool, ...] = (True,) * 5) -> dict:
    """A statement of `prove_time.py`'s summary as a `stage_prove.sh` group writes it, cut to what the table reads."""

    return {"device": device, "word": word, "n": n, "k_log": k_log, "accepted": list(accepted), "timed": timed,
            "prove_total_s": seconds}


def warm_ran(s: Store, tmp: Path, run: str, statements: dict | None, config: str = "warm=1 runs=4") -> str:
    """`run` in the store with one card's group (`out/gpu0/out/`) holding `statements` (None: a run that wrote none), a
    warm-session run of record."""

    files = tmp / "warm-files" / run
    group = files / "out" / "gpu0" / "out"
    group.mkdir(parents=True)
    (files / "out" / "steps.txt").write_text("ran")
    if statements is not None:
        (group / "summary.json").write_text(json.dumps({"statements": statements}))
        (group / "config.txt").write_text(f"device=gpu units=census:X@full ns=16 64 statements= {config}\n")
    s.ran(run, files)
    return s.record(run, R.WARMED)


def test_a_warm_session_run_prices_its_own_types_under_its_own_label_with_its_warm_and_timed_sessions(tmp_path):
    """A type with no steady-state row takes its own statement proved on one card from warm sessions: the fullest
    statement's median GPU-seconds over its blocks, marked `warm` and named with the run's WARM and RUNS, never the
    steady state's label. A type measured at steady state keeps that row, the totals stay modeled, and a statement or
    run that measures nothing is listed with why."""

    s, _ = recorded(tmp_path)
    before = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    off = next(r for r in before["rows"] if r["source"] in (P.FAMILY, P.SINGLE) and r["k_log"])
    on = next(r for r in before["rows"] if r["source"] == P.STEADY)
    full = 2**35 >> off["k_log"]
    run = warm_ran(s, tmp_path, "r20261115-000000-0001", {
        "gpu/off-full": warm_statement(off["block"], full, off["k_log"], 100.0),
        "gpu/off-half": warm_statement(off["block"], max(1, full // 2), off["k_log"], 1.0),
        "gpu/on": warm_statement(on["block"], 2**35 >> on["k_log"], on["k_log"], 1e-6),
        "cpu/on-cpu": warm_statement(on["block"], 1, on["k_log"], 1.0, device="cpu"),
        "gpu/short": warm_statement(off["block"], full, off["k_log"], 1.0, timed=3, accepted=(True,) * 4),
        "gpu/refused": warm_statement(off["block"], full, off["k_log"], 1.0, accepted=(True, True, False, True, True)),
        "gpu/nameless": warm_statement(None, full, off["k_log"], 1.0),
        "gpu/stray": warm_statement("Stray_v1{N=1}", 1, 27, 1.0)})
    empty = warm_ran(s, tmp_path, "r20261115-000000-0002", None)
    doc = R.build("flock", "glm53", store=s.fresh("second"), root=REPO)
    rows = {r["block"]: r for r in doc["rows"]}

    got = rows[off["block"]]
    assert (got["source"], got["record"], got["statement_blocks"]) == (P.WARM, run, full)
    assert got["seconds"] == pytest.approx(100.0 / full)
    assert got["note"].startswith(f"`off-full`, {full} of a full statement's {full} blocks, on one card: after 1 warm "
                                  "session, the median of 4 timed ones (WARM=1, RUNS=4)")
    assert rows[on["block"]]["source"] == P.STEADY and rows[on["block"]]["seconds"] == on["seconds"]
    by = {x["source"]: x["block_types"] for x in doc["sources"]}
    assert by[P.WARM] == 1 and by[P.STEADY] == {x["source"]: x["block_types"] for x in before["sources"]}[P.STEADY]
    assert doc["totals"]["status"] == "modeled"
    assert doc["totals"]["steady_time_share"] == pytest.approx(sum(r["time_share"] for r in doc["rows"]
                                                                   if r["source"] == P.STEADY))
    assert doc["inputs"][-1] == {"slot": R.WARMED, "record": "prove-glm53-flock-warm", "targets": [run],
                                 "what": "2 block type(s) proved on one card from warm sessions"}
    assert [(x["target"], x["why"]) for x in doc["refused"]] == [
        (empty, "holds no `out/gpu*/out/summary.json` with its `out/gpu*/out/config.txt`, which `stage_prove.sh` writes"),
        (run, "`on-cpu`: not proved on a card"),
        (run, "`nameless`: names no census type (`word`)"),
        (run, "`refused`: not every session was accepted ([True, True, False, True, True])"),
        (run, "`short`: 3 timed of 4 sessions, not RUNS=4 after WARM=1"),
        (run, "`stray`: its block type, `Stray_v1{N=1}`, is no block type of the census under its cut")]

    line = next(x for x in R.render(doc).splitlines() if x.startswith(f"| `{off['block']}`"))
    assert f"| {P.SOURCES[P.WARM]} |" in line and "(WARM=1, RUNS=4)" in line and P.SOURCES[P.STEADY] not in line

    assert R.warm_rows({"statements": {}}, "device=gpu units=x", run) == [
        "its `out/gpu*/out/config.txt` states no `warm=W runs=R`, or its `out/gpu*/out/summary.json` no `statements`"]
