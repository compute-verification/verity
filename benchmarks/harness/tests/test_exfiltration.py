"""`verity bench exfiltration` with no GPU and no node: the rule that reads a scope's rows off at a share of total
compute, on small rows; then the store side on a real store with a directory remote that holds the planner's rows as
`fixtures/exfiltration/` holds them (two runs' rows, cut to the fields the command reads and to the case it takes plus
three it must pass over; `records.json` names the runs they were cut from): the figures a hand computation gave on
those rows, the network certifier's part beside them at the planner's requests, every input's mark, each refusal by
name, and the command."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from verity_bench import exfiltration as E

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "exfiltration"
YEAR, OTHERS = json.loads((FIXTURES / "records.json").read_text())["compute-audit"]

# ---- the rule, on small rows -------------------------------------------------------------------------------------------------


def row(beta: float, total: float, bits: float, **more) -> dict:
    return {"scope": "world year", "step": "s", "mix": "m", "b10": beta, "total": total, "bound_bits": bits,
            "target_bits": 50.0, "weights_bits": 100.0, "delta_seen": 0.01, "S": 8.0, "gpu_seconds": 4.0, "serve": 0.01,
            "prove": total - 0.01, **more}


#: the total is 0.02 + 2β and the bound 1/β, so at 10% β is 0.04 and the bound 25 bits, counted twice at λ = 2
TOY = [row(0.02, 0.06, 50.0), row(0.03, 0.08, 100 / 3), row(0.05, 0.12, 20.0)]


def test_a_scope_is_read_at_the_budget_where_its_total_is_the_share():
    at = E.at_share("world year", "r1", TOY[::-1], 0.10)
    assert at["beta"] == pytest.approx(0.04) and at["inside"] and at["residual"] < 1e-15
    assert at["bound"]["bits"] == pytest.approx(25.0) and at["lambda"] == 2.0 and at["bits"] == pytest.approx(50.0)
    assert at["GB"] == pytest.approx(50.0 / 8e9) and (at["compute"], at["period"]) == ("world", "year")
    assert at["parts"] == pytest.approx({"serve": 0.01, "prove": 0.09})
    assert (at["delta"], at["output_bits"], at["gpu_seconds"], at["weights_bits"]) == (0.01, 8.0, 4.0, 100.0)
    assert at["pinned"] == [0.02, 0.03, 0.05] and at["priced"] == {}
    # at a pinned budget it is that row's bound, and past the last the line through the nearest two, said to be so
    assert E.at_share("world year", "r1", TOY, 0.08)["bound"]["bits"] == pytest.approx(100 / 3)
    past = E.at_share("world year", "r1", TOY, 0.14)
    assert past["beta"] == pytest.approx(0.06) and not past["inside"] and past["bound"]["bits"] == pytest.approx(100 / 6)

    with pytest.raises(ValueError, match="r1's world year rows pin 1 audit budgets, 1 distinct"):
        E.at_share("world year", "r1", TOY[:1], 0.10)
    with pytest.raises(ValueError, match="pin 3 audit budgets, 2 distinct"):
        E.at_share("world year", "r1", [*TOY[:2], TOY[0]], 0.10)
    with pytest.raises(ValueError, match="rows differ in delta, gpu_seconds, which one plan holds fixed"):
        E.at_share("world year", "r1", [*TOY[:2], TOY[2] | {"delta_seen": 0.02, "gpu_seconds": 5.0}], 0.10)
    with pytest.raises(ValueError, match="spend 2.00% of the compute before any audit: no audit budget brings the total to 1.00%"):
        E.at_share("world year", "r1", TOY, 0.01)


def test_proving_time_is_summed_by_how_its_block_families_are_priced():
    split = {"families": {"A_v1": {"priced": ["measured"], "share": 0.5}, "B_v1": {"priced": ["computed"], "share": 0.125},
                          "C_v1": {"priced": ["computed", "measured"], "share": 0.25},
                          "D_v1": {"priced": ["measured"], "share": 0.125}}}
    at = E.at_share("world year", "r1", [r | {"split": split} for r in TOY], 0.10)
    assert at["priced"] == {"measured": {"share": 0.625, "families": ["A_v1", "D_v1"]},
                            "measured and computed": {"share": 0.25, "families": ["C_v1"]},
                            "computed": {"share": 0.125, "families": ["B_v1"]}}


# ---- the store side ----------------------------------------------------------------------------------------------------------


class Store:
    """A machine holding the planner's runs: a local store with a directory remote, each run's attempt and files on it."""

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

    def ran(self, run: str, rows: list[dict] | None, *, name: str = E.ROWS_FILE) -> str:
        """A run's attempt on the remote, its run record holding `rows` as the planner's rows (None: it kept no files)."""

        from research.store import canon

        record = None
        if rows is not None:
            path = self.tmp / "run-files" / f"{run}-{self._ts}" / name
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            record = self.store.put_artifact("run-record/v1", {}, tree=path.parents[len(Path(name).parts) - 1])
            self.store.push(record)
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "bash", "version": None},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T00:00:00Z"}, "cost": None,
            "status": {"state": "done", "rc": 0, "validation": None},
            "outputs": {"result": None, "run_record": record}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))
        return run

    def label(self, target: str, key: str = E.RUN_OF_RECORD, value: str = E.RECORD) -> str:
        self._ts += 1
        self.store.add_label(target, key, value, by="benchmarks", ts=f"2026-01-01T00:{self._ts // 60:02d}:{self._ts % 60:02d}Z")
        self.store.push_labels([target])
        return target


def rows(run: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / run / "rows.jsonl").read_text().splitlines()]


def mine(run: str) -> list[dict]:
    """A fixture run's rows of the case the command takes."""

    return [r for r in rows(run) if (r["world"], r["row"]) == (E.WORLD, E.ROW) and r["step"].startswith(E.STEP)]


def recorded(tmp: Path) -> Store:
    """The fixtures' two runs in a store, each labelled the record: the world's year, and the other five scopes."""

    s = Store(tmp)
    for run in (YEAR, OTHERS):
        s.label(s.ran(run, rows(run)))
    return s


def test_the_hand_computation_is_reproduced_from_the_rows_it_read(tmp_path):
    """The figures a hand computation gave on these rows, to the digits it kept: the world year's audit budget at 10%, its
    bound at λ = 1 and at the rows' λ, each part of the 10%, the figure with the network check's own proving taken off the
    budget, and the other five scopes'. A fresh machine gets them from the store's remote alone."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert (doc["schema"], doc["unit"], doc["status"], doc["runs"], doc["refused"]) == (E.SCHEMA, "GB", "modeled", [OTHERS, YEAR], [])
    by = {p["scope"]: p for p in doc["compute_audit"]}
    year = by["world year"]
    assert doc["value"] == year["GB"] == pytest.approx(8455.87878178061, rel=1e-12)
    assert year["beta"] == pytest.approx(0.05542142320531889, rel=1e-12) and year["inside"] and year["residual"] < 1e-9
    assert year["bound"]["GB"] == pytest.approx(8053.217887410105, rel=1e-12) and year["lambda"] == pytest.approx(1.05)
    assert (year["run"], year["mix"], year["delta"], year["output_bits"]) == (YEAR, "B128", 0.01, 28000.0)
    assert year["step"].startswith(E.STEP) and year["pinned"] == [0.035, 0.045, 0.055, 0.065]
    assert {k: round(100 * v, 3) for k, v in year["parts"].items()} == {
        "serve": 4.339, "wait": 2.293, "prove": 5.369, "replay": 0.173, "pool": 0.07, "edge": 0.044, "compile": 0.006}
    assert sum(v for k, v in year["parts"].items() if k != "wait") == pytest.approx(0.10, abs=1e-5)
    assert {k: round(v["share"], 4) for k, v in year["priced"].items()} == {
        "measured": 0.9857, "measured and computed": 0.0135, "computed": 0.0008}
    assert sum(v["share"] for v in year["priced"].values()) == pytest.approx(1.0)
    assert year["priced"]["measured"]["families"][:2] == ["MoeExpertFp8CoordinateChunk_v1", "MoeExpertFp8GemmW_v1"]
    assert {scope: round(p["GB"], 1) for scope, p in by.items() if scope != "world year"} == {
        "world week": 8150.7, "world month": 8263.0, "datacenter week": 7510.9, "datacenter month": 7624.4,
        "datacenter year": 7819.2}
    assert {p["run"] for scope, p in by.items() if scope != "world year"} == {OTHERS}
    net = doc["network_check"]
    assert net["share"] == 0.003 and net["GB"] == pytest.approx(8939.28531191602, rel=1e-12) and net["inside"]
    assert net["beta"] == pytest.approx(0.05242440579315158, rel=1e-12)
    assert E.build(share=0.097, network_check=0, store=recorded(tmp_path / "again").fresh(), root=REPO)["value"] == net["GB"]


def test_the_network_certifiers_part_is_beside_it_at_the_planners_requests(tmp_path):
    """Both parts count the plan's requests, which are fewer than the census's, over the same GPU-seconds; the certifier's
    rows are its own calculator's at those requests; and the two are summed under each of its rows by the form: its term,
    plus the compute audit's part or the entries where the row charges them and they are fewer."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert [(p["compute"], p["period"]) for p in doc["periods"]] == [
        (c, p) for c in ("datacenter", "world") for p in ("week", "month", "year")]
    year = doc["periods"][-1]
    assert year["requests"] == pytest.approx(1.7784e16, rel=1e-4) and year["gpu_seconds"] == pytest.approx(1.182473856e16)
    planned = E.plan(REPO)
    assert year["requests"] == pytest.approx(planned["scopes"]["world year"]["requests"], rel=1e-12)
    assert f"sweep.cases()['{E.PLAN}']" in planned["source"] and planned["request_tokens"]["value"] == 3320.5

    lb = E._load(REPO, "network_traces", "leak_bounds")
    assert planned["scopes"]["world year"]["requests"] < lb.inputs()["world_requests"].value        # the census's request
    theirs = lb.document(lb.inputs(world_requests=year["requests"]), only=("world",), periods=("year",))["rows"]
    assert [{k: v for k, v in r.items() if k != "requests"} for r in doc["certifier"]["rows"]] == [
        {k: v for k, v in r.items() if k != "requests"} for r in theirs]
    scan = doc["certifier"]["rows"][0]
    assert scan["row"] == "scan" and scan["term"]["1"]["bits"] == pytest.approx(26 * year["requests"])

    audit, together = year["compute_audit"], year["together"]
    assert audit["GB"] == doc["value"] and audit["bits_a_request"] == pytest.approx(audit["bits"] / year["requests"])
    assert audit["of_weights"] == pytest.approx(doc["value"] / doc["weights"]["GB"]) == pytest.approx(22.449, abs=5e-4)
    assert list(together) == list(E.TOGETHER) == ["scan", "N2", "all three", "design"]
    assert together["scan"]["certifier_GB"] == pytest.approx(57_798_028, abs=1)
    assert together["scan"]["GB"] == pytest.approx(together["scan"]["certifier_GB"] + audit["GB"])
    assert together["N2"]["certifier_GB"] == pytest.approx(10_104, abs=1)
    assert together["N2"]["GB"] == pytest.approx(together["N2"]["certifier_GB"] + audit["GB"])
    assert (together["all three"]["certifier_GB"], together["design"]["certifier_GB"]) == (0.0, 0.0)
    assert together["all three"]["GB"] == together["design"]["GB"] == pytest.approx(audit["GB"])
    # the entries stand in for the compute audit's part where the scan charges them and they are fewer
    assert scan["entries"]["bits"] > audit["bits"]
    week = doc["periods"][0]
    assert week["compute_audit"]["GB"] == pytest.approx(7510.9, abs=0.05) and week["compute_audit"]["run"] == OTHERS
    assert week["requests"] == pytest.approx(planned["scopes"]["datacenter week"]["requests"], rel=1e-12)


def test_every_input_is_marked_measured_modeled_or_assumed_with_its_source(tmp_path):
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    by = {i["input"]: i for i in doc["inputs"]}
    assert all(i["mark"] in E.MARKS and i["source"] and i["part"] in ("compute audit", "network certifier", "both")
               for i in doc["inputs"])
    assert {k: (by[k]["part"], by[k]["mark"], by[k]["value"]) for k in (
        "planner_rows", "compute_share", "delta", "output_bits", "network_check_share", "advice")} == {
        "planner_rows": ("compute audit", "modeled", 24), "compute_share": ("compute audit", "assumed", 0.1),
        "delta": ("compute audit", "assumed", 0.01), "output_bits": ("compute audit", "assumed", 28000.0),
        "network_check_share": ("compute audit", "assumed", 0.003), "advice": ("compute audit", "assumed", 0)}
    assert by["lambda"]["mark"] == "assumed" and by["lambda"]["value"] == pytest.approx(1.05)
    assert f"`{E.RECORD}` ({OTHERS}, {YEAR})" in by["planner_rows"]["source"] and "6 scopes" in by["planner_rows"]["source"]
    assert (by["proving priced measured"]["mark"], by["proving priced computed"]["mark"],
            by["proving priced measured and computed"]["mark"]) == ("measured", "modeled", "modeled")
    # the world's requests are the planner's, marked so, where the certifier's command takes them as a given option
    requests = by["world_requests"]
    assert (requests["part"], requests["mark"]) == ("both", "modeled") and "benchmarks/spot_figures/breakeven" in requests["source"]
    assert (by["weights"]["part"], by["weights"]["mark"]) == ("both", "measured")
    assert (by["response_ids"]["part"], by["response_ids"]["mark"]) == ("network certifier", "assumed")
    assert (by["mfu_serve"]["part"], by["mfu_serve"]["mark"], by["mfu_serve"]["value"]) == ("both", "assumed", 0.4)
    # a request is the plan's, not the census's the certifier's calculator would count the requests from
    assert (by["request_tokens"]["value"], by["request_tokens"]["mark"]) == (3320.5, "modeled")
    assert by["token_macs"]["mark"] == "modeled" and by["token_macs"]["value"] == pytest.approx(7.925e10, rel=1e-3)
    assert "128k serve plan" in by["request_tokens"]["source"] and "reprice.C343" in by["token_macs"]["source"]
    world = by["world_machines"]["value"] * by["machine_gpus"]["value"] * 365 * 86400
    assert requests["value"] == pytest.approx(world * by["mfu_serve"]["value"] * by["peak_flops"]["value"]
                                              / (by["request_tokens"]["value"] * 2 * by["token_macs"]["value"]), rel=1e-9)

    text = E.render(doc)
    assert text.startswith("# verity bench exfiltration: GLM-5.3's exfiltration capacity over a monitoring period, at 10% of "
                           "total compute\n")
    assert "- **the compute audit's part: 8,456 GB** (modeled), 22.45 copies of the weights (GLM-5.3 in FP4, 376.7 GB)" in text
    assert "β = 5.542%: its bound, 8,053 GB, times λ = 1.05." in text
    assert "- **the network certifier's part: 57,798,028 GB** (modeled, on an assumed class) as the scan's check charges it today" in text
    assert "8,456 GB with the lengths held, the timing taken once and the entries charged through the compute audit" in text
    assert "the audit budget is β = 5.242% and the compute audit's part 8,939 GB." in text
    assert f"- compute-audit: {OTHERS}, {YEAR} (`{E.RECORD}`): the planner's rows, 24 of row `(i) honest`" in text
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in E.MARKS}
    assert f"of {len(doc['inputs'])} inputs, {marks['measured']} are measured, {marks['modeled']} modeled and {marks['assumed']} assumed" in text
    assert "| world year | " + YEAR + " | 5.542% | 8,053 | 8,456 | 22.45 | 4.339% | 2.293% | 5.369% |" in text
    assert "| world | year | 1.182e+16 | 1.778e+16 | 8,456 | 0.003804 | 57,798,028 | 10,104 | 0 | 0 | 57,806,484 | 18,560 | 8,456 | 8,456 |" in text
    assert "Refused" not in text and text.count("\n| datacenter | ") == 3


def test_an_unlabelled_superseded_or_ambiguous_record_is_refused_by_name(tmp_path):
    s = Store(tmp_path)
    year, others = s.ran(YEAR, rows(YEAR)), s.ran(OTHERS, rows(OTHERS))

    def build(**kw) -> dict:
        return E.build(store=s.store, root=REPO, **kw)

    with pytest.raises(ValueError, match=f"no planner rows of record: nothing in the store carries `run_of_record={E.RECORD}`"
                                         f".*research data label RUN run_of_record {E.RECORD} --by LANE"):
        build()
    s.label(others)
    with pytest.raises(ValueError, match=f"no run of record holds the world year's rows \\({OTHERS} hold datacenter month, "):
        build()
    s.label(year)
    assert build()["value"] == pytest.approx(8455.87878178061)

    # a second run holding the world's year: refused until one is superseded, then the newer one is read
    newer = s.label(s.ran("r20261111-000000-0001", [r | {"bound_bits": 2 * r["bound_bits"]} for r in mine(YEAR)]))
    with pytest.raises(ValueError, match=f"{newer} and {YEAR} both hold the world year's rows and neither is superseded: label "
                                         "the one the other replaces .*superseded_by NEWER"):
        build()
    s.label(YEAR, E.SUPERSEDED_BY, newer)
    doc = build()
    assert doc["value"] == pytest.approx(2 * 8455.87878178061) and doc["runs"] == [newer, OTHERS]
    assert doc["refused"] == [{"target": YEAR, "record": E.RECORD, "why": f"superseded by {newer}"}]
    assert f"- {YEAR} (`{E.RECORD}`): superseded by {newer}" in E.render(doc)

    # runs of record that can't be read are left out and listed
    s.label(s.ran("r20261111-000000-0002", None))
    s.label(s.ran("r20261111-000000-0003", [{}], name="out/other.jsonl"))
    s.label(s.ran("r20261111-000000-0004", [r | {"error": "timed out"} for r in mine(YEAR)]))
    s.label("r20261111-000000-0005")
    assert {x["target"][-1]: x["why"] for x in build()["refused"] if x["target"] != YEAR} == {
        "2": "kept no run files", "3": "holds no planner rows (`out/rows.jsonl`)",
        "4": "holds no row `(i) honest` at a step `+ the verifier as measured...` of the world `ours`",
        "5": "not in the store or on its remote"}

    # another row of the plan, a step two steps begin with, and a row no run holds
    both = recorded(tmp_path / "both").store
    other = E.build(store=both, root=REPO, row="(iii) honest")
    assert other["runs"] == [YEAR] and other["refused"][0]["target"] == OTHERS and other["value"] < 8455
    with pytest.raises(ValueError, match=f"`\\+ ` begins 2 steps of {YEAR}'s row `\\(i\\) honest`: give more of the one meant"):
        E.build(store=both, root=REPO, step="+ ")
    with pytest.raises(ValueError, match="no run of record holds the planner's rows: .*holds no row `\\(iv\\)`"):
        build(row="(iv)")
    with pytest.raises(ValueError, match="the share of total compute is in \\(0, 1\\) and the network check's own below it"):
        build(share=0.1, network_check=0.1)


def test_the_two_parts_must_cover_the_same_compute_and_weights(tmp_path):
    def build(name: str, **change) -> dict:
        s = Store(tmp_path / name)
        s.label(s.ran(YEAR, [r | change for r in mine(YEAR)]))
        return E.build(store=s.store, root=REPO)

    assert build("same")["value"] == pytest.approx(8455.87878178061)
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 1e\\+16 GPU-seconds on the "
                                         "compute audit's side and 1.18247e\\+16 on the network certifier's"):
        build("compute", gpu_seconds=1e16)
    with pytest.raises(ValueError, match="the two parts are shares of different weights: the rows of record hold 3,000,000,000,000"):
        build("weights", weights_bits=3_000_000_000_000, target_bits=3_000_000_000_000 / 1.05)


# ---- the command -------------------------------------------------------------------------------------------------------------


def test_verity_bench_exfiltration_is_the_commands_runner(tmp_path, monkeypatch, capsys):
    """`verity bench exfiltration` runs benchmarks/exfiltration/run.py, which prints the figure from the store alone."""

    s = recorded(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))

    def run(*argv: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(REPO / "benchmarks" / "exfiltration" / "run.py"), run_name="__main__")
        out = capsys.readouterr()
        return done.value.code, out.out, out.err

    code, out, _ = run()
    assert code == 0 and out.startswith("# verity bench exfiltration: GLM-5.3's exfiltration capacity")
    assert "- **the compute audit's part: 8,456 GB** (modeled)" in out and "## Inputs" in out
    code, out, _ = run("--json", "--share", "0.097", "--network-check-share", "0")
    doc = json.loads(out)
    assert code == 0 and doc["schema"] == E.SCHEMA and doc["value"] == pytest.approx(8939.28531191602)
    code, out, err = run("--row", "(iv)")
    assert code == 2 and out == "" and "verity bench exfiltration: no run of record holds the planner's rows" in err
    # this machine's catalog alone holds no label
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "alone"))
    code, _, err = run("--local-only")
    assert code == 2 and "no planner rows of record: nothing in the store carries" in err
