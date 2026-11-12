"""`verity bench exfiltration` with no GPU and no node: the rule that reads a scope's rows off at a share of total
compute and the rule that reads how its proving is priced from the row's plan, on small rows; then the store side on a
real store with a directory remote that holds the planner's runs as `fixtures/exfiltration/` holds them (`records.json`
names the runs they were cut from: one run's rows, cut to the fields the command reads and to the case it takes plus
two it must pass over; the run of that case's plan over the world's year, its plan cut to each cell's block and
share; and a run whose rows hold their own plans, cut to two scopes): the figures a hand computation gave on those
rows. The runs' case is row (i); here its rows stand for the command's row, the plan of record's, under that row's name
(`as_row`), and row (i) is made up from them at `ASSUMED` times their bound (`as_assumed`), printed only under its
label, with what prices each row's proving or that its rows don't record it. Then the network certifier's part beside
them at the requests its calculator counts by default, the sampling plan's; its figures of record at two release schedules, held to its own document; what rests on the set-time card printed
only under the calculator's label; the output count beside the sums that take it at 0; every figure's error, summed
over everything it covers, at the scans' δ the term states, and no amount where it is 1 or more, on term documents
with made-up errors; every input's mark; each refusal by name; and the command."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import pytest

from verity_bench import exfiltration as E

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "exfiltration"
RECORDS = json.loads((FIXTURES / "records.json").read_text())["compute-audit"]
ROWS, PLAN, PLANNED = RECORDS["rows"], RECORDS["plan"], RECORDS["planned"]
#: the world year's figure a hand computation gave on the fixture's rows, in GB: the command's row's here
YEAR_GB = 14239.837095861332
#: the row the fixture's runs were cut from, and the bound of the row (i) the tests make up from its rows over theirs
CUT_ROW = "(i) honest"
ASSUMED = 1.25


def as_row(x: dict) -> dict:
    """A row or a plan of the fixture's case as the command's row."""

    return x | {"row": E.ROW} if x.get("row") == CUT_ROW else x


def as_assumed(x: dict) -> dict:
    """A row of the fixture's case as a made-up row (i), its bound `ASSUMED` times theirs."""

    return x | {"row": E.ASSUMED_ROW, "bound_bits": ASSUMED * x["bound_bits"]}


def term():
    """The network certifier's calculator, as the command loads it."""

    return E._load(REPO, "network_traces", "leak_bounds")


def delta(lb, response_ids: int = E.RESPONSE_IDS) -> Fraction:
    """The scans' δ the command gives the calculator where it is given none: the one at which the world's year, the
    row it prints that sums the most execution phases, carries `PERIOD_ERROR`."""

    year = lb.document(lb.inputs(response_ids=response_ids), only=("world",), periods=("year",))["rows"][0]
    return E.PERIOD_ERROR / Fraction(year["phases"])


def printed(text: str) -> list[tuple[str, str]]:
    """Everything a print shows, each with what it is printed under: a line of prose under itself, and a table's cell
    under its column's header and its row's name."""

    out, head = [], None
    for line in text.splitlines():
        if not line.startswith("|"):
            out, head = [*out, (line, line)], None
            continue
        row = [c.strip() for c in line.strip("|").split("|")]
        if head is None:
            head = row
        elif set(row) != {"---"}:
            out += [(f"{h} | {row[0]}", c) for h, c in zip(head, row, strict=True)]
    return out

# ---- the rules, on small rows ------------------------------------------------------------------------------------------------


def row(beta: float, total: float, bits: float, **more) -> dict:
    return {"scope": "world year", "step": "s", "mix": "m", "b10": beta, "total": total, "bound_bits": bits,
            "target_bits": 50.0, "weights_bits": 100.0, "delta_seen": 0.01, "S": 8.0, "gpu_seconds": 4.0, "serve": 0.01,
            "prove": total - 0.01, "q": 0.5, "replay_sample": 3.0, "requests": 7.0, **more}


#: the total is 0.02 + 2β and the bound 1/β, so at 10% β is 0.04 and the bound 25 bits, counted twice at λ = 2; the
#: rows are priced over 3 / 0.5 = 6 requests, where their `requests` field says 7
TOY = [row(0.02, 0.06, 50.0), row(0.03, 0.08, 100 / 3), row(0.05, 0.12, 20.0)]


def test_a_scope_is_read_at_the_budget_where_its_total_is_the_share():
    at = E.at_share("world year", "r1", TOY[::-1], 0.10)
    assert at["beta"] == pytest.approx(0.04) and at["inside"] and at["residual"] < 1e-15
    assert at["bound"]["bits"] == pytest.approx(25.0) and at["lambda"] == 2.0 and at["bits"] == pytest.approx(50.0)
    assert at["GB"] == pytest.approx(50.0 / 8e9) and (at["compute"], at["period"]) == ("world", "year")
    assert at["parts"] == pytest.approx({"serve": 0.01, "prove": 0.09})
    assert (at["delta"], at["output_bits"], at["gpu_seconds"], at["weights_bits"]) == (0.01, 8.0, 4.0, 100.0)
    assert at["pinned"] == [0.02, 0.03, 0.05] and at["priced"] is None
    # the requests are the count the rows are priced over, not the field that names another request's
    assert at["requests"] == 6.0
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
    with pytest.raises(ValueError, match="rows differ in requests, which one plan holds fixed"):
        E.at_share("world year", "r1", [*TOY[:2], TOY[2] | {"q": 0.25}], 0.10)
    with pytest.raises(ValueError, match="rows don't say what requests they are priced over \\(`replay_sample / q`\\)"):
        E.at_share("world year", "r1", [{k: v for k, v in r.items() if k != "q"} for r in TOY], 0.10)
    with pytest.raises(ValueError, match="r1's world year rows are 2 variants of the row \\(h1lin, h8lin\\), each on its own "
                                         "premises: a run of record holds one"):
        E.at_share("world year", "r1", [r | {"h8": h} for r in TOY for h in ("h8lin", "h1lin")], 0.10)
    with pytest.raises(ValueError, match="spend 2.00% of the compute before any audit: no audit budget brings the total to 1.00%"):
        E.at_share("world year", "r1", TOY, 0.01)


def test_proving_time_is_priced_by_the_rows_plan_not_by_the_rows_own_shares():
    """The shares are the plan's cells', summed by block family; how a family is priced is the rows' account, whose own
    shares, written before the row's levers multiply its blocks, are not read. A plan is held to its rows."""

    split = {"families": {"A_v1": {"priced": ["measured"], "share": 0.97}, "B_v1": {"priced": ["computed"], "share": 0.01},
                          "C_v1": {"priced": ["computed", "measured"], "share": 0.01},
                          "D_v1": {"priced": ["measured"], "share": 0.01}}}
    rows = [r | {"split": split} for r in TOY]
    plan = {"scope": "world year", "beta": 0.04, "q": 0.25, "replay_sample": 1.5, "bound_bits": 25.0,
            "cells": [{"block": "B_v1{K=1}", "share": 0.5}, {"block": "A_v1{K=1}", "share": 0.125},
                      {"block": "C_v1{K=2}", "share": 0.125}, {"block": "B_v1{K=2}", "share": 0.0625},
                      {"block": "D_v1", "share": 0.0625}, {"block": "E_v1{K=1}", "share": 0.0625}],
            "left_out": [{"block": "Copy[8 bits]", "share": 0.0625}]}
    priced = E.at_share("world year", "r1", rows, 0.10, ("r2", plan))["priced"]
    assert (priced["of"], priced["run"], priced["beta"]) == ("plan", "r2", 0.04)
    assert priced["ways"] == {
        "computed": {"share": 0.5625, "families": {"B_v1": 0.5625}},
        "measured": {"share": 0.1875, "families": {"A_v1": 0.125, "D_v1": 0.0625}},
        "measured and computed": {"share": 0.125, "families": {"C_v1": 0.125}},
        E.UNSTATED: {"share": 0.125, "families": {"E_v1": 0.0625, "Copy[8 bits]": 0.0625}}}
    assert sum(w["share"] for w in priced["ways"].values()) == 1.0
    assert E.at_share("world year", "r1", rows, 0.10)["priced"] is None

    with pytest.raises(ValueError, match="r2's plan of the world year is priced over 12 requests and r1's rows over 6: it is "
                                         "not these rows' plan"):
        E.at_share("world year", "r1", rows, 0.10, ("r2", plan | {"q": 0.125}))
    with pytest.raises(ValueError, match="r2's plan of the world year bounds 0.0 GB at β = 4.0000%, where r1's rows give "
                                         "0.0 GB: it is not these rows' plan"):
        E.at_share("world year", "r1", rows, 0.10, ("r2", plan | {"bound_bits": 50.0}))
    assert E.at_share("world year", "r1", rows, 0.10, ("r2", plan | {"bound_bits": 25.0 * (1 + 5e-4)}))["priced"]

    # rows whose `split` says it is at the row's q and sampled replays hold their own plans: the one at the pinned
    # budget nearest is read, shares and all; a plan of record is read before it; a `split` at another q is no plan
    own = [r | {"split": split | {"q": r["q"], "replay_sample": r["replay_sample"] * (1 + 1e-12)}} for r in TOY]
    priced = E.at_share("world year", "r1", own, 0.10)["priced"]
    assert (priced["of"], priced["run"], priced["beta"]) == ("rows", "r1", 0.03)
    assert {k: (w["share"], list(w["families"])) for k, w in priced["ways"].items()} == {
        "measured": (0.98, ["A_v1", "D_v1"]), "computed": (0.01, ["B_v1"]), "measured and computed": (0.01, ["C_v1"])}
    assert E.at_share("world year", "r1", own, 0.115)["priced"]["beta"] == 0.05
    assert E.at_share("world year", "r1", own, 0.10, ("r2", plan))["priced"]["of"] == "plan"
    other = [r | {"split": split | {"q": r["q"] / 2, "replay_sample": r["replay_sample"]}} for r in TOY]
    assert E.at_share("world year", "r1", other, 0.10)["priced"] is None


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

    def ran(self, run: str, rows: list[dict] | Mapping[str, list[dict] | str] | None, *, plan: dict | None = None,
            name: str = E.ROWS_FILES[0]) -> str:
        """A run's attempt on the remote, its run record holding `rows` as the planner's rows, in the file `name` or
        by file (a file's rows, or its text), and `plan` as its plan (neither: it kept no files)."""

        from research.store import canon

        record = None
        if rows is not None or plan is not None:
            tree = self.tmp / "run-files" / f"{run}-{self._ts}"
            files = rows if isinstance(rows, Mapping) else {name: rows}
            for path, text in (*((f, its if isinstance(its, str) else its and "".join(json.dumps(r) + "\n" for r in its))
                                 for f, its in files.items()),
                               (E.PLAN_FILE, plan and json.dumps(plan))):
                if text:
                    (tree / path).parent.mkdir(parents=True, exist_ok=True)
                    (tree / path).write_text(text)
            record = self.store.put_artifact("run-record/v1", {}, tree=tree)
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


def rows(run: str = ROWS) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / run / "rows.jsonl").read_text().splitlines()]


def plan() -> dict:
    """The fixture's plan, the command's row's."""

    return as_row(json.loads((FIXTURES / PLAN / "mix.json").read_text()))


def mine(scope: str | None = None) -> list[dict]:
    """The fixture's rows of the case the command takes, as its row: the row's variant on one request count, at eight
    heads."""

    return [as_row(r) for r in rows() if (r["world"], r["row"], r["h8"]) == (E.WORLD, CUT_ROW, "h8lin")
            and r["step"].startswith(E.STEP) and scope in (None, r["scope"])]


def record() -> dict[str, list[dict]]:
    """What a run of record holds of the fixture's rows, one variant of each row, by file as the planner's driver
    writes them: row (i) (made up) and the rows it must pass over in the first, the command's row in the second."""

    other = [r for r in rows() if r["row"] != CUT_ROW]
    return {E.ROWS_FILES[0]: [as_assumed(r) for r in mine()] + other, E.ROWS_FILES[1]: mine()}


def recorded(tmp: Path, *, planned: bool = True) -> Store:
    """The fixtures' runs in a store, each labelled the record: the rows over the two year scopes, and with `planned`
    the run that holds the world year's plan and the plan's own row."""

    s = Store(tmp)
    s.label(s.ran(ROWS, record()))
    if planned:
        s.label(s.ran(PLAN, [as_row(r) for r in rows(PLAN)], plan=plan()))
    return s


def test_the_hand_computation_is_reproduced_from_the_rows_it_read(tmp_path):
    """The figures a hand computation gave on these rows, to the digits it kept: the world year's audit budget at 10%, its
    bound at λ = 1 and at the rows' λ, each part of the 10%, the figure with the network check's own proving taken off the
    budget, and the datacenter year's. A fresh machine gets them from the store's remote alone."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert (doc["schema"], doc["unit"], doc["status"], doc["runs"], doc["refused"]) == (E.SCHEMA, "GB", "modeled", [PLAN, ROWS], [])
    by = {p["scope"]: p for p in doc["compute_audit"]}
    assert list(by) == ["datacenter year", "world year"]
    year = by["world year"]
    assert doc["value"] == year["GB"] == pytest.approx(YEAR_GB, rel=1e-12)
    assert year["beta"] == pytest.approx(0.05542238712065398, rel=1e-12) and year["inside"] and year["residual"] < 1e-9
    assert year["bound"]["GB"] == pytest.approx(13561.74961510603, rel=1e-12) and year["lambda"] == pytest.approx(1.05)
    assert (year["run"], year["mix"], year["delta"], year["output_bits"]) == (ROWS, "B128", 0.01, 28000.0)
    assert year["step"].startswith(E.STEP) and year["pinned"] == [0.05, 0.055, 0.06, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0]
    assert {k: round(100 * v, 3) for k, v in year["parts"].items()} == {
        "serve": 4.339, "wait": 2.293, "prove": 5.372, "replay": 0.17, "pool": 0.07, "edge": 0.044, "compile": 0.005}
    assert sum(v for k, v in year["parts"].items() if k != "wait") == pytest.approx(0.10, abs=1e-5)
    other = by["datacenter year"]
    assert other["GB"] == pytest.approx(13707.861447630507, rel=1e-12) and other["run"] == ROWS and other["priced"] is None
    assert other["beta"] == pytest.approx(0.055420109073442365, rel=1e-12)
    net = doc["network_check"]
    assert net["share"] == 0.003 and net["GB"] == pytest.approx(15053.889111429447, rel=1e-12) and net["inside"]
    assert net["beta"] == pytest.approx(0.052425317582956925, rel=1e-12)
    assert E.build(share=0.097, network_check=0, store=recorded(tmp_path / "again").fresh(), root=REPO)["value"] == net["GB"]


def test_the_rows_requests_are_the_count_they_are_priced_over(tmp_path):
    """A row is priced over its sampled replays / the probability a request is sampled, which is the planner's count and
    the network certifier's part's, and not the row's `requests` field. Rows priced over another count are refused: the
    fixture's variant of the row whose indexer is stated against fewer linears is."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    year = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    field = {r["requests"] for r in mine("world year")}
    assert field == {2.02900114160274e16} and year["requests"] == pytest.approx(1.7784008649596718e16, rel=1e-12)
    assert year["requests"] == pytest.approx(doc["periods"][-1]["requests"], rel=1e-12)
    assert year["requests"] == pytest.approx(term().plan()["world_requests"].value, rel=1e-12)
    assert "The row is priced over 1.778e+16 requests, the count the network certifier's part is over." in E.render(doc)

    s = Store(tmp_path / "other")
    s.label(s.ran(ROWS, [as_row(r) for r in rows() if (r["row"], r["h8"]) == (CUT_ROW, "h8")]))
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 2.11575e\\+16 requests on "
                                         "the compute audit's side and 1.7784e\\+16 on the network certifier's"):
        E.build(store=s.store, root=REPO)
    # the run as the planner's driver wrote it, several variants of the row in one scope, is refused by name, and
    # read at the variant named
    both = Store(tmp_path / "both")
    both.label(both.ran(ROWS, [as_row(r) for r in rows()]))
    with pytest.raises(ValueError, match=f"{ROWS}'s world year rows are 2 variants of the row \\(h8, h8lin\\), each on its own "
                                         "premises: a run of record holds one, or --variant names the one meant"):
        E.build(store=both.store, root=REPO)
    named = E.build(store=both.store, root=REPO, variant="h8lin")
    assert named["value"] == pytest.approx(YEAR_GB, rel=1e-12) and named["variant"] == "h8lin" and doc["variant"] is None
    assert "in its variant `h8lin`, read at the audit budget" in E.render(named)


def test_how_the_proving_is_priced_is_the_plans_account(tmp_path):
    """The world year's proving time by how its block families are priced, from the row's plan of record: the sparse
    attention's split part, priced by computation, is over half of it, where the rows' own shares, written before the
    row's levers multiply its blocks, put it under a thousandth and 98.6% in families priced from measured statements.
    Without a plan of record the command says none is read and marks no proving measured."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    priced = next(p for p in doc["compute_audit"] if p["scope"] == "world year")["priced"]
    said = mine("world year")[0]["split"]["families"]
    assert sum(f["share"] for f in said.values() if f["priced"] == ["measured"]) == pytest.approx(0.9857, abs=1e-4)
    assert said["SparseMlaSplitPart_v1"]["share"] < 1e-3 and said["SparseMlaSplitPart_v1"]["priced"] == ["computed"]
    assert (priced["run"], priced["beta"]) == (PLAN, plan()["beta"]) and priced["beta"] == pytest.approx(0.0554224, abs=1e-7)
    ways = priced["ways"]
    assert sum(w["share"] for w in ways.values()) == pytest.approx(1.0, abs=1e-9)
    assert {k: round(w["share"], 4) for k, w in ways.items()} == {
        "computed": 0.5236, "measured": 0.4713, "measured and computed": 0.0051, E.UNSTATED: 0.0}
    assert list(ways["computed"]["families"])[0] == "SparseMlaSplitPart_v1"
    assert ways["computed"]["families"]["SparseMlaSplitPart_v1"] == pytest.approx(0.5229, abs=5e-4)
    mla = sum(s for w in ways.values() for f, s in w["families"].items() if f.startswith("SparseMla"))
    assert mla == pytest.approx(0.5826, abs=5e-4)
    by = {i["input"]: i for i in doc["inputs"]}
    assert (by["proving priced measured"]["mark"], by["proving priced measured"]["value"]) == ("measured", ways["measured"]["share"])
    assert (by["proving priced computed"]["mark"], by["proving priced measured and computed"]["mark"]) == ("modeled", "modeled")
    assert f"({PLAN}) `out/mix.json`: the row's plan at β = 5.542%" in by["proving priced computed"]["source"]
    text = E.render(doc)
    assert (f"- of the row's proving time, by its plan, β = 5.542% ({PLAN}), the share in block families priced each way is "
            "52.36% computed, 47.13% measured, 0.5086% measured and computed, 0.00179% in no stated way: the largest "
            "families are SparseMlaSplitPart_v1 (52.29%, computed), MoeExpertFp8CoordinateChunk_v1 (27.19%, measured), "
            "MoeExpertFp8GemmW_v1 (13%, measured).") in text
    assert f", and the world year's plan ({PLAN})" in text
    assert f"| world year | {PLAN} | 5.542% | computed | 52.358% | 7 | SparseMlaSplitPart_v1 52.291%, " in text and "98.6" not in text

    # a plan of another row's, or of the row on other premises, is not these rows' plan
    for change, why in (({"q": plan()["q"] / 2}, f"{PLAN}'s plan of the world year is priced over 3.5568e\\+16 requests and "
                                                   f"{ROWS}'s rows over 1.7784e\\+16: it is not these rows' plan"),
                        ({"bound_bits": plan()["bound_bits"] / 2}, f"{PLAN}'s plan of the world year bounds 6,78\\d.\\d GB at "
                                                                   f"β = 5.5422%, where {ROWS}'s rows give 13,56\\d.\\d GB")):
        s = Store(tmp_path / next(iter(change)))
        s.label(s.ran(ROWS, record()))
        s.label(s.ran(PLAN, None, plan=plan() | change))
        with pytest.raises(ValueError, match=why):
            E.build(store=s.store, root=REPO)

    bare = E.build(store=recorded(tmp_path / "bare", planned=False).fresh(), root=REPO)
    assert bare["value"] == doc["value"] and bare["runs"] == [ROWS]
    assert all(p["priced"] is None for p in bare["compute_audit"])
    assert not [i for i in bare["inputs"] if i["input"].startswith("proving priced")]
    assert [i["input"] for i in bare["inputs"] if i["mark"] == "measured"] == ["weights"]
    text = E.render(bare)
    assert ("- how the row's proving time is priced is not read, and none of it is marked measured: the rows' `split` is not "
            "their own plan (it doesn't say the row's `q` and `replay_sample`: it is written before the row's levers "
            "multiply its blocks), and no run of record holds the row's plan (`out/mix.json`).") in text
    assert "1 is measured" in text and "block families priced" not in text and "| priced |" not in text


def test_rows_that_hold_their_own_plans_are_read_with_them(tmp_path):
    """A run whose rows each hold their own plan (a `split` at the row's q and sampled replays) needs no plan beside
    it: the figures a hand computation gave on those rows over the world's year and week, each scope's proving priced
    by its row at the pinned budget nearest, and the rows' `requests` the count they are priced over."""

    s = Store(tmp_path)
    s.label(s.ran(PLANNED, [as_row(r) for r in rows(PLANNED)]))
    doc = E.build(store=s.fresh(), root=REPO)
    week, year = doc["compute_audit"]
    assert (doc["runs"], doc["refused"], week["scope"], year["scope"]) == ([PLANNED], [], "world week", "world year")
    assert doc["value"] == year["GB"] == pytest.approx(14228.634535615152, rel=1e-12)
    assert year["beta"] == pytest.approx(0.055422344454158994, rel=1e-12)
    assert week["GB"] == pytest.approx(13961.63619929689, rel=1e-12)
    assert doc["network_check"]["GB"] == pytest.approx(15042.046132203845, rel=1e-12)
    assert {r["requests"] for r in rows(PLANNED) if r["scope"] == "world year"} == {year["requests"]}
    assert year["requests"] == pytest.approx(doc["periods"][-1]["requests"], rel=1e-12)
    for p, computed, measured in ((week, 0.5346, 0.4613), (year, 0.5240, 0.4717)):
        priced = p["priced"]
        assert (priced["of"], priced["run"], priced["beta"]) == ("rows", PLANNED, 0.055)
        assert {k: round(w["share"], 4) for k, w in priced["ways"].items()} == {
            "computed": computed, "measured": measured, "measured and computed": pytest.approx(0.0042, abs=2e-4), "modeled": 0.0}
        assert sum(w["share"] for w in priced["ways"].values()) == pytest.approx(1.0, abs=1e-9)
        assert list(priced["ways"]["computed"]["families"])[0] == "SparseMlaSplitPart_v1"
    by = {i["input"]: i for i in doc["inputs"]}
    assert (by["proving priced measured"]["mark"], by["proving priced computed"]["mark"]) == ("measured", "modeled")
    assert (f"({PLANNED}) `split`: the row's own plan at the pinned budget nearest, β = 5.500%"
            in by["proving priced measured"]["source"])
    text = E.render(doc)
    assert "- **the compute audit's part: 14,229 GB** (modeled), 37.78 copies of the weights" in text
    assert (f"- of the row's proving time, by its own plan at the pinned budget nearest, β = 5.500% ({PLANNED}), the share in "
            "block families priced each way is 52.4% computed, 47.17% measured, 0.4277% measured and computed, 0.001791% "
            "modeled: the largest families are SparseMlaSplitPart_v1 (52.33%, computed)") in text
    assert f"| world week | {PLANNED} (rows) | 5.500% | computed | 53.457% | 7 | SparseMlaSplitPart_v1 53.390%, " in text
    assert ("| world | week | 2.268e+14 | 3.411e+14 | 13,962 | 0.3275 | 0.01 | 973,357 | 35,627 | 504,396 | 35,434 | 8.259e-20 | "
            "1.829e-08 | 987,319 | 49,589 | 518,357 | 49,395 | 0.01 | 468,962 |") in text
    assert ("| world | week | 0 | 13,962 | 937,924 | 193.8 | 468,962 | 0 | 951,885 | 14,155 | 482,924 | 13,962 | 8.259e-20 | "
            "1.829e-08 | 0.01 |") in text
    assert "'s plan (" not in text


def test_the_network_certifiers_part_is_beside_it_at_the_requests_its_calculator_counts(tmp_path, monkeypatch):
    """Both parts count the sampling plan's requests, which the network certifier's calculator counts by default and
    which are fewer than the census's, over the same GPU-seconds. The command holds no count of its own: the count and
    what its request is are the calculator's inputs as it marks them, none of them given, and where the calculator
    counts other requests than the rows of record are priced over the command refuses. The certifier's rows are its own
    calculator's for the class the command reads; and the two are summed under each of its rows by the form: its term
    at each jitter bound, plus the compute audit's part or the entries where the row charges them and they are fewer."""

    lb = term()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert [(p["compute"], p["period"]) for p in doc["periods"]] == [
        (c, p) for c in ("datacenter", "world") for p in ("week", "month", "year")]
    year = doc["periods"][-1]
    assert year["requests"] == pytest.approx(1.7784e16, rel=1e-4) and year["gpu_seconds"] == pytest.approx(1.182473856e16)
    counted = lb.inputs()
    assert year["requests"] == pytest.approx(counted["world_requests"].value, rel=1e-12)
    assert counted["world_requests"].value < lb.inputs(requests="census")["world_requests"].value  # the census's request
    by = {i["input"]: i for i in doc["inputs"]}
    for name in ("request_count", "world_requests", "request_tokens", "token_macs"):
        assert (by[name]["part"], by[name]["value"], by[name]["mark"], by[name]["source"]) == (
            "both", counted[name].value, counted[name].mark, counted[name].source)
    assert by["request_count"]["value"] == "plan" and by["request_tokens"]["value"] == 3320.5
    assert not [i["input"] for i in doc["inputs"] if i["value"] == "given" or "--world-requests" in i["source"]]

    theirs = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS, scan_delta=delta(lb)), only=("world",), periods=("year",))
    assert doc["certifier"]["rows"] == theirs["rows"] and [r["row"] for r in theirs["rows"]][-2:] == [E.OUTPUT_COUNT, "design"]
    assert (doc["certifier"]["schema"], doc["certifier"]["request_count"], doc["certifier"]["response_ids"]) == (
        E.TERM_SCHEMA, "plan", 1024)
    assert (doc["certifier"]["of_record"], doc["certifier"]["set_time"]) == ("2", "1")
    scan = doc["certifier"]["rows"][0]
    assert scan["row"] == "scan" and scan["term"]["1"]["bits"] == pytest.approx(22 * year["requests"])

    audit, together = year["compute_audit"], year["together"]
    assert audit["GB"] == doc["value"] and audit["bits_a_request"] == pytest.approx(audit["bits"] / year["requests"])
    assert audit["of_weights"] == pytest.approx(doc["value"] / doc["weights"]["GB"]) == pytest.approx(37.805, abs=5e-4)
    assert list(together) == list(E.TOGETHER) == ["scan", "N2", "all three", "design, no stop block", "design"]
    rows = {r["row"]: r for r in theirs["rows"]}
    for name, t in together.items():
        card = t["with_set_time_card"]
        assert (t["certifier_GB"], card["certifier_GB"]) == (rows[name]["term"]["2"]["GB"], rows[name]["term"]["1"]["GB"])
        assert t["GB"] == pytest.approx(t["certifier_GB"] + audit["GB"])
        assert card["GB"] == pytest.approx(card["certifier_GB"] + audit["GB"])
    # the entries stand in for the compute audit's part where the scan charges them and they are fewer
    assert scan["entries"]["bits"] > audit["bits"]
    dc = doc["periods"][2]
    assert dc["compute_audit"]["GB"] == pytest.approx(13707.9, abs=0.05) and dc["compute_audit"]["run"] == ROWS
    assert dc["requests"] == pytest.approx(year["requests"] * dc["gpu_seconds"] / year["gpu_seconds"], rel=1e-12)
    # a scope no run of record holds has the network certifier's part alone
    week = doc["periods"][0]
    assert week["compute_audit"] is None and week["together"]["N2"]["GB"] is None and week["together"]["N2"]["certifier_GB"] > 0
    assert week["together"]["N2"]["with_set_time_card"]["GB"] is None

    # the count read is the calculator's: were its plan another, the rows of record would be priced over other requests
    plan = lb.plan()
    monkeypatch.setattr(lb, "plan", lambda: plan | {"world_requests": plan["world_requests"]._replace(
        value=plan["world_requests"].value / 2)})
    with pytest.raises(ValueError, match="the two parts cover different compute over the datacenter year: 4.74291e\\+12 "
                                         "requests on the compute audit's side and 2.37145e\\+12 on the network certifier's"):
        E.build(store=recorded(tmp_path / "halved").fresh(), root=REPO)


def test_the_network_certifiers_figures_of_record_are_at_two_release_schedules(tmp_path, monkeypatch):
    """The network certifier's figures of record are its document's at D_j = 2, held here to that document: the scan's
    over the world's year is the document's `value`, on the class its calculator assumes by default and on the class
    the command reads, and the design's is the jitter alone on either. The headline and the sums are those figures,
    and the calculator's own figure is printed beside the command's where their classes differ. A document of
    another schema is refused by name."""

    lb = term()
    theirs = lb.document(lb.inputs())
    design = next(r for r in theirs["rows"] if (r["scope"], r["period"], r["row"]) == ("world", "year", "design"))
    assert (theirs["schema"], theirs["request_count"], lb.OF_RECORD) == (E.TERM_SCHEMA, "plan", 2)
    assert theirs["value"] == pytest.approx(59_645_644, abs=0.5) and design["term"]["2"]["GB"] == pytest.approx(1_847_615, abs=0.5)

    wide = E.build(store=recorded(tmp_path / "wide").fresh(), root=REPO, response_ids=lb.RESPONSE_IDS)
    sums = wide["periods"][-1]["together"]
    assert (sums["scan"]["certifier_GB"], sums["design"]["certifier_GB"]) == (theirs["value"], design["term"]["2"]["GB"])
    assert sums["scan"]["with_set_time_card"]["certifier_GB"] == theirs["with_set_time_card"]["value"]
    assert wide["certifier"]["default_class"] == {"response_ids": 4096, "eps": 2 ** -20, "value": theirs["value"], "unit": "GB"}
    text = E.render(wide)
    assert ("- **the network certifier's part: 59,645,644 GB** (modeled, on an assumed class of at most 4,096 ids a response) "
            "as the scan's check charges it today, 26.83 bits a request, and 1,857,720 GB as N2 charges it today, ") in text
    assert ("1,847,615 GB by the design no run has behind it, once a served audit has sampled a stop block, each at D_j = 2, "
            "its term of record, and each failing with at most ε = 9.537e-07 (2^-20) over the world's year, at a scan's δ of "
            "8.259e-20 (2^-63.39).\n") in text
    assert "assumes by default" not in text

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    narrow = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))
    sums = doc["periods"][-1]["together"]
    assert E.RESPONSE_IDS == 1024 and sums["scan"]["certifier_GB"] == narrow["value"] == pytest.approx(50_753_639, abs=0.5)
    assert sums["design"]["certifier_GB"] == design["term"]["2"]["GB"]
    assert sums["scan"]["GB"] == pytest.approx(narrow["value"] + YEAR_GB) and sums["design"]["GB"] == pytest.approx(1_861_855, abs=0.5)
    assert doc["certifier"]["default_class"] == wide["certifier"]["default_class"]
    text = E.render(doc)
    assert ("- **the network certifier's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids a response) "
            "as the scan's check charges it today, 22.83 bits a request, and 1,857,720 GB as N2 charges it today, "
            "26,300,627 GB by the design no run has behind it, before a served audit has sampled a stop block, 1,847,615 GB by "
            "the design no run has behind it, once a served audit has sampled a stop block, each at D_j = 2, its term of "
            "record, and each failing with at most ε = 9.537e-07 (2^-20) over the world's year, at a scan's δ of 8.259e-20 "
            "(2^-63.39). On the class of at most 4,096 ids a response that `verity bench certifier_term` assumes by "
            "default, the scan's is 59,645,644 GB.\n") in text
    assert ("at D_j = 2, with no advice priced and with a response's output count taken at 0 under `all three` and `design`: "
            "50,767,879 GB as the scan's check charges it today; 1,871,959 GB as N2 charges it today; 26,314,867 GB by the "
            "design no run has behind it, before a served audit has sampled a stop block; 1,861,855 GB by the design no run "
            "has behind it, once a served audit has sampled a stop block; each failing with at most 0.01, the two parts' "
            "errors summed.\n") in text
    assert ("| world | year | 1.182e+16 | 1.778e+16 | 14,240 | 0.006406 | 0.01 | 50,753,639 | 1,857,720 | 26,300,627 | 1,847,615 | "
            "8.259e-20 | 9.537e-07 | 50,767,879 | 1,871,959 | 26,314,867 | 1,861,855 | 0.01 | 24,453,012 |") in text
    assert "| scan | 24,453,012 | 24,453,012 | 50,753,639 | 134,745 | 48,906,024 | 4.108e+10 | 8.259e-20 | 9.537e-07 |" in text

    document = lb.document
    monkeypatch.setattr(lb, "document", lambda *a, **kw: document(*a, **kw) | {"schema": "certifier-term/v1"})
    with pytest.raises(ValueError, match="the network certifier's document is `certifier-term/v1` and this command reads "
                                         "`certifier-term/v2`"):
        E.build(store=recorded(tmp_path / "older").fresh(), root=REPO)


def test_what_rests_on_the_set_time_card_is_printed_only_under_the_calculators_label(tmp_path, monkeypatch):
    """A figure at D_j = 1, and every figure of a row whose changes include the timing's, is what a network certifier
    with a set-time card would be charged, and is printed only under the label the calculator's document gives it: in
    a line, a column or a row that carries the label. The label is the document's, read from the row's own field and
    from its top: the command's code doesn't spell it, and prints another where the document says another."""

    lb = term()
    label = lb.SET_TIME
    assert label and label not in Path(E.__file__).read_text()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    theirs = {(r["scope"], r["period"], r["row"]): r for r in lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    for p in doc["periods"]:
        for name, t in p["together"].items():
            assert t["label"] == theirs[p["compute"], p["period"], name]["with_set_time_card"]
            assert t["label"] == (label if name == "all three" else None) and t["with_set_time_card"]["label"] == label
    assert {r["row"] for r in doc["certifier"]["rows"] if r["with_set_time_card"] == label} == {"timing", "all three"}

    text = E.render(doc)
    shown = printed(text)
    at_one = [(under, cell) for under, cell in shown if "D_j = 1" in under]
    timed = [(under, cell) for under, cell in shown
             if under.rpartition(" | ")[2].startswith(("timing", "all three")) or ", all three" in under.partition(" | ")[0]]
    assert (len(at_one), len(timed)) == (60, 30) and all(label in under for under, _ in at_one + timed)
    # the scan's term and its two sums at D_j = 1, N2's sum there, and the sum under `all three`
    for figure, least in (("48,906,024", 5), ("48,920,264", 2), ("24,344", 2), ("14,240 GB with the lengths held", 1)):
        under = [under for under, cell in shown if figure in cell]
        assert len(under) >= least and all(label in u for u in under), figure
    assert "- the network certifier's part with a set-time card (not yet tested): 0 GB with the lengths held by a stop block" in text
    assert ("- together, with a set-time card (not yet tested): 14,240 GB with the lengths held by a stop block, the timing "
            "taken once and the entries charged through the compute audit; and at D_j = 1, 48,920,264 GB as the scan's "
            "check charges it today; 24,344 GB as N2 charges it today; 24,467,252 GB by the design no run has behind it, "
            "before a served audit has sampled a stop block; 14,240 GB by the design no run has behind it, once a served "
            "audit has sampled a stop block; error 0.01.\n") in text
    assert ("| all three, with a set-time card (not yet tested) | 0 | 0 | 0 | 0 | 0 | the compute side's | 8.259e-20 | "
            "9.537e-07 |") in text
    assert "\n## Each monitoring period over all compute, with a set-time card (not yet tested): in GB\n" in text
    assert ("| world | year | 0 | 14,240 | 48,906,024 | 10,104 | 24,453,012 | 0 | 48,920,264 | 24,344 | 24,467,252 | 14,240 | "
            "8.259e-20 | 9.537e-07 | 0.01 |") in text
    # no figure of record is under the label: the headline's and the table of record's lines don't carry it
    assert not [line for line in text.splitlines() if label in line and ("its term of record, " in line or "at D_j = 2, with no" in line)]
    assert label not in text.split("## Each monitoring period over all compute: the two parts side by side")[1].split("\n## ")[0]

    other = "under a label no code of the command holds"
    monkeypatch.setattr(lb, "SET_TIME", other)
    again = E.render(E.build(store=recorded(tmp_path / "again").fresh(), root=REPO))
    assert label not in again and again.count(other) == text.count(label) and again == text.replace(label, other)


def test_the_output_count_is_beside_the_sums_that_take_it_at_zero(tmp_path):
    """Until a served audit has sampled a stop block the server chooses a response's output count. The network
    certifier's row for the design before its stop block charges it, as its schedules, and the command reads it from
    there: beside the sums under the rows whose schedules are 0, which take it at 0, marked assumed, in every
    monitoring period and where the print says no advice is priced."""

    lb = term()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    theirs = {(r["scope"], r["period"], r["row"]): r for r in lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    for p in doc["periods"]:
        count, row = p["output_count"], theirs[p["compute"], p["period"], E.OUTPUT_COUNT]
        assert (count["row"], count["mark"], count["at_zero_in"]) == ("design, no stop block", "assumed", ["all three", "design"])
        assert (count["GB"], count["bits"]) == (row["schedules"]["GB"], row["schedules"]["bits"])
        assert count["bits_a_request"] == pytest.approx(11) and count["bits"] == pytest.approx(11 * p["requests"])
        sums = p["together"]
        assert sums[E.OUTPUT_COUNT]["certifier_GB"] - sums["design"]["certifier_GB"] == pytest.approx(count["GB"])
        assert all(theirs[p["compute"], p["period"], name]["schedules"]["bits"] == 0 for name in count["at_zero_in"])
    year = doc["periods"][-1]
    assert year["output_count"]["GB"] == pytest.approx(24_453_012, abs=0.5)
    assert year["together"][E.OUTPUT_COUNT]["GB"] - year["together"]["design"]["GB"] == pytest.approx(year["output_count"]["GB"])
    by = {i["input"]: i for i in doc["inputs"]}
    assert (by["output_count"]["part"], by["output_count"]["value"], by["output_count"]["mark"]) == ("network certifier", 0, "assumed")
    assert by["output_count"]["source"] == (
        "not counted in the sums under `all three` and `design`: until a served audit has sampled a stop block the server "
        "chooses a response's output count, which the network certifier's row `design, no stop block` charges 11 bits a "
        "request as its schedules")
    text = E.render(doc)
    assert "with no advice priced and with a response's output count taken at 0 under `all three` and `design`: " in text
    assert ("- the output count: 24,453,012 GB (assumed), 11 bits a request. Until a served audit has sampled a stop block the "
            "server chooses a response's output count, and the network certifier's row `design, no stop block` charges it "
            "as its schedules. The sums under `all three` and `design` take it at 0, as they price no advice.\n") in text
    assert "| together, design | together's error | output count (assumed) |\n" in text
    assert "the sum under `design` takes it at 0." in text
    assert "The sum under `all three` takes the output count at 0 too." in text
    # on the class the calculator assumes by default a connection's count is 13 bits
    wide = E.build(store=recorded(tmp_path / "wide").fresh(), root=REPO, response_ids=lb.RESPONSE_IDS)
    count = wide["periods"][-1]["output_count"]
    assert count["bits_a_request"] == pytest.approx(13) and count["GB"] == pytest.approx(28_899_014, abs=0.5)


def test_every_figure_carries_its_error_at_the_scans_delta_the_term_states(tmp_path):
    """Given no δ, the command gives the calculator the one at which the row it prints that sums the most execution
    phases, the world's year, carries `PERIOD_ERROR`, so that row's ε is that error and every other row's less. Each
    scope's monitoring period carries its rows' own ε, their execution phases times that δ, the compute audit's δ
    beside it, and their sum for the two together. The δ is the one the term's document states among its inputs,
    marked as the calculator marks it, with the command's reason as its source. No amount moves with it."""

    lb = term()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    given = delta(lb)
    assert E.PERIOD_ERROR == Fraction(1, 2 ** 20) and given == Fraction(1, 12108532285440000000) != lb.SCAN_DELTA
    theirs = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS, scan_delta=given), only=E.SCOPES)
    said, by = doc["certifier"]["scan_delta"], {i["input"]: i for i in doc["inputs"]}
    assert (said["value"], said["mark"]) == (theirs["inputs"]["scan_delta"]["value"], "assumed") and said["value"] == float(given)
    assert said["source"] == (
        "each execution phase's scan's δ, this command's, in place of the calculator's default: 9.537e-07 (2^-20), the error "
        "a monitoring period's figure is to carry at most, over the 1.155e+13 execution phases of the world's year, the row "
        "printed that sums the most, so that every row printed carries that error or less. The network certifier's result "
        "of record takes its δ so (benchmarks/certifier_term/README.md), and no term depends on it")
    assert by["scan_delta"] == {"part": "network certifier", "input": "scan_delta", **said}

    rows = {(r["scope"], r["period"]): r for r in theirs["rows"]}
    for p in doc["periods"]:
        r, audit = rows[p["compute"], p["period"]], p["compute_audit"]
        assert p["eps"] == r["eps"] == float(Fraction(r["phases"]) * given) and p["phases"] == r["phases"]
        assert p["error"] == (None if audit is None else p["eps"] + audit["delta"])
    dc, year = doc["periods"][2], doc["periods"][-1]
    assert year["eps"] == 2 ** -20 == max(p["eps"] for p in doc["periods"]) and year["phases"] == 11547596250000
    assert (year["compute_audit"]["delta"], year["error"]) == (0.01, 0.01 + 2 ** -20)
    assert dc["eps"] == pytest.approx(2.543e-10, rel=1e-3) and dc["error"] == 0.01 + dc["eps"]
    assert doc["periods"][0]["error"] is None  # a scope no run of record holds has no sum, and so no error of one
    # the amounts are the calculator's at its own default δ, where every one of these rows' ε is 1 or more
    default = {(r["scope"], r["period"], r["row"]): r for r in lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    for p in doc["periods"]:
        for name, t in p["together"].items():
            r = default[p["compute"], p["period"], name]
            assert r["eps"] >= 1 and t["certifier_GB"] == r["term"]["2"]["GB"]
    assert default["world", "year", "scan"]["eps"] == pytest.approx(11_012_646, abs=0.5)

    text = E.render(doc)
    assert ("- the compute audit's part fails with at most δ = 0.01 over the monitoring period. The network certifier's ε "
            "over the world's year is each execution phase's scan's δ (assumed), 8.259e-20 (2^-63.39), summed over every "
            "link's execution phases, 1.155e+13 of them: 9.537e-07 (2^-20). The two together fail with at most the sum of "
            "both, 0.01. A figure whose error, summed over everything it covers, is 1 or more is printed as \"no bound at "
            "this period\", with that error and no amount.\n") in text
    assert ("- the network certifier's part with a set-time card (not yet tested): 0 GB with the lengths held by a stop "
            "block, the timing taken once and the entries charged through the compute audit; and at D_j = 1, ") in text
    assert "once a served audit has sampled a stop block; ε = 9.537e-07 (2^-20).\n" in text
    assert ("| compute | period | GPU-seconds | requests | compute audit | bits a request | compute audit's δ | certifier, scan | "
            "certifier, N2 | certifier, design, no stop block | certifier, design | scan's δ | certifier's ε | together, scan | "
            "together, N2 | together, design, no stop block | together, design | together's error | output count (assumed) |\n") in text
    assert ("| datacenter | year | 3.154e+12 | 4.743e+12 | 13,708 | 23.12 | 0.01 | 13,536 | 495.4 | 7,014 | 492.8 | 8.259e-20 | "
            "2.543e-10 | 27,244 | 14,203 | 20,722 | 14,201 | 0.01 | 6,521 |") in text
    assert ("| datacenter | week | 6.048e+10 | 9.096e+10 | no rows of record | none | none | 259.6 | 9.502 | 134.5 | 9.45 | "
            "8.259e-20 | 4.878e-12 | none | none | none | none | none | 125.1 |") in text
    shown = printed(text)
    assert {cell for under, cell in shown if under.startswith("scan's δ | ")} == {"8.259e-20"}
    assert len([cell for under, cell in shown if under.startswith("scan's δ | ")]) == 8 + 6 + 6
    assert E.NO_BOUND not in text.replace(f"\"{E.NO_BOUND}\"", "")


def made_up(document, eps: Mapping[tuple[str, str], float], scan_delta: float | None = None, **row: float):
    """The calculator's ``document`` with made-up errors: each scope's monitoring period of ``eps`` at that ε on
    every row, but for the rows ``row`` names, which over the world's year are each at its own; and with
    ``scan_delta`` the δ the document states."""

    def made(*a, **kw) -> dict:
        doc = document(*a, **kw)
        stated = doc["inputs"]["scan_delta"] | ({} if scan_delta is None else {"value": scan_delta})
        rows = [r | {"eps": eps.get((r["scope"], r["period"]), r["eps"])} for r in doc["rows"]]
        return doc | {"inputs": doc["inputs"] | {"scan_delta": stated},
                      "rows": [r | {"eps": row.get(r["row"], r["eps"])} if (r["scope"], r["period"]) == ("world", "year")
                               else r for r in rows]}

    return made


def test_a_figure_whose_error_is_one_or_more_prints_no_bound_and_no_amount(tmp_path, monkeypatch):
    """On a term document with made-up errors. A scope's monitoring period whose ε is under 1 prints the network
    certifier's amounts with that ε; one whose ε is 1 or more prints "no bound at this period" with its ε and no
    amount, in the document and in every line, column and row of the print, the output count among them. The two
    together carry the sum of both parts' errors, and print no amount where that is 1 or more though the network
    certifier's own is under. The headline is the world's year's, and follows it. The compute audit's part keeps
    its amounts. The δ printed is the one the document states, whatever the command gave the calculator."""

    lb = term()
    document = lb.document
    theirs = {(r["scope"], r["period"], r["row"]): r for r in document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    monkeypatch.setattr(lb, "document", made_up(document, {
        ("datacenter", "week"): 0.25, ("datacenter", "year"): 0.995, ("world", "week"): 1.0, ("world", "year"): 3.5},
        scan_delta=0.125))
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    week, _, dc, none, _, year = doc["periods"]
    assert [(p["eps"], p["error"]) for p in (week, dc, none, year)] == [(0.25, None), (0.995, 1.005), (1.0, None), (3.5, 3.51)]
    assert doc["certifier"]["scan_delta"]["value"] == 0.125 != float(delta(lb))
    assert doc["value"] == pytest.approx(YEAR_GB) and year["compute_audit"]["GB"] == doc["value"]

    def amounts(p: dict) -> list:
        """A scope's monitoring period's figures of the network certifier, then of the two together."""

        both = [t for s in p["together"].values() for t in (s, s["with_set_time_card"])]
        count = p["output_count"]
        return [[t["certifier_GB"] for t in both] + [count["GB"], count["bits"], count["bits_a_request"]], [t["GB"] for t in both]]

    # under 1: the network certifier's amounts, the calculator's; with no rows of record there is nothing to sum
    for p in (week, dc):
        assert amounts(p)[0][:2] == [theirs[p["compute"], p["period"], "scan"]["term"][dj]["GB"] for dj in ("2", "1")]
        assert None not in amounts(p)[0] and set(amounts(p)[1]) == {None}
    assert dc["compute_audit"]["GB"] == pytest.approx(13707.9, abs=0.05)  # 0.995 + 0.01 is 1 or more: no sum
    # 1 or more, 1 itself among them: no amount of the network certifier's, the output count's among them, and no sum
    for p in (none, year):
        assert set(amounts(p)[0]) == set(amounts(p)[1]) == {None}
        assert p["output_count"]["at_zero_in"] == ["all three", "design"] and p["gpu_seconds"] and p["requests"]
    assert [sorted(r.keys() & set(E.AMOUNTS)) for r in doc["certifier"]["rows"]] == [[]] * 8
    assert {r["eps"] for r in doc["certifier"]["rows"]} == {3.5} and doc["certifier"]["rows"][0]["row"] == "scan"
    assert doc["certifier"]["default_class"] == {"response_ids": 4096, "eps": 3.5, "value": None, "unit": "GB"}

    text = E.render(doc)
    assert "- **the compute audit's part: 14,240 GB** (modeled), 37.81 copies of the weights" in text
    assert ("- **the network certifier's part: no bound at this period** (modeled, on an assumed class of at most 1,024 ids a "
            "response). Its ε over the world's year is 3.5, and at 1 or more it bounds nothing, so no amount is printed under "
            "any of its rows.\n") in text
    assert ("- **together: no bound at this period**. The two parts' errors over the world's year sum to 3.51, and at 1 or "
            "more the sum bounds nothing, so no amount is printed for the two together.\n") in text
    assert "- the output count: no bound at this period (assumed), under the network certifier's ε. Until a served audit" in text
    assert ("The network certifier's ε over the world's year is each execution phase's scan's δ (assumed), 0.125 (2^-3), "
            "summed over every link's execution phases, 1.155e+13 of them: 3.5. The two together fail with at most the sum "
            "of both, 3.51.") in text
    head = text.split("\n## Inputs")[0]
    assert " GB" not in "".join(line for line in head.splitlines()
                                if line.startswith(("- **the network certifier's part", "- **together", "- the output count")))
    assert "with a set-time card" not in head and "its term of record" not in head
    # none of the world year's amounts of the network certifier is printed, at either jitter bound or summed
    for figure in ("50,753,639", "1,857,720", "26,300,627", "1,847,615", "48,906,024", "24,453,012", "50,767,879", "1,861,855",
                   "59,645,644", "22.83 bits a request", "11 bits a request"):
        assert figure not in text, figure
    assert "| scan | " + "no bound at this period | " * 6 + "0.125 | 3.5 |" in text
    assert text.count("| " + "no bound at this period | " * 6 + "0.125 | 3.5 |\n") == 8
    no = "no bound at this period | "
    assert ("| world | year | 1.182e+16 | 1.778e+16 | 14,240 | 0.006406 | 0.01 | " + no * 4 + "0.125 | 3.5 | " + no * 4
            + "3.51 | no bound at this period |") in text
    assert "| world | year | " + no * 10 + "0.125 | 3.5 | 3.51 |" in text
    assert ("| world | week | 2.268e+14 | 3.411e+14 | no rows of record | none | none | " + no * 4 + "0.125 | 1 | "
            + "none | " * 4 + "none | no bound at this period |") in text
    assert "| world | week | no bound at this period | none | " + no * 4 + "none | " * 4 + "0.125 | 1 | none |" in text
    assert ("| datacenter | year | 3.154e+12 | 4.743e+12 | 13,708 | 23.12 | 0.01 | 13,536 | 495.4 | 7,014 | 492.8 | 0.125 | 0.995 | "
            + no * 4 + "1.005 | 6,521 |") in text
    assert ("| datacenter | year | 0 | no bound at this period | 13,043 | 2.695 | 6,521 | 0 | " + no * 4
            + "0.125 | 0.995 | 1.005 |") in text
    assert ("| datacenter | week | 6.048e+10 | 9.096e+10 | no rows of record | none | none | 259.6 | 9.502 | 134.5 | 9.45 | 0.125 | "
            "0.25 | none | none | none | none | none | 125.1 |") in text
    # the δ printed is the document's own on every line that carries one, and the one the command gave is nowhere
    shown = printed(text)
    assert {cell for under, cell in shown if under.startswith("scan's δ | ")} == {"0.125"} and "8.259e-20" not in text
    # every cell that is no bound is on a line that carries the error that says so
    lines = [line for line in text.splitlines() if line.startswith("|") and E.NO_BOUND in line]
    assert len(lines) == 8 + 3 + 3 and all(any(f"| {e} |" in line for e in ("3.5", "1", "1.005")) for line in lines)

    # the network certifier's part is bounded and the sum is not: its amounts are printed and the sum's are not
    monkeypatch.setattr(lb, "document", made_up(document, {("world", "year"): 0.995}))
    text = E.render(E.build(store=recorded(tmp_path / "sum").fresh(), root=REPO))
    assert "- **the network certifier's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids" in text
    assert "its term of record, and each failing with at most ε = 0.995 (2^-0.007232) over the world's year" in text
    assert "- **together: no bound at this period**. The two parts' errors over the world's year sum to 1.005" in text
    assert "- the output count: 24,453,012 GB (assumed), 11 bits a request." in text and "- together, " not in text
    # one ε a scope and monitoring period: rows of one that carry two are refused by name
    monkeypatch.setattr(lb, "document", made_up(document, {("world", "year"): 0.5}, N2=2.0))
    with pytest.raises(ValueError, match="the network certifier's rows over the world year carry 2 different ε \\(0.5, 2\\), and "
                                         "this command prints one a scope and monitoring period"):
        E.build(store=recorded(tmp_path / "two").fresh(), root=REPO)


def test_every_input_is_marked_measured_modeled_or_assumed_with_its_source(tmp_path):
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    by = {i["input"]: i for i in doc["inputs"]}
    assert all(i["mark"] in E.MARKS and i["source"] and i["part"] in ("compute audit", "network certifier", "both")
               for i in doc["inputs"])
    assert {k: (by[k]["part"], by[k]["mark"], by[k]["value"]) for k in (
        "planner_rows", "compute_share", "delta", "output_bits", "network_check_share", "advice")} == {
        "planner_rows": ("compute audit", "modeled", 18), "compute_share": ("compute audit", "assumed", 0.1),
        "delta": ("compute audit", "assumed", 0.01), "output_bits": ("compute audit", "assumed", 28000.0),
        "network_check_share": ("compute audit", "assumed", 0.003), "advice": ("compute audit", "assumed", 0)}
    assert by["lambda"]["mark"] == "assumed" and by["lambda"]["value"] == pytest.approx(1.05)
    assert f"`{E.RECORD}` ({PLAN}, {ROWS})" in by["planner_rows"]["source"] and "2 scopes" in by["planner_rows"]["source"]
    # the world's requests are the sampling plan's, as the certifier's calculator reads them from the planner
    requests = by["world_requests"]
    assert (requests["part"], requests["mark"]) == ("both", "modeled") and "benchmarks/spot_figures/breakeven" in requests["source"]
    assert (by["request_count"]["part"], by["request_count"]["value"], by["request_count"]["mark"]) == ("both", "plan", "assumed")
    assert (by["weights"]["part"], by["weights"]["mark"]) == ("both", "measured")
    assert (by["response_ids"]["part"], by["response_ids"]["value"], by["response_ids"]["mark"]) == (
        "network certifier", 1024, "assumed")
    assert (by["mfu_serve"]["part"], by["mfu_serve"]["mark"], by["mfu_serve"]["value"]) == ("both", "assumed", 0.4)
    # a request is the plan's, not the census's
    assert (by["request_tokens"]["value"], by["request_tokens"]["mark"]) == (3320.5, "modeled")
    assert by["token_macs"]["mark"] == "modeled" and by["token_macs"]["value"] == pytest.approx(7.925e10, rel=1e-3)
    assert "128k serve plan" in by["request_tokens"]["source"] and "reprice.workload" in by["token_macs"]["source"]
    world = by["world_machines"]["value"] * by["machine_gpus"]["value"] * 365 * 86400
    assert requests["value"] == pytest.approx(world * by["mfu_serve"]["value"] * by["peak_flops"]["value"]
                                              / (by["request_tokens"]["value"] * 2 * by["token_macs"]["value"]), rel=1e-9)

    text = E.render(doc)
    assert text.startswith("# verity bench exfiltration: GLM-5.3's exfiltration capacity over a monitoring period, at 10% of "
                           "total compute\n")
    assert "- **the compute audit's part: 14,240 GB** (modeled), 37.81 copies of the weights (GLM-5.3 in FP4, 376.7 GB)" in text
    assert "β = 5.542%: its bound, 13,562 GB, times λ = 1.05." in text
    assert ("- **the network certifier's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids a response) "
            "as the scan's check charges it today") in text
    assert "14,240 GB with the lengths held by a stop block, the timing taken once and the entries charged through the compute audit" in text
    assert "the audit budget is β = 5.243% and the compute audit's part 15,054 GB." in text
    assert f"- compute-audit: {PLAN}, {ROWS} (`{E.RECORD}`): the planner's rows, 18 of row `(ii) honest`" in text
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in E.MARKS}
    assert f"of {len(doc['inputs'])} inputs, {marks['measured']} are measured, {marks['modeled']} modeled and {marks['assumed']} assumed" in text
    assert "| world year | " + ROWS + " | 5.542% | 13,562 | 14,240 | 37.81 | 1.778e+16 | 4.339% | 2.293% | 5.372% |" in text
    assert "| world | year | 1.182e+16 | 1.778e+16 | 14,240 | 0.006406 | 0.01 | 50,753,639 | 1,857,720 | " in text
    assert "| world | week | 2.268e+14 | 3.411e+14 | no rows of record | none | " in text
    assert "Refused" not in text and text.count("\n| datacenter | ") == 6


def test_an_unlabelled_superseded_or_ambiguous_record_is_refused_by_name(tmp_path):
    s = Store(tmp_path)
    held, planned = s.ran(ROWS, record()), s.ran(PLAN, [as_row(r) for r in rows(PLAN)], plan=plan())
    year = s.ran("r20261111-000000-0000", mine("world year"))

    def build(**kw) -> dict:
        return E.build(store=s.store, root=REPO, **kw)

    with pytest.raises(ValueError, match=f"no planner rows of record: nothing in the store carries `run_of_record={E.RECORD}`"
                                         f".*research data label RUN run_of_record {E.RECORD} --by LANE"):
        build()
    # a plan alone is no figure: its own row, at its budget, is not read off
    s.label(planned)
    with pytest.raises(ValueError, match=f"no run of record holds the planner's rows: {PLAN} hold a plan alone"):
        build()
    other = s.label(s.ran("r20261111-000000-0009", mine("datacenter year")))
    with pytest.raises(ValueError, match=f"no run of record holds the world year's rows \\({other} holds datacenter year\\)"):
        build()
    s.label(other, E.SUPERSEDED_BY, held)
    s.label(held)
    assert build()["value"] == pytest.approx(YEAR_GB)

    # a second run holding the world's year: refused until one is superseded, then the newer one is read
    with pytest.raises(ValueError, match=f"{ROWS} and {year} both hold the world year's rows and neither is superseded: label "
                                         "the one the other replaces .*superseded_by NEWER"):
        s.label(year) and build()
    s.label(year, E.SUPERSEDED_BY, held)
    newer = s.label(s.ran("r20261112-000000-0001", [r | {"bound_bits": 2 * r["bound_bits"]} for r in mine("world year")]))
    s.label(planned, E.SUPERSEDED_BY, newer)
    s.label(held, E.SUPERSEDED_BY, newer)
    doc = build()
    assert doc["value"] == pytest.approx(2 * YEAR_GB) and doc["runs"] == [newer]
    assert {x["target"]: x["why"] for x in doc["refused"]}[ROWS] == f"superseded by {newer}"
    assert f"- {ROWS} (`{E.RECORD}`): superseded by {newer}" in E.render(doc)
    # a second plan of the scope is refused the same way, and a plan of a scope with no rows is left out and listed
    a = s.label(s.ran("r20261112-000000-000a", None, plan=plan() | {"bound_bits": 2 * plan()["bound_bits"]}))
    assert build()["compute_audit"][0]["priced"]["run"] == a
    b = s.label(s.ran("r20261112-000000-000b", None, plan=plan()))
    with pytest.raises(ValueError, match=f"{b} and {a} both hold the world year's plan and neither is superseded"):
        build()
    s.label(b, E.SUPERSEDED_BY, a)
    c = s.label(s.ran("r20261112-000000-000c", None, plan=plan() | {"scope": "world week"}))
    assert {x["target"]: x["why"] for x in build()["refused"]}[c] == (
        "holds the world week's plan, and no run of record holds that scope's rows")

    # runs of record that can't be read are left out and listed
    s.label(s.ran("r20261112-000000-0002", None))
    s.label(s.ran("r20261112-000000-0003", [{}], name="out/other.jsonl"))
    s.label(s.ran("r20261112-000000-0004", [r | {"error": "timed out"} for r in mine("world year")]))
    s.label("r20261112-000000-0005")
    assert {x["target"][-1]: x["why"] for x in build()["refused"] if x["target"].startswith("r20261112") and x["target"][-1] < "a"} == {
        "2": "kept no run files", "3": "holds no planner rows (`out/rows.jsonl`, `out/three/rows.jsonl`) and no plan "
                                                "(`out/mix.json`)",
        "4": "holds no row `(ii) honest` at a step `+ the verifier as measured...` of the world `ours`",
        "5": "not in the store or on its remote"}

    # another row of the plan, a step two steps begin with, and a row no run holds
    both = recorded(tmp_path / "both").store
    third = E.build(store=both, root=REPO, row="(iii) honest")
    assert third["runs"] == [ROWS] and third["refused"][0]["target"] == PLAN and third["value"] == pytest.approx(2459.2, abs=0.05)
    assert third["compute_audit"][0]["priced"] is None
    steps = Store(tmp_path / "steps")
    steps.label(steps.ran(ROWS, [r | {"step": r["step"] + more} for r in mine("world year") for more in ("", ", again")]))
    with pytest.raises(ValueError, match=f"`\\+ ` begins 2 steps of {ROWS}'s row `\\(ii\\) honest`: give more of the one meant"):
        E.build(store=steps.store, root=REPO, step="+ ")
    with pytest.raises(ValueError, match="no run of record holds the planner's rows: .*holds no row `\\(iv\\)`"):
        build(row="(iv)")
    with pytest.raises(ValueError, match="the share of total compute is in \\(0, 1\\) and the network check's own below it"):
        build(share=0.1, network_check=0.1)


def test_the_two_parts_must_cover_the_same_compute_and_weights(tmp_path):
    def build(name: str, **change) -> dict:
        s = Store(tmp_path / name)
        s.label(s.ran(ROWS, [r | change for r in mine("world year")]))
        return E.build(store=s.store, root=REPO)

    assert build("same")["value"] == pytest.approx(YEAR_GB)
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 1e\\+16 GPU-seconds on the "
                                         "compute audit's side and 1.18247e\\+16 on the network certifier's"):
        build("compute", gpu_seconds=1e16)
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 2e\\+16 requests on the "
                                         "compute audit's side and 1.7784e\\+16 on the network certifier's"):
        build("requests", q=1e-7, replay_sample=2e9)
    with pytest.raises(ValueError, match="the two parts are shares of different weights: the rows of record hold 3,000,000,000,000"):
        build("weights", weights_bits=3_000_000_000_000, target_bits=3_000_000_000_000 / 1.05)


def test_the_figure_is_the_plan_of_records_row_and_row_i_is_only_under_its_label(tmp_path):
    """The figure is the plan of record's row, and `together` sums it. Row (i), whose proving the planner prices at its
    assumed prices, is read over the world's year beside it and printed only under its label, and the document holds
    it as a field of its own; as the row read, the figure carries the label. A run holds its rows in two files, a
    case's over a scope read from the first that holds them, and a later file that holds other ones leaves the run
    out. A row (i) no run of record holds, or that covers other compute, is said to be unread, and the figure stands."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    head, year = next(p for p in doc["compute_audit"] if p["scope"] == "world year"), doc["periods"][-1]
    assert (E.ROW, E.ASSUMED_ROW, doc["row"], doc["label"]) == ("(ii) honest", "(i) honest", "(ii) honest", None)
    assert doc["value"] == head["GB"] == year["compute_audit"]["GB"] == pytest.approx(YEAR_GB, rel=1e-12)
    for t in year["together"].values():
        assert t["GB"] == pytest.approx(t["certifier_GB"] + YEAR_GB) and t["with_set_time_card"]["GB"] == pytest.approx(
            t["with_set_time_card"]["certifier_GB"] + YEAR_GB)
    a = doc["at_assumed_prices"]
    assert (a["row"], a["label"], a["run"], a["delta"], a["inside"], a["why"], a["sources"]) == (
        E.ASSUMED_ROW, E.ASSUMED_LABEL, ROWS, 0.01, True, None, None)
    assert a["GB"] == pytest.approx(ASSUMED * YEAR_GB, rel=1e-12) and a["beta"] == pytest.approx(head["beta"], rel=1e-12)
    assert json.loads(json.dumps(doc))["at_assumed_prices"]["GB"] == a["GB"] != doc["value"]

    text = E.render(doc)
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in text
    assert "It is the planner's row `(ii) honest`, the plan of record's row, with every modeled lever at its measured value" in text
    line = (f"- row `(i) honest`, at the planner's assumed prices: 17,800 GB, at β = 5.542% ({ROWS}), failing with at most "
            "δ = 0.01 over the monitoring period. It is not the figure. What prices its proving time: own row not recorded, "
            "family row not recorded, planner's price not recorded; at the planner's assumed default (`reprice.DEFAULT`) "
            "not recorded.")
    assert [x for x in text.splitlines() if "(i) honest" in x or "17,8" in x] == [line]

    own = E.build(store=recorded(tmp_path / "own").fresh(), root=REPO, row=E.ASSUMED_ROW)
    assert (own["label"], own["at_assumed_prices"]) == (E.ASSUMED_LABEL, None)
    assert own["value"] == pytest.approx(ASSUMED * YEAR_GB, rel=1e-12)
    text = E.render(own)
    assert "- **the compute audit's part, at the planner's assumed prices: 17,800 GB** (modeled)" in text
    assert "- row `(i) honest`, at" not in text

    s = Store(tmp_path / "alone")
    s.label(s.ran(ROWS, mine()))
    alone = E.build(store=s.store, root=REPO)
    assert alone["value"] == pytest.approx(YEAR_GB) and alone["at_assumed_prices"]["GB"] is None
    assert (f"- row `(i) honest`, at the planner's assumed prices: not read: no run of record holds the planner's rows: "
            f"{ROWS} holds no row `(i) honest` at a step") in E.render(alone)
    files = record()
    s = Store(tmp_path / "other")
    s.label(s.ran(ROWS, files | {E.ROWS_FILES[0]: [r | {"gpu_seconds": 1e16} for r in files[E.ROWS_FILES[0]]
                                                   if r["row"] == E.ASSUMED_ROW]}))
    other = E.build(store=s.store, root=REPO)
    assert other["value"] == pytest.approx(YEAR_GB)
    assert other["at_assumed_prices"]["why"] == "its rows differ from the figure's in gpu_seconds"

    # the driver writes row (i)'s world year to both files: the same rows, in any order, are read once
    twice = [r for r in files[E.ROWS_FILES[0]] if r["row"] == E.ASSUMED_ROW and r["scope"] == "world year"]
    s = Store(tmp_path / "same")
    s.label(s.ran(ROWS, files | {E.ROWS_FILES[1]: files[E.ROWS_FILES[1]] + twice[::-1]}))
    assert E.build(store=s.store, root=REPO)["at_assumed_prices"]["GB"] == pytest.approx(a["GB"], rel=1e-12)
    s = Store(tmp_path / "differ")
    s.label(s.ran(ROWS, files | {E.ROWS_FILES[1]: files[E.ROWS_FILES[1]] + [r | {"bound_bits": 2 * r["bound_bits"]}
                                                                           for r in twice]}))
    with pytest.raises(ValueError, match=f"no run of record holds the planner's rows: {ROWS} holds row `\\(i\\) honest` over "
                                         "the world year in `out/rows.jsonl` and in `out/three/rows.jsonl`, and the two "
                                         "differ"):
        E.build(store=s.store, root=REPO)


def test_what_prices_the_rows_proving_is_printed_by_source_or_not_recorded(tmp_path):
    """Beside each row printed, the share of its proving each source prices, as its own plan records it: its `sources`,
    else its `basis` by source, each of own row, family row and the planner's price at 0 where it prices none, and the
    share at the planner's assumed default. A row whose plan records neither, or that is read with a plan of record,
    prints "not recorded" for each."""

    assert E._sources({"families": {}}) is None
    basis = {"reprice.DEFAULT": 0.25, "steady": 0.5, "steady: art:00": 0.125, "family": 0.0625, "reprice.FAMILY": 0.03,
             "reprice.CACHE": 0.0124, "headline.copy_stratum": 0.0201}
    assert E._sources({"basis": basis}) == {
        "by": {"own row": 0.625, "family row": 0.0625, "planner's price": pytest.approx(0.2924), "headline.copy_stratum": 0.0201},
        "default": 0.25}
    sources = {"own row": 0.99, "planner's price": 0.01}
    assert E._sources({"sources": sources, "basis": basis}) == {
        "by": {"own row": 0.99, "family row": 0.0, "planner's price": 0.01}, "default": 0.25}
    assert E._sources({"sources": sources})["default"] is None

    s = Store(tmp_path)
    s.label(s.ran(PLANNED, [as_row(r) | {"split": r["split"] | {"basis": basis}} for r in rows(PLANNED)]
                  + [as_assumed(r) | {"split": r["split"] | {"basis": {"reprice.DEFAULT": 0.979, "reprice.CACHE": 0.021}}}
                     for r in rows(PLANNED)]))
    doc = E.build(store=s.fresh(), root=REPO)
    assert doc["at_assumed_prices"]["sources"] == {
        "by": {"own row": 0.0, "family row": 0.0, "planner's price": pytest.approx(1.0)}, "default": 0.979}
    text = E.render(doc)
    assert (f"- what prices the row's proving time, as its rows record it (its own plan at β = 5.500%, {PLANNED}): own row "
            "62.5%, family row 6.25%, planner's price 29.24%, headline.copy_stratum 2.01%; at the planner's assumed default "
            "(`reprice.DEFAULT`) 25%.\n") in text
    assert ("What prices its proving time: own row 0%, family row 0%, planner's price 100%; at the planner's assumed default "
            "(`reprice.DEFAULT`) 97.9%.\n") in text

    # the fixture's own plans record no source, and a plan of record none either
    none = ("own row not recorded, family row not recorded, planner's price not recorded; at the planner's assumed default "
            "(`reprice.DEFAULT`) not recorded.\n")
    s = Store(tmp_path / "bare")
    s.label(s.ran(PLANNED, [as_row(r) for r in rows(PLANNED)]))
    bare = E.build(store=s.store, root=REPO)
    assert next(p for p in bare["compute_audit"] if p["scope"] == "world year")["priced"]["sources"] is None
    assert "- what prices the row's proving time, as its rows record it: " + none in E.render(bare)
    planned = E.build(store=recorded(tmp_path / "planned").fresh(), root=REPO)
    assert planned["compute_audit"][1]["priced"]["of"] == "plan"
    assert "- what prices the row's proving time, as its rows record it: " + none in E.render(planned)


# ---- the share of compute a served run measures ------------------------------------------------------------------------------

#: a made-up served run's record, as its reader gives it: the window served 1,000 s on 8 GPUs, 8,000 GPU-seconds, and the
#: audit's 960 of them, 12%: two proving sessions on a GPU each (480), staging (20), the replay (240), the interior
#: commitment (20) and the serving charge over stock on the same load (8,000 - 7,800)
SERVED = {"run": "r20261113-000000-5e7d", "serving": {"seconds": 1000.0, "gpus": 8},
          "prove": [{"seconds": 300.0, "gpus": 1}, {"seconds": 180.0, "gpus": 1}], "staging": [{"seconds": 20.0, "gpus": 1}],
          "replay": [{"seconds": 30.0, "gpus": 8}], "interior": [{"seconds": 5.0, "gpus": 4}],
          "stock": {"seconds": 975.0, "gpus": 8}}


def test_with_no_served_run_the_share_is_not_measured_and_the_given_share_is_assumed(tmp_path):
    doc = E.build(store=recorded(tmp_path).store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["run"], cs["value"], cs["mark"], cs["why"], cs["GB"]) == (None, None, None, E.NO_RUN, None)
    assert cs["given"] == {"share": 0.1, "mark": "assumed", "GB": doc["value"]} and doc["value"] == pytest.approx(YEAR_GB)
    by = {i["input"]: i for i in doc["inputs"]}
    assert by["compute_share"]["mark"] == "assumed" and E.NO_RUN in by["compute_share"]["source"]
    assert not any(i["input"].startswith("served_") or i["mark"] == "missing" for i in doc["inputs"])
    text = E.render(doc)
    assert ("\n\n**The audit's share of compute: not measured: no served run's record.** Every figure below is read at the "
            "given 10% of total compute (assumed).\n\n") in text
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in text and "## The audit's share of compute on" not in text


def test_a_served_runs_record_gives_the_share_measured_and_the_part_at_it(tmp_path):
    s = recorded(tmp_path)
    doc = E.build(served=SERVED, store=s.store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["run"], cs["mark"], cs["why"], cs["missing"], cs["serving_gpu_s"]) == (SERVED["run"], "measured", None, [], 8000.0)
    assert cs["value"] == pytest.approx(0.12, rel=1e-12)
    assert {k: (p["gpu_s"], p["mark"]) for k, p in cs["parts"].items()} == {
        "prove": (480.0, "measured"), "staging": (20.0, "measured"), "replay": (240.0, "measured"),
        "interior": (20.0, "measured"), "serving_charge": (200.0, "measured")}
    # the part at the measured share is the part the command reads at that share given; the given 10%'s is beside it
    at = E.build(share=0.12, store=s.store, root=REPO)["value"]
    assert cs["GB"] == pytest.approx(at, rel=1e-12) and cs["inside"] and cs["no_GB"] is None
    assert cs["given"] == {"share": 0.1, "mark": "assumed", "GB": doc["value"]} and doc["value"] == pytest.approx(YEAR_GB)
    by = {i["input"]: i for i in doc["inputs"]}
    assert {k: by[f"served_{k}_gpu_s"]["mark"] for k in (*E.SERVED_PARTS, "serving")} == dict.fromkeys(
        (*E.SERVED_PARTS, "serving"), "measured")
    assert (by["served_compute_share"]["mark"], by["served_compute_share"]["value"]) == ("measured", cs["value"])
    assert by["compute_share"]["mark"] == "assumed" and SERVED["run"] in by["compute_share"]["source"]
    text = E.render(doc)
    assert (f"**The audit's share of compute: 12.000% (measured)**, on {SERVED['run']}: the audit's GPU-seconds over the "
            "window's serving, 8,000 GPU-seconds, in parts prove 6.000%, staging 0.250%, replay 3.000%, interior 0.250%, "
            f"serving charge 2.500%. At that share, β = {E._part(cs['beta'])}, the compute audit's part over the world's "
            f"year is {E._n(at)} GB, beside 14,240 GB at the given 10% of total compute (assumed)") in text
    assert f"| serving_charge | {E.SERVED_PARTS['serving_charge'][0]} | 200 | 2.500% | measured | the served GPU-seconds less" in text

    # with no stock serve of the load, the serving charge is the serve tax's cost of the arm over stock, modeled
    cost = {k: v for k, v in SERVED.items() if k != "stock"} | {
        "serving_cost": 0.025 / 0.975, "serving_cost_source": "r20261110-000000-0000, load chat"}
    cs = E.build(served=cost, store=s.store, root=REPO)["compute_share"]
    assert cs["value"] == pytest.approx(0.12, rel=1e-12) and cs["mark"] == "modeled"
    assert cs["parts"]["serving_charge"]["mark"] == "modeled" and "r20261110-000000-0000, load chat" in cs["parts"][
        "serving_charge"]["source"]


def test_a_field_the_record_lacks_names_its_gap_and_no_share_is_claimed(tmp_path):
    s = recorded(tmp_path)
    lacking = {k: v for k, v in SERVED.items() if k not in ("staging", "stock")} | {
        "prove": [{"seconds": 300.0}, {"seconds": 180.0, "gpus": 1}]}
    doc = E.build(served=lacking, store=s.store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["value"], cs["mark"], cs["GB"]) == (None, None, None) and doc["value"] == pytest.approx(YEAR_GB)
    assert {g["field"]: g["lacks"] for g in cs["missing"]} == {
        "prove": ["prove.gpus"], "staging": ["staging"], "serving_charge": ["stock"]}
    assert cs["why"] == "not measured: " + "; ".join(
        f"{SERVED['run']}'s record lacks {lacks} ({E.SERVED_PARTS[k][1]})"
        for k, lacks in (("prove", "`prove.gpus`"), ("staging", "`staging`"), ("serving_charge", "`stock`")))
    assert {k: p["mark"] for k, p in cs["parts"].items()} == {
        "prove": "missing", "staging": "missing", "replay": "measured", "interior": "measured", "serving_charge": "missing"}
    by = {i["input"]: i for i in doc["inputs"]}
    assert {k: by[k]["mark"] for k in ("served_prove_gpu_s", "served_staging_gpu_s", "served_replay_gpu_s",
                                       "served_compute_share")} == {
        "served_prove_gpu_s": "missing", "served_staging_gpu_s": "missing", "served_replay_gpu_s": "measured",
        "served_compute_share": "missing"}
    assert E.SERVED_PARTS["staging"][1] in by["served_staging_gpu_s"]["source"]
    text = E.render(doc)
    assert f"\n\n**The audit's share of compute: {cs['why']}.** Every figure below is read at the given 10%" in text
    assumed = sum(i["mark"] == "assumed" for i in doc["inputs"])
    assert f"modeled, {assumed} assumed and 4 missing." in text
    assert "| staging | " + E.SERVED_PARTS["staging"][0] + " | none | none | missing | `staging`; lacks `staging`: " in text
    # the window's serving lacking: every part is unclaimed as a share, and the gap is the serve's
    cs = E.served_share({k: v for k, v in SERVED.items() if k != "serving"})
    assert cs["value"] is None and cs["missing"][0] == {"field": "serving", "lacks": ["serving"], "why": E.SERVING_GAP}
    assert {p["share"] for p in cs["parts"].values()} == {None}

    for bad, why in (({"replay": [{"seconds": 30.0, "gpus": 0}]}, "holds replay at 30.0 seconds on 0 GPUs"),
                     ({"serving": {"seconds": 0.0, "gpus": 8}}, "holds the window's serving at 0.0 GPU-seconds"),
                     ({"stock": None, "serving_cost": -1}, "holds a serving cost of -1: a fraction over -1")):
        with pytest.raises(ValueError, match=f"{SERVED['run']}'s record {why}"):
            E.served_share(SERVED | bad)
    with pytest.raises(ValueError, match="a served run's record is one JSON object, not list"):
        E.served_share([SERVED])


# ---- the stop class's verdicts -----------------------------------------------------------------------------------------------

#: the sha256 of the print and of the `--json` document on the fixtures' runs with no verdicts of record: the command's
#: output before it read any. A change elsewhere that moves the print on purpose pins the new digests here
NO_VERDICTS = {"text": "61fde301a626b1a1bf2698f6b48ce9fffe6928a444f24ee38962d5c1de0588de",
               "json": "9105f3ca602a3ea5f1463be4f67c9747c2bbc8ea7ddc7a831bde6883eef81816"}
#: made-up verdicts: a verifier, what stands between its verdicts and proved, the run that holds the records, and each
#: verdict's own run
VERIFIER = {"module": "made_up.stop_verifier", "commit": "0123456789ab" * 3 + "0123"}
OPEN = "the two-round form's soundness is not proved"
HOLDER = "r20261113-000000-a000"
ALL = "the stop record, the response tie, the request tie and the compute audit's second stage"


def made_up_verdict(piece: str, control: str | None = None, **more) -> dict:
    """A made-up verdict's record: the honest run's on ``piece`` accepted, or a ``control`` on it refused, by `VERIFIER`
    and proved by nothing."""

    n = list(E.PIECES).index(piece) + (0 if control is None else 4 + list(E.CONTROLS).index(control) * 4)
    return {"piece": piece, "control": control, "run": f"r20261113-000000-a{n + 1:03d}", "verifier": VERIFIER,
            "verdict": "accept" if control is None else "refuse", "proved_by": None, "open": OPEN} | more


def complete() -> list[dict]:
    """The four pieces accepted, and a changed id refused on the response tie and a changed `max_tokens` on the request
    tie, each by the verifier that accepted that piece."""

    return [*(made_up_verdict(p) for p in E.PIECES), made_up_verdict("response_tie", "id"),
            made_up_verdict("request_tie", "max_tokens")]


def with_verdicts(tmp: Path, verdicts: list[dict] | str, run: str = HOLDER) -> Store:
    """The fixture's rows of record, and ``run`` labelled the stop class's record holding ``verdicts`` (or that text)."""

    s = recorded(tmp, planned=False)
    s.label(s.ran(run, {E.VERDICTS_FILE: verdicts}), value=E.VERDICTS)
    return s


def digests(doc: dict) -> dict:
    return {"text": hashlib.sha256(E.render(doc).encode()).hexdigest(),
            "json": hashlib.sha256(json.dumps(doc, indent=1, ensure_ascii=False).encode()).hexdigest()}


def test_with_no_verdicts_of_record_the_print_is_as_it_was_byte_for_byte(tmp_path):
    """With nothing in the store carrying the stop class's record, the print and the document are byte for byte those
    the command gave before it read verdicts (`NO_VERDICTS`), the output count assumed as it was."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert "stop_class" not in doc and E.VERDICTS not in E.render(doc)
    assert digests(doc) == NO_VERDICTS, (
        f"the print with no verdicts of record moved: {digests(doc)}; where a change elsewhere moved it on purpose, pin "
        "these in NO_VERDICTS")
    assert {i["input"]: i["mark"] for i in doc["inputs"]}["output_count"] == "assumed"


def test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured(tmp_path):
    """Every piece accepted and every control refused by the verifier that accepted its piece, none of them proved: the
    output count's 0 is `run, not proved`, printed "run (Python verifier, not proved)" with each verdict's run and
    verifier, and the summary counts it apart from measured. The class prints `complete (run, not proved: ...; Python
    verifier for ...)`, with what is open in words and no issue's id, and no line calls it proved."""

    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    doc = E.build(store=with_verdicts(tmp_path, complete()).store, root=REPO)
    stop = doc["stop_class"]
    said = f"complete (run, not proved: {OPEN}; Python verifier for {ALL})"
    assert (stop["record"], stop["runs"], stop["complete"], stop["mark"], stop["lacks"], stop["said"]) == (
        E.VERDICTS, [HOLDER], True, E.RUN_NOT_PROVED, [], said)
    assert [(v["piece"], v["control"]) for v in stop["verdicts"]] == [
        ("stop_record", None), ("response_tie", None), ("response_tie", "id"), ("request_tie", None),
        ("request_tie", "max_tokens"), ("second_stage", None)]
    by, before = {i["input"]: i for i in doc["inputs"]}, {i["input"]: i for i in base["inputs"]}
    count = by["output_count"]
    assert (count["part"], count["value"], count["mark"]) == ("network certifier", 0, E.RUN_NOT_PROVED)
    assert count["source"].startswith(
        f"taken at 0 in the sums under `all three` and `design`: the stop class's verdicts of record, `{E.VERDICTS}` "
        f"({HOLDER}), are {said}: the stop record, `accept` on r20261113-000000-a001 by `made_up.stop_verifier` at "
        f"{VERIFIER['commit']}, not proved: {OPEN}; the response tie, `accept` on r20261113-000000-a002 by ")
    assert ("the response tie, the control with a changed id, `refuse` on r20261113-000000-a006 by `made_up.stop_verifier`"
            in count["source"])
    assert count["source"].endswith("Without them the server chooses a response's output count, which the network "
                                    "certifier's row `design, no stop block` charges 11 bits a request as its schedules")
    assert {k: i for k, i in by.items() if k != "output_count"} == {k: i for k, i in before.items() if k != "output_count"}
    assert doc["periods"] == base["periods"] and doc["value"] == base["value"]

    text = E.render(doc)
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in E.MARKS}
    assert (marks[E.RUN_NOT_PROVED], marks["measured"], marks["assumed"] + 1) == (
        1, sum(i["mark"] == "measured" for i in base["inputs"]), sum(i["mark"] == "assumed" for i in base["inputs"]))
    assert (f"of {len(doc['inputs'])} inputs, {marks['measured']} is measured, {marks['modeled']} modeled, "
            f"{marks['assumed']} assumed and 1 run (Python verifier, not proved).\n") in text
    assert f"| network certifier | output_count | 0 | bit/request | run (Python verifier, not proved) | {count['source']} |" in text
    assert (f"take it at 0, as they price no advice. The stop class's verdicts of record (`{E.VERDICTS}`, {HOLDER}): "
            f"{said}.\n") in text
    assert f"- the stop class: {HOLDER} (`{E.VERDICTS}`): 6 verdicts, {said}\n" in text
    assert [line for line in text.splitlines() if "proved" in line.replace("not proved", "")] == []

    # a piece proved and the others not: still run, not proved, and the Python verifier is the others'
    mixed = [v | {"proved_by": "Made.Up.StopRecordSound", "open": None} if v["piece"] == "stop_record" else v
             for v in complete()]
    stop = E.build(store=with_verdicts(tmp_path / "mixed", mixed).store, root=REPO)["stop_class"]
    assert (stop["mark"], stop["said"]) == (E.RUN_NOT_PROVED, f"complete (run, not proved: {OPEN}; Python verifier "
                                            "for the response tie, the request tie and the compute audit's second stage)")


def test_proved_by_on_every_piece_makes_it_measured(tmp_path):
    """Only where every piece's record names the Lean guarantee that proves its verifier is the output count's 0
    measured; it is then counted with the measured inputs and nothing prints as not proved."""

    proved = [v | {"proved_by": f"Made.Up.{v['piece']}", "open": None} for v in complete()]
    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    doc = E.build(store=with_verdicts(tmp_path, proved).store, root=REPO)
    stop = doc["stop_class"]
    assert (stop["complete"], stop["mark"], stop["said"]) == (True, "measured", (
        "complete (measured: proved by `Made.Up.stop_record`, `Made.Up.response_tie`, `Made.Up.request_tie` and "
        "`Made.Up.second_stage`)"))
    count = next(i for i in doc["inputs"] if i["input"] == "output_count")
    assert count["mark"] == "measured" and "proved by `Made.Up.second_stage`" in count["source"]
    measured = sum(i["mark"] == "measured" for i in base["inputs"])
    text = E.render(doc)
    assert f"inputs, {measured + 1} are measured, " in text and "not proved" not in text
    assert "| network certifier | output_count | 0 | bit/request | measured | taken at 0 " in text


@pytest.mark.parametrize("field", E.VERDICT_FIELDS)
def test_a_verdicts_record_missing_a_field_is_refused_by_its_name(tmp_path, field):
    lacking = {k: v for k, v in made_up_verdict("stop_record").items() if k != field}
    with pytest.raises(ValueError, match=f"^here lacks `{field}`: a verdict's record holds `piece`, `control`, `run`, "
                                         "`verifier`, `verdict`, `proved_by`, `open`"):
        E.verdict(lacking, "here")
    s = with_verdicts(tmp_path, [*complete()[:3], lacking])
    with pytest.raises(ValueError, match=f"^{HOLDER}'s `{E.VERDICTS_FILE}` line 4 lacks `{field}`"):
        E.build(store=s.store, root=REPO)


def test_a_verdicts_record_whose_field_holds_what_it_cant_is_refused(tmp_path):
    v = made_up_verdict("stop_record")
    with pytest.raises(ValueError, match="^here lacks `verifier.commit`: a verdict's verifier is its module and the commit"):
        E.verdict(v | {"verifier": {"module": "m"}}, "here")
    with pytest.raises(ValueError, match="^here lacks `verifier.module`, `verifier.commit`"):
        E.verdict(v | {"verifier": "m"}, "here")
    for change, why in (({"piece": "stop"}, "`piece` is one of stop_record, response_tie, request_tie, second_stage"),
                        ({"control": "length"}, "`control` is null on the honest run, else one of id, max_tokens"),
                        ({"run": ""}, "`run` is the id of the run that produced the verdict"),
                        ({"verdict": "pass"}, "`verdict` is one of accept, not complete, refuse"),
                        ({"proved_by": ""}, "`proved_by` is null, or the Lean guarantee whose proof covers the verifier"),
                        ({"open": None}, "`open` states what stands between the verdict and proved, while `proved_by` is "
                                         "null"),
                        *(({"open": issue}, "`open` names an issue, and the print carries no issue's id: state what is open "
                                            "in words \\(e.g. `the two-round form's soundness is not proved`\\), and put the "
                                            "issue on the board") for issue in ("PRF-419", "waits on PRF-419", "NET-386 and more"))):
        with pytest.raises(ValueError, match=f"^here holds .*: {why}$"):
            E.verdict(v | change, "here")
    assert E.verdict(v | {"proved_by": "G", "open": None, "verifier": VERIFIER | {"more": 1}}, "here")["verifier"] == VERIFIER
    assert E.verdict(v | {"open": "SHA-512's collision resistance is assumed"}, "here")["open"].startswith("SHA-512")
    with pytest.raises(ValueError, match="^here is a verdict's record, one JSON object, not list"):
        E.verdict([v], "here")
    s = with_verdicts(tmp_path, json.dumps(v) + "\n\n{not json\n")
    with pytest.raises(ValueError, match=f"^{HOLDER}'s `{E.VERDICTS_FILE}` line 3 is no JSON object"):
        E.build(store=s.store, root=REPO)


def test_without_both_negative_controls_refused_the_class_is_not_complete_and_says_what_is_missing(tmp_path):
    """The class is complete only with every piece accepted, a changed id and a changed `max_tokens` each refused, and
    each refusal by the verifier that accepted the honest run's verdict on its piece. Short of that the inputs print as
    they do with no verdicts, and the output count's line names what is missing."""

    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    honest = complete()[:4]
    other = VERIFIER | {"commit": "f" * 40}
    for name, verdicts, lacks in (
            ("none", honest, ["no negative control with a changed id", "no negative control with a changed `max_tokens`"]),
            ("one", [*honest, made_up_verdict("response_tie", "id")],
             ["no negative control with a changed `max_tokens`"]),
            ("accepted", [*complete()[:5], made_up_verdict("request_tie", "max_tokens", verdict="accept")],
             ["the control with a changed `max_tokens` on the request tie is `accept` on r20261113-000000-a011, not "
              "refused"]),
            ("other", [*complete()[:5], made_up_verdict("request_tie", "max_tokens", verifier=other)],
             ["the control with a changed `max_tokens` is refused on r20261113-000000-a011 by `made_up.stop_verifier` at "
              f"{'f' * 40}, not by a verifier that accepted the honest run's verdict on the request tie"]),
            ("unaccepted", [*complete()[:2], made_up_verdict("request_tie", verdict="not complete"), *complete()[3:]],
             ["the request tie is `not complete` on r20261113-000000-a003",
              "the control with a changed `max_tokens` is refused on r20261113-000000-a011 by `made_up.stop_verifier` at "
              f"{VERIFIER['commit']}, not by a verifier that accepted the honest run's verdict on the request tie"]),
            ("missing", complete()[1:], ["no verdict on the stop record"])):
        doc = E.build(store=with_verdicts(tmp_path / name, verdicts).store, root=REPO)
        stop = doc["stop_class"]
        assert (stop["complete"], stop["mark"], stop["lacks"]) == (False, None, lacks), name
        assert doc["inputs"] == base["inputs"], name
        text = E.render(doc)
        said = "not complete: " + "; ".join(lacks)
        assert stop["said"] == said and f"(`{E.VERDICTS}`, {HOLDER}): {said}.\n" in text, name
        assert "complete (run" not in text and "run (Python verifier" not in text, name
        assert text.split("\n## Inputs")[1] == E.render(base).split("\n## Inputs")[1], name


def test_a_verdict_two_runs_hold_or_a_superseded_run_is_refused_by_name(tmp_path):
    s = with_verdicts(tmp_path, complete())
    again = s.label(s.ran("r20261113-000000-a100", complete()[:1], name=E.VERDICTS_FILE), value=E.VERDICTS)
    with pytest.raises(ValueError, match=f"^{HOLDER} and {again} both hold a verdict on the stop record and neither is "
                                         "superseded: label the one the other replaces .*superseded_by NEWER"):
        E.build(store=s.store, root=REPO)
    s.label(again, E.SUPERSEDED_BY, HOLDER)
    doc = E.build(store=s.store, root=REPO)
    assert doc["stop_class"]["complete"] and {x["target"]: x["why"] for x in doc["refused"]} == {
        again: f"superseded by {HOLDER}"}
    assert f"- {again} (`{E.VERDICTS}`): superseded by {HOLDER}" in E.render(doc)
    with pytest.raises(ValueError, match=f"^{HOLDER} holds two verdicts on the request tie, the control with a changed "
                                         "`max_tokens`"):
        E.build(store=with_verdicts(tmp_path / "twice", [*complete(), complete()[-1]]).store, root=REPO)
    empty = recorded(tmp_path / "empty", planned=False)
    empty.label(empty.ran(HOLDER, [{}], name="out/other.jsonl"), value=E.VERDICTS)
    doc = E.build(store=empty.store, root=REPO)
    assert doc["refused"] == [{"target": HOLDER, "record": E.VERDICTS, "why": f"holds no verdicts (`{E.VERDICTS_FILE}`)"}]
    assert doc["stop_class"]["runs"] == [] and not doc["stop_class"]["complete"]
    assert f"(`{E.VERDICTS}`, no run read): not complete: no verdict on the stop record; " in E.render(doc)


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
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in out and "## Inputs" in out
    assert "- row `(i) honest`, at the planner's assumed prices: 17,800 GB, at β = 5.542%" in out
    assert "- **the network certifier's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids a response)" in out
    code, out, _ = run("--json", "--share", "0.097", "--network-check-share", "0")
    doc = json.loads(out)
    assert code == 0 and doc["schema"] == E.SCHEMA and doc["value"] == pytest.approx(15053.889111429447)
    assert doc["at_assumed_prices"]["GB"] == pytest.approx(ASSUMED * doc["value"], rel=1e-12)
    code, out, _ = run("--row", "(i) honest")
    assert code == 0 and "- **the compute audit's part, at the planner's assumed prices: 17,800 GB** (modeled)" in out
    code, out, _ = run("--json", "--response-ids", "4096")
    assert code == 0 and json.loads(out)["periods"][-1]["together"]["scan"]["certifier_GB"] == pytest.approx(59_645_644, abs=0.5)
    # at the δ the calculator assumes by default, a scan's 2^-20, every row printed sums to 1 or more: no amount of the
    # network certifier's, and its ε over the world's year
    code, out, _ = run("--json", "--scan-delta", "1/1048576")
    doc = json.loads(out)
    assert code == 0 and doc["value"] == pytest.approx(YEAR_GB) and doc["certifier"]["scan_delta"]["value"] == 2 ** -20
    assert doc["certifier"]["scan_delta"]["source"] == "each execution phase's scan's δ, given (--scan-delta)"
    assert min(p["eps"] for p in doc["periods"]) == pytest.approx(56.3, abs=0.05)
    assert doc["periods"][-1]["eps"] == pytest.approx(11_012_646, abs=0.5)
    assert {t[k] for p in doc["periods"] for t in p["together"].values() for k in ("certifier_GB", "GB")} == {None}
    code, out, _ = run("--scan-delta", "1/1048576")
    assert code == 0 and "- **the compute audit's part: 14,240 GB** (modeled)" in out
    assert ("- **the network certifier's part: no bound at this period** (modeled, on an assumed class of at most 1,024 ids a "
            "response). Its ε over the world's year is 11,012,646, and at 1 or more it bounds nothing") in out
    assert "50,753,639" not in out and out.count("\n| world | year | ") == 2
    code, out, err = run("--scan-delta", "2")
    assert code == 2 and out == "" and "verity bench exfiltration: " in err and "the scan's δ in (0, 1)" in err
    code, out, err = run("--row", "(iv)")
    assert code == 2 and out == "" and "verity bench exfiltration: no run of record holds the planner's rows" in err
    # a served run's record from its reader: the share it measures, beside the given one
    (served := tmp_path / "served.json").write_text(json.dumps(SERVED))
    code, out, _ = run("--served-run", str(served))
    assert code == 0 and f"**The audit's share of compute: 12.000% (measured)**, on {SERVED['run']}" in out
    code, out, _ = run("--json", "--served-run", str(served))
    assert code == 0 and json.loads(out)["compute_share"]["value"] == pytest.approx(0.12)
    code, out, err = run("--served-run", str(tmp_path / "none.json"))
    assert code == 2 and out == "" and "verity bench exfiltration: " in err and "none.json" in err
    # this machine's catalog alone holds no label
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "alone"))
    code, _, err = run("--local-only")
    assert code == 2 and "no planner rows of record: nothing in the store carries" in err
