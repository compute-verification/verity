"""`verity bench exfiltration` with no GPU and no node: the rule that reads a scope's rows off at a share of total
compute and the rule that reads how its proving is priced from the row's plan, on small rows; then the store side on a
real store with a directory remote that holds the planner's runs as `fixtures/exfiltration/` holds them (`records.json`
names the runs they were cut from: one run's rows, cut to the fields the command reads and to the case it takes plus
two it must pass over; the run of that case's plan over the world's year, its plan cut to each cell's block and
share; and a run whose rows hold their own plans, cut to two scopes): the figures a hand computation gave on those
rows. The runs' case is row (i); here its rows stand for the command's row, the plan of record's, under that row's name
(`as_row`), and row (i) is made up from them at `ASSUMED` times their bound (`as_assumed`), printed only under its
label, with what prices each row's proving or that its rows don't record it. Then the network attester's part beside
them at the requests its calculator counts by default, the sampling plan's; its figures of record at two release schedules, held to its own document; what rests on the set-time card printed
only under the calculator's label; the output count beside the sums that take it at 0, which the stop verdicts'
reader's answer names, and it names none while the reader asks the rule without a schedule; every figure's error, summed
over everything it covers, at the scans' δ the term states, and no amount where it is 1 or more, on term documents
with made-up errors; every input's mark; each refusal by name; and the command."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from exfiltration_helpers import (
    ASSUMED,
    PLANNED,
    REPO,
    SERVED,
    YEAR_GB,
    Store,
    as_row,
    recorded,
    rows,
    term,
)
from verity_bench import exfiltration as E


def test_verity_bench_exfiltration_is_the_commands_runner(tmp_path, monkeypatch, capsys):
    """`verity bench exfiltration` runs benchmarks/exfiltration/run.py, which prints the figure from the store alone."""

    s = recorded(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    # the holes' block over the fixture runs' made-up commit, which no repository holds: test_spot_status.py decides them
    monkeypatch.setattr(E.spot_status, "commit_known", lambda root, commit: False)
    monkeypatch.setattr(E.spot_status, "LOCK", Path(__file__).parent / "fixtures" / "spot_status" / "lock.json")

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
    assert "\n\n## SPOT's resampling holes in the code that ran\n" in out and "- label: **SPOT**\n" in out
    assert "\n- redrawn sample: " in out and "\n- re-committed execution phase: " in out and ": closed\n" not in out
    code, out, _ = run("--json", "--share", "0.097", "--network-check-share", "0")
    doc = json.loads(out)
    assert code == 0 and doc["schema"] == E.SCHEMA and doc["value"] == pytest.approx(15053.889111429447)
    assert doc["spot_holes"]["label"] == "SPOT" and doc["spot_holes"]["runs"] == len(doc["runs"])
    assert doc["at_assumed_prices"]["GB"] == pytest.approx(ASSUMED * doc["value"], rel=1e-12)
    code, out, _ = run("--row", "(i) honest")
    assert code == 0 and "- **the compute audit's part, at the planner's assumed prices: 17,800 GB** (modeled)" in out
    code, out, _ = run("--json", "--response-ids", "4096")
    assert code == 0 and json.loads(out)["periods"][-1]["together"]["scan"]["certifier_GB"] == pytest.approx(59_645_644, abs=0.5)
    # at the δ the calculator assumes by default, a scan's 2^-20, every row printed sums to 1 or more: no amount of the
    # network attester's, and its ε over the world's year
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


def test_under_a_research_run_the_runner_records_its_three_figures_and_prints_the_same(tmp_path, monkeypatch, capsys):
    """Under a research run, `verity bench exfiltration` writes run.py's metrics into the run's result through the kit:
    the headline figure, the compute audit's part and the network attester's, each over every monitoring period of the
    document it has a value over at the share in GB, with its `period` and `over`, the world's year last and as it was
    (so a reader keyed by name alone reads it), its mark heading its first input and its error over its period in the
    next, then the runs of record it read; the document as its evidence and what ran in the fingerprint. What it prints
    is the same in a run as outside one, where it writes nothing. A period with no value is left out, but the world's
    year, where the network attester's part at an ε of 1 or more has no value and says why."""

    s = recorded(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261113-000000-0000")
    monkeypatch.setattr(E.spot_status, "commit_known", lambda root, commit: False)
    monkeypatch.setattr(E.spot_status, "LOCK", Path(__file__).parent / "fixtures" / "spot_status" / "lock.json")
    monkeypatch.setattr(E.kit, "host_stamp", lambda: {"card": "none", "gpus": [], "driver": None, "cuda": None,
                                                      "node": "cpu-node-0", "commit": "c" * 40, "why": "no nvidia-smi"})
    writers, writer = [], E.kit.writer
    monkeypatch.setattr(E.kit, "writer", lambda out: writers.append(out) or writer(out))
    runner = REPO / "benchmarks" / "exfiltration" / "run.py"
    metrics = runpy.run_path(str(runner))["METRICS"]

    def run(*argv: str, within: Path | None = None) -> str:
        if within is None:
            monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
        else:
            monkeypatch.setenv("RESEARCH_RUN_DIR", str(within))
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(runner), run_name="__main__")
        assert done.value.code == 0
        return capsys.readouterr().out

    outside, doc = run(), json.loads(run("--json"))
    assert writers == []
    first = tmp_path / "run-1"
    assert run(within=first) == outside and len(writers) == 1
    got = json.loads((first / "result.json").read_text())
    assert got["run_id"] == "r20261113-000000-0000" and got["measurement_files"] == [E.DOCUMENT]
    assert json.loads((first / E.DOCUMENT).read_text()) == doc
    assert got["workload_fingerprint"] == {
        "source_sha": None, "benchmark": "exfiltration", "row": E.ROW, "step": doc["compute_audit"][0]["step"],
        "variant": None, "share": E.SHARE, "network_check_share": E.NETWORK_CHECK_SHARE, "response_ids": E.RESPONSE_IDS,
        "scan_delta": doc["certifier"]["scan_delta"]["value"], "served_run": None, "runs": doc["runs"]}
    assert got["validation"]["status"] == "passed" and got["validation"]["evidence"] == [E.DOCUMENT]
    # only what the document has a value over, the world's year last: the fixtures' rows of record hold the compute
    # audit's part over the datacenter's year and the world's alone
    headline, audit_gb, net_gb = metrics
    assert [(m["name"], m["period"], m["over"]) for m in got["measurements"]] == [
        (net_gb, "week", "datacenter"), (net_gb, "month", "datacenter"),
        (headline, "year", "datacenter"), (audit_gb, "year", "datacenter"), (net_gb, "year", "datacenter"),
        (net_gb, "week", "world"), (net_gb, "month", "world"),
        (headline, "year", "world"), (audit_gb, "year", "world"), (net_gb, "year", "world")]
    assert all(m["value"] is not None and "why" not in m for m in got["measurements"])
    # a reader that keys them by name alone, the last of a name winning, reads the world's year
    by = {m["name"]: m for m in got["measurements"]}
    assert list(metrics) == ["benchmarks/exfiltration-gb-world-year", "benchmarks/exfiltration-compute-audit-gb",
                             "benchmarks/exfiltration-network-certifier-gb"]
    assert [by[name] for name in metrics] == got["measurements"][-3:]
    head, audit, net = (by[name] for name in metrics)
    year = doc["periods"][-1]
    assert {(m["period"], m["over"]) for m in by.values()} == {("year", "world")}
    assert {m["unit"] for m in by.values()} == {"GB a world year at 10% of compute"}
    assert [list(m)[:6] for m in by.values()] == [["name", "value", "unit", "mark", "period", "over"]] * 3
    assert [m["mark"] for m in by.values()] == ["modeled"] * 3
    assert head["value"] == audit["value"] == doc["value"] == pytest.approx(YEAR_GB)
    assert net["value"] == year["together"]["scan"]["certifier_GB"] == pytest.approx(50_753_639, abs=0.5)
    assert head["inputs"][0].startswith("modeled: the headline figure, the planner's row `(ii) honest`, read where its "
                                        "total is 10% of compute, β = ")
    assert audit["inputs"][0].startswith("modeled: the compute audit's part, the planner's row `(ii) honest`, ")
    assert head["inputs"][1:] == audit["inputs"][1:] == ["its error: δ = 0.01 over the world's year, the compute audit's",
                                                         *doc["runs"]]
    assert net["inputs"] == [
        ("modeled (run, not proved): the network certifier's term of record, its `scan` row as the scan's check charges "
        "it today at D_j = 2, on an assumed class of at most 1,024 ids a response"),
        (f"its error: ε = {E._p(year['eps'])} over the world's year, at a scan's δ of "
        f"{E._p(doc['certifier']['scan_delta']['value'])} (assumed)")]
    assert all("why" not in m and m["ci95"] is None and m["host"]["node"] == "cpu-node-0" for m in by.values())
    # every period of the document, each figure the document's own over it
    at = {(m["name"], m["period"], m["over"]): m for m in got["measurements"]}
    for p in doc["periods"]:
        scope, span = f"{p['compute']} {p['period']}", f"{p['compute']}'s {p['period']}"
        net = at[net_gb, p["period"], p["compute"]]
        assert (net["unit"], net["mark"]) == (f"GB a {scope} at 10% of compute", "modeled")
        assert net["value"] == p["together"]["scan"]["certifier_GB"]
        assert net["inputs"][1] == (f"its error: ε = {E._p(p['eps'])} over the {span}, at a scan's δ of "
                                    f"{E._p(doc['certifier']['scan_delta']['value'])} (assumed)")
        if p["compute_audit"] is None:
            assert (headline, p["period"], p["compute"]) not in at and (audit_gb, p["period"], p["compute"]) not in at
            continue
        head, audit = at[headline, p["period"], p["compute"]], at[audit_gb, p["period"], p["compute"]]
        assert {(m["unit"], m["mark"]) for m in (head, audit)} == {(f"GB a {scope} at 10% of compute", "modeled")}
        assert head["value"] == audit["value"] == p["compute_audit"]["GB"]
        assert head["inputs"][1:] == audit["inputs"][1:] == [
            f"its error: δ = {E._n(p['compute_audit']['delta'])} over the {span}, the compute audit's", *doc["runs"]]

    # where every ε is 1 or more, the network attester's part is written over the world's year alone, with no value
    second = tmp_path / "run-2"
    run("--scan-delta", "1/1048576", within=second)
    written = json.loads((second / "result.json").read_text())["measurements"]
    nets = [m for m in written if m["name"] == net_gb]
    assert [(m["period"], m["over"]) for m in nets] == [("year", "world")] and written[-1] == nets[0]
    assert nets[0]["value"] is None and nets[0]["why"] == f"{E.NO_BOUND}: ε = 11,012,646 over the world's year"
    assert [(m["name"], m["period"], m["over"]) for m in written if m["value"] is not None] == [
        (headline, "year", "datacenter"), (audit_gb, "year", "datacenter"), (headline, "year", "world"),
        (audit_gb, "year", "world")]
    with pytest.raises(ValueError, match="^no figure of the document is named total: a metric records one of headline, "):
        E.measurements(doc, {"benchmarks/x": "total"})


def test_the_legend_line_shows_a_rows_mark_as_its_calculator_gives_it(tmp_path, monkeypatch):
    """A row of the network attester's legend that carries a mark (the calculator's `scan` row's accept, run, not
    proved) prints it beside the row's standing, as the mark, in that line alone; the network attester's recorded
    figure carries it beside its own mark. A mark this command has no name for is refused by the row's name."""

    lb = term()
    document = lb.document

    def marked(mark: str):
        def made(*a, **kw) -> dict:
            doc = document(*a, **kw)
            return doc | {"legend": doc["legend"] | {"scan": doc["legend"]["scan"] | {"mark": mark}}}

        return made

    monkeypatch.setattr(lb, "document", marked(None))
    base = E.render(E.build(store=recorded(tmp_path / "base").fresh(), root=REPO)).splitlines()
    monkeypatch.setattr(lb, "document", marked(E.RUN_NOT_PROVED))
    doc = E.build(store=recorded(tmp_path / "marked").fresh(), root=REPO)
    scan = doc["certifier"]["legend"]["scan"]
    assert [line for line in E.render(doc).splitlines() if line not in base] == [
        f"- scan ({scan['standing']}; run, not proved): {scan['counts']}"]
    net = {(m["period"], m["over"]): m for m in E.measurements(doc, {"benchmarks/x": "network_certifier"})}["year", "world"]
    assert net["mark"] == "modeled" and net["inputs"][0].startswith("modeled (run, not proved): the network certifier's ")
    monkeypatch.setattr(lb, "document", marked("proved"))
    with pytest.raises(ValueError, match="^the network certifier's rows scan carry a mark this command doesn't print"):
        E.build(store=recorded(tmp_path / "refused").fresh(), root=REPO)


#: a row's verification phase as the planner writes it, with its binding bound: the row's own at the world's year,
#: but what Lean proves counted in compressions, made up so that both of Lean's cases print
PHASE = {
    "name": "datacenter", "seconds": 600, "machines": 12500.0, "replay_blocks": 90237979.8, "sampled": 55,
    "sessions": 406359.6, "limit": "its largest session in tables is 4 of its type, past the GPU prover's 1",
    "binding": {
        "what": "the two-stage first-hit finder's bound by classes of gates", "status": "modeled",
        "proof_blocks": 207715585.0, "tables": 4,
        "assumed": [
            {"input": "classes of gates a sampled proof block", "value": 4130, "unit": "class", "source": "no count gives it"},
            {"input": "symbols of a table's level-0 codewords", "value": 8388608, "unit": "symbol", "source": "the most"},
            {"input": "one commitment, one sample; every committed window is presented", "value": None, "unit": "",
             "source": "what a sum over verification phases takes of the prover"}],
        "design": {"sessions": 406360, "classes": 857865366050, "k": 49375943, "reruns": 10368948031,
                   "calls_a_run": "2^80.00", "rounds": 1788796724, "phases": 197078976,
                   "longest_call": {"bytes": 65536, "compressions": 513}},
        "counted": {
            "calls": {"phase": {"bound": "2^-101.08", "centibits": 10108}, "phases": {"bound": "2^-73.53", "centibits": 7353},
                      "lean": {"theorem": None, "nearest": "εclsAt_le_table1", "bits": 100,
                               "past": {"e": {"design": 10368948031, "instance": 2592237061}}}},
            "compressions": {"phase": {"bound": "2^-92.08", "centibits": 9208},
                             "phases": {"bound": "2^-64.52", "centibits": 6452},
                             "lean": {"theorem": "εclsAt_le_table1_513", "bits": 100}}}}}


def test_the_security_level_is_the_rows_binding_bound_named_by_what_it_is_over(tmp_path):
    """Beside the GB the command prints the binding bound its recorded planner row holds: over one
    verification phase and over a world-year of phases, each counted in calls and in compressions and named by what
    it is over, marked modeled; the inputs it assumes, each listed and marked assumed; and what Lean proves of each,
    by its theorem's name, or that nothing is proved at these sizes. It computes none of it, moves no GB, and says so
    where the rows hold no bound."""

    def built(phase: dict | None) -> dict:
        s = Store(tmp_path / str(len(list(tmp_path.iterdir()))))
        s.label(s.ran(PLANNED, [as_row(r) | ({} if phase is None else {"verification_phase": phase})
                                for r in rows(PLANNED)]))
        return E.build(store=s.fresh(), root=REPO)

    bare, doc = built(None), built(PHASE)
    assert doc["value"] == bare["value"] and len(doc["inputs"]) == len(bare["inputs"]) + 3
    bound = next(p for p in doc["compute_audit"] if p["scope"] == "world year")["binding"]
    assert (bound["run"], bound["beta"], bound["scope"], bound["sampled"]) == (PLANNED, 0.055, "world year", 55)
    assert bound["counted"] == PHASE["binding"]["counted"] and bound["design"] == PHASE["binding"]["design"]
    assert all(p["binding"] is None for p in bare["compute_audit"])
    assumed = [i for i in doc["inputs"] if i["part"] == E.SECURITY]
    assert [(i["input"], i["value"], i["mark"]) for i in assumed] == [
        ("classes of gates a sampled proof block", 4130, "assumed"),
        ("symbols of a table's level-0 codewords", 8388608, "assumed"),
        ("one commitment, one sample; every committed window is presented", None, "assumed")]
    assert all(i["source"].startswith(f"`{E.RECORD}` ({PLANNED}) `verification_phase.binding`: ") for i in assumed)

    lines = [line for line in E.render(doc).splitlines() if E.SECURITY in line and line.startswith("- ")]
    assert len(lines) == 3 and lines[0].startswith(
        "- **the security level** (modeled): the probability that an accepted run opens a value off the committed "
        "trace, by the bound Lean states of a two-stage sample counted by classes of gates "
        "(`CompiledTwo.eventTwoX_le_cls_at_budget`), every run at 2^80.00 calls of SHA-512, at "
        f"the row's own design at the pinned budget nearest, β = 5.500% ({PLANNED}). **Over one verification phase: at "
        "most 2^-101.08** counted in calls, as the Lean counts, and 2^-92.08 in compressions. **Over a world-year of "
        "phases**, 197,078,976 of them: **at most 2^-73.53** in calls and 2^-64.52 in compressions. A verification "
        "phase is a datacenter's 600 s on 12,500 machines: its first stage samples 55 replay blocks, and its second "
        "207,715,585 of their proof blocks in 406,360 sessions, the largest of 4 tables.")
    assert lines[1] == (
        "- the security level assumes: 4,130 classes of gates a sampled proof block; 8,388,608 symbols of a table's "
        "level-0 codewords; one commitment, one sample; every committed window is presented. It counts each session's "
        "knowledge error and the run's own opening checks as 0.")
    assert lines[2] == (
        "- what Lean proves of the security level: over one verification phase counted in calls, not proved at these "
        "sizes: the nearest instance, `Claims.Flock.Numbers.εclsAt_le_table1`, at most 2^-100, has reruns a session "
        "(e_s) at most 2,592,237,061 there and 10,368,948,031 here. Counted in compressions, "
        "`Claims.Flock.Numbers.εclsAt_le_table1_513` holds the part of it that is linear in the prover's calls to at "
        "most 2^-100, beside each session's knowledge error and the rewinding part. Over a world-year of phases, not "
        "proved at these sizes: no theorem sums "
        "a monitoring period's verification phases.")
    # no figure of it is called bits: each is a probability over what its clause names
    assert not any(" bits" in line for line in lines)
    none = [line for line in E.render(bare).splitlines() if E.SECURITY in line]
    assert none == [("- **the security level**: not read. The rows of record hold no binding bound "
                    "(`verification_phase.binding`): the planner wrote them before it computed one. Run it again and "
                    f"label its run (`research data label RUN {E.RUN_OF_RECORD} {E.RECORD} --by LANE`).")]
