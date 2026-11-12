"""`verity bench certifier_term --out DIR` (`run.py`): the result it writes beside what it prints.

- what it prints, as text and as `--json`, is the calculator's own to the character, with `--out` and without;
- the result is the harness kit's: its schema, one measurement a name, each in the kit's fields under the run and
  the host stamp, and the directory holds nothing else;
- the names are regular and every one says modeled: the term of record, and for each scope and monitoring period its
  row's free, capacity, verified and error, in bits, bits, bits and a probability;
- free is the document's term for the scan at two release schedules, the error its execution phases' scans' δ summed,
  and the metric the document's value, each held to the calculator's table too;
- the metric and free stand against the document's target as the document's own multiple of it says, and no other
  measurement has a target or a verdict;
- capacity is read from a registered uplink's `beta`, one way: the scope's machines times it times the period's
  ticks, and verified is capacity less free; with no registration both have no value and say why;
- a row whose error is 1 or more has no bound at its period: its free and verified, and the metric where the row is
  the term of record's, have no value and say why, and its error is written. At the calculator's default δ those are
  the datacenter's and the world's rows; at the δ of the result of record every row carries its amounts;
- every measurement's inputs name the class the term is at and the scan's δ its error is at;
- a registration that isn't the model's machine, a file that is no log, and a second result in one directory are
  refused."""
from __future__ import annotations

import json
import runpy
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run as R  # noqa: E402
from verity.hardware import registration  # noqa: E402
from verity.hardware.site import OUTSIDE, TICK, Link, Loc  # noqa: E402
from verity_bench import kit  # noqa: E402

LB = R.calculator()
RUN = "r20000101-000000-aaaa"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "c" * 40}
#: a document that reads no planner and asks git nothing: the world's requests given, and its commit named
GIVEN = ["--commit", "c0ffee", "--world-requests", "1e16"]
#: each scope and monitoring period's execution phases, as the calculator's table counts them
PHASES = {(r.scope.name, r.period): r.phases for r in LB.table(LB.inputs(world_requests=1e16)) if r.name == "scan"}
#: the error the result of record is run at over the world's year, and the scan's δ that gives it: that error over
#: the world year's execution phases
AT_MOST = Fraction(1, 2 ** 20)
DELTA = AT_MOST / PHASES["world", "year"]
#: `GIVEN` at that δ, under which every row is bounded
BOUNDED = [*GIVEN, "--scan-delta", str(DELTA)]
#: each amount's unit, by its name's last word
AMOUNTS = {"free": "bit", "capacity": "bit", "verified": "bit", "error": "probability"}
#: the measurements of a result with every scope and monitoring period: the term of record, and each row's amounts
EVERY = 1 + len(LB.SCOPES) * len(LB.PERIODS) * len(AMOUNTS)
#: the uplink's two halves' capacities in bits a tick: the host's side, which is the boundary link, and the outside's
BETA, BEYOND = 400, 100


@pytest.fixture(autouse=True)
def stamped(monkeypatch):
    """The kit's writer under a run of its own and with no run directory, so that the result is the given directory's,
    stamping with `STAMP` rather than asking the host."""
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(kit, "host_stamp", lambda: dict(STAMP))


def bench(capsys, *argv: str) -> str:
    """What the runner prints on ``argv``, which it exits 0 on."""
    assert R.main(list(argv)) == 0
    return capsys.readouterr().out


def result(out: Path) -> dict[str, dict]:
    """The measurements of the result in ``out``, by name, in the order written."""
    doc = json.loads((out / kit.RESULT).read_text())
    assert (doc["schema"], doc["run_id"], sorted(doc)) == (kit.RESULT_SCHEMA, RUN, ["measurements", "run_id", "schema"])
    return {m["name"]: m for m in doc["measurements"]}


def document(capsys, *argv: str) -> dict:
    """The calculator's own document on ``argv``, as it prints it."""
    assert LB.main([*argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def scans(doc: dict) -> dict[tuple[str, str], dict]:
    """The document's rows of the scan, by scope and monitoring period."""
    return {(r["scope"], r["period"]): r for r in doc["rows"] if r["row"] == "scan"}


def absent(m: dict) -> list[str]:
    """Why a measurement has no value, as its inputs say it."""
    return [i for i in m["inputs"] if i.startswith("absent: ")]


def log(tmp_path: Path, beta: int = BETA, *more: registration.Entry, composite: str = "onsite",
        links: tuple[Link, ...] | None = None, name: str = "registration") -> Path:
    """A committed registration log, in the file ``name``, of one host behind a network certifier that splits its
    uplink, the host's side at ``beta`` bits a tick, with a memory challenger's link, which is no boundary link, and
    a day's declaration, which is no site entry; or of ``links`` in place of those three."""
    links = links or (Link("uplink0", ("node0", "cert0"), beta, 100_000), Link("uplink0-out", ("cert0", OUTSIDE), BEYOND, 0),
                      Link("chal0-link", ("chal0", "node0"), 7, 1_000_000))
    first = registration.site(locs=(Loc("site0", "site"), Loc("node0", "host", "site0"),
                                    Loc("cert0", "certifier", "site0", "auditor", key="dummy-cert0-key"),
                                    Loc("chal0", "challenger", "site0", "auditor")), links=links)
    declared = registration.Declaration(0, "0" * 128)
    path = tmp_path / f"{name}.json"
    path.write_bytes(registration.committed(composite, registration.chain(composite, [first, declared, *more])))
    return path


def test_what_the_command_prints_is_the_calculators_to_the_character(tmp_path, capsys, monkeypatch):
    for i, argv in enumerate(([], ["--json"], ["--scope", "node", "--period", "week", "--failure-rate", "1/1000"],
                              ["--json", "--requests", "census", "--lanes", "512", "--target-gb", "150"],
                              ["--scan-delta", str(DELTA), "--response-ids", str(LB.RESPONSE_IDS // 2)])):
        assert LB.main([*GIVEN, *argv]) == 0
        alone = capsys.readouterr()
        assert bench(capsys, *GIVEN, *argv) == alone.out
        assert bench(capsys, "--out", str(tmp_path / str(i)), *GIVEN, *argv) == alone.out
        assert bench(capsys, *GIVEN, *argv, "--registration", str(log(tmp_path)), "--out", str(tmp_path / f"{i}r")) == alone.out
        assert sorted(p.name for p in (tmp_path / str(i)).iterdir()) == [kit.RESULT]
    # as `verity bench` runs it, with the result written beside
    monkeypatch.setattr(sys, "argv", ["run.py", "--json", *GIVEN, "--out", str(tmp_path / "main")])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(HERE.parent / "run.py"), run_name="__main__")
    assert done.value.code == 0 and json.loads(capsys.readouterr().out) == document(capsys, *GIVEN)
    assert list(result(tmp_path / "main")) == list(result(tmp_path / "1"))


def test_the_result_is_the_kits_and_nothing_else_is_written(tmp_path, capsys, monkeypatch):
    written = []
    measurement = kit.Writer.measurement
    monkeypatch.setattr(kit.Writer, "measurement", lambda self, **m: written.append(measurement(self, **m)) or written[-1])
    fields = set(kit.writer(tmp_path / "another").measurement(name="a measurement", value=1, unit="bit"))
    for at, argv in enumerate((GIVEN, BOUNDED)):
        out = tmp_path / str(at)
        del written[:]
        bench(capsys, "--out", str(out), *argv)
        # every measurement of the file is one the kit's writer returned, in its fields, under this run and its stamp
        assert json.loads((out / kit.RESULT).read_text())["measurements"] == written and len(written) == EVERY
        for m in result(out).values():
            assert set(m) == fields and (m["run"], m["commit"], m["host"], m["ci95"]) == (RUN, STAMP["commit"], STAMP, None)
        assert sorted(p.name for p in out.iterdir()) == [kit.RESULT]


def test_the_names_are_regular_and_every_one_says_modeled(tmp_path, capsys):
    bench(capsys, "--out", str(tmp_path), *GIVEN)
    got = result(tmp_path)
    assert list(got) == ["certifier_term/modeled/metric",
                         *(f"certifier_term/modeled/{scope}/{period}/{amount}"
                           for scope in LB.SCOPES for period in LB.PERIODS for amount in AMOUNTS)]
    assert all(name.split("/")[:2] == ["certifier_term", "modeled"] for name in got)
    assert all(m["unit"] == AMOUNTS.get(name.rsplit("/", 1)[1], "GB") for name, m in got.items())
    # the shared fields' last words, the error's among them
    assert R.ERROR == "error"
    assert {name.rsplit("/", 1)[1] for name in got} == {"metric", "free", "capacity", "verified", "error"}
    # only the rows the command was asked for, with the term of record
    bench(capsys, "--out", str(tmp_path / "one"), *GIVEN, "--scope", "datacenter", "--period", "month")
    assert list(result(tmp_path / "one")) == ["certifier_term/modeled/metric",
                                              *(f"certifier_term/modeled/datacenter/month/{amount}" for amount in AMOUNTS)]


def test_free_is_the_documents_term_and_the_error_its_phases_deltas_summed(tmp_path, capsys):
    bench(capsys, "--out", str(tmp_path), *BOUNDED)
    got, doc = result(tmp_path), document(capsys, *BOUNDED)
    rows = scans(doc)
    table = LB.table(LB.inputs(world_requests=1e16, scan_delta=DELTA))
    exact = {(r.scope.name, r.period): r for r in table if r.name == "scan"}
    assert list(rows) == [(scope, period) for scope in LB.SCOPES for period in LB.PERIODS] and LB.OF_RECORD == 2
    for (scope, period), r in rows.items():
        name, row = f"certifier_term/modeled/{scope}/{period}", exact[scope, period]
        # the term at two release schedules, and the union of the scans' δ, as the table has them
        assert got[f"{name}/free"]["value"] == r["term"]["2"]["bits"] == float(row.terms[2].certifier)
        error = got[f"{name}/error"]
        assert error["value"] == r["eps"] == float(row.eps) == float(row.phases * DELTA) < 1
        # the error's inputs name the calculator's own key for it
        assert error["inputs"][1] == R.UNION and "eps, the calculator's key: a union bound over the period's" in R.UNION
        assert got[f"{name}/free"]["inputs"][0].startswith("modeled: benchmarks/network_traces/leak_bounds.py's row scan "
                                                           f"at D_j = 2, scope {scope}, period {period}, ")
        # the document's inputs, each with its value, its mark and its source, the scan's δ as the command gives it
        assert got[f"{name}/free"]["inputs"][1:] == [R.at_delta(LB, doc) if k == "scan_delta" else R.stated(k, i)
                                                     for k, i in doc["inputs"].items()]
    # the term of record: the world's year's, in GB
    metric, year = got["certifier_term/modeled/metric"], got["certifier_term/modeled/world/year/free"]
    assert (metric["value"], metric["unit"]) == (doc["value"], "GB")
    assert metric["value"] == float(exact["world", "year"].terms[2].certifier / LB.GB)
    assert metric["inputs"][0] == f"modeled: {doc['name']}" and metric["inputs"][1:] == year["inputs"][1:]


def test_the_metric_and_free_stand_against_the_documents_target(tmp_path, capsys):
    for at, target_gb in enumerate((LB.TARGET_GB, 1e-3, 1e12)):
        argv = [] if target_gb == LB.TARGET_GB else ["--target-gb", str(target_gb)]
        bench(capsys, "--out", str(tmp_path / str(at)), *BOUNDED, *argv)
        got = result(tmp_path / str(at))
        said = {at: r["term"]["2"]["of_target"] for at, r in scans(document(capsys, *BOUNDED, *argv)).items()}
        for (scope, period), multiple in said.items():
            free = got[f"certifier_term/modeled/{scope}/{period}/free"]
            assert free["target"] == target_gb * LB.GB
            assert free["verdict"] == ("over target" if multiple > 1 else "within target")
        metric = got["certifier_term/modeled/metric"]
        assert (metric["target"], metric["verdict"]) == (
            target_gb, "over target" if said["world", "year"] > 1 else "within target")
        # a target under every row's term, and one over every row's
        assert at == 0 or {m["verdict"] for m in got.values()} == {None, "over target" if at == 1 else "within target"}
        assert all((m["target"], m["verdict"]) == (None, None) for name, m in got.items()
                   if name.rsplit("/", 1)[1] in ("capacity", "verified", "error"))


def test_capacity_is_read_from_the_registered_uplink_and_verified_is_capacity_less_free(tmp_path, capsys):
    path = log(tmp_path)
    bench(capsys, "--out", str(tmp_path / "out"), "--registration", str(path), *BOUNDED)
    got, doc = result(tmp_path / "out"), document(capsys, *BOUNDED)
    basis = [R.at_class(doc), R.at_delta(LB, doc)]
    composite, entries = registration.log_of(json.loads(path.read_text()))
    scopes = LB.scopes(LB.inputs(world_requests=1e16))
    assert TICK == Fraction(1, 10 ** 9) and [s.name for s in scopes] == list(LB.SCOPES) and scopes[0].machines == 1
    for scope in scopes:
        for period, seconds in LB.PERIODS.items():
            name = f"certifier_term/modeled/{scope.name}/{period}"
            capacity, free, verified = (got[f"{name}/{amount}"] for amount in ("capacity", "free", "verified"))
            # each machine's one certified boundary link, the host's side of its uplink, over the period's nanoseconds
            assert capacity["value"] == scope.machines * BETA * seconds * 10 ** 9 and type(capacity["value"]) is int
            assert verified["value"] == capacity["value"] - free["value"]
            assert verified["inputs"] == [f"{name}/capacity", f"{name}/free", *basis]
            assert capacity["inputs"] == [
                f"registration {registration.head(composite, entries)}, read as given: host node0's certified boundary "
                f"link(s) uplink0 at beta {BETA}, in bits each way a tick of 1/1000000000 s (verity.hardware.site)",
                R.SCALED, f"modeled: benchmarks/network_traces/leak_bounds.py's row scan at D_j = 2, scope {scope.name}, "
                f"period {period}, request count given", *basis]
    assert "beta is each way and the term is egress, so one direction" in R.SCALED
    # the registration's own number: another beta is another capacity, and nothing else moves
    bench(capsys, "--out", str(tmp_path / "twice"), "--registration", str(log(tmp_path, 2 * BETA)), *BOUNDED)
    for name, m in result(tmp_path / "twice").items():
        if name.endswith("/capacity"):
            assert m["value"] == 2 * got[name]["value"]
        elif not name.endswith("/verified"):
            assert m == got[name]


def test_with_no_registration_capacity_and_verified_have_no_value_and_say_why(tmp_path, capsys):
    bench(capsys, "--out", str(tmp_path), *BOUNDED)
    got, doc = result(tmp_path), document(capsys, *BOUNDED)
    basis = [R.at_class(doc), R.at_delta(LB, doc)]
    for scope in LB.SCOPES:
        for period in LB.PERIODS:
            name = f"certifier_term/modeled/{scope}/{period}"
            capacity, verified = got[f"{name}/capacity"], got[f"{name}/verified"]
            assert (capacity["value"], capacity["inputs"][0], capacity["inputs"][2:]) == (None, R.NO_REGISTRATION, basis)
            assert capacity["inputs"][1].startswith("modeled: benchmarks/network_traces/leak_bounds.py's row scan at ")
            assert verified["value"] is None
            assert verified["inputs"] == [f"{name}/capacity", f"{name}/free", R.NO_REGISTRATION, *basis]
    assert R.NO_REGISTRATION.startswith("absent: no registration is given")
    assert sum(m["value"] is None for m in got.values()) == 2 * len(LB.SCOPES) * len(LB.PERIODS)


def test_a_row_whose_error_is_1_or_more_carries_no_amount(tmp_path, capsys):
    """At the calculator's default δ the union bound over a datacenter's or the world's execution phases is over 1, so
    those rows have no bound at their period, the term of record's among them; a node's rows carry their amounts."""
    assert R.NO_BOUND == "no bound at this period"
    for at, argv in enumerate((GIVEN, ["--commit", "c0ffee"])):
        text = bench(capsys, "--out", str(tmp_path / str(at)), "--registration", str(log(tmp_path)), *argv)
        # what is printed is the calculator's still, its terms for those rows among it
        assert LB.main(argv) == 0 and capsys.readouterr().out == text
        got, doc = result(tmp_path / str(at)), document(capsys, *argv)
        assert doc["inputs"]["scan_delta"]["value"] == float(LB.SCAN_DELTA) and len(got) == EVERY
        none = set()
        for (scope, period), r in scans(doc).items():
            name = f"certifier_term/modeled/{scope}/{period}"
            free, capacity, verified, error = (got[f"{name}/{amount}"] for amount in AMOUNTS)
            # the error and the capacity are written whatever the error is
            assert error["value"] == r["eps"] and capacity["value"] > 0 and not absent(error) and not absent(capacity)
            if r["eps"] < 1:
                assert free["value"] == r["term"]["2"]["bits"] and free["verdict"] in ("within target", "over target")
                assert verified["value"] == capacity["value"] - free["value"] and not absent(free) and not absent(verified)
                continue
            none.add(scope)
            why = [f"absent: no bound at this period: the row's error, {r['eps']}, is 1 or more"]
            assert (free["value"], free["verdict"], absent(free)) == (None, None, why)
            assert (verified["value"], absent(verified)) == (None, why)
        assert none == {"datacenter", "world"}
        # the term of record's row is the world's year's, so the metric has no value either
        metric, year = got["certifier_term/modeled/metric"], scans(doc)["world", "year"]
        assert (metric["value"], metric["verdict"], metric["unit"]) == (None, None, "GB")
        assert absent(metric) == [f"absent: no bound at this period: the row's error, {year['eps']}, is 1 or more"]
    assert "request count plan" in got["certifier_term/modeled/world/year/free"]["inputs"][0]


def test_a_row_at_error_exactly_1_has_no_bound(tmp_path, capsys):
    """The boundary: at a scan's δ of one over a node's year's execution phases the node's year is at error 1 exactly,
    which bounds nothing, and its month, under 1, carries its amounts."""
    delta = 1 / PHASES["node", "year"]
    bench(capsys, "--out", str(tmp_path), *GIVEN, "--scan-delta", str(delta), "--scope", "node")
    got = result(tmp_path)
    year, month = (f"certifier_term/modeled/node/{period}" for period in ("year", "month"))
    assert got[f"{year}/error"]["value"] == 1 and got[f"{year}/free"]["value"] is None
    assert absent(got[f"{year}/free"]) == ["absent: no bound at this period: the row's error, 1.0, is 1 or more"]
    assert 0 < got[f"{month}/error"]["value"] < 1 and got[f"{month}/free"]["value"] > 0 and not absent(got[f"{month}/free"])


def test_the_metric_follows_the_world_s_year_whatever_rows_the_command_asks_for(tmp_path, capsys):
    """A command that asks for a node's week alone still writes the term of record, whose row it didn't ask for: the
    metric has a value where the world's year is bounded and none where it isn't, though the node's week is."""
    asked = ["--scope", "node", "--period", "week"]
    for at, (argv, bounded) in enumerate(((GIVEN, False), (BOUNDED, True))):
        text = bench(capsys, "--out", str(tmp_path / str(at)), *argv, *asked)
        assert LB.main([*argv, *asked]) == 0 and capsys.readouterr().out == text
        got, doc = result(tmp_path / str(at)), document(capsys, *argv)
        metric, year = got["certifier_term/modeled/metric"], scans(doc)["world", "year"]
        assert list(got) == ["certifier_term/modeled/metric", *(f"certifier_term/modeled/node/week/{a}" for a in AMOUNTS)]
        assert got["certifier_term/modeled/node/week/free"]["value"] > 0
        assert (year["eps"] < 1) is bounded and metric["value"] == (doc["value"] if bounded else None)
        assert absent(metric) == ([] if bounded else [f"absent: no bound at this period: the row's error, {year['eps']}, "
                                                      "is 1 or more"])
    # a document whose term of record isn't that row's is refused
    with pytest.raises(SystemExit, match="isn't its row scan's over the world's year at D_j = 2"):
        R.record(LB, GIVEN, {**doc, "value": doc["value"] + 1})


def test_the_result_of_record_s_command_bounds_every_row(tmp_path, capsys):
    """The command the README gives for the result of record: the class given, and the scan's δ the error it is run at
    over the world's year, divided by the world year's execution phases as the calculator's own document counts them.
    Every row then carries its amounts, and the world's year's error is that error and no more."""
    ids = str(LB.RESPONSE_IDS // 2)
    asked = ["--commit", "c0ffee", "--response-ids", ids]
    year = scans(document(capsys, *asked, "--scope", "world", "--period", "year"))["world", "year"]
    assert Fraction(year["phases"]) == PHASES["world", "year"]
    delta = AT_MOST / Fraction(year["phases"])
    argv = [*asked, "--scan-delta", str(delta)]
    bench(capsys, "--out", str(tmp_path), "--registration", str(log(tmp_path)), *argv)
    got, doc = result(tmp_path), document(capsys, *argv)
    assert len(got) == EVERY and all(m["value"] is not None and not absent(m) for m in got.values())
    assert got["certifier_term/modeled/metric"]["value"] == doc["value"]
    errors = {at: got[f"certifier_term/modeled/{at[0]}/{at[1]}/error"]["value"] for at in scans(doc)}
    assert errors["world", "year"] == max(errors.values()) <= float(AT_MOST) and min(errors.values()) > 0
    assert all(got[f"certifier_term/modeled/{scope}/{period}/free"]["value"] == r["term"]["2"]["bits"]
               for (scope, period), r in scans(doc).items())


def test_every_measurement_names_the_class_and_the_scans_delta(tmp_path, capsys):
    """The most ids a response holds and each execution phase's scan's δ, read from the document: the defaults', and
    another's where the command gives one, whose terms are another class's and whose errors another δ's."""
    another = LB.RESPONSE_IDS // 2
    got = {}
    asked = {"default": [], "class": ["--response-ids", str(another)], "delta": ["--scan-delta", str(DELTA)]}
    for at, argv in asked.items():
        bench(capsys, "--out", str(tmp_path / at), "--registration", str(log(tmp_path)), *GIVEN, *argv)
        doc, got[at] = document(capsys, *GIVEN, *argv), result(tmp_path / at)
        ids, delta = doc["inputs"]["response_ids"], doc["inputs"]["scan_delta"]
        klass = f"response_ids = {ids['value']} id ({ids['mark']}): {ids['source']}"
        # the default δ as the document states it; one the command gives as given and assumed, no scan having run at it
        given = (f"scan_delta = {float(DELTA)} (assumed): each execution phase's scan's δ as the command gives it "
                 "(--scan-delta), to the document's float; no scan has run at it")
        said = given if at == "delta" else f"scan_delta = {float(LB.SCAN_DELTA)} (assumed): {delta['source']}"
        assert ids["value"] == (another if at == "class" else LB.RESPONSE_IDS) and delta["mark"] == "assumed"
        assert all(klass in m["inputs"] and said in m["inputs"] for m in got[at].values())
        assert all(sum(i.startswith(("response_ids = ", "scan_delta = ")) for i in m["inputs"]) == 2 for m in got[at].values())
    # another class moves a bounded row's term and nothing of its capacity or its error
    node = [name for name in got["default"] if "/node/" in name]
    assert all((got["default"][name]["value"] == got["class"][name]["value"]) == name.endswith(("/capacity", "/error"))
               for name in node)
    # another δ moves every row's error, and no term the default δ bounds
    assert all((got["default"][name]["value"] == got["delta"][name]["value"]) != name.endswith("/error") for name in node)


def test_refusals(tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as done:
        R.main(["--registration", str(log(tmp_path)), *GIVEN])
    assert done.value.code == 2 and "--registration goes with --out" in capsys.readouterr().err
    bare = (Link("uplink0", ("node0", OUTSIDE), BETA, 0),)
    beside = (Link("uplink0", ("node0", "cert0"), BETA, 0), Link("uplink0-out", ("cert0", OUTSIDE), BETA, 0),
              Link("uplink1", ("node0", OUTSIDE), BETA, 0))
    second = registration.site(locs=(Loc("node1", "host", "site0"),))
    for path, why in ((log(tmp_path, composite="remote", links=bare, name="bare"), r"are \['uplink0'\], \[\] of them"),
                      (log(tmp_path, links=beside, name="beside"), r"\['uplink0', 'uplink1'\], \['uplink0'\] of them"),
                      (log(tmp_path, BETA, second, name="second"), r"it registers the hosts \['node0', 'node1'\]")):
        with pytest.raises(SystemExit, match=f"doesn't register the model's machine, one host whose 1 boundary.*{why}"):
            R.main(["--out", str(out), "--registration", str(path), *GIVEN])
    (tmp_path / "none.json").write_text(json.dumps({"composite": "onsite"}))
    for path in (tmp_path / "none.json", tmp_path / "missing.json"):
        with pytest.raises(SystemExit, match="is no registration log"):
            R.main(["--out", str(out), "--registration", str(path), *GIVEN])
    # a refusal prints nothing of the calculator's and writes nothing
    assert capsys.readouterr().out == "" and not out.exists()
    bench(capsys, "--out", str(out), *GIVEN)
    first = (out / kit.RESULT).read_bytes()
    with pytest.raises(SystemExit, match="already holds a measurement named certifier_term/modeled/metric"):
        R.main(["--out", str(out), *GIVEN])
    assert capsys.readouterr().out == "" and (out / kit.RESULT).read_bytes() == first


def test_help_lists_the_runners_options_after_the_calculators(capsys):
    with pytest.raises(SystemExit) as done:
        LB.main(["--help"])
    theirs = capsys.readouterr().out
    with pytest.raises(SystemExit) as again:
        R.main(["--help"])
    text = capsys.readouterr().out
    assert (done.value.code, again.value.code) == (0, 0) and text.startswith(theirs)
    own = text[len(theirs):]
    assert own.startswith("\nthe runner's own options:\n  --out DIR ") and "\n  --registration FILE " in own
