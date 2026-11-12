"""`verity bench certifier_term --out DIR` (`run.py`): the result it writes beside what it prints.

- what it prints, as text and as `--json`, is the calculator's own to the character, with `--out` and without;
- the result is the harness kit's: its schema, one measurement a name, each in the kit's fields under the run and
  the host stamp, and the directory holds nothing else;
- the names are regular and every one says modeled: the term of record, and for each scope and monitoring period its
  row's free, capacity, verified and eps, in bits, bits, bits and a probability;
- free is the document's term for the scan at two release schedules, eps its execution phases' scans' δ summed, and
  the metric the document's value, each held to the calculator's table too;
- the metric and free stand against the document's target as the document's own multiple of it says, and no other
  measurement has a target or a verdict;
- capacity is read from a registered uplink's `beta`, one way: the scope's machines times it times the period's
  ticks, and verified is capacity less free; with no registration both have no value and say why;
- every measurement's inputs name the class the term is at, as the document states it;
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
#: each amount's unit, by its name's last word; the error's last word is the runner's `ERROR`
AMOUNTS = {"free": "bit", "capacity": "bit", "verified": "bit", R.ERROR: "probability"}
#: the uplink's two halves' capacities in bits a tick: the host's side, which is the boundary link, and the outside's
BETA, BEYOND = 400, 100


@pytest.fixture(autouse=True)
def stamped(monkeypatch):
    """The kit's writer under a run of its own, stamping with `STAMP` rather than asking the host."""
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
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


def document(capsys, *argv: str, given: tuple[str, ...] = tuple(GIVEN)) -> dict:
    """The calculator's own document on ``given`` and ``argv``, as it prints it."""
    assert LB.main([*given, *argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


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
                              ["--json", "--requests", "census", "--lanes", "512", "--target-gb", "150"])):
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
    assert done.value.code == 0 and json.loads(capsys.readouterr().out) == document(capsys)
    assert list(result(tmp_path / "main")) == list(result(tmp_path / "1"))


def test_the_result_is_the_kits_and_nothing_else_is_written(tmp_path, capsys, monkeypatch):
    written = []
    measurement = kit.Writer.measurement
    monkeypatch.setattr(kit.Writer, "measurement", lambda self, **m: written.append(measurement(self, **m)) or written[-1])
    out = tmp_path / "out"
    bench(capsys, "--out", str(out), *GIVEN)
    # every measurement of the file is one the kit's writer returned, in its fields, under this run and its stamp
    assert json.loads((out / kit.RESULT).read_text())["measurements"] == written and len(written) == 37
    fields = set(kit.writer(tmp_path / "another").measurement(name="a measurement", value=1, unit="bit"))
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
    # only the rows the command was asked for, with the term of record
    bench(capsys, "--out", str(tmp_path / "one"), *GIVEN, "--scope", "datacenter", "--period", "month")
    assert list(result(tmp_path / "one")) == ["certifier_term/modeled/metric",
                                              *(f"certifier_term/modeled/datacenter/month/{amount}" for amount in AMOUNTS)]


def test_free_is_the_documents_term_and_eps_its_phases_deltas_summed(tmp_path, capsys):
    bench(capsys, "--out", str(tmp_path), *GIVEN)
    got, doc = result(tmp_path), document(capsys)
    rows = {(r["scope"], r["period"]): r for r in doc["rows"] if r["row"] == "scan"}
    exact = {(r.scope.name, r.period): r for r in LB.table(LB.inputs(world_requests=1e16)) if r.name == "scan"}
    assert list(rows) == [(scope, period) for scope in LB.SCOPES for period in LB.PERIODS] and LB.OF_RECORD == 2
    for (scope, period), r in rows.items():
        name, row = f"certifier_term/modeled/{scope}/{period}", exact[scope, period]
        # the term at two release schedules, and the union of the scans' δ, as the table has them
        assert got[f"{name}/free"]["value"] == r["term"]["2"]["bits"] == float(row.terms[2].certifier)
        error = got[f"{name}/{R.ERROR}"]
        assert error["value"] == r["eps"] == float(row.eps) == float(row.phases * LB.SCAN_DELTA)
        assert error["inputs"][1] == R.UNION and "eps, the calculator's key: a union bound over the period's" in R.UNION
        assert got[f"{name}/free"]["inputs"][0].startswith("modeled: benchmarks/network_traces/leak_bounds.py's row scan "
                                                           f"at D_j = 2, scope {scope}, period {period}, ")
        # the document's inputs, each with its value, its mark and its source
        assert got[f"{name}/free"]["inputs"][1:] == [R.stated(k, i) for k, i in doc["inputs"].items()]
    # the term of record: the world's year's, in GB
    metric, year = got["certifier_term/modeled/metric"], got["certifier_term/modeled/world/year/free"]
    assert (metric["value"], metric["unit"]) == (doc["value"], "GB")
    assert metric["value"] == float(exact["world", "year"].terms[2].certifier / LB.GB)
    assert metric["inputs"][0] == f"modeled: {doc['name']}" and metric["inputs"][1:] == year["inputs"][1:]


def test_the_metric_and_free_stand_against_the_documents_target(tmp_path, capsys):
    for at, target_gb in enumerate((LB.TARGET_GB, 1e-3, 1e12)):
        argv = [] if target_gb == LB.TARGET_GB else ["--target-gb", str(target_gb)]
        bench(capsys, "--out", str(tmp_path / str(at)), *GIVEN, *argv)
        got = result(tmp_path / str(at))
        doc = document(capsys, *argv)
        said = {(r["scope"], r["period"]): r["term"]["2"]["of_target"] for r in doc["rows"] if r["row"] == "scan"}
        for (scope, period), multiple in said.items():
            free = got[f"certifier_term/modeled/{scope}/{period}/free"]
            assert free["target"] == target_gb * LB.GB
            assert free["verdict"] == ("over target" if multiple > 1 else "within target")
        metric = got["certifier_term/modeled/metric"]
        assert (metric["target"], metric["verdict"]) == (
            target_gb, "over target" if said["world", "year"] > 1 else "within target")
        assert {m["verdict"] for m in got.values()} == {None, *(["over target"] if at < 2 else []),
                                                        *(["within target"] if at != 1 else [])}
        assert all((m["target"], m["verdict"]) == (None, None) for name, m in got.items()
                   if name.rsplit("/", 1)[1] in ("capacity", "verified", R.ERROR))


def test_capacity_is_read_from_the_registered_uplink_and_verified_is_capacity_less_free(tmp_path, capsys):
    path = log(tmp_path)
    bench(capsys, "--out", str(tmp_path / "out"), "--registration", str(path), *GIVEN)
    got, klass = result(tmp_path / "out"), R.at_class(document(capsys))
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
            assert verified["inputs"] == [f"{name}/capacity", f"{name}/free", klass]
            assert capacity["inputs"] == [
                f"registration {registration.head(composite, entries)}, read as given: host node0's certified boundary "
                f"link(s) uplink0 at beta {BETA}, in bits each way a tick of 1/1000000000 s (verity.hardware.site)",
                R.SCALED, f"modeled: benchmarks/network_traces/leak_bounds.py's row scan at D_j = 2, scope {scope.name}, "
                f"period {period}, request count given", klass]
    assert "beta is each way and the term is egress, so one direction" in R.SCALED
    # the registration's own number: another beta is another capacity, and nothing else moves
    bench(capsys, "--out", str(tmp_path / "twice"), "--registration", str(log(tmp_path, 2 * BETA)), *GIVEN)
    for name, m in result(tmp_path / "twice").items():
        if name.endswith("/capacity"):
            assert m["value"] == 2 * got[name]["value"]
        elif not name.endswith("/verified"):
            assert m == got[name]


def test_with_no_registration_capacity_and_verified_have_no_value_and_say_why(tmp_path, capsys):
    bench(capsys, "--out", str(tmp_path), *GIVEN)
    got, klass = result(tmp_path), R.at_class(document(capsys))
    for scope in LB.SCOPES:
        for period in LB.PERIODS:
            name = f"certifier_term/modeled/{scope}/{period}"
            capacity, verified = got[f"{name}/capacity"], got[f"{name}/verified"]
            assert (capacity["value"], capacity["inputs"][0], capacity["inputs"][2:]) == (None, R.NO_REGISTRATION, [klass])
            assert capacity["inputs"][1].startswith("modeled: benchmarks/network_traces/leak_bounds.py's row scan at ")
            assert verified["value"] is None
            assert verified["inputs"] == [f"{name}/capacity", f"{name}/free", R.NO_REGISTRATION, klass]
    assert R.NO_REGISTRATION.startswith("absent: no registration is given")
    assert sum(m["value"] is None for m in got.values()) == 18


def test_every_measurement_names_the_class_the_term_is_at(tmp_path, capsys):
    """The most ids a response holds, read from the document: the default's, and another's where the command is given
    one, whose terms are another class's."""
    paths = {}
    for ids in (None, 1024):
        argv = [] if ids is None else ["--response-ids", str(ids)]
        paths[ids] = tmp_path / str(ids)
        bench(capsys, "--out", str(paths[ids]), "--registration", str(log(tmp_path)), *GIVEN, *argv)
        entry = document(capsys, *argv)["inputs"][R.CLASS]
        klass = f"response_ids = {entry['value']} id ({entry['mark']}): {entry['source']}"
        assert R.CLASS == "response_ids" and entry["value"] == (LB.RESPONSE_IDS if ids is None else ids)
        assert all(klass in m["inputs"] for m in result(paths[ids]).values())
    default, other = result(paths[None]), result(paths[1024])
    assert LB.RESPONSE_IDS != 1024
    assert all((default[name]["value"] == m["value"]) == name.endswith(("/capacity", f"/{R.ERROR}")) for name, m in other.items())


def test_the_result_of_the_commands_defaults(tmp_path, capsys):
    """The term of record's row at the sampling plan's request count, which the default command counts."""
    text = bench(capsys, "--out", str(tmp_path), "--commit", "c0ffee")
    assert LB.main(["--commit", "c0ffee"]) == 0 and capsys.readouterr().out == text
    got, doc = result(tmp_path), document(capsys, given=("--commit", "c0ffee"))
    assert doc["request_count"] == "plan" and len(got) == 37
    assert got["certifier_term/modeled/metric"]["value"] == doc["value"]
    rows = {(r["scope"], r["period"]): r for r in doc["rows"] if r["row"] == "scan"}
    assert all(got[f"certifier_term/modeled/{scope}/{period}/free"]["value"] == r["term"]["2"]["bits"]
               for (scope, period), r in rows.items())
    assert "request count plan" in got["certifier_term/modeled/world/year/free"]["inputs"][0]


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
