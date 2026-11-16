"""The core benchmarks' interface (`benchmarks/README.md`'s "The core benchmarks' interface") as the kit holds it:
`record_dir` reads a directory with nothing of research's and a run id only through it, past its withdrawals (which
`record` also gives),
`ran_with` takes only the fixed words and states the deployed challenger's unverified coins as the record's word, and `report` prints the line after
the first, says where it is not an audit and where its record is made up, and gives no number for a run the Lean
verifier didn't accept. Each directory that declares itself a core command (`[tool.verity.interface] core` in its
`pyproject.toml`) is held too: its README has the three parts, its command calls `kit.report` with `ran_with` and
`accepted`, and each measurement name its source gives is a row of its `metrics.toml` with a goal."""
from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from verity_bench import kit

ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = ROOT / "benchmarks"
HARNESS = BENCHMARKS / "harness" / "pyproject.toml"
HEADINGS = ("## What you may change", "## What is fixed", "## How a cheating change is refused")
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
RUN = "r20000101-000000-aaaa"
#: declared directories rule 6 doesn't hold yet, each with what it writes that is no row: its owner drops its entry in the
#: change that makes every number it prints a row of its metrics.toml
DASHBOARD_NOT_YET = {
    "benchmarks/compute": "its verified, capacity, free and error measurements, and its share under constants of their own",
    "benchmarks/memory": "its verified, capacity, free and error measurements, and its share under the name metric",
}


def declared(benchmarks: Path) -> dict[str, str]:
    """Each directory under `benchmarks` that declares itself a core command, by its path from the repository, with the
    command it declares: its `pyproject.toml`'s ``[tool.verity.interface] core``."""
    found = {}
    for toml in sorted(benchmarks.glob("*/pyproject.toml")):
        core = tomllib.loads(toml.read_text()).get("tool", {}).get("verity", {}).get("interface", {}).get("core")
        if core is not None:
            found[f"benchmarks/{toml.parent.name}"] = core
    return found


def measurement_names(tree: ast.Module) -> list[str]:
    """The measurement names a source gives: the `name` of each dict literal that also has a `value` (a measurement
    `kit.report` takes), as a string constant or a module-level constant of one; any other as its source text in
    angle brackets, which no row's id is."""
    constants = {t.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                 for t in node.targets if isinstance(t, ast.Name)}
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
            if {"name", "value"} <= keys.keys():
                v = keys["name"]
                names.append(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else
                             constants.get(v.id, f"<{v.id}>") if isinstance(v, ast.Name) else f"<{ast.unparse(v)}>")
    return names


def unfollowed(directory: Path, dashboard: bool = True) -> list[str]:
    """What a directory lacks of the interface: a heading of its README's three parts, a `kit.report` call in its own
    Python files given both `ran_with` and `accepted`, or (with `dashboard`, rule 6) a row of its `metrics.toml`, with a
    `goal`, for each measurement name those files give (and at least one)."""
    if not (directory / "README.md").is_file():
        lacks = [f"a README.md with its three parts ({', '.join(HEADINGS)})"]
    else:
        readme = (directory / "README.md").read_text()
        lacks = [h for h in HEADINGS if not re.search(rf"^{re.escape(h)}\s*$", readme, re.M)]
    trees = [ast.parse(py.read_text()) for py in sorted(directory.glob("*.py"))]
    calls = [node for tree in trees for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "report"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "kit"]
    if not trees:
        lacks.append("a runner: a .py file directly in the directory whose kit.report call gives ran_with and accepted")
    elif not any({"ran_with", "accepted"} <= {k.arg for k in c.keywords} for c in calls):
        lacks.append("a kit.report(..., ran_with=..., accepted=...) call")
    if not dashboard:
        return lacks
    toml = directory / "metrics.toml"
    rows = {r["id"]: r for r in tomllib.loads(toml.read_text()).get("metric", [])} if toml.is_file() else {}
    names = sorted({n for tree in trees for n in measurement_names(tree)})
    if not names:
        lacks.append("a measurement under a row of metrics.toml")
    lacks += [f"a row of metrics.toml for the measurement {n}" for n in names if n not in rows]
    lacks += [f"a goal on metrics.toml's row {n}" for n in names if n in rows and not rows[n].get("goal")]
    return lacks


def test_each_directory_that_declares_itself_a_core_command_follows_the_interface():
    inputs = set(tomllib.loads(HARNESS.read_text())["tool"]["verity"]["tests"]["inputs"])
    for directory, core in declared(BENCHMARKS).items():
        where = f"{directory}/pyproject.toml declares [tool.verity.interface] core = {core!r}"
        assert isinstance(core, str) and core.startswith("verity bench "), f"{where}: give the command, 'verity bench NAME'"
        unread = [p for p in (directory, f"{directory}/README.md") if p not in inputs]
        assert not unread, (f"{where}, and this test reads {', '.join(unread)}: list it under [tool.verity.tests] inputs "
                            "in benchmarks/harness/pyproject.toml")
        lacks = unfollowed(ROOT / directory, directory not in DASHBOARD_NOT_YET)
        assert lacks == [], f"{where} and lacks {'; '.join(lacks)}: add them, or drop the key until it follows"


def test_the_holder_tells_a_command_that_follows_from_one_that_does_not(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    for d in (good, bad):
        d.mkdir()
    (good / "README.md").write_text("# good\n\n" + "\n\n".join(f"{h}\n\nwords" for h in HEADINGS) + "\n")
    (good / "run.py").write_text("from verity_bench import kit\n\nMETRIC = 'good/share'\n\ndef main():\n"
                                 "    measured = [{'name': METRIC, 'value': 1.0, 'unit': 'share'},\n"
                                 "                {'name': 'good/rate', 'value': 2.0, 'unit': 'Hz', 'member': 'clock.sm'}]\n"
                                 "    return kit.report('good', 't', {}, measured, "
                                 "ran_with=kit.ran_with('local challenger', 'development'), accepted=True)\n")
    (good / "metrics.toml").write_text('[[metric]]\nid = "good/share"\ngoal = "good/goal"\n\n'
                                       '[[metric]]\nid = "good/rate"\ngoal = "good/goal"\n')
    (bad / "README.md").write_text("# bad\n\n## What you may change\n")
    (bad / "run.py").write_text("from verity_bench import kit\n\ndef main():\n    kit.report('bad', 't', {})\n")
    assert unfollowed(good) == []
    assert unfollowed(bad) == [*HEADINGS[1:], "a kit.report(..., ran_with=..., accepted=...) call",
                               "a measurement under a row of metrics.toml"]
    (bad / "run.py").write_text("from verity_bench import kit\n\ndef main(r):\n    kit.report('bad', 't', {}, "
                                "[{'name': 'bad/share', 'value': 1.0}, {'name': f'free at {r}', 'value': 2.0}, "
                                "{'name': 'bad/aside', 'value': 3.0}], ran_with=None, accepted=True)\n")
    (bad / "metrics.toml").write_text('[[metric]]\nid = "bad/share"\n')
    assert unfollowed(bad) == [*HEADINGS[1:], "a row of metrics.toml for the measurement <f'free at {r}'>",
                               "a row of metrics.toml for the measurement bad/aside", "a goal on metrics.toml's row bad/share"]
    bare = tmp_path / "bare"
    bare.mkdir()
    assert unfollowed(bare, dashboard=False) == [
        f"a README.md with its three parts ({', '.join(HEADINGS)})",
        "a runner: a .py file directly in the directory whose kit.report call gives ran_with and accepted"]


def test_a_core_command_is_declared_in_its_own_pyproject(tmp_path):
    for name, text in [("good", '[tool.verity.interface]\ncore = "verity bench good"\n'),
                       ("plain", '[tool.verity.tests]\ninputs = []\n'), ("other", "[project]\nname = 'x'\n")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text(text)
    (tmp_path / "loose").mkdir()
    assert declared(tmp_path) == {"benchmarks/good": "verity bench good"}


# ---- record_dir


def no_research(monkeypatch):
    monkeypatch.setitem(sys.modules, "research.store.local", None)


def test_a_directory_is_its_record_and_imports_nothing_of_researchs(tmp_path, monkeypatch):
    no_research(monkeypatch)
    assert kit.record_dir(tmp_path) == tmp_path and kit.record_dir(str(tmp_path)) == tmp_path
    assert isinstance(kit.record_dir(tmp_path), Path) and kit.record(tmp_path) == (tmp_path, [])


@pytest.mark.parametrize("arg", ["no/such/dir", "r2000-bad", f"{RUN}x"])
def test_what_is_neither_a_directory_nor_a_run_id_is_refused_in_one_line(arg, monkeypatch):
    no_research(monkeypatch)
    with pytest.raises(kit.NoRecord, match="no such directory, and not a run id") as e:
        kit.record_dir(arg)
    assert "\n" not in str(e.value) and isinstance(e.value, ValueError)


def test_a_run_id_without_research_is_refused_naming_the_fix(monkeypatch):
    no_research(monkeypatch)
    with pytest.raises(kit.NoRecord) as e:
        kit.record_dir(RUN)
    assert str(e.value) == (f"{RUN}: a run id is read through research, which isn't installed here: give --from the "
                            "run's record directory, or install research (uv sync --all-packages)")


FILES = ["out/a.json", "out/b.json", "out/pieces/1.bin", "out/pieces/2.bin"]


def stand_in_research(monkeypatch, attempts: dict, fetched: Path, withdrawals: dict | None = None) -> list:
    """research's store client, as `record_dir` reads it: an attempt's run record holding `FILES`, the labels a
    withdrawal leaves on a target (`withdrawals`, a target's reasons), pulled first, and a fetch to `fetched`."""
    asked = []

    class StoreError(Exception):
        pass

    class LocalStore:
        def get_attempt(self, run):
            if run == "r20000101-000000-dead":
                raise StoreError("the store is down")
            return attempts.get(run)

        def listing(self, art):
            return [{"path": f} for f in FILES]

        def labels(self, target, key=None):
            assert key == "withdrawn"
            return [SimpleNamespace(key=key, value=v, by="infra") for v in (withdrawals or {}).get(target, [])]

        def fetch(self, art, *, paths=None):
            asked.append(("fetch", art, paths))
            return fetched

    local, sync, vocab = (ModuleType(f"research.store.{m}") for m in ("local", "labels_sync", "vocab"))
    local.LocalStore, local.StoreError = LocalStore, StoreError
    sync.read_through = lambda store, targets, warn: asked.append(("pull", targets))
    vocab.WITHDRAWN = "withdrawn"
    for module in (local, sync, vocab):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return asked


def test_a_run_id_is_fetched_through_research_to_its_records_directory(tmp_path, monkeypatch):
    asked = stand_in_research(monkeypatch, {RUN: {"outputs": {"run_record": f"art:{RUN}"}}, "r20000101-000000-bbbb": {}},
                              tmp_path)
    assert kit.record_dir(RUN, paths=["out/a.json"]) == tmp_path and isinstance(kit.record_dir(RUN), Path)
    assert asked[:2] == [("pull", [RUN, f"art:{RUN}"]), ("fetch", f"art:{RUN}", ["out/a.json"])]
    assert kit.record(RUN, paths=["out/a.json"]) == (tmp_path, [])
    with pytest.raises(kit.NoRecord, match="the evidence store holds no record of it"):
        kit.record_dir("r20000101-000000-bbbb")
    with pytest.raises(kit.NoRecord, match=r"can't give its record \(the store is down\)"):
        kit.record_dir("r20000101-000000-dead")


@pytest.mark.parametrize(("target", "reason", "paths"), [
    (f"art:{RUN}", "private values were published", ["out/a.json"]),                       # withdrawn whole
    (RUN, "private values were published", ["out/a.json"]),                                # the run's own label
    (f"art:{RUN}", "private values were published (only out/pieces)", ["out/pieces/1.bin"]),  # a part holding a file read
    (f"art:{RUN}", "private values were published (only out/b.json, out/a.json)", ["out/a.json"]),
    (f"art:{RUN}", "private values were published (only the sealed bundle)", ["out/a.json"]),  # names no file of it
    (f"art:{RUN}", "private values were published (only out/pieces)", None),                 # every file is read
])
def test_a_run_withdrawn_whole_or_where_it_is_read_is_refused_before_anything_is_fetched(tmp_path, monkeypatch, target,
                                                                                       reason, paths):
    asked = stand_in_research(monkeypatch, {RUN: {"outputs": {"run_record": f"art:{RUN}"}}}, tmp_path, {target: [reason]})
    for read in (kit.record_dir, kit.record):
        with pytest.raises(kit.NoRecord) as e:
            read(RUN, paths=paths)
        assert str(e.value) == f"{RUN} was withdrawn ({reason!r}, by infra): nothing of it is fetched"
    assert [a for a in asked if a[0] == "fetch"] == []


def test_a_withdrawal_of_only_other_files_is_reported_and_printed_beside_the_number(tmp_path, monkeypatch, out, capsys):
    reason = "the private value of a coin can be recovered (only out/pieces, out/b.json)"
    stand_in_research(monkeypatch, {RUN: {"outputs": {"run_files": f"art:{RUN}"}}}, tmp_path, {f"art:{RUN}": [reason]})
    record = kit.record(RUN, paths=["out/a.json"])
    assert record == (tmp_path, [f"{reason!r}, by infra"]) and kit.record_dir(RUN, paths=["out/a.json"]) == tmp_path
    assert kit.report("x", "x: 0.25", DOC, withdrawn=record.withdrawn) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["x: 0.25", f"  part of the run's record was withdrawn, none of it a file read here: {reason!r}, by infra"]
    assert json.loads(lines[-1]) == DOC | {"withdrawn": [f"{reason!r}, by infra"]}


# ---- ran_with


def test_ran_with_is_one_line_in_the_fixed_words_and_its_object():
    assert kit.ran_with("deployed challenger", "deployed", verified=True) == (
        "ran with: coins deployed challenger; parameters deployed",
        {"coins": "deployed challenger", "parameters": "deployed", "detail": None, "verified": True})
    line, said = kit.ran_with("in-process", "development", "2^-40 at 64 sampled proof blocks")
    assert line == "ran with: coins in-process; parameters development (2^-40 at 64 sampled proof blocks)"
    assert said["detail"] == "2^-40 at 64 sampled proof blocks" and kit.COINS == ("deployed challenger", "local challenger", "in-process")
    with pytest.raises(ValueError, match="'deployed challenger', 'local challenger', 'in-process'"):
        kit.ran_with("challenger, local", "deployed")
    with pytest.raises(ValueError, match="'deployed', 'development'"):
        kit.ran_with("local challenger", "not stated")


def test_the_deployed_challengers_unverified_coins_keep_their_line_and_say_verified_false():
    line, said = kit.ran_with("deployed challenger", "deployed", "signed")
    assert line == "ran with: coins deployed challenger; parameters deployed (signed)"
    assert said == {"coins": "deployed challenger", "parameters": "deployed", "detail": "signed", "verified": False}
    with pytest.raises(ValueError, match="verified is True or False"):
        kit.ran_with("deployed challenger", "deployed", verified="yes")
    assert kit.ran_with("local challenger", "deployed", verified=True)[1]["verified"] is True
    assert "verified" not in kit.ran_with("in-process", "development")[1]


# ---- report


@pytest.fixture
def out(tmp_path, monkeypatch) -> Path:
    for name in ("RESEARCH_RUN_DIR", "RESEARCH_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OUT", str(tmp_path / "out"))
    monkeypatch.setattr(kit, "host_stamp", lambda: dict(STAMP))
    return tmp_path / "out"


DOC = {"value": 0.25, "unit": "share", "commit": "c" * 40, "run": RUN}
MEASURED = [{"name": "a/metric", "value": 0.25, "unit": "share", "target": 0.5, "verdict": "within target",
             "interval": [0.2, 0.3]}]


def said(capsys, out: Path) -> tuple[list[str], dict, dict]:
    lines = capsys.readouterr().out.splitlines()
    doc = json.loads(lines[-1])
    assert (out / "x.json").read_text() == lines[-1] + "\n"
    return lines[:-1], doc, {m["name"]: m for m in json.loads((out / "result.json").read_text())["measurements"]}


@pytest.mark.parametrize(("coins", "parameters", "verified", "audit"), [("deployed challenger", "deployed", True, True),
                                                                        ("deployed challenger", "development", True, False),
                                                                        ("deployed challenger", "deployed", False, False),
                                                                        ("deployed challenger", "development", False, False),
                                                                        ("local challenger", "deployed", True, False),
                                                                        ("local challenger", "deployed", False, False),
                                                                        ("in-process", "development", False, False)])
def test_the_ran_with_line_follows_the_first_and_the_kit_says_where_it_is_not_an_audit(
        coins, parameters, verified, audit, out, capsys):
    pair = kit.ran_with(coins, parameters, "64 sampled proof blocks", verified=verified)
    assert kit.report("x", "x: 0.25\n  more", DOC, MEASURED, ran_with=pair, accepted=True) == 0
    text, doc, got = said(capsys, out)
    own = ["coins by the record's own word, not checked"] if coins == "deployed challenger" and not verified else []
    assert text == ["x: 0.25", pair[0], *([] if audit else ["a measurement, not an audit"]), *own, "  more"]
    assert list(doc) == [*DOC, "ran_with", "audit"] and doc["ran_with"] == pair[1] and doc["audit"] is audit
    assert got["a/metric"]["value"] == 0.25


@pytest.mark.parametrize("said", [{"coins": "deployed challenger", "parameters": "deployed", "detail": None},
                                  {"coins": "deployed challenger", "parameters": "deployed", "detail": None,
                                   "verified": "yes"}])
def test_a_pair_not_made_by_ran_with_is_refused_so_a_records_word_is_no_audit(said, out):
    with pytest.raises(ValueError):
        kit.report("x", "x: 0.25", DOC, ran_with=("ran with: coins deployed challenger; parameters deployed", said))
    assert not out.exists()


def test_a_made_up_record_says_so_under_the_audit_line_and_every_measurement_carries_it(out, capsys):
    pair = kit.ran_with("deployed challenger", "deployed", verified=True)
    measured = [MEASURED[0] | {"extra": {"rests_on": "a fixture"}}, {"name": "b", "value": None, "unit": "MAC",
                                                                      "why": "none"}]
    assert kit.report("x", "x: 0.25\n  more", DOC, measured, ran_with=pair, accepted=True, made_up=True) == 0
    text, doc, got = said(capsys, out)
    assert text == ["x: 0.25", pair[0], "a measurement, not an audit", "a made-up record, not a run's", "  more"]
    assert doc == DOC | {"ran_with": pair[1], "audit": False, "made_up": True}
    assert [(m["mark"], m["inputs"][0], m["made_up"]) for m in got.values()] == [("modeled", "modeled", True)] * 2
    assert got["a/metric"]["rests_on"] == "a fixture"
    with pytest.raises(ValueError, match="marked 'modeled'"):
        kit.report("y", "y", DOC, MEASURED, made_up=True, mark="measured")


def test_a_made_up_record_without_ran_with_says_so_under_the_first_line(out, capsys):
    assert kit.report("x", "x: no number\n  more", DOC, MEASURED, made_up=True, accepted=False, why="refused") == 1
    text, doc, got = said(capsys, out)
    assert text == ["x: no number: refused", "a made-up record, not a run's", "  more"]
    assert doc["made_up"] is True and "audit" not in doc
    assert (got["a/metric"]["value"], got["a/metric"]["mark"], got["a/metric"]["made_up"]) == (None, "modeled", True)


def test_a_record_that_predates_the_field_says_not_stated_and_is_no_audit(out, capsys):
    assert kit.report("x", "x: 0.25", DOC, ran_with=None) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["x: 0.25", "ran with: not stated", "a measurement, not an audit"]
    assert json.loads(lines[-1]) == DOC | {"ran_with": None, "audit": False}


def test_a_run_the_lean_verifier_did_not_accept_has_no_number_and_exits_1(out, capsys):
    pair = kit.ran_with("local challenger", "development")
    assert kit.report("x", "  more", DOC, MEASURED, ran_with=pair, accepted=False, why="the Lean verifier refused") == 1
    text, doc, got = said(capsys, out)
    assert text == ["x: no number: the Lean verifier refused", pair[0], "a measurement, not an audit", "  more"]
    assert (doc["value"], doc["no_number"], list(doc)[:4]) == (None, "the Lean verifier refused", list(DOC))
    m = got["a/metric"]
    assert (m["value"], m["why"], m["target"], m.get("verdict"), m.get("ci95")) == (
        None, "the Lean verifier refused", 0.5, None, None)
    with pytest.raises(ValueError, match="give why"):
        kit.report("x", "", DOC, accepted=False)


def test_with_no_number_the_texts_headline_is_replaced_so_its_number_never_prints(out, capsys):
    pair = kit.ran_with("local challenger", "development")
    assert kit.report("x", "x: 0.25 of the rate\n  more", DOC, MEASURED, ran_with=pair, accepted=False, why="refused") == 1
    text, doc, _ = said(capsys, out)
    assert text == ["x: no number: refused", pair[0], "a measurement, not an audit", "  more"]
    assert not any("0.25" in line for line in text) and doc["value"] is None
    # a first line that isn't the headline, a title or a note, is the command's detail and stays
    assert kit.report("x", "A title\n  a note", DOC, accepted=False, why="refused") == 1
    assert capsys.readouterr().out.splitlines()[:3] == ["x: no number: refused", "A title", "  a note"]
    assert kit.report("x", "x: 0.25", DOC, accepted=False, why="refused") == 1
    assert capsys.readouterr().out.splitlines()[0] == "x: no number: refused"


def test_given_none_of_the_interfaces_arguments_report_prints_and_writes_as_before(out, capsys):
    assert kit.report("x", "x: 0.25", DOC, MEASURED) == 0
    text, doc, got = said(capsys, out)
    assert text == ["x: 0.25"] and doc == DOC
    assert (got["a/metric"]["verdict"], got["a/metric"]["mark"]) == ("within target", "measured")
