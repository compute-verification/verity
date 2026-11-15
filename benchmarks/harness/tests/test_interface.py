"""The core benchmarks' interface (`benchmarks/README.md`'s "The core benchmarks' interface") as the kit holds it:
`record_dir` reads a directory with nothing of research's and a run id only through it, `ran_with` takes only the fixed
words, and `report` prints the line after the first, says where it is not an audit and gives no number for a run the
Lean verifier didn't accept. The section's table is held too: each listed directory's README has the three parts, and
its command calls `kit.report` with `ran_with` and `accepted`."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from verity_bench import kit

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "benchmarks" / "README.md"
SECTION = "## The core benchmarks' interface"
HEADINGS = ("## What you may change", "## What is fixed", "## How a cheating change is refused")
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
RUN = "r20000101-000000-aaaa"


def table(readme: str) -> list[dict[str, str]]:
    """The section's table of the commands that follow the interface: one row each, by its header's names."""
    section = readme.split(SECTION, 1)[1].split("\n## ", 1)[0]
    lines = [x for x in section.splitlines() if x.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    assert head == ["command", "directory", "owner"] and set(lines[1].replace("|", "").strip()) <= set("-: ")
    return [dict(zip(head, (c.strip().strip("`") for c in x.strip("|").split("|")))) for x in lines[2:]]


def unfollowed(directory: Path) -> list[str]:
    """What a directory lacks of the interface: a heading of its README's three parts, or a `kit.report` call in its
    own Python files given both `ran_with` and `accepted`."""
    readme = (directory / "README.md").read_text() if (directory / "README.md").is_file() else ""
    lacks = [h for h in HEADINGS if not re.search(rf"^{re.escape(h)}\s*$", readme, re.M)]
    calls = [node for py in sorted(directory.glob("*.py")) for node in ast.walk(ast.parse(py.read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "report"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "kit"]
    if not any({"ran_with", "accepted"} <= {k.arg for k in c.keywords} for c in calls):
        lacks.append("a kit.report(..., ran_with=..., accepted=...) call")
    return lacks


def test_each_command_the_section_lists_follows_the_interface():
    rows = table(README.read_text())
    for row in rows:
        assert row["command"].startswith("verity bench ") and row["owner"], row
        assert (ROOT / row["directory"]).is_dir(), f"{row['directory']}: no such directory"
        assert unfollowed(ROOT / row["directory"]) == [], f"{row['command']} ({row['directory']}) lacks these"


def test_the_holder_tells_a_command_that_follows_from_one_that_does_not(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    for d in (good, bad):
        d.mkdir()
    (good / "README.md").write_text("# good\n\n" + "\n\n".join(f"{h}\n\nwords" for h in HEADINGS) + "\n")
    (good / "run.py").write_text("from verity_bench import kit\n\ndef main():\n    return kit.report('good', 't', {}, "
                                 "ran_with=kit.ran_with('local challenger', 'development'), accepted=True)\n")
    (bad / "README.md").write_text("# bad\n\n## What you may change\n")
    (bad / "run.py").write_text("from verity_bench import kit\n\ndef main():\n    kit.report('bad', 't', {})\n")
    assert unfollowed(good) == []
    assert unfollowed(bad) == [*HEADINGS[1:], "a kit.report(..., ran_with=..., accepted=...) call"]
    readme = f"x\n\n{SECTION}\n\nwords\n\n| command | directory | owner |\n|---|---|---|\n| `verity bench good` | `good` | infra |\n\n## Next\n"
    assert table(readme) == [{"command": "verity bench good", "directory": "good", "owner": "infra"}]


# ---- record_dir


def no_research(monkeypatch):
    monkeypatch.setitem(sys.modules, "research.store.local", None)


def test_a_directory_is_its_record_and_imports_nothing_of_researchs(tmp_path, monkeypatch):
    no_research(monkeypatch)
    assert kit.record_dir(tmp_path) == tmp_path and kit.record_dir(str(tmp_path)) == tmp_path


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


def stand_in_research(monkeypatch, attempts: dict, fetched: Path) -> list:
    """research's store client, as `record_dir` reads it: an attempt's run record, fetched to `fetched`."""
    asked = []

    class StoreError(Exception):
        pass

    class LocalStore:
        def get_attempt(self, run):
            if run == "r20000101-000000-dead":
                raise StoreError("the store is down")
            return attempts.get(run)

        def fetch(self, art, *, paths=None):
            asked.append((art, paths))
            return fetched

    module = ModuleType("research.store.local")
    module.LocalStore, module.StoreError = LocalStore, StoreError
    monkeypatch.setitem(sys.modules, "research.store.local", module)
    return asked


def test_a_run_id_is_fetched_through_research_to_its_records_directory(tmp_path, monkeypatch):
    asked = stand_in_research(monkeypatch, {RUN: {"outputs": {"run_record": f"art:{RUN}"}}, "r20000101-000000-bbbb": {}},
                              tmp_path)
    assert kit.record_dir(RUN, paths=["out/a.json"]) == tmp_path and asked == [(f"art:{RUN}", ["out/a.json"])]
    with pytest.raises(kit.NoRecord, match="the evidence store holds no record of it"):
        kit.record_dir("r20000101-000000-bbbb")
    with pytest.raises(kit.NoRecord, match=r"can't give its record \(the store is down\)"):
        kit.record_dir("r20000101-000000-dead")


# ---- ran_with


def test_ran_with_is_one_line_in_the_fixed_words_and_its_object():
    assert kit.ran_with("deployed challenger", "deployed") == (
        "ran with: coins deployed challenger; parameters deployed",
        {"coins": "deployed challenger", "parameters": "deployed", "detail": None})
    line, said = kit.ran_with("in-process", "development", "2^-40 at 64 draws")
    assert line == "ran with: coins in-process; parameters development (2^-40 at 64 draws)"
    assert said["detail"] == "2^-40 at 64 draws" and kit.COINS == ("deployed challenger", "local challenger", "in-process")
    with pytest.raises(ValueError, match="'deployed challenger', 'local challenger', 'in-process'"):
        kit.ran_with("challenger, local", "deployed")
    with pytest.raises(ValueError, match="'deployed', 'development'"):
        kit.ran_with("local challenger", "not stated")


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


@pytest.mark.parametrize(("coins", "parameters", "audit"), [("deployed challenger", "deployed", True),
                                                            ("deployed challenger", "development", False),
                                                            ("local challenger", "deployed", False),
                                                            ("in-process", "development", False)])
def test_the_ran_with_line_follows_the_first_and_the_kit_says_where_it_is_not_an_audit(
        coins, parameters, audit, out, capsys):
    pair = kit.ran_with(coins, parameters, "64 draws")
    assert kit.report("x", "x: 0.25\n  more", DOC, MEASURED, ran_with=pair, accepted=True) == 0
    text, doc, got = said(capsys, out)
    assert text == ["x: 0.25", pair[0], *([] if audit else ["a measurement, not an audit"]), "  more"]
    assert list(doc) == [*DOC, "ran_with", "audit"] and doc["ran_with"] == pair[1] and doc["audit"] is audit
    assert got["a/metric"]["value"] == 0.25


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


def test_given_none_of_the_interfaces_arguments_report_prints_and_writes_as_before(out, capsys):
    assert kit.report("x", "x: 0.25", DOC, MEASURED) == 0
    text, doc, got = said(capsys, out)
    assert text == ["x: 0.25"] and doc == DOC
    assert (got["a/metric"]["verdict"], got["a/metric"]["mark"]) == ("within target", "measured")
