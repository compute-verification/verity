"""`verity lean deployed`: the deployed guarantees' rows, from locks the tests write (the pure `rows`, and a temporary git
repository for the git layer), and once from the tree's own locks, whose gaps must be the Lean check's `trust_gaps`."""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

import pytest

import verity_cli
from verity_cli import deployed

ROOT = Path(__file__).resolve().parents[4]
TODAY = datetime.date(2030, 6, 15)
CHECK = deployed.lean_check()
BASE = {"Base.CR": "collision resistance"}


def lock(records: dict | None = None, deployed_: list | None = None, trust_gap: dict | None = None, **sections) -> dict:
    """A lock deploying ``deployed_`` of ``records`` ({guarantee: {kind: [premise, ...]}}), `Base.CR` its trust base, and
    ``sections`` ({`open`|`assumptions`|`domain`: {premise: entry}}) as given."""
    out = {"format": 1, "trust": {"base": BASE}, **sections}
    if records is not None:
        out["guarantees"] = {g: {"owner": "@g", "type_hash": f"t-{g}", "unfolded_hash": f"u-{g}", "premises": p}
                             for g, p in records.items()}
    if deployed_ is not None:
        out["deployed"] = deployed_
    if trust_gap is not None:
        out["trust_gap"] = trust_gap
    return out


def entry(owner: str | None = "@o", issue: str | None = "X-1", due: str | None = "2030-12-31") -> dict:
    return {k: v for k, v in (("owner", owner), ("issue", issue), ("due", due), ("digest", "d")) if v is not None}


def run(locks: dict[str, dict | str], today: datetime.date = TODAY) -> dict:
    texts = {p: v if isinstance(v, str) else json.dumps(v) for p, v in locks.items()}
    return deployed.rows(texts, today, "c" * 40, CHECK)


def premise(out: dict, name: str) -> dict:
    return next(p for p in out["premises"] if p["name"] == name)


def test_a_guarantee_at_gap_zero_and_one_with_a_gap():
    out = run({"pkg/lean-audit.json": lock(
        {"P.Zero": {"assumptions": ["Base.CR"]}, "P.Gapped": {"assumptions": ["Base.CR", "P.Assumed"], "open": ["P.Open"]},
         "P.NotDeployed": {"open": ["P.Elsewhere"]}},
        ["P.Zero", "P.Gapped"], {"P.Gapped": ["P.Assumed", "P.Open"]},
        open={"P.Open": entry(), "P.Elsewhere": entry()}, assumptions={"P.Assumed": entry(None, None, None),
                                                                       "Base.CR": entry(None, None, None)})})
    assert list(out) == ["format", "commit", "today", "locks", "guarantees", "premises", "totals"]
    assert (out["format"], out["commit"], out["today"], out["locks"]) == (1, "c" * 40, "2030-06-15", ["pkg/lean-audit.json"])
    assert out["guarantees"] == [
        {"name": "P.Gapped", "lock": "pkg/lean-audit.json", "owner": "g", "type_hash": "t-P.Gapped",
         "unfolded_hash": "u-P.Gapped", "gap": ["P.Assumed", "P.Open"], "lacks_witness": None, "held_differs": None},
        {"name": "P.Zero", "lock": "pkg/lean-audit.json", "owner": "g", "type_hash": "t-P.Zero", "unfolded_hash": "u-P.Zero",
         "gap": [], "lacks_witness": None, "held_differs": None}]
    assert [(p["name"], p["kind"], p["in_gap_of"]) for p in out["premises"]] == [
        ("P.Assumed", "assumption", ["P.Gapped"]), ("P.Elsewhere", "open", []), ("P.Open", "open", ["P.Gapped"])]
    assert list(premise(out, "P.Open")) == ["name", "kind", "owner", "issue", "due", "listed", "overdue", "in_gap_of"]
    assert out["totals"] == {"deployed": 2, "gap_zero": 1, "gap_premises": 2, "open": 2, "overdue": 0, "owners_differ": 0,
                             "no_honest_witness": 0,
                             "by_owner": {"g": {"deployed": 2, "gap_premises": 0, "open": 0, "overdue": 0, "no_honest_witness": 0},
                                          "o": {"deployed": 0, "gap_premises": 1, "open": 2, "overdue": 0, "no_honest_witness": 0}}}


def test_a_gap_reaches_the_premises_of_a_guarantee_a_domain_hypothesis_cites():
    """The gap is `trust_gaps`' closure, which follows a domain hypothesis's `cites` to the guarantee it states."""
    out = run({"pkg/lean-audit.json": lock({"P.Top": {"domain": ["P.Cited"]}, "P.Below": {"open": ["P.Deep"]}}, ["P.Top"],
                                           open={"P.Deep": entry()}, domain={"P.Cited": {"owner": "@o", "cites": "P.Below"}})})
    assert out["guarantees"][0]["gap"] == ["P.Deep"]
    assert out["guarantees"][0]["held_differs"] == {"only_held": [], "only_computed": ["P.Deep"]}


def test_a_premise_two_locks_list_is_counted_once_with_equal_fields_and_with_two_owners():
    """An owner prints bare: `@x` and `x` are one owner."""
    a = lock({"A.G": {"open": ["S.Same", "S.Split"]}}, ["A.G"], {"A.G": ["S.Same", "S.Split"]},
             open={"S.Same": entry("@x"), "S.Split": entry("@x", due="2030-06-01")})
    b = lock({"B.G": {"open": ["S.Split"]}}, ["B.G"], {"B.G": ["S.Split"]},
             open={"S.Same": entry("x"), "S.Split": entry("@y", due="2030-07-01")})
    out = run({"b/lean-audit.json": b, "a/lean-audit.json": a})
    assert out["locks"] == ["a/lean-audit.json", "b/lean-audit.json"]
    assert premise(out, "S.Same") == {
        "name": "S.Same", "kind": "open", "owner": "x", "issue": "X-1", "due": "2030-12-31",
        "listed": [{"lock": "a/lean-audit.json", "kind": "open", "owner": "x", "issue": "X-1", "due": "2030-12-31"},
                   {"lock": "b/lean-audit.json", "kind": "open", "owner": "x", "issue": "X-1", "due": "2030-12-31"}],
        "overdue": False, "in_gap_of": ["A.G"]}
    split = premise(out, "S.Split")
    assert (split["owner"], split["issue"], split["due"]) == (None, "X-1", None)
    assert [e["owner"] for e in split["listed"]] == ["x", "y"]
    assert split["overdue"] is True, "the earliest of the locks' dates is before today"
    assert split["in_gap_of"] == ["A.G", "B.G"]
    t = out["totals"]
    assert (t["deployed"], t["gap_premises"], t["open"], t["overdue"], t["owners_differ"]) == (2, 2, 2, 1, 1)
    assert t["by_owner"]["x"] == {"deployed": 0, "gap_premises": 2, "open": 2, "overdue": 1, "no_honest_witness": 0}
    assert t["by_owner"]["y"] == {"deployed": 0, "gap_premises": 1, "open": 1, "overdue": 1, "no_honest_witness": 0}


def test_a_choice_item_lists_the_definitions_its_guarantee_reaches_with_the_earliest_due():
    """A deployed guarantee whose `trust_gap` entry holds a `choice:<head>` item keeps it while a record of its closure
    lists a definition `choosing` lists: the item is of kind `choice`, listed once for each such definition with its
    owner, issue and due, its `due` the earliest and `overdue` when that is before today.  Cured, the item leaves the gap
    the lock alone computes, and the held entry differs."""
    item = "choice:Base.CR"
    chosen = {"D.late": {"owner": "@proofs", "issue": "X-2", "due": "2030-06-01"},
              "D.later": {"owner": "@soundness", "issue": "X-3", "due": "2030-07-01"}, "D.other": entry()}
    out = run({"pkg/lean-audit.json": lock({"P.G": {"assumptions": ["Base.CR"], "choosing": ["D.late", "D.later"]}},
                                           ["P.G"], {"P.G": [item]}, choosing=chosen)})
    assert out["guarantees"][0]["gap"] == [item] and out["guarantees"][0]["held_differs"] is None
    row = premise(out, item)
    assert (row["kind"], row["owner"], row["issue"], row["due"], row["overdue"], row["in_gap_of"]) == (
        "choice", None, None, "2030-06-01", True, ["P.G"])
    assert row["listed"] == [
        {"lock": "pkg/lean-audit.json", "kind": "choice", "owner": "proofs", "issue": "X-2", "due": "2030-06-01",
         "definition": "D.late"},
        {"lock": "pkg/lean-audit.json", "kind": "choice", "owner": "soundness", "issue": "X-3", "due": "2030-07-01",
         "definition": "D.later"}]
    assert not run({"pkg/lean-audit.json": lock({"P.G": {"choosing": ["D.late"]}}, ["P.G"], {"P.G": [item]}, choosing=chosen)},
                   datetime.date(2030, 6, 1))["premises"][0]["overdue"]
    cured = run({"pkg/lean-audit.json": lock({"P.G": {"assumptions": ["Base.CR"]}}, ["P.G"], {"P.G": [item]},
                                             choosing=chosen)})
    assert cured["guarantees"][0]["gap"] == [] and cured["guarantees"][0]["held_differs"] == {
        "only_held": [item], "only_computed": []}


def test_a_lock_that_holds_only_open_premises():
    out = run({"only/lean-audit.json": {"open": {"O.One": entry(), "O.Two": entry()}}})
    assert out["guarantees"] == []
    assert [p["name"] for p in out["premises"]] == ["O.One", "O.Two"]
    assert {k: v for k, v in out["totals"].items() if k != "by_owner"} == {
        "deployed": 0, "gap_zero": 0, "gap_premises": 0, "open": 2, "overdue": 0, "owners_differ": 0, "no_honest_witness": 0}


@pytest.mark.parametrize(("due", "overdue"), [("2030-06-14", True), ("2030-06-15", False), ("2030-06-16", False),
                                              ("June", False)])
def test_a_premise_is_overdue_only_when_its_due_date_is_before_today(due, overdue):
    out = run({"pkg/lean-audit.json": {"open": {"D.P": entry(due=due)}}})
    assert premise(out, "D.P")["overdue"] is overdue
    assert out["totals"]["overdue"] == int(overdue)


def test_a_lock_without_owners_issues_or_dates_reads_as_nulls():
    """A lock may leave out `owner`, `issue` and `due`."""
    bare = {"guarantees": {"N.G": {"premises": {"open": ["N.P"]}}}, "deployed": ["N.G"], "open": {"N.P": {"digest": "d"}},
            "assumptions": {"N.A": "not an object"}}
    out = run({"pkg/lean-audit.json": bare})
    assert out["guarantees"] == [{"name": "N.G", "lock": "pkg/lean-audit.json", "owner": None, "type_hash": None,
                                  "unfolded_hash": None, "gap": ["N.P"], "lacks_witness": None,
                                  "held_differs": {"only_held": [], "only_computed": ["N.P"]}}]
    assert out["premises"] == [{"name": "N.P", "kind": "open", "owner": None, "issue": None, "due": None,
                                "listed": [{"lock": "pkg/lean-audit.json", "kind": "open", "owner": None, "issue": None,
                                            "due": None}],
                                "overdue": False, "in_gap_of": ["N.G"]}]
    assert out["totals"]["by_owner"] == {}


def test_a_held_gap_that_differs_from_the_computed_one_is_reported_and_one_that_agrees_is_null():
    pkg = lock({"H.G": {"open": ["H.Now"]}, "H.Same": {"open": ["H.Now"]}}, ["H.G", "H.Same"],
               {"H.G": ["H.Gone"], "H.Same": ["H.Now"]}, open={"H.Now": entry()})
    out = run({"pkg/lean-audit.json": pkg})
    assert [(r["name"], r["gap"], r["held_differs"]) for r in out["guarantees"]] == [
        ("H.G", ["H.Now"], {"only_held": ["H.Gone"], "only_computed": ["H.Now"]}), ("H.Same", ["H.Now"], None)]


def test_each_row_says_which_witness_halves_its_lock_shows_it_lacks():
    """Null where the witnesses rule doesn't hold the guarantee; else the halves its `witnesses` entry names by no non-empty
    string or list, whatever `unwitnessed` says beside it."""
    names = ("W.Free", "W.Both", "W.Bare", "W.Half", "W.Empty")
    pkg = lock({g: {} for g in names}, list(names), witnesses={
        "W.Both": {"honest": ["W.h"], "nontrivial": "W.n"}, "W.Half": {"honest": ["W.h"], "nontrivial": None},
        "W.Empty": {"honest": [], "nontrivial": ""}}, unwitnessed={"W.Bare": "t", "W.Half": {"type_hash": "t"}})
    for g in names:
        pkg["guarantees"][g]["owner"] = "@" + g[2:].lower()
    out = run({"pkg/lean-audit.json": pkg})
    assert {r["name"]: r["lacks_witness"] for r in out["guarantees"]} == {
        "W.Free": None, "W.Both": [], "W.Bare": ["honest", "nontrivial"], "W.Half": ["nontrivial"],
        "W.Empty": ["honest", "nontrivial"]}
    assert out["totals"]["no_honest_witness"] == 2
    assert {o: v["no_honest_witness"] for o, v in out["totals"]["by_owner"].items()} == {
        "free": 0, "both": 0, "bare": 1, "half": 0, "empty": 1}


def test_a_premise_takes_the_first_kind_of_the_sections_that_list_it_and_each_listing_keeps_its_own():
    """`open` before `assumption` before `domain`, across locks, and each `listed` entry is its lock's section; a gap's
    premise no section lists has no kind and no listing."""
    a = lock({"K.G": {"open": ["K.Both", "K.Unlisted"], "assumptions": ["K.AssumedToo"]}}, ["K.G"],
             open={"K.Both": entry()}, assumptions={"K.AssumedToo": entry(None, None, None)})
    b = {"domain": {"K.Both": {"owner": "@o", "why": "w"}, "K.AssumedToo": {"owner": "@o", "why": "w"}}}
    out = run({"a/lean-audit.json": a, "b/lean-audit.json": b})
    assert [(p["name"], p["kind"], len(p["listed"])) for p in out["premises"]] == [
        ("K.AssumedToo", "assumption", 2), ("K.Both", "open", 2), ("K.Unlisted", None, 0)]
    assert premise(out, "K.Both")["listed"] == [
        {"lock": "a/lean-audit.json", "kind": "open", "owner": "o", "issue": "X-1", "due": "2030-12-31"},
        {"lock": "b/lean-audit.json", "kind": "domain", "owner": "o", "issue": None, "due": None}]
    assert [e["kind"] for e in premise(out, "K.AssumedToo")["listed"]] == ["assumption", "domain"]
    assert premise(out, "K.Unlisted")["listed"] == []
    assert premise(out, "K.Both")["issue"] is None, "the domain entry gives no issue, so the locks don't agree"
    assert out["totals"]["open"] == 1


@pytest.mark.parametrize(("text", "why"), [
    ("{not json", "pkg/lean-audit.json: not JSON ("),
    ("[]", "pkg/lean-audit.json: not a JSON object"),
    (json.dumps({"deployed": "R.G"}), "pkg/lean-audit.json: `deployed` is not a list of names"),
    (json.dumps(lock({"R.Other": {}}, ["R.G", "R.Other"])),
     "pkg/lean-audit.json: `deployed` names R.G, which `guarantees` has no record of"),
    (json.dumps({"deployed": ["R.G"]}), "pkg/lean-audit.json: `deployed` names R.G, which `guarantees` has no record of"),
    (json.dumps({"witnesses": ["W.G"]}), "pkg/lean-audit.json: `witnesses` is not an object"),
    (json.dumps({"unwitnessed": "W.G"}), "pkg/lean-audit.json: `unwitnessed` is not an object"),
])
def test_a_lock_it_cannot_read_is_refused_by_name(text, why):
    with pytest.raises(deployed.Refused) as e:
        run({"ok/lean-audit.json": {"open": {}}, "pkg/lean-audit.json": text})
    assert str(e.value).startswith(f"verity lean deployed: {why}")
    assert "\n" not in str(e.value)


def git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                           *argv], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, files: dict[str, dict | str]) -> str:
    for rel, v in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(v if isinstance(v, str) else json.dumps(v))
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "locks")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    return tmp_path


def test_it_reads_the_tracked_locks_at_rev_from_git_alone_leaving_out_the_auditors_fixtures(repo, monkeypatch, capsys):
    fixtures = CHECK.FIXTURES.relative_to(CHECK.ROOT).as_posix()
    first = commit(repo, {"a/lean-audit.json": lock({"G.A": {"open": ["G.P"]}}, ["G.A"], open={"G.P": entry()}),
                          f"{fixtures}/control/lean-audit.json": "{not json", "a/other.json": "{not json"})
    commit(repo, {"a/lean-audit.json": "{not json", "b/lean-audit.json": {"open": {}}})
    (repo / "a" / "lean-audit.json").write_text("uncommitted edits are not read either")
    monkeypatch.chdir(repo / "a")
    assert verity_cli.main(["lean", "deployed", "--rev", first[:10], "--today", "2030-06-15"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["commit"], out["today"], out["locks"]) == (first, "2030-06-15", ["a/lean-audit.json"])
    assert [r["gap"] for r in out["guarantees"]] == [["G.P"]]


def test_a_refused_lock_ends_it_with_exit_1_and_one_line(repo, monkeypatch, capsys):
    commit(repo, {"a/lean-audit.json": "{not json"})
    monkeypatch.chdir(repo)
    assert verity_cli.main(["lean", "deployed", "--today", "2030-06-15"]) == 1
    said = capsys.readouterr()
    assert said.out == ""
    assert said.err.startswith("verity lean deployed: a/lean-audit.json: not JSON (") and said.err.count("\n") == 1


def test_a_rev_that_is_no_commit_ends_it_with_exit_1(repo, monkeypatch, capsys):
    commit(repo, {"a/lean-audit.json": {"open": {}}})
    monkeypatch.chdir(repo)
    assert verity_cli.main(["lean", "deployed", "--rev", "no-such-rev", "--today", "2030-06-15"]) == 1
    assert capsys.readouterr().err.startswith("verity lean deployed: no commit 'no-such-rev' here")


def test_the_help_says_where_today_and_the_gap_are_taken(capsys):
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["lean", "deployed", "--help"])
    assert e.value.code == 0
    said = " ".join(capsys.readouterr().out.split())
    assert "default today's date in America/Los_Angeles" in said
    assert "The gap is computed by the running checkout's Lean check (tools/verity/lean/check.py), on REV's locks." in said


def test_on_the_trees_own_locks_each_gap_is_the_lean_checks_and_each_gap_premise_has_a_row():
    commit_, texts = deployed.locks_at(ROOT, "HEAD", CHECK)
    out = deployed.rows(texts, TODAY, commit_, CHECK)
    assert out["locks"] == sorted(texts)
    names = {p["name"] for p in out["premises"]}
    for r in out["guarantees"]:
        assert r["gap"] == CHECK.trust_gaps(json.loads(texts[r["lock"]]))[r["name"]], r["name"]
        assert sorted(set(r["gap"]) - names) == [], r["name"]
