"""`spot_status`: each resampling hole is closed only where its fix's commit is an ancestor of the commit a run's record
names, decided in a tiny git repository of made-up commits; a run whose code can't be read is not known; the label is
`SPOT soundness` only when both holes are closed; and the theorem's open premises are the fixture lock's."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_bench import cli, prove_records
from verity_bench import exfiltration as E
from verity_bench import spot_status as S

#: a lock's shape under a name no Lean tool takes for a package's lock
LOCK = Path(__file__).parent / "fixtures" / "spot_status" / "lock.json"
REDRAWN, RECOMMITTED = S.HOLES
BASE, ONE, BOTH = "r20261112-000001-aaaa", "r20261112-000002-bbbb", "r20261112-000003-cccc"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository of three commits in a line (the base, the first fix, the second) and a side commit off the base,
    with the fixes' data file naming the two fixes."""

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    commits = {}
    for name in ("base", "fix1", "fix2"):
        _git(root, "commit", "-q", "--allow-empty", "-m", name)
        commits[name] = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "-b", "side", commits["base"])
    _git(root, "commit", "-q", "--allow-empty", "-m", "side")
    commits["side"] = _git(root, "rev-parse", "HEAD")
    data = tmp_path / "spot_holes.json"
    data.write_text(json.dumps({"fixes": {REDRAWN: commits["fix1"], RECOMMITTED: commits["fix2"]}}))
    monkeypatch.setattr(S, "FIXES", data)
    monkeypatch.setattr(S, "LOCK", LOCK)
    return SimpleNamespace(root=root, data=data, **commits)


class Store:
    def __init__(self, sources: dict[str, dict | None]):
        self.sources = sources

    def get_attempt(self, run: str) -> dict | None:
        src = self.sources[run]
        return None if src is None else {"id": run, "source": src}


def _status(repo, **runs: dict | None) -> dict:
    return S.status(list(runs), store=Store(runs), root=repo.root)


def _by_hole(doc: dict) -> dict[str, str]:
    return {h["hole"]: h["status"] for h in doc["holes"]}


def test_both_holes_open_at_the_base(repo):
    doc = _status(repo, **{BASE: {"commit": repo.base, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert "- label: **SPOT**\n- redrawn sample: open\n" in text and "- re-committed execution phase: open\n" in text
    assert f"  - at every commit read: the fix's commit {repo.fix1[:12]} is not an ancestor" in text
    assert "proved" not in text


def test_one_hole_closed_where_its_fix_is_an_ancestor(repo):
    doc = _status(repo, **{ONE: {"commit": repo.fix1, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    assert "- redrawn sample: closed\n- re-committed execution phase: open\n" in S.render(doc)


def test_both_closed_flips_the_label(repo):
    doc = _status(repo, **{BOTH: {"commit": repo.fix2, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "closed"} and doc["label"] == "SPOT soundness"
    text = S.render(doc)
    assert "- label: **SPOT soundness**\n- redrawn sample: closed\n- re-committed execution phase: closed\n" in text
    assert f"- runs read: 1, at 1 commit: `{repo.fix2[:12]}` (1 run)" in text


def test_a_side_commit_holds_neither_fix(repo):
    """An ancestor, not a later date: a commit off the base after both fixes still has neither."""

    doc = _status(repo, **{BOTH: {"commit": repo.side, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"}


def test_one_open_run_holds_the_hole_open_over_the_rest(repo):
    doc = _status(repo, **{BASE: {"commit": repo.base, "dirty": False}, BOTH: {"commit": repo.fix2, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    assert f"  - at `{repo.fix2[:12]}`: closed\n" in S.render(doc)


def test_a_commit_this_repository_lacks_is_not_known(repo):
    lost = "d" * 40
    doc = _status(repo, **{BOTH: {"commit": lost, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"} and doc["label"] == "SPOT"
    assert f"doesn't hold the run's commit {lost} (`git fetch origin {lost}`)" in S.render(doc)


def test_a_run_whose_code_cant_be_read_is_not_known_and_never_closes(repo):
    doc = _status(repo, **{BOTH: {"commit": repo.fix2, "dirty": False}, ONE: {"commit": repo.fix2, "dirty": True},
                           BASE: None})
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert f"  - {ONE}: not known: it ran a dirty tree at {repo.fix2[:12]}" in text
    assert f"  - {BASE}: not known: the store holds no record of it" in text
    assert _by_hole(S.status([], store=Store({}), root=repo.root)) == {REDRAWN: "not known", RECOMMITTED: "not known"}


def test_a_hole_with_no_fix_declared_is_open(repo):
    repo.data.write_text(json.dumps({"fixes": {REDRAWN: repo.fix1}}))
    doc = _status(repo, **{BOTH: {"commit": repo.fix2, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    assert "  - at every commit read: no fix's commit is declared" in S.render(doc)
    lost = _status(repo, **{BOTH: {"commit": "d" * 40, "dirty": False}})
    assert _by_hole(lost) == {REDRAWN: "not known", RECOMMITTED: "open"}
    assert _by_hole(S.status([], store=Store({}), root=repo.root)) == {REDRAWN: "not known", RECOMMITTED: "open"}


def test_the_fixes_file_names_only_holes_by_full_commit(repo):
    repo.data.write_text(json.dumps({"fixes": {"another hole": repo.fix1}}))
    with pytest.raises(ValueError, match="names another hole, which is no hole of"):
        S.fixes(repo.data)
    repo.data.write_text(json.dumps({"fixes": {REDRAWN: repo.fix1[:12]}}))
    with pytest.raises(ValueError, match="not a full commit id"):
        S.fixes(repo.data)


def test_the_tree_declares_only_holes():
    assert set(S.fixes(S.FIXES)) <= set(S.HOLES)


def test_the_theorem_and_its_open_premises_are_the_locks(tmp_path):
    t = S.theorem(tmp_path, LOCK)
    assert t["name"] == "Verity.Sampled.remote_soundness_deployedReady"
    assert t["open_premises"] == ["Verity.Sampled.Instance.SessionsReady", "Verity.Vocab.PouwSound"]
    lock = json.loads(LOCK.read_text())
    lock["guarantees"]["Verity.Sampled.onsite_soundness_deployedReady"] = {"premises": {"open": ["Verity.Vocab.WipeSound"]}}
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    assert S.theorem(tmp_path, tmp_path / "lock.json")["name"] == "Verity.Sampled.onsite_soundness_deployedReady"
    del lock["guarantees"]["Verity.Sampled.onsite_soundness_deployedReady"]
    del lock["guarantees"]["Verity.Sampled.remote_soundness_deployedReady"]
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    assert S.theorem(tmp_path, tmp_path / "lock.json")["name"] is None


def test_the_block_cites_the_theorem_with_its_open_premises(repo):
    text = S.render(_status(repo, **{BASE: {"commit": repo.base, "dirty": False}}))
    assert ("- theorem: `Verity.Sampled.remote_soundness_deployedReady`, on the 2 open premises its record in "
            f"`{LOCK}` takes: `Verity.Sampled.Instance.SessionsReady`, `Verity.Vocab.PouwSound`") in text


def _entry(run: str, commit: str | None, **more) -> dict:
    """A run's entry in a served record's `runs`, as its reader writes it."""

    return {"run": run, "commit": commit, "dirty": False, "why": None, "machine": "a-made-up-machine"} | more


def _served(commit: str | None, **runs: dict | None) -> dict:
    """A served record as its reader gives it, its second stage at ``commit``, with only the runs ``runs`` given."""

    return {"run": BOTH, "commit": commit, "runs": dict.fromkeys(S.SERVED_RUNS) | runs}


def test_a_served_record_is_read_at_its_own_commits(repo):
    """The block over a served run reads the commit its record names and each of its four runs' commits, listed; with
    all four read, a hole whose fix is an ancestor of every one is closed, and both closed is `SPOT soundness`."""

    record = _served(repo.fix1, **{name: _entry(ONE, repo.fix1) for name in S.SERVED_RUNS})
    doc = S.served(record, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert "is an ancestor of the commit the served run's record names and of each of its runs' commits." in text
    assert "- redrawn sample: closed\n- re-committed execution phase: open\n" in text
    assert (f"- the code {BOTH} ran: its record's `commit`, the second stage's, `{repo.fix1[:12]}`; its runs:\n"
            + "".join(f"  - {name}: {ONE} at `{repo.fix1[:12]}` on a-made-up-machine\n" for name in S.SERVED_RUNS)) in text
    assert "- runs read:" not in text and "not given" not in text
    both = _served(repo.fix2, **{name: _entry(BOTH, repo.fix2) for name in S.SERVED_RUNS})
    assert S.served(both, root=repo.root)["label"] == "SPOT soundness"


def test_a_hole_never_reads_closed_where_some_of_the_served_runs_were_not_given(repo):
    """The rehearsal's shape: the reader was given only the first and second stages, so the serve and the replay are not
    given and no commit of theirs is read. A hole whose fix is an ancestor of every commit read is closed at the runs
    read, naming those not given, never bare closed, and the label stays `SPOT` with both holes so."""

    record = _served(repo.fix2, first=_entry(ONE, repo.fix2), second=_entry(BOTH, repo.fix2))
    doc = S.served(record, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: S.CLOSED_READ, RECOMMITTED: S.CLOSED_READ} and doc["label"] == "SPOT"
    assert (doc["served"]["read"], doc["served"]["not_given"]) == (["first", "second"], ["serve", "replay"])
    text = S.render(doc)
    assert ("- label: **SPOT**\n"
            "- redrawn sample: closed at the runs read (first, second); serve, replay not given\n"
            "- re-committed execution phase: closed at the runs read (first, second); serve, replay not given\n") in text
    assert (f"  - serve: {S.NOT_GIVEN}\n  - first: {ONE} at `{repo.fix2[:12]}` on a-made-up-machine\n"
            f"  - replay: {S.NOT_GIVEN}\n  - second: {BOTH} at `{repo.fix2[:12]}` on a-made-up-machine\n") in text
    assert ": closed\n" not in text and "SPOT soundness**" not in text
    # an open hole stays open, and one the read commits can't settle stays not known: only closed is qualified
    one = S.served(_served(repo.fix1, first=_entry(ONE, repo.fix1), second=_entry(BOTH, repo.fix1)), root=repo.root)
    assert _by_hole(one) == {REDRAWN: S.CLOSED_READ, RECOMMITTED: "open"}
    lost = S.served(_served("d" * 40, second=_entry(BOTH, "d" * 40)), root=repo.root)
    assert _by_hole(lost) == {REDRAWN: "not known", RECOMMITTED: "not known"}


def test_a_hole_is_closed_on_a_served_run_only_at_every_one_of_its_commits(repo):
    """One run at a commit that lacks a fix holds the hole open over the record's own commit, which holds it."""

    record = _served(repo.fix2, first=_entry(ONE, repo.base), second=_entry(BOTH, repo.fix2))
    doc = S.served(record, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert f"  - at `{repo.fix2[:12]}`: closed\n" in text and f"  - at `{repo.base[:12]}`: open: the fix's commit" in text


def test_a_served_commit_this_repository_lacks_or_a_run_with_no_commit_is_not_known(repo):
    lost = "d" * 40
    doc = S.served(_served(lost, second=_entry(BOTH, lost)), root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"}
    assert f"doesn't hold the run's commit {lost} (`git fetch origin {lost}`)" in S.render(doc)
    nameless = _served(repo.fix2, first=_entry(ONE, None, why="its directory holds no `job.json`"),
                       replay=_entry(BASE, repo.fix2, dirty=True), second=_entry(BOTH, repo.fix2))
    doc = S.served(nameless, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert f"  - first: {ONE} on a-made-up-machine: not known: its directory holds no `job.json`\n" in text
    assert f"  - replay: {BASE} at `{repo.fix2[:12]}` on a-made-up-machine: not known: it ran a dirty tree at" in text
    doc = S.served({"run": BOTH, "runs": {"second": _entry(BOTH, repo.fix2)}}, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"}
    assert ("its record's `commit`, the second stage's, not known: the record names no `commit`, the commit its second "
            "stage ran; its runs:") in S.render(doc)
    for bad, why in (({"runs": {"third": None}}, "an object of serve, first, replay, second, not third"),
                     ({"runs": {"first": "r1"}}, "`runs.first` is a run's entry, one JSON object, not str")):
        with pytest.raises(ValueError, match=re.escape(why)):
            S.served({"run": BOTH, "commit": repo.fix2} | bad, root=repo.root)


def test_exfiltration_reads_the_holes_at_a_served_records_commit(repo, tmp_path, monkeypatch, capsys):
    """`verity bench exfiltration --served-run` reads the holes at the served run's own commits, not at the planner's
    runs of record (here at the base, where both are open); without it, at those runs. Proofs' section on what the
    audit showed (`verity_bench.served_row`) is printed between the figures and the holes where the tree holds it, and
    is skipped where it doesn't."""

    store = Store({BASE: {"commit": repo.base, "dirty": False}})
    monkeypatch.setattr(S.jobs, "StoreCompletion", lambda **k: SimpleNamespace(store=store))
    monkeypatch.setattr(S.jobs, "repository_root", lambda *a: repo.root)
    monkeypatch.setattr(E, "build", lambda **k: {"runs": [BASE]})
    monkeypatch.setattr(E, "render", lambda doc: "the figures")
    (served := tmp_path / "served.json").write_text(json.dumps(
        _served(repo.fix2, **{name: _entry(BOTH, repo.fix2) for name in S.SERVED_RUNS})))

    find_spec, name = importlib.util.find_spec, "verity_bench.served_row"
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == name else find_spec(n, *a))
    assert E.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("the figures\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- label: **SPOT**\n- redrawn sample: open\n" in out and "- runs read: 1, at 1 commit" in out
    assert E.main(["--served-run", str(served)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("the figures\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- label: **SPOT soundness**\n- redrawn sample: closed\n- re-committed execution phase: closed\n" in out
    assert f"its record's `commit`, the second stage's, `{repo.fix2[:12]}`; its runs:\n" in out

    section = types.ModuleType(name)
    section.account = lambda record: {"run": record["run"]}
    section.render = lambda shown: f"## what {shown['run']}'s audit showed"
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: importlib.machinery.ModuleSpec(n, None)
                        if n == name else find_spec(n, *a))
    monkeypatch.setitem(sys.modules, name, section)
    monkeypatch.setattr(sys.modules["verity_bench"], "served_row", section, raising=False)
    assert E.main(["--served-run", str(served)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"the figures\n\n## what {BOTH}'s audit showed\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- label: **SPOT soundness**\n" in out
    assert E.main(["--json", "--served-run", str(served)]) == 0
    held = json.loads(capsys.readouterr().out)
    assert held["served_run"] == {"run": BOTH} and held["spot_holes"]["served"]["commit"] == repo.fix2
    assert E.main(["--served-run", str(tmp_path / "none.json")]) == 2


def test_prove_spot_prints_the_block_over_its_input_runs(repo, monkeypatch, capsys):
    """`verity bench prove --system spot` reads the runs among its inputs (not its arts) and prints the block after its
    table, and holds it in `--json`; `--system flock` prints none."""

    doc = {"inputs": [{"targets": ["art:" + "a" * 64, BOTH]}, {"targets": [BOTH, ONE]}]}
    monkeypatch.setattr(prove_records, "build", lambda *a, **k: json.loads(json.dumps(doc)))
    monkeypatch.setattr(prove_records, "render", lambda d, rows: "the table\n")
    store = Store({BOTH: {"commit": repo.fix2, "dirty": False}, ONE: {"commit": repo.fix1, "dirty": False}})
    monkeypatch.setattr(S.jobs, "StoreCompletion", lambda **k: SimpleNamespace(store=store))
    monkeypatch.setattr(S.jobs, "repository_root", lambda *a: repo.root)
    args = argparse.Namespace(system="spot", workload="glm53", sampling=None, provers=1, cores=10, local_only=True,
                              steady_rows=None, json=False, rows=None)
    assert cli._prove(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("the table\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- redrawn sample: closed\n- re-committed execution phase: open\n" in out and "- runs read: 2, at 2 commits" in out
    assert cli._prove(argparse.Namespace(**{**vars(args), "json": True})) == 0
    held = json.loads(capsys.readouterr().out)["spot_holes"]
    assert held["label"] == "SPOT" and held["runs"] == 2
    assert cli._prove(argparse.Namespace(**{**vars(args), "system": "flock"})) == 0
    assert capsys.readouterr().out == "the table\n"
