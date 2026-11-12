"""`spot_status`: each resampling hole is closed only where its fix's commit is an ancestor of the commit a run's record
names, decided in a tiny git repository of made-up commits; a run whose code can't be read is not known; the label is
`SPOT soundness` only when both holes are closed; and the theorem's open premises are the fixture lock's."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_bench import cli, prove_records
from verity_bench import spot_status as S

LOCK = Path(__file__).parent / "fixtures" / "spot_status" / "lean-audit.json"
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
