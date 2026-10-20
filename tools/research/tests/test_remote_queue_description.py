"""`research run --queue` places by the cluster's current description, not by the one a lane's branch was cut with (INF-156: a
branch from before main renewed the nodes' stops and shares had the queue refuse vy-nebius-1 and vy-nebius-2 as stopped). The
`cluster` package here is a stand-in that answers with the description and kinds it was given, so only `queue_placement`'s
choice is under test."""
from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

from research import remote

FAKE_MAIN = """\
import json, sys
from pathlib import Path
desc = Path(sys.argv[2])
text = desc.read_text().strip()
kinds = sorted(p.name for p in (desc.resolve().parent.parent / "kinds").iterdir())
if text.startswith("stopped"):
    print(json.dumps({"refused": f"n1: {text}"}))
else:
    print(json.dumps({"node": "n1", "machine": "n1", "why": text, "kinds": kinds, "job": {}, "warnings": ["adhoc"]}))
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def _describe(repo: Path, text: str, msg: str) -> str:
    (repo / "infra/cluster/descriptions/nebius.toml").write_text(text + "\n")
    _git(repo, "commit", "-qam", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path) -> Path:
    d = tmp_path / "repo"
    cluster = d / "infra" / "cluster"
    (cluster / "src" / "cluster").mkdir(parents=True)
    (cluster / "src" / "cluster" / "__init__.py").write_text("")
    (cluster / "src" / "cluster" / "submit.py").write_text("")
    (cluster / "src" / "cluster" / "__main__.py").write_text(FAKE_MAIN)
    (cluster / "descriptions").mkdir()
    (cluster / "kinds").mkdir()
    (cluster / "kinds" / "lean.toml").write_text("")
    (cluster / "descriptions" / "nebius.toml").write_text("stopped at 19 Oct 15:00Z\n")
    _git(d, "init", "-q", "-b", "lane")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "cut")
    return d


def test_a_branch_cut_before_main_renewed_the_nodes_is_placed_by_mains_description(repo):
    cut = _git(repo, "rev-parse", "HEAD")
    renewed = _describe(repo, "renewed to 20 Oct 15:00Z", "renew")
    _git(repo, "update-ref", "refs/remotes/origin/main", renewed)
    _git(repo, "reset", "-q", "--hard", cut)
    (repo / "work.txt").write_text("the lane's own change\n")
    _git(repo, "add", "work.txt")
    _git(repo, "commit", "-qm", "lane work")

    out = remote.queue_placement(repo, "r1", {"project": "verity", "kind": "lean-audit"})
    assert out["why"] == "renewed to 20 Oct 15:00Z" and out["kinds"] == ["lean.toml"], "main's stops, the tree's kinds"
    assert out["warnings"][0] == "adhoc" and out["warnings"][1].startswith(
        f"placed by origin/main's infra/cluster/descriptions/nebius.toml (its {renewed[:10]}, which this tree's HEAD lacks)")
    assert (repo / "infra/cluster/descriptions/nebius.toml").read_text().startswith("stopped"), "the tree is left as it is"

    _git(repo, "update-ref", "refs/remotes/origin/main", _describe(repo, "stopped again", "stop"))
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    with pytest.raises(remote.Refused, match=r"the queue refuses the job: n1: stopped again \(placed by origin/main's"):
        remote.queue_placement(repo, "r2", {"project": "verity"})


def test_a_tree_with_mains_description_or_without_git_places_by_its_own(repo, tmp_path):
    with pytest.raises(remote.Refused, match=r"n1: stopped at 19 Oct 15:00Z$"):
        remote.queue_placement(repo, "r1", {"project": "verity"})           # no origin/main: the tree's, no note

    _git(repo, "update-ref", "refs/remotes/origin/main", _describe(repo, "renewed", "renew"))
    assert remote.queue_placement(repo, "r2", {})["warnings"] == ["adhoc"]
    _describe(repo, "the branch's own edit", "edit")                        # main's change is in HEAD: the branch's edit stands
    assert remote.queue_placement(repo, "r3", {})["why"] == "the branch's own edit"

    shipped = tmp_path / "archive"
    _git(repo, "archive", "-o", str(tmp_path / "tree.tar"), "HEAD")
    with tarfile.open(tmp_path / "tree.tar") as t:
        t.extractall(shipped, filter="data")
    assert remote.queue_placement(shipped, "r4", {})["why"] == "the branch's own edit", "no git: the tree's"
