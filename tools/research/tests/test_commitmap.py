"""`research.commitmap`: after a history rewrite, the merge gate reads a record of an old commit for its image only through an
accepted map and only when the tree is the same; it refuses old history; `research git remap` moves a clone's local work over."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research import commitmap as CM
from research.cli import main as research_main
from research.merge import main as merge_main
from research.store.local import LocalStore
from test_merge_gate import CHECK_FILES, CHECKED

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_AUTHOR_DATE": "2026-09-29T00:00:00Z", "GIT_COMMITTER_DATE": "2026-09-29T00:00:00Z"}


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env={**os.environ, **ENV, **(env or {})}).stdout.strip()


def _commit(repo: Path, name: str, text: str, msg: str) -> str:
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _image(repo: Path, commit: str, parents: list[str], note: str = "") -> str:
    """What a rewrite makes of `commit`: the same tree (or another, with `note`), new parents, the message kept."""
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}")
    if note:
        _git(repo, "read-tree", tree)
        blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=note, capture_output=True, text=True, check=True).stdout.strip()
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},stub.txt")
        tree = _git(repo, "write-tree")
        _git(repo, "read-tree", "HEAD")
    msg = _git(repo, "log", "-1", "--format=%B", commit)
    return _git(repo, "commit-tree", tree, *[x for p in parents for x in ("-p", p)], "-m", msg,
                env={"GIT_COMMITTER_DATE": "2026-10-01T00:00:00Z"})                  # a new commit, as a rewrite makes one


def _publish(store: LocalStore, tmp_path: Path, lines: str, verdicts: list[tuple[str, str]]) -> str:
    f = tmp_path / "commit-map.txt"
    f.write_text(lines)
    art = store.put_artifact(CM.KIND, {"from": "old", "to": "new"}, None, file=f)
    for by, value in verdicts:
        store.add_label(art, CM.LABEL, value, by=by)
    return art


def _gate_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    main = tmp_path / "main"
    (main / "mytools").mkdir(parents=True)
    (main / "mytools" / "decl_check.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1')\n")
    (main / "tools" / "verity" / "check").mkdir(parents=True)
    for rel, text in CHECK_FILES.items():
        (main / rel).write_text(text)
    (main / ".gitignore").write_text("__pycache__/\n")
    (main / "a.txt").write_text("a\n")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    feat = tmp_path / "feat"
    _git(main, "worktree", "add", "-q", "-b", "feat", str(feat))
    return main, feat, _commit(feat, "a.txt", "a\nb\n", "feat")


def _record(tmp_path: Path, checkout: Path, monkeypatch) -> None:
    monkeypatch.chdir(checkout)
    assert research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                          "--cwd", "source", "--", *CHECKED]) == 0


# ---- the map and the gate --------------------------------------------------------------------------------------------------------------

def test_parse_reads_the_payload_and_filter_repos_own_file():
    o, n = "a" * 40, "b" * 40
    assert CM.parse(f"old new\n{o} {n} 1\n{'c' * 40} {'0' * 40} 0\n") == {o: (n, True)}
    assert CM.parse(f"old                                      new\n{o} {n}\n") == {o: (n, False)}


def test_an_attempt_of_an_old_commit_counts_for_its_image_only_through_an_accepted_map_with_the_same_tree(tmp_path, monkeypatch, capsys):
    main, feat, x = _gate_repo(tmp_path)
    _record(tmp_path, feat, monkeypatch)
    y = _image(main, x, [_git(main, "rev-parse", "main")])            # the rewrite's image: same tree, another parent chain
    z = _image(main, x, [_git(main, "rev-parse", "main")], note="stub")
    _git(main, "branch", "feat2", y)
    _git(main, "branch", "feat3", z)
    monkeypatch.chdir(main)
    assert merge_main(["feat2", "--dry-run"]) == 3 and "no passing `check` attempt" in capsys.readouterr().err
    st = LocalStore()
    art = _publish(st, tmp_path, f"{x} {y} 1\n{'e' * 40} {z} 0\n", [("some-lane", CM.ACCEPTED)])
    assert merge_main(["feat2", "--dry-run"]) == 3                     # accepted by someone who may not accept a map
    st.add_label(art, CM.LABEL, CM.ACCEPTED, by="daniel")
    assert merge_main(["feat2", "--dry-run"]) == 0 and "may be merged" in capsys.readouterr().out
    st.add_label(art, CM.LABEL, CM.RETRACTED, by="daniel", ts="2099-01-01T00:00:00Z")
    assert merge_main(["feat2", "--dry-run"]) == 3                     # retracted: the map no longer applies
    st2 = tmp_path / "store2"
    monkeypatch.setenv("RESEARCH_STORE", str(st2))
    _record(tmp_path, feat, monkeypatch)
    monkeypatch.chdir(main)
    other = LocalStore()
    _publish(other, tmp_path, f"{x} {z} 1\n", [("daniel", CM.ACCEPTED)])      # a map claiming the tree held, where it did not
    capsys.readouterr()
    assert merge_main(["feat3", "--dry-run"]) == 3 and "no passing `check` attempt" in capsys.readouterr().err


def test_the_gate_refuses_a_ref_that_holds_old_history(tmp_path, monkeypatch, capsys):
    main, feat, x = _gate_repo(tmp_path)
    _record(tmp_path, feat, monkeypatch)
    y = _image(main, x, [_git(main, "rev-parse", "main")])
    _publish(LocalStore(), tmp_path, f"{x} {y} 1\n", [("daniel", CM.ACCEPTED)])
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    err = capsys.readouterr().err
    assert "1 commit(s) a history rewrite replaced" in err and "research git remap" in err


def test_build_marks_which_commits_kept_their_tree(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    a = _commit(repo, "f.txt", "1\n", "a")
    b = _commit(repo, "f.txt", "2\n", "b")
    a2 = _image(repo, a, [])
    b2 = _image(repo, b, [a2], note="stub")
    (repo / "cm").write_text(f"old                                      new\n{a} {a2}\n{b} {b2}\n")
    out = CM.build(repo, repo, repo / "cm")
    assert CM.parse(out) == {a: (a2, True), b: (b2, False)}


# ---- remap and the CLI ------------------------------------------------------------------------------------------------------------------

def _clone_with_local_work(tmp_path: Path) -> tuple[Path, CM.Maps, dict[str, str]]:
    """Old history A-B on main, a local branch `work` = B-L1-L2 (checked out) and `mixed` = L1 merged with old C; the rewrite's
    images A'-B'-C' are in the repository as they would be after `git fetch`."""
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    a = _commit(repo, "f.txt", "1\n", "A")
    b = _commit(repo, "f.txt", "2\n", "B")
    c = _commit(repo, "g.txt", "c\n", "C")
    _git(repo, "checkout", "-q", "-b", "work", b)
    l1 = _commit(repo, "w.txt", "1\n", "L1")
    l2 = _commit(repo, "w.txt", "2\n", "L2 names B " + b[:8])
    _git(repo, "branch", "mixed", l1)
    _git(repo, "checkout", "-q", "mixed")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge old C", c)
    m = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "work")
    a2 = _image(repo, a, [])
    b2 = _image(repo, b, [a2])
    c2 = _image(repo, c, [b2])
    _git(repo, "update-ref", "refs/remotes/origin/main", c2)
    maps = CM.Maps({a: (a2, True), b: (b2, True), c: (c2, True)})
    return repo, maps, {"a": a, "b": b, "c": c, "l1": l1, "l2": l2, "m": m, "a2": a2, "b2": b2, "c2": c2}


def test_remap_recreates_local_commits_over_the_images(tmp_path):
    repo, maps, s = _clone_with_local_work(tmp_path)
    dry = CM.remap(repo, maps, ["work"], dry_run=True, log=lambda *_: None)
    assert dry[0]["recreated"] == 2 and _git(repo, "rev-parse", "work") == s["l2"]
    rep = {r["branch"]: r for r in CM.remap(repo, maps, log=lambda *_: None)}
    assert rep["work"]["recreated"] + rep["mixed"]["recreated"] == 3                   # L1 once, for both branches; L2; the merge
    assert rep["main"]["new"] == s["c2"] and rep["main"]["recreated"] == 0              # a branch with no local work: its image
    new = _git(repo, "rev-parse", "work")
    assert _git(repo, "rev-parse", f"{new}^{{tree}}") == _git(repo, "rev-parse", f"{s['l2']}^{{tree}}")
    assert _git(repo, "rev-parse", f"{new}~2") == s["b2"] and _git(repo, "log", "-1", "--format=%B", new) == f"L2 names B {s['b'][:8]}"
    assert _git(repo, "log", "-1", "--format=%an <%ae> %ad", new) == _git(repo, "log", "-1", "--format=%an <%ae> %ad", s["l2"])
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "work" and not _git(repo, "status", "--porcelain")
    merge = _git(repo, "rev-parse", "mixed")
    assert _git(repo, "rev-parse", f"{merge}^2") == s["c2"] and _git(repo, "rev-parse", f"{merge}^1~1") == s["b2"]
    assert not maps.old_history(repo, "work") and not maps.old_history(repo, "mixed")
    assert CM.remap(repo, maps, log=lambda *_: None)[0]["moved"] is False               # idempotent: nothing old is left


def test_old_history_names_what_a_ref_holds_of_the_old_commits(tmp_path):
    repo, maps, s = _clone_with_local_work(tmp_path)
    assert set(maps.old_history(repo, "work")) == {s["a"], s["b"]}
    assert maps.old_history(repo, "work", "main") == []                 # main holds old history too: nothing new of work's is old
    assert maps.old_history(repo, "origin/main") == [] and not CM.Maps().old_history(repo, "work")


def test_the_git_cli_resolves_abbreviations_and_checks_a_ref(tmp_path, capsys):
    repo, maps, s = _clone_with_local_work(tmp_path)
    mf = tmp_path / "map.txt"
    mf.write_text("".join(f"{o} {n} 1\n" for o, (n, _) in maps.pairs.items()))
    assert research_main(["git", "old2new", s["b"][:8], "--map", str(mf)]) == 0
    assert capsys.readouterr().out == f"{s['b']} {s['b2']}\n"
    assert research_main(["git", "old2new", "0123456", "--map", str(mf)]) == 1
    assert research_main(["git", "check", "work", "--repo", str(repo), "--map", str(mf)]) == 1
    assert research_main(["git", "check", "origin/main", "--repo", str(repo), "--map", str(mf)]) == 0
    assert research_main(["git", "remap", "--repo", str(repo), "--map", str(mf)]) == 0
    assert research_main(["git", "check", "work", "--repo", str(repo), "--map", str(mf)]) == 0


def test_with_no_map_nothing_changes(tmp_path):
    empty = CM.Maps.load(LocalStore(tmp_path / "s"))
    assert not empty and empty.equivalent("f" * 40) == ["f" * 40] and empty.new("f" * 40) is None
    with pytest.raises(SystemExit):
        CM.main(["old2new"])
