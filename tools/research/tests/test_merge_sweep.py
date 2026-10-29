"""`research merge`'s pull request sweep: after a merge lands on the remote's branch, every open pull request whose head is on it
is closed with a comment naming the commit it landed in, and every one based on a branch that landed is retargeted onto it.  The
remote is a bare repository in tmp and GitHub is a fake `gh`, so nothing here reaches the network."""
from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path

import research.merge as M
from research.cli import main as research_main
from test_merge_gate import CHECK_FILES, CHECKED


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


class FakeGitHub:
    """`gh pr list | close | edit` over a dict of open pull requests, recording every mutating call."""

    def __init__(self, prs: list[dict]):
        self.open = {p["number"]: dict(p) for p in prs}
        self.merged: list[dict] = []
        self.mutations: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> str:
        if args[:2] == ("pr", "list"):
            assert "merged" not in args, "a listing of merged PRs by number misses those that waited in the queue"
            return json.dumps(list(self.open.values()))
        if args[:2] == ("api", "graphql"):
            assert f"query={M.MERGED_QUERY}" in args and "orderBy: {field: UPDATED_AT, direction: DESC}" in M.MERGED_QUERY
            return json.dumps(self.merged)
        self.mutations.append(args)
        n = int(args[2])
        if args[:2] == ("pr", "close"):
            del self.open[n]
        elif args[:2] == ("pr", "edit"):
            self.open[n]["baseRefName"] = args[args.index("--base") + 1]
        return ""


def _world(tmp_path: Path):
    """A bare `origin`, a clone on main, and branches A <- B <- E and C <- D pushed with a pull request each; B (with A) merged into
    main and pushed.  Returns (clone, fake GitHub, {branch: tip}, the merge commit)."""
    origin, repo = tmp_path / "origin.git", tmp_path / "main"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    repo.mkdir()
    (repo / "mytools").mkdir()
    (repo / "mytools" / "decl_check.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1')\n")
    (repo / "tools" / "verity" / "check").mkdir(parents=True)
    for rel, text in CHECK_FILES.items():
        (repo / rel).write_text(text)
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    tips = {}
    for name, base in (("A", "main"), ("B", "A"), ("E", "B"), ("C", "main"), ("D", "C")):
        _git(repo, "checkout", "-q", "-b", name, base)
        (repo / f"{name}.txt").write_text(name + "\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
        _git(repo, "push", "-q", "origin", name)
        tips[name] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge train T (B)", "B")
    _git(repo, "push", "-q", "origin", "main")
    gh = FakeGitHub([{"number": n, "headRefName": h, "headRefOid": tips[h], "baseRefName": b}
                     for n, h, b in ((1, "A", "main"), (2, "B", "A"), (3, "C", "main"), (4, "D", "C"), (5, "E", "B"))])
    return repo, gh, tips, _git(repo, "rev-parse", "HEAD")


def test_the_sweep_closes_what_landed_and_retargets_its_children(tmp_path):
    repo, gh, _tips, merge = _world(tmp_path)
    lines: list[str] = []
    assert M.sweep(repo, gh=gh, log=lines.append) == ([1, 2], [5])
    note = f"Merged into main at {merge[:12]} (Merge train T (B))."
    assert gh.mutations == [("pr", "edit", "5", "--base", "main"),
                            ("pr", "close", "1", "--comment", note), ("pr", "close", "2", "--comment", note)]
    assert sorted(gh.open) == [3, 4, 5] and gh.open[5]["baseRefName"] == "main" and gh.open[4]["baseRefName"] == "C"
    assert any(f"close #1 (A): {note}" in line for line in lines)
    assert _heads(repo) == {"main", "C", "D", "E"}                                # A and B landed, and no open PR uses them


def _heads(repo: Path) -> set[str]:
    return {line.split("refs/heads/")[1] for line in _git(repo, "ls-remote", "--heads", "origin").splitlines()}


def test_the_sweep_keeps_a_landed_branch_that_moved_past_its_landed_head(tmp_path):
    """A commit pushed to A after A landed isn't on main: A's PR closes, and the branch stays for its owner."""
    repo, gh, _tips, _merge = _world(tmp_path)
    _git(repo, "checkout", "-q", "A")
    (repo / "late.txt").write_text("late\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "late")
    _git(repo, "push", "-q", "origin", "A")
    _git(repo, "checkout", "-q", "main")
    lines: list[str] = []
    assert M.sweep(repo, gh=gh, log=lines.append) == ([1, 2], [5])
    assert _heads(repo) == {"main", "A", "C", "D", "E"} and any(line.startswith("research merge: kept A") for line in lines)


def test_the_sweep_closes_a_pr_whose_pinned_head_landed_and_whose_head_since_only_merged_main(tmp_path):
    """PRs that merged `main` into heads a tip had pinned: the tip lands them as the pinned heads, so the sweep closes such a
    PR, saying so, and keeps its branch.  One whose head since adds content stays open."""
    repo, gh, tips, _merge = _world(tmp_path)
    _git(repo, "checkout", "-q", "-b", "train", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", f"Merge #3 ({tips['C'][:8]})", tips["C"])
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge train U (#3)", "train")
    _git(repo, "push", "-q", "origin", "main")
    landed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "C")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", tips["B"])
    _git(repo, "push", "-q", "origin", "C")
    gh.open[3]["headRefOid"] = moved = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "D")
    (repo / "D2.txt").write_text("more\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "more")
    _git(repo, "checkout", "-q", "main")
    lines: list[str] = []
    closed, _ = M.sweep(repo, gh=gh, log=lines.append)
    assert 3 in closed and 4 not in closed
    note = f"Merged into main at {landed[:12]} (Merge train U (#3)). Its head since, {moved[:12]}, adds nothing to main."
    assert ("pr", "close", "3", "--comment", note) in gh.mutations
    assert "C" in _heads(repo)


def test_the_sweep_is_idempotent_and_its_dry_run_changes_nothing(tmp_path):
    repo, gh, _tips, _merge = _world(tmp_path)
    lines: list[str] = []
    assert M.sweep(repo, dry_run=True, gh=gh, log=lines.append) == ([1, 2], [5])
    assert gh.mutations == [] and sorted(gh.open) == [1, 2, 3, 4, 5]
    assert _heads(repo) == {"main", "A", "B", "C", "D", "E"} and any("would delete origin's branch A" in line for line in lines)
    assert "would retarget #5 from B" in lines[0] and "would close #1" in lines[1]
    M.sweep(repo, gh=gh, log=lambda *_: None)
    done = list(gh.mutations)
    lines.clear()
    assert M.sweep(repo, gh=gh, log=lines.append) == ([], [])
    assert gh.mutations == done and "no open pull request is on origin/main" in lines[-1]


def test_the_sweep_reads_the_remote_branch_not_the_checkout(tmp_path):
    """C merged into the local main but not pushed: nothing about C moves until the push."""
    repo, gh, _tips, _merge = _world(tmp_path)
    M.sweep(repo, gh=gh, log=lambda *_: None)
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge C", "C")
    lines: list[str] = []
    assert M.sweep(repo, gh=gh, log=lines.append) == ([], [])
    assert "main has commits origin/main lacks" in lines[0]
    _git(repo, "push", "-q", "origin", "main")
    assert M.sweep(repo, gh=gh, log=lambda *_: None) == ([3], [4])


def test_landed_in_names_the_merge_or_the_commit_itself(tmp_path):
    repo, _gh, tips, merge = _world(tmp_path)
    assert M.landed_in(repo, tips["A"], merge) == merge
    assert M.landed_in(repo, tips["B"], merge) == merge
    init = _git(repo, "rev-parse", "HEAD^1")
    assert M.landed_in(repo, init, merge) == init                               # on the first-parent line
    _git(repo, "checkout", "-q", "-b", "F")
    (repo / "F.txt").write_text("F\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "F")
    f = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "F")                                  # a fast-forward: F's commit is on the line
    assert M.landed_in(repo, f, f) == f and M.landed_in(repo, merge, f) == merge and M.landed_in(repo, tips["A"], f) == merge


def test_a_merge_sweeps_after_it_lands_and_says_to_push_first(tmp_path, monkeypatch, capsys):
    """`research merge C` with a passing check: the merge lands locally, its sweep closes nothing of C's before the push and says
    so; `research merge --sweep` after the push closes C and retargets D."""
    repo, gh, _tips, _merge = _world(tmp_path)
    monkeypatch.setattr(M, "_gh_cli", lambda _repo, *args: gh(*args))
    M.sweep(repo, gh=gh, log=lambda *_: None)
    _git(repo, "checkout", "-q", "C")
    _git(repo, "merge", "-q", "--no-ff", "-m", "main into C", "main")          # the gate needs main's tip in C
    _git(repo, "push", "-q", "origin", "C")
    gh.open[3]["headRefOid"] = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    assert research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool",
                          "mytools.decl_check:T", "--cwd", "source", "--", *CHECKED]) == 0
    _git(repo, "checkout", "-q", "main")
    capsys.readouterr()
    assert M.main(["C"]) == 0
    out = capsys.readouterr().out
    assert "merged C" in out and "push it, then `research merge --sweep`" in out and sorted(gh.open) == [3, 4, 5]
    _git(repo, "push", "-q", "origin", "main")
    assert M.main(["--sweep", "--dry-run"]) == 0 and sorted(gh.open) == [3, 4, 5]
    assert M.main(["--sweep"]) == 0
    assert sorted(gh.open) == [4, 5] and gh.open[4]["baseRefName"] == "main"


def _checked_c(tmp_path, monkeypatch, gh, repo):
    """C with main merged in, pushed, and a passing check of its tip; the checkout back on main."""
    monkeypatch.setattr(M, "_gh_cli", lambda _repo, *args: gh(*args))
    M.sweep(repo, gh=gh, log=lambda *_: None)
    _git(repo, "checkout", "-q", "C")
    _git(repo, "merge", "-q", "--no-ff", "-m", "main into C", "main")
    _git(repo, "push", "-q", "origin", "C")
    gh.open[3]["headRefOid"] = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    assert research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool",
                          "mytools.decl_check:T", "--cwd", "source", "--", *CHECKED]) == 0
    _git(repo, "checkout", "-q", "main")


def test_a_merge_push_of_main_off_drive_is_refused_and_pushes_nothing(tmp_path, monkeypatch, capsys):
    """GitHub's rulesets let only drive's App move `main`; a pure move lands from the control pod (`drive.sh merge`)."""
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    monkeypatch.delenv("RESEARCH_DRIVE")
    before = _git(repo, "ls-remote", "origin", "refs/heads/main")
    assert M.main(["C", "--push", "-m", "Merge train TC (c)"]) == 1
    assert "/data/infra/drive.sh merge" in capsys.readouterr().err
    assert _git(repo, "ls-remote", "origin", "refs/heads/main") == before


def test_a_merge_with_push_lands_on_the_remote_then_sweeps(tmp_path, monkeypatch, capsys):
    """Trains landed through `research merge` and a push by hand, so the sweep ran before the push and closed nothing: with --push
    the merge is on the remote before the sweep reads it, and what the tool runs after a landing starts on the commit pushed."""
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    landed: list[tuple] = []
    monkeypatch.setattr(M, "after_land", lambda *a, **_k: landed.append(a))
    capsys.readouterr()
    assert M.main(["C", "--push", "-m", "Merge train TC (c)"]) == 0
    out = capsys.readouterr().out
    assert "pushed main to origin" in out and "push it, then" not in out
    assert _git(repo, "rev-parse", "origin/main") == _git(repo, "rev-parse", "main")
    assert landed == [(repo, "check", _git(repo, "rev-parse", "origin/main"))]
    assert sorted(gh.open) == [4, 5] and gh.open[4]["baseRefName"] == "main" and "C" not in _heads(repo)


def test_after_a_landing_the_declarations_command_starts_detached_on_the_commit(tmp_path):
    """`Tool.after_land`, read from the target's checkout: started in a session of its own (it outlives the merge) with the landed
    commit appended; a declaration without one, or a tool the checkout doesn't declare, starts nothing."""
    (tmp_path / "mytools").mkdir()
    (tmp_path / "mytools" / "decl.py").write_text("from research.store.tool import Tool\n"
                                                  "T = Tool(name='check', after_land=['python', 'post.py', '--x'])\n"
                                                  "U = Tool(name='other')\n")
    started: list[tuple] = []
    spawn = lambda argv, **kw: started.append((argv, kw["cwd"], kw["start_new_session"], kw["stdin"]))  # noqa: E731
    lines: list[str] = []
    assert M.after_land(tmp_path, "mytools.decl:T", "abc123def4567", log=lines.append, spawn=spawn) == ["python", "post.py", "--x",
                                                                                                         "abc123def4567"]
    assert started == [(["python", "post.py", "--x", "abc123def4567"], tmp_path, True, subprocess.DEVNULL)]
    assert "started `python post.py --x abc123def4567`" in lines[0]
    assert M.after_land(tmp_path, "mytools.decl:U", "abc", spawn=spawn) is None
    assert M.after_land(tmp_path, "nothere.decl:T", "abc", spawn=spawn) is None and len(started) == 1
    assert M.after_land(tmp_path, "mytools.decl:T", "abc", spawn=spawn, attempt="r20261019-010000-0001") == [
        "python", "post.py", "--x", "abc", "--attempt", "r20261019-010000-0001"], "drive's landing names the check it landed on"


def test_the_after_landing_command_runs_in_a_worktree_of_the_landed_commit_that_goes_when_it_ends(tmp_path):
    """Drive's queue clone is reset and cleaned every tick, so a `post_train.py` started there would lose the `.venv` `uv run`
    made there; the command gets a tree of its own, which the caller's checkout moving can't touch."""
    repo, rec = tmp_path / "repo", tmp_path / "rec"
    (repo / "mytools").mkdir(parents=True)
    (repo / "mytools" / "decl.py").write_text(
        "from research.store.tool import Tool\n"
        f"T = Tool(name='check', after_land=['sh', '-c', 'pwd > {rec}; git rev-parse HEAD >> {rec}; echo $0 >> {rec}'])\n")
    for args in (("init", "-q", "-b", "main"), ("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")):
        subprocess.run(["git", *args], cwd=repo, check=True)
    landed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "later.txt").write_text("the checkout moves on\n")
    procs: list[subprocess.Popen] = []
    M.after_land(repo, "mytools.decl:T", landed, log=lambda *_: None, spawn=lambda argv, **kw: procs.append(subprocess.Popen(argv, **kw)))
    assert procs[0].wait() == 0
    where, head, arg = rec.read_text().split()
    assert Path(where) != repo and Path(where).name.startswith(f"research-after-land-{landed[:12]}-")
    assert (head, arg) == (landed, landed)
    assert not Path(where).exists()
    assert subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.count("worktree ") == 1


def test_a_refused_push_sweeps_nothing_and_fails(tmp_path, monkeypatch, capsys):
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    monkeypatch.setattr(M, "after_land", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("nothing landed on the remote")))
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    (other / "x.txt").write_text("x\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "someone else landed first")
    _git(other, "push", "-q", "origin", "main")
    assert M.main(["C", "--push"]) == 1
    assert "push of main to origin refused, nothing swept" in capsys.readouterr().err
    assert sorted(gh.open) == [3, 4, 5] and "C" in _heads(repo)


def test_a_merge_without_the_remote_skips_the_sweep(tmp_path, monkeypatch, capsys):
    repo, _gh, _tips, _merge = _world(tmp_path)
    _git(repo, "remote", "remove", "origin")
    monkeypatch.setattr(M, "_gh_cli", lambda *_: (_ for _ in ()).throw(AssertionError("gh must not run")))
    monkeypatch.chdir(repo)
    M._sweep_after(repo, "main", "origin")
    assert "no remote 'origin'; pull request sweep skipped" in capsys.readouterr().out


# ---- the steps after a landing follow origin's main, not the push

def _steps(monkeypatch) -> list[tuple]:
    """Every after-landing step `caught_up` runs, in order: ("alert", old, new), ("mirror", sha), ("after_land", sha),
    ("report", start, tip), ("landed", start, tip)."""
    import research.spec_alert as S

    ran: list[tuple] = []
    monkeypatch.setattr(S, "alert", lambda _repo, old, new, **_k: ran.append(("alert", old, new)) or True)
    monkeypatch.setattr(M, "push_mirror", lambda _repo, sha, _name: ran.append(("mirror", sha)) or True)
    monkeypatch.setattr(M, "after_land", lambda _repo, _tool, sha, **_k: ran.append(("after_land", sha)))
    monkeypatch.setattr(M, "report_landings", lambda _repo, start, tip, **_k: ran.append(("report", start, tip)) or True)
    monkeypatch.setattr(M, "label_landed", lambda _repo, start, tip, **_k: ran.append(("landed", start, tip)) or True)
    monkeypatch.setenv("RESEARCH_MERGE_MIRROR", "vy-nebius-1")
    return ran


def _marker(repo: Path) -> str:
    out = _git(repo, "ls-remote", "origin", M.AFTER_LAND_REF)
    return out.split()[0] if out else ""


def test_a_landing_pushed_another_way_gets_its_after_landing_steps_on_the_next_merge_run_once(tmp_path, monkeypatch, capsys):
    """A landing that reaches main without `research merge --push` would skip the spec DM, node 1's mirror and
    `post_train.py`.  Each `research merge` run catches them up from the marker, once, whoever runs it; a dry run runs
    none, and a remote with no marker yet only gets one."""
    repo, gh, _tips, merge = _world(tmp_path)
    monkeypatch.setattr(M, "_gh_cli", lambda _repo, *args: gh(*args))
    ran = _steps(monkeypatch)
    monkeypatch.chdir(repo)
    assert M.main(["--sweep"]) == 0
    assert ran == [] and _marker(repo) == merge and f"had no {M.AFTER_LAND_REF}; set it to main ({merge[:12]})" in capsys.readouterr().out
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    for name in ("x", "y"):
        (other / f"{name}.txt").write_text(name + "\n")
        _git(other, "add", "-A")
        _git(other, "commit", "-q", "-m", f"{name}, pushed by hand")
    _git(other, "push", "-q", "origin", "main")
    tip = _git(other, "rev-parse", "HEAD")
    assert M.main(["--sweep", "--dry-run"]) == 0 and ran == [] and _marker(repo) == merge
    assert M.main(["--sweep"]) == 0
    assert ran == [("alert", merge, tip), ("mirror", tip), ("after_land", tip), ("report", merge, tip), ("landed", merge, tip)]
    assert _marker(repo) == tip
    assert f"last ran for {merge[:12]}; catching up 2 commit(s) of main to {tip[:12]}" in capsys.readouterr().out
    assert M.main(["--sweep"]) == 0 and len(ran) == 5, "caught up once"


def test_the_after_landing_command_gets_the_attempt_drive_landed_the_tip_on(tmp_path, monkeypatch):
    """`post_train.py` would refuse every landing drive makes with deferred steps, since none has a passed check in the store
    yet; drive's `runs` names the attempt of the tip it pushed, and `after_land` gets it.  A tip it doesn't name gets none."""
    origin, repo = tmp_path / "origin.git", tmp_path / "repo"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(repo))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "a")
    _git(repo, "push", "-q", "origin", "HEAD:main", f"HEAD:{M.AFTER_LAND_REF}", f"HEAD:{M.SPEC_ALERTED_REF}")
    _steps(monkeypatch)
    got: list[tuple] = []
    monkeypatch.setattr(M, "after_land", lambda _repo, _tool, sha, **k: got.append((sha, k.get("attempt"))))
    for rid in ("r20261019-010000-0001", None):
        _git(repo, "commit", "-q", "--allow-empty", "-m", "Merge #1 into next")
        tip = _git(repo, "rev-parse", "HEAD")
        _git(repo, "push", "-q", "origin", "HEAD:main")
        assert M.caught_up(repo, "origin", "check", log=lambda *_: None, runs={"0" * 40: "r20261019-000000-0000"} |
                           ({tip: rid} if rid else {})) == tip
        assert got[-1] == (tip, rid or ""), rid


def test_a_merge_with_push_runs_the_steps_from_the_marker_over_what_it_and_the_landings_it_missed_brought(tmp_path, monkeypatch,
                                                                                                        capsys):
    repo, gh, _tips, merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    before = _git(repo, "rev-parse", f"{merge}^1")
    _git(repo, "push", "-q", "origin", f"{before}:{M.AFTER_LAND_REF}")       # the steps last ran before B's train landed
    ran = _steps(monkeypatch)
    assert M.main(["C", "--push"]) == 0, capsys.readouterr().err
    tip = _git(repo, "rev-parse", "origin/main")
    assert ran == [("alert", before, merge), ("mirror", merge), ("after_land", merge), ("report", before, merge),  # B's train
                   ("landed", before, merge),
                   ("alert", merge, tip), ("mirror", tip), ("after_land", tip), ("report", merge, tip),          # the push's: C
                   ("landed", merge, tip)]
    assert _marker(repo) == tip


def _lean_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    """A repository with a Lake package (`pkg/`, its lock and a source): (repo, base, a commit changing the source on it, a
    commit on that changing only a README)."""
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    (repo / "pkg").mkdir()
    (repo / "pkg/lean-audit.json").write_text("{}\n")
    (repo / "pkg/A.lean").write_text("def a := 1\n")
    (repo / "README.md").write_text("r\n")
    shas = []
    for path, text in ((None, None), ("pkg/A.lean", "def a := 2\n"), ("README.md", "r2\n")):
        if path:
            (repo / path).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", path or "base")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return repo, *shas


def test_a_landing_that_touches_a_lean_package_starts_the_lean_cache_publish_and_one_that_does_not_does_not(tmp_path,
                                                                                                         monkeypatch):
    """A Lean cache published only daily leaves a landing's Lean waiting on a stale cache.  A landing whose range
    touches a Lake package's source starts infra's publisher with the landed SHA; one that touches none, or an absent script,
    starts nothing; every case is one line, with the seconds the script took to start."""
    repo, base, lean, docs = _lean_repo(tmp_path)
    script = tmp_path / "lean_publish.sh"
    monkeypatch.setenv(M.LEAN_PUBLISH_ENV, str(script))
    calls, lines = [], []
    run = lambda argv, **_k: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", "")  # noqa: E731

    assert not M.lean_publish(repo, base, docs, log=lines.append, run=run)
    assert calls == [] and lines[-1].endswith("is absent") and str(script) in lines[-1]
    script.write_text("#!/bin/sh\n")
    assert M.lean_publish(repo, base, docs, log=lines.append, run=run, clock=iter([10.0, 11.5]).__next__)
    assert calls == [[str(script), docs]], "a range with a Lean change in it counts, whatever its tip changed"
    assert "started" in lines[-1] and "1.5 s" in lines[-1] and M.LEAN_PUBLISH_LOG in lines[-1]
    assert not M.lean_publish(repo, lean, docs, log=lines.append, run=run)
    assert len(calls) == 1 and lines[-1].endswith("touches no Lean package")
    monkeypatch.setenv(M.LEAN_PUBLISH_ENV, "")
    assert not M.lean_publish(repo, base, lean, log=lines.append, run=run) and len(calls) == 1 and "off" in lines[-1]


def test_a_lean_cache_publish_that_fails_is_logged_and_the_landing_stands(tmp_path, monkeypatch):
    """The publisher's nonzero exit, its timeout or its failure to run is one line naming why and the rerun; `caught_up` still
    runs the steps after it and moves its marker, so a landing never fails on the publish."""
    repo, base, lean, _docs = _lean_repo(tmp_path)
    script = tmp_path / "lean_publish.sh"
    script.write_text("#!/bin/sh\n")
    monkeypatch.setenv(M.LEAN_PUBLISH_ENV, str(script))
    lines: list[str] = []

    def timeout(argv, **k):
        raise subprocess.TimeoutExpired(argv, k["timeout"])

    for run, why in ((lambda argv, **_k: subprocess.CompletedProcess(argv, 3, "", "cpu-4 unreachable\n"),
                      "exited 3: cpu-4 unreachable"),
                     (timeout, f"no exit within {M.LEAN_PUBLISH_TIMEOUT_S} s"),
                     (lambda argv, **_k: (_ for _ in ()).throw(PermissionError("not executable")), "PermissionError")):
        assert not M.lean_publish(repo, base, lean, log=lines.append, run=run, clock=iter([0.0, 2.0]).__next__)
        assert "not started" in lines[-1] and why in lines[-1] and "2.0 s" in lines[-1] and f"rerun: {script} {lean}" in lines[-1]

    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", f"{base}:refs/heads/main", f"{base}:{M.AFTER_LAND_REF}", f"{base}:{M.SPEC_ALERTED_REF}")
    ran = _steps(monkeypatch)
    script.write_text('#!/bin/sh\necho "$1" > "$(dirname "$0")/published"\nexit 3\n')
    script.chmod(0o755)
    _git(repo, "push", "-q", "origin", f"{lean}:refs/heads/main")
    assert M.caught_up(repo, "origin", "check", log=lines.append) == lean
    assert (tmp_path / "published").read_text().strip() == lean and any(f"{script} exited 3" in x for x in lines[-4:])
    assert ran[-2:] == [("report", base, lean), ("landed", base, lean)] and _marker(repo) == lean


def test_a_spec_dm_that_failed_is_sent_again_by_the_next_sweep_which_fails_until_it_goes(tmp_path, monkeypatch, capsys):
    """A DM that fails (on the Slack broker's socket, say) while the after-landing marker moves anyway would need a hand retry.
    The DM's own marker stays put until `alert` returns True, so the next run resends it, and `--sweep` exits 1 while it
    lags main; the mirror and `after_land` still run once."""
    import research.spec_alert as S

    repo, gh, _tips, merge = _world(tmp_path)
    monkeypatch.setattr(M, "_gh_cli", lambda _repo, *args: gh(*args))
    ran = _steps(monkeypatch)
    monkeypatch.chdir(repo)
    assert M.main(["--sweep"]) == 0 and _marker(repo) == merge
    assert _git(repo, "ls-remote", "origin", M.SPEC_ALERTED_REF).split()[0] == merge
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    (other / "x.txt").write_text("x\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "x, pushed by hand")
    _git(other, "push", "-q", "origin", "main")
    tip = _git(other, "rev-parse", "HEAD")
    sent = iter([False, True])
    monkeypatch.setattr(S, "alert", lambda _repo, old, new, **_k: ran.append(("alert", old, new)) or next(sent))
    capsys.readouterr()
    assert M.main(["--sweep"]) == 1
    err, out = capsys.readouterr()[::-1]
    assert ran == [("alert", merge, tip), ("mirror", tip), ("after_land", tip), ("report", merge, tip), ("landed", merge, tip)]
    assert _marker(repo) == tip
    assert f"{M.SPEC_ALERTED_REF} stays at {merge[:12]}" in out and f"{M.SPEC_ALERTED_REF} is {merge[:12]}, behind main" in err
    assert M.main(["--sweep"]) == 0
    assert ran[5:] == [("alert", merge, tip)], "only the DM is retried"
    assert _git(repo, "ls-remote", "origin", M.SPEC_ALERTED_REF).split()[0] == tip
    assert M.main(["--sweep"]) == 0 and len(ran) == 6, "sent once"


def test_a_marker_main_does_not_hold_runs_nothing_and_says_how_to_reset_it(tmp_path, monkeypatch, capsys):
    repo, gh, tips, _merge = _world(tmp_path)
    monkeypatch.setattr(M, "_gh_cli", lambda _repo, *args: gh(*args))
    _git(repo, "push", "-q", "origin", f"{tips['D']}:{M.AFTER_LAND_REF}")
    ran = _steps(monkeypatch)
    monkeypatch.chdir(repo)
    assert M.main(["--sweep"]) == 0 and ran == [] and _marker(repo) == tips["D"]
    assert f"last ran for {tips['D'][:12]}, which isn't in origin's main" in capsys.readouterr().out


# ---- main to node 1's bare repository, once origin has it

def _node1(tmp_path, monkeypatch):
    """vy-nebius-1's bare repository in tmp, pushed to through `git_push_url` with `sh -c` standing in for its ssh."""
    bare = tmp_path / "node1.git"
    _git(tmp_path, "init", "-q", "--bare", str(bare))
    monkeypatch.setenv("RESEARCH_MERGE_MIRROR", "vy-nebius-1")
    asked = []
    monkeypatch.setattr(M, "_mirror_target", lambda name: (asked.append(name), (["sh", "-c"], str(bare)))[1])
    return bare, asked


def _main_of(bare: Path) -> str | None:
    r = subprocess.run(["git", "--git-dir", str(bare), "rev-parse", "-q", "--verify", "refs/heads/main"], capture_output=True,
                       text=True)
    return r.stdout.strip() or None


def test_a_merge_with_push_pushes_main_to_node_1_after_origin(tmp_path, monkeypatch, capsys):
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    bare, asked = _node1(tmp_path, monkeypatch)
    capsys.readouterr()
    assert M.main(["C", "--push"]) == 0
    out, main = capsys.readouterr().out, _git(repo, "rev-parse", "main")
    assert _git(repo, "rev-parse", "origin/main") == main == _main_of(bare) and asked == ["vy-nebius-1"]
    assert out.index("pushed main to origin") < out.index(f"pushed main ({main[:12]}) to vy-nebius-1:{bare}")


def test_a_refused_node_1_push_leaves_the_merge_landed_and_names_the_command_that_reruns_it(tmp_path, monkeypatch, capsys):
    """Node 1's main is somewhere the merge doesn't contain: the plain push is refused, never forced, and the merge stands."""
    repo, gh, tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    bare, _asked = _node1(tmp_path, monkeypatch)
    _git(repo, "push", "-q", "--no-verify", str(bare), f"{tips['D']}:refs/heads/main")
    capsys.readouterr()
    assert M.main(["C", "--push"]) == 0
    err, main = capsys.readouterr().err, _git(repo, "rev-parse", "main")
    assert _git(repo, "rev-parse", "origin/main") == main and _main_of(bare) == tips["D"]
    line, = [ln for ln in err.splitlines() if "not pushed to vy-nebius-1" in ln]
    assert f"main ({main[:12]}) not pushed to vy-nebius-1 (git push exited 1: " in line and "rejected" in line
    assert line.endswith(f"; rerun: cd {repo} && research merge --mirror {main}")
    _git(tmp_path, "--git-dir", str(bare), "update-ref", "-d", "refs/heads/main")
    assert M.main(["--mirror", main]) == 0 and _main_of(bare) == main


def test_a_hand_landed_main_reaches_node_1_past_the_checkouts_push_guard(tmp_path, monkeypatch, capsys):
    """The guard judges pushes to origin's main; node 1's copy follows origin, so a main landed by hand, which has no check
    attempt of its tree, still reaches it."""
    repo, _gh, _tips, _merge = _world(tmp_path)
    bare, _asked = _node1(tmp_path, monkeypatch)
    _git(repo, "config", "core.hooksPath", str(repo / ".git" / "hooks"))
    assert M.install_push_guard(repo) == repo / ".git" / "hooks" / "pre-push"
    (repo / "by-hand.txt").write_text("landed by hand\n")
    _git(repo, "add", "by-hand.txt")
    _git(repo, "commit", "-qm", "landed by hand")
    main = _git(repo, "rev-parse", "HEAD")
    plain = subprocess.run(["git", "push", "-q", str(bare), f"{main}:refs/heads/main"], cwd=repo, capture_output=True, text=True)
    assert plain.returncode != 0 and _main_of(bare) is None, "the guard refuses it as a push to main"
    monkeypatch.chdir(repo)
    assert M.main(["--mirror", main]) == 0 and _main_of(bare) == main


def test_a_machine_that_cannot_be_resolved_fails_only_the_node_1_push(tmp_path, monkeypatch, capsys):
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    monkeypatch.setenv("RESEARCH_MERGE_MIRROR", "vy-nebius-1")
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "no-machines.toml"))
    assert M.main(["C", "--push"]) == 0
    err = capsys.readouterr().err
    assert "not pushed to vy-nebius-1 (Refused: machine 'vy-nebius-1' is not in" in err and "research merge --mirror" in err
    assert _git(repo, "rev-parse", "origin/main") == _git(repo, "rev-parse", "main")


def test_a_node_1_push_with_no_answer_is_cut_off_and_the_merge_stands(tmp_path, monkeypatch, capsys):
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    bare, _asked = _node1(tmp_path, monkeypatch)
    bounds = []

    def hangs(argv, **kw):
        bounds.append(kw["timeout"])
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(M, "push_mirror", functools.partial(M.push_mirror, run=hangs))
    assert M.main(["C", "--push"]) == 0
    assert bounds == [M.MIRROR_TIMEOUT_S] and _main_of(bare) is None
    assert f"not pushed to vy-nebius-1 (no answer within {M.MIRROR_TIMEOUT_S} s); rerun:" in capsys.readouterr().err


def test_nothing_goes_to_node_1_when_origin_refuses_main(tmp_path, monkeypatch, capsys):
    repo, gh, _tips, _merge = _world(tmp_path)
    _checked_c(tmp_path, monkeypatch, gh, repo)
    bare, asked = _node1(tmp_path, monkeypatch)
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    (other / "x.txt").write_text("x\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "someone else landed first")
    _git(other, "push", "-q", "origin", "main")
    assert M.main(["C", "--push"]) == 1
    assert asked == [] and _main_of(bare) is None and "vy-nebius-1" not in capsys.readouterr().err


# ---- every landing is reported in the merge thread

def test_each_landing_is_one_line_naming_what_it_merged_and_its_checked_run(tmp_path):
    """Every landing is reported, and a stacked member can read closed on GitHub, so the lines come from main's first-parent log: a train's merges (`Merge #N`, `Merge
    origin/BRANCH`, or a pinned commit) through ``prs``, and the runs its `Checked:` trailer names, the lander's `*` id too."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "root")
    start = _git(repo, "rev-parse", "HEAD")
    heads = {}
    for name in ("feature", "pinned", "seven"):
        _git(repo, "checkout", "-q", "-b", name, start)
        (repo / name).write_text(name)
        _git(repo, "add", name)
        _git(repo, "commit", "-q", "-m", name)
        heads[name] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "train", start)
    _git(repo, "merge", "-q", "--no-ff", "-m", f"Merge #7 ({heads['seven'][:8]})", "seven")
    _git(repo, "merge", "-q", "--no-ff", "-m", f"Merge origin/feature ({heads['feature'][:8]})", "feature")
    _git(repo, "merge", "-q", "--no-ff", "-m", f"Merge {heads['pinned'][:12]} ({heads['pinned'][:8]})", "pinned")
    train = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", f"Merge {train[:12]} ({train[:8]})\n\nChecked: r20261010-104434-7273 (full "
         "check: passed), r20261010-*-9827 (suites)\nSkipped-check: none", "train")
    landing = _git(repo, "rev-parse", "HEAD")
    (repo / "hand").write_text("h")
    _git(repo, "add", "hand")
    _git(repo, "commit", "-q", "-m", "a fix pushed by hand")
    hand = _git(repo, "rev-parse", "HEAD")
    prs = {"feature": 9, heads["pinned"]: 8}
    assert M.landing_lines(repo, start, hand, prs) == [
        f"landed on main: {landing[:12]} #7 + #9 + #8; checked by r20261010-104434-7273, r20261010-*-9827",
        f"landed on main: {hand[:12]} “a fix pushed by hand”; checked by no run (no Check: trailer)"]
    assert M.landing_lines(repo, start, landing)[0].startswith(f"landed on main: {landing[:12]} #7 + feature + {heads['pinned'][:12]};")
    assert M.landing_lines(repo, hand, hand) == []


def test_each_run_a_landing_names_gets_one_landed_label_on_the_remote_and_a_rerun_writes_none(tmp_path, monkeypatch):
    """Which check landed what is a `landed` label on the run (main's commit, at the commit's time),
    so the store answers it without reading git; the lander's `*` id resolves to the one run with its suffix, and an id that
    names no run is said, never guessed."""
    from research.store.local import LocalStore

    vault = tmp_path / "vault"
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    for rid in ("r20261010-104434-7273", "r20261010-110000-9827", "r20261011-010203-abcd"):
        (vault / "attempts").mkdir(parents=True, exist_ok=True)
        (vault / "attempts" / f"{rid}.json").write_text("{}")
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "root")
    start = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "Merge train\n\nChecked: r20261010-104434-7273 (full check: passed), "
         "r20261010-*-9827 (suites)\nSkipped-check: none")
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "Merge B\n\nCheck: r20261011-010203-abcd")
    second = _git(repo, "rev-parse", "HEAD")
    assert [(r, c) for r, c, _ts in M.landed_runs(repo, start, second)] == [
        ("r20261010-104434-7273", first), ("r20261010-*-9827", first), ("r20261011-010203-abcd", second)]
    _git(repo, "commit", "-q", "--allow-empty", "-m", "queue: fast-forwarded, no trailer")
    third = _git(repo, "rev-parse", "HEAD")
    assert [(r, c) for r, c, _ts in M.landed_runs(repo, second, third, {third: "r20261011-020000-beef"})] == [
        ("r20261011-020000-beef", third)], "a queue commit's run from `runs`, as landing_lines reads it"
    st, said = LocalStore(), []
    assert M.label_landed(repo, start, second, store=st, log=said.append), said
    for rid, commit in (("r20261010-104434-7273", first), ("r20261010-110000-9827", first), ("r20261011-010203-abcd", second)):
        lab, = st.labels(rid, "landed")
        assert (lab.value, lab.by) == (commit, M.LANDED_BY) and lab.ts >= f"2026-10-1{rid[8]}T"
        assert list((vault / "labels").rglob("*.json")), "pushed"
    pushed = sorted((vault / "labels").rglob("*.json"))
    assert len(pushed) == 3
    assert M.label_landed(repo, "", second, store=st, log=said.append) and sorted((vault / "labels").rglob("*.json")) == pushed
    assert all(len(st.labels(r, "landed")) == 1 for r in ("r20261010-104434-7273", "r20261011-010203-abcd")), "a rerun writes none"
    monkeypatch.chdir(repo)
    assert M.main(["--label-landed", start]) == 0 and sorted((vault / "labels").rglob("*.json")) == pushed, "the backfill"
    (vault / "attempts" / "r20261010-120000-9827.json").write_text("{}")
    st2 = LocalStore(tmp_path / "fresh")
    assert not M.label_landed(repo, start, first, store=st2, log=said.append)
    assert "no `landed` label on r20261010-*-9827: the id names no single run on the remote" in said[-2]
    assert st2.labels("r20261010-104434-7273", "landed")[0].value == first
    monkeypatch.setattr(M, "landed_runs", lambda *_a: [("r20261011-010203-abcd", second, "2026-09-28T23:00:00Z")])
    st3 = LocalStore(tmp_path / "third")
    assert M.label_landed(repo, start, second, store=st3, log=said.append)
    assert st3.labels("r20261011-010203-abcd", "landed")[0].ts == "2026-10-11T01:02:03Z", "the lander's fixed date: the run's start"


def test_checked_heads_are_the_train_heads_their_own_check_ran_on_less_a_train_a_later_revert_names(tmp_path, monkeypatch,
                                                                                                    capsys):
    """`label_landed` labels every commit of a train on its one check run, but the run checked only its own source commit, so
    `checked_heads` keeps the label whose value is that commit: each train's head. A head whose train holds a commit a later
    `git revert` names is left out, and so are labels on commits off the line or on runs the index lacks."""
    from research.store import bucket_index as BI
    from research.store.local import LocalStore

    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    sha = {}
    for name in ("root", "a1", "a2", "b1", "c1", "c2"):
        _git(repo, "commit", "-q", "--allow-empty", "-m", f"Queue: {name}")
        sha[name] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", f"Revert c1\n\nThis reverts commit {sha['c1']}, reversing\nchanges.")
    sha["rev"] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "Queue: d1")
    sha["d1"] = _git(repo, "rev-parse", "HEAD")
    trains = {"r20261020-010000-aaaa": ("a2", ["a1", "a2"]), "r20261020-020000-bbbb": ("b1", ["b1"]),
              "r20261020-030000-cccc": ("c2", ["c1", "c2"]), "r20261020-040000-dddd": ("d1", ["rev", "d1"])}
    store = tmp_path / "store"
    monkeypatch.setenv("RESEARCH_STORE", str(store))
    monkeypatch.delenv(BI.READER_ENV, raising=False)
    idx = BI.SqliteIndex(BI.default_path(LocalStore().root))
    rows = [BI.Row(rid, "run", "check", None, {"id": rid, "source": {"commit": sha[head]}}) for rid, (head, _) in trains.items()]
    labels = [(rid, sha[c]) for rid, (_, members) in trains.items() for c in members]
    labels += [("r20261020-050000-eeee", sha["b1"]), ("r20261020-020000-bbbb", "f" * 40)]
    rows += [BI.Row(f"label:{i}", "label", "landed", None, {"target": rid, "key": "landed", "value": c, "by": M.LANDED_BY})
             for i, (rid, c) in enumerate(labels)]
    idx.add(rows)
    assert M.checked_heads(repo, "main", idx) == [(sha["a2"], "r20261020-010000-aaaa"), (sha["b1"], "r20261020-020000-bbbb"),
                                                 (sha["d1"], "r20261020-040000-dddd")]
    assert M.checked_heads(repo, sha["c2"], idx)[-1] == (sha["c2"], "r20261020-030000-cccc"), "before its revert, c2 counts"
    idx.close()
    monkeypatch.chdir(repo)
    assert research_main(["queue", "checked-heads", "--tip", "main"]) == 0
    assert capsys.readouterr().out.split() == [sha["a2"], sha["b1"], sha["d1"]]
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "empty"))
    assert research_main(["queue", "checked-heads"]) == 1
    assert "no store index" in capsys.readouterr().err


def test_the_landings_go_to_the_merge_thread_quietly_and_a_failed_post_leaves_them_standing(tmp_path, capsys):
    from research.monitors import MERGE_THREAD
    from research.slack import is_quiet

    repo, gh, _tips, merge = _world(tmp_path)
    start = _git(repo, "rev-parse", f"{merge}^1")
    calls = []

    class Api:
        def call(self, method, **kw):
            calls.append((method, kw))

    reg = {"channels": {"agent-coordination": {"id": "C1"}}}
    assert M.report_landings(repo, start, merge, api=Api(), reg=reg, gh=gh, log=lambda _: None)
    (method, kw), = calls
    assert method == "chat.postMessage" and kw["channel"] == "C1" and kw["thread_ts"] == MERGE_THREAD
    assert kw["text"].startswith(f"landed on main: {merge[:12]} “Merge train T (B)”; checked by no run") and is_quiet(kw["text"])

    class Down:
        def call(self, method, **kw):
            raise OSError("broker socket closed")

    assert not M.report_landings(repo, start, merge, api=Down(), reg=reg, gh=gh)
    assert f"the landings {start[:12]}..{merge[:12]} weren't posted to the merge thread: OSError" in capsys.readouterr().err
    assert M.report_landings(repo, merge, merge, api=Down(), reg=reg, gh=gh), "nothing landed, nothing to post"


def test_the_sweep_deletes_the_head_branch_of_a_pr_github_marked_merged_itself(tmp_path):
    """Drive lands a PR's commits; GitHub then marks the PR merged, so it was never open for the sweep, and its branch would
    stay.  The merged PRs are the most recently updated ones (`MERGED_QUERY`; the fake refuses `gh pr list --state merged`):
    the newest 200 by number would leave out PRs that had waited in the queue, and their branches would stay."""
    repo, gh, tips, _merge = _world(tmp_path)
    del gh.open[1], gh.open[2]
    gh.merged = [{"number": 1, "headRefName": "A", "headRefOid": tips["A"]}, {"number": 2, "headRefName": "B", "headRefOid": tips["B"]},
                 {"number": 9, "headRefName": "D", "headRefOid": tips["D"]}]                       # D isn't on main: kept
    M.sweep(repo, gh=gh, log=[].append)
    assert _heads(repo) == {"main", "C", "D", "E"}
    log = []
    M.sweep(repo, gh=gh, log=log.append)                                  # A and B are gone: nothing to delete, nothing kept
    assert not [x for x in log if "branch A" in x or "branch B" in x or "kept" in x], log


def test_main_line_is_the_checked_heads_newest_first(tmp_path, monkeypatch):
    """`merge.main_line` is `checked_heads`' commits reversed, so position 0 is main's head: the verifier's `--main`
    file lists the checked train heads newest first, and a pin's index is the checked trains it is behind."""
    heads = [("c0", "run-0"), ("c1", "run-1"), ("c2", "run-2")]           # checked_heads is oldest first
    monkeypatch.setattr(M, "checked_heads", lambda repo, tip, idx: heads)
    assert M.main_line(tmp_path, "origin/main", object()) == ["c2", "c1", "c0"]


def test_the_branch_sweep_deletes_what_is_on_main_and_archives_idle_unmerged_work(tmp_path):
    now = 1_800_000_000                                                   # the sweep's clock, injected

    repo, gh, tips, _merge = _world(tmp_path)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for name, base, days in (("old-landed", "main", 2), ("new-landed", "main", 0), ("old-work", "C", 5), ("new-work", "C", 1),
                             ("cursor/verity-ci-3d4e", "C", 9)):
        _git(repo, "checkout", "-q", "-b", name, base)
        (repo / f"{name.replace('/', '_')}.txt").write_text(name)
        _git(repo, "add", "-A")
        when = f"@{now - days * 86400} +0000"
        subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", name], cwd=repo, check=True,
                       env={**env, "GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when})
        _git(repo, "push", "-q", "origin", name)
    _git(repo, "checkout", "-q", "main")
    for name in ("old-landed", "new-landed"):
        _git(repo, "merge", "-q", "--no-ff", "-m", f"land {name}", name)
    _git(repo, "push", "-q", "origin", "main")
    del gh.open[1], gh.open[2]                                            # A and B landed; C, D and E stay open
    for name in ("A", "B"):                                               # landed with the train, and fresh at ``now``
        _git(repo, "push", "-q", "origin", f":refs/heads/{name}")
    log = []
    assert M.sweep_branches(repo, now=now, gh=gh, log=log.append, limit=0) == (0, 0), "at its limit, it pushes nothing more"
    assert "delete 0 of 1 branch(es)" in log[0] and "archive 0 of 1 unmerged" in log[0] and len(log) == 1, log
    # old-landed deleted, by name; old-work, the head of a PR drive closed as stale, kept
    log = []
    assert M.sweep_branches(repo, now=now, gh=gh, log=log.append, keep=["old-work"]) == (1, 0)
    assert "old-work" in _heads(repo) and "old-landed" not in _heads(repo)
    assert "research merge: delete: old-landed" in log, log
    assert M.sweep_branches(repo, now=now, gh=gh, log=[].append) == (0, 1)  # without it, old-work archived
    assert _heads(repo) == {"main", "C", "D", "E", "new-landed", "new-work", "cursor/verity-ci-3d4e"}
    assert "refs/archive/heads/old-work" in _git(repo, "ls-remote", "origin", "refs/archive/*")
