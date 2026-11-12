"""`lean_prebuild.py`: the change it reads, the tree it ships, and the run it launches, which a later launch must find as a mark
audit by a member's head."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("lean_prebuild_under_test", ROOT / "tools" / "verity" / "check" / "lean_prebuild.py")
LP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LP)
COMMON = LP.COMMON


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _commit(repo: Path, path: str, text: str, message: str = "") -> str:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", message or f"{path}: {text.strip()}")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    """`main` at two commits, with `origin/main` at its tip, and two branches off its first: `a` and `b` change files of their
    own, and `clash` changes the file `a` does."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    first = _commit(r, "Base.lean", "def base := 0\n")
    tip = _commit(r, "Main.lean", "def main := 1\n")
    _git(r, "update-ref", "refs/remotes/origin/main", tip)
    made = {"first": first, "main": tip}
    for name, path, text in (("a", "A.lean", "def a := 1\n"), ("b", "B.lean", "def b := 2\n"), ("clash", "A.lean", "def a := 3\n")):
        _git(r, "checkout", "-q", "-b", name, first)
        made[name] = _commit(r, path, text)
    _git(r, "checkout", "-q", "main")
    return r, made


def _launched(repo: Path, argv: list[str], urls: Path | None = None) -> tuple[int, list[list[str]], list[dict]]:
    """`main(argv)` in ``repo`` with a launcher that only records: its exit, each `research` call, and what each shipped tree
    held (its head, its files, its parents)."""
    calls, trees = [], []

    def run(args: list[str], cwd: Path) -> int:
        calls.append(args)
        trees.append({"head": _git(cwd, "rev-parse", "HEAD"), "files": sorted(p.name for p in cwd.glob("*.lean")),
                      "parents": _git(cwd, "rev-list", "--parents", "-n", "1", "HEAD").split()[1:]})
        return 0

    rc = LP.main(argv, repo=repo, run=run, run_id=lambda: "r20260101-000000-abcd", signed=lambda tree, d: urls)
    return rc, calls, trees


def _store(*commands: list[str]) -> sqlite3.Connection:
    """The store's index of attempts, as far as a launch's lookups read it: one row a command, the newest first."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE attempts (id TEXT, tool TEXT, state TEXT, start_utc TEXT, manifest_json TEXT)")
    for i, argv in enumerate(commands):
        db.execute("INSERT INTO attempts VALUES (?, ?, 'done', ?, ?)",
                   (f"run-{i}", argv[0], f"2026-01-01T00:00:{59 - i:02d}Z", json.dumps({"argv": argv, "params": {"argv": argv}})))
    return db


def _found(db: sqlite3.Connection, runs: str, heads: list[str], limit: int = 12) -> list[list[str]]:
    """The runs each of a launch's lookups of kind ``runs`` takes (`common.lookups`), as `lean_audit.shared_bundles` asks and
    cuts them, every run counted as one with a bundle."""
    ended = "state IN ('done', 'failed', 'cancelled') ORDER BY start_utc DESC LIMIT ?"
    return [[rid for (rid,) in db.execute(f"SELECT id FROM attempts WHERE {which} AND {ended}", args)][:most]
            for which, args, most in COMMON.lookups(runs, limit, heads)]


def test_the_next_batch_is_mains_tip_with_each_head_merged_in_order_read_against_main(repo):
    r, c = repo
    rc, calls, trees = _launched(r, ["origin/main", "a", "b", "--on", "vy-node", "--mem-gb", "200", "--as", "ci"])
    assert rc == 0 and len(calls) == 1
    assert trees[0]["files"] == ["A.lean", "B.lean", "Base.lean", "Main.lean"] and trees[0]["parents"][1] == c["b"]
    args = calls[0]
    work = args[args.index("--") + 1:]
    assert work == LP.workload(c["main"], [c["a"], c["b"]]) and work[3:] == [LP.NAME, c["main"], c["a"], c["b"]]
    assert args[:args.index("--source")] == ["run", "--id", "r20260101-000000-abcd", "--on", "vy-node", "--pool", "research", "--as",
                                             "ci", "--project", "verity", "--mem-gb", "200", "--timeout", f"{LP.TIMEOUT_S:g}"]
    assert "--send" not in args and args[args.index("--cwd") + 1] == "source"
    assert _git(r, "worktree", "list").count("\n") == 0, "the shipped worktree is removed once the run is launched"


def test_a_later_launch_finds_the_run_as_a_mark_audit_by_a_members_head_and_as_no_check(repo):
    r, c = repo
    _, calls, _ = _launched(r, ["origin/main", "a", "--on", "vy-node", "--mem-gb", "200", "--as", "ci"])
    ours = ["env", "CHECK_SLOTS=/x/research/slots", "python3", "tools/verity/check/slot.py", "--", *calls[0][calls[0].index("--") + 1:]]
    mark = ["sh", "-c", f"python3 quick_tier.py --head {'9' * 40} --lean-audit 'python3 {LP.WARM}'"]
    db = _store(*([mark] * 20), ours)
    by_head, newest = _found(db, COMMON.MARK_AUDITS, [c["a"]])
    assert by_head == ["run-20"] and "run-20" not in newest, "twenty newer marks push it out of the newest, and its head still finds it"
    assert _found(db, COMMON.MARK_AUDITS, [c["b"]])[0] == []
    assert _found(db, COMMON.RECORDED_CHECKS, []) == [[]], "a prebuild is no recorded check: no launch takes a check's verdicts from it"
    assert f'--out "$RESEARCH_RUN_DIR/{COMMON.MARK_AUDIT_OUT}"' in LP.SCRIPT


def test_the_workload_audits_from_the_base_and_takes_the_urls_only_when_they_were_sent(tmp_path):
    bin_, run_dir = tmp_path / "bin", tmp_path / "run"
    bin_.mkdir()
    (run_dir / "inputs").mkdir(parents=True)
    (bin_ / "uv").write_text('#!/bin/sh\nprintf "%s\\n" "${CHECK_LEAN_DEPS_URLS-unset}" "$@"\n')
    (bin_ / "uv").chmod(0o755)
    kind, script, *rest = LP.workload("b" * 40, ["h" * 40])
    assert kind == "bash" and script == "-lc" and rest[0] == LP.SCRIPT

    def said() -> list[str]:
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "RESEARCH_RUN_DIR": str(run_dir), "CHECK_LEAN_DEPS_URLS": "/stale"}
        return subprocess.run(["bash", "-c", *rest], env=env, check=True, capture_output=True, text=True).stdout.splitlines()

    audit = ["run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", LP.WARM, "--base", "b" * 40, "--out",
             f"{run_dir}/{COMMON.MARK_AUDIT_OUT}"]
    assert said() == ["/stale", *audit], "with none sent, the run's own setting stands"
    (run_dir / LP.URLS).write_text("{}")
    assert said() == [f"{run_dir}/{LP.URLS}", *audit]


def test_the_urls_the_launcher_signed_are_sent_with_the_run(repo, tmp_path):
    r, _ = repo
    urls = tmp_path / "lean-deps-urls.json"
    urls.write_text("{}")
    _, calls, _ = _launched(r, ["a", "--on", "vy-node", "--mem-gb", "64", "--as", "ci"], urls)
    assert calls[0][calls[0].index("--send") + 1] == str(urls) and LP.URLS.endswith(urls.name)


def test_one_head_is_read_against_where_its_branch_left_main(repo):
    r, c = repo
    rc, calls, trees = _launched(r, ["a", "--on", "vy-node", "--mem-gb", "64", "--as", "ci"])
    assert rc == 0 and trees[0]["head"] == c["a"] and calls[0][-2:] == [c["first"], c["a"]]


def test_a_landing_is_read_against_the_tip_before_it_and_names_the_prs_the_queue_put_in_it(repo):
    r, c = repo
    head = "c" * 40
    landed = _commit(r, "Landed.lean", "def landed := 1\n", f"{COMMON.QUEUE_SUBJECT}7 a change\n\nHead: {head}\n")
    _git(r, "update-ref", "refs/remotes/origin/main", landed)
    rc, calls, _ = _launched(r, ["origin/main", "--base", c["main"], "--on", "vy-node", "--mem-gb", "64", "--as", "ci"])
    assert rc == 0 and calls[0][-3:] == [c["main"], landed, head]


def test_whole_reads_against_the_first_commit_so_every_package_is_reached(repo):
    r, c = repo
    rc, calls, _ = _launched(r, ["origin/main", "--whole", "--on", "vy-node", "--mem-gb", "64", "--as", "ci"])
    assert rc == 0 and calls[0][-2:] == [c["first"], c["main"]]


@pytest.mark.parametrize("argv, why", [
    (["origin/main"], "holds no change from"),
    (["origin/main", "a", "clash"], "doesn't merge onto"),
    (["origin/main", "--base", "a"], "doesn't hold"),
    (["nowhere"], "names no commit"),
])
def test_a_refusal_launches_nothing_and_says_what_to_give(repo, capsys, argv, why):
    r, _ = repo
    rc, calls, _ = _launched(r, [*argv, "--on", "vy-node", "--mem-gb", "64", "--as", "ci"])
    assert rc == 2 and calls == [] and why in capsys.readouterr().err
    assert _git(r, "worktree", "list").count("\n") == 0


def test_a_dry_run_says_the_run_and_launches_nothing(repo, capsys):
    r, c = repo
    rc, calls, _ = _launched(r, ["origin/main", "a", "--on", "vy-node", "--mem-gb", "64", "--as", "ci", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and calls == [] and "research run --id" in out and c["a"] in out


def test_memory_and_the_lane_are_asked_for_never_assumed(repo, monkeypatch):
    r, _ = repo
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    for argv in (["a", "--on", "vy-node", "--as", "ci"], ["a", "--on", "vy-node", "--mem-gb", "64"],
                 ["a", "--mem-gb", "64", "--as", "ci"], ["origin/main", "--whole", "--base", "a", "--on", "vy-node", "--mem-gb", "64", "--as", "ci"]):
        with pytest.raises(SystemExit):
            LP.main(argv, repo=r, run=lambda *a: 0)
