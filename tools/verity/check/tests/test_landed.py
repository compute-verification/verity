"""`landed.py`, the landing hook: post_train.py, then ci.toml's after-landing jobs on main, a failure posted for the lander."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_landed", ROOT / "tools" / "verity" / "check" / "landed.py")
L = importlib.util.module_from_spec(spec)
sys.modules["repo_landed"] = L
spec.loader.exec_module(L)


def test_the_twin_that_ran_decides_and_twins_that_never_started_are_none():
    done = lambda rc: {"state": "done" if rc == 0 else "failed", "rc": rc}  # noqa: E731
    assert L.outcome({"a": done(75), "b": done(0)}, ["a", "b"]) == ("passed", "b")
    assert L.outcome({"a": done(1), "b": done(75)}, ["a", "b"]) == ("failed", "a")
    assert L.outcome({"a": done(75), "b": done(75)}, ["a", "b"]) == ("none", "a")
    assert L.outcome({"a": {"state": "running"}, "b": done(75)}, ["a", "b"])[0] == "running"
    assert L.outcome({"b": done(75)}, ["a", "b"])[0] == "running", "a run research status can't read yet may still start"


def _hook(monkeypatch, train_out: str, train_rc: int, rounds: list[dict]):
    started, posts, slept = [], [], []
    monkeypatch.setattr(L.subprocess, "Popen", lambda argv, **k: started.append(argv))

    def run(argv, **k):
        if argv[:2] == ["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh")]:
            assert argv[2:] == ["f" * 12, "auto", "--after-landing"]
            return subprocess.CompletedProcess(argv, train_rc, train_out, "")
        return subprocess.CompletedProcess(argv, 0, "Merge #1\n", "")

    monkeypatch.setattr(L.subprocess, "run", run)
    monkeypatch.setattr(L, "states", lambda ids: rounds.pop(0) if len(rounds) > 1 else rounds[0])
    monkeypatch.setattr(L, "post", posts.append)
    monkeypatch.setattr(L, "SLEEP", slept.append)
    return started, posts, slept


def test_a_landing_starts_post_train_and_its_after_landing_check_and_a_failure_is_posted(monkeypatch):
    running, failed = {"r20261012-040000-aaaa": {"state": "running"}}, {"r20261012-040000-aaaa": {"state": "failed", "rc": 1}}
    started, posts, slept = _hook(monkeypatch, "train: ...\nRUN r20261012-040000-aaaa\n", 0, [running, failed])
    assert L.main(["f" * 12]) == 1
    assert started and started[0][-2:] == [str(ROOT / "tools" / "verity" / "check" / "post_train.py"), "f" * 12]
    assert slept == [L.POLL_S] and len(posts) == 1
    assert "FAILED on r20261012-040000-aaaa" in posts[0] and "circuit-check" in posts[0] and "lean-suites" in posts[0]


def test_a_pass_posts_nothing_and_a_launch_that_registered_nothing_leaves_the_post_to_train_sh(monkeypatch):
    _, posts, _ = _hook(monkeypatch, "RUN r20261012-040000-aaaa r20261012-040000-bbbb\n", 0, [{"r20261012-040000-aaaa": {"state": "failed", "rc": 75}, "r20261012-040000-bbbb": {"state": "done", "rc": 0}}])
    assert L.main(["f" * 12, "--no-post-train"]) == 0 and posts == []
    _, posts, _ = _hook(monkeypatch, "train: no run launched\n", 1, [{}])
    assert L.main(["f" * 12, "--no-post-train"]) == 1 and posts == []
    _, posts, _ = _hook(monkeypatch, "RUN r20261012-040000-aaaa\n", 0, [{"r20261012-040000-aaaa": {"state": "failed", "rc": 75}}])
    assert L.main(["f" * 12, "--no-post-train"]) == 1 and "never took a slot" in posts[0]
