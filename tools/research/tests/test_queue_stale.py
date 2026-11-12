"""The stale rule (`research.stale`) in `research queue status`: a PR that isn't marked ready at its head and has been idle
`stale_days` (drive.toml's, at `main`) is marked stale, by its head's commit date and its labels, the store's labels pulled
in one batch however many PRs are open.  On `test_queue`'s throwaway origin with its fake GitHub; commit dates and the
queue's clock are injected."""
from __future__ import annotations

import dataclasses
import json
import time

import pytest

import research.queue as Q
from research.queue import DRIVE_FLAGS, PR
from research.stale import DAY_S, stale_prs
from test_queue import World, _git

T0 = 1_800_000_000.0                                    # the queue's clock, injected
DAY = int(DAY_S)


def _at(world: World, monkeypatch, files: dict[str, str], when: float, base: str = "origin/main") -> str:
    """`World.change` committed at ``when``."""
    monkeypatch.setenv("GIT_COMMITTER_DATE", f"@{int(when)} +0000")
    try:
        return world.change(files, base)
    finally:
        monkeypatch.delenv("GIT_COMMITTER_DATE")


def _iso(when: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_status_marks_an_unmarked_pr_idle_three_days_stale_by_its_head_and_its_labels(world, monkeypatch):
    gh = world.gh
    gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, T0 - 4 * DAY), ready=False)
    gh.put(2, _at(world, monkeypatch, {"b.txt": "2\n"}, T0 - DAY), ready=False)                     # pushed yesterday
    gh.put(3, _at(world, monkeypatch, {"c.txt": "3\n"}, T0 - 10 * DAY))                             # marked ready
    gh.put(4, _at(world, monkeypatch, {"d.txt": "4\n"}, T0 - 10 * DAY), labels=("hold",), ready=False)
    gh.put(5, h5 := _at(world, monkeypatch, {"e.txt": "5\n"}, T0 - 10 * DAY), ready=False)        # ejected yesterday
    gh.put(6, h6 := _at(world, monkeypatch, {"f.txt": "6\n"}, T0 - 10 * DAY), ready=False)        # reviewed 2 days ago
    gh.prs[7] = PR(7, _at(world, monkeypatch, {"g.txt": "7\n"}, T0 - 5 * DAY), "change 7", "Lane `lane-7`.", base="feature",
                   branch="cursor/w-7")
    _git(world.origin, "update-ref", "refs/pull/7/head", gh.prs[7].head)
    marks = {(5, h5): _iso(T0 - DAY)}
    q = world.queue(marked_at=lambda n, h: marks.get((n, h)))
    reviews = {(6, h6): [_iso(T0 - 2 * DAY)]}
    entries = q.status(lambda prs: stale_prs(q, prs, T0, 3, lambda n, h: reviews.get((n, h), [])))
    stale = {n for n, e in entries.items() if e.idle_since is not None}
    assert stale == {1, 7}
    assert entries[1].idle_since == T0 - 4 * DAY
    assert entries[1].why.startswith(f"stale, idle since {_iso(T0 - 4 * DAY)[:10]}: not marked ready at ")
    assert entries[7].state == "waiting" and entries[7].why.endswith("based on feature, and not marked ready at "
                                                                      f"{gh.prs[7].head[:8]}")
    assert stale_prs(q, gh.open_prs(), T0, 0) == {}, "0 is off"


def test_status_reads_stale_days_from_drive_toml_at_main(world, monkeypatch, capsys):
    """The command's own path: `stale_days` and `stale_keep` from drive.toml at `main`, and `stale`/`idle_since` in its JSON."""
    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(world.tmp / "store.toml"))
    sha = world.change({DRIVE_FLAGS: 'on = ["m1"]\nstale_days = 3\nstale_close_days = 7\nstale_keep = [3]\n'})
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/heads/main")
    world.gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, 1_000_000_000), ready=False)         # 2001: long idle
    world.gh.put(2, world.change({"b.txt": "2\n"}), ready=False)                                     # just now
    world.gh.put(3, _at(world, monkeypatch, {"c.txt": "3\n"}, 1_000_000_000), ready=False)         # as idle, but kept
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    assert Q.main(["status", "--json", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo",
                   "o/r", "--store", str(world.tmp / "store")]) == 0
    rows = {r["pr"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows[1]["stale"] and rows[1]["idle_since"] == "2001-09-09T01:46:40Z"
    assert rows[1]["why"].startswith("stale, idle since 2001-09-09: "), rows[1]["why"]
    assert not rows[2]["stale"] and rows[2]["idle_since"] is None
    assert not rows[3]["stale"] and rows[3]["idle_since"] is None


def test_status_pulls_the_stores_labels_in_one_batch_however_many_prs_are_open(world, monkeypatch, capsys):
    """`research queue status` pulled the store's labels once for admission, once for the PRs based off `main` and once for the
    stale ones, and a PR none of those covered alone; one batch of every open PR's labels (each of its heads', which a head
    with none of its own may take), whatever their number, keeps the command within seconds on an agent VM."""
    from research.store import labels_sync

    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(world.tmp / "store.toml"))
    sha = world.change({DRIVE_FLAGS: 'on = ["m1"]\nstale_days = 3\n'})
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/heads/main")
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    batches: list[int] = []
    for name in ("pull_targets", "pull_prefixes"):
        pull = getattr(labels_sync, name)
        monkeypatch.setattr(labels_sync, name, lambda store, todo, jobs=1, pull=pull: batches.append(len(todo))
                            or pull(store, todo, jobs=jobs))
    monkeypatch.setattr(labels_sync, "pull_prefix", lambda *a, **k: pytest.fail("status pulls no PR's heads one PR at a time"))
    argv = ["status", "--json", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo", "o/r",
            "--store", str(world.tmp / "store")]

    def add(n: int) -> None:
        world.gh.put(n, _at(world, monkeypatch, {f"{n}.txt": "x\n"}, 1_000_000_000), labels=("hold",) if n % 4 == 0 < n % 5 else ())
        if n % 5 == 0:
            world.gh.prs[n] = dataclasses.replace(world.gh.prs[n], base="feature", branch=f"cursor/w-{n}")

    for count in (5, 20):
        for n in range(len(world.gh.prs) + 1, count + 1):
            add(n)
        batches.clear()
        assert Q.main(argv) == 0
        rows = {r["pr"]: r for r in json.loads(capsys.readouterr().out)}
        assert len(rows) == count and rows[1]["stale"] and rows[5]["why"].startswith("stale") and "held" in rows[4]["why"]
        assert batches == [count], "one pull of every open PR's labels: none for admission, off-main or stale alone"
