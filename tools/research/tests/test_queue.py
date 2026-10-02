"""`research queue`: `next` is `main` plus the admitted PRs, one merge commit each; `main` fast-forwards to the newest commit on it
with a passing `check` of that exact commit.  Every test runs on a throwaway bare origin, with a fake GitHub that serves the open
PRs and writes their `refs/pull/{n}/head`."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.cli import main as research_main
from research.queue import PR, SHADOW_BRANCH, Queue, QueueError, Rules, grant_target
from research.store import ids, labels_sync, vocab
from research.store.local import LocalStore
from research.store.model import Label

RULES = """\
hold_label = "hold"

[[grant]]
role = "vllm-coordinator"
paths = ["integrations/vllm/"]

[[grant]]
role = "statement-reviewer"
lean_audit = ["pins", "reads"]
"""
AGENT = "bc-605d7c89-ca73-5a32-a582-ee77c49e762a"


def PASS(pr: PR) -> None:
    """A quick tier that passes."""
    return None


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


class FakeGitHub:
    def __init__(self, origin: Path):
        self.origin, self.prs, self.ready = origin, {}, set()

    def put(self, n: int, head: str, *, body: str | None = None, draft: bool = False, labels: tuple[str, ...] = (), ready: bool = True) -> None:
        """A PR at `head`; `ready`: its owner has said so at this head (`research queue ready`), as every PR here has unless told."""
        _git(self.origin, "update-ref", f"refs/pull/{n}/head", head)
        self.prs[n] = PR(n, head, f"change {n}", f"Lane `lane-{n}`." if body is None else body, draft, labels)
        if ready:
            self.ready.add((n, head))

    def open_prs(self) -> list[PR]:
        return list(self.prs.values())


class World:
    def __init__(self, tmp: Path):
        self.tmp, self.origin, self.dev = tmp, tmp / "origin.git", tmp / "dev"
        _git(tmp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        for rel, text in {"a.txt": "a\n", "b.txt": "b\n", "c.txt": "c\n", "ci/queue.toml": RULES, ".gitignore": "__pycache__/\n",
                          "pkg/lean-audit.json": json.dumps({"pins": {"T": "1"}, "reads": {}, "dependencies": {"M": "1"}}),
                          "mytools/decl_check.py": "from research.store.tool import Tool\nT = Tool(name='check', version='1')\n"}.items():
            (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dev / rel).write_text(text)
        _git(self.dev, "init", "-q", "-b", "main")
        _git(self.dev, "add", "-A")
        _git(self.dev, "commit", "-q", "-m", "init")
        _git(self.dev, "remote", "add", "origin", str(self.origin))
        _git(self.dev, "push", "-q", "origin", "main")
        self.gh, self.grants, self.logs = FakeGitHub(self.origin), {}, []
        self.store = LocalStore(tmp / "store")

    def change(self, files: dict[str, str], base: str = "origin/main") -> str:
        """A commit on `base` writing `files`; returns its sha."""
        _git(self.dev, "fetch", "-q", "origin")
        _git(self.dev, "checkout", "-q", "--detach", base)
        for rel, text in files.items():
            (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dev / rel).write_text(text)
        _git(self.dev, "add", "-A")
        _git(self.dev, "commit", "-q", "-m", f"change {sorted(files)}")
        sha = _git(self.dev, "rev-parse", "HEAD")
        _git(self.dev, "push", "-q", "origin", f"{sha}:refs/dev/{sha}")         # the object on the origin, as a PR's head is
        return sha

    def queue(self, **kw) -> Queue:
        return Queue(self.tmp / "q" / "verity", remote=str(self.origin), github=self.gh, grants=lambda n, h: self.grants.get((n, h), set()),
                     ready=lambda n, h: (n, h) in self.gh.ready,
                     store=self.store, notes=self.tmp / "notes", log=self.logs.append, **kw)

    def on_next(self, q: Queue) -> list[int]:
        q.fetch()
        return [m.pr for m in q.chain()]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_ready_prs_merge_onto_next_in_number_order_drafts_too_and_held_ownerless_or_unmarked_prs_wait(world):
    world.gh.put(3, world.change({"c.txt": "c3\n"}))
    world.gh.put(1, world.change({"a.txt": "a1\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}), draft=True)
    world.gh.put(4, world.change({"d.txt": "d\n"}), labels=("hold",))
    world.gh.put(5, world.change({"e.txt": "e\n"}), body="no owner here")
    world.gh.put(6, world.change({"f.txt": "f\n"}), body=f"[Open in Web](https://cursor.com/agents/{AGENT})")
    world.gh.put(7, world.change({"g.txt": "g\n"}), draft=True, labels=("hold",))
    world.gh.put(8, h8 := world.change({"h.txt": "h\n"}), draft=True, ready=False)
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2, 3, 6] == world.on_next(q)
    assert members[3].owner == AGENT and members[0].owner == "lane-1"
    assert entries[4].why == entries[7].why == "held (`hold` label)" and entries[5].why.startswith("no owner")
    assert entries[8].why == f"not marked ready at {h8[:8]}: `research queue ready 8`"
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    assert _git(world.origin, "rev-parse", f"{tip}^1^1^1^1") == _git(world.origin, "rev-parse", "main")    # one merge commit each
    msg = _git(world.origin, "log", "-1", "--format=%B", tip)
    assert msg.startswith("Queue: #6 change 6") and f"Head: {world.gh.prs[6].head}" in msg
    status = q.status()
    assert [n for n, e in sorted(status.items()) if e.state == "queued"] == [1, 2, 3, 6] and status[3].position == 3


def test_a_new_push_moves_its_pr_to_the_tip_and_keeps_the_commits_before_it(world):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    first, _ = q.sync()
    world.gh.put(2, world.change({"b.txt": "2, again\n"}))
    again, _ = q.sync()
    assert [m.pr for m in again] == [1, 3, 2]
    assert again[0].commit == first[0].commit                  # the unchanged prefix is kept, not re-merged
    assert again[2].head == world.gh.prs[2].head


def test_a_conflicting_pr_is_left_out_and_its_owner_told_which_queued_pr_touched_the_file(world):
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, world.change({"a.txt": "two\n", "b.txt": "b2\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "refused" and entries[2].why == "conflicts with `next` in a.txt (touched by #1)"
    notices = sorted((world.tmp / "notes" / "lanes" / "lane-2").glob("*-handoff-from-merge-queue-2-conflict.md"))
    assert len(notices) == 1 and "touched by #1" in notices[0].read_text()
    q.sync()
    assert len(list((world.tmp / "notes" / "lanes" / "lane-2").glob("*.md"))) == 1      # one notice per head
    assert q.status()[2].why == "conflicts with `next` in a.txt (touched by #1) (at the last sync)"


def test_a_pr_containing_another_goes_in_after_it_and_waits_while_it_waits(world):
    parent = world.change({"integrations/vllm/x.py": "x = 1\n"})
    world.gh.put(5, parent)
    world.gh.put(4, world.change({"c.txt": "stacked\n"}, base=parent))
    q = world.queue()
    members, entries = q.sync()
    assert members == [] and entries[5].why == f"waiting for the vllm-coordinator grant at {parent[:8]}"
    assert entries[4].state == "waiting" and entries[4].after == (5,)      # it carries #5's change, grant included
    world.grants[(5, parent)] = {"vllm-coordinator"}
    world.grants[(4, world.gh.prs[4].head)] = {"vllm-coordinator"}     # #4 carries #5's change, so it needs the grant too
    members, _ = q.sync()
    assert [m.pr for m in members] == [5, 4]


def test_grants_follow_paths_and_lean_audit_sections_at_the_exact_head(world):
    audit = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    world.gh.put(1, world.change({"integrations/vllm/x.py": "x = 1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "2"}})}))
    world.gh.put(3, world.change({"pkg/lean-audit.json": json.dumps({**audit, "dependencies": {"M": "2"}})}))
    q = world.queue()
    _, entries = q.sync()
    assert entries[1].why.startswith("waiting for the vllm-coordinator grant")
    assert entries[2].why.startswith("waiting for the statement-reviewer grant")
    assert entries[3].state == "queued"                         # a dependency digest isn't a statement
    world.grants[(1, world.gh.prs[1].head)] = {"vllm-coordinator"}
    assert 1 in [m.pr for m in q.sync()[0]]
    world.gh.put(1, world.change({"integrations/vllm/x.py": "x = 2\n"}))
    members, entries = q.sync()
    assert 1 not in [m.pr for m in members] and entries[1].state == "waiting"     # a new head needs a new grant


def test_an_ejected_head_stays_out_until_its_pr_is_pushed_again(world):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    q.sync()
    q.eject(2, "breaks suite `verity` at r20260929-000000-abcd", "test")
    assert world.on_next(q) == [1, 2, 3]                        # an ejection takes effect at the next sync
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 3] and entries[2].why.startswith("ejected at")
    notice = next((world.tmp / "notes" / "lanes" / "lane-2").glob("*-eject.md")).read_text()
    assert "breaks suite `verity`" in notice
    world.gh.put(2, world.change({"b.txt": "fixed\n"}))
    assert [m.pr for m in q.sync()[0]] == [1, 3, 2]


def test_a_main_moved_outside_the_queue_rebuilds_next_on_it(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    q = world.queue()
    q.sync()
    moved = world.change({"d.txt": "landed by hand\n"})
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    members, _ = q.sync()
    assert [m.pr for m in members] == [1, 2]
    _git(world.origin, "merge-base", "--is-ancestor", moved, "refs/heads/next")      # raises unless next is on the new main


def test_the_clone_follows_the_callers_origin_so_a_url_that_stopped_working_does_not_strand_it(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.queue().sync()
    moved = world.tmp / "origin-now.git"
    world.origin.rename(moved)                  # the URL the clone was made from fails now, as one with a rotated token does
    world.origin = world.gh.origin = moved
    members, _ = world.queue().sync()
    assert [m.pr for m in members] == [1]
    assert _git(world.tmp / "q" / "verity", "remote", "get-url", "origin") == str(moved)


def test_a_rewritten_pr_head_is_fetched_by_force_and_merged(world):
    old = world.change({"a.txt": "1\n"})
    world.gh.put(1, old)
    q = world.queue()
    q.sync()
    new = world.change({"a.txt": "rewritten\n"})                # not a descendant of the old head: a force-push
    world.gh.put(1, new)
    members, _ = q.sync()
    assert members[0].head == new and _git(world.origin, "show", "refs/heads/next:a.txt") == "rewritten"


def test_land_fast_forwards_main_to_the_newest_commit_with_a_passing_check_and_shadow_lands_nothing(world, monkeypatch):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    main_before = _git(world.origin, "rev-parse", "main")
    shadow = world.queue(shadow=True)
    shadow.sync()
    assert _git(world.origin, "rev-parse", f"refs/heads/{SHADOW_BRANCH}") and world.on_next(shadow) == [1, 2, 3]
    members, _ = world.queue().sync()
    monkeypatch.setenv("RESEARCH_STORE", str(world.tmp / "store"))
    for m, command in ((members[1], "true"), (members[2], "false")):
        tree = world.tmp / f"check-{m.pr}"
        _git(world.dev, "fetch", "-q", "origin")
        _git(world.dev, "worktree", "add", "-q", "--detach", str(tree), m.commit)
        monkeypatch.chdir(tree)
        research_main(["run", "--runs-dir", str(world.tmp / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                       "--cwd", "source", "--", command])
    shadow.land()                                               # whatever it finds, a shadow queue never moves `main`
    assert _git(world.origin, "rev-parse", "main") == main_before
    q = world.queue()
    assert q.land(dry_run=True) == members[1] and _git(world.origin, "rev-parse", "main") == main_before
    assert q.land() == members[1]
    assert _git(world.origin, "rev-parse", "main") == members[1].commit      # a fast-forward: exactly the checked commit
    assert world.on_next(q) == [3]


def test_pr_label_targets_and_grants(tmp_path, monkeypatch):
    head = "0123456789abcdef0123456789abcdef01234567"
    assert ids.is_pr(grant_target(12, head)) and ids.kind_of_id(f"pr:12@{head}") == "pr"
    assert not any(ids.is_pr(x) for x in (f"pr:0@{head}", f"pr:12@{head[:12]}", f"pr:x@{head}", f"12@{head}"))
    assert vocab.check(vocab.GRANT, "red-team") is None and vocab.check(vocab.GRANT, "anyone") is not None
    assert Label(target=f"pr:12@{head[:8]}", key="grant", value="review", by="t", ts="2026-09-29T05:30:00Z").errors()
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    reviewer, queue_host = LocalStore(tmp_path / "reviewer"), LocalStore(tmp_path / "queue")
    reviewer.add_label(grant_target(12, head), vocab.GRANT, "red-team", "red-team-flock-3", ts="2026-09-29T05:30:00Z")
    assert labels_sync.push_targets(reviewer, [grant_target(12, head)])["failed"] == []
    assert len(labels_sync.pull_targets(queue_host, [grant_target(12, head)])["pulled"]) == 1
    assert [lab.value for lab in queue_host.labels(grant_target(12, head), key=vocab.GRANT)] == ["red-team"]



class _Unreachable:
    """A configured remote whose endpoint is down: every call raises."""

    def describe(self):
        return "fs:/unreachable"

    def __getattr__(self, name):
        def fail(*a, **k):
            raise OSError("injected: endpoint unreachable")
        return fail


def test_grants_read_through_to_the_remote_and_a_failed_pull_grants_nothing(tmp_path, monkeypatch):
    from research.queue import StoreGrants

    head = "0123456789abcdef0123456789abcdef01234567"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    reviewer, queue_host = LocalStore(tmp_path / "reviewer"), LocalStore(tmp_path / "queue")
    reviewer.add_label(grant_target(12, head), vocab.GRANT, "red-team", "red-team-flock-3", ts="2026-09-29T05:30:00Z")
    reviewer.add_label(grant_target(12, head), vocab.READY, True, "lane-12", ts="2026-09-29T05:31:00Z")
    assert labels_sync.push_targets(reviewer, [grant_target(12, head)])["failed"] == []
    logs: list[str] = []
    grants = StoreGrants(queue_host, log=logs.append)
    real = LocalStore.remote.fget
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    assert grants(12, head) == set() and grants.ready(12, head) is False, "a failed pull is never a grant"
    assert len(logs) == 2 and all(x.startswith(f"research queue: labels of pr:12@{head}: pulling from fs:/unreachable failed (OSError")
                                  for x in logs)
    monkeypatch.setattr(LocalStore, "remote", property(real))
    assert grants(12, head) == {"red-team"} and grants.ready(12, head) is True and len(logs) == 2


def test_grants_with_no_remote_configured_say_so_once(tmp_path, monkeypatch):
    """A machine with no store remote sees only its own marks: `not marked ready` there may be a mark made elsewhere."""
    from research.queue import StoreGrants

    head = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    store = LocalStore(tmp_path / "store")
    assert store.remote is None
    logs: list[str] = []
    grants = StoreGrants(store, log=logs.append)
    assert grants.ready(12, head) is False and grants(12, head) == set() and grants.ready(13, head) is False
    assert len(logs) == 1 and logs[0].startswith("research queue: no evidence-store remote is configured here")
    store.add_label(grant_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-02T13:00:00Z")
    assert grants.ready(12, head) is True and len(logs) == 1


def test_rules_refuse_what_they_cannot_place():
    assert [r.role for r in Rules.parse(RULES).grants] == ["vllm-coordinator", "statement-reviewer"]
    for bad in ('[[grant]]\nrole = "whoever"\npaths = ["x/"]\n', '[[grant]]\nrole = "review"\n', 'colour = "red"\n',
                '[[grant]]\nrole = "review"\npaths = ["x/"]\nwhen = "always"\n'):
        with pytest.raises(QueueError):
            Rules.parse(bad)


def test_until_the_move_the_rules_are_read_from_tools_check(world):
    moved = world.change({"tools/check/queue.toml": RULES.replace('paths = ["integrations/vllm/"]', 'paths = ["integrations/vllm/", "!README.md"]')})
    _git(world.dev, "rm", "-q", "ci/queue.toml")
    _git(world.dev, "commit", "-q", "-m", "rules where they are before the move")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    assert moved
    world.gh.put(1, world.change({"integrations/vllm/x.py": "x = 1\n", "integrations/vllm/README.md": "r\n"}))
    world.gh.put(2, world.change({"integrations/vllm/README.md": "docs only\n"}))
    _, entries = world.queue().sync()
    assert entries[1].why.startswith("waiting for the vllm-coordinator grant") and entries[2].state == "queued"


def test_a_ready_label_holds_for_one_head_and_research_queue_ready_writes_it(world, capsys):
    from research.queue import StoreGrants, ready

    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    labels = StoreGrants(world.store, log=world.logs.append)
    assert labels.ready(1, h1) is False
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    assert f"#1 is ready at {h1[:12]} (pr:1@{h1})" in capsys.readouterr().out
    assert labels.ready(1, h1) is True
    q = Queue(world.tmp / "q" / "verity", remote=str(world.origin), github=world.gh, grants=labels, ready=labels.ready,
              store=world.store, notes=world.tmp / "notes", log=world.logs.append)
    members, _ = q.sync()
    assert [m.pr for m in members] == [1]
    h2 = world.change({"a.txt": "a2\n"})
    world.gh.put(1, h2, ready=False)                           # a new push: the label was for the old head
    members, entries = q.sync()
    assert members == [] and entries[1].why == f"not marked ready at {h2[:8]}: `research queue ready 1`"
    world.gh.put(1, h2, labels=("hold",), ready=False)
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    members, entries = q.sync()
    assert members == [] and entries[1].why == "held (`hold` label)"   # hold wins over ready
    assert ready(world.gh, world.store, 99, "lane-1", PASS) == 2


#: a `suites.py` that says what it was asked and what it found, and fails one test when FAKE_FAIL names it
FAKE_SUITES = """\
# it knows "--changed"
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
Path(os.environ["FAKE_SEEN"]).write_text(json.dumps({"argv": a, "files": sorted(p.name for p in Path.cwd().iterdir()),
                                                     "pythonpath": os.environ.get("PYTHONPATH")}))
fail = os.environ.get("FAKE_FAIL")
Path(a[a.index("--json") + 1]).write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed" if fail else "passed",
                                                       "failed_tests": [fail] if fail else None}]))
sys.exit(1 if fail else 0)
"""


def test_the_quick_tier_runs_the_suites_a_pr_changes_from_a_clean_checkout_at_its_head_and_a_failure_refuses(world, monkeypatch, capsys):
    from research import queue as Q

    main = world.change({"tools/check/suites.py": FAKE_SUITES})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    seen = world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("PYTHONPATH", str(world.tmp / "the-callers-tree"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert tier(world.gh.prs[1]) is None
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", main] and s["pythonpath"] is None   # from the PR's merge base, in its own tree
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    assert _git(wt, "rev-parse", "HEAD") == h1
    (wt / "stray.txt").write_text("left by the last PR\n")
    (wt / ".venv").mkdir()

    monkeypatch.setenv("FAKE_FAIL", "tests/test_a.py::test_a")
    assert tier(world.gh.prs[1]) == "pkg: tests/test_a.py::test_a"
    assert {"stray.txt", ".venv"} & set(json.loads(seen.read_text())["files"]) == {".venv"}     # cleaned, its environment kept
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: pkg: tests/test_a.py::test_a" in capsys.readouterr().err
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is False             # no label without a pass

    monkeypatch.delenv("FAKE_FAIL")
    h2 = world.change({"b.txt": "b2\n"})
    world.gh.put(1, h2, ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and _git(wt, "rev-parse", "HEAD") == h2
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h2) is True
    h3 = world.change({"tools/check/suites.py": FAKE_SUITES.replace('"--changed"', "nothing")})
    world.gh.put(1, h3, ready=False)
    assert tier(world.gh.prs[1]) is None
    assert json.loads(seen.read_text())["argv"][:2] == ["--quick", "--json"]     # a runner from before --changed: every suite
