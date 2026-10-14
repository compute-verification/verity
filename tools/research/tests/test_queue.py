"""`research queue`: `next` is `main` plus the admitted PRs, one merge commit each; `main` fast-forwards to the newest commit on it
with a passing `check` of that exact commit.  Every test runs on a throwaway bare origin, with a fake GitHub that serves the open
PRs and writes their `refs/pull/{n}/head`."""
from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from research.cli import _custody_ttl_problem, main as research_main
from research.pods.nebius import slot_only
from research.queue import PR, SHADOW_BRANCH, Queue, QueueError, Rules, credential_args, label_target
from research.store import ids, labels_sync, vocab
from research.store.local import LocalStore
from research.store.model import Label
from test_merge_gate import CHECK_FILES, CHECKED

RULES = 'hold_label = "hold"\n'
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
                          "pkg/lean-audit.json": json.dumps({"guarantees": {"T": "1"}, "reads": {}, "dependencies": {"M": "1"}}),
                          "mytools/decl_check.py": "from research.store.tool import Tool\nT = Tool(name='check', version='1')\n",
                          **CHECK_FILES}.items():
            (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dev / rel).write_text(text)
        _git(self.dev, "init", "-q", "-b", "main")
        _git(self.dev, "add", "-A")
        _git(self.dev, "commit", "-q", "-m", "init")
        _git(self.dev, "remote", "add", "origin", str(self.origin))
        _git(self.dev, "push", "-q", "origin", "main")
        self.gh, self.logs = FakeGitHub(self.origin), []
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

    def with_store_remote(self) -> None:
        """Give `store` a remote (a directory): `StoreLabels` answers only from labels it pulled from one."""
        (self.tmp / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{self.tmp / "vault"}"\n')
        self.store = LocalStore(self.tmp / "store", config_path=self.tmp / "store.toml")

    def queue(self, **kw) -> Queue:
        kw.setdefault("ready", lambda n, h: (n, h) in self.gh.ready)
        return Queue(self.tmp / "q" / "verity", remote=str(self.origin), github=self.gh,
                     store=self.store, log=self.logs.append, **kw)

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
    assert entries[8].why == f"not marked ready at {h8[:8]}: `research queue ready 8 --on MACHINE` (or --local; --mark without the tier)"
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    assert _git(world.origin, "rev-parse", f"{tip}^1^1^1^1") == _git(world.origin, "rev-parse", "main")    # one merge commit each
    msg = _git(world.origin, "log", "-1", "--format=%B", tip)
    assert msg.startswith("Queue: #6 change 6") and f"Head: {world.gh.prs[6].head}" in msg
    status = q.status()
    assert [n for n, e in sorted(status.items()) if e.state == "queued"] == [1, 2, 3, 6] and status[3].position == 3


def test_the_queues_merges_carry_the_time_they_are_made_whatever_date_the_caller_pins(world, monkeypatch):
    """12 Oct: the hand-landing recipe pinned 2026-09-28T23:00Z for reproducible merge ids, a merge on `next` built under it
    (ba87196a8) landed with that date, and `git log --since` over `main` stopped early there."""
    world.gh.put(1, world.change({"a.txt": "a1\n"}))
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-09-28T23:00:00+0000")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-09-28T23:00:00+0000")
    q = world.queue()
    [m] = q.sync()[0]
    assert "2026-09-28" not in _git(q.workdir, "log", "-1", "--format=%aI %cI", m.commit)


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


def test_an_api_head_ahead_of_its_pull_ref_waits_without_blocking_the_queue(world):
    """GitHub's REST PR head can update before refs/pull/N/head; the queue must not walk the absent REST commit or stop the tick."""
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, old := world.change({"b.txt": "old\n"}))
    q = world.queue()
    assert [m.pr for m in q.sync()[0]] == [1, 2]

    new = world.change({"b.txt": "new\n"})
    world.gh.prs[2] = PR(2, new, "change 2", "Lane `lane-2`.")
    world.gh.ready.add((2, new))
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "waiting"
    assert entries[2].why == f"GitHub's API head {new[:12]} and pull ref {old[:12]} disagree; retrying next sync"
    assert world.logs[-2] == f"research queue: #2's API head {new[:12]} and fetched pull ref {old[:12]} disagree; retrying next sync"

    world.gh.put(2, new)
    assert [m.pr for m in q.sync()[0]] == [1, 2]


def test_first_puts_a_ready_fix_at_the_bottom_of_next_ahead_of_members_that_keep_their_place(world):
    """12 Oct: a flaky test on `main` failed 8577 and its fix (#1858) sat behind the next train; `--first` restacks `next` so the
    next train starts with the fix.  A PR in `--first` that isn't ready is passed over, and another queue's sync without the
    flag keeps the new order."""
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    world.queue().sync()
    world.gh.put(9, world.change({"fix.txt": "fix\n"}))
    world.gh.put(8, world.change({"other.txt": "x\n"}), ready=False)
    members, _ = world.queue(first=(8, 9)).sync()
    assert [m.pr for m in members] == [9, 1, 2, 3]
    again, _ = world.queue().sync()
    assert [m.commit for m in again] == [m.commit for m in members]


def test_the_queues_clone_merges_lean_audit_records_with_the_record_driver(world):
    """`.gitattributes` names `lean-audit` for every lean-audit.json; the queue's own clone must define it, or two PRs that each
    add a record conflict as text (12 Oct: #1806)."""
    from research.clone import RECORD_DRIVER

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    tools = {f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}
    rec = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    base = world.change({**tools, ".gitattributes": "lean-audit.json merge=lean-audit\n"})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    for n, g in ((1, "A"), (2, "B")):
        world.gh.put(n, world.change({"pkg/lean-audit.json": json.dumps({**rec, "guarantees": {**rec["guarantees"], g: "1"}})}))
    q = world.queue()
    members, entries = q.sync()
    for k, v in RECORD_DRIVER.items():
        assert _git(q.workdir, "config", "--get", k) == v
    assert [m.pr for m in members] == [1, 2], entries[2].why
    merged = json.loads(_git(world.origin, "show", "refs/heads/next:pkg/lean-audit.json"))
    assert set(merged["guarantees"]) == {"T", "A", "B"}


def test_a_conflicting_pr_is_left_out_with_the_queued_pr_that_touched_the_file_and_no_note_is_written(world, monkeypatch):
    """13 Oct: a sync on @lean's VM wrote conflict handoffs for ci's and circuits' PRs into @lean's notes clone, where they
    waited for @lean's next notes push; drive tells each conflict in the merge thread (`Driver.refuse`)."""
    monkeypatch.setenv("RESEARCH_NOTES", str(world.tmp / "notes"))
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, world.change({"a.txt": "two\n", "b.txt": "b2\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "refused" and entries[2].why == "conflicts with `next` in a.txt (touched by #1)"
    assert q.load_state()["conflicts"]["2"]["why"] == entries[2].why
    assert not (world.tmp / "notes").exists()
    assert q.status()[2].why == "conflicts with `next` in a.txt (touched by #1) (at the last sync)"


def test_a_pr_containing_another_goes_in_after_it_and_waits_while_it_waits(world):
    parent = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, parent, ready=False)
    world.gh.put(4, world.change({"c.txt": "stacked\n"}, base=parent))
    q = world.queue()
    members, entries = q.sync()
    assert members == [] and entries[5].why.startswith(f"not marked ready at {parent[:8]}")
    assert entries[4].state == "waiting" and entries[4].after == (5,)
    world.gh.ready.add((5, parent))
    members, _ = q.sync()
    assert [m.pr for m in members] == [5, 4]


def test_an_ejected_head_stays_out_until_its_pr_is_pushed_again(world):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    q.sync()
    q.eject(2, "breaks suite `verity` at r20260929-000000-abcd", "test")
    assert world.on_next(q) == [1, 2, 3]                        # an ejection takes effect at the next sync
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 3] and entries[2].why.startswith("ejected at")
    assert "breaks suite `verity`" in entries[2].why                # the notice is `tell_ejected`'s post, tested below
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


HTTP_BACKEND = Path(subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout.strip()) / "git-http-backend"


class _GitHTTP(BaseHTTPRequestHandler):
    """`git http-backend` over `root` behind Basic auth as `u:p`, as GitHub serves a private repository: without credentials, a 401."""
    root: Path

    def do_GET(self):
        if self.headers.get("Authorization") != "Basic " + base64.b64encode(b"u:p").decode():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="origin"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        path, _, query = self.path.partition("?")
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        env = {"PATH": os.environ["PATH"], "GIT_PROJECT_ROOT": str(self.root), "GIT_HTTP_EXPORT_ALL": "1", "REMOTE_USER": "u",
               "PATH_INFO": path, "QUERY_STRING": query, "REQUEST_METHOD": self.command, "CONTENT_LENGTH": str(len(body)),
               "CONTENT_TYPE": self.headers.get("Content-Type", ""), "HTTP_CONTENT_ENCODING": self.headers.get("Content-Encoding", ""),
               "GIT_PROTOCOL": self.headers.get("Git-Protocol", "")}
        cgi = subprocess.run([str(HTTP_BACKEND)], input=body, env=env, capture_output=True, check=True).stdout
        head, _, out = cgi.partition(b"\r\n\r\n")
        headers = [line.split(": ", 1) for line in head.decode().split("\r\n") if line]
        self.send_response(int(next((v for k, v in headers if k == "Status"), "200").split()[0]))
        for k, v in headers:
            if k != "Status":
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_POST = do_GET

    def log_message(self, *args):
        pass


def _no_machine_git_config(tmp: Path, monkeypatch) -> None:
    """A cloud VM's view: no credential helper (nor prompt) from git's global or system config, only from a checkout's own."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp / "global.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_ASKPASS", "SSH_ASKPASS"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def served(world, monkeypatch):
    """`world`'s origin over http behind a password; its URL."""
    _no_machine_git_config(world.tmp, monkeypatch)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (_GitHTTP,), {"root": world.tmp}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/{world.origin.name}"
    srv.shutdown()
    srv.server_close()


@pytest.mark.skipif(not HTTP_BACKEND.exists(), reason="no git-http-backend in this git")
def test_the_clone_authenticates_as_the_callers_checkout_does_and_fails_rather_than_prompt_or_answer_stale(world, served):
    """A cloud VM's credential helper is in its checkout's own config (one that needs the path, as the VMs' broker does), which the
    queue's own clone never read, so `status` failed there at the clone.  It takes the checkout's settings for the remote now; and
    with no credentials git fails at once instead of prompting, also once the clone has fetched before."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _git(world.dev, "config", f"credential.{served}.helper", "")
    _git(world.dev, "config", "--add", f"credential.{served}.helper",
         "!f() { test \"$1\" = get && grep -qx path=origin.git && printf 'username=u\\npassword=p\\n'; }; f")
    _git(world.dev, "config", f"credential.{served}.useHttpPath", "true")

    def queue(auth: tuple[str, ...]) -> Queue:
        return Queue(world.tmp / "q" / "verity", remote=served, github=world.gh, ready=lambda n, h: True,
                     log=world.logs.append, auth=auth)

    with pytest.raises(QueueError, match=r"git clone .*terminal prompts disabled; no credential helper applies to the remote"):
        queue(()).status()
    auth = credential_args(world.dev, served)
    members, _ = queue(auth).sync()
    assert [m.pr for m in members] == [1] and _git(world.origin, "rev-parse", "refs/heads/next") == members[0].commit
    assert [(n, e.state) for n, e in queue(auth).status().items()] == [(1, "queued")]
    with pytest.raises(QueueError, match=r"git fetch .*could not read Username"):
        queue(()).status()


def test_credential_args_take_the_checkouts_own_settings_else_gh_where_no_helper_applies(tmp_path, monkeypatch):
    _no_machine_git_config(tmp_path, monkeypatch)
    url, repo, bin_dir = "https://verity-agent@github.com/o/r", tmp_path / "checkout", tmp_path / "bin"
    _git(tmp_path, "init", "-q", str(repo))
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/bin/sh\n")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    gh = ("-c", "credential.helper=", "-c", f"credential.helper=!{shlex.quote(str(bin_dir / 'gh'))} auth git-credential")
    assert credential_args(repo, url) == credential_args(tmp_path, url) == gh, "no helper anywhere, in a checkout or not: gh's"
    assert credential_args(repo, "/srv/origin.git") == credential_args(repo, "git@github.com:o/r") == (), "not http: no credentials"
    _git(repo, "config", "credential.https://verity-agent@github.com.helper", "")
    _git(repo, "config", "--add", "credential.https://verity-agent@github.com.helper", "!broker credential")
    _git(repo, "config", "credential.https://verity-agent@github.com.useHttpPath", "true")
    _git(repo, "config", "credential.https://elsewhere.example.helper", "!not-this-one")
    assert credential_args(repo, url) == ("-c", "credential.helper=", "-c", "credential.helper=!broker credential",
                                          "-c", "credential.usehttppath=true")
    (tmp_path / "global.gitconfig").write_text("[credential]\n\thelper = store\n")
    assert credential_args(tmp_path, url) == (), "a helper from git's global config serves the clone as it is"


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
    for m, command in ((members[1], CHECKED), (members[2], ("false",))):
        tree = world.tmp / f"check-{m.pr}"
        _git(world.dev, "fetch", "-q", "origin")
        _git(world.dev, "worktree", "add", "-q", "--detach", str(tree), m.commit)
        monkeypatch.chdir(tree)
        research_main(["run", "--runs-dir", str(world.tmp / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                       "--cwd", "source", "--", *command])
    shadow.land()                                               # whatever it finds, a shadow queue never moves `main`
    assert _git(world.origin, "rev-parse", "main") == main_before
    q = world.queue()
    assert q.land(dry_run=True) == members[1] and _git(world.origin, "rev-parse", "main") == main_before
    assert q.land() == members[1]
    assert _git(world.origin, "rev-parse", "main") == members[1].commit      # a fast-forward: exactly the checked commit
    assert world.on_next(q) == [3]


def test_pr_label_targets(tmp_path, monkeypatch):
    head = "0123456789abcdef0123456789abcdef01234567"
    assert ids.is_pr(label_target(12, head)) and ids.kind_of_id(f"pr:12@{head}") == "pr"
    assert not any(ids.is_pr(x) for x in (f"pr:0@{head}", f"pr:12@{head[:12]}", f"pr:x@{head}", f"12@{head}"))
    assert Label(target=f"pr:12@{head[:8]}", key="ready", value=True, by="t", ts="2026-09-29T05:30:00Z").errors()
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane, queue_host = LocalStore(tmp_path / "lane"), LocalStore(tmp_path / "queue")
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-09-29T05:30:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert len(labels_sync.pull_targets(queue_host, [label_target(12, head)])["pulled"]) == 1
    assert [lab.value for lab in queue_host.labels(label_target(12, head), key=vocab.READY)] == [True]


class _Unreachable:
    """A configured remote whose endpoint is down: every call raises."""

    def describe(self):
        return "fs:/unreachable"

    def __getattr__(self, name):
        def fail(*a, **k):
            raise OSError("injected: endpoint unreachable")
        return fail


def test_every_git_command_that_talks_to_the_remote_aborts_a_stalled_transfer(tmp_path, monkeypatch):
    import research.queue as Q

    q = World(tmp_path).queue()
    seen = []
    real = Q._git
    monkeypatch.setattr(Q, "_git", lambda cwd, *a, **k: seen.append(a) or real(cwd, *a, **k))
    q._clone()
    q.fetch()
    remote = [a for a in seen if {"clone", "fetch", "push", "ls-remote"} & set(a)]
    assert remote and all(a[: len(Q.STALL)] == Q.STALL for a in remote), seen


def test_ready_marks_read_through_to_the_remote_and_without_that_pull_the_queue_refuses_to_answer(tmp_path, monkeypatch):
    """A failed pull, an unusable remote, or none at all (a cloud VM's shell that lost its store secrets) used to leave the queue
    answering from this machine's labels, with a warning at most, so its ready marks could be stale.  Each is `STALE` now."""
    from research.queue import StoreLabels

    head = "0123456789abcdef0123456789abcdef01234567"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane, queue_host = LocalStore(tmp_path / "lane"), LocalStore(tmp_path / "queue")
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-09-29T05:31:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    logs: list[str] = []
    labels = StoreLabels(queue_host, log=logs.append)
    real = LocalStore.remote.fget
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    with pytest.raises(QueueError, match=rf"^STALE: the labels of pr:12@{head} weren't pulled from the store's remote "
                                         r"\(pulling from fs:/unreachable failed \(OSError: injected"):
        labels.ready(12, head)
    monkeypatch.setattr(LocalStore, "remote", property(real))
    assert labels.ready(12, head) is True and logs == []
    # the labels are on this machine now, and still don't answer without a pull
    absent, unusable = tmp_path / "absent.toml", tmp_path / "s3.toml"
    unusable.write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                        'access_key_id_env = "Q_AK_ABSENT"\nsecret_access_key_env = "Q_SK_ABSENT"\n')
    with pytest.raises(QueueError, match=r"^STALE: .*\(no remote configured"):
        StoreLabels(LocalStore(tmp_path / "queue", config_path=absent)).ready(12, head)
    with pytest.raises(QueueError, match=r"^STALE: .*Q_AK_ABSENT"):
        StoreLabels(LocalStore(tmp_path / "queue", config_path=unusable)).ready(12, head)
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", "R2_ENDPOINT,R2_BUCKET")
    with pytest.raises(QueueError, match=r"^STALE: .*R2_ENDPOINT, R2_BUCKET not in this shell .*fresh tmux login shell"):
        StoreLabels(LocalStore(tmp_path / "queue", config_path=absent)).ready(12, head)


def test_a_cutoff_admits_only_ready_marks_written_at_or_after_it(tmp_path, monkeypatch):
    """`--marked-after`: a mark then means "land this now" (top, 12 Oct), and older marks need nothing from their owners."""
    from research.queue import StoreLabels

    head = "0123456789abcdef0123456789abcdef01234567"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane = LocalStore(tmp_path / "lane")
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-12T05:59:59Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q1")).ready(12, head) is True
    assert StoreLabels(LocalStore(tmp_path / "q2"), since="2026-10-12T06:00:00Z").ready(12, head) is False
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-12T06:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q3"), since="2026-10-12T06:00:00Z").ready(12, head) is True
    lane.add_label(label_target(12, head), vocab.READY, False, "memory-accounting", ts="2026-10-12T10:20:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q4")).ready(12, head) is False         # an ejection, seen by every queue
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-12T11:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q5")).ready(12, head) is True          # until its owner marks it again


def test_ready_marks_are_pulled_for_every_pr_at_once_and_each_pr_alone_when_that_fails(tmp_path, monkeypatch):
    """12 Oct: one listing of the store's remote per open PR, in turn, was 105 listings, 10 s of every drive tick."""
    from research.queue import StoreLabels

    heads = {n: f"{n:040x}" for n in (1, 2, 3)}
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane = LocalStore(tmp_path / "lane")
    for n, value in ((1, True), (2, False)):
        lane.add_label(label_target(n, heads[n]), vocab.READY, value, f"lane-{n}", ts="2026-10-12T07:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(n, heads[n]) for n in (1, 2)])["failed"] == []
    pulls: list[tuple[int, int]] = []
    pull = labels_sync.pull_targets
    monkeypatch.setattr(labels_sync, "pull_targets", lambda store, targets, jobs=1: pulls.append((len(targets), jobs))
                        or pull(store, targets, jobs=jobs))
    labels = StoreLabels(LocalStore(tmp_path / "q1"))
    labels.prefetch(list(heads.items()))
    assert [labels.ready(n, heads[n]) for n in (1, 2, 3)] == [True, False, False]
    assert pulls == [(3, labels_sync.SYNC_JOBS)]

    pulls.clear()
    monkeypatch.setattr(labels_sync, "remote_labels_for", lambda r, t, f=labels_sync.remote_labels_for:
                        (_ for _ in ()).throw(OSError("listing refused")) if not pulls[1:] else f(r, t))
    labels = StoreLabels(LocalStore(tmp_path / "q2"))
    labels.prefetch(list(heads.items()))
    assert [labels.ready(n, heads[n]) for n in (1, 2, 3)] == [True, False, False]
    assert pulls == [(3, labels_sync.SYNC_JOBS), (1, 1), (1, 1), (1, 1)]


def test_admission_prefetches_the_marks_of_every_pr_it_will_ask_of(world):
    asked: list[list[tuple[int, str]]] = []
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}), labels=("hold",))
    world.gh.put(3, h3 := world.change({"c.txt": "3\n"}))
    world.queue(prefetch=asked.append).sync()
    assert asked == [[(1, h1), (3, h3)]]


def test_status_on_any_vm_applies_the_rules_cutoff_as_drive_does(world, monkeypatch, capsys):
    """12 Oct: `research queue status` on architecture's VM, run without `--marked-after`, listed week-old marks (#1145, #1736)
    as admitted though drive, with its cutoff, never took them.  The cutoff is `marked_after` in the rules on `main`, and a PR whose
    mark it ignores says so, so its owner re-marks it or closes it (12 Oct: circuits' #1784, #1786, #1801 sat unseen)."""
    import research.queue as Qm

    cfg = world.tmp / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{world.tmp / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    (world.dev / "ci" / "queue.toml").write_text(RULES + 'marked_after = "2026-10-12T06:00:00Z"\n')
    _git(world.dev, "commit", "-qam", "the cutoff")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    lane = LocalStore(world.tmp / "lane")
    for n, ts in ((1, "2026-10-05T09:00:00Z"), (2, "2026-10-12T07:00:00Z")):
        world.gh.put(n, head := world.change({f"{n}.txt": "x\n"}), ready=False)
        lane.add_label(label_target(n, head), vocab.READY, True, f"lane-{n}", ts=ts)
        assert labels_sync.push_targets(lane, [label_target(n, head)])["failed"] == []
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    argv = ["queue", "status", "--json", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin),
            "--repo", "o/verity", "--store", str(world.tmp / "queue-store")]
    assert research_main(argv) == 0
    got = {e["pr"]: (e["state"], e["why"]) for e in json.loads(capsys.readouterr().out)}
    assert got[1][0] == "waiting" and got[1][1].startswith("ignored: marked before cutoff") and "2026-10-05T09:00:00Z" in got[1][1]
    assert got[2][0] == "ready" and "the next sync merges it onto `next`" in got[2][1]
    assert research_main([*argv, "--marked-after", "2026-10-01T00:00:00Z"]) == 0
    assert {e["pr"]: e["state"] for e in json.loads(capsys.readouterr().out)} == {1: "ready", 2: "ready"}


def test_an_ejection_withdraws_the_heads_ready_mark_in_the_store(world):
    """12 Oct: #1824's ejection lived in the ejecting VM's queue state; drive, on another VM, rebuilt `next` with it."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q = world.queue()
    q.eject(1, "held for Daniel's decision 10", "memory-accounting")
    head = next(p.head for p in world.gh.open_prs() if p.number == 1)
    (lab,) = world.store.labels(label_target(1, head), key=vocab.READY)
    assert (lab.value, lab.by, lab.ref) == (False, "memory-accounting", "held for Daniel's decision 10")


def test_a_hand_ejection_is_posted_as_its_ejector_to_ci_and_the_owner_and_says_so_when_no_one_was_told(world, monkeypatch, capsys):
    """13 Oct: `research queue eject 1847` wrote its notice only as a handoff note on the ejecting VM, which syncs no notes, and
    logged that it told circuits, who heard through a relay."""
    import research.drive as D
    import research.queue as Qm

    cfg = world.tmp / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{world.tmp / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setenv("RESEARCH_NOTES", str(world.tmp / "notes"))
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    reg = world.dev / Qm.REGISTRY
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"handles": [{"handle": "ci"}, {"handle": "lane-1"}]}))
    _git(world.dev, "add", "-A")
    _git(world.dev, "commit", "-qm", "the registry")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    posts, ok = [], [True]
    monkeypatch.setattr(D, "merge_thread_post", lambda text, to, sender="ci": posts.append((text, to, sender)) or ok[0])
    argv = ["queue", "eject", "1", "--reason", "breaks suite `verity`", "--by", "ci", "--workdir", str(world.tmp / "q" / "verity"),
            "--remote", str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "queue-store")]
    assert research_main(argv) == 0
    ((text, to, sender),) = posts
    assert (to, sender) == (("@ci", "@lane-1"), "ci")
    assert text.startswith(f"@ci @lane-1 queue: #1 at {head[:8]} is ejected from `next` by ci: breaks suite `verity`")
    ok[0] = False
    assert research_main(argv) == 1
    err = capsys.readouterr().err
    assert "#1 is ejected, but posting its notice as @ci failed, so no one was told" in err
    assert "research msg send --as HANDLE --thread" in err and "--to @ci @lane-1" in err


def test_a_ready_mark_written_after_an_ejection_re_admits_the_same_head(world):
    """12 Oct: #1736, ejected at 2e050d8b and re-marked there by its owner, stayed off `next` on the ejecting VM's queue alone:
    the queue state's record of the ejection held it whatever the store's newest mark said."""
    from research.queue import StoreLabels

    world.with_store_remote()
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    target = label_target(1, head)
    world.store.add_label(target, vocab.READY, True, "lane-1", ts="2026-10-12T21:00:00Z")
    labels = StoreLabels(world.store)
    q = world.queue(ready=labels.ready, marked_at=labels.marked_at)
    assert [m.pr for m in q.sync()[0]] == [1]
    q.eject(1, "breaks the repository suite", "ci")
    members, entries = q.sync()
    assert members == [] and entries[1].why.startswith("ejected at")
    world.store.add_label(target, vocab.READY, True, "lane-1", ts="2999-01-01T00:00:00Z")
    assert [m.pr for m in q.sync()[0]] == [1]

    state = q.load_state()                      # an ejection recorded to the minute, with no withdrawn mark in the store
    state["ejected"]["1"] = {"head": head, "why": "w", "by": "ci", "at": "2999-01-01T00:00Z"}
    q.save_state(state)
    assert q.sync()[0] == [], "a mark from the ejection's own minute may be older than it"
    state["ejected"]["1"]["at"] = "2998-12-31T23:59Z"
    q.save_state(state)
    assert [m.pr for m in q.sync()[0]] == [1]


def test_rules_name_the_hold_label_and_refuse_anything_else():
    assert Rules.parse(RULES).hold_label == "hold" and Rules.parse("").hold_label == "hold"
    for bad in ('[[grant]]\nrole = "red-team"\npaths = ["x/"]\n', 'colour = "red"\n'):
        with pytest.raises(QueueError):
            Rules.parse(bad)


def test_until_the_move_the_rules_are_read_from_tools_check(world):
    world.change({"tools/verity/check/queue.toml": 'hold_label = "parked"\n'})
    _git(world.dev, "rm", "-q", "ci/queue.toml")
    _git(world.dev, "commit", "-q", "-m", "rules where they are before the move")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    world.gh.put(1, world.change({"x.txt": "x\n"}), labels=("parked",))
    world.gh.put(2, world.change({"y.txt": "y\n"}), labels=("hold",))
    _, entries = world.queue().sync()
    assert entries[1].why == "held (`parked` label)" and entries[2].state == "queued"


def test_a_ready_label_holds_for_one_head_and_research_queue_ready_writes_it(world, capsys):
    from research.queue import StoreLabels, ready

    world.with_store_remote()
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    labels = StoreLabels(world.store, log=world.logs.append)
    assert labels.ready(1, h1) is False
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    assert f"#1 is ready at {h1[:12]} (pr:1@{h1})" in capsys.readouterr().out
    assert labels.ready(1, h1) is True
    q = Queue(world.tmp / "q" / "verity", remote=str(world.origin), github=world.gh, ready=labels.ready,
              store=world.store, log=world.logs.append)
    members, _ = q.sync()
    assert [m.pr for m in members] == [1]
    h2 = world.change({"a.txt": "a2\n"})
    world.gh.put(1, h2, ready=False)                           # a new push: the label was for the old head
    members, entries = q.sync()
    assert members == [] and entries[1].why == f"not marked ready at {h2[:8]}: `research queue ready 1 --on MACHINE` (or --local; --mark without the tier)"
    world.gh.put(1, h2, labels=("hold",), ready=False)
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    members, entries = q.sync()
    assert members == [] and entries[1].why == "held (`hold` label)"   # hold wins over ready
    assert ready(world.gh, world.store, 99, "lane-1", PASS) == 2


def test_ready_mark_writes_the_mark_with_no_quick_tier(world, capsys):
    """#1650 lost 5.5 h to a ready post with no mark behind it; `--mark` needs neither a machine nor a tier."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    h = world.change({"m.txt": "m\n"})
    world.gh.put(3, h, ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly()) == 0
    assert "marked on lane-3's word, with no quick tier" in capsys.readouterr().out
    assert StoreLabels(world.store, log=world.logs.append).ready(3, h) is True


def test_a_head_with_the_tree_of_a_head_marked_ready_is_marked_without_running_the_tier_again(world, monkeypatch, capsys):
    """Merging a `main` that changes none of a PR's files (the repository move's merges did this) gave lanes a new head and a
    35-minute tier over the same files.  Only an identical tree, read with labels pulled from the remote, skips the tier."""
    from research import queue as Q

    class Counting:
        def __init__(self):
            self.repo, self.remote, self.ran = world.dev, str(world.origin), []

        def __call__(self, pr: PR) -> None:
            self.ran.append(pr.head)

    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_WRITE_THROUGH", "1")
    tier = Counting()
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and tier.ran == [h1]
    _move_main(world, {"a.txt": "a1\n", "b.txt": "b\n"})              # `main` makes the PR's own change
    world.gh.put(1, h2 := _merge_main(world, h1), ready=False)
    assert h2 != h1 and _git(world.dev, "rev-parse", f"{h2}^{{tree}}") == _git(world.dev, "rev-parse", f"{h1}^{{tree}}")
    capsys.readouterr()
    elsewhere = LocalStore(world.tmp / "elsewhere", config_path=world.tmp / "store.toml")   # h1's mark only on the remote
    assert Q.ready(world.gh, elsewhere, 1, "lane-1", tier) == 0 and tier.ran == [h1]
    assert f"#1's head {h2[:12]} has the tree of {h1[:12]}" in capsys.readouterr().out
    assert Q.StoreLabels(elsewhere).ready(1, h2) is True

    world.gh.put(1, h3 := world.change({"c.txt": "c3\n"}, base=h2), ready=False)
    assert Q.ready(world.gh, elsewhere, 1, "lane-1", tier) == 0 and tier.ran == [h1, h3]

    _move_main(world, {"a.txt": "a1\n", "b.txt": "b4\n"})
    world.gh.put(1, h4 := _merge_main(world, h3), ready=False)        # a merge of `main` that changes a file runs the tier
    assert Q.ready(world.gh, elsewhere, 1, "lane-1", tier) == 0 and tier.ran == [h1, h3, h4]

    h5 = _git(world.dev, "commit-tree", f"{h4}^{{tree}}", "-p", h4, "-m", "h4's tree again")
    _git(world.dev, "push", "-q", "origin", f"{h5}:refs/dev/{h5}")
    world.gh.put(1, h5, ready=False)
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    capsys.readouterr()
    Q.ready(world.gh, LocalStore(world.tmp / "cut-off", config_path=world.tmp / "store.toml"), 1, "lane-1", tier)
    assert tier.ran == [h1, h3, h4, h5]                               # labels it can't pull say nothing: the tier runs
    assert "STALE" in capsys.readouterr().err


#: a `suites.py` that says what it was asked and what it found, and fails one test when FAKE_FAIL names it
FAKE_SUITES = """\
# it knows "--changed"
import json, os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
umask = os.umask(0)
Path(os.environ["FAKE_SEEN"]).write_text(json.dumps({"argv": a, "files": sorted(p.name for p in Path.cwd().iterdir()),
                                                     "umask": umask, "cores": sorted(os.sched_getaffinity(0)),
                                                     "pythonpath": os.environ.get("PYTHONPATH"),
                                                     "built": Path("verity/core/service/flock/.lake/built").is_file(),
                                                     "lake": shutil.which("lake"),
                                                     "env": {k: os.environ.get(k) for k in ("VERITY_STORE_REQUIRED",
                                                             "CHECK_STORE_CUSTODY", "RESEARCH_RUN_ID", "GIT_AUTHOR_EMAIL",
                                                             "PYTHONPYCACHEPREFIX")}}))
fail = os.environ.get("FAKE_FAIL")
Path(a[a.index("--json") + 1]).write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed" if fail else "passed",
                                                       "failed_tests": [fail] if fail else None}]))
sys.exit(1 if fail else 0)
"""


def test_the_quick_tier_runs_the_suites_a_pr_changes_from_a_clean_checkout_at_its_head_and_a_failure_refuses(world, monkeypatch, capsys):
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    seen = world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("PYTHONPATH", str(world.tmp / "the-callers-tree"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert tier(world.gh.prs[1]) is None
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", main] and s["pythonpath"] is None   # from the PR's merge base, in its own tree
    cores = os.sched_getaffinity(0)
    assert s["umask"] == 0o002 and s["cores"] == sorted(cores - {min(cores)} if len(cores) > 1 else cores)   # a node's umask, off core 0
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    assert _git(wt, "rev-parse", "HEAD") == h1
    (wt / "stray.txt").write_text("left by the last PR\n")
    (wt / ".venv").mkdir()

    monkeypatch.setenv("FAKE_FAIL", "tests/test_a.py::test_a")
    assert tier(world.gh.prs[1]) == "pkg: tests/test_a.py::test_a"
    assert {"stray.txt", ".venv"} & set(json.loads(seen.read_text())["files"]) == {".venv"}     # cleaned, its environment kept
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: pkg: tests/test_a.py::test_a" in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False             # no label without a pass

    monkeypatch.delenv("FAKE_FAIL")
    h2 = world.change({"b.txt": "b2\n"})
    world.gh.put(1, h2, ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and _git(wt, "rev-parse", "HEAD") == h2
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is True
    h3 = world.change({"tools/verity/check/suites.py": FAKE_SUITES.replace('"--changed"', "nothing")})
    world.gh.put(1, h3, ready=False)
    assert tier(world.gh.prs[1]) is None
    assert json.loads(seen.read_text())["argv"][:2] == ["--quick", "--json"]     # a runner from before --changed: every suite


def test_the_quick_tier_builds_the_lean_verifier_before_its_tests_and_keeps_that_build_warm(world, monkeypatch):
    """The flock suite's workers each run `lake build` in the verifier; from the cold checkout `git clean` leaves, they collided."""
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "verity/core/service/flock/lakefile.toml": "name = \"flock\"\n"})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    bin_, seen, ran = world.tmp / "bin", world.tmp / "seen.json", world.tmp / "lake.log"
    bin_.mkdir()
    (bin_ / "lake").write_text('#!/bin/sh\necho "$PWD $*" >> "$FAKE_LAKE_LOG"\n[ -z "$FAKE_LAKE_FAIL" ] || exit 1\n'
                               'mkdir -p .lake && echo built > .lake/built\n')
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("FAKE_LAKE_LOG", str(ran))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_LAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert tier(world.gh.prs[1]) is None and json.loads(seen.read_text())["built"]
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    lean = wt / "verity" / "core" / "service" / "flock"
    assert ran.read_text().split("\n") == [f"{lean} build", ""]
    (lean / ".lake" / "built").write_text("from the last PR\n")
    (lean / "stray.olean").write_text("left by the last PR\n")
    monkeypatch.setenv("FAKE_LAKE_FAIL", "1")
    assert tier(world.gh.prs[1]) == "`lake build` in verity/core/service/flock exited 1 (its output is above)"
    assert (lean / ".lake" / "built").read_text() == "from the last PR\n" and not (lean / "stray.olean").exists()


def _move_main(world: World, files: dict[str, str]) -> str:
    moved = world.change(files)
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    return moved


def _merge_main(world: World, head: str) -> str:
    """``head`` with `main` merged into it, on the origin as a PR's head is."""
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", head)
    _git(world.dev, "merge", "-q", "--no-ff", "--no-edit", "origin/main")
    sha = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/dev/{sha}")
    return sha


#: a check slot's `slot.py` as the machine runs it: says how it was asked, then runs CMD
FAKE_SLOT = """\
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
Path(os.environ["FAKE_SLOT_SEEN"]).write_text(json.dumps(a))
cmd = a[a.index("--") + 1:]
os.execvp(cmd[0], cmd)
"""


class Machine:
    """`research run --on` and `research fetch` against a directory, as a machine serves them: the launch ships the `--source`
    checkout's HEAD (a shared clone of its repository, detached there; a dirty checkout is refused), puts each `--send` file in the
    run's inputs and runs the command in the shipped tree with RESEARCH_RUN_DIR and RESEARCH_RUN_ID, then writes status.json; a
    fetch copies the records into the local run dir.  ``outcome(run_dir, run_id, head)``, when given, stands in for the command.
    With ``custody_r2``, the launch stages the run's custody key under `<root>/requests/<run id>/custody/` first, as `--custody-r2`
    (the default with --on) does."""

    def __init__(self, tmp: Path, outcome=None, fetched: dict | None = None, reachable: bool = True, custody_r2: bool = False):
        self.root, self.outcome, self.fetched, self.reachable = tmp / "machine", outcome, fetched or {}, reachable
        self.custody_r2 = custody_r2
        self.launched: list[list[str]] = []

    def launch(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        self.launched.append(argv)
        flags = argv[:argv.index("--")]
        rid, src = flags[flags.index("--id") + 1], Path(flags[flags.index("--source") + 1])
        assert _git(src, "status", "--porcelain") == "", "`research run --on` ships only a clean checkout"
        head = _git(src, "rev-parse", "HEAD")
        tree, run = self.root / "src" / head, self.root / "runs" / rid
        if not tree.exists():
            tree.parent.mkdir(parents=True, exist_ok=True)
            repo = _git(src, "rev-parse", "--path-format=absolute", "--git-common-dir")
            _git(tree.parent, "clone", "-q", "--shared", "--no-checkout", repo, str(tree))
            _git(tree, "checkout", "-q", "--detach", head)
        (run / "inputs").mkdir(parents=True)
        for i, x in enumerate(flags):
            if x == "--send":
                shutil.copy(flags[i + 1], run / "inputs")
        if self.custody_r2:
            (key := self.root / "requests" / rid / "custody").mkdir(parents=True)
            (key / "store.toml").write_text('[remote]\nkind = "s3"\n')
            (key / "cred.json").write_text("{}\n")
        (Path(os.environ["RESEARCH_RUNS"]) / rid).mkdir(parents=True)
        if self.outcome is not None:
            self.outcome(run, rid, head)
        else:
            env = {**os.environ, "RESEARCH_RUN_DIR": str(run), "RESEARCH_RUN_ID": rid}
            rc = subprocess.run(argv[argv.index("--") + 1:], cwd=tree, env=env).returncode
            (run / "status.json").write_text(json.dumps({"state": "done" if rc == 0 else "failed", "rc": rc}))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fetch(self, local: Path) -> dict:
        if not self.reachable:
            raise OSError("injected: the machine is unreachable")
        for f in ("status.json", "result.json"):
            if (self.root / "runs" / local.name / f).exists():
                shutil.copy(self.root / "runs" / local.name / f, local / f)
        return dict(self.fetched)


def _remote_tier(world, monkeypatch, machine: Machine, **kw):
    """A `RemoteQuickTier` on machine `vy-test` for `world`, its clock the time its sleeps took; and
    those sleeps."""
    from research import queue as Q

    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_RUNS", str(world.tmp / "runs"))
    monkeypatch.setenv("FAKE_SLOT_SEEN", str(world.tmp / "slot-seen.json"))
    sleeps: list[float] = []
    return Q.RemoteQuickTier(world.dev, str(world.origin), "vy-test", by="lane-1", root=world.tmp / "ready", sleep=sleeps.append,
                             **{"clock": lambda: sum(sleeps), "launch": machine.launch, "fetch": machine.fetch,
                                **kw}), sleeps


def test_a_remote_quick_tier_runs_in_a_check_slot_on_the_machine_and_is_judged_as_the_local_one_is(world, monkeypatch, capsys):
    """The quick tier ran on the caller's VM, so a lane waited for its own tier and several PRs' tiers ran one at a time; `--on`
    runs it as a run on the machine, in a check slot, from the PR head's shipped tree, and the worktree's lock is free while it runs."""
    import fcntl

    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    seen = world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("PYTHONPATH", str(world.tmp / "the-callers-tree"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    machine = Machine(world.tmp)
    free_while_waiting = []

    def fetch(d: Path) -> dict:
        with open(next((world.tmp / "ready").glob("*.lock")), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)        # raises while the tier holds the worktree
            free_while_waiting.append(d.name)
        return machine.fetch(d)

    tier, sleeps = _remote_tier(world, monkeypatch, machine)
    tier.fetch = fetch
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    [argv] = machine.launched
    flags = argv[:argv.index("--")]
    rid = flags[flags.index("--id") + 1]
    assert flags[:4] == [sys.executable, "-m", "research", "run"] and "--require-result" in flags
    assert [flags[flags.index(k) + 1] for k in ("--on", "--project", "--cwd", "--declared-output")] == ["vy-test", "verity", "source", "out/*"]
    assert float(flags[flags.index("--timeout") + 1]) == Q.REMOTE_WAIT_S + Q.REMOTE_TIMEOUT_S
    ttl = flags[flags.index("--custody-ttl") + 1]
    assert int(ttl) >= Q.REMOTE_WAIT_S + Q.REMOTE_TIMEOUT_S + Q.REMOTE_GRACE_S and _custody_ttl_problem(ttl, flags) is None, \
        "the custody key outlasts the run and the wait for its upload, so `research run` takes the launch"
    assert [Path(flags[i + 1]).name for i, x in enumerate(flags) if x == "--send"] == ["quick_tier.py"]
    assert slot_only.through_slot(argv[argv.index("--") + 1:]), "cpu-1's launcher refuses a run whose command isn't slot.py's"
    assert json.loads((world.tmp / "slot-seen.json").read_text())[:8] == [
        "--by", "lane-1", "--wait-max", f"{Q.REMOTE_WAIT_S:g}", "--run-max", f"{Q.REMOTE_TIMEOUT_S:g}", "--", "sh"]
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", main] and s["pythonpath"] is None
    assert Path(s["argv"][s["argv"].index("--json") + 1]) == machine.root / "runs" / rid / "out" / "suites.json"
    assert free_while_waiting == [rid] and sleeps == [Q.REMOTE_POLL_S]
    assert f"the quick tier runs as {rid} on vy-test, in a check slot" in capsys.readouterr().out
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True

    monkeypatch.setenv("FAKE_FAIL", "tests/test_a.py::test_a")
    h2 = world.change({"b.txt": "b2\n"})
    world.gh.put(1, h2, ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: pkg: tests/test_a.py::test_a." in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is False


def test_a_remote_quick_tier_does_not_wait_for_a_local_tier_of_the_same_repository(world, monkeypatch):
    """A local tier holds its worktree's lock for the whole tier (about an hour on a lane's VM), and a remote tier from any worktree
    of the same repository took that lock to check out and ship, so every `--on` tier on the VM waited for it (two lanes, 4 Oct);
    a remote tier ships from a worktree of its own."""
    import fcntl
    import hashlib

    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(world.tmp / "seen.json"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)

    class NonBlocking:
        LOCK_EX, LOCK_NB = fcntl.LOCK_EX, fcntl.LOCK_NB

        @staticmethod
        def flock(f, op):
            fcntl.flock(f, op | fcntl.LOCK_NB)                  # a held lock raises here instead of hanging the test

    monkeypatch.setattr(Q, "fcntl", NonBlocking)
    common = _git(world.dev, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    (ready := world.tmp / "ready").mkdir()
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp))
    with open(ready / f"{hashlib.sha256(common.encode()).hexdigest()[:12]}.lock", "w") as local:
        fcntl.flock(local, fcntl.LOCK_EX)
        assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True


def _reports(status: dict | None = None, result=lambda doc: doc, raw: str | None = None):
    """A run's outcome on the machine: status.json `status` (done, exit 0, unless given) and the result.json of a pass at the head as
    quick_tier.py writes it, through `result` (None: no result.json), or `raw` as the file."""
    def outcome(run: Path, rid: str, head: str) -> None:
        (run / "status.json").write_text(json.dumps(status or {"state": "done", "rc": 0}))
        facts = {"kind": "quick-tier/v1", "head": head, "base": "0" * 40, "dirty": [], "lean": [], "changed": True, "rc": 0, "rows": [],
                 "logs": "out/logs", "error": None}
        doc = result({"schema": "research/result/v0.1", "kind": "quick-tier/v1", "run_id": rid, "validation": {"status": "passed"},
                      "tier": facts})
        if raw is not None:
            (run / "result.json").write_text(raw)
        elif doc is not None:
            (run / "result.json").write_text(json.dumps(doc))
    return outcome


REMOTE_OUTCOMES = {
    "a pass": ({}, None),
    "no result": ({"result": lambda d: None}, r"ended done \(exit 0\) without a result\.json"),
    "an unreadable result": ({"raw": "{not json"}, r"result\.json is unreadable: not valid JSON"),
    "another run's result": ({"result": lambda d: {**d, "run_id": "r20260101-000000-0000"}},
                             r"result\.json is unreadable: run_id 'r20260101-000000-0000' does not match this run"),
    "another kind": ({"result": lambda d: {**d, "kind": "check/v1"}}, r"result is a check/v1, not a quick-tier/v1 report"),
    "no kind": ({"result": lambda d: {k: v for k, v in d.items() if k != "kind"}}, r"result is a result with no kind, not a quick-tier/v1"),
    "no facts": ({"result": lambda d: {k: v for k, v in d.items() if k != "tier"}}, r"it reported no quick-tier/v1 facts"),
    "another head": ({"result": lambda d: {**d, "tier": {**d["tier"], "head": "f" * 40}}}, r"it ran at ffffffffffff, not at "),
    "a dirty tree": ({"result": lambda d: {**d, "tier": {**d["tier"], "dirty": ["?? stray.txt"]}}},
                     r"the checkout differed from \w{12} before anything ran: \?\? stray\.txt"),
    "a run that failed": ({"status": {"state": "failed", "rc": 1}}, r"facts show a pass, but it ended failed \(exit 1\)"),
    "a failed validation": ({"result": lambda d: {**d, "validation": {"status": "failed"}}}, r"ended done \(exit 0\) with validation failed"),
}


@pytest.mark.parametrize("case", list(REMOTE_OUTCOMES))
def test_a_remote_quick_tier_passes_only_on_a_readable_report_of_a_pass_at_the_head_from_a_run_that_ended_done(world, monkeypatch, capsys, case):
    from research import queue as Q

    kw, refusal = REMOTE_OUTCOMES[case]
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports(**kw)))
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == (0 if refusal is None else 1)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is (refusal is None)
    if refusal is not None:
        assert re.search(r"the quick tier failed: .*" + refusal, capsys.readouterr().err)


def _orphaned(world, monkeypatch, **reports):
    """A `research queue ready 1 --on vy-test` whose caller died while its run waited (a killed shell, a suspended VM): the run on the
    machine ends as ``reports`` says, and nothing marks the PR; the machine, the run's id and the head it was for."""
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports(**reports))
    tier, _ = _remote_tier(world, monkeypatch, machine)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)

    def died(d: Path) -> dict:
        raise KeyboardInterrupt("injected: the caller is gone")

    tier.fetch = died
    with pytest.raises(KeyboardInterrupt):
        Q.ready(world.gh, world.store, 1, "lane-1", tier)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    flags = machine.launched[-1]
    return machine, flags[flags.index("--id") + 1], h1


def test_judge_marks_a_pr_ready_from_its_quick_tiers_run_once_it_ended_without_its_caller(world, monkeypatch, capsys):
    """`ready --on` writes the mark only once its own wait sees the run end, so a tier that passed after its caller died left the PR
    unmarked, and the only way to mark it was to run the whole tier again."""
    from research import queue as Q

    machine, rid, h1 = _orphaned(world, monkeypatch)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 0
    assert f"#1 is ready at {h1[:12]} (pr:1@{h1})" in capsys.readouterr().out
    [label] = world.store.labels(label_target(1, h1), key=vocab.READY)
    assert label.value is True and label.by == "lane-1"
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True


def test_judge_marks_a_pr_whose_head_moved_to_the_same_tree_and_no_other(world, monkeypatch, capsys):
    """A lane that merged `main` (changing no file) while its tier ran had a passing run whose head no open PR has now."""
    from research import queue as Q

    machine, rid, h1 = _orphaned(world, monkeypatch)
    _move_main(world, {"a.txt": "a1\n", "b.txt": "b\n"})              # `main` makes the PR's own change
    world.gh.put(1, h2 := _merge_main(world, h1), ready=False)
    assert h2 != h1
    world.gh.put(2, world.change({"b.txt": "b2\n"}), ready=False)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 1           # no clone to compare trees in
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch, repo=world.dev, url=str(world.origin)) == 0
    assert f"#1's head {h2[:12]} has the tree of {h1[:12]}, where run {rid} ran the quick tier" in capsys.readouterr().out
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is True
    h3 = _git(world.dev, "commit-tree", f"{h2}^{{tree}}", "-p", h2, "-m", "the same tree")
    _git(world.dev, "push", "-q", "origin", f"{h3}:refs/dev/{h3}")
    world.gh.put(3, h3, ready=False)
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch, repo=world.dev, url=str(world.origin)) == 1
    assert "the head of each of #1, #3" in capsys.readouterr().err                          # two PRs with that tree: neither


JUDGE_REFUSALS = {
    "a run that hasn't ended": ({"status": {"state": "running"}}, r"run (\S+) hasn't ended \(last seen running\): `research inspect \1`"),
    "not a quick tier": ({"result": lambda d: {**d, "kind": "check/v1"}}, r"run \S+'s result is a check/v1, not a quick-tier/v1 report"),
    "no result": ({"result": lambda d: None}, r"run \S+ on its machine ended done \(exit 0\) without a result\.json"),
    "a failed tier": ({"status": {"state": "failed", "rc": 1}},
                      r"#1 is not marked ready at \w{12}: the quick tier failed: run \S+'s facts show a pass, but it ended failed"),
    "a head that moved": ({}, r"run \S+ ran the quick tier at \w{12}, which is no open PR's head now, nor its tree \(pushed again "
                              r"since, or closed\?\)"),
    "a run that isn't there": ({}, r"no run matching 'r-nowhere' under "),
}


@pytest.mark.parametrize("case", list(JUDGE_REFUSALS))
def test_judge_refuses_a_run_that_has_not_ended_is_not_a_quick_tier_failed_or_whose_pr_moved(world, monkeypatch, capsys, case):
    from research import queue as Q

    kw, refusal = JUDGE_REFUSALS[case]
    machine, rid, h1 = _orphaned(world, monkeypatch, **kw)
    if case == "a head that moved":
        world.gh.put(1, world.change({"a.txt": "a2\n"}), ready=False)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, "r-nowhere" if case == "a run that isn't there" else rid, "lane-1", fetch=machine.fetch,
                   repo=world.dev, url=str(world.origin)) == 1
    err = capsys.readouterr().err
    assert err.startswith("research queue judge: ") and err.count("\n") == 1 and re.search(refusal, err)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False


def test_a_remote_quick_tier_that_cannot_launch_or_is_never_seen_to_end_fails_after_a_bounded_wait(world, monkeypatch):
    from research import queue as Q

    monkeypatch.setattr(Q, "REMOTE_GRACE_S", 30)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    pr = world.gh.prs[1]
    refused = lambda argv, cwd: subprocess.CompletedProcess(argv, 3, "", "research run: refused: vy-test isn't in machines.toml\n")
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp), launch=refused)
    assert tier(pr) == "`research run --on vy-test` failed (exit 3): research run: refused: vy-test isn't in machines.toml" and sleeps == []
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports({"state": "running"})), timeout_s=60, wait_s=30)
    assert re.fullmatch(r"run r\S+ on vy-test didn't end within 120 s \(last seen running\): `research inspect r\S+`", tier(pr))
    assert sleeps == [Q.REMOTE_POLL_S] * 4, "the run's wait for a slot, its timeout and the grace, at one fetch per poll"
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports(), reachable=False), timeout_s=60, wait_s=30)
    assert re.fullmatch(r"run r\S+ on vy-test didn't end within 120 s \(last seen without a status\): .*", tier(pr)) and len(sleeps) == 4
    exited = Machine(world.tmp, outcome=lambda run, rid, head: None, fetched={"launcher_exited": {"last_line": "Traceback: boom"}})
    tier, sleeps = _remote_tier(world, monkeypatch, exited)
    assert re.fullmatch(r"run r\S+'s runner on vy-test exited before the run ended \(Traceback: boom\)", tier(pr)) and len(sleeps) == 1
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False


def test_a_remote_quick_tier_that_never_got_a_check_slot_never_started_and_did_not_fail(world, monkeypatch, capsys):
    """6 Oct: #1344's and compute-accounting's #1282/#1314 tiers spent their 3 h waiting in node 1's line and were reported failed.
    slot.py gives up at --wait-max with exit 75 before running anything; that is no verdict on the head, and says so."""
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports({"state": "failed", "rc": Q.SLOT_NOT_STARTED}, result=lambda d: None))
    tier, _ = _remote_tier(world, monkeypatch, machine)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == Q.READY_NOT_STARTED
    err = capsys.readouterr().err
    assert re.search(r"#1 is not marked ready at \w{12}: the quick tier never started: run \S+ on vy-test waited for a check slot "
                     r"until its --wait-max and never ran the tier", err) and "the quick tier failed" not in err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    rid = machine.launched[-1][machine.launched[-1].index("--id") + 1]
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 1
    assert "never ran the tier" in capsys.readouterr().err


def test_a_remote_quick_tier_on_auto_runs_on_the_machine_placed_and_without_one_never_starts(world, monkeypatch, capsys):
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports())
    asked = []
    placed = lambda spec, repo: asked.append(spec) or ("vy-test", "a check slot and a Lean check slot are free (check slots: d free)")
    tier, _ = _remote_tier(world, monkeypatch, machine, place=placed)
    tier.on = "auto"
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and asked == ["auto"]
    flags = machine.launched[-1][:machine.launched[-1].index("--")]
    assert flags[flags.index("--on") + 1] == "vy-test"
    assert "research queue: --on auto: vy-test: a check slot and a Lean check slot are free" in capsys.readouterr().out
    tier.place, tier.on = (lambda spec, repo: (None, "no machine's slots could be read")), "vy-a,vy-b"
    world.gh.put(1, h2 := world.change({"a.txt": "a2\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == Q.READY_NOT_STARTED and len(machine.launched) == 1
    assert "the quick tier never started: --on vy-a,vy-b: no machine's slots could be read" in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is False


def test_a_runner_seen_gone_just_after_it_recorded_its_end_is_not_lost(tmp_path):
    """10 Oct: memory-accounting's tier r20261010-134738-7625 passed with rc 0, and the wait reported it lost. A fetch copies
    status.json, then probes the runner's pid; a runner that records its end and exits in between reads as gone with a status
    that isn't terminal. The wait fetches once more before calling it lost, and only a runner still gone then is."""
    from research import queue as Q

    fetched = []

    def fetch(d: Path, ended: bool = True) -> dict:
        fetched.append(d)
        if len(fetched) == 1 or not ended:
            (d / "status.json").write_text(json.dumps({"state": "running", "pid": 7}))
            return {"runner_gone": {"why": "its runner (pid 7) is gone without recording an end"}}
        (d / "status.json").write_text(json.dumps({"state": "done", "rc": 0}))
        return {}

    (d := tmp_path / "runs" / "r1").mkdir(parents=True)
    tier = Q.RemoteQuickTier(tmp_path, "origin", "vy-test", poll_s=30, fetch=fetch, sleep=lambda s: None)
    assert tier._wait(d, "vy-test") == ({"state": "done", "rc": 0}, None) and len(fetched) == 2
    fetched.clear()
    tier.fetch = lambda d: fetch(d, ended=False)
    _, why = tier._wait(d, "vy-test")
    assert why == "run r1 on vy-test ended without recording it: its runner (pid 7) is gone without recording an end"
    assert len(fetched) == 2


def test_a_remote_quick_tiers_wait_is_bounded_in_time_however_long_each_fetch_takes(tmp_path):
    """The wait counted polls as if a fetch took no time, and `remote.fetch` may take its ssh's 900 s each poll: against a machine
    that takes the ssh and then hangs, the caller waited 180 polls of 930 s, about two days, not the run's timeout and the grace
    (red team, #989).  Simulated time only."""
    from research import queue as Q

    clock = [0.0]

    def advance(s: float) -> None:
        clock[0] += s

    def hung_fetch(d: Path) -> dict:
        clock[0] += 900
        return {}

    tier = Q.RemoteQuickTier(tmp_path, "origin", "vy-test", timeout_s=3600, wait_s=1800, poll_s=30, fetch=hung_fetch, sleep=advance,
                             clock=lambda: clock[0])
    (d := tmp_path / "runs" / "r1").mkdir(parents=True)
    _, why = tier._wait(d, "vy-test")
    bound = 1800 + 3600 + Q.REMOTE_GRACE_S
    assert why == f"run r1 on vy-test didn't end within {bound:g} s (last seen without a status): `research inspect r1`"
    assert bound <= clock[0] <= bound + 30 + 900, "it ends with the poll under way at the deadline"


def test_a_remote_quick_tier_gives_its_suites_the_store_as_a_recorded_check_does(world, monkeypatch):
    """On the machine the workload's environment holds no store credential, so `check.py` hands its steps the run's custody key
    (CHECK_STORE_CUSTODY) and VERITY_STORE_REQUIRED=1.  The tier handed neither: C-Flock's store-backed tests skipped as "the
    evidence store is not reachable", and the skip passed (red team, #989).  It hands both as check.py does, never the caller's,
    and nothing of the recording run's own; with no key staged, the store is still required.  The machine's PYTHONPYCACHEPREFIX
    reached the suites, so test_remote_cwd.py's runs wrote their bytecode there instead of under their own root and every PR whose
    tier ran tools/research on vy-nebius-1 failed it (#985, #1021)."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen := world.tmp / "seen.json"))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(world.tmp / "the-machines-pycache"))
    monkeypatch.setenv("CHECK_STORE_CUSTODY", str(world.tmp / "the-callers-custody"))
    monkeypatch.delenv("VERITY_STORE_REQUIRED", raising=False)
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    machine = Machine(world.tmp, custody_r2=True)
    tier, _ = _remote_tier(world, monkeypatch, machine)
    assert tier(world.gh.prs[1]) is None
    flags = machine.launched[-1]
    key = machine.root / "requests" / flags[flags.index("--id") + 1] / "custody"
    env = json.loads(seen.read_text())["env"]
    assert env["VERITY_STORE_REQUIRED"] == "1" and env["CHECK_STORE_CUSTODY"] == str(key.resolve())
    assert env["RESEARCH_RUN_ID"] is None and env["PYTHONPYCACHEPREFIX"] is None and env["GIT_AUTHOR_EMAIL"]
    machine.custody_r2 = False
    assert tier(world.gh.prs[1]) is None
    env = json.loads(seen.read_text())["env"]
    assert env["VERITY_STORE_REQUIRED"] == "1" and env["CHECK_STORE_CUSTODY"] is None


def test_a_quick_tier_with_no_lake_on_the_path_builds_with_elans_and_with_none_refuses(world, monkeypatch):
    """On a machine the run's PATH is a non-login shell's, without elan's bin, which check.py puts first itself.  The tier built
    the verifier only with `lake` on the PATH, so its Lean-dependent tests skipped and the PR was labelled ready (red team, #989).
    It takes `lake` from ~/.elan/bin, first on the suites' PATH too, and with neither refuses."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT, "verity/core/service/flock/lakefile.toml": 'name = "flock"\n'})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen := world.tmp / "seen.json"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp))
    (bare := world.tmp / "no-lake-bin").mkdir()
    for t in ("git", "python3", "sh", "mkdir"):
        (bare / t).symlink_to(shutil.which(t))
    monkeypatch.setenv("PATH", str(bare))
    monkeypatch.setenv("HOME", str(home := world.tmp / "home"))
    lean = "verity/core/service/flock"
    assert tier(world.gh.prs[1]) == f"no `lake` on the PATH or in {home}/.elan/bin to build {lean} with, and without that build its tests skip"
    (elan := home / ".elan" / "bin").mkdir(parents=True)
    (elan / "lake").write_text("#!/bin/sh\nmkdir -p .lake && echo built > .lake/built\n")
    (elan / "lake").chmod(0o755)
    assert tier(world.gh.prs[1]) is None
    s = json.loads(seen.read_text())
    assert s["built"] and s["lake"] == str(elan / "lake")


def test_research_queue_ready_takes_one_of_on_and_local(world, monkeypatch, capsys):
    monkeypatch.chdir(world.dev)
    for argv in (["queue", "ready", "1"], ["queue", "ready", "1", "--on", "vy-test", "--local"]):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and "ready takes one of --on MACHINE" in capsys.readouterr().err
    for argv, why in ((["queue", "judge"], "judge needs the quick tier's run id"),
                      (["queue", "eject", "r20261004-160000-abcd", "--reason", "x"], "eject: 'r20261004-160000-abcd' is not a PR number"),
                      (["queue", "eject", "1", "--reason", "x", "--dry-run"], "--dry-run is for sync, land, drive and report only"),
                      (["queue", "ready", "1", "--local", "--dry-run"], "--dry-run is for sync, land, drive and report only"),
                      (["queue", "drive"], "drive needs --on MACHINE"), (["queue", "report"], "report needs --on MACHINE")):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err


def test_research_queue_ready_and_judge_name_a_lane_never_merge_queue(world, monkeypatch, capsys):
    """Ready is the owner's word, and the merge gate refuses one by `merge-queue`: with no --by and no $RESEARCH_LANE, `ready`
    and `judge` refuse before writing anything; `eject` still defaults to merge-queue."""
    monkeypatch.chdir(world.dev)
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    for argv in (["queue", "ready", "1", "--local"], ["queue", "judge", "r20261004-160000-abcd"]):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and "writes the owner's ready mark: pass --by <lane>" in capsys.readouterr().err


def test_quick_tier_py_refuses_a_call_it_cannot_place(tmp_path, monkeypatch):
    from research import quick_tier

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    for argv in ([], ["--head", "h", "--base", "b"], ["--head", "h", "--"], ["--head", "h", "--base", "b", "--what", "x", "--", "c"]):
        assert quick_tier.main(argv) == 2
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    assert quick_tier.main(["--head", "h", "--base", "b", "--", "c"]) == 2 and not (tmp_path / "result.json").exists()


def test_a_quick_tier_in_a_check_slot_starts_lake_from_the_slots_warm_copy(tmp_path):
    """Each tier's checkout is fresh, so the verifier's `lake build` ran cold in every one and held a node-1 slot for most of a
    small PR's tier (#1298, 6 Oct). In a slot (CHECK_SLOT) it starts from the slot's last passing `.lake`; a warm build that fails
    is retried cold, and outside a slot nothing is kept."""
    from research import quick_tier

    lake = tmp_path / "lake"
    lake.write_text('#!/bin/sh\n[ -f .lake/warm ] && w=warm || w=cold\necho "$w" >> "$LOG"\n'
                    '[ "$w" = warm ] && [ -n "$FAIL_WARM" ] && exit 3\nmkdir -p .lake && echo x > .lake/warm\n')
    lake.chmod(0o755)
    log = tmp_path / "log"
    env = {"HOME": str(tmp_path / "home"), "PATH": os.environ["PATH"], "LOG": str(log), "CHECK_SLOT": "a"}
    d = "verity/core/service/flock"

    def run(n, **extra):
        (pkg := tmp_path / f"t{n}" / d).mkdir(parents=True)
        return quick_tier.build(str(lake), pkg, d, {**env, **extra})

    assert run(1) == {"dir": d, "rc": 0, "start": "cold"}
    warm = tmp_path / "home" / ".cache" / "verity" / "lake-slot-a" / "verity--core--service--flock"
    assert (warm / "warm").is_file()
    assert run(2) == {"dir": d, "rc": 0, "start": "warm"}
    assert run(3, FAIL_WARM="1") == {"dir": d, "rc": 0, "start": "cold"}
    assert log.read_text().split() == ["cold", "warm", "warm", "cold"]
    assert quick_tier.warm_dir({"HOME": env["HOME"]}, d) is None and quick_tier.warm_dir({**env, "CHECK_SLOT": "../x"}, d) is None
    env.pop("CHECK_SLOT")
    assert run(4) == {"dir": d, "rc": 0, "start": "cold"} and log.read_text().split()[-1] == "cold"
