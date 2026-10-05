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

from research.cli import main as research_main
from research.queue import PR, SHADOW_BRANCH, Queue, QueueError, Rules, credential_args, grant_target
from research.store import ids, labels_sync, vocab
from research.store.local import LocalStore
from research.store.model import Label
from test_merge_gate import CHECK_FILES, CHECKED

RULES = """\
hold_label = "hold"

[[grant]]
role = "vllm-coordinator"
paths = ["integrations/vllm/"]
by = "the vLLM coordinator"

[[grant]]
role = "statement-reviewer"
lean_audit = ["guarantees", "reads"]
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

    def with_store_remote(self) -> None:
        """Give `store` a remote (a directory): `StoreGrants` answers only from labels it pulled from one."""
        (self.tmp / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{self.tmp / "vault"}"\n')
        self.store = LocalStore(self.tmp / "store", config_path=self.tmp / "store.toml")

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
    assert entries[8].why == f"not marked ready at {h8[:8]}: `research queue ready 8 --on MACHINE` (or --local)"
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
    assert members == [] and entries[5].why == (
        f"waiting for the vllm-coordinator grant at {parent[:8]}: vllm-coordinator from the vLLM coordinator; granted with "
        f"`research data label pr:5@{parent} grant ROLE --by LANE` (ci/queue.toml)"), "who gives it, and the label that does"
    assert entries[4].state == "waiting" and entries[4].after == (5,)      # it carries #5's change, grant included
    world.grants[(5, parent)] = {"vllm-coordinator"}
    world.grants[(4, world.gh.prs[4].head)] = {"vllm-coordinator"}     # #4 carries #5's change, so it needs the grant too
    members, _ = q.sync()
    assert [m.pr for m in members] == [5, 4]


def test_grants_follow_paths_and_lean_audit_sections_at_the_exact_head(world):
    audit = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    world.gh.put(1, world.change({"integrations/vllm/x.py": "x = 1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": json.dumps({**audit, "guarantees": {"T": "2"}})}))
    world.gh.put(3, world.change({"pkg/lean-audit.json": json.dumps({**audit, "dependencies": {"M": "2"}})}))
    q = world.queue()
    _, entries = q.sync()
    assert entries[1].why.startswith("waiting for the vllm-coordinator grant")
    assert entries[2].why.startswith("waiting for the statement-reviewer grant")
    assert "statement-reviewer (no `by` in ci/queue.toml)" in entries[2].why
    assert entries[3].state == "queued"                         # a dependency digest isn't a statement
    world.grants[(1, world.gh.prs[1].head)] = {"vllm-coordinator"}
    assert 1 in [m.pr for m in q.sync()[0]]
    world.gh.put(1, world.change({"integrations/vllm/x.py": "x = 2\n"}))
    members, entries = q.sync()
    assert 1 not in [m.pr for m in members] and entries[1].state == "waiting"     # a new head needs a new grant


def test_a_file_renamed_out_of_a_grants_paths_needs_the_grant(tmp_path):
    """git reports a rename by its destination alone unless asked not to, which would leave out the grant the file's old path
    needs."""
    (tmp_path / "integrations" / "vllm").mkdir(parents=True)
    (tmp_path / "integrations" / "vllm" / "x.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "mv", "integrations/vllm/x.py", "x.py")
    _git(tmp_path, "commit", "-q", "-m", "move")
    assert Rules.parse(RULES).needs(tmp_path, base, _git(tmp_path, "rev-parse", "HEAD")) == {"vllm-coordinator"}


def test_a_lean_review_covers_only_the_change_it_printed_as_the_same_statements():
    """`reviewed`: an `audit.py --update` report stands in for the statement reviewer only on its own package, only when it printed
    nothing a reviewer must read, and only for entries it saw change: one another PR changed on `main` meanwhile isn't covered."""
    from research.queue import reviewed

    old, new = {"guarantees": {"P.t": "1", "P.u": "5"}, "reads": None}, {"guarantees": {"Q.t": "1q", "P.u": "5"}, "reads": {"Q": "d"}}
    rep = {"package": "lean", "statement_changes": False, "before": {"guarantees": {"P.t": "1", "P.u": "5"}, "reads": {}},
           "guarantees": {"Q.t": "1q", "P.u": "5"}, "reads": {"Q": "d"}}
    assert reviewed("lean/lean-audit.json", old, new, rep)
    assert reviewed("lean/lean-audit.json", old, new, rep | {"package": "/w/verity/lean/"})
    assert not reviewed("other/lean/lean-audit.json", old, new, rep) and not reviewed("lean-audit.json", old, new, rep)
    assert not reviewed("lean/lean-audit.json", old, new, rep | {"statement_changes": True})
    assert not reviewed("lean/lean-audit.json", old, new, {k: v for k, v in rep.items() if k != "before"})
    assert not reviewed("lean/lean-audit.json", {**old, "guarantees": {"P.t": "1", "P.u": "6"}}, new, rep), "P.u changed on main since"
    assert not reviewed("lean/lean-audit.json", old, {**new, "reads": "{not json"}, rep)


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
        return Queue(world.tmp / "q" / "verity", remote=served, github=world.gh, grants=lambda n, h: set(), ready=lambda n, h: True,
                     notes=world.tmp / "notes", log=world.logs.append, auth=auth)

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


def test_grants_read_through_to_the_remote_and_without_that_pull_the_queue_refuses_to_answer(tmp_path, monkeypatch):
    """A failed pull, an unusable remote, or none at all (a cloud VM's shell that lost its store secrets) used to leave the queue
    answering from this machine's labels, with a warning at most, so its ready marks could be stale.  Each is `STALE` now."""
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
    for read in (grants, grants.ready):
        with pytest.raises(QueueError, match=rf"^STALE: the labels of pr:12@{head} weren't pulled from the store's remote "
                                             r"\(pulling from fs:/unreachable failed \(OSError: injected"):
            read(12, head)
    monkeypatch.setattr(LocalStore, "remote", property(real))
    assert grants(12, head) == {"red-team"} and grants.ready(12, head) is True and logs == []
    # the labels are on this machine now, and still don't answer without a pull
    absent, unusable = tmp_path / "absent.toml", tmp_path / "s3.toml"
    unusable.write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                        'access_key_id_env = "Q_AK_ABSENT"\nsecret_access_key_env = "Q_SK_ABSENT"\n')
    with pytest.raises(QueueError, match=r"^STALE: .*\(no remote configured"):
        StoreGrants(LocalStore(tmp_path / "queue", config_path=absent)).ready(12, head)
    with pytest.raises(QueueError, match=r"^STALE: .*Q_AK_ABSENT"):
        StoreGrants(LocalStore(tmp_path / "queue", config_path=unusable)).ready(12, head)
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", "R2_ENDPOINT,R2_BUCKET")
    with pytest.raises(QueueError, match=r"^STALE: .*R2_ENDPOINT, R2_BUCKET not in this shell .*fresh tmux login shell"):
        StoreGrants(LocalStore(tmp_path / "queue", config_path=absent)).ready(12, head)


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

    world.with_store_remote()
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
    assert members == [] and entries[1].why == f"not marked ready at {h2[:8]}: `research queue ready 1 --on MACHINE` (or --local)"
    world.gh.put(1, h2, labels=("hold",), ready=False)
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    members, entries = q.sync()
    assert members == [] and entries[1].why == "held (`hold` label)"   # hold wins over ready
    assert ready(world.gh, world.store, 99, "lane-1", PASS) == 2


#: a `suites.py` that says what it was asked and what it found, and fails one test when FAKE_FAIL names it
FAKE_SUITES = """\
# it knows "--changed"
import json, os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
Path(os.environ["FAKE_SEEN"]).write_text(json.dumps({"argv": a, "files": sorted(p.name for p in Path.cwd().iterdir()),
                                                     "pythonpath": os.environ.get("PYTHONPATH"),
                                                     "built": Path("backends/flock/verifier/lean/.lake/built").is_file(),
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


def test_the_quick_tier_builds_each_lean_package_where_the_trees_layout_has_it(tmp_path):
    """`LEAN_BUILDS` names each package before and after the layout move; a head from either side builds the one it has."""
    from research import queue as Q

    old, new = Q.LEAN_BUILDS[0]
    assert Q.lean_builds(tmp_path) == [old], "neither: the first, so the build refuses as before"
    (tmp_path / new).mkdir(parents=True)
    assert Q.lean_builds(tmp_path) == [new]
    (tmp_path / old).mkdir(parents=True)
    assert Q.lean_builds(tmp_path) == [old]


def test_the_quick_tier_builds_the_lean_verifier_before_its_tests_and_keeps_that_build_warm(world, monkeypatch):
    """The flock suite's workers each run `lake build` in the verifier; from the cold checkout `git clean` leaves, they collided."""
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/check/suites.py": FAKE_SUITES, "backends/flock/verifier/lean/lakefile.toml": "name = \"flock\"\n"})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    bin_, seen, ran = world.tmp / "bin", world.tmp / "seen.json", world.tmp / "lake.log"
    bin_.mkdir()
    (bin_ / "lake").write_text('#!/bin/sh\necho "$PWD $*" >> "$FAKE_LAKE_LOG"\n[ -z "$FAKE_LAKE_FAIL" ] || exit 1\n'
                               'mkdir -p .lake && echo built > .lake/built\n')
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("FAKE_LAKE_LOG", str(ran))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_LAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert tier(world.gh.prs[1]) is None and json.loads(seen.read_text())["built"]
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    lean = wt / "backends" / "flock" / "verifier" / "lean"
    assert ran.read_text().split("\n") == [f"{lean} build", ""]
    (lean / ".lake" / "built").write_text("from the last PR\n")
    (lean / "stray.olean").write_text("left by the last PR\n")
    monkeypatch.setenv("FAKE_LAKE_FAIL", "1")
    assert tier(world.gh.prs[1]) == "`lake build` in backends/flock/verifier/lean exited 1 (its output is above)"
    assert (lean / ".lake" / "built").read_text() == "from the last PR\n" and not (lean / "stray.olean").exists()


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
    """A `RemoteQuickTier` on machine `vy-test` for `world`, its slot.py the fake one and its clock the time its sleeps took; and
    those sleeps."""
    from research import queue as Q

    world.with_store_remote()
    (world.tmp / "bin").mkdir(exist_ok=True)
    (slot := world.tmp / "bin" / "slot.py").write_text(FAKE_SLOT)
    monkeypatch.setenv("RESEARCH_RUNS", str(world.tmp / "runs"))
    monkeypatch.setenv("FAKE_SLOT_SEEN", str(world.tmp / "slot-seen.json"))
    sleeps: list[float] = []
    return Q.RemoteQuickTier(world.dev, str(world.origin), "vy-test", by="lane-1", root=world.tmp / "ready", sleep=sleeps.append,
                             **{"clock": lambda: sum(sleeps), "slot": slot, "launch": machine.launch, "fetch": machine.fetch,
                                **kw}), sleeps


def test_a_remote_quick_tier_runs_in_a_check_slot_on_the_machine_and_is_judged_as_the_local_one_is(world, monkeypatch, capsys):
    """The quick tier ran on the caller's VM, so a lane waited for its own tier and several PRs' tiers ran one at a time; `--on`
    runs it as a run on the machine, in a check slot, from the PR head's shipped tree, and the worktree's lock is free while it runs."""
    import fcntl

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
    assert float(flags[flags.index("--timeout") + 1]) == Q.REMOTE_TIMEOUT_S
    assert sorted(Path(flags[i + 1]).name for i, x in enumerate(flags) if x == "--send") == ["quick_tier.py", "slot.py"]
    assert json.loads((world.tmp / "slot-seen.json").read_text())[:4] == ["--by", "lane-1", "--", "python3"]
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", main] and s["pythonpath"] is None
    assert Path(s["argv"][s["argv"].index("--json") + 1]) == machine.root / "runs" / rid / "out" / "suites.json"
    assert free_while_waiting == [rid] and sleeps == [Q.REMOTE_POLL_S]
    assert f"the quick tier runs as {rid} on vy-test, in a check slot" in capsys.readouterr().out
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is True

    monkeypatch.setenv("FAKE_FAIL", "tests/test_a.py::test_a")
    h2 = world.change({"b.txt": "b2\n"})
    world.gh.put(1, h2, ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: pkg: tests/test_a.py::test_a." in capsys.readouterr().err
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h2) is False


def test_a_remote_quick_tier_does_not_wait_for_a_local_tier_of_the_same_repository(world, monkeypatch):
    """A local tier holds its worktree's lock for the whole tier (about an hour on a lane's VM), and a remote tier from any worktree
    of the same repository took that lock to check out and ship, so every `--on` tier on the VM waited for it (two lanes, 4 Oct);
    a remote tier ships from a worktree of its own."""
    import fcntl
    import hashlib

    from research import queue as Q

    main = world.change({"tools/check/suites.py": FAKE_SUITES})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/check/suites.py", "--quick"])
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
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is True


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
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is (refusal is None)
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
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is False
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
    [label] = world.store.labels(grant_target(1, h1), key=vocab.READY)
    assert label.value is True and label.by == "lane-1"
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is True


JUDGE_REFUSALS = {
    "a run that hasn't ended": ({"status": {"state": "running"}}, r"run (\S+) hasn't ended \(last seen running\): `research inspect \1`"),
    "not a quick tier": ({"result": lambda d: {**d, "kind": "check/v1"}}, r"run \S+'s result is a check/v1, not a quick-tier/v1 report"),
    "no result": ({"result": lambda d: None}, r"run \S+ on its machine ended done \(exit 0\) without a result\.json"),
    "a failed tier": ({"status": {"state": "failed", "rc": 1}},
                      r"#1 is not marked ready at \w{12}: the quick tier failed: run \S+'s facts show a pass, but it ended failed"),
    "a head that moved": ({}, r"run \S+ ran the quick tier at \w{12}, which is no open PR's head now \(pushed again since, or closed\?\)"),
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
    assert Q.judge(world.gh, world.store, "r-nowhere" if case == "a run that isn't there" else rid, "lane-1", fetch=machine.fetch) == 1
    err = capsys.readouterr().err
    assert err.startswith("research queue judge: ") and err.count("\n") == 1 and re.search(refusal, err)
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is False


def test_a_remote_quick_tier_that_cannot_launch_or_is_never_seen_to_end_fails_after_a_bounded_wait(world, monkeypatch):
    from research import queue as Q

    monkeypatch.setattr(Q, "REMOTE_GRACE_S", 30)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    pr = world.gh.prs[1]
    refused = lambda argv, cwd: subprocess.CompletedProcess(argv, 3, "", "research run: refused: vy-test isn't in machines.toml\n")
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp), launch=refused)
    assert tier(pr) == "`research run --on vy-test` failed (exit 3): research run: refused: vy-test isn't in machines.toml" and sleeps == []
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports({"state": "running"})), timeout_s=60)
    assert re.fullmatch(r"run r\S+ on vy-test didn't end within 90 s \(last seen running\): `research inspect r\S+`", tier(pr))
    assert sleeps == [Q.REMOTE_POLL_S] * 3, "the run's timeout and the grace, at one fetch per poll"
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports(), reachable=False), timeout_s=60)
    assert re.fullmatch(r"run r\S+ on vy-test didn't end within 90 s \(last seen without a status\): .*", tier(pr)) and len(sleeps) == 3
    exited = Machine(world.tmp, outcome=lambda run, rid, head: None, fetched={"launcher_exited": {"last_line": "Traceback: boom"}})
    tier, sleeps = _remote_tier(world, monkeypatch, exited)
    assert re.fullmatch(r"run r\S+'s runner on vy-test exited before the run ended \(Traceback: boom\)", tier(pr)) and len(sleeps) == 1
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp), slot=world.tmp / "no-slot.py")
    assert tier(pr).endswith("no-slot.py isn't there to take a check slot on vy-test with (run it from a checkout of the repository)")
    assert Q.StoreGrants(world.store, log=world.logs.append).ready(1, h1) is False


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

    tier = Q.RemoteQuickTier(tmp_path, "origin", "vy-test", timeout_s=3600, poll_s=30, fetch=hung_fetch, sleep=advance,
                             clock=lambda: clock[0])
    (d := tmp_path / "runs" / "r1").mkdir(parents=True)
    _, why = tier._wait(d)
    bound = 3600 + Q.REMOTE_GRACE_S
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

    main = world.change({"tools/check/suites.py": FAKE_SUITES})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/check/suites.py", "--quick"])
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

    main = world.change({"tools/check/suites.py": FAKE_SUITES, "backends/flock/verifier/lean/lakefile.toml": 'name = "flock"\n'})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen := world.tmp / "seen.json"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp))
    (bare := world.tmp / "no-lake-bin").mkdir()
    for t in ("git", "python3", "sh", "mkdir"):
        (bare / t).symlink_to(shutil.which(t))
    monkeypatch.setenv("PATH", str(bare))
    monkeypatch.setenv("HOME", str(home := world.tmp / "home"))
    lean = "backends/flock/verifier/lean"
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
                      (["queue", "eject", "r20261004-160000-abcd", "--reason", "x"], "eject: 'r20261004-160000-abcd' is not a PR number")):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err


def test_quick_tier_py_refuses_a_call_it_cannot_place(tmp_path, monkeypatch):
    from research import quick_tier

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    for argv in ([], ["--head", "h", "--base", "b"], ["--head", "h", "--"], ["--head", "h", "--base", "b", "--what", "x", "--", "c"]):
        assert quick_tier.main(argv) == 2
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    assert quick_tier.main(["--head", "h", "--base", "b", "--", "c"]) == 2 and not (tmp_path / "result.json").exists()
