"""`research queue`: `next` is `main` plus the admitted PRs, one merge commit each; `main` fast-forwards to the newest commit on it
with a passing `check` of that exact commit.  Every test runs on a throwaway bare origin, with a fake GitHub that serves the open
PRs and writes their `refs/pull/{n}/head`."""
from __future__ import annotations

import base64
import json
import os
import shlex
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

RULES = """\
hold_label = "hold"

[[grant]]
role = "vllm-coordinator"
paths = ["integrations/vllm/"]
by = "the vLLM coordinator"

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
    world.gh.put(2, world.change({"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "2"}})}))
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


def test_a_lean_review_covers_only_the_change_it_printed_as_the_same_statements():
    """`reviewed`: an `audit.py --update` report stands in for the statement reviewer only on its own package, only when it printed
    nothing a reviewer must read, and only for entries it saw change: one another PR changed on `main` meanwhile isn't covered."""
    from research.queue import reviewed

    old, new = {"pins": {"P.t": "1", "P.u": "5"}, "reads": None}, {"pins": {"Q.t": "1q", "P.u": "5"}, "reads": {"Q": "d"}}
    rep = {"package": "lean", "statement_changes": False, "before": {"pins": {"P.t": "1", "P.u": "5"}, "reads": {}},
           "pins": {"Q.t": "1q", "P.u": "5"}, "reads": {"Q": "d"}}
    assert reviewed("lean/lean-audit.json", old, new, rep)
    assert reviewed("lean/lean-audit.json", old, new, rep | {"package": "/w/verity/lean/"})
    assert not reviewed("other/lean/lean-audit.json", old, new, rep) and not reviewed("lean-audit.json", old, new, rep)
    assert not reviewed("lean/lean-audit.json", old, new, rep | {"statement_changes": True})
    assert not reviewed("lean/lean-audit.json", old, new, {k: v for k, v in rep.items() if k != "before"})
    assert not reviewed("lean/lean-audit.json", {**old, "pins": {"P.t": "1", "P.u": "6"}}, new, rep), "P.u changed on main since"
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
                                                     "pythonpath": os.environ.get("PYTHONPATH"),
                                                     "built": Path("backends/flock/verifier/lean/.lake/built").is_file()}))
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


RED_TEAM = '\n[[grant]]\nrole = "red-team"\npaths = ["backends/flock/", "!README.md"]\n'


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


def _carrying(world: World, rules: str | None = None) -> Queue:
    """A queue whose grants come from `world.store`'s remote, carried from a PR's other heads (`StoreGrants.carried`)."""
    from research.queue import StoreGrants

    world.with_store_remote()
    workdir = world.tmp / "q" / "verity"
    labels = StoreGrants(world.store, log=world.logs.append, repo=workdir)
    return Queue(workdir, remote=str(world.origin), github=world.gh, grants=labels, ready=lambda n, h: (n, h) in world.gh.ready,
                 store=world.store, notes=world.tmp / "notes", log=world.logs.append, reviews=labels.reviews, carry=labels.carried,
                 rules=Rules.parse(rules) if rules else None)


def _grant(world: World, pr: int, head: str, role: str) -> None:
    """``role``'s grant on `pr:{pr}@{head}`, written in a reviewer's own store and pushed to the remote."""
    reviewer = LocalStore(world.tmp / "reviewer", config_path=world.tmp / "store.toml")
    reviewer.add_label(grant_target(pr, head), vocab.GRANT, role, "red-team-x", ts="2026-10-03T16:00:00Z")
    assert labels_sync.push_targets(reviewer, [grant_target(pr, head)])["failed"] == []


def test_a_grant_carries_to_a_restack_with_a_byte_identical_lean_audit_change_and_the_queue_names_the_head_it_came_from(world):
    """Daniel's ruling (2 Oct): a statement review carries across a restack when the new head's `lean-audit.json` change is
    byte-identical to the one granted.  The clone holds the old head from the sync that admitted it; the grant was written on
    another machine and comes through the store's remote, pulled for every head of the PR at once."""
    q = _carrying(world)
    audit = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    pin = {"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "2"}})}
    world.gh.put(1, old := world.change(pin))
    _grant(world, 1, old, "statement-reviewer")
    assert [m.pr for m in q.sync()[0]] == [1]
    _move_main(world, {"d.txt": "landed meanwhile\n"})
    world.gh.put(1, new := world.change(pin))                   # the same change on the new `main`
    members, _ = q.sync()
    assert [(m.pr, m.head) for m in members] == [(1, new)]
    assert (f"research queue: grant statement-reviewer carried from pr:1@{old} to {new[:12]} "
            f"(restack; lean-audit.json change identical)") in world.logs


def test_a_grant_carries_to_a_merge_of_main_and_not_to_any_other_lean_audit_change(world):
    """A merge of `main` carries; a restated pin doesn't, and neither does the same pin change restacked onto a `main` that changed
    the same `lean-audit.json` meanwhile, since its change now starts from other bytes (fail closed: a new grant, or a lean_review)."""
    q = _carrying(world)
    audit = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    world.gh.put(1, old := world.change({"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "2"}})}))
    _grant(world, 1, old, "statement-reviewer")
    q.sync()
    _move_main(world, {"d.txt": "landed meanwhile\n"})
    world.gh.put(1, merged := _merge_main(world, old))
    assert [m.head for m in q.sync()[0]] == [merged]
    assert (f"research queue: grant statement-reviewer carried from pr:1@{old} to {merged[:12]} "
            f"(merge of main; lean-audit.json change identical)") in world.logs

    def refused(h: str) -> None:
        world.gh.put(1, h)
        members, entries = q.sync()
        assert members == [] and entries[1].why.startswith(f"waiting for the statement-reviewer grant at {h[:8]}")
        assert (f"research queue: grant statement-reviewer on pr:1@{old} doesn't carry to {h[:12]}: its lean-audit.json change "
                f"isn't byte-identical (pkg/lean-audit.json)") in world.logs

    refused(world.change({"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "3"}})}))
    _move_main(world, {"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "1", "U": "1"}})})
    refused(world.change({"pkg/lean-audit.json": json.dumps({**audit, "pins": {"T": "2", "U": "1"}})}))


def test_a_red_team_grant_on_flock_follows_the_same_restack_rule(world):
    """A path grant carries the same way: to a restack of the change it saw, never to one that changes another file too, a
    `lean-audit.json` the granted head didn't change included."""
    q = _carrying(world, RULES + RED_TEAM)
    audit = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    flock = {"backends/flock/x.py": "x = 1\n"}
    world.gh.put(1, old := world.change(flock))
    _grant(world, 1, old, "red-team")
    assert [m.pr for m in q.sync()[0]] == [1]
    _move_main(world, {"d.txt": "landed meanwhile\n"})
    world.gh.put(1, new := world.change(flock))
    assert [m.head for m in q.sync()[0]] == [new]
    assert f"research queue: grant red-team carried from pr:1@{old} to {new[:12]} (restack; no lean-audit.json change)" in world.logs
    for extra, why in (({"backends/flock/y.py": "y = 1\n"}, "its change against origin/main differs (backends/flock/y.py)"),
                       ({"a.txt": "a, too\n"}, "its change against origin/main differs (a.txt)"),
                       ({"pkg/lean-audit.json": json.dumps({**audit, "dependencies": {"M": "2"}})},
                        "its lean-audit.json change isn't byte-identical (pkg/lean-audit.json)")):
        world.gh.put(1, h := world.change({**flock, **extra}))
        members, entries = q.sync()
        assert members == [] and entries[1].why.startswith(f"waiting for the red-team grant at {h[:8]}")
        assert f"research queue: grant red-team on pr:1@{old} doesn't carry to {h[:12]}: {why}" in world.logs


def test_a_grant_on_a_head_git_cannot_compare_carries_nothing(world):
    """Fail closed: a head the clone never fetched, one that exists nowhere, and one with no history in common with `main` (same
    tree, so only the missing merge base refuses it) give the new head nothing."""
    from research.queue import carry

    q = _carrying(world, RULES + RED_TEAM)
    q.sync()                                                    # the clone first: a local clone hardlinks every object there is
    flock = {"backends/flock/x.py": "x = 1\n"}
    unseen = world.change(flock)                                # on the origin, but never a PR head the queue fetched
    ghost = "0123456789abcdef" * 2 + "01234567"
    for h in (unseen, ghost):
        _grant(world, 1, h, "red-team")
    _move_main(world, {"d.txt": "landed meanwhile\n"})
    world.gh.put(1, new := world.change(flock))
    members, entries = q.sync()
    assert members == [] and entries[1].why.startswith(f"waiting for the red-team grant at {new[:8]}")
    for h in (unseen, ghost):
        assert (f"research queue: grant red-team on pr:1@{h} doesn't carry to {new[:12]}: {h[:12]} isn't in {q.workdir}, so its "
                f"change can't be compared (fetch it)") in world.logs
    orphan = _git(world.dev, "commit-tree", f"{new}^{{tree}}", "-m", "the same files, no history")
    assert carry(world.dev, "origin/main", orphan, new) == (False, f"{orphan[:12]} shares no history with origin/main")
    assert carry(world.dev, "origin/main", unseen, new) == (True, "restack; no lean-audit.json change")   # where git can compare


def test_without_a_pull_of_every_head_of_the_pr_nothing_carries_and_the_queue_says_stale(tmp_path, monkeypatch):
    from research.queue import StoreGrants

    head = "0123456789abcdef0123456789abcdef01234567"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    grants = StoreGrants(LocalStore(tmp_path / "queue", config_path=cfg), repo=tmp_path)
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    assert grants.carried(12, head, set()) == set()             # nothing to carry, nothing pulled
    with pytest.raises(QueueError, match=r"^STALE: the labels of pr:12@\* weren't pulled from the store's remote \(OSError: injected.*"
                                         r"not carrying any$"):
        grants.carried(12, head, {"red-team"})
