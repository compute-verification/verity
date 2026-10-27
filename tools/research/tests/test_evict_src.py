"""`research data evict --src-dir`: the hourly eviction of `research run`'s shipped source trees (`evict.evict_src`).  Node 1 hit
74% inodes on 4 Oct with nothing pruning `/workspace/research/src`, and @infra pruned it by hand under root's three conditions:
the tree is recoverable on the node, nothing uses it, and no run record holds it as its source's only copy.  Each condition keeps a
tree here, a pass that can't establish one deletes nothing, and a tree meeting all of them goes."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.store import evict as EV

NOW = 2_000_000_000.0
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=ENV).stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / "f").write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", text)
    return _git(repo, "rev-parse", "HEAD")


def _age(path: Path, hours: float) -> None:
    when = NOW - hours * 3600
    for p in sorted(path.rglob("*"), reverse=True) + [path]:
        os.utime(p, (when, when), follow_symlinks=False)


def tree(src: Path, sha: str, hours: float = 3.0, ready: str | None = "self") -> Path:
    """What a git ship makes: a checkout of the node's repository detached at `sha` with READY.json beside it, or for a commit the
    repository lacks, a plain directory."""
    t, gd = src / sha, src.parent / "git" / "verity.git"
    if subprocess.run(["git", "--git-dir", str(gd), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0:
        _git(src.parent, "clone", "-q", "--shared", "--no-checkout", str(gd), str(t))
        _git(t, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha)
    else:
        t.mkdir(parents=True)
        (t / "f").write_text(sha)
    if ready:
        (t / EV.READY_FILE).write_text(json.dumps({"schema": "research/source-ready/v0.1", "identity": sha if ready == "self" else ready}))
    _age(t, hours)
    return t


def fake_proc(proc: Path, pid: int, *, cwd: str = "/", cmdline: str = "python3", environ: str = "", maps: str = "",
              fds: tuple[str, ...] = (), kthread: bool = False) -> None:
    d = proc / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "stat").write_text(f"{pid} (p {pid}) S 0 1 1 0 -1 {EV.PF_KTHREAD if kthread else 0x400100} 0 0\n")
    (d / "comm").write_text(f"p{pid}\n")
    os.symlink(cwd, d / "cwd")
    os.symlink("/", d / "root")
    os.symlink("/usr/bin/python3", d / "exe")
    for i, f in enumerate(fds):
        os.symlink(f, d / "fd" / str(i))
    (d / "maps").write_text(maps)
    (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    (d / "environ").write_text(environ)


@pytest.fixture
def node(tmp_path):
    """A node's research root: trees of a (on origin through main's history), b (main's tip) and c (a branch origin never saw),
    oldest first; origin a bare repository; the node's bare repository holding each as refs/research/src/<sha>; a /proc of an
    init process and a kernel thread."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    a, b = _commit(work, "a"), _commit(work, "b")
    _git(work, "checkout", "-q", "-b", "local")
    c = _commit(work, "c")
    origin, root = tmp_path / "origin.git", tmp_path / "research"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(work, "push", "-q", str(origin), "main")
    gd = root / "git" / "verity.git"
    gd.parent.mkdir(parents=True)
    _git(tmp_path, "init", "-q", "--bare", str(gd))
    for s in (a, b, c):
        _git(work, "push", "-q", str(gd), f"{s}:refs/research/src/{s}")
    src, proc = root / "src", tmp_path / "proc"
    fake_proc(proc, 1)
    fake_proc(proc, 2, kthread=True, cwd=f"{src}/{a}")
    for s, h in ((a, 5), (b, 4), (c, 3)):
        tree(src, s, h)
    return SimpleNamespace(root=root, src=src, origin=str(origin), proc=proc, a=a, b=b, c=c, tmp=tmp_path)


def run(n, **kw) -> dict:
    kw = {"origin": n.origin, "now": NOW, "proc": n.proc, "jobs_dirs": [], "clock": lambda: 0.0, **kw}
    return EV.evict_src(n.src, **kw)


def kept(rep: dict) -> dict[str, list[str]]:
    return {k["tree"]: k["why"] for k in rep["kept"]}


def write_run(n, name: str, *, state: str | None = "done", source: str | None = None, hours: float = 3.0, store: dict | None = None,
              root: str = "runs", tree_sha: str | None = None) -> Path:
    rd = n.root / root / name
    rd.mkdir(parents=True)
    sha = tree_sha or n.a
    job = {"run": {"id": name, "remote": {"source_sha": source, "source_dir": f"{n.src}/{sha}"}}}
    if store is not None:
        job["store"] = {"conditions": {"source": store}}
    (rd / "job.json").write_text(json.dumps(job))
    if state:
        (rd / "status.json").write_text(json.dumps({"state": state, "transitions": [{"state": state}]}))
    _age(rd, hours)
    return rd


def _usage(t: Path) -> tuple[int, int]:
    files = [p for p in t.rglob("*") if not p.is_dir() or p.is_symlink()]
    return len(list(t.rglob("*"))) + 1, sum(p.lstat().st_size for p in files)


def test_trees_meeting_every_condition_go_oldest_first_and_the_rest_are_kept_with_reasons(node):
    usage = {s: _usage(node.src / s) for s in (node.a, node.b)}
    rep = run(node)
    assert rep["error"] is None and rep["trees"] == 3
    assert [d["tree"] for d in rep["deleted"]] == [node.a, node.b], "oldest first; a is on origin through main's history"
    assert [(d["inodes"], d["bytes"]) for d in rep["deleted"]] == [usage[node.a], usage[node.b]], "every file, directory and the tree"
    assert rep["deleted_inodes"] == usage[node.a][0] + usage[node.b][0]
    assert kept(rep) == {node.c: [f"1: not recoverable on the node: on no ref of {node.origin}"]} and rep["kept_by"] == {"1": 1}
    assert sorted(set(os.listdir(node.src)) - {EV.LOCK}) == [node.c], "nothing left aside"
    assert (node.src / node.c / EV.READY_FILE).exists()


def test_a_run_that_ended_and_a_slot_holder_that_is_gone_do_not_keep_a_tree(node):
    write_run(node, "r1", source=node.a)
    (node.root / "locks").mkdir()
    (node.root / "locks" / "check-a.who").write_text(json.dumps({"pid": 4242, "run": "r1", "tree": f"{node.src}/{node.a}"}) + "\n")
    rep = run(node)
    assert node.a in [d["tree"] for d in rep["deleted"]]


def _cwd(n):
    fake_proc(n.proc, 100, cwd=f"{n.src}/{n.a}/tools")


def _environ(n):
    fake_proc(n.proc, 101, environ=f"PYTHONPATH={n.src}/{n.a}/tools/research/src\0HOME=/home/research\0")


def _open_file(n):
    fake_proc(n.proc, 102, fds=(f"{n.src}/{n.a}/README.md",))


def _map(n):
    fake_proc(n.proc, 103, maps=f"7f0000000000-7f0000001000 r-xp 00000000 00:00 0 {n.src}/{n.a}/lib.so\n")


def _cmdline(n):
    fake_proc(n.proc, 104, cmdline=f"python3 {n.src}/{n.a}/tools/verity/check/check.py")


def _running(n):
    write_run(n, "r2", state="running", source=n.a)


def _no_status(n):
    write_run(n, "r3", state=None, source=n.a)


def _aside_run(n):
    write_run(n, "r4", state="running", source=n.a, root="runs-aside")


def _queued(n):
    (n.root / "requests" / "r5").mkdir(parents=True)
    (n.root / "requests" / "r5" / "request.json").write_text(json.dumps({"cwd": f"{n.src}/{n.a}"}))


def _request_running(n):
    _queued(n)
    (n.root / "runs" / "r5").mkdir(parents=True)
    (n.root / "runs" / "r5" / "status.json").write_text(json.dumps({"state": "running"}))


def _holder(n):
    (n.root / "locks").mkdir()
    (n.root / "locks" / "check-a.who").write_text(json.dumps({"pid": 1, "run": "r6", "tree": f"{n.src}/{n.a}", "by": "proofs"}) + "\n")


def _lean_holder(n):
    (n.root / "locks").mkdir()
    (n.root / "locks" / "lean-check-0.who").write_text(json.dumps({"pid": 1, "run": None, "tree": f"{n.src}/{n.a}", "took": "x"}))


def _waiter(n):
    (n.root / "locks" / "check-line").mkdir(parents=True)
    (n.root / "locks" / "check-line" / "0179-1").write_text(json.dumps({"pid": 1, "run": "r7", "tree": f"{n.src}/{n.a}"}))


def _recent_change(n):
    (n.src / n.a / "NEW").write_text("shipped just now")
    os.utime(n.src / n.a / "NEW", (NOW - 600, NOW - 600))
    os.utime(n.src / n.a, (NOW - 600, NOW - 600))


def _recent_run(n):
    write_run(n, "r8", source=n.a, hours=0.5)


@pytest.mark.parametrize("setup, why", [
    (_cwd, "2: process 100 (p100) uses it"), (_environ, "2: process 101"), (_open_file, "2: process 102"), (_map, "2: process 103"),
    (_cmdline, "2: process 104"), (_running, "2: run r2 is running"), (_no_status, "2: run r3 is without a status"),
    (_aside_run, "2: run r4 is running"), (_queued, "2: request r5 has no run dir yet (queued)"),
    (_request_running, "2: request r5's run is running"), (_holder, "2: slot holder check-a.who (pid 1)"),
    (_lean_holder, "2: slot holder lean-check-0.who"), (_waiter, "2: slot waiter check-line/0179-1"),
    (_recent_change, "recent: changed 10 min ago"), (_recent_run, "recent: run r8 was recorded within 1 h"),
], ids=lambda x: x.__name__.strip("_") if callable(x) else None)
def test_a_tree_in_use_is_kept_and_the_pass_goes_on(node, setup, why):
    setup(node)
    rep = run(node)
    assert rep["error"] is None
    assert any(w.startswith(why) for w in kept(rep)[node.a]), kept(rep)[node.a]
    assert (node.src / node.a / EV.READY_FILE).exists() and [d["tree"] for d in rep["deleted"]] == [node.b]


def test_a_job_queued_for_the_fill_that_names_a_tree_keeps_it(node):
    q = node.tmp / "fill" / "queue"
    q.mkdir(parents=True)
    (q / "job-1.sh").write_text(f"cd /x/research/src/{node.a} && python3 run.py\n")
    rep = run(node, jobs_dirs=[q, node.tmp / "fill" / "running"])
    assert any(w.startswith("2: ") and "job-1.sh names it" in w for w in kept(rep)[node.a])


def _no_ready(n):
    (n.src / n.a / EV.READY_FILE).unlink()
    _age(n.src / n.a, 5)


def _foreign_ready(n):
    (n.src / n.a / EV.READY_FILE).write_text(json.dumps({"identity": n.c}))
    _age(n.src / n.a, 5)


def _other_source(n):
    write_run(n, "r9", source=n.c)


def _other_commit(n):
    write_run(n, "r10", source=n.a, store={"commit": n.c, "dirty": False})


def _dirty(n):
    write_run(n, "r11", source=n.a, store={"commit": n.a, "dirty": True})


def _no_source(n):
    write_run(n, "r12", source=None)


def _bad_job(n):
    rd = write_run(n, "r13", source=n.a)
    (rd / "job.json").write_text(f'{{"cwd": "{n.src}/{n.a}", ')
    _age(rd, 3)


@pytest.mark.parametrize("setup, why", [
    (_no_ready, "3: no READY.json"), (_foreign_ready, "3: READY.json names"), (_other_source, "3: run r9 names it, and its source is"),
    (_other_commit, "3: run r10 names it, and its source is"), (_dirty, "3: run r11's source is dirty"),
    (_no_source, "3: run r12 names it and records no source commit"), (_bad_job, "3: run r13's job.json does not parse"),
], ids=lambda x: x.__name__.strip("_") if callable(x) else None)
def test_a_tree_that_may_be_a_runs_only_copy_of_its_source_is_kept(node, setup, why):
    setup(node)
    rep = run(node)
    assert rep["error"] is None and any(w.startswith(why) for w in kept(rep)[node.a]), kept(rep)[node.a]
    assert (node.src / node.a).is_dir()


def test_a_run_refused_before_its_workload_does_not_keep_the_tree_once_it_is_old(node):
    """On node 2 (16 Oct), 43 trees and 1.15M inodes were kept only by refused runs (launch.json, refused.json, no job.json), as
    "records no source commit": a refused run ran nothing, so the tree can't be the only copy of what it ran."""
    for name, hours in (("r15", 3.0), ("r16", 0.5)):
        rd = node.root / "runs" / name
        rd.mkdir(parents=True)
        (rd / "launch.json").write_text(json.dumps({"run_id": name, "cwd": f"{node.src}/{node.a if name == 'r15' else node.b}"}))
        (rd / "refused.json").write_text(json.dumps({"run_id": name, "reason": "memory"}))
        (rd / "job.json").write_text(json.dumps({"run": {"id": name}}))
        _age(rd, hours)
    rep = run(node)
    assert rep["error"] is None and node.a in [d["tree"] for d in rep["deleted"]], kept(rep)
    assert kept(rep)[node.b] == ["recent: run r16 was recorded within 1 h"], "a refused run still counts as recent"


def _captured(n, *, marker: dict | None, digest: str = "d" * 64, cap_digest: str | None = None) -> None:
    rd = write_run(n, "r14", source=n.a, store={"commit": n.a, "dirty": True, "dirty_digest": digest})
    job = json.loads((rd / "job.json").read_text())
    job["store"]["dirty_capture"] = {"path": "source-dirty.tar", "bytes": 10, "sha256": "e" * 64, "dirty_digest": cap_digest or digest}
    (rd / "job.json").write_text(json.dumps(job))
    if marker is not None:
        (rd / ".custody").write_text(json.dumps(marker))
    _age(rd, 3)


PRESERVED = {"preserved": True, "files": {"job.json": [1, "f" * 64], "source-dirty.tar": [10, "e" * 64]}}


def test_a_dirty_run_whose_preserved_record_holds_its_capture_no_longer_keeps_the_tree(node):
    """The 10 Oct src sweep deleted trees that were seven runs' only copy of their dirty sources; the runner now captures a dirty
    source into the run dir and custody uploads it, so once the record is preserved the tree is not the only copy."""
    _captured(node, marker=PRESERVED)
    rep = run(node)
    assert rep["error"] is None and node.a in [d["tree"] for d in rep["deleted"]], kept(rep)


@pytest.mark.parametrize("marker, digest, cap_digest", [
    (None, "d" * 64, None),
    ({**PRESERVED, "preserved": False}, "d" * 64, None),
    ({"preserved": True, "files": {"source-dirty.tar": [10, "0" * 64]}}, "d" * 64, None),
    ({"preserved": True, "files": {"job.json": [1, "f" * 64]}}, "d" * 64, None),
    (PRESERVED, "too-large", None),
    (PRESERVED, "d" * 64, "a" * 64),
], ids=["no-custody", "not-preserved", "other-bytes", "not-in-record", "too-large", "other-digest"])
def test_a_dirty_run_keeps_the_tree_unless_its_preserved_record_holds_that_capture(node, marker, digest, cap_digest):
    _captured(node, marker=marker, digest=digest, cap_digest=cap_digest)
    rep = run(node)
    assert any(w.startswith("3: run r14's source is dirty, and no preserved record") for w in kept(rep).get(node.a, [])), kept(rep)
    assert (node.src / node.a).is_dir()


def test_a_research_keep_record_leaves_the_tree_to_retention(node):
    from research import retention as RT
    RT.keep(node.src / node.a, owner="@infra", expires=None, priority="keep", root=node.root.parent, why="pinned")
    _age(node.src / node.a, 5)
    rep = run(node)
    assert kept(rep)[node.a] == ["record: a `research keep` record covers it; `research retention` decides"]


def _all_kept(node, rep: dict, error: str) -> None:
    assert rep["error"] and rep["error"].startswith(error), rep["error"]
    assert rep["deleted"] == [] and sorted(kept(rep)) == sorted([node.a, node.b, node.c])
    assert all(any(w.startswith("unestablished: ") for w in why) for why in kept(rep).values())
    assert all((node.src / s / EV.READY_FILE).exists() for s in (node.a, node.b, node.c))


def test_an_origin_that_cannot_be_listed_deletes_nothing(node):
    _all_kept(node, run(node, origin=str(node.tmp / "nowhere.git")), "1: git ls-remote")


def test_an_origin_with_no_refs_deletes_nothing(node):
    empty = node.tmp / "empty.git"
    _git(node.tmp, "init", "-q", "--bare", str(empty))
    _all_kept(node, run(node, origin=str(empty)), "1: ")


def test_without_the_nodes_repository_an_ancestor_cannot_be_placed_and_nothing_goes(node):
    import shutil
    shutil.rmtree(node.root / "git")
    _all_kept(node, run(node), "1: git cat-file in")


def test_an_origin_url_s_credential_never_reaches_the_report(node):
    rep = run(node, origin="https://x-access-token:" "dummy@127.0.0.1:1/none.git")
    assert "dummy" not in json.dumps(rep) and "//***@127.0.0.1:1" in rep["origin"]


def test_a_process_it_cannot_read_deletes_nothing(node, monkeypatch):
    real = os.readlink

    def readlink(p, *a, **k):
        if str(p) == str(node.proc / "1" / "cwd"):
            raise PermissionError(13, "Permission denied")
        return real(p, *a, **k)

    monkeypatch.setattr(EV.os, "readlink", readlink)
    _all_kept(node, run(node), "2: ")
    assert "CAP_SYS_PTRACE" in run(node)["error"]


RUNPOD_NGINX = """events { worker_connections 2048; }
http {
    server { listen 9091; location / { proxy_pass http://localhost:9090; } error_page 502 /502.html;
             location = /502.html { root /usr/share/nginx/html; } }
}
"""


def _runpod_proxy(node, monkeypatch, *, conf: str = RUNPOD_NGINX, uid: int = 65534, master: str = "nginx: master process /usr/sbin/nginx",
                  name: str = "nginx") -> Path:
    """RunPod's proxy: a root nginx master (39) the pass reads and its worker (40) as `nobody`, which it can't."""
    fake_proc(node.proc, 39, cmdline=master)
    os.remove(node.proc / "39" / "exe")
    os.symlink("/usr/sbin/nginx", node.proc / "39" / "exe")
    fake_proc(node.proc, 40, cmdline="nginx: worker process")
    (node.proc / "40" / "status").write_text(f"Name:\t{name}\nPPid:\t39\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    real = os.readlink

    def readlink(p, *a, **k):
        if str(p).startswith(str(node.proc / "40") + "/"):
            raise PermissionError(13, "Permission denied")
        return real(p, *a, **k)

    monkeypatch.setattr(EV.os, "readlink", readlink)
    path = node.tmp / "nginx.conf"
    path.write_text(conf)
    return path


def test_runpods_proxy_worker_is_passed_over_unread_and_its_master_is_read(node, monkeypatch):
    rep = run(node, nginx_conf=_runpod_proxy(node, monkeypatch))
    assert rep["error"] is None and [d["tree"] for d in rep["deleted"]] == [node.a, node.b], rep
    tree(node.src, node.a, 5)
    os.remove(node.proc / "39" / "cwd")
    os.symlink(f"{node.src}/{node.a}", node.proc / "39" / "cwd")
    rep = run(node, nginx_conf=node.tmp / "nginx.conf")
    assert rep["error"] is None and kept(rep)[node.a] == ["2: process 39 (p39) uses it"], rep


@pytest.mark.parametrize("how", ["conf names the root", "conf includes", "conf loads a module", "root worker", "master with -c",
                                 "not nginx"])
def test_any_other_process_it_cannot_read_still_deletes_nothing(node, monkeypatch, how):
    top = Path(os.path.abspath(node.src)).parts[1]
    kw = {"conf names the root": {"conf": RUNPOD_NGINX + f"# root /{top}/x;\n"},
          "conf includes": {"conf": RUNPOD_NGINX + "include /etc/nginx/sites-enabled/*;\n"},
          "conf loads a module": {"conf": "load_module modules/x.so;\n" + RUNPOD_NGINX},
          "root worker": {"uid": 0},
          "master with -c": {"master": "nginx: master process /usr/sbin/nginx -c /elsewhere.conf"},
          "not nginx": {"name": "python3"}}[how]
    rep = run(node, nginx_conf=_runpod_proxy(node, monkeypatch, **kw))
    _all_kept(node, rep, "2: ")
    assert "/40 is unreadable" in rep["error"]


def test_an_empty_proc_deletes_nothing(node):
    empty = node.tmp / "noproc"
    empty.mkdir()
    _all_kept(node, run(node, proc=empty), "2: ")


def test_a_runs_root_or_a_record_it_cannot_read_deletes_nothing(node, monkeypatch):
    rd = write_run(node, "r14", source=node.a)
    real_read = Path.read_text

    def read_text(self, *a, **k):
        if self == rd / "job.json":
            raise PermissionError(13, "Permission denied")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    _all_kept(node, run(node), f"2: cannot read {rd / 'job.json'}")
    monkeypatch.setattr(Path, "read_text", real_read)
    real_list = os.listdir

    def listdir(p="."):
        if str(p) == str(node.root / "runs"):
            raise PermissionError(13, "Permission denied")
        return real_list(p)

    monkeypatch.setattr(EV.os, "listdir", listdir)
    _all_kept(node, run(node), f"2: cannot list {node.root / 'runs'}")


def test_a_user_that_appears_before_the_deletion_keeps_the_tree_and_after_the_rename_moves_it_back(node, monkeypatch):
    """Each tree is checked again just before it goes, then renamed aside (never executable by name) and checked once more."""
    real, calls = EV.proc_uses, []

    def late(pattern, proc, *rest):
        calls.append(1)
        found = real(pattern, proc, *rest)
        if len(calls) == 2:                      # the pre-check of a
            found.setdefault(node.a, set()).add("2: process 7 (late) uses it")
        if len(calls) == 4:                      # b's check after its rename
            found.setdefault(node.b, set()).add("2: process 8 (later) uses it")
        return found

    monkeypatch.setattr(EV, "proc_uses", late)
    ino = os.stat(node.src / node.b).st_ino
    rep = run(node)
    assert kept(rep)[node.a] == ["2: process 7 (late) uses it"]
    assert kept(rep)[node.b] == ["2: moved back", "2: process 8 (later) uses it"]
    assert rep["deleted"] == [] and os.stat(node.src / node.b).st_ino == ino and (node.src / node.b / EV.READY_FILE).exists()
    assert sorted(set(os.listdir(node.src)) - {EV.LOCK}) == sorted([node.a, node.b, node.c])


def test_a_check_that_fails_after_the_rename_puts_the_tree_back_and_stops(node, monkeypatch):
    real, calls = EV.proc_uses, []

    def failing(pattern, proc, *rest):
        calls.append(1)
        if len(calls) == 3:
            raise EV.Unestablished("2: /proc/9 is unreadable")
        return real(pattern, proc, *rest)

    monkeypatch.setattr(EV, "proc_uses", failing)
    rep = run(node)
    assert rep["error"] == "2: /proc/9 is unreadable" and rep["deleted"] == []
    assert sorted(set(os.listdir(node.src)) - {EV.LOCK}) == sorted([node.a, node.b, node.c]) and (node.src / node.a / EV.READY_FILE).exists()


def test_an_interrupted_deletion_is_finished_unless_something_uses_it(node):
    gone, used = node.src / f"{EV.ASIDE}{node.c}-0000", node.src / f"{EV.ASIDE}{'e' * 40}-0001"
    for d in (gone, used):
        (d / "deep").mkdir(parents=True)
        (d / "deep" / "f").write_text("half deleted")
    fake_proc(node.proc, 300, cwd=f"{used}/deep")
    rep = run(node)
    assert {"tree": gone.name, "leftover": True, "inodes": 3, "bytes": 12} in rep["deleted"] and not gone.exists()
    assert kept(rep)[used.name] == ["2: process 300 (p300) uses it"] and used.exists()


def test_no_deletion_starts_after_the_budget(node):
    t = [0.0]
    rep = run(node, clock=lambda: t[0], budget_s=60, log=lambda s: t.__setitem__(0, 61.0) if s.startswith("src: deleted ") else None)
    assert [d["tree"] for d in rep["deleted"]] == [node.a]
    assert kept(rep)[node.b] == ["budget: no deletion starts after 1 min; the next pass goes on"] and (node.src / node.b).is_dir()


def test_a_dry_run_deletes_nothing_and_says_what_would_go(node):
    rep = run(node, dry_run=True)
    assert [d["tree"] for d in rep["deleted"]] == [node.a, node.b] and rep["deleted"][0]["age_h"] == 5.0
    assert all((node.src / s / EV.READY_FILE).exists() for s in (node.a, node.b, node.c))


def test_no_src_dir_is_nothing_to_do(tmp_path):
    rep = EV.evict_src(tmp_path / "research" / "src", origin=str(tmp_path / "nowhere"), proc=tmp_path, jobs_dirs=[], now=NOW)
    assert rep["error"] is None and rep["trees"] == 0 and rep["deleted"] == rep["kept"] == []


def test_the_cli_runs_the_pass_whatever_the_target_and_reports_it(node, monkeypatch, capsys):
    from research.store.cli import main
    seen = {}

    def fake(src, **kw):
        seen.update(src=src, **kw)
        return {"deleted": [], "kept": [], "error": None}

    monkeypatch.setattr(EV, "evict_src", fake)
    assert main(["evict", "--target-free-gb", "0", "--src-dir", str(node.src), "--src-origin", node.origin, "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["src"] == {"deleted": [], "kept": [], "error": None}
    assert seen["src"] == node.src and seen["origin"] == node.origin and seen["dry_run"] is True
    assert seen["keep_s"] == EV.SRC_KEEP_S
    main(["evict", "--target-free-gb", "0", "--src-dir", str(node.src), "--src-keep-h", "0.5", "--dry-run", "--json"])
    capsys.readouterr()
    assert seen["keep_s"] == 1800, "a node short of inodes keeps recent trees for less"


def test_by_default_a_tree_is_recoverable_when_the_nodes_own_repository_holds_its_commit(node):
    """Root, 4 Oct 17:27Z: condition 1 is recoverability, so the node's `git/verity.git` (refs/research/src/<sha>, what `research
    run` re-ships from) is its repository and no credential for GitHub is needed: c, a branch origin never saw, goes; a tree whose
    commit the node doesn't hold stays."""
    stray = "f" * 40
    tree(node.src, stray, 6)
    rep = run(node, origin=None)
    assert rep["error"] is None and rep["origin"] == str(node.root / "git" / "verity.git")
    assert node.c in {d["tree"] for d in rep["deleted"]} and not (node.src / node.c).exists()
    assert kept(rep)[stray] == [f"1: not recoverable on the node: on no ref of {node.root / 'git' / 'verity.git'}"]


def test_a_tree_that_is_not_exactly_its_commit_is_kept(node):
    """Red team (bc-ddc07abc) on node 1: trees with lean-audit.json rewritten in place after the ship would have gone.  The commit is
    recoverable, its tree's own changes are not: a tracked file edited, an untracked file however deep, another HEAD, no .git."""
    (node.src / node.a / "f").write_text("edited in place: no commit has this")
    (node.src / node.b / "notes" / "deep").mkdir(parents=True)
    (node.src / node.b / "notes" / "deep" / "result.json").write_text('{"only": "copy"}')
    _git(node.src / node.c, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", node.b)
    for s in (node.a, node.b, node.c):
        _age(node.src / s, 5)
    rep = run(node, origin=None)
    assert rep["deleted"] == [] and rep["error"] is None
    why = kept(rep)
    assert why[node.a] == ["1: not recoverable on the node: 1 local change(s) no commit holds, e.g. '" + _status_line(node.src / node.a) + "'"]
    assert why[node.b] == ["1: not recoverable on the node: 1 local change(s) no commit holds, e.g. '? notes/'"]
    assert why[node.c] == [f"1: not recoverable on the node: its HEAD is {node.b}"]
    _git(node.src / node.c, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", node.c)
    import shutil
    shutil.rmtree(node.src / node.c / ".git")
    assert kept(run(node, origin=None))[node.c] == ["1: not recoverable on the node: no .git to compare it with its commit"]


def _status_line(t: Path) -> str:
    out = _git(t, "status", "--porcelain=v2", "-z", "--untracked-files=normal")
    return next(e for e in out.split("\0") if e and e != f"? {EV.READY_FILE}")[-120:]


def test_a_git_worktree_made_from_a_tree_keeps_it(node):
    """ci-bundle.sh and train.sh `git worktree add` from a checkout: its git data lives in the tree, its processes elsewhere."""
    _git(node.src / node.a, "worktree", "add", "-q", "--detach", str(node.tmp / "wt"), node.a)
    _age(node.src / node.a, 5)
    assert "2: a git worktree made from it (.git/worktrees) may be in use" in kept(run(node))[node.a]


def test_one_pass_at_a_time(node, monkeypatch):
    """A second pass during the first would take its aside tree for an interrupted pass's leftover and delete it while the first
    is still finding a user; it does nothing instead."""
    real, calls, inner = EV.proc_uses, [], []

    def hooked(pattern, proc, *rest):
        calls.append(1)
        found = real(pattern, proc, *rest)
        if len(calls) == 3:                      # a's check after its rename: another pass starts
            inner.append(EV.evict_src(node.src, origin=node.origin, now=NOW, proc=node.proc, jobs_dirs=[], clock=lambda: 0.0))
            found.setdefault(node.a, set()).add("2: process 77 (lean) uses it")
        return found

    monkeypatch.setattr(EV, "proc_uses", hooked)
    rep = run(node)
    assert inner[0]["error"] == f"another pass holds {node.src / EV.LOCK}" and inner[0]["deleted"] == []
    assert kept(rep)[node.a] == ["2: moved back", "2: process 77 (lean) uses it"] and (node.src / node.a / EV.READY_FILE).exists()


def test_a_replace_ref_or_graft_in_the_nodes_repository_places_nothing(node):
    """Anyone who can push to the node's repository could graft a commit under a tip origin has."""
    gd = node.root / "git" / "verity.git"
    _git(node.tmp, "--git-dir", str(gd), "replace", "--graft", node.b, node.c)
    assert node.c in kept(run(node))
    (gd / "info").mkdir(exist_ok=True)
    (gd / "info" / "grafts").write_text(f"{node.b} {node.c}\n")
    rep = run(node)
    assert rep["error"] == f"1: {gd / 'info' / 'grafts'} would rewrite the history the ancestor walk reads" and node.c in kept(rep)


@pytest.mark.parametrize("how", ["relative-cmdline", "double-slash-env", "dot-env"])
def test_a_process_naming_a_tree_however_it_spells_the_path_keeps_it(node, how):
    a = node.a
    if how == "relative-cmdline":
        fake_proc(node.proc, 500, cwd=str(node.root), cmdline=f"python3 src/{a}/tools/verity/check/check.py")
    elif how == "double-slash-env":
        fake_proc(node.proc, 500, environ=f"PYTHONPATH={node.root}//src/{a}/tools/research/src\0")
    else:
        fake_proc(node.proc, 500, environ=f"PYTHONPATH={node.src}/./{a}/tools/research/src\0")
    assert kept(run(node))[a] == ["2: process 500 (p500) uses it"]


def test_a_thread_group_whose_leader_exited_is_read_through_its_threads(node):
    d = node.proc / "600"
    fake_proc(node.proc, 600)
    for n in ("cwd", "root", "exe"):
        os.unlink(d / n)
    (d / "cmdline").write_text("")
    (d / "task" / "601").mkdir(parents=True)
    os.symlink(f"{node.src}/{node.a}/backends", d / "task" / "601" / "cwd")
    assert kept(run(node))[node.a] == ["2: process 600 (p600) uses it"]


def test_a_record_of_an_unexpected_shape_keeps_its_tree_and_an_error_after_the_rename_puts_it_back(node, monkeypatch):
    rd = write_run(node, "r20", source=node.a, store={"commit": node.a})
    job = json.loads((rd / "job.json").read_text())
    job["store"]["conditions"]["source"] = "a string"
    (rd / "job.json").write_text(json.dumps(job))
    _age(rd, 3)
    assert "3: run r20's store.conditions.source is not an object" in kept(run(node))[node.a]
    import shutil
    shutil.rmtree(rd)
    real, calls = EV.record_uses, []

    def boom(*a, **k):
        calls.append(1)
        if len(calls) == 3:                      # a's check after its rename
            raise AttributeError("unexpected")
        return real(*a, **k)

    monkeypatch.setattr(EV, "record_uses", boom)
    with pytest.raises(AttributeError):
        run(node)
    assert (node.src / node.a / EV.READY_FILE).exists() and not [n for n in os.listdir(node.src) if n.startswith(EV.ASIDE)]
