"""research.liveness: what live work names (requests, Kueue objects, fill jobs, every process), what a source tree holds besides its
commit, and node 1's job trees. A fake /proc and fake kubectl, one real child process, and real git repositories in tmp dirs."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from research import liveness as L
from research.liveness import JobTrees, Live, LivenessError, Unknown, live_pods, outside_commit

SHA = "ab" * 20


def proc(base: Path, pid: int, *, cwd: str = "/", fds: tuple[str, ...] = (), cmdline: str = "python", environ: str = "",
         maps: str = "") -> Path:
    """A fake /proc/<pid>: cwd and root links, fd links, and maps, cmdline, environ and comm."""
    d = base / str(pid)
    (d / "fd").mkdir(parents=True)
    os.symlink(cwd, d / "cwd")
    os.symlink("/", d / "root")
    for i, f in enumerate(fds):
        os.symlink(f, d / "fd" / str(i))
    (d / "maps").write_text(maps)
    (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    (d / "environ").write_bytes(environ.replace(" ", "\0").encode())
    (d / "comm").write_text(cmdline.split()[0] + "\n")
    return d


def live(tmp_path: Path, **kw) -> Live:
    (tmp_path / "proc").mkdir(exist_ok=True)
    return Live(**{"research": tmp_path / "research", "fill": tmp_path / "fill", "proc": tmp_path / "proc", "kube": None, "units": (),
                   **kw})


def test_a_process_names_a_path_through_its_cwd_files_maps_argv_or_environment_and_never_a_longer_name(tmp_path):
    p = live(tmp_path)
    proc(p.proc, 10, cwd="/w/src/" + SHA + "/tools")
    proc(p.proc, 11, fds=("/tmp/pytest-of-research/pytest-12/x.log",))
    proc(p.proc, 12, environ="PYTHONPATH=/w/jobs/src/pod-a:/x HOME=/root")
    proc(p.proc, 13, maps="7f00 r-xp /w/scratch/lib.so\n", cmdline="python -c 'open(\"/w/q\")'")
    pats = [SHA, "/w/src/" + SHA, "/tmp/pytest-of-research/pytest-1", "/tmp/pytest-of-research/pytest-12", "/w/jobs/src/pod-a",
            "/w/jobs/src/pod", "/w/scratch", "/w/q", "/w/nothing"]
    assert p.names(pats) == {SHA: ["proc:10(python)"], "/w/src/" + SHA: ["proc:10(python)"],
                             "/tmp/pytest-of-research/pytest-12": ["proc:11(python)"], "/w/jobs/src/pod-a": ["proc:12(python)"],
                             "/w/scratch": ["proc:13(python)"], "/w/q": ["proc:13(python)"]}


def test_a_request_names_while_unfinished_and_its_runner_alive_or_not_yet_launched(tmp_path):
    p = live(tmp_path)
    proc(p.proc, 77)
    for rid, status, launch in [("r20261003-a", None, None), ("r20261003-b", "running", 77), ("r20261003-c", "done", 77),
                                ("r20261003-d", "running", 99), ("r20261003-e", None, None)]:
        q = tmp_path / "research/requests" / rid
        q.mkdir(parents=True)
        (q / "request.json").write_text(json.dumps({"cwd": f"/w/src/{SHA}", "id": rid}))
        run = tmp_path / "research/runs" / rid
        run.mkdir(parents=True)
        if status:
            (run / "status.json").write_text(json.dumps({"state": status}))
        if launch:
            (run / "launch.json").write_text(json.dumps({"runner_pid": launch}))
    (tmp_path / "research/runs/r20261003-e/refused.json").write_text("{}")
    assert p.names([SHA]) == {SHA: ["request:r20261003-a", "request:r20261003-b"]}


def test_unfinished_pods_and_workloads_and_queued_or_running_fill_jobs_name_what_they_carry(tmp_path):
    objs = {"pods": [{"metadata": {"name": "p1"}, "status": {"phase": "Running"}, "spec": {"volumes": [{"hostPath": {"path": "/w/a"}}]}},
                     {"metadata": {"name": "p2"}, "status": {"phase": "Succeeded"}, "spec": {"x": "/w/a"}}],
            "workloads.kueue.x-k8s.io": [{"metadata": {"name": "w1"}, "status": {"conditions": [{"type": "Finished", "status": "True"}]},
                                          "spec": {"x": "/w/a"}},
                                         {"metadata": {"name": "w2"}, "status": {}, "spec": {"x": "/w/a/b"}}]}
    p = live(tmp_path, kube=lambda kind: objs[kind])
    (tmp_path / "fill/queue").mkdir(parents=True)
    (tmp_path / "fill/queue/j1.sh").write_text("cd /w/a && run\n")
    (tmp_path / "fill/done").mkdir()
    (tmp_path / "fill/done/j0.sh").write_text("cd /w/a\n")
    assert p.names(["/w/a"]) == {"/w/a": ["pods:p1", "workloads:w2", "fill:queue/j1.sh"]}
    assert live_pods(lambda kind: objs[kind]) == {"p1"}


def test_installed_units_and_held_fill_jobs_name_their_trees_while_nothing_runs_from_them(tmp_path):
    """node 2's vy-cluster-agent.service names its tree while stopped (its STOP file, exit 2, the start limit); a held fill job is
    moved back to queue/ by hand. A unit file, a drop-in and a link, in the system's and users' unit dirs; finished jobs name nothing."""
    system, user = tmp_path / "etc/systemd/system", tmp_path / "home/research/.config/systemd/user"
    (system / "vy-cluster-agent.service.d").mkdir(parents=True)
    (system / "vy-cluster-agent.service").write_text(f"[Service]\nWorkingDirectory=/w/src/{SHA}\n")
    (system / "vy-cluster-agent.service.d/env.conf").write_text("[Service]\nEnvironment=PYTHONPATH=/w/b/src\n")
    (user / "default.target.wants").mkdir(parents=True)
    os.symlink("/w/c/infra-pool-publish.service", user / "default.target.wants/infra-pool-publish.service")
    (tmp_path / "fill/held-overnight").mkdir(parents=True)
    (tmp_path / "fill/held-overnight/proofs-pn2g.sh").write_text("cd /w/d && python -m bench\n")
    for d in L.FILL_ENDED:
        (tmp_path / "fill" / d).mkdir()
        (tmp_path / "fill" / d / "old.sh").write_text("cd /w/e\n")
    p = live(tmp_path, units=(system, tmp_path / "home/*/.config/systemd/user", tmp_path / "run/systemd/transient"))
    assert p.names([SHA, "/w/b", "/w/c", "/w/d", "/w/e"]) == {
        SHA: [f"unit:{system}/vy-cluster-agent.service"], "/w/b": [f"unit:{system}/vy-cluster-agent.service.d/env.conf"],
        "/w/c": [f"unit:{user}/default.target.wants/infra-pool-publish.service"], "/w/d": ["fill:held-overnight/proofs-pn2g.sh"]}


def test_a_process_names_what_the_script_its_arguments_name_defaults_to(tmp_path):
    """node 2's custody push loop (`bash push.sh` in tmux) runs node 1's tree from CUSTODY_SRC_N1, else push.sh's SRC_N1 default,
    which only the script's text carries. A file over SCRIPT_MAX, or that isn't regular, is no script."""
    other, big = "cd" * 20, "ef" * 20
    push = tmp_path / "custody-push"
    push.mkdir()
    (push / "push.sh").write_text(f'SRC_N1="${{CUSTODY_SRC_N1:-/workspace/research/src/{SHA}}}"\n')
    (tmp_path / "big.bin").write_bytes(b"x" * L.SCRIPT_MAX + big.encode())
    p = live(tmp_path)
    proc(p.proc, 20, cwd=str(push), cmdline="/bin/bash ./push.sh")
    proc(p.proc, 21, cmdline=f"/bin/bash {push}/push.sh", environ=f"CUSTODY_SRC_N1=/workspace/research/src/{other}")
    proc(p.proc, 22, cwd=str(tmp_path), cmdline="python big.bin custody-push missing.sh")
    assert p.names([SHA, other, big]) == {SHA: ["proc:20(/bin/bash)", "proc:21(/bin/bash)"], other: ["proc:21(/bin/bash)"]}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_unit_or_a_script_it_cant_read_raises(tmp_path):
    units = tmp_path / "units"
    units.mkdir()
    (units / "a.service").write_text("")
    (tmp_path / "run.sh").write_text("")
    (tmp_path / "s").mkdir()
    for path, scan, why in ((units / "a.service", live(tmp_path, units=(units,)), "gc reads every unit file: run it as root"),
                            (units, live(tmp_path, units=(units,)), "gc reads every unit file: run it as root"),
                            (tmp_path / "run.sh", live(tmp_path / "s"), "which a process's arguments name")):
        if path.name == "run.sh":
            proc(scan.proc, 30, cwd=str(tmp_path), cmdline="bash run.sh")
        os.chmod(path, 0)
        try:
            with pytest.raises(LivenessError, match=f"can't read {re.escape(str(path))}.*{why}"):
                scan.names(["/w"])
        finally:
            os.chmod(path, 0o755)


SSH = """import json, os, sys
d = os.environ["FAKE_SSH"]
name = os.path.basename(sys.argv[0])
with open(os.path.join(d, name + ".json"), "w") as f:
    json.dump({"argv": sys.argv[1:], "stdin": "" if name == "runuser" else sys.stdin.read()}, f)
if name == "runuser":
    os.execvp(sys.argv[4], sys.argv[4:])
rc, out, err = json.load(open(os.path.join(d, "answer.json")))
sys.stdout.write(out)
sys.stderr.write(err)
sys.exit(rc)
"""


def test_a_peer_is_asked_over_ssh_and_an_answer_it_cant_give_raises(tmp_path, monkeypatch):
    """SshPeer runs `research retention names` on the other node, as research when this is root; fake ssh and runuser on PATH."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name in ("ssh", "runuser"):
        (bin_ / name).write_text(f"#!{sys.executable}\n{SSH}")
        os.chmod(bin_ / name, 0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SSH", str(tmp_path))
    remote = ("sudo", "-n", "research", "retention", "names", "--json", "-")
    peer = L.SshPeer("vy-nebius-2", "vy-n2", remote)

    def answer(rc: int, out, err: str = "") -> None:
        (tmp_path / "answer.json").write_text(json.dumps([rc, out if isinstance(out, str) else json.dumps(out), err]))

    def asked(name: str = "ssh") -> dict:
        return json.loads((tmp_path / f"{name}.json").read_text())

    monkeypatch.setattr(L.os, "geteuid", lambda: 1000)
    answer(1, {"schema": "research/retention-names/v1", "names": {SHA: ["proc:7(bash)"], "/x": ["fill:queue/j"], "/unasked": ["x"]}})
    assert peer([SHA, "/x", "/y", SHA, "a\nb"]) == {SHA: ["vy-nebius-2:proc:7(bash)"], "/x": ["vy-nebius-2:fill:queue/j"]}
    ssh = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "vy-n2", *remote]
    assert asked() == {"argv": ssh, "stdin": f"{SHA}\n/x\n/y\n"} and not (tmp_path / "runuser.json").exists()
    os.unlink(tmp_path / "ssh.json")
    assert peer(["a\nb"]) == {} and not (tmp_path / "ssh.json").exists(), "nothing to ask"

    monkeypatch.setattr(L.os, "geteuid", lambda: 0)
    answer(0, {"names": {}})
    assert peer([SHA]) == {} and asked("runuser")["argv"] == ["-u", "research", "--", "ssh", *ssh]
    for rc, out, err, why in ((3, {"schema": "research/retention-names/v1", "refused": "status.txt does not say it"}, "", "status.txt does not say it"),
                              (255, "", "ssh: connect to host vy-n2 port 22: Connection refused\n", "Connection refused"),
                              (0, "not JSON", "", "exit 0, no answer"), (2, {"names": {}}, "usage: research retention names", "usage")):
        answer(rc, out, err)
        with pytest.raises(LivenessError, match=rf"can't ask vy-nebius-2 what it names \(vy-n2: .*{why}"):
            peer([SHA])

    answer(1, {"names": {SHA: ["unit:/etc/systemd/system/vy-cluster-agent.service"]}})
    p = live(tmp_path, peers=[peer])
    proc(p.proc, 3, cwd=f"/w/src/{SHA}")
    assert p.names([SHA, "/z"]) == {SHA: ["proc:3(python)", "vy-nebius-2:unit:/etc/systemd/system/vy-cluster-agent.service"]}


def test_a_scan_that_cannot_look_raises_and_a_process_gone_mid_scan_names_nothing(tmp_path):
    def failing(kind):
        raise LivenessError("kubectl get pods exited 1: connection refused")

    with pytest.raises(LivenessError, match="connection refused"):
        live(tmp_path, kube=failing).names(["/w/a"])
    p = live(tmp_path)
    proc(p.proc, 5, cwd="/w/a")
    shut = proc(p.proc, 6)
    os.rmdir(shut / "fd")
    (shut / "fd").write_text("")
    with pytest.raises(LivenessError, match="can't read .*/6: .*run it as root"):
        p.names(["/w/a"])
    assert live(tmp_path, pids=[5, 4242]).names(["/w/a"]) == {"/w/a": ["proc:5(python)"]}


def test_a_real_process_whose_cwd_is_in_a_tree_names_it(tmp_path):
    tree = tmp_path / "src" / SHA
    tree.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], cwd=tree, stdin=subprocess.PIPE)
    scan = Live(research=tmp_path, fill=tmp_path, kube=None, pids=[child.pid], units=())
    try:
        named = scan.names([SHA, str(tmp_path / "src"), str(tmp_path / "elsewhere")])
    finally:
        child.stdin.close()
        child.wait()
    assert set(named) == {SHA, str(tmp_path / "src")}
    assert all(refs == [refs[0]] and refs[0].startswith(f"proc:{child.pid}(") for refs in named.values())
    assert scan.names([SHA]) == {}, "an exited process names nothing"


# ---- outside the commit -------------------------------------------------------------------------------------------------------------------

LA = "backends/flock/verifier/lean/soundness/lean-audit.json"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def repo(tmp_path: Path, files: dict[str, str] | None = None) -> tuple[Path, str]:
    """A committed checkout named by its commit (with .gitignore hiding results/ and .venv/), and that commit."""
    d = tmp_path / "checkout"
    d.mkdir()
    files = {".gitignore": "results/\n.venv/\n", LA: "{}\n", "fixtures/artifacts.json": json.dumps({"art:" + "1" * 64: "fixtures/big"}),
             "README.md": "x\n", **(files or {})}
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    git(d, "init", "-q")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "c")
    sha = git(d, "rev-parse", "HEAD")
    return d.rename(tmp_path / sha), sha


def test_a_checkout_holds_untracked_ignored_and_changed_files_but_never_caches_or_registered_fixtures(tmp_path):
    d, _ = repo(tmp_path)
    assert outside_commit(d, tmp_path / "absent.git") == []
    for rel in (".venv/lib/x.py", "tools/__pycache__/a.pyc", "fixtures/big/blob.bin", "backends/x/.lake/packages/m/a.olean"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("cache")
    assert outside_commit(d, tmp_path / "absent.git") == []
    (d / LA).write_text('{"changed": 1}\n')
    assert outside_commit(d, tmp_path / "absent.git") == [f"M {LA}"]
    (d / "results/run1").mkdir(parents=True)
    (d / "results/run1/out.json").write_text("{}")
    (d / "notes.txt").write_text("mine")
    assert sorted(outside_commit(d, tmp_path / "absent.git")) == sorted([f"M {LA}", "results/run1/out.json", "notes.txt"])


def test_a_tree_shipped_without_git_is_compared_with_the_bare_repos_commit_and_its_ready_time(tmp_path):
    d, sha = repo(tmp_path)
    bare = tmp_path / "verity.git"
    git(tmp_path, "clone", "-q", "--bare", str(d), str(bare))
    tree = tmp_path / "src" / sha
    (tmp_path / "src").mkdir()
    subprocess.run(["cp", "-a", str(d), str(tree)], check=True)
    subprocess.run(["rm", "-rf", str(tree / ".git")], check=True)
    with pytest.raises(Unknown, match="no READY.json"):
        outside_commit(tree, bare)
    (tree / "READY.json").write_text("{}")
    shipped = os.stat(tree / "READY.json").st_mtime
    for f in tree.rglob("*"):
        os.utime(f, (shipped - 100, shipped - 100), follow_symlinks=False)
    assert outside_commit(tree, bare) == []
    os.utime(tree / "README.md", (shipped + 120, shipped + 120))
    (tree / "out.log").write_text("a run's")
    assert sorted(outside_commit(tree, bare)) == ["changed README.md", "out.log"]
    with pytest.raises(Unknown, match=r"the bare repo can't list the commit \(git ls-tree exited 128 and said 'fatal: "):
        outside_commit(tree.rename(tmp_path / "src" / ("0" * 40)), bare)


def test_work_committed_or_stashed_in_a_checkout_is_outside_its_commit_and_the_bare_repos_branches_are_not(tmp_path):
    """A git-shipped tree is a `clone --shared` of the bare repo detached at its commit (remote._git_ship_script), so the bare
    repo's branches arrive as remote-tracking refs. An agent's commit there leaves the tree clean against HEAD; a stash too."""
    d, sha = repo(tmp_path)
    git(d, "checkout", "-qb", "other")
    (d / "new.md").write_text("x\n")
    git(d, "add", "new.md")
    git(d, "commit", "-qm", "other")
    bare = tmp_path / "verity.git"
    git(tmp_path, "clone", "-q", "--bare", str(d), str(bare))
    tree = tmp_path / "src" / sha
    git(tmp_path, "clone", "-q", "--shared", "--no-checkout", str(bare), str(tree))
    git(tree, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha)
    git(tree, "branch", "mine", "origin/other")
    assert outside_commit(tree, bare) == [], "the bare repo has every commit its branches reach"
    (tree / "notes.md").write_text("an agent's work\n")
    git(tree, "add", "notes.md")
    git(tree, "commit", "-qm", "work")
    work = git(tree, "rev-parse", "HEAD")
    assert outside_commit(tree, bare) == [f"HEAD at {work}", f"commit {work}"]
    git(tree, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha)
    assert outside_commit(tree, bare) == [f"commit {work}"], "left behind, but in HEAD's reflog"
    (tree / "README.md").write_text("edited\n")
    git(tree, "stash", "-q")
    out = outside_commit(tree, bare)
    assert out[0].startswith("stash@{0}: WIP on") and f"commit {work}" in out and all(x.startswith("commit ") for x in out[1:])


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_what_git_cant_read_as_the_owner_or_warns_about_is_unknown_never_nothing(tmp_path, monkeypatch):
    """As root, gc runs git as the tree's owner: about a directory only root can read, git warns `could not open directory`,
    lists nothing and exits 0. That warning, a nonzero exit, and any path the owner (or this user) can't read are Unknown."""
    d, _ = repo(tmp_path)
    (d / "out").mkdir()
    (d / "out/result.json").write_text("{}")
    (d / ".venv").mkdir()
    for path, mode, back, euid in ((d / "out", 0, 0o755, None), (d / "out/result.json", 0o200, 0o644, None),
                                   (d / "out", 0o305, 0o755, 0), (d / "README.md", 0o244, 0o644, 0)):
        os.chmod(path, mode)
        rel = os.path.relpath(path, d) + ("/" if path.is_dir() else "")
        try:
            with monkeypatch.context() as m:
                if euid is not None:
                    m.setattr(L.os, "geteuid", lambda: euid)
                with pytest.raises(Unknown, match=f"^{re.escape(rel)} can't be read by uid {os.getuid()}, as whom git runs"):
                    outside_commit(d, tmp_path / "absent.git")
        finally:
            os.chmod(path, back)
    assert outside_commit(d, tmp_path / "absent.git") == ["out/result.json"]
    os.chmod(d / ".venv", 0)
    try:
        with pytest.raises(Unknown, match=re.escape("git status exited 0 and said \"warning: could not open directory '.venv/'")):
            outside_commit(d, tmp_path / "absent.git")
    finally:
        os.chmod(d / ".venv", 0o755)
    monkeypatch.setattr(L, "_git", lambda args, _owner: subprocess.CompletedProcess(args, 128, b"", b""))
    with pytest.raises(Unknown, match="^git rev-parse HEAD exited 128$"):
        outside_commit(d, tmp_path / "absent.git")


# ---- job trees ------------------------------------------------------------------------------------------------------------------------

def test_job_trees_age_a_content_copy_by_its_newest_by_pod_file_and_hold_what_a_live_pod_reads(tmp_path):
    j = tmp_path / "jobs/src"
    for name in ("pod-alive", "pod-dead", "0123456789abcdef", "fedcba9876543210", "x.partial.1", ".trash", "by-pod"):
        (j / name).mkdir(parents=True)
    (j / "0123456789abcdef/.copied").write_text("")
    os.symlink(j / "pod-dead", j / "link")
    (j / "by-pod/alive").write_text(f"{j}/0123456789abcdef\n")
    (j / "by-pod/gone").write_text(f"{j}/0123456789abcdef\n")
    (j / "by-pod/elsewhere").write_text("/other/0123456789abcdef\n")
    os.utime(j / "by-pod/gone", (4102444800, 4102444800))
    jt = JobTrees(j, {"alive", "pod-alive"})
    assert dict(jt.copies()) == {"0123456789abcdef": None, "fedcba9876543210": "no .copied, so job_tree.sh may be remaking it",
                                 "pod-alive": None, "pod-dead": None}
    assert jt.age_stamp("0123456789abcdef") == 4102444800
    assert (jt.held("pod-alive"), jt.held("pod-dead"), jt.held("0123456789abcdef")) == ("live pod alive", "", "by-pod of live pod alive")
    dead = JobTrees(j, set())
    assert dead.held("0123456789abcdef") == ""
    assert dead.held("0123456789abcdef", since=4102444800) == "by-pod written since the rename: gone"
