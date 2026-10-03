"""research.retention_rules through `research retention gc` and `report`: the resource steward's delete-without-asking policy as
default rules. Fixture trees in tmp dirs (real git checkouts as shipped source trees), a fake /proc, fake kubectl, a scripted
liveness for the checks after a rename, and a clock in 2031, so every file a test writes is old unless the test dates it."""
from __future__ import annotations

import fcntl
import io
import itertools
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research import retention as R
from research import retention_rules as RR
from research.cli import main as research_main
from research.liveness import Live, LivenessError
from research.store import custody as CU
from research.store.local import LocalStore

NOW = datetime(2031, 1, 1, tzinfo=timezone.utc)
T = NOW.timestamp()
N1, N2 = "vy-nebius-1", "vy-nebius-2"
LA = RR.LA
_n = itertools.count()


def clock() -> datetime:
    return NOW


def ago(path: Path, hours: float) -> None:
    os.utime(path, (T - hours * 3600, T - hours * 3600), follow_symlinks=False)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def tree(root: Path, lake: bool = False, attempt: bool = False) -> Path:
    """A shipped source tree: a checkout at research/src/<its commit> (results/ ignored), with the verifier's three Lake packages'
    .lake/packages when `lake`, and the store's attempt.py (a complete tree, RR.NEWEST) when `attempt`."""
    d = root.parent / f"checkout{next(_n)}"
    (d / LA).parent.mkdir(parents=True)
    (d / LA).write_text("{}\n")
    if attempt:
        (d / RR.ATTEMPT).parent.mkdir(parents=True)
        (d / RR.ATTEMPT).write_text("")
    (d / "README.md").write_text(f"{d.name}\n")
    (d / ".gitignore").write_text("results/\n.lake/\n")
    git(d, "init", "-q")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "c")
    dst = root / "research/src" / git(d, "rev-parse", "HEAD")
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.rename(dst)
    for k in RR.LAKE if lake else ():
        (dst / k / ".lake/packages/mathlib").mkdir(parents=True)
        (dst / k / ".lake/packages/mathlib/A.olean").write_bytes(b"o" * 100)
    return dst


def proc(base: Path, pid: int, *, cwd: str = "/", fds: tuple[str, ...] = ()) -> Path:
    d = base / str(pid)
    (d / "fd").mkdir(parents=True)
    os.symlink(cwd, d / "cwd")
    os.symlink("/", d / "root")
    for i, f in enumerate(fds):
        os.symlink(f, d / "fd" / str(i))
    for f, text in (("maps", ""), ("cmdline", "python"), ("environ", ""), ("comm", "python\n")):
        (d / f).write_text(text)
    return d


def pods(*running: str):
    objs = [{"metadata": {"name": n}, "status": {"phase": "Running"}, "spec": {}} for n in running]
    return lambda kind: objs if kind == "pods" else []


def fake_live(tmp_path: Path, root: Path, kube=None, **kw) -> Live:
    (tmp_path / "proc").mkdir(parents=True, exist_ok=True)
    return Live(**{"research": root / "research", "fill": root / "pouw/fill", "proc": tmp_path / "proc", "kube": kube, "units": (), **kw})


class Scripted:
    """A Live whose scans answer in turn: a dict {pattern: refs}, a callable run first and then answered, or an exception."""

    def __init__(self, *answers, kube=None):
        self.answers, self.kube, self.asked = list(answers), kube, []

    def names(self, patterns):
        self.asked.append(sorted(patterns))
        a = self.answers.pop(0)
        a = a() if callable(a) else a
        if isinstance(a, Exception):
            raise a
        return {p: r for p, r in a.items() if p in patterns}


QUIET = "2031-01-01T00:00:00Z free GPUs 7/8, kept free -, waiters -, timed False, window waiting False, backfill False\n"
UNITS = Path(RR.__file__).parent / "pods/nebius"


def status(root: Path, line: str = QUIET, age_s: float = 0) -> Path:
    """Node 2's fill dir, its runner's status.txt written `age_s` before NOW."""
    fill = root / "pouw/fill"
    fill.mkdir(parents=True, exist_ok=True)
    (fill / "status.txt").write_text(line)
    os.utime(fill / "status.txt", (T - age_s, T - age_s))
    return fill


def run_gc(root: Path, **kw) -> dict:
    """R.gc on node 2 by default, whose fill runner is quiet (a fresh status) unless the test made pouw/fill or chose the node."""
    if "node" not in kw and not (root / "pouw/fill").exists():
        status(root)
    kw = {"now": clock, "usage": lambda _p: (100, 10, 90), "remote_check": lambda _a, _p: None, "in_use": lambda _p: None, "node": N2,
          "places": RR.Places(root), **kw}
    return R.gc(root, **kw)


def acts(plan: dict, root: Path) -> dict[str, dict]:
    return {os.path.relpath(a["path"], root): a for a in plan["rules"]}


def lines(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


# ---- source trees ---------------------------------------------------------------------------------------------------------------------

def test_a_tree_nothing_names_goes_while_a_live_process_in_another_keeps_it_and_a_young_one_waits(tmp_path):
    root = tmp_path / "ws"
    gone, used, young = tree(root), tree(root), tree(root)
    ago(young, 5)
    live = fake_live(tmp_path, root)
    proc(live.proc, 10, cwd=f"{used}/tools")
    log = RR.Log(tmp_path / "deletions.jsonl", N2, "vy-retention-gc@vy-nebius-2", clock)

    dry = run_gc(root, live=live, log=log)
    assert gone.exists() and not log.path.exists(), "a dry run deletes and logs nothing"
    assert {k: (a["rule"], a["action"]) for k, a in acts(dry, root).items()} == {
        f"research/src/{gone.name}": ("src-tree", "delete"), f"research/src/{used.name}": ("src-tree", "keep")}

    plan = run_gc(root, live=live, log=log, apply=True)
    assert not gone.exists() and used.exists() and young.exists() and os.listdir(root / "research/src/.trash") == []
    a = acts(plan, root)
    assert a[f"research/src/{used.name}"]["reason"] == "named by proc:10(python)"
    assert a[f"research/src/{gone.name}"] == {**a[f"research/src/{gone.name}"], "action": "delete", "owner": "@infra", "error": None,
                                              "reason": RR.DELETES["src-tree"]}
    assert a[f"research/src/{gone.name}"]["files"] > 3 and plan["freed_bytes"] == a[f"research/src/{gone.name}"]["bytes"]
    assert [(x["action"], x["path"], x["rule"], x["by"], x["node"], x["t"]) for x in lines(log.path)] == [
        ("kept", str(used), "src-tree", "vy-retention-gc@vy-nebius-2", N2, "2031-01-01T00:00:00Z"),
        ("deleted", str(gone), "src-tree", "vy-retention-gc@vy-nebius-2", N2, "2031-01-01T00:00:00Z")]
    assert all(x["schema"] == RR.LOG_SCHEMA and x["owner"] == "@infra" for x in lines(log.path))


def test_a_real_process_working_in_a_tree_keeps_it(tmp_path):
    root = tmp_path / "ws"
    t = tree(root)
    child = subprocess.Popen(["python3", "-c", "import sys; sys.stdin.read()"], cwd=t, stdin=subprocess.PIPE)
    try:
        plan = run_gc(root, live=Live(research=root / "research", fill=root / "pouw/fill", kube=None, pids=[child.pid], units=()), apply=True)
    finally:
        child.stdin.close()
        child.wait()
    assert t.exists() and acts(plan, root)[f"research/src/{t.name}"]["reason"].startswith(f"named by proc:{child.pid}(")


def test_a_tree_an_installed_unit_or_a_held_fill_job_names_stays_while_nothing_runs_from_it(tmp_path):
    """node 2's vy-cluster-agent.service runs from research/src/1253f09e... and is often stopped (its STOP file, exit 2, the start
    limit, 15 s between restarts), when no process names the tree; deleting it then keeps the agent from starting again."""
    root = tmp_path / "ws"
    agent, held, done = tree(root), tree(root), tree(root)
    units = tmp_path / "etc/systemd/system"
    units.mkdir(parents=True)
    (units / "vy-cluster-agent.service").write_text(
        f"[Service]\nUser=research\nWorkingDirectory={agent}\nEnvironment=PYTHONPATH={agent}/tools/cluster/src PYTHONDONTWRITEBYTECODE=1\n"
        f"ExecStartPre=/usr/bin/test -f {agent}/READY.json\nExecStart=/usr/bin/python3 -m cluster agent --mode live\n")
    fill = status(root)
    for d, t in (("held-overnight", held), ("done", done)):
        (fill / d).mkdir()
        (fill / d / "proofs-pn2g.sh").write_text(f"#!/bin/bash\ncd {t} && python -m bench\n")
    for node in (N2, N1):
        a = acts(run_gc(root, live=fake_live(tmp_path, root, units=(units,)), apply=True, node=node), root)
        assert agent.exists() and a[f"research/src/{agent.name}"]["reason"] == f"named by unit:{units}/vy-cluster-agent.service"
        assert held.exists() and a[f"research/src/{held.name}"]["reason"] == "named by fill:held-overnight/proofs-pn2g.sh"
        assert not done.exists(), "a finished job names nothing"


def test_node_1_keeps_a_tree_node_2_names_and_refuses_when_node_2_cant_say(tmp_path, monkeypatch):
    """node 2's custody push loop runs node 1's research/src/<sha> over ssh (CUSTODY_SRC_N1, else push.sh's SRC_N1 default), so
    only node 2 sees what pins it; gc's own Live on node 1 asks node 2 (RR.peers: `research retention names` there, as root)."""
    root = tmp_path / "ws"
    pushed, gone = tree(root), tree(root)
    asked = []

    def node_2(patterns):
        asked.append(set(patterns))
        return {pushed.name: [f"{N2}:proc:4242(bash)"]}

    plan = run_gc(root, node=N1, live=fake_live(tmp_path, root, peers=[node_2]), apply=True)
    assert pushed.exists() and not gone.exists() and len(asked) == 2 and {pushed.name, gone.name} <= asked[0]
    assert acts(plan, root)[f"research/src/{pushed.name}"]["reason"] == f"named by {N2}:proc:4242(bash)"

    def unreachable(patterns):
        raise LivenessError(f"can't ask {N2} what it names (vy-n2: ssh: connect to host vy-n2 port 22: Connection refused)")

    plan = run_gc(root, node=N1, live=fake_live(tmp_path, root, peers=[unreachable]), apply=True)
    assert pushed.exists() and plan["refused"].startswith(f"can't establish what live work names (can't ask {N2} what it names")

    [peer] = RR.peers(N1)
    assert (peer.node, peer.host, peer.remote[:2], peer.remote[-6:]) == (N2, "vy-n2", ("sudo", "-n"),
                                                                         ("retention", "names", "--json", "--node", N2, "-"))
    assert RR.peers(N2) == ()
    seen = []
    monkeypatch.setattr(R, "Live", lambda **kw: seen.append(kw) or fake_live(tmp_path, root))
    run_gc(root, node=N1)
    assert seen[0]["peers"] == RR.peers(N1)


def test_node_1_keeps_its_newest_complete_tree_which_node_2_publishes_with(tmp_path):
    """n2_custody.sh takes node 1's newest tree holding the store's attempt.py anew each round, in a shell variable no scan sees."""
    root = tmp_path / "ws"
    older, newest, partial = tree(root, attempt=True), tree(root, attempt=True), tree(root)
    for t, h in ((older, 30), (newest, 20), (partial, 10)):
        ago(t, h)
    plan = run_gc(root, node=N1, live=fake_live(tmp_path, root), apply=True)
    assert newest.exists() and not older.exists() and not partial.exists()
    assert acts(plan, root)[f"research/src/{newest.name}"] == {**acts(plan, root)[f"research/src/{newest.name}"], "action": "keep",
                                                               "rule": "src-tree", "reason": RR.NEWEST}
    run_gc(root, live=fake_live(tmp_path, root), apply=True)
    assert not newest.exists(), "node 2 publishes with node 1's tree, not its own"


def test_names_answers_what_this_nodes_live_work_names_and_refuses_when_it_cant_tell(tmp_path, monkeypatch, capsys):
    root, sha = tmp_path / "ws", "ab" * 20
    root.mkdir()
    scan = Scripted({sha: ["proc:7(bash)"]}, {}, LivenessError("can't read /proc/9: Permission denied"))
    monkeypatch.setattr(R, "utc_now", clock)
    monkeypatch.setattr(R, "Live", lambda **kw: scan)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"/w/x\n\n{sha}\n"))
    assert research_main(["retention", "names", "--root", str(root), "--node", N1, "--json", "-", "/w/y"]) == 1
    assert json.loads(capsys.readouterr().out) == {"schema": "research/retention-names/v1", "node": N1, "now": "2031-01-01T00:00:00Z",
                                                   "patterns": 3, "names": {sha: ["proc:7(bash)"]}}
    assert scan.asked == [sorted(["/w/x", sha, "/w/y"])]
    assert research_main(["retention", "names", "--root", str(root), "--node", N1, sha]) == 0
    assert f"0 of 1 pattern(s) named on {N1}" in capsys.readouterr().out
    assert research_main(["retention", "names", "--root", str(root), "--node", N1, "--json", sha]) == 3
    assert json.loads(capsys.readouterr().out)["refused"] == "can't establish what live work names (can't read /proc/9: Permission denied)"
    assert research_main(["retention", "names", "--root", str(root), "--node", N2, "--json", sha]) == 3
    assert json.loads(capsys.readouterr().out)["refused"].startswith(f"{root}/pouw/fill is missing on {N2}")


def test_a_tree_named_after_its_rename_is_put_back_and_one_that_cant_be_is_stuck_until_the_next_pass(tmp_path):
    root = tmp_path / "ws"
    named, gone = tree(root), tree(root)
    live = Scripted({}, {named.name: ["proc:9(lake)"]})
    plan = run_gc(root, live=live, apply=True)
    assert named.exists() and not gone.exists() and (named / "README.md").exists()
    assert live.asked[1] == sorted([named.name, gone.name]), "the check after the renames looks for every renamed tree"
    assert acts(plan, root)[f"research/src/{named.name}"]["reason"] == "named after the rename: proc:9(lake)"

    def reshipped():
        named.mkdir()
        (named / "READY.json").write_text("{}")
        return {named.name: ["request:r20310101-a"]}

    plan = run_gc(root, live=Scripted({}, reshipped), apply=True)
    stuck = acts(plan, root)[f"research/src/{named.name}"]
    assert stuck["action"] == "stuck" and "can't be put back" in stuck["reason"] and f"a new {named} is there" in stuck["reason"]
    [left] = (root / "research/src/.trash").iterdir()
    assert (left / "README.md").exists()

    plan = run_gc(root, live=Scripted({}), apply=True)
    a = acts(plan, root)
    assert not left.exists() and a[os.path.relpath(left, root)]["rule"] == "trash"
    assert named.exists() and a[f"research/src/{named.name}"]["reason"].startswith("can't tell what it holds besides its commit")


def test_a_check_after_the_renames_that_fails_puts_everything_back_and_refuses(tmp_path):
    root = tmp_path / "ws"
    a, b = tree(root), tree(root)
    plan = run_gc(root, live=Scripted({}, LivenessError("kubectl get pods exited 1: refused")), apply=True)
    assert a.exists() and b.exists() and os.listdir(root / "research/src/.trash") == []
    assert plan["refused"].startswith("can't establish what live work names after the renames (kubectl get pods exited 1: refused)")
    assert {x["reason"] for x in plan["rules"]} == {f"put back undecided: {plan['refused']}"}


def test_liveness_it_cant_establish_refuses_before_any_deletion(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    t = tree(root)
    (root / "research/src/.trash/old-1").mkdir(parents=True)
    live = fake_live(tmp_path, root)
    unreadable = proc(live.proc, 6)
    os.rmdir(unreadable / "fd")
    (unreadable / "fd").write_text("")
    plan = run_gc(root, live=live, apply=True)
    assert plan["refused"].startswith("can't establish what live work names (can't read") and plan["rules"] == []
    assert t.exists() and (root / "research/src/.trash/old-1").exists()

    def failing(kind):
        raise LivenessError(f"kubectl get {kind} exited 1")

    assert run_gc(root, live=fake_live(tmp_path / "k", root, kube=failing), apply=True)["refused"].startswith(
        "can't establish what live work names (kubectl get pods exited 1)")
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    monkeypatch.setattr(R, "Live", lambda **kw: live)
    monkeypatch.setattr(RR, "LOCKS", (tmp_path / "gc.lock",))
    assert research_main(["retention", "gc", "--root", str(root), "--apply", "--by", "t", "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["refused"].startswith("can't establish") and t.exists()


def test_a_tree_holding_a_runs_output_waits_for_its_owner_and_an_unpreserved_run_is_never_deleted(tmp_path):
    root = tmp_path / "ws"
    t = tree(root)
    (t / "results/r1").mkdir(parents=True)
    (t / "results/r1/out.json").write_text("{}")
    run = root / "research/runs/r20300101-a"
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({"ownership": {"lane": "proofs"}}))
    (run / "out.bin").write_bytes(b"r" * 1000)
    (run / CU.MARKER).write_text(json.dumps({"schema": CU.SCHEMA, "run_id": run.name, "preserved": False, "host": N2}))
    plan = run_gc(root, live=fake_live(tmp_path, root), apply=True, force=True, low_pct=0, usage=lambda _p: (100, 95, 5))
    assert t.exists() and run.exists() and (run / "out.bin").exists()
    assert acts(plan, root)[f"research/src/{t.name}"]["reason"] == (
        "1 file(s) outside its commit, e.g. results/r1/out.json: they may be a run's output, so its owner decides")
    assert {a["path"]: a["action"] for a in plan["actions"]}["research/runs"] == "skip"
    assert not any(a["path"].startswith("research/src") for a in plan["actions"]), "the rules decide source trees, not the records"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_a_tree_git_cant_fully_read_is_kept_and_the_log_says_why(tmp_path):
    root = tmp_path / "ws"
    t = tree(root)
    (t / "out").mkdir()
    (t / "out/result.json").write_text("{}")
    os.chmod(t / "out", 0)
    log = RR.Log(tmp_path / "deletions.jsonl", N1, "t", clock)
    try:
        plan = run_gc(root, node=N1, live=Scripted({}), apply=True, log=log)
    finally:
        os.chmod(t / "out", 0o755)
    assert (t / "out/result.json").exists()
    why = f"can't tell what it holds besides its commit: out/ can't be read by uid {os.getuid()}, as whom git runs"
    assert acts(plan, root)[f"research/src/{t.name}"]["reason"].startswith(why)
    assert [(x["action"], x["reason"][:len(why)]) for x in lines(log.path)] == [("kept", why)]


# ---- proofs' audit trees, Lean dependencies -----------------------------------------------------------------------------------------------

def test_proofs_audit_trees_go_on_node_1_only_once_their_lean_audit_json_is_saved(tmp_path):
    root = tmp_path / "ws"
    a, b = tree(root), tree(root)
    for t in (a, b):
        (t / LA).write_text('{"audited": true}\n')
    saved = []

    def save(trees):
        saved.append(sorted(t.name for t in trees))
        assert all((t / LA).read_text() == '{"audited": true}\n' for t in trees), "saved before the rename"
        return "art:" + "e" * 64

    dry = acts(run_gc(root, node=N1, live=fake_live(tmp_path, root), save=save), root)
    assert saved == [] and dry[f"research/src/{a.name}"]["reason"] == "would save its lean-audit.json to the store, then delete it"
    assert dry[f"research/src/{a.name}"]["owner"] == "@proofs" and dry[f"research/src/{a.name}"]["rule"] == "audit-tree"
    plan = run_gc(root, node=N1, live=fake_live(tmp_path, root), save=save, apply=True)
    assert saved == [sorted([a.name, b.name])] and not a.exists() and not b.exists()
    assert acts(plan, root)[f"research/src/{a.name}"]["reason"] == f"its lean-audit.json is saved as art:{'e' * 64}, and nothing live names its commit"

    c = tree(root)
    (c / LA).write_text('{"audited": true}\n')

    def no_remote(trees):
        raise RR.SaveError("no store remote is configured here")

    plan = run_gc(root, node=N1, live=fake_live(tmp_path, root), save=no_remote, apply=True)
    assert c.exists() and acts(plan, root)[f"research/src/{c.name}"]["reason"] == (
        "its lean-audit.json is not saved, so it stays: no store remote is configured here")
    plan = run_gc(root, live=fake_live(tmp_path, root), save=save, apply=True)
    assert c.exists() and acts(plan, root)[f"research/src/{c.name}"]["reason"].startswith(f"1 file(s) outside its commit, e.g. M {LA}")


def test_the_default_saver_puts_nothing_without_a_remote_and_preserves_the_files_with_one(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    t = tree(root)
    (t / LA).write_text('{"audited": true}\n')
    with pytest.raises(RR.SaveError, match="no store remote is configured here"):
        RR.store_saver(str(tmp_path / "store"), N1, "t")([t])
    assert not (tmp_path / "store").exists()
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    art = RR.store_saver(str(tmp_path / "store"), N1, "vy-retention-gc@vy-nebius-1")([t])
    man = LocalStore(tmp_path / "store").get_manifest(art)
    assert man.kind == "evidence/v1" and man.meta["trees"] == [t.name] and man.meta["lane"] == "proofs"
    assert [f.path for f in man.payload.files] == [f"{t.name}/{LA}"]


def test_lean_dependencies_go_at_any_age_from_a_tree_that_stays_unless_something_names_the_tree(tmp_path):
    root = tmp_path / "ws"
    young, named = tree(root, lake=True), tree(root, lake=True)
    ago(young, 1)
    ago(named, 1)
    left = young / RR.LAKE[0] / ".lake/packages.rs-trash-5"
    left.mkdir()
    live = fake_live(tmp_path, root)
    proc(live.proc, 30, fds=(f"{named}/{RR.LAKE[2]}/.lake/packages/mathlib/A.olean",))
    plan = run_gc(root, node=N1, live=live, apply=True)
    assert young.exists() and (young / "README.md").exists() and not left.exists()
    assert not any((young / k / ".lake/packages").exists() for k in RR.LAKE)
    assert not list(young.glob("**/packages.rs-trash-*"))
    assert all((named / k / ".lake/packages/mathlib/A.olean").exists() for k in RR.LAKE)
    a = acts(plan, root)
    assert {a[f"research/src/{young.name}/{k}/.lake/packages"]["rule"] for k in RR.LAKE} == {"lean-deps"}
    assert a[f"research/src/{named.name}/{RR.LAKE[0]}/.lake/packages"]["reason"] == "named by proc:30(python)"
    plan = run_gc(root, live=fake_live(tmp_path, root), apply=True)
    assert all((named / k / ".lake/packages").exists() for k in RR.LAKE), "node 1's rule only"


# ---- job trees, scratch, records --------------------------------------------------------------------------------------------------------

def jobs(root: Path) -> Path:
    j = root / "jobs/src"
    for name in ("pod-dead", "pod-alive", "pod-young", "0123456789abcdef", "fedcba9876543210", "by-pod"):
        (j / name).mkdir(parents=True)
        (j / name / "README.md").write_text("x")
    (j / "0123456789abcdef/.copied").write_text("")
    (j / "by-pod/dead").write_text(f"{j}/0123456789abcdef\n")
    ago(j / "pod-young", -1)
    return j


def test_job_trees_go_on_node_1_when_no_live_pod_reads_them(tmp_path):
    root = tmp_path / "ws"
    j = jobs(root)
    plan = run_gc(root, node=N1, live=fake_live(tmp_path, root, kube=pods("alive")), apply=True)
    assert sorted(os.listdir(j)) == [".trash", "by-pod", "fedcba9876543210", "pod-alive", "pod-young"]
    assert os.listdir(j / ".trash") == []
    a = acts(plan, root)
    assert {k: (x["action"], x["reason"]) for k, x in a.items()} == {
        "jobs/src/pod-dead": ("delete", RR.DELETES["jobs-src"]), "jobs/src/0123456789abcdef": ("delete", RR.DELETES["jobs-src"]),
        "jobs/src/pod-alive": ("keep", "live pod alive"),
        "jobs/src/fedcba9876543210": ("keep", "no .copied, so job_tree.sh may be remaking it")}
    assert run_gc(root, node=N1, live=fake_live(tmp_path, root))["refused"].startswith("can't tell which pods are live (no k3s here)")


def test_a_job_tree_a_pod_takes_during_the_pass_is_put_back(tmp_path):
    root = tmp_path / "ws"
    j = jobs(root)

    def pod_starts():
        (j / "by-pod/late").write_text(f"{j}/0123456789abcdef\n")
        os.utime(j / "by-pod/late", (T + 10, T + 10))
        return {}

    plan = run_gc(root, node=N1, live=Scripted({}, pod_starts, kube=pods()), apply=True)
    assert (j / "0123456789abcdef/.copied").exists() and not (j / "pod-dead").exists()
    assert acts(plan, root)["jobs/src/0123456789abcdef"]["reason"] == "named after the rename: by-pod written since the rename: late"


def test_check_scratch_untouched_for_2_h_goes_unless_something_names_it_or_it_is_protected(tmp_path):
    root, pt, cache = tmp_path / "ws", tmp_path / "pytest-of-research", tmp_path / "cache"
    for d in ("pytest-1", "pytest-12", "pytest-2", "pytest-3", "pytest-5", "pytest-x"):
        (pt / d / "sub").mkdir(parents=True)
        (pt / d / "sub/f.txt").write_text("s")
    ago(pt / "pytest-12/sub/f.txt", 1)
    (pt / "pytest-5/kept.retention.json").write_text("{}")
    (cache / "lean-audit-scratch-a").mkdir(parents=True)
    (cache / "lean-deps/pkg").mkdir(parents=True)
    places = RR.Places(root, ((pt, "pytest-[0-9]*"), (cache, "lean-audit-scratch-*")), (pt / "pytest-3", cache / "lean-deps"))
    root.mkdir()
    live = fake_live(tmp_path, root)
    proc(live.proc, 40, fds=(f"{pt}/pytest-2/sub/f.txt",))
    plan = run_gc(root, places=places, live=live, apply=True)
    assert sorted(os.listdir(pt)) == ["pytest-12", "pytest-2", "pytest-3", "pytest-5", "pytest-x"]
    assert sorted(os.listdir(cache)) == ["lean-deps"]
    a = {Path(x["path"]).name: (x["action"], x["reason"]) for x in plan["rules"]}
    assert a == {"pytest-1": ("delete", RR.DELETES["check-scratch"]), "lean-audit-scratch-a": ("delete", RR.DELETES["check-scratch"]),
                 "pytest-2": ("keep", "named by proc:40(python)"), "pytest-3": ("keep", "protected: no rule deletes it"),
                 "pytest-5": ("keep", f"{pt}/pytest-5/kept.retention.json is a record, and it decides what beneath it goes")}


def test_a_record_wins_over_a_rule(tmp_path):
    root = tmp_path / "ws"
    t = tree(root)
    R.keep(t, root=root, owner="@proofs", expires=None, priority="keep", now=clock, node=N2)
    plan = run_gc(root, live=fake_live(tmp_path, root), apply=True)
    assert t.exists() and acts(plan, root)[f"research/src/{t.name}"]["reason"] == f"covered by {t}'s record (owner @proofs), which decides"


# ---- refusals ---------------------------------------------------------------------------------------------------------------------------

def test_node_2_runs_only_on_a_fresh_status_whose_first_line_is_quiet(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    t = tree(root)
    fill = root / "pouw/fill"
    unsure = "so a timed window can't be ruled out"

    def refused(**kw) -> str:
        plan = run_gc(root, live=fake_live(tmp_path, root), apply=True, node=N2, **kw)
        assert plan["rules"] == [] and t.exists(), plan["rules"]
        return plan["refused"]

    assert refused().startswith(f"{fill} is missing on {N2}, {unsure}")
    assert run_gc(root, live=fake_live(tmp_path, root), node=N1)["refused"] is None, "node 1 runs no timed windows"
    fill.mkdir(parents=True)
    assert refused().startswith(f"{fill}/status.txt can't be read")
    for line, age_s in ((QUIET, 61), (QUIET.replace("00:00:00Z", "23:58:59Z").replace("2031-01-01", "2030-12-31"), 0),
                        (QUIET.replace("00:00:00Z", "00:00:05Z"), 0), (QUIET, -5)):
        status(root, line, age_s)
        why = refused()
        assert "not both within the last 60 s" in why and unsure in why, (line, age_s)
    status(root, "free GPUs 7/8, timed False, window waiting False\n")
    assert refused().startswith(f"{fill}/status.txt's first line carries no time")
    for line in (QUIET.replace("timed False", "timed True"), QUIET.replace("waiting False", "waiting True"),
                 "2031-01-01T00:00:00Z free GPUs 7/8\nwas timed False, window waiting False\n"):
        status(root, line)
        assert refused().startswith(f"{fill}/status.txt does not say `timed False, window waiting False`"), line
    with pytest.raises(RR.Refused, match="does not say `timed False"):
        R.report(root, now=clock, usage=lambda _p: (100, 10, 90), node=N2)
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    out = tmp_path / "retention/dry-run.json"
    assert research_main(["retention", "gc", "--root", str(root), "--node", N2, "--out", str(out)]) == 3
    assert "refused: " in capsys.readouterr().err and "status.txt" in json.loads(out.read_text())["refused"]
    assert research_main(["retention", "report", "--root", str(root), "--node", N2, "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["refused"].startswith(f"{fill}/status.txt")
    status(root, QUIET, 59)
    assert run_gc(root, live=fake_live(tmp_path, root), apply=True)["refused"] is None and not t.exists()


def test_node_2_refuses_from_15_min_before_a_booked_window_until_it_ends(tmp_path):
    """fill/windows books windows without a GPU lease (node 2's merge checks), which the status never shows; the gc timer's
    06:30 can be a window's first second."""
    root = tmp_path / "ws"
    t = tree(root)
    fill = status(root)
    for booking in ("2030-12-31T23:55Z 10 # merge checks off node 2 until 00:05Z: no GPU booking",
                    "2031-01-01T00:00:00Z # 30 min by default", "2031-01-01T00:15Z 190 # circuits TP8, all 8 GPUs",
                    "2031-01-01T00:00:00+00:00 1", "2031-01-01T00:00 5 # no zone: UTC", "tomorrow 30 # can't parse"):
        (fill / "windows").write_text(f"# old\n\n2030-12-30T06:30Z 190\n{booking}\n")
        why = run_gc(root, live=fake_live(tmp_path, root), apply=True)["refused"]
        assert t.exists() and why and ("has a line gc can't read" in why and "tomorrow 30" in why if "tomorrow" in booking else
                                       f"books {booking.strip()!r}: nothing runs on this node from 15 min before" in why), why
    (fill / "windows").write_text("2030-12-31T23:50Z 10\n2031-01-01T00:16Z 60\n")
    assert run_gc(root, live=fake_live(tmp_path, root), apply=True)["refused"] is None and not t.exists()


@pytest.mark.parametrize("unit", ["vy-retention-gc@.service", "vy-retention-dry-run@.service"])
def test_the_units_start_on_node_2_only_on_a_fresh_status_whose_first_line_is_quiet(tmp_path, unit):
    """The ExecCondition, run by /bin/sh with node 2's fill dir moved to tmp: no `$` (systemd would expand it), and a status
    dated 2100 stands for a fresh one, since `find -mmin -1` takes a future mtime and the test reads no clock."""
    [cond] = re.findall(r"^ExecCondition=/bin/sh -c '(.*)'$", (UNITS / unit).read_text(), re.M)
    fill = tmp_path / "fill"
    assert "$" not in cond

    def starts(node: str) -> bool:
        return subprocess.run(["/bin/sh", "-c", cond.replace("/workspace/pouw/fill", str(fill)).replace("%i", node)]).returncode == 0

    assert not starts(N2) and starts(N1)
    fill.mkdir()
    for line, mtime, ok in ((QUIET, 0, False), (QUIET.replace("timed False", "timed True"), 4102444800, False),
                            ("t\n" + QUIET, 4102444800, False), (QUIET, 4102444800, True)):
        (fill / "status.txt").write_text(line)
        os.utime(fill / "status.txt", (mtime, mtime))
        assert starts(N2) is ok and starts(N1), (line, mtime)


def test_a_window_seen_mid_pass_stops_the_next_deletion_and_puts_back_what_is_aside(tmp_path, monkeypatch):
    """The guard is read again after report's walk, before the records pass and before every deletion: the rules', the
    trash's, and the records'."""
    root = tmp_path / "ws"
    a, b = tree(root), tree(root)
    (root / "research/src/.trash/old-1").mkdir(parents=True)
    fill = status(root)
    log = RR.Log(tmp_path / "deletions.jsonl", N2, "t", clock)

    def window_opens():
        status(root, QUIET.replace("timed False", "timed True"))
        return {}

    plan = run_gc(root, live=Scripted({}, window_opens), apply=True, log=log)
    assert a.exists() and b.exists() and os.listdir(root / "research/src/.trash") == [], "deleted nothing more, put both back"
    assert plan["refused"].endswith("(seen during the pass: what is renamed aside goes back, and nothing more is deleted)")
    assert [(x["action"], x["reason"].split(":")[0]) for x in lines(log.path)] == [
        ("deleted", "an interrupted pass left it, and nothing live names it"), ("kept", "put back undecided"),
        ("kept", "put back undecided")]

    status(root)
    rec = root / "data/old"
    rec.mkdir(parents=True)
    R.keep(rec, root=root, owner="@infra", expires="2030-01-01T00:00:00Z", priority="low", regenerate="make it", now=clock, node=N2)

    def in_use(_p):
        status(root, QUIET, 61)
        return None

    plan = run_gc(root, live=Scripted({}, {}), apply=True, force=True, low_pct=0, in_use=in_use)
    assert rec.exists() and plan["refused"].endswith("(seen during the records pass: nothing more is deleted)")
    assert not a.exists() and not b.exists(), "the rules ran first"

    orig = R.scan

    def scan(*args, **kw):
        status(root, QUIET.replace("waiting False", "waiting True"))
        return orig(*args, **kw)

    status(root)
    monkeypatch.setattr(R, "scan", scan)
    with pytest.raises(RR.Refused, match=r"window waiting False`.*\(seen after the walk\)$"):
        R.report(root, now=clock, usage=lambda _p: (100, 10, 90), node=N2)


def test_apply_needs_the_locks_and_refuses_while_another_pass_holds_one(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    monkeypatch.setattr(RR, "LOCKS", (tmp_path / "gc.lock", tmp_path / "steward.lock"))
    with open(tmp_path / "steward.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert research_main(["retention", "gc", "--root", str(root), "--apply", "--by", "t"]) == 3
        assert f"{tmp_path / 'steward.lock'} is held" in capsys.readouterr().err
    assert research_main(["retention", "gc", "--root", str(root), "--apply", "--by", "t"]) == 0


# ---- the report -------------------------------------------------------------------------------------------------------------------------

def test_the_report_shows_the_rules_paths_and_asks_the_owner_of_each_big_unowned_area(tmp_path, monkeypatch, capsys):
    root, pt = tmp_path / "ws", tmp_path / "pytest-of-research"
    t = tree(root)
    for rel, job in (("research/runs/r1", {"ownership": {"lane": "circuits"}}), ("research/runs/r2", {"store": {"campaign": "pouw-c"}}),
                     ("research/runs/r3", {})):
        (root / rel).mkdir(parents=True)
        (root / rel / "job.json").write_text(json.dumps(job))
    for rel in ("hf/models--m/blob", "cache/uv/wheel", "jobs/cov/bundle-1/b.bin", "research/runs/r3/out.bin"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"w" * 2048)
    (pt / "pytest-1").mkdir(parents=True)
    places = RR.Places(root, ((pt, "pytest-[0-9]*"),))
    rep = R.report(root, now=clock, usage=lambda _p: (100, 10, 90), node=N1, places=places, ask_gib=1e-6)
    rows = {r["path"]: r for r in rep["rows"]}
    assert rep["schema"] == "research/retention-report/v1"
    src = rows[f"research/src/{t.name}"]
    assert (src["source"], src["rule"], src["owner"], src["state"]) == ("rule", "src-tree", "@infra", "expired")
    assert src["record"]["expires"] == R.fmt_utc(datetime.fromtimestamp(os.lstat(t).st_mtime, timezone.utc) + RR.RULES["src-tree"].age)
    assert rows[str(pt / "pytest-1")]["rule"] == "check-scratch"
    assert {a["path"]: (a["owners"], a["what"]) for a in rep["asks"]} == {
        "hf": (list(R.WEIGHTS), "weights"), "cache": (["@infra"], "not in the policy: infra finds its owner"),
        "jobs": (["@infra"], "not in the policy: infra finds its owner"),
        "research/runs": (["@circuits", "@infra", "@pouw-c"], "run dirs")}
    (root / "jobs/src").mkdir()
    (root / "jobs/src/pod-a").mkdir()
    rep = R.report(root, now=clock, usage=lambda _p: (100, 10, 90), node=N1, places=places, ask_gib=1e-6)
    assert {a["path"]: a["owners"] for a in rep["asks"]}["jobs/cov"] == ["@circuits"]
    assert {r["path"]: r["rule"] for r in rep["rows"]}["jobs/src/pod-a"] == "jobs-src"

    monkeypatch.setattr(R, "disk_usage", lambda _p: (100, 10, 90))
    assert research_main(["retention", "report", "--root", str(root), "--node", N1, "--ask-gib", "0.000001"]) == 1
    out = capsys.readouterr().out
    assert f"ASK {' '.join(R.WEIGHTS)}: hf, 2.0 KiB unowned (weights)" in out and f"research/src/{t.name} (rule src-tree)" in out
    assert research_main(["retention", "report", "--root", str(root), "--node", N1]) == 0
