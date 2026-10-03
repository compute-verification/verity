"""research.retention_rules through `research retention gc` and `report`: the resource steward's delete-without-asking policy as
default rules. Fixture trees in tmp dirs (real git checkouts as shipped source trees), a fake /proc, fake kubectl, a scripted
liveness for the checks after a rename, and a clock in 2031, so every file a test writes is old unless the test dates it."""
from __future__ import annotations

import fcntl
import itertools
import json
import os
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


def tree(root: Path, lake: bool = False) -> Path:
    """A shipped source tree: a checkout at research/src/<its commit> (results/ ignored), with the verifier's three Lake packages'
    .lake/packages when `lake`."""
    d = root.parent / f"checkout{next(_n)}"
    (d / LA).parent.mkdir(parents=True)
    (d / LA).write_text("{}\n")
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


def fake_live(tmp_path: Path, root: Path, kube=None) -> Live:
    (tmp_path / "proc").mkdir(parents=True, exist_ok=True)
    return Live(research=root / "research", fill=root / "pouw/fill", proc=tmp_path / "proc", kube=kube)


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


def run_gc(root: Path, **kw) -> dict:
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
        plan = run_gc(root, live=Live(research=root / "research", fill=root / "pouw/fill", kube=None, pids=[child.pid]), apply=True)
    finally:
        child.stdin.close()
        child.wait()
    assert t.exists() and acts(plan, root)[f"research/src/{t.name}"]["reason"].startswith(f"named by proc:{child.pid}(")


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
    assert {x["reason"] for x in plan["rules"]} == {"named after the rename: the check after the rename failed: kubectl get pods exited 1: "
                                                   "refused"}


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
    plan = run_gc(root, node=N2, live=fake_live(tmp_path, root), save=save, apply=True)
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
    plan = run_gc(root, node=N2, live=fake_live(tmp_path, root), apply=True)
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

def test_node_2_refuses_in_a_timed_window_and_when_its_status_cant_be_read(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    t = tree(root)
    fill = root / "pouw/fill"
    fill.mkdir(parents=True)
    plan = run_gc(root, live=fake_live(tmp_path, root), apply=True)
    assert plan["refused"].startswith(f"{fill}/status.txt can't be read") and plan["rules"] == [] and t.exists()
    (fill / "status.txt").write_text("2031-01-01T00:00:00Z free GPUs 0/8, timed True, window waiting False\n")
    assert run_gc(root, live=fake_live(tmp_path, root), apply=True)["refused"].startswith(f"{fill}/status.txt does not say timed False")
    with pytest.raises(RR.Refused, match="does not say timed False"):
        R.report(root, now=clock, usage=lambda _p: (100, 10, 90), node=N2)
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    out = tmp_path / "retention/dry-run.json"
    assert research_main(["retention", "gc", "--root", str(root), "--node", N2, "--out", str(out)]) == 3
    assert "refused: " in capsys.readouterr().err and json.loads(out.read_text())["refused"].startswith(f"{fill}/status.txt does not")
    assert research_main(["retention", "report", "--root", str(root), "--json"]) == 3
    assert json.loads(capsys.readouterr().out) == {"schema": R.REPORT_SCHEMA, "refused": json.loads(out.read_text())["refused"]}
    (fill / "status.txt").write_text("2031-01-01T00:00:00Z free GPUs 7/8, timed False, window waiting False\n")

    def window_opens():
        (fill / "status.txt").write_text("2031-01-01T00:00:00Z free GPUs 0/8, timed True, window waiting False\n")
        return {}

    plan = run_gc(root, live=Scripted(window_opens), apply=True)
    assert plan["refused"].endswith("(it changed during this pass: nothing deleted)") and t.exists()
    (fill / "status.txt").write_text("2031-01-01T00:00:00Z free GPUs 7/8, timed False, window waiting False\n")
    assert run_gc(root, live=fake_live(tmp_path, root), apply=True)["refused"] is None and not t.exists()


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
