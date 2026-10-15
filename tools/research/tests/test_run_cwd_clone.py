"""`research run --cwd clone`: the workload runs in its own clone of the source root, never in the tree other runs of the commit share.
Proofs ran one Lean audit twice at one sha with `--cwd source`; cancelling the duplicate ran its EXIT trap, which deleted the shared
`PKG/.lake/packages`, and the live audit failed after 4779 of 4781 targets (3 Oct 2026).  The machine is a directory and ssh is
`sh -c`, as in test_remote_preflight.py."""
from __future__ import annotations

import errno
import importlib.machinery
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from research import clone, gates, remote
from research import runs as R
from research.cli import _clone_imports, main
from research.pythonpath import import_roots_dir, pythonpath
from research.store import attempt as A
from research.telemetry.procs import stat_fields
from waits import Gate, wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
# a member of the workload's group that ignores SIGTERM and keeps writing into its cwd, so the clone keeps filling while it is
# removed; once its cwd is gone it waits for a signal.  It says it is ready (SIGTERM ignored) on stdout.
WRITER = """
import os, signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.write(1, b'r')
i = 0
while True:
    try:
        os.makedirs('busy', exist_ok=True)
        open(f'busy/{i % 1000}', 'w').close()
    except OSError:
        signal.pause()
    i += 1
"""
# leaves a build tree in its cwd, says where it ran, at which commit and where it imported the workspace module `leaky` from (when
# the tree has one, `_workspace`), then ends as argv[1] says: an exit status, or `term`, the SIGTERM to its process group that
# `research cancel` sends, which leaves a WRITER (its pid in seen.json `outliver`) running
PROBE = f"""
import importlib.util, json, os, signal, subprocess, sys
os.makedirs('.lake/packages/dep')
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
mod = importlib.util.find_spec('leaky') and __import__('leaky')
seen = {{'cwd': os.getcwd(), 'head': head, 'module': mod and mod.__file__, 'wrote': mod and str(mod.OUT), 'mark': getattr(mod, 'MARK', None)}}
end = sys.argv[1] if len(sys.argv) > 1 else '0'
if end == 'term':
    child = subprocess.Popen([sys.executable, '-c', {WRITER!r}], stdout=subprocess.PIPE)
    child.stdout.read(1)
    seen['outliver'] = child.pid
json.dump(seen, open(os.path.join(os.environ['RESEARCH_RUN_DIR'], 'seen.json'), 'w'))
os.killpg(0, signal.SIGTERM) if end == 'term' else sys.exit(int(end))
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _gone(pid: int) -> bool:
    st = stat_fields(pid)
    return st is None or st["state"] in ("Z", "X")


@pytest.fixture
def outlivers():
    """The WRITERs a test's runs left; any still running at the end are killed (the old runner left them writing)."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        if not _gone(pid):
            os.kill(pid, signal.SIGKILL)


def _workspace(repo: Path) -> None:
    """A workspace member whose module writes beside its own __file__ when imported, as a plotting module writes its figures."""
    (repo / "pyproject.toml").write_text('[project]\nname = "root"\nversion = "0"\n[tool.uv.workspace]\nmembers = ["m"]\n')
    (repo / "m" / "src" / "leaky").mkdir(parents=True)
    (repo / "m" / "pyproject.toml").write_text('[project]\nname = "m"\nversion = "0"\n')
    (repo / "m" / "src" / "leaky" / "__init__.py").write_text(
        "from pathlib import Path\nOUT = Path(__file__).resolve().parents[2] / 'reports' / 'plot.txt'\nOUT.parent.mkdir()\n"
        "OUT.write_text('figure')\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a workspace member")


def _root_package(d: Path) -> None:
    """A workspace whose root is itself the package `verity`, as this repository's is, beside a member `m`: a machine's PYTHONPATH
    reaches `verity` through the tree's shim (`pythonpath.shim_dir`), never through the root."""
    (d / "verity").mkdir(parents=True)
    (d / "verity" / "__init__.py").write_text("")
    (d / "m" / "src" / "other").mkdir(parents=True)
    (d / "m" / "src" / "other" / "__init__.py").write_text("")
    (d / "m" / "pyproject.toml").write_text('[project]\nname = "m"\nversion = "0"\n')
    (d / "pyproject.toml").write_text('[project]\nname = "verity"\nversion = "0"\n[build-system]\nrequires = ["hatchling"]\n'
                                      '[tool.hatch.build.targets.wheel]\npackages = ["verity"]\n[tool.uv.workspace]\nmembers = ["m"]\n')


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A committed checkout, the invoking directory; local clones go under the test's tmp dir."""
    d = tmp_path / "repo"
    (d / "pkg").mkdir(parents=True)
    (d / "pkg" / "lakefile.toml").write_text("name = 'pkg'\n")
    _git(d, "init", "-q")
    _git(d, "add", ".")
    _git(d, "commit", "-q", "-m", "one")
    monkeypatch.chdir(d)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    return d


@pytest.fixture
def machine(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "machine"
    (tmp_path / "machines.toml").write_text(f'[machines.m]\nprovider = "runpod"\npod_id = "p1"\nproject = "p"\nroot = "{root}"\npython = "{PY}"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "pod_name": "m", "cost_per_hr": 0.1,
                                                         "rate_observed_utc": "t0", "gpu": None})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    return root


def _run(runs: Path, cwd: str, *end: str) -> int:
    return main(["run", "--runs-dir", str(runs), "--no-sampler", "--cwd", cwd, "--", PY, "-c", PROBE, *end])


def _on(tmp_path: Path, repo: Path, cwd: str, *end: str) -> int:
    return main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--source", str(repo), "--no-custody-r2",
                 "--no-sampler", "--cwd", cwd, "--", PY, "-c", PROBE, *end])


def _seen(run_dir: Path) -> dict:
    return json.loads((run_dir / "seen.json").read_text())


def _clones(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "tmp").glob("research-*"))


@pytest.mark.parametrize("end, rc", [("0", 0), ("3", 3), ("term", 143)])
def test_a_local_run_gets_a_clone_at_head_outside_its_run_dir_removed_however_it_ends(tmp_path, repo, outlivers, end, rc, capsys):
    assert _run(tmp_path / "runs", "clone", end) == rc
    [run_dir] = (tmp_path / "runs").iterdir()
    seen = _seen(run_dir)
    if end == "term":
        outlivers.append(seen["outliver"])
        assert _gone(seen["outliver"]), "the group member that ignored SIGTERM is SIGKILLed before the clone is removed"
        assert "1 process(es) of the workload's group" in capsys.readouterr().err
    tree = Path(seen["cwd"])
    assert seen["head"] == _git(repo, "rev-parse", "HEAD") and tree.name == "tree"
    assert tree.parent.parent == (tmp_path / "tmp").resolve() and not tree.is_relative_to(run_dir.resolve())
    assert not tree.parent.exists() and _clones(tmp_path) == [], "removed once the workload exited"
    assert not (repo / ".lake").exists() and _git(repo, "status", "--porcelain") == "", "the checkout is untouched"
    job = json.loads((run_dir / "job.json").read_text())["run"]
    assert job["cwd_basis"] == "clone" and Path(job["cwd"]).resolve() == tree
    assert job["cwd_clone"] == {"root": job["cwd"], "of": str(repo), "commit": seen["head"]}
    s = R.summarize(run_dir)
    assert (s["state"], s["rc"]) == ("done" if rc == 0 else "failed", rc)


def test_clone_sub_runs_in_sub_inside_the_clone(tmp_path, repo, capsys):
    assert _run(tmp_path / "runs", "clone/pkg") == 0
    [run_dir] = (tmp_path / "runs").iterdir()
    cwd = Path(_seen(run_dir)["cwd"])
    assert (cwd.name, cwd.parent.name) == ("pkg", "tree") and not cwd.parent.exists()
    assert json.loads((run_dir / "job.json").read_text())["run"]["cwd_basis"] == "clone" and not (repo / "pkg" / ".lake").exists()
    head = _git(repo, "rev-parse", "HEAD")
    for sub in ("absent", ".."):
        assert _run(tmp_path / "refused", f"clone/{sub}") == 3
        assert f"refused: --cwd clone/{sub}: {sub} is not a directory of commit {head}" in capsys.readouterr().err
    assert _clones(tmp_path) == [] and not list((tmp_path / "refused").glob("r*"))


def test_a_source_root_that_cannot_be_cloned_is_refused_naming_cwd_source(tmp_path, repo, monkeypatch, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert _run(tmp_path / "runs", "clone") == 3
    err = capsys.readouterr().err
    assert f"refused: --cwd clone: {plain} is not the top level of a git checkout" in err and "pass --cwd source" in err
    monkeypatch.chdir(repo / "pkg")
    assert _run(tmp_path / "runs", "clone") == 3
    assert f"(it is inside {repo.resolve()})" in capsys.readouterr().err
    monkeypatch.chdir(repo)
    (repo / "new.py").write_text("x = 1\n")
    assert _run(tmp_path / "runs", "clone") == 3
    err = capsys.readouterr().err
    assert "has uncommitted or untracked files, which a clone of HEAD would not carry" in err and "--cwd source" in err
    assert not list((tmp_path / "runs").glob("r*")) and _clones(tmp_path) == [], "refused before a run dir; no clone left"


def test_untracked_files_count_whatever_status_shows(tmp_path, repo, capsys):
    """`status.showUntrackedFiles=no` hid an untracked file from `git status --porcelain`, so a checkout the clone would not
    match passed as clean; `ship_source` read the same."""
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "pkg" / "new.py").write_text("x = 1\n")
    assert _run(tmp_path / "runs", "clone") == 3
    assert "has uncommitted or untracked files" in capsys.readouterr().err and _clones(tmp_path) == []
    with pytest.raises(remote.Refused, match="has uncommitted changes"):
        remote.ship_source(["false"], str(tmp_path / "machine"), repo)


def test_a_clone_merges_lean_audit_json_with_the_record_driver(tmp_path, repo):
    """A clone's config starts empty, so a workload that merged in it merged a `lean-audit.json` line by line."""
    clone.make(repo, tmp_path / "c")
    assert _git(tmp_path / "c", "config", "merge.lean-audit.driver") == gates.RECORD_DRIVER["merge.lean-audit.driver"]


def test_the_callers_git_dir_does_not_point_the_clone_at_the_source(tmp_path, repo, monkeypatch):
    """GIT_DIR in the caller's environment (a git hook, a wrapper) made the clone's `checkout --detach` act on the source."""
    branch = _git(repo, "symbolic-ref", "-q", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(repo / ".git"))
    assert _run(tmp_path / "runs", "clone") == 0
    monkeypatch.delenv("GIT_DIR")
    assert _git(repo, "symbolic-ref", "-q", "HEAD") == branch and _git(repo, "status", "--porcelain") == ""
    assert _clones(tmp_path) == []


def test_a_clone_that_cannot_be_removed_is_reported_and_the_run_keeps_its_status(tmp_path, repo, capsys):
    """The workload replaces the clone's directory with a symlink to a directory of the test's: removing it must neither follow
    the link nor fail the run."""
    victim = tmp_path / "victim"
    (victim / "keep").mkdir(parents=True)
    wl = (f"import os, shutil; home = os.path.dirname(os.getcwd()); os.chdir('/'); shutil.rmtree(home); "
          f"os.symlink({str(victim)!r}, home)")
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--cwd", "clone", "--", PY, "-c", wl]) == 0
    [left] = _clones(tmp_path)
    err = capsys.readouterr().err
    assert f"could not remove {left} (it is a symbolic link); remove it by hand" in err
    assert left.is_symlink() and (victim / "keep").is_dir()
    [run_dir] = (tmp_path / "runs").iterdir()
    assert R.summarize(run_dir)["state"] == "done"


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes a read-only directory's entries anyway")
def test_a_clone_the_workload_left_read_only_is_removed(tmp_path, repo, capsys):
    wl = ("import os; os.makedirs('ro/x'); open('ro/x/f', 'w').close(); os.chmod('ro/x', 0o555); os.chmod('ro', 0o555); "
          "os.makedirs('nr/y'); open('nr/y/f', 'w').close(); os.chmod('nr', 0o000); "
          "os.makedirs('nx'); open('nx/f', 'w').close(); os.chmod('nx', 0o600); os.chmod('.', 0o555)")
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--cwd", "clone", "--", PY, "-c", wl]) == 0
    assert "could not remove" not in capsys.readouterr().err and _clones(tmp_path) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes a read-only directory's entries anyway")
@pytest.mark.filterwarnings("ignore:onerror argument is deprecated:DeprecationWarning")
@pytest.mark.parametrize("python", ["this", "3.11"])
def test_remove_makes_an_unreadable_clone_directory_writable(tmp_path, monkeypatch, python):
    if python == "3.11":
        monkeypatch.setattr(clone, "sys", type("Sys", (), {"version_info": (3, 11)}))   # rmtree's onerror, before onexc
    (tmp_path / "d" / "ro").mkdir(parents=True)
    (tmp_path / "d" / "ro" / "f").write_text("")
    (tmp_path / "d" / "ro").chmod(0o555)
    (tmp_path / "d").chmod(0o000)
    assert clone.remove(tmp_path / "d") is None and not (tmp_path / "d").exists()


def test_an_unexpected_error_before_the_workload_still_removes_the_clone(tmp_path, repo, monkeypatch):
    import research.cli as C

    def fails(*a, **k):
        raise ValueError("a store preparation error")

    monkeypatch.setattr(C, "_prepare_store", fails)
    with pytest.raises(ValueError):
        _run(tmp_path / "runs", "clone")
    assert _clones(tmp_path) == []


def test_what_stops_a_clone_being_made_is_a_refusal(tmp_path, repo, monkeypatch):
    (tmp_path / "file").write_text("")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "file"))
    with pytest.raises(clone.Refused, match="could not create a directory for the clone .*pass --cwd source"):
        clone.home(tmp_path / "runs", "r1", machine=False)
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "scratch").write_text("")
    with pytest.raises(clone.Refused, match="could not create a directory for the clone"):
        clone.home(tmp_path / "root" / "runs", "r1", machine=True)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    with pytest.raises(clone.Refused, match="could not run git rev-parse"):
        clone.make(repo, tmp_path / "c")


def test_a_machine_that_cannot_make_the_clone_directory_refuses_the_run(tmp_path, repo):
    root = tmp_path / "root"
    sha = _git(repo, "rev-parse", "HEAD")
    subprocess.run(["git", "clone", "-q", "--shared", str(repo), str(root / "src" / sha)], check=True)
    (root / "scratch").write_text("")
    out = remote.launch_request(_request(root, root / "src" / sha, sha, "r-x"))
    assert not out["started"] and out["fault"] == "CloneRefused", out          # refused before its first record: never `launched`
    wait_runner(root / "runs" / "r-x")
    refused = json.loads((root / "runs" / "r-x" / "refused.json").read_text())
    assert refused["fault"] == "CloneRefused" and "could not create a directory for the clone (NotADirectoryError" in refused["reason"]


def test_remove_says_why_it_could_not(tmp_path):
    (tmp_path / "d" / "x").mkdir(parents=True)
    (tmp_path / "f").write_text("x")
    assert clone.remove(tmp_path / "d") is None and not (tmp_path / "d").exists()
    assert clone.remove(tmp_path / "absent") is None
    assert clone.remove(tmp_path / "f").startswith("NotADirectoryError")


def test_remove_tries_again_while_the_clone_is_still_filling(tmp_path, monkeypatch):
    (tmp_path / "d" / "x").mkdir(parents=True)
    (tmp_path / "e").mkdir()
    real, calls = shutil.rmtree, []

    def filling(path, *a, **k):
        calls.append(path)
        if len(calls) == 1 or path == tmp_path / "e":
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", filling)
    assert clone.remove(tmp_path / "d") is None and len(calls) == 2 and not (tmp_path / "d").exists()
    why = clone.remove(tmp_path / "e")
    assert why.startswith("OSError: [Errno 39] Directory not empty") and why.endswith(f"after {clone.REMOVE_ATTEMPTS} attempts")


def test_a_machine_clone_first_removes_what_finished_runs_left_in_scratch_and_never_a_live_runs(tmp_path, capsys):
    root = tmp_path / "root"
    exited = subprocess.Popen([PY, "-c", "pass"])
    exited.wait()
    gate = Gate(tmp_path / "gate")
    member = subprocess.Popen([PY, "-c", Gate.WAIT, str(gate.path)], start_new_session=True)   # a workload group still running
    victim = tmp_path / "victim"
    (victim / "keep").mkdir(parents=True)
    runs = {"done": (exited.pid, {"state": "done", "pgid": exited.pid}), "refused": (exited.pid, None),
            "runner": (os.getpid(), {"state": "done", "pgid": exited.pid}), "group": (exited.pid, {"state": "failed", "pgid": member.pid}),
            "running": (exited.pid, {"state": "running", "pgid": exited.pid}), "link": (exited.pid, {"state": "done", "pgid": exited.pid})}
    try:
        for rid, (runner, status) in runs.items():
            (root / "runs" / rid).mkdir(parents=True)
            (root / "runs" / rid / "launch.json").write_text(json.dumps({"runner_pid": runner}))
            (root / "runs" / rid / ("status.json" if status else "refused.json")).write_text(json.dumps(status or {"fault": "CloneRefused"}))
            if rid != "link":
                (root / "scratch" / rid / "tree" / "busy").mkdir(parents=True)
        (root / "scratch" / "unknown" / "tree").mkdir(parents=True)        # no run dir: nothing says it is over
        (root / "scratch" / "link").symlink_to(victim)
        assert clone.home(root / "runs", "new", machine=True) == root / "scratch" / "new"
    finally:
        gate.release()
        member.wait()
    assert sorted(p.name for p in (root / "scratch").iterdir()) == ["group", "link", "new", "runner", "running", "unknown"]
    assert (victim / "keep").is_dir()
    err = capsys.readouterr().err
    assert f"removed {root / 'scratch' / 'done'}, the clone of finished run done" in err and "finished run refused" in err


def test_on_a_machine_the_clone_is_the_shipped_commit_beside_src_and_the_shared_tree_is_untouched(tmp_path, repo, machine, outlivers,
                                                                                                    capsys):
    _workspace(repo)
    assert _on(tmp_path, repo, "clone/pkg", "term") == 0
    assert "workload cwd: --cwd clone/pkg (the run's own clone of the shipped tree" in capsys.readouterr().err
    [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
    run_dir = machine / "runs" / rid
    wait_runner(run_dir)
    sha = _git(repo, "rev-parse", "HEAD")
    shared = (machine / "src" / sha).resolve()
    tree = (machine / "scratch" / rid / "tree").resolve()
    seen = _seen(run_dir)
    outlivers.append(seen["outliver"])
    assert _gone(seen["outliver"]) and "outlived it" in (run_dir / "launcher.log").read_text(), "SIGKILLed before the removal"
    assert Path(seen["cwd"]) == tree / "pkg" and seen["head"] == sha
    assert Path(seen["module"]) == tree / "m" / "src" / "leaky" / "__init__.py", "the workspace's code is imported from the clone"
    assert Path(seen["wrote"]) == tree / "m" / "reports" / "plot.txt", "so a write beside its __file__ lands in the clone"
    assert not (machine / "scratch" / rid).exists(), "removed after the SIGTERM `research cancel` sends to the workload's group"
    assert not (shared / "pkg" / ".lake").exists(), "the shared tree's build trees are not this run's"
    assert not (shared / "m" / "reports").exists() and not list(shared.rglob("__pycache__")), "nor its module's writes or bytecode"
    job = json.loads((run_dir / "job.json").read_text())["run"]
    assert job["cwd_basis"] == "clone" and job["cwd_clone"]["commit"] == sha and job["cwd_clone"]["of"] == str(shared)
    assert str(tree / "m" / "src") in job["remote"]["pythonpath"] and not [p for p in job["remote"]["pythonpath"] if str(shared) in p]
    assert A.build(run_dir)["execution"]["cwd_clone"] == job["cwd_clone"]
    assert R.summarize(run_dir)["state"] == "failed"


def test_on_a_machine_a_clone_run_executes_the_shipped_code_not_the_shared_trees_current_files(tmp_path, repo, machine):
    """Another run of the commit may have changed the shared tree's files (a `--cwd source` codegen); the clone run's record says it
    ran a clean checkout of the commit, so that is the code it imports."""
    _workspace(repo)
    sha = _git(repo, "rev-parse", "HEAD")

    def go(n: int) -> dict:
        runs = tmp_path / f"runs{n}"
        assert main(["run", "--runs-dir", str(runs), "--on", "m", "--project", "p", "--source", str(repo), "--no-custody-r2",
                     "--no-sampler", "--cwd", "clone", "--", PY, "-c", PROBE]) == 0
        [rid] = [p.name for p in runs.iterdir()]
        wait_runner(machine / "runs" / rid)
        return _seen(machine / "runs" / rid)

    assert go(0)["mark"] is None
    mod = machine / "src" / sha / "m" / "src" / "leaky" / "__init__.py"
    mod.write_text(mod.read_text() + "MARK = 'changed in the shared tree'\n")
    assert go(1)["mark"] is None


def test_the_trees_shim_is_replaced_by_one_of_the_clone_so_nothing_on_pythonpath_reaches_the_shipped_tree(tmp_path, monkeypatch):
    """On a machine the tree's shim (<root>/cache/pyshim/<key>, `verity -> <tree>/verity`) is not under the tree, so it stayed on
    PYTHONPATH and its link took `import verity` to the shared tree: Lean audits under `--cwd clone` refused when difftest_vectors.py
    found `verity` imported from there (runs r20261006-001503-1357, r20261006-032546-c760).  The clone's shim lives beside the clone
    and goes with it; the shared one stays for the runs that share it."""
    of, home = tmp_path / "src" / "sha", tmp_path / "scratch" / "r1"
    _root_package(of)
    tree = Path(shutil.copytree(of, home / "tree"))
    path = pythonpath(str(of), import_roots_dir(of), extra="/mine", shim_base=tmp_path / "cache" / "pyshim").split(":")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path))
    remote_info = {"pythonpath": list(path)}
    moved = _clone_imports(of, tree, home, remote_info)["PYTHONPATH"].split(os.pathsep)
    assert remote_info["pythonpath"] == moved and moved[1:] == [str(tree / "m" / "src"), "/mine"], "in place, in order"
    assert Path(moved[0]).parent == home / "pyshim" and os.listdir(moved[0]) == ["verity"]
    reached = [os.path.realpath(p) for e in moved for p in (e, *(os.path.join(e, c) for c in (os.listdir(e) if os.path.isdir(e) else ())))]
    assert not [p for p in reached if Path(p).is_relative_to(of.resolve())], "no entry, nor a link in one, reaches the shipped tree"
    spec = importlib.machinery.PathFinder.find_spec("verity", moved)
    assert Path(spec.origin).resolve() == (tree / "verity" / "__init__.py").resolve()
    assert (Path(path[0]) / "verity").resolve() == (of / "verity").resolve(), "the shared tree's shim is left as it was"
    assert clone.remove(home) is None and Path(path[0]).is_dir()


def test_on_a_machine_a_clone_run_imports_the_root_package_from_the_clone(tmp_path, repo, machine):
    _root_package(repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "the root is a package")
    probe = ("import json, os, verity; json.dump({'verity': os.path.realpath(verity.__file__), 'path': os.environ['PYTHONPATH']}, "
             "open(os.path.join(os.environ['RESEARCH_RUN_DIR'], 'seen.json'), 'w'))")
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--source", str(repo), "--no-custody-r2",
                 "--no-sampler", "--cwd", "clone/m", "--", PY, "-c", probe]) == 0              # `-c` in the root would find it in its cwd
    [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
    run_dir = machine / "runs" / rid
    wait_runner(run_dir)
    tree = (machine / "scratch" / rid / "tree").resolve()
    seen = _seen(run_dir)
    assert Path(seen["verity"]) == tree / "verity" / "__init__.py", "not the shared tree's, through its shim"
    shims = [p for p in seen["path"].split(os.pathsep) if "pyshim" in p]
    assert shims and all(Path(p).parent == machine / "scratch" / rid / "pyshim" for p in shims)
    shared = str((machine / "src" / _git(repo, "rev-parse", "HEAD")).resolve())
    recorded = json.loads((run_dir / "job.json").read_text())["run"]["remote"]["pythonpath"]
    assert not [p for p in recorded if shared in p or "cache/pyshim" in p], "the record names the clone's shim, not the shared tree's"
    assert not (machine / "scratch" / rid).exists()


def test_the_launcher_refuses_a_tree_shipped_without_git(tmp_path, repo, machine, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_SHIP", "stream")
    assert _on(tmp_path, repo, "clone") == 3
    err = capsys.readouterr().err
    assert "was shipped without git (transport stream), so it has no .git to clone.  Pass --cwd source" in err
    assert not (machine / "runs").exists(), "nothing claimed"


def test_the_launcher_refuses_a_shipped_research_package_that_predates_clone(tmp_path, repo, machine, capsys):
    old = repo / "tools" / "research" / "src" / "research"
    old.mkdir(parents=True)
    (old / "__init__.py").write_text("")
    (old / "remote.py").write_text("# before --cwd clone\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "an old research package")
    assert _on(tmp_path, repo, "clone") == 3
    err = capsys.readouterr().err
    assert "--cwd clone: the research package in" in err and "predates it (no research/clone.py)" in err and "pass --cwd source" in err
    assert not (machine / "src").exists() and not (machine / "tool").exists(), "refused before anything shipped"
    assert remote.knows_clone(None) and remote.knows_clone(Path(remote.__file__).parent) and not remote.knows_clone(old)
    assert remote.workload_cwd(["--cwd", "clone/pkg"], None).startswith("--cwd clone/pkg (the run's own clone of the shipped tree")
    assert remote.workload_cwd(["--cwd", "source"], None) == "--cwd source"


def _request(root: Path, src: Path, sha: str, run_id: str, *, custody: bool = False) -> Path:
    """A launch request for a `--cwd clone` run of `src`; `custody`: with a staged (dummy) --custody-r2 key."""
    (src / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": sha, "commit": sha}))
    (root / "requests" / run_id).mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": str(src),
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler", "--cwd", "clone"], "command": [PY, "-c", PROBE], "exclusive": False,
           "remote": {"machine": "m", "root": str(root), "source_sha": sha, "source_dir": str(src)}, "custody": custody}
    if custody:
        (root / "requests" / run_id / remote.CUSTODY_DIR).mkdir()
        (root / "requests" / run_id / remote.CUSTODY_DIR / "cred.json").write_text(json.dumps({"access_key_id": "dummy"}))
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p


def test_the_machine_refuses_a_tree_it_cannot_clone_as_the_shipped_commit(tmp_path, repo):
    """The launcher's refusals rest on what it shipped; the runner checks the tree it is given."""
    root = tmp_path / "root"
    head = _git(repo, "rev-parse", "HEAD")
    moved, plain = "d" * 40, "e" * 40
    subprocess.run(["git", "clone", "-q", "--shared", str(repo), str(root / "src" / moved)], check=True)
    (root / "src" / plain).mkdir(parents=True)
    for sha, why in ((moved, f"is at {head}, not {moved}, the commit the run was shipped for"),
                     (plain, "is not the top level of a git checkout")):
        out = remote.launch_request(_request(root, root / "src" / sha, sha, f"r-{sha[0]}", custody=True))
        assert not out["started"] and out["fault"] == "CloneRefused" and why in out["refused"], out
        run_dir = root / "runs" / f"r-{sha[0]}"
        wait_runner(run_dir)
        refused = json.loads((run_dir / "refused.json").read_text())
        assert refused["fault"] == "CloneRefused" and why in refused["reason"], refused
        assert R.summarize(run_dir)["state"] == "refused" and not (run_dir / "job.json").exists()
        assert not (root / "scratch" / f"r-{sha[0]}").exists()
        assert not (root / "requests" / f"r-{sha[0]}" / remote.CUSTODY_DIR / "cred.json").exists(), "nothing will publish: the key goes now"
