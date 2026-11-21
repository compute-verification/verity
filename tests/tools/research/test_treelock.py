"""A `--cwd source` workload that builds Lean takes the shared tree's lock first (research/treelock.py): two `lake build`s of one
commit on a machine wrote one `.lake` at once and clang died of SIGBUS on the other's truncated .c files (INF-641). It takes it
last of what it waits for: after its GPU lease and its check slot, so a run waiting in a line holds no tree."""
from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

from research import cli, remote, treelock
from research.cli import main
from research.pods.nebius import slot_only
from waits import FakeClock

PY = sys.executable
SLOT_PY = Path(__file__).resolve().parents[3] / "tools" / "check" / "slot.py"
#: a `lake` that says whether its tree's lock (job.json run.lean_tree_lock) is held while it runs: a second open of it can't take it
FAKE_LAKE = f"""#!{PY}
import fcntl, json, os, sys
d = os.environ["RESEARCH_RUN_DIR"]
fd = os.open(json.load(open(os.path.join(d, "job.json")))["run"]["lean_tree_lock"], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held = "free"
except BlockingIOError:
    held = "held"
open(os.path.join(d, "lake.txt"), "w").write(" ".join([held, os.getcwd(), *sys.argv[1:]]))
"""


@pytest.mark.parametrize("command, builds", [
    (["bash", "-lc", "uv sync -q && lake build Flock"], True),
    (["python3", "tools/verity/lean/lean_changed.py", "--base", "x"], True),
    (["python3", "tools/check/slot.py", "--by", "t", "--", "nice", "-n", "19", "python3", "tools/check/check.py"], True),
    (["bash", "-lc", "uv sync -q --all-packages && uv run tools/check/suites.py --quick"], True),
    (["uv", "run", "--extra", "torch-cpu", "verity", "check", "lean"], True),
    (["/home/research/.elan/bin/lake", "env", "lean", "X.lean"], True),
    (["python", "bench.py", "--out", "o"], False),
    (["bash", "-c", "du -sh .lake/build"], False),
    (["echo", "lakeside", "check.pyc"], False),
    (["verity", "bench", "one_stage"], False),
])
def test_what_builds_lean(command, builds):
    assert treelock.builds_lean(command) is builds


def test_a_waiter_takes_the_lock_once_its_holder_lets_go_and_gives_up_after_its_wait(tmp_path, capsys):
    lock = tmp_path / "tree.lock"
    holder = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(holder, fcntl.LOCK_EX)
    clock = FakeClock()
    clock.at(60, lambda: os.close(holder))
    fd = treelock.acquire(str(lock), 300, "/t", clock=clock.module.monotonic, sleep=clock.sleep)
    assert fd is not None and clock.now == 60
    err = capsys.readouterr().err
    assert err.count("waiting up to 300 s for another run's Lean build in the shared tree /t") == 1 and "--cwd clone" in err
    assert treelock.acquire(str(lock), 300, "/t", clock=clock.module.monotonic, sleep=clock.sleep) is None, "until the wait is spent"
    assert clock.now == 360
    assert treelock.main([str(lock), "0", "/t", "--", "true"]) == treelock.EX_NOT_STARTED
    err = capsys.readouterr().err
    assert "nothing ran" in err and "launch it with `research run --cwd clone`" in err
    os.close(fd)


def test_a_lean_build_in_the_shared_tree_runs_holding_its_lock_and_other_runs_take_none(tmp_path, monkeypatch):
    tree, bin_dir, runs = tmp_path / "tree", tmp_path / "bin", tmp_path / "runs"
    tree.mkdir()
    bin_dir.mkdir()
    (bin_dir / "lake").write_text(FAKE_LAKE)
    (bin_dir / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)
    monkeypatch.chdir(tree)
    assert main(["run", "--runs-dir", str(runs), "--no-sampler", "--cwd", "source", "--", "lake", "build", "Flock"]) == 0
    [run_dir] = [d for d in runs.iterdir() if d.is_dir()]
    lock = json.loads((run_dir / "job.json").read_text())["run"]["lean_tree_lock"]
    assert lock == str(treelock.lock_path(runs, tree)) and Path(lock).parent == runs
    assert (run_dir / "lake.txt").read_text() == f"held {tree.resolve()} build Flock", "the build ran in the tree, holding its lock"
    for i, (args, code) in enumerate(((["--cwd", "source"], "print('bench')"), ([], "print('lake build')"))):
        run_id = f"r20261109-000000-000{i}"
        assert main(["run", "--runs-dir", str(runs), "--id", run_id, "--no-sampler", *args, "--", PY, "-c", code]) == 0
        assert "lean_tree_lock" not in json.loads((runs / run_id / "job.json").read_text())["run"], "no Lean build, or not the tree"


LEASE = ["gpu-lease", "1", "--wait", "--mem-gb", "0", "--"]
POOLED = slot_only.in_pool("research", "t", ["bash", "-lc", "lake build"], 600, 60)


@pytest.mark.parametrize("req, spawn, rest", [
    ({}, ["lake", "build"], ["lake", "build"]),
    ({}, ["python3", "tools/check/slot.py", "--by", "t", "--", "lake", "build", "--", "X"], ["lake", "build", "--", "X"]),
    ({}, POOLED, ["nice", "-n", "19", "bash", "-lc", "lake build"]),
    ({}, slot_only.node_copy(POOLED), ["nice", "-n", "19", "bash", "-lc", "lake build"]),
    ({"gpus": 1}, [*LEASE, "lake", "build"], ["lake", "build"]),
    ({"gpus": 1}, [*LEASE, *POOLED], ["nice", "-n", "19", "bash", "-lc", "lake build"]),
    ({"workload_wrap": ["gpu-lease", "2", "--"], "wrap": ["srun"]}, ["gpu-lease", "2", "--", "lake", "build"], ["lake", "build"]),
    ({"gpus": 1}, ["lake", "build"], ["lake", "build"]),
    ({}, ["python3", "tools/check/slot.py", "--status"], ["python3", "tools/check/slot.py", "--status"]),
    ({}, ["bash", "-lc", "python3 tools/check/slot.py -- lake build"], ["bash", "-lc", "python3 tools/check/slot.py -- lake build"]),
])
def test_the_trees_lock_goes_after_every_wrapper_that_waits_for_a_lease_or_a_slot(req, spawn, rest):
    """The lock's wrapper starts only once the wrappers before it have what they wait for: it is the last wait of a run."""
    at = cli._past_waits(req, spawn)
    assert spawn[at:] == rest


def _slotted(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    """A tree with the real slot.py at its place, a machine with one check slot `a`, and `lake` and `python3` on PATH:
    (tree, runs, the tree's lock, the slot's lock)."""
    tree, bin_dir, runs, locks = tmp_path / "tree", tmp_path / "bin", tmp_path / "runs", tmp_path / "locks"
    (tree / "tools" / "check").mkdir(parents=True)
    (tree / "tools" / "check" / "slot.py").write_bytes(SLOT_PY.read_bytes())
    bin_dir.mkdir()
    locks.mkdir()
    (locks / "slots").write_text(f"a {min(os.sched_getaffinity(0))}\n")
    (bin_dir / "lake").write_text(FAKE_LAKE)
    (bin_dir / "lake").chmod(0o755)
    (bin_dir / "python3").symlink_to(PY)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)
    monkeypatch.setenv("CHECK_SLOTS", str(locks / "slots"))
    monkeypatch.delenv("CHECK_SLOT", raising=False)
    monkeypatch.chdir(tree)
    return tree, runs, treelock.lock_path(runs, tree), locks / "check-a.lock"


def _hold(lock: Path) -> int:
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


SLOTTED = ["python3", "tools/check/slot.py", "--by", "t", "--wait-max", "1e-9", "--", "lake", "build", "Flock"]


def test_a_run_waiting_for_a_check_slot_holds_no_tree(tmp_path, monkeypatch):
    """A run in a slot's line held the shared tree's lock through its wait, and every other Lean build of the commit waited
    behind a run that was building nothing. The lock's wrapper now starts inside the slot: a run that gets no slot never
    opens the tree's lock, which a run that had waited on it would have made."""
    tree, runs, lock, slot_lock = _slotted(tmp_path, monkeypatch)
    busy = _hold(slot_lock)
    rc = main(["run", "--runs-dir", str(runs), "--no-sampler", "--cwd", "source", "--", *SLOTTED])
    os.close(busy)
    [run_dir] = [d for d in runs.iterdir() if d.is_dir()]
    assert rc == treelock.EX_NOT_STARTED and "no check slot within --wait-max 1e-09 s" in (run_dir / "stderr.log").read_text()
    assert json.loads((run_dir / "job.json").read_text())["run"]["lean_tree_lock"] == str(lock)
    assert not lock.exists() and not (run_dir / "lake.txt").exists(), "it waited for a slot and left: it never came to the tree's lock"


@pytest.mark.parametrize("command, cwd, at", [
    (["bash", "-lc", "uv sync -q && cd security && lake build"], None, "security"),
    (["bash", "-lc", "cd security/Proofs; LEAN_NUM_THREADS=4 lake build Proofs 2>&1"], None, "security/Proofs"),
    (["lake", "-d", "security/Properties", "build"], None, "security/Properties"),
    (["/home/research/.elan/bin/lake", "build", "--dir=security"], None, "security"),
    (["lake", "build", "Definitions"], "clone/security", "clone/security"),
    (["bash", "-c", "cd /workspace/research/src/abc/security && lake build"], None, "/workspace/research/src/abc/security"),
    (["python3", "tools/verity/lean/lean_changed.py", "--records", "security", "--update"], None, None),
    (["verity", "lean", "try", "security/Definitions/Core.lean", "--on", "auto"], None, None),
    (["python3", "tools/verity/lean/lean_audit.py", "security"], None, None),
    (["python3", "tools/verity/lean/check.py", "--build", "--artifacts", "a", "security"], None, None),
    (["python3", "tools/check/slot.py", "--by", "t", "--", "nice", "-n", "19", "python3", "tools/check/check.py"], None, None),
    (["uv", "run", "--extra", "torch-cpu", "verity", "check", "lean"], None, None),
    (["bash", "-lc", "cd verity/protocols/proof/flock/verifier && lake build"], None, None),
    (["bash", "-lc", "cd verity/protocols/proof/flock/prover && cargo build --release && cargo test"], None, None),
    (["lake", "env", "lean", "security/Definitions/Core.lean"], "source", None),
    (["lake", "build"], "source", None),
], ids=["cd", "cd-sub", "dash-d", "dir-eq", "cwd", "absolute", "records", "lean-try", "lean-audit", "check-artifacts",
        "check-shard", "verity-check", "flock-verifier", "flock-rust", "lake-env", "source-root"])
def test_a_raw_build_of_an_analysis_package_is_told_apart_from_the_lean_tools_and_the_codes_lean(command, cwd, at):
    assert treelock.analysis_build(command, cwd) == at


RAW = ["bash", "-lc", "cd security && lake build"]


@pytest.fixture
def launched(monkeypatch):
    """Each launch `research run` went on to, in place of making it: its command and what it was launched with."""
    seen: list[tuple[list[str], str | None]] = []
    monkeypatch.setattr(cli, "_run_placed", lambda ns, harness, command, clones: seen.append((command, ns.cold_lean_build)) or 0)
    monkeypatch.setattr(cli, "_auto_placement", lambda ns, harness: seen.append((["auto"], None)) or "placed")
    return seen


@pytest.mark.parametrize("where", [["--on", "vy-nowhere", "--project", "p"], ["--on", "auto"],
                                   ["--queue", "--source", ".", "--project", "p"]], ids=["on", "auto", "queue"])
def test_a_raw_build_of_an_analysis_package_on_a_machine_is_refused_at_launch_with_the_fix(tmp_path, capsys, launched, where):
    assert main(["run", "--runs-dir", str(tmp_path), *where, "--", *RAW]) == 2
    err = capsys.readouterr().err
    assert "research run: refused: the workload runs a raw `lake build` of an analysis package (security)" in err
    assert "`verity lean try FILE --on auto`" in err and "`python3 tools/verity/lean/lean_changed.py --records PKG --update`" in err
    assert "--cold-lean-build REASON (recorded in job.json)" in err
    assert launched == [] and list(tmp_path.iterdir()) == [], "nothing launched, no run dir taken"


def test_a_cold_build_on_purpose_launches_and_the_machine_records_why(tmp_path, launched):
    argv = ["run", "--runs-dir", str(tmp_path), "--on", "vy-nowhere", "--project", "p", "--cold-lean-build", "a fresh toolchain"]
    assert main([*argv, "--", *RAW]) == 0 and launched == [(RAW, "a fresh toolchain")]


def test_the_flag_travels_to_the_machine_and_its_record_says_why(tmp_path, monkeypatch):
    sent: dict = {}
    monkeypatch.setattr(remote, "launch", lambda **kw: sent.update(kw) or (_ for _ in ()).throw(remote.Refused("stop here")))
    argv = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "vy-nowhere", "--project", "p", "--no-custody-r2",
            "--cold-lean-build", "a fresh toolchain", "--", *RAW]
    assert main(argv) == 3
    at = sent["harness_args"].index("--cold-lean-build")
    assert sent["harness_args"][at + 1] == "a fresh toolchain"
    run_id = "r20261118-000000-0001"
    assert main(["run", "--runs-dir", str(tmp_path / "local"), "--id", run_id, "--no-sampler", "--local-only", "a test",
                 "--cold-lean-build", "a fresh toolchain", "--", PY, "-c", "pass"]) == 0
    assert json.loads((tmp_path / "local" / run_id / "job.json").read_text())["run"]["cold_lean_build"] == "a fresh toolchain"


def test_a_local_run_and_the_machine_side_are_not_judged(tmp_path, launched):
    """A local run builds in a tree of its caller's; on the machine the launcher has judged the command already."""
    assert main(["run", "--runs-dir", str(tmp_path), "--", *RAW]) == 0
    assert main(["run", "--runs-dir", str(tmp_path), "--id", "r20261118-000000-abcd", "--claimed", "--", *RAW]) == 0
    assert launched == [(RAW, None), (RAW, None)]
