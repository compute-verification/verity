"""Per-run workload cwd on a machine: every run of one commit shares <root>/src/<sha>/, so the workload runs in its own run dir and a
relative `result.json` can no longer race (research-secondary found two concurrent runs of one commit writing the same file)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from research import remote
from research import runs as R
from research.cli import main
from research.store import attempt as A
from waits import wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
SHA = "c" * 40

# writes a relative result.json naming its run; given a FIFO it meets the other run there (ra opens it to write, rb to read, and
# neither open returns before both runs have written), then reads it back
RACER = ("import json, os, sys; me = os.environ['RESEARCH_RUN_ID']; json.dump({'who': me}, open('result.json', 'w')); "
         "sys.argv[1:] and open(sys.argv[1], 'w' if me == 'ra' else 'r').close(); "
         "import treemod; print(json.load(open('result.json'))['who'], treemod.X, os.getcwd())")


def _tree(root: Path) -> Path:
    src = root / "src" / SHA
    src.mkdir(parents=True)
    (src / "treemod.py").write_text("X = 7\n")
    (src / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": SHA, "commit": SHA, "archive_sha256": "0" * 64,
                                                      "files": 1, "verified": "archive-sha256+file-manifest"}))
    return src


def _request(root: Path, src: Path, run_id: str, *, harness=(), meet: Path | None = None) -> Path:
    (root / "requests" / run_id).mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": str(src),
           "env": {"PYTHONPATH": str(src)}, "harness_args": ["--stage", "t", "--no-sampler", *harness],
           "command": [PY, "-c", RACER, *([str(meet)] if meet else [])],
           "exclusive": False, "cost": {"provider": "runpod", "pod_id": "p1", "rate_usd_per_hour": 0.06, "rate_observed_utc": "t0", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p1", "root": str(root), "source_sha": SHA, "source_dir": str(src)}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p


def _wait(run_dir: Path) -> dict:
    wait_runner(run_dir)
    return R.summarize(run_dir)


def test_two_concurrent_runs_of_one_commit_do_not_share_a_cwd(tmp_path, monkeypatch):
    # inside a recorded run the outer runner's own prefix would win (`_runner_env` only sets a default)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    root = tmp_path / "root"
    src = _tree(root)
    before = sorted(p.relative_to(src).as_posix() for p in src.rglob("*"))
    meet = tmp_path / "meet"
    os.mkfifo(meet)
    outs = [remote.launch_request(_request(root, src, rid, meet=meet)) for rid in ("ra", "rb")]
    assert all(o["started"] for o in outs)
    for rid in ("ra", "rb"):
        run_dir = root / "runs" / rid
        s = _wait(run_dir)
        assert s["state"] == "done", (run_dir / "stderr.log").read_text()
        who, x, cwd = (run_dir / "stdout.log").read_text().split()
        assert who == rid and x == "7", "each run reads back its own relative result.json; the tree is still importable"
        assert Path(cwd).resolve() == run_dir.resolve()
        assert json.loads((run_dir / "result.json").read_text()) == {"who": rid}
        job = json.loads((run_dir / "job.json").read_text())
        assert Path(job["run"]["cwd"]).resolve() == run_dir.resolve() and job["run"]["cwd_basis"] == "run_dir"
        assert A.build(run_dir)["execution"]["cwd_basis"] == "run_dir"
    assert sorted(p.relative_to(src).as_posix() for p in src.rglob("*")) == before, "nothing is written into the shared tree (no result.json, no __pycache__)"
    assert list((root / "cache" / "pycache").rglob("treemod*.pyc")), "bytecode of the tree's modules goes to <root>/cache/pycache"


def test_cwd_source_is_the_explicit_opt_in(tmp_path):
    root = tmp_path / "root"
    src = _tree(root)
    assert remote.launch_request(_request(root, src, "rs", harness=["--cwd", "source"]))["started"]
    run_dir = root / "runs" / "rs"
    assert _wait(run_dir)["state"] == "done"
    assert json.loads((src / "result.json").read_text()) == {"who": "rs"} and not (run_dir / "result.json").exists()
    job = json.loads((run_dir / "job.json").read_text())
    assert Path(job["run"]["cwd"]).resolve() == src.resolve() and job["run"]["cwd_basis"] == "source"


def test_launcher_names_the_cwd_and_a_tree_that_predates_it(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    for d, body in ((new, 'WORKLOAD_CWD_DEFAULT = "run_dir"\n'), (old, "# before per-run cwd\n")):
        d.mkdir()
        (d / "remote.py").write_text(body)
    assert remote.workload_cwd(["--stage", "x"], None) == "run_dir"
    assert remote.workload_cwd([], new) == "run_dir"
    assert remote.workload_cwd([], old).startswith("source (the shipped tree's research package predates")
    assert remote.workload_cwd(["--cwd", "source/integrations/vllm"], old) == "--cwd source/integrations/vllm"
    assert remote.workload_cwd(["--cwd=/abs"], None) == "--cwd /abs"


def test_remote_cwd_source_without_a_shipped_tree_is_refused(tmp_path, capsys):
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--cwd", "source", "--", "true"])
    assert rc == 2 and "relative to the shipped source tree, and there is none" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists(), "refused before a run dir or any machine contact"


def test_local_cwd_source_is_the_invoking_directory(tmp_path, monkeypatch):
    here = tmp_path / "checkout"
    here.mkdir()
    monkeypatch.chdir(here)
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--cwd", "source", "--", PY, "-c", "open('out.txt', 'w').write('x')"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    job = json.loads((run_dir / "job.json").read_text())
    assert (here / "out.txt").exists() and job["run"]["cwd_basis"] == "source" and Path(job["run"]["cwd"]).resolve() == here.resolve()
