"""A tool's preflight (`Tool.preflight`) on the machine: `research run --on` refuses a machine that fails it before the run is claimed,
so an unprepared pod is an error, never a verdict.  The whole launch runs locally: ssh is `sh -c`, the machine a directory, and the
tree a git repository whose declaration names the preflight."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research import remote
from research import runs as R
from research.cli import main
from research.store.local import LocalStore
from waits import wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
TOOL = "tools.fake.tool:FAKE"
DECL = "from research.store.tool import Tool\nFAKE = Tool(name='fake', version='1', preflight=['pf.py', '--say', 'hi'])\n"
# prints what it saw; fails when the run's --env says so
PREFLIGHT = """import os, sys
inputs = os.environ.get("RESEARCH_PREFLIGHT_INPUTS")
print("args", *sys.argv[1:])
print("cwd", os.getcwd())
print("sent", *sorted(os.listdir(inputs)) if inputs and os.path.isdir(inputs) else ["-"])
print("custody", os.environ.get("RESEARCH_PREFLIGHT_CUSTODY", "-"))
if os.environ.get("PF_FAIL"):
    print("uv is not on the PATH", file=sys.stderr)
    print("prepare the pod: bash tools/verity/check/pod_setup.sh", file=sys.stderr)
    sys.exit(1)
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path) -> Path:
    d = tmp_path / "repo"
    (d / "tools" / "fake").mkdir(parents=True)
    (d / "tools" / "fake" / "tool.py").write_text(DECL)
    (d / "pf.py").write_text(PREFLIGHT)
    _git(d, "init", "-q")
    _git(d, "add", ".")
    _git(d, "commit", "-q", "-m", "one")
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


def _run(tmp_path, repo, *extra: str, which: tuple[str, ...] = ("--tool", TOOL)) -> int:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sent.txt").write_text("x")
    return main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--source", str(repo), *which,
                 "--no-custody-r2", "--send", str(tmp_path / "sent.txt"), "--no-sampler", *extra, "--", PY, "-c", "print('ran')"])


def test_a_failing_preflight_refuses_the_run_before_anything_runs(tmp_path, repo, machine, capsys):
    rc = _run(tmp_path, repo, "--env", "PF_FAIL=1")
    err = capsys.readouterr().err
    assert rc == 3 and "refused on m (PreflightFailed)" in err
    assert "uv is not on the PATH; prepare the pod: bash tools/verity/check/pod_setup.sh" in err, "the preflight's stderr is the reason"
    assert "research: preflight pf.py --say hi on m: FAILED" in err
    [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
    run_dir = machine / "runs" / rid
    refused = json.loads((run_dir / "refused.json").read_text())
    assert refused["fault"] == "PreflightFailed" and refused["preflight"]["rc"] == 1
    assert not (run_dir / "launch.json").exists() and not (run_dir / "job.json").exists(), "no runner was started"
    assert R.summarize(run_dir)["state"] == "refused"
    assert LocalStore().get_attempt(rid) is None, "nothing recorded: a refusal is not a verdict"
    local = json.loads((tmp_path / "runs" / rid / "remote.json").read_text())
    assert local["phase"] == "refused" and local["preflight"]["stderr"][0] == "uv is not on the PATH"


def test_a_passing_preflight_runs_in_the_tree_with_the_runs_inputs_and_is_recorded(tmp_path, repo, machine, capsys):
    assert _run(tmp_path, repo) == 0
    err = capsys.readouterr().err
    [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
    run_dir = machine / "runs" / rid
    wait_runner(run_dir)
    assert R.summarize(run_dir)["state"] == "done" and (run_dir / "stdout.log").read_text() == "ran\n"
    pf = json.loads((run_dir / "job.json").read_text())["run"]["remote"]["preflight"]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert pf["rc"] == 0 and pf["argv"] == ["pf.py", "--say", "hi"]
    assert pf["stdout"] == ["args --say hi", f"cwd {machine / 'src' / sha}", "sent sent.txt", "custody -"]
    assert "research: preflight pf.py --say hi on m: passed" in err and "research: preflight: sent sent.txt" in err
    assert (run_dir / "inputs" / "sent.txt").read_text() == "x", "the run still gets its --send files"
    assert LocalStore().get_attempt(rid) is not None


def test_preflight_names_a_tools_preflight_for_a_run_of_anything_else(tmp_path, repo, machine, capsys):
    """A Lean record regeneration on a check pod is no `--tool check` run, and must not be recorded as one."""
    assert _run(tmp_path, repo, "--env", "PF_FAIL=1", which=("--preflight", TOOL)) == 3
    assert "refused on m (PreflightFailed)" in capsys.readouterr().err
    assert _run(tmp_path / "again", repo, which=("--preflight", TOOL)) == 0
    [rid] = [p.name for p in (tmp_path / "again" / "runs").iterdir()]
    wait_runner(machine / "runs" / rid)
    job = json.loads((machine / "runs" / rid / "job.json").read_text())
    assert job["run"]["remote"]["preflight"]["rc"] == 0 and job["store"]["tool"]["name"] != "fake", "not recorded as a run of the tool"


def test_preflight_refuses_what_it_cannot_run(tmp_path, repo, machine, capsys):
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--preflight", TOOL, "--", "true"]) == 2
    assert "from the tree --source ships" in capsys.readouterr().err
    assert _run(tmp_path / "other", repo, which=("--preflight", "tools.other.tool:X")) == 3
    assert "does not declare that tool" in capsys.readouterr().err
    plain = _tree(tmp_path, decl="from research.store.tool import Tool\nFAKE = Tool(name='fake')\n", name="plain")
    with pytest.raises(remote.PreflightFailed, match="names no preflight"):
        remote.tool_preflight([], plain, asked=TOOL)
    old = tmp_path / "old-research"
    old.mkdir()
    (old / "remote.py").write_text("# before --preflight\n")
    assert remote.knows_preflight(None) and not remote.knows_preflight(old) and remote.knows_preflight(Path(remote.__file__).parent)


def _request(root: Path, run_id: str, tree: Path | None, *, custody: bool = False, env=None) -> Path:
    stage = root / "requests" / run_id
    stage.mkdir(parents=True)
    if custody:
        (stage / remote.CUSTODY_DIR).mkdir()
        (stage / remote.CUSTODY_DIR / "cred.json").write_text("{}")
    sha = "a" * 40
    if tree is not None:
        (tree / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": sha, "archive_sha256": "0" * 64}))
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY,
           "cwd": str(tree) if tree else None,
           "env": env or {}, "harness_args": ["--stage", "t", "--no-sampler", "--tool", TOOL], "command": [PY, "-c", "pass"], "exclusive": False,
           "cost": {}, "remote": {"machine": "m", "pod_id": "p1", **({"source_sha": sha, "source_dir": str(tree)} if tree else {})},
           **({"custody": {"expires_utc": "2099-01-01T00:00:00Z"}} if custody else {})}
    (stage / "request.json").write_text(json.dumps(req))
    return stage / "request.json"


def _tree(tmp_path: Path, decl: str = DECL, name: str = "tree") -> Path:
    t = tmp_path / name
    (t / "tools" / "fake").mkdir(parents=True)
    (t / "tools" / "fake" / "tool.py").write_text(decl)
    (t / "pf.py").write_text(PREFLIGHT)
    return t


def test_the_preflight_gets_the_custody_dir_and_a_refusal_deletes_the_key(tmp_path):
    root = tmp_path / "root"
    req = _request(root, "r1", _tree(tmp_path), custody=True, env={"PF_FAIL": "1"})
    out = remote.launch_request(req)
    assert out["started"] is False and out["fault"] == "PreflightFailed"
    assert f"custody {root / 'requests' / 'r1' / remote.CUSTODY_DIR}" in out["preflight"]["stdout"]
    assert not (root / "requests" / "r1" / remote.CUSTODY_DIR / "cred.json").exists(), "nothing will publish: the key goes now"
    again = remote.launch_request(req)                      # the launcher re-delivered after an ssh drop
    assert again["existing"] and again["fault"] == "PreflightFailed" and again["refused"] == out["refused"]


def test_a_refusal_after_the_preflight_passed_deletes_the_staged_credentials(tmp_path):
    """The preflight admits, then --exclusive finds another run live: nothing will ever take the sent URLs out of the request dir."""
    root = tmp_path / "root"
    req = _request(root, "r1", _tree(tmp_path), custody=True)
    doc = json.loads(req.read_text())
    doc["exclusive"] = True
    req.write_text(json.dumps(doc))
    (root / "runs" / "r0").mkdir(parents=True)
    (root / "runs" / "r0" / "status.json").write_text('{"state": "running"}')
    (req.parent / "inputs").mkdir()
    (req.parent / "inputs" / "lean-deps-urls.json").write_text('{"k": "https://r2/o?X-Amz-Signature=s"}')
    (req.parent / "inputs" / "sent.txt").write_text("x")
    out = remote.launch_request(req)
    assert out["started"] is False and out["fault"] == "Refused" and "other run(s) are live" in out["refused"], "after the preflight"
    assert not (req.parent / "inputs" / "lean-deps-urls.json").exists() and (req.parent / "inputs" / "sent.txt").exists()
    assert not (req.parent / remote.CUSTODY_DIR / "cred.json").exists()


def test_an_unreadable_declaration_refuses_and_none_means_no_preflight(tmp_path):
    root = tmp_path / "root"
    bad = _tree(tmp_path, decl="raise RuntimeError('broken')\n")
    out = remote.launch_request(_request(root, "r1", bad))
    assert out["fault"] == "PreflightFailed" and "cannot be read (RuntimeError: broken)" in out["refused"]
    assert remote.tool_preflight(["--tool", TOOL], None) is None, "no shipped tree: nothing to run a preflight from"
    assert remote.tool_preflight(["--stage", "t"], bad) is None, "no --tool"
    assert remote.tool_preflight(["--tool", "tools.other.tool:X"], bad) is None, "the tree does not declare the tool"
    plain = _tree(tmp_path, decl="from research.store.tool import Tool\nFAKE = Tool(name='fake')\n", name="plain")
    assert remote.tool_preflight(["--tool", TOOL], plain) is None, "a declaration without a preflight"
    out = remote.launch_request(_request(root, "r2", plain))
    assert out["started"] and "preflight" not in out
    wait_runner(root / "runs" / "r2")
