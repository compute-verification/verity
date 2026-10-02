"""`research run/status/inspect` over the promoted harness, in a temporary runs root."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from research import runs as R
from research.cli import main


def test_run_records_and_inspects(tmp_path, capsys):
    rc = main(["run", "--runs-dir", str(tmp_path), "--project", "t", "--stage", "smoke", "--no-sampler", "--",
               sys.executable, "-c", "print('hello'); import sys; sys.exit(3)"])
    assert rc == 3, "research run exits with the workload's status"
    runs = R.list_runs(tmp_path)
    assert len(runs) == 1
    s = runs[0]
    assert s["project"] == "t" and s["stage"] == "smoke" and s["rc"] == 3
    assert s["state"] in ("done", "failed")
    assert (tmp_path / s["id"] / "stdout.log").read_text().strip() == "hello"
    job = json.loads((tmp_path / s["id"] / "job.json").read_text())
    assert job["schema"].startswith("telemetry/job/")
    assert job["ownership"]["job"] == s["id"]

    assert main(["inspect", s["id"][:12], "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert s["id"] in out and "rc=3" in out

    assert main(["status", "--runs-dir", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"][0]["id"] == s["id"]


RESULT_OF_ENV = ("import os, json; json.dump({'foo': os.environ.get('RSEC_FOO'), 'tok': os.environ.get('RSEC_API_TOKEN'), 'cwd': os.getcwd()}, "
                 "open('result.json', 'w'))")


def test_local_env_is_applied_to_the_workload_recorded_and_scoped(tmp_path, monkeypatch):
    src = tmp_path / "checkout"
    src.mkdir()
    monkeypatch.chdir(src)
    monkeypatch.delenv("RSEC_FOO", raising=False)
    monkeypatch.setenv("RSEC_API_TOKEN", "callers")
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--env", "RSEC_FOO=bar", "--env", "RSEC_API_TOKEN=s3cret",
               "--", sys.executable, "-c", RESULT_OF_ENV])
    assert rc == 0
    run_dir = next((tmp_path / "runs").iterdir())
    seen = json.loads((run_dir / "result.json").read_text())
    assert seen["foo"] == "bar" and seen["tok"] == "s3cret", "the workload sees --env"
    assert Path(seen["cwd"]).resolve() == run_dir.resolve() and list(src.iterdir()) == [], "default cwd is the run dir: the checkout stays clean"
    assert "RSEC_FOO" not in os.environ and os.environ["RSEC_API_TOKEN"] == "callers", "--env is scoped to the workload"
    params = json.loads((run_dir / "job.json").read_text())["store"]["params"]
    assert params["env"] == {"RSEC_FOO": "bar", "RSEC_API_TOKEN": "<redacted>"}
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--env", "NOEQUALS", "--", "true"]) == 2


def test_cwd_overrides_the_run_dir_default(tmp_path, monkeypatch):
    src = tmp_path / "checkout"
    src.mkdir()
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--cwd", str(src), "--", sys.executable, "-c", RESULT_OF_ENV]) == 0
    assert Path(json.loads((src / "result.json").read_text())["cwd"]).resolve() == src.resolve()


def test_machine_side_records_the_requests_env(tmp_path):
    from research.cli import _request_env

    req = tmp_path / "request.json"
    req.write_text(json.dumps({"schema": "x", "env": {"A": "1", "B_TOKEN": "t"}}))
    assert _request_env(json.dumps({"machine": "m", "request": str(req)})) == {"A": "1", "B_TOKEN": "t"}
    assert _request_env(json.dumps({"machine": "m"})) == {} and _request_env(None) == {}
    assert _request_env(json.dumps({"request": str(tmp_path / "absent.json")})) == {}


def test_run_requires_separator_and_command(tmp_path):
    assert main(["run", "--runs-dir", str(tmp_path), "echo", "x"]) == 2
    assert main(["run", "--runs-dir", str(tmp_path), "--"]) == 2
    assert not list(tmp_path.iterdir())


def test_remote_requires_project_and_listed_machine(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--", "true"]) == 2
    assert "--project is required" in capsys.readouterr().err
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--project", "verity", "--", "true"])
    err = capsys.readouterr().err
    assert rc == 3 and "not in" in err and "refused" in err
    (tmp_path / "machines.toml").write_text('[machines.pod-x]\nprovider = "runpod"\npod_id = "p1"\nproject = "veritor"\n')
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--project", "verity", "--", "true"])
    err = capsys.readouterr().err
    assert rc == 3 and "belongs to project(s) ['veritor']" in err, "a foreign pod is refused before any contact"
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 2 and all((r / "remote.json").exists() for r in runs), "refused launches are still recorded runs"


def test_resolve_prefix_and_ambiguity(tmp_path):
    for name in ("r20260921-100000-aaaa", "r20260921-100001-bbbb"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "job.json").write_text("{}")
    assert R.resolve(tmp_path, "r20260921-100000").name.endswith("aaaa")
    with pytest.raises(LookupError):
        R.resolve(tmp_path, "r20260921")
    with pytest.raises(FileNotFoundError):
        R.resolve(tmp_path, "nope")


def test_no_packages_import():
    """`research` must not depend on the Verity library, frontends, backends, or integrations (README: it treats
    the workload as opaque). Covers `verity` and any `verity_*` extension distribution."""
    import pathlib
    import re

    src = pathlib.Path(R.__file__).parent
    offenders = []
    for p in src.rglob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*(from|import)\s+(verity\w*|packages|veritor|backends|frontends|integrations)\b", line):
                offenders.append(f"{p.relative_to(src)}: {line.strip()}")
    assert not offenders, offenders


STUB_SUBMIT = """\
import json, sys
open(sys.argv[0] + '.argv', 'w').write(json.dumps(sys.argv[1:]))
a = sys.argv
val = lambda k, d=None: a[a.index(k) + 1] if k in a else d
if val('--gpus') == '9':
    print(json.dumps({'refused': 'vy-nebius-2: not 9 GPUs'})); sys.exit(3)
print(json.dumps({'node': 'vy-nebius-2', 'machine': 'vy-nebius-2', 'why': 'stub', 'job': {'gpus': int(val('--gpus', 0))},
                  'wrap': ['env', 'HELD=1'], 'workload_wrap': ['lease', '--'], 'kind': val('--kind'),
                  'question': val('--question', ''), 'max_s': int(val('--max-s', 1200)), 'warnings': ['stub warning']}))
"""


def _queue_source(tmp_path) -> Path:
    pkg = tmp_path / "src-tree" / "tools" / "cluster" / "src" / "cluster"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "submit.py").write_text("")
    (pkg / "__main__.py").write_text(STUB_SUBMIT)
    return tmp_path / "src-tree"


def test_run_queue_places_the_job_then_launches_on_the_placed_machine_with_its_wrapper(tmp_path, monkeypatch, capsys):
    """`research run --queue`: the declared resources reach `cluster submit` from the --source tree, and the run launches as an
    `--on` run on the machine it names, carrying the wrappers, its kind and its phase; the runner's --timeout is the job's time
    plus, for a gpu phase, the time the lease may take to be granted."""
    from research import cli, remote
    src, seen = _queue_source(tmp_path), []

    def launch(**kw):
        seen.append(kw)
        return {"phase": "launched", "pod_id": "p1", "remote_run_dir": "/r", "launch": {"runner_pid": 1}}
    monkeypatch.setattr(remote, "launch", launch)
    monkeypatch.setenv("RESEARCH_LANE", "lane-x")
    base = ["run", "--runs-dir", str(tmp_path / "runs"), "--queue", "--source", str(src), "--project", "verity"]
    assert main([*base, "--mem-gb", "64", "--gpus", "1", "--max-min", "10", "--preemptible", "--quiet", "--class", "bench",
                 "--kind", "vllm.commit", "--", "true"]) == 0
    err = capsys.readouterr().err
    assert "the queue places run" in err and "warning: stub warning" in err
    argv = json.loads((src / "tools" / "cluster" / "src" / "cluster" / "__main__.py.argv").read_text())
    assert argv[:2] == ["submit", str(src / "tools" / "cluster" / "descriptions" / "nebius.toml")]
    for k, v in (("--project", "verity"), ("--submitter", "lane-x"), ("--class", "bench"), ("--gpus", "1"), ("--mem-gb", "64"),
                 ("--max-s", "600"), ("--preempt", "requeue")):
        assert argv[argv.index(k) + 1] == v, k
    assert "--quiet" in argv
    [kw] = seen
    assert kw["machine_name"] == "vy-nebius-2" and kw["queue"]["wrap"] == ["env", "HELD=1"] and kw["mem_gb"] is None
    assert kw["queue"]["workload_wrap"] == ["lease", "--"] and (kw["queue"]["kind"], kw["queue"]["phase"]) == ("vllm.commit", "gpu")
    assert kw["queue"]["disk_gb"] == 20 and "--disk-gb" not in argv
    assert kw["harness_args"][kw["harness_args"].index("--timeout") + 1] == str(600 + 60 * cli.GPU_WAIT_MIN)
    assert main([*base, "--mem-gb", "8", "--max-min", "10", "--", "true"]) == 0
    assert seen[-1]["queue"]["kind"] == "adhoc" and seen[-1]["queue"]["phase"] == "cpu"
    assert seen[-1]["harness_args"][seen[-1]["harness_args"].index("--timeout") + 1] == "600"
    assert main([*base, "--mem-gb", "8", "--max-min", "10", "--phase", "cpu", "--gpus", "1", "--", "true"]) == 2
    assert "a cpu phase takes no GPUs" in capsys.readouterr().err
    assert main([*base, "--kind", "lean-audit", "--question", "does it check?", "--disk-gb", "5", "--", "true"]) == 0
    argv = json.loads((src / "tools" / "cluster" / "src" / "cluster" / "__main__.py.argv").read_text())
    assert argv[argv.index("--disk-gb") + 1] == "5.0" and seen[-1]["queue"]["disk_gb"] == 5.0
    assert "--mem-gb" not in argv and "--max-s" not in argv and argv[argv.index("--question") + 1] == "does it check?"
    assert seen[-1]["harness_args"][seen[-1]["harness_args"].index("--timeout") + 1] == "1200", "the kind's time"
    assert main([*base, "--mem-gb", "8", "--max-min", "10", "--phase", "gpu", "--", "true"]) == 2

    assert main([*base, "--mem-gb", "8", "--gpus", "9", "--max-min", "1", "--", "true"]) == 3
    assert "the queue refuses the job: vy-nebius-2: not 9 GPUs" in capsys.readouterr().err
    assert len(seen) == 3


def test_run_queue_needs_a_source_tree_with_the_queue_and_queue_flags_need_queue(tmp_path, capsys):
    base = ["run", "--runs-dir", str(tmp_path / "runs"), "--queue", "--project", "verity"]
    assert main([*base, "--mem-gb", "8", "--max-min", "1", "--", "true"]) == 2
    assert "--source GITDIR" in capsys.readouterr().err
    assert main([*base, "--source", str(tmp_path), "--mem-gb", "8", "--max-min", "1", "--", "true"]) == 3
    assert "no tools/cluster" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())
    plain = ["run", "--runs-dir", str(tmp_path / "runs")]
    assert main([*plain, "--gpus", "1", "--cpus", "4", "--", "true"]) == 2
    assert "--gpus, --cpus declare a job to the queue and mean nothing without --queue" in capsys.readouterr().err
    assert main([*plain, "--kind", "lean-audit", "--", "true"]) == 2


def test_a_tool_package_that_predates_the_queue_is_told_apart(tmp_path):
    from research import remote
    pkg = tmp_path / "research"
    pkg.mkdir()
    (pkg / "remote.py").write_text("def tool_identity(): pass\n")
    assert remote.knows_queue(None) and remote.knows_queue(Path(remote.__file__).parent)
    assert not remote.knows_queue(pkg) and not remote.knows_queue(tmp_path / "nope")


def test_an_unknown_command_says_how_old_the_checkout_is(tmp_path, capsys, monkeypatch):
    import subprocess

    from research.cli import checkout_age

    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True, env=env).stdout
    git("init", "-q")
    for i in range(3):
        git("commit", "-q", "--allow-empty", "-m", str(i))
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    sha = git("rev-parse", "--short", "HEAD~2").strip()
    git("checkout", "-q", "HEAD~2")
    assert checkout_age(tmp_path) == f"This checkout is at {sha}, 2 commits behind origin/main as of its last fetch."
    git("checkout", "-q", "origin/main")
    assert checkout_age(tmp_path).endswith("level with origin/main as of its last fetch.")
    git("update-ref", "-d", "refs/remotes/origin/main")
    assert checkout_age(tmp_path) == f"This checkout is at {git('rev-parse', '--short', 'HEAD').strip()}."
    assert checkout_age(tmp_path / "nope") == ""

    monkeypatch.setattr("research.cli.checkout_age", lambda: f"This checkout is at {sha}.")
    assert main(["no-such-command"]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"research: unknown command 'no-such-command'. This checkout is at {sha}. If 'no-such-command' is "
                          "newer than that, update the checkout (`git pull`) and retry.\n\n`research` command line.")
    monkeypatch.setattr("research.cli.checkout_age", lambda: "")
    assert main(["no-such-command"]) == 2
    assert capsys.readouterr().err.startswith("research: unknown command 'no-such-command'\n\n`research` command line.")
