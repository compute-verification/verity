"""What every `research run` job starts with unless it says otherwise (`result.env_defaults`): oneDNN held off AMX, for a run
here, a run a machine claimed and a job user's unit. No test needs a CPU with AMX: each reads the environment a workload got."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from research import cli, jobuser, result

NAME, CAP = result.ONEDNN_ISA
#: a workload that writes down the variable as its process starts with it
SEES = f"import json, os; json.dump({{'isa': os.environ.get({NAME!r})}}, open('result.json', 'w'))"


def _run(runs: Path, *flags: str, command: tuple[str, ...] = (sys.executable, "-c", SEES)) -> tuple[str | None, dict | None]:
    """One run, the only one of its runs root: (the variable as its workload saw it, the job's environment as its record lists it)."""
    assert cli.main(["run", "--runs-dir", str(runs), "--no-sampler", *flags, "--", *command]) == 0
    (run_dir,) = runs.iterdir()
    seen = json.loads((run_dir / "result.json").read_text())["isa"]
    return seen, json.loads((run_dir / "job.json").read_text())["store"]["params"].get("env")


def test_the_default_is_the_cap_unless_the_environment_names_the_variable_and_a_job_users_unit_keeps_its_name():
    assert result.env_defaults({}) == {NAME: CAP} == {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}
    assert result.env_defaults({NAME: "AVX2", "OTHER": "1"}) == {NAME: "AVX2"}
    assert set(result.env_defaults({})) <= jobuser.PASS_NAMES and not jobuser.CREDENTIAL.search(NAME)


def test_a_job_starts_with_onednn_held_off_amx_and_its_record_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    assert _run(tmp_path / "runs") == (CAP, {NAME: CAP})
    assert NAME not in dict(os.environ), "set for the workload: the runner's own environment is as it was"


def test_a_value_the_job_or_the_runners_environment_names_stays(tmp_path, monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    assert _run(tmp_path / "env", "--env", f"{NAME}=AVX2") == ("AVX2", {NAME: "AVX2"})
    own = ("env", f"{NAME}=AVX512_CORE", sys.executable, "-c", SEES)
    assert _run(tmp_path / "command", command=own) == ("AVX512_CORE", {NAME: CAP}), "the job's own command decides last"
    monkeypatch.setenv(NAME, "AVX512_CORE_VNNI")
    assert _run(tmp_path / "inherited") == ("AVX512_CORE_VNNI", {NAME: "AVX512_CORE_VNNI"})
    assert _run(tmp_path / "both", "--env", f"{NAME}=AVX2") == ("AVX2", {NAME: "AVX2"})
    assert dict(os.environ)[NAME] == "AVX512_CORE_VNNI"


def test_a_run_a_machine_claimed_gets_it_under_its_requests_env(tmp_path, monkeypatch):
    """Machine side, the launch request's env is the run's --env: a request that names the variable keeps its value."""
    monkeypatch.delenv(NAME, raising=False)
    for case, env, want in (("plain", {"RSEC_ROW": "1"}, CAP), ("named", {NAME: "AVX2"}, "AVX2")):
        runs, req = tmp_path / case, tmp_path / f"{case}.json"
        (runs / "q1").mkdir(parents=True)
        req.write_text(json.dumps({"env": env}))
        claimed = ("--id", "q1", "--claimed", "--remote-json", json.dumps({"request": str(req)}))
        assert _run(runs, *claimed) == (want, {**env, NAME: want}), case


def test_a_job_users_unit_gets_it_and_a_value_the_run_names_wins(tmp_path, monkeypatch):
    """A job user's unit starts from a clean environment, built by job-exec (`jobuser.job_env`) from the one the runner spawned it
    with and the names the run passes: it keeps the variable, at the default or at what the run's --env says."""
    monkeypatch.delenv(NAME, raising=False)
    monkeypatch.setattr(jobuser, "check_user", lambda name: SimpleNamespace(pw_name=name, pw_uid=990, pw_gid=990))
    monkeypatch.setattr(jobuser, "UNREACHABLE", ())
    monkeypatch.setattr(cli, "_survive_stray_signals", lambda: None)
    finish, spawned = jobuser.finish, []

    def as_root(argv, **kw):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(jobuser, "finish", lambda *a, **k: finish(*a, run=as_root, **k))

    def harness(a, cmd):
        from research.telemetry.job import set_status

        spawned.append((cmd, dict(os.environ)))
        set_status(Path(a.attempt_dir), "done", rc=0, signal=None)
        return 0

    monkeypatch.setattr("research.telemetry.run.run", harness)
    base = ["run", "--runs-dir", str(tmp_path / "runs"), "--job-user", "vyjob", "--no-sampler"]
    assert cli.main([*base, "--", "true"]) == 0 and cli.main([*base, "--env", f"{NAME}=AVX2", "--", "true"]) == 0
    units = []
    for cmd, environ in spawned:
        passed = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--pass"]
        units.append((passed, jobuser.job_env(environ, passed)[NAME]))
    assert units == [([], CAP), ([NAME], "AVX2")]
    assert NAME not in dict(os.environ)
