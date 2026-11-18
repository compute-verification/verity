"""A machine's host by its name: `research pods env NAME ...` prints `export <NAME>_HOST=<host>` from the register, `research run
--env-host NAME` puts that variable in the workload's environment, and a run's record keeps the variable's name with its value
redacted, as a secret-named key's is (`telemetry.job.SECRET_KEY`). The register is one made under tmp_path, with documentation
addresses only; nothing here reads the network."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from research import cli, remote
from research.env import DIRTY_CAPTURE
from research.pods import registry
from research.store.attempt import store_root
from research.telemetry import job as job_rule

CHALLENGER, AUDITOR, JUMP, IN_COMMAND = "192.0.2.10", "198.51.100.7", "203.0.113.5", "203.0.113.6"
#: a workload that writes down the challenger's variable as its process starts with it
SEES = "import json, os; json.dump({'host': os.environ.get('VY_CHALLENGER_1_HOST')}, open('result.json', 'w'))"


@pytest.fixture
def reg(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "notes" / "machines.d"
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(d))
    monkeypatch.setenv("RESEARCH_NOTES", str(d.parent))
    registry.write_entry(d, "vy-challenger-1", {"provider": "ssh", "host": CHALLENGER, "project": "verity"})
    registry.write_entry(d, "vy-auditor-2", {"provider": "ssh", "host": AUDITOR, "project": "verity"})
    registry.write_entry(d, "vy-pod", {"provider": "runpod", "pod_id": "abc123def45678", "project": "verity"})
    return d


def _files(*roots: Path) -> dict[Path, str]:
    """Every file under the roots but the capture of a dirty checkout, which holds this test's own source."""
    return {p: p.read_text(errors="replace") for r in roots for p in r.rglob("*") if p.is_file() and p.name != DIRTY_CAPTURE}


def test_the_variable_is_the_name_in_capitals_with_dashes_as_underscores_then_host():
    names = ("vy-challenger-1", "vy-auditor-1", "vy-auditor-2", "vy-nebius-2")
    assert [registry.host_var(n) for n in names] == ["VY_CHALLENGER_1_HOST", "VY_AUDITOR_1_HOST", "VY_AUDITOR_2_HOST", "VY_NEBIUS_2_HOST"]


def test_pods_env_prints_one_export_line_per_named_machine_from_the_register(reg, capsys):
    assert cli.main(["pods", "env", "vy-auditor-2", "vy-challenger-1"]) == 0
    out = capsys.readouterr().out
    assert out == f"export VY_AUDITOR_2_HOST={AUDITOR}\nexport VY_CHALLENGER_1_HOST={CHALLENGER}\n"


def test_an_unregistered_name_is_refused_naming_the_file_and_nothing_is_printed_for_any(reg, capsys):
    assert cli.main(["pods", "env", "vy-challenger-1", "vy-nope"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and str(reg / "vy-nope.toml") in err and CHALLENGER not in err
    assert cli.main(["pods", "env", "vy-challenger-1", "vy-pod"]) == 2, "a RunPod pod has no host in the register"
    out, err = capsys.readouterr()
    assert out == "" and "RunPod" in err


def test_env_host_puts_the_machines_variable_in_a_local_runs_environment_and_refuses_an_unregistered_one(reg, tmp_path, capsys):
    runs = tmp_path / "runs"
    assert cli.main(["run", "--runs-dir", str(runs), "--no-sampler", "--env-host", "vy-challenger-1", "--", sys.executable, "-c", SEES]) == 0
    (run_dir,) = runs.iterdir()
    assert json.loads((run_dir / "result.json").read_text())["host"] == CHALLENGER
    names = set(os.environ)
    assert "VY_CHALLENGER_1_HOST" not in names, "set for the workload alone"
    capsys.readouterr()
    assert cli.main(["run", "--runs-dir", str(runs), "--env-host", "vy-nope", "--", "true"]) == 2
    assert str(reg / "vy-nope.toml") in capsys.readouterr().err and len(list(runs.iterdir())) == 1, "refused before any run dir"


def test_env_host_goes_to_the_machine_in_the_launch_requests_env(reg, tmp_path, monkeypatch):
    staged = {}

    def launch(**kw):
        staged.update(kw["env"])
        return {"phase": "refused", "launch": {"refused": "a test's stub", "fault": "Refused"}}

    monkeypatch.setattr(remote, "launch", launch)
    argv = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "vy-auditor-2", "--project", "verity", "--no-custody-r2",
            "--env-host", "vy-challenger-1", "--env-host", "vy-auditor-2", "--", "true"]
    assert cli.main(argv) == 3
    assert (staged.get("VY_CHALLENGER_1_HOST"), staged.get("VY_AUDITOR_2_HOST")) == (CHALLENGER, AUDITOR)


def test_the_record_keeps_a_host_variables_name_and_redacts_its_value(reg, tmp_path, monkeypatch):
    """For --env-host, for an --env X_HOST= given by hand, and on the machine for a request's env: no file of the run dir or the
    store holds the address."""
    store = store_root()
    for case, flags in (("env-host", ["--env-host", "vy-challenger-1"]), ("by-hand", ["--env", f"VY_CHALLENGER_1_HOST={CHALLENGER}"])):
        runs = tmp_path / case
        assert cli.main(["run", "--runs-dir", str(runs), "--no-sampler", *flags, "--", "true"]) == 0, case
        (run_dir,) = runs.iterdir()
        env = json.loads((run_dir / "job.json").read_text())["store"]["params"]["env"]
        assert env.get("VY_CHALLENGER_1_HOST") == "<redacted>", case
        assert [p for p, text in _files(run_dir, store).items() if CHALLENGER in text] == [], case
    runs, req = tmp_path / "claimed", tmp_path / "request.json"
    (runs / "q1").mkdir(parents=True)
    req.write_text(json.dumps({"env": {"VY_AUDITOR_2_HOST": AUDITOR}}))
    claimed = ["--id", "q1", "--claimed", "--remote-json", json.dumps({"request": str(req)})]
    assert cli.main(["run", "--runs-dir", str(runs), "--no-sampler", *claimed, "--", "true"]) == 0
    env = json.loads((runs / "q1" / "job.json").read_text())["store"]["params"]["env"]
    assert env.get("VY_AUDITOR_2_HOST") == "<redacted>"
    assert [p for p, text in _files(runs / "q1").items() if AUDITOR in text] == []


def test_fleet_jump_and_a_host_variable_in_the_command_are_redacted_too_and_other_keys_are_not(tmp_path):
    """FLEET_JUMP given with --env is in no file of the run dir or the store. A `K=V` word of the command is redacted in job.json
    and the attempt by the same rule (`redact_argv`), and its value is hidden in status.json's copy of the command."""
    runs, store = tmp_path / "runs", store_root()
    argv = ["run", "--runs-dir", str(runs), "--no-sampler", "--env", f"FLEET_JUMP=research@{JUMP}", "--env", "PLAIN=kept",
            "--", "env", f"VY_X_HOST={IN_COMMAND}", "true"]
    assert cli.main(argv) == 0
    (run_dir,) = runs.iterdir()
    job = json.loads((run_dir / "job.json").read_text())
    env = job["store"]["params"]["env"]
    assert (env.get("FLEET_JUMP"), env.get("PLAIN")) == ("<redacted>", "kept")
    assert job["execution"]["command"] == ["env", "VY_X_HOST=<redacted>", "true"]
    assert [p for p, text in _files(run_dir, store).items() if JUMP in text or IN_COMMAND in text] == []


#: a workload that writes down what it was given, so a test sees the run receive the real value
GIVEN = "import json, os, sys; json.dump({'env': os.environ.get('GIVEN'), 'argv': sys.argv[1:]}, open('result.json', 'w'))"


def _records(run_dir: Path, store: Path) -> dict[Path, str]:
    """The run's records and the store's copies: every file but what the workload itself wrote (result.json)."""
    return {p: t for p, t in _files(run_dir, store).items() if p.name != "result.json"}


@pytest.mark.parametrize("form", ["challenger", "gate", "literal"])
def test_an_address_inside_another_keys_value_or_typed_in_the_command_is_named_in_every_record(reg, tmp_path, form):
    """The three forms seen: `gate:research@ADDRESS` under a `*_CHALLENGER` key (a registered machine's address: its name stands
    in), `ssh ... research@ADDRESS` under a `*_GATE` key whose address is the value of an unregistered `*_HOST` variable (the
    redaction mark stands in), and a registered address typed in the command where `${NAME_HOST}` was meant. The workload gets
    the real value; no record of the run dir or the store holds it."""
    unregistered = "192.0.2.77"
    env, words, addr, said = {
        "challenger": ([f"VY_CHALLENGER=gate:research@{CHALLENGER}"], [], CHALLENGER, "gate:research@vy-challenger-1"),
        "gate": ([f"OTHER_HOST={unregistered}", f"VY_GATE=ssh -p 22 research@{unregistered} true"], [], unregistered,
                 "ssh -p 22 research@<redacted> true"),
        "literal": ([], [f"research@{AUDITOR}:/srv"], AUDITOR, "research@vy-auditor-2:/srv"),
    }[form]
    given = env[-1].partition("=")[2] if env else words[0]
    runs, store = tmp_path / "runs", store_root()
    flags = [x for e in env for x in ("--env", e)] + ["--env", f"GIVEN={given}"]
    assert cli.main(["run", "--runs-dir", str(runs), "--no-sampler", *flags, "--", sys.executable, "-c", GIVEN, *words]) == 0
    (run_dir,) = runs.iterdir()
    got = json.loads((run_dir / "result.json").read_text())
    assert (got["env"], got["argv"]) == (given, words), "the run receives the real value"
    job = json.loads((run_dir / "job.json").read_text())
    recorded = job["store"]["params"]["env"]["GIVEN"] if not words else job["execution"]["command"][-1]
    assert recorded == said
    assert [p.name for p, text in _records(run_dir, store).items() if addr in text] == []


def test_a_claimed_runs_records_name_the_requests_hidden_addresses(tmp_path):
    """On the machine there is no register: the request's own `hidden_addresses`, which the launcher wrote, decide the names, the
    target machine's own address among them."""
    runs, req = tmp_path / "claimed", tmp_path / "request.json"
    (runs / "q2").mkdir(parents=True)
    req.write_text(json.dumps({"env": {"VY_GATE": f"ssh research@{CHALLENGER} true"},
                               remote.HIDDEN_ADDRESSES: {CHALLENGER: "vy-challenger-1", AUDITOR: "vy-auditor-2"}}))
    claimed = ["--id", "q2", "--claimed", "--remote-json", json.dumps({"request": str(req), "host": AUDITOR, "machine": "vy-auditor-2"})]
    assert cli.main(["run", "--runs-dir", str(runs), "--no-sampler", *claimed, "--", "echo", f"research@{CHALLENGER}"]) == 0
    job = json.loads((runs / "q2" / "job.json").read_text())
    assert (job["run"]["remote"]["host"], job["execution"]["command"]) == ("vy-auditor-2", ["echo", "research@vy-challenger-1"])
    assert job["store"]["params"]["env"]["VY_GATE"] == "ssh research@vy-challenger-1 true"
    records = {p.name: t for p, t in _files(runs / "q2").items() if p.name != "stdout.log"}
    assert [n for n, text in records.items() if CHALLENGER in text or AUDITOR in text] == []
    assert (runs / "q2" / "stdout.log").read_text() == f"research@{CHALLENGER}\n", "the workload's own output is its own"


def test_the_launcher_ships_names_for_the_registered_addresses_its_request_holds_and_no_others(reg, tmp_path, monkeypatch):
    """The target's own address and a registered one typed in the command are named; a registered machine the request never
    mentions is not sent to the machine."""
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "192.0.2.30", "port": 22, "user": "research", "pod_name": "vy-pod",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "wake", lambda m, ssh: None)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: "/r/tool/x")
    staged = []

    def stage(ssh, root, run_id, request, sends, private=None):
        staged.append(request)
        raise remote.Refused("staged")

    monkeypatch.setattr(remote, "stage_request", stage)
    with pytest.raises(remote.Refused, match="staged"):
        remote.launch(run_id="h1", local_dir=tmp_path / "runs" / "h1", machine_name="vy-pod", project="verity", source=None,
                      command=["ssh", f"research@{CHALLENGER}", "true"], harness_args=["--stage", "t"], sends=[], cwd=None,
                      env={"OTHER_HOST": JUMP}, exclusive=False)
    assert staged[0][remote.HIDDEN_ADDRESSES] == {"192.0.2.30": "vy-pod", CHALLENGER: "vy-challenger-1", JUMP: "<redacted>"}


def test_an_address_is_hidden_only_as_a_whole_token():
    """A longer address or name that starts or ends with it is another machine's, and is left as it is."""
    names = {CHALLENGER: "vy-challenger-1"}
    doc = {"a": f"${{H:-{CHALLENGER}}}", "b": f"{CHALLENGER}1 x{CHALLENGER} {CHALLENGER}.5", f"k@{CHALLENGER}": [f"{CHALLENGER}."]}
    assert job_rule.hide_addresses(doc, names) == {"a": "${H:-vy-challenger-1}", "b": doc["b"], "k@vy-challenger-1": ["vy-challenger-1."]}
    assert job_rule.address_names({"X_HOST": "1", "Y_HOST": JUMP}, {"vy-challenger-1": CHALLENGER}, ["Z_HOST=" + IN_COMMAND]) == {
        CHALLENGER: "vy-challenger-1", JUMP: "<redacted>", IN_COMMAND: "<redacted>"}
