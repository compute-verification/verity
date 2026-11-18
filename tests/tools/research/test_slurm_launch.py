"""`research run --on NAME` for a `provider = "slurm"` machine (pods/slurm.py): the payload it builds with infra/vcf/payload.sh,
the sbatch arguments the run's flags give, the machine and properties it hands the job, its refusals, `status` and `fetch` once
the run is published, and the register's listing.  The `via` machine is `sh -c` on a local directory, where a tmux and a
vcf-job-run.sh stub record what they were given; its entry's host is a dummy."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

from research import cli, remote
from research.pods import connect, registry, slurm

PY = sys.executable
SSH_BASE = connect.ssh_base
PROPS = '[properties]\ncards = { value = 8, read = "sinfo" }\nlease = { value = "none", read = "Slurm allocates the cards" }\n'
NODE = ('provider = "slurm"\nvia = "vy-via"\npartition = "main"\nnodelist = "n-0"\nreservation = "r-1"\ngres = "gpu:1"\n'
        f'max_minutes = 30\nproject = "verity"\n\n{PROPS}')
SHOW = ("import os, pathlib; d = pathlib.Path(os.environ['RESEARCH_RUN_DIR']) / 'inputs'; "
        "print(os.environ.get('RESEARCH_MACHINE'), sorted(p.name for p in d.iterdir()) if d.is_dir() else [])")


def _exe(p: Path, text: str) -> Path:
    p.write_text(text)
    p.chmod(0o755)
    return p


@pytest.fixture
def site(tmp_path, monkeypatch):
    """The register (vy-via, an ssh machine, and vcf-node), the via machine's directories and stubs, and a source checkout."""
    reg = tmp_path / "notes" / "machines.d"
    reg.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(reg))
    monkeypatch.setenv("RESEARCH_NOTES", str(reg.parent))
    (reg / "vy-via.toml").write_text('provider = "ssh"\nhost = "via.invalid"\nproject = "verity"\n')
    (reg / "vcf-node.toml").write_text(NODE)
    via, stubs = tmp_path / "via", tmp_path / "bin"
    stubs.mkdir()
    _exe(stubs / "tmux", '#!/bin/sh\ncase "$1" in new-session) shift 4; sh -c "$1"; exit 0;; *) exit 1;; esac\n')
    monkeypatch.setenv("PATH", str(stubs), prepend=os.pathsep)
    job_run = _exe(stubs / "vcf-job-run.sh", '#!/bin/sh\nd=$(dirname "$0")\nprintf "%s\\n" "$@" > "$d/args"\n'
                                              'printf "%s\\n%s\\n" "$SEND" "$OUT" > "$d/where"\necho "00:00:00Z job 4242"\n')
    monkeypatch.setattr(slurm, "VIA_LAUNCHES", str(via / "launches"))
    monkeypatch.setattr(slurm, "VIA_JOBS", str(via / "jobs"))
    monkeypatch.setattr(slurm, "JOB_RUN", str(job_run))
    monkeypatch.setattr(connect, "ssh_base", lambda target: ["sh", "-c"])
    src = tmp_path / "src"
    for rel in (slurm.MARKER, "infra/vcf/vcf-job-run.sh", "README"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel + "\n")
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    return {"reg": reg, "via": via, "stubs": stubs, "src": src, "runs": tmp_path / "runs"}


def _run(site: dict, *flags: str, command: tuple[str, ...] = ("python", "-c", "pass")) -> int:
    return cli.main(["run", "--runs-dir", str(site["runs"]), "--on", "vcf-node", "--project", "verity", "--source", str(site["src"]),
                     *flags, "--", *command])


def test_a_launch_submits_the_payload_through_via_with_the_runs_flags_as_sbatch_arguments(site, capsys):
    """Rule: --gpus, --cpus, --mem-gb and --timeout become the job's sbatch arguments and minutes, beside the entry's node and
    reservation; weakening: a run that reaches the node with the entry's defaults whatever it asked."""
    assert _run(site, "--gpus", "2", "--cpus", "8", "--mem-gb", "40.5", "--timeout", "20m", "--env", "K=V") == 0
    err = capsys.readouterr().err
    (run_dir,) = site["runs"].iterdir()
    rid = run_dir.name
    assert (site["stubs"] / "args").read_text().split("\n")[:-1] == [
        rid, "main", "20", slurm.JOB_SCRIPT, "--gres=gpu:2", "--cpus-per-task=8", "--mem=41G", "--nodelist=n-0", "--reservation=r-1"]
    send, out = (site["stubs"] / "where").read_text().split()
    assert (send, out) == (f"{site['via']}/launches/{rid}/payload.tar.gz", f"{site['via']}/jobs/{rid}")
    assert f"run {rid} submitted to vcf-node as Slurm job 4242" in err and f"research status {rid}" in err
    rec = json.loads((run_dir / "remote.json").read_text())
    assert (rec["provider"], rec["via"], rec["slurm_job"], rec["minutes"]) == ("slurm", "vy-via", "4242", 20)
    assert "via.invalid" not in err + json.dumps(rec)

    with tarfile.open(send, "r:gz") as tf:
        names = tf.getnames()
        request = tf.extractfile("request.sh").read().decode()
        copy = tf.extractfile("machines.d/vcf-node.toml").read().decode()
    assert {"tree/README", f"tree/{slurm.MARKER}", "request.sh", "machines.d/vcf-node.toml"} <= set(names)
    for line in (f"VCF_RUN_ID={rid}", "VCF_MACHINE=vcf-node", "VCF_TIMEOUT_MIN=10", "VCF_ENV=(K=V )", "VCF_CMD=(python -c pass )"):
        assert line in request.splitlines()
    assert tomllib.loads(copy) == {"machine": "vcf-node", "properties": tomllib.loads(PROPS)["properties"]}
    assert [v for v in ("vy-via", "via.invalid", "partition", "r-1") if v in copy] == []


def test_no_gpus_asked_takes_the_entrys_gres_and_zero_takes_none(site):
    assert _run(site) == 0
    assert (site["stubs"] / "args").read_text().split("\n")[2:5] == ["30", slurm.JOB_SCRIPT, "--gres=gpu:1"]
    assert _run(site, "--gpus", "0", "--timeout", "15m") == 0
    assert (site["stubs"] / "args").read_text().split("\n")[2:5] == ["15", slurm.JOB_SCRIPT, "--nodelist=n-0"]


@pytest.mark.parametrize("flags, command, said", [
    (("--timeout", "45m"), None, r"asks 45 min .*above vcf-node's max_minutes 30 \(its register entry\): pass --timeout 30m or less"),
    (("--max-min", "31"), None, r"above vcf-node's max_minutes 30"),
    (("--pool", "research", "--as", "infra"), None, r"--pool: vcf-node has no pool or slot of ours: drop it; Slurm gives the job"),
    (("--gpu-host",), None, r"--gpu-host: no card lease of ours runs on vcf-node: ask the cards with --gpus N"),
    ((), ("gpu-lease", "1", "--", "python", "-c", "pass"), r"asks gpu-lease for cards.*ask the cards with --gpus N"),
    (("--wait",), None, r"--wait: its records reach the store only once `via` publishes them"),
    (("--env-host", "vy-via"), None, r"--env-host: it would send a machine's address to the cluster"),
])
def test_a_run_a_slurm_machine_cannot_take_is_refused_by_name_before_anything_is_sent(site, capsys, flags, command, said):
    assert _run(site, *flags, **({"command": command} if command else {})) == 2
    assert re.search(said, capsys.readouterr().err)
    assert not site["via"].exists() and not (site["runs"].exists() and any(site["runs"].iterdir()))


def test_a_commit_before_the_slurm_launch_is_refused_and_leaves_no_run(site, capsys):
    git = ["git", "-C", str(site["src"]), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "rm", "-q", slurm.MARKER], check=True)
    subprocess.run([*git, "commit", "-qm", "old"], check=True)
    assert _run(site) == 3
    assert "predates the Slurm launch" in capsys.readouterr().err and not any(site["runs"].iterdir())


class Store:
    """The store's two calls a slurm run's fetch makes: its attempt, and its run record laid out in a directory."""

    def __init__(self, attempt: dict | None, files: dict[str, str]):
        self.attempt, self.files = attempt, files

    def get_attempt(self, run_id: str) -> dict | None:
        return self.attempt

    def fetch(self, art: str, to: Path) -> Path:
        for rel, text in self.files.items():
            (to / rel).parent.mkdir(parents=True, exist_ok=True)
            (to / rel).write_text(text)
        return to


def test_status_reads_vias_log_until_the_run_is_published_and_fetch_reads_the_store_after(site, capsys, monkeypatch):
    assert _run(site) == 0
    (run_dir,) = site["runs"].iterdir()
    rid = run_dir.name
    (site["via"] / "jobs" / rid).mkdir(parents=True)
    (site["via"] / "jobs" / rid / "vcf-job.log").write_text("00:00:00Z job 4242\n00:02:00Z state RUNNING\n")
    monkeypatch.setattr(slurm, "_open_store", lambda: Store(None, {}))
    capsys.readouterr()
    assert cli.main(["status", "--runs-dir", str(site["runs"]), rid]) == 0
    out = capsys.readouterr()
    assert "not in the store yet; vy-via's log of Slurm job 4242: 00:02:00Z state RUNNING" in out.err and "submitted" in out.out

    published = Store({"outputs": {"run_record": "art:0"}}, {"status.json": json.dumps({"state": "done", "rc": 0}),
                                                             "stdout.log": "hi\n", "remote.json": "{}"})
    monkeypatch.setattr(slurm, "_open_store", lambda: published)
    assert cli.main(["fetch", "--runs-dir", str(site["runs"]), rid]) == 0
    assert f"{rid} observed" in (said := capsys.readouterr().out) and "done rc=0" in said
    assert (run_dir / "stdout.log").read_text() == "hi\n"
    assert json.loads((run_dir / "remote.json").read_text())["phase"] == "published"


def test_another_vm_finds_a_published_slurm_run_in_the_store(site, capsys, monkeypatch):
    rid = "r20261118-020000-abcd"
    monkeypatch.setattr(remote, "adopt", lambda *a, **k: (_ for _ in ()).throw(remote.Refused("no machine holds it")))
    monkeypatch.setattr(slurm, "_open_store", lambda: Store({"execution": {"machine": "vcf-node"}, "outputs": {"run_record": "art:0"}},
                                                           {"status.json": json.dumps({"state": "done", "rc": 0})}))
    assert cli.main(["status", "--runs-dir", str(site["runs"]), rid]) == 0
    assert rid in capsys.readouterr().out and (site["runs"] / rid / "status.json").is_file()


def test_a_local_only_run_with_machine_gets_its_name_and_properties_copy(site, capsys):
    """The contract's form for a job already on the node: the register there holds the payload's copy."""
    runs = site["runs"]
    assert cli.main(["run", "--runs-dir", str(runs), "--machine", "vcf-node", "--", PY, "-c", SHOW]) == 0
    (run_dir,) = runs.iterdir()
    assert (run_dir / "stdout.log").read_text() == "vcf-node ['vcf-node.toml']\n"
    assert tomllib.loads((run_dir / "inputs" / "vcf-node.toml").read_text())["machine"] == "vcf-node"
    assert cli.main(["run", "--runs-dir", str(runs), "--machine", "nowhere", "--", PY, "-c", SHOW]) == 0
    assert "nowhere []\n" in {(d / "stdout.log").read_text() for d in runs.iterdir()}
    assert cli.main(["run", "--runs-dir", str(runs), "--on", "vcf-node", "--project", "verity", "--machine", "vcf-node",
                     "--", PY, "-c", "pass"]) == 2
    assert "--machine NAME names the machine a --local-only run is already on" in capsys.readouterr().err


def test_the_register_lists_a_slurm_machine_and_refuses_one_that_names_an_address(site, capsys):
    rows = {r["name"]: r for r in registry.rows(remote.machines_path())}
    assert (rows["vcf-node"]["source"], rows["vcf-node"]["pod_id"], rows["vcf-node"]["problem"]) == ("machines.d", "slurm via vy-via", None)
    entry = tomllib.loads(NODE)
    assert registry.validate("vcf-node", {**entry, "host": "dummy.invalid"}).startswith("host: a slurm machine's entry names no address")
    assert "partition 'main; true'" in registry.validate("vcf-node", {**entry, "partition": "main; true"})
    assert "max_minutes" in registry.validate("vcf-node", {**entry, "max_minutes": "30"})
    with pytest.raises(remote.Refused, match=r"reached only through batch jobs its via machine vy-via submits: `research run --on"):
        SSH_BASE("vcf-node")


def test_payload_sh_refuses_properties_without_a_machine(site):
    r = subprocess.run(["bash", str(slurm.PAYLOAD_SH), "HEAD", "--properties", "x", "--", "true"], cwd=site["src"], capture_output=True,
                       text=True)
    assert r.returncode == 2 and "--properties is the copy of --machine NAME's table" in r.stderr
