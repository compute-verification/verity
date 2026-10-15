"""`research doctor`: each check against fakes of what it probes (no network), and no value of any secret in what it prints."""
from __future__ import annotations

import base64
import json
import math
import os
import select
import shlex
import ssl
import subprocess
import threading
import urllib.error
from pathlib import Path

import pytest

from research import cli, doctor, notes_sync, runs, secret_hint, slack
from research.doctor import FINDING, OK, SKIP, Check
from research.pods import runpod
from research.store import remote as store_remote

ALL_NAMES = sorted({n for need in doctor.SECRETS.values() for alts in need for n in doctor._alts(alts)})
LANE = [n for alts in doctor.needs("lane") for n in alts]


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for n in [*ALL_NAMES, "JOBS_TOKEN"]:    # JOBS_TOKEN: these tests' name that only the VM injects, set on a real VM
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(secret_hint, "RECOVERED", [])
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "absent-machines.toml"))
    monkeypatch.setenv("RESEARCH_NOTES", str(tmp_path / "absent-notes"))
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))


class Runs:
    """`doctor.run` faked: each argv's answer by its program, every call kept (argv, cwd, the env it set)."""

    def __init__(self, monkeypatch, **answers):
        self.answers, self.calls = answers, []
        monkeypatch.setattr(doctor, "run", self)

    def __call__(self, argv, *, cwd=None, env=None):
        self.calls.append((argv, cwd, env or {}))
        return self.answers[Path(argv[0]).name]


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _clone(tmp_path: Path, origin: str | None = None) -> Path:
    bare = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(bare))
    _git("clone", "-q", str(bare), str(tmp_path / "clone"))
    clone = tmp_path / "clone"
    (clone / "a").write_text("a\n")
    _git("add", "a", cwd=clone)
    _git("commit", "-qm", "a", cwd=clone)
    if origin:
        _git("remote", "set-url", "origin", origin, cwd=clone)
    return clone


# ---- secrets --------------------------------------------------------------------------------------------------------------------------

def test_secrets_name_what_the_role_lacks_and_where_to_add_it(monkeypatch):
    for n in LANE[:-1]:
        monkeypatch.setenv(n, "dummy")
    c = doctor.check_secrets("lane")
    assert (c.status, c.detail) == (FINDING, f"not set here: {LANE[-1]}") and "Cursor Dashboard" in c.fix
    monkeypatch.setenv(LANE[-1], "dummy")
    assert doctor.check_secrets("lane") == Check("secrets", OK, "the 7 names the lane role needs are set")
    assert doctor.check_secrets("coordinator").detail == "the 7 names the coordinator role needs are set"    # a lane's, and Slack's broker
    c = doctor.check_secrets("infra")
    assert c.detail == "not set here: NEBIUS_SA_ID, NEBIUS_SA_PUBLIC_KEY_ID, NEBIUS_SA_PRIVATE_KEY_B64|NEBIUS_SA_PRIVATE_KEY"
    for n in ("NEBIUS_SA_ID", "NEBIUS_SA_PUBLIC_KEY_ID", "NEBIUS_SA_PRIVATE_KEY"):                       # either form of the key will do
        monkeypatch.setenv(n, "dummy")
    assert doctor.check_secrets("infra").status == OK


def test_a_secret_the_vm_injects_and_lost_is_a_finding_whose_fix_is_a_fresh_shell_or_a_new_vm(monkeypatch):
    """ci's VM on 3 Oct: the injected values gone from every shell after an exec-daemon restart, unnoticed."""
    for n in LANE:
        monkeypatch.setenv(n, "dummy")
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", ",".join([*LANE, "JOBS_TOKEN"]))
    c = doctor.check_secrets("lane")
    assert c.detail == "not set here: JOBS_TOKEN" and "this VM injects it" in c.fix
    assert secret_hint.FRESH_SHELL in c.fix and "a new agent VM gets them at boot" in c.fix and "Dashboard" not in c.fix
    monkeypatch.setenv("JOBS_TOKEN", "dummy")
    monkeypatch.setattr(secret_hint, "RECOVERED", ["R2_BUCKET", "JOBS_TOKEN"])
    c = doctor.check_secrets("lane")
    assert c.status == OK and c.detail.startswith("the 7 names the lane role needs and the 8 this VM injects are set; 2 taken back from "
                                                  "the tmux server because this shell had lost them (R2_BUCKET, JOBS_TOKEN)")


# ---- what a check prints ------------------------------------------------------------------------------------------------------------

def _world(monkeypatch, tmp_path, *, failing: bool):
    """Every name set to a value of its own, and every probe faked to fail (or to pass)."""
    values = {n: f"not-to-be-printed-{i:02d}" for i, n in enumerate([*ALL_NAMES, "JOBS_TOKEN", "SLACK_BOT_TOKEN"])}
    for n, v in values.items():
        monkeypatch.setenv(n, v)
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", ",".join(values))
    if failing:
        monkeypatch.delenv("JOBS_TOKEN")
        Runs(monkeypatch, gh=(1, ""), git=(128, "fatal: Authentication failed for 'https://github.com/o/r.git/'"),
             ssh=(255, "research@192.0.2.10: Permission denied (publickey)."), uv=(1, " + pyyaml==6.0.3\nerror: The environment is outdated"))
        monkeypatch.setattr(store_remote, "load_remote", lambda: store_remote._remote_from(store_remote.env_remote_config(), "environment"))

        def post(url, data, headers, timeout=None):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        monkeypatch.setattr(doctor, "server_date", lambda: 0.0)
    else:
        Runs(monkeypatch, gh=(0, ""), git=(0, ""), ssh=(0, ""), uv=(0, "Would make no changes"))
        cfg = tmp_path / "store.toml"
        cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "remote"}"\n')
        monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))

        def post(url, data, headers, timeout=None):
            return {"ok": True}
        monkeypatch.setattr(doctor, "server_date", lambda: doctor.clock())
    monkeypatch.setattr(slack, "post", post)
    root = _clone(tmp_path)
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "pyvenv.cfg").write_text("home = /usr\n")
    monkeypatch.setattr(doctor, "checkout", lambda: root)
    monkeypatch.setattr(doctor, "active_venv", lambda: None)
    monkeypatch.setenv("RESEARCH_NOTES", str(root))
    (tmp_path / "machines.toml").write_text('[machines.node-a]\nprovider = "ssh"\nhost = "192.0.2.10"\nuser = "research"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(runpod, "SSH_KEY", tmp_path / "key")
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "known_hosts")
    (tmp_path / "key").write_text("dummy\n")
    monkeypatch.setattr(runs, "free_bytes", lambda p: 10**8 if failing else 100 * 10**9)
    return values


@pytest.mark.parametrize("failing", [True, False])
def test_no_value_of_any_secret_reaches_the_output_and_the_exit_code_says_whether_anything_was_found(monkeypatch, tmp_path, capsys, failing):
    values = _world(monkeypatch, tmp_path, failing=failing)
    assert cli.main(["doctor", "--role", "infra"]) == (1 if failing else 0)
    text = capsys.readouterr().out
    assert cli.main(["doctor", "--role", "infra", "--json"]) == (1 if failing else 0)
    out = capsys.readouterr().out
    doc = json.loads(out)
    leaked = [n for n, v in values.items() if v in text + out]
    assert leaked == []
    names = [c["name"] for c in doc["checks"]]
    assert names == ["secrets", "github", "git-push", "notes-push", "store", "slack", "ssh:node-a", "venv", "disk", "clock"]
    assert doc["schema"] == "research/doctor/v1" and doc["role"] == "infra" and doc["ok"] is not failing
    statuses = {c["name"]: c["status"] for c in doc["checks"]}
    assert set(statuses.values()) == ({FINDING} if failing else {OK}), statuses
    if failing:
        by = {c["name"]: c for c in doc["checks"]}
        assert by["store"]["detail"].startswith("the store's remote is misconfigured: environment remote: R2_BUCKET is not the research store's bucket")
        assert by["slack"]["detail"] == "auth.test: HTTP 401" and by["secrets"]["detail"] == "not set here: JOBS_TOKEN"
        assert text.startswith("research doctor (infra): 10 findings\nFINDING  secrets ")


# ---- github, git-push, notes-push ---------------------------------------------------------------------------------------------------

def test_gh_auth_status_s_exit_code_is_the_github_check(monkeypatch):
    runs_ = Runs(monkeypatch, gh=(0, "Logged in ... Token: gho_****"))
    assert doctor.check_github().status == OK
    argv, _, env = runs_.calls[0]
    assert argv == ["gh", "auth", "status", "--hostname", "github.com"] and env["GH_PROMPT_DISABLED"] == "1"
    runs_.answers["gh"] = (1, "")
    c = doctor.check_github()
    assert c.detail == "`gh auth status` exits 1: gh's token is missing, expired or revoked" and "a new agent VM renews" in c.fix
    runs_.answers["gh"] = (127, "")
    assert doctor.check_github().detail == "gh is not on PATH"


def test_a_dry_run_push_to_a_reachable_origin_is_ok_and_sends_nothing_and_runs_no_hook(tmp_path):
    clone = _clone(tmp_path)
    hook = clone / ".git" / "hooks" / "pre-push"
    hook.write_text(f"#!/bin/sh\ntouch {tmp_path / 'hook-ran'}\nexit 1\n")
    hook.chmod(0o755)
    assert doctor.check_git_push(clone) == Check("git-push", OK, f"origin of {clone} accepts a push (dry run)")
    assert not (tmp_path / "hook-ran").exists()
    assert subprocess.run(["git", "-C", str(tmp_path / "origin.git"), "for-each-ref"], capture_output=True, text=True).stdout == ""
    _git("remote", "remove", "origin", cwd=clone)
    c = doctor.check_git_push(clone)
    assert c.status == SKIP and "a checkout may be local on purpose (a notes clone may not)" in c.detail
    assert doctor.check_git_push(None).status == SKIP


@pytest.mark.parametrize("said, what, credential", [
    ("remote: Permission to o/r.git denied to cursor[bot].\nfatal: unable to access 'https://github.com/o/r.git/': "
     "The requested URL returned error: 403", "GitHub refused the credential write access", True),
    ("fatal: could not read Username for 'https://github.com': terminal prompts disabled", "no usable credential", True),
    ("fatal: unable to access 'https://github.com/o/r.git/': Could not resolve host: github.com", "the remote is unreachable", False),
    ("ERROR: Repository not found.", "no such repository", True),
    ("error: something else", "git exits 128", False),
])
def test_git_s_refusals_are_named_and_only_a_credential_s_fix_names_the_credential(monkeypatch, tmp_path, said, what, credential):
    runs_ = Runs(monkeypatch, git=(128, said))
    c = doctor.push_probe("git-push", tmp_path, [], "origin", "THE-CREDENTIAL-FIX")
    assert c.status == FINDING and what in c.detail and said not in c.detail + c.fix
    assert ("THE-CREDENTIAL-FIX" in c.fix) is credential
    shown = shlex.join(["git", "-C", str(tmp_path), "push", "--dry-run", "--no-verify", "origin", f"HEAD:{doctor.PROBE_REF}"])
    assert c.fix.endswith(f"`{shown}` shows git's message")
    argv, _, env = runs_.calls[0]
    assert argv[-2:] == ["origin", f"HEAD:{doctor.PROBE_REF}"] and env["GIT_TERMINAL_PROMPT"] == "0"
    assert "--dry-run" in argv and "--no-verify" in argv
    runs_.answers["git"] = (None, "")
    assert "gave no answer" in doctor.push_probe("git-push", tmp_path, [], "origin", "x").detail


def test_notes_push_needs_a_clone_and_pushes_through_the_token_s_helper(monkeypatch, tmp_path):
    c = doctor.check_notes_push()
    assert c.status == FINDING and c.detail == f"{tmp_path / 'absent-notes'} is not a git clone: notes written here reach nobody"
    clone = _clone(tmp_path, origin="https://github.com/danielreuter/research-notes.git")
    monkeypatch.setenv("RESEARCH_NOTES", str(clone))
    runs_ = Runs(monkeypatch, git=(128, "remote: Permission to danielreuter/research-notes.git denied to cursor[bot]."))
    c = doctor.check_notes_push()
    assert c.status == FINDING and "set RESEARCH_NOTES_TOKEN (a Cursor secret)" in c.fix
    assert runs_.calls[-1][0][-2] == "origin"                                     # no token: the remote as configured
    monkeypatch.setenv("RESEARCH_NOTES_TOKEN", "not-to-be-printed")
    c = doctor.check_notes_push()
    assert "RESEARCH_NOTES_TOKEN was refused" in c.fix
    argv = runs_.calls[-1][0]
    assert argv[-2] == "https://x-access-token@github.com/compute-verification/research-notes.git" and "credential.helper=" in argv
    assert not any("not-to-be-printed" in a for a in argv)                         # the helper reads it from the environment
    assert "push --dry-run --no-verify https://x-access-token@github.com/compute-verification/research-notes.git HEAD:" in c.fix
    assert "credential.helper=" in c.fix and "not-to-be-printed" not in c.fix      # the hint is the command the probe ran
    runs_.answers["git"] = (0, "")
    assert doctor.check_notes_push().status == OK
    _git("remote", "remove", "origin", cwd=clone)
    c = doctor.check_notes_push()
    assert c.status == FINDING and c.detail.startswith(f"the notes clone {clone} has no origin: notes committed here reach nobody")


# ---- store, slack ---------------------------------------------------------------------------------------------------------------

class _Remote:
    def __init__(self, error=None, answer=None):
        self.error, self.answer = error, answer

    def head(self, key):
        if self.error:
            raise self.error
        assert key == doctor.STORE_PROBE_KEY
        return self.answer


def test_the_store_check_reads_the_remote_s_answer_to_a_head(monkeypatch, tmp_path):
    c = doctor.check_store()                                                      # conftest: an absent store.toml
    assert c.status == FINDING and c.detail.startswith("no store remote") and "R2_ENDPOINT, R2_BUCKET" in c.fix
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "remote"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    assert doctor.check_store() == Check("store", OK, "the store's remote (FsRemote) answers")
    for code in (400, 401, 403):
        monkeypatch.setattr(store_remote, "load_remote", lambda code=code: _Remote(urllib.error.HTTPError("u", code, "x", {}, None)))
        c = doctor.check_store()
        assert c.detail == f"the store's remote (_Remote) refused a HEAD: HTTP {code}" and "AWS_SESSION_TOKEN" in c.fix
    for code, why in ((503, "it is failing"), (429, "it is rate limiting"), (418, "no answer a HEAD of an absent key gets")):
        monkeypatch.setattr(store_remote, "load_remote", lambda code=code: _Remote(urllib.error.HTTPError("u", code, "x", {}, None)))
        c = doctor.check_store()
        assert c.detail == f"the store's remote (_Remote) answered a HEAD with HTTP {code}: {why}"
        assert c.fix.startswith("retry later") and "AWS_" not in c.fix                # not the credential's fault
    monkeypatch.setattr(store_remote, "load_remote", lambda: _Remote(TimeoutError("not shown")))
    assert doctor.check_store().detail == "the store's remote (_Remote) gave no answer: TimeoutError"
    monkeypatch.setattr(store_remote, "load_remote", lambda: _Remote(answer={"bytes": 0, "etag": None}))
    c = doctor.check_store()
    assert c == Check("store", FINDING, f"the store's remote (_Remote) answered for {doctor.STORE_PROBE_KEY}, a key that can't exist", c.fix)
    assert "answers every request" in c.fix

    def bad():
        raise store_remote.ConfigError("x.toml: [remote] type s3 needs bucket")
    monkeypatch.setattr(store_remote, "load_remote", bad)
    assert doctor.check_store().detail == "the store's remote is misconfigured: x.toml: [remote] type s3 needs bucket"


def test_slack_is_checked_for_handle_holders_only_with_auth_test(monkeypatch):
    assert "slack" not in [n for n, _ in doctor.checks("lane", None)]
    assert "slack" in [n for n, _ in doctor.checks("coordinator", None)] and "slack" in [n for n, _ in doctor.checks("infra", None)]
    seen = []

    def post(url, data, headers, timeout=None):
        seen.append((url, timeout))
        return answer
    monkeypatch.setattr(slack, "post", post)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "dummy")
    answer = {"ok": True}
    assert doctor.check_slack() == Check("slack", OK, "auth.test is ok (direct)")
    assert seen == [(slack.DEFAULT_API + "auth.test", doctor.PROBE_TIMEOUT_S)]
    answer = {"ok": False, "error": "invalid_auth"}
    c = doctor.check_slack()
    assert (c.status, c.detail) == (FINDING, "auth.test: invalid_auth") and "using-slack" in c.fix


def test_slack_without_a_token_asks_the_broker_with_this_agent_s_identity(monkeypatch):
    c = doctor.check_slack()                                                      # conftest: no Cursor identity socket here
    assert c.status == FINDING and c.detail.startswith("the Cursor identity socket the Slack broker needs is unavailable")
    seen = []
    monkeypatch.setattr(slack, "oidc_token", lambda aud: seen.append(aud) or "dummy-identity")

    def post(url, data, headers, timeout=None):
        seen.append((url, json.loads(data)["method"], headers["Authorization"], timeout))
        if refuse:
            raise urllib.error.HTTPError(url, refuse, "x", {}, None)
        return {"ok": True}
    monkeypatch.setattr(slack, "post", post)
    refuse = None
    assert doctor.check_slack() == Check("slack", OK, "auth.test is ok (broker)")
    assert seen == [slack.BROKER, (slack.BROKER, "auth.test", "Bearer dummy-identity", doctor.PROBE_TIMEOUT_S)]
    refuse = 401
    c = doctor.check_slack()
    assert (c.status, c.detail) == (FINDING, "auth.test: the Slack broker rejected this agent's Cursor identity (HTTP 401); set "
                                             "SLACK_BOT_TOKEN or ask @infra")
    refuse = 503
    assert "the Slack broker is down or switched off (HTTP 503)" in doctor.check_slack().detail


# ---- ssh ------------------------------------------------------------------------------------------------------------------------

def test_ssh_probes_each_machine_with_an_ssh_host_as_research_run_reaches_it(monkeypatch, tmp_path):
    (tmp_path / "machines.toml").write_text('[machines.node-a]\nprovider = "ssh"\nhost = "192.0.2.10"\nuser = "research"\n'
                                            '[machines.pod-b]\nprovider = "runpod"\npod_id = "p"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(runpod, "SSH_KEY", tmp_path / "key")
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "known_hosts")
    (tmp_path / "key").write_text("dummy\n")
    assert list(doctor.ssh_machines()) == ["node-a"]
    assert [n for n, _ in doctor.checks("lane", None)] == ["secrets", "github", "git-push", "notes-push", "store", "ssh:node-a",
                                                           "venv", "disk", "clock"]
    m = doctor.ssh_machines()["node-a"]
    runs_ = Runs(monkeypatch, ssh=(0, ""))
    assert doctor.check_ssh("node-a", m) == Check("ssh:node-a", OK, "research@192.0.2.10:22 answers")
    argv = runs_.calls[0][0]
    assert argv[-2:] == ["research@192.0.2.10", "true"] and "BatchMode=yes" in argv and f"ConnectTimeout={doctor.SSH_CONNECT_S}" in argv
    assert str(tmp_path / "key") in argv
    runs_.answers["ssh"] = (255, "research@192.0.2.10: Permission denied (publickey).")
    c = doctor.check_ssh("node-a", m)
    assert c.detail == "research@192.0.2.10:22: refused the key" and "ask @infra to add its public key" in c.fix
    runs_.answers["ssh"] = (255, "ssh: connect to host 192.0.2.10 port 22: Connection timed out")
    assert doctor.check_ssh("node-a", m).detail == "research@192.0.2.10:22: unreachable"


def test_two_ssh_machines_on_a_fresh_state_dir_both_get_the_whole_key(monkeypatch, tmp_path):
    # the probes resolved the key from RUNPOD_SSH_KEY_B64 in their threads at once: 481 of 500 runs failed, some reading it empty
    key = b"dummy-ssh-key " * 40 + b"\n"
    monkeypatch.setenv("RUNPOD_SSH_KEY_B64", base64.b64encode(key).decode())
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "fresh-state"))
    monkeypatch.setattr(runpod, "SSH_KEY", tmp_path / "absent-key")
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "absent" / "known_hosts")
    (tmp_path / "machines.toml").write_text("".join(f'[machines.node-{c}]\nprovider = "ssh"\nhost = "192.0.2.{i}"\nuser = "research"\n'
                                                    for i, c in enumerate("ab", 10)))
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    read = []

    def ssh(argv, *, cwd=None, env=None):
        read.append(Path(argv[argv.index("-i") + 1]).read_bytes())
        return 0, ""
    monkeypatch.setattr(doctor, "run", ssh)
    probes = [p for p in doctor.checks("lane", None) if p[0].startswith("ssh:")]
    assert [c.status for c in doctor.gather(probes)] == [OK, OK] and read == [key, key]


def test_on_a_fresh_vm_the_doctor_reads_no_registry_rather_than_clone_the_notes(monkeypatch, tmp_path):
    # no machines.toml, no RESEARCH_MACHINES_D, no notes clone, RESEARCH_NOTES_TOKEN set: the registry's read cloned the notes repo
    # first, for up to 900 s before the deadline started, and wrote ~/.research/notes
    monkeypatch.delenv("RESEARCH_MACHINES_D")
    monkeypatch.setenv("RESEARCH_NOTES_TOKEN", "dummy")
    clones = []
    monkeypatch.setattr(notes_sync, "ensure_clone", lambda root: clones.append(root))
    (name, probe), = [p for p in doctor.checks("lane", None) if p[0].startswith("ssh")]
    assert clones == [] and not (tmp_path / "absent-notes").exists()
    assert probe() == Check("ssh", SKIP, "no machine in machines.toml (the registry is in the notes clone, which is not here) names "
                                         "an ssh host")
    notes = tmp_path / "notes"                                                    # a clone already here: its registry is read
    (notes / "machines.d").mkdir(parents=True)
    (notes / "machines.d" / "node-c.toml").write_text('provider = "ssh"\nhost = "192.0.2.12"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_NOTES", str(notes))
    assert list(doctor.ssh_machines()) == ["node-c"]


def test_a_machines_entry_of_the_wrong_shape_is_passed_over_or_a_finding_never_a_crash(monkeypatch, tmp_path):
    # `[machines]\nx = 1` crashed the command with AttributeError and exit 1, which reads as "a finding"
    (tmp_path / "machines.toml").write_text('[machines]\nx = 1\n[machines.node-a]\nprovider = "ssh"\nhost = "192.0.2.10"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    assert list(doctor.ssh_machines()) == ["node-a"]
    (tmp_path / "machines.toml").write_text("machines = 1\n")
    (name, probe), = [p for p in doctor.checks("lane", None) if p[0].startswith("ssh")]
    c = probe()
    assert (name, c.status, c.detail) == ("ssh", FINDING, "the machines config is unreadable: AttributeError")
    assert str(tmp_path / "machines.toml") in c.fix


def test_ssh_in_a_shell_that_lost_the_key_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(runpod, "SSH_KEY", tmp_path / "absent-key")
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", "RUNPOD_SSH_KEY_B64")
    c = doctor.check_ssh("node-a", {"provider": "ssh", "host": "192.0.2.10", "user": "research"})
    assert c.status == FINDING and secret_hint.FRESH_SHELL in c.detail
    (tmp_path / "machines.toml").write_text("[machines\n")
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    (name, probe), = [p for p in doctor.checks("lane", None) if p[0].startswith("ssh")]
    assert name == "ssh" and probe().detail == "the machines config is unreadable: TOMLDecodeError"


# ---- venv, disk, clock ----------------------------------------------------------------------------------------------------------

def test_the_venv_is_checked_against_the_lock_without_touching_it(monkeypatch, tmp_path):
    root, other = tmp_path / "wt", tmp_path / "shared" / ".venv"
    assert doctor.check_venv(None).status == SKIP
    root.mkdir()
    assert doctor.check_venv(root).status == SKIP                                 # no uv.lock
    (root / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(doctor, "active_venv", lambda: None)
    c = doctor.check_venv(root)
    assert c.detail == f"no venv at {root / '.venv'}" and c.fix == f"`cd {root} && uv sync --all-packages --extra torch-cpu`"
    for v in (root / ".venv", other):
        v.mkdir(parents=True)
        (v / "pyvenv.cfg").write_text("home = /usr\n")
    runs_ = Runs(monkeypatch, uv=(0, "Would make no changes"))
    assert doctor.check_venv(root).status == OK
    argv, cwd, env = runs_.calls[0]
    assert argv == [*doctor.UV_SYNC, "--locked", "--check", "--offline", "--no-install-workspace", "--inexact"]
    assert cwd == root and env["UV_PROJECT_ENVIRONMENT"] == str(root / ".venv")
    runs_.answers["uv"] = (1, "Would install 2 packages\n + pyyaml==6.0.3\n + zstd==1.0\nerror: The environment is outdated; run `uv sync`")
    c = doctor.check_venv(root)
    assert c.detail == f"{root / '.venv'} lacks what {root / 'uv.lock'} pins: pyyaml==6.0.3, zstd==1.0" and c.fix.startswith(f"`cd {root} && uv sync")
    monkeypatch.setattr(doctor, "active_venv", lambda: other)                     # a worktree run with another checkout's venv
    c = doctor.check_venv(root)
    assert runs_.calls[-1][2]["UV_PROJECT_ENVIRONMENT"] == str(other)
    assert c.fix.startswith(f"{other} is another checkout's") and f"would point its editables at {root}" in c.fix
    runs_.answers["uv"] = (2, "error: The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.")
    assert doctor.check_venv(root).fix == f"`uv lock` in {root}, and commit it"
    runs_.answers["uv"] = (127, "")
    assert doctor.check_venv(root).detail == "uv is not on PATH"


def test_disk_is_the_floor_research_run_refuses_under(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "free_bytes", lambda p: 500 * 10**9)
    c = doctor.check_disk(tmp_path)
    assert c.status == OK and c.detail.startswith("500.0 GB free at the least on ") and "over the floor" in c.detail
    monkeypatch.setattr(runs, "free_bytes", lambda p: 5 * 10**8)
    c = doctor.check_disk(tmp_path)
    assert c.status == FINDING and "0.50 GB free on the filesystem of" in c.detail and "1.5 GB floor" in c.detail
    assert c.fix.startswith("free space first: `research data evict --target-free-gb 3 --runs`")
    monkeypatch.setenv(runs.DISK_FLOOR_ENV, "0")
    assert doctor.check_disk(tmp_path).status == OK


def test_the_clock_is_compared_with_a_date_header(monkeypatch):
    now = iter([1000.0, 1000.2] * 3)
    monkeypatch.setattr(doctor, "clock", lambda: next(now))
    monkeypatch.setattr(doctor, "server_date", lambda: 999.6)
    assert doctor.check_clock() == Check("clock", OK, "within 30 s of cloudflare.com's (+0.0 s)")
    monkeypatch.setattr(doctor, "server_date", lambda: 900.0)
    c = doctor.check_clock()
    assert c.detail == "this clock is +100 s off cloudflare.com's" and "timedatectl" in c.fix

    def unreachable():
        raise urllib.error.URLError("no route")
    monkeypatch.setattr(doctor, "server_date", unreachable)
    assert doctor.check_clock().detail == "no Date from cloudflare.com to compare with: URLError"

    def unverified():
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate is not yet valid"))
    monkeypatch.setattr(doctor, "server_date", unverified)
    c = doctor.check_clock()
    assert c.detail == "no Date from cloudflare.com to compare with: its certificate did not verify" and "a clock far off" in c.fix


def test_server_date_reads_the_date_header(monkeypatch):
    class Answer:
        headers = {"Date": "Sat, 03 Oct 2026 19:32:28 GMT"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    seen = []
    monkeypatch.setattr(doctor.urllib.request, "urlopen", lambda req, timeout: seen.append((req.full_url, req.get_method())) or Answer())
    assert doctor.server_date() == 1791055948.0 and seen == [(doctor.CLOCK_URL, "HEAD")]


# ---- the runner and the command -------------------------------------------------------------------------------------------------

def test_a_probe_that_raises_or_hangs_is_a_finding_and_never_shows_its_message():
    def raises():
        raise RuntimeError("not-to-be-printed")
    released = threading.Event()
    assert doctor.gather([("a", raises)]) == [Check("a", FINDING, "the probe itself failed: RuntimeError", "tell @infra, with `research doctor --json`")]
    (c,) = doctor.gather([("b", lambda: released.wait() and Check("b", OK, "late"))], deadline_s=0)
    released.set()
    assert (c.status, c.detail) == (FINDING, "no answer within 0 s")


def test_what_a_probe_runs_is_killed_before_the_deadline_with_everything_it_started(monkeypatch, tmp_path):
    """A timeout killed only the child: git's remote helpers and ssh's lived on, and a probe of two steps could last the whole
    deadline (notes-push: 10 s of rev-parse, then 15 s of push)."""
    (c,) = doctor.gather([("a", lambda: Check("a", OK, str(doctor._left(math.inf))))], deadline_s=60)
    assert 0 <= float(c.detail) <= 60 - doctor.KILL_MARGIN_S
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    started = []

    class TimesOutOnceStarted(subprocess.Popen):
        def communicate(self, input=None, timeout=None):
            started.append(int(self.stdout.readline()))             # the grandchild is running: now the time is up
            raise subprocess.TimeoutExpired(self.args, timeout)
    monkeypatch.setattr(doctor.subprocess, "Popen", TimesOutOnceStarted)
    assert doctor.run(["sh", "-c", f"cat {fifo} & echo $!; wait"]) == (None, "")
    try:
        pidfd = os.pidfd_open(started[0])
    except ProcessLookupError:                                       # gone already
        return
    select.select([pidfd], [], [])                                   # the grandchild (blocked on the fifo for good) has exited
    os.close(pidfd)


def test_research_doctor_exits_0_ok_1_on_a_finding_2_on_usage(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "checkout", lambda: None)
    result = [Check("x", OK, "fine"), Check("y", SKIP, "n/a")]
    monkeypatch.setattr(doctor, "checks", lambda role, root: [(c.name, lambda c=c: c) for c in result])
    assert cli.main(["doctor"]) == 0
    assert capsys.readouterr().out == "research doctor (lane): ok\nok       x  fine\nskip     y  n/a\n"
    result.append(Check("z", FINDING, "broken", "mend it"))
    assert cli.main(["doctor", "--role", "coordinator"]) == 1
    assert capsys.readouterr().out.splitlines()[0] == "research doctor (coordinator): 1 finding"
    assert cli.main(["doctor", "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert (doc["findings"], doc["checks"][2]) == (1, {"name": "z", "status": FINDING, "detail": "broken", "fix": "mend it"})
    with pytest.raises(SystemExit) as e:
        cli.main(["doctor", "--role", "nobody"])
    assert e.value.code == 2
