"""`research doctor`: each check against fakes of what it probes (no network), and no value of any secret in what it prints."""
from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
from pathlib import Path

import pytest

from research import cli, doctor, runs, secret_hint, slack
from research.doctor import FINDING, OK, SKIP, Check
from research.pods import runpod
from research.store import remote as store_remote

ALL_NAMES = sorted({n for need in doctor.SECRETS.values() for alts in need for n in doctor._alts(alts)})
LANE = [n for alts in doctor.needs("lane") for n in alts]


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for n in ALL_NAMES:
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


def test_a_dry_run_push_to_a_reachable_origin_is_ok_and_sends_nothing(tmp_path):
    clone = _clone(tmp_path)
    assert doctor.check_git_push(clone) == Check("git-push", OK, f"origin of {clone} accepts a push (dry run)")
    assert subprocess.run(["git", "-C", str(tmp_path / "origin.git"), "for-each-ref"], capture_output=True, text=True).stdout == ""
    _git("remote", "remove", "origin", cwd=clone)
    assert doctor.check_git_push(clone).status == SKIP
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
    assert ("THE-CREDENTIAL-FIX" in c.fix) is credential and "shows git's message" in c.fix
    argv, _, env = runs_.calls[0]
    assert argv[-2:] == ["origin", f"HEAD:{doctor.PROBE_REF}"] and "--dry-run" in argv and env["GIT_TERMINAL_PROMPT"] == "0"
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
    assert argv[-2] == "https://x-access-token@github.com/danielreuter/research-notes.git" and "credential.helper=" in argv
    assert not any("not-to-be-printed" in a for a in argv)                         # the helper reads it from the environment
    runs_.answers["git"] = (0, "")
    assert doctor.check_notes_push().status == OK


# ---- store, slack ---------------------------------------------------------------------------------------------------------------

class _Remote:
    def __init__(self, error=None):
        self.error = error

    def head(self, key):
        if self.error:
            raise self.error
        assert key == doctor.STORE_PROBE_KEY
        return None


def test_the_store_check_reads_the_remote_s_answer_to_a_head(monkeypatch, tmp_path):
    c = doctor.check_store()                                                      # conftest: an absent store.toml
    assert c.status == FINDING and c.detail.startswith("no store remote") and "R2_ENDPOINT, R2_BUCKET" in c.fix
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "remote"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    assert doctor.check_store() == Check("store", OK, "the store's remote (FsRemote) answers")
    monkeypatch.setattr(store_remote, "load_remote", lambda: _Remote(urllib.error.HTTPError("u", 403, "Forbidden", {}, None)))
    c = doctor.check_store()
    assert c.detail == "the store's remote (_Remote) refused a HEAD: HTTP 403" and "AWS_SESSION_TOKEN" in c.fix
    monkeypatch.setattr(store_remote, "load_remote", lambda: _Remote(TimeoutError("not shown")))
    assert doctor.check_store().detail == "the store's remote (_Remote) gave no answer: TimeoutError"

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
