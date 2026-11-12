"""`research run --job-user USER`: the workload runs as USER in a transient system unit (no new privileges, no home, an empty /run,
a clean environment), the runner keeps its records as itself, and nothing reads the run dir until the unit is gone and what the
job left there is safe to read.  The unit makes no namespace but the types `--job-namespaces` lists, of user, mnt and net only,
and a `--store-read` key reaches it.  No test needs a real job user or systemd: the command runner is injected and each argv
asserted."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import cli, jobuser, remote
from research.store import custody as CU

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
RUN = "/workspace/research/runs/r1"
NAMESPACES = ("mnt", "net", "user")             # what `--job-namespaces user,mnt,net` becomes: sorted, as the unit's property lists them


@pytest.fixture(autouse=True)
def no_helper(monkeypatch, tmp_path):
    """Whether this machine has the lock-out's vy-sudo decides how a job asks for root: every test starts on one without it."""
    monkeypatch.setattr(jobuser, "VY_SUDO", str(tmp_path / "no-vy-sudo"))


def passwd(name="vyjob", uid=990, gid=990):
    return SimpleNamespace(pw_name=name, pw_uid=uid, pw_gid=gid, pw_dir="/nonexistent", pw_shell="/usr/sbin/nologin")


class Group:
    def __init__(self, name, gid, mem=()):
        self.gr_name, self.gr_gid, self.gr_mem = name, gid, list(mem)


class Runner:
    """subprocess.run as the unit's commands see it: every argv recorded, `find`'s stdout (or vy-sudo job-reclaim's) given."""

    def __init__(self, find_out=""):
        self.calls, self.find_out = [], find_out

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        out = self.find_out if argv[2:3] == ["find"] or argv[3:4] == ["job-reclaim"] else ""
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()


def test_the_unit_is_a_sandboxed_system_service_of_the_job_user():
    u = jobuser.Unit(name="research-job-r1", user="vyjob", run_dir=RUN, cwd=RUN, env_file=f"{RUN}/control/job.env", cpus="0-3,8",
                     memory=2**30, nice=5)
    props = [
        "NoNewPrivileges=yes", "RestrictSUIDSGID=yes", "CapabilityBoundingSet=", "AmbientCapabilities=",
        "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes", "PrivateIPC=yes", "ProtectProc=invisible",
        "ProtectKernelTunables=yes", "ProtectKernelModules=yes", "ProtectKernelLogs=yes", "ProtectControlGroups=yes",
        "ProtectClock=yes", "ProtectHostname=yes", "RestrictNamespaces=yes", "LockPersonality=yes", "RestrictRealtime=yes",
        "SystemCallArchitectures=native", "KeyringMode=private", "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "IPAddressDeny=link-local", "UMask=0022",
        "TemporaryFileSystem=/run:ro", "TemporaryFileSystem=/dev/shm:mode=1777,nosuid,nodev",
        "BindReadOnlyPaths=-/run/systemd/resolve/resolv.conf", "BindReadOnlyPaths=-/run/systemd/resolve/stub-resolv.conf",
        "BindPaths=-/run/gpu-lease", "BindPaths=-/run/gpu-lease-circuits",
        "InaccessiblePaths=-/etc/verity", "InaccessiblePaths=-/etc/rancher", "InaccessiblePaths=-/var/lib/rancher",
        "InaccessiblePaths=-/var/lib/kubelet",
        "TemporaryFileSystem=/workspace/research/runs:ro", "InaccessiblePaths=-/workspace/research/requests",
        f"BindPaths={RUN}", f"ReadWritePaths={RUN}", f"EnvironmentFile={RUN}/control/job.env",
        "KillMode=control-group", "SendSIGKILL=yes", "TimeoutStopSec=10",
        "AllowedCPUs=0-3,8", "MemoryMax=1073741824", "Nice=5",
    ]
    assert jobuser.unit_argv(u) == [
        "sudo", "-n", "systemd-run", "--unit=research-job-r1", "--uid=vyjob", "--wait", "--pipe", "--collect", "--quiet",
        "--service-type=exec", "--expand-environment=no", f"--working-directory={RUN}", *(x for p in props for x in ("-p", p)), "--"]
    bare = jobuser.unit_argv(jobuser.Unit(name="n", user="vyjob", run_dir=RUN, cwd=RUN, env_file="/e"))
    assert not [p for p in bare if p.startswith(("AllowedCPUs=", "MemoryMax=", "Nice="))], "only the limits the run was started with"


def test_of_the_resolvers_directory_the_unit_sees_the_two_resolv_conf_files_and_not_its_sockets():
    """A bind of the directory would bring systemd-resolved's own socket back into the emptied /run, and that socket looks names
    up for whoever connects, from a network namespace with no interface too."""
    props = jobuser.unit_argv(jobuser.Unit(name="n", user="vyjob", run_dir=RUN, cwd=RUN, env_file="/e"))
    back = [p.split("=", 1)[1].lstrip("-") for p in props if p.startswith(("BindReadOnlyPaths=", "BindPaths=")) and "=-/run" in p]
    assert back == ["/run/systemd/resolve/resolv.conf", "/run/systemd/resolve/stub-resolv.conf", *jobuser.LEASE_DIRS]
    assert "TemporaryFileSystem=/run:ro" in props and not [p for p in props if p.endswith("/run/systemd/resolve")]


def test_a_unit_may_make_exactly_the_namespace_types_its_run_listed_and_nothing_else_of_it_changes():
    u = jobuser.Unit(name="n", user="vyjob", run_dir=RUN, cwd=RUN, env_file="/e")
    default = jobuser.unit_argv(u)
    assert default.count("RestrictNamespaces=yes") == 1, "none unless the run lists them"
    for listed, prop in ((("user", "net", "mnt"), "RestrictNamespaces=mnt net user"), (("user",), "RestrictNamespaces=user")):
        assert jobuser.unit_argv(dataclasses.replace(u, namespaces=listed)) == [prop if p == "RestrictNamespaces=yes" else p for p in default]
    with pytest.raises(jobuser.Refused, match="'pid' is not a namespace type"):
        jobuser.unit_argv(dataclasses.replace(u, namespaces=("user", "pid")))


def test_a_run_may_list_user_mnt_and_net():
    assert jobuser.check_namespaces("user,mnt,net") == jobuser.check_namespaces(["net", "user", "mnt", "user"]) == NAMESPACES
    assert jobuser.check_namespaces("user") == ("user",)


@pytest.mark.parametrize("listed", ["pid", "user,pid", "ipc", "uts", "cgroup", "time", "", "user,", "user mnt", " user", "USER", "yes", "no",
                                    "~user", ["user", "pid"], [], [["user"]], True])
def test_any_other_namespace_type_or_spelling_is_refused(listed):
    """`yes`, `no` and a `~` list are RestrictNamespaces= values too (none, every type, all but those), so none may pass as a name."""
    with pytest.raises(jobuser.Refused, match="is not a namespace type a job's unit may be allowed"):
        jobuser.check_namespaces(listed)


def test_the_job_gets_the_runs_identity_its_gpus_and_what_its_env_names_and_no_other_credential():
    environ = {"RESEARCH_RUN_ID": "r1", "RESEARCH_RUN_DIR": RUN, "OUT": f"{RUN}/out", "RESEARCH_NOTES_TOKEN": "t", "PYTHONPATH": "/tool:/tree",
               "CUDA_VISIBLE_DEVICES": "2", "GPU_LEASE_ID": "7", "HF_TOKEN": "h", "AWS_SECRET_ACCESS_KEY": "a", "SSH_AUTH_SOCK": "/s",
               "KUBECONFIG": "/k", "PATH": "/home/research/.local/bin:/usr/bin", "HOME": "/home/research", "OMP_NUM_THREADS": "4",
               "XDG_RUNTIME_DIR": "/run/user/1001", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus", "TMUX": "/tmp/t,1,0"}
    assert jobuser.job_env(environ, ["HF_TOKEN", "OMP_NUM_THREADS"]) == {
        "PATH": jobuser.SYSTEM_PATH, "HOME": "/tmp", "LANG": "C.UTF-8", "RESEARCH_RUN_ID": "r1", "RESEARCH_RUN_DIR": RUN,
        "OUT": f"{RUN}/out", "PYTHONPATH": "/tool:/tree", "CUDA_VISIBLE_DEVICES": "2", "GPU_LEASE_ID": "7", "HF_TOKEN": "h", "OMP_NUM_THREADS": "4"}


def test_the_env_file_keeps_a_value_as_given():
    assert jobuser.env_file_text({"B": "plain", "A": 'q"b\\d$x`y'}) == 'A="q\\"b\\\\d\\$x\\`y"\nB="plain"\n'


@pytest.mark.parametrize("pw, groups, why", [
    (None, [], "no such user here"),
    (passwd(uid=0), [], "uid 0 is root"),
    (passwd(uid=1001), [], "uid 1001 is the runner"),
    (passwd(gid=1001), [Group("research", 1001)], "in group(s) research"),
    (passwd(), [Group("vyjob", 990), Group("sudo", 27, ["vyjob"])], "in group(s) sudo"),
    (passwd(), [Group("vyjob", 990), Group("docker", 998, ["vyjob"]), Group("research", 1001, ["vyjob"])], "docker, research"),
])
def test_admission_refuses_root_the_runner_and_a_privileged_or_the_runners_group(pw, groups, why):
    def getpwnam(name):
        if pw is None:
            raise KeyError(name)
        return pw

    with pytest.raises(jobuser.Refused, match=re.escape(why)):
        jobuser.check_user("vyjob", getpwnam=getpwnam, getgrall=lambda: groups, uid=1001, gid=1001)
    ok = jobuser.check_user("vyjob", getpwnam=lambda n: passwd(), getgrall=lambda: [Group("vyjob", 990), Group("video", 44, ["vyjob"])],
                            uid=1001, gid=1001)
    assert ok.pw_uid == 990


def test_a_run_dir_the_unit_cant_reach_or_name_is_refused():
    assert jobuser.check_run_dir("/srv/research/runs/r1", "r1") == "/srv/research/runs/r1"
    for d, rid, why in (("/home/research/.research/runs/r1", "r1", "under /home"), ("/tmp/runs/r1", "r1", "under /tmp"),
                        ("/srv/runs/a b", "a b", "can't carry"), ("/srv/runs/r%n", "r%n", "can't carry")):
        with pytest.raises(jobuser.Refused, match=why):
            jobuser.check_run_dir(d, rid)


class Proc:
    def __init__(self, rc, during=None):
        self.rc, self.during = rc, during

    def wait(self):
        if self.during:
            self.during()
        return self.rc


def run_dir(tmp_path) -> Path:
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    return d


def test_job_exec_runs_the_command_in_the_unit_from_an_env_file_only_the_runner_reads(tmp_path, monkeypatch):
    d = run_dir(tmp_path)
    monkeypatch.chdir(d)
    monkeypatch.setattr(jobuser, "allowed_cpus", lambda: "0-3")
    monkeypatch.setattr(jobuser, "memory_max", lambda: 2**30)
    environ = {"RESEARCH_RUN_DIR": str(d), "RESEARCH_RUN_ID": "r1", "FOO": "bar", "HF_TOKEN": "h", "HOME": "/home/research"}
    env_file = d / "control" / "job.env"
    seen, runs = [], Runner()

    def during():
        seen.append((stat.S_IMODE(env_file.stat().st_mode), env_file.read_text()))

    def popen(argv, **kw):
        seen.append((argv, kw))
        return Proc(7, during)

    rc = jobuser.main(["--user", "vyjob", "--unit", "research-job-r1", "--pass", "FOO", "--", "python", "-c", "x"], popen=popen,
                      run=runs, environ=environ, check=lambda u: passwd())
    unit = jobuser.Unit(name="research-job-r1", user="vyjob", run_dir=str(d), cwd=str(d), env_file=str(env_file), cpus="0-3",
                        memory=2**30, nice=os.nice(0))
    assert rc == 7 and seen[0] == ([*jobuser.unit_argv(unit), "python", "-c", "x"], {"start_new_session": True})
    assert seen[1] == (0o600, jobuser.env_file_text({"PATH": jobuser.SYSTEM_PATH, "HOME": "/tmp", "LANG": "C.UTF-8", "FOO": "bar",
                                                     "RESEARCH_RUN_DIR": str(d), "RESEARCH_RUN_ID": "r1"}))
    assert not env_file.exists() and runs.calls == [], "no stop: the unit ended by itself"


def test_a_signal_stops_the_unit_and_job_exec_exits_128_plus_it(tmp_path, monkeypatch):
    d = run_dir(tmp_path)
    monkeypatch.chdir(d)
    before = signal.getsignal(signal.SIGTERM)
    runs = Runner()
    rc = jobuser.main(["--user", "vyjob", "--unit", "research-job-r1", "--", "python", "-c", "x"], run=runs, check=lambda u: passwd(),
                      popen=lambda argv, **kw: Proc(1, lambda: os.kill(os.getpid(), signal.SIGTERM)),
                      environ={"RESEARCH_RUN_DIR": str(d)})
    assert rc == 128 + signal.SIGTERM
    assert runs.calls == [["sudo", "-n", "systemctl", "stop", "--no-block", "research-job-r1"],
                          ["sudo", "-n", "systemctl", "stop", "research-job-r1"]]
    assert signal.getsignal(signal.SIGTERM) is before and not (d / "control" / "job.env").exists()


def test_job_exec_refuses_before_anything_runs(tmp_path, monkeypatch):
    d = run_dir(tmp_path)
    monkeypatch.chdir(d)

    def refuse(u):
        raise jobuser.Refused("no")

    def popen(*a, **k):
        pytest.fail("nothing runs")

    assert jobuser.main(["--user", "vyjob", "--unit", "u", "--", "true"], popen=popen, check=refuse,
                        environ={"RESEARCH_RUN_DIR": str(d)}) == jobuser.EX_CONFIG
    assert jobuser.main(["--user", "vyjob", "--unit", "u x", "--", "true"], popen=popen, check=lambda u: passwd(),
                        environ={"RESEARCH_RUN_DIR": str(d)}) == jobuser.EX_CONFIG
    for listed in ("user,pid", "yes", ""):
        assert jobuser.main(["--user", "vyjob", "--unit", "u", "--namespaces", listed, "--", "true"], popen=popen, check=lambda u: passwd(),
                            environ={"RESEARCH_RUN_DIR": str(d)}) == jobuser.EX_CONFIG
    assert not (d / "control").exists()


def test_job_exec_gives_the_unit_the_namespaces_it_was_given(tmp_path, monkeypatch):
    d = run_dir(tmp_path)
    monkeypatch.chdir(d)
    seen = []
    assert jobuser.main(["--user", "vyjob", "--unit", "u", "--namespaces", "user,net,mnt", "--", "true"], run=Runner(),
                        popen=lambda argv, **kw: seen.append(argv) or Proc(0), check=lambda u: passwd(),
                        environ={"RESEARCH_RUN_DIR": str(d)}) == 0
    assert [p for p in seen[0] if p.startswith("RestrictNamespaces=")] == ["RestrictNamespaces=mnt net user"]


def test_the_runner_reads_nothing_until_the_unit_is_gone_and_what_the_job_left_is_a_file_or_a_directory(tmp_path):
    d = run_dir(tmp_path)
    os.chmod(d, 0o750)
    assert jobuser.prepare(str(d)) == 0o750
    assert stat.S_IMODE(d.stat().st_mode) == 0o1777 and (d / "control").is_dir()
    assert stat.S_IMODE((d / "out").stat().st_mode) == 0o1777, "the job writes its $OUT"
    (d / "control" / "job.env").write_text("X=1\n")
    runs = Runner(find_out=f"{d}/out/link\n{d}/fifo\n")
    removed, still = jobuser.finish(str(d), "research-job-r1", "vyjob", 2**31 - 2, 0o750, run=runs)
    assert runs.calls == [
        ["sudo", "-n", "systemctl", "stop", "research-job-r1"],
        ["sudo", "-n", "find", "-P", str(d), "-mindepth", "1", "-user", "vyjob", "!", "-type", "d", "!", "-type", "f", "-print", "-delete"],
        ["sudo", "-n", "chown", "-R", "-P", "--from=vyjob", f"{os.getuid()}:{os.getgid()}", "--", str(d)]]
    assert removed == ["out/link", "fifo"] and still == []
    assert stat.S_IMODE(d.stat().st_mode) == stat.S_IMODE((d / "out").stat().st_mode) == 0o750
    assert not (d / "control" / "job.env").exists()
    (d / "left").write_text("")
    assert sorted(jobuser.finish(str(d), "u", "vyjob", os.getuid(), 0o750, run=Runner())[1]) == ["control", "left", "out"], \
        "what the chown didn't give back is reported"


def test_where_the_nodes_vy_sudo_has_the_jobs_forms_every_root_command_goes_through_it(tmp_path, monkeypatch):
    """A locked-out node gives the runner no other root, so a job must already ask this way before the node's sudo is cut down."""
    vy_sudo = tmp_path / "vy-sudo"
    monkeypatch.setattr(jobuser, "VY_SUDO", str(vy_sudo))
    assert not jobuser.helper() and jobuser.root() == ("sudo", "-n"), "a node without the lock-out's files"
    vy_sudo.write_text('FORMS = ("renice", "systemd-run", "systemctl")\n')
    assert not jobuser.helper() and jobuser.root() == ("sudo", "-n"), "a helper from before the job's forms"
    vy_sudo.write_text('FORMS = ("renice", "systemd-run", "systemctl", "job-reclaim")\n')
    assert jobuser.helper() and jobuser.root() == ("sudo", "-n", str(vy_sudo))
    d = run_dir(tmp_path)
    u = jobuser.Unit(name="research-job-r1", user="vyjob", run_dir=str(d), cwd=str(d), env_file=f"{d}/control/job.env")
    plain = jobuser.unit_argv(u, via=False)
    assert jobuser.unit_argv(u) == ["sudo", "-n", str(vy_sudo), *plain[2:]], "the same unit, asked for through the helper"
    monkeypatch.chdir(d)
    runs = Runner()
    rc = jobuser.main(["--user", "vyjob", "--unit", "research-job-r1", "--", "python", "-c", "x"], run=runs, check=lambda u: passwd(),
                      popen=lambda argv, **kw: Proc(1, lambda: os.kill(os.getpid(), signal.SIGTERM)),
                      environ={"RESEARCH_RUN_DIR": str(d)})
    assert rc == 128 + signal.SIGTERM
    assert runs.calls == [["sudo", "-n", str(vy_sudo), "systemctl", "stop", "--no-block", "research-job-r1"],
                          ["sudo", "-n", str(vy_sudo), "systemctl", "stop", "research-job-r1"]]
    jobuser.prepare(str(d))
    runs = Runner(find_out=f"{d}/out/link\n{d}/fifo\n")
    removed, still = jobuser.finish(str(d), "research-job-r1", "vyjob", 2**31 - 2, 0o750, run=runs)
    assert runs.calls == [["sudo", "-n", str(vy_sudo), "systemctl", "stop", "research-job-r1"],
                          ["sudo", "-n", str(vy_sudo), "job-reclaim", str(d), "vyjob"]], "no find or chown of its own as root"
    assert removed == ["out/link", "fifo"] and still == []
    assert stat.S_IMODE(d.stat().st_mode) == stat.S_IMODE((d / "out").stat().st_mode) == 0o750


def test_through_vy_sudo_a_unit_keeps_its_namespaces_and_finish_removes_the_store_toml_with_the_env_file(tmp_path, monkeypatch):
    """Who is asked for root changes nothing a run asked for: the unit's argv differs only in its first words, and both of the
    runner's files in control/ are gone before job-reclaim, which gives back only what the job user owns."""
    vy_sudo = tmp_path / "vy-sudo"
    vy_sudo.write_text('FORMS = ("systemd-run", "systemctl", "job-reclaim")\n')
    monkeypatch.setattr(jobuser, "VY_SUDO", str(vy_sudo))
    d = run_dir(tmp_path)
    u = jobuser.Unit(name="research-job-r1", user="vyjob", run_dir=str(d), cwd=str(d), env_file=f"{d}/control/job.env",
                     namespaces=("user", "net"))
    argv = jobuser.unit_argv(u)
    assert argv == ["sudo", "-n", str(vy_sudo), *jobuser.unit_argv(u, via=False)[2:]]
    assert [p for p in argv if p.startswith("RestrictNamespaces=")] == ["RestrictNamespaces=net user"]
    staged = tmp_path / "store.toml"
    staged.write_text("[remote]\n")
    jobuser.prepare(str(d), store_config=staged)
    (d / "control" / "job.env").write_text("X=1\n")
    assert sorted(p.name for p in (d / "control").iterdir()) == ["job.env", "store.toml"]
    runs = Runner()
    jobuser.finish(str(d), "research-job-r1", "vyjob", 2**31 - 2, 0o750, run=runs)
    assert runs.calls == [["sudo", "-n", str(vy_sudo), "systemctl", "stop", "research-job-r1"],
                          ["sudo", "-n", str(vy_sudo), "job-reclaim", str(d), "vyjob"]]
    assert list((d / "control").iterdir()) == [], "neither the env file nor the store.toml copy outlives the unit"


def _request(tmp_path, job_user, **more):
    root = tmp_path / "root"
    (root / "requests" / "j1").mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": "j1", "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": None,
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler"], "command": ["true"], "exclusive": False, "job_user": job_user,
           "cost": {}, "remote": {"machine": "m"}, **more}
    p = root / "requests" / "j1" / "request.json"
    p.write_text(json.dumps(req))
    return p, root / "runs" / "j1"


def test_a_machine_refuses_a_job_user_it_cant_admit_and_nothing_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: pytest.fail("no runner starts"))
    req, d = _request(tmp_path, "no-such-user-zz9")
    out = remote.launch_request(req)
    assert out["started"] is False and "no such user here" in out["refused"]
    assert "no such user here" in json.loads((d / "refused.json").read_text())["reason"]


def test_a_machine_starts_the_runner_with_the_job_user_it_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(jobuser, "check_user", lambda name: passwd(name))
    monkeypatch.setattr(jobuser, "UNREACHABLE", ())
    started = []
    monkeypatch.setattr(remote.subprocess, "Popen", lambda argv, **kw: started.append(argv) or type("P", (), {"pid": 4242})())
    monkeypatch.setattr(remote, "await_first_record", lambda proc, fd: remote.FIRST_RECORD)     # the fake runner never acknowledges
    req, _ = _request(tmp_path, "vyjob")
    assert remote.launch_request(req)["started"]
    argv = started[0]
    assert argv[argv.index("--job-user"):argv.index("--job-user") + 4] == ["--job-user", "vyjob", "--stage", "t"]
    assert json.loads(argv[argv.index("--remote-json") + 1])["job_user"] == "vyjob"


def test_a_machine_starts_the_runner_with_the_namespaces_its_request_lists_and_refuses_a_request_listing_any_other(tmp_path, monkeypatch):
    monkeypatch.setattr(jobuser, "check_user", lambda name: passwd(name))
    monkeypatch.setattr(jobuser, "UNREACHABLE", ())
    started = []
    monkeypatch.setattr(remote.subprocess, "Popen", lambda argv, **kw: started.append(argv) or type("P", (), {"pid": 4242})())
    monkeypatch.setattr(remote, "await_first_record", lambda proc, fd: remote.FIRST_RECORD)     # the fake runner never acknowledges
    req, _ = _request(tmp_path / "a", "vyjob", job_namespaces=["user", "net", "mnt"])
    assert remote.launch_request(req)["started"]
    argv = started[0]
    assert argv[argv.index("--job-user"):argv.index("--job-user") + 5] == ["--job-user", "vyjob", "--job-namespaces", "mnt,net,user", "--stage"]
    assert json.loads(argv[argv.index("--remote-json") + 1])["job_namespaces"] == list(NAMESPACES)
    req, d = _request(tmp_path / "b", "vyjob", job_namespaces=["user", "pid"])
    out = remote.launch_request(req)
    assert out["started"] is False and "'pid' is not a namespace type" in out["refused"] and len(started) == 1
    assert "'pid' is not a namespace type" in json.loads((d / "refused.json").read_text())["reason"]


def test_a_launch_sends_the_machine_the_namespaces_with_the_job_user(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": "/r", "python": "python3"})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "user": "research", "pod_name": "n1",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "wake", lambda m, ssh: None)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: "/r/tool/x")
    staged = []

    def stage(ssh, root, run_id, request, sends, private=None):
        staged.append(request)
        raise remote.Refused("staged")

    monkeypatch.setattr(remote, "stage_request", stage)
    for run_id, listed in (("j1", NAMESPACES), ("j2", ())):
        with pytest.raises(remote.Refused, match="staged"):
            remote.launch(run_id=run_id, local_dir=tmp_path / "runs" / run_id, machine_name="n1", project="verity", command=["true"],
                          harness_args=["--stage", "t"], source=None, sends=[], cwd=None, env={}, exclusive=False, job_user="vyjob",
                          job_namespaces=listed)
    assert (staged[0]["job_user"], staged[0]["job_namespaces"]) == ("vyjob", list(NAMESPACES)) and "job_namespaces" not in staged[1]


def test_a_tree_that_predates_job_users_is_refused_before_anything_ships(tmp_path, monkeypatch):
    pkg = tmp_path / "tree" / "tools" / "research" / "src" / "research"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "remote.py").write_text("def tool_identity(): pass\n")
    assert remote.knows_job_user(None) and remote.knows_job_user(Path(SRC) / "research") and not remote.knows_job_user(pkg)
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": "/workspace/research", "python": "python3"})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "user": "research", "pod_name": "n1",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "ship_tool", lambda *a: pytest.fail("nothing ships"))
    with pytest.raises(remote.Refused, match="--job-user: the research package in .* predates it"):
        remote.launch(run_id="j1", local_dir=tmp_path / "runs" / "j1", machine_name="n1", project="verity", command=["true"],
                      harness_args=[], source=tmp_path / "tree", sends=[], cwd=None, env={}, exclusive=False, job_user="vyjob")
    (pkg / "jobuser.py").write_text("")             # a tree with job users whose units make no namespace
    (pkg / "remote.py").write_text('def tool_identity(): pass\nKEY = "job_user"\n')
    assert remote.knows_job_user(pkg) and not remote.knows_job_user(pkg, namespaces=True)
    assert remote.knows_job_user(None, namespaces=True) and remote.knows_job_user(Path(SRC) / "research", namespaces=True)
    with pytest.raises(remote.Refused, match="--job-namespaces: the research package in .* predates it"):
        remote.launch(run_id="j2", local_dir=tmp_path / "runs" / "j2", machine_name="n1", project="verity", command=["true"],
                      harness_args=[], source=tmp_path / "tree", sends=[], cwd=None, env={}, exclusive=False, job_user="vyjob",
                      job_namespaces=("user",))


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A local `research run --job-user vyjob`: its harness a stand-in that records what it was given to spawn, in what environment,
    how open the run dir was then and the store.toml the job would find there, and finish()'s root commands recorded."""
    monkeypatch.setattr(jobuser, "check_user", lambda name: passwd(name))
    monkeypatch.setattr(jobuser, "UNREACHABLE", ())
    monkeypatch.setattr(cli, "_survive_stray_signals", lambda: None)
    got, runs = {}, Runner()
    real_finish = jobuser.finish
    monkeypatch.setattr(jobuser, "finish", lambda *a, **k: got.setdefault("finish", real_finish(*a, run=runs, **k)))

    def harness(a, cmd):
        from research.telemetry.job import set_status

        cfg = Path(a.attempt_dir) / jobuser.STORE_CONFIG
        got.update(cmd=cmd, mode=stat.S_IMODE(Path(a.attempt_dir).stat().st_mode), env=dict(os.environ),
                   store_config=(stat.S_IMODE(cfg.stat().st_mode), cfg.read_text()) if cfg.exists() else None)
        set_status(Path(a.attempt_dir), "done", rc=0, signal=None)
        return 0

    monkeypatch.setattr("research.telemetry.run.run", harness)
    return tmp_path / "runs", got, runs


def test_a_local_run_spawns_job_exec_and_keeps_the_workloads_command_in_its_record(local):
    runs_root, got, runs = local
    assert cli.main(["run", "--runs-dir", str(runs_root), "--id", "r20260101-000000-0a11", "--job-user", "vyjob", "--env", "FOO=1", "--no-sampler",
                     "--", "echo", "hi"]) == 0
    d = runs_root / "r20260101-000000-0a11"
    assert got["cmd"] == jobuser.spawn_argv(["echo", "hi"], "vyjob", "research-job-r20260101-000000-0a11", ["FOO"]) and got["mode"] == 0o1777
    job = json.loads((d / "job.json").read_text())
    assert job["execution"]["command"] == ["echo", "hi"] and job["run"]["job_user"] == "vyjob"
    assert [c[2] for c in runs.calls] == ["systemctl", "find", "chown"] and got["finish"] == ([], [])
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(d.stat().st_mode) == 0o777 & ~umask, "the run dir's own mode back"


def test_a_queued_gpu_phase_runs_job_exec_inside_its_lease(local, tmp_path):
    runs_root, got, _ = local
    lease = ["gpu-lease", "1", "--wait", "--max-min", "30", "--"]
    req = tmp_path / "request.json"
    req.write_text(json.dumps({remote.QUEUE_WORKLOAD_WRAP: lease, "env": {"OMP_NUM_THREADS": "4"}}))
    (runs_root / "q1").mkdir(parents=True)
    assert cli.main(["run", "--runs-dir", str(runs_root), "--id", "q1", "--claimed", "--remote-json", json.dumps({"request": str(req)}),
                     "--job-user", "vyjob", "--no-sampler", "--", *lease, "nvidia-smi"]) == 0
    assert got["cmd"] == [*lease, *jobuser.spawn_argv(["nvidia-smi"], "vyjob", "research-job-q1", ["OMP_NUM_THREADS"])]


def test_a_run_lists_the_namespaces_its_unit_may_make_and_its_record_states_them(local):
    runs_root, got, _ = local
    base = ["run", "--runs-dir", str(runs_root), "--job-user", "vyjob", "--no-sampler"]
    assert cli.main([*base, "--id", "r20260101-000000-0c31", "--job-namespaces", "user,net,mnt", "--", "echo", "hi"]) == 0
    assert got["cmd"] == jobuser.spawn_argv(["echo", "hi"], "vyjob", "research-job-r20260101-000000-0c31", [], NAMESPACES)
    assert got["cmd"][got["cmd"].index("--namespaces") + 1] == "mnt,net,user"
    assert json.loads((runs_root / "r20260101-000000-0c31" / "job.json").read_text())["run"]["job_namespaces"] == list(NAMESPACES)
    assert cli.main([*base, "--id", "r20260101-000000-0c32", "--", "echo", "hi"]) == 0
    assert "--namespaces" not in got["cmd"], "none unless the run lists them"
    assert "job_namespaces" not in json.loads((runs_root / "r20260101-000000-0c32" / "job.json").read_text())["run"]


def test_namespaces_without_a_job_user_or_outside_user_mnt_and_net_are_refused_before_anything_runs(tmp_path, monkeypatch, capsys):
    runs_root = tmp_path / "runs"
    for name in ("launch", "queue_placement"):
        monkeypatch.setattr(remote, name, lambda *a, **k: pytest.fail("nothing is placed or launched"))
    for args, why in ((["--job-namespaces", "user"], "--job-namespaces lists what a job user's unit may make (--job-user USER)"),
                      (["--job-user", "vyjob", "--job-namespaces", "user,pid"], "'pid' is not a namespace type"),
                      (["--on", "m", "--project", "p", "--job-user", "vyjob", "--job-namespaces", "yes"], "'yes' is not a namespace type"),
                      (["--queue", "--source", str(tmp_path), "--project", "p", "--job-user", "vyjob", "--job-namespaces", "uts"],
                       "'uts' is not a namespace type")):
        assert cli.main(["run", "--runs-dir", str(runs_root), *args, "--", "true"]) == 2
        assert why in capsys.readouterr().err
    assert not runs_root.exists()


def test_a_run_on_a_machine_hands_the_launch_its_namespaces(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(remote, "launch", lambda **kw: seen.append((kw["job_user"], kw["job_namespaces"])) or
                        {"phase": "launched", "pod_id": "p1", "remote_run_dir": "/r", "launch": {"runner_pid": 1}})
    base = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--job-user", "vyjob"]
    assert cli.main([*base, "--job-namespaces", "user,net,mnt", "--", "true"]) == 0 and cli.main([*base, "--", "true"]) == 0
    assert seen == [("vyjob", NAMESPACES), ("vyjob", ())]


def test_a_store_read_key_reaches_the_unit_and_without_one_no_aws_name_passes(local, tmp_path, monkeypatch):
    """The machine's runner holds the key as launch_request left it: the three variables in its environment and a store.toml among
    the launch requests, which the unit doesn't see."""
    runs_root, got, _ = local
    staged = tmp_path / "root" / "requests" / "q1"
    read = staged / remote.STORE_READ_DIR
    read.mkdir(parents=True)
    toml = f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n'
    (read / CU.STORE_TOML).write_text(toml)
    (read / CU.STORE_TOML).chmod(0o600)
    (staged / "request.json").write_text(json.dumps({"env": {}}))
    key = dict.fromkeys(CU.KEY_ENV, "dummy")
    for k, v in {**key, "RESEARCH_STORE_CONFIG": str(read / CU.STORE_TOML)}.items():
        monkeypatch.setenv(k, v)
    base = ["run", "--runs-dir", str(runs_root), "--claimed", "--remote-json", json.dumps({"request": str(staged / "request.json")}),
            "--job-user", "vyjob", "--no-sampler"]
    for run_id in ("q1", "q2"):
        (runs_root / run_id).mkdir(parents=True)
    (read / CU.CRED_FILE).write_text("{}")
    assert cli.main([*base, "--id", "q1", "--store-read-dir", str(read), "--", "true"]) == 0
    passed = [got["cmd"][i + 1] for i, x in enumerate(got["cmd"]) if x == "--pass"]
    assert passed == sorted(CU.KEY_ENV)
    assert got["store_config"] == (0o644, toml), "a copy in the runner's control/, which the job reads and can't write"
    env = jobuser.job_env(got["env"], passed)
    assert {k: env[k] for k in key} == key and env["RESEARCH_STORE_CONFIG"] == str(runs_root / "q1" / "control" / "store.toml")
    assert not (runs_root / "q1" / "control" / "store.toml").exists() and not (read / CU.CRED_FILE).exists(), "gone with the workload"
    assert cli.main([*base, "--id", "q2", "--", "true"]) == 0
    assert "--pass" not in got["cmd"] and got["store_config"] is None
    assert got["env"]["RESEARCH_STORE_CONFIG"] == str(read / CU.STORE_TOML), "the runner's own environment came back after the first run"
    assert not [k for k in jobuser.job_env(got["env"], []) if k.startswith("AWS_")], "what the runner holds stays the runner's"


def test_a_run_dir_the_job_user_still_owns_part_of_is_not_published(local, monkeypatch, capsys):
    runs_root, _, _ = local
    monkeypatch.setattr(jobuser, "finish", lambda *a, **k: ([], ["out/x"]))
    published = []
    monkeypatch.setattr(cli, "_publish_attempt", lambda *a, **k: published.append(a))
    assert cli.main(["run", "--runs-dir", str(runs_root), "--id", "r20260101-000000-0b21", "--job-user", "vyjob", "--no-sampler", "--", "true"]) == 3
    assert published == [] and "refused to publish: 1 entry" in capsys.readouterr().err


def test_a_job_user_this_machine_cant_admit_is_refused_before_the_run_dir_exists(tmp_path, capsys):
    runs_root = tmp_path / "runs"
    assert cli.main(["run", "--runs-dir", str(runs_root), "--job-user", "no-such-user-zz9", "--", "true"]) == 3
    assert "no such user here" in capsys.readouterr().err and not (runs_root / "r1").exists() and not list(runs_root.glob("r*"))
