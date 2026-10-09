"""The Nebius lock-out (pods/nebius/lockout): vy-seal-runs seals a run dir only when custody's own rule says it ended, never
follows a symlink, and frees a sealed run's disk only for retention and eviction; vy-sudo lets through only the forms it names,
as the caller; vy-lockout's allowlist is the committed sudoers file, every sudoers file parses, the fallback expires on its own
and only vy-admin restores it, and its steps say what they would do without changing anything. Pure Python over tmp trees: the
caller's uid stands in for root, and the clock, the runner and /proc are fakes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from research import retention as R
from research.pods import registry
from research.store import attempt as AT
from research.store import custody as CU
from research.store import evict as EV
from research.store import sealed as SL

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
LOCKOUT = NEBIUS / "lockout"
ME, MY_GID = os.getuid(), os.getgid()
CALLER, CALLER_GID = (ME, MY_GID) if ME else (1000, 1000)   # vy-sudo refuses root as its caller: a root runner (a pod) tests as 1000
T0 = 1_600_000_000.0                     # every file's mtime: quiet, and before VERIFIED_FROM
VERIFIED_FROM = 4102444800.0
ART = "art:" + "a" * 64
NOW = datetime(2026, 10, 1, 19, 30, tzinfo=timezone.utc)
not_root = pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


SR = load("vy_seal_runs", LOCKOUT / "seal_runs.py")
VS = load("vy_sudo", LOCKOUT / "vy_sudo.py")
LO = load("vy_lockout", LOCKOUT / "lockout.py")


def run(root: Path, name: str, *, state: str | None = "done", marker: bool = True, preserved: bool = True,
        files: dict[str, bytes] | None = None) -> Path:
    """A run dir with status.json (`state`), the `files`, and the `.custody` marker the runner writes, all quiet since T0."""
    d = root / name
    d.mkdir(parents=True)
    d.chmod(0o755)
    if state:
        (d / "status.json").write_text(json.dumps({"state": state}))
    for rel, body in {"stdout.log": b"y" * 100, **(files or {})}.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(body)
    if marker:
        listed = {r: [(d / r).stat().st_size, hashlib.sha256((d / r).read_bytes()).hexdigest()] for r in CU.run_files(d)}
        (d / CU.MARKER).write_text(json.dumps({"schema": CU.SCHEMA, "run_id": name, "run_record": ART, "preserved": preserved,
                                               "verified_from": VERIFIED_FROM, "verified_utc": "2026-09-21T00:00:00Z",
                                               "files": listed}))
    quiet(d)
    return d


def quiet(d: Path, t: float = T0) -> None:
    for base, dirs, fs in os.walk(d):
        for n in dirs + fs:
            os.utime(os.path.join(base, n), (t, t), follow_symlinks=False)
    os.utime(d, (t, t))


def writable(d: Path) -> None:
    """Undo a seal by hand (what vy-admin would do, and what tmp_path's cleanup needs)."""
    for base, dirs, fs in os.walk(d):
        os.chmod(base, 0o755)
        for f in fs:
            p = os.path.join(base, f)
            if not os.path.islink(p):
                os.chmod(p, 0o644)


@pytest.fixture
def roots(tmp_path):
    rs = [tmp_path / "research" / "runs", tmp_path / "jobs" / "runs"]
    for r in rs:
        r.mkdir(parents=True)
    yield rs
    for r in rs:
        writable(r)


# ---- vy-seal-runs -------------------------------------------------------------------------------------------------------------

def test_the_seal_rule_is_custodys_own(roots):
    assert (SR.TERMINAL, SR.MARKER, SR.SCHEMA, SR.EXCLUDED) == (AT.TERMINAL, CU.MARKER, CU.SCHEMA, CU.EXCLUDED)
    assert (SR.EVICTED_FILE, SR.EVICTED_SCHEMA, SR.TRIMMED_LISTING) == (EV.EVICTED_FILE, EV.EVICTED_SCHEMA, CU.TRIMMED_LISTING)
    r = roots[0]
    dirs = [run(r, "a"), run(r, "b", state="running"), run(r, "c", state=None), run(r, "d", marker=False),
            run(r, "e", preserved=False), run(r, "f", state="failed"), run(r, "g", state="cancelled")]
    (run(r, "h") / "late.txt").write_text("after custody")
    changed = run(r, "i")
    (changed / "stdout.log").write_bytes(b"z" * 100)
    os.utime(changed / "stdout.log", (VERIFIED_FROM + 1, VERIFIED_FROM + 1))
    fetched = run(r, "j", marker=False)
    (fetched / "preserved.json").write_text("{}")
    os.symlink("/etc", r / "a" / "link")
    for d in [*dirs, r / "h", changed, fetched]:
        assert SR.marker_problem(d) == CU.marker_problem(d), d.name
        state = json.loads((d / "status.json").read_text())["state"] if (d / "status.json").exists() else None
        assert (SR.not_ended(d) is None) == (state in AT.TERMINAL and CU.has_custody(d)[0]), d.name
    assert [d.name for d in sorted(r.iterdir()) if SR.not_ended(d) is None] == ["a", "f", "g", "j"]


def test_a_pass_seals_ended_quiet_runs_up_to_its_cap_and_a_dry_run_changes_nothing(roots, tmp_path):
    ended = [run(roots[0], n) for n in ("r1", "r2")] + [run(roots[1], "r3")]
    live = run(roots[0], "r0", state="running")
    fresh = run(roots[0], "r4")
    quiet(fresh, T0 + 3300)
    os.chmod(roots[0], 0o775)
    ledger = tmp_path / "state" / "sealed.jsonl"
    lines: list[str] = []
    kw = dict(quiet_s=600, now=T0 + 3600, uid=ME, gid=MY_GID, ledger=str(ledger), say=lines.append)

    rep = SR.seal_pass([str(r) for r in roots], dry_run=True, **kw)
    assert rep["sealed"] == [str(ended[0]), str(ended[1]), str(ended[2])] and rep["skipped"] == 2
    assert all(os.stat(d).st_mode & 0o200 for d in ended) and not ledger.exists(), "a dry run changes nothing"

    rep = SR.seal_pass([str(r) for r in roots], dry_run=False, max_dirs=2, **kw)
    assert rep["sealed"] == [str(ended[0]), str(ended[1])] and any("cap of 2" in ln for ln in lines)
    for d in ended[:2]:
        for base, dirs, fs in os.walk(d):
            assert stat.S_IMODE(os.stat(base).st_mode) == 0o555
            assert all(stat.S_IMODE(os.stat(os.path.join(base, f)).st_mode) == 0o444 for f in fs)
        assert SR.not_ended(d) is None, "sealing changes no mtime, so custody still vouches for it"
    assert os.stat(ended[2]).st_mode & 0o200
    assert [json.loads(ln)["path"] for ln in ledger.read_text().splitlines()] == [str(ended[0]), str(ended[1])]
    assert json.loads(ledger.read_text().splitlines()[0])["mode"] == "0o755"

    rep = SR.seal_pass([str(r) for r in roots], dry_run=False, **kw)
    assert rep["sealed"] == [str(ended[2])], "a sealed dir is not a candidate again"
    assert os.stat(live).st_mode & 0o200 and os.stat(fresh).st_mode & 0o200, "never a live run, nor one changed within the window"
    assert any("not root's and sticky" in ln for ln in lines), "a runs root its owner can rename in is named"


def test_sealing_follows_no_symlink(roots, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x")
    os.chmod(outside / "secret", 0o600)
    d = run(roots[0], "r1")
    os.symlink(outside / "secret", d / "to-file")
    os.symlink(outside, d / "to-dir")
    os.symlink(outside, roots[0] / "r2")
    quiet(d)
    rep = SR.seal_pass([str(roots[0])], dry_run=False, now=T0 + 3600, uid=ME, gid=MY_GID, ledger=None, say=lambda s: None)
    assert rep["sealed"] == [str(d)] and not rep["errors"]
    assert stat.S_IMODE(os.stat(outside / "secret").st_mode) == 0o600 and stat.S_IMODE(os.stat(outside).st_mode) & 0o200
    assert os.path.islink(d / "to-file") and stat.S_IMODE(os.stat(d).st_mode) == 0o555


def test_delete_and_evict_refuse_what_is_not_a_sealed_ended_run_directly_under_a_root(roots, capsys):
    rs = [str(r) for r in roots]
    open_run = run(roots[0], "r1")
    unended = run(roots[0], "r2", marker=False)
    nested = run(roots[0], "r3")
    for d in (unended, nested):
        SR.seal_dir(os.open(d.parent, os.O_RDONLY), d.name, ME, MY_GID)
    cases = {str(open_run): "not sealed", str(unended): "custody doesn't vouch", str(nested / "sub"): "not a run dir directly under",
             f"{roots[0]}/../runs/r3": "absolute, normalized", "r3": "absolute, normalized"}
    for path, why in cases.items():
        assert SR.delete(path, rs, uid=ME) == SR.REFUSED
        assert why in capsys.readouterr().out, path
        assert SR.evict(path, rs, 1, uid=ME, gid=MY_GID) == SR.REFUSED
        assert why in capsys.readouterr().out, path
    assert open_run.exists() and unended.exists()
    gone: list[Path] = []
    assert SR.delete(str(nested), rs, uid=ME, remove=gone.append) == 0 and gone == [nested]
    if os.geteuid() != 0:
        assert SR.main(["seal", "--dry-run"]) == SR.REFUSED


def test_evict_removes_only_the_big_files_custody_listed_unchanged_and_records_them(roots, monkeypatch, capsys):
    d = run(roots[0], "r1", files={"big.bin": b"b" * 5000, "grown.bin": b"g" * 4000, "small.json": b"{}"})
    (d / "grown.bin").write_bytes(b"g" * 4500)
    quiet(d)
    monkeypatch.setattr(SR, "not_ended", lambda p: None)
    monkeypatch.setattr(SR, "is_sealed", lambda st, uid=0: True)
    assert SR.evict(str(d), [str(roots[0])], 1000, uid=ME, gid=MY_GID) == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert [f["path"] for f in rep["files"]] == ["big.bin"] and rep["evicted_bytes"] == 5000 and rep["artifact"] == ART
    assert not (d / "big.bin").exists() and (d / "grown.bin").exists() and (d / "small.json").exists()
    rec = json.loads((d / SR.EVICTED_FILE).read_text())
    assert rec["schema"] == EV.EVICTED_SCHEMA and rec["restore"] == f"research data fetch {ART} --to {d}"
    assert stat.S_IMODE(os.stat(d / SR.EVICTED_FILE).st_mode) == 0o444


# ---- retention and eviction on a sealed run --------------------------------------------------------------------------------------

class Disk:
    def __init__(self, watched: list[Path]):
        self.watched = watched

    def __call__(self, _p):
        used = 90 + 5 * sum(p.exists() for p in self.watched)
        return 100, used, 100 - used


@pytest.fixture
def helper(monkeypatch, tmp_path):
    """SL's root helper as this user: vy-seal-runs' own checks, then the removal a non-root process can do once it unseals."""
    monkeypatch.setattr(SL, "SEALED_UID", ME)
    roots = [str(tmp_path / "research" / "runs"), str(tmp_path / "jobs" / "runs")]
    calls: list[list[str]] = []

    def call(*args: str) -> subprocess.CompletedProcess:
        calls.append(list(args))
        out: list[str] = []
        rc = SR.delete(args[1], roots, uid=ME, say=out.append, remove=lambda p: (writable(p), shutil.rmtree(p))) \
            if args[0] == "delete" else 2
        return subprocess.CompletedProcess(args, rc, "\n".join(out), "\n".join(out) if rc else "")

    monkeypatch.setattr(SL, "_call", call)
    return calls


def _sealed(roots: list[Path], name: str, **kw) -> Path:
    d = run(roots[0], name, files={"job.json": json.dumps({"ownership": {"lane": "compute-accounting"}}).encode()}, **kw)
    SR.seal_dir(os.open(d.parent, os.O_RDONLY), d.name, ME, MY_GID)
    return d


@not_root
def test_gc_deletes_a_sealed_run_through_the_helper_which_it_refused_as_read_only_before(roots, tmp_path, helper):
    d = _sealed(roots, "r1")
    kw = dict(now=lambda: NOW, usage=Disk([d]), remote_check=lambda a, p: None, in_use=lambda p: None, node=None, force=True,
              low_pct=0, apply=True)
    SL.SEALED_UID = ME + 1                      # not the sealer's: gc treats it as any read-only tree
    plan = R.gc(tmp_path, **kw)
    (act,) = [a for a in plan["actions"] if a["path"] == "research/runs/r1"]
    assert "read-only" in act["error"] and d.exists() and helper == []
    SL.SEALED_UID = ME
    plan = R.gc(tmp_path, **kw)
    (act,) = [a for a in plan["actions"] if a["path"] == "research/runs/r1"]
    assert act["action"] == "delete" and not act["error"] and not d.exists()
    assert helper == [["delete", str(d)]]


@not_root
def test_rm_deletes_a_sealed_run_whole_and_refuses_a_path_inside_one_or_what_the_helper_refuses(roots, tmp_path, helper):
    d, kept = _sealed(roots, "r1"), _sealed(roots, "r2")
    kw = dict(approved_by="@compute-accounting", ref="slack:1", root=tmp_path, in_use=lambda p: None, now=lambda: NOW, node="n",
              apply=True, by="https://agent", log=tmp_path / "deletions.jsonl")
    plan = R.rm([d / "stdout.log", d], **kw)
    inside, whole = plan["actions"]
    assert inside["action"] == "refuse" and "inside a sealed run dir" in inside["reason"]
    assert whole["action"] == "delete" and not whole["error"] and not d.exists()
    writable(kept)
    (kept / "late.txt").write_text("after custody")
    SR.seal_dir(os.open(kept.parent, os.O_RDONLY), kept.name, ME, MY_GID)
    (act,) = R.rm([kept], **kw)["actions"]
    assert act["action"] == "failed" and "custody doesn't vouch" in act["error"] and kept.exists(), "the helper checks custody again"


def test_evict_runs_hands_a_sealed_run_to_the_helper_and_skips_it_when_refused(tmp_path, monkeypatch):
    rd = tmp_path / "runs" / "r1"
    rd.mkdir(parents=True)
    (rd / "big.bin").write_bytes(b"b" * 10)
    plan = [{"run": "r1", "dir": str(rd), "artifact": ART, "files": [{"path": "big.bin", "bytes": 10, "sha256": "s", "artifact": ART}],
             "restore": "r"}]
    monkeypatch.setattr(EV, "plan_runs", lambda *a, **k: plan)
    monkeypatch.setattr(SL, "sealed", lambda p: True)
    got: list = []
    monkeypatch.setattr(SL, "evict", lambda p, n: got.append((p, n)) or {"evicted_bytes": 10, "files": plan[0]["files"]})
    rep = EV.evict_runs(None, tmp_path / "runs", target_free_bytes=100, min_bytes=5, free_fn=lambda: 0)
    assert got == [(rd, 5)] and rep["runs"] == [{"run": "r1", "artifact": ART, "files": 1, "bytes": 10}]
    assert (rd / "big.bin").exists() and not (rd / EV.EVICTED_FILE).exists(), "the helper, not this process, writes into it"

    def refuse(p, n):
        raise OSError("refused: not sealed")
    monkeypatch.setattr(SL, "evict", refuse)
    said: list[str] = []
    rep = EV.evict_runs(None, tmp_path / "runs", target_free_bytes=100, min_bytes=5, free_fn=lambda: 0, log=said.append)
    assert rep["runs"] == [] and rep["evicted_bytes"] == 0 and any("refused to evict" in s for s in said)


def test_sealed_needs_a_directory_of_the_sealers_without_write_bits(tmp_path, monkeypatch):
    monkeypatch.setattr(SL, "SEALED_UID", ME)
    d = tmp_path / "r1"
    d.mkdir()
    assert not SL.sealed(d)
    os.chmod(d, 0o555)
    assert SL.sealed(d) == (os.geteuid() != 0)
    monkeypatch.setattr(SL, "SEALED_UID", ME + 1)
    assert not SL.sealed(d) and not SL.sealed(tmp_path / "missing")
    os.chmod(d, 0o755)


# ---- vy-sudo --------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def proc(tmp_path):
    p = tmp_path / "proc"
    (p / "123").mkdir(parents=True)
    return p


def test_renice_and_taskset_touch_only_the_callers_processes(proc):
    assert VS.renice(["-n", "-5", "-p", "123"], ME, proc) == ["/usr/bin/renice", "-n", "-5", "-p", "123"]
    for args, uid in ((["-n", "-5", "-p", "123"], ME + 1), (["-n", "-21", "-p", "123"], ME), (["-n", "1", "-p", "999"], ME),
                      (["-n", "1", "-g", "123"], ME), (["-n", "x", "-p", "123"], ME)):
        with pytest.raises(VS.Refused):
            VS.renice(args, uid, proc)
    for args in (["-pc", "0-3,8", "123"], ["-p", "-c", "0-3,8", "123"], ["--pid", "--cpu-list", "0-3,8", "123"]):
        assert VS.taskset(args, ME, proc) == ["/usr/bin/taskset", "-p", "-c", "0-3,8", "123"]
    assert VS.taskset(["-apc", "4", "123"], ME, proc)[1] == "-a"
    for args, uid in ((["-pc", "0-3", "123"], ME + 1), (["-pc", "0;id", "123"], ME), (["-x", "-pc", "0", "123"], ME),
                      (["-c", "0", "123"], ME), (["-pc", "0", "123", "124"], ME)):
        with pytest.raises(VS.Refused):
            VS.taskset(args, uid, proc)


FILL = ["--quiet", "--scope", "--collect", "--slice=vy-timing.slice", "--unit", "fill-timing-1", "--nice=0", "-p", "AllowedCPUs=8-15",
        "-p", "MemoryMax=64G", "-p", "MemorySwapMax=0", "--", "setpriv", f"--reuid={CALLER}", f"--regid={CALLER_GID}", "--init-groups", "--",
        "/usr/local/bin/gpu-lease", "1", "--", "taskset", "-c", "8-15", "bash", "-c", "x"]


def test_systemd_run_runs_a_scope_as_the_caller_and_a_unit_only_as_a_job_user(tmp_path):
    out = VS.systemd_run(FILL, CALLER, CALLER_GID)
    assert out[0] == "/usr/bin/systemd-run" and f"--uid={CALLER}" in out and f"--gid={CALLER_GID}" in out and "--unit=fill-timing-1" in out
    assert out[out.index("--") + 1:] == ["/usr/local/bin/gpu-lease", "1", "--", "taskset", "-c", "8-15", "bash", "-c", "x"]
    env = tmp_path / "job.env"
    env.write_text("A=1\n")
    os.chown(env, CALLER, CALLER_GID)
    job = lambda n: (_ for _ in ()).throw(VS.Refused("privileged")) if n in ("root", "0") else type("pw", (), {"pw_gid": 1000})()
    unit = ["--unit=job-1", "--uid=ubuntu", "--wait", "--pipe", "--collect", "--quiet", "--service-type=exec", "--expand-environment=no",
            "--working-directory=/workspace", "-p", f"EnvironmentFile={env}", "-p", "CapabilityBoundingSet=", "-p", "NoNewPrivileges=yes",
            "--", "python3", "job.py"]
    assert VS.systemd_run(unit, CALLER, CALLER_GID, user=job)[-2:] == ["python3", "job.py"]
    bad = [
        ["--scope", "--uid=0", "--", "id"], ["--scope", "-p", "ExecStartPre=+/bin/sh", "--", "id"], ["--scope", "-p", "User=root", "--", "id"],
        ["--scope", "-p", "CapabilityBoundingSet=CAP_SYS_ADMIN", "--", "id"], ["--scope", "--slice=system.slice", "--", "id"],
        ["--scope", "--setenv=LD_PRELOAD=/x", "--", "id"], ["--scope", "--", ], ["--scope", "id"], ["--wait", "--", "id"],
        ["--uid=root", "--wait", "--", "id"], ["--uid=ubuntu", "-p", "EnvironmentFile=/etc/passwd", "--", "id"],
        ["--scope", "--nice=-30", "--", "id"], ["--scope", "--unit=../x", "--", "id"], ["--uid=ubuntu", "--scope", "--", "id"],
    ]
    for args in bad:
        with pytest.raises(VS.Refused):
            VS.systemd_run(args, CALLER, CALLER_GID, user=job)


def test_plan_takes_the_caller_from_sudo_and_passes_only_harmless_variables(proc):
    env = {"SUDO_UID": str(CALLER), "SUDO_GID": str(CALLER_GID)}
    argv, e = VS.plan(["env", "PATH=/home/research/.local/bin:/usr/bin", "HOME=/home/research", "systemd-run", *FILL], env)
    assert argv[0] == "/usr/bin/systemd-run" and e["PATH"].startswith("/home/research") and e["HOME"] == "/home/research"
    for args, environ in ((["env", "LD_PRELOAD=/x", "systemd-run", *FILL], env), (["env", "PYTHONPATH=/x", "systemd-run", *FILL], env),
                          (["env", "PATH=/x", "bash", "-c", "id"], env), (["renice", "-n", "1", "-p", "1"], {}),
                          (["renice", "-n", "1", "-p", "1"], {"SUDO_UID": "0", "SUDO_GID": "0"}), (["bash"], env),
                          (["du-bundles", "/etc"], env), ([], env)):
        with pytest.raises(VS.Refused):
            VS.plan(args, environ)
    assert VS.plan(["du-bundles"], env)[0] == ["/usr/bin/find", *VS.BUNDLE_ROOTS, "-xdev", "-name", "replay_bundle_p*", "-prune",
                                               "-exec", "/usr/bin/du", "-sb", "{}", "+"]
    with pytest.raises(VS.Refused):
        VS.plan(["hf-ns", "BASH_ENV=/tmp/x", "--", "/workspace/n2_build.sh", "a"], env)
    hf = VS.plan(["hf-ns", "--", "/workspace/n2_build.sh", "a"], env)[0]
    assert hf[:4] == ["/usr/bin/unshare", "--mount", "--propagation", "private"] and hf[-6:] == [*VS.HF_BIND, str(CALLER), str(CALLER_GID),
                                                                                               "/workspace/n2_build.sh", "a"]
    if os.geteuid() != 0:
        assert VS.main(["renice", "-n", "1", "-p", "1"]) == VS.REFUSED


def test_systemctl_stops_only_a_scope_whose_processes_are_all_the_callers(proc, tmp_path):
    cg = tmp_path / "cgroup"
    (cg / "x.slice" / "fill-timing-1.scope" / "sub").mkdir(parents=True)
    (cg / "x.slice" / "fill-timing-1.scope" / "cgroup.procs").write_text("123\n")
    (cg / "x.slice" / "fill-timing-1.scope" / "sub" / "cgroup.procs").write_text("777\n")   # exited: no /proc entry
    show = lambda u: "/x.slice/fill-timing-1.scope"
    assert VS.systemctl(["stop", "fill-timing-1.scope"], ME, show, cg, proc) == [VS.SYSTEMCTL, "stop", "fill-timing-1.scope"]
    assert VS.systemctl(["kill", "--signal=SIGTERM", "fill-timing-1.scope"], ME, show, cg, proc)[2] == "--signal=SIGTERM"
    for args, uid in ((["stop", "fill-timing-1.scope"], ME + 1), (["stop", "vy-custody.service"], ME), (["start", "x.scope"], ME),
                      (["stop", "--signal=KILL", "x.scope"], ME), (["kill", "--signal=SIGSYS", "x.scope"], ME),
                      (["stop", "a.scope", "b.scope"], ME)):
        with pytest.raises(VS.Refused):
            VS.systemctl(args, uid, show, cg, proc)


def test_job_user_is_never_root_or_a_privileged_account():
    for name in ("root", "0", "no-such-user-vy"):
        with pytest.raises(VS.Refused):
            VS.job_user(name)


# ---- vy-lockout: the allowlist --------------------------------------------------------------------------------------------------

def test_the_committed_allowlist_is_renders_and_every_rule_names_its_evidence():
    assert (LOCKOUT / "sudoers" / LO.ALLOWLIST).read_text() == LO.render(), "regenerate: python3 lockout/lockout.py render"
    for r in LO.ALLOW + LO.UBUNTU_ALLOW:
        assert r.cmd.startswith("/") and r.evidence.strip()
        assert Path(r.cmd).name not in ("k3s", "kubectl", "bash", "sh", "env", "python3", "xargs", "find", "tee", "rm", "cp", "mv"), r
    assert "k3s" not in LO.render() and "kubectl" not in LO.render()


def _parses(text: str) -> bool:
    """A sudoers file as this test reads it where visudo isn't installed: every logical line a Defaults, Cmnd_Alias or user spec."""
    import re
    logical = re.sub(r"\\\n", " ", text)
    ok = re.compile(r"^(Defaults(:[a-z_][a-z0-9_-]*)? \S.*|Cmnd_Alias [A-Z][A-Z0-9_]* = .+|[a-z_][a-z0-9_-]* ALL=\([^)]*\) NOPASSWD: \S.*)$")
    return all(ok.match(ln.strip()) for ln in logical.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))


@pytest.mark.parametrize("name", ["10-vy-admin", "50-vy-research", "fallback"])
def test_every_sudoers_file_parses(name, tmp_path):
    text = LO.fallback_text(VERIFIED_FROM) if name == "fallback" else (LOCKOUT / "sudoers" / name).read_text()
    assert _parses(text), name
    if shutil.which("visudo") or os.path.exists("/usr/sbin/visudo"):
        visudo = shutil.which("visudo") or "/usr/sbin/visudo"
        f = tmp_path / name
        f.write_text(text)
        r = subprocess.run([visudo, "-cf", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr or r.stdout


def test_allowed_is_sudos_match_over_the_allowlist():
    ok = [("research", "root", "/usr/local/sbin/vy-sudo env PATH=/x systemd-run --scope -- id"),
          ("research", "root", "/usr/local/sbin/vy-sudo du-bundles"), ("research", "root", "/usr/bin/true"),
          ("research", "root", "/usr/local/sbin/vy-seal-runs delete /workspace/research/runs/r1"),
          ("research", "ubuntu", "/usr/bin/bash -c anything"), ("research", "research", "/usr/bin/tmux new -d"),
          ("ubuntu", "root", "/usr/local/sbin/vy-seal-runs evict /workspace/jobs/runs/r1 --min-bytes 1")]
    no = [("research", "root", "/usr/local/bin/k3s kubectl get pods"), ("research", "root", "/usr/bin/true x"),
          ("research", "root", "/usr/local/sbin/vy-sudo du-bundles /etc"), ("research", "root", "/usr/bin/xargs -0r du -sb"),
          ("research", "root", "/usr/bin/python3 -c x"), ("research", "vy-admin", "/usr/bin/id"),
          ("ubuntu", "root", "/usr/local/sbin/vy-sudo renice -n 1 -p 1"), ("lane", "root", "/usr/bin/true")]
    assert all(LO.allowed(*c) for c in ok) and not any(LO.allowed(*c) for c in no)


def test_the_audit_counts_refused_forms_and_prints_no_argument_values():
    journal = "\n".join([
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/local/bin/k3s kubectl get pods -n default",
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/local/bin/k3s kubectl get pods -n kueue",
        "  research : PWD=/ ; USER=root ; COMMAND=/usr/bin/env TOKEN=s3cr3tvalue HOME=/root /usr/bin/x",
        "  research : PWD=/ ; USER=root ; COMMAND=/usr/bin/renice -n -5 -p 4242",
        "  research : PWD=/ ; USER=root ; COMMAND=/usr/bin/true",
        "  research : command not allowed ; PWD=/ ; USER=root ; COMMAND=/usr/bin/id",
        "  research : PWD=/ ; USER=ubuntu ; COMMAND=/usr/bin/env HOME=/x y",
        "pam_unix(sudo:session): session opened for user root",
    ])
    rows = LO.audit(journal)
    assert rows[0] == (2, "research->root /usr/local/bin/kNs kubectl get pods")
    assert (1, "research->root /usr/bin/renice -n -N -p") in rows and len(rows) == 3
    assert "s3cr3t" not in json.dumps(rows) and any("TOKEN=*" in k for _, k in rows)


# ---- vy-lockout: the fallback, the checks and the steps -------------------------------------------------------------------------

def runner(log: list[list[str]], fail: tuple[str, ...] = ()):
    """Runs rm, mv, cp and install for real, visudo for real when it is installed (else accepts), and fakes everything else."""
    visudo = shutil.which("visudo") or ("/usr/sbin/visudo" if os.path.exists("/usr/sbin/visudo") else None)

    def run(argv: list[str]) -> subprocess.CompletedProcess:
        log.append(argv)
        if argv[0] in fail:
            return subprocess.CompletedProcess(argv, 1, "", "no")
        if argv[0] == "visudo":
            if visudo and argv[1] == "-cf":
                return subprocess.run([visudo, *argv[1:]], capture_output=True, text=True)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] in ("rm", "mv", "cp") or argv[:2] == ["install", "-d"]:
            return subprocess.run(argv, capture_output=True, text=True)
        return subprocess.CompletedProcess(argv, 0, "", "")
    return run


def test_the_fallback_expires_by_its_own_line_and_only_vy_admin_puts_it_back(tmp_path):
    t = 1_791_000_000.0
    assert LO.expiry(LO.fallback_text(t)) == t and LO.expiry("research ALL=(ALL) NOPASSWD: ALL\n") is None
    log: list[list[str]] = []
    ctx = LO.Ctx(apply=True, run=runner(log), say=lambda s: None)
    path = tmp_path / LO.FALLBACK
    for env, hours in (({"SUDO_USER": "research"}, 24), ({}, 24), ({"SUDO_USER": "vy-admin"}, 25), ({"SUDO_USER": "vy-admin"}, 0)):
        with pytest.raises(LO.Failed):
            LO.fallback(ctx, "on", t, hours, tmp_path, env)
    assert not path.exists()
    assert LO.fallback(ctx, "on", t, 2, tmp_path, {"SUDO_USER": "vy-admin"}).startswith("fallback on until")
    assert stat.S_IMODE(path.stat().st_mode) == 0o440 and LO.expiry(path.read_text()) == t + 7200
    assert LO.fallback(ctx, "expire", t + 7199, sudoers_d=tmp_path).startswith("fallback holds") and path.exists()
    assert LO.fallback(ctx, "expire", t + 7200, sudoers_d=tmp_path) == "fallback removed (expired)" and not path.exists()
    path.write_text("research ALL=(ALL) NOPASSWD: ALL\n")
    assert "no readable" in LO.fallback(ctx, "expire", t, sudoers_d=tmp_path) and not path.exists()
    assert LO.fallback(ctx, "expire", t, sudoers_d=tmp_path) == "no fallback"


def _st(uid: int, mode: int) -> os.stat_result:
    return os.stat_result((mode, 1, 1, 1, uid, 0, 0, 0, 0, 0))


def test_untrusted_names_every_path_or_parent_someone_but_root_can_change():
    fs = {"/": _st(0, 0o40755), "/w": _st(0, 0o41777), "/w/ok": _st(0, 0o100755), "/w/mine": _st(1001, 0o100755),
          "/w/gw": _st(0, 0o100775), "/r": _st(1001, 0o40755), "/r/x": _st(0, 0o100755), "/t": _st(0, 0o41777)}

    def lstat(p):
        if p not in fs:
            raise FileNotFoundError(p)
        return fs[p]
    assert LO.untrusted(["/w/ok"], lstat) == []
    assert LO.untrusted(["/w/mine", "/w/gw", "/r/x", "/w/gone"], lstat) == [
        "/w/mine: owner uid 1001", "/w/gw: mode 775", "/r: owner uid 1001", "/w/gone: missing"]
    assert LO.untrusted(["/t"], lstat) == ["/t: mode 1777"], "a sticky dir is fine above a path, not as one a root unit writes in"


def test_kubeconfigs_and_the_tmux_command_a_switch_back_runs():
    assert LO.kubeconfig_admin("users:\n- name: a\n  user:\n    client-certificate-data: X\n    client-key-data: Y\n")
    assert not LO.kubeconfig_admin(LO.research_kubeconfig("tok", "CA"))
    text = (NEBIUS / "units" / "vy-n1-lease@.service").read_text()
    cmd = LO.tmux_command(text, "vy-n1-lease@provers.service")
    assert "vy-n1-lease@provers.env" in cmd and "%i" not in cmd and "$$" not in cmd and "cd /home/research &&" in cmd
    assert cmd.endswith(">> /workspace/verity-guest/n1-lease-provers.log 2>&1")
    assert (LO.unit_file("vy-n1-lease@provers.service"), LO.service_of("vy-custody.timer")) == ("vy-n1-lease@.service", "vy-custody.service")


def test_a_switch_stops_the_tmux_loop_before_its_unit_starts_and_a_dry_run_only_says_so(tmp_path, monkeypatch):
    said: list[str] = []
    LO.step_switch(LO.Ctx(apply=False, say=said.append), "vy-nebius-1", ["vy-gpu-stray.service"], back=False)
    assert [s.split()[0] for s in said] == ["would:", "would:", "would:", "vy-gpu-stray.service:"] and "C-c" in said[0]
    monkeypatch.setattr(LO, "UNITS", tmp_path)
    (tmp_path / "vy-gpu-stray.service").write_text("[Service]\nExecStart=/bin/true\n")
    log: list[list[str]] = []
    base = runner(log)

    def run(argv):
        if "list-panes" in argv:
            log.append(argv)
            return subprocess.CompletedProcess(argv, 0, "bash\n", "")
        return base(argv)
    waits: list[float] = []
    LO.step_switch(LO.Ctx(apply=True, run=run, say=said.append), "vy-nebius-1", ["vy-gpu-stray.service"], back=False, wait=waits.append)
    steps = [a[5] if a[0] == "runuser" else a[1] for a in log]
    assert steps == ["has-session", "send-keys", "list-panes", "kill-session", "enable", "is-active"] and waits == []
    with pytest.raises(LO.Failed):
        LO.step_switch(LO.Ctx(apply=False, say=said.append), "vy-nebius-1", ["vy-nope.service"], back=False)


def test_root_units_refuse_while_what_they_run_or_write_isnt_roots_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(LO, "untrusted", lambda paths: [f"{paths[0]}: owner uid 1001"])
    with pytest.raises(LO.Failed, match="not root's alone"):
        LO.step_root_units(LO.Ctx(apply=False, say=lambda s: None), "vy-nebius-1", back=False)
    monkeypatch.setattr(LO, "untrusted", lambda paths: [])
    said: list[str] = []
    LO.step_root_units(LO.Ctx(apply=False, say=said.append), "vy-nebius-1", back=False, units=tmp_path)
    drops = [s for s in said if LO.DROPIN in s]
    assert len(drops) == sum(lp.root for lp in LO.SWITCH["vy-nebius-1"]) and not list(tmp_path.iterdir())


def test_the_sudoers_step_needs_a_clean_preflight_installs_both_files_and_rolls_back(tmp_path):
    lib, sd, backup = tmp_path / "lib", tmp_path / "sudoers.d", tmp_path / "backup"
    lib.mkdir()
    sd.mkdir()
    (sd / LO.CLOUD_INIT).write_text("research ALL=(ALL) NOPASSWD:ALL\n")
    (lib / LO.ALLOWLIST).write_text(LO.render())
    log: list[list[str]] = []
    ctx = LO.Ctx(apply=True, run=runner(log), say=lambda s: None)
    t = 1_791_000_000.0
    bad = [LO.Check("vy-admin", False, "missing"), LO.Check("py-spy", True, "")]
    with pytest.raises(LO.Failed, match="preflight fails: vy-admin"):
        LO.step_sudoers(ctx, "vy-nebius-1", [], t, lib, sd, backup, checks=bad)
    assert not (sd / LO.ALLOWLIST).exists()
    (lib / LO.ALLOWLIST).write_text(LO.render() + "research ALL=(ALL) NOPASSWD: ALL\n")
    with pytest.raises(LO.Failed, match="isn't `vy-lockout render`'s"):
        LO.step_sudoers(ctx, "vy-nebius-1", ["vy-admin"], t, lib, sd, backup, checks=bad)
    (lib / LO.ALLOWLIST).write_text(LO.render())
    LO.step_sudoers(ctx, "vy-nebius-1", ["vy-admin"], t, lib, sd, backup, checks=bad)
    assert (sd / LO.ALLOWLIST).read_text() == LO.render() and LO.expiry((sd / LO.FALLBACK).read_text()) == t + 24 * 3600
    assert not (sd / LO.CLOUD_INIT).exists() and (backup / "sudoers" / LO.CLOUD_INIT).exists()
    assert ["systemctl", "enable", "--now", "vy-sudo-fallback-expire.timer"] in log
    LO.step_rollback(ctx, sd, backup)
    assert (sd / LO.CLOUD_INIT).exists() and not (sd / LO.ALLOWLIST).exists() and not (sd / LO.FALLBACK).exists()


def test_preflight_fails_each_check_on_an_unprepared_node(tmp_path):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "group").write_text("lxd:x:110:ubuntu\nsystemd-journal:x:999:\n")
    (tmp_path / "home" / "research" / ".kube").mkdir(parents=True)
    (tmp_path / "home" / "research" / ".kube" / "config").write_text("    client-key-data: X\n")
    for r in LO.RUNS_ROOTS:
        (tmp_path / r.lstrip("/")).mkdir(parents=True)

    def run(argv):
        if argv[0] == "getcap":
            return subprocess.CompletedProcess(argv, 0, f"{LO.PY_SPY} cap_sys_ptrace=ep\n", "")
        if "has-session" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 1, "", "")
    checks = {c.name: c for c in LO.preflight("vy-nebius-1", run, tmp_path)}
    failing = {n for n, c in checks.items() if not c.ok}
    assert {"vy-admin", "vy-admin-login", "ubuntu-unprivileged", "py-spy", "kubeconfig", "journal-group",
            "switched:vy-gpu-stray.service", "root-unit:vy-node1-dispatch.service", "tool:/usr/local/sbin/vy-sudo"} <= failing
    assert all(checks[n].fix for n in failing)


def test_the_plan_has_every_step_with_its_rollback():
    text = LO.plan_text("vy-nebius-1")
    assert [ln.split(".")[0] for ln in text.splitlines() if not ln.startswith(" ")] == [str(i) for i in range(1, 10)]
    assert text.count("   rollback: ") == 9 and "vy-node1-dispatch.service (tmux node1-dispatch)" in text


# ---- vy-admin-setup -------------------------------------------------------------------------------------------------------------

keygen = pytest.mark.skipif(not shutil.which("ssh-keygen"), reason="no ssh-keygen")


@keygen
def test_admin_setup_takes_only_public_keys_from_its_file_and_prints_fingerprints(tmp_path):
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "dan@laptop", "-f", str(tmp_path / "k")], check=True)
    pub = (tmp_path / "k.pub").read_text().strip()
    script = LOCKOUT / "admin_setup.sh"
    sh = lambda f, *a: subprocess.run(["bash", str(script), str(f), *a], capture_output=True, text=True)
    keys = tmp_path / "keys"
    keys.write_text(f"# Dan\n{pub}\n\n")
    r = sh(keys, "--check")
    assert r.returncode == 0 and "key: 256 SHA256:" in r.stdout and pub.split()[1] not in r.stdout + r.stderr
    for body, why in (("", "holds no public key"), ("# only a comment\n", "holds no public key"),
                      ((tmp_path / "k").read_text(), "private key"), (f'command="id" {pub}\n', "not one public key"),
                      ("ssh-ed25519 AAAAnotakey x\n", "ssh-keygen can't read it")):
        keys.write_text(body)
        r = sh(keys, "--check")
        assert r.returncode == 1 and why in r.stderr, (body[:20], r.stderr)
        assert "PRIVATE" not in r.stdout + r.stderr or why == "private key"
    assert sh(tmp_path / "missing", "--check").returncode == 1
    if os.geteuid() != 0:
        keys.write_text(pub + "\n")
        r = sh(keys)
        assert r.returncode == 3 and "run as root" in r.stderr
