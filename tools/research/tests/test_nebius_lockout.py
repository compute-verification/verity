"""The Nebius lock-out (pods/nebius/lockout): vy-seal-runs seals a run dir only when custody's own rule says it ended, never
follows a symlink, and frees a sealed run's disk only for retention and eviction; vy-sudo lets through only the forms it names,
as the caller, and takes back for the caller only what a job user left in the caller's own run dir; vy-lockout's allowlists (the
cluster nodes' and a CPU node's) are the committed sudoers files, every sudoers file parses, a CPU node's lets a run keep its forms
and refuses root by any other name, the sudoers step is refused while the admin login doesn't work, the fallback expires on its
own and only vy-admin restores it, and its steps say what they would do without changing anything. Pure Python over tmp trees:
the caller's uid stands in for root, and the clock, the runner, the login session and /proc are fakes."""
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
from types import SimpleNamespace

import pytest
from research import jobuser
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
    kw = dict(now=lambda: NOW, usage=Disk([d]), remote_check=lambda a, p: R.Stored("verified", ""), in_use=lambda p: None, node=None, force=True,
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


def _pw(**fields):
    """vy-sudo's lookup of a job user, giving a passwd entry of only `fields`."""
    return lambda name: type("pw", (), fields)()


def _unit(**props):
    """vy-sudo's reading of a unit, giving `props`."""
    return lambda unit: props


def test_systemctl_stops_a_jobs_unit_and_never_one_that_runs_as_root():
    job = _unit(LoadState="loaded", User="vyjob")
    for args in (["stop", "research-job-r1"], ["stop", "research-job-r1.service"]):
        assert VS.systemctl(args, ME, props=job) == [VS.SYSTEMCTL, "stop", "research-job-r1.service"]
    assert VS.systemctl(["stop", "--no-block", "research-job-r1"], ME, props=job) == [VS.SYSTEMCTL, "stop", "--no-block",
                                                                                     "research-job-r1.service"]
    gone = _unit(LoadState="not-found", User="")
    assert VS.systemctl(["stop", "research-job-r1"], ME, props=gone)[-1] == "research-job-r1.service", "a stop of a unit that is gone"
    for args, props in ((["stop", "research-job-r1"], _unit(LoadState="loaded", User="root")),
                        (["stop", "research-job-r1"], _unit(LoadState="loaded", User="0")),
                        (["stop", "research-job-r1"], _unit(LoadState="loaded", User="")),
                        (["stop", "research-job-r1"], _unit(LoadState="loaded")),
                        (["stop", "research-job-r1", "ssh.service"], job), (["stop", "--now", "research-job-r1"], job),
                        (["stop", "--no-block", "--no-block", "research-job-r1"], job), (["stop", "research-job-a/b"], job),
                        (["stop", "research-job-"], job), (["kill", "research-job-r1"], job), (["start", "research-job-r1"], job),
                        (["stop", "ssh.service"], job), (["stop", "--no-block", "vy-seal-runs.service"], job)):
        with pytest.raises(VS.Refused):
            VS.systemctl(args, ME, props=props)


def _left_by_a_job(tmp_path: Path) -> tuple[Path, Path]:
    """A run dir as a job leaves it, and a directory outside it that links in the run dir point at."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x")
    d = tmp_path / "runs" / "r1"
    (d / "out" / "sub").mkdir(parents=True)
    (d / "out" / "result.json").write_text("{}")
    (d / "out" / "sub" / "deep.bin").write_bytes(b"b")
    os.symlink(outside / "secret", d / "out" / "to-file")
    os.symlink(outside, d / "to-dir")
    os.mkfifo(d / "out" / "sub" / "fifo")
    return d, outside


def test_job_reclaim_gives_the_caller_what_the_job_user_left_and_removes_what_is_neither_a_file_nor_a_directory(tmp_path, monkeypatch):
    d, outside = _left_by_a_job(tmp_path)
    chowned: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "fchown", lambda fd, u, g: chowned.append((os.readlink(f"/proc/self/fd/{fd}"), u, g)))
    removed: list[str] = []
    the_job = _pw(pw_uid=ME)                                        # this user's files stand in for the job user's
    assert VS.reclaim([str(d), "vyjob"], ME, MY_GID, user=the_job, say=removed.append) == 0
    assert sorted(removed) == [f"{d}/out/sub/fifo", f"{d}/out/to-file", f"{d}/to-dir"]
    assert sorted(chowned) == [(f"{d}/out", ME, MY_GID), (f"{d}/out/result.json", ME, MY_GID), (f"{d}/out/sub", ME, MY_GID),
                               (f"{d}/out/sub/deep.bin", ME, MY_GID)], "each by the descriptor it checked, the run dir itself left"
    assert (outside / "secret").read_text() == "x" and sorted(n.name for n in outside.iterdir()) == ["secret"], "no link was followed"
    assert not [p for p in d.rglob("*") if p.is_symlink() or p.is_fifo()]


def test_job_reclaim_leaves_what_is_not_the_job_users_and_stays_out_of_a_directory_that_is_neither_ones(tmp_path, monkeypatch):
    d, _outside = _left_by_a_job(tmp_path)
    chowned: list[str] = []
    monkeypatch.setattr(os, "fchown", lambda fd, u, g: chowned.append(os.readlink(f"/proc/self/fd/{fd}")))
    removed: list[str] = []
    other = _pw(pw_uid=ME + 1)                                      # a job user that owns nothing here
    assert VS.reclaim([str(d), "vyjob"], ME, MY_GID, user=other, say=removed.append) == 0
    assert removed == [] and chowned == [] and (d / "to-dir").is_symlink() and (d / "out" / "sub" / "fifo").is_fifo()

    real = os.fstat

    def fstat(fd):                                                  # out/sub as a third user's directory
        st = real(fd)
        if os.readlink(f"/proc/self/fd/{fd}") != f"{d}/out/sub":
            return st
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, ME + 2, st.st_gid, st.st_size, 0, 0, 0))
    monkeypatch.setattr(os, "fstat", fstat)
    the_job = _pw(pw_uid=ME)
    assert VS.reclaim([str(d), "vyjob"], ME, MY_GID, user=the_job, say=removed.append) == 0
    assert f"{d}/out/sub" not in chowned and f"{d}/out/sub/deep.bin" not in chowned and (d / "out" / "sub" / "fifo").is_fifo()
    assert sorted(removed) == [f"{d}/out/to-file", f"{d}/to-dir"]


def test_job_reclaim_refuses_a_run_dir_that_is_not_the_callers_own_directory(tmp_path):
    d, outside = _left_by_a_job(tmp_path)
    os.symlink(d, tmp_path / "link")
    the_job = _pw(pw_uid=ME)

    def no_user(name):
        raise VS.Refused("not a job user")
    for args, uid, user in (([str(d), "vyjob"], ME + 1, the_job), ([str(tmp_path / "link"), "vyjob"], ME, the_job),
                            ([f"{d}/../r1", "vyjob"], ME, the_job), (["runs/r1", "vyjob"], ME, the_job),
                            ([str(d / "out" / "result.json"), "vyjob"], ME, the_job), ([str(d / "missing"), "vyjob"], ME, the_job),
                            ([str(d)], ME, the_job), ([str(d), "vyjob", "x"], ME, the_job), ([str(d), "root"], ME, no_user)):
        with pytest.raises(VS.Refused):
            VS.reclaim(args, uid, MY_GID, user=user, say=lambda s: pytest.fail("nothing is removed"))
    assert (d / "to-dir").is_symlink() and (outside / "secret").exists()
    with pytest.raises(VS.Refused):
        VS.plan(["job-reclaim", str(d), "vyjob"], {"SUDO_UID": str(CALLER), "SUDO_GID": str(CALLER_GID)})


def test_every_root_command_of_a_job_is_one_vy_sudo_takes_and_both_allowlists_name(tmp_path):
    """jobuser.py builds a job's root commands and vy-sudo parses them: a property or an option one adds and the other doesn't
    pass would fail every --job-user run on a locked node."""
    d = tmp_path / "runs" / "r1"
    (d / "control").mkdir(parents=True)
    env = d / "control" / "job.env"
    env.write_text("A=1\n")
    os.chown(env, CALLER, CALLER_GID)
    unit = jobuser.Unit(name="research-job-r1", user="vyjob", run_dir=str(d), cwd=str(d), env_file=str(env), cpus="0-3,8",
                        memory=2**30, nice=5)
    argv = [*jobuser.unit_argv(unit, via=True), "python3", "job.py"]
    assert argv[:4] == ["sudo", "-n", jobuser.VY_SUDO, "systemd-run"]
    out = VS.systemd_run(argv[4:], CALLER, CALLER_GID, user=_pw(pw_gid=990))
    assert out[0] == "/usr/bin/systemd-run" and "--uid=vyjob" in out and out[-2:] == ["python3", "job.py"]
    assert [a for a in out if a.startswith("--unit=")] == ["--unit=research-job-r1"] and out.count("-p") == argv.count("-p")
    calls: list[list[str]] = []

    def run(a, **kw):
        calls.append(list(a))
        return subprocess.CompletedProcess(a, 0, "", "")
    os.chmod(d, 0o1777)
    (d / "out").mkdir()
    jobuser.finish(str(d), unit.name, "vyjob", 2**31 - 2, 0o755, run=run, via=True)
    stop, reclaim = calls
    assert stop == ["sudo", "-n", jobuser.VY_SUDO, "systemctl", "stop", "research-job-r1"]
    assert VS.systemctl(stop[4:], CALLER, props=_unit(LoadState="loaded", User="vyjob"))[-1] == "research-job-r1.service"
    assert reclaim == ["sudo", "-n", jobuser.VY_SUDO, "job-reclaim", str(d), "vyjob"]
    for p in LO.PROFILES:
        for a in (argv, stop, [*stop[:5], "--no-block", stop[5]], reclaim):
            assert LO.allowed("research", "root", " ".join(a[2:]), p), (p.file, a[3])
    assert jobuser.helper(str(LOCKOUT / "vy_sudo.py")), "the helper this tree installs has the job's forms"


@pytest.mark.parametrize("listed", [("user",), ("mnt",), ("net",), ("user", "mnt"), ("user", "net"), ("mnt", "net"), ("user", "mnt", "net")])
def test_vy_sudo_and_both_allowlists_take_a_unit_that_may_make_the_namespaces_its_run_listed(listed, tmp_path):
    """`research run --job-namespaces LIST` reaches the unit as RestrictNamespaces=<the list>, one argument with spaces in it.
    vy-sudo passes a property it lists with the value it was given, so which lists a run may give stays research's rule
    (jobuser.check_namespaces); were the value held to `yes` here, such a run would fail at its unit's start on a locked node."""
    d = tmp_path / "runs" / "r1"
    (d / "control").mkdir(parents=True)
    env = d / "control" / "job.env"
    env.write_text("A=1\n")
    os.chown(env, CALLER, CALLER_GID)
    unit = jobuser.Unit(name="research-job-r1", user="vyjob", run_dir=str(d), cwd=str(d), env_file=str(env), namespaces=listed)
    argv = [*jobuser.unit_argv(unit, via=True), "true"]
    prop = "RestrictNamespaces=" + " ".join(sorted(listed))
    assert argv.count(prop) == 1
    out = VS.systemd_run(argv[4:], CALLER, CALLER_GID, user=_pw(pw_gid=990))
    assert [a for a in out if a.startswith("RestrictNamespaces=")] == [prop] and out[out.index(prop) - 1] == "-p"
    assert out.count("-p") == argv.count("-p"), "every property of the unit reaches systemd-run"
    for p in LO.PROFILES:
        assert LO.allowed("research", "root", " ".join(argv[2:]), p), p.file


def _unit_ns(*props: str) -> list[str]:
    """vy-sudo's arguments for a job's unit carrying each `RestrictNamespaces=<value>` of `props`, each as its own `-p`."""
    head = ["--unit=research-job-r1", "--uid=vyjob", "--wait", "--pipe"]
    return [*head, *(x for v in props for x in ("-p", f"RestrictNamespaces={v}")), "--", "true"]


def test_vy_sudo_refuses_a_restrict_namespaces_value_that_would_widen_the_unit():
    """RestrictNamespaces= only narrows at `yes` (no namespace) or a space-separated list out of NAMESPACES (only those); `no`, a
    `~` list, a type outside the set, or anything else widens the unit, so vy-sudo refuses it. The set is the launcher's
    (jobuser.check_namespaces), held equal here because vy-sudo is a lone `python3 -I` script on the node and can't import it."""
    assert set(VS.NAMESPACES) == set(jobuser.NAMESPACES)
    job = _pw(pw_gid=990)
    allowed = ("yes", "user", "mnt", "net", "user mnt", "net user", "user mnt net", "user user", "  user   mnt ")
    for value in allowed:
        out = VS.systemd_run(_unit_ns(value), CALLER, CALLER_GID, user=job)
        assert [a for a in out if a.startswith("RestrictNamespaces=")] == [f"RestrictNamespaces={value}"], value
    widening = ("", "no", "false", "0", "off", "~", "~user", "~user mnt", "yes mnt", "YES", "Yes", "User", "MNT",
                "user,mnt", "user,net", "pid", "cgroup", "uts", "ipc", "user pid", "bogus", "all")
    for value in widening:
        with pytest.raises(VS.Refused, match="RestrictNamespaces"):
            VS.systemd_run(_unit_ns(value), CALLER, CALLER_GID, user=job)
    # a property repeated with different values: vy-sudo checks each occurrence, so a widening one is caught wherever it sits,
    # whichever systemd would take last
    assert VS.systemd_run(_unit_ns("user", "mnt"), CALLER, CALLER_GID, user=job).count("RestrictNamespaces=user") == 1
    for pair in (("user", "no"), ("no", "user"), ("yes", "~net")):
        with pytest.raises(VS.Refused, match="RestrictNamespaces"):
            VS.systemd_run(_unit_ns(*pair), CALLER, CALLER_GID, user=job)
    with pytest.raises(VS.Refused, match="widen the unit: only .* mnt, net, user"):
        VS.systemd_run(_unit_ns("no"), CALLER, CALLER_GID, user=job)


def test_job_user_is_never_root_or_a_privileged_account():
    for name in ("root", "0", "no-such-user-vy"):
        with pytest.raises(VS.Refused):
            VS.job_user(name)


# ---- vy-sudo: which accounts are job users --------------------------------------------------------------------------------------

DAEMONS = ("polkitd", "messagebus", "_chrony")
SUDO_ENV = {"SUDO_UID": str(CALLER), "SUDO_GID": str(CALLER_GID)}


class Accounts:
    """Stand-ins for vy-sudo's `pwd` and `grp`: `users` is name -> (uid, primary group), `groups` name -> its member list."""

    def __init__(self, users: dict[str, tuple[int, str]], groups: dict[str, tuple[str, ...]]):
        gid = {g: 0 if g == "root" else CALLER + 100 + i for i, g in enumerate(groups)}
        self.users = [SimpleNamespace(pw_name=n, pw_uid=u, pw_gid=gid.get(g, CALLER + 99)) for n, (u, g) in users.items()]
        self.groups = [SimpleNamespace(gr_name=g, gr_gid=gid[g], gr_mem=list(mem)) for g, mem in groups.items()]

    @staticmethod
    def _one(rows, field, value):
        for r in rows:
            if getattr(r, field) == value:
                return r
        raise KeyError(value)

    def getpwnam(self, name):
        return self._one(self.users, "pw_name", name)

    def getpwuid(self, uid):
        return self._one(self.users, "pw_uid", uid)

    def getgrgid(self, gid):
        return self._one(self.groups, "gr_gid", gid)

    def getgrall(self):
        return list(self.groups)


def _node(monkeypatch, listed: tuple[str, ...] = ("vyjob",), group: bool = True, admin_group: str = "vy-admin") -> Accounts:
    """vy-sudo on a node with these accounts: root (and `toor`, uid 0 too), vy-admin, three daemons' (each in its own group and
    no other), `vyjob` (made for jobs), `ops` (in sudo), `imager` (primary group disk) and `byprimary` (primary group the job
    users'), with `listed` as the job users' group's member list; `group=False` is a node that has no such group."""
    uid = {n: CALLER + i for i, n in enumerate(("vy-admin", *DAEMONS, "vyjob", "ops", "imager", "byprimary"), 1)}
    own = ("vy-admin", *DAEMONS, "vyjob", "ops")
    users = {"root": (0, "root"), "toor": (0, "root"), **{n: (uid[n], n) for n in own}, "vy-admin": (uid["vy-admin"], admin_group),
             "imager": (uid["imager"], "disk"), "byprimary": (uid["byprimary"], VS.JOB_GROUP)}
    groups = {"root": (), **{n: () for n in own}, "users": (), "disk": (), "sudo": ("ops",),
              **({VS.JOB_GROUP: tuple(listed)} if group else {})}
    acc = Accounts(users, groups)
    monkeypatch.setattr(VS, "pwd", acc)
    monkeypatch.setattr(VS, "grp", acc)
    return acc


def _unit_as(who: str) -> list[str]:
    """vy-sudo's arguments for a job's unit that runs as `who`."""
    return ["systemd-run", "--unit=research-job-r1", f"--uid={who}", "--wait", "--pipe", "--collect", "--quiet", "--", "python3", "job.py"]


def test_a_daemons_account_is_no_job_user_for_a_unit_or_for_job_reclaim(tmp_path, monkeypatch):
    """A daemon's account (polkitd, messagebus, _chrony) is in no privileged group, and some are root by another name: only a
    list of the accounts that are job users keeps a caller from running a unit as one or taking its files."""
    acc = _node(monkeypatch)
    d, _outside = _left_by_a_job(tmp_path)
    for who in (*DAEMONS, *(str(acc.getpwnam(n).pw_uid) for n in DAEMONS)):
        for argv in (_unit_as(who), ["env", "PATH=/usr/bin", *_unit_as(who)]):
            with pytest.raises(VS.Refused, match="is not a job user"):
                VS.plan(argv, SUDO_ENV)
        with pytest.raises(VS.Refused, match="is not a job user"):
            VS.reclaim([str(d), who], ME, MY_GID, say=lambda s: pytest.fail("nothing is removed"))
    assert (d / "to-dir").is_symlink() and (d / "out" / "sub" / "fifo").is_fifo()


def test_an_account_is_a_job_user_once_root_lists_it_in_the_job_users_group(tmp_path, monkeypatch):
    d, _outside = _left_by_a_job(tmp_path)
    _node(monkeypatch, listed=())
    with pytest.raises(VS.Refused, match="vyjob is not a job user"):
        VS.plan(_unit_as("vyjob"), SUDO_ENV)
    with pytest.raises(VS.Refused, match="vyjob is not a job user"):
        VS.reclaim([str(d), "vyjob"], ME, MY_GID)
    acc = _node(monkeypatch, listed=("vyjob",))                     # after `usermod -aG`
    uid = acc.getpwnam("vyjob").pw_uid
    for who in ("vyjob", str(uid)):
        argv, _env = VS.plan(_unit_as(who), SUDO_ENV)
        assert argv[0] == "/usr/bin/systemd-run" and f"--uid={who}" in argv and argv[-2:] == ["python3", "job.py"]
    assert VS.job_user("vyjob").pw_uid == uid
    removed: list[str] = []
    assert VS.reclaim([str(d), "vyjob"], ME, MY_GID, say=removed.append) == 0 and removed == [], "nothing here is vyjob's"


def test_the_job_users_group_as_a_primary_group_makes_no_job_user(monkeypatch):
    """Only the group's member list counts: `getent group` then prints every job user, and a gid that comes to equal the
    group's grants nothing."""
    _node(monkeypatch, listed=("vyjob",))
    with pytest.raises(VS.Refused, match=f"byprimary is not a job user.*{VS.JOB_GROUP} as its primary group doesn't count.*"
                                         f"usermod -aG {VS.JOB_GROUP} byprimary"):
        VS.plan(_unit_as("byprimary"), SUDO_ENV)
    _node(monkeypatch, listed=("vyjob", "byprimary"))
    assert VS.plan(_unit_as("byprimary"), SUDO_ENV)[0][0] == "/usr/bin/systemd-run"
    _node(monkeypatch, group=False)
    with pytest.raises(VS.Refused, match="byprimary's group .* has no entry"):
        VS.plan(_unit_as("byprimary"), SUDO_ENV)


def test_a_listed_account_in_a_privileged_group_is_still_refused(monkeypatch):
    """The list adds a condition and takes none away: an account listed while it is in sudo, or whose own group is disk."""
    _node(monkeypatch, listed=("vyjob", "ops", "imager"))
    for who, why in (("ops", "ops is in sudo: not a job user"), ("imager", "imager is in disk: not a job user")):
        with pytest.raises(VS.Refused, match=why):
            VS.plan(_unit_as(who), SUDO_ENV)
        with pytest.raises(VS.Refused, match=why):
            VS.reclaim(["/nonexistent", who], CALLER, CALLER_GID)
    _node(monkeypatch, listed=())
    for who in ("ops", "imager"):
        with pytest.raises(VS.Refused) as e:
            VS.plan(_unit_as(who), SUDO_ENV)
        assert "usermod" not in str(e.value), "listing it would not make it a job user, so the refusal doesn't say to"


def test_root_and_vy_admin_are_never_job_users_whatever_their_groups(monkeypatch):
    acc = _node(monkeypatch, listed=("root", "toor", "vy-admin", "vyjob"), admin_group="users")
    for who in ("root", "0", "toor", "vy-admin", str(acc.getpwnam("vy-admin").pw_uid)):
        with pytest.raises(VS.Refused, match="can't be a job user"):
            VS.plan(_unit_as(who), SUDO_ENV)
        with pytest.raises(VS.Refused, match="can't be a job user"):
            VS.reclaim(["/nonexistent", who], CALLER, CALLER_GID)


def test_the_refusal_of_an_unlisted_account_names_who_lists_it_and_with_what_command(monkeypatch, capsys):
    """What the caller reads on stderr (and a `--job-user` run in its log): vy-admin can act on it without the source."""
    fix = f"sudo groupadd -f --system {VS.JOB_GROUP} && sudo usermod -aG {VS.JOB_GROUP} vyjob"
    monkeypatch.setattr(os, "geteuid", lambda: 0)                   # as sudo runs it
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("nothing runs"))
    for name, value in SUDO_ENV.items():
        monkeypatch.setenv(name, value)
    for group in (True, False):                                     # the group with other members, and a node without the group
        _node(monkeypatch, listed=("ops",), group=group)
        for argv in (_unit_as("vyjob"), ["job-reclaim", "/nonexistent", "vyjob"]):
            assert VS.main(argv) == VS.REFUSED
            err = capsys.readouterr().err
            assert err.startswith(f"vy-sudo: refused: vyjob is not a job user: it is not listed in group {VS.JOB_GROUP}"), err
            assert fix in err and "root (vy-admin, after the lock-out)" in err and "made to run jobs and nothing else" in err


def test_the_plan_the_readme_and_research_give_the_command_that_makes_a_job_user(monkeypatch):
    """No tool makes a job user, so the command is where an operator reads it, and each place names vy-sudo's group."""
    group = VS.JOB_GROUP
    assert jobuser.JOB_GROUP == LO.JOB_GROUP == group
    make, grant = f"sudo groupadd -f --system {group}", f"sudo usermod -aG {group} USER"
    for text in (LO.plan_text("vy-nebius-1"), LO.plan_text(CPU13), (LOCKOUT / "README.md").read_text(), VS.__doc__):
        assert make in text and grant in text
    deploy = next(ln for ln in LO.plan_text(CPU13).splitlines() if ln.startswith("1. deploy"))
    assert make in deploy and grant in deploy, "vy-sudo is the one asked from the deploy on, so the plan has it there"

    def missing(name):
        raise KeyError(name)
    with pytest.raises(jobuser.Refused) as e:
        jobuser.check_user("newjob", getpwnam=missing, getgrall=list)
    said = str(e.value)
    assert f"{make} && sudo useradd --system --no-create-home --shell /usr/sbin/nologin --groups {group} newjob" in said
    acc = _node(monkeypatch, listed=())                             # research's own admission has no such rule: vy-sudo's alone
    assert jobuser.check_user("vyjob", getpwnam=acc.getpwnam, getgrall=acc.getgrall, uid=CALLER, gid=CALLER_GID).pw_name == "vyjob"


# ---- vy-lockout: the allowlist --------------------------------------------------------------------------------------------------

def test_each_committed_allowlist_is_its_profiles_render_and_every_rule_names_its_evidence():
    assert [p.file for p in LO.PROFILES] == ["50-vy-research", "50-vy-research-cpu"] and LO.CLUSTER.file == LO.ALLOWLIST
    for p in LO.PROFILES:
        node = "vy-nebius-1" if p is LO.CLUSTER else "vy-nebius-cpu-13"
        assert (LOCKOUT / "sudoers" / p.file).read_text() == LO.render(p), f"regenerate: python3 lockout/lockout.py render --node {node}"
        for r in p.rules + p.ubuntu:
            assert r.cmd.startswith("/") and r.evidence.strip()
            assert Path(r.cmd).name not in ("k3s", "kubectl", "bash", "sh", "env", "python3", "xargs", "find", "tee", "rm", "cp", "mv",
                                            "chown", "systemd-run", "apt-get", "rsync", "tail", "cat", "nft"), r
        assert "k3s" not in LO.render(p) and "kubectl" not in LO.render(p)
    assert "ubuntu" not in LO.render(LO.CPU), "nothing on a CPU node runs as ubuntu, which is in sudo, adm and lxd there"
    assert {(r.cmd, r.args) for r in LO.CPU.rules if "vy-s" in r.cmd} <= {(r.cmd, r.args) for r in LO.ALLOW}, \
        "a CPU node's helper forms are forms the cluster nodes have too"


def test_a_node_has_the_profile_of_its_kind_and_no_other_machine_has_one(tmp_path):
    assert LO.profile("vy-nebius-1") is LO.profile("vy-nebius-2") is LO.CLUSTER
    assert LO.profile("vy-nebius-cpu-13") is LO.profile("vy-nebius-cpu-1") is LO.CPU
    for name in ("", "vy-nebius-3", "vy-nebius-cpu-0", "vy-nebius-cpu-13 ", "vy-nebius-cpu-x", "vy-challenger-1", "vy-auditor-1"):
        with pytest.raises(ValueError):
            LO.profile(name)
        with pytest.raises(SystemExit):
            LO.main(["plan", name])
    assert LO.staged_profile(tmp_path) is LO.CLUSTER
    (tmp_path / LO.ALLOWLIST).write_text(LO.render(LO.CPU))
    assert LO.staged_profile(tmp_path) is LO.CPU


def _parses(text: str) -> bool:
    """A sudoers file as this test reads it where visudo isn't installed: every logical line a Defaults, Cmnd_Alias or user spec."""
    import re
    logical = re.sub(r"\\\n", " ", text)
    ok = re.compile(r"^(Defaults(:[a-z_][a-z0-9_-]*)? \S.*|Cmnd_Alias [A-Z][A-Z0-9_]* = .+|[a-z_][a-z0-9_-]* ALL=\([^)]*\) NOPASSWD: \S.*)$")
    return all(ok.match(ln.strip()) for ln in logical.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))


@pytest.mark.parametrize("name", ["10-vy-admin", "50-vy-research", "50-vy-research-cpu", "fallback"])
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


def _text_allows(text: str, user: str, runas: str, command: str) -> bool:
    """What a committed sudoers file itself grants, read from its text: `USER ALL=(RUNAS, ...) NOPASSWD: ALL | ALIAS`, with an
    alias's commands matched as sudo matches them (the path exactly; the arguments by the pattern, `""` for none)."""
    import fnmatch
    import re
    logical = re.sub(r"\\\n\s*", " ", text)
    aliases = {m.group(1): [c.strip() for c in m.group(2).split(", ")] for m in re.finditer(r"^Cmnd_Alias (\w+) = (.+)$", logical, re.M)}
    path, _, args = command.partition(" ")
    for who, runs, what in re.findall(r"^([a-z_][a-z0-9_-]*) ALL=\(([^)]*)\) NOPASSWD: (\S+)$", logical, re.M):
        if who != user or runas not in [r.strip() for r in runs.split(",")]:
            continue
        if what == "ALL":
            return True
        for spec in aliases[what]:
            cmd, _, pat = spec.partition(" ")
            if cmd == path and (args == "" if pat == '""' else pat != "" and fnmatch.fnmatchcase(args, LO._pattern(pat))):
                return True
    return False


RUN = "/workspace/research/runs/r1"
#: what a run and the store ask of root on a CPU node, as sudo logs each
CPU_OK = [
    ("research", "root", "/usr/local/sbin/vy-sudo systemd-run --unit=research-job-r1 --uid=nobody --wait --pipe -p NoNewPrivileges=yes -- true"),
    ("research", "root", "/usr/local/sbin/vy-sudo systemctl stop --no-block research-job-r1"),
    ("research", "root", f"/usr/local/sbin/vy-sudo job-reclaim {RUN} nobody"),
    ("research", "root", f"/usr/local/sbin/vy-seal-runs delete {RUN}"),
    ("research", "root", f"/usr/local/sbin/vy-seal-runs evict {RUN} --min-bytes 1"),
    ("research", "root", "/usr/bin/systemctl start --no-block vy-lean-cache-daily.service"),
    ("research", "root", "/usr/bin/true"),
    ("research", "research", "/usr/bin/flock -n /workspace/research/slots/0 true"),
]
#: root by any other name, each refused there: a shell, an editor, the packet filter, another user's files, a write under /etc, ...
CPU_REFUSED = [
    ("research", "root", "/usr/bin/bash"), ("research", "root", "/usr/bin/bash -s"), ("research", "root", "/usr/bin/sh -c id"),
    ("research", "root", "/usr/bin/su -"), ("research", "root", "/usr/bin/env bash"), ("research", "root", "/usr/bin/python3 -c x"),
    ("research", "root", "/usr/bin/vi /etc/sudoers"), ("research", "root", "/usr/bin/nano /etc/passwd"),
    ("research", "root", "/usr/sbin/visudo"), ("research", "root", "sudoedit /etc/sudoers.d/50-vy-research"),
    ("research", "root", "/usr/sbin/nft list ruleset"), ("research", "root", "/usr/sbin/nft flush ruleset"),
    ("research", "root", "/usr/sbin/iptables -L"),
    ("research", "root", "/usr/bin/cat /root/.ssh/id_ed25519"), ("research", "root", "/usr/bin/cat /home/vy-admin/.ssh/authorized_keys"),
    ("research", "root", "/usr/bin/tail -c 4096 /root/executor/state.log"), ("research", "root", "/usr/bin/ls /root"),
    ("research", "root", "/usr/bin/find /home/ubuntu -type f"), ("research", "root", "/usr/bin/rsync --server --sender . /root/"),
    ("research", "root", "/usr/bin/tee /etc/sudoers.d/60-me"), ("research", "root", "/usr/bin/cp x /etc/cron.d/x"),
    ("research", "root", "/usr/bin/install -m 0440 x /etc/sudoers.d/x"), ("research", "root", "/usr/bin/sed -i s/a/b/ /etc/passwd"),
    ("research", "root", "/usr/bin/mv /tmp/x /etc/research/deadline"), ("research", "root", "/usr/bin/chown research /etc/shadow"),
    ("research", "root", "/usr/bin/true x"), ("research", "root", "/usr/bin/id"),
    ("research", "root", "/usr/bin/systemctl start vy-lean-cache-daily.service"),
    ("research", "root", "/usr/bin/systemctl start --no-block vy-lean-cache-daily.service ssh.service"),
    ("research", "root", "/usr/bin/systemctl stop ssh"), ("research", "root", "/usr/bin/systemctl daemon-reload"),
    ("research", "root", "/usr/bin/systemd-run --uid=root --pipe bash"), ("research", "root", "/usr/bin/apt-get install -y x"),
    ("research", "root", "/usr/sbin/usermod -aG sudo research"), ("research", "root", "/usr/bin/passwd root"),
    ("research", "root", "/usr/local/sbin/vy-lockout fallback on"), ("research", "root", "/usr/local/sbin/vy-lockout rollback --apply"),
    ("research", "root", "/usr/local/sbin/vy-admin-setup /tmp/keys"), ("research", "root", "/usr/local/sbin/vy-seal-runs seal"),
    ("research", "root", "/usr/local/sbin/vy-sudo hf-ns -- /x"), ("research", "root", "/usr/local/sbin/vy-sudo du-bundles"),
    ("research", "root", "/usr/local/sbin/vy-sudo renice -n 1 -p 1"), ("research", "root", "/usr/local/bin/vy-lean-sandbox probe"),
    ("research", "root", "/usr/local/sbin/vy-sudo"), ("research", "root", "/usr/bin/ip netns exec x bash"),
    ("research", "ubuntu", "/usr/bin/bash -c anything"), ("research", "vy-admin", "/usr/bin/id"), ("research", "lean-build", "/usr/bin/id"),
    ("ubuntu", "root", f"/usr/local/sbin/vy-seal-runs evict {RUN} --min-bytes 1"), ("ubuntu", "root", "/usr/bin/true"),
    ("nobody", "root", "/usr/bin/true"), ("vy-admin", "root", "/usr/bin/true"),
]


def test_a_cpu_nodes_allowlist_lets_a_run_keep_its_forms_and_refuses_root_by_any_other_name():
    text = (LOCKOUT / "sudoers" / LO.CPU.file).read_text()
    for case in CPU_OK:
        assert LO.allowed(*case, LO.CPU) and _text_allows(text, *case), case
    for case in CPU_REFUSED:
        assert not LO.allowed(*case, LO.CPU) and not _text_allows(text, *case), case
    cluster = (LOCKOUT / "sudoers" / LO.CLUSTER.file).read_text()
    for case in (*CPU_OK, *CPU_REFUSED):
        assert LO.allowed(*case) == _text_allows(cluster, *case), ("the cluster nodes' file and ALLOW agree too", case)


def test_a_cpu_nodes_audit_counts_what_its_own_allowlist_would_refuse():
    journal = "\n".join([
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/bin/sh -c 'set -e; c=0123abcd; echo x'",
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/bin/sh -c 'set -e; c=4567ef01; echo x'",
        "  research : PWD=/workspace/research/runs/r1 ; USER=root ; COMMAND=/usr/bin/rm -f /workspace/tmp/tmp.BuFloNGHNH",
        "  research : PWD=/workspace/research/runs/r2 ; USER=root ; COMMAND=/usr/bin/rm -f /workspace/tmp/tmp.iNf4LgE",
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/bin/systemctl start --no-block vy-lean-cache-daily.service",
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/local/sbin/vy-sudo job-reclaim /workspace/research/runs/r1 nobody",
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/bin/systemd-run --unit=research-job-r1 --uid=nobody --wait true",
        "  research : PWD=/home/research ; USER=research ; COMMAND=/usr/bin/flock -n /x true",
        "  research : PWD=/home/research ; USER=ubuntu ; COMMAND=/usr/bin/id",
        "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/bin/true",
        "  research : a password is required ; PWD=/home/research ; USER=root ; COMMAND=/usr/sbin/nft list ruleset",
        "  research : a password is required ; PWD=/home/research ; USER=root ; COMMAND=/usr/local/sbin/vy-sudo renice -n 1 -p 1",
    ])
    rows = LO.audit(journal, LO.CPU)
    assert (1, "research->root /usr/sbin/nft list ruleset") in rows, "what sudo -n refused once the fallback is gone counts as well"
    assert (1, "research->root /usr/local/sbin/vy-sudo renice -n N") in rows
    assert rows[:2] == [(2, "research->root /usr/bin/rm -f /workspace/tmp/tmp.*"), (2, "research->root /usr/bin/sh -c 'set -e;")]
    assert {k.split(" ", 2)[1] for _, k in rows} == {"/usr/bin/sh", "/usr/bin/rm", "/usr/bin/systemd-run", "/usr/bin/id", "/usr/sbin/nft",
                                                     "/usr/local/sbin/vy-sudo"}
    assert (1, "research->ubuntu /usr/bin/id") in rows and (1, "research->ubuntu /usr/bin/id") not in LO.audit(journal)
    assert "0123abcd" not in json.dumps(rows)


def test_the_audit_command_a_cpu_nodes_plan_prints_runs_as_anyone_against_that_nodes_allowlist(monkeypatch, capsys):
    asked: list[list[str]] = []
    journal = ("  research : PWD=/home/research ; USER=ubuntu ; COMMAND=/usr/bin/id\n"
               "  research : PWD=/home/research ; USER=root ; COMMAND=/usr/bin/true\n")
    monkeypatch.setattr(LO, "_run", lambda argv, **kw: asked.append(argv) or subprocess.CompletedProcess(argv, 0, journal, ""))
    monkeypatch.setattr(os, "geteuid", lambda: 1001)
    command = next(ln for ln in LO.plan_text(CPU13).splitlines() if "vy-lockout audit" in ln).split("`")[1].split()
    assert command[:2] == ["vy-lockout", "audit"] and LO.main(command[1:]) == 1, "a refused form is the audit's nonzero exit"
    assert asked == [["journalctl", "SYSLOG_IDENTIFIER=sudo", "--since", "-24h", "-o", "cat"]]
    out = capsys.readouterr().out
    assert "research->ubuntu /usr/bin/id" in out and "sudoers/50-vy-research-cpu" in out
    monkeypatch.setattr(LO, "staged_profile", lambda: LO.CPU)
    assert LO.main(["audit"]) == 1 and "sudoers/50-vy-research-cpu" in capsys.readouterr().out, "the staged file says which node this is"
    monkeypatch.setattr(LO, "staged_profile", lambda: LO.CLUSTER)
    assert LO.main(["audit"]) == 0 and "sudoers/50-vy-research)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        LO.main(["audit", "--node", "vy-auditor-1"])
    capsys.readouterr()


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
    with pytest.raises(LO.Failed, match="isn't `vy-lockout render --node vy-nebius-1`'s"):
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
    assert {"vy-admin", "vy-admin-login", "vy-admin-session", "ubuntu-unprivileged", "py-spy", "kubeconfig", "journal-group", "allowlist",
            "switched:vy-gpu-stray.service", "root-unit:vy-node1-dispatch.service", "tool:/usr/local/sbin/vy-sudo"} <= failing
    assert all(checks[n].fix for n in failing)


CPU13 = "vy-nebius-cpu-13"
ADMIN_UID, RESEARCH_UID = 1002, 1001
KEYS_AT = 1_791_000_000                      # when vy-admin's keys and sudoers file were installed


def _prepared_cpu_node(root: Path) -> None:
    """A CPU node after the deploy, vy-admin and harden steps, as preflight reads it."""
    for d in ("etc/sudoers.d", "home/vy-admin/.ssh", "usr/local/sbin", "usr/local/lib/vy-lockout"):
        (root / d).mkdir(parents=True)
    (root / "etc" / "passwd").write_text(f"root:x:0:0::/root:/bin/bash\nresearch:x:{RESEARCH_UID}:{RESEARCH_UID}::/home/research:/bin/bash\n"
                                         f"vy-admin:x:{ADMIN_UID}:{ADMIN_UID}::/home/vy-admin:/bin/bash\n")
    (root / "etc" / "group").write_text("sudo:x:27:ubuntu\nadm:x:4:ubuntu\nlxd:x:110:\nsystemd-journal:x:999:research\n")
    keys, admin = root / "home/vy-admin/.ssh/authorized_keys", root / "etc/sudoers.d" / LO.ADMIN
    keys.write_text("ssh-ed25519 AAAA lead@laptop\n")
    admin.write_text((LOCKOUT / "sudoers" / LO.ADMIN).read_text())
    for f in (keys, admin):
        os.utime(f, (KEYS_AT, KEYS_AT))
    for tool in ("vy-sudo", "vy-seal-runs", "vy-lockout"):
        (root / "usr/local/sbin" / tool).write_text("#!/usr/bin/python3 -I\n")
    (root / "usr/local/lib/vy-lockout" / LO.ALLOWLIST).write_text(LO.render(LO.CPU))


def _sshd(*logins: float):
    """A runner whose sshd journal holds a key login as vy-admin at each of `logins` (and whose every other command finds nothing)."""
    def run(argv):
        if argv[0] == "journalctl":
            out = "".join(f"{t:.6f} node sshd[41]: Accepted publickey for vy-admin from bastion port 22 ssh2: ED25519 SHA256:x\n" for t in logins)
            return subprocess.CompletedProcess(argv, 0 if logins else 1, out, "")
        return subprocess.CompletedProcess(argv, 1, "", "")
    return run


def test_a_prepared_cpu_node_passes_preflight_with_no_cluster_check(tmp_path):
    _prepared_cpu_node(tmp_path)
    checks = LO.preflight(CPU13, _sshd(KEYS_AT + 60), tmp_path, session=lambda: ADMIN_UID)
    assert [c.name for c in checks if not c.ok] == []
    names = [c.name for c in checks]
    assert not [n for n in names if n.startswith(("switched:", "root-unit:"))], "a CPU node has no loop to switch"
    assert {"vy-admin", "vy-admin-login", "vy-admin-session", "allowlist", "journal-group", "tool:/usr/local/sbin/vy-sudo"} <= set(names)
    (tmp_path / "usr/local/lib/vy-lockout" / LO.ALLOWLIST).write_text(LO.render(LO.CLUSTER))
    checks = {c.name: c for c in LO.preflight(CPU13, _sshd(KEYS_AT + 60), tmp_path, session=lambda: ADMIN_UID)}
    assert not checks["allowlist"].ok and "50-vy-research-cpu" in checks["allowlist"].detail, "the cluster nodes' file is not a CPU node's"


@pytest.mark.parametrize("why, logins, session, broken, failing", [
    ("nobody has logged in as vy-admin", (), ADMIN_UID, None, {"vy-admin-login"}),
    ("the last login is from before the keys in place now", (KEYS_AT - 60,), ADMIN_UID, None, {"vy-admin-login"}),
    ("the step is run from research's login, through research's sudo", (KEYS_AT + 60,), RESEARCH_UID, None, {"vy-admin-session"}),
    ("the session has no login uid", (KEYS_AT + 60,), None, None, {"vy-admin-session"}),
    ("vy-admin has no key", (KEYS_AT + 60,), ADMIN_UID, "keys", {"vy-admin"}),
    ("vy-admin has no sudoers file", (KEYS_AT + 60,), ADMIN_UID, "sudoers", {"vy-admin"}),
    ("there is no vy-admin", (), None, "user", {"vy-admin", "vy-admin-login", "vy-admin-session"}),
])
def test_the_sudoers_step_is_refused_on_a_cpu_node_while_the_admin_login_does_not_work(tmp_path, why, logins, session, broken, failing):
    """Root must work through vy-admin before research loses it: a node locked without that needs its disk mounted elsewhere."""
    root, sd, backup = tmp_path / "node", tmp_path / "node" / "etc" / "sudoers.d", tmp_path / "backup"
    _prepared_cpu_node(root)
    if broken in ("keys", "user"):
        (root / "home/vy-admin/.ssh/authorized_keys").unlink()
    if broken in ("sudoers", "user"):
        (sd / LO.ADMIN).unlink()
    if broken == "user":
        (root / "etc" / "passwd").write_text(f"research:x:{RESEARCH_UID}:{RESEARCH_UID}::/home/research:/bin/bash\n")
    (sd / LO.CLOUD_INIT).write_text("research ALL=(ALL) NOPASSWD:ALL\n")
    before = sorted(p.name for p in sd.iterdir())
    checks = LO.preflight(CPU13, _sshd(*logins), root, session=lambda: session)
    assert {c.name for c in checks if not c.ok} == failing, why
    log: list[list[str]] = []
    ctx = LO.Ctx(apply=True, run=runner(log), say=lambda s: None)
    with pytest.raises(LO.Failed, match="preflight fails: vy-admin"):
        LO.step_sudoers(ctx, CPU13, [], KEYS_AT + 120.0, root / "usr/local/lib/vy-lockout", sd, backup, checks=checks)
    assert sorted(p.name for p in sd.iterdir()) == before and not backup.exists() and log == [], "nothing was installed or moved"


def test_the_sudoers_step_installs_a_cpu_nodes_own_allowlist_and_rolls_back(tmp_path):
    root, sd, backup = tmp_path / "node", tmp_path / "node" / "etc" / "sudoers.d", tmp_path / "backup"
    _prepared_cpu_node(root)
    lib = root / "usr/local/lib/vy-lockout"
    (sd / LO.CLOUD_INIT).write_text("research ALL=(ALL) NOPASSWD:ALL\n")
    checks = LO.preflight(CPU13, _sshd(KEYS_AT + 60), root, session=lambda: ADMIN_UID)
    log: list[list[str]] = []
    ctx = LO.Ctx(apply=True, run=runner(log), say=lambda s: None)
    t = KEYS_AT + 120.0
    (lib / LO.ALLOWLIST).write_text(LO.render(LO.CLUSTER))
    with pytest.raises(LO.Failed, match="isn't `vy-lockout render --node vy-nebius-cpu-13`'s"):
        LO.step_sudoers(ctx, CPU13, [], t, lib, sd, backup, checks=checks)
    assert not (sd / LO.ALLOWLIST).exists() and (sd / LO.CLOUD_INIT).exists()
    (lib / LO.ALLOWLIST).write_text(LO.render(LO.CPU))
    LO.step_sudoers(ctx, CPU13, [], t, lib, sd, backup, checks=checks)
    assert (sd / LO.ALLOWLIST).read_text() == LO.render(LO.CPU) and LO.expiry((sd / LO.FALLBACK).read_text()) == t + 24 * 3600
    assert stat.S_IMODE((sd / LO.ALLOWLIST).stat().st_mode) == 0o440
    assert not (sd / LO.CLOUD_INIT).exists() and (backup / "sudoers" / LO.CLOUD_INIT).exists()
    assert ["systemctl", "enable", "--now", "vy-sudo-fallback-expire.timer"] in log
    LO.step_rollback(ctx, sd, backup)
    assert (sd / LO.CLOUD_INIT).read_text() == "research ALL=(ALL) NOPASSWD:ALL\n"
    assert not (sd / LO.ALLOWLIST).exists() and not (sd / LO.FALLBACK).exists() and (sd / LO.ADMIN).exists(), "vy-admin keeps root"


def test_the_session_is_the_login_uid_the_kernel_keeps(tmp_path):
    f = tmp_path / "loginuid"
    assert LO.login_uid(str(f)) is None
    for text, uid in (("1002\n", 1002), ("4294967295\n", None), ("", None), ("x", None)):
        f.write_text(text)
        assert LO.login_uid(str(f)) == uid
    (tmp_path / "passwd").write_text("root:x:0:0::/root:/bin/bash\nvy-admin:x:1002:1002::/home/vy-admin:/bin/bash\n")
    assert LO.uid_of("vy-admin", str(tmp_path / "passwd")) == 1002 and LO.uid_of("vy", str(tmp_path / "passwd")) is None
    assert LO.uid_of("vy-admin", str(tmp_path / "missing")) is None


def test_harden_back_puts_back_only_what_the_node_has(tmp_path, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(LO, "PY_SPY", str(tmp_path / "py-spy"))
    LO.step_harden(LO.Ctx(apply=False, say=said.append), back=True)
    assert not [s for s in said if "setcap" in s] and any("gpasswd -d research systemd-journal" in s for s in said), "a CPU node has no py-spy"
    (tmp_path / "py-spy").write_text("")
    LO.step_harden(LO.Ctx(apply=False, say=said.append), back=True)
    assert [s for s in said if "setcap" in s]


def test_the_plan_has_every_step_with_its_rollback():
    text = LO.plan_text("vy-nebius-1")
    assert [ln.split(".")[0] for ln in text.splitlines() if not ln.startswith(" ")] == [str(i) for i in range(1, 10)]
    assert text.count("   rollback: ") == 9 and "vy-node1-dispatch.service (tmux node1-dispatch)" in text
    cpu = LO.plan_text(CPU13)
    steps = [ln.split(":")[0] for ln in cpu.splitlines() if not ln.startswith(" ")]
    assert steps == ["1. deploy", "2. vy-admin", "3. seal", "4. harden", "5. sudoers", "6. watch"], "no loop to switch, no cluster"
    assert cpu.count("   rollback: ") == 6 and not [w for w in ("tmux", "kubeconfig", "root.env") if w in cpu]
    assert f"vy-lockout sudoers --node {CPU13} --apply" in cpu and "--login vy-admin" in cpu and f"audit --since=-24h --node {CPU13}" in cpu
    assert "sudo sshd -t && sudo systemctl reload ssh" in cpu and f"{LO.SSHD_ADMIN}; sshd -t && systemctl reload ssh" in cpu


def test_a_cpu_node_that_gets_the_lock_out_gets_every_file_of_it_and_the_sshd_file_that_lets_vy_admin_in():
    """vm_setup.sh's sshd allows only research: a node staged without the AllowUsers file can never pass preflight's login check."""
    import tomllib
    files = tomllib.loads((NEBIUS / "deploy.toml").read_text())["file"]
    on = {n: {f["path"]: f["src"] for f in files if n in f.get("nodes", []) and "src" in f} for f in files for n in f.get("nodes", [])
          if LO.CPU_NODE.fullmatch(n)}
    staged = {n: paths for n, paths in on.items() if any(p.startswith(str(LO.LIB)) for p in paths)}
    assert CPU13 in staged
    lockout = "tools/research/src/research/pods/nebius/lockout/"
    for node, paths in staged.items():
        assert paths[str(LO.LIB / LO.ALLOWLIST)] == lockout + "sudoers/" + LO.CPU.file, (node, "its own allowlist, under the staged name")
        assert paths[str(LO.LIB / LO.ADMIN)] == lockout + "sudoers/" + LO.ADMIN
        assert paths[LO.SSHD_ADMIN].endswith("/11-vy-admin.conf"), node
        assert {"/usr/local/sbin/vy-lockout", "/usr/local/sbin/vy-sudo", "/usr/local/sbin/vy-seal-runs", "/usr/local/sbin/vy-admin-setup",
                "/etc/systemd/system/vy-seal-runs.timer", "/etc/systemd/system/vy-seal-runs.service",
                "/etc/systemd/system/vy-sudo-fallback-expire.timer", "/etc/systemd/system/vy-sudo-fallback-expire.service"} <= set(paths), node
    text = (NEBIUS / "challenger" / "11-vy-admin.conf").read_text()
    assert [ln for ln in text.splitlines() if ln and not ln.startswith("#")] == ["AllowUsers vy-admin"]


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
