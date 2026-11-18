"""A machine that is not ours: a node of a Slurm cluster, reached only through batch jobs that the registered machine `via`
submits with infra/vcf's scripts, under the cluster's key only `via` holds.  Its register entry, machines.d/NAME.toml:

    provider = "slurm"
    via = "vy-control-verity"     # the registered machine that holds the cluster's key and submits the job
    partition = "main"
    nodelist = "h100-sxm-8-0"     # optional
    reservation = "RES"           # optional
    gres = "gpu:1"                # optional: the job's --gres when the run gives no --gpus
    max_minutes = 30
    project = "verity"
    [properties]                  # optional: what a job may do there, copied into the run as inputs/NAME.toml

The entry names no address, user, key or account: `via`'s scripts hold those.  `research run --on NAME --project P [--source
GITDIR] ... -- CMD` builds infra/vcf/payload.sh's payload of GITDIR's HEAD commit (default: the checkout here), ships it to `via`
by `research pods ssh`'s path, and starts the commit's own vcf-job-run.sh there in a session of its own (setsid), so the launch returns once
Slurm has the job, and the job outlives the launching VM.  On the node the job runs CMD under `research run --local-only --id RUN
--machine NAME`; vcf-job-run.sh copies its run dir back to `via` and publishes it to the store under RUN, where `research status
RUN` and `research fetch RUN` find it.  No lease, slot or pool of ours exists there.
"""
from __future__ import annotations

import argparse
import math
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .. import remote
from . import registry
from ..telemetry.events import utc_now_ms
from ..telemetry.job import write_json_atomic

PAYLOAD_SH = Path(__file__).resolve().parents[5] / "infra" / "vcf" / "payload.sh"
#: on `via`: one directory per run, holding its payload and the launch's log, and vcf-job-run.sh's OUT, where the job's output lands
VIA_LAUNCHES = "/data/vcf/launches"
VIA_JOBS = "/data/vcf-jobs"
#: the payload's own scripts, run on `via` from the payload unpacked there
JOB_RUN = "tree/infra/vcf/vcf-job-run.sh"
JOB_SCRIPT = "tree/infra/vcf/vcf-bench.sh"
#: a commit without this file predates `research run --machine`, which the job runs on the node
MARKER = "tools/research/src/research/pods/slurm.py"
#: of a job's minutes, those uv sync takes on the node before the command starts: the command's own timeout is the rest
SETUP_MIN = 10
SUBMIT_WAIT_S = 300
PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
NODELIST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,_\[\]-]{0,255}")
GRES = re.compile(r"[A-Za-z0-9][A-Za-z0-9:_-]{0,63}")
JOB_LINE = re.compile(r" job (\d+)$", re.M)
#: what a slurm entry never holds: the cluster's address and login stay in `via`'s scripts
NOT_HELD = ("host", "port", "user", "account", "key", "identity", "pod_id")


def problem(entry: dict) -> str | None:
    """Why `entry` cannot be a `provider = "slurm"` machine, or None.  Every value reaches a shell on `via`, so each is plain."""
    if held := [k for k in NOT_HELD if k in entry]:
        return f"{', '.join(held)}: a slurm machine's entry names no address, login or key (its `via` machine's scripts hold them)"
    for k, pat, need in (("via", PLAIN, True), ("partition", PLAIN, True), ("nodelist", NODELIST, False),
                         ("reservation", PLAIN, False), ("gres", GRES, False)):
        v = entry.get(k)
        if (v is not None or need) and not (isinstance(v, str) and pat.fullmatch(v)):
            return f"{k} {v!r} is not {'a registered machine name' if k == 'via' else 'a plain Slurm name'} ({pat.pattern})"
    if not (type(entry.get("max_minutes")) is int and entry["max_minutes"] > 0):
        return f"max_minutes {entry.get('max_minutes')!r} is not a positive whole number of minutes"
    return None


def refusal(ns: argparse.Namespace, harness_opts: list[str], command: list[str], m: dict) -> str | None:
    """Why `research run --on NAME` refuses this run for slurm machine `m`, with what to pass instead, or None."""
    name = ns.on
    instead = (
        ("--pool", ns.pool, f"{name} has no pool or slot of ours: drop it; Slurm gives the job the cores, memory and cards that "
                            "--cpus, --mem-gb and --gpus ask"),
        ("--gpu-ids", ns.gpu_ids, "Slurm picks the cards: ask how many with --gpus N"),
        ("--gpu-host", ns.gpu_host, f"no card lease of ours runs on {name}: ask the cards with --gpus N, which the job holds whole"),
        ("--queue", ns.queue, "the cluster queues the job itself: drop --queue"),
        ("--wait", ns.wait, "its records reach the store only once `via` publishes them: launch without --wait, then "
                            "`research status RUN`"),
        ("--env-host", ns.env_host, "it would send a machine's address to the cluster: drop it"),
        ("--send", ns.send, "the payload carries the commit alone: commit the file"),
        ("--exclusive", ns.exclusive, "Slurm decides who shares the node: drop it"),
        ("--mem-wait", ns.mem_wait, "Slurm holds the job until its memory is free: drop it"),
        ("--job-user", ns.job_user, "the job runs as the cluster's account: drop it"),
        ("--store-read", ns.store_read, "the cluster holds no store key: drop it"),
        ("--preflight", ns.preflight, "drop it"), ("--stay", ns.stay, "drop it"),
        ("--custody-r2/--custody-ttl", ns.custody_r2 is not None or ns.custody_ttl, f"{m['via']} publishes the run with its own key: "
                                                                                     "drop them"),
        ("--expect/--require-result/--scratch", ns.expect or ns.require_result or ns.scratch, "drop them"),
    )
    if given := [(flag, why) for flag, on, why in instead if on]:
        return "; ".join(f"{flag}: {why}" for flag, why in given)
    if any("gpu-lease" in a for a in command):
        return (f"the command asks gpu-lease for cards, and no card lease of ours runs on {name}: drop it and ask the cards with "
                "--gpus N (Slurm hands the job those cards alone)")
    cwd = remote.flag_value(harness_opts, "--cwd")
    if cwd not in (None, "source"):
        return f"--cwd {cwd}: the job runs in the payload's tree (--cwd source)"
    rest, skip = [], False
    for a in harness_opts:
        if skip:
            skip = False
        elif a in ("--timeout", "--cwd"):
            skip = True
        elif not a.startswith(("--timeout=", "--cwd=")):
            rest.append(a)
    if rest:
        return f"{' '.join(rest)}: not carried to a slurm machine's run (its command, --tool, --env, --timeout and --cwd source are)"
    return None


def minutes(ns: argparse.Namespace, harness_opts: list[str], m: dict) -> int:
    """The job's minutes: --timeout's or --max-min's (the larger), else the entry's max_minutes.  ValueError above max_minutes."""
    t = remote.timeout_seconds(remote.flag_value(harness_opts, "--timeout"))
    asked = [x for x in (math.ceil(t / 60) if t else None, ns.max_min) if x]
    most = m["max_minutes"]
    n = max(asked) if asked else most
    if n > most:
        raise ValueError(f"the run asks {n} min (--timeout / --max-min), above {ns.on}'s max_minutes {most} (its register entry): "
                         f"pass --timeout {most}m or less, or split the work into runs of at most {most} min")
    return n


def sbatch_args(ns: argparse.Namespace, m: dict) -> list[str]:
    """--gpus N as --gres=gpu:N (none: the entry's gres; 0: none), --cpus, --mem-gb, and the entry's nodelist and reservation."""
    gres = f"gpu:{ns.gpus}" if ns.gpus else None if ns.gpus == 0 else m.get("gres")
    out = [f"--gres={gres}"] if gres else []
    out += [f"--cpus-per-task={ns.cpus}"] if ns.cpus else []
    out += [f"--mem={math.ceil(ns.mem_gb)}G"] if ns.mem_gb else []
    return out + [f"--{k}={m[k]}" for k in ("nodelist", "reservation") if m.get(k)]


def _git(src: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise remote.Refused(f"git {' '.join(args)} in {src}: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()


def build_payload(src: Path, run_id: str, name: str, m: dict, cmd_min: int, ns: argparse.Namespace, command: list[str],
                  into: Path) -> str:
    """Write payload.sh's payload of `src`'s HEAD commit to `into`, carrying the run's id, machine and command timeout; -> the commit."""
    commit = _git(src, "rev-parse", "--verify", "HEAD^{commit}")
    if subprocess.run(["git", "-C", str(src), "cat-file", "-e", f"{commit}:{MARKER}"], capture_output=True).returncode != 0:
        raise remote.Refused(f"commit {commit[:12]} of {src} predates the Slurm launch (no {MARKER}, so the node's `research run` "
                             "has no --machine): launch from a commit that has it (merge origin/main)")
    if _git(src, "status", "--porcelain", "--untracked-files=no"):
        print(f"research run --on {name}: {src} has uncommitted changes, which the payload leaves out: it carries commit "
              f"{commit[:12]} alone", file=sys.stderr)
    args = ["bash", str(PAYLOAD_SH), commit, "--run-id", run_id, "--project", ns.project, "--machine", name, "--timeout-min", str(cmd_min)]
    args += ["--tool", ns.tool] if ns.tool else []
    for e in ns.env:
        args += ["--env", e]
    with tempfile.TemporaryDirectory(prefix="research-slurm-") as t:
        if (props := registry.properties_copy(name, m)) is not None:
            (Path(t) / f"{name}.toml").write_text(props)
            args += ["--properties", str(Path(t) / f"{name}.toml")]
        with into.open("wb") as out:
            r = subprocess.run([*args, "--", *command], cwd=src, stdout=out, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise remote.Refused(f"payload.sh failed (rc {r.returncode}): {r.stderr.decode(errors='replace').strip()[-300:]}")
    return commit


def _via_ssh(via: str) -> list[str]:
    from . import connect

    return connect.ssh_base(via)


def start(via: str, run_id: str, payload: Path, job_args: list[str]) -> str:
    """Ship the payload to `via`, start vcf-job-run.sh there in a session of its own (setsid; its pid in launch.pid), and wait (on
    `via`, at most
    SUBMIT_WAIT_S) for Slurm's job id, which it returns.  Refused when the copy fails or the job is not submitted."""
    ssh, q = _via_ssh(via), shlex.quote
    d, out = f"{VIA_LAUNCHES}/{run_id}", f"{VIA_JOBS}/{run_id}"
    see = f"`research pods ssh {via} -- tail {d}/launch.log` says why"
    with payload.open("rb") as f:
        r = subprocess.run([*ssh, f"mkdir -p {q(d)} && cat > {q(d)}/payload.tar.gz"], stdin=f, capture_output=True, timeout=1800)
    if r.returncode != 0:
        raise remote.Refused(f"the payload's copy to {via} failed (ssh rc {r.returncode}): `research pods ssh {via} -- true` shows "
                             "ssh's message")
    job = f"echo $$ > launch.pid; SEND={q(d)}/payload.tar.gz OUT={q(out)} bash {shlex.join([JOB_RUN, *job_args])} > launch.log 2>&1"
    grep = "grep -o -m1 ' job [0-9][0-9]*$' launch.log 2>/dev/null"
    script = (f"cd {q(d)} && tar -xzf payload.tar.gz tree/infra/vcf && setsid -f bash -c {q(job)} < /dev/null > /dev/null 2>&1 || exit 3; "
              "for i in $(seq 50); do [ -s launch.pid ] && break; sleep 0.1; done; "
              f"for i in $(seq {SUBMIT_WAIT_S // 2}); do {grep} && exit 0; "
              f"kill -0 \"$(cat launch.pid 2>/dev/null)\" 2>/dev/null || {{ {grep} && exit 0; tail -n 1 launch.log; exit 5; }}; sleep 2; done; "
              "exit 4")
    r = subprocess.run([*ssh, script], capture_output=True, timeout=SUBMIT_WAIT_S + 120)
    text = r.stdout.decode(errors="replace")
    if r.returncode == 0 and (hit := JOB_LINE.search(text)):
        return hit.group(1)
    if r.returncode == 4:
        raise remote.Refused(f"Slurm gave no job id within {SUBMIT_WAIT_S} s; the job may still be submitted: {see}")
    # the log's fixed phrase alone (`not submitted`, `the payload's copy failed`): what follows it may be the cluster's words
    said = text.strip().splitlines()[-1].split(" ", 1)[-1].split(":", 1)[0] if r.returncode == 5 and text.strip() else f"rc {r.returncode}"
    raise remote.Refused(f"the job was not started on {via} ({said}): {see}")


def launch(ns: argparse.Namespace, harness_opts: list[str], command: list[str], m: dict) -> int:
    """`research run --on NAME` for slurm machine `m` (module docstring)."""
    from .. import runs as R

    name = ns.on
    try:
        remote.resolve_machine(name, ns.project)
        if why := refusal(ns, harness_opts, command, m):
            raise ValueError(why)
        n = minutes(ns, harness_opts, m)
    except (ValueError, remote.Refused) as e:
        print(f"research run --on {name}: refused: {e}", file=sys.stderr)
        return 2
    cmd_min = max(n - SETUP_MIN, n // 2, 1)
    sbatch = sbatch_args(ns, m)
    try:
        src = Path(_git(Path(ns.source or ".").expanduser(), "rev-parse", "--show-toplevel"))
        run_dir = R.take_run_dir(R.runs_root(ns.runs_dir), ns.id)
    except (remote.Refused, R.RunDirTaken) as e:
        print(f"research run --on {name}: refused: {e}", file=sys.stderr)
        return 2
    run_id = run_dir.name
    try:
        with tempfile.TemporaryDirectory(prefix="research-slurm-") as t:
            commit = build_payload(src, run_id, name, m, cmd_min, ns, command, Path(t) / "payload.tar.gz")
            job = start(m["via"], run_id, Path(t) / "payload.tar.gz", [run_id, m["partition"], str(n), JOB_SCRIPT, *sbatch])
    except remote.Refused as e:
        R.drop_run_dir(run_dir)
        print(f"research run --on {name}: refused: {e}", file=sys.stderr)
        return 3
    d, out = f"{VIA_LAUNCHES}/{run_id}", f"{VIA_JOBS}/{run_id}"
    write_json_atomic(run_dir / "remote.json", {
        "schema": "research/remote/v0.1", "run_id": run_id, "machine": name, "provider": "slurm", "via": m["via"], "project": ns.project,
        "phase": "launched", "created_utc": utc_now_ms(), "observed_utc": None, "last_observed": None, "command": command,
        "launch_args": ["--stage", ns.stage, *harness_opts], "source_sha": commit, "slurm_job": job, "minutes": n, "sbatch": sbatch,
        "via_dir": d, "via_out": out})
    print(f"research: run {run_id} submitted to {name} as Slurm job {job} ({n} min, its command {cmd_min} of them; "
          f"{' '.join(sbatch) or 'no sbatch options'}), from a session of its own on {m['via']} (pid in {d}/launch.pid)", file=sys.stderr)
    print(f"research: its output lands in {m['via']}:{out}/ and its launch log is {m['via']}:{d}/launch.log; once {m['via']} has "
          f"published it, `research status {run_id}` and `research fetch {run_id}` read it from the store. The job does not depend "
          "on this session staying alive", file=sys.stderr)
    return 0


def _open_store() -> Any:
    from ..store.attempt import open_store

    return open_store()


def fetch(local_dir: Path, rec: dict, *, store: Any = None) -> dict:
    """`remote.fetch` of a slurm run: its run record from the store once `via` has published it, else `via`'s last log line."""
    run_id = rec["run_id"]
    st = store or _open_store()
    record = ((st.get_attempt(run_id) or {}).get("outputs") or {}).get("run_record")
    rec["observed_utc"] = utc_now_ms()
    if record:
        with tempfile.TemporaryDirectory(prefix="research-slurm-") as t:
            st.fetch(record, to=Path(t))
            for p in sorted(Path(t).rglob("*")):
                rel = p.relative_to(t)
                if p.is_file() and rel.as_posix() != "remote.json":
                    (local_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, local_dir / rel)
        s = remote._read_json(local_dir / "status.json") or {}
        rec.update(phase="published", run_record=record, preserved=True,
                   last_observed={"state": s.get("state"), "rc": s.get("rc"), "at_utc": rec["observed_utc"]})
        rec.pop("last_fetch_error", None)
    else:
        q = shlex.quote
        r = remote.ssh_run(_via_ssh(rec["via"]), ["sh", "-c", f"tail -n 1 {q(rec['via_out'])}/vcf-job.log 2>/dev/null || "
                                                             f"tail -n 1 {q(rec['via_dir'])}/launch.log"], timeout=120)
        if r.returncode != 0:
            rec["last_fetch_error"] = {"at_utc": rec["observed_utc"], "rc": r.returncode,
                                       "stderr": f"`research pods ssh {rec['via']} -- true` shows ssh's message"}
        else:
            rec.pop("last_fetch_error", None)
            rec["via_said"] = r.stdout.decode(errors="replace").strip()[-200:]
            print(f"research: {run_id} is not in the store yet; {rec['via']}'s log of Slurm job {rec.get('slurm_job')}: "
                  f"{rec['via_said'] or '(empty)'}", file=sys.stderr)
    write_json_atomic(local_dir / "remote.json", rec)
    return rec


def adopt(runs_root: Path, run_id: str, *, store: Any = None) -> Path | None:
    """The local run dir of a published run another VM launched onto a slurm machine, made from the store; None when the store
    has no such run."""
    st = store or _open_store()
    try:
        att = st.get_attempt(run_id) or {}
    except Exception as e:  # noqa: BLE001  (no store, an unreachable remote: the run is then not found, and this says why)
        print(f"research: the store could not be asked for {run_id} ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    on = (att.get("execution") or {}).get("machine")
    m = remote.load_machines().get(on) if on else None
    if not (m and m.get("provider") == "slurm" and (att.get("outputs") or {}).get("run_record")):
        return None
    local = runs_root / run_id
    local.mkdir(parents=True, exist_ok=False)
    fetch(local, {"schema": "research/remote/v0.1", "run_id": run_id, "machine": on, "provider": "slurm", "via": m["via"],
                  "phase": "adopted", "created_utc": utc_now_ms()}, store=st)
    return local
