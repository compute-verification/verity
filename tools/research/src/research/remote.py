from __future__ import annotations

import concurrent.futures
import errno
import fcntl
import hashlib
import io
import json
import os
import secrets
import select
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
from pathlib import Path, PurePosixPath
from typing import IO, Any, Callable

from . import clone
from .telemetry.events import utc_now_ms
from .telemetry.job import write_json_atomic

REQUEST_SCHEMA = "research/request/v0.1"
PRESERVED_SCHEMA = "research/preserved/v0.1"
READY_SCHEMA = "research/source-ready/v0.1"
READY_FILE = "READY.json"
_DIRT_MARK = "RESEARCH-TRACKED-DIRT"
LEGACY_COMPLETE = ".complete"
SHIP_SOURCE_TIMEOUT_S = 600
SHIP_MIN_BYTES_PER_S = 64 * 1024
SHIP_CHUNK = 1 << 20
STALE_PARTIAL_MIN = 30
_RC_DIGEST = 75
WORKLOAD_CWD_DEFAULT = "run_dir"
SMALL_FILES = ("job.json", "status.json", "failure.json", "result.json", "launch.json", "refused.json", "events.jsonl", "slot.json")
TERMINAL = ("done", "failed", "cancelled")
FETCHED_FILE = ".fetched"
FETCHED_SCHEMA = "research/fetched/v0.1"
TAR_CHANGED_OK = '; rc=$?; [ "$rc" -eq 1 ] && rc=0; exit "$rc"'
CUSTODY_DIR = "custody"
STORE_READ_DIR = "store-read"
LAUNCH_ACK = ".launch-ack"
LAUNCH_LOCK = ".launch.lock"
CLAIM_MARK = ".claimed"
STAGE_TAKEN_RC = 17
FIRST_RECORD = "job.json"
RUNNER_FAILED_LINES = 5
CANCEL_GRACE_S = 20
GUARD_IDLE_MIN_DEFAULT = 90
GUARD_ENVIRON = "/proc/1/environ"
DIRECT_GPUS_FILE = "/etc/vy/direct-gpus"
GPU_LEASE_HOST_FILE = "/etc/vy/gpu-lease-host"
DIRECT_CPUS_FILE = "/etc/vy/direct-cpus"
RUNS_VIA_SLOT_FILE = "/etc/vy/runs-via-slot"
NODE_SLOT_ONLY_FILE = "/usr/local/lib/vy-monitoring/slot_only.py"
NODE_TOOL_RULES_FILE = "/usr/local/lib/vy-monitoring/tool_rules.py"
AUDITOR_FILE = "/etc/vy/auditor"
GUARD_SCRIPT = Path(__file__).resolve().parent / "pods" / "sh" / "pod_guard.sh"
POD_TARGET_RETRIES = 6
POD_TARGET_RETRY_S = 2.0
_TOLD_KEPT: set[str] = set()
PREFLIGHT_TIMEOUT_S = 180
PREFLIGHT_TAIL = 40
LAUNCH_DELIVERIES = 3


def launch_deadline(mem_wait_s: float | None) -> float:
    return 120 + PREFLIGHT_TIMEOUT_S + (mem_wait_s or 0)


def launch_pause(attempt: int) -> float:
    return 3 * (attempt + 1)


def launch_limit(mem_wait_s: float | None) -> float:
    return sum(launch_deadline(mem_wait_s) + launch_pause(k) for k in range(LAUNCH_DELIVERIES))


class Refused(Exception):
    pass


class SourceNotReady(Refused):
    pass


class ResearchVersionMismatch(Refused):
    pass


class MemoryNotAvailable(Refused):
    pass


class PreflightFailed(Refused):
    def __init__(self, reason: str, report: dict[str, Any] | None = None):
        super().__init__(reason)
        self.report = report


class RunnerFailed(Refused):
    pass


class RunIdTaken(Refused):
    pass


class CannotStart(Refused):
    pass


NOT_STARTED = "the runner could not be started"
ARGS_TOO_LONG = f"[Errno {errno.E2BIG}] "


def cannot_start(answer: dict[str, Any]) -> str | None:
    reason, fault = str(answer.get("refused") or ""), answer.get("fault")
    older = fault == RunnerFailed.__name__ and reason.startswith(NOT_STARTED) and ARGS_TOO_LONG not in reason
    return reason if reason and (fault == CannotStart.__name__ or older) else None


def machines_path() -> Path:
    return Path(os.environ.get("RESEARCH_MACHINES", Path.home() / ".research" / "machines.toml")).expanduser()


def load_machines(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or machines_path()
    ms = tomllib.loads(p.read_text()).get("machines", {}) if p.exists() else {}
    if path is None:
        from .pods import registry

        ms = registry.merged(ms, p)
    return ms


DEFAULT_ROOT = "/workspace/research"
DEPLOY_MANIFEST = Path(__file__).resolve().parent / "pods" / "nebius" / "deploy.toml"


def restricted(machines: dict[str, dict[str, Any]]) -> set[str]:
    nodes = tomllib.loads(DEPLOY_MANIFEST.read_text()).get("nodes", {}) if DEPLOY_MANIFEST.is_file() else {}
    admin = {n for n, c in nodes.items() if isinstance(c, dict) and c.get("admin") is True}
    return {n for n, m in machines.items() if isinstance(m, dict) and (m.get("auditor") or n in admin)}


def resolve_machine(name: str, project: str | None, machines: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    ms = load_machines() if machines is None else machines
    m = ms.get(name)
    if m is None:
        from .pods import registry

        if why := registry.problems.get(name):
            raise Refused(f"machine {name!r} refused: {why}")
        d = registry.registry_dir(machines_path())
        where = f"{machines_path()}" + (f" or {d}" if d else "")
        raise Refused(f"machine {name!r} is not in {where}; only listed machines may be used (research pods register)")
    if not project:
        raise Refused("--project is required for a remote run (the machine is bound to a project)")
    allowed = m.get("projects") or ([m["project"]] if m.get("project") else [])
    if project not in allowed:
        raise Refused(f"machine {name!r} belongs to project(s) {allowed}, not {project!r}; refusing to touch it")
    out = dict(m)
    out.setdefault("root", DEFAULT_ROOT)
    out.setdefault("python", "python3")
    out["name"] = name
    return out


def guard_idle_min(machine: dict[str, Any]) -> float | None:
    g = machine.get("guard")
    if g is None or g is False:
        return None
    if g is True:
        return float(GUARD_IDLE_MIN_DEFAULT)
    if isinstance(g, (int, float)) and g > 0:
        return float(g)
    raise Refused(f"machine {machine.get('name')!r}: guard = {g!r} is neither true/false nor a positive number of idle minutes")


def ssh_host_target(machine: dict[str, Any]) -> dict[str, Any]:
    who = f"machine {machine.get('name') or '?'}"
    host, port, user = machine.get("host"), machine.get("port", 22), machine.get("user", "root")
    if not isinstance(host, str) or not host:
        raise Refused(f"{who}: provider ssh needs host = \"...\"")
    if not isinstance(port, int) or not 0 < port < 65536:
        raise Refused(f"{who}: port {port!r} is not a TCP port")
    if not isinstance(user, str) or not user:
        raise Refused(f"{who}: user {user!r} is not a user name")
    return {"host": host, "port": port, "user": user, "pod_name": machine.get("name"), "cost_per_hr": machine.get("rate_usd_per_hour"),
            "rate_observed_utc": utc_now_ms(), "gpu": machine.get("gpu")}


def pod_target(machine: dict[str, Any]) -> dict[str, Any]:
    if machine.get("provider") == "ssh":
        return ssh_host_target(machine)
    if machine.get("provider") != "runpod":
        raise Refused(f"provider {machine.get('provider')!r} not supported (runpod or ssh)")
    from .pods import runpod

    pod_id = machine["pod_id"]
    who = f"pod {pod_id} (machine {machine.get('name') or '?'})"
    kept, age_s = runpod.kept_address(pod_id) or (None, 0.0)
    asked = kept is not None and age_s < runpod.ADDRESS_TTL_S
    if asked and runpod.kept_stands(pod_id, kept):
        return kept
    try:
        pod = runpod.get(pod_id)
    except runpod.PodError as e:
        if e.code == 404:
            runpod.forget_address(pod_id)
            raise Refused(f"{who} no longer exists on runpod (HTTP 404: terminated?); no implicit provisioning") from None
        if kept and not asked and runpod.kept_stands(pod_id, kept):
            if pod_id not in _TOLD_KEPT:
                _TOLD_KEPT.add(pod_id)
                print(f"research: {who}: runpod API unavailable ({str(e)[:200]}); using its address as read "
                      f"{kept.get('rate_observed_utc') or '?'}, where the host holds the key recorded for it", file=sys.stderr)
            return kept
        raise Refused(f"{who}: runpod API unavailable: {e}") from None
    if not pod or (pod.get("desiredStatus") or "").upper() != "RUNNING":
        runpod.forget_address(pod_id)
        raise Refused(f"{who} is {pod.get('desiredStatus') if pod else 'not found'}; no implicit provisioning")
    tgt = runpod.ssh_target(pod)
    for _ in range(POD_TARGET_RETRIES):
        if tgt is not None:
            break
        time.sleep(POD_TARGET_RETRY_S)
        pod = runpod.get(machine["pod_id"]) or pod
        tgt = runpod.ssh_target(pod)
    if tgt is None:
        raise Refused(f"{who} has no ssh port mapped yet")
    target = {"host": tgt[0], "port": tgt[1], "user": "root", "pod_name": pod.get("name"), "cost_per_hr": pod.get("costPerHr"),
              "rate_observed_utc": utc_now_ms(), "gpu": (pod.get("machine") or {}).get("gpuTypeId")}
    runpod.keep_address(pod_id, target)
    return target


def ssh_argv(host: str, port: int, *, user: str = "root", connect_timeout: int = 15) -> list[str]:
    from .pods import runpod

    base = runpod.ssh_command(host, port, user=user)
    return [*base[:-1], "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}", base[-1]]


def sender(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    if cid := env.get("CURSOR_CONVERSATION_ID", "").strip():
        return f"https://cursor.com/agents/{cid}"
    return env.get("RESEARCH_AGENT_URL") or None


WAKE_WAIT_S, WAKE_POLL_S = 900.0, 15.0
NEBIUS_KEY_ENV = ("NEBIUS_SA_ID", "NEBIUS_SA_PUBLIC_KEY_ID", "NEBIUS_SA_PRIVATE_KEY_B64")
TRUST_UPGRADE = ("ls /etc/ssl/certs | grep -q -E 'emSign_ECC_Root_CA_-_G3|Comodo_AAA_Services_root' || exit 0; "
                 "sudo -n apt-get -o DPkg::Lock::Timeout=120 update -qq && sudo -n env DEBIAN_FRONTEND=noninteractive "
                 "NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=120 install -y -qq --only-upgrade ca-certificates 'libnss3*'")
CONTROL_WAKE = "PYTHONPATH=/data/infra/store-index-src/src python3.12 -m research.pods.nebius.wake start"


def wake_on_control(iid: str, *, run=None) -> str | None:
    from .pods import budgets, door

    host = budgets.guard_host()
    try:
        r = door.ask(["wake", iid], f"{CONTROL_WAKE} {shlex.quote(iid)}", host=host, timeout=180, run=run)
    except Exception as e:  # noqa: BLE001  (no RunPod key or API, a timeout: the pod can't be asked)
        raise Refused(f"the control pod {host} can't be reached ({type(e).__name__}: {str(e)[:200]})") from None
    if r.returncode != 0:
        why = (r.stderr or r.stdout).strip().splitlines()
        raise Refused(f"the control pod {host} refused: {why[-1][:400] if why else f'rc {r.returncode}'}{door.hint(host, r.stderr)}")
    return json.loads(r.stdout.strip().splitlines()[-1]).get("was")


def wake(machine: dict[str, Any], ssh: list[str], *, nebius=None, control=None, sleep=time.sleep, clock=time.monotonic) -> str | None:
    iid = machine.get("nebius_instance_id")
    if not iid or ssh_run(ssh, ["true"], timeout=30).returncode == 0:
        return None
    if nebius is None and control is None and not all(os.environ.get(k) for k in NEBIUS_KEY_ENV):
        control = wake_on_control
    if control is not None:
        try:
            state = control(iid)
        except Refused as e:
            raise Refused(f"{machine['name']} doesn't answer on ssh, and with no Nebius key here it was to be started by the control "
                          f"pod: {e}") from None
        how = "the control pod started it"
    else:
        if nebius is None:
            from .pods.nebius.fleet import nb as nebius
        try:
            state = (nebius("compute", "instance", "get", "--id", iid).get("status") or {}).get("state")
            if state in ("STOPPED", "STOPPING"):
                try:
                    nebius("compute", "instance", "start", "--id", iid, "--async")
                except ValueError:
                    pass
        except (RuntimeError, OSError, ValueError) as e:
            raise Refused(f"{machine['name']} doesn't answer on ssh and couldn't be started through the Nebius API ({e}); with no "
                          f"Nebius credentials here, ask @infra to start {iid}") from None
        how = "started it through the Nebius API"
    print(f"research: {machine['name']} was {state}; {how}, waiting up to {WAKE_WAIT_S / 60:g} min for its ssh (disks kept)",
          file=sys.stderr)
    end = clock() + WAKE_WAIT_S
    while clock() < end:
        sleep(WAKE_POLL_S)
        if ssh_run(ssh, ["true"], timeout=30).returncode == 0:
            r = ssh_run(ssh, ["sh", "-c", TRUST_UPGRADE], timeout=600)
            if r.returncode != 0:
                print(f"research: {machine['name']}: its ca-certificates upgrade failed (exit {r.returncode}); "
                      f"`research deploy drift --on {machine['name']}` lists a distrusted root it still trusts", file=sys.stderr)
            return f"started from {state}"
    raise Refused(f"{machine['name']} ({iid}) was started from {state} but its ssh didn't answer in {WAKE_WAIT_S / 60:g} min")


def ssh_run(ssh: list[str], remote_argv: list[str], *, stdin: bytes | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    cmd = [*ssh, " ".join(shlex.quote(a) for a in remote_argv)]
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)


def ssh_stream(ssh: list[str], remote_argv: list[str], feed: Callable[[IO[bytes]], None], *, timeout: float) -> subprocess.CompletedProcess:
    cmd = [*ssh, " ".join(shlex.quote(a) for a in remote_argv)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out: list[bytes] = []
    err: list[bytes] = []
    failed: list[BaseException] = []

    def pump() -> None:
        try:
            feed(proc.stdin)
        except (BrokenPipeError, ConnectionResetError, ValueError):
            pass
        except BaseException as e:  # noqa: BLE001
            failed.append(e)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=pump, daemon=True),
               threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True),
               threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)]
    for t in threads:
        t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        for t in threads:
            t.join(10)
        raise
    for t in threads:
        t.join()
    if failed:
        raise failed[0]
    return subprocess.CompletedProcess(cmd, proc.returncode, b"".join(out), b"".join(err))


def ssh_read(ssh: list[str], remote_argv: list[str], consume: Callable[[IO[bytes]], None], *, timeout: float) -> subprocess.CompletedProcess:
    cmd = [*ssh, " ".join(shlex.quote(a) for a in remote_argv)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        try:
            consume(proc.stdout)
            while proc.stdout.read(SHIP_CHUNK):
                pass
        except BaseException:
            proc.kill()
            raise
        finally:
            timer.cancel()
            proc.stdout.close()
            rc = proc.wait()
            err.seek(0)
            stderr = err.read()[-8000:]
    return subprocess.CompletedProcess(cmd, rc, b"", stderr)


def _node_module(var: str, default: str, name: str) -> Any | None:
    p = Path(os.environ.get(var) or default)
    if p.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, p)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def node_slot_only() -> Any:
    if (mod := _node_module("NODE_SLOT_ONLY_FILE", NODE_SLOT_ONLY_FILE, "vy_node_slot_only")) is not None:
        return mod
    from .pods.nebius import slot_only

    return slot_only


def node_tool_rules() -> Any:
    if (mod := _node_module("NODE_TOOL_RULES_FILE", NODE_TOOL_RULES_FILE, "vy_node_tool_rules")) is not None:
        return mod
    from .pods.nebius import tool_rules

    return tool_rules


def node_slot_test(command: list[str]) -> str:
    from .pods.nebius import slot_only

    q, copy = shlex.quote, slot_only.NODE_SLOT
    opts = dict.fromkeys(a for a in command[4:command.index("--", 4)] if a.startswith("--"))
    knows = "".join(f" && grep -qsF -e {q(o)} {q(copy)}" for o in opts)
    return (f"{{ test -f {q(copy)}{knows}; }} || exit 1; m=open; test -e {q(RUNS_VIA_SLOT_FILE)} && m=marked; r=none; "
            f"test -e {q(NODE_SLOT_ONLY_FILE)} && {{ grep -qsF -e {q(copy)} {q(NODE_SLOT_ONLY_FILE)} && r=names || r=lacks; }}; "
            f"echo $m $r")


def node_slot_unruled(answer: str, tool_pkg: Path | None) -> str | None:
    from .pods.nebius import slot_only

    def names(file: str, word: str) -> bool:
        if tool_pkg is None:
            return True
        try:
            return word in (tool_pkg / file).read_text()
        except (OSError, UnicodeDecodeError):
            return False

    marked, _, rule = answer.strip().partition(" ")
    if marked not in ("marked", "open") or rule not in ("names", "lacks", "none"):
        return f"answered {answer.strip()[:80]!r} when asked whether its own slot.py takes this run"
    if rule == "lacks":
        return "has a slot_only.py that predates its own slot.py, and would take this run for one outside a check slot"
    if marked == "marked" and not (rule == "names" and names("remote.py", "def node_slot_only")) \
            and not names("pods/nebius/slot_only.py", slot_only.NODE_SLOT):
        return ("starts a run only through a check slot, and the research package this run ships predates the machine's own "
                "slot.py: it would refuse this run")
    return None


def source_tool_package(source: Path | None) -> Path | None:
    if source is None:
        return None
    cand = Path(source).expanduser() / "tools" / "research" / "src" / "research"
    return cand if (cand / "__init__.py").is_file() else None


def tool_tarball(src: Path | None = None) -> tuple[bytes, str]:
    src = Path(src).resolve() if src is not None else Path(__file__).resolve().parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in sorted(src.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc",):
                continue
            ti = tf.gettarinfo(str(p), arcname=str(Path("research") / p.relative_to(src)))
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(p, "rb") as f:
                tf.addfile(ti, f)
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()[:16]


def tool_identity(pkg: Path | str | None = None) -> str:
    src = Path(pkg).resolve() if pkg is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(src.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        h.update(f"{p.relative_to(src).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()[:16]


def knows_version_check(pkg: Path | None) -> bool:
    if pkg is None:
        return True
    try:
        return "def tool_identity" in (Path(pkg) / "remote.py").read_text()
    except OSError:
        return False


def knows_custody(pkg: Path | None) -> bool:
    if pkg is None:
        return True
    return (Path(pkg) / "store" / "custody.py").is_file() and "CUSTODY_DIR" in (Path(pkg) / "remote.py").read_text()


def knows_queue(pkg: Path | None) -> bool:
    if pkg is None:
        return True
    try:
        return "QUEUE_WORKLOAD_WRAP" in (Path(pkg) / "remote.py").read_text() and (Path(pkg) / "disk.py").is_file()
    except OSError:
        return False


QUEUE_WRAP = "wrap"
QUEUE_WORKLOAD_WRAP = "workload_wrap"


NODE_DEADLINE = "/etc/research/deadline"


def live_stop(machine: str, project: str) -> tuple[int | None, str]:
    try:
        m = resolve_machine(machine, project)
        tgt = pod_target(m)
        r = ssh_run(ssh_argv(tgt["host"], tgt["port"], user=tgt.get("user", "root"), connect_timeout=10), ["cat", NODE_DEADLINE],
                    timeout=30)
    except (Refused, OSError, subprocess.TimeoutExpired) as e:
        return None, f"{type(e).__name__}: {e}"
    if r.returncode:
        return None, f"`cat {NODE_DEADLINE}` exited {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}"
    words = r.stdout.decode(errors="replace").split()
    return (int(words[0]) if words and words[0].isdigit() else 0), NODE_DEADLINE


def queue_placement(source: Path | None, run_id: str, spec: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    from . import launcher
    from .store import tools_registry

    tree = launcher.code(source)
    src = (Path(tree) / (tools_registry.resolve("infra/cluster", lambda p: (Path(tree) / p).is_dir()) or "infra/cluster")
           if tree is not None else None)
    if src is None or not (src / "src" / "cluster" / "submit.py").is_file():
        which = f"the trusted tree ${launcher.ENV}" if launcher.tree() else "the --source tree"
        raise Refused(f"--queue: {which} has no infra/cluster with `cluster submit` (the queue's description and client)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    description = src / "descriptions" / "nebius.toml"
    argv = [sys.executable, "-m", "cluster", "submit", str(description), "--run-id", run_id,
            "--now", str(int(time.time() if now is None else now))]
    model = src / "src" / "cluster" / "model.py"
    if model.is_file() and "def with_stop" in model.read_text():
        nodes = [spec["node"]] if spec.get("node") else sorted(tomllib.loads(description.read_text()).get("nodes", {}))
        for name in nodes:
            stop, why = live_stop(name, spec.get("project") or "")
            if stop is None:
                print(f"research run --queue: {name}'s live deadline is unread ({why}): placing there with the description's "
                      f"stops_at", file=sys.stderr)
            else:
                argv += ["--stop", f"{name}={stop}"]
    for k, v in spec.items():
        if v is True:
            argv.append(f"--{k}")
        elif v not in (None, False):
            argv += [f"--{k}", str(v)]
    r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60, cwd=src / "src" if launcher.tree() else None)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise Refused(f"--queue: `cluster submit` failed (exit {r.returncode}): {r.stderr.strip()[-400:]}") from None
    if "refused" in out:
        raise Refused(f"the queue refuses the job: {out['refused']}")
    return out


def knows_job_user(pkg: Path | None, namespaces: bool = False) -> bool:
    if pkg is None:
        return True
    try:
        text = (Path(pkg) / "remote.py").read_text()
        return (Path(pkg) / "jobuser.py").is_file() and '"job_user"' in text and (not namespaces or '"job_namespaces"' in text)
    except OSError:
        return False


def knows_gpus(pkg: Path | None) -> bool:
    if pkg is None:
        return True
    try:
        return "def gpu_lease_wrap" in (Path(pkg) / "remote.py").read_text()
    except OSError:
        return False


def knows_clone(pkg: Path | None) -> bool:
    return pkg is None or (Path(pkg) / "clone.py").is_file()


def knows_preflight(pkg: Path | None) -> bool:
    if pkg is None:
        return True
    try:
        return "preflight_tool" in (Path(pkg) / "remote.py").read_text()
    except OSError:
        return False


def knows_pss_bound(pkg: Path | None) -> bool:
    if pkg is None:
        return True
    try:
        return "pss_max_rss_gb" in (Path(pkg) / "telemetry" / "sample.py").read_text()
    except OSError:
        return False


def pss_bound_problem(harness_args: list[str], source: Path | None) -> str | None:
    asked = []
    every = flag_value(harness_args, "--pss-every")
    try:
        if every is not None and float(every) <= 0:
            asked.append(f"--pss-every {every}")
    except ValueError:
        pass
    if (bound := flag_value(harness_args, "--pss-max-rss-gb")) is not None:
        asked.append(f"--pss-max-rss-gb {bound}")
    if not asked or knows_pss_bound(pkg := source_tool_package(source)):
        return None
    return (f"{' and '.join(asked)}: the research package in {source} predates them ({pkg}/telemetry/sample.py has no "
            f"--pss-max-rss-gb), and it is what the machine runs: there --pss-every 0 means a PSS round (smaps_rollup of every process, "
            f"under its mmap lock) every round, about once a second, and --pss-max-rss-gb fails the run.  Rebase the --source tree onto "
            f"main, or drop the flag")


def check_tool(expected: str | None) -> dict[str, str]:
    here = str(Path(__file__).resolve().parent)
    got = tool_identity(here)
    if expected and got != expected:
        raise ResearchVersionMismatch(f"the research package running on this machine ({here}, identity {got}) is not the one the launcher "
                                      f"shipped (identity {expected}): another `research` shadows the snapshot on sys.path, or the snapshot "
                                      f"was altered; nothing started")
    return {"identity": got, "path": here}


def launch_request_failure(rc: int, stdout: bytes, stderr: bytes) -> Refused:
    err = stderr.decode(errors="replace")
    for line in reversed(stdout.decode(errors="replace").strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("refused"):
            cls = {"ResearchVersionMismatch": ResearchVersionMismatch, "PreflightFailed": PreflightFailed,
                   "RunnerFailed": RunnerFailed}.get(str(doc.get("fault")), Refused)
            return cls(str(doc["refused"]))
    if rc == 2 and "--expect-tool" in err:
        return ResearchVersionMismatch("the research package that answered `launch-request` on the machine predates the version check, so it "
                                       f"is not the snapshot the launcher shipped (another `research` shadows it on sys.path?): {err.strip()[-300:]}")
    return Refused(f"launch-request failed on the machine: {err[-400:]}")


def ship_tool(ssh: list[str], root: str, src: Path | None = None) -> str:
    data, sha = tool_tarball(src)
    dest = f"{root}/tool/{sha}"
    probe = ssh_run(ssh, ["test", "-f", f"{dest}/.complete"])
    if probe.returncode == 0:
        return dest
    r = ssh_run(ssh, ["sh", "-c", f"mkdir -p {shlex.quote(dest)} && tar -xf - -C {shlex.quote(dest)} && touch {shlex.quote(dest)}/.complete"], stdin=data)
    if r.returncode != 0:
        raise Refused(f"shipping the tool snapshot failed: {r.stderr.decode(errors='replace')[-400:]}")
    return dest


class _Hashing:
    def __init__(self, raw: IO[bytes]):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        b = self.raw.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


def _manifest_of(stream: IO[bytes]) -> dict[str, Any]:
    src = _Hashing(stream)
    files: dict[str, str] = {}
    total = 0
    with tarfile.open(fileobj=src, mode="r|") as tf:
        for m in tf:
            if not m.isreg():
                continue
            h = hashlib.sha256()
            f = tf.extractfile(m)
            while f is not None and (chunk := f.read(SHIP_CHUNK)):
                h.update(chunk)
                total += len(chunk)
            files[m.name.removeprefix("./")] = h.hexdigest()
    while src.read(SHIP_CHUNK):
        pass
    return {"archive_sha256": src.h.hexdigest(), "files": files, "bytes": total, "size": src.n}


def archive_manifest(tar_bytes: bytes) -> dict[str, Any]:
    return _manifest_of(io.BytesIO(tar_bytes))


GIT_ARCHIVE_LIMITS = ("-c", "core.deltaBaseCacheLimit=8m", "-c", "core.packedGitWindowSize=4m", "-c", "core.packedGitLimit=16m")


def _git_archive(git_dir: Path) -> subprocess.Popen:
    from . import launcher

    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(launcher.git(*GIT_ARCHIVE_LIMITS, "archive", "--format=tar", "HEAD"), cwd=git_dir, stdout=subprocess.PIPE,
                            stderr=err, env=launcher.git_env())
    proc.archive_stderr = err
    return proc


def _git_archive_done(proc: subprocess.Popen) -> str | None:
    proc.stdout.close()
    if proc.wait() == 0:
        return None
    proc.archive_stderr.seek(0)
    return f"rc {proc.returncode}: {proc.archive_stderr.read().decode(errors='replace')[-300:].strip()}"


def archive_manifest_git(git_dir: Path) -> dict[str, Any]:
    proc = _git_archive(git_dir)
    try:
        man = _manifest_of(proc.stdout)
    finally:
        err = _git_archive_done(proc)
    if err:
        raise Refused(f"git archive HEAD failed in {git_dir}: {err}")
    return man


def _archive_feed(git_dir: Path, sent: dict[str, Any]) -> Callable[[IO[bytes]], None]:
    def feed(pipe: IO[bytes]) -> None:
        proc = _git_archive(git_dir)
        h, n = hashlib.sha256(), 0
        try:
            while chunk := proc.stdout.read(SHIP_CHUNK):
                pipe.write(chunk)
                h.update(chunk)
                n += len(chunk)
            pipe.flush()
        except (BrokenPipeError, ConnectionResetError):
            sent["cut"] = True
            raise
        finally:
            sent.update(sha256=h.hexdigest(), bytes=n, git_error=_git_archive_done(proc))
    return feed


def ship_timeout(archive_bytes: int) -> float:
    return max(float(SHIP_SOURCE_TIMEOUT_S), archive_bytes / SHIP_MIN_BYTES_PER_S)


def _compressed(ssh: list[str]) -> list[str]:
    return [ssh[0], "-C", *ssh[1:]] if Path(ssh[0]).name == "ssh" else ssh


def _sha_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        h, _, name = line.partition("  ")
        if h and name:
            out[name.removeprefix("./")] = h.strip()
    return out


def _rm_rf(ssh: list[str], path: str) -> None:
    try:
        ssh_run(ssh, ["rm", "-rf", path], timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


def read_ready(path: Path) -> dict[str, Any]:
    p = Path(path) / READY_FILE
    try:
        doc = json.loads(p.read_text())
    except FileNotFoundError:
        legacy = (Path(path) / LEGACY_COMPLETE).exists()
        raise SourceNotReady(f"{path} has no {READY_FILE}" + (f" (only the legacy {LEGACY_COMPLETE} marker; ship it again to verify it)" if legacy
                                                              else " (never verified: a truncated or interrupted ship); ship it again")) from None
    except (OSError, ValueError) as e:
        raise SourceNotReady(f"{p} is unreadable: {e}") from None
    if not isinstance(doc, dict) or doc.get("schema") != READY_SCHEMA or not doc.get("identity"):
        raise SourceNotReady(f"{p} is not a {READY_SCHEMA} document")
    return doc


def require_ready_source(path: Path | str, identity: str) -> dict[str, Any]:
    doc = read_ready(Path(path))
    if doc["identity"] != identity:
        raise SourceNotReady(f"{path}/{READY_FILE} names identity {doc['identity']}, the run asked for {identity}; refusing to execute")
    if dirt := tracked_dirt(Path(path), identity):
        raise SourceNotReady(f"{path} is shared by every run of {identity}, and an earlier run edited its tracked files ({', '.join(dirt[:3])}); "
                             f"refusing to execute from it: relaunch (the launcher sets it aside and ships the commit afresh), or use --cwd clone")
    return doc


def _head_dirt(head: str, identity: str) -> list[str]:
    return [] if head == identity else [f"HEAD is at {head or 'no commit'}, not at {identity}"]


def tracked_dirt(path: Path, identity: str) -> list[str]:
    if not (Path(path) / ".git").exists():
        return []
    head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, capture_output=True, text=True)
    return _head_dirt(head, identity) + ([line[3:] for line in r.stdout.splitlines() if line.strip()] if r.returncode == 0 else [])


def _ready_doc(identity: str, tree: str | None, man: dict[str, Any], verified: str) -> dict[str, Any]:
    return {"schema": READY_SCHEMA, "identity": identity, "commit": identity, "tree": tree, "archive_sha256": man["archive_sha256"],
            "files": len(man["files"]), "bytes": man["bytes"], "shipped_at_utc": utc_now_ms(), "launcher": os.uname().nodename, "verified": verified}


def _probe_source(ssh: list[str], dest: str) -> tuple[str, dict[str, Any] | None]:
    q = shlex.quote(dest)
    r = ssh_run(ssh, ["sh", "-c", f"if [ -f {q}/{READY_FILE} ]; then cat {q}/{READY_FILE}; echo; echo {_DIRT_MARK}; "
                                  f"if [ -e {q}/.git ]; then echo \"HEAD $(git -C {q} rev-parse --verify -q HEAD 2>/dev/null)\"; "
                                  f"git -C {q} status --porcelain --untracked-files=no 2>/dev/null | head -5; fi; "
                                  f"elif [ -e {q} ]; then echo LEGACY; else echo ABSENT; fi"])
    out, _, dirt = r.stdout.decode(errors="replace").partition(_DIRT_MARK)
    out = out.strip()
    if r.returncode != 0:
        raise Refused(f"probing {dest} failed: {r.stderr.decode(errors='replace')[-400:]}")
    if out == "ABSENT":
        return "absent", None
    if out == "LEGACY":
        return "legacy", None
    try:
        doc = json.loads(out)
    except ValueError:
        raise SourceNotReady(f"{dest}/{READY_FILE} is not JSON; inspect the machine") from None
    lines = [line for line in dirt.splitlines() if line.strip()]
    if lines and (found := _head_dirt(lines[0][5:], doc.get("identity")) + [line[3:] for line in lines[1:]]):
        return "dirty", {**doc, "dirt": found}
    return "ready", doc


def _verify_legacy(ssh: list[str], dest: str, man: dict[str, Any]) -> list[str]:
    r = ssh_run(ssh, ["sh", "-c", f"cd {shlex.quote(dest)} && find . -type f ! -name {LEGACY_COMPLETE} -print0 | xargs -0 sha256sum"], timeout=SHIP_SOURCE_TIMEOUT_S)
    if r.returncode != 0:
        return [f"could not hash the existing tree: {r.stderr.decode(errors='replace')[-300:]}"]
    have = _sha_lines(r.stdout.decode(errors="replace"))
    want = man["files"]
    bad = [f"missing {p}" for p in sorted(set(want) - set(have))] + [f"extra {p}" for p in sorted(set(have) - set(want))]
    bad += [f"differs {p}" for p in sorted(set(want) & set(have)) if want[p] != have[p]]
    return bad


def _write_ready(ssh: list[str], dest: str, doc: dict[str, Any]) -> None:
    r = ssh_run(ssh, ["sh", "-c", f"cat > {shlex.quote(dest)}/{READY_FILE}"], stdin=(json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())
    if r.returncode != 0:
        raise SourceNotReady(f"could not write {dest}/{READY_FILE}: {r.stderr.decode(errors='replace')[-300:]}")


SHIP_ENV = "RESEARCH_SHIP"
POD_GIT_REPO = "git/verity.git"
_RC_NO_GIT = 69
NO_GIT = "the machine has no git"
NO_CHECKOUT = "no_checkout"
_RC_TREE = 76


def _stream_forced() -> bool:
    return os.environ.get(SHIP_ENV, "").strip().lower() == "stream"

NEEDS_GIT = ("tools/check/suites.py", "tools/check/check.py", "tools/verity/lean/lean_changed.py", "verity check")


def needs_git(command: list[str]) -> str | None:
    words = " ".join(command)
    return next((p for p in NEEDS_GIT if p in words), None)


def _ext_escape(arg: str) -> str:
    return arg.replace("%", "%%").replace(" ", "% ")


def git_push_url(ssh: list[str], repo: str) -> str:
    return "ext::" + " ".join(_ext_escape(a) for a in ssh) + " " + _ext_escape(f"git receive-pack {shlex.quote(repo)}")


def _stale_partials_sweep(root: str) -> str:
    return (f"for d in {shlex.quote(root)}/src/*.partial-*; do [ -d \"$d\" ] && [ -z \"$(find \"$d\" -mmin -{STALE_PARTIAL_MIN} -print 2>/dev/null "
            f"| head -1)\" ] && rm -rf \"$d\"; done; ")


def _git_ship_script(root: str, repo: str, partial: str, sha: str) -> str:
    q, g = shlex.quote(partial), shlex.quote(repo)
    return (_stale_partials_sweep(root) +
            f"trap 'rm -rf {q}' EXIT; trap 'exit 1' HUP INT TERM PIPE; set -e; "
            f"t=$(git --git-dir={g} rev-parse --verify -q {sha}^{{tree}}) || exit {_RC_TREE}; [ \"$t\" = \"$1\" ] || exit {_RC_TREE}; "
            f"git clone -q --shared --no-checkout {g} {q}; git -C {q} -c advice.detachedHead=false checkout -q --detach {sha}; "
            f"mkdir -p {q}/.git/info; echo /{READY_FILE} >> {q}/.git/info/exclude; "
            f"git --git-dir={g} archive --format=tar {sha} > {q}/.git/ship.tar; sha256sum {q}/.git/ship.tar | cut -d' ' -f1; "
            f"rm -f {q}/.git/ship.tar; trap - EXIT HUP INT TERM PIPE; "
            f"cd {q} && find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum")


def _push_main(ssh: list[str], repo: str, git_dir: Path, size: int) -> None:
    from . import launcher

    main = subprocess.run(launcher.git("rev-parse", "--verify", "-q", "refs/remotes/origin/main^{commit}"), cwd=git_dir,
                          capture_output=True, text=True, env=launcher.git_env()).stdout.strip()
    if not main:
        return
    try:
        subprocess.run(launcher.git("-c", "protocol.ext.allow=always", "push", "-q", git_push_url(ssh, repo), f"{main}:refs/heads/main"),
                       cwd=git_dir, capture_output=True, text=True, timeout=ship_timeout(size), env=launcher.git_env())
    except subprocess.TimeoutExpired as e:
        print(f"research: the push of origin/main to the machine timed out after {e.timeout:.0f} s, so its main may be older", file=sys.stderr, flush=True)


def _push_commit(ssh: list[str], repo: str, git_dir: Path, sha: str, size: int) -> str | None:
    r = ssh_run(ssh, ["sh", "-c", f"command -v git >/dev/null 2>&1 || exit {_RC_NO_GIT}; mkdir -p {shlex.quote(str(Path(repo).parent))} && "
                                  f"{{ [ -d {shlex.quote(repo)} ] || git init -q --bare {shlex.quote(repo)}; }}"], timeout=120)
    if r.returncode == _RC_NO_GIT:
        return NO_GIT
    if r.returncode != 0:
        return f"could not create {repo}: {r.stderr.decode(errors='replace')[-200:].strip()}"
    from . import launcher

    try:
        push = subprocess.run(launcher.git("-c", "protocol.ext.allow=always", "push", "-q", git_push_url(ssh, repo),
                                           f"{sha}:refs/research/src/{sha}"),
                              cwd=git_dir, capture_output=True, text=True, timeout=ship_timeout(2 * size), env=launcher.git_env())
    except subprocess.TimeoutExpired:
        return "git push timed out"
    if push.returncode != 0:
        return f"git push failed: {push.stderr.strip()[-200:]}"
    _push_main(ssh, repo, git_dir, size)
    return None


def _ship_git(ssh: list[str], root: str, git_dir: Path, sha: str, tree: str | None, man: dict[str, Any],
              dest: str) -> tuple[str | None, str | None, bool]:
    repo = f"{root}/{POD_GIT_REPO}"
    if not tree:
        return None, "no tree hash for HEAD", False
    if why := _push_commit(ssh, repo, git_dir, sha, man["size"]):
        return None, why, False
    partial = f"{dest}.partial-{secrets.token_hex(4)}"
    try:
        r = ssh_run(ssh, ["sh", "-c", _git_ship_script(root, repo, partial, sha), "sh", tree], timeout=ship_timeout(man["size"]))
    except subprocess.TimeoutExpired:
        _rm_rf(ssh, partial)
        return None, "extraction on the machine timed out", True
    if r.returncode != 0:
        _rm_rf(ssh, partial)
        why = "the pushed commit's tree is not the launcher's" if r.returncode == _RC_TREE else r.stderr.decode(errors="replace")[-200:].strip()
        return None, f"extraction failed (rc {r.returncode}): {why}", True
    lines = r.stdout.decode(errors="replace").splitlines()
    have = _sha_lines("\n".join(lines[1:]))
    want = man["files"]
    if have != want:
        _rm_rf(ssh, partial)
        bad = [f"missing {p}" for p in sorted(set(want) - set(have))] + [f"extra {p}" for p in sorted(set(have) - set(want))]
        bad += [f"differs {p}" for p in sorted(set(want) & set(have)) if want[p] != have[p]]
        return None, f"extracted tree does not match the archive manifest ({len(bad)}: {'; '.join(bad[:3])})", True
    man["pod_archive_sha256"] = lines[0].strip() if lines else None
    return partial, None, True


_RC_DIRT = 77


def _add_git(ssh: list[str], root: str, git_dir: Path, sha: str, tree: str | None, dest: str, doc: dict[str, Any]) -> str | None:
    repo = f"{root}/{POD_GIT_REPO}"
    if not tree:
        return "no tree hash for HEAD"
    if why := _push_commit(ssh, repo, git_dir, sha, int(doc.get("bytes") or 0)):
        return why
    nonce = secrets.token_hex(4)
    g, d, p = shlex.quote(repo), shlex.quote(dest), shlex.quote(f"{dest}.git-{nonce}")
    new = shlex.quote(f"{dest}/.git/{READY_FILE}.{nonce}")
    script = (f"trap 'rm -rf {p}' EXIT; set -e; "
              f"t=$(git --git-dir={g} rev-parse --verify -q {sha}^{{tree}}) || exit {_RC_TREE}; [ \"$t\" = \"$1\" ] || exit {_RC_TREE}; "
              f"if [ ! -e {d}/.git ]; then git clone -q --shared --no-checkout {g} {p}; git -C {p} update-ref --no-deref HEAD {sha}; "
              f"git -C {p} read-tree {sha}; mkdir -p {p}/.git/info; echo /{READY_FILE} >> {p}/.git/info/exclude; "
              f"mv -T {p}/.git {d}/.git || [ -d {d}/.git ]; fi; "
              f"[ \"$(git -C {d} rev-parse HEAD)\" = {sha} ] || exit {_RC_DIRT}; "
              f"[ -z \"$(git -C {d} status --porcelain --untracked-files=no)\" ] || exit {_RC_DIRT}; "
              f"cat > {new}; mv {new} {d}/{READY_FILE}")
    ready = {**doc, "checkout": True, "transport": f"{doc.get('transport') or 'unrecorded'}+git", "git_added_utc": utc_now_ms()}
    try:
        r = ssh_run(ssh, ["sh", "-c", script, "sh", tree], stdin=(json.dumps(ready, indent=1, sort_keys=True) + "\n").encode(),
                    timeout=ship_timeout(int(doc.get("bytes") or 0)))
    except subprocess.TimeoutExpired:
        return "adding git on the machine timed out"
    if r.returncode == _RC_DIRT:
        return None
    if r.returncode != 0:
        why = "the pushed commit's tree is not the launcher's" if r.returncode == _RC_TREE else r.stderr.decode(errors="replace")[-200:].strip()
        return f"adding git failed (rc {r.returncode}): {why}"
    return None


def _give_git(ssh: list[str], root: str, git_dir: Path, sha: str, tree: str | None, dest: str, doc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if why := _add_git(ssh, root, git_dir, sha, tree, dest, doc):
        print(f"research: could not make {dest} a checkout ({why})", file=sys.stderr)
        return "ready", {**doc, NO_CHECKOUT: why}
    return _probe_source(ssh, dest)


def _mark_ready(ssh: list[str], partial: str, dest: str, sha: str, doc: dict[str, Any]) -> dict[str, Any]:
    q = shlex.quote(partial)
    _write_ready(ssh, partial, doc)
    r = ssh_run(ssh, ["sh", "-c", f"if [ -e {shlex.quote(dest)} ]; then echo EXISTS; else mv {q} {shlex.quote(dest)} && echo MOVED; fi"])
    outcome = r.stdout.decode(errors="replace").strip()
    if r.returncode != 0 or outcome not in ("MOVED", "EXISTS"):
        _rm_rf(ssh, partial)
        raise SourceNotReady(f"could not move {partial} into place: {r.stderr.decode(errors='replace')[-300:]}")
    if outcome == "EXISTS":
        _rm_rf(ssh, partial)
    state, have = _probe_source(ssh, dest)
    if state != "ready" or have.get("identity") != sha:
        raise SourceNotReady(f"{dest} appeared without a READY.json for {sha} (another writer?); refusing to execute from it")
    return have


def _quarantine(ssh: list[str], root: str, dest: str, sha: str, why: str) -> None:
    quarantine = f"{root}/src/.quarantine/{sha}-{secrets.token_hex(4)}"
    r = ssh_run(ssh, ["sh", "-c", f"mkdir -p {shlex.quote(root)}/src/.quarantine && mv {shlex.quote(dest)} {shlex.quote(quarantine)}"])
    if r.returncode != 0:
        raise SourceNotReady(f"{dest} is {why} and could not be moved aside: {r.stderr.decode(errors='replace')[-300:]}")
    print(f"research: {dest} is {why}; moved to {quarantine}, shipping afresh", file=sys.stderr)


def ship_source(ssh: list[str], root: str, git_dir: Path) -> tuple[str, str, dict[str, Any]]:
    from . import launcher

    launcher.tree(git_dir)
    env = launcher.git_env()
    st = subprocess.run(launcher.git("status", "--porcelain", "--untracked-files=all"), cwd=git_dir, capture_output=True, text=True, env=env)
    if st.returncode != 0:
        raise Refused(f"{git_dir} is not a git checkout")
    if st.stdout.strip():
        raise Refused(f"{git_dir} has uncommitted changes; commit them (an unpushed commit is fine) or the snapshot would be mislabeled as HEAD")
    sha = subprocess.run(launcher.git("rev-parse", "HEAD"), cwd=git_dir, capture_output=True, text=True, check=True, env=env).stdout.strip()
    tree = subprocess.run(launcher.git("rev-parse", "HEAD^{tree}"), cwd=git_dir, capture_output=True, text=True, env=env).stdout.strip() or None
    dest = f"{root}/src/{sha}"
    state, doc = _probe_source(ssh, dest)
    push = not _stream_forced()
    if state == "ready" and not doc.get("checkout") and doc.get("identity") == sha and push:
        state, doc = _give_git(ssh, root, git_dir, sha, tree, dest, doc)
    if state == "ready":
        if doc.get("identity") != sha or doc.get("schema") != READY_SCHEMA:
            raise SourceNotReady(f"{dest}/{READY_FILE} names identity {doc.get('identity')!r}, not {sha}; inspect the machine before shipping over it")
        return sha, dest, doc

    def checked_out(have: dict[str, Any], push: bool, unpushed: str | None = None) -> tuple[str, str, dict[str, Any]]:
        if have.get("checkout") or not push:
            return sha, dest, {**have, NO_CHECKOUT: unpushed} if unpushed and not have.get("checkout") else have
        state, have = _give_git(ssh, root, git_dir, sha, tree, dest, have)
        if state != "ready":
            raise SourceNotReady(f"{dest}'s tracked files are not {sha}'s ({', '.join(have.get('dirt', [])[:3])}): a run edited the "
                                 f"shared tree; relaunch, and the ship sets it aside and ships the commit afresh")
        return sha, dest, have

    man = archive_manifest_git(git_dir)
    if state == "dirty":
        if doc.get("identity") != sha or doc.get("schema") != READY_SCHEMA:
            raise SourceNotReady(f"{dest}/{READY_FILE} names identity {doc.get('identity')!r}, not {sha}; inspect the machine before shipping over it")
        _quarantine(ssh, root, dest, sha, f"a READY checkout whose tracked files an earlier run edited ({', '.join(doc['dirt'][:3])})")
    if state == "legacy":
        bad = _verify_legacy(ssh, dest, man)
        if not bad:
            doc = _ready_doc(sha, tree, man, "legacy-sha256-per-file")
            _write_ready(ssh, dest, doc)
            return checked_out(doc, push)
        _quarantine(ssh, root, dest, sha, f"an unverified tree that did not match the archive ({len(bad)} file(s): {'; '.join(bad[:3])})")
    unpushed = None
    if push:
        gpartial, why, push = _ship_git(ssh, root, git_dir, sha, tree, man, dest)
        if gpartial is not None:
            doc = {**_ready_doc(sha, tree, man, "git-push+tree+file-sha256"), "transport": "git", "checkout": True,
                   "pod_archive_sha256": man.get("pod_archive_sha256")}
            return checked_out(_mark_ready(ssh, gpartial, dest, sha, doc), push)
        print(f"research: git ship to {dest} not used ({why}); shipping the archive stream", file=sys.stderr)
        unpushed = why
    partial = f"{dest}.partial-{secrets.token_hex(4)}"
    q = shlex.quote(partial)
    script = (_stale_partials_sweep(root) +
              f"trap 'rm -rf {q}' EXIT; trap 'exit 1' HUP INT TERM PIPE; set -e; mkdir -p {q}; cat > {q}/.archive.tar; "
              f"got=$({{ sha256sum {q}/.archive.tar 2>/dev/null || shasum -a 256 {q}/.archive.tar; }} | cut -d' ' -f1); echo \"$got\"; "
              f"[ \"$got\" = \"$1\" ] || exit {_RC_DIGEST}; "
              f"tar -xmf {q}/.archive.tar -C {q}; rm -f {q}/.archive.tar; trap - EXIT HUP INT TERM PIPE; cd {q} && find . -type f | LC_ALL=C sort")
    sent: dict[str, Any] = {}
    timeout = ship_timeout(man["size"])
    try:
        r = ssh_stream(_compressed(ssh), ["sh", "-c", script, "sh", man["archive_sha256"]], _archive_feed(git_dir, sent), timeout=timeout)
    except subprocess.TimeoutExpired:
        _rm_rf(ssh, partial)
        raise SourceNotReady(f"shipping the source snapshot to {dest} exceeded {timeout:.0f}s ({man['size']} bytes at >= {SHIP_MIN_BYTES_PER_S // 1024} KiB/s); "
                             f"the partial directory was removed, nothing is READY") from None
    lines = r.stdout.decode(errors="replace").splitlines()
    got_sha = lines[0].strip() if lines else ""
    if r.returncode == _RC_DIGEST or (r.returncode == 0 and got_sha != man["archive_sha256"]):
        _rm_rf(ssh, partial)
        why = "truncated transfer?"
        if sent.get("git_error") and not sent.get("cut"):
            why = f"git archive failed while shipping ({sent['git_error']})"
        elif sent.get("sha256") and sent["sha256"] != man["archive_sha256"] and sent.get("bytes") == man["size"]:
            why = f"git archive HEAD is not reproducible here (the second pass hashed {sent['sha256']})"
        raise SourceNotReady(f"source archive sha256 mismatch: the machine received {got_sha or '<nothing>'}, the launcher sent {man['archive_sha256']} "
                             f"({sent.get('bytes', 0)} of {man['size']} bytes written); {why} nothing was extracted, nothing is READY")
    if r.returncode != 0:
        _rm_rf(ssh, partial)
        raise SourceNotReady(f"shipping the source snapshot failed (rc {r.returncode}): {r.stderr.decode(errors='replace')[-400:]}; nothing is READY")
    got_files = {ln.strip().removeprefix("./") for ln in lines[1:] if ln.strip()}
    want_files = set(man["files"])
    if got_files != want_files:
        _rm_rf(ssh, partial)
        missing, extra = sorted(want_files - got_files), sorted(got_files - want_files)
        raise SourceNotReady(f"source tree manifest mismatch after extraction: {len(missing)} missing, {len(extra)} extra "
                             f"(e.g. {(missing or extra)[:3]}); nothing is READY")
    doc = {**_ready_doc(sha, tree, man, "archive-sha256+file-manifest"), "transport": "stream"}
    return checked_out(_mark_ready(ssh, partial, dest, sha, doc), push, unpushed)


def stage_request(ssh: list[str], root: str, run_id: str, request: dict[str, Any], sends: list[Path],
                  private: list[tuple[str, bytes]] | None = None) -> str:
    stage_dir = f"{root}/requests/{run_id}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        ti = tarfile.TarInfo("request.json")
        body = (json.dumps(request, indent=1) + "\n").encode()
        ti.size = len(body)
        tf.addfile(ti, io.BytesIO(body))
        for p in sends:
            if not p.is_file():
                raise Refused(f"--send {p}: not a file")
            tf.add(str(p), arcname=f"inputs/{p.name}")
        for name, data in private or []:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), 0o600
            tf.addfile(ti, io.BytesIO(data))
    q = shlex.quote(stage_dir)
    cmd = (f"mkdir -p {shlex.quote(f'{root}/requests')} && {{ mkdir {q} || {{ test -e {q} && exit {STAGE_TAKEN_RC}; exit 1; }}; }} && "
           f"tar -xf - -C {q}")
    if private:
        cmd = f"umask 077 && {cmd} && chmod 600 " + " ".join(shlex.quote(f"{stage_dir}/{n}") for n, _ in private)
    r = ssh_run(ssh, ["sh", "-c", cmd], stdin=buf.getvalue(), timeout=600)
    if r.returncode == STAGE_TAKEN_RC:
        raise RunIdTaken(f"run id {run_id} is another launch's on the machine ({stage_dir} is staged already), so nothing was staged "
                         "or started; launch again: a launch without --id picks a new id")
    if r.returncode != 0:
        raise Refused(f"staging the request failed: {r.stderr.decode(errors='replace')[-400:]}")
    return f"{stage_dir}/request.json"


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError) as e:
        print(f"research: {p} unread, read as none ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return None


def _json_line(line: str) -> dict[str, Any]:
    try:
        d = json.loads(line)
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def _holder_live(runs_root: Path, holder: str) -> bool:
    d = runs_root / holder
    st = _read_json(d / "status.json")
    if st and st.get("state") in TERMINAL:
        return False
    launch = _read_json(d / "launch.json") or {}
    if st is None and not _pid_alive(launch.get("runner_pid")):
        return False
    if st and st.get("state") not in TERMINAL and not _pid_alive(st.get("pid")) and not _pid_alive(launch.get("runner_pid")):
        return False
    return True


def take_exclusive(runs_root: Path, run_id: str) -> None:
    lock = runs_root / ".exclusive.lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock.read_text().strip() if lock.exists() else ""
            if holder and holder != run_id and _holder_live(runs_root, holder):
                raise Refused(f"machine is exclusively held by run {holder} (still live); not starting {run_id}")
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(run_id + "\n")
        return
    raise Refused("could not take the exclusive lock (contended twice)")


def exclusive_holder(runs_root: Path) -> str | None:
    lock = runs_root / ".exclusive.lock"
    try:
        holder = lock.read_text().strip()
    except FileNotFoundError:
        return None
    if holder and _holder_live(runs_root, holder):
        return holder
    try:
        lock.unlink()
    except FileNotFoundError:
        pass
    return None


def release_exclusive(runs_root: Path, run_id: str) -> None:
    lock = runs_root / ".exclusive.lock"
    try:
        if lock.read_text().strip() == run_id:
            lock.unlink()
    except FileNotFoundError:
        pass


def active_runs(runs_root: Path, exclude: str | None = None) -> list[str]:
    out = []
    if not runs_root.is_dir():
        return out
    for d in runs_root.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == exclude:
            continue
        st = _read_json(d / "status.json")
        launch = _read_json(d / "launch.json") or {}
        if st is None:
            if launch.get("runner_pid") and _pid_alive(launch["runner_pid"]) and not (d / "refused.json").exists():
                out.append(d.name)
        elif st.get("state") not in TERMINAL:
            out.append(d.name)
    return out


def _environ_of(path: str) -> dict[str, str]:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep:
            out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def _guard_pid(gdir: Path) -> int | None:
    if not (gdir / "guard.pid").exists():
        return None
    try:
        pid = int((gdir / "guard.pid").read_text().strip())
    except (OSError, ValueError) as e:
        print(f"research: {gdir / 'guard.pid'} unread, so no guard daemon is known ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return None
    if not _pid_alive(pid):
        return None
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        cmd = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True).stdout
    return pid if "pod_guard.sh daemon" in cmd else None


def ensure_guard(root: Path, idle_min: float, *, pod_id: str | None = None, environ: str | None = None) -> dict[str, Any]:
    environ = environ or GUARD_ENVIRON
    penv = _environ_of(environ)
    me = penv.get("RUNPOD_POD_ID")
    if not me:
        return {"installed": False, "reason": f"not a RunPod pod (no RUNPOD_POD_ID in {environ}): no guard"}
    if pod_id and me != pod_id:
        return {"installed": False, "reason": f"this machine is pod {me}, the request names pod {pod_id}: no guard"}
    if not penv.get("RUNPOD_API_KEY"):
        return {"installed": False, "reason": f"pod {me} has no pod-scoped RUNPOD_API_KEY in {environ}: guard NOT installed, the pod will "
                                              "not terminate itself (the account key is never copied to a pod)"}
    gdir = root / "guard"
    gdir.mkdir(parents=True, exist_ok=True)
    base = {"pod_id": me, "idle_min": idle_min, "state": str(gdir / "state.json"), "log": str(gdir / "guard.log")}
    pid = _guard_pid(gdir)
    if pid:
        return {"installed": True, "started": False, "pid": pid, **base}
    from .pods import workproc

    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
    env.update(workproc.guard_env(), GUARD_TERMINAL=" ".join(TERMINAL), GUARD_ENVIRON=environ)
    with open(gdir / "daemon.log", "ab") as log:
        proc = subprocess.Popen(["bash", str(GUARD_SCRIPT), "daemon", str(root), f"{idle_min:g}"], cwd=str(gdir), env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        pid = _guard_pid(gdir)
        if pid or proc.poll() is not None:
            break
        time.sleep(0.05)
    pid = pid or _guard_pid(gdir)
    if not pid:
        return {"installed": False, "reason": f"the guard daemon did not start (exit {proc.poll()}; see {gdir / 'daemon.log'})", **base}
    return {"installed": True, "started": pid == proc.pid, "pid": pid, **base}


def tool_preflight(harness_args: list[str], tree: Path | str | None, asked: str | None = None) -> list[str] | None:
    spec = asked or flag_value(harness_args, "--tool")
    flag = "--preflight" if asked else "--tool"
    if not spec or tree is None:
        if asked:
            raise PreflightFailed(f"--preflight {asked}: the run ships no tree to run it from; nothing started")
        return None
    from .store import tools_registry

    if tools_registry.locate(spec, Path(tree)) is None:
        if asked:
            raise PreflightFailed(f"--preflight {asked}: {tree} does not declare that tool; nothing started")
        return None
    try:
        decl = tools_registry.from_tree(spec, Path(tree))
    except Exception as e:  # noqa: BLE001
        raise PreflightFailed(f"{flag} {spec}: its declaration in {tree} cannot be read ({type(e).__name__}: {e}), so its preflight is "
                              f"unknown; nothing started") from None
    argv = list(getattr(decl, "preflight", None) or [])
    if not argv and asked:
        raise PreflightFailed(f"--preflight {asked}: its declaration in {tree} names no preflight; nothing started")
    return argv or None


def _tail(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()][-PREFLIGHT_TAIL:]


def run_preflight(argv: list[str], tree: Path | str, env: dict[str, str]) -> dict[str, Any]:
    cmd = [sys.executable, *argv]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, cwd=str(tree), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=PREFLIGHT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise PreflightFailed(f"preflight {shlex.join(argv)} did not finish in {PREFLIGHT_TIMEOUT_S} s; nothing started",
                              {"argv": argv, "rc": None, "seconds": PREFLIGHT_TIMEOUT_S}) from None
    rep = {"argv": argv, "rc": r.returncode, "seconds": round(time.monotonic() - t0, 1), "stdout": _tail(r.stdout), "stderr": _tail(r.stderr)}
    if r.returncode:
        why = "; ".join(rep["stderr"] or rep["stdout"][-5:]) or "no output"
        raise PreflightFailed(f"the machine failed the preflight {shlex.join(argv)} (exit {r.returncode}): {why}", rep)
    return rep


def launch_request(request_path: Path) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    with open(request_path.parent / LAUNCH_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = _launch_request(request_path)
    try:
        req = json.loads(request_path.read_text())
        g = req.get("guard")
    except (OSError, ValueError):
        req, g = {}, None
    if req.get("lease_extend_s") and out.get("started"):
        from .pods import lease

        try:
            out["lease"] = lease.machine_extend(float(req["lease_extend_s"]), f"run-{req.get('run_id')}")
        except Exception as e:  # noqa: BLE001  (the run is started; the answer says the lease could not be touched)
            out["lease"] = {"leased": None, "error": f"{type(e).__name__}: {e}"}
    if g:
        try:
            out["guard"] = ensure_guard(Path(req["runs_root"]).parent, float(g.get("idle_min") or GUARD_IDLE_MIN_DEFAULT),
                                        pod_id=(req.get("remote") or {}).get("pod_id"))
        except Exception as e:  # noqa: BLE001
            out["guard"] = {"installed": False, "reason": f"guard failed to start: {type(e).__name__}: {e}"}
    return out


def _launch_request(request_path: Path) -> dict[str, Any]:
    req = json.loads(request_path.read_text())
    if req.get("schema") != REQUEST_SCHEMA:
        raise Refused(f"unknown request schema {req.get('schema')!r}")
    run_id = req["run_id"]
    runs_root = Path(req["runs_root"])
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / run_id
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError:
        launch = _read_json(run_dir / "launch.json") or {}
        st = _read_json(run_dir / "status.json") or {}
        refused = _read_json(run_dir / "refused.json")
        return {"claimed": False, "run_id": run_id, "run_dir": str(run_dir), "existing": True, "state": st.get("state"),
                "runner_pid": launch.get("runner_pid"), "refused": (refused or {}).get("reason"), "fault": (refused or {}).get("fault"),
                "this_request": (request_path.parent / CLAIM_MARK).is_file()}
    (request_path.parent / CLAIM_MARK).write_text(f"{run_id}\n")
    concurrent = active_runs(runs_root, exclude=run_id)
    src_dir = None
    mem_rec: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    remote = dict(req.get("remote") or {})
    remote["request"] = str(request_path)
    try:
        if (marker := Path(os.environ.get("RUNS_VIA_SLOT_FILE") or RUNS_VIA_SLOT_FILE)).exists():
            slot_only = node_slot_only()
            if not slot_only.through_slot(req["command"]):
                raise Refused(slot_only.refusal(marker))
        if (marker := Path(os.environ.get("AUDITOR_FILE") or AUDITOR_FILE)).exists():
            rule = node_slot_only()
            if not hasattr(rule, "auditor_allows"):
                from .pods.nebius import slot_only as rule
            if not rule.auditor_allows(req["command"], marker):
                raise Refused(rule.auditor_refusal(marker))
        if why := node_tool_rules().refusal(Path(__file__).resolve().parent):
            raise Refused(why)
        if remote.get("source_sha"):
            src_dir = remote.get("source_dir") or str(runs_root.parent / "src" / str(remote["source_sha"]))
            require_ready_source(src_dir, str(remote["source_sha"]))
        env = _runner_env(req, runs_root, src_dir, remote)
        if req.get("custody") and not (request_path.parent / CUSTODY_DIR / "cred.json").is_file():
            raise Refused(f"--custody-r2: no staged key under {request_path.parent / CUSTODY_DIR}; the run could not publish its custody")
        if req.get("store_read"):
            from .store import custody as CU

            try:
                env.update(CU.reader_env(request_path.parent / STORE_READ_DIR))
            except (OSError, ValueError, KeyError):
                raise Refused(f"--store-read: no staged key under {request_path.parent / STORE_READ_DIR}") from None
        if req.get("job_user"):
            from . import jobuser

            try:
                jobuser.check_user(req["job_user"])
                jobuser.check_run_dir(run_dir, run_id)
                if req.get("job_namespaces"):
                    remote["job_namespaces"] = list(jobuser.check_namespaces(req["job_namespaces"]))
            except jobuser.Refused as e:
                raise Refused(str(e)) from None
            remote["job_user"] = req["job_user"]
        holder = exclusive_holder(runs_root)
        if holder and holder != run_id:
            raise Refused(f"machine is exclusively held by run {holder} (still live); not starting {run_id}")
        if pf_argv := tool_preflight(req.get("harness_args") or [], src_dir, req.get("preflight_tool")):
            pf_env = {**env, "RESEARCH_PREFLIGHT_INPUTS": str(request_path.parent / "inputs")}
            if req.get("custody"):
                pf_env["RESEARCH_PREFLIGHT_CUSTODY"] = str(request_path.parent / CUSTODY_DIR)
            remote["preflight"] = preflight = run_preflight(pf_argv, src_dir, pf_env)
        if req.get("mem_gb"):
            from . import mem as M

            mem_rec = M.admit(float(req["mem_gb"]), ramlock_dir=Path(req.get("ramlock_dir") or M.RAMLOCK_DIR), run_id=run_id, run_dir=run_dir,
                              wait_s=float(req.get("mem_wait_s") or 0), note=f"research run {run_id}: {shlex.join(req['command'])}")
        if req.get("exclusive"):
            take_exclusive(runs_root, run_id)
            if concurrent:
                release_exclusive(runs_root, run_id)
                raise Refused(f"--exclusive: other run(s) are live on this machine: {', '.join(concurrent)}; not starting {run_id}")
        ack = request_path.parent / LAUNCH_ACK
        try:
            log = open(run_dir / "launcher.log", "ab")
            ack.unlink(missing_ok=True)
            os.mkfifo(ack, 0o600)
            ack_fd = os.open(ack, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            raise CannotStart(f"could not prepare the launch ({type(e).__name__}: {e}); not starting {run_id}") from None
    except Refused as e:
        _release_launch(req, request_path, mem_rec)
        pf = {"preflight": e.report} if isinstance(e, PreflightFailed) and e.report else {}
        write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": str(e), "fault": type(e).__name__, "at_utc": utc_now_ms(), **pf})
        return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": str(e), "fault": type(e).__name__, **pf}
    staged_inputs = request_path.parent / "inputs"
    if staged_inputs.is_dir():
        os.replace(staged_inputs, run_dir / "inputs")
    cost = dict(req.get("cost") or {})
    cost["shared"] = bool(concurrent)
    cost["concurrent_runs_at_launch"] = concurrent
    if mem_rec:
        remote["mem"] = mem_rec
        env["RESEARCH_MEM_RESERVATION"] = mem_rec["reservation"]
        env["RESEARCH_RUN_MEM_GB"] = f"{float(req['mem_gb']):g}"
    cpus = direct_cpus()
    if cpus:
        remote["cpus"] = f"{format_cpus(cpus)}: {os.environ.get('DIRECT_CPUS_FILE') or DIRECT_CPUS_FILE} (a command may pin itself elsewhere)"
    argv = [req.get("python") or sys.executable, "-m", "research", "run", "--runs-dir", str(runs_root), "--id", run_id, "--claimed",
            "--cost-json", json.dumps(cost), "--remote-json", json.dumps(remote)]
    if req.get("exclusive"):
        argv += ["--exclusive-lock", str(runs_root / ".exclusive.lock")]
    if req.get("custody"):
        argv += ["--custody-dir", str(request_path.parent / CUSTODY_DIR)]
    if req.get("store_read"):
        argv += ["--store-read-dir", str(request_path.parent / STORE_READ_DIR)]
    if req.get("job_user"):
        argv += ["--job-user", req["job_user"]]
        if remote.get("job_namespaces"):
            argv += ["--job-namespaces", ",".join(remote["job_namespaces"])]
    argv += [*(req.get("harness_args") or []), "--", *(req.get(QUEUE_WORKLOAD_WRAP) or []), *gpu_lease_wrap(req), *req["command"]]
    if req.get(QUEUE_WRAP):
        argv = [*req[QUEUE_WRAP], *argv]
        if cores := (req.get("queue") or {}).get("cores"):
            argv = [req.get("python") or sys.executable, "-m", "research", "cpu-hold", "--cpus", str(cores["n"]), "--from",
                    cores["cpus"], "--wrap", str(len(req[QUEUE_WRAP])), "--who", run_id,
                    *(["--rank", str(cores["rank"])] if cores.get("rank") else []), "--ack",
                    str(request_path.parent / LAUNCH_ACK), "--", *argv]
    if (req.get("queue") or {}).get("disk_gb"):
        argv = [req.get("python") or sys.executable, "-m", "research", "disk-hold", "--gb", str(req["queue"]["disk_gb"]),
                "--path", str(runs_root), "--ack", str(request_path.parent / LAUNCH_ACK), "--", *argv]
    started = {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": True, **({"preflight": preflight} if preflight else {}),
               **({"gpus": remote["gpus"]} if remote.get("gpus") else {})}
    try:
        try:
            proc = subprocess.Popen(argv, cwd=req.get("cwd") or None, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
        except OSError as e:
            _release_launch(req, request_path, mem_rec)
            reason = f"{NOT_STARTED}: {type(e).__name__}: {e}"
            fault = (RunnerFailed if e.errno == errno.E2BIG else CannotStart).__name__
            write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": reason, "fault": fault, "at_utc": utc_now_ms()})
            return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": reason, "fault": fault}
        started["runner_pid"] = proc.pid
        try:
            if mem_rec:
                M.set_pid(mem_rec["reservation"], proc.pid)
            write_json_atomic(run_dir / "launch.json", {"run_id": run_id, "claimed_utc": utc_now_ms(), "runner_pid": proc.pid, "argv": argv,
                                                        "cwd": req.get("cwd"), "request": str(request_path), "hostname": os.uname().nodename})
            first = await_first_record(proc, ack_fd)
        except Exception as e:  # noqa: BLE001  (the runner is started: an error here leaves its launch unconfirmed, never refused)
            return {**started, "first_record": None, "unconfirmed": f"{type(e).__name__}: {e}"}
    finally:
        log.close()
        os.close(ack_fd)
        ack.unlink(missing_ok=True)
    if first is None:
        return _runner_failed(req, request_path, run_dir, proc, mem_rec, started)
    return {**started, "first_record": first}


def ack_launch(path: Path | str, what: str) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno not in (errno.ENOENT, errno.ENXIO):
            print(f"research: could not acknowledge the launch on {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return
    try:
        if stat.S_ISFIFO(os.fstat(fd).st_mode):
            os.write(fd, what.encode() + b"\n")
    except OSError:
        pass
    finally:
        os.close(fd)


def await_first_record(proc: subprocess.Popen, ack_fd: int) -> str | None:
    try:
        pid_fd = os.pidfd_open(proc.pid)
    except OSError:
        pid_fd, w = os.pipe()
        threading.Thread(target=lambda: (proc.wait(), os.close(w)), daemon=True, name=f"exit-{proc.pid}").start()
    watch = [ack_fd, pid_fd]
    try:
        while True:
            ready, _, _ = select.select(watch, [], [])
            try:
                got = os.read(ack_fd, 256)
            except BlockingIOError:
                got = None
            if got:
                return got.decode(errors="replace").split()[0]
            if pid_fd in ready:
                proc.wait()
                return None
            if got == b"":
                watch = [pid_fd]
    finally:
        os.close(pid_fd)


def _release_launch(req: dict[str, Any], request_path: Path, mem_rec: dict[str, Any] | None) -> None:
    if mem_rec:
        from .mem import release

        release(mem_rec["reservation"], req["run_id"])
    if req.get("exclusive"):
        release_exclusive(Path(req["runs_root"]), req["run_id"])
    if req.get("custody"):
        (request_path.parent / CUSTODY_DIR / "cred.json").unlink(missing_ok=True)
    from .store.custody import drop_credentials

    drop_credentials(request_path.parent)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _runner_failed(req: dict[str, Any], request_path: Path, run_dir: Path, proc: subprocess.Popen, mem_rec: dict[str, Any] | None,
                   started: dict[str, Any]) -> dict[str, Any]:
    run_id = req["run_id"]
    if (run_dir / FIRST_RECORD).exists():
        return {**started, "first_record": None, "rc": proc.returncode,
                "unconfirmed": f"no ack arrived, but the runner wrote {FIRST_RECORD} (exit {proc.returncode})"}
    if _group_alive(proc.pid):
        return {**started, "first_record": None, "rc": proc.returncode,
                "unconfirmed": f"pid {proc.pid} exited {proc.returncode} before {FIRST_RECORD}, but its process group lives on"}
    if own := _read_json(run_dir / "refused.json"):
        reason, fault = str(own.get("reason")), str(own.get("fault") or "Refused")
    else:
        try:
            tail = _tail((run_dir / "launcher.log").read_text(errors="replace")) if (run_dir / "launcher.log").exists() else []
        except OSError as e:
            print(f"research: {run_dir / 'launcher.log'} unread ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            tail = []
        reason = (f"the runner exited {proc.returncode} before writing {FIRST_RECORD}: "
                  + ("; ".join(tail[-RUNNER_FAILED_LINES:]) or "nothing in launcher.log"))
        fault = RunnerFailed.__name__
        write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": reason, "fault": fault, "rc": proc.returncode,
                                                     "log_tail": tail, "at_utc": utc_now_ms()})
    _release_launch(req, request_path, mem_rec)
    return {"claimed": True, "run_id": run_id, "run_dir": str(run_dir), "started": False, "refused": reason, "fault": fault,
            "runner_pid": proc.pid, "rc": proc.returncode}


def _research_bin(tool: str, runs_root: Path) -> Path:
    d = runs_root.parent / "cache" / "bin" / Path(tool).name
    text = f'#!/bin/sh\nPYTHONPATH={shlex.quote(tool)}${{PYTHONPATH:+:$PYTHONPATH}} exec {shlex.quote(sys.executable)} -m research "$@"\n'
    shim = d / "research"
    try:
        if shim.read_text() == text:
            return d
    except OSError:
        pass
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".research.{os.getpid()}"
    tmp.write_text(text)
    tmp.chmod(0o755)
    os.replace(tmp, shim)
    return d


def _runner_env(req: dict[str, Any], runs_root: Path, src_dir: str | None, remote: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (req.get("env") or {}).items()})
    if remote.get("source_sha"):
        env["RESEARCH_SOURCE_SHA"] = str(remote["source_sha"])
        env.setdefault("PYTHONPYCACHEPREFIX", str(runs_root.parent / "cache" / "pycache"))
        from .pythonpath import import_roots_dir, pythonpath

        derived = import_roots_dir(src_dir)
        env["PYTHONPATH"] = pythonpath(src_dir, derived, extra=env.get("PYTHONPATH"), shim_base=runs_root.parent / "cache" / "pyshim")
        remote["pythonpath"] = [p for p in env["PYTHONPATH"].split(":") if p]
        remote["pythonpath_derived"] = {"roots": derived["roots"], "missing": derived["missing"], "source": derived["source"],
                                        **({"shim": derived["shim"]} if derived["shim"] else {})}
    path = [p for p in (env.get("PATH") or os.defpath).split(os.pathsep) if p]
    if req.get("tool"):
        env["PYTHONPATH"] = req["tool"] + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["RESEARCH_TOOL_SNAPSHOT"] = Path(req["tool"]).name
        path.insert(0, str(_research_bin(req["tool"], runs_root)))
    if (elan := Path.home() / ".elan" / "bin").is_dir() and str(elan) not in path:
        path.append(str(elan))
    env["PATH"] = os.pathsep.join(path)
    gpus = req.get("gpus")
    if _kueue_owns_every_gpu():
        if gpus and not req.get(QUEUE_WRAP):
            raise Refused(f"--gpus {gpus}: every GPU here is Kueue's ({DIRECT_GPUS_FILE}); GPU work goes through `research run --queue --gpus N`")
        env["CUDA_VISIBLE_DEVICES"] = ""
        remote["gpus"] = f"none: {DIRECT_GPUS_FILE} says every GPU is Kueue's (CUDA_VISIBLE_DEVICES='')"
    elif _lease_owns_every_gpu() and not req.get(QUEUE_WRAP):
        env["CUDA_VISIBLE_DEVICES"] = ""
        remote["gpus"] = (f"{gpus} through `gpu-lease {gpus} --wait` (CUDA_VISIBLE_DEVICES names them)" if gpus else
                          f"none: gpu-lease hands out every GPU here ({GPU_LEASE_HOST_FILE}), so CUDA_VISIBLE_DEVICES=''; declare "
                          f"--gpus N to wait for N of them, or call gpu-lease in the workload")
    elif gpus and not req.get(QUEUE_WRAP):
        raise Refused(f"--gpus {gpus}: this machine's GPUs aren't handed out by gpu-lease (no {GPU_LEASE_HOST_FILE}); drop --gpus")
    return env


def gpu_lease_wrap(req: dict[str, Any]) -> list[str]:
    gpus = req.get("gpus")
    if not gpus or req.get(QUEUE_WRAP):
        return []
    return ["gpu-lease", str(int(gpus)), "--wait", "--mem-gb", str(int(req.get("mem_gb") or 0)), "--"]


def direct_cpus() -> set[int]:
    path = Path(os.environ.get("DIRECT_CPUS_FILE") or DIRECT_CPUS_FILE)
    if not path.exists():
        return set()
    try:
        text = path.read_text()
    except OSError as e:
        print(f"research: {path} unread, so a direct run names no CPUs ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return set()
    want: set[int] = set()
    for part in text.replace("\n", ",").split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            want |= set(range(int(lo), int(hi or lo) + 1))
    return want & os.sched_getaffinity(0)


def format_cpus(cpus: set[int]) -> str:
    out, run = [], []
    for c in sorted(cpus):
        if run and c != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = []
        run.append(c)
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _kueue_owns_every_gpu() -> bool:
    try:
        return Path(os.environ.get("GPU_LEASE_ALLOWED_FILE") or DIRECT_GPUS_FILE).read_text().strip() == "none"
    except OSError:
        return False


def _lease_owns_every_gpu() -> bool:
    return Path(os.environ.get("GPU_LEASE_HOST_FILE") or GPU_LEASE_HOST_FILE).exists()


def flag_value(args: list[str], flag: str) -> str | None:
    val = None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


def set_flag(args: list[str], flag: str, value: str) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a == flag:
            skip = True
        elif not a.startswith(flag + "="):
            out.append(a)
    return [*out, flag, value]


def shipped_identity(sha: str, ready: dict[str, Any]) -> dict[str, Any]:
    return {"commit": sha, "tree": ready.get("tree"), "dirty": False, "dirty_digest": None}


def lease_extend_s(harness_args: list[str]) -> float | None:
    from .pods.lease import RUN_MARGIN_S

    t = timeout_seconds(flag_value(harness_args, "--timeout"))
    return t + RUN_MARGIN_S if t and t > 0 else None


def timeout_seconds(text: str | None) -> float | None:
    from .store.credentials import parse_ttl

    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return float(parse_ttl(text))


def lease_line(lz: dict[str, Any] | None, asked: float | None, machine: str) -> str:
    if asked is None:
        return (f"{machine}'s lease was not extended: the run has no --timeout, so a pod lease (if any) may end it; "
                f"`research pods extend {machine} --hours H` extends it by hand")
    if lz is None:
        return f"{machine} answered without a lease report (its research package predates leases): the lease, if any, was not extended"
    if lz.get("leased") is False:
        return f"{machine} has no lease ({lz.get('reason')}): nothing ends it but its guards"
    if lz.get("extended"):
        return f"{machine}'s lease now ends no sooner than {lz.get('expires_utc')} (the run's --timeout plus its publish margin)"
    return f"{machine}'s lease could NOT be extended ({lz.get('error')}): it may end the run; `research pods extend {machine} --hours H`"


def workload_cwd(harness_args: list[str], tool_pkg: Path | None) -> str:
    asked = flag_value(harness_args, "--cwd")
    if asked is not None and clone.sub(asked) is not None:
        return f"--cwd {asked} (the run's own clone of the shipped tree, <root>/scratch/<run-id>/tree, removed when the workload exits)"
    if asked is not None:
        return f"--cwd {asked}"
    if tool_pkg is not None:
        try:
            knows = "WORKLOAD_CWD_DEFAULT" in (Path(tool_pkg) / "remote.py").read_text()
        except OSError as e:
            print(f"research: the tool's remote.py unread, so it counts as not knowing --cwd ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            knows = False
        if not knows:
            return "source (the shipped tree's research package predates the per-run workload cwd: it runs in <root>/src/<sha>/)"
    return WORKLOAD_CWD_DEFAULT

def launch(*, run_id: str, local_dir: Path, machine_name: str, project: str, command: list[str], harness_args: list[str],
           source: Path | None, sends: list[Path], cwd: str | None, env: dict[str, str], exclusive: bool,
           mem_gb: float | None = None, mem_wait_s: float = 0.0, custody_ttl: str | None = None,
           preflight_tool: str | None = None, queue: dict[str, Any] | None = None, job_user: str | None = None,
           gpus: int | None = None, store_read: bool = False, job_namespaces: tuple[str, ...] = (),
           auto: dict[str, Any] | None = None, mem_limit: dict[str, Any] | None = None, taken: bool = False) -> dict[str, Any]:
    local_dir.mkdir(parents=True, exist_ok=taken)
    rec: dict[str, Any] = {"schema": "research/remote/v0.1", "run_id": run_id, "machine": machine_name, "project": project, "phase": "resolving",
                           "created_utc": utc_now_ms(), "observed_utc": None, "last_observed": None, "command": command, "launch_args": harness_args,
                           **({"queue": queue} if queue else {}), **({"auto": auto} if auto else {})}
    write_json_atomic(local_dir / "remote.json", rec)
    try:
        m = resolve_machine(machine_name, project)
        if m.get("auditor"):
            raise Refused(f"machine {machine_name!r} is an auditor device: it takes no research run, from anyone. Its service is "
                          f"reached only through its gate: {m['auditor']}")
        guard_min = guard_idle_min(m)
        custody_pub: dict[str, Any] | None = None
        custody_files: list[tuple[str, bytes]] = []
        if custody_ttl:
            if not knows_custody(source_tool_package(source)):
                raise Refused(f"--custody-r2: the research package in {source} predates run custody (no research/store/custody.py); "
                              f"the machine would not publish it.  Commit a tree that has it, or pass --no-custody-r2")
            from .store import credentials as C
            from .store import custody as CU

            try:
                custody_pub, custody_files = CU.mint_for_machine(custody_ttl)
            except C.MintError as e:
                raise Refused(f"--custody-r2: cannot mint the machine's key here: {e} (the parent key: the Cursor secrets, or source "
                              f"~/.config/verity/r2.env; --no-custody-r2 launches without run custody)") from None
            rec["custody"] = custody_pub
        if store_read:
            from .store import credentials as C
            from .store import custody as CU

            if not custody_ttl:
                raise Refused("--store-read needs run custody (the default with --on): without it the runner's own publish would try "
                              "the workload's read-only key")
            if not hasattr(CU, "mint_reader") or "store_read" not in (Path(source_tool_package(source)) / "remote.py").read_text():
                raise Refused(f"--store-read: the research package in {source} predates it; the machine would not pass the key on")
            try:
                read_pub, read_files = CU.mint_reader(custody_ttl or CU.DEFAULT_TTL)
            except C.MintError as e:
                raise Refused(f"--store-read: cannot mint a read-only key here: {e} (the parent key: the Cursor secrets, or source "
                              f"~/.config/verity/r2.env)") from None
            custody_files = [*custody_files, *read_files]
            rec["store_read"] = read_pub
        tgt = pod_target(m)
        user = tgt.get("user", "root")
        ssh = ssh_argv(tgt["host"], tgt["port"], user=user)
        rec.update(pod_id=m.get("pod_id"), host=tgt["host"], port=tgt["port"], user=user, root=m["root"], phase="shipping")
        if woke := wake(m, ssh):
            rec["woke"] = woke
        write_json_atomic(local_dir / "remote.json", rec)
        tool_pkg = source_tool_package(source)
        if mem_gb and tool_pkg is not None and not (tool_pkg / "mem.py").exists():
            raise Refused(f"--mem-gb: the research package in {source} predates memory admission (no research/mem.py); the machine would "
                          f"ignore the reservation.  Commit a tree that has it, or drop --mem-gb")
        if mem_limit and tool_pkg is not None and not (tool_pkg / "memmark.py").exists():
            raise Refused(f"--mem-limit-gb: the research package in {source} predates the memory mark (no research/memmark.py); the "
                          f"run's record would not say what was stated.  Commit a tree that has it, or drop --mem-limit-gb")
        if preflight_tool and not knows_preflight(tool_pkg):
            raise Refused(f"--preflight: the research package in {source} predates it, so the machine would not run the preflight.  Commit "
                          f"a tree that has it, or drop --preflight")
        if queue and not knows_queue(tool_pkg):
            raise Refused(f"--queue: the research package in {source} predates it, so the machine would run the job outside its "
                          f"allocation.  Commit a tree that has it")
        if job_user and not knows_job_user(tool_pkg):
            raise Refused(f"--job-user: the research package in {source} predates it, so the machine would run the workload as its "
                          f"own user.  Commit a tree that has it")
        if job_namespaces and not knows_job_user(tool_pkg, namespaces=True):
            raise Refused(f"--job-namespaces: the research package in {source} predates it, so the machine's unit would allow the "
                          f"workload no namespace.  Commit a tree that has it")
        if gpus and not knows_gpus(tool_pkg):
            raise Refused(f"--gpus: the research package in {source} predates it, so the machine would run the workload outside the "
                          f"GPU lease.  Rebase onto a main that has it, or wrap the command in `gpu-lease {gpus} --wait --`")
        cwd_clone = clone.sub(flag_value(harness_args, "--cwd")) is not None
        if cwd_clone and not knows_clone(tool_pkg):
            raise Refused(f"--cwd clone: the research package in {source} predates it (no research/clone.py), so the machine would look "
                          f"for a directory named clone in the shared tree.  Commit a tree that has it, or pass --cwd source")
        from .pods.nebius import slot_only, tool_rules

        if tool_pkg is not None and (why := tool_rules.refusal(tool_pkg)):
            raise Refused(why)
        tool_dir = ship_tool(ssh, m["root"], tool_pkg)
        if (own := slot_only.node_copy(command)) is not None:
            asked = ssh_run(ssh, ["sh", "-c", node_slot_test(command)])
            why = (node_slot_unruled(asked.stdout.decode(errors="replace"), tool_pkg) if asked.returncode == 0 else
                   "has no slot.py of its own that takes this run" if asked.returncode == 1 else
                   f"didn't say whether its own slot.py takes this run (exit {asked.returncode})")
            if why is None:
                command = own
            else:
                print(f"research: {machine_name} {why} ({slot_only.NODE_SLOT}): the run waits in its pool's line by its tree's "
                      f"{slot_only.SLOT_PATHS[0]} (`research deploy drift --on {machine_name}` lists what the machine has)",
                      file=sys.stderr)
            rec.update(command=command, slot_copy="machine" if command is own else "tree")
        rec["tool_from"] = str(tool_pkg) if tool_pkg else "launcher"
        rec["tool_identity"] = tool_identity(tool_pkg)
        rec["tool_checked"] = knows_version_check(tool_pkg)
        if not rec["tool_checked"]:
            print(f"research: the machine runs {tool_pkg}'s research package (identity {rec['tool_identity']}; this launcher's is "
                  f"{tool_identity()}), which predates the version check: it is not verified on the machine", file=sys.stderr)
        rec["workload_cwd"] = workload_cwd(harness_args, tool_pkg)
        if rec["workload_cwd"].startswith("source (") and source is not None:
            print(f"research: {rec['workload_cwd']}; concurrent runs of this commit share it", file=sys.stderr)
        tree_sha = None
        src_dir = None
        ready: dict[str, Any] = {}
        remote_cwd = cwd
        if source is not None:
            tree_sha, src_dir, ready = ship_source(ssh, m["root"], source)
            no_git = CannotStart if ready.get(NO_CHECKOUT) == NO_GIT else Refused
            if not ready.get("checkout") and (prog := needs_git(command)):
                why = f"{SHIP_ENV}=stream" if _stream_forced() else "the line above says why"
                raise no_git(f"{src_dir} has no .git, as git couldn't reach {machine_name} ({why}), and {prog} needs a git checkout: "
                             f"the run would wait for its slot, then fail.  Relaunch once a git push reaches {machine_name} ({SHIP_ENV} "
                             f"unset, git installed there): the launch then makes the shared tree a checkout in place, which --cwd "
                             f"clone needs too")
            if cwd_clone and not ready.get("checkout"):
                raise no_git(f"--cwd clone: {src_dir} was shipped without git (transport {ready.get('transport') or 'unrecorded'}), so it "
                             f"has no .git to clone.  Pass --cwd source to run in it as it is, shared with other runs of the commit")
            remote_cwd = src_dir if cwd is None else str(Path(src_dir) / cwd)
            if "--tree-sha" not in harness_args:
                harness_args = [*harness_args, "--tree-sha", tree_sha]
            harness_args = set_flag(harness_args, "--source-identity", json.dumps(shipped_identity(tree_sha, ready), sort_keys=True))
        request = {"schema": REQUEST_SCHEMA, "run_id": run_id, "runs_root": f"{m['root']}/runs", "tool": tool_dir, "python": m["python"],
                   "cwd": remote_cwd, "env": env, "harness_args": harness_args, "command": command, "exclusive": exclusive,
                   "mem_gb": mem_gb, "mem_wait_s": mem_wait_s, "ramlock_dir": m.get("ramlock"),
                   **({"mem_limit": mem_limit} if mem_limit else {}),
                   "guard": {"idle_min": guard_min} if guard_min else None,
                   "lease_extend_s": lease_extend_s(harness_args),
                   **({"custody": custody_pub} if custody_pub else {}),
                   **({"store_read": rec["store_read"]} if rec.get("store_read") else {}),
                   **({"submitted_by": by} if (by := sender()) else {}),
                   **({"preflight_tool": preflight_tool} if preflight_tool else {}),
                   **({"job_user": job_user} if job_user else {}),
                   **({"job_namespaces": list(job_namespaces)} if job_user and job_namespaces else {}),
                   **({"gpus": gpus} if gpus else {}),
                   **({QUEUE_WRAP: queue["wrap"], QUEUE_WORKLOAD_WRAP: queue.get("workload_wrap") or [], "queue": queue} if queue else {}),
                   **({"auto": auto} if auto else {}),
                   "cost": {"provider": m.get("provider", "runpod"), "pod_id": m.get("pod_id"), "machine": machine_name, "currency": "USD",
                            "rate_usd_per_hour": tgt["cost_per_hr"], "rate_observed_utc": tgt["rate_observed_utc"], "reason": None},
                   "remote": {"machine": machine_name, "pod_id": m.get("pod_id"), "pod_name": tgt["pod_name"], "host": tgt["host"], "port": tgt["port"],
                              "root": m["root"], "tool": tool_dir, "tool_identity": rec["tool_identity"] if rec["tool_checked"] else None,
                              "source_sha": tree_sha, "source_dir": src_dir,
                              "source_archive_sha256": ready.get("archive_sha256"), "source_files": ready.get("files"),
                              "source_verified": ready.get("verified"), "source_transport": ready.get("transport"),
                              "launched_from": os.uname().nodename}}
        req_path = stage_request(ssh, m["root"], run_id, request, sends, private=custody_files or None)
        rec.update(phase="launching", request=req_path, remote_run_dir=f"{m['root']}/runs/{run_id}", tool=tool_dir, source_sha=tree_sha,
                   source_archive_sha256=ready.get("archive_sha256"), rate_usd_per_hour=tgt["cost_per_hr"])
        write_json_atomic(local_dir / "remote.json", rec)
        last_err = None
        expect = ["--expect-tool", rec["tool_identity"]] if rec["tool_checked"] else []
        deadline = launch_deadline(mem_wait_s)
        for attempt in range(LAUNCH_DELIVERIES):
            try:
                r = ssh_run(ssh, ["env", f"PYTHONPATH={tool_dir}", m["python"], "-m", "research", "launch-request", *expect, req_path],
                            timeout=deadline)
            except subprocess.TimeoutExpired:
                last_err = f"no answer from launch-request in {deadline:g} s"
                time.sleep(launch_pause(attempt))
                continue
            if r.returncode == 0:
                out = json.loads(r.stdout.decode().strip().splitlines()[-1])
                if out.get("existing") and not out.get("this_request", attempt > 0):
                    _rm_rf(ssh, str(Path(req_path).parent))
                    raise RunIdTaken(f"run id {run_id} is another launch's on {machine_name} (its run dir {rec['remote_run_dir']} was "
                                     f"not claimed from this launch's request), so nothing of this launch started and what it staged "
                                     f"is removed; launch again: a launch without --id picks a new id")
                break
            last_err = r.stderr.decode(errors="replace")[-400:]
            if r.returncode != 255:
                raise launch_request_failure(r.returncode, r.stdout, r.stderr)
            time.sleep(launch_pause(attempt))
        else:
            rec.update(phase="unknown", note=f"{LAUNCH_DELIVERIES} deliveries of the staged request failed; it may or may not have started: "
                                             f"{last_err}")
            write_json_atomic(local_dir / "remote.json", rec)
            raise Refused(rec["note"])
        rec.update(phase="refused" if out.get("refused") else "launched", launch=out, observed_utc=utc_now_ms())
        if pf := out.get("preflight"):
            rec["preflight"] = pf
            print(f"research: preflight {shlex.join(pf['argv'])} on {machine_name}: {'FAILED' if out.get('refused') else 'passed'} "
                  f"in {pf.get('seconds')} s", file=sys.stderr)
            for line in pf.get("stdout") or []:
                print(f"research: preflight: {line}", file=sys.stderr)
        if guard_min:
            g = out.get("guard") or {"installed": False, "reason": "the machine's research package predates the guard (no `guard` in its answer)"}
            rec["guard"] = g
            if g.get("installed"):
                print(f"research: guard on pod {g.get('pod_id')}: pid {g.get('pid')} ({'started' if g.get('started') else 'already running'}), "
                      f"terminates the pod after {g.get('idle_min'):g} idle min once every run is fetched (`research fetch --all`); "
                      f"state {g.get('state')}", file=sys.stderr)
            else:
                print(f"research: NO guard on {machine_name}: {g.get('reason')}", file=sys.stderr)
        if out.get("gpus"):
            print(f"research: GPUs on {machine_name}: {out['gpus']}", file=sys.stderr)
        if not out.get("refused"):
            rec["lease"] = lz = out.get("lease")
            print(f"research: {lease_line(lz, request['lease_extend_s'], machine_name)}", file=sys.stderr)
        write_json_atomic(local_dir / "remote.json", rec)
        return rec
    except Refused as e:
        rec.update(phase=rec["phase"] if rec.get("phase") in ("unknown",) else "refused", error=str(e))
        write_json_atomic(local_dir / "remote.json", rec)
        raise


ADOPT_CONNECT_S, ADOPT_PROBE_S, ADOPT_JOBS = 5, 20, 16


def adopt(runs_root: Path, run_id: str, *, on: str | None = None, machines: dict[str, dict[str, Any]] | None = None,
          unique: bool = False) -> Path:
    from .store import ids

    if not ids.is_run_id(run_id):
        raise Refused(f"{run_id!r} is not a full run id (rYYYYMMDD-HHMMSS-xxxx), so it is not looked up on the machines")
    local = runs_root / run_id
    if local.exists():
        raise Refused(f"{local} exists already")
    ms = load_machines() if machines is None else machines
    ssh = {n: m for n, m in ms.items() if isinstance(m, dict) and m.get("provider") == "ssh"}
    skipped = [] if on else sorted(restricted(ssh))
    names = [on] if on else sorted(n for n in ssh if n not in skipped)
    not_asked = (f"; not asked: the auditor devices {', '.join(skipped)}, which refuse lanes' keys (--on NAME asks one)"
                 if skipped else "")
    answers: dict[str, str] = {}
    targets = []
    for name in names:
        m = ms.get(name)
        if m is None:
            raise Refused(f"machine {name!r} is not registered")
        try:
            tgt = pod_target(m)
        except Refused as e:
            answers[name] = f"{name}: {e}"
            continue
        m = {**m, "root": m.get("root") or DEFAULT_ROOT}
        targets.append((name, m, tgt, f"{m['root']}/runs/{run_id}"))
    if len(targets) > 1 or skipped:
        print(f"research: looking for {run_id} on {len(targets)} machine{'s' if len(targets) != 1 else ''} (--on NAME skips this)"
              f"{not_asked}", file=sys.stderr, flush=True)

    def probe(t: tuple) -> str | None:
        name, _, tgt, rd = t
        ssh = ssh_argv(tgt["host"], tgt["port"], user=tgt.get("user", "root"), connect_timeout=ADOPT_CONNECT_S)
        try:
            r = ssh_run(ssh, ["test", "-d", rd], timeout=ADOPT_PROBE_S)
        except subprocess.TimeoutExpired:
            return f"{name}: no answer in {ADOPT_PROBE_S} s"
        if r.returncode != 0:
            return f"{name}: " + (f"no {rd}" if r.returncode == 1 else f"ssh rc {r.returncode}: {r.stderr.decode(errors='replace').strip()[-160:]}")
        return None

    held = []
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(ADOPT_JOBS, len(targets))))
    try:
        futures = {ex.submit(probe, t): t for t in targets}
        for f in concurrent.futures.as_completed(futures):
            t = futures[f]
            if (why := f.result()) is not None:
                answers[t[0]] = why
                continue
            held.append(t)
            if not unique:
                break
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    held.sort(key=lambda h: h[0])
    if len(held) > 1:
        raise Refused(f"{' and '.join(h[0] for h in held)} each hold a run {run_id}: name one with --on")
    if held:
        name, m, tgt, rd = held[0]
        local.mkdir(parents=True, exist_ok=False)
        write_json_atomic(local / "remote.json", {
            "schema": "research/remote/v0.1", "run_id": run_id, "machine": name, "project": m.get("project"), "phase": "adopted",
            "created_utc": utc_now_ms(), "observed_utc": None, "last_observed": None, "command": None, "launch_args": [],
            "pod_id": m.get("pod_id"), "host": tgt["host"], "port": tgt["port"], "user": tgt.get("user", "root"), "root": m["root"],
            "remote_run_dir": rd})
        fetch(local)
        return local
    why = "; ".join(answers[n] for n in sorted(answers)) or ("" if skipped else "no ssh machine is registered")
    raise Refused(f"no machine holds run {run_id}" + (f" ({why})" if why else "") + not_asked)


def _runner_gone(ssh: list[str], local_dir: Path, pid: int, *, timeout: float = 120) -> str | None:
    env = (_read_json(local_dir / "job.json") or {}).get("environment") or {}
    if not env.get("boot_id"):
        return None
    try:
        probe = ssh_run(ssh, ["sh", "-c", f"cat /proc/sys/kernel/random/boot_id; hostname; kill -0 {pid} 2>/dev/null && echo alive "
                                          "|| echo gone"], timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f"research: the machine's liveness probe timed out after {e.timeout:.0f} s, so it is not known", file=sys.stderr, flush=True)
        return None
    lines = probe.stdout.decode(errors="replace").split() if probe.returncode == 0 else []
    if len(lines) != 3:
        return None
    boot, host, alive = lines
    if boot != env["boot_id"]:
        return f"the machine rebooted since its runner started (boot {env['boot_id']}, now {boot})"
    if host == env.get("hostname") and alive == "gone":
        return f"its runner (pid {pid}) is gone without recording an end"
    return None


def _remote_status(ssh: list[str], rd: str, *, timeout: float = 120) -> dict[str, Any]:
    try:
        probe = ssh_run(ssh, ["sh", "-c", f"cat {shlex.quote(rd)}/status.json"], timeout=timeout)
        doc = json.loads(probe.stdout.decode(errors="replace")) if probe.returncode == 0 else {}
    except (subprocess.TimeoutExpired, ValueError) as e:
        return {"unreadable": f"{type(e).__name__}: {e}"}
    return doc if isinstance(doc, dict) else {}


def fetch(local_dir: Path, *, everything: bool = False, also: tuple[str, ...] = (), timeout: float | None = None) -> dict[str, Any]:
    def within(s: float) -> float:
        return s if timeout is None else max(1.0, min(s, timeout))

    if bad := [n for n in also if not n or n.startswith("/") or any(p in ("", "..") or p.startswith(".") for p in n.split("/"))]:
        raise Refused(f"--also takes paths of files under the run dir, not {', '.join(map(repr, bad))}")
    rec = _read_json(local_dir / "remote.json")
    if not rec or not rec.get("remote_run_dir"):
        raise Refused(f"{local_dir} is not a launched remote run")
    ssh = ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root")
    rd = rec["remote_run_dir"]
    if everything:
        remote_cmd = ["sh", "-c", f"cd {shlex.quote(rd)} || exit 2; tar -cf - .{TAR_CHANGED_OK}"]
    else:
        names = " ".join(shlex.quote(n) for n in dict.fromkeys((*SMALL_FILES, *also)))
        remote_cmd = ["sh", "-c", f"cd {shlex.quote(rd)} || exit 2; ls {names} 2>/dev/null | tar -cf - -T -{TAR_CHANGED_OK}"]

    def extract(stream: IO[bytes]) -> None:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                name = m.name.removeprefix("./")
                if m.name.startswith("/") or ".." in Path(m.name).parts or name in ("", ".", "remote.json", FETCHED_FILE, FETCHED_FILE + ".tmp"):
                    continue
                tf.extract(m, local_dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))

    try:
        r = ssh_read(ssh, remote_cmd, extract, timeout=within(900))
    except (tarfile.TarError, OSError) as e:
        r = subprocess.CompletedProcess(remote_cmd, -1, b"", f"reading the run's tar stream: {type(e).__name__}: {e}".encode())
    if r.returncode != 0:
        rec["last_fetch_error"] = {"at_utc": utc_now_ms(), "rc": r.returncode, "stderr": r.stderr.decode(errors="replace")[-300:]}
        write_json_atomic(local_dir / "remote.json", rec)
        return rec
    rec["observed_utc"] = utc_now_ms()
    st = _read_json(local_dir / "status.json") or {}
    waits = st.get("state") == "running" and (_read_json(local_dir / "slot.json") or {}).get("state") == "waiting"
    rec["last_observed"] = {"state": "waiting" if waits else st.get("state"), "rc": st.get("rc"), "at_utc": rec["observed_utc"]}
    rec.pop("last_fetch_error", None)
    rec.pop("launcher_exited", None)
    rec.pop("runner_gone", None)
    if st.get("state") not in TERMINAL and isinstance(st.get("pid"), int):
        if why := _runner_gone(ssh, local_dir, st["pid"], timeout=within(120)):
            again = _remote_status(ssh, rd, timeout=within(120))
            if again.get("state") in TERMINAL:
                write_json_atomic(local_dir / "status.json", again)
                rec["last_observed"] = {"state": again["state"], "rc": again.get("rc"), "at_utc": rec["observed_utc"]}
            else:
                rec["runner_gone"] = {"at_utc": rec["observed_utc"], "runner_pid": st["pid"], "why": why,
                                      **({"recheck": again["unreadable"]} if "unreadable" in again else {})}
    pid = (_read_json(local_dir / "launch.json") or {}).get("runner_pid")
    if not st and isinstance(pid, int):
        try:
            probe = ssh_run(ssh, ["sh", "-c", f"cd {shlex.quote(rd)} && if kill -0 {pid} 2>/dev/null; then echo alive; "
                                              f"else echo exited; tail -n 1 launcher.log 2>/dev/null; fi"], timeout=within(120))
        except subprocess.TimeoutExpired as e:
            print(f"research: the runner's probe timed out after {e.timeout:.0f} s, so its state is not known", file=sys.stderr, flush=True)
            probe = None
        lines = probe.stdout.decode(errors="replace").splitlines() if probe and probe.returncode == 0 else []
        if lines and lines[0] == "exited":
            rec["launcher_exited"] = {"at_utc": rec["observed_utc"], "runner_pid": pid, "last_line": (lines[1] if len(lines) > 1 else "")[-300:]}
    if everything:
        manifest = _verify_and_manifest(ssh, rd, local_dir, rec)
        rec["preserved"] = manifest is not None
        if manifest is not None:
            rec["fetched_marker"] = _mark_fetched(ssh, rd, local_dir, manifest)
    write_json_atomic(local_dir / "remote.json", rec)
    return rec


CHECK_DIR = Path(__file__).resolve().parents[3] / "check"
NODE_PROGRAM = '''
import json, sys, types


def load(name, src, file):
    m = types.ModuleType(name)
    m.__file__ = file
    sys.modules[name] = m
    exec(compile(src, file, "exec"), m.__dict__)
    return m


mem = load("research_scopemem", SRC["scopemem"], "research/scopemem.py")
mods = (load("check_slot_for_lean_slot", SRC["slot"], "tools/check/slot.py"),
        load("check_lean_slot", SRC["lean"], "tools/check/lean_slot.py")) if SRC.get("slot") and SRC.get("lean") else None
print(json.dumps(mem.node(sys.argv[1:], slot_mods=mods)))
'''


def node_payload(check_dir: Path = CHECK_DIR) -> bytes:
    from . import scopemem

    def src(p: Path) -> str | None:
        if not p.exists():
            return None
        try:
            return p.read_text()
        except OSError as e:
            print(f"research: {p} unread, so the node reading goes without it ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            return None

    srcs = {"scopemem": Path(scopemem.__file__).read_text(), "slot": src(check_dir / "slot.py"), "lean": src(check_dir / "lean_slot.py")}
    return (f"SRC = {srcs!r}\n" + NODE_PROGRAM).encode()


def node_readings(local_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    by: dict[tuple, list[tuple[Path, dict[str, Any]]]] = {}
    out: dict[str, dict[str, Any]] = {}
    for d in local_dirs:
        rec = _read_json(d / "remote.json")
        if not rec or not rec.get("remote_run_dir"):
            out[str(d)] = {"error": f"{d} is not a launched remote run"}
            continue
        by.setdefault((rec["host"], rec["port"], rec.get("user") or "root"), []).append((d, rec))
    payload = node_payload() if by else b""

    def one(target: tuple, runs: list[tuple[Path, dict[str, Any]]]) -> None:
        machine = runs[0][1].get("machine")
        try:
            r = ssh_run(ssh_argv(target[0], target[1], user=target[2]), ["python3", "-", *(rec["remote_run_dir"] for _, rec in runs)],
                        stdin=payload, timeout=60)
            doc = json.loads(r.stdout) if r.returncode == 0 else None
            why = None if isinstance(doc, dict) else f"rc {r.returncode} {r.stderr.decode(errors='replace').strip()[-200:]}"
        except subprocess.TimeoutExpired:
            doc, why = None, "took over 60 s"
        except ValueError:
            doc, why = None, f"its answer is not JSON: {r.stdout[:120]!r}"
        except Exception as e:  # noqa: BLE001  (no ssh key here, ...): that machine's runs unread, the rest still shown
            doc, why = None, f"{type(e).__name__}: {e}"
        for d, rec in runs:
            run = (doc or {}).get("runs", {}).get(rec["remote_run_dir"])
            out[str(d)] = ({"run": run, "node": {k: v for k, v in doc.items() if k != "runs"}, "machine": machine} if run is not None
                           else {"error": f"reading {machine}: {why or 'no reading of this run'}"})

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, len(by))) as ex:
        list(ex.map(lambda kv: one(*kv), by.items()))
    return out


def py_spy(local_dir: Path) -> str:
    from .telemetry import pyspy

    rec = _read_json(local_dir / "remote.json")
    if not rec or not rec.get("remote_run_dir"):
        raise Refused(f"{local_dir} is not a launched remote run")
    limit = pyspy.MAX_PROCS * pyspy.DUMP_TIMEOUT_S + 60
    try:
        r = ssh_run(ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root"), ["python3", "-", rec["remote_run_dir"]],
                    stdin=Path(pyspy.__file__).read_bytes(), timeout=limit)
    except subprocess.TimeoutExpired:
        raise Refused(f"the dump on {rec.get('machine')} took over {limit:g} s") from None
    if r.returncode:
        raise Refused(f"no dump on {rec.get('machine')}: rc {r.returncode} {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout.decode(errors="replace")


def cancel(local_dir: Path, *, reason: str, by: str) -> dict[str, Any]:
    rec = fetch(local_dir)
    if err := rec.get("last_fetch_error"):
        raise Refused(f"the machine is unreachable ({err['stderr'].strip() or 'rc ' + str(err['rc'])}); nothing was signalled")
    st = _read_json(local_dir / "status.json") or {}
    if not st:
        if refused := _read_json(local_dir / "refused.json"):
            raise Refused(f"{local_dir.name} was refused on its machine ({refused.get('fault') or 'Refused'}: {refused.get('reason')}), "
                          f"so there is nothing to cancel")
        if (w := withdraw(local_dir, rec, reason=reason, by=by)) is not None:
            return w
        rec = fetch(local_dir)
        st = _read_json(local_dir / "status.json") or {}
        if not st:
            raise Refused(f"{local_dir.name} left its hold as the cancel reached it and its runner hasn't written status.json yet; "
                          f"nothing was signalled: run `research cancel` again")
    if st.get("state") != "running":
        raise Refused(f"{local_dir.name} is {st.get('state')}, so there is nothing to cancel")
    pgid = st.get("pgid")
    if not isinstance(pgid, int) or pgid <= 1:
        raise Refused(f"{local_dir.name}'s status.json names no workload process group (pgid {pgid!r})")
    tool = _shipped_tool(local_dir, rec)
    rd = rec["remote_run_dir"]
    tele = ["env", f"PYTHONPATH={tool}", "python3", "-m", "research.telemetry", "cancel-intent", "--actor", "manual",
            "--reason", f"{reason} (research cancel by {by})", "--target-pgid", str(pgid), "--expect-run-dir", rd, "--attempt-dir", rd,
            "--deliver"]

    def deliver(argv: list[str]) -> subprocess.CompletedProcess:
        script = (f"f={shlex.quote(rd)}/events.jsonl; n=$(wc -l < \"$f\" 2>/dev/null || echo 0); {shlex.join(argv)}; rc=$?; "
                  f"tail -n +$((n + 1)) \"$f\"; exit $rc")
        return ssh_run(ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root"), ["sh", "-c", script], timeout=120)

    r, old_tool = deliver([*tele, "--grace-s", f"{CANCEL_GRACE_S:g}"]), False
    stderr = r.stderr.decode(errors="replace").strip()
    if r.returncode == 2 and "--grace-s" in stderr:
        r, old_tool = deliver(tele), True
        stderr = r.stderr.decode(errors="replace").strip()
    if r.returncode == 2 and "--expect-run-dir" in stderr:
        raise Refused(f"{local_dir.name}'s research package ({tool}) predates the workload check, so nothing was signalled; on the machine, "
                      f"`kill -TERM -{pgid}` once /proc/<pid>/environ of a group member names RESEARCH_RUN_DIR={rd}")
    res = next((x for x in map(_json_line, r.stdout.decode(errors="replace").splitlines())
                if x.get("kind") == "cancel_result" and (x.get("target") or {}).get("pgid") == pgid), None)
    if res is None:
        raise Refused(f"the cancel recorded no result on the machine (rc {r.returncode}): {stderr[-300:]}")
    tgt = res.get("target") or {}
    scope = res.get("scope")
    if old_tool:
        scope = {"fallback": f"the run's research package ({tool}) predates the scope end: SIGTERM to its group and to the processes "
                             f"naming its run dir only, and nothing SIGKILLs what ignores it", "runner_kills_after_s": None}
    return {"pgid": pgid, "processes": tgt.get("members_total", len(tgt.get("members") or [])), "delivered": bool(res.get("delivered")),
            "why": res.get("reason"), "scope": scope}


def _shipped_tool(local_dir: Path, rec: dict[str, Any]) -> str:
    tool = rec.get("tool") or ((((_read_json(local_dir / "job.json") or {}).get("run") or {}).get("remote") or {}).get("tool"))
    if not tool:
        raise Refused(f"neither {local_dir.name}'s remote.json nor its job.json names a shipped tool to record the cancel with")
    snap, root = PurePosixPath(tool), rec.get("root")
    if not root or snap.parent != PurePosixPath(root) / "tool" or ":" in tool or snap.name in ("", ".", ".."):
        raise Refused(f"{local_dir.name}'s tool {tool!r} is not a snapshot under {root}/tool/; nothing was signalled")
    return tool


def withdraw(local_dir: Path, rec: dict[str, Any], *, reason: str, by: str) -> dict[str, Any] | None:
    from . import withdraw as W
    from .store import custody as CU

    args = {"run_dir": rec["remote_run_dir"], "reason": reason, "by": by, "cred_file": CU.CRED_FILE,
            "cred_dirs": [CUSTODY_DIR, STORE_READ_DIR], "credentials": list(CU.CREDENTIALS)}
    try:
        r = ssh_run(ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root"), ["python3", "-", json.dumps(args)],
                    stdin=Path(W.__file__).read_bytes(), timeout=W.EXIT_WAIT_S + 90)
    except subprocess.TimeoutExpired:
        raise Refused(f"the withdrawal on {rec.get('machine')} gave no answer in {W.EXIT_WAIT_S + 90:g} s; `research fetch "
                      f"{local_dir.name}` says what it did") from None
    out = _json_line((r.stdout.decode(errors="replace").strip().splitlines() or [""])[-1]) if r.returncode == 0 else {}
    name, what = local_dir.name, out.get("outcome")
    if what == "withdrawn":
        fetch(local_dir)
        return {"withdrawn": True, "delivered": True, "holds": out.get("holds") or [], "refused": out.get("refused"),
                "released": out.get("released") or []}
    if what == "started":
        return None
    why = {"refused": f"was refused on its machine ({out.get('fault')}: {out.get('reason')}), so there is nothing to cancel",
           "busy": f"is still being launched ({out.get('why')}); nothing was signalled: run `research cancel` again",
           "absent": f"is not on its machine ({out.get('why')}); nothing was signalled",
           "unstoppable": f"is held, but {out.get('why')}",
           "survived": f"is held, and its hold process(es) {out.get('pids')} outlived SIGTERM for {W.EXIT_WAIT_S:g} s; no refused.json "
                       f"was written",
           "no-hold": f"has no status.json and no hold process names its request, so nothing was signalled; its runner (pid "
                      f"{out.get('runner_pid')}) is " + ("alive and starting: run `research cancel` again" if out.get("runner_alive")
                                                         else "gone (`research fetch` and its launcher.log say why)")}.get(what)
    if why is None:
        raise Refused(f"the withdrawal of {name} failed on {rec.get('machine')} (rc {r.returncode}): "
                      f"{r.stderr.decode(errors='replace').strip()[-300:]}")
    raise Refused(f"{name} {why}")


def _file_sha256(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _verify_and_manifest(ssh: list[str], remote_dir: str, local_dir: Path, rec: dict[str, Any]) -> dict[str, Any] | None:
    r = ssh_run(ssh, ["sh", "-c", f"cd {shlex.quote(remote_dir)} && t=$(date +%s.%N 2>/dev/null); case $t in *N*|'') "
                                  f"t=$(python3 -c 'import time; print(time.time())' 2>/dev/null || date +%s);; esac; "
                                  f"echo \"T0 $t\" && find . -type f ! -name preserved.json ! -name {FETCHED_FILE} ! -name {FETCHED_FILE}.tmp "
                                  f"-print0 | xargs -0 sha256sum"], timeout=900)
    if r.returncode != 0:
        return None
    remote_sums, t0 = {}, None
    for line in r.stdout.decode().splitlines():
        if line.startswith("T0 "):
            t0 = float(line[3:])
            continue
        h, _, name = line.partition("  ")
        remote_sums[name.removeprefix("./")] = h
    files, mismatched = [], []
    for name, h in sorted(remote_sums.items()):
        p = local_dir / name
        if not p.is_file():
            mismatched.append(name)
            continue
        lh = _file_sha256(p)
        (files if lh == h else mismatched).append({"path": name, "bytes": p.stat().st_size, "sha256": lh} if lh == h else name)
    if mismatched:
        rec["preserve_error"] = {"at_utc": utc_now_ms(), "mismatched_or_missing": mismatched}
        return None
    manifest = {"schema": PRESERVED_SCHEMA, "run_id": rec["run_id"], "source": f"{rec['host']}:{remote_dir}", "destination": str(local_dir),
                "files": files, "completed_utc": utc_now_ms(), "verified_from_machine_epoch": t0,
                "note": "copy on the control machine; the machine's copy is the other one"}
    write_json_atomic(local_dir / "preserved.json", manifest)
    return manifest


def _mark_fetched(ssh: list[str], remote_dir: str, local_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest["files"]
    doc = {"schema": FETCHED_SCHEMA, "run_id": manifest["run_id"], "verified_from": manifest["verified_from_machine_epoch"],
           "files": len(files), "bytes": sum(f["bytes"] for f in files),
           "manifest_sha256": hashlib.sha256("".join(f"{f['sha256']}  {f['path']}\n" for f in files).encode()).hexdigest(),
           "fetched_utc": utc_now_ms(), "destination": f"{os.uname().nodename}:{local_dir}"}
    if doc["verified_from"] is None:
        return {"written": False, "reason": "the machine did not report its clock before hashing (no T0)"}
    q = shlex.quote(remote_dir)
    r = ssh_run(ssh, ["sh", "-c", f"cd {q} && cat > {FETCHED_FILE}.tmp && mv -f {FETCHED_FILE}.tmp {FETCHED_FILE}"],
                stdin=json.dumps(doc).encode(), timeout=120)
    if r.returncode != 0:
        return {"written": False, "reason": f"rc {r.returncode}: {r.stderr.decode(errors='replace')[-200:]}"}
    return {"written": True, "verified_from": doc["verified_from"], "files": doc["files"], "bytes": doc["bytes"]}
