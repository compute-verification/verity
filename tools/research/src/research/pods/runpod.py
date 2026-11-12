#!/usr/bin/env python3
"""RunPod, which now hosts only the control pod (`vy-control-verity`): create, list, get, ssh and terminate, plus the REST and
GraphQL calls the budgets guard and `research notes` make. Nebius runs every node (`pods/nebius/`).

Reads the API key from ``~/.runpod/config.toml`` and never prints it.  Talks to
the REST API (``https://rest.runpod.io/v1``).  RunPod counts the account's REST
calls and its GraphQL calls apart, each by the minute, the hour and the UTC day,
and answers 429 past a count.  So the two reads, one pod (``get``) and every pod
(``pods``), ask GraphQL when REST gives no answer, and hand back REST's shape
with the fields ``_as_rest`` lists; the balance is GraphQL's alone.

Every call that reaches RunPod is one line of ``api-calls.log`` under
``$XDG_STATE_HOME/research/runpod/`` (default ``~/.local/state``): the time, the
API, the call, the status, the counts RunPod says are left, and who called.
The address ``research`` last read for a pod is kept there too
(``address-<pod id>.json``), so a call that names the pod asks RunPod again only
when that address is over ``ADDRESS_TTL_S`` old or the host there does not hold
the host key recorded for it (``remote.pod_target``).

Who is at an address is RunPod's word or a host key, never that a port takes a
connection.  An address RunPod has just given is dialled as it always was
(``StrictHostKeyChecking=no``), and is kept only when the host there proves a
host key at that moment, which ``known_hosts()`` then records for the address
(``keep_address``).  A kept address is dialled without RunPod's fresh word only
after the host there has proved that same key in a handshake (``kept_stands``),
and every ssh command this process then builds for it checks the key strictly
(``ssh_command``).  A host refused for its key is not dialled: the address is
forgotten and RunPod asked.

Where the laptop files are absent (a cloud VM with only the Cursor secrets) the
environment stands in, and a file that exists always wins:
``RUNPOD_API_KEY`` for the API key (with the default GraphQL URL), and with
neither, the REST calls are asked of the control pod's gate (``pods/door.py``);
``RUNPOD_SSH_KEY_B64`` for the ssh key, decoded once to a mode-600 file under
``$XDG_STATE_HOME/research/runpod/`` (default ``~/.local/state``), with its
known_hosts beside it.  Neither value is ever printed.

    python -m research.pods.runpod create --name vy-control --gpu ... --disk 100
    python -m research.pods.runpod list
    python -m research.pods.runpod get POD_ID
    python -m research.pods.runpod ssh POD_ID                # prints the ssh command once port 22 is mapped
    python -m research.pods.runpod terminate POD_ID

Every pod this script creates carries the label ``veritor-campaign`` in its
name so a stray pod is recognisable from the console.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

REST = "https://rest.runpod.io/v1"
CONFIG = Path.home() / ".runpod" / "config.toml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
KNOWN_HOSTS = Path.home() / ".runpod" / "ssh" / "veritor-campaign-known_hosts"
GRAPHQL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CALL_LOG = "api-calls.log"
CALL_LOG_MAX = 4 << 20              # bytes; past it the log becomes `api-calls.log.1` (one older file is kept) and a new one starts
ADDRESS_TTL_S = 3600.0              # how long a pod's kept address is used before RunPod is asked again
ADDRESS_KEYS = ("host", "port", "user", "pod_name", "cost_per_hr", "rate_observed_utc", "gpu")    # a runpod machine's target, as kept
HOST_KEY_CONNECT_S = 5              # ssh's ConnectTimeout for the handshake that asks a host for its key (`host_key`) ...
HOST_KEY_S = 20.0                   # ... and the whole handshake's bound
#: what the GraphQL reads ask of a pod: the fields `_as_rest` turns into REST's
POD_FIELDS = ("id name desiredStatus costPerHr createdAt machineId machine { gpuTypeId } "
              "runtime { ports { ip isIpPublic privatePort publicPort type } }")
#: the clock of the kept addresses and the call log, and the runner of `host_key`'s ssh and `drop_key`'s ssh-keygen: a test's stand-ins
clock, run = time.time, subprocess.run
#: the kept addresses (host, port) this process dials without RunPod's fresh word (`kept_stands`): `ssh_command` checks the host
#: key at one strictly
_KEPT: set[tuple[str, str]] = set()
_SAID: set[str] = set()


class PodError(RuntimeError):
    """The provider could not answer for a pod: an HTTP error (404 = the pod no longer exists), no API config, no network, sshd never
    answering.  `code` is the HTTP status when there is one.  A library caller (`research.remote`) turns it into its own refusal; the
    standalone CLI (`main`) prints it and exits 1.  Never `SystemExit` from library code: that took `research fetch` / `status
    --refresh` down whole on a terminated pod."""

    def __init__(self, message: str, *, code: int | None = None, pod_id: str | None = None):
        super().__init__(message)
        self.code, self.pod_id = code, pod_id


def _config() -> dict[str, Any]:
    try:
        with CONFIG.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        key = os.environ.get("RUNPOD_API_KEY")
        if key:
            return {"apikey": key, "apiurl": GRAPHQL}
        raise PodError(f"no RunPod API config at {CONFIG} and RUNPOD_API_KEY is not set") from None


def _env_dir() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "research" / "runpod"


def ssh_key() -> Path:
    """The private key file: `SSH_KEY` when it exists, else `RUNPOD_SSH_KEY_B64` written once (mode 600, named by its content
    hash so a rotated secret gets a new file).  Neither set: `SSH_KEY`, so ssh names the missing file as before, except on a Cursor
    VM that injects the secret this shell lost (`research.secret_hint`, spelled out: this file runs standalone on a pod)."""
    b64 = os.environ.get("RUNPOD_SSH_KEY_B64")
    if not SSH_KEY.is_file() and not b64 and "RUNPOD_SSH_KEY_B64" in os.environ.get("CLOUD_AGENT_INJECTED_SECRET_NAMES", "").split(","):
        raise PodError("RUNPOD_SSH_KEY_B64 not in this shell though this VM injects it: rerun from a fresh tmux login shell")
    if SSH_KEY.is_file() or not b64:
        return SSH_KEY
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise PodError("RUNPOD_SSH_KEY_B64 is not valid base64") from None
    if not raw.endswith(b"\n"):
        raw += b"\n"
    path = _env_dir() / f"ssh-key-{hashlib.sha256(raw).hexdigest()[:16]}"
    try:
        if path.read_bytes() == raw and (path.stat().st_mode & 0o077) == 0:
            return path
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")     # threads of one process resolve it at once
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, raw)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return path


def known_hosts() -> Path:
    """`KNOWN_HOSTS` where its directory exists (the laptop), else a writable file beside the env-materialised key."""
    if KNOWN_HOSTS.parent.is_dir():
        return KNOWN_HOSTS
    d = _env_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d / "known_hosts"


def public_key() -> str:
    """The `.pub` beside `SSH_KEY` when it exists, else derived from the private key (`ssh-keygen -y`)."""
    pub = SSH_KEY.with_suffix(".pub")
    if pub.is_file():
        return pub.read_text().strip()
    out = subprocess.run(["ssh-keygen", "-y", "-f", str(ssh_key())], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise PodError(f"cannot derive the ssh public key (no {pub}; ssh-keygen rc {out.returncode})")
    return out.stdout.strip()


def _struck(text: str, cfg: dict[str, Any]) -> str:
    """`text` with the config's API key struck out.  A config that holds no key, or an empty one, strikes nothing: an empty key
    is between every two characters of any text."""
    key = str(cfg.get("apikey") or "")
    return text.replace(key, "<the API key>") if key else text


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    try:
        cfg = _config()
    except PodError as no_key:
        # a machine with no RunPod key asks the control pod's gate, which holds the account's (pods/door.py)
        try:
            from . import door
        except ImportError:                 # this file alone on a pod: no package around it
            raise no_key from None
        return door.api(method, path, body, no_key=str(no_key))
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{REST}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {cfg['apikey']}",
            "Content-Type": "application/json",
            "User-Agent": "veritor-campaign/0.1",
        },
    )
    pod_id = path[len("/pods/"):].split("/", 1)[0] if path.startswith("/pods/") else None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
            _note_call("rest", method, path, getattr(response, "status", None), getattr(response, "headers", None))
    except urllib.error.HTTPError as error:
        _note_call("rest", method, path, error.code, error.headers)
        detail = _struck(error.read().decode(errors="replace"), cfg)[:500]
        what = "not found (terminated, or never existed)" if error.code == 404 else detail
        raise PodError(f"{method} {path} -> HTTP {error.code}: {what}", code=error.code, pod_id=pod_id) from None
    except ValueError:                          # its text quotes the header it could not send, which holds the key
        _note_call("rest", method, path, None, None)
        raise PodError(f"{method} {path}: the RunPod config's API key holds a character no request can carry", pod_id=pod_id) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        _note_call("rest", method, path, None, None)
        raise PodError(f"{method} {path}: {type(error).__name__}: {getattr(error, 'reason', None) or error}", pod_id=pod_id) from None
    return json.loads(raw) if raw else None


def _graphql(query: str) -> Any:
    cfg = _config()
    request = urllib.request.Request(
        f"{cfg['apiurl']}?api_key={cfg['apikey']}",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "veritor-campaign/0.1",
        },
    )
    what = ".".join(re.findall(r"[A-Za-z_]\w*", query)[:2]) or "?"       # `myself.pods`, `myself.clientBalance`, `pod.input`
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            _note_call("graphql", "POST", what, getattr(response, "status", None), getattr(response, "headers", None))
    except urllib.error.HTTPError as error:
        _note_call("graphql", "POST", what, error.code, error.headers)
        raise
    except (http.client.InvalidURL, UnicodeError):      # its text quotes the request's target, which holds the key
        _note_call("graphql", "POST", what, None, None)
        raise PodError("GraphQL: the RunPod config's API URL or key holds a character no request can carry") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        _note_call("graphql", "POST", what, None, None)
        raise
    return json.loads(raw)


def _caller() -> str:
    """Who made a call, by the program's name and the subcommands after it: `research,pods,ssh`, `research,run`.  A word that
    is not a subcommand's shape ends it, so no argument's value is written."""
    argv = list(sys.argv) or ["?"]
    head = Path(argv[0])
    words = [head.parent.name if head.name == "__main__.py" else head.name]
    for arg in argv[1:3]:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,15}", arg):
            break
        words.append(arg)
    return ",".join(words) or "?"


def _say(what: str) -> None:
    """One line on stderr, once a process."""
    if what not in _SAID:
        _SAID.add(what)
        print(f"research: {what}", file=sys.stderr)


def _note_call(api: str, method: str, what: str, status: int | None, headers: Any) -> None:
    """One line of the call log (module docstring) for a call sent to RunPod: `status` is its HTTP status, `none` when no answer
    came, and the counts left are read from the answer's headers.  It is given neither the key nor the URL that carries it.
    Nothing here, a full disk included, may fail the call it records, which the budgets guard makes every poll: a line that
    can't be written is said once on stderr."""
    try:
        get = headers.get if headers is not None else (lambda _name: None)
        left = dict(re.findall(r'"(minute|hour|day)";r=(\d+)', str(get("ratelimit") or "")))    # REST: `"day";r=N;t=S, ...`
        for window in ("minute", "hour", "day"):                                                  # GraphQL: one header a window
            if window not in left and re.fullmatch(r"\d+", str(get(f"x-ratelimit-remaining-{window}") or "")):
                left[window] = str(get(f"x-ratelimit-remaining-{window}"))
        line = " ".join([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock())), api, method, what,
                         str(status) if status is not None else "none",
                         *(f"{window}={left[window]}" for window in ("day", "hour", "minute") if window in left),
                         f"by={_caller()}", f"pid={os.getpid()}"])
        path = _env_dir() / CALL_LOG
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_file() and path.stat().st_size > CALL_LOG_MAX:
            os.replace(path, path.with_name(f"{CALL_LOG}.1"))
        with path.open("a") as log:
            log.write(line + "\n")
    except Exception as error:  # noqa: BLE001
        _say(f"RunPod's call log can't be written, so its calls are in none: {type(error).__name__}: {error}")


def _graphql_data(query: str, pod_id: str | None = None) -> dict[str, Any]:
    """`data` of a GraphQL query; any failure (no answer, an HTTP error, `errors` in the answer) is a `PodError`.  The request's
    URL holds the API key, so an error's text takes nothing from the request: it quotes RunPod's answer with the key struck out
    should the answer repeat it, and of a failure whose own text can hold the URL it gives the type alone."""
    cfg = _config()

    def quoted(answer: str) -> str:
        return _struck(answer, cfg)[:300]

    try:
        doc = _graphql(query)
    except urllib.error.HTTPError as error:
        detail = quoted(error.read().decode(errors="replace"))
        raise PodError(f"GraphQL -> HTTP {error.code}: {detail}", code=error.code, pod_id=pod_id) from None
    except http.client.HTTPException as error:
        raise PodError(f"GraphQL: {type(error).__name__}", pod_id=pod_id) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise PodError(f"GraphQL: {type(error).__name__}: {quoted(str(getattr(error, 'reason', None) or error))}", pod_id=pod_id) from None
    if not isinstance(doc, dict) or doc.get("errors") or not isinstance(doc.get("data"), dict):
        errors = doc.get("errors") if isinstance(doc, dict) else doc
        raise PodError(f"GraphQL: {quoted(json.dumps(errors))}", pod_id=pod_id)
    return doc["data"]


def _as_rest(pod: dict[str, Any]) -> dict[str, Any]:
    """A GraphQL pod (`POD_FIELDS`) as REST gives one, with the fields `research` reads: id, name, desiredStatus, costPerHr,
    createdAt, machineId, machine.gpuTypeId, and publicIp and portMappings from its public TCP ports."""
    ports = [p for p in ((pod.get("runtime") or {}).get("ports") or []) if p.get("isIpPublic") and p.get("type") == "tcp"]
    ssh = [p for p in ports if p.get("privatePort") == 22] or ports
    return {**{k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "createdAt", "machineId")},
            "machine": pod.get("machine") or {}, "publicIp": (ssh[0].get("ip") if ssh else None) or "",
            "portMappings": {str(p["privatePort"]): p["publicPort"] for p in ports if p.get("privatePort") and p.get("publicPort")}}


def _graphql_pod(pod_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9]{1,64}", pod_id or ""):
        raise PodError(f"GraphQL: {pod_id!r} is not a pod id", pod_id=pod_id)
    pod = _graphql_data(f'{{ pod(input: {{podId: "{pod_id}"}}) {{ {POD_FIELDS} }} }}', pod_id).get("pod")
    if not pod:
        raise PodError(f"GraphQL pod {pod_id}: not found (terminated, or never existed)", code=404, pod_id=pod_id)
    return _as_rest(pod)


def _graphql_pods() -> list[dict[str, Any]]:
    listing = (_graphql_data(f"{{ myself {{ pods {{ {POD_FIELDS} }} }} }}").get("myself") or {}).get("pods")
    if not isinstance(listing, list):
        raise PodError("GraphQL: the answer lists no pods")
    return [_as_rest(p) for p in listing]


def _read(rest: Callable[[], Any], graphql: Callable[[], Any]) -> Any:
    """REST's answer, or GraphQL's when REST gives none: its rate limit (HTTP 429), a server error, or no answer at all.  Any
    other status is REST's answer (404: the pod is gone) and is raised as it came.  With no key here REST was asked of the control
    pod's gate, which has no verb for GraphQL, so its failure stands."""
    try:
        return rest()
    except PodError as refused:
        if refused.code is not None and refused.code != 429 and refused.code < 500:
            raise
        try:
            _config()
        except PodError:
            raise refused from None
        try:
            return graphql()
        except PodError as error:
            if error.code == 404:
                raise
            raise PodError(f"{refused}; {error}", code=refused.code, pod_id=refused.pod_id) from None


def pods() -> list[dict[str, Any]]:
    """Every pod of the account."""
    return _read(lambda: _request("GET", "/pods") or [], _graphql_pods)


def _address_file(pod_id: str) -> Path | None:
    return _env_dir() / f"address-{pod_id}.json" if re.fullmatch(r"[A-Za-z0-9]{1,64}", pod_id or "") else None


def kept_address(pod_id: str) -> tuple[dict[str, Any], float] | None:
    """(the target `remote.pod_target` last read for the pod, marked `kept`, its age in seconds), or None.  The mark is put on
    here and never stored: the file holds `ADDRESS_KEYS` and nothing else, so no file can say its address needs no check, and one
    that is not such a target is removed and reads as none."""
    path = _address_file(pod_id)
    if path is None or not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
        target, read_at = doc["target"], float(doc["read_at"])
        if set(target) != set(ADDRESS_KEYS) or not (isinstance(target["host"], str) and target["host"]) or type(target["port"]) is not int:
            raise ValueError("not a kept target")
    except (OSError, ValueError, KeyError, TypeError):
        forget_address(pod_id)
        return None
    return {**target, "kept": True}, max(0.0, clock() - read_at)


def keep_address(pod_id: str, target: dict[str, Any]) -> None:
    """Keep the target RunPod has just given for the pod, for its next lookup, if the host there now proves a host key that is
    from here the key recorded for the address (`_recorded`): a kept address always has the key its host held when RunPod gave
    it.  This process's ssh to the address is as RunPod's word allows from here (`ssh_command`).  An address that is not kept is
    read from RunPod at the next call."""
    host, port = target["host"], target["port"]
    _KEPT.discard((host, str(port)))
    path = _address_file(pod_id)
    if path is None:
        return
    if not _recorded(host, port):
        forget_address(pod_id)
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps({"target": {k: target[k] for k in ADDRESS_KEYS}, "read_at": clock()}))
        os.replace(tmp, path)
    except OSError as error:
        _say(f"pod {pod_id}'s address can't be kept, so RunPod is asked for it at each call: {type(error).__name__}: {error}")


def forget_address(pod_id: str) -> None:
    path = _address_file(pod_id)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        _say(f"pod {pod_id}'s kept address can't be removed: {type(error).__name__}: {error}")


def _known_as(host: str, port: int) -> str:
    """The address as known_hosts names it."""
    return host if port == 22 else f"[{host}]:{port}"


def host_key(host: str, port: int, *, fresh: bool = False) -> str:
    """Whether the host at `host:port` holds the host key `known_hosts()` records for that address, by one ssh handshake of its
    own (no shared connection) that checks the key and asks for no login: "held" when it proved the key (only then does ssh get
    as far as being denied a login it never asked for), "refused" when ssh refused it for its key (the host there proves another
    or, unless `fresh`, none is recorded for the address), "silent" when nothing there finished a handshake.  `fresh`: RunPod has
    just given the address, so a host with no key recorded there has its key recorded (ssh's `accept-new`) and holds it.  A kept
    address's use rests on this; that a port takes a connection says nothing of who is there."""
    base = _ssh_command(host, port, "root", checking="accept-new" if fresh else "yes")
    argv = [*base[:-1], "-o", "BatchMode=yes", "-o", "PreferredAuthentications=none", "-o", "ControlPath=none",
            "-o", f"ConnectTimeout={HOST_KEY_CONNECT_S}", base[-1], "true"]
    try:
        r = run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=HOST_KEY_S)
    except (OSError, subprocess.TimeoutExpired):
        return "silent"
    if "Host key verification failed" in r.stderr:
        return "refused"
    return "held" if r.returncode == 0 or re.search(r"^\S+@\S+: Permission denied \(", r.stderr, re.M) else "silent"


def drop_key(host: str, port: int) -> None:
    """Remove what `known_hosts()` records for the address."""
    name, recorded = _known_as(host, port), known_hosts()
    if not recorded.is_file():
        return
    try:
        r = run(["ssh-keygen", "-R", name, "-f", str(recorded)], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=HOST_KEY_S)
        failed = f"ssh-keygen exits {r.returncode}" if r.returncode else ""
    except (OSError, subprocess.TimeoutExpired) as error:
        failed = type(error).__name__
    if failed:
        _say(f"the host key recorded for {name} can't be removed ({failed}): until it is, RunPod is asked for that address at each call")


def _recorded(host: str, port: int) -> bool:
    """Whether the host at the address RunPod has just given proves a host key that is, from here, the key recorded for the
    address.  A key recorded for an earlier host there is dropped first: RunPod's word is fresher than the record, and is the
    only thing a record is dropped on."""
    answer = host_key(host, port, fresh=True)
    if answer == "refused":
        drop_key(host, port)
        answer = host_key(host, port, fresh=True)
    return answer == "held"


def kept_stands(pod_id: str, target: dict[str, Any]) -> bool:
    """Whether the pod's kept address may be dialled without RunPod's fresh word: the host there has just proved the key recorded
    for that address (`host_key`), which is the key its host held when RunPod gave it (`keep_address`).  From then on every ssh
    command this process builds for the address checks that key strictly (`ssh_command`), until RunPod gives the address again.
    A host refused for its key is not the pod as far as anything here can tell: the address is forgotten, so RunPod is asked."""
    at = (target["host"], str(target["port"]))
    answer = host_key(target["host"], target["port"])
    if answer == "held":
        _KEPT.add(at)
        return True
    if answer == "refused":
        forget_address(pod_id)
    return False


def create(args: argparse.Namespace) -> None:
    body = {
        "name": f"{args.name}-veritor-campaign",
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": args.gpu_count,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        # 22/tcp always; --expose-port adds raw TCP ports (a live verifier's listen port).  RunPod maps
        # each to a random public port (`portMappings`, `research pods list`); the pod-side process listens on the inner one.
        "ports": ["22/tcp", *(f"{p}/tcp" for p in (getattr(args, "expose_port", None) or []))],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": public_key()},
    }
    if args.min_vcpu:
        body["minVCPUPerGPU"] = args.min_vcpu
    if args.min_ram:
        body["minRAMPerGPU"] = args.min_ram
    if args.network_volume:
        # durable storage: the REST field `networkVolumeId` mounts the volume at /workspace (pod disk stays the
        # container disk); the pod is then pinned to the volume's data center (create fails by name if no such GPU/CPU is there).
        body["networkVolumeId"] = args.network_volume
        body["volumeMountPath"] = args.volume_mount
    if args.data_center:
        body["dataCenterIds"] = [args.data_center]
    if args.cpu:
        # CPU pod (no GPU): RunPod sizes RAM by flavor x vCPU (cpu3m / cpu5m = 8 GB per vCPU), so
        # ``--cpu cpu3m --vcpu 32`` is a 256 GB box.  GPU fields are ignored by the API for CPU pods.
        body["computeType"] = "CPU"
        body["cpuFlavorIds"] = [args.cpu]
        body["cpuFlavorPriority"] = "custom"
        body["vcpuCount"] = args.vcpu
        for key in ("gpuTypeIds", "gpuCount", "minVCPUPerGPU", "minRAMPerGPU"):
            body.pop(key, None)
    if getattr(args, "lease_expires", None):
        # `research pods create --max-hours`: the lease holds from boot (research.pods.lease.boot_fields)
        from .lease import boot_fields

        extra = boot_fields(args.image, args.lease_expires, boot=bool(getattr(args, "boot_lease", False)))
        body["env"].update(extra.pop("env"))
        body.update(extra)
    pod = _request("POST", "/pods", body)
    print(json.dumps(pod, indent=2))
    return pod


def create_args(c: argparse.ArgumentParser) -> None:
    """The `create` flags, shared with `research pods create` (research.pods.part)."""
    c.add_argument("--name", required=True)
    c.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--disk", type=int, default=100)
    c.add_argument("--gpu-count", type=int, default=1, help="GPUs per pod (tensor-parallel lanes)")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    c.add_argument("--min-vcpu", type=int, default=0)
    c.add_argument("--min-ram", type=int, default=0)
    c.add_argument("--cpu", default="", help="CPU pod flavor (cpu3m/cpu5m = 8 GB RAM per vCPU); no GPU")
    c.add_argument("--vcpu", type=int, default=16, help="vCPUs for a CPU pod (RAM = flavor GB/vCPU x this)")
    c.add_argument("--network-volume", default="", help="RunPod network volume id to mount (REST networkVolumeId); pins the pod to the volume's DC")
    c.add_argument("--volume-mount", default="/workspace", help="mount path of --network-volume (default /workspace)")
    c.add_argument("--data-center", default="", help="restrict to one data center id (e.g. EU-RO-1)")
    c.add_argument("--expose-port", action="append", type=int, default=[], metavar="PORT",
                   help="also expose this TCP port (repeatable; e.g. a live verifier's listen port). Public mapping: `pods list`")


def list_pods() -> None:
    for pod in pods():
        print(
            f"{pod.get('id')}  {pod.get('name')}  {pod.get('desiredStatus')}  "
            f"{(pod.get('machine') or {}).get('gpuTypeId')}  ${pod.get('costPerHr')}/h  "
            f"ip={pod.get('publicIp')} ports={pod.get('portMappings')}"
        )


def get(pod_id: str) -> dict[str, Any]:
    return _read(lambda: _request("GET", f"/pods/{pod_id}"), lambda: _graphql_pod(pod_id))


def ssh_target(pod: dict[str, Any]) -> tuple[str, int] | None:
    ip = pod.get("publicIp")
    ports = pod.get("portMappings") or {}
    port = ports.get("22")
    if ip and port:
        return str(ip), int(port)
    return None


def ssh_command(host: str, port: int, *, user: str = "root") -> list[str]:
    """The ssh command every `research` connection to a machine starts from (`remote.ssh_argv`, `connect.ssh_for`, and what they
    are made into: rsync's `-e`, git's `ext::`).  It records a host's key at the first connection and connects whatever the key,
    as RunPod's word or a registry entry names the address; at a kept address in use (`kept_stands`) it connects only to the
    host that proves the recorded key."""
    return _ssh_command(host, port, user, checking="yes" if (host, str(port)) in _KEPT else "no")


def _ssh_command(host: str, port: int, user: str, *, checking: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(ssh_key()),
        "-p",
        str(port),
        "-o",
        f"StrictHostKeyChecking={checking}",
        "-o",
        f"UserKnownHostsFile={known_hosts()}",
        "-o",
        "ServerAliveInterval=30",
        f"{user}@{host}",
    ]


def wait_ssh(pod_id: str, timeout_s: int = 1800) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = get(pod_id)
        target = ssh_target(pod)
        if target is not None:
            command = ssh_command(*target)
            probe = subprocess.run(
                # options must precede the user@host argument
                [
                    *command[:-1],
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "BatchMode=yes",
                    command[-1],
                    "true",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                return target
        time.sleep(15)
    raise PodError(f"pod {pod_id}: sshd did not answer within {timeout_s}s", pod_id=pod_id)


def terminate(pod_id: str) -> None:
    _request("DELETE", f"/pods/{pod_id}")
    print(f"terminated {pod_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create_args(sub.add_parser("create"))
    sub.add_parser("list")
    for name in ("get", "ssh", "terminate"):
        p = sub.add_parser(name)
        p.add_argument("pod_id")
    args = parser.parse_args()

    try:
        if args.command == "create":
            create(args)
        elif args.command == "list":
            list_pods()
        elif args.command == "get":
            print(json.dumps(get(args.pod_id), indent=2))
        elif args.command == "ssh":
            target = ssh_target(get(args.pod_id))
            if target is None:
                raise SystemExit("port 22 not mapped yet")
            print(" ".join(ssh_command(*target)))
        elif args.command == "terminate":
            terminate(args.pod_id)
    except PodError as e:                       # the standalone CLI: one line, exit 1, no traceback
        raise SystemExit(f"runpod: {e}") from None


if __name__ == "__main__":
    main()
