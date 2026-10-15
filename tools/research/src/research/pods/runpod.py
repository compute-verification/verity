#!/usr/bin/env python3
"""RunPod, which now hosts only the control pod (`vy-control-verity`): create, list, get, ssh and terminate, plus the REST and
GraphQL calls the budgets guard and `research notes` make. Nebius runs every node (`pods/nebius/`).

Reads the API key from ``~/.runpod/config.toml`` and never prints it.  Talks to
the REST API (``https://rest.runpod.io/v1``); the GraphQL endpoint is used only
for the account balance.

Where the laptop files are absent (a cloud VM with only the Cursor secrets) the
environment stands in, and a file that exists always wins:
``RUNPOD_API_KEY`` for the API key (with the default GraphQL URL);
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
import json
import os
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REST = "https://rest.runpod.io/v1"
CONFIG = Path.home() / ".runpod" / "config.toml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
KNOWN_HOSTS = Path.home() / ".runpod" / "ssh" / "veritor-campaign-known_hosts"
GRAPHQL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


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


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    cfg = _config()
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
    except urllib.error.HTTPError as error:
        detail = error.read()[:500].decode(errors="replace")
        what = "not found (terminated, or never existed)" if error.code == 404 else detail
        raise PodError(f"{method} {path} -> HTTP {error.code}: {what}", code=error.code, pod_id=pod_id) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
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
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


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
        # 22/tcp always; --expose-port adds raw TCP ports (a live verifier's listen port: lane live-verifier).  RunPod maps
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
        # [E10 NIGHT 2026-09-12] durable storage: the REST field `networkVolumeId` mounts the volume at /workspace (pod disk stays the
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
    for pod in _request("GET", "/pods") or []:
        print(
            f"{pod.get('id')}  {pod.get('name')}  {pod.get('desiredStatus')}  "
            f"{(pod.get('machine') or {}).get('gpuTypeId')}  ${pod.get('costPerHr')}/h  "
            f"ip={pod.get('publicIp')} ports={pod.get('portMappings')}"
        )


def get(pod_id: str) -> dict[str, Any]:
    pod = _request("GET", f"/pods/{pod_id}")
    return pod


def ssh_target(pod: dict[str, Any]) -> tuple[str, int] | None:
    ip = pod.get("publicIp")
    ports = pod.get("portMappings") or {}
    port = ports.get("22")
    if ip and port:
        return str(ip), int(port)
    return None


def ssh_command(host: str, port: int, *, user: str = "root") -> list[str]:
    return [
        "ssh",
        "-i",
        str(ssh_key()),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
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
