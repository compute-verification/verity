#!/usr/bin/env python3
"""Minimal RunPod lifecycle for a measurement campaign (local machine only).

Reads the API key from ``~/.runpod/config.toml`` and never prints it.  Talks to
the REST API (``https://rest.runpod.io/v1``); the GraphQL endpoint is used only
for the account balance.

    python zk/campaign/runpod.py balance
    python zk/campaign/runpod.py gpus                      # price/stock for the SKUs we care about
    python zk/campaign/runpod.py create --name veritor-4090 --gpu "NVIDIA GeForce RTX 4090" --disk 100
    python zk/campaign/runpod.py list
    python zk/campaign/runpod.py get POD_ID
    python zk/campaign/runpod.py ssh POD_ID                # prints the ssh command once port 22 is mapped
    python zk/campaign/runpod.py wait-ssh POD_ID           # blocks until sshd answers
    python zk/campaign/runpod.py terminate POD_ID
    python zk/campaign/runpod.py watchdog POD_ID --hours 9 # detached: terminate after N hours

Every pod this script creates carries the label ``veritor-campaign`` in its
name so a stray pod is recognisable from the console.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REST = "https://rest.runpod.io/v1"
CONFIG = Path.home() / ".runpod" / "config.toml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
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
        raise PodError(f"no RunPod API config at {CONFIG}") from None


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


def balance() -> None:
    result = _graphql("{ myself { clientBalance spendLimit currentSpendPerHr } }")
    print(json.dumps(result["data"]["myself"]))


def gpus() -> None:
    result = _graphql(
        "{ gpuTypes { id memoryInGb secureCloud lowestPrice(input:{gpuCount:1}) "
        "{ uninterruptablePrice stockStatus } } }"
    )
    for gpu in result["data"]["gpuTypes"]:
        if not any(tag in gpu["id"] for tag in ("4090", "5090", "H100", "A100", "L40")):
            continue
        price = gpu["lowestPrice"] or {}
        print(
            f"{gpu['id']:<32} {gpu['memoryInGb']:>3}GB  on-demand ${price.get('uninterruptablePrice')}"
            f"  stock={price.get('stockStatus')}"
        )


def create(args: argparse.Namespace) -> None:
    public_key = (SSH_KEY.with_suffix(".pub")).read_text().strip()
    body = {
        "name": f"{args.name}-veritor-campaign",
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": args.gpu_count,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": public_key},
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


def ssh_command(host: str, port: int) -> list[str]:
    return [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"UserKnownHostsFile={Path.home() / '.runpod' / 'ssh' / 'veritor-campaign-known_hosts'}",
        "-o",
        "ServerAliveInterval=30",
        f"root@{host}",
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


def watchdog(pod_id: str, hours: float) -> None:
    """Detach a process that terminates the pod after ``hours`` even if this session dies."""

    log = Path.home() / ".runpod" / f"watchdog-{pod_id}.log"
    script = (
        "import time, sys, subprocess\n"
        f"time.sleep({int(hours * 3600)})\n"
        f"subprocess.run([sys.executable, {str(Path(__file__).resolve())!r}, 'terminate', {pod_id!r}])\n"
    )
    command = [sys.executable, "-c", script]
    if os.uname().sysname == "Darwin":
        command = ["caffeinate", "-is", *command]
    with log.open("ab") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"watchdog pid {process.pid}: terminates {pod_id} in {hours} h (log {log})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("balance")
    sub.add_parser("gpus")
    create_args(sub.add_parser("create"))
    sub.add_parser("list")
    for name in ("get", "ssh", "wait-ssh", "terminate"):
        p = sub.add_parser(name)
        p.add_argument("pod_id")
    w = sub.add_parser("watchdog")
    w.add_argument("pod_id")
    w.add_argument("--hours", type=float, default=9.0)
    args = parser.parse_args()

    try:
        if args.command == "balance":
            balance()
        elif args.command == "gpus":
            gpus()
        elif args.command == "create":
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
        elif args.command == "wait-ssh":
            host, port = wait_ssh(args.pod_id)
            print(" ".join(ssh_command(host, port)))
        elif args.command == "terminate":
            terminate(args.pod_id)
        elif args.command == "watchdog":
            watchdog(args.pod_id, args.hours)
    except PodError as e:                       # the standalone CLI: one line, exit 1, no traceback
        raise SystemExit(f"runpod: {e}") from None


if __name__ == "__main__":
    main()
