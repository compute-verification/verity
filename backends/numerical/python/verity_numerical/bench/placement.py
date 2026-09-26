"""Where a cell's prover and verifier ran: machine identity, and whether the two are separate machines on a routed link.

A container hostname says nothing about the machine: RunPod places several pods on one host, and two such pods share its
kernel, CPUs and public IP and can talk over a host-private bridge (red-team-flock's ruling, coordinator 2026-09-26 12:00Z: a
co-resident verifier is not a separate verifier).  An identity is a dict with any of

    machine_id     the RunPod machine id (the pod API's ``machineId``)
    public_ip      the pod's public IP (the API's ``publicIp``, or ``$RUNPOD_PUBLIC_IP`` on the pod)
    product_uuid   ``/sys/class/dmi/id/product_uuid``
    boot_id        ``/proc/sys/kernel/random/boot_id``: containers of one host share its kernel's
    cpu_model      the host CPU model (``/proc/cpuinfo``)
    pod_id, hostname

and a link ``{peer, peer_ip, source_ip}`` is how the prover reached the verifier.  :func:`separation` says why a pair is not
shown to be on separate machines: a shared machine id, product uuid, boot id or public IP; a link to a loopback, link-local or
private address (a host bridge) unless the machine ids are known and differ; or too little identity to tell (neither a
machine id or DMI uuid on both sides, nor a public IP and CPU model on both).

    python3 placement.py probe [--peer HOST:PORT] [--out FILE]     stdlib only: runs on a pod before a cell's job
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

SHARED_KEYS = ("machine_id", "product_uuid", "boot_id", "public_ip")
PROBE = "backends/numerical/python/verity_numerical/bench/placement.py"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


def _cpu_model() -> str | None:
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


#: Address space a path to which need not leave the host: RFC 1918, shared (100.64/10), IPv6 unique-local.
HOST_LOCAL_NETS = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "fc00::/7"))


def is_host_private(ip: str | None) -> bool:
    """Loopback, link-local, private (RFC 1918), shared or unique-local address space: a path that need not leave the host."""
    if not ip:
        return False
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.is_loopback or a.is_link_local or any(a.version == n.version and a in n for n in HOST_LOCAL_NETS)


def link_to(peer: str) -> dict[str, Any]:
    """The route to ``peer`` (HOST:PORT): its resolved address and the local source address the kernel picks for it."""
    host, _, port = peer.rpartition(":")
    out: dict[str, Any] = {"peer": peer, "peer_ip": None, "source_ip": None}
    try:
        out["peer_ip"] = socket.getaddrinfo(host, int(port or 0), proto=socket.IPPROTO_TCP)[0][4][0]
        with socket.socket(socket.AF_INET6 if ":" in out["peer_ip"] else socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((out["peer_ip"], int(port or 9)))
            out["source_ip"] = s.getsockname()[0]
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def probe(peer: str | None = None) -> dict[str, Any]:
    """This machine's identity as seen from inside its container (and the link to ``peer``)."""
    ident = {"pod_id": os.environ.get("RUNPOD_POD_ID"), "public_ip": os.environ.get("RUNPOD_PUBLIC_IP"),
             "product_uuid": _read("/sys/class/dmi/id/product_uuid"), "boot_id": _read("/proc/sys/kernel/random/boot_id"),
             "cpu_model": _cpu_model(), "hostname": socket.gethostname(), "datacenter": os.environ.get("RUNPOD_DC_ID")}
    if peer:
        ident["link"] = link_to(peer)
    return ident


def from_job(run_dir: Path) -> dict[str, Any]:
    """The identity a run recorded: its ``placement.json`` (the probe), else what ``job.json`` has (boot id, hostname)."""
    ident: dict[str, Any] = {}
    job = run_dir / "job.json"
    if job.is_file():
        env = json.loads(job.read_text()).get("environment") or {}
        ident.update({k: env.get(k) for k in ("boot_id", "hostname", "pod_id") if env.get(k)})
    pl = run_dir / "placement.json"
    if pl.is_file():
        ident.update({k: v for k, v in json.loads(pl.read_text()).items() if v})
    return ident


def from_runpod(machine: str) -> dict[str, Any]:
    """A pod's identity from the RunPod API, by machine name (the machines registry's pod id) or pod id."""
    from research.pods import registry, runpod

    pod_id = machine
    d = os.environ.get("RESEARCH_MACHINES_D")
    if d and Path(d).is_dir():
        entries, _errors = registry.load_dir(Path(d))
        pod_id = (entries.get(machine) or {}).get("pod_id") or machine
    pod = runpod.get(pod_id)
    return {"pod_id": pod.get("id"), "machine_id": pod.get("machineId") or None, "public_ip": pod.get("publicIp") or None}


def merge(*idents: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i in idents:
        out.update({k: v for k, v in (i or {}).items() if v})
    return out


def separation(prover: Mapping[str, Any], verifier: Mapping[str, Any], link: Mapping[str, Any] | None = None) -> list[str]:
    """Why ``prover`` and ``verifier`` are not shown to be separate machines on a routed link (empty: they are)."""
    out = []
    for k in SHARED_KEYS:
        if prover.get(k) and prover.get(k) == verifier.get(k):
            out.append(f"prover and verifier share {k} {prover[k]} (one machine)")
    ids_differ = any(prover.get(k) and verifier.get(k) and prover[k] != verifier[k] for k in ("machine_id", "product_uuid"))
    ip_cpu = all(x.get(k) for x in (prover, verifier) for k in ("public_ip", "cpu_model")) and prover["public_ip"] != verifier["public_ip"]
    if not out and not (ids_differ or ip_cpu):
        out.append("machine identity not recorded: neither a machine id / DMI product uuid on both pods, nor a public IP and CPU model "
                   "on both, so the verifier is not shown to be on another machine")
    peer = (link or {}).get("peer_ip")
    if peer and is_host_private(peer) and not (prover.get("machine_id") and verifier.get("machine_id")
                                               and prover["machine_id"] != verifier["machine_id"]):
        out.append(f"the prover reaches the verifier at {peer}, a host-private address (a bridge on one machine, not a routed path)")
    if peer and prover.get("public_ip") and peer == prover["public_ip"]:
        out.append(f"the prover reaches the verifier at its own public IP {peer} (hairpin on one machine)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="placement.py")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe")
    p.add_argument("--peer", default=None)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    doc = probe(a.peer or None)
    text = json.dumps(doc, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
