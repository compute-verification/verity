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

and a link ``{peer, peer_ip, source_ip}`` is how the prover reached the verifier.  :func:`assess` says why a pair is not shown
to be on separate machines: a shared machine id, product uuid or boot id; a link to a host bridge (loopback, link-local, the
docker pool 172.16/12), or to another private address unless the machine ids are known and differ; too little identity to tell
(neither a machine id or DMI uuid on both sides, nor a public IP and CPU model on both); or a shared public IP without the proof
a datacenter NAT needs (machine id, DMI uuid and boot id all recorded and all different, and a measured RTT of at least
:data:`RTT_FLOOR_MS`).  A shared public IP that has that proof is accepted and reported (``shared_public_ip``).

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
#: What a pair sharing a public IP (a datacenter's egress NAT) must record, on both pods and all different.
NAT_IDS = ("machine_id", "product_uuid", "boot_id")
#: The measured prover-verifier round trip a shared-IP pair must reach: well above the 0.03 ms of two containers on one host.
RTT_FLOOR_MS = 0.1
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
#: Docker's bridge pool: a container-to-container path on one host (RunPod's routed private network is 10.x).
HOST_BRIDGE_NET = ipaddress.ip_network("172.16.0.0/12")


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


def is_host_bridge(ip: str | None) -> bool:
    """Loopback, link-local or the docker bridge pool (172.16/12, e.g. the 172.24.0.x co-resident path): never a routed link."""
    if not ip:
        return False
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.is_loopback or a.is_link_local or (a.version == 4 and a in HOST_BRIDGE_NET)


def assess(prover: Mapping[str, Any], verifier: Mapping[str, Any], link: Mapping[str, Any] | None = None, rtt_ms: float | None = None,
           final: bool = True) -> dict[str, Any]:
    """Whether ``prover`` and ``verifier`` are shown to be separate machines on a routed link: ``{"problems": [...],
    "shared_public_ip": bool, "evidence": {...}}`` (no problems: they are).

    A shared public IP (a datacenter's egress NAT) is one machine unless every one of :data:`NAT_IDS` is recorded on both pods and
    differs, the measured prover-verifier RTT is at least :data:`RTT_FLOOR_MS` and the link is no host bridge (coordinator
    2026-09-26 22:00Z, red-team-flock's 12:00Z item 1 made explicit).  With ``final=False`` (a plan: only the RunPod API's machine
    ids are known) the machine ids must differ and the DMI uuid, boot id and RTT are left to the runs (``evidence.pending``)."""
    out: list[str] = []
    for k in ("machine_id", "product_uuid", "boot_id"):
        if prover.get(k) and prover.get(k) == verifier.get(k):
            out.append(f"prover and verifier share {k} {prover[k]} (one machine)")
    peer = (link or {}).get("peer_ip")
    shared = bool(prover.get("public_ip")) and prover.get("public_ip") == verifier.get("public_ip")
    evidence: dict[str, Any] = {}
    if shared:
        evidence = {k: [prover.get(k), verifier.get(k)] for k in NAT_IDS} | {"public_ip": prover["public_ip"], "rtt_ms": rtt_ms,
                                                                             "rtt_floor_ms": RTT_FLOOR_MS}
        needed = NAT_IDS if final else ("machine_id",)
        missing = [k for k in needed if not (prover.get(k) and verifier.get(k))]
        if missing:
            out.append(f"prover and verifier share public_ip {prover['public_ip']}: a shared (NAT) IP needs {', '.join(NAT_IDS)} recorded on "
                       f"both pods and all different; missing: {', '.join(missing)}")
        if final and not (isinstance(rtt_ms, (int, float)) and rtt_ms >= RTT_FLOOR_MS):
            out.append(f"prover and verifier share public_ip {prover['public_ip']}: the measured RTT {rtt_ms!r} ms is not at least "
                       f"{RTT_FLOOR_MS} ms (a co-resident pair measured about 0.03 ms)")
        if not final:
            evidence["pending"] = [k for k in ("product_uuid", "boot_id") if not (prover.get(k) and verifier.get(k))] + ["rtt_ms"]
    else:
        ids_differ = any(prover.get(k) and verifier.get(k) and prover[k] != verifier[k] for k in ("machine_id", "product_uuid"))
        ip_cpu = all(x.get(k) for x in (prover, verifier) for k in ("public_ip", "cpu_model"))
        if not out and not (ids_differ or ip_cpu):
            out.append("machine identity not recorded: neither a machine id / DMI product uuid on both pods, nor a public IP and CPU model "
                       "on both, so the verifier is not shown to be on another machine")
    machines_differ = bool(prover.get("machine_id") and verifier.get("machine_id") and prover["machine_id"] != verifier["machine_id"])
    if peer and is_host_bridge(peer):
        out.append(f"the prover reaches the verifier at {peer}, a host bridge or loopback address (one machine, not a routed path)")
    elif peer and is_host_private(peer) and not machines_differ:
        out.append(f"the prover reaches the verifier at {peer}, a private address, and the machine ids are not known to differ")
    if peer and prover.get("public_ip") and peer == prover["public_ip"] and not shared:
        out.append(f"the prover reaches the verifier at its own public IP {peer} (hairpin on one machine)")
    return {"problems": out, "shared_public_ip": shared, "evidence": evidence}


def separation(prover: Mapping[str, Any], verifier: Mapping[str, Any], link: Mapping[str, Any] | None = None, rtt_ms: float | None = None,
               final: bool = True) -> list[str]:
    """Why ``prover`` and ``verifier`` are not shown to be separate machines on a routed link (empty: they are; :func:`assess`)."""
    return assess(prover, verifier, link, rtt_ms, final)["problems"]


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
