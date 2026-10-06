"""Where a cell's prover and verifier ran: machine identity, and whether the two are separate machines on a routed link.

A container hostname says nothing about the machine: RunPod places several pods on one host, and two such pods share its
kernel, CPUs and public IP and can talk over a host-private bridge (red-team-flock's ruling, coordinator 2026-09-26 12:00Z: a
co-resident verifier is not a separate verifier).  An identity is a dict with any of

    machine_id     the RunPod machine id (the pod API's ``machineId``)
    public_ip      the pod's public IP (the API's ``publicIp``, or ``$RUNPOD_PUBLIC_IP`` on the pod)
    product_uuid   ``/sys/class/dmi/id/product_uuid``
    boot_id        ``/proc/sys/kernel/random/boot_id``: containers of one host share its kernel's
    cpu_model      the host CPU model (``/proc/cpuinfo``)
    hypervisor     whether the CPU flags carry ``hypervisor`` (a virtual machine's guest)
    sys_vendor, product_name   ``/sys/class/dmi/id/``
    pod_id, hostname

and a link ``{peer, peer_ip, source_ip, rtt}`` is how the prover reached the verifier, ``rtt`` its TCP-connect round trip
(:func:`tcp_rtt`).  :func:`assess` says why a pair is not shown to be on separate machines: a shared machine id, product uuid or
boot id; a link to a host bridge (loopback, link-local, the docker pool 172.16/12), to the prover's own public IP, or to another
private address unless the machine ids are known and differ; too little identity to tell (neither a machine id or DMI uuid on
both sides, nor a public IP and CPU model on both); or a shared public IP without the proof a datacenter NAT needs
(:func:`_nat_exception`: both pods bare metal, their machine ids and boot ids (and DMI uuids, where readable) different, the link RunPod's global network, and a
TCP-connect RTT on it of at least :data:`RTT_FLOOR_MS`).  A shared public IP that has that proof is accepted and reported
(``shared_public_ip`` and its evidence).

    python3 placement.py probe [--peer HOST:PORT] [--rtt HOST:PORT] [--out FILE]     stdlib only: runs on a pod before a cell's job

The same checks are the deployment gate of the proof service's parties (:func:`gate`, :data:`GATE_RULES`): the developer's
firewall, and the auditor's network certifiers, memory challengers and challenger.  It runs on the machine, and in the cgroup,
of the role it checks, and refuses (exit 1, the reasons in its JSON verdict) unless

- every peer the role must be separate from (:func:`separate`: :func:`assess` finds nothing; with ``--link``, a routed link with
  a TCP-connect round trip of at least :data:`RTT_FLOOR_MS`) is, and every peer it must share a machine with, the one node a
  certifier or memory challenger guards, is that machine (:func:`same_machine`: inside the node's isolation boundary);
- the firewall is a VM guest;
- a role on dedicated cores may run on exactly ``--cores``, and no task outside its own cgroup may run on any of them
  (:func:`dedicated_cores` over :func:`read_tasks`' ``/proc``; kernel threads are not counted).

Every peer role in the rule is named (``--peer ROLE[:NAME]=FILE``, a probe record) or declared not yet deployed
(``--absent ROLE``, recorded in the verdict); a guarded node can never be absent.

    python3 placement.py gate --role ROLE [--self FILE|-] [--peer ROLE[:NAME]=FILE ...] [--absent ROLE ...]
                              [--link NAME=HOST:PORT ...] [--cores LIST] [--out FILE]
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
#: How many connects :func:`tcp_rtt` makes; the NAT exception needs every one to succeed (red-team-flock-3 R1).
RTT_CONNECTS = 30
#: The evidence's ``product_uuid`` for a pod whose DMI uuid could not be read (every RunPod container).
UNREADABLE = "unreadable"
RTT_METHOD = "tcp-connect: median and minimum of 30 kernel-level TCP connects on the session's route (no payload)"
#: RunPod's global network: a shared-IP pair's only accepted link is ``<verifier pod id>.runpod.internal`` resolving into 10/8.
RUNPOD_INTERNAL = ".runpod.internal"
GLOBAL_NET = ipaddress.ip_network("10.0.0.0/8")
#: ``/sys/class/dmi/id/sys_vendor`` prefixes of hypervisors' guests (red-team-flock-3 S1): a VM's DMI uuid and boot id are its own.
HYPERVISOR_VENDORS = ("QEMU", "KVM", "Xen", "VMware", "Microsoft Corporation", "innotek", "Parallels", "Red Hat", "OpenStack",
                      "Amazon EC2", "Google")
PROBE = "benchmarks/numerical/python/verity_numerical/bench/placement.py"


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


def _hypervisor_flag() -> bool | None:
    """Whether the CPU flags carry ``hypervisor`` (set by every hypervisor for its guests); ``None`` without /proc/cpuinfo flags."""
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.lower().startswith("flags"):
            return "hypervisor" in line.split(":", 1)[1].split()
    return None


def tcp_rtt(target: str, n: int = RTT_CONNECTS, timeout: float = 5.0) -> dict[str, Any]:
    """Round trips of ``n`` kernel-level TCP connects to ``target`` (HOST:PORT): ``connect()`` to the SYN-ACK, no payload, each socket
    closed at once.  Median and minimum in ms, the address connected to, and how many failed."""
    import statistics
    import time

    host, _, port = target.rpartition(":")
    out: dict[str, Any] = {"target": target, "target_ip": None, "n": 0, "failed": 0, "median_ms": None, "min_ms": None,
                           "method": RTT_METHOD}
    try:
        out["target_ip"] = socket.getaddrinfo(host, int(port), proto=socket.IPPROTO_TCP)[0][4][0]
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    ts = []
    family = socket.AF_INET6 if ":" in out["target_ip"] else socket.AF_INET
    for _ in range(n):
        s = socket.socket(family, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            t = time.perf_counter()
            s.connect((out["target_ip"], int(port)))
            ts.append((time.perf_counter() - t) * 1e3)
        except OSError:
            out["failed"] += 1
        finally:
            s.close()
    if ts:
        out.update(n=len(ts), median_ms=statistics.median(ts), min_ms=min(ts))
    return out


def _pod_env(key: str) -> str | None:
    """A RunPod container variable: the process environment, else the container's init process (``/proc/1/environ``), since a
    job started over ssh does not inherit the container's ``RUNPOD_*`` environment."""
    if os.environ.get(key):
        return os.environ[key]
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            k, sep, v = item.partition(b"=")
            if sep and k.decode(errors="replace") == key:
                return v.decode(errors="replace") or None
    except OSError:
        return None
    return None


def probe(peer: str | None = None, rtt: str | None = None) -> dict[str, Any]:
    """This machine's identity as seen from inside its container, the link to ``peer`` and, with ``rtt`` (HOST:PORT on the peer's
    route), the TCP-connect round trip to it."""
    ident = {"pod_id": _pod_env("RUNPOD_POD_ID"), "public_ip": _pod_env("RUNPOD_PUBLIC_IP"),
             "product_uuid": _read("/sys/class/dmi/id/product_uuid"), "boot_id": _read("/proc/sys/kernel/random/boot_id"),
             "cpu_model": _cpu_model(), "hypervisor": _hypervisor_flag(), "sys_vendor": _read("/sys/class/dmi/id/sys_vendor"),
             "product_name": _read("/sys/class/dmi/id/product_name"), "hostname": socket.gethostname(),
             "datacenter": _pod_env("RUNPOD_DC_ID")}
    if peer:
        ident["link"] = link_to(peer)
        if rtt:
            ident["link"]["rtt"] = tcp_rtt(rtt)
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
        ident = merge(ident, json.loads(pl.read_text()))
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


def rtt_target(peer: str | None) -> str | None:
    """Where the prover times its TCP connects for a global-network peer: the verifier pod's sshd on the same name and route.  Not
    the session port: the verifiers count every connection on it as a session."""
    host = (peer or "").rpartition(":")[0]
    return f"{host}:22" if host.endswith(RUNPOD_INTERNAL) else None


def remote_probe(machine: str, peer: str | None = None) -> dict[str, Any]:
    """:func:`probe` run on a pod over ``research pods ssh`` (this file on stdin), so a plan can see its DMI identity and RTT."""
    import subprocess

    args = ["probe", *(["--peer", peer] if peer else []), *(["--rtt", rtt_target(peer)] if rtt_target(peer) else [])]
    r = subprocess.run(["research", "pods", "ssh", machine, "--", "python3", "-", *args], input=Path(__file__).read_text(),
                       capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(f"probe on {machine}: rc {r.returncode}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout[r.stdout.index("{"):])


def merge(*idents: Mapping[str, Any] | None) -> dict[str, Any]:
    """Later identities override earlier ones; a value not recorded (``None``, empty) never overrides (``False`` is recorded)."""
    out: dict[str, Any] = {}
    for i in idents:
        out.update({k: v for k, v in (i or {}).items() if v is not None and v != "" and v != {}})
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


def is_hypervisor_vendor(vendor: str | None) -> bool:
    return bool(vendor) and any(str(vendor).strip().lower().startswith(v.lower()) for v in HYPERVISOR_VENDORS)


def _nat_exception(prover: Mapping[str, Any], verifier: Mapping[str, Any], link: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Why a pair sharing a public IP does not earn the NAT exception (coordinator 22:00Z; red-team-flock-3 S1-S5, 22:05Z), and its
    evidence: the machine id and boot id recorded and different, and the DMI uuid too unless both pods are bare metal (where it
    is compared only if read, and the evidence says :data:`UNREADABLE` where it was not); both pods bare metal (no ``hypervisor`` CPU flag, a DMI
    ``sys_vendor`` that is no hypervisor's); the link ``<verifier pod id>.runpod.internal`` resolving into 10/8; the TCP-connect RTT
    on that route, all :data:`RTT_CONNECTS` connects succeeding, with a median of at least :data:`RTT_FLOOR_MS`."""
    ip, out = prover["public_ip"], []
    pre = f"prover and verifier share public_ip {ip}"
    metal = True
    for role, x in (("prover", prover), ("verifier", verifier)):
        if x.get("hypervisor") is None or not x.get("sys_vendor"):
            metal = False
            out.append(f"{pre}: the {role} pod's hypervisor flag and DMI sys_vendor are not recorded (a VM's uuid and boot id are its own)")
        elif x["hypervisor"] is True or is_hypervisor_vendor(x["sys_vendor"]):
            metal = False
            out.append(f"{pre}: the {role} pod is a virtual machine (hypervisor flag {x['hypervisor']}, sys_vendor {x['sys_vendor']!r}): "
                       "two VMs of one host would pass the id checks")
    # Bare-metal containers share their host's kernel, so boot_id names the host; RunPod containers cannot read the DMI uuid
    # (coordinator 22:40Z).  A uuid that is read is still compared (``assess`` refuses a shared one).
    need = [k for k in NAT_IDS if k != "product_uuid" or not metal]
    missing = [k for k in need if not (prover.get(k) and verifier.get(k))]
    if missing:
        out.append(f"{pre}: the NAT exception needs {', '.join(need)} recorded on both pods and all different"
                   f"{'' if metal else ' (product_uuid may be unreadable only on bare metal)'}; missing: {', '.join(missing)}")
    host = (link.get("peer") or "").rpartition(":")[0]
    want = f"{verifier.get('pod_id')}{RUNPOD_INTERNAL}" if verifier.get("pod_id") else None
    peer_ip = link.get("peer_ip")
    try:
        in_global = peer_ip is not None and ipaddress.ip_address(peer_ip) in GLOBAL_NET
    except ValueError:
        in_global = False
    if not (want and host == want and in_global):
        out.append(f"{pre}: the link must be RunPod's global network, {want or '<verifier pod id>' + RUNPOD_INTERNAL} resolving into "
                   f"{GLOBAL_NET}; it is {host or '?'} at {peer_ip}")
    rtt = link.get("rtt") or {}
    ok_rtt = (str(rtt.get("method", "")).startswith("tcp-connect") and isinstance(rtt.get("median_ms"), (int, float))
              and rtt.get("target_ip") == peer_ip and rtt["median_ms"] >= RTT_FLOOR_MS)
    if not ok_rtt:
        out.append(f"{pre}: no TCP-connect RTT of at least {RTT_FLOOR_MS} ms measured on the link's own route (median "
                   f"{rtt.get('median_ms')!r} ms to {rtt.get('target_ip')!r}, method {rtt.get('method')!r}); a co-resident pair measured "
                   "about 0.03 ms")
    if rtt and not (rtt.get("failed") == 0 and isinstance(rtt.get("n"), int) and rtt["n"] >= RTT_CONNECTS):
        out.append(f"{pre}: the RTT needs all {RTT_CONNECTS} TCP connects to succeed; {rtt.get('n')!r} did and {rtt.get('failed')!r} failed")
    evidence = {k: [prover.get(k), verifier.get(k)] for k in NAT_IDS}
    evidence["product_uuid"] = [x or UNREADABLE for x in evidence["product_uuid"]]
    evidence.update({"public_ip": ip, "bare_metal": {role: {k: x.get(k) for k in ("hypervisor", "sys_vendor", "product_name", "cpu_model")}
                                                     for role, x in (("prover", prover), ("verifier", verifier))},
                     "link": {"name": host or None, "address": peer_ip}, "rtt": {k: rtt.get(k) for k in ("method", "median_ms", "min_ms", "n",
                                                                                                     "failed", "target", "target_ip")},
                     "rtt_floor_ms": RTT_FLOOR_MS})
    return out, evidence


def assess(prover: Mapping[str, Any], verifier: Mapping[str, Any], link: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whether ``prover`` and ``verifier`` are shown to be separate machines on a routed link: ``{"problems": [...],
    "shared_public_ip": bool, "evidence": {...}}`` (no problems: they are).  A shared public IP (a datacenter's egress NAT) is one
    machine unless the pair earns the NAT exception (:func:`_nat_exception`); ``evidence`` is what it rests on."""
    link = link or {}
    out: list[str] = []
    for k in NAT_IDS:
        if prover.get(k) and prover.get(k) == verifier.get(k):
            out.append(f"prover and verifier share {k} {prover[k]} (one machine)")
    peer = link.get("peer_ip")
    shared = bool(prover.get("public_ip")) and prover.get("public_ip") == verifier.get("public_ip")
    evidence: dict[str, Any] = {}
    if shared:
        why, evidence = _nat_exception(prover, verifier, link)
        out += why
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
    if peer and prover.get("public_ip") and peer == prover["public_ip"]:
        out.append(f"the prover reaches the verifier at its own public IP {peer} (a hairpin, never a separate machine's route)")
    return {"problems": out, "shared_public_ip": shared, "evidence": evidence}


def separation(prover: Mapping[str, Any], verifier: Mapping[str, Any], link: Mapping[str, Any] | None = None) -> list[str]:
    """Why ``prover`` and ``verifier`` are not shown to be separate machines on a routed link (empty: they are; :func:`assess`)."""
    return assess(prover, verifier, link)["problems"]


#: The deployment gate's rule per role (the proof-service architecture of 6 Oct 2026): the peer roles it must share a machine
#: with (exactly one: the node it guards, inside whose isolation boundary it runs), those it must be separate from (every one,
#: at least one each unless absent), whether it must be a VM guest, and whether it runs on dedicated cores.
GATE_RULES: dict[str, dict[str, Any]] = {
    "firewall": {"same": (), "separate": ("worker", "challenger"), "vm": True, "cores": True},
    "certifier": {"same": ("worker",), "separate": ("firewall", "challenger"), "vm": False, "cores": True},
    "memory-challenger": {"same": ("worker",), "separate": ("firewall", "challenger"), "vm": False, "cores": True},
    "challenger": {"same": (), "separate": ("firewall", "worker"), "vm": False, "cores": False},
}
PEER_ROLES = ("worker", "firewall", "challenger")
GATE_SCHEMA = "verity/placement-gate/v1"
PROC = "/proc"


def cpu_list(text: str | None) -> frozenset[int]:
    """A CPU list (``0-3,8``: ``Cpus_allowed_list``, a cpuset, ``taskset -c``) as a set; a malformed one raises ``ValueError``."""
    out: set[int] = set()
    for part in (text or "").replace(" ", "").split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(out)


def cpu_text(cpus: frozenset[int] | set[int]) -> str:
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _cgroup(lines: list[str]) -> str | None:
    """A task's cgroup: the v2 path (``0::/...``), else its v1 cpuset hierarchy's."""
    v1 = None
    for line in lines:
        hid, _, rest = line.partition(":")
        ctrl, _, path = rest.partition(":")
        if hid == "0" and not ctrl:
            return path
        if "cpuset" in ctrl.split(","):
            v1 = path
    return v1


def read_tasks(proc: str = PROC) -> dict[str, dict[str, Any]]:
    """Every task (thread) under ``proc`` by tid: its process, name, allowed CPUs (``Cpus_allowed_list``), cgroup and whether it
    is a kernel thread (kthreadd or its child).  A task that exits while it is read is left out."""
    out: dict[str, dict[str, Any]] = {}
    for status in Path(proc).glob("[0-9]*/task/[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
            cgroup = _cgroup(status.with_name("cgroup").read_text().splitlines())
        except OSError:
            continue
        f = {k: v.strip() for k, v in fields.items()}
        pid = status.parent.parent.parent.name
        out[status.parent.name] = {"pid": pid, "name": f.get("Name"), "cpus": f.get("Cpus_allowed_list"), "cgroup": cgroup,
                                   "kthread": pid == "2" or f.get("PPid") == "2" or f.get("Kthread") == "1"}
    return out


def dedicated_cores(cores: frozenset[int], tasks: Mapping[str, Mapping[str, Any]], own: str) -> tuple[list[str], dict[str, Any]]:
    """Why ``cores`` are not the role's alone (empty: they are), and the evidence: its own task ``own`` (a tid of ``tasks``) may run
    on exactly ``cores``, in a cgroup of its own (not the root), and no task outside that cgroup or below it may run on any of
    them; a task whose allowed CPUs cannot be read counts as on them.  Kernel threads are not counted: no cgroup moves them."""
    out: list[str] = []
    ev: dict[str, Any] = {"declared": cpu_text(cores), "own": None, "cgroup": None, "tasks": len(tasks), "kernel_threads": 0,
                          "intruders": {}}
    if not cores:
        out.append("no dedicated cores declared (--cores)")
    me = tasks.get(own)
    scope = ((me or {}).get("cgroup") or "").rstrip("/")
    if me is None or not scope:
        out.append(f"this process (task {own}) is not in a cgroup of its own: {(me or {}).get('cgroup')!r}")
        return out, ev
    try:
        mine = cpu_list(me.get("cpus"))
    except ValueError:
        mine = frozenset()
    ev.update(own=cpu_text(mine), cgroup=scope)
    if mine != cores:
        out.append(f"this process may run on CPUs {cpu_text(mine) or '?'}, not exactly the declared {cpu_text(cores)}")
    intruders: dict[str, dict[str, Any]] = ev["intruders"]
    for t in tasks.values():
        cg = t.get("cgroup") or ""
        if t.get("kthread"):
            ev["kernel_threads"] += 1
            continue
        if cg == scope or cg.startswith(scope + "/"):
            continue
        try:
            on = cpu_list(t.get("cpus")) & cores if t.get("cpus") else cores
        except ValueError:
            on = cores
        if on:
            x = intruders.setdefault(str(t.get("pid")), {"name": t.get("name"), "cgroup": cg, "cpus": set()})
            x["cpus"] |= on
    for x in intruders.values():
        x["cpus"] = cpu_text(x["cpus"])
    if intruders:
        shown = "; ".join(f"{x['name']} (pid {pid}, {x['cgroup'] or 'no cgroup'}) on {x['cpus']}"
                          for pid, x in sorted(intruders.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)[:5])
        more = f"; and {len(intruders) - 5} more" if len(intruders) > 5 else ""
        out.append(f"{len(intruders)} process(es) outside cgroup {scope} may run on the dedicated CPUs {cpu_text(cores)}: {shown}{more}")
    return out, ev


def same_machine(me: Mapping[str, Any], node: Mapping[str, Any]) -> list[str]:
    """Why ``me`` is not shown to run on ``node``'s machine, inside its isolation boundary (empty: it is): every identity both
    record (machine id, DMI product uuid, boot id) equal, and at least one recorded on both.  A VM guest's uuid and boot id are its
    own, so a process on the guest's host, outside the guest, matches neither."""
    both = [k for k in NAT_IDS if me.get(k) and node.get(k)]
    out = [f"its {k} {me[k]} is not the node's {node[k]}" for k in both if me[k] != node[k]]
    if not both:
        out.append(f"none of {', '.join(NAT_IDS)} is recorded on both, so it is not shown to be on the node's machine")
    return out


def separate(me: Mapping[str, Any], peer: Mapping[str, Any], link: Mapping[str, Any] | None = None) -> list[str]:
    """Why ``me`` and ``peer`` are not shown to be separate machines (empty: they are): :func:`assess`'s problems, and with a
    ``link`` (:func:`link_to` with its :func:`tcp_rtt`) every TCP connect succeeding at a median of at least :data:`RTT_FLOOR_MS`."""
    out = [p.replace("prover and verifier", "they").replace("the prover reaches the verifier", "it reaches the peer")
           for p in assess(me, peer, link)["problems"]]
    if link is not None:
        rtt = link.get("rtt") or {}
        med = rtt.get("median_ms")
        if not (isinstance(med, (int, float)) and med >= RTT_FLOOR_MS and rtt.get("failed") == 0 and rtt.get("n")):
            out.append(f"the link to it ({link.get('peer')}) has no TCP-connect round trip of at least {RTT_FLOOR_MS} ms with every "
                       f"connect succeeding (median {med!r} ms, {rtt.get('n')!r} connected, {rtt.get('failed')!r} failed)")
    return out


def gate(role: str, me: Mapping[str, Any], peers: list[tuple[str, str, Mapping[str, Any]]], absent: tuple[str, ...] = (),
         cores: frozenset[int] | None = None, tasks: Mapping[str, Mapping[str, Any]] | None = None, own: str | None = None,
         links: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The deployment verdict for ``role`` (a :data:`GATE_RULES` key) on the machine ``me`` (its :func:`probe`): ``peers`` are
    ``(peer role, name, probe record)``, ``absent`` the peer roles not deployed yet, ``links`` measured links by peer name, and
    ``cores``, ``tasks`` (:func:`read_tasks`) and ``own`` (this process's tid) the dedicated-cores check's inputs."""
    rule, links = GATE_RULES[role], dict(links or {})
    named = {want: [n for r, n, _ in peers if r == want] for want in PEER_ROLES}
    problems: list[str] = []
    for r in rule["same"]:
        if len(named[r]) != 1:
            problems.append(f"a {role} guards exactly one {r} (--peer {r}=FILE): {len(named[r])} named")
    for r in rule["separate"]:
        if not named[r] and r not in absent:
            problems.append(f"no {r} named: name every one (--peer {r}[:NAME]=FILE), or --absent {r} while none is deployed")
        if named[r] and r in absent:
            problems.append(f"{r} is both named and declared absent")
    for r in absent:
        if r not in rule["separate"]:
            problems.append(f"--absent {r}: a {role}'s rule needs {'its ' + r if r in rule['same'] else 'no ' + r}")
    checked = []
    for r, name, ident in peers:
        if r in rule["same"]:
            why, relation = same_machine(me, ident), "same machine"
        elif r in rule["separate"]:
            why, relation = separate(me, ident, links.pop(name, None)), "separate machine"
        else:
            why, relation = [f"a {role}'s rule names no {r}"], None
        problems += [f"{name} ({r}): {w}" for w in why]
        checked.append({"name": name, "role": r, "relation": relation, "problems": why})
    problems += [f"--link {name}: no peer of that name is checked separate" for name in links]
    if rule["vm"] and not (me.get("hypervisor") is True or is_hypervisor_vendor(me.get("sys_vendor"))):
        problems.append(f"a {role} runs in a VM guest of its own: this machine's hypervisor flag is {me.get('hypervisor')!r} and "
                        f"its DMI sys_vendor {me.get('sys_vendor')!r}")
    core_ev = None
    if rule["cores"] or cores is not None:
        why, core_ev = dedicated_cores(cores or frozenset(), tasks or {}, own or "")
        problems += [f"cores: {w}" for w in why]
    return {"schema": GATE_SCHEMA, "role": role, "passed": not problems, "problems": problems, "self": dict(me), "peers": checked,
            "absent": sorted(absent), "cores": core_ev}


def _gate_main(a: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    def load(path: str) -> dict[str, Any]:
        try:
            return json.loads(sys.stdin.read() if path == "-" else Path(path).read_text())
        except (OSError, ValueError) as e:
            ap.error(f"{path}: {type(e).__name__}: {e}")

    peers, names = [], set()
    for spec in a.peer:
        who, sep, path = spec.partition("=")
        r, _, name = who.partition(":")
        name = name or r
        if not sep or r not in PEER_ROLES or name in names:
            ap.error(f"--peer {spec}: ROLE[:NAME]=FILE, ROLE one of {', '.join(PEER_ROLES)}, each NAME once")
        names.add(name)
        peers.append((r, name, load(path)))
    links = {}
    for spec in a.link:
        name, sep, target = spec.partition("=")
        if not sep:
            ap.error(f"--link {spec}: NAME=HOST:PORT")
        links[name] = {**link_to(target), "rtt": tcp_rtt(target)}
    try:
        cores = cpu_list(a.cores) if a.cores is not None else None
    except ValueError:
        ap.error(f"--cores {a.cores}: a CPU list such as 124-127")
    rule = GATE_RULES[a.role]
    needs_cores = rule["cores"] or cores is not None
    verdict = gate(a.role, load(a.self_) if a.self_ else probe(), peers, tuple(a.absent), cores,
                   read_tasks(PROC) if needs_cores else None, str(os.getpid()), links)
    text = json.dumps(verdict, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    sys.stdout.write(text)
    for p in verdict["problems"]:
        sys.stderr.write(f"placement gate: {a.role}: {p}\n")
    return 0 if verdict["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="placement.py")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe")
    p.add_argument("--peer", default=None)
    p.add_argument("--rtt", default=None, help="HOST:PORT on the peer's route: the TCP-connect RTT target")
    p.add_argument("--out", default=None)
    g = sp.add_parser("gate", help="the deployment gate: exit 0 when ROLE may run here, 1 with the reasons when it may not")
    g.add_argument("--role", required=True, choices=sorted(GATE_RULES))
    g.add_argument("--self", dest="self_", default=None, metavar="FILE|-", help="this machine's probe record; default: probe it now")
    g.add_argument("--peer", action="append", default=[], metavar="ROLE[:NAME]=FILE", help="a peer's probe record")
    g.add_argument("--absent", action="append", default=[], choices=PEER_ROLES, help="a peer role not deployed yet")
    g.add_argument("--link", action="append", default=[], metavar="NAME=HOST:PORT", help="measure the route to a separate peer")
    g.add_argument("--cores", default=None, metavar="LIST", help="the role's dedicated CPUs, e.g. 124-127")
    g.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        return _gate_main(a, ap)
    doc = probe(a.peer or None, a.rtt or None)
    text = json.dumps(doc, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
