"""The mock datacenter: a `datacenter.Datacenter` declaration stood up on one Linux host as network namespaces, with the
real network certifier on each block's uplink, then the audit of what each certifier published.

The declaration is `datacenter.examples.leaf_spine`'s by default: four blocks (each a measured node of 8 GPUs, `rtx_node`), a
certifier on each block's uplink, two leaf switches joined by a spine, and one host outside every block. Any other
declaration whose blocks each have one uplink, to a certifier, and which has a device of kind `outside`, works too
(`--declaration FILE.py:NAME`, a `Datacenter` or a `Fleet`, or a function returning one; a fleet is validated and stood
up merged, `Fleet.merged`). `--fleet` is `datacenter.examples.two_sites`: two leaf-spines, `east` and `west`, joined by
a WAN between their spines, `--wan-latency-us` one way, with the outside host on the west site, so the east blocks'
links cross the WAN. `up` creates the namespaces of its netns plan (`graph.netns_plan`), all named `na-*`, with
netem on the fabric links (`--wan-jitter-us` of jitter on a WAN link). `down` stops every process in an `na-` namespace
and deletes the namespaces, and their veths with them; the root namespace never holds a veth. `status` lists them and
checks what can reach what, and in how long a round trip. All three run `ip` through `sudo -n`.

`run` runs one configuration on every block at once, each block with its own strategy (by default the blocks take
`honest`, `bits`, `late`, `extra` and `maintenance` in turn, so the default four blocks take the first four;
`--block b1=late` sets one). For each block, three processes, each in its
own namespace as the invoking user (`sudo ip netns exec NS setpriv ...`), on the host's monotonic clock from one origin:
`benchmarks/network_traces/active_replay.py`'s server, proxy and peer, on the namespaces' addresses, with the
certifier's own proxy unmodified:
- **the server** (`active_replay.Server`) inside the block, on its uplink's address at `SERVER_PORT`;
- **the proxy** (`verity.core.protocols.certifier.active.Proxy`) in the block's certifier namespace, its link on the
  certifier's fabric address at `LINK_PORT` and its upstream the block's server;
- **the peer** (`active_replay.Peer`) on the outside host, across the fabric.

The configuration is `test_active_replay`'s synthetic trace, three windows of 60 buckets of 100 ms, calibrated by the
library. A block's strategy is its server's: `honest`; `bits`, which writes each session's frames all at once or spread out
by a random bit (`active_replay.adversary_writes`); `late`, which holds a session in window 0 past every sync;
`extra`, honest but for one more event, at a position the Program doesn't have; or `maintenance`, an honest server
whose certifier is down for window 1. Each block is audited twice: by the
certifier's reference verifier, `audit.Audit`, on the raw and committed records (`active_replay.audit_configuration`),
and by the accounting layer's record check, `datacenter.check_records`, on the records as
`datacenter.observe.PortRecord`s against the ones the Program declares (`records`). For that check the declaration's
cadence is the certifiers' calibrated parameters (`records.cadence`), whatever it declared. The record check also runs
once on every block's records together, over the declaration's ports (a fleet's merged ones), with the account's charge
for the whole run. A
non-honest block's wire is compared with an honest block's of the same run (`active_replay.against_honest`).

With `--salt`, each certifier's per-window salt is on (`Proxy(salt=active.fresh_salt)`): its ingress side puts one salt
frame a window in the block's ingress record, carrying the placeholder and never the salt, and sends the salt to the
block's server on a connection of its own. Once a block's run is over, `run` makes the verifier's "reveal" call to its
proxy and keeps the salts in `<block>/salts.json`. The record check declares the ingress records of the windows it has
salts for, and the certifier's audit reads each window's salt frame and the salt's width and says where the salt frames
went (`active_replay.audit_salt`): one in each ingress window, none in egress.

Under `maintenance` the block is drained for window 1 (`active_replay.outage_plan`), and its proxy serves window 0,
closes the link and stops listening, then comes back for window 2 (`Proxy(start=2, position=p)`), and the peer
connects again: the certifier is inline, so for window 1 the link is dead and no record is published. Its records are
declared from its own drained plan, the Program its prover registers. The record check runs twice: against the
verifier's plan with `Maintenance(block, window 1)` (`verifier_plan`), which leaves that window out of the declared
records, and without it, which refuses both its port-windows "no record". Each comes with the account's charge for the
block (`charge`, `datacenter.account` on a profile that credits no work): without the plan, the refused port-windows
at the uplink's full capacity; with it, everything the block's devices hold. With `--salt` its proxy reveals the salts
of the windows it served, so the certifier's salt check, like its audit, finds nothing for window 1.

`seal` measures what a sealed pair link costs (the certifier's `Proxy(pair=…)`, its PROTOCOL.md "Sealed pair links"), in
bytes of frames per second per core, at each `--frame-bytes` (the calibrated 4096 and the tests' 512 by default), into
`<out>/seal.json`: the Python halves in one process (`seal_python`: `active.Sealer`, then `active.Opener`, each on its
CPU seconds); the same across the fabric, the first block's certifier sealing frames as fast as the link takes them to
the second's, which opens them (the roles `sealer` and `opener`, each its own core); the construction in C over
OpenSSL's SHA-512 (`seal_fast.c`, `fast_path`), whose units it first checks against the Python's, byte for byte; and
OpenSSL's own SHA-512 rate (`openssl speed`), the bound on any CPU path without SHA-512 instructions. It reports the CPU,
its SHA flags and the cores one 400 Gb/s uplink would take on each path. Every path seals under a public bench secret
(`BENCH_SECRET`), never a pair's.

    python mockdc.py up | status | down [--fleet --blocks 2]
    python mockdc.py run --out DIR [--salt] [--block b0=honest --block b1=late ...]
    python mockdc.py seal --out DIR [--frame-bytes 512 --frame-bytes 4096] [--megabytes 256]
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes.util
import dataclasses
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PATHS = [str(HERE), str(ROOT / "benchmarks" / "network_traces"), str(ROOT / "benchmarks" / "datacenter"), str(ROOT)]
sys.path[:0] = PATHS

import active_replay as ar  # noqa: E402
import graph  # noqa: E402
import records  # noqa: E402
from datacenter import (Carried, Datacenter, Fleet, Interval, Maintenance, Plan, Profile, Refused,  # noqa: E402
                        account, check_records)
from datacenter.examples import leaf_spine, two_sites  # noqa: E402
from verity.core.protocols.certifier import active, calibration  # noqa: E402
from verity.core.protocols.certifier.audit import Audit  # noqa: E402
from verity.core.protocols.certifier.grid import GridParams  # noqa: E402
from verity.ml.crypto import aead  # noqa: E402

T = 60
SERVER_PORT, LINK_PORT = 8000, 9000
STARTUP_S = 3.0
STRATEGIES = ("honest", "bits", "late", "extra", "maintenance")
OUTAGE_WINDOW = 1
RESHAPE = "bring the lab up in this shape (`mockdc.py down`, then `up` with the same declaration and options)"
# The fleet's options, which a shape built elsewhere (`workload.lab_run.shape_of`) may lack.
FLEET_OPTIONS = {"fleet": False, "wan_latency_us": 20_000, "wan_jitter_us": 2_000}


def option(args, name: str):
    return getattr(args, name, FLEET_OPTIONS[name])


def lab(args) -> Datacenter:
    if args.declaration:
        path, _, name = args.declaration.rpartition(":")
        spec = importlib.util.spec_from_file_location("declaration", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        dc = getattr(module, name)
        dc = dc() if callable(dc) else dc
    elif option(args, "fleet"):
        dc = two_sites(args.blocks, leaves=args.leaves, spines=args.spines, fabric_latency=args.fabric_latency_us / 1e6,
                       wan_latency=option(args, "wan_latency_us") / 1e6)
    else:
        dc = leaf_spine(args.blocks, leaves=args.leaves, spines=args.spines, fabric_latency=args.fabric_latency_us / 1e6)
    if isinstance(dc, Fleet):
        if problems := dc.validate():
            raise SystemExit("the fleet falls short of the model: " + "; ".join(problems))
        dc = dc.merged()
    graph.check(dc)
    if not any(d.kind == "outside" for d in dc.fabric):
        raise SystemExit(f"{dc.name} has no device of kind `outside` to run the peer on")
    return dc


def plan_of(args, dc: Datacenter) -> graph.NetnsPlan:
    return graph.netns_plan(dc, jitter=args.fabric_jitter_us / 1e6, wan_jitter=option(args, "wan_jitter_us") / 1e6)


def outside(dc: Datacenter) -> str:
    return next(d.name for d in dc.fabric if d.kind == "outside")


def sudo(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["sudo", "-n", *cmd], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"{' '.join(cmd)}: {r.stderr.strip() or r.returncode}")
    return r


def namespaces() -> list[str]:
    out = subprocess.run(["ip", "netns", "list"], capture_output=True, text=True, check=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.startswith(graph.NETNS_PREFIX)]


def differences(plan: graph.NetnsPlan) -> list[str]:
    """How the namespaces that are up differ from the plan's: one missing or not in it, or an interface without the
    plan's address (a lab brought up in another shape numbers its links differently)."""
    present, planned = set(namespaces()), {n.name for n in plan.namespaces}
    out = [f"{ns} missing" for ns in sorted(planned - present)]
    out += [f"{ns} not in the plan" for ns in sorted(present - planned)]
    for n in plan.namespaces:
        if n.name in present:
            have = {(f[1].split("@")[0], f[3]) for line in sudo(["ip", "-n", n.name, "-o", "-4", "addr", "show"]).stdout
                    .splitlines() if len(f := line.split()) > 3}
            out += [f"{n.name} {i.name} lacks {i.address}" for i in n.interfaces if (i.name, i.address) not in have]
    return out


def up(args) -> None:
    if existing := namespaces():
        raise SystemExit(f"{' '.join(existing)} exist: run `mockdc.py down` first")
    plan = plan_of(args, lab(args))
    for cmd in plan.up():
        r = sudo(cmd, check=False)
        if r.returncode:
            down(args)
            raise SystemExit(f"{' '.join(cmd)}: {r.stderr.strip() or r.returncode} (the namespaces are deleted again)")
    print(f"up: {len(plan.namespaces)} namespaces, {sum(len(n.interfaces) for n in plan.namespaces) // 2} links")
    status(args)


def down(args) -> None:
    names = namespaces()
    for ns in names:
        for pid in sudo(["ip", "netns", "pids", ns]).stdout.split():
            sudo(["kill", "-TERM", pid], check=False)
    for ns in names:
        sudo(["ip", "netns", "del", ns])
    if left := namespaces():
        raise SystemExit(f"still there: {' '.join(left)}")
    print(f"down: deleted {' '.join(names) or 'nothing'}")


def status(args) -> None:
    """Each namespace's interfaces and netem, then reachability, with the fastest round trip of a reachable address's
    three pings: the outside host reaches each certifier (across the WAN, in a fleet, for the far site's) and no block,
    a certifier reaches its block, and a block reaches nothing, its certifier included, since the certifier's guard
    drops whatever the block opens."""
    dc = lab(args)
    plan = plan_of(args, dc)
    width = max(len(n.name) for n in plan.namespaces)
    for n in plan.namespaces:
        ifs = ", ".join(f"{i.name} {i.address} -> {i.peer[0]}" + (f" [netem {' '.join(i.netem)}]" if i.netem else "")
                        for i in n.interfaces)
        print(f"{n.name:{width}} {n.role:9} forward={int(n.forward)}  {ifs}")
    if diff := differences(plan):
        raise SystemExit(f"{'; '.join(diff)}: {RESHAPE}")
    ext = outside(dc)
    ext_address = plan.namespace(ext).interfaces[0].address.split("/")[0]
    checks = []
    for b in (b.name for b in dc.blocks):
        _, cert = records.uplink(dc, b)
        checks += [(ext, plan.fabric_address(cert), True), (ext, plan.address(b, cert), False),
                   (cert, plan.address(b, cert), True), (b, plan.address(cert, b), False), (b, ext_address, False)]
    bad, width = 0, max(len(src) for src, _, _ in checks)
    for src, dst, want in checks:
        # The first packet's round trip includes resolving each hop's neighbour, so a reachable one gets three.
        r = sudo(["ip", "netns", "exec", graph.netns_name(src), "ping", "-c3" if want else "-c1", "-i0.2", "-W1", dst],
                 check=False)
        got, rtt = r.returncode == 0, re.search(r"= ([\d.]+)/", r.stdout)
        bad += got != want
        print(f"ping {src:{width}} -> {dst:12} {'reaches' if got else 'dropped'}"
              + (f" in {rtt[1]} ms" if got and rtt else "") + ("" if got == want else "  (WRONG)"))
    if bad:
        raise SystemExit(f"{bad} reachability check(s) wrong")


def trace() -> dict:
    """`test_active_replay`'s synthetic trace: three windows of T buckets, each with three sessions of four frames, one
    frame per bucket, the first ready 1 to 3 buckets after the anchor; each request arrives 30 ms into its anchor's
    bucket, the second one in two frames."""
    frames, arrival_ms = [], []
    for w in range(3):
        for j, (offset, latency) in enumerate(((0, 1), (10, 3), (20, 2))):
            anchor = w * T + offset
            frames += [{"session": f"s{w}.{j}", "index": i, "anchor": anchor, "ready": anchor + latency + i}
                       for i in range(4)]
            arrival_ms += [anchor * 100 + 30.0] * (2 if j == 1 else 1)
    frames.sort(key=lambda f: (f["ready"], f["anchor"], f["session"], f["index"]))
    return {"bucket_ms": 100, "window_buckets": T, "frame_bytes": 4096, "queue_capacity": 1 << 16,
            "configurations": [{"name": "synthetic/w0-3", "labels": {"run": "r0", "windows": [0, 3]},
                                "egress": [{"link": "responses", "frames": frames}],
                                "ingress": [{"link": "requests", "arrival_ms": arrival_ms}]}]}


def extra_plan(plan: dict) -> dict:
    """An honest server, each frame written in the bucket before it is ready, that also writes one more event on the
    first session, at the position after the Program's last."""
    release = {int(k): v for k, v in plan["model_release"].items()}
    keys = ar.program_keys(plan["sessions"], release, plan["windows"] * T)
    names = [s[0] for s in plan["sessions"]]
    writes = defaultdict(list)
    for p, (ready, _, rank, _) in enumerate(keys):
        writes[names[rank]].append([ready - 1, p])
    first = writes[names[keys[0][2]]]
    first.append([first[-1][0], len(keys)])
    return plan | {"strategy": "extra", "seed": 0, "writes": dict(writes)}


def strategy_plan(plan: dict, setup: ar.Setup, strategy: str) -> dict:
    if strategy == "honest":
        return plan
    if strategy == "extra":
        return extra_plan(plan)
    if strategy == "maintenance":
        return ar.outage_plan(plan, setup, OUTAGE_WINDOW) | {"strategy": "maintenance"}
    return ar.adversary_plan(plan, setup, strategy, seed=1)


def verifier_plan(dc: Datacenter, plans: dict) -> Plan:
    """The verifier's plan for the run's epoch: every block whose plan has an outage is out for maintenance in that
    window."""
    per, windows = dc.cadence.window_ticks, next(iter(plans.values()))["windows"]
    return Plan(Interval(0, windows * per), maintenance=[
        Maintenance(b, Interval(p["outage"]["window"] * per, (p["outage"]["window"] + 1) * per))
        for b, p in plans.items() if "outage" in p])


def charge(dc: Datacenter, plan: Plan, claims: list, block: str | None = None) -> dict:
    """The account of one block's claims under the verifier's plan for that block, or of every block's under the whole
    plan (`datacenter.account`, which charges every block its plan has out), on a profile that credits no work and
    registers no store: whether it is sound, and its covert budget, which is the carried windows' timing advice, every
    refused port-window at its uplink's full capacity, and everything a block holds if it is out for maintenance."""
    mine = plan if block is None else Plan(plan.epoch, maintenance=[m for m in plan.maintenance if m.block == block])
    acc = account(claims, Profile(dc, mine, {}, {}, {b.name: 0.0 for b in dc.blocks}, {}))
    ours = [x for x in acc.blocks if block is None or x.block == block]
    return {"sound": acc.sound, "covert_bits": acc.covert_bits, "timing_bits": acc.timing_bits,
            "refused_bits": acc.refused_bits, "serviced_bits": sum(v for b, v in acc.serviced.items() if b in
                                                                   {x.block for x in ours}),
            "carried_bits": sum(x.carried for x in ours)}


def describe(charged: dict) -> str:
    beyond = [f"{charged['refused_bits']:.3g} on refused windows"] * bool(charged["refused_bits"]) + \
        [f"{charged['serviced_bits']:.3g} held in maintenance"] * bool(charged["serviced_bits"])
    return (f"account {'sound' if charged['sound'] else 'NOT sound'}, covert budget {charged['covert_bits']:.3g} bits"
            + (f" ({', '.join(beyond)})" if beyond else ""))


# The three processes, each in its own namespace: active_replay's server, proxy and peer on the namespaces' addresses.
# Each prints one line when it is ready and reads the origin from its stdin; the server then reads one more line
# before it stops.

class Pipe:
    """A role's pipe to `run`, in the shape active_replay's roles take: what the role sends is its ready line, and what
    it receives is a line of its stdin, as JSON."""

    def send(self, value) -> None:
        print(json.dumps({"ready": True, "sent": value}), flush=True)

    def recv(self):
        return json.loads(sys.stdin.readline())


ROLES = {
    "server": lambda cfg, pipe: ar._server(cfg["plan"], cfg["params"], pipe, Path(cfg["out"]), cfg["host"],
                                           SERVER_PORT),
    "proxy": lambda cfg, pipe: ar._proxy(cfg["plan"], ar.Setup(cfg["report"]), (cfg["upstream"], SERVER_PORT), pipe,
                                         Path(cfg["out"]), host=cfg["host"], port=LINK_PORT,
                                         salt=cfg.get("salt", False)),
    "peer": lambda cfg, pipe: ar._peer(cfg["plan"], cfg["params"], LINK_PORT, pipe, Path(cfg["out"]), cfg["link"]),
    "sealer": lambda cfg, pipe: _sealer(cfg, pipe),
    "opener": lambda cfg, pipe: _opener(cfg, pipe),
}


def role(args) -> None:
    ar._quiet_runtime()
    asyncio.run(ROLES[args.name](json.loads(Path(args.config).read_text()), Pipe()))


class Process:
    """One role in its namespace, as the invoking user, its stderr in `<out>/<role>.log`."""

    def __init__(self, ns: str, name: str, cfg: dict) -> None:
        out = Path(cfg["out"])
        config = out / f"{name}.json"
        config.write_text(json.dumps(cfg))
        self.name, self.log = f"{ns}/{name}", out / f"{name}.log"
        cmd = ["sudo", "-n", "ip", "netns", "exec", ns, "setpriv", f"--reuid={os.getuid()}", f"--regid={os.getgid()}",
               "--clear-groups", "--", "env", f"PYTHONPATH={os.pathsep.join(PATHS)}", sys.executable, "-B", "-u",
               str(Path(__file__).resolve()), "role", name, str(config)]
        with open(self.log, "w") as err:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)

    def ready(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"{self.name} exited before it was ready ({self.proc.wait()}): {self.log.read_text()[-800:]}")
        return json.loads(line)

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def reveal(self) -> dict | None:
        """A salted proxy's answer to the verifier's "reveal" call once its run is over: every window's salt, or None
        if it exited without one."""
        self.send(json.dumps("reveal"))
        line = self.proc.stdout.readline()
        return json.loads(line)["sent"] if line else None

    def wait(self) -> int:
        code = self.proc.wait()
        if code:
            print(f"{self.name} exited {code}: {self.log.read_text()[-800:]}", file=sys.stderr)
        return code


def run(args) -> None:
    dc = lab(args)
    plan_ns = plan_of(args, dc)
    if diff := differences(plan_ns):
        raise SystemExit(f"{'; '.join(diff)}: {RESHAPE}")
    blocks = [b.name for b in dc.blocks]
    strategies = ({b.split("=")[0]: b.split("=")[1] for b in args.block} if args.block else
                  {b: STRATEGIES[k % len(STRATEGIES)] for k, b in enumerate(blocks)})
    if bad := {b: s for b, s in strategies.items() if b not in blocks or s not in STRATEGIES}:
        raise SystemExit(f"{bad}: a block of {blocks} and a strategy of {STRATEGIES}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    doc = trace()
    report = calibration.calibrate(doc)
    setup = ar.Setup(report)
    dc = dataclasses.replace(dc, cadence=records.cadence(setup))
    honest = ar.plan_configuration(doc["configurations"][0], setup)
    params = ar.wire_params(setup)
    (out / "report.json").write_text(json.dumps(report))
    ext = graph.netns_name(outside(dc))
    plans, procs = {}, {}
    for b, strategy in strategies.items():
        d = out / b
        d.mkdir(exist_ok=True)
        plans[b] = strategy_plan(honest, setup, strategy)
        (d / "plan.json").write_text(json.dumps(plans[b]))
        _, cert = records.uplink(dc, b)
        common = {"plan": plans[b], "params": params, "out": str(d)}
        server = Process(graph.netns_name(b), "server", common | {"host": plan_ns.address(b, cert)})
        server.ready()
        proxy = Process(graph.netns_name(cert), "proxy", common | {"report": report, "host": plan_ns.fabric_address(cert),
                                                                   "upstream": plan_ns.address(b, cert),
                                                                   "salt": args.salt})
        proxy.ready()
        peer = Process(ext, "peer", common | {"link": plan_ns.fabric_address(cert)})
        peer.ready()
        procs[b] = (server, proxy, peer)
    origin = time.monotonic() + STARTUP_S
    for ps in procs.values():
        for p in ps:
            p.send(repr(origin))
    print(f"running {strategies} from origin {origin:.3f}, {honest['windows']} windows of {T} buckets"
          + (", the certifiers' salt on" if args.salt else ""))
    codes = {}
    for b, (server, proxy, peer) in procs.items():
        if args.salt and (salts := proxy.reveal()) is not None:
            (out / b / "salts.json").write_text(json.dumps(salts))
        codes[b] = {"proxy": proxy.wait(), "peer": peer.wait()}
        server.send(json.dumps("stop"))
        codes[b]["server"] = server.wait()
        (out / b / "processes.json").write_text(json.dumps({"exit": codes[b], "origin": origin, "clock": "real",
                                                            "namespaces": [p.name for p in (server, proxy, peer)]}))
    summary = audit(out, dc, honest, plans, report, strategies, args.salt)
    summary["exit"] = codes
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if any(any(c.values()) for c in codes.values()):
        raise SystemExit("a process failed: see its log")


def audit(out: Path, dc: Datacenter, honest_plan: dict, plans: dict, report: dict, strategies: dict,
          salt: bool = False) -> dict:
    """Each block's audit by the certifier's verifier and by the accounting layer's record check, its timing, and its
    wire against an honest block's. With `salt`, each block's records are declared with the salts its proxy revealed
    (none if it revealed none, which the record check then refuses), and the certifier's audit reads each window's salt
    frame (`active_replay.audit_salt`).

    A block's records are declared from the honest plan, or, for a block down for maintenance, from its own drained
    plan, and checked against the verifier's plan (`verifier_plan`), which has that block out for its outage window, so
    the window owes no record. A block out for maintenance is also checked without it (`unplanned`). Each check comes
    with the account's charge for the block's claims (`charge`)."""
    setup = ar.Setup(report)
    verifier = verifier_plan(dc, plans)
    unplanned_plan = Plan(verifier.epoch)
    summary = {"declaration": dc.name, "cadence": {k: v for k, v in vars(dc.cadence).items() if k != "tick"},
               "wan": [[f"{e.a}-{e.b}", repr(e.capacity), repr(e.latency)] for e in dc.links if e.kind == "wan"],
               "plan": {"epoch": [verifier.epoch.start, verifier.epoch.end],
                        "maintenance": [[m.block, [m.during.start, m.during.end]] for m in verifier.maintenance]},
               "blocks": {}}
    honest = next((b for b, s in strategies.items() if s == "honest"), None)
    width = max(map(len, strategies.values()))
    every_published, every_declared = [], {}
    for b, strategy in strategies.items():
        a = ar.audit_configuration(plans[b], report, out / b, audit=Audit)
        (out / b / "audit.json").write_text(json.dumps(a))
        rows = [(r["link"], r["window"], r["raw"], r["committed"]) for r in a["rows"]]
        published = records.published(out / b, setup, dc, b)
        path = out / b / "salts.json"
        salts = {int(w): bytes.fromhex(s) for w, s in json.loads(path.read_text()).items()} if salt and path.exists() \
            else None
        mine = plans[b] if "outage" in plans[b] else honest_plan
        declared = records.declared(mine, setup, dc, b, salts, verifier)
        every_published += published
        every_declared |= declared
        claims = check_records(published, declared, dc.cadence)
        (out / b / "records.json").write_text(json.dumps([records.to_json(r) for r in published]))
        refused = [f"{c.on}: {c.why}" for c in claims if isinstance(c, Refused)]
        c = a["comparisons"]
        entry = {"strategy": strategy, "accepted": all(r[2] is None and r[3] is None for r in rows), "verdicts": rows,
                 "datacenter": {"accepted": not refused, "refused": refused,
                                "carried": sum(isinstance(x, Carried) for x in claims),
                                "advice_bits": sum(x.bits for x in claims if isinstance(x, Carried)),
                                "account": charge(dc, verifier, claims, b)},
                 "records_are_run_shapers": c["records_are_run_shapers"], "wire_is_the_records": c["wire_is_the_records"],
                 "released_not_the_programs": c["released_not_the_programs"],
                 "slack_ms": a["timing"]["slack_ms"],
                 "tick_late_ms_max": {k: v.get("max") for k, v in a["timing"]["tick_late_ms"].items()},
                 "ticks_over_slack": {k: v["over_slack"] for k, v in a["timing"]["tick_late_ms"].items()},
                 "wire_bucket_lag_ms": a["timing"]["wire_bucket_lag_ms"]}
        out_for = verifier.out(b, verifier.epoch)
        if out_for:
            bare = check_records(published, records.declared(mine, setup, dc, b, salts), dc.cadence)
            entry["datacenter"]["unplanned"] = {"accepted": not any(isinstance(x, Refused) for x in bare),
                                                "refused": [f"{x.on}: {x.why}" for x in bare if isinstance(x, Refused)],
                                                "account": charge(dc, unplanned_plan, bare, b)}
        if salt:
            entry["salt"] = a.get("salt")
        if honest is not None and b != honest and "outage" not in plans[b]:
            entry["against_honest"] = ar.against_honest(out / b, out / honest)
        summary["blocks"][b] = entry
        failed = [f"{link}[{w}] {raw or committed}" for link, w, raw, committed in rows if raw or committed]
        line = f"{b} {strategy:{width}} certifier audit {'ACCEPTED' if entry['accepted'] else 'REFUSED'}"
        line += f" ({', '.join(failed)})" if failed else ""
        if out_for:
            u = entry["datacenter"]["unplanned"]
            m = next(m for m in verifier.maintenance if m.block == b)
            line += ("; datacenter records without the plan " + ("ACCEPTED" if u["accepted"] else
                     f"REFUSED ({'; '.join(u['refused'])})") + f", {describe(u['account'])}"
                     + f"; with Maintenance({b}, {m.during}) " + ("ACCEPTED" if not refused else
                     f"REFUSED ({'; '.join(refused)})") + f", {describe(entry['datacenter']['account'])}")
        else:
            line += "; datacenter records " + ("ACCEPTED" if not refused else f"REFUSED ({'; '.join(refused)})")
        if "against_honest" in entry:
            line += f"; wire is {honest}'s: {entry['against_honest']['wire_is_honests']}"
        if salt:
            s = entry["salt"]
            line += "; salt: none revealed" if s is None else (
                f"; salt frames ingress {s['ingress_salt_frames']} egress {s['egress_salt_frames']}, salt check "
                f"{'ACCEPTED' if s['accepted'] else 'REFUSED'}, block received {s['block_received']}")
        print(line)
    claims = check_records(every_published, every_declared, dc.cadence)
    refused = [f"{c.on}: {c.why}" for c in claims if isinstance(c, Refused)]
    ports = sorted({repr(p) for p, _ in every_declared})
    summary["records"] = {"ports": ports, "of": len(dc.ports), "carried": sum(isinstance(x, Carried) for x in claims),
                          "refused": refused, "account": charge(dc, verifier, claims)}
    where = sorted({str(c.on) for c in claims if isinstance(c, Refused)})
    print(f"{dc.name}: datacenter records on {len(ports)} of its {len(dc.ports)} ports "
          + ("ACCEPTED" if not refused else f"REFUSED ({', '.join(where)})")
          + f", {describe(summary['records']['account'])}")
    return summary


# What a sealed pair link costs (`seal`).

SEAL_PORT = 9100
SEAL_FRAME_BYTES = (512, 4096)
#: what `seal` seals under: public, so nothing it measures or writes is a pair's key
BENCH_SECRET = hashlib.sha512(b"mockdc seal: a public bench secret, never a pair's").digest()
BENCH_SESSION = b"mockdc seal"
LINE_RATE_BPS = 400e9
CHUNK = 1024  # frames sealed, or opened, between two readings of the clock


def seal_grid(frame_bytes: int) -> GridParams:
    """The grid every path seals: the cost is per frame, so only the frame size matters."""
    return GridParams(buckets=600, allocation=4, frame_bytes=frame_bytes)


def frame_pool(frame_bytes: int) -> list[bytes]:
    rng = random.Random(frame_bytes)
    return [rng.randbytes(frame_bytes) for _ in range(64)]


def bench_keys() -> aead.Keys:
    return aead.keys(BENCH_SECRET, BENCH_SESSION, "a>b")


def rates(frame_bytes: int, frames: int, seconds: dict[str, float]) -> dict[str, float]:
    """Each timing's bytes of frames per second, and the cores one 400 Gb/s uplink takes at it."""
    out = {}
    for name, s in seconds.items():
        out[name] = s
        out[name.removesuffix("_s") + "_Bps"] = frames * frame_bytes / s if s > 0 else float("inf")
        out[name.removesuffix("_s") + "_cores_at_line_rate"] = LINE_RATE_BPS / 8 / out[name.removesuffix("_s") + "_Bps"]
    return out


def seal_python(frame_bytes: int, frames: int, cpu: Callable[[], float] = time.process_time) -> dict:
    """The certifier's two halves in this process, on one core: `frames` frames of `frame_bytes` bytes sealed
    (`active.Sealer`, the k-th at slot k of `seal_grid`), then opened (`active.Opener`), `CHUNK` at a time, each half
    on `cpu`'s seconds. A unit that doesn't open to its frame stops the bench."""
    grid, keys, pool = seal_grid(frame_bytes), bench_keys(), frame_pool(frame_bytes)
    sealer, opener = active.Sealer(keys, grid), active.Opener(keys, grid)
    seal_s = open_s = 0.0
    for k in range(0, frames, CHUNK):
        js = range(k, min(frames, k + CHUNK))
        c0 = cpu()
        units = [sealer.seal(*divmod(j, grid.allocation), pool[j % len(pool)]) for j in js]
        c1 = cpu()
        opened = [opener.open(u) for u in units]
        c2 = cpu()
        if opened != [pool[j % len(pool)] for j in js]:
            raise SystemExit(f"a sealed unit of {frame_bytes} bytes didn't open to its frame")
        seal_s, open_s = seal_s + c1 - c0, open_s + c2 - c1
    return {"frame_bytes": frame_bytes, "frames": frames} | rates(frame_bytes, frames, {"seal_cpu_s": seal_s,
                                                                                        "open_cpu_s": open_s})


def fast_vector(frame_bytes: int) -> str:
    """The SHA-512 of `seal_fast.c`'s first 64 units, as the Python seals them: frame k is `frame_bytes` bytes of k, at
    nonce (0, k, 0)."""
    k = bench_keys()
    return hashlib.sha512(b"".join(aead.seal(k, aead.nonce(0, j, 0), bytes([j]) * frame_bytes)
                                   for j in range(64))).hexdigest()


def build_fast(where: Path) -> Path | str:
    """`seal_fast.c` built against the host's libcrypto, or why it can't be."""
    cc, crypto = shutil.which("cc"), ctypes.util.find_library("crypto")
    if cc is None or crypto is None:
        return f"no {'C compiler' if cc is None else 'libcrypto'} on this host"
    exe = where / "seal_fast"
    built = subprocess.run([cc, "-O2", "-o", str(exe), str(HERE / "seal_fast.c"), f"-l:{crypto}"], capture_output=True,
                           text=True)
    return exe if built.returncode == 0 else f"seal_fast.c didn't build: {built.stderr[-400:]}"


def fast_path(exe: Path, frame_bytes: int, frames: int) -> dict:
    """The construction in C over OpenSSL's SHA-512 at `frame_bytes`, on one core: reported only once its units are the
    Python's, byte for byte (`fast_vector`), and every one opened."""
    k = bench_keys()
    ran = subprocess.run([str(exe), str(frame_bytes), str(frames), k.enc.hex(), k.mac.hex()], capture_output=True,
                         text=True)
    out = json.loads(ran.stdout) if ran.stdout else {}
    if ran.returncode or out.get("digest") != fast_vector(frame_bytes):
        raise SystemExit(f"seal_fast's units at {frame_bytes} bytes aren't the Python's ({ran.returncode}): {ran.stderr}")
    return {"frame_bytes": frame_bytes, "frames": frames} | rates(
        frame_bytes, frames, {n: out[n] for n in ("seal_cpu_s", "open_cpu_s", "seal_wall_s", "open_wall_s")})


def openssl_speed(sizes: tuple[int, ...] = (88, 1024, 8192), seconds: int = 2) -> dict:
    """OpenSSL's SHA-512 on one core, bytes hashed per second at each input size (`openssl speed -evp sha512 -mr`):
    at 88 bytes, one keystream block's input, a call is one compression, and the bound on any CPU path without SHA-512
    instructions is about 64 bytes of keystream per such call."""
    if shutil.which("openssl") is None:
        return {"skipped": "no openssl on this host"}
    out = {}
    for n in sizes:
        ran = subprocess.run(["openssl", "speed", "-mr", "-seconds", str(seconds), "-bytes", str(n), "-evp", "sha512"],
                             capture_output=True, text=True)
        rate = next((line.split(":")[3] for line in ran.stdout.splitlines() if line.startswith("+F:")), None)
        out[f"sha512_{n}B_Bps"] = float(rate) if rate is not None else None
    if out.get("sha512_88B_Bps"):
        out["keystream_bound_Bps"] = out["sha512_88B_Bps"] / 88 * 64
    version = subprocess.run(["openssl", "version"], capture_output=True, text=True).stdout.strip()
    return out | {"version": version}


def cpu_info() -> dict:
    """The host's CPU: its model, cores, load, and the flags that hash SHA in hardware (`sha_ni` is SHA-1 and SHA-256
    only on x86; `sha512` is the x86 SHA512 extension, or ARMv8.2's SHA-512 instructions)."""
    text = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    found = re.search(r"^(?:flags|Features)\s*:(.*)$", text, re.M)
    flags = set(found.group(1).split()) if found else set()
    model = re.search(r"^model name\s*:\s*(.*)$", text, re.M)
    return {"model": model.group(1) if model else None, "cores": os.cpu_count(), "load": os.getloadavg(),
            "flags": sorted(flags & {"sha_ni", "sha512", "sha2", "avx2", "avx512f", "avx512vl"})}


async def _sealer(cfg: dict, pipe: Pipe) -> None:
    """The pair's sending certifier: seals `frames` frames as fast as the link takes them and writes them to the
    opener across the fabric, then sends its CPU and wall seconds."""
    b, n = cfg["frame_bytes"], cfg["frames"]
    grid, pool = seal_grid(b), frame_pool(b)
    sealer = active.Sealer(bench_keys(), grid)
    pipe.send(None)
    pipe.recv()
    with socket.create_connection((cfg["link"], SEAL_PORT)) as s:
        c0, w0 = time.process_time(), time.perf_counter()
        for k in range(0, n, CHUNK):
            s.sendall(b"".join(sealer.seal(*divmod(j, grid.allocation), pool[j % len(pool)])
                               for j in range(k, min(n, k + CHUNK))))
        s.shutdown(socket.SHUT_WR)
        c1, w1 = time.process_time(), time.perf_counter()
    pipe.send({"cpu_s": c1 - c0, "wall_s": w1 - w0})


async def _opener(cfg: dict, pipe: Pipe) -> None:
    """The pair's receiving certifier: opens each unit it reads off the link, and sends how many opened, how many were
    refused, and its CPU and wall seconds from the connection on."""
    b = cfg["frame_bytes"]
    pool = frame_pool(b)
    opener = active.Opener(bench_keys(), seal_grid(b))
    size, buf, opened = opener.unit_bytes, bytearray(), 0
    with socket.create_server((cfg["host"], SEAL_PORT)) as srv:
        pipe.send(None)
        conn, _ = srv.accept()
        with conn:
            c0, w0 = time.process_time(), time.perf_counter()
            while data := conn.recv(1 << 20):
                buf += data
                whole = len(buf) - len(buf) % size
                for i in range(0, whole, size):
                    opened += opener.open(bytes(buf[i : i + size])) == pool[(opener.position - 1) % len(pool)]
                del buf[:whole]
            c1, w1 = time.process_time(), time.perf_counter()
    pipe.send({"opened": opened, "refused": opener.refused, "left": len(buf), "cpu_s": c1 - c0, "wall_s": w1 - w0})


def seal_fabric(args, out: Path, frame_bytes: int, frames: int) -> dict:
    """The first block's certifier seals to the second's across the lab's fabric, each in its namespace."""
    dc = lab(args)
    plan_ns = plan_of(args, dc)
    if diff := differences(plan_ns):
        raise SystemExit(f"{'; '.join(diff)}: {RESHAPE}")
    (_, a), (_, b) = (records.uplink(dc, x.name) for x in dc.blocks[:2])
    d = out / f"fabric-{frame_bytes}"
    d.mkdir(parents=True, exist_ok=True)
    common = {"out": str(d), "frame_bytes": frame_bytes, "frames": frames}
    opener = Process(graph.netns_name(b), "opener", common | {"host": plan_ns.fabric_address(b)})
    opener.ready()
    sealer = Process(graph.netns_name(a), "sealer", common | {"link": plan_ns.fabric_address(b)})
    sealer.ready()
    sealer.send(json.dumps("go"))
    sent, got = sealer.ready()["sent"], opener.ready()["sent"]
    if sealer.wait() or opener.wait() or got["refused"] or got["left"] or got["opened"] != frames:
        raise SystemExit(f"the fabric run at {frame_bytes} bytes didn't open every frame: {got}")
    return {"frame_bytes": frame_bytes, "frames": frames, "from": a, "to": b} | rates(
        frame_bytes, frames, {"seal_cpu_s": sent["cpu_s"], "seal_wall_s": sent["wall_s"], "open_cpu_s": got["cpu_s"],
                              "open_wall_s": got["wall_s"]})


def seal(args) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sizes = tuple(args.frame_bytes) or SEAL_FRAME_BYTES
    frames = {b: max(CHUNK, int(args.megabytes * 2**20) // b) for b in sizes}
    doc: dict = {"cpu": cpu_info(), "line_rate_bps": LINE_RATE_BPS, "openssl": openssl_speed()}
    doc["python"] = [seal_python(b, frames[b]) for b in sizes]
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_fast(Path(tmp))
        doc["c"] = [fast_path(exe, b, frames[b]) for b in sizes] if isinstance(exe, Path) else {"skipped": exe}
    doc["fabric"] = [seal_fabric(args, out, b, frames[b]) for b in sizes] if not args.no_fabric else \
        {"skipped": "--no-fabric"}
    (out / "seal.json").write_text(json.dumps(doc, indent=1))
    for path in ("python", "c", "fabric"):
        for r in doc[path] if isinstance(doc[path], list) else []:
            print(f"{path:6} B={r['frame_bytes']:5}: seal {r['seal_cpu_Bps'] / 1e6:7.1f} MB/s/core, open "
                  f"{r['open_cpu_Bps'] / 1e6:7.1f} MB/s/core; one 400 Gb/s uplink takes "
                  f"{max(r['seal_cpu_cores_at_line_rate'], r['open_cpu_cores_at_line_rate']):.0f} cores each end")
    print(f"openssl sha512: {doc['openssl']}; cpu: {doc['cpu']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("up", up), ("down", down), ("status", status), ("run", run), ("seal", seal)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--declaration", metavar="FILE.py:NAME", help="a Datacenter or Fleet, or a function returning "
                       "one, instead of datacenter.examples.leaf_spine's (which --blocks, --leaves, --spines and "
                       "--fabric-latency-us shape)")
        p.add_argument("--fleet", action="store_true", help="datacenter.examples.two_sites' fleet instead, with "
                       "--blocks, --leaves and --spines a site's, and --wan-latency-us its WAN's")
        p.add_argument("--blocks", type=int, default=4)
        p.add_argument("--leaves", type=int, default=2)
        p.add_argument("--spines", type=int, default=1)
        p.add_argument("--fabric-latency-us", type=int, default=2_000, help="each fabric hop's latency")
        p.add_argument("--fabric-jitter-us", type=int, default=500, help="each fabric hop's delay jitter")
        p.add_argument("--wan-latency-us", type=int, default=FLEET_OPTIONS["wan_latency_us"],
                       help="the WAN's one-way latency, with --fleet")
        p.add_argument("--wan-jitter-us", type=int, default=FLEET_OPTIONS["wan_jitter_us"],
                       help="a WAN link's delay jitter")
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--block", action="append", default=[], metavar="BLOCK=STRATEGY",
                           help=f"a block's server strategy, one of {STRATEGIES} (default: the blocks take them in turn)")
            p.add_argument("--salt", action="store_true", help="turn each certifier's per-window salt on")
        if name == "seal":
            p.add_argument("--out", required=True)
            p.add_argument("--frame-bytes", type=int, action="append", default=[],
                           help=f"a frame size to measure (default: {SEAL_FRAME_BYTES})")
            p.add_argument("--megabytes", type=float, default=256, help="the frames sealed on each path, in MiB")
            p.add_argument("--no-fabric", action="store_true", help="skip the run across the lab's fabric")
    r = sub.add_parser("role")
    r.set_defaults(fn=role)
    r.add_argument("name", choices=sorted(ROLES))
    r.add_argument("config")
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
