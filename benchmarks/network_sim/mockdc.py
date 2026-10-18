"""The mock datacenter: a `datacenter.Datacenter` declaration stood up on one Linux host as network namespaces, with the
real network certifier on each block's uplink, then the audit of what each certifier published.

The declaration is `graph.leaf_spine`'s by default: four blocks (each a measured node of 8 GPUs, `graph.rtx_node`), a
certifier on each block's uplink, two leaf switches joined by a spine, and one host outside every block. Any other
declaration whose blocks each have one uplink, to a certifier, and which has a device of kind `outside`, works too
(`--declaration FILE.py:NAME`, a `Datacenter` or a function returning one). `up` creates the namespaces of its netns
plan (`graph.netns_plan`), all named `na-*`, with netem on the fabric links. `down` stops every process in an `na-`
namespace and deletes the namespaces, and their veths with them; the root namespace never holds a veth. `status` lists
them and checks what can reach what. All three run `ip` through `sudo -n`.

`run` runs one configuration on every block at once, each block with its own strategy (by default the blocks take
`honest`, `bits`, `late` and `extra` in turn; `--block b1=late` sets one). For each block, three processes, each in its
own namespace as the invoking user (`sudo ip netns exec NS setpriv ...`), on the host's monotonic clock from one origin,
built from `benchmarks/network_traces/active_replay.py`'s classes and the certifier's own, unmodified:
- **the server** (`active_replay.Server`) inside the block, on its uplink's address;
- **the proxy** (`verity.core.protocols.certifier.active.Proxy`) in the block's certifier namespace, its link on the
  certifier's fabric address and its upstream the block's server;
- **the peer** (`active_replay.Peer`) on the outside host, across the fabric.

The configuration is `test_active_replay`'s synthetic trace, three windows of 60 buckets of 100 ms, calibrated by the
library. A block's strategy is its server's: `honest`; `bits`, which writes each session's frames all at once or spread out
by a random bit (`active_replay.adversary_writes`); `late`, which holds a session in window 0 past every sync; or
`extra`, honest but for one more event, at a position the Program doesn't have. Each block is audited twice: by the
certifier's reference verifier, `audit.Audit`, on the raw and committed records (`active_replay.audit_configuration`),
and by the accounting layer's record check on the records as `datacenter.observe.Record`s against the ones the Program
declares (`records`). A non-honest block's wire is compared with an honest block's of the same run
(`active_replay.against_honest`).

    python mockdc.py up | status | down
    python mockdc.py run --out DIR [--block b0=honest --block b1=late ...]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PATHS = [str(HERE), str(ROOT / "benchmarks" / "network_traces"), str(ROOT / "benchmarks" / "datacenter"), str(ROOT)]
sys.path[:0] = PATHS

import active_replay as ar  # noqa: E402
import graph  # noqa: E402
import records  # noqa: E402
from datacenter import Carried, Datacenter, Refused  # noqa: E402
from verity.core.protocols.certifier import calibration  # noqa: E402
from verity.core.protocols.certifier.active import Proxy  # noqa: E402
from verity.core.protocols.certifier.audit import Audit  # noqa: E402

T = 60
SERVER_PORT, LINK_PORT = 8000, 9000
STARTUP_S = 3.0
STRATEGIES = ("honest", "bits", "late", "extra")
RESHAPE = "bring the lab up in this shape (`mockdc.py down`, then `up` with the same declaration and options)"


def lab(args) -> Datacenter:
    if args.declaration:
        path, _, name = args.declaration.rpartition(":")
        spec = importlib.util.spec_from_file_location("declaration", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        dc = getattr(module, name)
        dc = dc() if callable(dc) else dc
    else:
        dc = graph.leaf_spine(args.blocks, leaves=args.leaves, spines=args.spines,
                              fabric_latency=args.fabric_latency_us / 1e6)
    graph.check(dc)
    if not any(d.kind == "outside" for d in dc.fabric):
        raise SystemExit(f"{dc.name} has no device of kind `outside` to run the peer on")
    return dc


def plan_of(args, dc: Datacenter) -> graph.NetnsPlan:
    return graph.netns_plan(dc, jitter=args.fabric_jitter_us / 1e6)


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
    """Each namespace's interfaces and netem, then reachability: the outside host reaches each certifier and no block,
    a certifier reaches its block, and a block reaches nothing, its certifier included, since the certifier's guard
    drops whatever the block opens."""
    dc = lab(args)
    plan = plan_of(args, dc)
    for n in plan.namespaces:
        ifs = ", ".join(f"{i.name} {i.address} -> {i.peer[0]}" + (f" [netem {' '.join(i.netem)}]" if i.netem else "")
                        for i in n.interfaces)
        print(f"{n.name:12} {n.role:9} forward={int(n.forward)}  {ifs}")
    if diff := differences(plan):
        raise SystemExit(f"{'; '.join(diff)}: {RESHAPE}")
    ext = outside(dc)
    ext_address = plan.namespace(ext).interfaces[0].address.split("/")[0]
    checks = []
    for b in (b.name for b in dc.blocks):
        _, cert = records.uplink(dc, b)
        checks += [(ext, plan.fabric_address(cert), True), (ext, plan.address(b, cert), False),
                   (cert, plan.address(b, cert), True), (b, plan.address(cert, b), False), (b, ext_address, False)]
    bad = 0
    for src, dst, want in checks:
        got = sudo(["ip", "netns", "exec", graph.netns_name(src), "ping", "-c1", "-W1", dst], check=False).returncode == 0
        bad += got != want
        print(f"ping {src:7} -> {dst:12} {'reaches' if got else 'dropped'}{'' if got == want else '  (WRONG)'}")
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
    return ar.adversary_plan(plan, setup, strategy, seed=1)


# The three processes, each in its own namespace. Each prints one line when it is ready and reads the origin from its
# stdin; the server then reads one more line before it stops.

async def _line() -> str:
    return await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)


def _ready(**info) -> None:
    print(json.dumps({"ready": True} | info), flush=True)


async def server_role(cfg: dict) -> None:
    server = ar.Server(cfg["plan"], cfg["params"])
    listening = await asyncio.start_server(server.handle, cfg["host"], SERVER_PORT)
    _ready()
    await server.serve(float(await _line()))
    await _line()
    listening.close()
    server.close()
    ar.write_logs(Path(cfg["out"]), server=server)


async def proxy_role(cfg: dict) -> None:
    setup, out = ar.Setup(cfg["report"]), Path(cfg["out"])
    publish = ar.Publisher(out, setup)
    advice = ar.prover_args(cfg["plan"], setup)
    done = asyncio.get_running_loop().create_future()
    holder: dict = {}

    async def on_link(reader, writer) -> None:
        if "link" in holder:  # one link: a second connection is never the peer's
            writer.close()
            return
        holder["link"] = writer.get_extra_info("peername")
        await holder["proxy"].run(reader, writer)
        done.set_result(None)

    listening = await asyncio.start_server(on_link, cfg["host"], LINK_PORT)
    _ready()
    origin = float(sys.stdin.readline())  # blocks the loop, so the peer's connection waits until the proxy exists
    proxy = holder["proxy"] = Proxy((cfg["upstream"], SERVER_PORT), setup.egress_params, setup.ingress_params,
                                    setup.sync_set, advice.pop("syncs"), origin=origin, publish=publish,
                                    egress_link=setup.egress_link, ingress_link=setup.ingress_link, **advice)
    await done
    listening.close()
    publish.close()
    ar.write_logs(out, proxy=proxy)
    (out / "link.json").write_text(json.dumps({"peer": holder["link"]}))


async def peer_role(cfg: dict) -> None:
    peer = ar.Peer(cfg["plan"], cfg["params"])
    reader, writer = await asyncio.open_connection(cfg["link"], LINK_PORT, limit=1 << 22)
    _ready(local=writer.get_extra_info("sockname"))
    origin = float(await _line())
    sending = asyncio.create_task(peer.send(writer, origin))
    await peer.read(reader, origin)
    sending.cancel()
    writer.close()
    ar.write_logs(Path(cfg["out"]), peer=peer)


ROLES = {"server": server_role, "proxy": proxy_role, "peer": peer_role}


def role(args) -> None:
    ar._quiet_runtime()
    asyncio.run(ROLES[args.name](json.loads(Path(args.config).read_text())))


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
                                                                   "upstream": plan_ns.address(b, cert)})
        proxy.ready()
        peer = Process(ext, "peer", common | {"link": plan_ns.fabric_address(cert)})
        peer.ready()
        procs[b] = (server, proxy, peer)
    origin = time.monotonic() + STARTUP_S
    for ps in procs.values():
        for p in ps:
            p.send(repr(origin))
    print(f"running {strategies} from origin {origin:.3f}, {honest['windows']} windows of {T} buckets")
    codes = {}
    for b, (server, proxy, peer) in procs.items():
        codes[b] = {"proxy": proxy.wait(), "peer": peer.wait()}
        server.send("stop")
        codes[b]["server"] = server.wait()
        (out / b / "processes.json").write_text(json.dumps({"exit": codes[b], "origin": origin, "clock": "real",
                                                            "namespaces": [p.name for p in (server, proxy, peer)]}))
    summary = audit(out, dc, honest, plans, report, strategies)
    summary["exit"] = codes
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if any(any(c.values()) for c in codes.values()):
        raise SystemExit("a process failed: see its log")


def audit(out: Path, dc: Datacenter, honest_plan: dict, plans: dict, report: dict, strategies: dict) -> dict:
    """Each block's audit by the certifier's verifier and by the accounting layer's record check, its timing, and its
    wire against an honest block's."""
    setup = ar.Setup(report)
    syncs = len(setup.sync_set.deltas) * len(setup.sync_set.rhos)
    summary = {"declaration": dc.name, "blocks": {}}
    honest = next((b for b, s in strategies.items() if s == "honest"), None)
    for b, strategy in strategies.items():
        a = ar.audit_configuration(plans[b], report, out / b, audit=Audit)
        (out / b / "audit.json").write_text(json.dumps(a))
        rows = [(r["link"], r["window"], r["raw"], r["committed"]) for r in a["rows"]]
        published = records.published(out / b, setup, dc, b)
        claims = records.audit(published, records.declared(honest_plan, setup, dc, b), syncs)
        (out / b / "records.json").write_text(json.dumps([records.to_json(r) for r in published]))
        refused = [f"{c.what}: {c.why}" for c in claims if isinstance(c, Refused)]
        c = a["comparisons"]
        entry = {"strategy": strategy, "accepted": all(r[2] is None and r[3] is None for r in rows), "verdicts": rows,
                 "datacenter": {"accepted": not refused, "refused": refused,
                                "carried": sum(isinstance(x, Carried) for x in claims),
                                "advice_bits": sum(x.bits for x in claims if isinstance(x, Carried))},
                 "records_are_run_shapers": c["records_are_run_shapers"], "wire_is_the_records": c["wire_is_the_records"],
                 "released_not_the_programs": c["released_not_the_programs"],
                 "slack_ms": a["timing"]["slack_ms"],
                 "tick_late_ms_max": {k: v.get("max") for k, v in a["timing"]["tick_late_ms"].items()},
                 "ticks_over_slack": {k: v["over_slack"] for k, v in a["timing"]["tick_late_ms"].items()},
                 "wire_bucket_lag_ms": a["timing"]["wire_bucket_lag_ms"]}
        if honest is not None and b != honest:
            entry["against_honest"] = ar.against_honest(out / b, out / honest)
        summary["blocks"][b] = entry
        failed = [f"{link}[{w}] {raw or committed}" for link, w, raw, committed in rows if raw or committed]
        line = f"{b} {strategy:6} certifier audit {'ACCEPTED' if entry['accepted'] else 'REFUSED'}"
        line += f" ({', '.join(failed)})" if failed else ""
        line += "; datacenter records " + ("ACCEPTED" if not refused else f"REFUSED ({'; '.join(refused)})")
        if "against_honest" in entry:
            line += f"; wire is {honest}'s: {entry['against_honest']['wire_is_honests']}"
        print(line)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("up", up), ("down", down), ("status", status), ("run", run)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--declaration", metavar="FILE.py:NAME", help="a Datacenter, or a function returning one, "
                       "instead of graph.leaf_spine's (which --blocks, --leaves, --spines and --fabric-latency-us shape)")
        p.add_argument("--blocks", type=int, default=4)
        p.add_argument("--leaves", type=int, default=2)
        p.add_argument("--spines", type=int, default=1)
        p.add_argument("--fabric-latency-us", type=int, default=2_000, help="each fabric hop's latency")
        p.add_argument("--fabric-jitter-us", type=int, default=500, help="each fabric hop's delay jitter")
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--block", action="append", default=[], metavar="BLOCK=STRATEGY",
                           help=f"a block's server strategy, one of {STRATEGIES} (default: the blocks take them in turn)")
    r = sub.add_parser("role")
    r.set_defaults(fn=role)
    r.add_argument("name", choices=sorted(ROLES))
    r.add_argument("config")
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
