"""certifier_timing: the bits an interval a server moves through when the network certifier releases its bytes.

    verity bench certifier_timing run     [--paced-ms 50] [--intervals 3000] [--attacks write,burst]
                                          [--observers plain,spin] [--repeat 1] [--tree DIR [--tree-commit SHA]]
                                          [--pin] [--gate-ms BEFORE,AFTER] [--waker-ms MS] [--interval-ms 100] [--out DIR]
    verity bench certifier_timing guard   [--intervals 600] [--interval-ms 100] [--out DIR]
    verity bench certifier_timing serve   --listen ADDR --port PORT [--attack write] [--certifier warden] [--ignore]
                                          [--wait-s 600] [--launch DEV,AFTER,OPEN[,offload]] [run's flags] --out DIR
    verity bench certifier_timing observe HOST:PORT [--certifier warden] [--observer spin] [--intervals 3000]
                                          [--wait-s 600] [--tree DIR] --out DIR
    verity bench certifier_timing score   DIR... [--observer OBSERVER.json]
    verity bench certifier_timing table   RESULT.json...
    verity bench certifier_timing target  [--interval-ms 100]

`run` first runs the guard, then each attack against the network certifier (`parties.py`: a server that picks a bit an
interval and times its writes by it, the certifier, and an observer on the client side, three processes on this host's
loopback and monotonic clock), `--repeat` times each with each observer, and prints one JSON line, which it also writes
to `--out`/result.json (`$OUT` under `research run`): `value`, the most bits an interval any attack showed
(`channel.measure`'s `bits`, of that attack's pooled runs), with its `unit`, its 95% `lower` and `upper`, the scored
intervals `n`, the `commit`, the `run`, the host, the target and every case; `ceiling` is the largest of the attacks'
upper bounds, what the run shows every attack built here under. `--paced-ms 0` is the certifier writing each interval
on its loop, and `--tree` another checkout's certifier (its `verity` first on the certifier's import path), measured by
this checkout's server and observer. It exits 1 when the guard fails, 2 when a case's parties disagree on the schedule
or the certifier failed a connection: the number is then of nothing.

The candidates a run can turn on beside the certifier, none of them a change to it: `--pin` gives the certifier's loop,
its pacer, the server and the observer a core each of the run's; `--gate-ms` keeps the certifier from reading the
server around each release (`parties.warden`); `--waker-ms` keeps the host's network path in use (`parties.waker`).

`guard` is the check that the measurement sees a channel: against a certifier that paces nothing (`relay`), a server
that delays an interval's bytes by half an interval on bit 1 must show at least `PLANTED_MIN` bits an interval, and
the same server writing as for bit 0 whatever its bits at most `IGNORED_MAX`.

`serve` and `observe` are one case with the observer on another host. `serve` runs the server and the certifier, which
listens on ADDR:PORT for `--wait-s`, and leaves the case's directory without its observer.json; `observe`, on the
other host, connects to it, timestamps the intervals on its own clock and leaves observer.json; `score DIR --observer
FILE` joins them. Nothing joins the two clocks: which interval a chunk is comes from its bytes and the certifier's
record of the connection, and `channel`'s `step` reads through the clocks' drift. `serve --certifier relay --attack
delay`, with and without `--ignore`, is the guard across the two hosts. `--launch` puts the release in the kernel's
hands, or the NIC's: for the case it sets a `taprio` gate on DEV, the device the certifier's bytes leave by, open for
OPEN ms from AFTER ms past each release time and shut for the rest of the interval, on the host's monotonic clock, or
with `offload` in the NIC, on its clock (which the host keeps to TAI). The server then aims at the gate's opening. It
takes CAP_NET_ADMIN, `tc`, `sch_taprio` and a device with more than one transmit queue.

`score` measures recorded runs again, each DIR a case's directory as `run` or `serve` left it (config.json,
server.json, observer.json, certifier.json), and `table` prints results' rows. `target` prints the budget's bits an
interval at an interval's length and the scored intervals it takes to show a number that small (`channel.target`,
`channel.samples_to_show`).

Every run's timestamps stay in its directory; the tree holds none of them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import channel  # noqa: E402
import parties  # noqa: E402
from parties import ATTACKS, CERTIFIERS, HEADER_BYTES, OBSERVERS, distinct_cores, plan_of, span_s  # noqa: E402

SCHEMA = "certifier-timing/result/v1"
#: the guard's two bounds, in bits an interval
PLANTED_MIN, IGNORED_MAX = 0.9, 0.05
#: a server write that ended this long after its time is counted late (`late_writes`), in µs
LATE_US = 250.0
#: the certifier's schedule besides τ and the pacer's offset: one cell an interval, released D intervals after admission
SCHEDULE = {"cell_bytes": 256, "slots": 1, "delay": 3, "window": 600, "admit_lag": 1}
#: what a case's score keeps of its configuration
KEEP = ("attack", "ignore", "certifier", "paced_ms", "observer", "interval_ms", "intervals", "pin", "gate_ms",
        "waker_ms", "launch")


def config(args: argparse.Namespace, *, attack: str, observer: str, certifier: str = "warden", ignore: bool = False,
           intervals: int | None = None) -> dict:
    """One case's public configuration, without its origin. The relay takes none of the candidates."""
    relay = certifier == "relay"
    paced, launch = 0.0 if relay else args.paced_ms, None if relay else getattr(args, "launch", None)
    return SCHEDULE | {
        "attack": attack, "ignore": ignore, "certifier": certifier, "observer": observer,
        "interval_ms": args.interval_ms, "paced_ms": paced, "aim_ms": paced + (launch["after_ms"] if launch else 0.0),
        "intervals": intervals or args.intervals, "lead_ms": args.lead_ms, "burst": args.burst,
        "unit": SCHEDULE["slots"] * (SCHEDULE["cell_bytes"] - HEADER_BYTES), "start_s": args.start_s,
        "pin": False if relay else args.pin, "gate_ms": None if relay else args.gate_ms,
        "waker_ms": None if relay else args.waker_ms, "launch": launch,
        "cores": distinct_cores(sorted(os.sched_getaffinity(0)))}


def launch_commands(launch: dict, base_s: float, interval_ms: int) -> tuple[list[str], list[str]]:
    """`--launch`'s two commands: the `taprio` gate on the device's root, whose cycle is one interval from `base_s`
    (seconds on the gate's clock: the host's monotonic clock, or with `offload` the NIC's), open for `open_ms` and
    then shut; and its removal. Every priority is one traffic class on the device's first queue."""
    tau, opened = interval_ms * 1_000_000, round(launch["open_ms"] * 1e6)
    if not 0 < opened < tau:
        raise ValueError("a launch gate is open for part of an interval")
    clock = ["flags", "0x2"] if launch["offload"] else ["clockid", "CLOCK_MONOTONIC"]
    return (["tc", "qdisc", "replace", "dev", launch["dev"], "parent", "root", "handle", "100", "taprio",
             "num_tc", "1", "map", *["0"] * 16, "queues", "1@0", "base-time", str(round(base_s * 1e9)),
             "sched-entry", "S", "01", str(opened), "sched-entry", "S", "00", str(tau - opened), *clock],
            ["tc", "qdisc", "del", "dev", launch["dev"], "root"])


def _launch(cfg: dict) -> list[str]:
    """Set the case's launch gate, its first opening `aim_ms` past the origin's boundary; the command that removes
    it."""
    base = cfg["origin"] + cfg["aim_ms"] / 1000
    if cfg["launch"]["offload"]:
        base += time.clock_gettime(time.CLOCK_TAI) - time.monotonic()
    add, remove = launch_commands(cfg["launch"], base, cfg["interval_ms"])
    done = subprocess.run(add, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"--launch: `{' '.join(add)}` failed ({done.stderr.strip()}): it takes CAP_NET_ADMIN, "
                           f"sch_taprio and a device with more than one transmit queue")
    return remove


def _line(process: subprocess.Popen, what: str) -> dict:
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"the {what} ended before it was ready (exit {process.wait()})")
    return json.loads(line)


def live(cfg: dict, out: Path, tree: Path = ROOT) -> None:
    """Run one case's parties and leave config.json, server.json, certifier.json and observer.json in `out`; the
    certifier imports `verity` from `tree`. With `listen` the observer is another host's (`observe`): the certifier
    listens there for `wait_s`, the line that says so is printed, and no observer.json is left."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg | {"out": str(out), "origin": time.monotonic() + cfg["start_s"]}
    cfg["stop"] = cfg["origin"] + cfg.get("wait_s", 0.0) + span_s(cfg)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    party = [sys.executable, str(HERE / "parties.py")]
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(tree), os.environ.get("PYTHONPATH")]))}
    started: list[subprocess.Popen] = []

    def start(role: str, *extra: str, **kw) -> subprocess.Popen:
        with open(out / f"{role}.log", "w") as log:
            started.append(subprocess.Popen([*party, role, str(out / "config.json"), *extra],
                                            stdout=subprocess.PIPE, stderr=log, text=True, **kw))
        return started[-1]

    remove = None
    try:
        if cfg["launch"]:
            remove = _launch(cfg)
        server = start("server")
        if cfg["waker_ms"]:
            _line(start("waker"), "waker")
        certifier = start("certifier", f"upstream={_line(server, 'server')['port']}", env=env)
        ready = _line(certifier, "certifier")
        if cfg.get("listen"):
            print(json.dumps({"listening": {"port": ready["port"], "wait_s": cfg["wait_s"]}}), flush=True)
            waited = [server]
        else:
            waited = [start("observer", *(f"{k}={ready[k]}" for k in ("port", "hello", "chunk"))), server]
        limit = cfg["stop"] - time.monotonic() + 30.0
        codes = [p.wait(limit) for p in waited]
        for p in started:
            if p not in waited:
                p.send_signal(signal.SIGTERM)
                codes.append(p.wait(30.0))
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
                process.wait()
        if remove:
            subprocess.run(remove, capture_output=True)
    if any(codes):
        tails = {p.args[2]: (out / f"{p.args[2]}.log").read_text()[-800:] for p in started if p.returncode}
        raise RuntimeError(f"a party failed in {out}: {json.dumps(tails)}")


def samples(cfg: dict, server: dict, observer: dict) -> tuple[list[tuple[float, int]], list[str]]:
    """The run's samples in interval order, (the offset of the interval's first byte at the observer after its
    release time, in seconds; the server's bit for it), and what doesn't fit: a scored chunk whose bytes name
    another interval, and on one host an observer's segment whose first arrival isn't in the interval the server
    took its connection to start in. An observer on another host has its own clock: its offsets are after the
    release times counted from its first arrival, the one before a connection's first interval."""
    plan, tau = plan_of(cfg), cfg["interval_ms"] / 1000
    coins = {t: bit for t, bit, _ in server["rows"]}
    firsts = [c["first"] for c in server["connections"] if "first" in c]
    ahead = cfg["delay"] if cfg["certifier"] == "warden" else 0
    remote, labels = observer.get("remote", False), observer.get("labels")
    origin = None if remote else round(cfg["origin"] * 1e9)
    rows, problems = {}, []
    for k, segment in enumerate(observer["segments"]):
        if remote:
            if k >= len(firsts):
                problems.append(f"connection {k}: the observer's, and none of the server's")
                break
            t0 = firsts[k] - ahead
            origin = segment[0] - round(plan.release(t0) * 1e9) if origin is None else origin
        else:
            t0 = math.floor(((segment[0] - origin) / 1e9 - plan.paced_s) / tau)
            if k < len(firsts) and firsts[k] != t0 + ahead:
                problems.append(f"connection {k}: the server took interval {firsts[k]} for its first bytes, and the "
                                f"observer's first arrival puts it at {t0 + ahead} (admit_lag)")
        other = 0
        for i, at in enumerate(segment):
            if t0 + i in coins:
                rows[t0 + i] = ((at - origin) / 1e9 - plan.release(t0 + i), coins[t0 + i])
                other += labels is not None and labels[k][i] != (t0 + i) & 0xFF
        if other:
            problems.append(f"connection {k}: {other} of the observer's scored intervals hold another interval's bytes")
    return [rows[t] for t in sorted(rows)], problems


def score(case: Path, observed: Path | None = None) -> dict:
    """One recorded case (a directory `live` left; `observed`: its observer.json, where another host wrote it): its
    configuration, `channel.measure` of the observer's arrivals (`client`) and of the certifier's own times (`sender`:
    a paced one's write ends, else its ticks' lateness), and what makes it no measurement (`problems`)."""
    cfg, server, certifier = (json.loads((case / f"{n}.json").read_text()) for n in ("config", "server", "certifier"))
    observer = json.loads((observed or case / "observer.json").read_text())
    pairs, problems = samples(cfg, server, observer)
    coins = {t: bit for t, bit, _ in server["rows"]}
    sender = None
    if cfg["certifier"] == "warden":
        if cfg["paced_ms"]:
            times = {r[0]: r[3] / 1e6 for r in certifier["releases"]["rows"]}
        else:
            times = {t: x / 1e6 for t, x in enumerate(certifier["tick_late_us"])}
        sender = channel.measure([(times[t], coins[t]) for t in sorted(coins) if t in times])
        problems += [f"the certifier failed connection {c['ordinal']} ({c['failed']})"
                     for c in certifier["connections"] if c["failed"]]
        admitted = [c["admitted"] for c in sorted(certifier["connections"], key=lambda c: c["ordinal"])
                    if c.get("admitted") is not None]
        firsts = [c["first"] for c in server["connections"] if "first" in c]
        problems += [f"connection {k}: the certifier admitted it in interval {s}, so its first bytes leave in "
                     f"{s + cfg['delay']}, and the server took {first} (admit_lag)"
                     for k, (s, first) in enumerate(zip(admitted, firsts)) if s + cfg["delay"] != first]
        over = sum(x > certifier["slack_ms"] * 1000 for x in certifier["tick_late_us"])
        problems += [f"{over} ticks ran past the slack"] * bool(over)
    problems += [f"the server's connection: {c['error']}" for c in server["connections"] if "error" in c]
    return {k: cfg.get(k) for k in KEEP} | {
        "observer": observer["observer"], "dir": case.name, "remote": observer.get("remote", False),
        "client": channel.measure(pairs), "sender": sender,
        "problems": problems, "late_writes": sum(late > LATE_US for _, _, late in server["rows"]),
        "cpus": {"server": server["cpus"], "observer": observer["cpus"], "certifier": certifier["cpus"],
                 "pacer": certifier.get("pacer_cpus")}}


def guard(args: argparse.Namespace, out: Path) -> dict:
    """The planted channel and its control, each one live run against the relay; `ok` when the first shows at least
    `PLANTED_MIN` bits an interval and the second at most `IGNORED_MAX`."""
    doc = {}
    for name, ignore in (("planted", False), ("ignored", True)):
        cfg = config(args, attack="delay", observer="plain", certifier="relay", ignore=ignore,
                     intervals=args.guard_intervals)
        live(cfg, out / f"guard-{name}")
        doc[name] = score(out / f"guard-{name}")
    planted, ignored = doc["planted"]["client"]["bits"], doc["ignored"]["client"]["bits"]
    doc["ok"] = (planted is not None and planted >= PLANTED_MIN and ignored is not None and ignored <= IGNORED_MAX
                 and not doc["planted"]["problems"] and not doc["ignored"]["problems"])
    doc["bounds"] = {"planted_min": PLANTED_MIN, "ignored_max": IGNORED_MAX}
    return doc


def _git(tree: Path) -> str | None:
    p = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def host() -> dict:
    """Where a `run`'s parties are: all three on this host, on its loopback and its monotonic clock."""
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
    idle = Path("/sys/devices/system/cpu/cpuidle/current_driver")
    return {"name": platform.node(), "kernel": platform.release(), "cpu": model,
            "cpus": sorted(os.sched_getaffinity(0)), "cpuidle": idle.read_text().strip() if idle.exists() else None,
            "parties": "the server, the certifier and the observer are processes on this one host, over loopback",
            "clock": "CLOCK_MONOTONIC of this host (time.monotonic), read by all three"}


def summarize(cases: list[dict]) -> dict:
    """The cases as one number: `rows`, the cases by (attack, observer), each pooled over its runs (`channel.pool`),
    most bits first; `value`, `lower`, `upper` and `n`, the first row's; `ceiling`, the largest upper bound of any
    row; and every case's problems."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cases:
        groups.setdefault((c["attack"], c["observer"]), []).append(c)
    rows = [{"attack": a, "observer": o} | channel.pool([c["client"] for c in group])
            | {"sender": channel.pool([c["sender"] for c in group if c["sender"]])}
            for (a, o), group in groups.items()]
    rows.sort(key=lambda r: -(r["bits"] or 0.0))
    best = rows[0] if rows and rows[0]["bits"] is not None else None
    return {"value": best and best["bits"], "unit": "bits/interval", "lower": best and best["lower"],
            "upper": best and best["upper"], "n": best and best["n"],
            "ceiling": max((r["upper"] for r in rows if r["bits"] is not None), default=None),
            "best": best and {"attack": best["attack"], "observer": best["observer"]},
            "problems": sorted({p for c in cases for p in c["problems"]}), "rows": rows}


def result(args: argparse.Namespace, guarded: dict, cases: list[dict]) -> dict:
    tree = args.tree or ROOT
    commit = os.environ.get("RESEARCH_SOURCE_SHA") or _git(HERE)
    goal = channel.target(args.interval_ms / 1000)
    summary = summarize(cases)
    goal["met"] = [summary["ceiling"] is not None and summary["ceiling"] <= b for b in goal["bits"]]
    return {"schema": SCHEMA, "benchmark": "certifier_timing"} | summary | {
        "commit": commit, "run": os.environ.get("RESEARCH_RUN_ID"),
        "certifier": {"tree": str(tree), "commit": args.tree_commit or (_git(tree) if args.tree else commit),
                      "paced_ms": args.paced_ms, "pin": args.pin, "gate_ms": args.gate_ms,
                      "waker_ms": args.waker_ms} | SCHEDULE,
        "interval_ms": args.interval_ms, "host": host(), "target": goal,
        "guard": {"ok": guarded["ok"], "planted": guarded["planted"]["client"]["bits"],
                  "ignored": guarded["ignored"]["client"]["bits"], "bounds": guarded["bounds"]},
        "cases": cases}


def _table(docs: list[dict]) -> str:
    lines = ["| run | certifier | paced ms | pin | gate ms | waker ms | attack | observer | bits/interval | 95% | scored "
             "| sender |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in docs:
        c = d["certifier"]
        for r in d["rows"]:
            if r["bits"] is None:
                continue
            sender = r["sender"].get("bits")
            lines.append(f"| {d['run'] or '-'} | {(c['commit'] or '-')[:10]} | {c['paced_ms']:g} | "
                         f"{'yes' if c['pin'] else 'no'} | {c['gate_ms'] or '-'} | {c.get('waker_ms') or '-'} | "
                         f"{r['attack']} | {r['observer']} | {r['bits']:.4f} | [{r['lower']:.4f}, {r['upper']:.4f}] | "
                         f"{r['n']} | {'-' if sender is None else f'{sender:.4f}'} |")
    return "\n".join(lines)


def _hello(certifier: str, tree: Path) -> tuple[str, int]:
    """What an observer sends a certifier of `tree` and the bytes of one interval: the relay takes the request as it
    is, the network certifier in its cells."""
    if certifier == "relay":
        return parties.HELLO.hex(), SCHEDULE["slots"] * (SCHEDULE["cell_bytes"] - HEADER_BYTES)
    code = ("import sys; from verity.core.protocols.certifier.warden import cells; "
            "print(cells(bytes.fromhex(sys.argv[1]), int(sys.argv[2])).hex())")
    done = subprocess.run([sys.executable, "-c", code, parties.HELLO.hex(), str(SCHEDULE["cell_bytes"])],
                          env=os.environ | {"PYTHONPATH": str(tree)}, capture_output=True, text=True, check=True)
    return done.stdout.strip(), SCHEDULE["cell_bytes"] * SCHEDULE["slots"]


def _launch_arg(text: str) -> dict:
    dev, after, opened, *rest = text.split(",")
    if rest not in ([], ["offload"]):
        raise argparse.ArgumentTypeError("--launch DEV,AFTER,OPEN[,offload]")
    return {"dev": dev, "after_ms": float(after), "open_ms": float(opened), "offload": bool(rest)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench certifier_timing", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "guard", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--interval-ms", type=int, default=100, help="τ, the certifier's bucket")
        p.add_argument("--start-s", type=float, default=6.0, help="from a case's start to its origin")
        p.add_argument("--out", type=Path, help="default $OUT, else a temporary directory")
    sub.choices["run"].add_argument("--guard-intervals", type=int, default=600,
                                    help="the intervals of each of the guard's two runs")
    sub.choices["guard"].add_argument("--intervals", dest="guard_intervals", type=int, default=600)
    sub.choices["guard"].set_defaults(lead_ms=2.0, burst=32, paced_ms=0.0, pin=False, gate_ms=None, waker_ms=None)
    for name in ("run", "serve"):
        p = sub.choices[name]
        p.add_argument("--paced-ms", type=float, default=50.0,
                       help="the pacer's offset after each interval's boundary; 0: each interval is written on the loop")
        p.add_argument("--intervals", type=int, default=3000, help="the intervals of each case")
        p.add_argument("--lead-ms", type=float, default=2.0)
        p.add_argument("--burst", type=int, default=32)
        p.add_argument("--tree", type=Path, help="the checkout whose certifier is measured (default: this one)")
        p.add_argument("--tree-commit", help="--tree's commit, where it isn't a git checkout")
        p.add_argument("--pin", action="store_true")
        p.add_argument("--gate-ms", type=lambda s: [float(x) for x in s.split(",")], metavar="BEFORE,AFTER")
        p.add_argument("--waker-ms", type=float)
    p = sub.choices["run"]
    p.add_argument("--attacks", default="write,burst")
    p.add_argument("--observers", default="plain,spin")
    p.add_argument("--repeat", type=int, default=1, help="runs of each attack with each observer, in turn")
    p = sub.choices["serve"]
    p.add_argument("--listen", required=True, help="the address the certifier listens on, the other host's way in")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--wait-s", type=float, default=600.0, help="how long the certifier waits for the observer")
    p.add_argument("--attack", default="write", choices=ATTACKS)
    p.add_argument("--certifier", default="warden", choices=CERTIFIERS)
    p.add_argument("--ignore", action="store_true", help="the server writes as for bit 0 whatever its bits")
    p.add_argument("--launch", type=_launch_arg, metavar="DEV,AFTER,OPEN[,offload]")
    p = sub.add_parser("observe")
    p.add_argument("at", metavar="HOST:PORT")
    p.add_argument("--certifier", default="warden", choices=CERTIFIERS)
    p.add_argument("--observer", default="spin", choices=OBSERVERS)
    p.add_argument("--intervals", type=int, default=3000)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--wait-s", type=float, default=600.0, help="how long it tries the certifier")
    p.add_argument("--tree", type=Path, help="the checkout whose certifier it is (default: this one)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--observer", type=Path, help="one DIR's observer.json, where another host wrote it")
    sub.add_parser("table").add_argument("results", nargs="+", type=Path)
    sub.add_parser("target").add_argument("--interval-ms", type=int, default=100)
    args = ap.parse_args(argv)
    if args.command == "target":
        goal = channel.target(args.interval_ms / 1000)
        print(json.dumps(goal | {"to_show": [channel.samples_to_show(x) for x in goal["bits"]]}))
        return 0
    if args.command == "score":
        if args.observer and len(args.dirs) != 1:
            ap.error("--observer is one DIR's")
        cases = [score(d, args.observer) for d in args.dirs]
        print(json.dumps({"schema": SCHEMA, "benchmark": "certifier_timing"} | summarize(cases) | {"cases": cases}))
        return 0
    if args.command == "table":
        print(_table([json.loads(p.read_text()) for p in args.results]))
        return 0
    if args.command == "observe":
        name, _, port = args.at.rpartition(":")
        hello, chunk = _hello(args.certifier, args.tree or ROOT)
        args.out.mkdir(parents=True, exist_ok=True)
        doc = parties.observer(SCHEDULE | {
            "observer": args.observer, "intervals": args.intervals, "interval_ms": args.interval_ms, "pin": False,
            "wait_s": args.wait_s, "connect": name, "port": int(port), "hello": hello, "chunk": chunk})
        (args.out / "observer.json").write_text(json.dumps(doc | {"host": platform.node()}))
        print(json.dumps({"observed": str(args.out / "observer.json"), "intervals": sum(map(len, doc["segments"]))}))
        return 0 if doc["segments"] else 2
    out = args.out or Path(os.environ.get("OUT") or tempfile.mkdtemp(prefix="certifier-timing-"))
    out.mkdir(parents=True, exist_ok=True)
    if args.pin and len(distinct_cores(sorted(os.sched_getaffinity(0)))) < 4:
        ap.error("--pin takes four cores: the certifier's loop, its pacer, the server and the observer")
    if args.command == "serve":
        cfg = config(args, attack=args.attack, observer="another host's", certifier=args.certifier, ignore=args.ignore)
        if args.attack == "delay" and args.certifier == "warden":
            ap.error("the network certifier fails a delayed interval closed: --attack delay goes with --certifier relay")
        if cfg["launch"] and not cfg["aim_ms"] + cfg["launch"]["open_ms"] < args.interval_ms:
            ap.error("--launch: the gate opens and shuts inside the interval its release time is in")
        live(cfg | {"listen": args.listen, "port": args.port, "wait_s": args.wait_s}, out, args.tree or ROOT)
        print(json.dumps({"served": str(out), "commit": os.environ.get("RESEARCH_SOURCE_SHA") or _git(HERE),
                          "run": os.environ.get("RESEARCH_RUN_ID"), "host": host()["name"]}))
        return 0
    guarded = guard(args, out)
    if args.command == "guard":
        print(json.dumps({"guard": guarded}))
        return 0 if guarded["ok"] else 1
    attacks, observers = args.attacks.split(","), args.observers.split(",")
    if (bad := [a for a in attacks if a not in ATTACKS or a == "delay"] + [o for o in observers if o not in OBSERVERS]):
        ap.error(f"--attacks are of write, burst (the network certifier fails a delayed interval closed) and "
                 f"--observers of {', '.join(OBSERVERS)}: not {', '.join(bad)}")
    cases = []
    for k in range(args.repeat):
        for attack in attacks:
            for observer in observers:
                case = out / f"{attack}-{observer}-{k}"
                live(config(args, attack=attack, observer=observer), case, args.tree or ROOT)
                cases.append(score(case))
    doc = result(args, guarded, cases)
    (out / "result.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "cases"}))
    return 1 if not guarded["ok"] else 2 if doc["problems"] or doc["value"] is None else 0


if __name__ == "__main__":
    sys.exit(main())
