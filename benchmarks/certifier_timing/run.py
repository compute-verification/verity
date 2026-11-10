"""certifier_timing: the bits an interval a server moves through when the network certifier releases its bytes.

    verity bench certifier_timing run    [--paced-ms 50] [--intervals 3000] [--attacks write,burst] [--observers plain,spin]
                                         [--repeat 1] [--tree DIR [--tree-commit SHA]] [--pin] [--gate-ms BEFORE,AFTER]
                                         [--interval-ms 100] [--out DIR]
    verity bench certifier_timing guard  [--intervals 600] [--interval-ms 100] [--out DIR]
    verity bench certifier_timing score  DIR...
    verity bench certifier_timing table  RESULT.json...
    verity bench certifier_timing target [--interval-ms 100]

`run` first runs the guard, then each attack against the network certifier (`parties.py`: a server that picks a bit an
interval and times its writes by it, the certifier, and an observer on the client side, three processes on this host's
loopback and monotonic clock), `--repeat` times each with each observer, and prints one JSON line, which it also writes
to `--out`/result.json (`$OUT` under `research run`): `value`, the most bits an interval any attack showed
(`channel.measure`'s `bits`, of that attack's pooled runs), with its `unit`, its 95% `lower` and `upper`, the scored
intervals `n`, the `commit`, the `run`, the host, the target and every case. `--paced-ms 0` is the certifier writing
each interval on its loop, and `--tree` another checkout's certifier (its `verity` first on the certifier's import
path), measured by this checkout's server and observer. `--pin` gives the certifier's loop, its pacer, the server and
the observer a core each of the run's, and `--gate-ms` keeps the certifier from reading the server around each release
(`parties.warden`). It exits 1 when the guard fails, 2 when a case's parties disagree on the schedule or the
certifier failed a connection: the number is then of nothing.

`guard` is the check that the measurement sees a channel: against a certifier that paces nothing (`relay`), a server
that delays an interval's bytes by half an interval on bit 1 must show at least `PLANTED_MIN` bits an interval, and
the same server writing as for bit 0 whatever its bits at most `IGNORED_MAX`.

`score` measures recorded runs again, each DIR a case's directory as `run` left it (config.json, server.json,
observer.json, certifier.json), and `table` prints results' rows. `target` prints the budget's bits an interval at an
interval's length and the scored intervals it takes to show a number that small (`channel.target`,
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
from parties import ATTACKS, HEADER_BYTES, OBSERVERS, distinct_cores, plan_of  # noqa: E402

SCHEMA = "certifier-timing/result/v1"
#: the guard's two bounds, in bits an interval
PLANTED_MIN, IGNORED_MAX = 0.9, 0.05
#: a server write that ended this long after its time is counted late (`late_writes`), in µs
LATE_US = 250.0
#: the certifier's schedule besides τ and the pacer's offset: one cell an interval, released D intervals after admission
SCHEDULE = {"cell_bytes": 256, "slots": 1, "delay": 3, "window": 600, "admit_lag": 1}


def config(args: argparse.Namespace, *, attack: str, observer: str, certifier: str = "warden", ignore: bool = False,
           intervals: int | None = None) -> dict:
    """One case's public configuration, without its origin."""
    relay = certifier == "relay"
    return SCHEDULE | {
        "attack": attack, "ignore": ignore, "certifier": certifier, "observer": observer,
        "interval_ms": args.interval_ms, "paced_ms": 0.0 if relay else args.paced_ms,
        "intervals": intervals or args.intervals, "lead_ms": args.lead_ms, "burst": args.burst,
        "unit": SCHEDULE["slots"] * (SCHEDULE["cell_bytes"] - HEADER_BYTES), "start_s": args.start_s,
        "pin": False if relay else args.pin, "gate_ms": None if relay else args.gate_ms,
        "cores": distinct_cores(sorted(os.sched_getaffinity(0)))}


def _line(process: subprocess.Popen, what: str) -> dict:
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"the {what} ended before it was ready (exit {process.wait()})")
    return json.loads(line)


def live(cfg: dict, out: Path, tree: Path = ROOT) -> None:
    """Run one case's three parties and leave config.json, server.json, certifier.json and observer.json in `out`;
    the certifier imports `verity` from `tree`."""
    out.mkdir(parents=True, exist_ok=True)
    tau = cfg["interval_ms"] / 1000
    cfg = cfg | {"out": str(out), "origin": time.monotonic() + cfg["start_s"]}
    cfg["stop"] = cfg["origin"] + (cfg["delay"] + cfg["intervals"] + 12) * tau + 2.0
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    party = [sys.executable, str(HERE / "parties.py")]
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(tree), os.environ.get("PYTHONPATH")]))}
    started: list[subprocess.Popen] = []

    def start(role: str, *extra: str, **kw) -> subprocess.Popen:
        with open(out / f"{role}.log", "w") as log:
            started.append(subprocess.Popen([*party, role, str(out / "config.json"), *extra],
                                            stdout=subprocess.PIPE, stderr=log, text=True, **kw))
        return started[-1]

    try:
        server = start("server")
        certifier = start("certifier", f"upstream={_line(server, 'server')['port']}", env=env)
        ready = _line(certifier, "certifier")
        observer = start("observer", *(f"{k}={ready[k]}" for k in ("port", "hello", "chunk")))
        limit = cfg["stop"] - time.monotonic() + 30.0
        codes = [observer.wait(limit), server.wait(limit)]
        certifier.send_signal(signal.SIGTERM)
        codes.append(certifier.wait(30.0))
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
                process.wait()
    if any(codes):
        tails = {p.args[2]: (out / f"{p.args[2]}.log").read_text()[-800:] for p in started if p.returncode}
        raise RuntimeError(f"a party failed in {out}: {json.dumps(tails)}")


def samples(cfg: dict, server: dict, observer: dict) -> tuple[list[tuple[float, int]], list[str]]:
    """The run's samples in interval order, (the offset of the interval's first byte at the observer after its
    release time, in seconds; the server's bit for it), and what doesn't fit: an observer's segment whose first
    arrival isn't in the interval the server took its connection to start in."""
    plan, tau, origin = plan_of(cfg), cfg["interval_ms"] / 1000, round(cfg["origin"] * 1e9)
    coins = {t: bit for t, bit, _ in server["rows"]}
    firsts = [c["first"] for c in server["connections"] if "first" in c]
    ahead = cfg["delay"] if cfg["certifier"] == "warden" else 0
    rows, problems = {}, []
    for k, segment in enumerate(observer["segments"]):
        t0 = math.floor(((segment[0] - origin) / 1e9 - plan.paced_s) / tau)
        if k < len(firsts) and firsts[k] != t0 + ahead:
            problems.append(f"connection {k}: the server took interval {firsts[k]} for its first bytes, and the "
                            f"observer's first arrival puts it at {t0 + ahead} (admit_lag)")
        for i, at in enumerate(segment):
            if t0 + i in coins:
                rows[t0 + i] = ((at - origin) / 1e9 - plan.release(t0 + i), coins[t0 + i])
    return [rows[t] for t in sorted(rows)], problems


def score(case: Path) -> dict:
    """One recorded case (a directory `live` left): its configuration, `channel.measure` of the observer's arrivals
    (`client`) and of the certifier's own times (`sender`: a paced one's write ends, else its ticks' lateness), and
    what makes it no measurement (`problems`)."""
    cfg, server, observer, certifier = (json.loads((case / f"{n}.json").read_text())
                                        for n in ("config", "server", "observer", "certifier"))
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
        over = sum(x > certifier["slack_ms"] * 1000 for x in certifier["tick_late_us"])
        problems += [f"{over} ticks ran past the slack"] * bool(over)
    problems += [f"the server's connection: {c['error']}" for c in server["connections"] if "error" in c]
    keep = ("attack", "ignore", "certifier", "paced_ms", "observer", "interval_ms", "intervals", "pin", "gate_ms")
    return {k: cfg[k] for k in keep} | {
        "dir": case.name, "client": channel.measure(pairs), "sender": sender, "problems": problems,
        "late_writes": sum(late > LATE_US for _, _, late in server["rows"]),
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
    """Where the run's parties are: all three on this host, on its loopback and its monotonic clock."""
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
    idle = Path("/sys/devices/system/cpu/cpuidle/current_driver")
    return {"name": platform.node(), "kernel": platform.release(), "cpu": model,
            "cpus": sorted(os.sched_getaffinity(0)), "cpuidle": idle.read_text().strip() if idle.exists() else None,
            "parties": "the server, the certifier and the observer are processes on this one host, over loopback",
            "clock": "CLOCK_MONOTONIC of this host (time.monotonic), read by all three"}


def summarize(cases: list[dict]) -> list[dict]:
    """The cases by (attack, observer), each pooled over its runs (`channel.pool`), most bits first."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cases:
        groups.setdefault((c["attack"], c["observer"]), []).append(c)
    rows = [{"attack": a, "observer": o} | channel.pool([c["client"] for c in group])
            | {"sender": channel.pool([c["sender"] for c in group if c["sender"]])}
            for (a, o), group in groups.items()]
    return sorted(rows, key=lambda r: -(r["bits"] or 0.0))


def result(args: argparse.Namespace, guarded: dict, cases: list[dict]) -> dict:
    rows = summarize(cases)
    best = rows[0] if rows and rows[0]["bits"] is not None else None
    tree = args.tree or ROOT
    commit = os.environ.get("RESEARCH_SOURCE_SHA") or _git(HERE)
    return {"schema": SCHEMA, "benchmark": "certifier_timing",
            "value": best and best["bits"], "unit": "bits/interval", "lower": best and best["lower"],
            "upper": best and best["upper"], "n": best and best["n"],
            "best": best and {"attack": best["attack"], "observer": best["observer"]},
            "commit": commit, "run": os.environ.get("RESEARCH_RUN_ID"),
            "certifier": {"tree": str(tree), "commit": args.tree_commit or (_git(tree) if args.tree else commit),
                          "paced_ms": args.paced_ms, "pin": args.pin, "gate_ms": args.gate_ms} | SCHEDULE,
            "interval_ms": args.interval_ms, "host": host(), "target": channel.target(args.interval_ms / 1000),
            "guard": {"ok": guarded["ok"], "planted": guarded["planted"]["client"]["bits"],
                      "ignored": guarded["ignored"]["client"]["bits"], "bounds": guarded["bounds"]},
            "problems": sorted({p for c in cases for p in c["problems"]}), "rows": rows, "cases": cases}


def _table(docs: list[dict]) -> str:
    lines = ["| run | certifier | paced ms | pin | gate ms | attack | observer | bits/interval | 95% | scored | sender |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in docs:
        c = d["certifier"]
        for r in d["rows"]:
            if r["bits"] is None:
                continue
            sender = r["sender"].get("bits")
            lines.append(f"| {d['run'] or '-'} | {(c['commit'] or '-')[:10]} | {c['paced_ms']:g} | "
                         f"{'yes' if c['pin'] else 'no'} | {c['gate_ms'] or '-'} | {r['attack']} | {r['observer']} | "
                         f"{r['bits']:.4f} | [{r['lower']:.4f}, {r['upper']:.4f}] | {r['n']} | "
                         f"{'-' if sender is None else f'{sender:.4f}'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench certifier_timing", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "guard"):
        p = sub.add_parser(name)
        p.add_argument("--interval-ms", type=int, default=100, help="τ, the certifier's bucket")
        p.add_argument("--guard-intervals" if name == "run" else "--intervals", dest="guard_intervals", type=int,
                       default=600, help="the intervals of each of the guard's two runs")
        p.add_argument("--start-s", type=float, default=6.0, help="from a case's start to its origin")
        p.add_argument("--out", type=Path, help="default $OUT, else a temporary directory")
    sub.choices["guard"].set_defaults(lead_ms=2.0, burst=32, paced_ms=0.0, pin=False, gate_ms=None)
    p = sub.choices["run"]
    p.add_argument("--paced-ms", type=float, default=50.0,
                   help="the pacer's offset after each interval's boundary; 0: each interval is written on the loop")
    p.add_argument("--intervals", type=int, default=3000, help="the intervals of each case")
    p.add_argument("--attacks", default="write,burst")
    p.add_argument("--observers", default="plain,spin")
    p.add_argument("--repeat", type=int, default=1, help="runs of each attack with each observer, in turn")
    p.add_argument("--lead-ms", type=float, default=2.0)
    p.add_argument("--burst", type=int, default=32)
    p.add_argument("--tree", type=Path, help="the checkout whose certifier is measured (default: this one)")
    p.add_argument("--tree-commit", help="--tree's commit, where it isn't a git checkout")
    p.add_argument("--pin", action="store_true")
    p.add_argument("--gate-ms", type=lambda s: [float(x) for x in s.split(",")], metavar="BEFORE,AFTER")
    sub.add_parser("score").add_argument("dirs", nargs="+", type=Path)
    sub.add_parser("table").add_argument("results", nargs="+", type=Path)
    sub.add_parser("target").add_argument("--interval-ms", type=int, default=100)
    args = ap.parse_args(argv)
    if args.command == "target":
        goal = channel.target(args.interval_ms / 1000)
        print(json.dumps(goal | {"to_show": [channel.samples_to_show(x) for x in goal["bits"]]}))
        return 0
    if args.command == "score":
        print(json.dumps([score(d) for d in args.dirs]))
        return 0
    if args.command == "table":
        print(_table([json.loads(p.read_text()) for p in args.results]))
        return 0
    out = args.out or Path(os.environ.get("OUT") or tempfile.mkdtemp(prefix="certifier-timing-"))
    out.mkdir(parents=True, exist_ok=True)
    guarded = guard(args, out)
    if args.command == "guard":
        print(json.dumps({"guard": guarded}))
        return 0 if guarded["ok"] else 1
    attacks, observers = args.attacks.split(","), args.observers.split(",")
    if (bad := [a for a in attacks if a not in ATTACKS or a == "delay"] + [o for o in observers if o not in OBSERVERS]):
        ap.error(f"--attacks are of write, burst (the network certifier fails a delayed interval closed) and "
                 f"--observers of {', '.join(OBSERVERS)}: not {', '.join(bad)}")
    if args.pin and len(distinct_cores(sorted(os.sched_getaffinity(0)))) < 4:
        ap.error("--pin takes four cores: the certifier's loop, its pacer, the server and the observer")
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
