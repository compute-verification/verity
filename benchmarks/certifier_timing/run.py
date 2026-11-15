"""certifier_timing: the bits an interval a server moves through when the network certifier releases its bytes.

    verity bench certifier_timing run     [--paced-ms 50] [--intervals 3000] [--attacks write,burst]
                                          [--observers plain,spin] [--repeat 1] [--tree DIR [--tree-commit SHA]]
                                          [--pin] [--gate-ms BEFORE,AFTER] [--waker-ms MS] [--awake]
                                          [--interval-ms 100] [--out DIR]
    verity bench certifier_timing guard   [--intervals 600] [--interval-ms 100] [--out DIR]
    verity bench certifier_timing serve   --listen ADDR --port PORT [--attack write] [--certifier network] [--ignore]
                                          [--wait-s 600] [--launch DEV,AFTER,OPEN[,MODE]] [run's flags] --out DIR
    verity bench certifier_timing observe HOST:PORT [--certifier network] [--observer spin] [--intervals 3000]
                                          [--wait-s 600] [--tree DIR] --out DIR
    verity bench certifier_timing score   DIR... [--observer FILE|DIR]
    verity bench certifier_timing table   RESULT.json...
    verity bench certifier_timing target  [--interval-ms 100]

`run` first runs the guard, then each attack against the network certifier (`parties.py`: a server that picks a bit an
interval and times its writes by it, the certifier, and an observer on the client side, three processes on this host's
loopback and monotonic clock), `--repeat` times each with each observer, and prints one JSON line, which it also writes
to `--out`/result.json (`$OUT` under `research run`): `value`, the most bits an interval any attack showed
(`channel.measure`'s `bits`, of that attack's pooled runs), with its `unit`, its 95% `lower` and `upper`, the scored
intervals `n`, the `commit`, the `run`, the host, the target and every case; `ceiling` is the largest of the attacks'
upper bounds, what the run shows every attack built here under. Every case is scored on each of the observer's two
stamps (`STAMPS`): its own clock's once a read returned, and the kernel's on the bytes as they came off the network.
A row holds the first and, under `kernel`, the second; `value` and `ceiling` are over both, and the guard holds on
both. `--paced-ms 0` is the certifier writing each interval
on its loop, and `--tree` another checkout's certifier (its `verity` first on the certifier's import path), measured by
this checkout's server and observer. It exits 1 when the guard fails, 2 when a case's parties disagree on the schedule
or the certifier failed a connection: the number is then of nothing.

The candidates a run can turn on beside the certifier, none of them a change to it: `--pin` gives the certifier's loop,
its pacer, the server and the observer a core each of the run's; `--gate-ms` keeps the certifier from reading the
server around each release (`parties.network`); `--waker-ms` keeps the host's network path in use (`parties.waker`);
`--awake` keeps every one of the run's cores from halting (`parties.awake`).

`guard` is the check that the measurement sees a channel: against a certifier that paces nothing (`relay`), a server
that delays an interval's bytes by half an interval on bit 1 must show at least `PLANTED_MIN` bits an interval, and
the same server writing as for bit 0 whatever its bits at most `IGNORED_MAX`.

`serve` and `observe` are one case with the observer on another host. `serve` runs the server and the certifier, which
listens on ADDR:PORT for `--wait-s` and takes one connection, and leaves the case's directory without its
observer.json; `observe`, on the other host, tries it until it answers, timestamps the intervals on its own clock and
leaves observer.json; `score DIR --observer FILE` joins them, and `score DIR... --observer SEEN` many, each DIR's
observer.json at SEEN/<DIR's name>/. Nothing joins the two clocks: which interval a chunk is comes from its bytes and
the certifier's record of the connection, and `channel`'s `step` reads through the clocks' drift. `serve --certifier
relay --attack delay`, with and without `--ignore`, is the guard across the two hosts, and `serve --ignore` an
attack's control through the network certifier: `score` reads its guard from the first two and keeps the control out
of its number.

`--launch` puts the release in the kernel's hands, or the NIC's. For the case it sets a `taprio` schedule on DEV, the
device the certifier's bytes leave by, of two classes: the certifier's connections, which send at
`parties.LAUNCH_PRIORITY`, pass for OPEN ms from AFTER ms past each release time and wait for the rest of the
interval; everything else of the host's passes always. The server then aims at the gate's opening. MODE says who
holds the gate (`LAUNCH_MODES`): `gate`, the kernel's timer on the host's monotonic clock; `gate-nic`, the NIC, the
whole schedule offloaded (its clock, which the host keeps to TAI); `txtime`, the kernel stamps each packet with its
gate's opening and an ETF queue holds it till `ETF_DELTA_NS` before then, in software; `txtime-nic`, the same with
the NIC launching each packet at its stamp. A socket's own `SO_TXTIME` plays no part: TCP sets no packet's time from
it. `--launch` takes the right to administer the host's network, `tc`, `sch_taprio` (and `sch_etf`) and a device
with two transmit queues; it keeps the device's queue counts in launch.json, and a case whose gated queue carried
fewer packets than it has intervals is no measurement.

`score` measures recorded runs again, each DIR a case's directory as `run` or `serve` left it (config.json,
server.json, observer.json, certifier.json), with `run`'s exit codes, and `table` prints results' rows. `target` prints the budget's bits an
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
sys.path[:0] = [str(HERE), str(ROOT / "tools" / "config")]
sys.path.append(str(ROOT))

import channel  # noqa: E402
import parties  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402
from parties import (  # noqa: E402
    ATTACKS, CERTIFIERS, HEADER_BYTES, LAUNCH_PRIORITY, OBSERVERS, distinct_cores, plan_of, span_s,
)

SCHEMA = "certifier-timing/result/v1"
#: the guard's two bounds, in bits an interval
PLANTED_MIN, IGNORED_MAX = 0.9, 0.05
#: a server write that ended this long after its time is counted late (`late_writes`), and a pacer's release that
#: started this long after its (`late_releases`), in µs
LATE_US = 250.0
#: the certifier's schedule besides τ and the pacer's offset: one cell an interval, released D intervals after admission
SCHEDULE = {"cell_bytes": 256, "slots": 1, "delay": 3, "window": 600, "admit_lag": 1}
#: the observer's two stamps of an interval's first byte, and the key a case's score and a row hold each under: its
#: own clock's once its read returned, and the kernel's on the bytes as it took them in (`parties.kernel_stamp`)
STAMPS = {"process": "client", "kernel": "kernel"}
#: the wall clock may move this far against the monotonic one over a case, in ns, before the kernel's stamps, which
#: are the wall clock's, are no measurement
WALL_MOVED_NS = 10_000
#: what a case's score keeps of its configuration
KEEP = ("attack", "ignore", "certifier", "paced_ms", "observer", "interval_ms", "intervals", "pin", "gate_ms",
        "waker_ms", "awake", "launch", "commit", "certifier_commit", "run", "host")
#: who holds a launch gate: the kernel's timer, the NIC (the schedule offloaded), the kernel's ETF queue under the
#: schedule's stamps, the NIC's launch time under them
LAUNCH_MODES = ("gate", "gate-nic", "txtime", "txtime-nic")
#: under `txtime`, how long the kernel allows a packet from the schedule to the device, and how long before its stamp
#: the ETF queue lets it go (in software: when it leaves), in ns
TXTIME_DELAY_NS, ETF_DELTA_NS = 500_000, 200_000
CLOCK_TAI = 11


def _git(tree: Path) -> str | None:
    p = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def _commit() -> str | None:
    return SHARED.RESEARCH_SOURCE_SHA or _git(HERE)


def config(args: argparse.Namespace, *, attack: str, observer: str, certifier: str = "network", ignore: bool = False,
           intervals: int | None = None) -> dict:
    """One case's public configuration, without its origin. The relay takes none of the candidates. Under a launch
    gate the server aims at when the gate lets the bytes go: its opening, or under `txtime` `ETF_DELTA_NS` before."""
    relay = certifier == "relay"
    paced, launch = 0.0 if relay else args.paced_ms, None if relay else getattr(args, "launch", None)
    aim = paced + (launch["after_ms"] - (ETF_DELTA_NS / 1e6 if launch["mode"] == "txtime" else 0.0) if launch else 0.0)
    tree = getattr(args, "tree", None)
    return SCHEDULE | {
        "attack": attack, "ignore": ignore, "certifier": certifier, "observer": observer,
        "interval_ms": args.interval_ms, "paced_ms": paced, "aim_ms": aim,
        "intervals": intervals or args.intervals, "lead_ms": args.lead_ms, "burst": args.burst,
        "unit": SCHEDULE["slots"] * (SCHEDULE["cell_bytes"] - HEADER_BYTES), "start_s": args.start_s,
        "pin": False if relay else args.pin, "gate_ms": None if relay else args.gate_ms,
        "waker_ms": None if relay else args.waker_ms, "awake": False if relay else args.awake, "launch": launch,
        "cores": distinct_cores(sorted(os.sched_getaffinity(0))),
        "commit": _commit(), "run": SHARED.RESEARCH_RUN_ID, "host": platform.node(),
        "certifier_commit": (getattr(args, "tree_commit", None) or _git(tree)) if tree and not relay else _commit()}


def launch_commands(launch: dict, base_s: float, interval_ms: int) -> tuple[list[list[str]], list[str]]:
    """`--launch`'s commands: what sets the schedule on the device's root, and what removes it. Its cycle is one
    interval from `base_s` (seconds on the gate's clock: the host's monotonic clock under `gate`, else TAI), of two
    classes on a queue each: priority `LAUNCH_PRIORITY`'s passes for `open_ms` and then waits, every other always.
    Under `txtime` an ETF queue holds the first class's packets to the stamps the schedule gives them."""
    tau, opened, dev, mode = interval_ms * 1_000_000, round(launch["open_ms"] * 1e6), launch["dev"], launch["mode"]
    if not 0 < opened < tau:
        raise ValueError("a launch gate is open for part of an interval")
    clock = {"gate": ["clockid", "CLOCK_MONOTONIC"], "gate-nic": ["flags", "0x2"]}.get(
        mode, ["flags", "0x1", "txtime-delay", str(TXTIME_DELAY_NS), "clockid", "CLOCK_TAI"])
    setup = [["tc", "qdisc", "replace", "dev", dev, "parent", "root", "handle", "100", "taprio", "num_tc", "2",
              "map", *("1" if p == LAUNCH_PRIORITY else "0" for p in range(16)), "queues", "1@0", "1@1",
              "base-time", str(round(base_s * 1e9)), "sched-entry", "S", "03", str(opened),
              "sched-entry", "S", "01", str(tau - opened), *clock]]
    if mode.startswith("txtime"):
        setup.append(["tc", "qdisc", "replace", "dev", dev, "parent", "100:2", "etf", "clockid", "CLOCK_TAI",
                      "delta", str(ETF_DELTA_NS), *(["offload"] if mode == "txtime-nic" else []), "skip_sock_check"])
    return setup, ["tc", "qdisc", "del", "dev", dev, "root"]


def base_time(opening_s: float, now_s: float, interval_ms: int) -> float:
    """A launch schedule's base time: the gate's opening at `opening_s`, moved back by whole intervals to between one
    and two before `now_s`. The schedule is the same from any of its openings, and an I225 refuses one that is still
    to come."""
    return opening_s - (math.floor((opening_s - now_s) * 1000 / interval_ms) + 2) * interval_ms / 1000


def _launch(cfg: dict) -> list[str]:
    """Set the case's launch gate, which opens `after_ms` past each release time from the origin's on; the command
    that removes it."""
    base = base_time(cfg["origin"] + (cfg["paced_ms"] + cfg["launch"]["after_ms"]) / 1000, time.monotonic(),
                     cfg["interval_ms"])
    if cfg["launch"]["mode"] != "gate":
        base += time.clock_gettime(CLOCK_TAI) - time.monotonic()
    setup, remove = launch_commands(cfg["launch"], base, cfg["interval_ms"])
    for command in setup:
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            subprocess.run(remove, capture_output=True)
            raise RuntimeError(f"--launch: `{' '.join(command)}` failed ({done.stderr.strip()}): it takes the "
                               f"right to administer the network, sch_taprio (and sch_etf) and a device with two "
                               f"transmit queues")
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
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(tree), *SHARED.PYTHONPATH])}
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
        if cfg["awake"]:
            _line(start("awake"), "awake")
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
            queues = subprocess.run(["tc", "-s", "-j", "qdisc", "show", "dev", cfg["launch"]["dev"]],
                                    capture_output=True, text=True)
            (out / "launch.json").write_text(queues.stdout if queues.returncode == 0 and queues.stdout else "[]")
            subprocess.run(remove, capture_output=True)
    if any(codes):
        tails = {p.args[2]: (out / f"{p.args[2]}.log").read_text()[-800:] for p in started if p.returncode}
        raise RuntimeError(f"a party failed in {out}: {json.dumps(tails)}")


def samples(cfg: dict, server: dict, observer: dict, kernel: bool = False) -> tuple[list[tuple[float, int]], list[str]]:
    """The run's samples in interval order, (the offset of the interval's first byte at the observer after its
    release time, in seconds; the server's bit for it), and what doesn't fit: a scored chunk whose bytes name
    another interval, and on one host an observer's segment whose first arrival isn't in the interval the server
    took its connection to start in. An observer on another host has its own clock: its offsets are after the
    release times counted from its first arrival, the one before a connection's first interval.

    With `kernel` the offsets are of the kernel's stamps (`observer["kernel"]`) in place of the observer's own
    clock's, and what doesn't fit is the stamps' alone: which interval a chunk is stays what the observer's own
    arrivals and the chunk's bytes say, a scored interval whose read carried no stamp is left out and counted, and
    another host's offsets are counted from its first stamp."""
    plan, tau = plan_of(cfg), cfg["interval_ms"] / 1000
    coins = {t: bit for t, bit, _ in server["rows"]}
    firsts = [c["first"] for c in server["connections"] if "first" in c]
    ahead = cfg["delay"] if cfg["certifier"] == "network" else 0
    remote, labels = observer.get("remote", False), observer.get("labels")
    origin = None if remote else round(cfg["origin"] * 1e9)
    rows, fits, stamps, bare = {}, [], [], 0
    for k, segment in enumerate(observer["segments"]):
        if remote:
            if k >= len(firsts):
                fits.append(f"connection {k}: the observer's, and none of the server's")
                break
            t0 = firsts[k] - ahead
        else:
            t0 = math.floor(((segment[0] - origin) / 1e9 - plan.paced_s) / tau)
            if k < len(firsts) and firsts[k] != t0 + ahead:
                fits.append(f"connection {k}: the server took interval {firsts[k]} for its first bytes, and the "
                            f"observer's first arrival puts it at {t0 + ahead} (admit_lag)")
        times = observer["kernel"][k] if kernel else segment
        if origin is None and (i0 := next((i for i, at in enumerate(times) if at), None)) is not None:
            origin = times[i0] - round(plan.release(t0 + i0) * 1e9)
        other = 0
        for i, at in enumerate(times):
            if t0 + i in coins:
                if at:
                    rows[t0 + i] = ((at - origin) / 1e9 - plan.release(t0 + i), coins[t0 + i])
                bare += not at
                other += labels is not None and labels[k][i] != (t0 + i) & 0xFF
        if other:
            fits.append(f"connection {k}: {other} of the observer's scored intervals hold another interval's bytes")
    if bare:
        stamps.append(f"the kernel stamped none of the bytes of {bare} of the observer's scored intervals")
    return [rows[t] for t in sorted(rows)], stamps if kernel else fits


def stamped(cfg: dict, server: dict, observer: dict) -> tuple[dict | None, list[str]]:
    """A case by the kernel's stamps: `channel.measure` of them with `merged`, the observer's intervals whose stamp
    is the next one's too, and what makes them no measurement. The kernel merges segments that wait to be read and
    gives them the last one's stamp, so a merged interval's is of a later interval's bytes: it stays in, as late as
    the observer's own stamp of it is. None for an observer that recorded none."""
    if observer.get("kernel") is None:
        return None, []
    pairs, problems = samples(cfg, server, observer, kernel=True)
    moved = abs(observer["wall_less_monotonic_ns"][1] - observer["wall_less_monotonic_ns"][0])
    if moved > WALL_MOVED_NS:
        problems.append(f"the observer's wall clock moved {moved / 1000:.0f} µs against its monotonic clock over the "
                        f"case, and the kernel's stamps are the wall clock's")
    merged = sum(a == b != 0 for segment in observer["kernel"] for a, b in zip(segment, segment[1:]))
    return channel.measure(pairs) | {"merged": merged}, problems


def score(case: Path, observed: Path | None = None) -> dict:
    """One recorded case (a directory `live` left; `observed`: its observer.json, where another host wrote it): its
    configuration, `channel.measure` of the observer's arrivals on its own clock (`client`), by the kernel's stamps
    (`kernel`, `stamped`; `unstamped`: why its host's kernel gave none) and of the certifier's own times (`sender`:
    a paced one's write ends, else its ticks' lateness), how many times the certifier's host's cores entered an idle
    state that stops them while it ran (`halted`; None where the host gave no count, `parties.halts`), and what
    makes it no measurement (`problems`). A record whose certifier is none of `CERTIFIERS` is another benchmark's,
    and is refused."""
    cfg, server, certifier = (json.loads((case / f"{n}.json").read_text()) for n in ("config", "server", "certifier"))
    if cfg["certifier"] not in CERTIFIERS:
        raise SystemExit(f"{case}: its certifier is {cfg['certifier']!r}, none of {', '.join(CERTIFIERS)}: score it "
                         f"with the benchmark that recorded it")
    observer = json.loads((observed or case / "observer.json").read_text())
    pairs, problems = samples(cfg, server, observer)
    kernel, unfit = stamped(cfg, server, observer)
    problems += unfit
    coins = {t: bit for t, bit, _ in server["rows"]}
    sender = late_releases = gated = None
    halts = certifier.get("halts") or [None]
    counted = all(isinstance(count, int) for count in halts)
    if cfg["certifier"] == "network":
        if cfg["paced_ms"]:
            times = {r[0]: r[3] / 1e6 for r in certifier["releases"]["rows"]}
            late_releases = sum(r[2] > LATE_US for r in certifier["releases"]["rows"])
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
    if cfg.get("launch"):
        queues = json.loads(f.read_text()) if (f := case / "launch.json").exists() else []
        gated = {k: sum(q.get(k, 0) for q in queues if q.get("parent") == "100:2") for k in ("packets", "drops")}
        if gated["packets"] < cfg["intervals"]:
            problems.append(f"the launch gate's queue carried {gated['packets']} packets, under one for each of "
                            f"the case's {cfg['intervals']} intervals: its bytes left another way")
        if gated["drops"]:
            problems.append(f"the launch gate's queue dropped {gated['drops']} packets")
    problems += [f"the server's connection: {c['error']}" for c in server["connections"] if "error" in c]
    return {k: cfg.get(k) for k in KEEP} | {
        "observer": observer["observer"], "dir": case.name, "remote": observer.get("remote", False),
        "observer_host": observer.get("host", cfg.get("host")), "observer_run": observer.get("run", cfg.get("run")),
        "client": channel.measure(pairs), "kernel": kernel, "unstamped": observer.get("unstamped") or None,
        "sender": sender, "halted": halts[1] - halts[0] if counted else None,
        "problems": problems, "late_writes": sum(late > LATE_US for _, _, late in server["rows"]),
        "late_releases": late_releases, "gated": gated,
        "cpus": {"server": server["cpus"], "observer": observer["cpus"], "certifier": certifier["cpus"],
                 "pacer": certifier.get("pacer_cpus")}}


def role(case: dict) -> str:
    """What a case is to the number: an `attack` on the network certifier or its `control` (the server ignores its
    bits), or of the guard, through the relay, the `planted` channel or its `ignored` twin."""
    if case["certifier"] == "relay":
        return "ignored" if case["ignore"] else "planted"
    return "control" if case["ignore"] else "attack"


def guard(args: argparse.Namespace, out: Path) -> list[dict]:
    """The planted channel and its control, each one live run against the relay."""
    cases = []
    for name, ignore in (("planted", False), ("ignored", True)):
        cfg = config(args, attack="delay", observer="plain", certifier="relay", ignore=ignore,
                     intervals=args.guard_intervals)
        live(cfg, out / f"guard-{name}")
        cases.append(score(out / f"guard-{name}"))
    return cases


def verdict(cases: list[dict]) -> dict | None:
    """The guard among `cases`, if they hold one: `ok` when there is a planted case and an ignored one, every planted
    one shows at least `PLANTED_MIN` bits an interval, every ignored one at most `IGNORED_MAX`, and none has a
    problem; `planted` the least a planted one showed and `ignored` the most an ignored one did, on the observer's
    own clock. `kernel` holds the same two by the kernel's stamps, which the guard holds on as well wherever a case
    of it has them; None where none has."""
    of = {name: [c for c in cases if role(c) == name] for name in ("planted", "ignored")}
    if not of["planted"] and not of["ignored"]:
        return None
    read = {}
    for stamp, key in STAMPS.items():
        bits = {name: [(c.get(key) or {}).get("bits") for c in group] for name, group in of.items()}
        measured = all(group and None not in group for group in bits.values())
        planted, ignored = (min(bits["planted"]), max(bits["ignored"])) if measured else (None, None)
        read[stamp] = {"planted": planted, "ignored": ignored,
                       "ok": measured and planted >= PLANTED_MIN and ignored <= IGNORED_MAX}
    stamps = any(c.get("kernel") for group in of.values() for c in group)
    return {"ok": read["process"]["ok"] and (read["kernel"]["ok"] or not stamps)
            and not any(c["problems"] for group in of.values() for c in group),
            "planted": read["process"]["planted"], "ignored": read["process"]["ignored"],
            "kernel": {k: read["kernel"][k] for k in ("planted", "ignored")} if stamps else None,
            "bounds": {"planted_min": PLANTED_MIN, "ignored_max": IGNORED_MAX}}


def host() -> dict:
    """Where a `run`'s parties are: all three on this host, on its loopback and its monotonic clock."""
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
    idle = Path("/sys/devices/system/cpu/cpuidle/current_driver")
    return {"name": platform.node(), "kernel": platform.release(), "cpu": model,
            "cpus": sorted(os.sched_getaffinity(0)), "cpuidle": idle.read_text().strip() if idle.exists() else None,
            "parties": "the server, the certifier and the observer are processes on this one host, over loopback",
            "clock": "CLOCK_MONOTONIC of this host (time.monotonic), read by all three; the kernel's stamps are its "
                     "wall clock's, moved to it by the two clocks' difference as the observer started"}


def summarize(cases: list[dict]) -> dict:
    """The cases as one number: `rows`, the cases by (role, attack, observer), each pooled over its runs
    (`channel.pool`) on the observer's own clock and, under `kernel`, by the kernel's stamps (with `merged`, the
    runs' sum), with `halted`, the runs' sum (None where a run's host counted none), the attacks first, most bits
    on either stamp first; `value`, `lower`, `upper` and `n`, of the attack and stamp that showed most (`best`);
    `ceiling`, the largest upper bound of any attack on either stamp; and every case's problems."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for c in cases:
        groups.setdefault((role(c), c["attack"], c["observer"]), []).append(c)
    rows = []
    for (r, a, o), group in groups.items():
        stamps = [c["kernel"] for c in group if c.get("kernel")]
        rows.append({"role": r, "attack": a, "observer": o} | channel.pool([c["client"] for c in group]) | {
            "kernel": channel.pool(stamps) | {"merged": sum(m["merged"] for m in stamps)},
            "sender": channel.pool([c["sender"] for c in group if c["sender"]]),
            "halted": None if any(c.get("halted") is None for c in group) else sum(c["halted"] for c in group)})
    reads = {id(r): [(stamp, m) for stamp, m in (("process", r), ("kernel", r["kernel"])) if m["bits"] is not None]
             for r in rows}
    rows.sort(key=lambda r: (r["role"] != "attack", -max((m["bits"] for _, m in reads[id(r)]), default=0.0)))
    attacks = [(r, stamp, m) for r in rows if r["role"] == "attack" for stamp, m in reads[id(r)]]
    row, stamp, best = max(attacks, key=lambda x: x[2]["bits"], default=(None, None, None))
    return {"value": best and best["bits"], "unit": "bits/interval", "lower": best and best["lower"],
            "upper": best and best["upper"], "n": best and best["n"],
            "ceiling": max((m["upper"] for _, _, m in attacks), default=None),
            "best": best and {"attack": row["attack"], "observer": row["observer"], "stamp": stamp},
            "problems": sorted({p for c in cases for p in c["problems"]}), "rows": rows}


def result(cases: list[dict], **more) -> dict:
    """What `run` and `score` print: the cases' number, its target at their interval, their guard, and the cases."""
    summary = summarize(cases)
    tau = {c["interval_ms"] for c in cases}
    goal = channel.target(tau.pop() / 1000) if len(tau) == 1 else None
    if goal:
        goal["met"] = [summary["ceiling"] is not None and summary["ceiling"] <= b for b in goal["bits"]]
    return {"schema": SCHEMA, "benchmark": "certifier_timing"} | summary | {
        "commit": _commit(), "run": SHARED.RESEARCH_RUN_ID, "target": goal,
        "guard": verdict(cases)} | more | {"cases": cases}


def _table(docs: list[dict]) -> str:
    """Results' rows as a Markdown table, each with what its cases ran with."""
    head = ["run", "host", "observer's host", "certifier", "paced ms", "pin", "gate ms", "waker ms", "awake",
            "launch", "role", "attack", "observer", "bits/interval", "95%", "scored", "sender",
            "by the kernel's stamp", "95%", "merged", "halted"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d in docs:
        for r in d["rows"]:
            if r["bits"] is None:
                continue
            c = next(c for c in d["cases"] if (role(c), c["attack"], c["observer"]) == (r["role"], r["attack"],
                                                                                         r["observer"]))
            launch, sender, stamps = c.get("launch"), r["sender"].get("bits"), r["kernel"]
            runs = dict.fromkeys(filter(None, (c.get("run") or d.get("run"), c.get("observer_run"))))
            cells = [" and ".join(runs) or "-", c.get("host") or "-", c.get("observer_host") or "-",
                     (c.get("certifier_commit") or "-")[:10], f"{c['paced_ms']:g}", "yes" if c["pin"] else "no",
                     c["gate_ms"] or "-", c.get("waker_ms") or "-", "yes" if c.get("awake") else "no",
                     f"{launch['mode']} {launch['after_ms']:g}+{launch['open_ms']:g}" if launch else "-",
                     r["role"], r["attack"], r["observer"], f"{r['bits']:.4f}",
                     f"[{r['lower']:.4f}, {r['upper']:.4f}]", r["n"], "-" if sender is None else f"{sender:.4f}",
                     *(["-"] * 3 if stamps["bits"] is None else
                       [f"{stamps['bits']:.4f}", f"[{stamps['lower']:.4f}, {stamps['upper']:.4f}]", stamps["merged"]]),
                     "-" if r["halted"] is None else r["halted"]]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(lines)


def _hello(certifier: str, tree: Path) -> tuple[str, int]:
    """What an observer sends a certifier of `tree` and the bytes of one interval: the relay takes the request as it
    is, the network certifier in its cells."""
    if certifier == "relay":
        return parties.HELLO.hex(), SCHEDULE["slots"] * (SCHEDULE["cell_bytes"] - HEADER_BYTES)
    code = ("import sys; from verity.protocols.certifier.verifier.warden import cells; "
            "print(cells(bytes.fromhex(sys.argv[1]), int(sys.argv[2])).hex())")
    done = subprocess.run([sys.executable, "-c", code, parties.HELLO.hex(), str(SCHEDULE["cell_bytes"])],
                          env=os.environ | {"PYTHONPATH": str(tree)}, capture_output=True, text=True, check=True)
    return done.stdout.strip(), SCHEDULE["cell_bytes"] * SCHEDULE["slots"]


def _launch_arg(text: str) -> dict:
    dev, after, opened, *mode = text.split(",")
    if len(mode) > 1 or (mode and mode[0] not in LAUNCH_MODES):
        raise argparse.ArgumentTypeError(f"--launch DEV,AFTER,OPEN[,MODE], MODE of {', '.join(LAUNCH_MODES)}")
    return {"dev": dev, "after_ms": float(after), "open_ms": float(opened), "mode": mode[0] if mode else "gate"}


def _observed(case: Path, where: Path | None) -> Path | None:
    """A case's observer.json: its own, the file `where`, or the one under the directory `where` by the case's name."""
    return where if where is None or where.is_file() else where / case.name / "observer.json"


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
    sub.choices["guard"].set_defaults(lead_ms=2.0, burst=32, paced_ms=0.0, pin=False, gate_ms=None, waker_ms=None,
                                      awake=False)
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
        p.add_argument("--awake", action="store_true")
    p = sub.choices["run"]
    p.add_argument("--attacks", default="write,burst")
    p.add_argument("--observers", default="plain,spin")
    p.add_argument("--repeat", type=int, default=1, help="runs of each attack with each observer, in turn")
    p = sub.choices["serve"]
    p.add_argument("--listen", required=True, help="the address the certifier listens on, the other host's way in")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--wait-s", type=float, default=600.0, help="how long the certifier waits for the observer")
    p.add_argument("--attack", default="write", choices=ATTACKS)
    p.add_argument("--certifier", default="network", choices=CERTIFIERS)
    p.add_argument("--ignore", action="store_true", help="the server writes as for bit 0 whatever its bits")
    p.add_argument("--launch", type=_launch_arg, metavar="DEV,AFTER,OPEN[,MODE]")
    p = sub.add_parser("observe")
    p.add_argument("at", metavar="HOST:PORT")
    p.add_argument("--certifier", default="network", choices=CERTIFIERS)
    p.add_argument("--observer", default="spin", choices=OBSERVERS)
    p.add_argument("--intervals", type=int, default=3000)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--wait-s", type=float, default=600.0, help="how long it tries the certifier")
    p.add_argument("--tree", type=Path, help="the checkout whose certifier it is (default: this one)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--observer", type=Path, metavar="FILE|DIR",
                   help="one DIR's observer.json, or where each DIR's is, at <DIR's name>/observer.json")
    sub.add_parser("table").add_argument("results", nargs="+", type=Path)
    sub.add_parser("target").add_argument("--interval-ms", type=int, default=100)
    args = ap.parse_args(argv)
    if args.command == "target":
        goal = channel.target(args.interval_ms / 1000)
        print(json.dumps(goal | {"to_show": [channel.samples_to_show(x) for x in goal["bits"]]}))
        return 0
    if args.command == "score":
        if args.observer and args.observer.is_file() and len(args.dirs) != 1:
            ap.error("--observer FILE is one DIR's")
        doc = result([score(d, _observed(d, args.observer)) for d in args.dirs])
        print(json.dumps(doc))
        return 1 if doc["guard"] and not doc["guard"]["ok"] else 2 if doc["problems"] else 0
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
        (args.out / "observer.json").write_text(json.dumps(doc | {
            "host": platform.node(), "run": SHARED.RESEARCH_RUN_ID, "commit": _commit()}))
        print(json.dumps({"observed": str(args.out / "observer.json"), "intervals": sum(map(len, doc["segments"]))}))
        return 0 if doc["segments"] else 2
    out = args.out or Path(SHARED.OUT or tempfile.mkdtemp(prefix="certifier-timing-"))
    out.mkdir(parents=True, exist_ok=True)
    if args.pin and len(distinct_cores(sorted(os.sched_getaffinity(0)))) < 4:
        ap.error("--pin takes four cores: the certifier's loop, its pacer, the server and the observer")
    if args.command == "serve":
        cfg = config(args, attack=args.attack, observer="another host's", certifier=args.certifier, ignore=args.ignore)
        if args.attack == "delay" and args.certifier == "network":
            ap.error("the network certifier fails a delayed interval closed: --attack delay goes with --certifier relay")
        if cfg["launch"] and not args.paced_ms + cfg["launch"]["after_ms"] + cfg["launch"]["open_ms"] < args.interval_ms:
            ap.error("--launch: the gate opens and shuts inside the interval its release time is in")
        if cfg["launch"] and not args.paced_ms:
            ap.error("--launch: the gate is after a pacer's release time (--paced-ms)")
        live(cfg | {"listen": args.listen, "port": args.port, "wait_s": args.wait_s}, out, args.tree or ROOT)
        print(json.dumps({"served": str(out), "commit": _commit(), "run": SHARED.RESEARCH_RUN_ID,
                          "host": platform.node()}))
        return 0
    cases = guard(args, out)
    if args.command == "guard":
        print(json.dumps({"guard": verdict(cases), "cases": cases}))
        return 0 if verdict(cases)["ok"] else 1
    attacks, observers = args.attacks.split(","), args.observers.split(",")
    if (bad := [a for a in attacks if a not in ATTACKS or a == "delay"] + [o for o in observers if o not in OBSERVERS]):
        ap.error(f"--attacks are of write, burst (the network certifier fails a delayed interval closed) and "
                 f"--observers of {', '.join(OBSERVERS)}: not {', '.join(bad)}")
    for k in range(args.repeat):
        for attack in attacks:
            for observer in observers:
                case = out / f"{attack}-{observer}-{k}"
                live(config(args, attack=attack, observer=observer), case, args.tree or ROOT)
                cases.append(score(case))
    doc = result(cases, interval_ms=args.interval_ms, host=host(), certifier={
        "tree": str(args.tree or ROOT), "commit": cases[-1]["certifier_commit"], "paced_ms": args.paced_ms,
        "pin": args.pin, "gate_ms": args.gate_ms, "waker_ms": args.waker_ms, "awake": args.awake} | SCHEDULE)
    (out / "result.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "cases"}))
    return 1 if not doc["guard"]["ok"] else 2 if doc["problems"] or doc["value"] is None else 0


if __name__ == "__main__":
    sys.exit(main())
