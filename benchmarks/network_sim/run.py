"""The network night on the lab host in one command: `verity bench network_sim --out DIR`.

`verity` isn't on vy-nebius-cpu-2's path, so there it is `python3 benchmarks/network_sim/run.py --out DIR`. In one
process tree it runs:
1. the accounting layer's demo, `python -m datacenter.demo` (`DIR/demo.txt`);
2. the mock datacenter in five blocks (`mockdc.py down`, then `up --blocks 5`), whose blocks take its five strategies,
   run once (`DIR/mockdc/`) and once with the certifiers' salt on (`DIR/mockdc-salt/`);
3. the fleet (`up --fleet --blocks 2`, `datacenter.examples.two_sites`): two sites of two blocks joined by a 20 ms WAN
   between their spines, the outside host on the west site, run twice with the strategies swapped between the sites,
   so that each of the first four is played across the WAN once (`DIR/fleet/`, `DIR/fleet-swapped/`);
4. the lab back in its default four blocks, and the workload's three runs of `python -m workload.lab_run run`: honest,
   `--tamper b2:late` and `--tamper b1:extra` (`DIR/workload/<run>/`), each at a first-frame latency D of
   `DELTA_BUCKETS`. They need a Python with torch: `--python`, else this one if it has torch, else one
   that `uv` builds in a scratch directory, as the workload's recorded run does.

Each step's output is printed and kept in `DIR/<step>.txt`. It refuses to start while a process runs in a lab namespace,
as lab_run does, and however it ends it leaves the lab up in its default shape with nothing running in it.

Then it reads each step's `summary.json` and writes `DIR/summary.json` and the table `DIR/summary.txt`: each block's
verdicts under the certifier's audit, mockdc's datacenter record check (with the account's charge) and, on lab_run's
served links, N2's served verdict on each link-window, the median wire lag,
the worst proxy tick against its slack, and for each mockdc run its record check on every port of the declaration (a
fleet's merged ones), and the answers. The verdicts are compared with the expected ones (`expected_block`,
`expected_records` and `WORKLOAD`), and the exit status is 1 if any differs, a step exited nonzero or a proxy tick ran
past its slack, so the night is its own regression check.

    python3 benchmarks/network_sim/run.py --out DIR [--python PY]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOCKDC = [sys.executable, "-B", str(HERE / "mockdc.py")]

STRATEGIES = ("honest", "bits", "late", "extra", "maintenance")  # mockdc's, which its blocks take in turn
WINDOW = 6000  # ticks: mockdc's windows are 60 buckets of 100 ms
FIVE, FLEET = ("--blocks", "5"), ("--fleet", "--blocks", "2")
# mockdc's runs: the lab's shape, whether the certifiers' salt is on, and each block's strategy. The fleet's two runs
# put each strategy on each site once.
MOCKDC_RUNS = {
    "mockdc": (FIVE, False, {f"b{k}": s for k, s in enumerate(STRATEGIES)}),
    "mockdc-salt": (FIVE, True, {f"b{k}": s for k, s in enumerate(STRATEGIES)}),
    "fleet": (FLEET, False, {"east.b0": "honest", "east.b1": "late", "west.b0": "bits", "west.b1": "extra"}),
    "fleet-swapped": (FLEET, False, {"east.b0": "bits", "east.b1": "extra", "west.b0": "honest", "west.b1": "late"}),
}
COVERAGE = "status:coverage-never-established"
CHEATED = ("occupancy differs from the declared profile", "a frame is not a committed crossing")
DOWN = "no record: the certifier was down or its record was withheld"
SALT = {"accepted": True, "violations": [], "ingress_salt_frames": [1, 1, 1], "egress_salt_frames": 0,
        "egress_holds_salt": False, "block_received": True}
# lab_run's D, which it otherwise calibrates from the host's timing of the sessions: `late` holds b2's first response
# past D, so the window its refusal lands in moves with D, and these verdicts are D = 12's (1.2 s, over four times the
# slowest session the lab host has timed, 263 ms).
DELTA_BUCKETS = 12
# lab_run's runs: its arguments, the blocks of b0 … b3 that an audit refuses, each as the certifier's refused rows and
# the link-windows N2's served counts refuse (every other block is accepted by both, on each of its `WORKLOAD_WINDOWS`
# link-windows), and the answers. N2 is counts only, so it refuses the windows whose egress counts aren't the ones their
# ingress counts derive, each refused by the served counts' check itself: late's held response and extra's one more
# event, after b1's last response.
WORKLOAD = {
    "honest": ([], {}, "equal"),
    "late": (["--tamper", "b2:late", "--expect", "refused"], {"b2": ([["responses", 0, COVERAGE, COVERAGE]], [0])}, None),
    "extra": (["--tamper", "b1:extra", "--expect", "refused"], {"b1": ([], [1])}, "equal"),
}
WORKLOAD_BLOCKS = ["b0", "b1", "b2", "b3"]
WORKLOAD_WINDOWS = 5  # lab_run's windows at D = 12


def expected_block(strategy: str, block: str, salt: bool) -> dict:
    """What mockdc's audits say of a block that plays `strategy`, in `mockdc_verdicts`'s shape: the certifier's refused
    rows, the record check's refusals under the verifier's plan, whether the account is sound and what it charges
    besides timing advice, the same without the plan for a block down for maintenance, whether a block other than the
    honest one put the honest wire on the fabric, and with `salt` the certifier's salt check, which finds nothing for
    the window a block is down."""
    def refused(w: int, whys, directions=("out",)) -> list[str]:
        return [f"port {block}.cert:{d} [{w * WINDOW}, {(w + 1) * WINDOW}): {why}" for d in directions for why in whys]

    honest = {"strategy": strategy, "certifier": [], "datacenter": [], "sound": True, "charged": ["timing"]}
    e = {"honest": honest,
         "bits": honest | {"wire_is_honests": True},
         "late": honest | {"certifier": [["responses", 0, COVERAGE, COVERAGE]], "datacenter": refused(0, CHEATED),
                           "sound": False, "charged": ["timing", "refused"], "wire_is_honests": False},
         "extra": honest | {"certifier": [["responses", 2, "mismatch", "mismatch"]], "datacenter": refused(2, CHEATED),
                            "sound": False, "charged": ["timing", "refused"], "wire_is_honests": False},
         "maintenance": honest | {"certifier": [[link, 1, "no-verdict", "no-verdict"] for link in ("responses", "requests")],
                                  "charged": ["timing", "serviced"], "unplanned": refused(1, [DOWN], ("out", "in")),
                                  "unplanned_sound": False, "unplanned_charged": ["timing", "refused"]}}[strategy]
    if salt:
        e = e | {"salt": SALT | {"accepted": False, "violations": [[1, "salt"]], "ingress_salt_frames": [1, 1]}
                 if strategy == "maintenance" else SALT}
    return e


def expected_records(strategies: dict[str, str]) -> dict:
    """What mockdc's record check on every block's records together says, in `records_verdicts`' shape: it covers each
    block's two ports, which are all of the declaration's, and refuses what each block's own check refuses, in block
    order; the account for the whole run charges timing advice, the refused port-windows and what a block down for
    maintenance holds."""
    refused = [r for b, s in strategies.items() for r in expected_block(s, b, False)["datacenter"]]
    return {"ports": sorted(f"{b}.cert:{d}" for b in strategies for d in ("in", "out")), "of": 2 * len(strategies),
            "refused": refused, "sound": not refused,
            "charged": ["timing"] + ["refused"] * bool(refused) + ["serviced"] * ("maintenance" in strategies.values())}


def refused_rows(rows: list) -> list[list]:
    return [list(r) for r in rows if r[2] or r[3]]


def charged(account: dict) -> list[str]:
    return [k for k in ("timing", "refused", "serviced") if account[f"{k}_bits"]]


def mockdc_verdicts(entry: dict) -> dict:
    """A block's verdicts in mockdc's `summary.json`."""
    d = entry["datacenter"]
    out = {"strategy": entry["strategy"], "certifier": refused_rows(entry["verdicts"]), "datacenter": d["refused"],
           "sound": d["account"]["sound"], "charged": charged(d["account"])}
    if (u := d.get("unplanned")) is not None:
        out |= {"unplanned": u["refused"], "unplanned_sound": u["account"]["sound"],
                "unplanned_charged": charged(u["account"])}
    if "against_honest" in entry:
        out["wire_is_honests"] = entry["against_honest"]["wire_is_honests"]
    if "salt" in entry:
        out["salt"] = entry["salt"]
    return out


def records_verdicts(records: dict) -> dict:
    """The record check on every port in mockdc's `summary.json`."""
    return {"ports": records["ports"], "of": records["of"], "refused": records["refused"],
            "sound": records["account"]["sound"], "charged": charged(records["account"])}


def workload_verdicts(block: dict) -> dict:
    """A block's verdicts in lab_run's `summary.json`: the certifier's refused rows, and N2's as its link-windows, the
    ones it refuses and the ones its served counts' check alone refuses."""
    n2 = block.get("n2")
    return {"certifier": refused_rows(block["certifier"]["verdicts"]),
            "n2": None if n2 is None else {"windows": n2["windows"], "refused": n2["refused"],
                                           "counts_refused": [w for w, ok in enumerate(n2["served_counts"]) if not ok]}}


def tick(block: dict) -> dict:
    """The block's worst proxy tick against its slack, and the median lag from a bucket's start to its bytes reaching
    the peer, which holds the path's latency, the WAN's included."""
    late = [v for v in block["tick_late_ms_max"].values() if v is not None]
    return {"late_ms": max(late, default=0.0), "slack_ms": block["slack_ms"],
            "over": sum(block["ticks_over_slack"].values()), "lag_ms": block.get("wire_bucket_lag_ms", {}).get("p50")}


def differences(where: str, expected: dict, observed: dict) -> list[str]:
    return [f"{where} {k}: expected {expected.get(k)!r}, got {observed.get(k)!r}"
            for k in sorted(expected.keys() | observed.keys()) if expected.get(k) != observed.get(k)]


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def summarize(out: Path, exits: dict[str, int]) -> dict:
    """The night's summary from its steps' exit codes and the `summary.json` each wrote under `out`: each block's
    verdicts, the account's charge and its worst proxy tick, the answers, and every mismatch with the expected verdicts
    (a step that exited nonzero, a missing summary, a differing verdict, a proxy tick past its slack)."""
    mismatches = [f"{step} exited {rc}" for step, rc in exits.items() if rc]
    runs = {}

    def block(run: str, b: str, case: str, want: dict, got: dict, entry: dict, **more) -> dict:
        t = tick(entry)
        bad = differences(f"{run} {b}", want, got)
        bad += [f"{run} {b}: {t['over']} proxy ticks past the {t['slack_ms']:g} ms slack"] * bool(t["over"])
        mismatches.extend(bad)
        return {"case": case, "verdicts": got, **more, "tick": t, "ok": not bad}

    for run, (_, salt, strategies) in MOCKDC_RUNS.items():
        if (s := read(out / run / "summary.json")) is None:
            mismatches.append(f"{run}: no summary.json")
            continue
        if list(s["blocks"]) != list(strategies):
            mismatches.append(f"{run}: blocks {list(s['blocks'])}, not {list(strategies)}")
        blocks = {}
        for b, entry in s["blocks"].items():
            if b not in strategies:
                continue
            d = entry["datacenter"]
            accounts = {"account": d["account"]} | ({"unplanned_account": d["unplanned"]["account"]}
                                                    if "unplanned" in d else {})
            blocks[b] = block(run, b, entry["strategy"], expected_block(strategies[b], b, salt), mockdc_verdicts(entry),
                              entry, **accounts)
        runs[run] = {"declaration": s.get("declaration"), "wan": s.get("wan"), "blocks": blocks, "plan": s.get("plan")}
        if (r := s.get("records")) is None:
            mismatches.append(f"{run}: no record check on every port")
            continue
        got = records_verdicts(r)
        bad = differences(f"{run} every port", expected_records(strategies), got)
        mismatches.extend(bad)
        runs[run]["records"] = {"verdicts": got, "account": r["account"], "ok": not bad}
    for name, (_, refused, answers) in WORKLOAD.items():
        run = f"workload/{name}"
        if (s := read(out / run / "summary.json")) is None:
            mismatches.append(f"{run}: no summary.json")
            continue
        if list(s["blocks"]) != WORKLOAD_BLOCKS:
            mismatches.append(f"{run}: blocks {list(s['blocks'])}, not {WORKLOAD_BLOCKS}")
        blocks = {}
        for b, x in s["blocks"].items():
            rows, n2 = refused.get(b, ([], []))
            want = {"certifier": rows, "n2": {"windows": WORKLOAD_WINDOWS, "refused": n2, "counts_refused": n2}}
            blocks[b] = block(run, b, x["tamper"] or "-", want, workload_verdicts(x), x)
        got = None if s["answers"] is None else "equal" if s["answers"] == s["reference"] else "differ"
        if got != answers:
            mismatches.append(f"{run} answers: expected {answers!r}, got {got!r} ({s['answers']} against {s['reference']})")
        runs[run] = {"blocks": blocks, "answers": s["answers"], "reference": s["reference"]}
    ticks = [(x["tick"], f"{run} {b}") for run, r in runs.items() for b, x in r["blocks"].items()]
    worst = max(ticks, key=lambda t: t[0]["late_ms"], default=None)
    return {"ok": not mismatches, "mismatches": mismatches, "exit": exits, "runs": runs,
            "worst_tick": None if worst is None else worst[0] | {"at": worst[1]},
            "ticks_over_slack": sum(t["over"] for t, _ in ticks)}


def render(summary: dict) -> str:
    """`summary.txt`: a row per block of each run, and for a mockdc run one more for its record check on every port
    (`all`), then the answers, the worst tick and the mismatches."""
    def certifier(v: dict) -> str:
        rows = v["certifier"]
        text = "accepted" if not rows else "refused " + ", ".join(f"{link}[{w}] {raw or com}" for link, w, raw, com in rows)
        if (s := v.get("salt")) is not None:
            text += "; salt " + ("accepted" if s["accepted"] else "refused " + ", ".join(f"[{w}]" for w, _ in s["violations"]))
        return text

    def records(refused: list[str], by_block: bool = False) -> str:
        by: dict[tuple[str, str], dict[str, None]] = {}
        for r in refused:
            block, rest = r.split(": ")[0].removeprefix("port ").split(".cert:")
            direction, during = rest.split(" ", 1)
            by.setdefault((block if by_block else "", during), {})[direction] = None
        return "accepted" if not refused else "refused " + "; ".join(
            f"{b + ' ' if b else ''}{', '.join(d)} {w}" for (b, w), d in by.items())

    def account(a: dict) -> str:
        return f"{a['covert_bits']:.3g} {'sound' if a['sound'] else 'NOT sound'}"

    def served(v: dict | None) -> str:
        if v is None:
            return "-"
        refused = ", ".join(f"[{w}]" for w in v["refused"])
        return f"{v['windows'] - len(v['refused'])}/{v['windows']}" + (f" refused {refused}" if refused else "")

    rows = [("run", "block", "case", "certifier audit", "records", "N2 served", "covert budget (bits)", "wire lag",
             "worst tick", "")]
    for run, r in summary["runs"].items():
        for b, x in r["blocks"].items():
            v = x["verdicts"]
            second, charge = "-", ""
            if "datacenter" in v:
                second, charge = "records " + records(v["datacenter"]), account(x["account"])
                if "unplanned" in v:
                    second += f"; unplanned {records(v['unplanned'])}"
                    charge += f"; unplanned {account(x['unplanned_account'])}"
            t = x["tick"]
            rows.append((run, b, x["case"], certifier(v), second, served(v.get("n2")), charge,
                         "" if t["lag_ms"] is None else f"{t['lag_ms']:.1f} ms",
                         f"{t['late_ms']:.1f}/{t['slack_ms']:g} ms" + (f", {t['over']} over" if t["over"] else ""),
                         "ok" if x["ok"] else "MISMATCH"))
        if (a := r.get("records")) is not None:
            v = a["verdicts"]
            rows.append((run, "all", f"{len(v['ports'])} ports", "", "records " + records(v["refused"], by_block=True),
                         "", account(a["account"]), "", "", "ok" if a["ok"] else "MISMATCH"))
    widths = [max(len(row[k]) for row in rows) for k in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows]
    answers = [f"{run.split('/')[1]} " + ("none" if r["answers"] is None else
               f"{r['answers']} {'equal' if r['answers'] == r['reference'] else 'differ from'} the reference")
               for run, r in summary["runs"].items() if "answers" in r]
    lines += ["", "answers: " + ("; ".join(answers) or "no workload run")]
    if (w := summary["worst_tick"]) is not None:
        lines.append(f"worst proxy tick: {w['late_ms']:.1f} ms late against {w['slack_ms']:g} ms of slack ({w['at']}), "
                     f"{summary['ticks_over_slack']} ticks past the slack in the night")
    lines.append("steps: " + ", ".join(f"{s} {rc}" for s, rc in summary["exit"].items()))
    if summary["ok"]:
        lines.append("every verdict is the expected one")
    else:
        lines += [f"{len(summary['mismatches'])} MISMATCHES:", *(f"  {m}" for m in summary["mismatches"])]
    return "\n".join(lines) + "\n"


def finish(out: Path, exits: dict[str, int]) -> int:
    summary = summarize(out, exits)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    text = render(summary)
    (out / "summary.txt").write_text(text)
    print(text, end="")
    return 0 if summary["ok"] else 1


# The lab.

def step(name: str, argv: list[str], out: Path, cwd: Path = HERE) -> int:
    """Run one step, its output printed and appended to `out/<name>.txt`."""
    log = out / f"{name}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"== {name}: {' '.join(argv)}", flush=True)
    with open(log, "a") as fh, subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True) as p:
        for line in p.stdout:
            fh.write(line)
            sys.stdout.write(line)
        rc = p.wait()
    sys.stdout.flush()
    return rc


def busy() -> dict[str, list[str]]:
    """The lab's namespaces that have a process in them."""
    listed = subprocess.run(["ip", "netns", "list"], capture_output=True, text=True, check=True).stdout
    names = [line.split()[0] for line in listed.splitlines() if line.startswith("na-")]
    return {ns: pids for ns in names if (pids := subprocess.run(["sudo", "-n", "ip", "netns", "pids", ns],
                                                                capture_output=True, text=True, check=True).stdout.split())}


def shape(out: Path, *options: str) -> int:
    """The lab taken down and brought up again with `options` (`lab.txt`)."""
    return step("lab", [*MOCKDC, "down"], out) or step("lab", [*MOCKDC, "up", *options], out)


def torch_python(given: str | None, scratch: Path, out: Path) -> tuple[str | None, int]:
    if given:
        return given, 0
    if importlib.util.find_spec("torch") is not None:
        return sys.executable, 0
    venv = scratch / "venv"
    py = str(venv / "bin" / "python")
    for argv in (["uv", "venv", "-q", str(venv), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
                 ["uv", "pip", "install", "-q", "--python", py, "numpy"],
                 ["uv", "pip", "install", "-q", "--python", py, "torch", "--index-url",
                  "https://download.pytorch.org/whl/cpu"]):
        if rc := step("python", argv, out):
            return None, rc
    return py, 0


def night(out: Path, python: str | None, exits: dict[str, int]) -> None:
    """The steps, each step's exit code in `exits` as it ends."""
    exits["demo"] = step("demo", [sys.executable, "-B", "-m", "datacenter.demo"], out, ROOT / "benchmarks" / "datacenter")
    for options in dict.fromkeys(o for o, _, _ in MOCKDC_RUNS.values()):
        lab = " ".join(("lab", *options))
        exits[lab] = shape(out, *options)
        for run, (o, salt, strategies) in MOCKDC_RUNS.items():
            if o == options and not exits[lab]:
                blocks = [x for b, s in strategies.items() for x in ("--block", f"{b}={s}")]
                exits[run] = step(run, [*MOCKDC, "run", *options, *["--salt"] * salt, *blocks, "--out", str(out / run)],
                                  out)
    exits["lab default"] = shape(out)
    with tempfile.TemporaryDirectory() as scratch:
        py, exits["python"] = torch_python(python, Path(scratch), out)
        if py is not None and not exits["lab default"]:
            for name, (args, _, _) in WORKLOAD.items():
                run = f"workload/{name}"
                exits[run] = step(run, [py, "-B", "-m", "workload.lab_run", "run", "--out", str(out / run),
                                        "--delta-buckets", str(DELTA_BUCKETS), *args], out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", help="a Python with torch for the workload's runs (default: this one if it has torch, "
                    "else one built with uv in a scratch directory)")
    args = ap.parse_args(argv)
    if running := busy():
        raise SystemExit(f"the lab is busy, processes {running} run in its namespaces: wait until its run ends")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    exits: dict[str, int] = {}
    try:
        night(out, args.python, exits)
    finally:
        if busy() or step("lab", [*MOCKDC, "status"], out):
            exits["lab restored"] = shape(out)
        status = finish(out, exits)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
