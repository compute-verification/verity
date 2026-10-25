"""`verity bench all-protocols`'s runner: every period, one run of the torch loop audited by every protocol's leg.

    python -m verity_all_protocols.runner --out DIR --tools FILE [--periods N] [--period-s S] [--node NAME]
        [--tamper P,...] [--law L] [--cap N] [--requests 3,7] [--steps 2] [--onsite EXE] [--scratch DIR]

Per period P (0-based): the verifier registers the loop's circuit from its shape and run id (``serve --circuit-only``),
the loop serves once (`integrations/torch_loop`, run id ``RUN/pP``, so each period has a root of its own) and writes its
window into scratch; each leg of `legs.LEGS` audits that window (`decide`); and ``DIR/pP/report.json`` holds the period's
report, also appended to ``DIR/reports.jsonl``:

    {"format": "verity/all-protocols-period/v1", "period": P, "node": NAME, "run": RUN, "loop": {...},
     "protocols": {LEG: ROW, ...}, "composed": {...}, "horizon": {...}, "seconds": {...}}

A row is ``{"verdict": "missing", "owner": LANE}`` for a protocol with no leg, ``{"verdict": "error", "owner", "reason"}``
for a leg that couldn't decide, and otherwise its answer's verdict with ``protocol``, ``owner``, ``commitment``,
``reason``, ``quantity`` (what its acceptance certifies; null unless it accepts), ``delta`` (its profile's), ``runs`` and
the whole ``answer``. ``composed`` is `compose.composed`'s, ``horizon`` `compose.horizon`'s over the periods so far.

``--tools FILE``: JSON naming the built executables (`legs.Period.tools`). ``--tamper P,...``: those periods' loop records
outputs that are not its model's (`serve --tamper`) and its prover skips its self-check, so a drawn unit's proof is of a
false witness. Only the window's public files are kept in ``DIR`` (never a prover's file: its rows and salts).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import compose as C
from .legs import LEGS, Entry, Leg, Missing, Period, check_answer

FORMAT = "verity/all-protocols-period/v1"
ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "integrations" / "torch_loop"
PUBLIC = ("registration.json", "window.json", "index.json", "circuit.json", "serve.json")


def row_of(entry: Entry, leg: Leg) -> dict[str, Any]:
    """A leg's report row from its answer (checked to be a core answer first)."""
    a = check_answer(leg.answer)
    return {"protocol": a["protocol"], "owner": entry.owner, "verdict": a["verdict"], "reason": a["reason"],
            "commitment": a["commitment"], "quantity": leg.quantity if a["verdict"] == "accept" else None,
            "delta": (a["profile"] or {}).get("delta"), "runs": list(leg.runs), "answer": a}


def decide(period: Period, legs: Sequence[Entry] = LEGS, earlier: Sequence[dict] = (), period_s: float = 0,
           onsite: Sequence[str] | None = None, exe: str | None = None,
           clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Every leg's row on ``period``, the composed verdict and the horizon over ``earlier`` reports and this one."""
    rows: dict[str, dict] = {}
    cases: dict[str, dict | None] = {}
    seconds: dict[str, float] = {}
    for e in legs:
        if isinstance(e.leg, Missing):
            rows[e.name] = {"verdict": "missing", "owner": e.leg.owner}
            continue
        t = clock()
        out = period.out / e.name
        out.mkdir(parents=True, exist_ok=True)
        try:
            got = e.leg(dataclasses.replace(period, out=out))
            rows[e.name], cases[e.name] = row_of(e, got), got.case
        except Exception as x:  # noqa: BLE001: a leg's failure is its row's error, and the other legs still run
            (out / "error.txt").write_text(traceback.format_exc())
            rows[e.name] = {"verdict": "error", "owner": e.owner, "reason": f"{type(x).__name__}: {x}"[:2000]}
        seconds[e.name] = round(clock() - t, 1)
    report = {"format": FORMAT, "period": period.index, "node": period.node, "run": os.environ.get("RESEARCH_RUN_ID"),
              "loop": {"circuit": period.circuit, "run_root": period.run_root, "law": period.law, "cap": period.cap,
                       "tampered": bool(period.prover_env)},
              "protocols": rows, "seconds": seconds,
              "composed": C.composed(rows, {e.name: e.component for e in legs}, cases, onsite, exe)}
    report["horizon"] = C.horizon([*earlier, report], period_s)
    return report


def serve(argv: Sequence[str], scratch: Path) -> dict[str, Any]:
    """The torch loop's ``serve`` with ``argv``: its JSON line."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(LOOP), os.environ.get("PYTHONPATH")]))}
    p = subprocess.run([sys.executable, "-m", "verity_torch_loop.serve", *argv], cwd=ROOT, env=env, capture_output=True,
                       text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"verity_torch_loop.serve {' '.join(argv)} exited {p.returncode}: {p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def loop_period(index: int, run: str, a: argparse.Namespace, scratch: Path, out: Path) -> tuple[Period, dict]:
    """The period's loop: the circuit registered before it runs, then its window in scratch; its public files in ``out``."""
    shape = ["--law", a.law, "--cap", str(a.cap), "--requests", a.requests, "--steps", str(a.steps), "--run-id",
             f"{run}/p{index}"]
    circuit = serve([*shape, "--circuit-only"], scratch)["circuit"]
    window = scratch / f"p{index}" / "window"
    tampered = index in a.tamper
    got = serve([*shape, "--out", str(window), *(["--tamper"] if tampered else [])], scratch)
    (window / "serve.json").write_text(json.dumps(got, sort_keys=True))
    kept = out / "window"
    kept.mkdir(parents=True, exist_ok=True)
    for f in [*PUBLIC, *(p.name for p in window.glob("m*-public.bin"))]:
        if (window / f).exists():
            shutil.copy(window / f, kept / f)
    period = Period(index=index, node=a.node, records=window, out=out, circuit=circuit, run_root=got["run"], law=a.law,
                    cap=a.cap, tools=a.tools, prover_env={"FC_ZK_SELF_CHECK": "skip"} if tampered else {})
    return period, {"circuit_before": circuit, **got}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="JSON: the built executables by name (legs.Period.tools)")
    ap.add_argument("--periods", type=int, default=1)
    ap.add_argument("--period-s", type=float, default=3600, help="the period's length: each period starts this long after "
                    "the one before (at once when the one before took longer), and the month's horizon counts them")
    ap.add_argument("--node", default=os.environ.get("RESEARCH_MACHINE") or os.uname().nodename)
    ap.add_argument("--tamper", default="", help="periods whose loop tampers with its outputs, e.g. 1,3")
    ap.add_argument("--law", default="subset:2")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--requests", default="3,7")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--onsite", help="onsite's executable (`onsite check`), for when its components are ports")
    ap.add_argument("--scratch", help="where the loop's windows go (default a temporary directory, removed after)")
    a = ap.parse_args(argv)
    a.tamper = {int(x) for x in a.tamper.split(",") if x.strip()}
    a.tools = json.loads(Path(a.tools).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = os.environ.get("RESEARCH_RUN_ID") or "local"
    reports: list[dict] = []
    rc = 0
    with tempfile.TemporaryDirectory(dir=a.scratch) as scratch:
        t0 = time.monotonic()
        for p in range(a.periods):
            wait = t0 + p * a.period_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            pdir = out / f"p{p}"
            t = time.monotonic()
            try:
                period, loop = loop_period(p, run, a, Path(scratch), pdir)
            except Exception as x:  # noqa: BLE001: a failed loop is the period's error, and the next period still runs
                report = {"format": FORMAT, "period": p, "node": a.node, "run": run,
                          "error": f"the loop: {type(x).__name__}: {x}"[:3000]}
                rc = 1
            else:
                loop_s = round(time.monotonic() - t, 1)
                report = decide(period, LEGS, reports, a.period_s, exe=a.onsite)
                report["loop"].update({k: loop.get(k) for k in ("population", "members", "registration", "mismatched")})
                report["seconds"]["loop"] = loop_s
                reports.append(report)
                rc |= any(r["verdict"] == "error" for r in report["protocols"].values())
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
            with open(out / "reports.jsonl", "a") as f:
                f.write(json.dumps(report, sort_keys=True) + "\n")
            print("ALLP " + json.dumps(summary_line(report), sort_keys=True), flush=True)
    return rc


def summary_line(report: dict[str, Any]) -> dict[str, Any]:
    """The period's one line: each protocol's verdict and quantity, the composed verdict and the month's bound."""
    if "error" in report:
        return {"period": report["period"], "error": report["error"]}
    return {"period": report["period"], "tampered": report["loop"]["tampered"],
            "protocols": {n: {"verdict": r["verdict"], **({"quantity": r["quantity"]} if r.get("quantity") else {}),
                              **({"reason": r["reason"][:300]} if r.get("reason") else {})}
                          for n, r in report["protocols"].items()},
            "composed": {k: report["composed"].get(k) for k in ("verdict", "reason")},
            "horizon": {"integrity": report["horizon"]["integrity"], "month": report["horizon"]["month"]}}


if __name__ == "__main__":
    raise SystemExit(main())
