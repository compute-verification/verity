"""`verity bench all-protocols`'s runner: every period, one run of the torch loop audited by every protocol's leg.

    python -m verity_all_protocols.runner --out DIR --tools FILE [--periods N] [--period-s S] [--node NAME]
        [--tamper P,...] [--law L] [--cap N] [--requests 3,7] [--steps 2] [--onsite EXE] [--scratch DIR] [--registered]

Per period P (0-based): the verifier registers the loop's circuit from its shape and run id (``serve --circuit-only``),
the loop serves once (`integrations/torch_loop`, run id ``RUN/pP``, so each period has a root of its own) and writes its
window into scratch with ``period.json`` beside it; each protocol of `table.TABLE` audits it through its leg (`decide`);
and ``DIR/pP/report.json`` holds the period's report, also appended to ``DIR/reports.jsonl``:

    {"format": "verity/all-protocols-period/v1", "period": P, "node": NAME, "run": RUN, "loop": {...},
     "protocols": {NAME: ROW, ...}, "composed": {...}, "horizon": {...}, "seconds": {...}}

A row is ``{"verdict": "missing", "owner": LANE}`` for a protocol with no leg, ``{"verdict": "error", "owner", "reason"}``
for a leg that couldn't decide, and otherwise its answer's verdict with ``protocol``, ``owner``, ``commitment``,
``reason``, ``quantity`` (its answer's public outputs, what acceptance certifies; null unless it accepts), ``delta`` (its
profile's), ``epsilon`` (a sampled protocol's bound on wrong units as a share of its units), ``runs`` and the whole
``answer``. ``composed`` is `compose.composed`'s, ``horizon`` `compose.horizon`'s over the periods so far.

``--tools FILE``: JSON naming the built executables (``flock-circuit``, ``flock-verify``, ``flock-firewall``,
``m0-pythonpath``). ``--tamper P,...``: those periods' loop records outputs that are not its model's (`serve --tamper`)
and its prover skips its self-check, so a drawn unit's proof is of a false witness. Only the window's public files are
kept in ``DIR`` (never a prover's file: its rows and salts). ``--registered``: the loop reads its weights as registered
values (`serve --registered`), committed once before the first period (`serve --commit-weights`), their opening in
scratch and the verifier's pin in ``DIR/weights-pin.json``, which ``period.json`` names (``registered``) and the
integrity leg's audit holds every period's reads to; its members are then shared-row.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import compose as C
from .table import ANSWERS, PERIOD, TABLE, Entry, LegFn, check_answer, leg_of

FORMAT = "verity/all-protocols-period/v1"
ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "integrations" / "torch_loop"
PUBLIC = ("registration.json", "window.json", "index.json", "circuit.json", "serve.json", PERIOD)
#: ``--registered``'s weights, committed once a run: their opening (the prover's salts) in scratch, the verifier's pin
#: of their roots in ``DIR``
WEIGHTS_OPENING, WEIGHTS_PIN = "weights-opening.json", "weights-pin.json"


def epsilon(a: Mapping[str, Any]) -> str | None:
    """A sampled protocol's bound on wrong units (``public.bound``) over its checked level's size (`Core.Integrity`), or
    None for an answer that has no such bound."""
    try:
        bound, size = a["public"]["bound"], a["profile"]["levels"][-1]["size"]
    except (KeyError, IndexError, TypeError):
        return None
    return str(Fraction(bound, size)) if isinstance(bound, int) and isinstance(size, int) and size > 0 else None


def row_of(entry: Entry, a: Any, out: Path) -> dict[str, Any]:
    """A leg's report row from its answer (checked to be a core answer first)."""
    a = check_answer(a)
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    run = os.environ.get("RESEARCH_RUN_ID")
    accepted = a["verdict"] == "accept"
    return {"protocol": a["protocol"], "owner": entry.owner, "verdict": a["verdict"], "reason": a["reason"],
            "commitment": a["commitment"], "quantity": a["public"] if accepted else None,
            "delta": (a["profile"] or {}).get("delta") if isinstance(a["profile"], dict) else None,
            "epsilon": epsilon(a) if accepted else None, "runs": [*([run] if run else []), *runs], "answer": a}


def decide(period_dir: Path, node: str, out: Path, table: Sequence[Entry] = TABLE,
           legs: Callable[[str], LegFn | None] = leg_of, earlier: Sequence[dict] = (), period_s: float = 0,
           onsite: Sequence[str] | None = None, exe: str | None = None,
           clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Every protocol's row on the period in ``period_dir``, in ``table``'s order, each leg's answer written to
    ``period_dir/answers/<component>.json`` for the legs after it; the composed verdict; the horizon over ``earlier``
    reports and this one."""
    rows: dict[str, dict] = {}
    cases: dict[str, dict | None] = {}
    seconds: dict[str, float] = {}
    (period_dir / ANSWERS).mkdir(exist_ok=True)
    for e in table:
        fn = legs(e.component)
        if fn is None:
            rows[e.name] = {"verdict": "missing", "owner": e.owner}
            continue
        t = clock()
        lout = out / e.name
        lout.mkdir(parents=True, exist_ok=True)
        try:
            rows[e.name] = row_of(e, fn(period_dir, node, lout), lout)
        except Exception as x:  # noqa: BLE001: a leg's failure is its row's error, and the other legs still run
            (lout / "error.txt").write_text(traceback.format_exc())
            rows[e.name] = {"verdict": "error", "owner": e.owner, "reason": f"{type(x).__name__}: {x}"[:2000]}
        else:
            (period_dir / ANSWERS / f"{e.component}.json").write_text(json.dumps(rows[e.name]["answer"], sort_keys=True))
            case = lout / f"{e.component}.case.json"
            cases[e.name] = json.loads(case.read_text()) if case.exists() else None
        seconds[e.name] = round(clock() - t, 1)
    p = json.loads((period_dir / PERIOD).read_text())
    report = {"format": FORMAT, "period": p["period"], "node": node, "run": os.environ.get("RESEARCH_RUN_ID"),
              "loop": {"circuit": p["circuit"], "run_root": p["run_root"], "law": p["law"], "cap": p["cap"],
                       "tampered": bool(p.get("prover_env"))},
              "protocols": rows, "seconds": seconds,
              "composed": C.composed(rows, {e.name: e.component if e.onsite else None for e in table}, cases, onsite, exe)}
    report["horizon"] = C.horizon([*earlier, report], period_s)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(period_dir / ANSWERS, out / ANSWERS, dirs_exist_ok=True)
    return report


def serve(argv: Sequence[str]) -> dict[str, Any]:
    """The torch loop's ``serve`` with ``argv``: its JSON line."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(LOOP), os.environ.get("PYTHONPATH")]))}
    p = subprocess.run([sys.executable, "-m", "verity_torch_loop.serve", *argv], cwd=ROOT, env=env, capture_output=True,
                       text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"verity_torch_loop.serve {' '.join(argv)} exited {p.returncode}: {p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def loop_period(index: int, run: str, a: argparse.Namespace, scratch: Path, out: Path) -> tuple[Path, dict]:
    """The period's loop: the circuit registered before it runs, then its window in scratch with ``period.json``; its
    public files in ``out/window``. -> the period's directory and what the loop printed."""
    weights = a.weights
    shape = ["--law", a.law, "--cap", str(a.cap), "--requests", a.requests, "--steps", str(a.steps), "--run-id",
             f"{run}/p{index}", *(["--registered"] if weights else [])]
    circuit = serve([*shape, "--circuit-only"])["circuit"]
    window = scratch / f"p{index}"
    tampered = index in a.tamper
    got = serve([*shape, "--out", str(window), *(["--tamper"] if tampered else []),
                 *(["--weights", str(weights["opening"])] if weights else [])])
    (window / "serve.json").write_text(json.dumps(got, sort_keys=True))
    (window / PERIOD).write_text(json.dumps({
        "period": index, "periods": a.periods, "node": a.node, "circuit": circuit, "run_root": got["run"], "law": a.law,
        "cap": a.cap, "tools": a.tools, "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {},
        **({"registered": str(weights["pin"])} if weights else {})}, sort_keys=True))
    kept = out / "window"
    kept.mkdir(parents=True, exist_ok=True)
    for f in [*PUBLIC, *(p.name for p in window.glob("m*-public.bin"))]:
        if (window / f).exists():
            shutil.copy(window / f, kept / f)
    return window, {"circuit_before": circuit, **got}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="JSON: the built executables by name")
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
    ap.add_argument("--registered", action="store_true", help="the loop reads its weights as registered values "
                    "(`serve --registered`), committed once before the first period")
    a = ap.parse_args(argv)
    a.tamper = {int(x) for x in a.tamper.split(",") if x.strip()}
    a.tools = json.loads(Path(a.tools).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = os.environ.get("RESEARCH_RUN_ID") or "local"
    reports: list[dict] = []
    rc = 0
    with tempfile.TemporaryDirectory(dir=a.scratch) as scratch:
        a.weights = {"opening": Path(scratch) / WEIGHTS_OPENING, "pin": out / WEIGHTS_PIN} if a.registered else None
        if a.weights:
            serve(["--commit-weights", str(a.weights["opening"]), "--pin", str(a.weights["pin"])])
        t0 = time.monotonic()
        for p in range(a.periods):
            wait = t0 + p * a.period_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            pdir = out / f"p{p}"
            t = time.monotonic()
            try:
                period_dir, loop = loop_period(p, run, a, Path(scratch), pdir)
            except Exception as x:  # noqa: BLE001: a failed loop is the period's error, and the next period still runs
                report = {"format": FORMAT, "period": p, "node": a.node, "run": run,
                          "error": f"the loop: {type(x).__name__}: {x}"[:3000]}
                rc = 1
            else:
                loop_s = round(time.monotonic() - t, 1)
                report = decide(period_dir, a.node, pdir, earlier=reports, period_s=a.period_s, exe=a.onsite)
                report["loop"].update({k: loop.get(k) for k in ("population", "members", "registration", "mismatched")})
                report["seconds"]["loop"] = loop_s
                reports.append(report)
                rc |= any(r["verdict"] == "error" for r in report["protocols"].values())
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
            with open(out / "reports.jsonl", "a") as f:
                f.write(json.dumps(report, sort_keys=True) + "\n")
            print("ALLP " + json.dumps(summary_line(report), sort_keys=True), flush=True)
    return int(rc)


def summary_line(report: dict[str, Any]) -> dict[str, Any]:
    """The period's one line: each protocol's verdict and quantity, the composed verdict and the month's bound."""
    if "error" in report:
        return {"period": report["period"], "error": report["error"]}
    return {"period": report["period"], "tampered": report["loop"]["tampered"],
            "protocols": {n: {"verdict": r["verdict"], **({"quantity": r["quantity"]} if r.get("quantity") else {}),
                              **({"epsilon": r["epsilon"], "delta": r["delta"]} if r.get("epsilon") else {}),
                              **({"reason": r["reason"][:300]} if r.get("reason") else {})}
                          for n, r in report["protocols"].items()},
            "composed": {k: report["composed"].get(k) for k in ("verdict", "reason")},
            "horizon": {"integrity": report["horizon"]["integrity"], "month": report["horizon"]["month"]}}


if __name__ == "__main__":
    raise SystemExit(main())
