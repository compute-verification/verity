"""`verity bench all-protocols`'s runner: every period, one run of the torch loop audited by every protocol's leg.

    python -m verity_all_protocols.runner --out DIR --tools FILE [--periods N] [--period-s S] [--node NAME]
        [--tamper P,...] [--law L] [--cap N] [--requests 3,7] [--steps 2] [--onsite EXE] [--scratch DIR] [--registered]
        [--tie [--tie-tamper P,...]]

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

``--tie``: the loop's outputs tied to what the network certifier relayed (`tie`). Before the first period the
communication leg's session is relayed (its `relay`, in scratch, where the leg finds it) with the reading program's
digest (``serve --circuit-only``'s, the same every period), and the network certifier commits the output it relayed in
each link-window as the registered value ``out`` and signs its head with its session's own key. Each period, before
the loop serves, the verifier pins the root of the period's link-window's record (``DIR/pP/out-record.json``) from
that record alone under the deployment's certifier key and session (``DIR/pP/pin.json``, beside the weights' pin,
which ``period.json`` then names), and the loop serves with each output unit's ``o0`` a registered read of ``out``,
opened from the rows and salts the certifier handed the prover (`serve --tie --out-opening`). ``period.json``'s ``tie``
names the value,
the port, the stratum the integrity audit proves whole (`tie.STRATUM`), the relay's run and link-window and the pinned
root. ``--tie-tamper P,...``: those periods' output is the ``CERTIFIER_TAMPER=byte`` relay's, one output byte changed on
the egress, which Lean's registered read of ``o0`` must refuse. ``--tie`` needs a work law (``--law work:K``, default
``work:64``; ``subset:2`` without it): a5 proves the output class only beside a streamed work draw, so any other law is
refused here rather than in every period's audit.
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

from . import compose as C, tie as T
from .table import ANSWERS, PERIOD, TABLE, Entry, LegFn, check_answer, leg_of

FORMAT = "verity/all-protocols-period/v1"
ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "integrations" / "torch_loop"
PUBLIC = ("registration.json", "window.json", "index.json", "circuit.json", "serve.json", PERIOD)
#: ``--registered``'s weights, committed once a run: their opening (the prover's salts) in scratch, the verifier's pin
#: of their roots in ``DIR``
WEIGHTS_OPENING, WEIGHTS_PIN = "weights-opening.json", "weights-pin.json"
#: ``--law``'s default, and its default with ``--tie``, whose output class a5 proves only beside a streamed work draw
LAW, TIE_LAW = "subset:2", "work:64"


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
                       "tampered": bool(p.get("prover_env")),
                       **({"tie": {k: p["tie"][k] for k in ("strategy", "window", "stratum", "root")}}
                          if p.get("tie") else {})},
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


def shape_of(index: int, run: str, a: argparse.Namespace) -> list[str]:
    """The loop's shape and period ``index``'s run id, as ``serve`` takes them."""
    return ["--law", a.law, "--cap", str(a.cap), "--requests", a.requests, "--steps", str(a.steps), "--run-id",
            f"{run}/p{index}", *(["--registered"] if a.weights else []),
            *(["--tie"] if getattr(a, "tie", False) else [])]


def loop_period(index: int, run: str, a: argparse.Namespace, scratch: Path, out: Path) -> tuple[Path, dict]:
    """The period's loop: the circuit registered before it runs, then its window in scratch with ``period.json``; its
    public files in ``out/window``. With ``--tie``, the out record the network certifier signed for the period's
    link-window pinned before the loop serves (`tie.period`), the period's pins ``out/pin.json`` (the weights' and
    ``out``'s). -> the period's directory and what the loop printed."""
    weights, tied = a.weights, getattr(a, "tie", False)
    shape = shape_of(index, run, a)
    registered = serve([*shape, "--circuit-only"])
    circuit, pins, tie = registered["circuit"], weights and weights["pin"], None
    window = scratch / f"p{index}"
    tampered = index in a.tamper
    if tied:
        rows = len(a.requests.split(",")) * a.steps * a.relay["h"]
        tie = T.period(a.relay, index, "byte" if index in a.tie_tamper else "honest", registered["program"], rows,
                       scratch, out)
        pins = out / "pin.json"
        pins.write_text(json.dumps([*([json.loads(weights["pin"].read_text())] if weights else []), tie.pop("pin")],
                                   indent=1, sort_keys=True))
    got = serve([*shape, "--out", str(window), *(["--tamper"] if tampered else []),
                 *(["--weights", str(weights["opening"])] if weights else []),
                 *(["--out-opening", str(tie.pop("opening"))] if tie else [])])
    (window / "serve.json").write_text(json.dumps(got, sort_keys=True))
    (window / PERIOD).write_text(json.dumps({
        "period": index, "periods": a.periods, "node": a.node, "circuit": circuit, "run_root": got["run"], "law": a.law,
        "cap": a.cap, "tools": a.tools, "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {},
        **({"registered": str(pins)} if pins else {}), **({"tie": tie} if tie else {})}, sort_keys=True))
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
    ap.add_argument("--law", help=f"the draw's law (default {LAW}, and {TIE_LAW} with --tie, which needs a work law)")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--requests", default="3,7")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--onsite", help="onsite's executable (`onsite check`), for when its components are ports")
    ap.add_argument("--scratch", help="where the loop's windows go (default a temporary directory, removed after)")
    ap.add_argument("--registered", action="store_true", help="the loop reads its weights as registered values "
                    "(`serve --registered`), committed once before the first period")
    ap.add_argument("--tie", action="store_true", help="each output unit's o0 reads the output the network certifier "
                    "relayed in the period's link-window, committed and signed by it and pinned from its record (`tie`)")
    ap.add_argument("--tie-tamper", default="", help="with --tie: periods whose output is the CERTIFIER_TAMPER=byte "
                    "relay's, one egress byte changed, e.g. 0")
    a = ap.parse_args(argv)
    a.tamper = {int(x) for x in a.tamper.split(",") if x.strip()}
    a.tie_tamper = {int(x) for x in a.tie_tamper.split(",") if x.strip()}
    if a.tie_tamper and not a.tie:
        ap.error("--tie-tamper needs --tie")
    a.law = a.law or (TIE_LAW if a.tie else LAW)
    if a.tie and a.law.partition(":")[0] != "work":
        ap.error(f"--tie proves the output class whole beside a streamed work draw, and a5 refuses it under any other "
                 f"law (output-class-needs-a-streamed-work-draw): give --law work:K (the default with --tie is "
                 f"{TIE_LAW}), not {a.law}")
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
        if a.tie:
            program = serve([*shape_of(0, run, a), "--circuit-only"])["program"]
            a.relay = T.relay(Path(scratch), [int(x) for x in a.requests.split(",")], a.steps, a.periods, a.tools,
                              bool(a.tie_tamper), program)
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
            **({"tie": report["loop"]["tie"]["strategy"]} if report["loop"].get("tie") else {}),
            "protocols": {n: {"verdict": r["verdict"], **({"quantity": r["quantity"]} if r.get("quantity") else {}),
                              **({"epsilon": r["epsilon"], "delta": r["delta"]} if r.get("epsilon") else {}),
                              **({"reason": r["reason"][:300]} if r.get("reason") else {})}
                          for n, r in report["protocols"].items()},
            "composed": {k: report["composed"].get(k) for k in ("verdict", "reason")},
            "horizon": {"integrity": report["horizon"]["integrity"], "month": report["horizon"]["month"]}}


if __name__ == "__main__":
    raise SystemExit(main())
