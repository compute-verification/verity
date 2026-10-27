"""`verity example toy-onsite`'s runner: every window, one run of the toy audited by every protocol's leg, and the
result over each monitoring period of a datacenter.

    python -m verity_all_protocols.runner --out DIR --tools FILE [--windows N] [--start I] [--window-s S] [--jobs 1]
        [--session-jobs 1] [--node NAME] [--tamper W,...] [--law L] [--delta D] [--cap N] [--requests 3,7] [--steps 2] [--pouw h1]
        [--deployment ID] [--onsite EXE] [--scratch DIR] [--registered] [--wired] [--tie [--tie-tamper W,...]] [--of-record]
        [--check]

A window is one stretch of one machine's serving and its audits, and a monitoring period is the many windows on every
machine of a deployment a result is about (`ARCHITECTURE.md`'s Glossary). Per window W (``--start`` on): the verifier
registers the loop's circuit from its shape and run id (``serve --circuit-only``), the loop serves once
(`integrations/torch_loop`, run id ``RUN/wW``, so each window has a root of its own) and writes its window into scratch
with ``period.json`` beside it (the legs' records, `table.PERIOD`); each protocol of `table.TABLE` audits it through its leg
(`decide`). Outer proofs go through the firewall one at a time, one firewalled session on the run whichever window it
belongs to, so the windows are audited one after another and each audit's drawn members' proof sessions too (``--jobs``
and ``--session-jobs`` take only 1); only what never goes through the firewall, the next window's loop, overlaps an
audit. ``DIR/wW/report.json`` holds the window's report, also appended to ``DIR/reports.jsonl``:

    {"format": "verity/all-protocols-window/v1", "window": W, "node": NAME, "run": RUN, "loop": {...},
     "protocols": {NAME: ROW, ...}, "composed": {...}, "seconds": {...}}

After each window ``DIR/monitoring-period.json`` is `monitoring.monitoring` over the windows so far: integrity's pooled
total over a week, a month and a year of the census deployment (``--deployment``) at δ = 1% and 10⁻⁶, the per-window
figure beside it, and what is proved and what scaled. Its line (``ALLP``) leads with it. Windows run on several machines
(``--start`` apart) pool with ``python -m verity_all_protocols.monitoring DIR1/reports.jsonl DIR2/reports.jsonl``.

A row is ``{"verdict": "missing", "owner": LANE}`` for a protocol with no leg, ``{"verdict": "error", "owner", "reason"}``
for a leg that couldn't decide, and otherwise its answer's verdict with ``protocol``, ``owner``, ``commitment``,
``reason``, ``quantity`` (its answer's public outputs, what acceptance certifies; null unless it accepts), ``delta`` (its
profile's), ``epsilon`` (a sampled protocol's bound on wrong units as a share of its units), ``runs`` and the whole
``answer``. ``composed`` is `compose.composed`'s.

``--tools FILE``: JSON naming the built executables (``flock-circuit``, ``flock-verify``, ``flock-firewall``,
``m0-pythonpath``, ``pous-run-protocol``, ``pous-setup``, ``remote-audit-protocol``). ``--tamper W,...``: those windows'
loop records outputs that are not its model's (`serve --tamper`) and its prover skips its self-check, so a drawn unit's
proof is of a false witness. ``--pouw h1`` (until ALLP-H1LOOP's ``serve --pouw h1`` lands): the loop also writes PoUW's
window (its linears as the pinned H1 tile) to ``<window>/pouw``, and ``verity_torch_loop.pouw_audit`` audits it into
``<window>/pouw/answer.json`` before the legs run. Only the window's public files are kept in ``DIR`` (never a prover's
file: its rows and salts). ``--registered``: the loop reads its weights as registered values (`serve --registered`),
committed once before the first window (`serve --commit-weights`), their opening in scratch and the verifier's pin in
``DIR/weights-pin.json``, which ``period.json`` names (``registered``) and the integrity leg's audit holds every window's
reads to; its members are then shared-row. ``period.json``'s ``periods`` is the windows the run's one certifier session
serves (`legs/certifier.py`), ``--windows``, so each window's ``period`` (its place in this run) is one of them.

``--tie``: the loop's outputs tied to what the network certifier relayed (`tie`). Before the first window the
communication leg's session is relayed (its `relay`, in scratch, where the leg finds it) with the output value's shape
(``serve --circuit-only``'s ``out_shape``, the same every window: the reading program's digest and the circuit's
registered read of ``out``, one word a row, or with ``--wired`` a down_proj unit's whole row), and the network certifier
commits the output it relayed in each link-window as the registered value ``out`` at that shape and signs its head with
its session's own key. Each window, before
the loop serves, the verifier pins the root of its period's link-window's record (``DIR/wW/out-record.json``) from
that record alone under the deployment's certifier key and session (``DIR/wW/pin.json``, beside the weights' pin,
which ``period.json`` then names), and the loop serves with each output unit's ``o0`` a registered read of ``out``,
opened from the rows and salts the certifier handed the prover (`serve --tie --out-opening`). ``period.json``'s ``tie``
names the value, the port, the stratum the integrity audit proves whole (`tie.STRATUM`), the relay's run and
link-window and the pinned root. ``--tie-tamper W,...``: those windows' output is the ``CERTIFIER_TAMPER=byte``
relay's, one output byte changed on the egress, which Lean's registered read of ``o0`` must refuse. ``--tie`` needs a
work law (``--law work:K``, default ``work:64``, and `WIRED_TIE_LAW` with ``--wired``): a5 proves the output class only
beside a streamed work draw, so any other law is refused here rather than in every window's audit.

``--wired``: the loop's wired circuit (`serve --wired`), its weights with ``--registered`` committed one row a linear.
With ``--tie`` each down_proj unit's ``o0`` reads its row of ``out``, request-major, and the next step's plain quantize
unit's ``x`` the same row (`loop.TorchLoop.wired_circuit`).

``--of-record``: S1's run of record (`OF_RECORD`), the one way a run counts as its evidence. It sets ``--tie --pouw h1
--registered --law work:64 --jobs 1 --session-jobs 1`` (with ``--wired`` ``--law`` `WIRED_TIE_LAW`; and
``CERTIFIER_TAMPER=byte`` under ``--tie-tamper``) and refuses a command that contradicts any of them, with the reason
(`of_record`). ``--of-record --wired`` is the joint run, every layer chained and the output tied; an unwired one names
the gap in its mode (`WIRING_GAP`): the wiring is cited from the wired run of record (`COMPANION`). Every
run writes its mode, whether it is of record, and its exact flags to ``DIR/mode.json``, each window's ``period.json`` and
report (``mode``) and its ``ALLP`` line, so a run's evidence says whether it counts. ``--check`` only checks the command
and prints its mode, which ``run.sh`` does before it builds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import compose as C, monitoring as M, tie as T
from .table import ANSWERS, PERIOD, TABLE, Entry, LegFn, check_answer, leg_of

FORMAT = "verity/all-protocols-window/v1"
ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "integrations" / "torch_loop"
PUBLIC = ("registration.json", "window.json", "index.json", "circuit.json", "serve.json", PERIOD)
#: ``--registered``'s weights, committed once a run: their opening (the prover's salts) in scratch, the verifier's pin
#: of their roots in ``DIR``
WEIGHTS_OPENING, WEIGHTS_PIN = "weights-opening.json", "weights-pin.json"
#: ``--law``'s default (k = 64: about 74 wrong units over a whole monitoring period at δ = 1% and n = 1024), and its
#: default with ``--tie``, whose output class a5 proves only beside a streamed work draw
LAW, TIE_LAW = "subset:64", "work:64"
CAP = 8
#: the tie's work law on the wired toy, which has 14 units, so its work law draws at most k = 14
WIRED_TIE_LAW = "work:14"
#: S1's run of record (``--of-record``): the network certifier's output tie (``--tie``, network-accounting's), PoUW's
#: window (``--pouw h1``, whose leg refuses a window without it), the weights as registered values (``--registered``:
#: unregistered, they count as free input), the tie's work law (`WIRED_TIE_LAW` with ``--wired``), and one firewalled
#: session at a time
OF_RECORD = {"tie": True, "pouw": "h1", "registered": True, "law": TIE_LAW, "jobs": 1, "session_jobs": 1}
#: what an unwired run of record leaves to the wired one (``--of-record --wired``), named in its mode
WIRING_GAP = "the toy's wiring is cited from the wired run of record (--of-record --wired)"
#: the wired run of record, every layer chained and the output tied to what the network certifier relayed
COMPANION = "--of-record --wired --windows 2 --tie-tamper 0"


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
           legs: Callable[[str], LegFn | None] = leg_of, onsite: Sequence[str] | None = None, exe: str | None = None,
           clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Every protocol's row on the window in ``period_dir``, in ``table``'s order, each leg's answer written to
    ``period_dir/answers/<component>.json`` for the legs after it, and the composed verdict."""
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
    report = {"format": FORMAT, "window": p["window"], "node": node, "run": os.environ.get("RESEARCH_RUN_ID"),
              "loop": {"circuit": p["circuit"], "run_root": p["run_root"], "law": p["law"], "cap": p["cap"],
                       "delta": p.get("delta"), "tampered": bool(p.get("prover_env")), "pouw": p.get("pouw"),
                       **({"tie": {k: p["tie"][k] for k in ("strategy", "window", "stratum", "root")}}
                          if p.get("tie") else {})},
              "mode": p.get("mode"), "protocols": rows, "seconds": seconds,
              "composed": C.composed(rows, {e.name: e.component if e.onsite else None for e in table}, cases, onsite, exe)}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(period_dir / ANSWERS, out / ANSWERS, dirs_exist_ok=True)
    return report


def _python(module: str, argv: Sequence[str], log: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(LOOP), os.environ.get("PYTHONPATH")]))}
    if log is None:
        return subprocess.run([sys.executable, "-m", module, *argv], cwd=ROOT, env=env, capture_output=True, text=True,
                              check=False)
    with open(log.with_suffix(".log"), "w") as o, open(log.with_suffix(".err"), "w") as e:
        return subprocess.run([sys.executable, "-m", module, *argv], cwd=ROOT, env=env, stdout=o, stderr=e, check=False)


def serve(argv: Sequence[str]) -> dict[str, Any]:
    """The torch loop's ``serve`` with ``argv``: its JSON line."""
    p = _python("verity_torch_loop.serve", argv)
    if p.returncode != 0:
        raise RuntimeError(f"verity_torch_loop.serve {' '.join(argv)} exited {p.returncode}: {p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def shape_of(index: int, run: str, a: argparse.Namespace) -> list[str]:
    """The loop's shape and window ``index``'s run id, as ``serve`` takes them."""
    return ["--law", a.law, "--cap", str(a.cap), "--requests", a.requests, "--steps", str(a.steps), "--run-id",
            f"{run}/w{index}", *(["--pouw", a.pouw] if a.pouw else []), *(["--registered"] if a.weights else []),
            *(["--tie"] if getattr(a, "tie", False) else []), *(["--wired"] if getattr(a, "wired", False) else [])]


def loop_window(index: int, run: str, a: argparse.Namespace, scratch: Path, out: Path) -> tuple[Path, dict]:
    """The window's loop: the circuit registered before it runs, then its window in scratch with ``period.json`` (and with
    ``--pouw h1`` PoUW's window in its ``pouw/``); its public files in ``out/window``. With ``--tie``, the out record the
    network certifier signed for its period's link-window pinned before the loop serves (`tie.period`), the window's pins
    ``out/pin.json`` (the weights' and ``out``'s). -> the window's directory and what the loop printed."""
    weights, tied = a.weights, getattr(a, "tie", False)
    shape = shape_of(index, run, a)
    registered = serve([*shape, "--circuit-only"])
    pins, tie = weights and weights["pin"], None
    window = scratch / f"w{index}"
    tampered = index in a.tamper
    if tied:
        tie = T.period(a.relay, index - getattr(a, "start", 0), "byte" if index in a.tie_tamper else "honest",
                       registered["out_shape"], scratch, out)
        pins = out / "pin.json"
        pins.write_text(json.dumps([*([json.loads(weights["pin"].read_text())] if weights else []), tie.pop("pin")],
                                   indent=1, sort_keys=True))
    got = serve([*shape, "--out", str(window), *(["--tamper"] if tampered else []),
                 *(["--weights", str(weights["opening"])] if weights else []),
                 *(["--out-opening", str(tie.pop("opening"))] if tie else [])])
    (window / "serve.json").write_text(json.dumps(got, sort_keys=True))
    (window / PERIOD).write_text(json.dumps({
        "window": index, "period": index - getattr(a, "start", 0), "periods": a.windows, "node": a.node,
        "circuit": registered["circuit"], "run_root": got["run"], "law": a.law, "delta": a.delta, "cap": a.cap,
        "tools": a.tools,
        "pouw": {"mode": a.pouw, "circuit": registered.get("pouw")} if a.pouw else None,
        "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {},
        "mode": getattr(a, "mode", None),
        **({"registered": str(pins)} if pins else {}), **({"tie": tie} if tie else {})}, sort_keys=True))
    kept = out / "window"
    kept.mkdir(parents=True, exist_ok=True)
    for f in [*PUBLIC, *(p.name for p in window.glob("m*-public.bin"))]:
        if (window / f).exists():
            shutil.copy(window / f, kept / f)
    return window, {"circuit_before": registered["circuit"], **got}


def pouw_audit(period_dir: Path, out: Path) -> dict[str, Any]:
    """PoUW's audit of the window's H1 window (``verity_torch_loop.pouw_audit``, ALLP-H1LOOP's), before the legs: it writes
    ``<window>/pouw/answer.json``, which PoUW's leg reads; its proving tree stays in scratch, and only its ``pouw.json`` and
    log are kept in ``out``. -> its rc and ``POUW`` line."""
    p = json.loads((period_dir / PERIOD).read_text())
    t = p["tools"]
    out.mkdir(parents=True, exist_ok=True)
    argv = ["--window", str(period_dir / "pouw"), "--circuit", p["pouw"]["circuit"], "--run", p["run_root"],
            "--m0-pythonpath", t["m0-pythonpath"], "--bin", t["flock-circuit"], "--lean", t["flock-verify"],
            "--firewall", t["flock-firewall"], "--out", str(period_dir.parent / f"{period_dir.name}-pouw"),
            *(["--forged"] if p.get("prover_env") else [])]
    r = _python("verity_torch_loop.pouw_audit", argv, out / "pouw-audit")
    kept = period_dir.parent / f"{period_dir.name}-pouw" / "pouw.json"
    if kept.exists():
        shutil.copy(kept, out / "pouw.json")
    lines = [x for x in (out / "pouw-audit.log").read_text().splitlines() if x.startswith("POUW ")]
    return {"rc": r.returncode, "line": json.loads(lines[-1].split(" ", 1)[1]) if lines else None}


def window(index: int, run: str, a: argparse.Namespace, scratch: Path, out: Path,
           loop: tuple[Path, dict] | Exception) -> dict[str, Any]:
    """One window's report from its loop's output (or the loop's failure): PoUW's H1 audit first under ``--pouw``, then
    every leg (`decide`)."""
    wdir = out / f"w{index}"
    if isinstance(loop, Exception):
        return {"format": FORMAT, "window": index, "node": a.node, "run": run,
                "error": f"the loop: {type(loop).__name__}: {loop}"[:3000]}
    period_dir, got = loop
    t = time.monotonic()
    pre = pouw_audit(period_dir, wdir / "work") if a.pouw else None
    report = decide(period_dir, a.node, wdir, exe=a.onsite)
    report["loop"].update({k: got.get(k) for k in ("population", "members", "registration", "mismatched")})
    report["seconds"]["loop"] = got.get("_loop_s")
    report["seconds"]["audits"] = round(time.monotonic() - t, 1)
    if pre is not None:
        report["pouw_audit"] = pre
    return report


#: AGENTS.md's live-coins rule (a5's `ONE_AT_A_TIME`), which keeps the run to one firewalled session at a time
ONE_AT_A_TIME = ("outer proofs go through the firewall one at a time (AGENTS.md, \"Live coins only\"): one firewalled "
                 "session at a time on the run, whichever window it belongs to")


def only_one(what: str) -> Callable[[str], int]:
    """An option whose only value is 1 (`ONE_AT_A_TIME`), kept so commands that pass it still run."""
    def parse(s: str) -> int:
        if s.strip() != "1":
            raise argparse.ArgumentTypeError(f"{s}: {ONE_AT_A_TIME}, so {what}; pass 1 or omit it")
        return 1
    return parse


one_window = only_one("the windows are audited one at a time")
one_session = only_one("an audit runs its sessions one at a time")


def of_record(a: argparse.Namespace) -> str | None:
    """``--of-record``'s flags (`OF_RECORD`, its law `WIRED_TIE_LAW` with ``--wired``) set on ``a``, or why the command
    contradicts them. ``--tie`` and ``--pouw h1`` have no other value to contradict them, and ``--jobs`` and
    ``--session-jobs`` take only 1 (`ONE_AT_A_TIME`)."""
    want = of_record_flags(a)
    if a.law not in (None, want["law"]):
        return (f"S1's run of record draws under {want['law']} (the tie's output class needs a streamed work draw, and "
                f"the run's figures are stated at k = {want['law'].partition(':')[2]}), not --law {a.law}: drop --law")
    tamper = os.environ.get("CERTIFIER_TAMPER")
    if a.tie_tamper and tamper not in (None, "byte"):
        return (f"--tie-tamper ties a window to the CERTIFIER_TAMPER=byte relay (one egress byte changed), not "
                f"CERTIFIER_TAMPER={tamper}: unset it")
    for k, v in want.items():
        setattr(a, k, v)
    if a.tie_tamper:
        os.environ["CERTIFIER_TAMPER"] = "byte"
    return None


def of_record_flags(a: argparse.Namespace) -> dict[str, Any]:
    """The flags a run of record runs with: `OF_RECORD`, under ``--wired`` at `WIRED_TIE_LAW`."""
    return {**OF_RECORD, **({"law": WIRED_TIE_LAW} if a.wired else {})}


def of_record_unmet(a: argparse.Namespace) -> list[str]:
    """The `OF_RECORD` flags a run that says it is of record doesn't run with (none once `of_record` set them)."""
    return [k for k, v in of_record_flags(a).items() if getattr(a, k) != v]


def mode_of(a: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    """The run's mode: whether it is S1's run of record, the command line as given, and the flags it ran with."""
    return {"of_record": bool(a.of_record), "argv": list(argv),
            "flags": {"tie": a.tie, "pouw": a.pouw, "law": a.law, "jobs": a.jobs, "session_jobs": a.session_jobs,
                      "windows": a.windows, "start": a.start, "tamper": sorted(a.tamper),
                      "tie_tamper": sorted(a.tie_tamper), "registered": a.registered, "wired": a.wired,
                      "requests": a.requests, "steps": a.steps, "cap": a.cap, "delta": a.delta,
                      "certifier_tamper": os.environ.get("CERTIFIER_TAMPER")},
            **({"gap": WIRING_GAP, "companion": COMPANION} if a.of_record and not a.wired else {})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="JSON: the built executables by name")
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="the first window's index (another machine's windows start apart)")
    ap.add_argument("--window-s", type=float, default=0, help="each window's loop starts this long after the one before (at "
                    "once when the one before took longer)")
    ap.add_argument("--jobs", type=one_window, default=1, help="1 only: the windows are audited one at a time (ONE_AT_A_TIME)")
    ap.add_argument("--session-jobs", type=one_session, default=1,
                    help="1 only: each integrity audit runs its sessions one at a time (ONE_AT_A_TIME)")
    ap.add_argument("--node", default=os.environ.get("RESEARCH_MACHINE") or os.uname().nodename)
    ap.add_argument("--tamper", default="", help="windows whose loop tampers with its outputs, e.g. 1,3")
    ap.add_argument("--law", help=f"the proof service's law of each window (default {LAW}, and {TIE_LAW} with --tie, "
                    "which needs a work law)")
    ap.add_argument("--delta", default="1/100", help="each window's audit record's own δ")
    ap.add_argument("--cap", type=int, help=f"units a member (default {CAP}, and with --wired the largest divisor of the "
                    "request count up to it: a wired member must not straddle the step-0 up_proj units and the rest)")
    ap.add_argument("--requests", default="3,7")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--pouw", choices=["h1"], help="the loop's PoUW window and its audit (ALLP-H1LOOP's, until it lands)")
    ap.add_argument("--deployment", default=M.REFERENCE, help="the census deployment the monitoring periods are over")
    ap.add_argument("--onsite", help="onsite's executable (`onsite check`), for when its components are ports")
    ap.add_argument("--scratch", help="where the loop's windows go (default a temporary directory, removed after)")
    ap.add_argument("--registered", action="store_true", help="the loop reads its weights as registered values "
                    "(`serve --registered`), committed once before the first window")
    ap.add_argument("--tie", action="store_true", help="each output unit's o0 reads the output the network certifier "
                    "relayed in the window's link-window, committed and signed by it and pinned from its record (`tie`)")
    ap.add_argument("--tie-tamper", default="", help="with --tie: windows whose output is the CERTIFIER_TAMPER=byte "
                    "relay's, one egress byte changed, e.g. 0")
    ap.add_argument("--wired", action="store_true", help="the loop's wired circuit (`serve --wired`: each linear and each "
                    "quantize one unit, every activation a tie; with --registered the weights one row a linear)")
    ap.add_argument("--of-record", action="store_true", help="S1's run of record: --tie --pouw h1 --registered --law "
                    f"work:64 ({WIRED_TIE_LAW} with --wired) --jobs 1 --session-jobs 1, refusing a command that "
                    "contradicts them (OF_RECORD)")
    ap.add_argument("--check", action="store_true", help="only check the command and print its mode (run.sh does, "
                    "before it builds, so a refused run of record costs nothing)")
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    a.tamper = {int(x) for x in a.tamper.split(",") if x.strip()}
    a.tie_tamper = {int(x) for x in a.tie_tamper.split(",") if x.strip()}
    if a.of_record and (refused := of_record(a)):
        ap.error(f"--of-record: {refused}")
    if a.of_record and (unmet := of_record_unmet(a)):
        ap.error(f"--of-record: S1's run of record runs with {of_record_flags(a)}, and this one doesn't for "
                 f"{', '.join(unmet)}")
    if a.tie_tamper and not a.tie:
        ap.error("--tie-tamper needs --tie")
    r = len(a.requests.split(","))
    a.cap = a.cap or (math.gcd(CAP, r) if a.wired else CAP)
    if a.wired and r % a.cap:
        ap.error(f"--wired --cap {a.cap}: a wired member must not straddle the step-0 up_proj units and the rest "
                 f"(`loop`), so the cap divides the request count, {r}: give --cap {math.gcd(a.cap, r)} or omit it")
    a.law = a.law or (((WIRED_TIE_LAW if a.wired else TIE_LAW) if a.tie else LAW))
    if a.tie and a.law.partition(":")[0] != "work":
        ap.error(f"--tie proves the output class whole beside a streamed work draw, and a5 refuses it under any other "
                 f"law (output-class-needs-a-streamed-work-draw): give --law work:K (the default with --tie is "
                 f"{TIE_LAW}, and {WIRED_TIE_LAW} with --wired), not {a.law}")
    if a.check:
        print(json.dumps(mode_of(a, argv), sort_keys=True))
        return 0
    a.tools = json.loads(Path(a.tools).read_text())
    dep = M.deployment(a.deployment)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.mode = mode_of(a, argv)
    (out / "mode.json").write_text(json.dumps(a.mode, indent=1, sort_keys=True))
    run = os.environ.get("RESEARCH_RUN_ID") or "local"
    reports: list[dict] = []
    lock = threading.Lock()

    def finish(report: dict[str, Any]) -> None:
        with lock:
            reports.append(report)
            wdir = out / f"w{report['window']}"
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
            with open(out / "reports.jsonl", "a") as f:
                f.write(json.dumps(report, sort_keys=True) + "\n")
            m = M.monitoring(sorted(reports, key=lambda r: r["window"]), dep)
            (out / "monitoring-period.json").write_text(json.dumps(m, indent=1, sort_keys=True))
            print("ALLP " + json.dumps({"monitoring_period": M.lead(m), "window": summary_line(report)}, sort_keys=True),
                  flush=True)

    # one worker: every firewalled session (PoUW's audit, each leg) is a window's audit, and they run one after another,
    # while this thread serves the next window's loop
    with tempfile.TemporaryDirectory(dir=a.scratch) as scratch, ThreadPoolExecutor(1) as pool:
        a.weights = {"opening": Path(scratch) / WEIGHTS_OPENING, "pin": out / WEIGHTS_PIN} if a.registered else None
        if a.weights:
            serve(["--commit-weights", str(a.weights["opening"]), "--pin", str(a.weights["pin"]),
                   *(["--wired"] if a.wired else [])])
        if a.tie:
            shape = serve([*shape_of(a.start, run, a), "--circuit-only"])["out_shape"]
            a.relay = T.relay(Path(scratch), [int(x) for x in a.requests.split(",")], a.steps, a.windows, a.tools,
                              bool(a.tie_tamper), shape, out)
        t0 = time.monotonic()
        jobs = []
        for i, w in enumerate(range(a.start, a.start + a.windows)):
            wait = t0 + i * a.window_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            t = time.monotonic()
            try:
                loop: tuple[Path, dict] | Exception = loop_window(w, run, a, Path(scratch), out / f"w{w}")
                loop[1]["_loop_s"] = round(time.monotonic() - t, 1)
            except Exception as x:  # noqa: BLE001: a failed loop is the window's error, and the next window still runs
                loop = x
            jobs.append(pool.submit(lambda w=w, loop=loop: finish(window(w, run, a, Path(scratch), out, loop))))
        for j in jobs:
            j.result()
    bad = [r for r in reports if "error" in r or any(x["verdict"] == "error" for x in r["protocols"].values())]
    return int(bool(bad))


def summary_line(report: dict[str, Any]) -> dict[str, Any]:
    """The window's line: each protocol's verdict and quantity, and the composed verdict."""
    if "error" in report:
        return {"window": report["window"], "error": report["error"]}
    mode = report.get("mode") or {}
    return {"window": report["window"], "of_record": bool(mode.get("of_record")),
            **({"gap": mode["gap"]} if mode.get("gap") else {}), "tampered": report["loop"]["tampered"],
            **({"tie": report["loop"]["tie"]["strategy"]} if report["loop"].get("tie") else {}),
            "protocols": {n: {"verdict": r["verdict"],
                              **({"bound": r["quantity"].get("bound")} if isinstance(r.get("quantity"), dict)
                                 and "bound" in r["quantity"] else {}),
                              **({"epsilon": r["epsilon"], "delta": r["delta"]} if r.get("epsilon") else {}),
                              **({"reason": r["reason"][:300]} if r.get("reason") else {})}
                          for n, r in report["protocols"].items()},
            "composed": {k: report["composed"].get(k) for k in ("verdict", "reason")}}


if __name__ == "__main__":
    raise SystemExit(main())
