"""`verity bench all-protocols`'s runner: every window, one run of the torch loop audited by every protocol's leg, and the
result over each monitoring period of a datacenter.

    python -m verity_all_protocols.runner --out DIR --tools FILE [--windows N] [--start I] [--window-s S] [--jobs J]
        [--node NAME] [--tamper W,...] [--law L] [--delta D] [--cap N] [--requests 3,7] [--steps 2] [--pouw h1]
        [--deployment ID] [--onsite EXE] [--scratch DIR]

A window is one stretch of one machine's serving and its audits, and a monitoring period is the many windows on every
machine of a deployment a result is about (`ARCHITECTURE.md`'s Glossary). Per window W (``--start`` on): the verifier
registers the loop's circuit from its shape and run id (``serve --circuit-only``), the loop serves once
(`integrations/torch_loop`, run id ``RUN/wW``, so each window has a root of its own) and writes its window into scratch
with ``period.json`` beside it (the legs' records, `table.PERIOD`); each protocol of `table.TABLE` audits it through its leg
(`decide`), up to ``--jobs`` windows at once, each audit on ports of its own; and ``DIR/wW/report.json`` holds the window's
report, also appended to ``DIR/reports.jsonl``:

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
``m0-pythonpath``). ``--tamper W,...``: those windows' loop records outputs that are not its model's (`serve --tamper`)
and its prover skips its self-check, so a drawn unit's proof is of a false witness. ``--pouw h1`` (until ALLP-H1LOOP's
``serve --pouw h1`` lands): the loop also writes PoUW's window (its linears as the pinned H1 tile) to ``<window>/pouw``,
and ``verity_torch_loop.pouw_audit`` audits it into ``<window>/pouw/answer.json`` before the legs run. Only the window's
public files are kept in ``DIR`` (never a prover's file: its rows and salts).
"""
from __future__ import annotations

import argparse
import json
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

from . import compose as C, monitoring as M
from .table import ANSWERS, PERIOD, TABLE, Entry, LegFn, check_answer, leg_of

FORMAT = "verity/all-protocols-window/v1"
ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "integrations" / "torch_loop"
PUBLIC = ("registration.json", "window.json", "index.json", "circuit.json", "serve.json", PERIOD)


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
                       "delta": p.get("delta"), "tampered": bool(p.get("prover_env")), "pouw": p.get("pouw")},
              "protocols": rows, "seconds": seconds,
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


def loop_window(index: int, run: str, a: argparse.Namespace, scratch: Path, out: Path) -> tuple[Path, dict]:
    """The window's loop: the circuit registered before it runs, then its window in scratch with ``period.json`` (and with
    ``--pouw h1`` PoUW's window in its ``pouw/``); its public files in ``out/window``. -> the window's directory and what
    the loop printed."""
    shape = ["--law", a.law, "--cap", str(a.cap), "--requests", a.requests, "--steps", str(a.steps), "--run-id",
             f"{run}/w{index}", *(["--pouw", a.pouw] if a.pouw else [])]
    registered = serve([*shape, "--circuit-only"])
    window = scratch / f"w{index}"
    tampered = index in a.tamper
    got = serve([*shape, "--out", str(window), *(["--tamper"] if tampered else [])])
    (window / "serve.json").write_text(json.dumps(got, sort_keys=True))
    (window / PERIOD).write_text(json.dumps({
        "window": index, "period": index, "node": a.node, "circuit": registered["circuit"], "run_root": got["run"],
        "law": a.law, "delta": a.delta, "cap": a.cap, "tools": a.tools,
        "pouw": {"mode": a.pouw, "circuit": registered.get("pouw")} if a.pouw else None,
        "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {}}, sort_keys=True))
    kept = out / "window"
    kept.mkdir(parents=True, exist_ok=True)
    for f in [*PUBLIC, *(p.name for p in window.glob("m*-public.bin"))]:
        if (window / f).exists():
            shutil.copy(window / f, kept / f)
    return window, {"circuit_before": registered["circuit"], **got}


def pouw_audit(period_dir: Path, out: Path) -> dict[str, Any]:
    """PoUW's audit of the window's H1 window (``verity_torch_loop.pouw_audit``, ALLP-H1LOOP's), before the legs: it writes
    ``<window>/pouw/answer.json``, which PoUW's leg reads. -> its rc and ``POUW`` line."""
    p = json.loads((period_dir / PERIOD).read_text())
    t = p["tools"]
    out.mkdir(parents=True, exist_ok=True)
    argv = ["--window", str(period_dir / "pouw"), "--circuit", p["pouw"]["circuit"], "--run", p["run_root"],
            "--m0-pythonpath", t["m0-pythonpath"], "--bin", t["flock-circuit"], "--lean", t["flock-verify"],
            "--firewall", t["flock-firewall"], "--out", str(period_dir.parent / f"{period_dir.name}-pouw"),
            *(["--forged"] if p.get("prover_env") else [])]
    r = _python("verity_torch_loop.pouw_audit", argv, out / "pouw-audit")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="JSON: the built executables by name")
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="the first window's index (another machine's windows start apart)")
    ap.add_argument("--window-s", type=float, default=0, help="each window's loop starts this long after the one before (at "
                    "once when the one before took longer)")
    ap.add_argument("--jobs", type=int, default=1, help="windows audited at once, each on ports of its own")
    ap.add_argument("--node", default=os.environ.get("RESEARCH_MACHINE") or os.uname().nodename)
    ap.add_argument("--tamper", default="", help="windows whose loop tampers with its outputs, e.g. 1,3")
    ap.add_argument("--law", default="subset:64", help="the proof service's law of each window (k = 64: about 74 wrong "
                    "units over a whole monitoring period at δ = 1% and n = 1024)")
    ap.add_argument("--delta", default="1/100", help="each window's audit record's own δ")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--requests", default="3,7")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--pouw", choices=["h1"], help="the loop's PoUW window and its audit (ALLP-H1LOOP's, until it lands)")
    ap.add_argument("--deployment", default=M.REFERENCE, help="the census deployment the monitoring periods are over")
    ap.add_argument("--onsite", help="onsite's executable (`onsite check`), for when its components are ports")
    ap.add_argument("--scratch", help="where the loop's windows go (default a temporary directory, removed after)")
    a = ap.parse_args(argv)
    a.tamper = {int(x) for x in a.tamper.split(",") if x.strip()}
    a.tools = json.loads(Path(a.tools).read_text())
    dep = M.deployment(a.deployment)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = os.environ.get("RESEARCH_RUN_ID") or "local"
    reports: list[dict] = []
    lock = threading.Lock()
    if a.jobs > 1 and "RAYON_NUM_THREADS" in os.environ:
        os.environ["RAYON_NUM_THREADS"] = str(max(1, int(os.environ["RAYON_NUM_THREADS"]) // a.jobs))

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

    with tempfile.TemporaryDirectory(dir=a.scratch) as scratch, ThreadPoolExecutor(max(1, a.jobs)) as pool:
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
    return {"window": report["window"], "tampered": report["loop"]["tampered"],
            "protocols": {n: {"verdict": r["verdict"],
                              **({"bound": r["quantity"].get("bound")} if isinstance(r.get("quantity"), dict)
                                 and "bound" in r["quantity"] else {}),
                              **({"epsilon": r["epsilon"], "delta": r["delta"]} if r.get("epsilon") else {}),
                              **({"reason": r["reason"][:300]} if r.get("reason") else {})}
                          for n, r in report["protocols"].items()},
            "composed": {k: report["composed"].get(k) for k in ("verdict", "reason")}}


if __name__ == "__main__":
    raise SystemExit(main())
