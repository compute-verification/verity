"""Replay recorded served_zk windows through PoUW's check: the Python reference (`core_check`) and its Lean port
(`pouw-check serve`, and `pouw-check-difftest`'s `opening`), each case against the verdict the run recorded.

  python replay_core.py OUT [--runs DIR ...] [--v2 DIR|TAR]    OUT/replay.json and one line on stdout

Without ``--runs`` it selects every `job.sh` attempt in the evidence store and fetches its run record's ``out/``; without
``--v2`` (A4's tree, or a tarball of it) it fetches A4's v2 run (`V2_ART`).  For each run with a summary:

* **core cases** are the checks `core_check.case` decides, mapped to (params, registration, the service's answer):
  ``receive-honest`` and the registration controls ``receive-foreign``, ``receive-extra``, ``receive-raised`` and
  ``register-extra`` (the request's registration with the calls' shapes added, as `job.sh` asks the committer), each paired
  with the honest outcome's answer, and every ``outcome-*`` the run wrote (its ``outcome.json`` and ``audit.json``).  The
  params are the request's (`params.json`), the check item and the query the request registration's: ``receive-honest``
  REGISTERED says they were the verifier's own.  A REGISTERED or ACCEPT check must get accept; a REFUSED registration a
  reject by a registration rule (before ``outcome:``); a REJECT outcome a reject.  An accept's public outputs must be the
  recorded ``outcome.json``'s.
* **openings** (``open-u<t>``, ``open-filler``, ``open-foreign-*``), as `job.sh` maps them, go to `core_check.opening`
  and Lean's: every recorded window predates the 336-byte entry rows, so these compare Python with Lean only.
* the other checks (coins, the stream's state, the firewall, Lean's tile verifier, replays, shapes) are not core_check's.

A run of an older record version (no registration, or another scheme) and the v2 registrations run through both sides
for agreement only.  Python and Lean must give the same outcome on every input, reason included.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import verity_difftest as D

from verity.core.protocols.pouw import core_check as CC

REPO = Path(__file__).resolve().parents[3]
LEAN = REPO / "verity/core/protocols/pouw/lean/check"
V2_ART = "art:08afd6b23db47295569411eb024868fb39e6dbae6a11a3835657be263321621d"
REGISTRATION_RULES = ("window", "public-items", "level", "public-value", "population", "law", "query", "scheme")
CONTROLS = {"receive-foreign": "other/registration.json", "receive-extra": "registration-extra.json",
            "receive-raised": "registration-raised.json", "register-extra": None}


def research(*args: str) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"research {' '.join(args)}: exit {r.returncode}\n{r.stderr[-2000:]}")
    return r.stdout


def store_runs(into: Path) -> list[Path]:
    """Every served_zk `job.sh` attempt's run record ``out/``, fetched under `into`."""
    rows = research("data", "sql", "select a.id, o.artifact from attempts a join attempt_outputs o on o.attempt = a.id "
                    "where a.manifest_json like '%served_zk/job.sh%' and o.name = 'run_record' order by a.start_utc")
    runs = []
    for line in rows.splitlines()[1:]:
        run, art = line.split("\t")
        research("data", "fetch", art, "--to", str(into / run), "--path", "out/*")
        runs.append(into / run)
    return runs


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def hexed(j: Any) -> str:
    return D.canonical(j).encode().hex()


def ratio_of(x: Any, want: str) -> str:
    """A recorded profile's δ as `Core.Integrity`'s ratio string: the window's own when the float is it."""
    if isinstance(x, str):
        return x
    return want if float(Fraction(want)) == x else str(Fraction(x).limit_denominator(10**9))


def answer_of(outcome_dir: Path, delta: str) -> dict:
    """The service's answer (`Core.Answer`) one ``outcome-*`` directory records."""
    o, a = load(outcome_dir / "outcome.json"), load(outcome_dir / "audit.json")
    base = {"protocol": CC.ONE_STAGE, "commitment": a["registration"]}
    if o["accepted"]:
        prof = {**o["profile"], "delta": ratio_of(o["profile"]["delta"], delta)}
        return {**base, "verdict": "accept", "reason": None, "public": {"draw": a["audit"]["draw"]}, "profile": prof}
    codes = (o.get("verdict") or {}).get("codes") or ["reject"]
    return {**base, "verdict": "reject", "reason": ", ".join(codes), "public": None, "profile": None}


def params_of(out: Path, reg: dict) -> dict:
    p = load(out / "request/params.json")
    return {"check": reg["window"]["public"]["check"], "context": p["context"], "delta": p["delta"], "eps": p["eps"],
            "grid_bits": p["grid_bits"], "query": reg["query"], "range": p["range"]}


def current(reg: Any) -> bool:
    return isinstance(reg, dict) and reg.get("scheme") == CC.SCHEME and any(
        r.get("port") == "entries" for r in reg.get("roots", []))


def core_cases(out: Path, checks: dict) -> list[tuple[str, dict, str, Any]]:
    """(check, case, recorded verdict, the recorded public outputs on accept) of every check core_check decides."""
    reg = load(out / "request/registration.json")
    params = params_of(out, reg)
    honest = answer_of(out / "outcome-honest", params["delta"]) if (out / "outcome-honest").is_dir() else None
    cases = []
    for name, rec in sorted(checks.items()):
        if name == "receive-honest" or name in CONTROLS:
            if honest is None:
                continue
            path = CONTROLS.get(name)
            r = load(out / path) if path else copy.deepcopy(reg)
            if name == "register-extra":
                r["window"]["public"]["shapes"] = [[1, 4096, 4096]]
            pub = load(out / "outcome-honest/outcome.json")["public"] if name == "receive-honest" else None
            cases.append((name, {"params": params, "commitment": hexed(r), "evidence": hexed(honest)}, rec["verdict"], pub))
        elif name.startswith("outcome-") and (out / name / "outcome.json").is_file():
            ans = answer_of(out / name, params["delta"])
            pub = load(out / name / "outcome.json")["public"] if ans["verdict"] == "accept" else None
            cases.append((name, {"params": params, "commitment": hexed(reg), "evidence": hexed(ans)}, rec["verdict"], pub))
    return cases


def opening_inputs(out: Path, checks: dict, summary: dict) -> list[tuple[str, dict]]:
    """The openings `job.sh`'s ``opencheck`` judged, as `pouw-check-difftest`'s ``opening`` inputs."""
    regs = {s: load(out / s / "registration.json") for s in ("request", "other")}
    drawn = summary.get("units") or []
    u0 = drawn[0] if drawn else None
    ou = u0 if u0 is not None and u0 < regs["other"]["population"] else 0
    plan = {f"open-u{t}": ("request", t, f"u{t}", f"u{t}", True) for t in drawn}
    plan["open-filler"] = ("request", summary.get("filler_unit"), "filler", "filler", False)
    plan["open-foreign-proof"] = ("request", u0, "other-tile", f"u{u0}", True)
    plan["open-foreign-commit"] = ("other", ou, f"u{u0}", "other-tile", False)
    got = []
    for name, (side, unit, pub, op, is_drawn) in plan.items():
        if name not in checks or unit is None or not (out / pub / "pub.bin").is_file():
            continue
        p = load(out / side / "params.json")
        reg = regs[side]
        got.append((name, {"params": {"check": {**reg["window"]["public"]["check"], "entries": p["entries"]},
                                      "context": p["context"], "delta": p["delta"], "eps": p["eps"],
                                      "grid_bits": p["grid_bits"], "query": reg["query"], "range": p["range"]},
                           "commitment": hexed(reg), "units": drawn if is_drawn else None, "unit": unit,
                           "pub": (out / pub / "pub.bin").read_bytes().hex(), "opening": load(out / op / "opening.json")}))
    return got


def opening_reference(x: dict) -> Any:
    p = CC.params(CC.canonical(D.canonical(x["params"]).encode()))
    c = CC.commitment(bytes.fromhex(x["commitment"]))
    return CC.opening(p, c, x["units"], x["unit"], bytes.fromhex(x["pub"]), x["opening"])


class Sides:
    """Python and Lean on one input each: `core` (a core_check case) and `opening`."""

    def __init__(self) -> None:
        self.core = D.Server(D.ProtocolPort("pouw-check", CC.case, inputs=None, package=LEAN).argv())
        self.opening = D.Server(D.Port("pouw-check-difftest", "opening", opening_reference, inputs=None,
                                       package=LEAN).argv())

    def case(self, case: dict) -> tuple[dict, bool]:
        """Python's outcome, and whether Lean's is the same, reason included."""
        py = CC.case(copy.deepcopy(case))
        return py, D.canonical(self.core.exchange(case)) == D.canonical(py)

    def open(self, x: dict) -> tuple[Any, bool]:
        try:
            py: tuple[str, Any] = ("output", opening_reference(copy.deepcopy(x)))
        except CC.Refused as e:
            py = ("reject", str(e))
        lean = self.opening.call("opening", x)
        same = lean == ("output", D.canonical(py[1])) if py[0] == "output" else lean[0] == "reject"
        return py, same


def registration_stage(reason: str | None) -> bool:
    return reason is not None and reason.split(":")[0] in REGISTRATION_RULES


def judged(recorded: str, a: dict, public: Any) -> bool:
    """Whether core_check's outcome `a` is the recorded verdict."""
    if recorded in ("REGISTERED", "ACCEPT"):
        return a["verdict"] == "accept" and (public is None or D.canonical(a["public"]) == D.canonical(public))
    if recorded == "REFUSED":
        return a["verdict"] == "reject" and registration_stage(a["reason"])
    return recorded == "REJECT" and a["verdict"] == "reject"


def replay_run(sides: Sides, out: Path) -> dict:
    summary = load(out / "summary.json")
    checks = summary.get("checks") or {}
    reg_p = out / "request/registration.json"
    reg = load(reg_p) if reg_p.is_file() else None
    row: dict[str, Any] = {"checks": len(checks), "cases": [], "openings": [], "apart": []}
    if not current(reg):
        row["version"] = "no registration" if reg is None else (reg.get("scheme") or "no scheme")
        try:
            older = core_cases(out, checks) if reg is not None else []
        except (KeyError, TypeError, OSError) as e:
            row["unmapped"] = f"{type(e).__name__}: {e}"
            older = []
        for name, case, recorded, _ in older:
            a, same = sides.case(case)
            row["cases"].append({"check": name, "recorded": recorded, "verdict": a["verdict"], "reason": a["reason"],
                                 "lean_same": same})
        return row
    row["version"] = CC.SCHEME
    mapped = set()
    for name, case, recorded, public in core_cases(out, checks):
        a, same = sides.case(case)
        mapped.add(name)
        row["cases"].append({"check": name, "recorded": recorded, "verdict": a["verdict"], "reason": a["reason"],
                             "lean_same": same, "same_verdict": judged(recorded, a, public)})
    for name, x in opening_inputs(out, checks, summary):
        py, same = sides.open(x)
        mapped.add(name)
        row["openings"].append({"check": name, "recorded": checks[name]["verdict"], "python": py, "lean_same": same})
    row["apart"] = sorted(set(checks) - mapped)
    return row


def replay_v2(sides: Sides, root: Path) -> list[dict]:
    """A4's v2 registrations, each through both sides: none is a PoUW window (no served window), and its
    `registered` values wait on R7."""
    rows = []
    for p in sorted(root.rglob("registration.json")):
        reg = load(p)
        case = {"params": {"check": {"entries": 2, "scheme": CC.SCHEME, "template": "v2", "tile_work": 1},
                           "context": "v2", "delta": "1/10", "eps": "1/2", "grid_bits": 3, "query": reg["query"],
                           "range": [1, 1 << 40]},
                "commitment": hexed(reg), "evidence": hexed({"protocol": CC.ONE_STAGE, "verdict": "reject",
                                                            "reason": "v2", "commitment": None, "public": None,
                                                            "profile": None})}
        a, same = sides.case(case)
        rows.append({"registration": str(p.relative_to(root)), "format": reg.get("format"), "verdict": a["verdict"],
                     "reason": a["reason"], "lean_same": same})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--runs", type=Path, nargs="*")
    ap.add_argument("--v2", type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs = a.runs if a.runs is not None else store_runs(Path(tmp) / "runs")
        v2 = a.v2
        if v2 is None:
            research("data", "fetch", V2_ART, "--to", str(Path(tmp) / "v2"))
            v2 = Path(tmp) / "v2"
        elif v2.is_file():
            with tarfile.open(v2) as tf:
                tf.extractall(Path(tmp) / "v2", filter="data")
            v2 = Path(tmp) / "v2"
        sides = Sides()
        report: dict[str, Any] = {"runs": {}, "v2": {"art": V2_ART, "registrations": replay_v2(sides, v2)}}
        for r in runs:
            out = r / "out" if (r / "out").is_dir() else r
            if not (out / "summary.json").is_file():
                report["runs"][r.name] = {"version": "no summary (the job failed before its checks)"}
                continue
            report["runs"][r.name] = replay_run(sides, out)
        sides.core.close()
        sides.opening.close()
    cur = {k: v for k, v in report["runs"].items() if v.get("version") == CC.SCHEME}
    cases = [c for v in cur.values() for c in v["cases"]]
    opens = [o for v in cur.values() for o in v["openings"]]
    older = [c for v in report["runs"].values() if v.get("version") != CC.SCHEME for c in v.get("cases", [])]
    every = cases + opens + older + report["v2"]["registrations"]
    t = {"runs": len(report["runs"]), "current": len(cur), "cases": len(cases),
         "same_verdict": sum(c["same_verdict"] for c in cases),
         "controls": sum(c["recorded"] in ("REFUSED", "REJECT") for c in cases),
         "controls_same": sum(c["same_verdict"] for c in cases if c["recorded"] in ("REFUSED", "REJECT")),
         "controls_by_rule": sum(c["recorded"] in ("REFUSED", "REJECT") and registration_stage(c["reason"])
                                 for c in cases),
         "lean_same": sum(c["lean_same"] for c in every), "compared": len(every), "openings": len(opens),
         "older_runs": sorted(k for k in report["runs"] if k not in cur), "older_cases": len(older),
         "v2": len(report["v2"]["registrations"]),
         "apart": dict(sorted(Counter(n.split("-")[0] for v in cur.values() for n in v["apart"]).items()))}
    report["totals"] = t
    t["line"] = (f"{t['current']} runs, {t['same_verdict']}/{t['cases']} same verdict, {t['controls']} negative controls "
                 f"({t['controls_same']} same, {t['controls_by_rule']} refused by a registration rule, the rest the "
                 f"service's reject); Lean = Python on {t['lean_same']}/{t['compared']} inputs "
                 f"({t['openings']} pre-layout openings, {t['older_cases']} cases of {len(t['older_runs'])} older runs, "
                 f"{t['v2']} v2 registrations apart)")
    (a.out / "replay.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(t["line"])
    return 0 if t["same_verdict"] == t["cases"] and t["lean_same"] == t["compared"] else 1


if __name__ == "__main__":
    sys.exit(main())
