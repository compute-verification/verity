"""The block record: what each subprotocol writes for one server-time block and what the composed verdict reads,
one document ``verity/block-record/v1``. It extends the all-protocols runner's per-period report
(``verity/all-protocols-period/v1``, and its per-window ``verity/all-protocols-window/v1``) with the block's server and
steps, every leg in one shape, the composed verdict with the JointHorizonLive error sum, and the monitoring period.

**The record.**

* ``server``: the node the block ran on; ``server_time``: the block's steps, ``[{server, steps: [lo, hi)}]`` (the circuits'
  blocks schema).
* ``legs``: ``integrity``, ``pouw``, ``pous``, ``bandwidth``, ``certifier``, ``wipe``, each ``null`` when the block ran no
  such audit, else ``{protocol, owner, server, source: real|fixture, answer, quantity, error, error_from, at, runs}``:
  ``answer`` the core verifier's answer (verdict, reason, commitment, public); ``quantity`` the leg's certificate
  ``{symbol, value, unit, ...}`` (integrity bound, W_cert, S_cert, Q_cert, K, the wipe's foreign_bits at its reads ``k``
  and ratio ``r``, ``r`` null in a wipe from before ``verity/wipe/v0`` published it); ``error`` the probability over the leg's
  own coins that it accepts while its claim fails, as the run established it, read from ``error_from``; ``at`` the
  certificate at δ = 1/100 and 1/10^6 (core's `DELTAS`): the run's own wherever ``error`` ≤ δ (a claim holding except with
  ``error`` holds except with δ), integrity's recomputed from its draw at δ (`integrity_bound`), else not established.
  A leg whose ``server`` is not the block's is carried but not composed.
* ``composed``: ``verdict`` (`compose`: reject when a composed leg rejects or errs, accept when every leg ran on the block's
  server and accepted, else pending with what is ``missing``), the run's own composed answer (``onsite``) when it wrote
  one, and ``joint_horizon``: `JOINT_HORIZON`'s error over this block, the composed legs' errors summed, and at each δ of
  `DELTAS` that δ times the legs established at it. The statement names five audits; the wipe is ``outside_statement``.
* ``monitoring_period``: the census deployment (a parameter: ``{id, machines, window_s, ...}``) and, per period of
  core's `PERIODS`, its blocks over all compute (machines × blocks a machine) and JointHorizonLive's union bound over them at this
  block's errors (``error_sum`` × blocks), and the per-block error each δ of `DELTAS` needs over the period (δ / blocks).

`check` recomputes every derived field from the legs. `from_period` builds a record from a runner report and extra legs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.protocols.pous.wipe_monitoring import DELTAS, PERIODS
from verity.core.protocols.profile import Stratified

FORMAT = "verity/block-record/v1"
EXTENDS = ("verity/all-protocols-period/v1", "verity/all-protocols-window/v1")
LEGS = ("integrity", "pouw", "pous", "bandwidth", "certifier", "wipe")
#: the runner's protocol keys, by leg
RUNNER_KEYS = {"integrity": "integrity", "pouw": "work", "pous": "space", "bandwidth": "bandwidth", "certifier": "communication"}
JOINT_HORIZON = {"name": "Accounting.Guarantees.JointHorizonLive", "proved": True,
                 "file": "Security/Definitions/Accounting/Guarantees/JointLive.lean",
                 "audits": ["integrity", "pouw", "pous", "bandwidth", "certifier"]}


def frac(x: Any) -> Fraction:
    return Fraction(str(x))


def integrity_bound(n: int, kd: int, delta: Fraction) -> int:
    """The most wrong units of ``n`` an accepted uniform sample of ``kd`` leaves except with ``δ``: the integrity profile's
    count bound (`Stratified.count_bound`), the largest ``K`` whose sampling term ``C(n − K, kd)/C(n, kd)`` is at least
    ``δ``."""
    return Stratified.uniform(n, kd).count_bound(delta)


def _draw(answer: Mapping[str, Any]) -> tuple[int, int]:
    level = answer["profile"]["levels"][-1]
    return int(level["size"]), int(answer["profile"]["drawn"][level["name"]])


def _quantity(leg: str, row: Mapping[str, Any]) -> tuple[dict[str, Any], Fraction, str]:
    """A leg's certificate, its error and where the error is read from, out of its runner row (or the wipe's leg)."""
    q, a = row.get("quantity") or {}, row["answer"]
    pub = a.get("public") or {}
    if leg == "integrity":
        n, kd = _draw(a)
        return ({"symbol": "integrity bound", "value": int(q["bound"]), "unit": "wrong units", "of": n, "drawn": kd},
                frac(row["delta"]), "delta")
    if leg == "pouw":
        w = q["work"]
        return ({"symbol": "W_cert", "value": int(w["credited"]), "unit": "work", "tiles": int(w["credited_tiles"])},
                frac(row["delta"]), "delta")
    if leg == "pous":
        return ({"symbol": "S_cert", "value": int(q["s_bits"]), "unit": "bits", "certified": bool(q["certified"])},
                frac(q["eps"]), "quantity.eps")
    if leg == "bandwidth":
        return ({"symbol": "Q_cert", "value": int(q["q_bytes"]), "unit": "bytes"}, frac(q["eps"]), "quantity.eps")
    if leg == "certifier":
        return ({"symbol": "K", "value": int(q["K"]), "unit": "bits", "windows": int(q["windows"])}, frac(q["eps"]),
                "quantity.eps")
    if leg == "wipe":
        k = pub.get("k", (a.get("profile") or {}).get("k"))
        return ({"symbol": "foreign_bits", "value": int(pub["foreign_bits"]), "unit": "bits", "k": k, "r": pub.get("r"),
                 "M": int(pub["memory_bits"]), "certified": bool(pub["certified"])}, frac(pub["eps"]), "answer.public.eps")
    raise ValueError(f"{leg}: not a leg of {LEGS}")


def at_deltas(leg: str, quantity: Mapping[str, Any], error: Fraction) -> dict[str, Any]:
    """The leg's certificate at each δ of `DELTAS` (module docstring)."""
    out: dict[str, Any] = {}
    for d in DELTAS:
        if leg == "integrity":
            b = integrity_bound(quantity["of"], quantity["drawn"], d)
            out[str(d)] = {"holds": True, "value": b, "epsilon": str(Fraction(b, quantity["of"])), "by": "draw"}
        elif error <= d:
            out[str(d)] = {"holds": True, "value": quantity["value"], "epsilon": None, "by": "run"}
        else:
            out[str(d)] = {"holds": False, "why": f"the run established it except with {error}, more than {d}"}
    return out


def leg(name: str, row: Mapping[str, Any], *, server: str, runs: Sequence[str], source: str = "real") -> dict[str, Any] | None:
    """One leg of the record from its runner row (or, for the wipe, its leg document); None for a row the runner marks
    ``missing``."""
    if row.get("verdict") == "missing":
        return None
    if source not in ("real", "fixture"):
        raise ValueError(f"source {source!r}: real or fixture")
    a = row["answer"]
    quantity, error, error_from = _quantity(name, row)
    return {"protocol": a["protocol"], "owner": row.get("owner"), "server": server, "source": source,
            "answer": {k: a.get(k) for k in ("verdict", "reason", "commitment", "public")},
            "quantity": quantity, "error": str(error), "error_from": error_from, "at": at_deltas(name, quantity, error),
            "runs": list(runs)}


def compose(server: str, legs: Mapping[str, Any]) -> dict[str, Any]:
    """The composed verdict and the JointHorizonLive error over the block (module docstring)."""
    here = {k: v for k, v in legs.items() if v is not None and v["server"] == server}
    elsewhere = sorted(k for k, v in legs.items() if v is not None and v["server"] != server)
    missing = sorted(k for k in LEGS if k not in here)
    verdicts = {k: v["answer"]["verdict"] for k, v in here.items()}
    if any(v in ("reject", "error") for v in verdicts.values()):
        verdict = "reject"
    elif not missing and all(v == "accept" for v in verdicts.values()):
        verdict = "accept"
    else:
        verdict = "pending"
    accepted = sorted(k for k, v in verdicts.items() if v == "accept")
    terms = {k: here[k]["error"] for k in accepted}
    jh = {"statement": JOINT_HORIZON, "terms": terms, "error_sum": str(sum((frac(e) for e in terms.values()), Fraction(0))),
          "outside_statement": sorted(k for k in terms if k not in JOINT_HORIZON["audits"]),
          "at": {str(d): {"error_sum": str(d * sum(1 for k in accepted if here[k]["at"][str(d)]["holds"])),
                          "not_established": sorted(k for k in accepted if not here[k]["at"][str(d)]["holds"])}
                 for d in DELTAS}}
    return {"verdict": verdict, "legs": verdicts, "missing": missing, "elsewhere": elsewhere, "joint_horizon": jh}


def monitoring_period(deployment: Mapping[str, Any], error_sum: Fraction) -> dict[str, Any]:
    """The monitoring period over all compute (module docstring), a block being one of a machine's ``window_s`` windows."""
    periods = {}
    for name, seconds in PERIODS:
        per_machine = seconds // int(deployment["window_s"])
        blocks = int(deployment["machines"]) * per_machine
        union = error_sum * blocks
        periods[name] = {"seconds": seconds, "blocks_per_machine": per_machine, "blocks": blocks,
                         "joint_horizon": {"error_sum": str(union), "vacuous": union >= 1},
                         "per_block_error_for": {str(d): str(d / blocks) for d in DELTAS}}
    return {"deployment": dict(deployment), "deltas": [str(d) for d in DELTAS], "periods": periods}


def from_period(report: Mapping[str, Any], *, steps: tuple[int, int], deployment: Mapping[str, Any],
                extra: Mapping[str, Mapping[str, Any]] | None = None, source: str = "real") -> dict[str, Any]:
    """The record of one runner report (``verity/all-protocols-period/v1`` or ``-window/v1``) whose loop served ``steps``
    ``[lo, hi)`` on its node. ``extra``: legs the runner doesn't run, by leg name, each ``{row, server, runs, source}`` with
    ``row`` the leg's document (``{answer, ...}``)."""
    if report.get("format") not in EXTENDS:
        raise ValueError(f"{report.get('format')}: not one of {EXTENDS}")
    lo, hi = steps
    if not 0 <= lo < hi:
        raise ValueError(f"steps [{lo}, {hi}): not a nonempty range")
    server = report["node"]
    legs: dict[str, Any] = {}
    for name, key in RUNNER_KEYS.items():
        row = report["protocols"].get(key)
        legs[name] = None if row is None else leg(name, row, server=server, runs=row.get("runs") or [report["run"]],
                                                  source=source)
    legs["wipe"] = None
    for name, x in (extra or {}).items():
        legs[name] = leg(name, x["row"], server=x["server"], runs=x["runs"], source=x.get("source", "real"))
    composed = compose(server, legs)
    onsite = report.get("composed") or {}
    composed["onsite"] = {k: onsite.get(k) for k in ("protocol", "verdict", "reason")} if onsite else None
    which = {"period": report["period"]} if "period" in report else {"window": report["window"]}
    return {"format": FORMAT, "extends": report["format"], "run": report["run"], **which, "server": server,
            "server_time": [{"server": server, "steps": [lo, hi]}], "legs": legs, "composed": composed,
            "monitoring_period": monitoring_period(deployment, frac(composed["joint_horizon"]["error_sum"]))}


def check(record: Mapping[str, Any]) -> dict[str, str]:
    """Every derived field of a record recomputed from its legs and its deployment: ``{key: why}``, empty when it holds."""
    bad: dict[str, str] = {}
    if record.get("format") != FORMAT:
        return {"format": f"not {FORMAT}"}
    server = record["server"]
    if record["server_time"] != [{"server": server, "steps": record["server_time"][0]["steps"]}] or not (
            0 <= record["server_time"][0]["steps"][0] < record["server_time"][0]["steps"][1]):
        bad["server_time"] = "not one nonempty block [lo, hi) of the record's server"
    if sorted(record["legs"]) != sorted(LEGS):
        bad["legs"] = f"not the legs {LEGS}"
        return bad
    for name, lg in record["legs"].items():
        if lg is None:
            continue
        if lg["source"] not in ("real", "fixture"):
            bad[f"leg:{name}"] = "its source is neither real nor fixture"
        if lg["at"] != at_deltas(name, lg["quantity"], frac(lg["error"])):
            bad[f"leg:{name}"] = "its certificates at the deltas are not its quantity's at its error"
    want = compose(server, record["legs"])
    got = {k: v for k, v in record["composed"].items() if k != "onsite"}
    for k in want:
        if got.get(k) != want[k]:
            bad[f"composed:{k}"] = "not what its legs compose to"
    if record["monitoring_period"] != monitoring_period(record["monitoring_period"]["deployment"],
                                                        frac(want["joint_horizon"]["error_sum"])):
        bad["monitoring_period"] = "not the deployment's periods at the block's error sum"
    return bad


# --- the records from the runs' outputs ----------------------------------------------------------------------------------

#: `benchmarks/census/deployments.json`'s reference deployment as the runs' monitoring reports recorded it
REFERENCE = {"id": "dc-12500-machines-600s", "machines": 12500, "gpus_per_machine": 8, "window_s": 600, "role": "reference",
             "label": "reference datacenter: 12,500 eight-GPU machines (100,000 GPUs), every machine audited in 10-minute windows"}


def from_runs(r2: Path, r1: Path, wipe: Path) -> dict[str, dict[str, Any]]:
    """The fixtures from the runs' outputs: period 0 of one all-protocols run (``r2``: its ``reports.jsonl``), window 10 of
    another (``r1``) and a wipe run (``wipe``: its ``results.json``, with ``leg/wipe.json`` beside it). Each loop served 2
    requests for 2 steps: ``[0, 2)``. The wipe's server is its session report's host, the hostname the runner names its
    node by (its ``placement`` names the node's alias)."""
    p0 = next(json.loads(x) for x in r2.read_text().splitlines() if json.loads(x)["period"] == 0)
    w10 = next(json.loads(x) for x in r1.read_text().splitlines() if json.loads(x)["window"] == 10)
    row = json.loads((wipe.parent / "leg" / "wipe.json").read_text())
    w = {"row": row, "server": row["report"]["host"], "runs": ["r20261025-224222-91e2"]}
    return {"block-record-r2-period0": from_period(p0, steps=(0, 2), deployment=REFERENCE, extra={"wipe": w}),
            "block-record-r1-window10": from_period(w10, steps=(0, 2), deployment=REFERENCE, extra={"wipe": w})}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The block record: check one, or build the fixture records from the runs' outputs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("record")
    t = sub.add_parser("from-runs")
    t.add_argument("--r2", required=True, type=Path)
    t.add_argument("--r1", required=True, type=Path)
    t.add_argument("--wipe", required=True, type=Path)
    t.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        bad = check(json.loads(Path(a.record).read_text()))
        print(json.dumps({"ok": not bad, "bad": bad}, sort_keys=True))
        return 0 if not bad else 1
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rec in from_runs(a.r2, a.r1, a.wipe).items():
        (a.out / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n")
        print(a.out / f"{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
