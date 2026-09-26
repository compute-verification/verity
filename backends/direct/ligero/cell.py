"""A B-interactive Table 2 cell's result (``bench.cell``'s driver ``drivers/b_interactive.py``, pod script ``cell.sh``).

    python -m backends.direct.ligero.cell finish --sweep DIR --probe probe.json [--loopback LOOP/result.json] --verifier HOST:PORT \\
        [--operator LANE] --out $RESEARCH_RUN_DIR/result.json
    python -m backends.direct.ligero.cell register --prover-run DIR [--verifier-run DIR] --lane LANE --out META

``finish`` runs on the prover pod after the sweep: the plateau point's ``bench-vu --verifier`` result (every recorded rep a live
session with a verifier on another pod, which drew the coins and ran the Rust verifier), with what the same run measured around
it -- the open-connection RTT and one-flow bandwidth to the verifier (``live probe --pings``, before the sweep), the same-run
loopback probe (one ``bench-vu`` at the plateau size against a verifier on 127.0.0.1: the prover's own handling per round trip)
-- and the interaction record the views read (``bench.views.interaction``).  ``register`` stamps the producing lane and the
verifier run's session verdicts on that document for ``research data put``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PER_PROOF_DEPTH = 3          # PROTOCOL.md 8c: the verifier's commitment, then its two openings


def _meas(doc: dict[str, Any]) -> dict[str, Any]:
    return {m["name"]: m.get("value") for m in doc.get("measurements") or [] if isinstance(m, dict) and "name" in m}


def _set(doc: dict[str, Any], name: str, value: Any, unit: str = "") -> None:
    ms = doc.setdefault("measurements", [])
    for m in ms:
        if m.get("name") == name:
            m["value"] = value
            return
    ms.append({"name": name, "value": value, "unit": unit})


def plateau(sweep_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    sw = json.loads((sweep_dir / "sweep.json").read_text())
    p = next(q for q in sw["points"] if q["point"] == sw["plateau_point"])
    return sw, p, sweep_dir / p["dir"]


def loopback_round_seconds(doc: dict[str, Any]) -> tuple[float, int]:
    """(seconds per round trip, round trips per session) of a loopback ``bench-vu --verifier`` result: the median rep's network
    wait over its session's openings (``live.BenchLive.fields``)."""
    ev = ((doc.get("validation") or {}).get("evidence") or {}).get("live_verifier") or {}
    sessions = max(1, int(ev.get("sessions") or 1))
    per_session = int(ev.get("round_trips") or 0) // sessions
    wait = _meas(doc).get("net.wait_seconds")
    if not per_session or not isinstance(wait, (int, float)):
        raise ValueError("loopback result without a live session's round trips and network wait")
    return float(wait) / per_session, per_session


def finish(sweep_dir: Path, probe: dict[str, Any], loopback: dict[str, Any] | None, verifier: str, run_dir: Path,
           operator: str | None = None) -> dict[str, Any]:
    sw, p, pdir = plateau(sweep_dir)
    doc = json.loads((pdir / "result.json").read_text())
    m = _meas(doc)
    fp = doc["workload_fingerprint"]
    if "net.wait_seconds" not in m or not ((doc.get("validation") or {}).get("evidence") or {}).get("live_verifier"):
        raise ValueError(f"plateau point {p['dir']} did not run against a live verifier (bench-vu --verifier)")
    n_proofs = int(fp["N_subbatches"])
    pipeline = int((fp.get("software") or {}).get("backend", {}).get("pipeline") or 1) or 1
    rounds = PER_PROOF_DEPTH * math.ceil(n_proofs / max(1, pipeline))
    if isinstance(m.get("net.rtt_ms"), (int, float)):
        _set(doc, "live.opening_rtt_ms_median", m["net.rtt_ms"], "ms")
    _set(doc, "net.rtt_ms", float(probe["ping_ms_median"]), "ms")
    if isinstance(probe.get("mbps_median"), (int, float)):
        _set(doc, "net.bandwidth_bps", float(probe["mbps_median"]) * 1e6, "b/s")
    _set(doc, "rounds.per_proof_depth", PER_PROOF_DEPTH, "count")
    _set(doc, "rounds.sequential_depth", rounds, "count")
    inter: dict[str, Any] = {
        "rtt_method": f"net.rtt_ms = {probe['ping_method']}, prover pod -> verifier pod, measured in this run before the sweep; the "
                      f"session's own opening round trips (queued behind the proof stream) are live.opening_rtt_ms_median",
        "bandwidth_method": "net.bandwidth_bps = one-flow throughput to the verifier pod (live probe: NOP frames of "
                            f"{probe['flows'][0]['bytes'] >> 20 if probe.get('flows') else '?'} MB acknowledged by the verifier, median)",
        "rounds_rule": f"rounds.sequential_depth = {PER_PROOF_DEPTH} (8c: commitment + two openings per proof) x ceil({n_proofs} "
                       f"sub-batch proofs / {pipeline} in flight)",
        "verifier_operator": operator,
        "probe": {k: probe.get(k) for k in ("ping_ms_median", "ping_ms_min", "connect_ms", "rtt_ms", "mbps_median", "tcp_cc",
                                             "prover_dc", "verifier_dc")},
    }
    if loopback is not None:
        per_round, per_session = loopback_round_seconds(loopback)
        _set(doc, "live.loopback_round_seconds", per_round, "s")
        lm = _meas(loopback)
        inter["loopback_method"] = (f"same-run loopback probe: one bench-vu of the plateau's {int(lm.get('vu.count') or 0)} VUs against a "
                                    f"live verifier on 127.0.0.1 in this prover pod, its median rep's network wait "
                                    f"{float(lm['net.wait_seconds']):.4f} s over the session's {per_session} openings")
    doc["interaction"] = inter
    fp["verifier"] = {"placement": verifier, "process": "live verifier (backends.direct.ligero.live serve, Rust ligero-verify) on a "
                                                       "separate pod; it draws the coins", "operator": operator}
    # the plateau's dump (bench-vu wrote its artifacts relative to $RESEARCH_RUN_DIR): the run-files of this result
    doc["artifacts"] = [a for a in doc.get("artifacts") or [] if (run_dir / a).exists()]
    doc["cell_sweep"] = {"id": sw["id"], "stopped": sw["stopped"], "plateau_point": sw["plateau_point"],
                         "points": [[q["total_vus"], q["vu_per_second"]] for q in sw["points"]], "dir": pdir.name}
    return doc


def _tree_sha(run: Path) -> str | None:
    try:
        return (json.loads((run / "job.json").read_text()).get("computation") or {}).get("tree_sha")
    except (OSError, ValueError):
        return None


def register(prover_run: Path, verifier_run: Path | None, lane: str) -> dict[str, Any]:
    doc = json.loads((prover_run / "result.json").read_text())
    doc["lane"] = doc["registered_by"] = lane
    # a tree shipped by `research run --source` has no git: its commit is the runner's record of what it shipped
    fp, sha = doc.get("workload_fingerprint") or {}, _tree_sha(prover_run)
    if sha:
        if not fp.get("commit"):
            fp["commit"] = sha
        be = (fp.get("software") or {}).get("backend")
        if isinstance(be, dict) and not be.get("commit"):
            be["commit"] = sha
    if verifier_run is not None:
        sessions = []
        for f in sorted(verifier_run.rglob("index.jsonl")):
            for line in f.read_text().splitlines():
                try:
                    sessions.append(json.loads(line))
                except ValueError:
                    continue
        mine = [x for x in sessions if x.get("run_id") == prover_run.name]
        doc.setdefault("validation", {}).setdefault("evidence", {})["verifier_run"] = {
            "run": verifier_run.name, "tree_sha": _tree_sha(verifier_run), "sessions": len(mine),
            "accepted": sum(1 for x in mine if x.get("accepted")),
            "note": "the verifier pod's own session index (backends.direct.ligero.live serve --out), recorded by its research run"}
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="backends.direct.ligero.cell")
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("finish")
    f.add_argument("--sweep", required=True, type=Path)
    f.add_argument("--probe", required=True, type=Path)
    f.add_argument("--loopback", default=None, type=Path)
    f.add_argument("--verifier", required=True)
    f.add_argument("--operator", default=None)
    f.add_argument("--run-dir", default=None, type=Path)
    f.add_argument("--out", required=True, type=Path)
    r = sp.add_parser("register")
    r.add_argument("--prover-run", required=True, type=Path)
    r.add_argument("--verifier-run", default=None, type=Path)
    r.add_argument("--lane", required=True)
    r.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "finish":
        lb = json.loads(a.loopback.read_text()) if a.loopback and a.loopback.is_file() else None
        doc = finish(a.sweep, json.loads(a.probe.read_text()), lb, a.verifier, a.run_dir or a.out.parent, a.operator)
    else:
        doc = register(a.prover_run, a.verifier_run, a.lane)
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    ix = __import__("verity_numerical.bench.views", fromlist=["interaction"])
    rec = ix.interaction(doc)
    print(json.dumps({"out": str(a.out), "interaction": rec, "problem": ix.interaction_problem(rec)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
