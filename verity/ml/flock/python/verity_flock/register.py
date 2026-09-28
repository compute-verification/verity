"""Re-register a Flock cell's ``bench-result/v1`` from its recorded runs, with what the red team and the renderer require.

    python -m verity_flock.register --prover-run DIR --verifier-run DIR --lane LANE --out meta.json \\
        [--loopback-seconds S --loopback-method TEXT] [--rtt-ms MS --rtt-method TEXT] [--supersedes art:...]

then ``research data put --kind bench-result/v1 --meta @meta.json --ref run_files=<the run's run files> --preserve``.
Adds: the prover's and verifier's source commits and ``flock-pure-gpu`` sha256s (each run's ``job.json`` tree sha and
``out/binary.sha256``); the union bound over the batch's sub-batch proofs; ``net.rtt_ms`` as a network round trip (the
coin-wait figure moves to ``live.coin_wait_per_round_ms``); ``live.loopback_round_seconds`` with its method.
Nothing measured is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .bench import PER_PROOF_LOG2


def _run(d: Path) -> dict:
    job = json.loads((d / "job.json").read_text())
    sha = (d / "out" / "binary.sha256").read_text().split()[0] if (d / "out" / "binary.sha256").exists() else None
    return {"run": d.name, "commit": job["computation"]["tree_sha"], "binary_sha256": sha,
            "host": (job.get("environment") or {}).get("hostname")}


def _live(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.startswith("LIVE\t"):
            r = json.loads(line.split("\t", 1)[1], strict=False)
            if not r.get("warm") and r.get("accepted"):
                out.append(r)
    return out


def coin_waits(point_dir: Path) -> dict:
    """The coin wait of each timed session without the verifier's final replay and the proof uploads: the transport's
    ``wait_s`` is read after ``Finish``, so it includes ``verdict.verify_s``, which ``e2e_s`` does not."""
    import statistics
    subs = sorted(point_dir.glob("prove-s*.txt"))
    per = [_live(f) for f in subs]
    runs = list(zip(*per))
    wait = statistics.median(sum(max(0.0, r["wait_s"] - r["verdict"]["verify_s"]) for r in run) for run in runs)
    handle = statistics.median(sum(r["verdict"].get("handle_s", 0.0) for r in run) for run in runs)
    calls = sum(r["calls"] for r in per[0][:1]) if len(per) == 1 else sum(x[0]["calls"] for x in per)
    lb = [max(0.0, r["wait_s"] - r["verdict"]["verify_s"]) / r["calls"] for f in sorted(point_dir.glob("loopback-prove-s*.txt")) for r in _live(f)]
    return {"wait": wait, "handle": handle, "calls": calls, "loopback": statistics.median(lb) if lb else None, "loopback_sessions": len(lb)}


def correct_waits(doc: dict, cw: dict) -> dict:
    ms = {m["name"]: m for m in doc["measurements"]}
    t_total = ms["t.total"]["value"]
    vals = {"net.wait_seconds": cw["wait"], "live.network_wait_seconds": max(0.0, cw["wait"] - cw["handle"]),
            "live.prover_compute_seconds": t_total - cw["wait"],
            "live.coin_wait_per_round_ms": 1e3 * max(0.0, cw["wait"] - cw["handle"]) / max(1, cw["calls"])}
    if cw["loopback"] is not None:
        vals["live.loopback_round_seconds"] = cw["loopback"]
    for k, v in vals.items():
        if k in ms:
            ms[k]["value"] = v
        else:
            doc["measurements"].append({"name": k, "value": v, "unit": "ms" if k.endswith("_ms") else "s"})
    inter = doc.setdefault("interaction", {})
    inter["wait_correction"] = ("coin waits exclude the verifier's end-of-session replay and the proof uploads: per session "
                                "wait_s - verdict.verify_s (the transport's wait_s is read after Finish; e2e_s and t.total never "
                                "included them); the loopback probe's per-round figure is corrected the same way")
    if cw["loopback"] is not None and inter.get("loopback_method") and "wait_s - verify_s" not in inter["loopback_method"]:
        inter["loopback_method"] = inter["loopback_method"].replace("(wait_s / calls)", "((wait_s - verify_s) / calls)")
    return doc


def patch(doc: dict, prover: dict, verifier: dict, a) -> dict:
    fp = doc["workload_fingerprint"]
    k = int(fp.get("N_subbatches") or 1)
    sec = fp["security"]
    sec.update({"per_proof_log2": PER_PROOF_LOG2, "proofs": k, "achieved_log2": PER_PROOF_LOG2 + math.log2(k),
                "union": "union bound over the batch's sequential sub-batch proofs (red-team-flock grant of flock-pure-block/v2)"})
    be = fp["software"]["backend"]
    be["commit"], be["binary_sha256"] = prover["commit"], prover["binary_sha256"]
    fp["software"]["verifier"] = {"binary": f"{getattr(a, 'verifier_binary', 'flock-pure-gpu')} serve (CPU build)", "commit": verifier["commit"],
                                  "binary_sha256": verifier["binary_sha256"], "run": verifier["run"], "host": verifier["host"],
                                  "operator": (doc.get("interaction") or {}).get("verifier_operator")}
    ms = {m["name"]: m for m in doc["measurements"]}
    inter = doc.setdefault("interaction", {})
    if "live.coin_wait_per_round_ms" not in ms and "net.rtt_ms" in ms:
        ms["net.rtt_ms"]["name"] = "live.coin_wait_per_round_ms"
        inter["coin_wait_method"] = inter.pop("rtt_method", None)
        ms = {m["name"]: m for m in doc["measurements"]}
    if a.rtt_ms is not None:
        if "net.rtt_ms" in ms:
            ms["net.rtt_ms"]["value"] = a.rtt_ms
        else:
            doc["measurements"].append({"name": "net.rtt_ms", "value": a.rtt_ms, "unit": "ms"})
        inter["rtt_method"] = a.rtt_method
    if a.loopback_seconds is not None and "live.loopback_round_seconds" not in ms:
        doc["measurements"].append({"name": "live.loopback_round_seconds", "value": a.loopback_seconds, "unit": "s"})
        inter["loopback_method"] = a.loopback_method
    doc.update({"lane": a.lane, "registered_by": a.lane,
                "derived_from": {"prover_run": prover["run"], "verifier_run": verifier["run"], "supersedes": a.supersedes},
                "registration": "re-registered from the recorded runs; measurements unchanged except the renamed/added "
                                "interaction fields named in interaction.*_method"})
    doc.pop("artifacts", None)
    doc.pop("measurement_files", None)
    return doc


def main(argv=None) -> int:
    from verity_numerical.bench import contract

    ap = argparse.ArgumentParser(prog="verity_flock.register")
    ap.add_argument("--prover-run", required=True)
    ap.add_argument("--verifier-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lane", required=True, help="the producing lane (the result's lane / registered_by: the non-producer rule "
                                                  "excludes its labels from verification)")
    ap.add_argument("--loopback-seconds", type=float, default=None)
    ap.add_argument("--loopback-method", default=None)
    ap.add_argument("--rtt-ms", type=float, default=None)
    ap.add_argument("--rtt-method", default=None)
    ap.add_argument("--supersedes", default=None)
    ap.add_argument("--verifier-binary", default="flock-pure-gpu")
    a = ap.parse_args(argv)
    pd, vd = Path(a.prover_run), Path(a.verifier_run)
    raw = json.loads((pd / "result.json").read_text())
    raw.pop("kind", None)  # ir_bench's run result is a bench-point/v1; the cell registered here is the bench-result/v1
    point = pd / raw["artifacts"][0]
    doc = patch(raw, _run(pd), _run(vd), a)
    if any(point.glob("prove-s*.txt")):
        doc = correct_waits(doc, coin_waits(point))
    else:
        doc.setdefault("interaction", {})["wait_correction"] = "not recomputed: the run's session lines (prove-s*.txt) are not present"
    problems = contract.validate(doc)
    doc["validation"]["contract_problems"] = problems
    from verity_numerical.bench import views
    ix = views.interaction(doc, interactive=True)
    inter_problem = views.interaction_problem(ix)
    doc["validation"]["interaction_problem"] = inter_problem
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({"problems": problems, "interaction": inter_problem or "PASS (renderer's rule: loopback per round is prover "
                      "compute, the excess is network wait, checked at the run's measured RTT)", "live": {m["name"]: m["value"] for m in doc["measurements"]
                                                     if m["name"].startswith(("live.", "net.", "t.total"))},
                      "achieved_log2": doc["workload_fingerprint"]["security"]["achieved_log2"],
                      "verifier": doc["workload_fingerprint"]["software"]["verifier"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
