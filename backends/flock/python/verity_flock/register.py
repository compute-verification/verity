"""Re-register a Flock cell's ``bench-result/v1`` from its recorded runs, with what the red team and the renderer require.

    python -m verity_flock.register --prover-run DIR --verifier-run DIR --out meta.json \\
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


def patch(doc: dict, prover: dict, verifier: dict, a) -> dict:
    fp = doc["workload_fingerprint"]
    k = int(fp.get("N_subbatches") or 1)
    sec = fp["security"]
    sec.update({"per_proof_log2": PER_PROOF_LOG2, "proofs": k, "achieved_log2": PER_PROOF_LOG2 + math.log2(k),
                "union": "union bound over the batch's sequential sub-batch proofs (red-team-flock grant of flock-pure-block/v2)"})
    be = fp["software"]["backend"]
    be["commit"], be["binary_sha256"] = prover["commit"], prover["binary_sha256"]
    fp["software"]["verifier"] = {"binary": "flock-pure-gpu serve (CPU build)", "commit": verifier["commit"],
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
    doc.update({"lane": "flock-backend", "registered_by": "flock-backend",
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
    ap.add_argument("--loopback-seconds", type=float, default=None)
    ap.add_argument("--loopback-method", default=None)
    ap.add_argument("--rtt-ms", type=float, default=None)
    ap.add_argument("--rtt-method", default=None)
    ap.add_argument("--supersedes", default=None)
    a = ap.parse_args(argv)
    pd, vd = Path(a.prover_run), Path(a.verifier_run)
    doc = patch(json.loads((pd / "result.json").read_text()), _run(pd), _run(vd), a)
    problems = contract.validate(doc)
    doc["validation"]["contract_problems"] = problems
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({"problems": problems, "achieved_log2": doc["workload_fingerprint"]["security"]["achieved_log2"],
                      "verifier": doc["workload_fingerprint"]["software"]["verifier"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
