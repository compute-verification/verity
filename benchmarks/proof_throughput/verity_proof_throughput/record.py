"""``record``: the metric's run, the prover points and the Lean verifier on each of their bundles, then its readings."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence

METRIC = "proofs/proof-throughput-blocks-per-second"
UNIT = "proof blocks a second"
MODEL = {"bf16-k6144": "glm-5.3"}


def card_slug(card: str) -> str:
    return "cpu" if card == "none" else "-".join(card.lower().split()).removeprefix("nvidia-")


def member(workload: str, card: str, provers: int) -> str:
    return f"{MODEL[workload]}-{card_slug(card)}-{provers}-prover{'s' if provers > 1 else ''}"


def lease_span(phases: Path) -> tuple[float, float] | None:
    t = {}
    for line in phases.read_text().splitlines() if phases.exists() else []:
        f = line.split("\t")
        if len(f) == 3 and f[0] == "lease" and f[1] in ("start", "end"):
            t[f[1]] = float(f[2])
    return (t["start"], t["end"]) if len(t) == 2 else None


def throughput(points: Sequence[dict]) -> tuple[float | None, str]:
    """The proof blocks a second of provers proving at once: the sum of each one's rate, each from a verifier result
    that passed, and only when every prover's sessions overlapped every other's."""
    bad = [p["dir"] for p in points if (p.get("result") or {}).get("validation", {}).get("status") != "passed"]
    if bad:
        return None, f"no verifier result that passed for {', '.join(bad)}"
    spans = [p.get("span") for p in points]
    if None in spans:
        return None, "a prover's phases name no lease"
    if len(spans) > 1 and max(s for s, _ in spans) >= min(e for _, e in spans):
        return None, "the provers' sessions did not run at once"
    rates = [next((m["value"] for m in p["result"]["measurements"] if m["name"] == "vu.per_second"), None) for p in points]
    if any(not isinstance(r, (int, float)) for r in rates):
        return None, "a verifier result has no vu.per_second"
    return float(sum(rates)), f"{len(points)} prover(s) at once, each verified"


def _sh(root: Path, script: str, env: dict[str, str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["bash", f"benchmarks/proof_throughput/{script}"], cwd=root, env={**os.environ, **env},
                            stdout=log.open("w"), stderr=subprocess.STDOUT)


def run(root: Path, out: Path, workload: str, B: int, provers: Sequence[int], cores: Sequence[str], node: str,
        start: Callable[[Path, str, dict[str, str], Path], subprocess.Popen] = _sh) -> dict[int, list[dict]]:
    points: dict[int, list[dict]] = {}
    for n in sorted(provers):
        group = []
        for i in range(n):
            d = out / f"provers-{n}" / f"prover-{i}"
            env = {"WORKLOAD": workload, "B": str(B), "OUT": str(d), "RESULT": str(d / "result.json"),
                   "PT_PROVER_NODE": node, "SESSIONS_LOCK": str(d / "sessions.lock"), "LEASE_CORES": cores[i]}
            group.append({"dir": str(d.relative_to(out)), "path": d, "proc": start(root, "job.sh", env, d / "job.out")})
        for p in group:
            p["prover_rc"] = p.pop("proc").wait()
        for p in group:
            d = p["path"]
            bundle = (json.loads((d / "bundle.json").read_text()) if (d / "bundle.json").exists() else {}).get("id")
            if p["prover_rc"] == 0 and bundle:
                v = d / "verifier"
                env = {"BUNDLE": bundle if bundle.startswith("art:") else f"art:{bundle}", "OUT": str(v),
                       "RESULT": str(v / "result.json"), "PROVE_RUN": os.environ.get("RESEARCH_RUN_ID", "local")}
                p["verifier_rc"] = start(root, "verify.sh", env, v / "verify.out").wait()
                if (v / "result.json").exists():
                    p["result"] = json.loads((v / "result.json").read_text())
            p["span"] = lease_span(d / "phases.tsv")
        points[n] = group
    return points


def write(out: Path, workload: str, B: int, points: dict[int, list[dict]]) -> int:
    from verity_bench import kit

    w = kit.writer(out)

    def rel(p: dict) -> str:
        return str((out / p["dir"] / "verifier" / "result.json").resolve().relative_to(Path(w.out).resolve()))

    failed = []
    for n, group in sorted(points.items()):
        value, why = throughput(group)
        results = [rel(p) for p in group if p.get("result")]
        extra = {"workload": workload, "B": B, "provers": n, "results": results}
        m = w.measurement(METRIC, value, UNIT, why=None if value is not None else why, mark="measured",
                          member=member(workload, w.stamp["card"], n), extra=extra)
        print(json.dumps({k: m.get(k) for k in ("name", "value", "unit", "member")} | {"why": why}))
        if value is None:
            failed.append(f"{n} prover(s): {why}")
    w.validation("failed" if failed else "passed", "proof_throughput record",
                 "; ".join(failed) or "every prover's sessions accepted by the Lean verifier",
                 evidence=[rel(p) for g in points.values() for p in g if p.get("result")])
    return 1 if failed else 0
