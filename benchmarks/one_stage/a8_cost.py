"""A8's cost points as observations of the one cost model (`verity.core.protocols.sampled_proofs.costfit`, fitted in
benchmarks/sampling_plan, whose `inputs.py` holds these), and the times each session measured.

    python benchmarks/one_stage/a8_cost.py RUN [RUN ...] [--json]

RUN is an A8 run's directory: its records in ``out/a8`` (`a8.py --points`, each session's ``cost/<arm>-k<k>-r<rep>``), and
``gpu-lease-usage.jsonl`` where `a8_gpu.sh` ran each prover process under a GPU lease. Per session it prints, in seconds
and per drawn opening: the device phase (each proof rep's ``t.total``, what the prover's device, the GPU under ``--gpu``,
spends on the proof), the prover's e2e less its wait on the verifier, e2e, the process's wall, the seconds it held its
lease (the wall less the lease's wait, matched by the session's verifier port), and the process's CPU seconds. Then, of
each leased run's design arm (``fw``), the `costfit.Observation`s: per (staged, drawn) units, prove the mean seconds its
cold prover processes held the device, and stage the statement's staging. They carry no verify: a8.py times Lean's by
wall, and the model's verify is CPU time.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from verity.core.protocols.sampled_proofs.costfit import Observation

PARAMS = ("N", "C", "K", "MASKED", "BYTES", "ROWS", "LEAF", "P", "DR", "DP", "DV")


def held(run: Path) -> dict[str, int]:
    """Each leased prover process's held seconds, by the verifier port on its command line."""
    f = run / "gpu-lease-usage.jsonl"
    out: dict[str, int] = {}
    for line in (f.read_text().splitlines() if f.exists() else []):
        u = json.loads(line)
        port = u["cmd"].split("--verifier ", 1)[1].split()[0]
        if port in out:
            raise SystemExit(f"{f}: two leases for verifier {port}; a session can't be matched to its lease")
        out[port] = sum(g["held_s"] for g in u["gpus"])
    return out


def sessions(run: Path) -> list[dict]:
    """Every cost session of ``run``: cost.jsonl's row with its LIVE line's breakdown and its lease's held seconds."""
    out, leases, rows = run / "out" / "a8", held(run), []
    for line in (out / "cost" / "cost.jsonl").read_text().splitlines():
        r = json.loads(line)
        log = (out / "cost" / f"{r['arm']}-k{r['k']}-r{r['rep']}" / "prove.log").read_text().splitlines()
        live = next((json.loads(x[5:]) for x in log if x.startswith("LIVE\t")), {})
        b = live.get("buckets") or []
        rows.append({**r, "device_s": sum(x["t.total"] for x in b) if b and all("t.total" in x for x in b) else None,
                     "prover_s": r["e2e_s"] - (live.get("wait_e2e_s") or 0.0),
                     "held_s": leases.get(log[0].split("--verifier ", 1)[1].split()[0]) if "--verifier " in log[0] else None})
    return rows


def observations(run: Path, rows: list[dict]) -> list[Observation]:
    """The design arm's points of a leased run, each the mean of its cold processes."""
    s = json.loads((run / "out" / "a8" / "summary.json").read_text())
    template = "BlockOpeningCheck_v1{" + ",".join(f"{p}={s['statics'][p]}" for p in PARAMS) + "}"
    pts: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        if r["arm"] == "fw" and r["accepted"] and r["gpu"] and r["held_s"] is not None:
            pts.setdefault((r["staged"], r["k"]), []).append(r["held_s"])
    return [Observation(run.name, template, s["honest"]["ands"], c, k, prove=round(mean(p), 2), stage=s["honest"]["stage_s"])
            for (c, k), p in sorted(pts.items())]


TIMES = ("device_s", "prover_s", "e2e_s", "wall_s", "held_s", "cpu_s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    report = {}
    for run in a.runs:
        rows = sessions(run)
        report[run.name] = {"sessions": rows, "observations": [asdict(o) for o in observations(run, rows)]}
    if a.json:
        print(json.dumps(report, indent=1))
        return 0
    for name, r in report.items():
        print(f"{name}: " + "  ".join(f"{x} (/opening)" for x in TIMES))
        for s in r["sessions"]:
            cells = ["-" if s.get(x) is None else f"{s[x]:.1f} ({s[x] / s['k']:.4f})" for x in TIMES]
            print(f"  {s['arm']:5s} staged {s['staged']} k={s['k']} rep {s['rep']}{'' if s['accepted'] else ' REJECTED'}: "
                  + "  ".join(cells))
        for o in r["observations"]:
            print(f"  observation {json.dumps(o)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
