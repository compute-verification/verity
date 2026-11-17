from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from verity_bench import kit

METRIC = "memory-accounting/host-squaring-rate"


def cpu_model() -> str:
    lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    return next((x.split(":", 1)[1].strip() for x in lines if x.startswith("model name")), "unknown")


def record(out: Path, cpu: str) -> int:
    log, errors = out / "time.jsonl", out / "errors.txt"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.startswith("{")] if log.is_file() else []
    gates = sorted(out.glob("gate-*.json"))
    failed = errors.read_text(encoding="utf-8").splitlines() if errors.is_file() else []
    cores = [r["threads"] for r in rows if r["set"] == "cores"]
    if rows and not cores:
        failed.append("no timing on one thread a core")
    ok = bool(rows) and not failed
    w = kit.writer(out)
    w.validation("passed" if ok else "failed", "sq_gate.py",
                 f"{len(gates)} gates, {len(rows)} timings" + "".join(f"; {x}" for x in failed),
                 [os.path.relpath(p, w.out) for p in gates])
    if not ok:
        return 5
    best = max(rows, key=lambda r: r["sq_per_s_sum"])
    m = w.measurement(METRIC, best["sq_per_s_sum"], "squarings a second", mark="measured",
                      extra={"cpu": cpu, "cores": max(cores),
                             "threads": best["threads"], "compiler": best["cc"], "vectors": best["g"], "bits": best["bits"]})
    print(json.dumps({k: m[k] for k in ("name", "value", "unit", "cpu", "cores", "threads", "compiler", "vectors", "bits")}
                     | {"node": m["host"].get("node")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(record(Path(sys.argv[1]), cpu_model()))
