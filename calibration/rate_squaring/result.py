"""The run's result.json through the harness's kit (`kit.writer`), from the records `run.sh`'s `sq` or `ifma` left in
$RESEARCH_RUN_DIR/out: `ifma`'s fastest timing as metrics.toml's number, `sq`'s gates' validation with no number.

    python result.py ifma|sq OUT
"""
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


def record_sq(out: Path) -> int:
    """`sq`'s validation: passed where it timed and no form failed its gate, the gates its evidence and its records its
    files; it writes no number, so a failed gate exits 5."""
    log, errors = out / "time.jsonl", out / "errors.txt"
    rows = [x for x in log.read_text(encoding="utf-8").splitlines() if x.startswith("{")] if log.is_file() else []
    gates = sorted(out.glob("gate-*.json"))
    noted = errors.read_text(encoding="utf-8").splitlines() if errors.is_file() else []
    if not rows:
        noted.append("no timing")
    failed = not rows or any("failed its gate" in x or x.startswith("gate ") for x in noted)
    w = kit.writer(out)
    for name in ("time.jsonl", "probe.jsonl", "sass-mix.json", "errors.txt"):
        if (out / name).is_file():
            w.measurement_file(os.path.relpath(out / name, w.out))
    w.validation("failed" if failed else "passed", "sq_gate.py",
                 f"{len(gates)} gates, {len(rows)} timings" + "".join(f"; {x}" for x in noted),
                 [os.path.relpath(p, w.out) for p in gates])
    return 5 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("ifma", "sq"):
        sys.exit("usage: python result.py ifma|sq OUT")
    sys.exit(record(Path(sys.argv[2]), cpu_model()) if sys.argv[1] == "ifma" else record_sq(Path(sys.argv[2])))
