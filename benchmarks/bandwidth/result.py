from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from verity_bench import kit

METRIC = "memory-accounting/host-write-rate"


def cpu_model() -> str:
    lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    return next((x.split(":", 1)[1].strip() for x in lines if x.startswith("model name")), "unknown")


def record(out: Path, cpu: str) -> int:
    log = out / "write_host.jsonl"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.startswith("{")] if log.is_file() else []
    wrong = sum(r["wrong"] for r in rows)
    ok = bool(rows) and wrong == 0
    w = kit.writer(out)
    w.validation("passed" if ok else "failed", "write_host.c",
                 f"{len(rows)} records, {wrong} wrong of {sum(r['checked'] for r in rows)} checked",
                 [os.path.relpath(log, w.out)] if rows else [])
    if not ok:
        return 5
    best = max(rows, key=lambda r: r["gb_per_s"])
    m = w.measurement(METRIC, best["gb_per_s"], "GB a second", mark="measured",
                      extra={"cpu": cpu, "cores": max(r["threads"] for r in rows), "threads": best["threads"],
                             "kernel": best["kernel"], "set": best["label"]})
    print(json.dumps({k: m[k] for k in ("name", "value", "unit", "cpu", "cores", "threads", "kernel", "set")}
                     | {"node": m["host"].get("node")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(record(Path(sys.argv[1]), cpu_model()))
