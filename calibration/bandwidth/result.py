"""The run's result.json through the harness's kit (`kit.writer`), from the records a probe of `run.sh` left:
`write_host`'s fastest record as metrics.toml's number, and `chip`'s and `write_gpu`'s validation, their device
program's exit status, with no number.

    python result.py write_host OUT
    python result.py chip|write_gpu OUT RC
"""
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


#: each device probe's checker, its records (the first its evidence) and what its program checks before it reports
PROBES = {"chip": ("bwchip", ("chip.json", "ur.jsonl", "build.txt", "gpu.txt", "sass.txt", "uniform.txt"),
                   "each sweep's chains and chases against the host's"),
          "write_gpu": ("bwwrite", ("write.jsonl", "build.txt", "sass.txt", "gpu.txt", "dmon.txt", "rc.txt"),
                        "every arm on one card")}


def record_probe(probe: str, out: Path, rc: int) -> int:
    """A device probe's validation: passed where its program exited 0 and left its first record, with no number. It
    returns 0 either way: the probe exits with its program's status, as `run.sh` passes it on."""
    checker, files, what = PROBES[probe]
    held = [f for f in files if (out / f).is_file() and (out / f).stat().st_size]
    ok = rc == 0 and files[0] in held
    w = kit.writer(out)
    for f in held:
        w.measurement_file(os.path.relpath(out / f, w.out))
    w.validation("passed" if ok else "failed", checker,
                 f"{checker} exited {rc}: {what}" + ("" if files[0] in held else f"; no {files[0]}"),
                 [os.path.relpath(out / files[0], w.out)] if files[0] in held else [])
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["write_host"] and len(args) == 2:
        sys.exit(record(Path(args[1]), cpu_model()))
    if args[:1] in (["chip"], ["write_gpu"]) and len(args) == 3 and args[2].lstrip("-").isdigit():
        sys.exit(record_probe(args[0], Path(args[1]), int(args[2])))
    sys.exit("usage: python result.py write_host OUT | python result.py chip|write_gpu OUT RC")
