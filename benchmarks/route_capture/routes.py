"""What a route capture (route_capture_plugin, run.sh) holds, read back on any machine; stdlib only.

    routes.py summary CAPTURE_DIR     # JSON: every class's steps and tokens, its CUDA-graph dispatches, its profiled
                                      # steps, and the kernels those ran
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from verity_vllm.observe.kernel_trace import kernels_of_trace_events, load_trace


def ranks(capture: Path) -> dict[int, Path]:
    """Each recording rank's steps file."""
    return dict(sorted((int(p.name.split(".")[1].removeprefix("rank")), p) for p in capture.glob("steps.rank*.jsonl*")))


def steps(capture: Path, rank: int) -> Iterator[dict[str, Any]]:
    """The rows of one recording rank's steps file."""
    f = ranks(capture)[rank]
    with (gzip.open(f, "rt") if f.suffix == ".gz" else f.open()) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def kernels(trace: Path) -> Counter[str]:
    """A profiled step's device kernels (memcpys and memsets included) by name, with their launch counts."""
    return Counter(e.get("name", "") for e in kernels_of_trace_events(load_trace(trace)))


def summary(capture: Path, rank: int | None = None) -> dict[str, Any]:
    """Every class of one recording rank (the lowest unless ``rank`` names one), most tokens first."""
    rank = min(ranks(capture)) if rank is None else rank
    by: dict[str, dict[str, Any]] = defaultdict(lambda: {"steps": 0, "tokens": 0, "cg": Counter(), "profiled": []})
    total: Counter[str] = Counter()
    for r in steps(capture, rank):
        c = by[r["class"]]
        c["steps"] += 1
        c["tokens"] += r["tokens"]
        cg = r.get("cg") or {}
        c["cg"][f"{cg.get('cg_mode')}@{cg.get('num_tokens')}"] += 1
        if r.get("profiled"):
            c["profiled"].append(r["profiled"])
        total["steps"] += 1
        total["tokens"] += r["tokens"]
        total["prefill_tokens"] += sum(q for q, _, _ in r["prefill"])
        total["decode_tokens"] += r["decode"]
        total["prefix_hit_tokens"] += r["prefix_hit_tokens"]
    seen_in: Counter[str] = Counter()
    for k, c in by.items():
        ks: Counter[str] = Counter()
        for name in c["profiled"]:
            t = capture / "kineto" / f"rank{rank}" / name
            if t.is_file():
                ks |= kernels(t)
        c["kernels"] = dict(sorted(ks.items()))
        seen_in.update(ks.keys())
        c["cg"] = dict(c["cg"].most_common())
    classes = sorted(by.items(), key=lambda kv: -kv[1]["tokens"])
    return {"rank": rank, **total, "classes": len(by), "profiled": sum(len(c["profiled"]) for c in by.values()),
            "kernels": len(seen_in), "kernel_classes": dict(seen_in.most_common()),
            "by_class": [{"class": k, **c} for k, c in classes]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="every class's steps, tokens, dispatches and kernels")
    s.add_argument("capture", type=Path)
    s.add_argument("--rank", type=int)
    a = ap.parse_args(argv)
    json.dump(summary(a.capture, a.rank), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
