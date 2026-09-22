"""Combine the per-range parts of a sharded ``compact_p1_validate`` run spread over several machines (lane COMPACT-P1).

    python scripts/compact_p1_merge.py --plan results/a.plan.json --parts results/a.parts.jsonl results/b.parts.jsonl --out merged.json

Every machine ran the same driver on the same program with the same ``--chunks`` (so the same work-balanced
``bounds``), one forwards and one ``--reverse``; each appends a JSON line per finished range.  The merge keeps one
part per range (the first seen), requires the parts to tile ``[0, n)`` exactly and to agree on the commitment side,
and sums the verification-side totals the way :func:`compact_p1_validate.merge_parts` does in one process.  The
plan-side record (digest, per-kind counts, census comparison) is the ``.plan.json`` snapshot the driver writes before
validating; the machines' snapshots must carry the same partition digest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_p1_validate import merge_parts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, nargs="+", help=".plan.json snapshot(s); all must agree on the partition digest")
    ap.add_argument("--parts", required=True, nargs="+", help=".parts.jsonl files (one line per finished range)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    plans = [json.loads(Path(p).read_text()) for p in args.plan]
    digests = {p["plan"]["partition_digest"] for p in plans}
    if len(digests) != 1:
        print(f"plan snapshots disagree on the partition digest: {digests}", file=sys.stderr)
        return 2
    doc = dict(plans[0])
    n = doc["plan"]["vus"]

    # every finished range from every machine (a range done twice: first file wins), then a greedy tiling of [0, n)
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    seen: dict[str, int] = {}
    for path in args.parts:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = tuple(rec["part"]["range"])
            seen[path] = seen.get(path, 0) + 1
            candidates.setdefault(key, {**rec["part"], "source": path})
    if not candidates:
        print("no parts", file=sys.stderr)
        return 2
    chosen: list[dict[str, Any]] = []
    pos = 0
    ordered = sorted(candidates)
    while pos < n:
        starting = [k for k in ordered if k[0] == pos]
        if not starting:
            nxt = [k for k in ordered if k[0] > pos]
            if not nxt:
                break
            pos = nxt[0][0]  # a gap: merge_parts reports it
            continue
        best = max(starting, key=lambda k: k[1])  # a whole range over its subdivision (both are complete parts)
        chosen.append(candidates[best])
        pos = best[1]
    parts = chosen
    sources: dict[str, int] = {}
    for p in parts:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
    val = merge_parts(parts, None, workers=0, chunks=len(parts), n=n, seconds=0.0, rows=None)
    val["mode"] = f"sharded over {len(args.parts)} machines: {len(parts)} ranges chosen of {len(candidates)} finished ({sources})"
    val["seconds"] = None  # wall time is per machine; see ``worker_seconds_sum`` and the machines' logs
    val["validated"] = val["status"] == "valid"
    val["vus_validated"] = val["vu_stats"]["units"]
    doc["validation"] = val
    doc["merged_from"] = {"plans": args.plan, "parts": args.parts, "parts_seen_per_source": seen, "ranges_chosen": len(parts),
                          "ranges_finished": len(candidates)}
    doc["worker_maxrss_mb"] = val["worker_maxrss_mb"]
    Path(args.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"{val['status']}: {len(parts)} ranges from {sources}; {val['vus_validated']}/{n} VUs validated; gaps {val.get('gaps', [])}; "
          f"vu_stats={val['vu_stats']} cu_stats={val['cu_stats']}  -> {args.out}")
    return 0 if val["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
