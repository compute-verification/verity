"""A run's measurements as one JSON object a build (or a candidate), with the integer-issue ceiling derived from its SASS.

    python3 summary.py sha512 OUT     sha512_baseline.sh's run: per build, the raw rate and the 4 KiB row-seg leaves
    python3 summary.py hashes OUT     candidates.sh's run: per candidate, against full SHA-512 in the same rounds

The ceiling: SMs x 4 sub-partitions x the median SM clock x 32 lanes / the warp cycles a block (sass_count.py's bound:
2 cycles an ALU or FMA instruction, 1 an issue).
"""
from __future__ import annotations

import json
import os
import statistics
import sys


def lines(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.startswith("{")] if os.path.exists(path) else []


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def sms(out: str) -> int:
    for name in sorted(os.listdir(out)):
        for r in lines(os.path.join(out, name)) if name.endswith(".jsonl") else []:
            for v in (r, r.get("sha", {}), r.get("meta", {})):
                if isinstance(v, dict) and "sms" in v:
                    return int(v["sms"])
    return 0


def gpu_name(out: str) -> str:
    p = os.path.join(out, "gpu.txt")
    if not os.path.exists(p):
        return ""
    rows = [x for x in open(p).read().splitlines()[1:] if x and not x.startswith("index")]
    return rows[0].split(", ")[2] if rows else ""


def ceiling(sass: dict, n_sm: int, mhz: float) -> float:
    return n_sm * 4 * mhz * 1e6 * 32 / sass["warp_cycles_per_block_bound"] * 1e-9 if sass and n_sm else float("nan")


def sha512(out: str) -> None:
    n_sm = sms(out)
    straight = {}
    p = os.path.join(out, "sass-a0r0.json")
    if os.path.exists(p):
        straight = json.load(open(p))
    for t in sorted(x[len("sha-"):-len(".jsonl")] for x in os.listdir(out) if x.startswith("sha-") and x.endswith(".jsonl")):
        raw = [r["sha"] for r in lines(os.path.join(out, f"sha-{t}.jsonl"))]
        mhz = med([r["mhz"] for r in raw])
        sass = json.load(open(os.path.join(out, f"sass-{t}.json"))) if os.path.exists(os.path.join(out, f"sass-{t}.json")) else {}
        rec = {"build": t, "gpu": gpu_name(out), "sms": n_sm, "rounds": len(raw), "gcomp_per_s": med([r["gcomp_per_s"] for r in raw]),
               "gcomp_per_s_all": [r["gcomp_per_s"] for r in raw], "gb_per_s_blocks": med([r["gb_per_s"] for r in raw]),
               "mhz": mhz, "sass_total_per_block": sass.get("total_per_block"), "sass_pipes_per_block": sass.get("pipes_per_block"),
               "sass_per_block": sass.get("per_block"), "ceiling_gcomp_per_s": ceiling(sass, n_sm, mhz),
               "ceiling_from_straight_line_gcomp_per_s": ceiling(straight, n_sm, mhz)}
        rec["fraction_of_ceiling"] = rec["gcomp_per_s"] / rec["ceiling_from_straight_line_gcomp_per_s"]
        by_n: dict[int, list[dict]] = {}
        for r in lines(os.path.join(out, f"bench-seg-{t}.jsonl")):
            if "bench" in r:
                by_n.setdefault(r["n"], []).append(r)
        rec["leaves_4k_rowseg"] = {
            str(n): {"operand_mib": rs[0]["operand_bytes"] / 2**20, "rounds": len(rs),
                     "data_phase_ms": med([r["commit_ms"] for r in rs]), "data_phase_gb_per_s": med([r["operand_gb_per_s"] for r in rs]),
                     "graph_ms": med([r["graph_ms"] for r in rs]), "graph_gb_per_s": med([r["operand_gb_per_s_graph"] for r in rs]),
                     "salt_ms": med([r["salt_ms"] for r in rs]),
                     "with_salt_gb_per_s": med([r["operand_bytes"] / (r["commit_ms"] + r["salt_ms"]) * 1e-6 for r in rs]),
                     "gcomp_per_s": med([r["gcomp_per_s"] for r in rs]), "mhz": med([r["mhz"] for r in rs])}
            for n, rs in sorted(by_n.items())}
        print(json.dumps(rec))


def hashes(out: str) -> None:
    rows = lines(os.path.join(out, "bench.jsonl"))
    sass = {}
    for name in os.listdir(out):
        if name.startswith("sass-") and name.endswith(".json"):
            s = json.load(open(os.path.join(out, name)))
            sass[s["candidate"]] = s
    n_sm = sms(out)
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if "candidate" in r:
            by.setdefault((r["candidate"], r["mode"]), []).append(r)
    for (cand, mode), rs in sorted(by.items()):
        mhz = med([r["mhz"] for r in rs])
        rec = {"candidate": cand, "mode": mode, "gpu": gpu_name(out), "sms": n_sm, "reps": len(rs),
               "gcomp_per_s": med([r["gcomp_per_s"] for r in rs]), "gb_per_s": med([r["gb_per_s"] for r in rs]),
               "baseline_gcomp_per_s": med([r["base_gcomp_per_s"] for r in rs]), "baseline_gb_per_s": med([r["base_gb_per_s"] for r in rs]),
               "ratio_to_baseline_gb": med([r["gb_per_s"] / r["base_gb_per_s"] for r in rs]),
               "mhz": mhz, "base_mhz": med([r["base_mhz"] for r in rs])}
        s = sass.get(cand)
        if s and mode == "raw":
            rec["sass_total_per_block"] = s["total_per_block"]
            rec["sass_pipes_per_block"] = s["pipes_per_block"]
            rec["ceiling_gcomp_per_s"] = ceiling(s, n_sm, mhz)
            rec["instructions_per_byte"] = s["total_per_block"] / s["block_bytes"]
        print(json.dumps(rec))


if __name__ == "__main__":
    {"sha512": sha512, "hashes": hashes}[sys.argv[1]](sys.argv[2])
