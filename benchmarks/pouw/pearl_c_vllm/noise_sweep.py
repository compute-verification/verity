"""Pearl-C's served decode overhead over stock FP8 across a sweep's points, without hashing (noise and denoise only, the
``pearlc-nohash`` arm) and with it (``pearlc``), from the points' `e2e.py` outputs (noise_sweep.sh's ``out/points/*/e2e.json``).

  python noise_sweep.py RUN_OUT_DIR... --json sweep.json --md sweep.md

A point directory is ``<point>.r<k>[.t<n>]`` (replicate k, try n).  Per run, each Pearl-C rep is paired with the stock reps
timed around it (`CYCLE`: fp8, pearlc, fp8, pearlc-nohash, fp8, bf16, so ``pearlc`` rep i sits between fp8 reps 3i and 3i+1
and ``pearlc-nohash`` rep i between 3i+1 and 3i+2; ``fp8-graphs`` follows each fp8 rep), and its ratio is its step over
their mean.  A point's row is the median of its paired ratios over every rep of every run, with their min and max, against
eager stock FP8 and, where the run timed it, stock FP8 with vLLM's CUDA graphs.  A run in ABAB blocks has no pairs; its
ratios are its medians'.  Untimed screens (tier 2a): a no-hash ratio under 1.00 is flagged for the assessor.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ARMS = ("pearlc-nohash", "pearlc")
#: each arm's offsets among the three fp8 reps of its cycle (`e2e.CYCLE`)
AROUND = {"pearlc": (0, 1), "pearlc-nohash": (1, 2)}


def _decode(results: dict, mode: str) -> list[dict]:
    reps = results.get(mode, {}).get("reps", {})
    key = next((k for k in reps if k.startswith("decode_")), None)
    return sorted(reps[key], key=lambda r: r["rep"]) if key else []


def run_ratios(e2e: dict) -> dict:
    """One run's paired ratios: ``{base: {arm: [ratio per rep]}}`` for base ``fp8`` and ``fp8-graphs``, the per-rep hashing share
    (``pearlc`` less ``pearlc-nohash``, each over its own pair), and each mode's median step and SM clock."""
    res = e2e.get("results", {})
    out: dict = {"ratios": {}, "step_ms": {}, "sm_mhz": {}}
    for mode in ("fp8", "fp8-graphs", "bf16", *ARMS):
        reps = _decode(res, mode)
        if reps:
            out["step_ms"][mode] = statistics.median(r["step_s"] for r in reps) * 1e3
            mhz = [r["clock"]["sm_mhz"] for r in reps if "sm_mhz" in r.get("clock", {})]
            out["sm_mhz"][mode] = statistics.median(mhz) if mhz else None
    paired = e2e.get("interleave") == "rep"
    for base in ("fp8", "fp8-graphs"):
        b = _decode(res, base)
        if not b:
            continue
        got = {}
        for arm in ARMS:
            a = _decode(res, arm)
            if paired and len(b) >= 3 * len(a):
                got[arm] = [r["step_s"] / statistics.mean(b[3 * r["rep"] + o]["step_s"] for o in AROUND[arm]) for r in a]
            elif a:
                got[arm] = [statistics.median(r["step_s"] for r in a) / statistics.median(r["step_s"] for r in b)]
        if all(arm in got for arm in ARMS):
            got["hashing"] = [h - n for h, n in zip(got["pearlc"], got["pearlc-nohash"], strict=False)]
        out["ratios"][base] = got
    return out


def point_of(path: Path) -> tuple[str, str]:
    """``(point, replicate)`` from a point directory's name."""
    m = re.fullmatch(r"(.+)\.(r\d+)(?:\.t\d+)?", path.name)
    return (m.group(1), m.group(2)) if m else (path.name, "r1")


def summarize(dirs: list[Path]) -> dict:
    """Every point's rows over its finished runs (an ``e2e.json`` with no error, its arm's gate passed, every arm timed)."""
    points: dict = {}
    for d in dirs:
        for f in sorted(d.glob("*/e2e.json")):
            point, rep = point_of(f.parent)
            e2e = json.loads(f.read_text())
            ok = not e2e.get("error") and e2e.get("summary", {}).get("gates", {}).get("arm") and all(
                _decode(e2e.get("results", {}), m) for m in ("fp8", *ARMS))
            p = points.setdefault(point, {"runs": [], "failed": []})
            if not ok:
                p["failed"].append({"dir": str(f.parent), "error": (e2e.get("error") or "gate or arm missing")[:300]})
                continue
            p["runs"].append({"dir": str(f.parent), "replicate": rep, "run": e2e.get("run"), "scheme": e2e.get("scheme"),
                              "schedule": e2e.get("schedule"), "graphs": e2e.get("graphs"), "whole_step": e2e.get("whole_step"),
                              "whole_defer": e2e.get("whole_defer"), "rows_form": e2e.get("rows_form"), "decode": e2e.get("decode"),
                              "interleave": e2e.get("interleave"), **run_ratios(e2e)})
    for p in points.values():
        p["rows"] = {}
        for base in ("fp8", "fp8-graphs"):
            row = {}
            for arm in (*ARMS, "hashing"):
                xs = [x for r in p["runs"] for x in r["ratios"].get(base, {}).get(arm, [])]
                if xs:
                    row[arm] = {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs),
                                "runs": len([r for r in p["runs"] if r["ratios"].get(base, {}).get(arm)])}
            if row:
                row["flag"] = ("no-hash under 1.00x: for the assessor, not a win"
                               if row.get("pearlc-nohash", {}).get("median", 2) < 1.0 else None)
                p["rows"][base] = row
        steps = {m: statistics.median(r["step_ms"][m] for r in p["runs"] if m in r["step_ms"])
                 for m in ("fp8", "fp8-graphs", "bf16", *ARMS) if any(m in r["step_ms"] for r in p["runs"])}
        p["step_ms"] = steps
    return {"schema": "verity-pouw/noise-sweep/v1", "tier": "2a, untimed preemptible screens (not panel rows)",
            "points": dict(sorted(points.items()))}


def _cell(c: dict | None, digits: int = 3) -> str:
    if not c:
        return "-"
    return f"{c['median']:.{digits}f} [{c['min']:.{digits}f}, {c['max']:.{digits}f}] n={c['n']}/{c['runs']}"


def table(doc: dict) -> str:
    lines = ["| point | base | fp8 step ms | no-hash / fp8 | -h2 / fp8 | hashing share | runs | flag |",
             "|---|---|---|---|---|---|---|---|"]
    for name, p in doc["points"].items():
        runs = ", ".join(sorted({r["run"] or "?" for r in p["runs"]}))
        for base, row in p["rows"].items():
            lines.append(f"| {name} | {base} | {p['step_ms'].get(base, float('nan')):.2f} | {_cell(row.get('pearlc-nohash'))} | "
                         f"{_cell(row.get('pearlc'))} | {_cell(row.get('hashing'))} | {runs} | {row.get('flag') or ''} |")
        if p["failed"]:
            lines.append(f"| {name} | failed x{len(p['failed'])} | | | | | | {p['failed'][0]['error'][:80]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", type=Path, nargs="+", help="directories holding point directories (a run's out/points)")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--result", action="store_true",
                    help="write the run's result.json (each point's e2e.py wrote its own over it): passed when every point finished a run")
    a = ap.parse_args()
    doc = summarize(a.dirs)
    a.json.write_text(json.dumps(doc, indent=1))
    md = table(doc)
    if a.md:
        a.md.write_text(md)
    print(md)
    if a.result:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import run_result
        done = [n for n, p in doc["points"].items() if p["runs"]]
        ms = [(f"{n}.{base}.{arm}", c["median"], "x") for n, p in doc["points"].items() for base, row in p["rows"].items()
              for arm, c in row.items() if isinstance(c, dict)]
        run_result.write(a.json, len(done) == len(doc["points"]) > 0, "noise sweep: every point finished a gated run (tier 2a, untimed)",
                         f"{len(done)} of {len(doc['points'])} points finished", ms)


if __name__ == "__main__":
    main()
