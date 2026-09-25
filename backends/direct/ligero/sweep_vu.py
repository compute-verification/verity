"""The TABLES.md batch sweep for ``bench-vu``: double ``--total-vus`` from ``--start`` at fixed per-proof settings until two
successive doublings together raise the median throughput by less than 2 % (or a point fails, e.g. out of memory), then stamp
every point's ``result.json`` with the ``sweep`` and ``protocol`` blocks the views read (``bench.views._protocol``,
benchmark-standard §6.1) and the ``commit.seconds`` measurement (:data:`bench.views.COMMIT`) when the run timed its commitment
per rep (``--commit-per-rep``: ``commitment.seconds``).

    python -m backends.direct.ligero.sweep_vu --relation fp8-hopper+blake3 --out-dir D [--start 1024] [--max 65536] \\
        [--dump plateau|all|none] -- --zk --mode interactive --auth included-hash --commit-per-rep --batch 16384 --pipeline 4 \\
        --target -128 --reps 5 --device cuda

Throughput is ``end_to_end.vu_per_second`` (commitment + proving) when present, else VUs / ``t.total``.  Each point is its own
``run bench-vu`` process: its warm-up pass is untimed, its ``--reps`` timed runs give the median, and the timing guard writes
its ``contention`` verdict.  ``--dump plateau`` re-runs nothing: it dumps rep 1 of every point (``--dump-reps 1``) and deletes
the proof files of every non-plateau point once the sweep is decided (their manifests stay).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RULE = ("double total VUs from start at fixed per-proof settings until two successive doublings together raise the median "
        "throughput by < 2 % or a point fails; plateau = the point with the highest median throughput (kb/TABLES.md)")


def _meas(doc: dict[str, Any]) -> dict[str, Any]:
    return {m["name"]: m.get("value") for m in doc.get("measurements") or [] if isinstance(m, dict) and "name" in m}


def throughput(doc: dict[str, Any]) -> float | None:
    m = _meas(doc)
    if isinstance(m.get("end_to_end.vu_per_second"), (int, float)):
        return float(m["end_to_end.vu_per_second"])
    t, n = m.get("t.total"), m.get("vu.count")
    return float(n) / float(t) if isinstance(t, (int, float)) and t > 0 and isinstance(n, (int, float)) else None


def done(tputs: list[float]) -> bool:
    return len(tputs) >= 3 and tputs[-1] < 1.02 * tputs[-3]


def stamp(doc: dict[str, Any], sweep: dict[str, Any], reps: int) -> dict[str, Any]:
    cont = doc.get("contention") if isinstance(doc.get("contention"), dict) else {}
    doc["sweep"] = sweep
    doc["protocol"] = {"warm": True, "runs": reps, "statistic": "median", "contended": cont.get("contended"),
                       "warm_rule": "one full untimed pass (bench-vu warm-up) before the timed reps, same process"}
    m = _meas(doc)
    if "commitment.seconds" in m and "commit.seconds" not in m:
        doc["measurements"].append({"name": "commit.seconds", "value": m["commitment.seconds"], "unit": "s"})
        if "end_to_end.seconds" in m:
            doc["measurements"].append({"name": "e2e.seconds", "value": m["end_to_end.seconds"], "unit": "s"})
    return doc


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bench = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="sweep_vu")
    ap.add_argument("--relation", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=1024)
    ap.add_argument("--max", type=int, default=1 << 17)
    ap.add_argument("--dump", choices=("plateau", "all", "none"), default="plateau")
    ap.add_argument("--id", default=None, help="sweep id (default: a digest of relation, bench args and start time)")
    a = ap.parse_args(own)
    for flag in ("--total-vus", "--out", "--dump-dir", "--dump-reps"):
        if flag in bench:
            ap.error(f"{flag} is set per point by the sweep")
    reps = int(bench[bench.index("--reps") + 1]) if "--reps" in bench else 5
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sid = a.id or "sweep-" + hashlib.sha256(json.dumps([a.relation, bench, time.time()]).encode()).hexdigest()[:12]
    points: list[dict[str, Any]] = []
    total = a.start
    while total <= a.max:
        d = out / f"p{len(points)}-{total}"
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "backends.direct.ligero.run", "--relation", a.relation, "bench-vu", *bench,
               "--total-vus", str(total), "--out", str(d / "result.json")]
        if a.dump != "none":
            cmd += ["--dump-dir", str(d / "proofs"), "--dump-reps", "1"]
        print(f"=== {time.strftime('%H:%M:%SZ', time.gmtime())} point {len(points)} total {total}: {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        with open(d / "bench.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        doc = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else None
        tp = throughput(doc) if doc and rc == 0 else None
        points.append({"point": len(points), "total_vus": total, "rc": rc, "wall_s": round(time.time() - t0, 1),
                       "vu_per_second": tp, "dir": d.name,
                       "contended": (doc or {}).get("contention", {}).get("contended") if doc else None})
        print(f"    rc={rc} vu/s={tp} wall={points[-1]['wall_s']}s contended={points[-1]['contended']}", flush=True)
        if tp is None:
            break
        if done([p["vu_per_second"] for p in points if p["vu_per_second"] is not None]):
            break
        total *= 2
    ok = [p for p in points if p["vu_per_second"] is not None]
    if not ok:
        (out / "sweep.json").write_text(json.dumps({"id": sid, "relation": a.relation, "bench": bench, "points": points}, indent=1))
        return 1
    best = max(ok, key=lambda p: (p["vu_per_second"], -p["total_vus"]))
    stop = ("a point failed (rc %d)" % points[-1]["rc"]) if points[-1]["vu_per_second"] is None else \
        ("< 2 %% over two doublings" if done([p["vu_per_second"] for p in ok]) else "reached --max")
    summary = {"id": sid, "relation": a.relation, "bench": bench, "rule": RULE, "stopped": stop,
               "plateau_point": best["point"], "plateau_total_vus": best["total_vus"], "points": points}
    for p in ok:
        rj = out / p["dir"] / "result.json"
        sweep = {"id": sid, "point": p["point"], "plateau": p is best, "total_vus": p["total_vus"], "rule": RULE,
                 "stopped": stop, "points": [[q["total_vus"], q["vu_per_second"]] for q in points]}
        rj.write_text(json.dumps(stamp(json.loads(rj.read_text()), sweep, reps), indent=1, default=str))
        if a.dump == "plateau" and p is not best:
            for rep in (out / p["dir"] / "proofs").glob("rep*"):
                shutil.rmtree(rep, ignore_errors=True)
    (out / "sweep.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("id", "stopped", "plateau_point", "plateau_total_vus")}), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
