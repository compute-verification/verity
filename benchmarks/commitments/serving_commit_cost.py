#!/usr/bin/env python3
"""A cell's commitment time: the serving hook (``verity.ml.windows.rows.commit``, M0's ``frame-v3-sha512`` +
``hm96-sha512/v1`` row leaves) committing the first ``--n`` instances of an input set afresh, as a Table 1 cell's
``commit.seconds``.

    research run --tool serving_commit_cost --campaign C -- python benchmarks/commitments/serving_commit_cost.py \\
        --input-set DIR --input-set-art art:... --n 1024 [--reps 5] [--workers N]

Per call: fresh salts (a new OS salt key), every row's ``sha512/row/v1`` digest, ``b || c`` and ``frame-v3-sha512`` leaf (each
output port a row too, its words zero-padded to whole SHA-512 blocks), and one tree per port (``serving_rows.commit``).  The rows
are the set's own row ports, one row per instance and port as M0's per-instance statement reads them (no shared rows);
consecutive instances with the same row widths are one member (an attention class set spans key counts, so it is one member per
T).  The first call is ``commit.cold_seconds``; ``commit.seconds``
is the median of the next ``--reps`` (the renderer's ``views.COMMIT``).  The window's binding fields name the input set, not a
served run: they enter each port's domain id once, so they do not move the time.  Validation recomputes two rows' ``b || c`` with
core (``hm96.SHA512.commit_string`` over ``sha512_row_layout``) and compares them with what the hook committed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_layout
from verity_numerical.bench.input_sets import InputSet
from verity.ml.windows import rows as SR

RESULT_SCHEMA = "research/result/v0.1"


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input-set", required=True, help="an input-set directory (research data fetch ART --to DIR)")
    ap.add_argument("--input-set-art", default=None, help="its art: id, recorded in the fingerprint")
    ap.add_argument("--n", type=int, required=True, help="instances committed: the set's first N")
    ap.add_argument("--reps", type=int, default=5, help="warm calls after the cold one")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None, help="directory for result.json (default $RESEARCH_RUN_DIR)")
    ns = ap.parse_args(argv)
    if ns.n <= 0 or ns.reps <= 0 or ns.workers <= 0:
        ap.error("--n, --reps and --workers must be positive")
    return ns


def batch(s: InputSet, n: int) -> tuple[SR.Window, list[dict[str, dict[str, np.ndarray]]]]:
    """One member per run of consecutive instances with the same row widths (an attention class set: one per key count T, as
    serving commits each template instance's rows)."""
    sub = s.subcircuit
    names = [p.name for p in sub.inputs], [p.name for p in sub.outputs]
    arrays = {p: [np.asarray(x, dtype=np.uint16).reshape(-1) for x in s.port(p, 0, n)] for p in names[0] + names[1]}
    shape = [tuple(arrays[p][i].size for p in names[0] + names[1]) for i in range(n)]
    members, data, lo = [], [], 0
    while lo < n:
        hi = lo
        while hi < n and shape[hi] == shape[lo]:
            hi += 1
        rows = {p: np.stack(arrays[p][lo:hi]) for p in names[0]}
        outs = {p: np.stack(arrays[p][lo:hi]) for p in names[1]}
        members.append(SR.Member(f"{sub.id}#{len(members)}", tuple((p, int(a.shape[1])) for p, a in rows.items()),
                                 tuple((p, int(a.shape[1])) for p, a in outs.items()), hi - lo, lo))
        data.append({"rows": rows, "outs": outs})
        lo = hi
    window = SR.Window({"program": s.content_digest}, f"input-set:{s.name}", n, f"serving-commit-cost:{s.content_digest[:16]}", tuple(members))
    return window, data


def check_rows(c: SR.Committed, member: SR.Member) -> list[str]:
    bad = []
    inst = hm96.SHA512
    for p, words in member.all_rows:
        port = member.port(p)
        lay = sha512_row_layout(ROLE_X, 16, words)
        for i in sorted({0, member.n // 2}):
            row = np.ascontiguousarray(c.rows[port][i], dtype="<u2").tobytes()
            salt = c.salts[port][i * SR.SALT_BYTES:(i + 1) * SR.SALT_BYTES]
            want = inst.commit_string(inst.default_key, lay.digest(row), salt)
            if c.bc[port][i * SR.BC_BYTES:(i + 1) * SR.BC_BYTES] != want:
                bad.append(f"{port} row {i}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ns = parse(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    s = InputSet.open(ns.input_set)
    if ns.n > s.n:
        raise SystemExit(f"--n {ns.n} exceeds the set's {s.n} instances")
    window, data = batch(s, ns.n)
    ms = window.members
    calls = []
    for _ in range(1 + ns.reps):
        c = SR.commit(window, data, workers=ns.workers)
        calls.append(c.seconds)
    checked = [ms[0]] + ([ms[-1]] if len(ms) > 1 else [])
    bad = [x for m in checked for x in check_rows(c, m)]
    warm = calls[1:]
    med = {k: statistics.median(x[k] for x in warm) for k in ("total_s", "salts_s", "commit_s")}
    rows = sum(m.n * len(m.all_rows) for m in ms)
    row_bytes = sum(m.n * sum(2 * w for _, w in m.all_rows) for m in ms) / ns.n
    summary = {"set": s.name, "n": ns.n, "workers": ns.workers, "members": len(ms), "cold_s": calls[0]["total_s"], "warm_median_s": med["total_s"],
               "warm_s": [x["total_s"] for x in warm], "rows": rows, "row_bytes_per_instance": row_bytes,
               "roots": {p: r.hex() for p, r in c.roots.items()}, "rows_checked": sum(len(m.all_rows) * len({0, m.n // 2}) for m in checked), "rows_bad": bad}
    print(json.dumps({k: v for k, v in summary.items() if k != "roots"}))
    out = ns.out or os.environ.get("RESEARCH_RUN_DIR")
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "serving_commit_cost.json").write_text(json.dumps({"summary": summary, "calls": calls}, indent=1) + "\n")
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "serving_commit_cost", "scheme": SR.SCHEME, "row_schema": SR.ROW_SCHEMA,
                                     "input_set": {"id": s.name, "art": ns.input_set_art, "content_digest": s.content_digest,
                                                   "manifest_sha256": s.manifest_sha256(), "subcircuit": s.subcircuit.id, "range": [0, ns.n]},
                                     "rows": "one row per instance and row port, the output rows included (M0's per-instance statement, no shared rows); one member "
                                             "per run of instances with the same row widths (an attention class set: one per key count)",
                                     "members": len(ms),
                                     "implementation": "verity.ml.windows.rows.commit (hashlib SHA-512, numpy ChaCha20, fork pool)",
                                     "workers": ns.workers, "reps": ns.reps, "cpu": platform.processor() or platform.machine(),
                                     "nproc": os.cpu_count(), "python": sys.version.split()[0]},
            "validation": {"status": "passed" if not bad else "failed",
                           "checker": "two rows per port: b || c recomputed by core (hm96.SHA512.commit_string over sha512_row_layout)",
                           "detail": "ok" if not bad else ", ".join(bad), "evidence": []},
            "measurements": [
                {"name": "commit.seconds", "value": med["total_s"], "unit": "s"},
                {"name": "commit.cold_seconds", "value": calls[0]["total_s"], "unit": "s"},
                {"name": "commit.reps", "value": ns.reps, "unit": "count"},
                {"name": "commit.salts_seconds", "value": med["salts_s"], "unit": "s"},
                {"name": "commit.tree_seconds", "value": med["commit_s"], "unit": "s"},
                {"name": "commit.workers", "value": ns.workers, "unit": "count"},
                {"name": "commit.rows", "value": rows, "unit": "count"},
                {"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"},
            ],
            "measurement_files": ["serving_commit_cost.json"],
            "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
