"""CIR-263: the words a card run of glm53_mtp_eh_proj_capture_gpu kept (<out>/words: W once, and each launch's rows, x and
output), compared on a CPU as the capture's words phase compares them: `BG.compare` over the capture's own `jobs_of`, each launch
at the split the run's routes give its M (ROUTES: the list the run printed).  A run that stopped before its words phase left its
launches' words uncompared; this reads them.  Writes <out>/ehproj_words.json and prints, per launch, rows and words compared,
words that differ, reference coordinates and where the reference differs from the twin; exits 1 on any difference or error.

    cd adapters/vllm; python ehproj_words.py --words DIR --out DIR --routes '[[1, 104, null], ...]' [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tests.properties import glm53_bf16_gemm_capture_gpu as BG
from tests.properties import glm53_mtp_eh_proj_capture_gpu as X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    words, routes = Path(a.words), json.loads(a.routes)
    ms = sorted(int(d.name[1:]) for d in words.iterdir() if d.is_dir() and d.name.startswith("M"))
    cases, jobs = {}, []
    for M in ms:
        d = words / f"M{M}"
        rows = np.load(d / "rows.npy")
        if rows.tolist() != list(range(M)):
            print(f"M={M}: kept rows are not 0..M-1 ({rows.size} rows): skipped", flush=True)
            continue
        split = X.split_at(routes, M)
        job = {"model": X.MODEL, "phase": X.PHASE, "case": X.NAME, "M": M, "K": X.K, "N": X.N, "kernel": None, "split_k": split,
               "paths": {"x": str(d / "x.npy"), "w": str(words / "w.npy"), "out": str(d / "out.npy")}, "probe_paths": None}
        cases[M] = {"split": split, "totals": {k: 0 for k in X.TOTALS}, "first_mismatch": None, "errors": [], "jobs": 0}
        for j in X.jobs_of(job):
            jobs.append(j)
            cases[M]["jobs"] += 1
    print(f"{len(jobs)} jobs over M={sorted(cases)} on {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(BG.compare, jobs):
            c = cases[res["M"]]
            if "error" in res:
                c["errors"].append(res["error"])
                continue
            c["totals"]["rows"] += res["rows"][1] - res["rows"][0]
            for k in BG.TOTALS:
                c["totals"][k] += res[k]
            c["first_mismatch"] = c["first_mismatch"] or res["first_mismatch"]
    secs = time.time() - t0
    bad = 0
    for M, c in sorted(cases.items()):
        t = c["totals"]
        ok = not c["errors"] and t["words"] > 0 and t["mismatch"] == 0 and t["reference_coords"] > 0 and t["twin_mismatch"] == 0
        bad += not ok
        print(f"M={M:5d} split={c['split']} rows={t['rows']} words={t['words']} !={t['mismatch']} ref={t['reference_coords']} "
              f"twin!={t['twin_mismatch']} first={json.dumps(c['first_mismatch'])} errors={c['errors'][:1]}", flush=True)
    totals = {k: sum(c["totals"][k] for c in cases.values()) for k in X.TOTALS}
    doc = {"routes": routes, "cases": {f"M{M}": c for M, c in cases.items()}, "totals": totals, "seconds": secs,
           "workers": a.workers, "ok": bad == 0 and bool(cases)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ehproj_words.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"EHPROJ-WORDS {'EQUAL' if doc['ok'] else 'DIFFER'} totals={json.dumps(totals)} seconds={secs:.0f} "
          f"rows_per_worker_second={totals['rows'] / max(secs * a.workers, 1e-9):.3f}", flush=True)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
